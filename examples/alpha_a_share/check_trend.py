"""
单票趋势检查（基于日线 + 利弗莫尔规则）

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/check_trend.py 600487
  .venv/bin/python examples/alpha_a_share/check_trend.py 600487.SSE
  .venv/bin/python examples/alpha_a_share/check_trend.py 600487 --price 60.29   # 盘中临时价
  .venv/bin/python examples/alpha_a_share/check_trend.py 600487 --update          # 先拉最新日线
  .venv/bin/python examples/alpha_a_share/check_trend.py 01024 --update         # 港股（仅趋势）

说明：
  - 默认不需要手填价格，使用本地日线最新收盘价（收盘后最准确）
  - 盘中可用 --price 传入券商「最新价」做临时判断
  - A 股建仓建议依赖 predict_daily 信号；港股仅看趋势结构，不参与 ML/利弗莫尔
  - 无实时行情接口，结论以「最近一根日线」为准
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from vnpy.alpha import AlphaLab
from vnpy.trader.constant import Interval

from datafeed import download_daily_data
from livermore_positions_store import load_positions
from predict_daily import (
    ADD_SPACING_PCT,
    BREAKOUT_WINDOW,
    EARLY_ZONE_ARM_PCT,
    EARLY_ZONE_GIVEBACK_PCT,
    LIVERMORE_TOP_N,
    STOP_LOSS_PCT,
    TRAIL_ACTIVATE_PCT,
    TRAIL_PULLBACK_PCT,
    evaluate_held_action,
    is_n_day_breakout,
    suggest_target_shares,
)
from stock_universe import (
    HK_STOCK_LIST,
    STOCK_LIST,
    is_hk_vt_symbol,
    normalize_hk_code,
    stock_entry,
    stock_name,
)

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")


def normalize_vt_symbol(raw: str) -> str:
    s = raw.strip().upper()
    if s.endswith(".SEHK"):
        code = s.split(".", 1)[0]
        return f"{normalize_hk_code(code)}.SEHK"
    if "." in s:
        return s
    for code, exch, _ in STOCK_LIST:
        if code == s:
            return f"{code}.{exch}"
    if s.isdigit() and len(s) <= 5:
        hk = normalize_hk_code(s)
        for code, exch, _ in HK_STOCK_LIST:
            if code == hk:
                return f"{code}.{exch}"
    raise ValueError(
        f"未在股票池找到代码: {raw}（A股如 600487；港股如 01024 或 01024.SEHK）"
    )


def load_daily_series(lab: AlphaLab, vt_symbol: str, calendar_days: int = 120) -> list[dict]:
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=calendar_days)
    bars = lab.load_bar_data(vt_symbol, Interval.DAILY, start_dt, end_dt)
    rows: list[dict] = []
    for bar in bars:
        if bar.close_price and bar.close_price > 0:
            rows.append(
                {
                    "date": bar.datetime.strftime("%Y-%m-%d"),
                    "open": bar.open_price,
                    "high": bar.high_price,
                    "low": bar.low_price,
                    "close": bar.close_price,
                }
            )
    return rows


def is_n_day_breakdown(closes: list[float], n: int = BREAKOUT_WINDOW) -> bool:
    if len(closes) < n:
        return False
    window = closes[-n:]
    return window[-1] <= min(window) + 1e-12


def ma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def _fmt_rank(rank: int | None) -> str:
    return f"#{rank}" if rank is not None else "不在信号池"


def load_ml_context(lab: AlphaLab, vt_symbol: str) -> dict:
    """读取最新 ML 预测信号，判断是否在 Top5。"""
    signal_dir = Path(ALPHA_LAB_PATH) / "signal"
    files = sorted(signal_dir.glob("lgb_pred_*.parquet"))
    if not files:
        return {
            "available": False,
            "reason": "无预测信号文件，请先运行 predict_daily.py",
        }

    latest = files[-1]
    df = pl.read_parquet(latest).sort("predicted_return", descending=True)
    top5 = df.head(LIVERMORE_TOP_N)
    in_top5 = vt_symbol in top5["vt_symbol"].to_list()

    rank: int | None = None
    predicted_return: float | None = None
    for i, row in enumerate(df.iter_rows(named=True), 1):
        if row["vt_symbol"] == vt_symbol:
            rank = i
            predicted_return = row.get("predicted_return")
            break

    signal_date = latest.stem.replace("lgb_pred_", "")
    return {
        "available": True,
        "signal_date": signal_date,
        "signal_file": latest.name,
        "rank": rank,
        "in_top5": in_top5,
        "predicted_return": predicted_return,
        "top5_list": top5["vt_symbol"].to_list(),
    }


def estimate_stock_budget(book: dict, price: float, vt_symbol: str) -> float:
    equity = float(book.get("cash", 0) or 0)
    for sym, pos in (book.get("positions") or {}).items():
        shares = int(pos.get("shares", 0) or 0)
        if shares <= 0:
            continue
        if sym == vt_symbol:
            equity += shares * price
        else:
            # 其他持仓用成本粗估（本脚本未拉全市场行情）
            cost = float(pos.get("cost", 0) or 0)
            equity += shares * cost
    return equity / LIVERMORE_TOP_N if equity > 0 else 0.0


def build_trade_advice(
    r: dict,
    ml: dict,
    book: dict,
    held_hint: dict | None,
    *,
    is_hk: bool = False,
) -> dict:
    """生成建仓/持仓/清仓建议及说明。"""
    sig = r["signals"]
    pos = r["position"]
    price = r["price"]
    has_pos = bool(pos and int(pos.get("shares", 0) or 0) > 0)
    in_top5 = bool(ml.get("in_top5"))
    reasons: list[str] = []
    detail = ""

    stock_budget = estimate_stock_budget(book, price, r["vt_symbol"])

    if is_hk and not has_pos:
        if sig["breakdown_20d"]:
            action = "观望"
            reasons.append("创 20 日新低，趋势偏弱")
            detail = "港股仅趋势参考，不参与 predict_daily / 利弗莫尔建仓"
        elif sig["breakout_20d"]:
            action = "关注"
            reasons.append("创 20 日新高，短期趋势转强")
            reasons.append("港股未纳入 ML Top5 选股池")
            detail = "可跟踪趋势，但不走 A 股利弗莫尔自动建仓流程"
        else:
            action = "观望"
            reasons.append(f"未突破 20 日高点（需收盘 ≥ {r['high_20d']:.2f}）")
            detail = "港股仅趋势参考"
        if sig["below_ma20"]:
            reasons.append("收盘在 MA20 下方，动能偏弱")
        elif sig["above_ma20"]:
            reasons.append("收盘在 MA20 上方")
        return {"action": action, "reasons": reasons, "detail": detail}

    if has_pos:
        held = sig.get("held_action") or "持有"
        shares = int(pos.get("shares", 0) or 0)
        cost = float(pos.get("cost", 0) or 0)
        pnl = (price - cost) / cost if cost > 0 else 0.0

        if held in ("止损清仓", "清仓"):
            action = "清仓"
            reasons.append(f"管仓规则触发【{held}】：{sig.get('held_note', '')}")
            reasons.append(f"相对成本 {pnl * 100:+.1f}%，已达退出条件")
            detail = "建议次日/盘后定价卖出至 0 股"
        elif held == "减半":
            action = "减半"
            half = held_hint.get("shares", shares // 2) if held_hint else shares // 2
            reasons.append(sig.get("held_note", ""))
            reasons.append(
                f"移动止盈：先涨 {TRAIL_ACTIVATE_PCT:.0%} 后回撤 {TRAIL_PULLBACK_PCT:.0%} → 先卖一半"
            )
            detail = f"建议减至约 {half} 股（当前 {shares} 股）"
        elif held == "加仓":
            action = "加仓"
            stage = held_hint.get("stage", 2) if held_hint else 2
            tgt = suggest_target_shares(stock_budget, stage, price)
            add = max(0, tgt - shares)
            reasons.append(sig.get("held_note", ""))
            reasons.append(
                f"相对上次买入涨 ≥{ADD_SPACING_PCT:.0%} 且仍在 Top{LIVERMORE_TOP_N}"
                if in_top5
                else f"价格达加仓间距，但已掉出 Top{LIVERMORE_TOP_N}，策略不加仓"
            )
            if in_top5:
                detail = f"建议加约 {add} 股至约 {tgt} 股（单股预算约 {stock_budget:,.0f} 元）"
            else:
                action = "持仓"
                detail = "不加仓，继续按止损/早期盈利区/移动止盈持有"
        else:
            action = "持仓"
            reasons.append(sig.get("held_note", f"成本盈亏 {pnl * 100:+.1f}%"))
            if not in_top5:
                rank_s = _fmt_rank(ml.get("rank"))
                reasons.append(
                    f"已掉出 ML Top{LIVERMORE_TOP_N}（排名 {rank_s}），"
                    "策略不强制卖出，仍看止损/早期盈利区/移动止盈"
                )
            else:
                reasons.append(f"仍在 ML Top{LIVERMORE_TOP_N}（排名 #{ml.get('rank')}）")
            if sig["breakdown_20d"]:
                reasons.append("结构转弱：创20日新低，密切关注止损位")
            elif not sig["breakout_20d"]:
                reasons.append("未创20日新高，趋势动能一般，不加仓")
            detail = f"继续持有 {shares} 股，止损线约 {cost * (1 - STOP_LOSS_PCT):.2f}"

    else:
        if not ml.get("available"):
            action = "观望"
            reasons.append(ml.get("reason", "缺少 ML 信号"))
        elif not in_top5:
            action = "观望"
            rank = ml.get("rank")
            if rank:
                reasons.append(
                    f"ML 排名 #{rank}/{ml.get('total', '?')}，未进 Top{LIVERMORE_TOP_N}，不符合建仓候选"
                )
            else:
                reasons.append("不在最新预测信号中")
            reasons.append("利弗莫尔只在 Top5 内寻找突破建仓机会")
            detail = "不建仓，等待进入 Top5 且突破"
        elif sig["breakout_20d"]:
            action = "建仓"
            tgt = suggest_target_shares(stock_budget, 1, price)
            reasons.append(f"在 ML Top{LIVERMORE_TOP_N}（排名 #{ml.get('rank')}）")
            reasons.append(f"收盘创 {BREAKOUT_WINDOW} 日新高，满足突破建仓")
            if ml.get("predicted_return") is not None:
                reasons.append(f"3日预期收益约 {ml['predicted_return'] * 100:+.2f}%")
            detail = (
                f"建议首仓约 {tgt} 股（预算 60% 档，单股预算约 {stock_budget:,.0f} 元，整手100）"
            )
        else:
            action = "观望"
            reasons.append(f"在 ML Top{LIVERMORE_TOP_N}（排名 #{ml.get('rank')}），但尚未突破20日高点")
            reasons.append(f"需收盘 ≥ {r['high_20d']:.2f} 才触发建仓")
            if sig["breakdown_20d"]:
                reasons.append("同时出现20日新低，趋势偏弱，不宜抢先建仓")
            detail = "等待突破后再建仓"

    return {
        "action": action,
        "reasons": reasons,
        "detail": detail,
    }


def trend_verdict(signals: dict) -> tuple[str, str]:
    """返回 (评级, 一句话)"""
    score = 0
    if signals["breakout_20d"]:
        score += 2
    if signals["above_ma20"]:
        score += 1
    if signals["breakdown_20d"]:
        score -= 2
    if signals["below_ma20"]:
        score -= 1
    if signals.get("stop_hit"):
        score -= 3
    if signals.get("trail_exit"):
        score -= 2

    if score >= 2:
        return "偏强", "仍在上升趋势特征内（突破/均线之上）"
    if score <= -2:
        return "转弱", "出现跌破或管仓退出信号，趋势可能已变"
    return "震荡", "多空信号混杂，趋势方向不明确"


def analyze(
    lab: AlphaLab,
    vt_symbol: str,
    price_override: float | None = None,
    *,
    account: str | None = None,
) -> dict:
    series = load_daily_series(lab, vt_symbol)
    if len(series) < BREAKOUT_WINDOW:
        raise RuntimeError(
            f"日线不足 {BREAKOUT_WINDOW} 根，请先执行:\n"
            f"  .venv/bin/python examples/alpha_a_share/check_trend.py "
            f"{vt_symbol.split('.')[0]} --update\n"
            "若下载失败，检查系统代理（可临时 unset HTTP_PROXY HTTPS_PROXY）"
        )

    last = series[-1]
    bar_date = last["date"]
    bar_close = float(last["close"])
    bar_high = float(last["high"])

    closes = [r["close"] for r in series]
    lows = [r["low"] for r in series]

    # 分析价：默认用最新日线收盘；可选手填盘中价
    if price_override is not None:
        price = float(price_override)
        price_source = f"手动输入（基于 {bar_date} 日线结构判断）"
    else:
        price = bar_close
        price_source = f"日线收盘（{bar_date}）"

    closes_for_signal = closes[:-1] + [price]
    ma20 = ma(closes_for_signal, 20)

    book = load_positions(account=account)
    pos = (book.get("positions") or {}).get(vt_symbol)
    is_hk = is_hk_vt_symbol(vt_symbol)
    ml: dict = {"available": False}
    if not is_hk:
        ml = load_ml_context(lab, vt_symbol)
        if ml.get("available"):
            signal_df = pl.read_parquet(Path(ALPHA_LAB_PATH) / "signal" / ml["signal_file"])
            ml["total"] = len(signal_df)
    else:
        ml = {"available": False, "reason": "港股不参与 ML 预测"}

    held_hint: dict = {}
    in_top5 = bool(ml.get("in_top5"))

    signals = {
        "breakout_20d": is_n_day_breakout(closes_for_signal, BREAKOUT_WINDOW),
        "breakdown_20d": is_n_day_breakdown(closes_for_signal, BREAKOUT_WINDOW),
        "above_ma20": ma20 is not None and price >= ma20,
        "below_ma20": ma20 is not None and price < ma20,
        "stop_hit": False,
        "trail_exit": False,
        "held_action": None,
        "held_note": None,
    }

    if pos and int(pos.get("shares", 0) or 0) > 0:
        action, note, hint = evaluate_held_action(pos, price, bar_high, in_top5)
        signals["held_action"] = action
        signals["held_note"] = note
        held_hint = hint
        signals["stop_hit"] = action == "止损清仓"
        signals["trail_exit"] = action in ("减半", "清仓")

    rating, summary = trend_verdict(signals)
    trade = build_trade_advice(
        {
            "signals": signals,
            "position": pos,
            "price": price,
            "vt_symbol": vt_symbol,
            "high_20d": max(closes[-BREAKOUT_WINDOW:]),
        },
        ml,
        book,
        held_hint,
        is_hk=is_hk,
    )

    low_20 = min(lows[-BREAKOUT_WINDOW:])
    high_20 = max(closes[-BREAKOUT_WINDOW:])

    return {
        "vt_symbol": vt_symbol,
        "name": stock_name(vt_symbol),
        "bar_date": bar_date,
        "price": price,
        "price_source": price_source,
        "ma20": ma20,
        "high_20d": high_20,
        "low_20d": low_20,
        "signals": signals,
        "ml": ml,
        "trade": trade,
        "rating": rating,
        "summary": summary,
        "position": pos,
        "is_hk": is_hk,
    }


def print_report(r: dict) -> None:
    sig = r["signals"]
    ml = r.get("ml") or {}
    trade = r.get("trade") or {}

    print("=" * 72)
    print(f"  趋势检查  {r['vt_symbol']}  {r['name']}")
    print("=" * 72)
    print(f"  分析价: {r['price']:.3f}  ({r['price_source']})")
    if r["ma20"] is not None:
        print(f"  MA20:   {r['ma20']:.3f}  |  20日高/低: {r['high_20d']:.3f} / {r['low_20d']:.3f}")
    if ml.get("available"):
        ret_s = (
            f"{ml['predicted_return'] * 100:+.2f}%"
            if ml.get("predicted_return") is not None
            else "n/a"
        )
        print(
            f"  ML信号: {ml['signal_file']}  排名={_fmt_rank(ml.get('rank'))}  "
            f"Top{LIVERMORE_TOP_N}={'是' if ml.get('in_top5') else '否'}  3日预期={ret_s}"
        )
    elif r.get("is_hk"):
        print("  ML信号: 港股不参与 predict_daily / Top5 选股")
    else:
        print("  ML信号: 无（请先运行 predict_daily.py）")
    print()
    print("  结构信号:")
    print(f"    20日突破(新高): {'是' if sig['breakout_20d'] else '否'}")
    print(f"    20日跌破(新低): {'是' if sig['breakdown_20d'] else '否'}")
    print(f"    收盘在MA20上:   {'是' if sig['above_ma20'] else '否'}")
    print()
    print(f"  ▶ 操作建议: 【{trade.get('action', '—')}】")
    if trade.get("detail"):
        print(f"     {trade['detail']}")
    print("  说明:")
    for reason in trade.get("reasons") or ["（无）"]:
        print(f"    · {reason}")
    print()
    print(f"  趋势评级: 【{r['rating']}】  {r['summary']}")
    print()
    print("  规则摘要:")
    print(f"    建仓 = Top{LIVERMORE_TOP_N} + {BREAKOUT_WINDOW}日突破 | 加仓 = 涨{ADD_SPACING_PCT:.0%}且仍在Top5")
    print(
        f"    止损 = -{STOP_LOSS_PCT:.0%} | "
        f"早期区 +{EARLY_ZONE_ARM_PCT:.0%}~+{TRAIL_ACTIVATE_PCT:.0%} 吐回利润{EARLY_ZONE_GIVEBACK_PCT:.0%}清 | "
        f"移动止盈 = +{TRAIL_ACTIVATE_PCT:.0%}后回撤{TRAIL_PULLBACK_PCT:.0%}减半/清"
    )
    print()
    print("  提示: 默认用本地最新日线收盘；盘中请加 --price；收盘后 --update 再查")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="单票趋势检查（日线）")
    parser.add_argument("symbol", help="股票代码，如 600487 或 600487.SSE")
    parser.add_argument("--price", type=float, default=None, help="手动现价（盘中临时判断）")
    parser.add_argument("--update", action="store_true", help="检查前先更新日线数据")
    parser.add_argument("--account", default=None, help="持仓账户名（独立账本）")
    args = parser.parse_args()

    vt_symbol = normalize_vt_symbol(args.symbol)
    lab = AlphaLab(ALPHA_LAB_PATH)

    if args.update:
        code, exch, name = stock_entry(vt_symbol)
        print("更新日线...")
        download_daily_data(lab, [(code, exch, name)])

    report = analyze(lab, vt_symbol, price_override=args.price, account=args.account)
    print_report(report)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n✗ {exc}")
        raise SystemExit(1) from exc
