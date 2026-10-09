"""
盘中开仓 / 加仓扫描（无分钟线，默认用最新日线收盘价）。

开仓（Top5 未持仓）:
  - 触发：现价 ≥ 近 20 日收盘高点（不含最新一根时的前高）
  - 委托：限价 [触发价, 触发价×101%]，首仓约 60% 单股预算

加仓（已持仓 stage=1）:
  - 触发：现价 ≥ last_buy × 110% 且仍在 Top5
  - 委托：限价 [触发价, 触发价×101%]，加至第 2 档约 40%

用法：
  .venv/bin/python examples/alpha_a_share/intraday_add.py
  .venv/bin/python examples/alpha_a_share/intraday_add.py --price 600487:61.20
  .venv/bin/python examples/alpha_a_share/intraday_add.py --account 本人 --price 300274:88.5

说明：无实时/分钟行情；盘中请用 --price 传入券商最新价。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from vnpy.alpha import AlphaLab
from vnpy.trader.constant import Interval

from check_trend import normalize_vt_symbol, stock_name
from datafeed import download_daily_data
from livermore_positions_store import all_prediction_accounts, load_positions
from predict_daily import (
    ADD_SPACING_PCT,
    BREAKOUT_WINDOW,
    INTRADAY_MAX_CHASE_PCT,
    LIVERMORE_TOP_N,
    MIN_LOT,
    PYRAMID_FRACTIONS,
    STOP_LOSS_PCT,
    _account_label,
    breakout_trigger_price,
    evaluate_held_action,
    is_a_share_limit_up,
    is_n_day_breakout,
    load_latest_bar_prices,
    load_recent_closes,
    suggest_target_shares,
)
from stock_universe import STOCK_LIST

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")

# 距触发价 2% 以内视为「临近」
NEAR_TRIGGER_PCT = 0.02


def parse_price_spec(spec: str) -> tuple[str, float]:
    """解析 600487:61.2 / 600487.SSE=61.2 / 600487 61.2"""
    raw = spec.strip()
    for sep in (":", "=", "@"):
        if sep in raw:
            sym, px = raw.split(sep, 1)
            return normalize_vt_symbol(sym.strip()), float(px.strip())
    parts = raw.split()
    if len(parts) == 2:
        return normalize_vt_symbol(parts[0]), float(parts[1])
    raise ValueError(f"无法解析价格参数: {spec!r}（示例 600487:61.20）")


def load_top5_context(lab: AlphaLab) -> dict:
    """最新 ML Top5 与信号日期。"""
    signal_dir = Path(ALPHA_LAB_PATH) / "signal"
    files = sorted(signal_dir.glob("lgb_pred_*.parquet"))
    if not files:
        return {"available": False, "reason": "无预测信号，请先运行 predict_daily.py"}
    latest = files[-1]
    df = pl.read_parquet(latest).sort("predicted_return", descending=True)
    top5_df = df.head(LIVERMORE_TOP_N)
    top5_list = top5_df["vt_symbol"].to_list()
    return {
        "available": True,
        "signal_file": latest.name,
        "signal_date": latest.stem.replace("lgb_pred_", ""),
        "top5": set(top5_list),
        "top5_list": top5_list,
        "rank_map": {
            row["vt_symbol"]: i
            for i, row in enumerate(df.iter_rows(named=True), 1)
        },
    }


def estimate_equity(
    book: dict,
    vt_prices: dict[str, float],
    *,
    default_price: float | None = None,
    focus_vt: str | None = None,
) -> float:
    equity = float(book.get("cash", 0) or 0)
    for vt, pos in (book.get("positions") or {}).items():
        shares = int(pos.get("shares", 0) or 0)
        if shares <= 0:
            continue
        px = vt_prices.get(vt)
        if px is None and vt == focus_vt and default_price is not None:
            px = default_price
        if px is None:
            px = float(pos.get("cost", 0) or 0)
        equity += shares * px
    return equity


def classify_add_status(
    *,
    price: float,
    trigger: float,
    in_top5: bool,
    eligible: bool,
    stop_px: float,
) -> str:
    if not eligible:
        return "不可加"
    if price <= stop_px:
        return "止损区"
    if not in_top5:
        return "掉出Top5"
    if price >= trigger:
        return "可加仓"
    gap = (trigger - price) / trigger if trigger > 0 else 1.0
    if gap <= NEAR_TRIGGER_PCT:
        return "临近触发"
    return "等待触发"


def scan_position(
    *,
    vt: str,
    pos: dict,
    book: dict,
    top5: set[str],
    rank_map: dict[str, int],
    price: float | None,
    bar_high: float | None,
    lab: AlphaLab,
    end_dt: datetime,
    manual_prices: set[str],
) -> dict | None:
    shares = int(pos.get("shares", 0) or 0)
    if shares <= 0:
        return None

    cost = float(pos.get("cost", 0) or 0)
    last_buy = float(pos.get("last_buy", cost) or cost)
    stage = int(pos.get("stage", 1) or 1)
    halved = bool(pos.get("halved", False))

    if price is None:
        price, bar_high = load_latest_bar_prices(lab, vt, end_dt)
    if price is None or price <= 0 or cost <= 0 or last_buy <= 0:
        return None

    in_top5 = vt in top5
    eligible = not halved and stage < len(PYRAMID_FRACTIONS)
    trigger = last_buy * (1.0 + ADD_SPACING_PCT)
    max_chase = trigger * (1.0 + INTRADAY_MAX_CHASE_PCT)
    stop_px = cost * (1.0 - STOP_LOSS_PCT)

    stock_budget = estimate_equity(book, {vt: price}, default_price=price, focus_vt=vt)
    next_stage = stage + 1
    tgt_shares = suggest_target_shares(stock_budget, next_stage, price) if eligible else 0
    add_shares = max(0, tgt_shares - shares) if eligible else 0

    action, note, hint = evaluate_held_action(pos, price, bar_high, in_top5)
    status = classify_add_status(
        price=price,
        trigger=trigger,
        in_top5=in_top5,
        eligible=eligible,
        stop_px=stop_px,
    )

    gap_pct = (trigger - price) / trigger * 100 if trigger > 0 else 0.0
    if price >= trigger:
        gap_pct = 0.0

    return {
        "vt_symbol": vt,
        "name": stock_name(vt),
        "shares": shares,
        "cost": cost,
        "last_buy": last_buy,
        "stage": stage,
        "halved": halved,
        "price": price,
        "price_source": "手动" if vt in manual_prices else "日线收盘",
        "trigger": trigger,
        "max_chase": max_chase,
        "stop_px": stop_px,
        "in_top5": in_top5,
        "rank": rank_map.get(vt),
        "eligible": eligible,
        "status": status,
        "gap_pct": gap_pct,
        "add_shares": add_shares,
        "tgt_shares": tgt_shares,
        "stock_budget": stock_budget,
        "held_action": action,
        "held_note": note,
        "add_now": status == "可加仓" and add_shares >= MIN_LOT,
    }


def scan_accounts(
    lab: AlphaLab,
    accounts: list[str | None],
    price_overrides: dict[str, float],
    *,
    end_dt: datetime | None = None,
) -> list[dict]:
    end_dt = end_dt or datetime.now()
    ml = load_top5_context(lab)
    if not ml.get("available"):
        raise RuntimeError(ml.get("reason", "缺少 ML 信号"))

    top5 = ml["top5"]
    rank_map = ml["rank_map"]
    manual = set(price_overrides)

    rows: list[dict] = []
    for acc in accounts:
        book = load_positions(account=acc)
        for vt, pos in (book.get("positions") or {}).items():
            px = price_overrides.get(vt)
            bar_high = None
            if px is not None:
                _, bar_high = load_latest_bar_prices(lab, vt, end_dt)
            item = scan_position(
                vt=vt,
                pos=pos,
                book=book,
                top5=top5,
                rank_map=rank_map,
                price=px,
                bar_high=bar_high,
                lab=lab,
                end_dt=end_dt,
                manual_prices=manual,
            )
            if item is None:
                continue
            item["account"] = acc
            item["account_label"] = _account_label(acc)
            rows.append(item)

    rows.sort(
        key=lambda r: (
            {"可加仓": 0, "临近触发": 1, "等待触发": 2, "掉出Top5": 3, "止损区": 4, "不可加": 5}.get(
                r["status"], 9
            ),
            r.get("rank") or 999,
            r["vt_symbol"],
        )
    )
    return rows


def classify_entry_status(*, price: float, trigger: float) -> str:
    if price >= trigger - 1e-9:
        return "可开仓"
    gap = (trigger - price) / trigger if trigger > 0 else 1.0
    if gap <= NEAR_TRIGGER_PCT:
        return "临近触发"
    return "等待突破"


def scan_entry_accounts(
    lab: AlphaLab,
    accounts: list[str | None],
    price_overrides: dict[str, float],
    ml: dict,
    *,
    end_dt: datetime | None = None,
) -> list[dict]:
    """Top5 未持仓且未收盘突破 → 盘中开仓扫描。"""
    end_dt = end_dt or datetime.now()
    top5_list: list[str] = list(ml.get("top5_list") or [])
    rank_map: dict[str, int] = ml.get("rank_map") or {}
    manual = set(price_overrides)

    rows: list[dict] = []
    for acc in accounts:
        book = load_positions(account=acc)
        held = {
            vt
            for vt, p in (book.get("positions") or {}).items()
            if int((p or {}).get("shares", 0) or 0) > 0
        }
        px_map: dict[str, float] = dict(price_overrides)
        for vt in held:
            if vt not in px_map:
                px, _ = load_latest_bar_prices(lab, vt, end_dt)
                if px:
                    px_map[vt] = px
        equity = estimate_equity(book, px_map)
        stock_budget = equity / LIVERMORE_TOP_N if equity > 0 else 0.0

        for vt in top5_list:
            if vt in held:
                continue
            # 多取 1 根用于昨收判定涨停；已收盘突破仍给出盘中确认价
            closes_ext = load_recent_closes(lab, vt, end_dt, BREAKOUT_WINDOW + 1)
            closes = (
                closes_ext[-BREAKOUT_WINDOW:]
                if len(closes_ext) >= BREAKOUT_WINDOW
                else closes_ext
            )
            if len(closes) < 2:
                continue
            already = is_n_day_breakout(closes, BREAKOUT_WINDOW)
            trigger = breakout_trigger_price(closes, BREAKOUT_WINDOW)
            if trigger is None:
                continue
            price = price_overrides.get(vt)
            if price is None:
                price, _ = load_latest_bar_prices(lab, vt, end_dt)
            if price is None or price <= 0:
                continue

            pre_close = closes_ext[-2] if len(closes_ext) >= 2 else None
            # 手填盘中价时，昨收取最新一根日线收盘（closes_ext[-1]）
            if vt in manual and closes_ext:
                pre_close = closes_ext[-1]
            at_limit = bool(
                pre_close is not None and is_a_share_limit_up(vt, price, pre_close)
            )

            max_chase = trigger * (1.0 + INTRADAY_MAX_CHASE_PCT)
            limit_lo = trigger
            ref_px = trigger
            if price >= trigger - 1e-9:
                # 已触达/已突破：限价跟现价，避免挂单上限低于市价
                limit_lo = price
                ref_px = price
                max_chase = price * (1.0 + INTRADAY_MAX_CHASE_PCT)
            tgt = suggest_target_shares(stock_budget, 1, ref_px)
            if at_limit:
                status = "涨停暂不可买"
            elif already:
                status = "已收盘突破"
            else:
                status = classify_entry_status(price=price, trigger=trigger)
            gap_pct = max(0.0, (trigger - price) / trigger * 100) if trigger > 0 else 0.0
            if price >= trigger - 1e-9:
                gap_pct = 0.0

            rows.append(
                {
                    "account": acc,
                    "account_label": _account_label(acc),
                    "vt_symbol": vt,
                    "name": stock_name(vt),
                    "rank": rank_map.get(vt),
                    "price": price,
                    "price_source": "手动" if vt in manual else "日线收盘",
                    "trigger": trigger,
                    "limit_lo": limit_lo,
                    "max_chase": max_chase,
                    "tgt_shares": tgt,
                    "status": status,
                    "gap_pct": gap_pct,
                    "limit_up": at_limit,
                    "entry_now": status in ("可开仓", "已收盘突破") and tgt >= MIN_LOT,
                }
            )

    rows.sort(
        key=lambda r: (
            {
                "可开仓": 0,
                "已收盘突破": 1,
                "临近触发": 2,
                "等待突破": 3,
                "涨停暂不可买": 4,
            }.get(r["status"], 9),
            r.get("rank") or 999,
            r["account_label"],
            r["vt_symbol"],
        )
    )
    return rows


def print_entry_report(rows: list[dict], ml: dict) -> None:
    print("=" * 88)
    print("  盘中开仓扫描（Top5 + 20日突破 / 首仓 60%）")
    print("=" * 88)
    print(
        f"  ML 信号: {ml.get('signal_file', '—')}  "
        f"Top{LIVERMORE_TOP_N}: {', '.join(ml.get('top5_list') or [])}"
    )
    print(
        f"  规则: 现价≥近{BREAKOUT_WINDOW}日收盘高点开仓 | "
        f"限价 [触发, 触发×{1 + INTRADAY_MAX_CHASE_PCT:.0%}] | 首仓 {PYRAMID_FRACTIONS[0]:.0%} 预算"
    )
    print(
        "  价格说明: 无分钟/小时线；现价默认=最新日线收盘"
        "（盘前多为昨收，收盘后更新则为当日收盘）。盘中请 --price 传券商现价。"
    )
    print()

    if not rows:
        print("  （Top5 均已持仓，或无足够日线）")
        print("=" * 88)
        return

    buckets: dict[str, list] = {
        "可开仓": [],
        "已收盘突破": [],
        "临近触发": [],
        "等待突破": [],
        "涨停暂不可买": [],
    }
    for r in rows:
        buckets.setdefault(r["status"], []).append(r)

    for title, key in (
        ("★ 可立即开仓（现价已触达）", "可开仓"),
        ("★ 已收盘突破（可按突破建仓）", "已收盘突破"),
        ("◆ 临近触发（距触发价 ≤2%）", "临近触发"),
        ("○ 等待突破", "等待突破"),
        ("△ 涨停观望 / 暂不可买", "涨停暂不可买"),
    ):
        bucket = buckets.get(key) or []
        if not bucket:
            continue
        print(f"  —— {title} ({len(bucket)} 只) ——")
        for r in bucket:
            rank_s = f"#{r['rank']}" if r.get("rank") else "—"
            line = (
                f"    [{r['account_label']}] {r['vt_symbol']} ({r['name']})  "
                f"现价={r['price']:.2f}({r['price_source']})  "
                f"开仓≥{r['trigger']:.2f}  限价{r.get('limit_lo', r['trigger']):.2f}~{r['max_chase']:.2f}  "
                f"Top5={rank_s}"
            )
            if r["status"] == "涨停暂不可买":
                line += "  → 涨停难成交，不建议追板"
            elif r["status"] in ("可开仓", "已收盘突破", "临近触发") and r["tgt_shares"] >= MIN_LOT:
                line += f"  → 建议约 {r['tgt_shares']} 股(60%)"
            elif r["status"] == "等待突破":
                line += f"  → 还差 {r['gap_pct']:.1f}%"
            print(line)
        print()

    entry_now = [r for r in rows if r.get("entry_now")]
    if entry_now:
        print("  —— 盘中开仓委托清单 ——")
        for r in entry_now:
            print(
                f"    {r['vt_symbol']}  限价 {r.get('limit_lo', r['trigger']):.2f}~{r['max_chase']:.2f}  "
                f"数量约 {r['tgt_shares']} 股  账户={r['account_label']}"
            )
        print("  成交后: 写入 JSON → stage=1, cost/last_buy/high=成交价")
        print("          运行 snapshot_positions.py --account <名> --note \"盘中开仓\"")
    else:
        print("  （当前无「可开仓」；可挂触发价限价单，或 --price 传入现价再查）")

    print("=" * 88)


def print_report(rows: list[dict], ml: dict) -> None:
    print("=" * 88)
    print("  盘中加仓扫描（利弗莫尔金字塔第 2 档 / 40%）")
    print("=" * 88)
    print(
        f"  ML 信号: {ml.get('signal_file', '—')}  "
        f"Top{LIVERMORE_TOP_N}: {', '.join(ml.get('top5_list') or [])}"
    )
    print(
        f"  规则: 涨 {ADD_SPACING_PCT:.0%} 加仓 | "
        f"限价 [触发价, 触发价×{1 + INTRADAY_MAX_CHASE_PCT:.0%}] | 止损 -{STOP_LOSS_PCT:.0%}"
    )
    print(
        "  委托建议: 10:00 后挂单；价格触及触发价且仍在 Top5 时买至第 2 档；"
        "掉出 Top5 或进入止损区不加仓"
    )
    print()

    eligible_rows = [r for r in rows if r["eligible"]]
    skipped = [r for r in rows if not r["eligible"]]
    if not rows:
        print("  （无持仓）")
        print("=" * 88)
        return
    if not eligible_rows:
        print("  （持仓均已加满 stage=2 或已减半，无加仓档）")
        for r in skipped:
            print(
                f"    [{r['account_label']}] {r['vt_symbol']}  "
                f"stage={r['stage']} halved={r['halved']}  {r['shares']}股"
            )
        print("=" * 88)
        return

    buckets = {
        "可加仓": [],
        "临近触发": [],
        "等待触发": [],
        "掉出Top5": [],
        "止损区": [],
        "不可加": [],
    }
    for r in eligible_rows:
        buckets.setdefault(r["status"], []).append(r)

    for title, bucket in (
        ("★ 可立即加仓", buckets["可加仓"]),
        ("◆ 临近触发（距触发价 ≤2%）", buckets["临近触发"]),
        ("○ 等待触发", buckets["等待触发"]),
        ("× 掉出 Top5（不加仓）", buckets["掉出Top5"]),
        ("! 止损区（禁止加仓）", buckets["止损区"]),
    ):
        if not bucket:
            continue
        print(f"  —— {title} ({len(bucket)} 只) ——")
        for r in bucket:
            rank_s = f"#{r['rank']}" if r.get("rank") else "—"
            px_src = r["price_source"]
            line = (
                f"    [{r['account_label']}] {r['vt_symbol']} ({r['name']})  "
                f"{r['shares']}股 stage={r['stage']}  "
                f"现价={r['price']:.2f}({px_src})  触发≥{r['trigger']:.2f}  "
                f"限价≤{r['max_chase']:.2f}  Top5={rank_s}"
            )
            if r["status"] in ("可加仓", "临近触发") and r["add_shares"] >= MIN_LOT:
                line += f"  → 建议加 {r['add_shares']} 股至约 {r['tgt_shares']} 股"
            elif r["status"] == "等待触发":
                line += f"  → 还差 {r['gap_pct']:.1f}%"
            elif r["status"] == "掉出Top5":
                line += "  → 策略不加仓，持有管仓"
            print(line)
            if r["held_action"] in ("止损清仓", "清仓", "减半"):
                print(f"           └ 管仓: {r['held_action']} | {r['held_note']}")
        print()

    add_now = [r for r in rows if r.get("add_now")]
    if add_now:
        print("  —— 盘中委托清单 ——")
        for r in add_now:
            print(
                f"    {r['vt_symbol']}  限价 {r['trigger']:.2f}~{r['max_chase']:.2f}  "
                f"数量 {r['add_shares']} 股  账户={r['account_label']}"
            )
        print("  成交后: 更新 JSON → stage=2, last_buy=成交价, high=max(high,成交价)")
        print("          运行 snapshot_positions.py --account <名> --note \"盘中加仓\"")
    else:
        print("  （当前无「可加仓」标的；可挂触发价限价单等待，或用 --price 传入最新价再查）")

    if skipped:
        print(f"\n  —— 已满档 / 已减半 ({len(skipped)} 只，不加仓) ——")
        for r in skipped:
            print(
                f"    [{r['account_label']}] {r['vt_symbol']} ({r['name']})  "
                f"stage={r['stage']} halved={r['halved']}  {r['shares']}股"
            )

    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="盘中开仓/加仓扫描（全账户 / 单账户）")
    parser.add_argument("--account", default=None, help="仅扫描指定账户（默认全部）")
    parser.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="CODE:PX",
        help="盘中价，如 600487:61.20（可多次指定）",
    )
    parser.add_argument("--update", action="store_true", help="扫描前更新日线")
    args = parser.parse_args()

    price_overrides: dict[str, float] = {}
    for spec in args.price:
        try:
            vt, px = parse_price_spec(spec)
            price_overrides[vt] = px
        except (ValueError, TypeError) as e:
            print(f"价格参数错误: {e}")
            raise SystemExit(1) from e

    accounts = [args.account] if args.account else all_prediction_accounts()
    lab = AlphaLab(ALPHA_LAB_PATH)

    if args.update:
        print("更新日线...")
        download_daily_data(lab, STOCK_LIST)

    ml = load_top5_context(lab)
    if not ml.get("available"):
        print(f"✗ {ml.get('reason', '缺少 ML 信号')}")
        raise SystemExit(1)

    acct_label = _account_label(args.account) if args.account else f"全部({len(accounts)}个)"
    print(f"\n  扫描账户: {acct_label}")
    if price_overrides:
        print(f"  手动现价: {', '.join(f'{k}={v:.2f}' for k, v in price_overrides.items())}")
    else:
        print("  现价来源: 最新日线收盘（无分钟线；盘中请加 --price）")
    print()

    try:
        entry_rows = scan_entry_accounts(lab, accounts, price_overrides, ml)
        add_rows = scan_accounts(lab, accounts, price_overrides)
    except RuntimeError as e:
        print(f"✗ {e}")
        raise SystemExit(1) from e

    print_entry_report(entry_rows, ml)
    print()
    print_report(add_rows, ml)


if __name__ == "__main__":
    main()
