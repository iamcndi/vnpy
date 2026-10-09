"""
每日预测：加载 1/3/5 日持有期 LightGBM 模型 → 获取最新数据 → 输出多持有期选股信号

用法：每天收盘后执行：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/predict_daily.py

输出：控制台打印 Top5/Bottom5（含 1日/3日/5日预期）
      + 利弗莫尔建仓 / 管仓动作（读 alpha_data/livermore_positions.json）
      alpha_data/signal/ 下保存预测信号

持仓：
  - 当前快照：alpha_data/livermore_positions.json（每日预测读取）
  - 历史流水：alpha_data/livermore_positions_history.jsonl（复盘用，只追加不覆盖）
  更新后运行 snapshot_positions.py 记入历史；参考 livermore_positions.example.json
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.dataset import Segment
from vnpy.trader.constant import Interval

from a_share_limits import a_share_daily_limit_up_pct, is_a_share_limit_up
from datafeed import download_daily_data
from stock_universe import STOCK_LIST
from livermore_positions_store import (
    all_prediction_accounts,
    load_positions,
    resolve_paths,
    save_positions,
)
from horizons import (
    FORECAST_HORIZONS,
    PRIMARY_HORIZON,
    model_name,
    LEGACY_MODEL_NAME,
)


ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")
LOOKBACK_DAYS = 90
BASE_LONG_TOP_N_RATIO = 1.0 / 3.0
OPTIMIZED_CONFIG_PATH = os.path.join(ALPHA_LAB_PATH, "optimized_config.json")

PREDICTION_WINDOW_DAYS = 30
DISPLAY_TOP_N = 5
DISPLAY_BOTTOM_N = 5

# 与 backtest_1y_livermore.py（4万 / Top5）对齐
LIVERMORE_TOP_N = 5
BREAKOUT_WINDOW = 20
ADD_SPACING_PCT = 0.10
PYRAMID_FRACTIONS = (0.60, 0.40)
STOP_LOSS_PCT = 0.07
TRAIL_ACTIVATE_PCT = 0.15
TRAIL_PULLBACK_PCT = 0.08
# 早期盈利区：峰值浮盈 ∈ [+3%, +15%) 时，吐回峰值利润 70% → 清仓（一年回测最优夏普对照）
EARLY_ZONE_ARM_PCT = 0.03
EARLY_ZONE_GIVEBACK_PCT = 0.70
MIN_LOT = 100
# 盘中追高上限：触发价之上最多追 1%（开仓 / 加仓共用）
INTRADAY_MAX_CHASE_PCT = 0.01


def _stock_name(vt_symbol: str) -> str:
    for code, exch, name in STOCK_LIST:
        if f"{code}.{exch}" == vt_symbol:
            return name
    return ""


def _fmt_ret(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "   n/a"
    return f"{value * 100:>+6.2f}%"


def is_n_day_breakout(closes: list[float], n: int = BREAKOUT_WINDOW) -> bool:
    """最新收盘是否创近 n 日（含当日）新高；与利弗莫尔回测一致。"""
    if len(closes) < n:
        return False
    window = closes[-n:]
    return window[-1] >= max(window) - 1e-12


def breakout_trigger_price(closes: list[float], n: int = BREAKOUT_WINDOW) -> float | None:
    """
    盘中开仓触发价：现价需 ≥ 该价，才使「以现价替换最新一根收盘」后创近 n 日新高。
    无分钟线时仍用日线序列；现价由日线收盘或盘中 --price 提供。
    """
    if len(closes) < 2:
        return None
    if len(closes) < n:
        return max(closes[:-1])
    return max(closes[-n:-1])


def load_recent_closes(
    lab: AlphaLab,
    vt_symbol: str,
    end_dt: datetime,
    n: int = BREAKOUT_WINDOW,
) -> list[float]:
    """加载截止 end_dt 的最近 n 根日线收盘价（原始价）。"""
    start_dt = end_dt - timedelta(days=n * 3 + 30)
    bars = lab.load_bar_data(vt_symbol, Interval.DAILY, start_dt, end_dt)
    closes = [b.close_price for b in bars if b.close_price and b.close_price > 0]
    return closes[-n:] if len(closes) >= n else closes


def load_latest_bar_prices(
    lab: AlphaLab, vt_symbol: str, end_dt: datetime
) -> tuple[float | None, float | None]:
    """返回 (close, high)；无数据则为 (None, None)。"""
    start_dt = end_dt - timedelta(days=10)
    bars = lab.load_bar_data(vt_symbol, Interval.DAILY, start_dt, end_dt)
    if not bars:
        return None, None
    bar = bars[-1]
    close = bar.close_price if bar.close_price and bar.close_price > 0 else None
    high = bar.high_price if bar.high_price and bar.high_price > 0 else close
    return close, high


def load_latest_quote(
    lab: AlphaLab, vt_symbol: str
) -> tuple[datetime | None, float | None, datetime | None]:
    """本地日线最后一根: (K线日期, 收盘价, parquet 文件更新时间)。"""
    path = lab.daily_path / f"{vt_symbol}.parquet"
    if not path.exists():
        return None, None, None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    df = pl.read_parquet(path, columns=["datetime", "close"])
    if df.is_empty():
        return None, None, mtime
    last = df.tail(1).row(0, named=True)
    close = last["close"]
    if close is None or (isinstance(close, float) and (close <= 0 or np.isnan(close))):
        close = None
    return last["datetime"], close, mtime


def _fmt_quote(
    bar_dt: datetime | None, close: float | None, mtime: datetime | None
) -> tuple[str, str]:
    px = f"{close:.2f}" if close is not None else "n/a"
    if bar_dt is None and mtime is None:
        return px, "n/a"
    if bar_dt is None:
        return px, mtime.strftime("%Y-%m-%d %H:%M")
    bar_d = bar_dt.strftime("%Y-%m-%d")
    if mtime is None:
        return px, bar_d
    if mtime.date() == bar_dt.date():
        return px, f"{bar_d} {mtime.strftime('%H:%M')}"
    return px, f"{bar_d} (拉取 {mtime.strftime('%m-%d %H:%M')})"


def round_lot(shares: float | int, lot: int = MIN_LOT) -> int:
    s = int(shares)
    return s - (s % lot) if s >= lot else 0


def suggest_target_shares(stock_budget: float, stage: int, price: float) -> int:
    if stage <= 0 or price <= 0 or stock_budget <= 0:
        return 0
    frac = sum(PYRAMID_FRACTIONS[:stage])
    return round_lot(stock_budget * frac / price)


def evaluate_held_action(
    pos: dict,
    price: float,
    bar_high: float | None,
    in_candidates: bool,
) -> tuple[str, str, dict]:
    """根据单票持仓与今日价给出管仓动作。返回 (动作, 说明, 建议更新字段)。"""
    shares = int(pos.get("shares", 0) or 0)
    cost = float(pos.get("cost", 0) or 0)
    high = float(pos.get("high", 0) or 0)
    last_buy = float(pos.get("last_buy", cost) or cost)
    stage = int(pos.get("stage", 1) or 1)
    halved = bool(pos.get("halved", False))

    if shares <= 0 or cost <= 0 or price <= 0:
        return "跳过", "持仓字段不完整", {}

    peak = max(high, price, bar_high or 0.0)
    hint = {"high": peak}

    pnl = (price - cost) / cost
    if pnl <= -STOP_LOSS_PCT:
        return (
            "止损清仓",
            f"相对成本 {pnl * 100:+.1f}% ≤ -{STOP_LOSS_PCT * 100:.0f}%",
            {"shares": 0, "halved": False, "stage": 0},
        )

    peak_pnl = (peak - cost) / cost
    # 早期盈利区 (+3%~+15%)：吐回峰值利润 70% → 清仓（未达移动止盈激活线）
    if EARLY_ZONE_ARM_PCT <= peak_pnl < TRAIL_ACTIVATE_PCT and peak > cost:
        given = (peak - price) / (peak - cost)
        if given >= EARLY_ZONE_GIVEBACK_PCT:
            return (
                "清仓",
                f"早期盈利区：峰值浮盈 {peak_pnl * 100:+.1f}% "
                f"∈ [{EARLY_ZONE_ARM_PCT * 100:.0f}%, {TRAIL_ACTIVATE_PCT * 100:.0f}%)，"
                f"吐回利润 {given * 100:.1f}% ≥ {EARLY_ZONE_GIVEBACK_PCT:.0%}",
                {"shares": 0, "halved": False, "high": peak},
            )

    trail_armed = peak_pnl >= TRAIL_ACTIVATE_PCT
    if trail_armed and peak > 0:
        pullback = (peak - price) / peak
        if pullback >= TRAIL_PULLBACK_PCT:
            if not halved:
                half = round_lot(shares // 2)
                if half <= 0:
                    return (
                        "清仓",
                        f"移动止盈回撤 {pullback * 100:.1f}% 且不足一手，直接清",
                        {"shares": 0, "halved": True},
                    )
                return (
                    "减半",
                    f"已激活移动止盈，从高点回撤 {pullback * 100:.1f}% "
                    f"≥ {TRAIL_PULLBACK_PCT * 100:.0f}%",
                    {"shares": half, "halved": True, "high": peak},
                )
            return (
                "清仓",
                f"已减半后仍回撤 {pullback * 100:.1f}%",
                {"shares": 0, "halved": True},
            )

    if (
        not halved
        and stage < len(PYRAMID_FRACTIONS)
        and in_candidates
        and last_buy > 0
        and price >= last_buy * (1.0 + ADD_SPACING_PCT)
    ):
        return (
            "加仓",
            f"相对上次买入再涨 ≥{ADD_SPACING_PCT * 100:.0f}% 且仍在 Top{LIVERMORE_TOP_N}",
            {"stage": stage + 1, "last_buy": price, "high": peak},
        )

    note = f"成本盈亏 {pnl * 100:+.1f}% | 档位={stage} | 减半={halved}"
    if not in_candidates:
        note += " | 已掉出Top5(不强制卖)"
    return "持有", note, hint


def compute_price_levels(
    pos: dict,
    price: float,
    *,
    in_candidates: bool,
) -> list[tuple[str, float, str]]:
    """
    管仓关键价位：(动作, 触发价, 说明)。
    顺序：止损 → 加仓 → 减仓（早期区 / 移动止盈）。
    """
    shares = int(pos.get("shares", 0) or 0)
    cost = float(pos.get("cost", 0) or 0)
    high = float(pos.get("high", 0) or 0)
    last_buy = float(pos.get("last_buy", cost) or cost)
    stage = int(pos.get("stage", 1) or 1)
    halved = bool(pos.get("halved", False))

    if shares <= 0 or cost <= 0 or price <= 0:
        return []

    peak = max(high, price)
    peak_pnl = (peak - cost) / cost
    levels: list[tuple[str, float, str]] = []

    stop_px = cost * (1.0 - STOP_LOSS_PCT)
    levels.append(("止损", stop_px, f"收盘≤{stop_px:.2f} 清仓"))

    if not halved and stage < len(PYRAMID_FRACTIONS) and last_buy > 0:
        add_px = last_buy * (1.0 + ADD_SPACING_PCT)
        top5 = f"且仍在Top{LIVERMORE_TOP_N}" if in_candidates else f"(已掉出Top{LIVERMORE_TOP_N}不加仓)"
        levels.append(("加仓", add_px, f"收盘≥{add_px:.2f} {top5}"))

    early_arm = cost * (1.0 + EARLY_ZONE_ARM_PCT)
    trail_arm = cost * (1.0 + TRAIL_ACTIVATE_PCT)
    if peak > 0:
        if peak_pnl >= TRAIL_ACTIVATE_PCT:
            reduce_px = peak * (1.0 - TRAIL_PULLBACK_PCT)
            action = "清仓" if halved else "减半"
            levels.append(
                (
                    action,
                    reduce_px,
                    f"移动止盈：高点{peak:.2f} 回撤{TRAIL_PULLBACK_PCT:.0%}→收盘≤{reduce_px:.2f}",
                )
            )
        elif peak_pnl >= EARLY_ZONE_ARM_PCT and peak > cost:
            reduce_px = peak - (peak - cost) * EARLY_ZONE_GIVEBACK_PCT
            levels.append(
                (
                    "清仓",
                    reduce_px,
                    f"早期区：峰值{peak_pnl * 100:+.1f}% 吐回利润{EARLY_ZONE_GIVEBACK_PCT:.0%}"
                    f"→收盘≤{reduce_px:.2f}",
                )
            )
        else:
            levels.append(("参考", early_arm, f"峰值≥{early_arm:.2f} 进入早期区监控"))
            levels.append(("参考", trail_arm, f"峰值≥{trail_arm:.2f} 激活移动止盈"))

    return levels


def format_price_levels(
    levels: list[tuple[str, float, str]],
    price: float,
) -> str:
    """格式化价位一行；已达/已触发的项前加 *。"""
    if not levels:
        return ""
    parts: list[str] = []
    for action, px, note in levels:
        triggered = False
        if action == "加仓" and price >= px - 1e-9:
            triggered = True
        elif action in ("止损", "减半", "清仓") and price <= px + 1e-9:
            triggered = True
        mark = "*" if triggered else ""
        parts.append(f"{mark}{action} {note}")
    return "  价位: " + " | ".join(parts)


def build_df(lab: AlphaLab, vt_symbols: list[str]) -> pl.DataFrame:
    """构建归一化的因子数据集"""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS + 365)

    records: list[dict] = []
    for vt_symbol in vt_symbols:
        bars = lab.load_bar_data(vt_symbol, Interval.DAILY, start_dt, end_dt)
        for bar in bars:
            records.append({
                "datetime": bar.datetime,
                "vt_symbol": vt_symbol,
                "open": bar.open_price, "high": bar.high_price,
                "low": bar.low_price, "close": bar.close_price,
                "volume": bar.volume, "turnover": bar.turnover,
                "open_interest": bar.open_interest or 0.0,
            })

    df = pl.DataFrame(records).sort(["datetime", "vt_symbol"])

    first_close = df.group_by("vt_symbol").agg(pl.col("close").first().alias("close_0"))
    df = df.join(first_close, on="vt_symbol")
    for col in ("open", "high", "low", "close"):
        df = df.with_columns((pl.col(col) / pl.col("close_0")).alias(col))
    df = df.with_columns(
        ((pl.col("turnover") / (pl.col("volume") + 1e-12)) / pl.col("close_0")).alias("vwap")
    )
    df = df.drop("close_0")

    numeric_cols = [c for c in df.columns if c not in ("datetime", "vt_symbol")]
    mask = df.select(pl.sum_horizontal(pl.col(c) for c in numeric_cols)) == 0
    df = df.with_columns(
        [pl.when(mask.to_series()).then(float("nan")).otherwise(pl.col(c)).alias(c)
         for c in numeric_cols]
    )
    return df


def load_optimized_config() -> dict:
    """加载优化配置（如果存在），否则返回默认值"""
    default = {
        "optimized_params": {
            "long_top_n_ratio": BASE_LONG_TOP_N_RATIO,
            "min_signal_threshold": 0.0,
            "confidence_scale": 1.0,
        },
        "model_status": {
            "retrain_needed": False,
        },
    }

    config_path = Path(OPTIMIZED_CONFIG_PATH)
    if not config_path.exists():
        return default

    try:
        with open(config_path) as f:
            config = json.load(f)
            for key in default:
                if key not in config:
                    config[key] = default[key]
            for key in default["optimized_params"]:
                if key not in config.get("optimized_params", {}):
                    config.setdefault("optimized_params", {})[key] = default["optimized_params"][key]

        print(f"  ✓ 加载优化配置 (选股比率={config['optimized_params']['long_top_n_ratio']:.2f}, "
              f"置信度={config['optimized_params']['confidence_scale']:.2f}x)")

        return config
    except Exception as e:
        print(f"  ⚠ 加载优化配置失败: {e}，使用默认参数")
        return default


def should_warn_retrain(config: dict) -> bool:
    """仅在优化配置判定需要重训、且并非刚训练当天时提示。"""
    if not config.get("model_status", {}).get("retrain_needed", False):
        return False
    days = (config.get("retraining_schedule") or {}).get("days_since_last_train")
    if days is not None and int(days) <= 0:
        return False
    return True


def print_retrain_warning(config: dict) -> None:
    if not should_warn_retrain(config):
        return
    schedule = config.get("retraining_schedule") or {}
    reason = schedule.get("retrain_reason") or "评估指标建议重训"
    days = schedule.get("days_since_last_train")
    print("\n" + "=" * 60)
    print("  ⚠ 模型建议重训练")
    if days is not None:
        print(f"  距上次训练: {days} 天")
    print(f"  原因: {reason}")
    print("  请运行: 菜单 2→1  或  examples/alpha_a_share/run_ml.py")
    print("=" * 60)


def load_horizon_models(lab: AlphaLab) -> dict[int, object]:
    """加载各持有期模型；若多持有期文件缺失则回退旧单模型作为 3 日"""
    models: dict[int, object] = {}

    for days in FORECAST_HORIZONS:
        model = lab.load_model(model_name(days))
        if model is not None and getattr(model, "model", None) is not None:
            models[days] = model
            print(f"  ✓ 加载模型 {model_name(days)}")

    if PRIMARY_HORIZON not in models:
        legacy = lab.load_model(LEGACY_MODEL_NAME)
        if legacy is not None and getattr(legacy, "model", None) is not None:
            models[PRIMARY_HORIZON] = legacy
            print(f"  ✓ 回退加载旧模型 {LEGACY_MODEL_NAME} → {PRIMARY_HORIZON}日")

    if PRIMARY_HORIZON not in models:
        print("✗ 未找到主持有期模型，请先运行 run_ml.py 训练")
        sys.exit(1)

    missing = [d for d in FORECAST_HORIZONS if d not in models]
    if missing:
        print(f"  ⚠ 缺少持有期模型 {missing} 日，对应列将显示 n/a；请运行 run_ml.py 完整训练")

    return models


def _print_rank_row(
    rank: int,
    row: dict,
    n_long: int,
    n_total: int,
    horizons: list[int],
    quote_cache: dict[str, tuple[datetime | None, float | None, datetime | None]],
) -> None:
    vt = row["vt_symbol"]
    sig = row["signal"]
    if rank <= n_long:
        action = "BUY  "
    elif rank > n_total - n_long:
        action = "SELL "
    else:
        action = "HOLD "

    cols = "  ".join(_fmt_ret(row.get(f"ret_{d}d")) for d in horizons)
    px, ts = _fmt_quote(*quote_cache.get(vt, (None, None, None)))
    print(
        f"  {rank:>4d}  {vt:>12s}  {_stock_name(vt):>8s}  {cols}  {sig:>6.3f}  {action}"
        f"  {px:>7s}  {ts}"
    )


def _account_label(account: str | None) -> str:
    return account or "默认"


def _print_livermore_for_account(
    *,
    account: str | None,
    book: dict,
    lab: AlphaLab,
    latest_date: datetime,
    top_candidates: pl.DataFrame,
    candidate_set: set[str],
    price_cache: dict[str, tuple[float | None, float | None]],
) -> None:
    """单账户利弗莫尔建仓 / 管仓输出。"""
    positions_path, history_path = resolve_paths(account)
    held: dict[str, dict] = {
        vt: dict(p)
        for vt, p in (book.get("positions") or {}).items()
        if int((p or {}).get("shares", 0) or 0) > 0
    }

    entry_rows: list[dict] = []
    limit_up_rows: list[dict] = []
    watch_rows: list[dict] = []
    for row in top_candidates.iter_rows(named=True):
        vt = row["vt_symbol"]
        # 多取 1 根，用于昨收判定涨停
        closes_ext = load_recent_closes(lab, vt, latest_date, BREAKOUT_WINDOW + 1)
        closes = (
            closes_ext[-BREAKOUT_WINDOW:]
            if len(closes_ext) >= BREAKOUT_WINDOW
            else closes_ext
        )
        breakout = is_n_day_breakout(closes, BREAKOUT_WINDOW)
        trigger = breakout_trigger_price(closes, BREAKOUT_WINDOW)
        close = closes[-1] if closes else None
        pre_close = closes_ext[-2] if len(closes_ext) >= 2 else None
        at_limit = bool(
            close is not None
            and pre_close is not None
            and is_a_share_limit_up(vt, close, pre_close)
        )
        enriched = {
            **row,
            "close": close,
            "breakout": breakout,
            "closes_n": len(closes),
            "entry_trigger": trigger,
            "limit_up": at_limit,
        }
        if vt in held:
            watch_rows.append({**enriched, "held": True})
        elif breakout and at_limit:
            limit_up_rows.append(enriched)
        elif breakout:
            entry_rows.append(enriched)
        else:
            watch_rows.append(enriched)

    equity = float(book.get("cash", 0) or 0)
    for vt, pos in held.items():
        px, _ = price_cache.get(vt, (None, None))
        if px:
            equity += int(pos.get("shares", 0) or 0) * px
    stock_budget = equity / LIVERMORE_TOP_N if equity > 0 else 0.0

    acct = _account_label(account)
    print(
        f"  利弗莫尔（Top{LIVERMORE_TOP_N} / N={BREAKOUT_WINDOW} / X={ADD_SPACING_PCT:.0%} "
        f"/ 金字塔={list(PYRAMID_FRACTIONS)}）— 账户: {acct}"
    )
    print(
        f"  持仓文件: {positions_path}"
        f"  | cash={book.get('cash', 0)}  持仓数={len(held)}  "
        f"权益约={equity:,.0f}  单股预算约={stock_budget:,.0f}"
    )
    print(f"  历史流水: {history_path}  （复盘用，见 snapshot_positions.py --list）")
    if not book.get("updated") and not held:
        print("  提示: 空仓模板已就绪，成交后按 example 填写 positions")

    print(f"  —— 突破建仓 ({len(entry_rows)} 只) ——")
    if entry_rows:
        for row in entry_rows:
            detail = " / ".join(
                f"{d}日 {_fmt_ret(row.get(f'ret_{d}d')).strip()}"
                for d in FORECAST_HORIZONS
            )
            close = row.get("close")
            close_s = f"{close:.2f}" if close is not None else "n/a"
            tgt = suggest_target_shares(stock_budget, 1, close) if close else 0
            print(
                f"    建仓  {row['vt_symbol']} ({_stock_name(row['vt_symbol'])})  "
                f"收盘={close_s}  建议约{tgt}股(60%档)  →  {detail}"
            )
    else:
        print("    （今日无新突破建仓）")

    print(f"  —— 涨停观望 / 暂不可买 ({len(limit_up_rows)} 只) ——")
    if limit_up_rows:
        print("     突破信号有效，但收盘涨停难成交，不建议追板；开板后再评估")
        for row in limit_up_rows:
            detail = " / ".join(
                f"{d}日 {_fmt_ret(row.get(f'ret_{d}d')).strip()}"
                for d in FORECAST_HORIZONS
            )
            close = row.get("close")
            close_s = f"{close:.2f}" if close is not None else "n/a"
            print(
                f"    涨停  {row['vt_symbol']} ({_stock_name(row['vt_symbol'])})  "
                f"收盘={close_s}  →  {detail}"
            )
    else:
        print("    （无）")

    watch_only = [r for r in watch_rows if not r.get("held")]
    print(f"  —— 候选观望 ({len(watch_only)} 只) ——")
    print(
        f"     盘中开仓: 现价≥20日高点触发；限价 [触发, 触发×{1 + INTRADAY_MAX_CHASE_PCT:.0%}]；"
        f"首仓约60%预算 | 无分钟线时现价=最新日线收盘，盘中请手填价"
    )
    for row in watch_only:
        detail = " / ".join(
            f"{d}日 {_fmt_ret(row.get(f'ret_{d}d')).strip()}"
            for d in FORECAST_HORIZONS
        )
        close = row.get("close")
        trigger = row.get("entry_trigger")
        print(f"    观望  {row['vt_symbol']} ({_stock_name(row['vt_symbol'])})  →  {detail}")
        if trigger is None or close is None or close <= 0:
            continue
        max_chase = trigger * (1.0 + INTRADAY_MAX_CHASE_PCT)
        # 建议股数按触发价估算（触及后按该价附近成交）
        tgt = suggest_target_shares(stock_budget, 1, trigger) if stock_budget > 0 else 0
        gap_pct = max(0.0, (trigger - close) / trigger * 100) if trigger > 0 else 0.0
        if close >= trigger - 1e-9:
            dist = "现价已触达，可按突破建仓"
        else:
            dist = f"现价={close:.2f} 还差{gap_pct:.1f}%"
        print(
            f"           └盘中开仓: ≥{trigger:.2f}  限价≤{max_chase:.2f}  "
            f"建议约{tgt}股(60%)  {dist}"
        )

    print(f"  —— 持仓管仓 ({len(held)} 只) ——")
    high_updates: list[tuple[str, float, float]] = []
    if held:
        for vt, pos in held.items():
            close, bar_high = price_cache.get(vt, (None, None))
            if close is None:
                print(f"    跳过  {vt} ({_stock_name(vt)})  无最新行情")
                continue
            action, note, hint = evaluate_held_action(
                pos, close, bar_high, vt in candidate_set
            )
            shares = int(pos.get("shares", 0) or 0)
            cost = float(pos.get("cost", 0) or 0)
            extra = ""
            if action == "加仓":
                tgt = suggest_target_shares(stock_budget, int(hint.get("stage", 2)), close)
                add_lots = max(0, tgt - shares)
                extra = f"  建议加约{add_lots}股至约{tgt}股"
            elif action == "减半":
                extra = f"  目标剩约{hint.get('shares', 0)}股"
            elif action in ("止损清仓", "清仓"):
                extra = "  目标0股"
            new_high = float(hint.get("high") or 0)
            old_high = float(pos.get("high", 0) or 0)
            high_raised = new_high > old_high + 1e-12
            if high_raised:
                pos["high"] = new_high
                stored = (book.get("positions") or {}).get(vt)
                if isinstance(stored, dict):
                    stored["high"] = new_high
                high_updates.append((vt, old_high, new_high))
            print(
                f"    {action}  {vt} ({_stock_name(vt)})  "
                f"{shares}股 成本={cost:.2f} 收盘={close:.2f}  | {note}{extra}"
            )
            price_line = format_price_levels(
                compute_price_levels(pos, close, in_candidates=vt in candidate_set),
                close,
            )
            if price_line:
                print(f"           └{price_line.lstrip()}")
            if high_raised:
                print(
                    f"           └ 已更新 {vt} high 记录 {old_high:.2f} → {new_high:.2f}"
                )
        if high_updates:
            names = "、".join(
                f"{vt} {old:.2f}→{new:.2f}" for vt, old, new in high_updates
            )
            save_positions(
                book,
                note=f"predict_daily 刷新 high: {names}",
                source="predict_daily",
                path=positions_path,
            )
    else:
        print("    （无持仓；有仓后写入 livermore_positions.json）")


def predict(lab: AlphaLab, vt_symbols: list[str], opt_config: dict, *, account: str | None = None) -> None:
    """预测并输出多持有期选股信号"""
    print("构建最新因子数据集...")
    df = build_df(lab, vt_symbols)
    print(f"  数据行数: {len(df)}, 日期: {df['datetime'].min()} ~ {df['datetime'].max()}")

    latest_date = df["datetime"].max()
    pred_start = (latest_date - timedelta(days=PREDICTION_WINDOW_DAYS)).strftime("%Y-%m-%d")
    pred_end = latest_date.strftime("%Y-%m-%d")

    print("计算因子特征...")
    dataset = Alpha158(
        df=df,
        train_period=(pred_start, pred_end),
        valid_period=(pred_start, pred_end),
        test_period=(pred_start, pred_end),
    )
    dataset.prepare_data(max_workers=4)

    models = load_horizon_models(lab)
    feat_df = dataset.fetch_raw(Segment.TRAIN)

    latest_feat = feat_df.filter(pl.col("datetime") == latest_date)
    if latest_feat.is_empty():
        print(f"✗ 最新日期 {latest_date} 没有因子数据")
        return

    result = latest_feat.select(["datetime", "vt_symbol"])

    for days, model_obj in models.items():
        feature_cols = model_obj.feature_cols
        missing_cols = [c for c in feature_cols if c not in latest_feat.columns]
        if missing_cols:
            print(f"  ⚠ {days}日模型缺少特征列 {len(missing_cols)} 个，跳过")
            continue
        X = latest_feat.select(feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        preds = model_obj.model.predict(X)
        result = result.with_columns(pl.Series(f"ret_{days}d", preds))

    primary_col = f"ret_{PRIMARY_HORIZON}d"
    if primary_col not in result.columns:
        print(f"✗ 缺少主持有期列 {primary_col}")
        return

    # 兼容旧评估脚本：predicted_return = 主持有期预期
    result = result.with_columns(pl.col(primary_col).alias("predicted_return"))

    optim = opt_config["optimized_params"]
    long_top_n_ratio = optim["long_top_n_ratio"]
    min_signal_threshold = optim.get("min_signal_threshold", 0.0)

    pred_min = result["predicted_return"].min()
    pred_max = result["predicted_return"].max()
    signal_range = pred_max - pred_min + 1e-12

    result = result.with_columns(
        ((pl.col("predicted_return") - pred_min) / signal_range).alias("signal")
    ).sort("predicted_return", descending=True)

    n_total = len(result)
    # 排名表仍用优化配置的 long_top_n_ratio 标 BUY/SELL；利弗莫尔建仓另用 Top5
    n_long = max(DISPLAY_TOP_N, int(n_total * long_top_n_ratio))
    n_long = min(n_long, max(1, n_total // 2))

    top_candidates = result.head(LIVERMORE_TOP_N)
    if min_signal_threshold > 0:
        filtered = top_candidates.filter(pl.col("signal") >= min_signal_threshold)
        if len(filtered) >= 1:
            top_candidates = filtered.head(LIVERMORE_TOP_N)
            print(
                f"  信号阈值过滤利弗莫尔候选: → {len(top_candidates)} 只 "
                f"(阈值={min_signal_threshold:.2f})"
            )

    sell_list = result.tail(DISPLAY_BOTTOM_N).sort(
        "predicted_return", descending=False
    )

    candidate_set = {row["vt_symbol"] for row in top_candidates.iter_rows(named=True)}
    accounts = [account] if account is not None else all_prediction_accounts()

    books: dict[str | None, dict] = {}
    symbols_for_prices = set(candidate_set)
    for acc in accounts:
        pos_path, _ = resolve_paths(acc)
        book = load_positions(path=pos_path)
        books[acc] = book
        for vt, p in (book.get("positions") or {}).items():
            if int((p or {}).get("shares", 0) or 0) > 0:
                symbols_for_prices.add(vt)

    price_cache: dict[str, tuple[float | None, float | None]] = {}
    for vt in symbols_for_prices:
        price_cache[vt] = load_latest_bar_prices(lab, vt, latest_date)

    rows = list(result.iter_rows(named=True))
    show_all = n_total <= DISPLAY_TOP_N + DISPLAY_BOTTOM_N
    display_rows = rows if show_all else rows[:DISPLAY_TOP_N] + rows[-DISPLAY_BOTTOM_N:]
    quote_symbols = {row["vt_symbol"] for row in display_rows}
    quote_symbols.update(row["vt_symbol"] for row in sell_list.iter_rows(named=True))
    quote_cache: dict[str, tuple[datetime | None, float | None, datetime | None]] = {}
    for vt in quote_symbols:
        quote_cache[vt] = load_latest_quote(lab, vt)

    horizon_headers = "  ".join(f"{d}日预期" for d in FORECAST_HORIZONS)
    print(f"\n{'='*108}")
    print(f"  多持有期选股信号 — {latest_date.strftime('%Y-%m-%d')}  (共 {n_total} 只)")
    print(f"  排序依据: {PRIMARY_HORIZON}日预期 | 标签=今日收盘→未来N日收盘累计收益")
    print("  最新价=本地日线最后收盘；更新时间=K线日期 + parquet 拉取时刻")
    print(f"{'='*108}")
    print(
        f"  {'排名':>4s}  {'股票':>12s}  {'名称':>8s}  {horizon_headers}  "
        f"{'信号':>6s}  {'操作':>6s}  {'最新价':>7s}  更新时间"
    )
    print(
        f"  {'-'*4}  {'-'*12}  {'-'*8}  {'  '.join(['-'*7]*len(FORECAST_HORIZONS))}  "
        f"{'-'*6}  {'-'*6}  {'-'*7}  {'-'*16}"
    )

    if show_all:
        for rank, row in enumerate(rows, 1):
            _print_rank_row(rank, row, n_long, n_total, FORECAST_HORIZONS, quote_cache)
    else:
        print(f"  —— 预测最高 Top {DISPLAY_TOP_N} ({PRIMARY_HORIZON}日) ——")
        for rank, row in enumerate(rows[:DISPLAY_TOP_N], 1):
            _print_rank_row(rank, row, n_long, n_total, FORECAST_HORIZONS, quote_cache)

        skipped = n_total - DISPLAY_TOP_N - DISPLAY_BOTTOM_N
        if skipped > 0:
            print(f"  ... ({skipped} 只中间排名省略) ...")

        print(f"  —— 预测最低 Bottom {DISPLAY_BOTTOM_N} ({PRIMARY_HORIZON}日) ——")
        for rank, row in enumerate(rows[-DISPLAY_BOTTOM_N:], n_total - DISPLAY_BOTTOM_N + 1):
            _print_rank_row(rank, row, n_long, n_total, FORECAST_HORIZONS, quote_cache)

    print(f"{'='*108}")
    if len(accounts) > 1:
        labels = "、".join(_account_label(a) for a in accounts)
        print(f"  管仓账户 ({len(accounts)} 个): {labels}")
        print(f"{'='*88}")

    for acc in accounts:
        if len(accounts) > 1:
            print(f"\n{'─'*88}")
            print(f"  【{_account_label(acc)}】")
            print(f"{'─'*88}")
        _print_livermore_for_account(
            account=acc,
            book=books[acc],
            lab=lab,
            latest_date=latest_date,
            top_candidates=top_candidates,
            candidate_set=candidate_set,
            price_cache=price_cache,
        )

    print(f"\n  建议回避 Bottom{DISPLAY_BOTTOM_N}（非利弗莫尔强制卖出）:")
    for row in sell_list.iter_rows(named=True):
        detail = " / ".join(
            f"{d}日 {_fmt_ret(row.get(f'ret_{d}d')).strip()}"
            for d in FORECAST_HORIZONS
        )
        px, ts = _fmt_quote(*quote_cache.get(row["vt_symbol"], (None, None, None)))
        print(
            f"    {row['vt_symbol']} ({_stock_name(row['vt_symbol'])})  "
            f"最新价={px}  {ts}  →  {detail}"
        )

    save_cols = ["datetime", "vt_symbol", "predicted_return", "signal"]
    for days in FORECAST_HORIZONS:
        col = f"ret_{days}d"
        if col in result.columns:
            save_cols.append(col)

    signal_save = result.select(save_cols)
    lab.save_signal(f"lgb_pred_{latest_date.strftime('%Y%m%d')}", signal_save)
    print("\n  信号已保存至 AlphaLab")
    print("\n  下单后更新持仓并记入历史：")
    for acc in accounts:
        acct_hint = f" --account {acc}" if acc else ""
        label = _account_label(acc)
        print(
            f"    [{label}] .venv/bin/python examples/alpha_a_share/snapshot_positions.py"
            f"{acct_hint} --note \"说明\""
        )
    print(f"{'='*88}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LightGBM 每日选股预测 + 利弗莫尔管仓")
    parser.add_argument(
        "--account",
        default=None,
        help="仅输出指定账户管仓（默认：全部账户统一列出）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  LightGBM 每日选股预测 (多持有期 + 利弗莫尔管仓)")
    if args.account:
        print(f"  账户: {args.account}")
    else:
        print("  账户: 全部")
    print("=" * 60)

    opt_config = load_optimized_config()

    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]

    print("检查并更新最新行情数据...")
    download_daily_data(lab, STOCK_LIST)

    predict(lab, vt_symbols, opt_config, account=args.account)
    print_retrain_warning(opt_config)


if __name__ == "__main__":
    main()
