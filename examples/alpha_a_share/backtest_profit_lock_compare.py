"""
盈利锁利规则对比：原利弗莫尔 vs +3%→回撤至1%清仓

  原策略在 +3%~+15% 区间不会卖，只有 -7% 止损 或 +15% 后回撤 8% 才退出。
  本脚本只跑最优 X=10% 的两版对比（比全网格快）。

用法:
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/backtest_profit_lock_compare.py
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from backtest_1y_livermore import (  # noqa: E402
    ALPHA_LAB_PATH,
    BACKTEST_END,
    BACKTEST_START,
    INITIAL_CAPITAL,
    LivermoreStrategy,
    PROFIT_LOCK_ARM_PCT,
    PROFIT_LOCK_EXIT_PCT,
    TRAIL_ACTIVATE_PCT,
    TRAIL_PULLBACK_PCT,
    build_dataset_df,
    download_daily_data,
    main as _unused,
    pick_stats,
    run_one,
    setup_contracts,
)
from stock_universe import STOCK_LIST  # noqa: E402
from vnpy.alpha import AlphaLab  # noqa: E402
from vnpy.alpha.dataset import Segment  # noqa: E402
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158  # noqa: E402
from vnpy.alpha.model.lgb_model import LGBAlphaModel  # noqa: E402
import polars as pl  # noqa: E402

LOOKBACK_DAYS = 90
TRAIN_START = "2023-01-01"
TRAIN_END = "2025-06-28"
BEST_X = 0.10


def _line(name: str, s: dict) -> None:
    print(
        f"  {name:28s}  {s.get('annual_return', float('nan')):>10.2f}  "
        f"{s.get('total_return', float('nan')):>10.2f}  "
        f"{s.get('sharpe_ratio', float('nan')):>8.2f}  "
        f"{s.get('max_ddpercent', float('nan')):>10.2f}  "
        f"{s.get('total_trade_count', 0):>8}"
    )


def main() -> None:
    print("=" * 90)
    print("  盈利锁利规则对比（原利弗莫尔 vs +3%→1% 清仓）")
    print(f"  回测期: {BACKTEST_START} ~ {BACKTEST_END}  资金={INITIAL_CAPITAL:,}  X={BEST_X:.0%}")
    print(f"  原规则: 止损 -7% | 移动止盈 +{TRAIL_ACTIVATE_PCT:.0%} 后回撤 {TRAIL_PULLBACK_PCT:.0%}")
    print(
        f"  新规则: 峰值浮盈 ≥{PROFIT_LOCK_ARM_PCT:.0%} 后，"
        f"回落到 ≤{PROFIT_LOCK_EXIT_PCT:.0%} → 清仓（+3%~+15% 区间也会卖）"
    )
    print("=" * 90)

    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]
    setup_contracts(lab)

    print("\n[1/3] 增量下载日线...")
    download_daily_data(lab, STOCK_LIST, TRAIN_START)

    print("\n[2/3] 训练 LightGBM...")
    df = build_dataset_df(lab, vt_symbols, TRAIN_START, BACKTEST_END, LOOKBACK_DAYS)
    dataset = Alpha158(
        df=df,
        train_period=(TRAIN_START, TRAIN_END),
        valid_period=(BACKTEST_START, BACKTEST_END),
        test_period=(BACKTEST_START, BACKTEST_END),
    )
    dataset.prepare_data(max_workers=4)
    model = LGBAlphaModel()
    model.fit(dataset)
    preds = model.predict(dataset, Segment.TEST)
    test_raw = dataset.fetch_raw(Segment.TEST)
    signal_df = test_raw.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series("signal", preds)
    )

    print("\n[3/3] 回测对比...")
    base_setting = {"add_spacing_pct": BEST_X}
    lock_setting = {
        "add_spacing_pct": BEST_X,
        "profit_lock_arm_pct": PROFIT_LOCK_ARM_PCT,
        "profit_lock_exit_pct": PROFIT_LOCK_EXIT_PCT,
    }
    stats_base, eng_base = run_one(lab, vt_symbols, signal_df, LivermoreStrategy, base_setting)
    stats_lock, eng_lock = run_one(lab, vt_symbols, signal_df, LivermoreStrategy, lock_setting)
    st_base = eng_base.strategy
    st_lock = eng_lock.strategy

    result = {
        "X": BEST_X,
        "profit_lock_arm": PROFIT_LOCK_ARM_PCT,
        "profit_lock_exit": PROFIT_LOCK_EXIT_PCT,
        "original": {
            **pick_stats(stats_base),
            "stop_count": st_base.stop_count,
            "half_count": st_base.half_count,
            "trail_clear_count": st_base.trail_clear_count,
        },
        "with_profit_lock": {
            **pick_stats(stats_lock),
            "stop_count": st_lock.stop_count,
            "half_count": st_lock.half_count,
            "trail_clear_count": st_lock.trail_clear_count,
            "profit_lock_count": st_lock.profit_lock_count,
        },
    }

    print("\n" + "=" * 90)
    print(f"  {'策略':28s}  {'年化%':>10s}  {'总收益%':>10s}  {'夏普':>8s}  {'最大回撤%':>10s}  {'成交':>8s}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    _line(f"原利弗莫尔 X={BEST_X:.0%}", stats_base)
    _line(
        f"+锁利 {PROFIT_LOCK_ARM_PCT:.0%}→{PROFIT_LOCK_EXIT_PCT:.0%}",
        stats_lock,
    )

    print("\n  退出事件统计:")
    print(
        f"    原策略: 止损={result['original']['stop_count']} "
        f"减半={result['original']['half_count']} "
        f"回撤清仓={result['original']['trail_clear_count']}"
    )
    print(
        f"    加锁利: 止损={result['with_profit_lock']['stop_count']} "
        f"减半={result['with_profit_lock']['half_count']} "
        f"回撤清仓={result['with_profit_lock']['trail_clear_count']} "
        f"锁利清仓={result['with_profit_lock']['profit_lock_count']}"
    )
    print(
        f"\n  Δ年化={result['with_profit_lock']['annual_return'] - result['original']['annual_return']:+.2f}%  "
        f"Δ夏普={result['with_profit_lock']['sharpe_ratio'] - result['original']['sharpe_ratio']:+.2f}  "
        f"Δ回撤={result['with_profit_lock']['max_ddpercent'] - result['original']['max_ddpercent']:+.2f}%"
    )

    out_path = os.path.join(ALPHA_LAB_PATH, "livermore_profit_lock_compare.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  结果: {out_path}")
    print("\n✓ 对比完成!")


if __name__ == "__main__":
    main()
