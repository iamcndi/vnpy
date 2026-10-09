"""
早期盈利区：按峰值利润吐回比例 α 清仓（近两年，α=60%/70%）

  训练: 2023-01-01 ~ 2024-09-03
  回测: 2024-09-04 ~ 2026-09-04
  峰值浮盈 ∈ [+3%, +15%) 时，吐回峰值利润的 α → 清仓；
  达 +15% 后仍由原有移动止盈接管。不改每日预测。

用法:
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/backtest_early_zone_giveback.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

import polars as pl  # noqa: E402
from vnpy.alpha import AlphaLab  # noqa: E402
from vnpy.alpha.dataset import Segment  # noqa: E402
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158  # noqa: E402
from vnpy.alpha.model.lgb_model import LGBAlphaModel  # noqa: E402

from backtest_1y_livermore import (  # noqa: E402
    ALPHA_LAB_PATH,
    EARLY_ZONE_ARM_PCT,
    INITIAL_CAPITAL,
    LivermoreStrategy,
    TRAIL_ACTIVATE_PCT,
    TRAIL_PULLBACK_PCT,
    build_dataset_df,
    download_daily_data,
    pick_stats,
    run_one,
    setup_contracts,
)
from stock_universe import STOCK_LIST  # noqa: E402

LOOKBACK_DAYS = 90
TRAIN_START = "2023-01-01"
TRAIN_END = "2024-09-03"
BACKTEST_START = "2024-09-04"
BACKTEST_END = "2026-09-04"
BEST_X = 0.10
GIVEBACK_GRID = [0.60, 0.70]


def _patch_run_dates() -> None:
    """让 run_one 使用近两年区间（不改动原脚本常量）。"""
    import backtest_1y_livermore as bm

    bm.BACKTEST_START = BACKTEST_START
    bm.BACKTEST_END = BACKTEST_END


def _line(name: str, s: dict, extra: str = "") -> None:
    print(
        f"  {name:32s}  {s.get('annual_return', float('nan')):>10.2f}  "
        f"{s.get('sharpe_ratio', float('nan')):>8.2f}  "
        f"{s.get('max_ddpercent', float('nan')):>10.2f}  "
        f"{s.get('total_trade_count', 0):>8}{extra}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="跳过日线增量下载")
    args = parser.parse_args()

    _patch_run_dates()

    print("=" * 96)
    print("  早期盈利区利润吐回 α（清仓价=高点-(高点-成本)×α）")
    print(f"  训练: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  回测: {BACKTEST_START} ~ {BACKTEST_END}  资金={INITIAL_CAPITAL:,}  X={BEST_X:.0%}")
    print(
        f"  区间: 峰值浮盈 [{EARLY_ZONE_ARM_PCT:.0%}, {TRAIL_ACTIVATE_PCT:.0%})  "
        f"| ≥{TRAIL_ACTIVATE_PCT:.0%} 仍用移动止盈回撤 {TRAIL_PULLBACK_PCT:.0%}"
    )
    print(f"  α: {[f'{x:.0%}' for x in GIVEBACK_GRID]}")
    print("=" * 96)

    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]
    setup_contracts(lab)

    if not args.skip_download:
        print("\n[1/3] 增量下载日线...")
        download_daily_data(lab, STOCK_LIST, TRAIN_START)
    else:
        print("\n[1/3] 跳过下载（--skip-download）")

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

    print("\n[3/3] 回测...")
    stats_base, eng_base = run_one(
        lab, vt_symbols, signal_df, LivermoreStrategy, {"add_spacing_pct": BEST_X}
    )
    st_base = eng_base.strategy

    rows: list[dict] = []
    for i, alpha in enumerate(GIVEBACK_GRID, 1):
        print(f"  ({i}/{len(GIVEBACK_GRID)}) α={alpha:.0%} ...", flush=True)
        setting = {
            "add_spacing_pct": BEST_X,
            "profit_lock_arm_pct": EARLY_ZONE_ARM_PCT,
            "early_zone_giveback_pct": alpha,
        }
        stats, engine = run_one(lab, vt_symbols, signal_df, LivermoreStrategy, setting)
        st = engine.strategy
        row = {
            "kind": "giveback",
            "giveback_pct": alpha,
            **pick_stats(stats),
            "early_zone_exit_count": getattr(st, "early_zone_exit_count", 0),
            "stop_count": st.stop_count,
            "half_count": st.half_count,
            "trail_clear_count": st.trail_clear_count,
            "delta_annual_vs_base": stats.get("annual_return", 0) - stats_base.get("annual_return", 0),
            "delta_sharpe_vs_base": stats.get("sharpe_ratio", 0) - stats_base.get("sharpe_ratio", 0),
            "delta_dd_vs_base": stats.get("max_ddpercent", 0) - stats_base.get("max_ddpercent", 0),
        }
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda r: r["sharpe_ratio"] if r["sharpe_ratio"] == r["sharpe_ratio"] else -999,
        reverse=True,
    )
    best = ranked[0]

    print("\n" + "=" * 96)
    print(f"  {'策略':32s}  {'年化%':>10s}  {'夏普':>8s}  {'最大回撤%':>10s}  {'成交':>8s}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    _line("原利弗莫尔（无早期区规则）", stats_base)
    for r in rows:
        _line(
            f"+早期区吐回 α={r['giveback_pct']:.0%}",
            r,
            f"  早期清={r['early_zone_exit_count']}",
        )

    print(
        f"\n  原策略退出: 止损={st_base.stop_count} 减半={st_base.half_count} "
        f"回撤清={st_base.trail_clear_count}"
    )
    print(
        f"  最优 α={best['giveback_pct']:.0%} (按夏普)  "
        f"年化={best['annual_return']:.2f}%  夏普={best['sharpe_ratio']:.2f}  "
        f"回撤={best['max_ddpercent']:.2f}%  早期清={best['early_zone_exit_count']}"
    )
    print(
        f"  vs 原策略: Δ年化={best['delta_annual_vs_base']:+.2f}%  "
        f"Δ夏普={best['delta_sharpe_vs_base']:+.2f}  "
        f"Δ回撤={best['delta_dd_vs_base']:+.2f}%"
    )

    result = {
        "X": BEST_X,
        "train": [TRAIN_START, TRAIN_END],
        "backtest": [BACKTEST_START, BACKTEST_END],
        "early_zone_arm": EARLY_ZONE_ARM_PCT,
        "early_zone_max": TRAIL_ACTIVATE_PCT,
        "giveback_grid": GIVEBACK_GRID,
        "baseline": {
            **pick_stats(stats_base),
            "stop_count": st_base.stop_count,
            "half_count": st_base.half_count,
            "trail_clear_count": st_base.trail_clear_count,
        },
        "grid": rows,
        "best_by_sharpe": best,
    }
    out_path = os.path.join(ALPHA_LAB_PATH, "livermore_early_zone_giveback_2y.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  结果: {out_path}")
    print("\n✓ 利润吐回 α 网格回测完成!")


if __name__ == "__main__":
    main()
