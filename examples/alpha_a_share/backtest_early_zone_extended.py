"""
早期盈利区回撤 5%/8% — 扩展区间回测

  训练: 2023-01-01 ~ 2024-06-28
  回测: 2024-06-29 ~ 2026-08-28（约 2 年 2 个月）

对比: 原利弗莫尔 | +早期区回撤 5% | +早期区回撤 8%

用法:
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/backtest_early_zone_extended.py
  .venv/bin/python examples/alpha_a_share/backtest_early_zone_extended.py --skip-download
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
    LOOKBACK_DAYS,
    TRAIL_ACTIVATE_PCT,
    TRAIL_PULLBACK_PCT,
    build_dataset_df,
    download_daily_data,
    pick_stats,
    run_one,
    setup_contracts,
)
from stock_universe import STOCK_LIST  # noqa: E402

TRAIN_START = "2023-01-01"
TRAIN_END = "2024-06-28"
BACKTEST_START = "2024-06-29"
BACKTEST_END = "2026-08-28"
BEST_X = 0.10
PULLBACK_VARIANTS = [0.05, 0.08]


def _patch_run_dates():
    """让 run_one 使用扩展区间（不改动原脚本常量）。"""
    import backtest_1y_livermore as bm

    bm.BACKTEST_START = BACKTEST_START
    bm.BACKTEST_END = BACKTEST_END


def _line(name: str, s: dict, extra: str = "") -> None:
    print(
        f"  {name:30s}  {s.get('annual_return', float('nan')):>10.2f}  "
        f"{s.get('total_return', float('nan')):>10.2f}  "
        f"{s.get('sharpe_ratio', float('nan')):>8.2f}  "
        f"{s.get('max_ddpercent', float('nan')):>10.2f}  "
        f"{s.get('total_trade_count', 0):>8}{extra}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    _patch_run_dates()

    print("=" * 100)
    print("  早期盈利区回撤 — 扩展区间回测")
    print(f"  训练: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  回测: {BACKTEST_START} ~ {BACKTEST_END}  资金={INITIAL_CAPITAL:,}  X={BEST_X:.0%}")
    print(
        f"  早期区 [{EARLY_ZONE_ARM_PCT:.0%}, {TRAIL_ACTIVATE_PCT:.0%}) 回撤清仓 "
        f"| ≥{TRAIL_ACTIVATE_PCT:.0%} 移动止盈 {TRAIL_PULLBACK_PCT:.0%}"
    )
    print(f"  对比回撤: {[f'{x:.0%}' for x in PULLBACK_VARIANTS]}")
    print("=" * 100)

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
    print(f"  数据: {df['datetime'].min()} ~ {df['datetime'].max()}  行数={len(df)}")
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
    print(f"  回测信号行数: {len(signal_df)}")

    print("\n[3/3] 回测...")
    stats_base, eng_base = run_one(
        lab, vt_symbols, signal_df, LivermoreStrategy, {"add_spacing_pct": BEST_X}
    )
    st_base = eng_base.strategy

    variants: list[dict] = []
    for pb in PULLBACK_VARIANTS:
        print(f"  回撤={pb:.0%} ...", flush=True)
        setting = {
            "add_spacing_pct": BEST_X,
            "profit_lock_arm_pct": EARLY_ZONE_ARM_PCT,
            "early_zone_pullback_pct": pb,
        }
        stats, engine = run_one(lab, vt_symbols, signal_df, LivermoreStrategy, setting)
        st = engine.strategy
        variants.append({
            "pullback_pct": pb,
            **pick_stats(stats),
            "early_zone_exit_count": st.early_zone_exit_count,
            "stop_count": st.stop_count,
            "half_count": st.half_count,
            "trail_clear_count": st.trail_clear_count,
            "delta_annual_vs_base": stats.get("annual_return", 0) - stats_base.get("annual_return", 0),
            "delta_sharpe_vs_base": stats.get("sharpe_ratio", 0) - stats_base.get("sharpe_ratio", 0),
            "delta_dd_vs_base": stats.get("max_ddpercent", 0) - stats_base.get("max_ddpercent", 0),
        })

    print("\n" + "=" * 100)
    print(f"  {'策略':30s}  {'年化%':>10s}  {'总收益%':>10s}  {'夏普':>8s}  {'最大回撤%':>10s}  {'成交':>8s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    _line("原利弗莫尔", stats_base)
    for v in variants:
        _line(
            f"+早期区回撤 {v['pullback_pct']:.0%}",
            v,
            f"  早期清={v['early_zone_exit_count']}",
        )

    print(
        f"\n  原策略退出: 止损={st_base.stop_count} 减半={st_base.half_count} "
        f"回撤清={st_base.trail_clear_count}"
    )
    for v in variants:
        print(
            f"  回撤{v['pullback_pct']:.0%}: 早期清={v['early_zone_exit_count']} "
            f"止损={v['stop_count']} 减半={v['half_count']} 回撤清={v['trail_clear_count']}  "
            f"Δ年化={v['delta_annual_vs_base']:+.2f}% Δ夏普={v['delta_sharpe_vs_base']:+.2f} "
            f"Δ回撤={v['delta_dd_vs_base']:+.2f}%"
        )

    result = {
        "train": [TRAIN_START, TRAIN_END],
        "backtest": [BACKTEST_START, BACKTEST_END],
        "X": BEST_X,
        "early_zone_arm": EARLY_ZONE_ARM_PCT,
        "baseline": {
            **pick_stats(stats_base),
            "stop_count": st_base.stop_count,
            "half_count": st_base.half_count,
            "trail_clear_count": st_base.trail_clear_count,
        },
        "variants": variants,
    }
    out_path = os.path.join(ALPHA_LAB_PATH, "livermore_early_zone_extended.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  结果: {out_path}")
    print("\n✓ 扩展区间回测完成!")


if __name__ == "__main__":
    main()
