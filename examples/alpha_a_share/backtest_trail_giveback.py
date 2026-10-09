"""
早期区 α=70% + 移动止盈也按利润吐回（一年回测）

  训练: 2023-01-01 ~ 2025-06-28
  回测: 2025-06-29 ~ 2026-06-29
  早期区 (+3%~+15%)：吐回峰值利润 70% → 清仓
  ≥15% 后：同样吐回 70% → 先减半再清（对照仍用原 8% 高点回撤）
  不改每日预测。

用法:
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/backtest_trail_giveback.py --skip-download
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
TRAIN_END = "2025-06-28"
BACKTEST_START = "2025-06-29"
BACKTEST_END = "2026-06-29"
BEST_X = 0.10
ALPHA = 0.70


def _patch_run_dates() -> None:
    import backtest_1y_livermore as bm

    bm.BACKTEST_START = BACKTEST_START
    bm.BACKTEST_END = BACKTEST_END


def _line(name: str, s: dict, extra: str = "") -> None:
    print(
        f"  {name:36s}  {s.get('annual_return', float('nan')):>10.2f}  "
        f"{s.get('sharpe_ratio', float('nan')):>8.2f}  "
        f"{s.get('max_ddpercent', float('nan')):>10.2f}  "
        f"{s.get('total_trade_count', 0):>8}{extra}"
    )


def _run(lab, vt_symbols, signal_df, setting: dict):
    stats, engine = run_one(lab, vt_symbols, signal_df, LivermoreStrategy, setting)
    st = engine.strategy
    row = {
        **pick_stats(stats),
        "early_zone_exit_count": getattr(st, "early_zone_exit_count", 0),
        "stop_count": st.stop_count,
        "half_count": st.half_count,
        "trail_clear_count": st.trail_clear_count,
    }
    return row, st


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    _patch_run_dates()

    print("=" * 100)
    print("  早期区 α=70% + 移动止盈利润吐回（一年）")
    print(f"  训练: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  回测: {BACKTEST_START} ~ {BACKTEST_END}  资金={INITIAL_CAPITAL:,}  X={BEST_X:.0%}")
    print(
        f"  早期区 [{EARLY_ZONE_ARM_PCT:.0%}, {TRAIL_ACTIVATE_PCT:.0%}) 吐回{ALPHA:.0%}清 | "
        f"≥{TRAIL_ACTIVATE_PCT:.0%} 对照=高点回撤{TRAIL_PULLBACK_PCT:.0%} / 实验=吐回{ALPHA:.0%}"
    )
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
    variants = [
        ("原利弗莫尔（无早期区，移动8%）", {"add_spacing_pct": BEST_X}),
        (
            f"早期吐回{ALPHA:.0%} + 移动高点回撤{TRAIL_PULLBACK_PCT:.0%}",
            {
                "add_spacing_pct": BEST_X,
                "profit_lock_arm_pct": EARLY_ZONE_ARM_PCT,
                "early_zone_giveback_pct": ALPHA,
            },
        ),
        (
            f"早期吐回{ALPHA:.0%} + 移动也吐回{ALPHA:.0%}",
            {
                "add_spacing_pct": BEST_X,
                "profit_lock_arm_pct": EARLY_ZONE_ARM_PCT,
                "early_zone_giveback_pct": ALPHA,
                "trail_giveback_pct": ALPHA,
            },
        ),
    ]
    rows: list[dict] = []
    for i, (name, setting) in enumerate(variants, 1):
        print(f"  ({i}/{len(variants)}) {name} ...", flush=True)
        row, _st = _run(lab, vt_symbols, signal_df, setting)
        row["name"] = name
        row["setting"] = setting
        rows.append(row)

    base = rows[0]
    for row in rows:
        row["delta_annual_vs_base"] = row.get("annual_return", 0) - base.get("annual_return", 0)
        row["delta_sharpe_vs_base"] = row.get("sharpe_ratio", 0) - base.get("sharpe_ratio", 0)
        row["delta_dd_vs_base"] = row.get("max_ddpercent", 0) - base.get("max_ddpercent", 0)

    print("\n" + "=" * 100)
    print(f"  {'策略':36s}  {'年化%':>10s}  {'夏普':>8s}  {'最大回撤%':>10s}  {'成交':>8s}")
    print(f"  {'-'*36}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    for row in rows:
        extra = (
            f"  早期清={row['early_zone_exit_count']}"
            f" 减半={row['half_count']} 回撤清={row['trail_clear_count']}"
        )
        _line(row["name"], row, extra)

    exp = rows[-1]
    print(
        f"\n  实验 vs 原策略: Δ年化={exp['delta_annual_vs_base']:+.2f}%  "
        f"Δ夏普={exp['delta_sharpe_vs_base']:+.2f}  "
        f"Δ回撤={exp['delta_dd_vs_base']:+.2f}%"
    )

    result = {
        "alpha": ALPHA,
        "train": [TRAIN_START, TRAIN_END],
        "backtest": [BACKTEST_START, BACKTEST_END],
        "variants": rows,
    }
    out_path = os.path.join(ALPHA_LAB_PATH, "livermore_trail_giveback_1y.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  结果: {out_path}")
    print("\n✓ 移动止盈利润吐回一年回测完成!")


if __name__ == "__main__":
    main()
