"""
双轨对比（不下单）：现有 lgb_pred_* vs qlib_pred_*

层 1：IC / 分层收益（标签=未来 3 日累计收益 close_{t+3}/close_t - 1）
层 2：等权 TopN、持有 3 日的代理净值

用法：
  .venv/bin/python examples/alpha_a_share/compare_vnpy_qlib.py
  .venv/bin/python examples/alpha_a_share/compare_vnpy_qlib.py --top-n 5
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from horizons import PRIMARY_HORIZON  # noqa: E402
from stock_universe import STOCK_LIST  # noqa: E402

ALPHA_LAB_PATH = Path(PROJECT_DIR) / "alpha_data"
SIGNAL_DIR = ALPHA_LAB_PATH / "signal"
DAILY_DIR = ALPHA_LAB_PATH / "daily"
EVAL_DIR = ALPHA_LAB_PATH / "evaluation"


def _load_signals(prefix: str) -> pl.DataFrame:
    files = sorted(SIGNAL_DIR.glob(f"{prefix}_*.parquet"))
    if not files:
        return pl.DataFrame()
    dfs = []
    for f in files:
        df = pl.read_parquet(f)
        if "signal" not in df.columns:
            if "predicted_return" in df.columns:
                df = df.with_columns(pl.col("predicted_return").alias("signal"))
            else:
                continue
        df = df.select(
            [
                pl.col("datetime").cast(pl.Datetime("us")).dt.truncate("1d").alias("datetime"),
                pl.col("vt_symbol").cast(pl.Utf8),
                pl.col("signal").cast(pl.Float64),
            ]
        ).with_columns(pl.lit(prefix).alias("source"))
        dfs.append(df)
    return pl.concat(dfs) if dfs else pl.DataFrame()


def _load_close_panel(vt_symbols: list[str]) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for vt in vt_symbols:
        path = DAILY_DIR / f"{vt}.parquet"
        if not path.is_file():
            continue
        df = pl.read_parquet(path).select(
            [
                pl.col("datetime").cast(pl.Datetime("us")).dt.truncate("1d").alias("datetime"),
                pl.lit(vt).alias("vt_symbol"),
                pl.col("close").cast(pl.Float64),
            ]
        )
        rows.append(df)
    if not rows:
        raise SystemExit("无日线数据")
    return pl.concat(rows).sort(["vt_symbol", "datetime"])


def _attach_forward_return(panel: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """按股票内交易日顺序：close[t+h]/close[t]-1。"""
    return (
        panel.sort(["vt_symbol", "datetime"])
        .with_columns(
            pl.col("close").shift(-horizon).over("vt_symbol").alias("close_fwd"),
        )
        .with_columns(
            (pl.col("close_fwd") / pl.col("close") - 1.0).alias("actual_return"),
        )
        .drop("close_fwd")
    )


def _daily_ic(merged: pl.DataFrame) -> pl.DataFrame:
    metrics = []
    for dt in merged["datetime"].unique().sort():
        day = merged.filter(pl.col("datetime") == dt)
        if len(day) < 5:
            continue
        x = day["signal"].to_numpy()
        y = day["actual_return"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        x, y = x[mask], y[mask]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        ric, _ = spearmanr(x, y)
        metrics.append({"datetime": dt, "rank_ic": float(ric), "n": int(mask.sum())})
    return pl.DataFrame(metrics) if metrics else pl.DataFrame()


def _quantile_stats(merged: pl.DataFrame, n_groups: int = 5) -> pl.DataFrame:
    rows = []
    for dt in merged["datetime"].unique().sort():
        day = merged.filter(pl.col("datetime") == dt).drop_nulls(["signal", "actual_return"])
        if len(day) < n_groups:
            continue
        day = day.sort("signal")
        # 近似等分
        idx = np.array_split(np.arange(len(day)), n_groups)
        for g, ix in enumerate(idx, start=1):
            if len(ix) == 0:
                continue
            part = day[ix.tolist()]
            rows.append(
                {
                    "datetime": dt,
                    "group": g,
                    "mean_return": float(part["actual_return"].mean()),
                    "mean_signal": float(part["signal"].mean()),
                }
            )
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .group_by("group")
        .agg(
            pl.col("mean_return").mean().alias("avg_return"),
            pl.col("mean_signal").mean().alias("avg_signal"),
            pl.len().alias("days"),
        )
        .sort("group")
    )


def _topn_hold_backtest(merged: pl.DataFrame, top_n: int, horizon: int) -> dict:
    """
    代理策略：每个信号日等权买入 signal TopN，持有 horizon 个交易日收益取平均；
    再按信号日顺序累乘（重叠持有简化为信号日非重叠抽样：每 horizon 日再平衡一次）。
    """
    dates = merged["datetime"].unique().sort().to_list()
    if len(dates) < horizon + 1:
        return {"error": "日期不足"}

    # 非重叠再平衡：每 horizon 个信号日取一次
    pick_dates = dates[::horizon]
    rets = []
    turnovers = []
    prev_set: set[str] | None = None
    equity = 1.0
    curve = []
    for dt in pick_dates:
        day = merged.filter(pl.col("datetime") == dt).drop_nulls(["signal", "actual_return"])
        if len(day) < top_n:
            continue
        top = day.sort("signal", descending=True).head(top_n)
        names = set(top["vt_symbol"].to_list())
        r = float(top["actual_return"].mean())
        if not np.isfinite(r):
            continue
        equity *= 1.0 + r
        rets.append(r)
        if prev_set is None:
            turnovers.append(1.0)
        else:
            # 单边换手近似
            inter = len(names & prev_set)
            turnovers.append(1.0 - inter / top_n)
        prev_set = names
        curve.append({"datetime": dt, "equity": equity, "period_return": r})

    if not rets:
        return {"error": "无有效再平衡点"}
    arr = np.array(rets, dtype=float)
    eq = np.array([c["equity"] for c in curve], dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return {
        "n_rebalance": len(rets),
        "total_return": float(eq[-1] - 1.0),
        "avg_period_return": float(arr.mean()),
        "max_drawdown": float(dd.min()),
        "avg_turnover": float(np.mean(turnovers)),
        "curve": curve,
    }


def _summarize_source(name: str, signals: pl.DataFrame, actuals: pl.DataFrame, top_n: int, horizon: int) -> dict:
    if signals.is_empty():
        return {"source": name, "error": "无信号文件"}
    merged = signals.join(
        actuals.select(["datetime", "vt_symbol", "actual_return"]),
        on=["datetime", "vt_symbol"],
        how="inner",
    ).drop_nulls(["signal", "actual_return"])
    if merged.is_empty():
        return {"source": name, "error": "信号与收益无法对齐"}

    ic_df = _daily_ic(merged)
    q_df = _quantile_stats(merged)
    bt = _topn_hold_backtest(merged, top_n=top_n, horizon=horizon)

    mean_ic = float(ic_df["rank_ic"].mean()) if not ic_df.is_empty() else float("nan")
    std_ic = float(ic_df["rank_ic"].std()) if not ic_df.is_empty() and len(ic_df) > 1 else float("nan")
    icir = mean_ic / std_ic if std_ic and np.isfinite(std_ic) and abs(std_ic) > 1e-12 else float("nan")

    coverage = merged.select(["datetime", "vt_symbol"]).unique().height
    days = merged["datetime"].n_unique()

    return {
        "source": name,
        "signal_rows": len(signals),
        "merged_rows": len(merged),
        "days": days,
        "coverage_pairs": coverage,
        "mean_rank_ic": mean_ic,
        "icir": icir,
        "quantile": q_df,
        "ic_daily": ic_df,
        "backtest": bt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="vnpy vs qlib 信号对比（不下单）")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    args = ap.parse_args()

    print("=" * 60)
    print("  vnpy vs Qlib 双轨对比（只报告、不下单）")
    print(f"  TopN={args.top_n}  hold={args.horizon}d  label=close[t+{args.horizon}]/close[t]-1")
    print("=" * 60)

    vt_symbols = [f"{c}.{e}" for c, e, _ in STOCK_LIST]
    print("\n[1] 加载信号...")
    vnpy_sig = _load_signals("lgb_pred")
    qlib_sig = _load_signals("qlib_pred")
    print(f"  lgb_pred: {len(vnpy_sig)} 行 / 文件前缀匹配")
    print(f"  qlib_pred: {len(qlib_sig)} 行")

    print("\n[2] 计算实际前向收益...")
    panel = _attach_forward_return(_load_close_panel(vt_symbols), args.horizon)

    # 公平对比：取两边信号日期交集
    if not vnpy_sig.is_empty() and not qlib_sig.is_empty():
        common_dates = (
            set(vnpy_sig["datetime"].unique().to_list())
            & set(qlib_sig["datetime"].unique().to_list())
        )
        if common_dates:
            print(f"\n[3] 评估（重叠信号日={len(common_dates)}）...")
            vnpy_sig = vnpy_sig.filter(pl.col("datetime").is_in(list(common_dates)))
            qlib_sig = qlib_sig.filter(pl.col("datetime").is_in(list(common_dates)))
        else:
            print("\n[3] 评估（无重叠日期，各自全量）...")
    else:
        print("\n[3] 评估...")

    results = [
        _summarize_source("vnpy_lgb", vnpy_sig, panel, args.top_n, args.horizon),
        _summarize_source("qlib_lgb", qlib_sig, panel, args.top_n, args.horizon),
    ]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for r in results:
        print(f"\n── {r['source']} ──")
        if r.get("error"):
            print(f"  ✗ {r['error']}")
            continue
        print(
            f"  信号日={r['days']}  对齐样本={r['merged_rows']}  "
            f"RankIC={r['mean_rank_ic']:+.4f}  ICIR={r['icir']:+.3f}"
        )
        q = r["quantile"]
        if not q.is_empty():
            print("  分层收益:")
            for row in q.iter_rows(named=True):
                print(
                    f"    Q{row['group']}: avg_ret={row['avg_return']:+.4f}  "
                    f"avg_signal={row['avg_signal']:+.4f}"
                )
        bt = r["backtest"]
        if bt.get("error"):
            print(f"  TopN 代理回测: {bt['error']}")
        else:
            print(
                f"  Top{args.top_n} 代理: 累计={bt['total_return']:+.2%}  "
                f"最大回撤={bt['max_drawdown']:+.2%}  "
                f"均换手={bt['avg_turnover']:.2f}  再平衡={bt['n_rebalance']}"
            )
            curve = pl.DataFrame(bt["curve"])
            curve_path = EVAL_DIR / f"vnpy_vs_qlib_curve_{r['source']}.parquet"
            curve.write_parquet(curve_path)
            print(f"  曲线: {curve_path}")

        summary_rows.append(
            {
                "source": r["source"],
                "days": r["days"],
                "merged_rows": r["merged_rows"],
                "mean_rank_ic": r["mean_rank_ic"],
                "icir": r["icir"],
                "topn_total_return": bt.get("total_return"),
                "topn_max_drawdown": bt.get("max_drawdown"),
                "topn_avg_turnover": bt.get("avg_turnover"),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if not r["ic_daily"].is_empty():
            r["ic_daily"].write_parquet(EVAL_DIR / f"vnpy_vs_qlib_ic_{r['source']}.parquet")

    if summary_rows:
        out = EVAL_DIR / "vnpy_vs_qlib_summary.parquet"
        pl.DataFrame(summary_rows).write_parquet(out)
        md = EVAL_DIR / "vnpy_vs_qlib_summary.md"
        lines = [
            "# vnpy vs Qlib 对比摘要",
            "",
            f"- TopN={args.top_n}, horizon={args.horizon}",
            f"- 生成时间: {summary_rows[0]['generated_at']}",
            "",
            "| source | days | RankIC | ICIR | TopN累计 | 最大回撤 |",
            "|--------|------|--------|------|----------|----------|",
        ]
        for s in summary_rows:
            lines.append(
                f"| {s['source']} | {s['days']} | {s['mean_rank_ic']:+.4f} | "
                f"{s['icir']:+.3f} | {s['topn_total_return']:+.2%} | "
                f"{s['topn_max_drawdown']:+.2%} |"
            )
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n✓ 摘要: {out}")
        print(f"✓ Markdown: {md}")
    else:
        print("\n✗ 无可用对比结果（是否尚未导出 qlib_pred_*？）")


if __name__ == "__main__":
    main()
