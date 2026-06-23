"""
预测评估系统：加载已保存的预测信号，与实际收益对比，计算评估指标

核心指标：
  - IC (Information Coefficient): 预测信号与实际收益的Spearman秩相关
  - Rank IC: 分日计算的秩相关，取均值
  - Hit Rate: 方向准确率
  - Top/Bottom Spread: 多头组 vs 空头组的实际收益差

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/evaluate_predictions.py
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import numpy as np
from scipy.stats import spearmanr, pearsonr

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from vnpy.alpha import AlphaLab
from vnpy.trader.constant import Interval, Exchange


ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")
EVAL_DIR = os.path.join(ALPHA_LAB_PATH, "evaluation")
FORWARD_SHIFT = 3          # 标签: close_{t+3} / close_{t+1} - 1

STOCK_LIST: list[tuple[str, str, str]] = [
    ("002245", "SZSE", "蔚蓝锂芯"), ("600487", "SSE", "亨通光电"),
    ("600089", "SSE", "特变电工"), ("002532", "SZSE", "天山铝业"),
    ("300316", "SZSE", "晶盛机电"), ("300843", "SZSE", "胜蓝股份"),
    ("300438", "SZSE", "鹏辉能源"), ("000338", "SZSE", "潍柴动力"),
    ("300661", "SZSE", "圣邦股份"), ("300507", "SZSE", "苏奥传感"),
    ("301511", "SZSE", "德福科技"), ("300442", "SZSE", "润泽科技"),
    ("301498", "SZSE", "乖宝宠物"), ("002299", "SZSE", "圣农发展"),
    ("601717", "SSE", "中创智领"), ("002639", "SZSE", "雪人集团"),
    ("601665", "SSE", "齐鲁银行"), ("600580", "SSE", "卧龙电驱"),
    ("301217", "SZSE", "铜冠铜箔"), ("300484", "SZSE", "蓝海华腾"),
    ("518800", "SSE", "黄金ETF国泰"), ("688523", "SSE", "航天环宇"),
    ("300433", "SZSE", "蓝思科技"), ("300811", "SZSE", "铂科新材"),
    ("300136", "SZSE", "信维通信"),
]


def load_all_signals(lab: AlphaLab) -> pl.DataFrame:
    """加载所有已保存的预测信号"""
    files = sorted(lab.signal_path.glob("lgb_pred_*.parquet"))
    if not files:
        print("✗ 没有找到预测信号文件")
        return pl.DataFrame()

    dfs = [pl.read_parquet(f).with_columns(pl.lit(f.stem).alias("filename")) for f in files]
    all_signals = pl.concat(dfs)
    print(f"  ✓ 加载 {len(files)} 个信号文件, 共 {len(all_signals)} 条记录")
    return all_signals


def compute_actual_returns(lab: AlphaLab, vt_symbols: list[str]) -> pl.DataFrame:
    """用交易日索引计算实际前向收益 (close_{t+F} / close_{t+1} - 1)"""
    all_records = []
    for vt_symbol in vt_symbols:
        bars = lab.load_bar_data(vt_symbol, Interval.DAILY, datetime(2023, 1, 1), datetime.now())
        if len(bars) < FORWARD_SHIFT + 1:
            continue
        dates = [b.datetime for b in bars]
        close_map = {b.datetime: b.close_price for b in bars}
        for i, dt in enumerate(dates):
            if i + FORWARD_SHIFT < len(dates):
                c1 = close_map[dates[i + 1]]
                c3 = close_map[dates[i + FORWARD_SHIFT]]
                ret = (c3 / c1 - 1) if (c1 > 0 and c3 != c1) else float("nan")
            else:
                ret = float("nan")
            all_records.append({"datetime": dt, "vt_symbol": vt_symbol, "actual_return": ret, "close": close_map[dt]})
    df = pl.DataFrame(all_records).sort(["datetime", "vt_symbol"])
    print(f"  ✓ 计算 {len(df)} 条实际收益记录")
    return df


def compute_daily_ic(signals: pl.DataFrame, actuals: pl.DataFrame) -> pl.DataFrame:
    """计算每日的 IC / Rank IC / Hit Rate"""
    merged = signals.join(actuals, on=["datetime", "vt_symbol"], how="inner")
    if merged.is_empty() or "actual_return" not in merged.columns:
        return pl.DataFrame()

    daily_metrics = []
    for dt in merged["datetime"].unique().sort():
        day_df = merged.filter(pl.col("datetime") == dt)
        if len(day_df) < 5:
            continue
        preds = day_df["signal"].to_numpy()
        actuals_arr = day_df["actual_return"].to_numpy()
        mask = ~np.isnan(actuals_arr)
        if mask.sum() < 5:
            continue
        preds, actuals_arr = preds[mask], actuals_arr[mask]

        # 检查是否有足够的方差来计算相关性
        if np.std(preds) == 0 or np.std(actuals_arr) == 0:
            continue

        ic_val, ic_p = pearsonr(preds, actuals_arr)
        rank_ic_val, rank_ic_p = spearmanr(preds, actuals_arr)

        median_pred = float(np.median(preds))
        pred_up = preds >= median_pred
        actual_up = actuals_arr > 0
        hit_rate = (pred_up == actual_up).sum() / len(preds)

        n = len(preds)
        top_n = max(1, n // 3)
        sorted_idx = np.argsort(preds)
        top_return = float(np.mean(actuals_arr[sorted_idx[-top_n:]]))
        bottom_return = float(np.mean(actuals_arr[sorted_idx[:top_n]]))

        daily_metrics.append({
            "datetime": dt, "n_stocks": len(preds),
            "ic": float(ic_val), "ic_pvalue": float(ic_p),
            "rank_ic": float(rank_ic_val), "rank_ic_pvalue": float(rank_ic_p),
            "hit_rate": float(hit_rate),
            "top_return": top_return, "bottom_return": bottom_return,
            "top_bottom_spread": top_return - bottom_return,
            "mean_actual_return": float(np.mean(actuals_arr)),
        })
    return pl.DataFrame(daily_metrics)


def compute_quantile_returns(signals: pl.DataFrame, actuals: pl.DataFrame, n_groups: int = 5) -> pl.DataFrame:
    """按信号强度分 n 组, 计算每组平均实际收益"""
    merged = signals.join(actuals, on=["datetime", "vt_symbol"], how="inner")
    if merged.is_empty() or "actual_return" not in merged.columns:
        return pl.DataFrame()
    records = []
    for dt in merged["datetime"].unique().sort():
        day_df = merged.filter(pl.col("datetime") == dt)
        if len(day_df) < n_groups * 2:
            continue
        day_df = day_df.sort("signal")
        group_size = len(day_df) // n_groups
        for g in range(n_groups):
            gs = g * group_size
            ge = gs + group_size if g < n_groups - 1 else len(day_df)
            gdf = day_df[gs:ge]
            ar = gdf["actual_return"].to_numpy()
            records.append({"datetime": dt, "group": g + 1, "n_stocks": len(gdf),
                            "mean_signal": float(np.mean(gdf["signal"].to_numpy())),
                            "mean_actual_return": float(np.mean(ar)),
                            "median_actual_return": float(np.median(ar))})
    return pl.DataFrame(records).sort(["datetime", "group"])


def compute_rolling_metrics(daily_metrics: pl.DataFrame, rolling_window: int = 10) -> pl.DataFrame:
    if daily_metrics.is_empty():
        return pl.DataFrame()
    return daily_metrics.sort("datetime").with_columns([
        pl.col("ic").rolling_mean(window_size=rolling_window, min_samples=3).alias("ic_ma"),
        pl.col("rank_ic").rolling_mean(window_size=rolling_window, min_samples=3).alias("rank_ic_ma"),
        pl.col("hit_rate").rolling_mean(window_size=rolling_window, min_samples=3).alias("hit_rate_ma"),
        pl.col("top_bottom_spread").rolling_mean(window_size=rolling_window, min_samples=3).alias("spread_ma"),
    ])


def print_evaluation_report(daily_metrics: pl.DataFrame, rolling_metrics: pl.DataFrame, quantile_returns: pl.DataFrame) -> None:
    if daily_metrics.is_empty():
        print("\n✗ 无可评估的预测数据")
        return
    print(f"\n{'='*70}\n  预测评估报告\n{'='*70}")
    print(f"\n  【总体统计】")
    print(f"  {'指标':<25s} {'值':>12s}\n  {'-'*25} {'-'*12}")
    print(f"  {'IC (均值)':<25s} {daily_metrics['ic'].mean():>+12.4f}")
    print(f"  {'Rank IC (均值)':<25s} {daily_metrics['rank_ic'].mean():>+12.4f}")
    print(f"  {'Hit Rate (均值)':<25s} {daily_metrics['hit_rate'].mean():>12.2%}")
    print(f"  {'IC > 0 占比':<25s} {(daily_metrics['ic'] > 0).mean() * 100:>12.1f}%")
    print(f"  {'Top-Bottom Spread':<25s} {daily_metrics['top_bottom_spread'].mean():>+12.4f}")
    print(f"  {'评估天数':<25s} {len(daily_metrics):>12d}")

    if not quantile_returns.is_empty():
        print(f"\n  【分组收益(按信号强度)】")
        print(f"  {'组':>6s}  {'信号均值':>10s}  {'实际收益均值':>14s}  {'中位数':>10s}")
        gs = quantile_returns.group_by("group").agg([
            pl.col("mean_signal").mean().alias("avg_signal"),
            pl.col("mean_actual_return").mean().alias("avg_return"),
            pl.col("median_actual_return").mean().alias("med_return"),
        ]).sort("group")
        labels = ["Q1(最弱)", "Q2", "Q3", "Q4", "Q5(最强)"]
        for row in gs.iter_rows(named=True):
            label = labels[int(row["group"]) - 1] if int(row["group"]) <= 5 else f"G{row['group']}"
            print(f"  {label:>6s}  {row['avg_signal']:>10.4f}  {row['avg_return']:>+14.4f}  {row['med_return']:>+10.4f}")

    if not rolling_metrics.is_empty():
        print(f"\n  【滚动模型质量监测(滑动窗口=10)】")
        print(f"  {'日期':<12s} {'IC':>8s} {'RankIC':>8s} {'HitRate':>10s} {'Spread':>10s}")
        for row in rolling_metrics.tail(5).iter_rows(named=True):
            dt_str = row["datetime"].strftime("%m-%d")
            ic = row.get("ic_ma", 0) or 0
            ric = row.get("rank_ic_ma", 0) or 0
            hr = row.get("hit_rate_ma", 0) or 0
            sp = row.get("spread_ma", 0) or 0
            print(f"  {dt_str:<12s} {ic:>+8.4f} {ric:>+8.4f} {hr:>10.2%} {sp:>+10.4f}")


def save_evaluation_results(lab: AlphaLab, daily_metrics: pl.DataFrame, rolling_metrics: pl.DataFrame, quantile_returns: pl.DataFrame, merged: pl.DataFrame) -> None:
    eval_dir = Path(EVAL_DIR)
    eval_dir.mkdir(parents=True, exist_ok=True)
    if not daily_metrics.is_empty():
        daily_metrics.write_parquet(eval_dir / "daily_metrics.parquet")
    if not rolling_metrics.is_empty():
        rolling_metrics.write_parquet(eval_dir / "rolling_metrics.parquet")
    if not quantile_returns.is_empty():
        quantile_returns.write_parquet(eval_dir / "quantile_returns.parquet")
    if not merged.is_empty():
        merged.write_parquet(eval_dir / "prediction_vs_actual.parquet")
    print(f"\n  评估结果已保存至: {eval_dir}/")


def main() -> None:
    print("=" * 70)
    print("  预测评估系统 — 预测信号 vs 实际收益")
    print("=" * 70)
    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]

    print("\n[1/5] 加载历史预测信号...")
    signals = load_all_signals(lab)
    if signals.is_empty():
        return

    print("\n[2/5] 计算实际前向收益...")
    actuals = compute_actual_returns(lab, vt_symbols)

    print("\n[3/5] 计算每日评估指标...")
    daily_metrics = compute_daily_ic(signals, actuals)
    if daily_metrics.is_empty():
        print("  ⚠ 所有预测日期均无可比较的实际收益(需等待数据积累)")
        return

    print("\n[4/5] 分组收益分析 + 滚动监测...")
    quantile_returns = compute_quantile_returns(signals, actuals, n_groups=5)
    rolling_metrics = compute_rolling_metrics(daily_metrics, rolling_window=10)
    merged = signals.join(actuals, on=["datetime", "vt_symbol"], how="inner")

    print("\n[5/5] 生成评估报告...")
    print_evaluation_report(daily_metrics, rolling_metrics, quantile_returns)
    save_evaluation_results(lab, daily_metrics, rolling_metrics, quantile_returns, merged)
    print("\n✓ 评估完成!")


if __name__ == "__main__":
    main()
