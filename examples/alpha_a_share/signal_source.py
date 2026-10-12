"""
回测用信号加载：从 alpha_data/signal/{prefix}_*.parquet 拼成引擎所需 signal_df。

prefix 示例：lgb_pred、qlib_pred
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

SIGNAL_PREFIX = {
    "vnpy": "lgb_pred",  # 仅用于「读已有文件」模式；默认回测仍现场训练
    "file": "lgb_pred",
    "qlib": "qlib_pred",
}


def load_signal_parquet(
    signal_dir: str | Path,
    prefix: str,
    start: str,
    end: str,
) -> pl.DataFrame:
    """
    读取 prefix_*.parquet，过滤 [start, end]，返回列 datetime / vt_symbol / signal。
    """
    signal_dir = Path(signal_dir)
    files = sorted(signal_dir.glob(f"{prefix}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到信号: {signal_dir}/{prefix}_*.parquet")

    dfs: list[pl.DataFrame] = []
    for f in files:
        df = pl.read_parquet(f)
        if "signal" not in df.columns:
            if "predicted_return" in df.columns:
                df = df.with_columns(pl.col("predicted_return").alias("signal"))
            else:
                continue
        dfs.append(
            df.select(
                [
                    pl.col("datetime").cast(pl.Datetime("us")).dt.truncate("1d").alias("datetime"),
                    pl.col("vt_symbol").cast(pl.Utf8),
                    pl.col("signal").cast(pl.Float64),
                ]
            )
        )
    if not dfs:
        raise ValueError(f"信号文件无可用列: prefix={prefix}")

    out = pl.concat(dfs).unique(subset=["datetime", "vt_symbol"], keep="last")
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    out = out.filter(
        (pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt)
    ).sort(["datetime", "vt_symbol"])

    if out.is_empty():
        raise ValueError(f"信号在 [{start}, {end}] 内为空: prefix={prefix}")
    return out


def resolve_prefix(source: str) -> str:
    key = (source or "vnpy").strip().lower()
    if key not in SIGNAL_PREFIX and key not in ("train",):
        raise ValueError(f"未知 signal-source={source!r}，可选: vnpy|qlib|file")
    return SIGNAL_PREFIX.get(key, "lgb_pred")
