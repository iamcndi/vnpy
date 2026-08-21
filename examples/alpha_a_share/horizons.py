"""
多持有期预测配置

标签统一为「今日收盘 → 未来 N 日收盘」的累计收益：
  ts_delay(close, -N) / close - 1
"""

from __future__ import annotations

FORECAST_HORIZONS: list[int] = [1, 3, 5]
PRIMARY_HORIZON: int = 3  # 主排序 / 回测 / 兼容旧模型名


def horizon_label(days: int) -> str:
    """未来 N 日累计收益标签表达式"""
    return f"ts_delay(close, -{days}) / close - 1"


def model_name(days: int) -> str:
    return f"lgb_alpha_158_{days}d"


def feature_cols_name(days: int) -> str:
    return f"feature_cols_{days}d.json"


# 兼容旧版单模型文件名（等同于 3 日模型）
LEGACY_MODEL_NAME = "lgb_alpha_158"
LEGACY_FEATURE_COLS = "feature_cols.json"
