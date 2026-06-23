
"""
预测优化系统：基于评估结果，动态优化模型和交易参数

核心优化逻辑：
  1. 自适应阈值：根据近期 Rank IC 动态调整多头/空头分组比率
  2. 置信度校准：根据近期 Hit Rate 调整持仓权重
  3. 信号衰减检测：监控 Rank IC 趋势，预警模型衰减
  4. 重训练触发：当评估指标低于阈值时，触发模型重训练
  5. 特征漂移监控：跟踪特征重要性变化

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/optimize_prediction.py

输出：
  - 控制台打印优化建议
  - alpha_data/evaluation/ 下保存优化配置
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


ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")
EVAL_DIR = os.path.join(ALPHA_LAB_PATH, "evaluation")
MODEL_NAME = "lgb_alpha_158"

DEFAULT_PARAMS = {
    "retrain_ic_threshold": 0.05,
    "retrain_hit_rate_threshold": 0.45,
    "long_top_n_ratio": 1.0 / 3.0,
    "max_confidence_scale": 1.5,
    "min_confidence_scale": 0.3,
    "rolling_window": 10,
    "base_retrain_interval_days": 30,
}

OPTIMIZED_CONFIG_PATH = os.path.join(ALPHA_LAB_PATH, "optimized_config.json")


def load_evaluation_results() -> tuple:
    """加载已保存的评估结果"""
    eval_dir = Path(EVAL_DIR)
    daily_metrics = pl.DataFrame()
    rolling_metrics = pl.DataFrame()
    quantile_returns = pl.DataFrame()

    if (eval_dir / "daily_metrics.parquet").exists():
        daily_metrics = pl.read_parquet(eval_dir / "daily_metrics.parquet")
    if (eval_dir / "rolling_metrics.parquet").exists():
        rolling_metrics = pl.read_parquet(eval_dir / "rolling_metrics.parquet")
    if (eval_dir / "quantile_returns.parquet").exists():
        quantile_returns = pl.read_parquet(eval_dir / "quantile_returns.parquet")

    print(f"  ✓ 加载评估结果: 每日指标 {len(daily_metrics)} 条, 滚动指标 {len(rolling_metrics)} 条")
    return daily_metrics, rolling_metrics, quantile_returns


def compute_adaptive_thresholds(rolling_metrics: pl.DataFrame, daily_metrics: pl.DataFrame) -> dict:
    """根据近期评估指标，自适应调整多头比率和信号强度阈值"""
    config = dict(DEFAULT_PARAMS)

    if rolling_metrics.is_empty() or daily_metrics.is_empty():
        print("  ⚠ 评估数据不足，使用默认参数")
        return config

    recent_rank_ic = rolling_metrics.tail(5)["rank_ic_ma"].mean()
    recent_hit_rate = rolling_metrics.tail(5)["hit_rate_ma"].mean()
    recent_spread = rolling_metrics.tail(5)["spread_ma"].mean()

    print(f"\n  【近期模型表现】")
    print(f"  Rank IC (5日均值): {recent_rank_ic:+.4f}")
    print(f"  Hit Rate (5日均值): {recent_hit_rate:.2%}")
    print(f"  Top-Bottom Spread (5日均值): {recent_spread:+.4f}")

    # 自适应多头比率
    base_ratio = DEFAULT_PARAMS["long_top_n_ratio"]
    if recent_rank_ic > 0.15:
        adjusted_ratio = min(base_ratio * 1.3, 0.5)
        reason = "IC 高，扩大选股范围"
    elif recent_rank_ic > 0.08:
        adjusted_ratio = base_ratio
        reason = "IC 正常，保持默认"
    elif recent_rank_ic > 0.03:
        adjusted_ratio = max(base_ratio * 0.7, 0.15)
        reason = "IC 偏低，缩小范围"
    else:
        adjusted_ratio = max(base_ratio * 0.5, 0.1)
        reason = "IC 接近0，大幅缩小范围"

    config["long_top_n_ratio"] = adjusted_ratio
    print(f"\n  【自适应阈值调整】")
    print(f"  选股比率: {base_ratio:.2%} → {adjusted_ratio:.2%}  ({reason})")

    # 信号强度阈值
    if recent_rank_ic > 0.10:
        config["min_signal_threshold"] = 0.3
    elif recent_rank_ic > 0.05:
        config["min_signal_threshold"] = 0.5
    else:
        config["min_signal_threshold"] = 0.7

    print(f"  最小信号阈值: {config['min_signal_threshold']:.2f}")

    # 重训练判断
    model_decayed = recent_rank_ic < DEFAULT_PARAMS["retrain_ic_threshold"]
    config["retrain_recommended"] = model_decayed
    if model_decayed:
        print(f"  ⚠ 模型严重衰减！推荐立即重训练")

    return config


def compute_confidence_scaling(rolling_metrics: pl.DataFrame) -> float:
    """基于近期准确率计算仓位缩放因子"""
    if rolling_metrics.is_empty():
        return 1.0

    recent_hit_rate = rolling_metrics.tail(5)["hit_rate_ma"].mean()
    recent_ic = rolling_metrics.tail(5)["ic_ma"].mean()

    hit_rate_score = (recent_hit_rate - 0.5) * 10
    ic_score = recent_ic * 5

    confidence = 1.0 + hit_rate_score + ic_score
    confidence = np.clip(confidence, DEFAULT_PARAMS["min_confidence_scale"], DEFAULT_PARAMS["max_confidence_scale"])

    print(f"\n  【置信度校准】")
    print(f"  近期 Hit Rate: {recent_hit_rate:.2%} (得分: {hit_rate_score:+.2f})")
    print(f"  近期 IC: {recent_ic:+.4f} (得分: {ic_score:+.2f})")
    print(f"  仓位缩放因子: {confidence:.2f}x")

    return confidence


def check_feature_drift(lab: AlphaLab, daily_metrics: pl.DataFrame) -> dict:
    """检查特征重要性漂移"""
    model = lab.load_model(MODEL_NAME)
    if model is None or model.model is None:
        print("  ⚠ 无法加载模型")
        return {"drift_detected": False}

    feat_path = lab.model_path.parent.joinpath("feature_cols.json")
    if not feat_path.exists():
        return {"drift_detected": False}

    with open(feat_path) as f:
        feature_cols = json.load(f)

    importances = model.model.feature_importances_
    top_features = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    if len(daily_metrics) >= 20:
        early_ic = daily_metrics.head(10)["ic"].mean()
        recent_ic = daily_metrics.tail(10)["ic"].mean()
        ic_decline = early_ic - recent_ic
        drift_detected = ic_decline > 0.10
    else:
        ic_decline = 0.0
        drift_detected = False
        early_ic = recent_ic = 0.0

    print(f"\n  【特征漂移监控】")
    print(f"  模型特征数: {len(feature_cols)}")
    print(f"  Top-5 特征: {', '.join(n for n, _ in top_features[:5])}")
    if len(daily_metrics) >= 20:
        print(f"  IC 变化: {early_ic:+.4f} → {recent_ic:+.4f}")

    if drift_detected:
        print(f"  ⚠ 检测到特征漂移")

    return {"drift_detected": drift_detected, "ic_decline": ic_decline, "top_features": top_features[:10]}


def generate_optimized_config(thresholds: dict, confidence_scale: float, drift_info: dict, daily_metrics: pl.DataFrame) -> dict:
    """生成优化后的配置文件"""
    config = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optimized_params": {
            "long_top_n_ratio": thresholds["long_top_n_ratio"],
            "min_signal_threshold": thresholds.get("min_signal_threshold", 0.0),
            "confidence_scale": confidence_scale,
        },
        "model_status": {
            "retrain_needed": thresholds.get("retrain_recommended", False),
            "drift_detected": drift_info.get("drift_detected", False),
            "feature_count": len(drift_info.get("top_features", [])),
            "ic_decline": drift_info.get("ic_decline", 0.0),
        },
        "evaluation_summary": {
            "total_days": len(daily_metrics),
            "mean_ic": float(daily_metrics["ic"].mean()) if not daily_metrics.is_empty() else 0.0,
            "mean_hit_rate": float(daily_metrics["hit_rate"].mean()) if not daily_metrics.is_empty() else 0.0,
        },
        "retraining_schedule": {
            "days_since_last_train": None,
            "recommended_retrain_date": None,
            "retrain_reason": None,
        },
    }

    model_path = Path(ALPHA_LAB_PATH) / "model" / f"{MODEL_NAME}.pkl"
    if model_path.exists():
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        days_since = (datetime.now() - mtime).days
        config["retraining_schedule"]["days_since_last_train"] = days_since

        if thresholds.get("retrain_recommended", False):
            config["retraining_schedule"]["recommended_retrain_date"] = "now"
            config["retraining_schedule"]["retrain_reason"] = "模型预测能力衰减"
        elif days_since > DEFAULT_PARAMS["base_retrain_interval_days"]:
            config["retraining_schedule"]["recommended_retrain_date"] = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            config["retraining_schedule"]["retrain_reason"] = f"距上次训练{days_since}天"

    return config


def print_optimization_report(config: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  优化建议报告")
    print(f"{'='*70}")

    params = config["optimized_params"]
    status = config["model_status"]
    schedule = config["retraining_schedule"]
    summary = config["evaluation_summary"]

    print(f"\n  【参数优化建议】")
    print(f"  选股比率:           {params['long_top_n_ratio']:.4f}")
    print(f"  最小信号阈值:       {params['min_signal_threshold']:.2f}")
    print(f"  仓位缩放因子:       {params['confidence_scale']:.2f}x")

    print(f"\n  【模型状态】")
    print(f"  需要重训练:         {'是' if status['retrain_needed'] else '否'}")
    print(f"  特征漂移:           {'检测到' if status['drift_detected'] else '未检测到'}")

    print(f"\n  【评估摘要】")
    print(f"  评估天数:           {summary['total_days']}")
    print(f"  均值 IC:            {summary['mean_ic']:+.4f}")
    print(f"  均值 Hit:           {summary['mean_hit_rate']:.2%}")

    print(f"\n  【重训练计划】")
    if schedule["days_since_last_train"] is not None:
        print(f"  距上次训练:         {schedule['days_since_last_train']} 天")
    if schedule["recommended_retrain_date"]:
        print(f"  建议重训练:         {schedule['recommended_retrain_date']}")
    if schedule["retrain_reason"]:
        print(f"  原因:               {schedule['retrain_reason']}")

    print(f"\n{'='*70}")


def save_optimized_config(config: dict) -> None:
    with open(OPTIMIZED_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  优化配置已保存至: {OPTIMIZED_CONFIG_PATH}")


def main() -> None:
    print("=" * 70)
    print("  预测优化系统 — 自适应参数优化")
    print("=" * 70)

    lab = AlphaLab(ALPHA_LAB_PATH)

    print("\n[1/5] 加载评估结果...")
    daily_metrics, rolling_metrics, _ = load_evaluation_results()

    print("\n[2/5] 计算自适应阈值...")
    thresholds = compute_adaptive_thresholds(rolling_metrics, daily_metrics)

    print("\n[3/5] 计算置信度缩放...")
    confidence_scale = compute_confidence_scaling(rolling_metrics)

    print("\n[4/5] 检查特征漂移...")
    drift_info = check_feature_drift(lab, daily_metrics)

    print("\n[5/5] 生成优化配置...")
    config = generate_optimized_config(thresholds, confidence_scale, drift_info, daily_metrics)

    print_optimization_report(config)
    save_optimized_config(config)

    print("\n✓ 优化完成！")


if __name__ == "__main__":
    main()
