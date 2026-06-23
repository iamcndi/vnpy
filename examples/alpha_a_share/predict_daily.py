
"""
每日预测：加载训练好的 LightGBM 模型 → 获取最新数据 → 计算因子 → 输出选股信号

用法：每天收盘后执行：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/predict_daily.py

输出：控制台打印今日预测信号排名 + 推荐买入列表
      alpha_data/signal/ 下保存预测信号
      (后续可运行 evaluate_predictions.py 评估预测质量)
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
from vnpy.alpha.model.template import AlphaModel
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData

from vnpy.alpha.model.lgb_model import LGBAlphaModel
import lightgbm as lgb

from datafeed import download_daily_data


# ═══════════════════════════════════════════════════════════════
# 配置（基础 - 优化配置会从 optimized_config.json 读取）
# ═══════════════════════════════════════════════════════════════

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")
MODEL_NAME = "lgb_alpha_158"
LOOKBACK_DAYS = 90
BASE_LONG_TOP_N_RATIO = 1.0 / 3.0
OPTIMIZED_CONFIG_PATH = os.path.join(ALPHA_LAB_PATH, "optimized_config.json")

STOCK_LIST: list[tuple[str, str, str]] = [
    ("002245", "SZSE", "蔚蓝锂芯"),
    ("600487", "SSE", "亨通光电"),
    ("600089", "SSE", "特变电工"),
    ("002532", "SZSE", "天山铝业"),
    ("300316", "SZSE", "晶盛机电"),
    ("300843", "SZSE", "胜蓝股份"),
    ("300438", "SZSE", "鹏辉能源"),
    ("000338", "SZSE", "潍柴动力"),
    ("300661", "SZSE", "圣邦股份"),
    ("300507", "SZSE", "苏奥传感"),
    ("301511", "SZSE", "德福科技"),
    ("300442", "SZSE", "润泽科技"),
    ("301498", "SZSE", "乖宝宠物"),
    ("002299", "SZSE", "圣农发展"),
    ("601717", "SSE", "中创智领"),
    ("002639", "SZSE", "雪人集团"),
    ("601665", "SSE", "齐鲁银行"),
    ("600580", "SSE", "卧龙电驱"),
    ("301217", "SZSE", "铜冠铜箔"),
    ("300484", "SZSE", "蓝海华腾"),
    ("518800", "SSE", "黄金ETF国泰"),
    ("688523", "SSE", "航天环宇"),
    ("300433", "SZSE", "蓝思科技"),
    ("300811", "SZSE", "铂科新材"),
    ("300136", "SZSE", "信维通信"),
]

PREDICTION_WINDOW_DAYS = 30   # 因子计算窗口


# ═══════════════════════════════════════════════════════════════
# 构建预测数据集
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# 加载优化配置（自适应参数）
# ═══════════════════════════════════════════════════════════════

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
            # 合并默认值，保证兼容性
            for key in default:
                if key not in config:
                    config[key] = default[key]
            for key in default["optimized_params"]:
                if key not in config.get("optimized_params", {}):
                    config.setdefault("optimized_params", {})[key] = default["optimized_params"][key]

        print(f"  ✓ 加载优化配置 (选股比率={config['optimized_params']['long_top_n_ratio']:.2f}, "
              f"置信度={config['optimized_params']['confidence_scale']:.2f}x)")

        if config["model_status"].get("retrain_needed", False):
            print(f"  ⚠ 模型需要重训练！运行 run_ml.py 重新训练")

        return config
    except Exception as e:
        print(f"  ⚠ 加载优化配置失败: {e}，使用默认参数")
        return default


# ═══════════════════════════════════════════════════════════════
# 加载模型并预测
# ═══════════════════════════════════════════════════════════════

def load_model_and_cols(lab: AlphaLab) -> tuple:
    """加载保存的模型和特征列名"""
    model = lab.load_model(MODEL_NAME)
    if model is None:
        print("✗ 模型未找到，请先运行 run_ml.py 训练模型")
        sys.exit(1)

    feat_path = lab.model_path.parent.joinpath("feature_cols.json")
    with open(feat_path) as f:
        feature_cols = json.load(f)

    return model, feature_cols


def predict(lab: AlphaLab, vt_symbols: list[str], opt_config: dict) -> None:
    """预测并输出选股信号"""
    print("构建最新因子数据集...")
    df = build_df(lab, vt_symbols)
    print(f"  数据行数: {len(df)}, 日期: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # 取最新的完整交易月作为"预测期"（实际只取最后一天的预测结果）
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

    # 加载模型
    model_loaded, feature_cols = load_model_and_cols(lab)
    feat_df = dataset.fetch_raw(Segment.TRAIN)

    # 只取最新一天做预测
    latest_feat = feat_df.filter(pl.col("datetime") == latest_date)
    if latest_feat.is_empty():
        print(f"✗ 最新日期 {latest_date} 没有因子数据")
        return

    X = latest_feat.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 模型预测
    lgb_model = model_loaded.model
    preds = lgb_model.predict(X)

    # 整理结果：包含原始预测值和归一化信号
    result = latest_feat.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series("predicted_return", preds)
    )

    # 读取优化参数
    optim = opt_config["optimized_params"]
    LONG_TOP_N_RATIO = optim["long_top_n_ratio"]
    MIN_SIGNAL_THRESHOLD = optim.get("min_signal_threshold", 0.0)
    CONFIDENCE_SCALE = optim.get("confidence_scale", 1.0)

    # 构建信号（min-max归一化到 [0, 1]）
    pred_min = result["predicted_return"].min()
    pred_max = result["predicted_return"].max()
    signal_range = pred_max - pred_min + 1e-12

    result = result.with_columns(
        ((pl.col("predicted_return") - pred_min) / signal_range).alias("signal")
    ).sort("predicted_return", descending=True)

    # 应用最小信号阈值过滤（IC低时收紧，只保留高确信度信号）
    if MIN_SIGNAL_THRESHOLD > 0:
        filtered = result.filter(pl.col("signal") >= MIN_SIGNAL_THRESHOLD)
        if len(filtered) >= 3:
            print(f"  信号阈值过滤: {len(result)} → {len(filtered)} 只 (阈值={MIN_SIGNAL_THRESHOLD:.2f})")
            result = filtered

    n_long = max(1, int(len(result) * LONG_TOP_N_RATIO))

    # ── 输出 ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  选股信号 — {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    print(f"  {'排名':>4s}  {'股票':>12s}  {'预测收益':>10s}  {'信号':>6s}  {'操作':>6s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*6}")

    buy_list = result.head(n_long)

    for rank, row in enumerate(result.iter_rows(named=True), 1):
        vt = row["vt_symbol"]
        ret = row["predicted_return"]
        sig = row["signal"]
        if rank <= n_long:
            action = "BUY  "
        elif rank > len(result) - n_long:
            action = "SELL "
        else:
            action = "HOLD "

        name = ""
        for code, exch, n in STOCK_LIST:
            if f"{code}.{exch}" == vt:
                name = n
                break

        print(f"  {rank:>4d}  {vt:>12s}  {ret*100:>+8.2f}%  {sig:>6.3f}  {action}")

    print(f"{'='*60}")
    print(f"  推荐买入 ({len(buy_list)} 只):")
    for row in buy_list.iter_rows(named=True):
        vt = row["vt_symbol"]
        ret = row["predicted_return"]
        sig = row["signal"]
        name = ""
        for code, exch, n in STOCK_LIST:
            if f"{code}.{exch}" == vt:
                name = n
                break
        print(f"    {vt} ({name})  →  预测收益 {ret*100:+.2f}%,  信号 {sig:.3f}")

    # 保存信号到文件（包含 raw 预测值 + 归一化信号）
    signal_save = result.select([
        "datetime", "vt_symbol",
        "predicted_return",
        "signal",
    ])

    lab.save_signal(f"lgb_pred_{latest_date.strftime('%Y%m%d')}", signal_save)
    print(f"\n  信号已保存至 AlphaLab")
    print(f"\n  提示: 运行 evaluate_predictions.py 可评估预测质量")
    print(f"        运行 optimize_prediction.py 可自适应优化参数")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  LightGBM 每日选股预测")
    print("=" * 60)

    # 加载优化配置（如果存在）
    opt_config = load_optimized_config()
    if opt_config["model_status"].get("retrain_needed", False):
        print("  ⚠ 建议尽快重训练模型")

    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]

    # Step 1: 更新数据
    print("检查并更新最新行情数据...")
    download_daily_data(lab, STOCK_LIST)

    # Step 2: 预测并输出
    predict(lab, vt_symbols, opt_config)


if __name__ == "__main__":
    main()
