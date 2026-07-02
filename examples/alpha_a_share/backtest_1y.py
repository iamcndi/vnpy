"""
多因子 Alpha 一年回测系统

基于 predict_daily.py 的多因子逻辑（Alpha158 + LightGBM），回测最近一年。

流程：
  1. 加载日线数据 → 构建 Alpha158 因子数据集
  2. 训练期训练 LightGBM → 回测期滚动预测
  3. 用模型预测值做多前 1/3 股票
  4. 输出统计指标

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/backtest_1y.py
"""

import os
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import polars as pl

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from vnpy.alpha import AlphaLab, BacktestingEngine, AlphaStrategy, logger
from vnpy.alpha.dataset import Segment
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.lgb_model import LGBAlphaModel
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData

from datafeed import download_daily_data


# ═══════════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════════

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")

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

# 一年回测期：2025-06-29 ~ 2026-06-29
BACKTEST_START = "2025-06-29"
BACKTEST_END   = "2026-06-29"

# 训练期：从 2023-01-01 到回测期开始前
TRAIN_START = "2023-01-01"
TRAIN_END   = "2025-06-28"

LOOKBACK_DAYS = 90
INITIAL_CAPITAL = 1_000_000
ANNUAL_TRADING_DAYS = 250
LONG_TOP_N_RATIO = 1.0 / 3.0
FORECAST_DAYS = 3

COMMISSION_RATE_BUY  = 0.00025
COMMISSION_RATE_SELL = 0.00125


# ═══════════════════════════════════════════════════════════════
# 2. 构建数据集
# ═══════════════════════════════════════════════════════════════

def build_dataset_df(
    lab: AlphaLab,
    vt_symbols: list[str],
    start: str,
    end: str,
    lookback_days: int,
) -> pl.DataFrame:
    """构建归一化的因子数据集 DataFrame"""
    start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=lookback_days)
    end_dt   = datetime.strptime(end, "%Y-%m-%d")

    records: list[dict] = []
    for vt_symbol in vt_symbols:
        bars = lab.load_bar_data(vt_symbol, Interval.DAILY, start_dt, end_dt)
        if not bars:
            continue
        for bar in bars:
            records.append({
                "datetime": bar.datetime,
                "vt_symbol": vt_symbol,
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
                "turnover": bar.turnover,
                "open_interest": bar.open_interest or 0.0,
            })

    df = pl.DataFrame(records).sort(["datetime", "vt_symbol"])

    # 统一价格归一化
    first_close = df.group_by("vt_symbol").agg(pl.col("close").first().alias("close_0"))
    df = df.join(first_close, on="vt_symbol")

    for col in ("open", "high", "low", "close"):
        df = df.with_columns((pl.col(col) / pl.col("close_0")).alias(col))

    df = df.with_columns(
        ((pl.col("turnover") / (pl.col("volume") + 1e-12)) / pl.col("close_0")).alias("vwap")
    )
    df = df.drop("close_0")

    # 停牌日置 NaN
    numeric_cols = [c for c in df.columns if c not in ("datetime", "vt_symbol")]
    mask = df.select(pl.sum_horizontal(pl.col(c) for c in numeric_cols)) == 0
    df = df.with_columns(
        [pl.when(mask.to_series()).then(float("nan")).otherwise(pl.col(c)).alias(c)
         for c in numeric_cols]
    )

    return df


# ═══════════════════════════════════════════════════════════════
# 3. ML 信号策略
# ═══════════════════════════════════════════════════════════════

class MLSignalStrategy(AlphaStrategy):
    """ML 信号选股：做多预测信号前 1/3 股票"""

    price_add_pct = 0.003

    def on_init(self) -> None:
        self.write_log("ML 信号策略初始化完成")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        signal = self.get_signal()
        if signal.is_empty():
            return

        signal = signal.sort("signal", descending=True)
        n_long = max(1, int(len(signal) * LONG_TOP_N_RATIO))
        long_set = set(signal.head(n_long)["vt_symbol"].to_list())

        portfolio_value = self.get_portfolio_value()
        capital_per_stock = portfolio_value / n_long

        for vt_symbol, bar in bars.items():
            if vt_symbol in long_set and bar.close_price > 0:
                target = int(capital_per_stock / bar.close_price)
            else:
                target = 0
            self.set_target(vt_symbol, target)

        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade) -> None:
        pass


# ═══════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  多因子 Alpha 一年回测 (Alpha158 + LightGBM)")
    print(f"  回测期: {BACKTEST_START} ~ {BACKTEST_END}")
    print("=" * 60)

    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]

    # ── Step 1: 更新数据 ─────────────────────────────────────
    print("\n[1/6] 检查/下载最新日线数据...")
    download_daily_data(lab, STOCK_LIST, TRAIN_START)

    # ── Step 2: 构建因子数据集 ───────────────────────────────
    print("\n[2/6] 构建因子数据集...")
    df = build_dataset_df(lab, vt_symbols, TRAIN_START, BACKTEST_END, LOOKBACK_DAYS)
    print(f"  数据范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"  行数: {len(df)}, 股票数: {df['vt_symbol'].n_unique()}")

    # ── Step 3: Alpha158 因子 ────────────────────────────────
    print("\n[3/6] 创建 Alpha158 因子数据集（训练/回测分窗）...")
    dataset = Alpha158(
        df=df,
        train_period=(TRAIN_START, TRAIN_END),
        valid_period=(BACKTEST_START, BACKTEST_END),
        test_period=(BACKTEST_START, BACKTEST_END),
    )
    print(f"  训练期: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  回测期: {BACKTEST_START} ~ {BACKTEST_END}")

    # ── Step 4: 计算因子 ─────────────────────────────────────
    print("\n[4/6] 并行计算 158 个因子 + 标签...")
    dataset.prepare_data(max_workers=4)
    print(f"  因子列数: {len(dataset.feature_expressions)}")

    # ── Step 5: 训练模型并生成信号 ───────────────────────────
    print("\n[5/6] 训练 LightGBM 模型...")
    model = LGBAlphaModel()
    model.fit(dataset)

    # 生成回测期预测信号
    preds = model.predict(dataset, Segment.TEST)
    test_raw = dataset.fetch_raw(Segment.TEST)

    signal_df = test_raw.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series("signal", preds)
    )
    print(f"  信号记录数: {len(signal_df)}")
    print(f"  信号日期范围: {signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")

    # 因子重要性 Top10
    if model.model:
        importance = sorted(
            zip(model.feature_cols, model.model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        print(f"\n  因子重要性 Top10:")
        for name, imp in importance[:10]:
            print(f"    {name:15s}: {imp}")

    # ── Step 6: 回测 ─────────────────────────────────────────
    print("\n[6/6] 运行回测...")
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(BACKTEST_START, "%Y-%m-%d"),
        end=datetime.strptime(BACKTEST_END, "%Y-%m-%d"),
        capital=INITIAL_CAPITAL,
        risk_free=0.0,
        annual_days=ANNUAL_TRADING_DAYS,
    )
    engine.add_strategy(MLSignalStrategy, {}, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics = engine.calculate_statistics()

    # ── 输出 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  多因子 Alpha 一年回测结果")
    print("=" * 60)
    for k, v in statistics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>18.2f}")
        else:
            print(f"  {k:30s}: {v}")
    print("-" * 60)
    print(f"  交易笔数:     {engine.trade_count:>15d}")
    print(f"  因子数量:     {len(model.feature_cols):>15d}")

    # 保存模型
    MODEL_SAVE_NAME = "lgb_alpha_158_1y"
    lab.save_model(MODEL_SAVE_NAME, model)
    feat_path = lab.model_path.parent.joinpath("feature_cols_1y.json")
    with open(feat_path, "w") as f:
        json.dump(model.feature_cols, f)
    print(f"\n  模型已保存: {MODEL_SAVE_NAME}")
    print(f"  特征列已保存: {feat_path}")
    print("\n✓ 回测完成!")


if __name__ == "__main__":
    main()
