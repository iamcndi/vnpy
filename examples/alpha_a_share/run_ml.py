"""
ML 因子学习回测系统

使用 Alpha158 因子 + LightGBM 模型，在训练期学习因子权重，在回测期
用模型预测值作为选股信号，验证 ML 因子合成的效果。

流程：
  1. 加载日线数据 → 构建 Alpha158 因子数据集
  2. 按时间轴划分训练/回测期（历史训练 → 近一年回测）
  3. 分别训练 1日 / 3日 / 5日持有期 LightGBM 模型
  4. 用主持有期(3日)预测值做多前 1/3 股票并回测
  5. 输出统计指标

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/run_ml.py
"""

import os
import sys
import json
from datetime import datetime, timedelta

import polars as pl

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from vnpy.alpha import (
    AlphaLab, AlphaStrategy,
    BacktestingEngine,
)
from vnpy.alpha.dataset import Segment
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.dataset.template import calculate_feature
from vnpy.trader.constant import Interval
from vnpy.trader.object import BarData

from datafeed import download_daily_data, data_end_date
from stock_universe import STOCK_LIST
from horizons import (
    FORECAST_HORIZONS,
    PRIMARY_HORIZON,
    horizon_label,
    model_name,
    feature_cols_name,
    LEGACY_MODEL_NAME,
    LEGACY_FEATURE_COLS,
)
from vnpy.alpha.model.lgb_model import LGBAlphaModel


# ═══════════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════════

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")

TRAIN_START = "2023-01-01"
BACKTEST_DAYS = 365
LOOKBACK_DAYS = 90

INITIAL_CAPITAL = 1_000_000
LONG_TOP_N_RATIO = 1.0 / 3.0
COMMISSION_RATE_BUY = 0.00025
COMMISSION_RATE_SELL = 0.00125


def setup_contracts(lab: AlphaLab) -> None:
    """为股票池写入回测所需的价格精度/费率"""
    for code, exchange_str, _name in STOCK_LIST:
        lab.add_contract_setting(
            f"{code}.{exchange_str}",
            long_rate=COMMISSION_RATE_BUY,
            short_rate=COMMISSION_RATE_SELL,
            size=1,
            pricetick=0.01,
        )
    print(f"  ✓ 已配置 {len(STOCK_LIST)} 个合约信息")


def resolve_periods(lab: AlphaLab, vt_symbols: list[str]) -> tuple[str, str, str, str]:
    """根据本地日线最新日期，确定训练/回测区间"""
    latest: datetime | None = None
    for vt_symbol in vt_symbols:
        bars = lab.load_bar_data(
            vt_symbol, Interval.DAILY, datetime(2023, 1, 1), datetime.now()
        )
        if bars:
            dt = bars[-1].datetime
            if latest is None or dt > latest:
                latest = dt

    if latest is None:
        test_end = data_end_date()
    else:
        test_end = latest.strftime("%Y-%m-%d")

    test_end_dt = datetime.strptime(test_end, "%Y-%m-%d")
    test_start_dt = test_end_dt - timedelta(days=BACKTEST_DAYS - 1)
    train_end_dt = test_start_dt - timedelta(days=1)

    train_start_dt = datetime.strptime(TRAIN_START, "%Y-%m-%d")
    if train_end_dt <= train_start_dt:
        train_end_dt = train_start_dt + timedelta(days=180)
        test_start_dt = train_end_dt + timedelta(days=1)

    return (
        TRAIN_START,
        train_end_dt.strftime("%Y-%m-%d"),
        test_start_dt.strftime("%Y-%m-%d"),
        test_end,
    )


def build_dataset_df(
    lab: AlphaLab,
    vt_symbols: list[str],
    lookback_start: str,
    data_end: str,
) -> pl.DataFrame:
    """构建归一化的因子数据集 DataFrame"""
    start_dt = datetime.strptime(lookback_start, "%Y-%m-%d")
    end_dt = datetime.strptime(data_end, "%Y-%m-%d")

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


def rebind_label(dataset: Alpha158, days: int) -> None:
    """复用已算好的因子，仅替换持有期标签"""
    dataset.set_label(horizon_label(days))
    label_series = calculate_feature((dataset.df, "label", dataset.label_expression))
    dataset.result_df = dataset.result_df.with_columns(label_series)

    raw_df = dataset.result_df.fill_null(float("nan"))
    select_columns: list[str] = ["datetime", "vt_symbol"] + raw_df.columns[dataset.df.width:]
    dataset.raw_df = raw_df.select(select_columns).sort(["datetime", "vt_symbol"])
    dataset.infer_df = dataset.raw_df
    dataset.learn_df = dataset.raw_df


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


def train_horizon_models(
    lab: AlphaLab,
    dataset: Alpha158,
) -> dict[int, LGBAlphaModel]:
    """训练各持有期模型并保存；因子只算一次，标签按持有期替换"""
    models: dict[int, LGBAlphaModel] = {}

    for i, days in enumerate(FORECAST_HORIZONS):
        print(f"\n  ── 训练 {days} 日持有期模型 ({i + 1}/{len(FORECAST_HORIZONS)}) ──")
        rebind_label(dataset, days)

        model = LGBAlphaModel()
        model.fit(dataset)
        models[days] = model

        name = model_name(days)
        lab.save_model(name, model)
        feat_path = lab.model_path.parent.joinpath(feature_cols_name(days))
        with open(feat_path, "w") as f:
            json.dump(model.feature_cols, f)
        print(f"    ✓ 已保存: {name}  (特征 {len(model.feature_cols)} 列)")

        if days == PRIMARY_HORIZON:
            lab.save_model(LEGACY_MODEL_NAME, model)
            legacy_feat = lab.model_path.parent.joinpath(LEGACY_FEATURE_COLS)
            with open(legacy_feat, "w") as f:
                json.dump(model.feature_cols, f)
            print(f"    ✓ 兼容旧名: {LEGACY_MODEL_NAME}")

            if model.model:
                importance = sorted(
                    zip(model.feature_cols, model.model.feature_importances_),
                    key=lambda x: x[1], reverse=True,
                )
                print("    因子重要性 Top10:")
                for fname, imp in importance[:10]:
                    print(f"      {fname:15s}: {imp}")

    return models


def main() -> None:
    print("=" * 60)
    print("  ML 因子学习回测系统 (LightGBM + Alpha158 多持有期)")
    print(f"  持有期: {FORECAST_HORIZONS} 日  |  主排序/回测: {PRIMARY_HORIZON} 日")
    print("=" * 60)

    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]

    print("\n[0/5] 更新日线数据...")
    download_daily_data(lab, STOCK_LIST, TRAIN_START)

    train_start, train_end, test_start, test_end = resolve_periods(lab, vt_symbols)

    print("\n[1/5] 构建因子数据集...")
    data_start_dt = datetime.strptime(TRAIN_START, "%Y-%m-%d") - timedelta(days=LOOKBACK_DAYS)
    data_start = data_start_dt.strftime("%Y-%m-%d")

    df = build_dataset_df(lab, vt_symbols, data_start, test_end)
    print(f"  数据范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"  行数: {len(df)}, 股票数: {df['vt_symbol'].n_unique()}")

    print("\n[2/5] 创建 Alpha158 因子数据集（训练/回测分窗）...")
    dataset = Alpha158(
        df=df,
        train_period=(train_start, train_end),
        valid_period=(test_start, test_end),
        test_period=(test_start, test_end),
    )
    dataset.set_label(horizon_label(PRIMARY_HORIZON))
    print(f"  训练期: {train_start} ~ {train_end}")
    print(f"  回测期: {test_start} ~ {test_end}")

    print("\n[3/5] 计算 158 个因子 + 主持有期标签...")
    dataset.prepare_data(max_workers=4)
    print(f"  因子列数: {len(dataset.feature_expressions)}")

    print("\n[4/5] 训练多持有期 LightGBM 模型...")
    models = train_horizon_models(lab, dataset)
    primary_model = models[PRIMARY_HORIZON]

    rebind_label(dataset, PRIMARY_HORIZON)
    preds = primary_model.predict(dataset, Segment.TEST)
    test_raw = dataset.fetch_raw(Segment.TEST)
    signal_df = test_raw.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series("signal", preds)
    )
    print(f"  主持有期信号记录数: {len(signal_df)}")

    print(f"\n[5/5] 运行回测 (持有期={PRIMARY_HORIZON}日)...")
    setup_contracts(lab)
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(test_start, "%Y-%m-%d"),
        end=datetime.strptime(test_end, "%Y-%m-%d"),
        capital=INITIAL_CAPITAL,
        risk_free=0.0,
        annual_days=250,
    )
    engine.add_strategy(MLSignalStrategy, {}, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    try:
        statistics = engine.calculate_statistics()
    except AttributeError as exc:
        print(f"  ⚠ 回测统计失败（模型已保存，可继续预测）: {exc}")
        print("\n✓ 多持有期模型训练完成!")
        return

    print("\n" + "=" * 60)
    print(f"  ML 回测结果 (持有期={PRIMARY_HORIZON}日信号)")
    print("=" * 60)
    for k, v in statistics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>18.2f}")
        else:
            print(f"  {k:30s}: {v}")
    print("-" * 60)
    print(f"  交易笔数:  {engine.trade_count:>14d}")
    print(f"  因子数量:  {len(primary_model.feature_cols):>14d}")
    print(f"  已保存模型: {', '.join(model_name(d) for d in FORECAST_HORIZONS)}")

    print("\n提示:")
    print("  engine.show_chart()                     # 资金曲线")
    print("  engine.show_performance('000300.SSE')   # vs 沪深300")
    print("  predict_daily.py 将同时输出 1/3/5 日预期收益")
    print("\n✓ 回测完成!")


if __name__ == "__main__":
    main()
