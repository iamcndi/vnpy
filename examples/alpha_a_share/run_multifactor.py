"""
多因子 Alpha 回测系统

使用 Alpha158 因子库（158 个量价因子），构建复合选股信号，回测 A 股。

流程：
  1. 从 parquet 加载日线数据 → 构建归一化 DataFrame
  2. 创建 Alpha158 数据集 → 并行计算 158 个因子
  3. 截面排序归一化 → 多因子等权合成信号
  4. 运行组合回测（做多信号前 1/3 股票）
  5. 输出统计指标

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/run_multifactor.py
"""

import os
import sys
from datetime import datetime, timedelta
from functools import partial

import polars as pl

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from vnpy.alpha import AlphaLab, BacktestingEngine, AlphaStrategy, logger
from vnpy.alpha.dataset import Segment, process_cs_rank_norm
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData


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

BACKTEST_START = "2023-01-01"
BACKTEST_END   = "2025-06-20"

INITIAL_CAPITAL = 1_000_000
ANNUAL_TRADING_DAYS = 250
LONG_TOP_N_RATIO = 1.0 / 3.0
LOOKBACK_DAYS = 90          # 因子计算需要的额外历史窗口

COMMISSION_RATE_BUY  = 0.00025
COMMISSION_RATE_SELL = 0.00125


# ═══════════════════════════════════════════════════════════════
# 2. 构建归一化因子数据集
# ═══════════════════════════════════════════════════════════════

def build_dataset_df(
    lab: AlphaLab,
    vt_symbols: list[str],
    start: str,
    end: str,
    lookback_days: int,
) -> pl.DataFrame:
    """构建因子数据集 DataFrame（统一价格归一化）"""
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

    # 统一价格归一化：除以各股票首个收盘价
    first_close = df.group_by("vt_symbol").agg(pl.col("close").first().alias("close_0"))
    df = df.join(first_close, on="vt_symbol")

    for col in ("open", "high", "low", "close"):
        df = df.with_columns((pl.col(col) / pl.col("close_0")).alias(col))

    # VWAP 也做归一化
    df = df.with_columns(
        ((pl.col("turnover") / (pl.col("volume") + 1e-12)) / pl.col("close_0")).alias("vwap")
    )

    df = df.drop("close_0")

    # 停牌日置为 NaN
    numeric_cols = [c for c in df.columns if c not in ("datetime", "vt_symbol")]
    mask = df.select(pl.sum_horizontal(pl.col(c) for c in numeric_cols)) == 0
    df = df.with_columns(
        [pl.when(mask.to_series()).then(float("nan")).otherwise(pl.col(c)).alias(c)
         for c in numeric_cols]
    )

    return df


# ═══════════════════════════════════════════════════════════════
# 3. 策略
# ═══════════════════════════════════════════════════════════════

class MultiFactorStrategy(AlphaStrategy):
    """多因子选股：复合信号做多前 N% 股票，等权持仓"""

    price_add_pct = 0.003

    def on_init(self) -> None:
        self.write_log("多因子策略初始化完成")

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
    print("  多因子 Alpha 回测系统 (Alpha158)")
    print("=" * 60)

    lab = AlphaLab(ALPHA_LAB_PATH)
    start_dt = datetime.strptime(BACKTEST_START, "%Y-%m-%d")
    end_dt   = datetime.strptime(BACKTEST_END, "%Y-%m-%d")
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]

    # ── Step 1: 加载数据 ─────────────────────────────────────
    print("\n[1/5] 加载数据并构建因子数据集...")
    df = build_dataset_df(lab, vt_symbols, BACKTEST_START, BACKTEST_END, LOOKBACK_DAYS)
    print(f"  数据范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"  行数: {len(df)}, 股票数: {df['vt_symbol'].n_unique()}")
    print(f"  列: {df.columns}")

    # ── Step 2: Alpha158 多因子 ──────────────────────────────
    print("\n[2/5] 创建 Alpha158 多因子数据集...")
    dataset = Alpha158(
        df=df,
        train_period=(BACKTEST_START, BACKTEST_END),
        valid_period=(BACKTEST_END, BACKTEST_END),
        test_period=(BACKTEST_END, BACKTEST_END),
    )
    print(f"  注册因子: {len(dataset.feature_expressions)} 个")

    # ── Step 3: 并行计算因子 ────────────────────────────────
    print("\n[3/5] 并行计算 158 个因子...")
    dataset.prepare_data(max_workers=4)

    # ── Step 4: 构建复合信号 ────────────────────────────────
    print("\n[4/5] 构建复合因子信号...")

    # 取出原始因子数据（含 datetime, vt_symbol + 原始列 + 因子列）
    raw_df = dataset.fetch_raw(Segment.TRAIN)
    feature_cols = [c for c in raw_df.columns if c not in ("datetime", "vt_symbol")]

    # 选择部分代表性因子：各窗口的动量、趋势、波动率、价量相关等
    selected_prefixes = ("roc_", "ma_", "std_", "beta_", "corr_",
                         "cord_", "cntp_", "sump_", "rsv_")
    selected = [c for c in feature_cols
                if any(c.startswith(p) for p in selected_prefixes)
                and not c.startswith("open_")
                and not c.startswith("open_")]
    print(f"  总因子数: {len(feature_cols)}")
    print(f"  选用于信号: {len(selected)} 个")

    # 截面排序归一化（每个因子在每个时点做 cs_rank → [0, 1]）
    rank_df = raw_df.select(["datetime", "vt_symbol"])
    for i, name in enumerate(selected):
        col_rank = raw_df.select([
            "datetime", "vt_symbol",
            pl.col(name).rank("ordinal").over("datetime").cast(pl.Float64).alias(f"r{i}")
        ])
        rank_df = rank_df.join(col_rank, on=["datetime", "vt_symbol"], how="left")

    # 等权加总 → 复合信号
    rank_cols = [f"r{i}" for i in range(len(selected))]
    signal_df = rank_df.with_columns(
        pl.sum_horizontal(rank_cols).alias("signal")
    ).select(["datetime", "vt_symbol", "signal"])

    print(f"  信号记录数: {len(signal_df)}")
    print(f"  信号日期范围: {signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")

    # ── Step 5: 回测 ────────────────────────────────────────
    print("\n[5/5] 运行回测...")
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=start_dt,
        end=end_dt,
        capital=INITIAL_CAPITAL,
        risk_free=0.0,
        annual_days=ANNUAL_TRADING_DAYS,
    )
    engine.add_strategy(MultiFactorStrategy, {}, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics = engine.calculate_statistics()

    # ── 输出 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  多因子 Alpha 回测结果")
    print("=" * 60)
    for k, v in statistics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>18.2f}")
        else:
            print(f"  {k:30s}: {v}")
    print("-" * 60)
    print(f"  交易笔数:     {engine.trade_count:>15d}")
    print(f"  因子数量:     {len(selected):>15d}")
    print(f"  日志条数:     {len(engine.logs):>15d}")

    # ── 提示 ─────────────────────────────────────────────────
    print("\n提示:")
    print("  engine.show_chart()                        # 资金曲线")
    print("  engine.show_performance('000300.SSE')      # vs 沪深300")
    print("\n✓ 回测完成!")


if __name__ == "__main__":
    main()
