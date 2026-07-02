"""
多因子 Alpha 一年回测 + 止损止盈 + 回撤清仓熔断

策略逻辑：
  1. 个股止损 7%：持仓个股亏损达 7% → 卖出该股
  2. 个股止盈 20%：持仓个股盈利达 20% → 卖出该股
  3. 组合回撤熔断 10%：组合总市值较峰值回撤达 10% → 清仓所有持仓
  4. 熔断冷却期 5 个交易日后恢复交易

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/backtest_1y_circuit.py
"""

import os
import sys
import json
from datetime import datetime, timedelta
from collections import defaultdict

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
from vnpy.trader.constant import Interval, Exchange, Direction, Offset
from vnpy.trader.object import BarData, TradeData

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

BACKTEST_START = "2025-06-29"
BACKTEST_END   = "2026-06-29"
TRAIN_START = "2023-01-01"
TRAIN_END   = "2025-06-28"

LOOKBACK_DAYS = 90
INITIAL_CAPITAL = 1_000_000
ANNUAL_TRADING_DAYS = 250
LONG_TOP_N_RATIO = 1.0 / 3.0

COMMISSION_RATE_BUY  = 0.00025
COMMISSION_RATE_SELL = 0.00125

# 策略参数
STOP_LOSS_PCT = 0.07       # 个股止损 7%
TAKE_PROFIT_PCT = 0.20     # 个股止盈 20%
PORTFOLIO_DD_PCT = 0.10    # 组合回撤清仓阈值 10%
COOLDOWN_DAYS = 5          # 熔断后冷却交易日数


# ═══════════════════════════════════════════════════════════════
# 2. 构建数据集
# ═══════════════════════════════════════════════════════════════

def build_dataset_df(lab, vt_symbols, start, end, lookback_days):
    start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=lookback_days)
    end_dt   = datetime.strptime(end, "%Y-%m-%d")

    records = []
    for vt_symbol in vt_symbols:
        bars = lab.load_bar_data(vt_symbol, Interval.DAILY, start_dt, end_dt)
        if not bars:
            continue
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
# 3. 带熔断的策略
# ═══════════════════════════════════════════════════════════════

class CircuitBreakerStrategy(AlphaStrategy):
    """ML 信号选股 + 个股止损止盈 + 组合回撤清仓熔断"""

    price_add_pct = 0.003
    stop_loss_pct = STOP_LOSS_PCT
    take_profit_pct = TAKE_PROFIT_PCT
    portfolio_dd_pct = PORTFOLIO_DD_PCT
    cooldown_days = COOLDOWN_DAYS

    def on_init(self) -> None:
        self.cost_basis: dict[str, float] = {}
        self.sltp_triggered: dict[str, bool] = defaultdict(bool)
        # 组合级熔断状态
        self.high_watermark: float = 0.0        # 组合市值峰值
        self.circuit_triggered: bool = False     # 是否已触发熔断
        self.circuit_day_count: int = 0          # 熔断后天数
        self.circuit_trigger_count: int = 0      # 熔断触发总次数
        self.circuit_trigger_dates: list[str] = []  # 熔断触发日期
        self.write_log(
            f"熔断策略初始化 SL={self.stop_loss_pct:.0%} TP={self.take_profit_pct:.0%} "
            f"DD={self.portfolio_dd_pct:.0%} 冷却={self.cooldown_days}天"
        )

    def on_bars(self, bars: dict[str, BarData]) -> None:
        portfolio_value = self.get_portfolio_value()

        # ── 0. 更新组合峰值 ──────────────────────────────────
        if portfolio_value > self.high_watermark:
            self.high_watermark = portfolio_value

        # ── 1. 熔断冷却期 ───────────────────────────────────
        if self.circuit_triggered:
            self.circuit_day_count += 1
            # 冷却期内清仓
            for vt_symbol in bars:
                self.set_target(vt_symbol, 0)
            self.execute_trading(bars, price_add=self.price_add_pct)
            # 冷却期结束
            if self.circuit_day_count >= self.cooldown_days:
                self.circuit_triggered = False
                self.circuit_day_count = 0
                # 重置峰值：以当前市值为新基准，避免反复触发
                self.high_watermark = self.get_portfolio_value()
                self.write_log("熔断冷却期结束，恢复交易，重置峰值基准")
            return

        # ── 2. 检查组合回撤熔断 ─────────────────────────────
        if self.high_watermark > 0:
            drawdown_pct = (portfolio_value - self.high_watermark) / self.high_watermark
            if drawdown_pct <= -self.portfolio_dd_pct:
                self.circuit_triggered = True
                self.circuit_day_count = 0
                self.circuit_trigger_count += 1
                dt_str = str(self.strategy_engine.datetime.date()) if self.strategy_engine.datetime else "?"
                self.circuit_trigger_dates.append(dt_str)
                self.write_log(
                    f"★ 组合回撤熔断! 峰值={self.high_watermark:,.0f} "
                    f"现值={portfolio_value:,.0f} 回撤={drawdown_pct:.1%}"
                )
                # 清仓所有
                for vt_symbol in bars:
                    self.set_target(vt_symbol, 0)
                self.execute_trading(bars, price_add=self.price_add_pct)
                return

        # ── 3. 检查个股止损止盈 ──────────────────────────────
        sltp_sell_set: set[str] = set()
        for vt_symbol, bar in bars.items():
            pos = self.get_pos(vt_symbol)
            if pos <= 0:
                self.cost_basis.pop(vt_symbol, None)
                self.sltp_triggered.pop(vt_symbol, None)
                continue

            cost = self.cost_basis.get(vt_symbol, 0)
            if cost <= 0:
                continue

            # 前一日已触发 → 今日强制卖出
            if self.sltp_triggered.get(vt_symbol, False):
                sltp_sell_set.add(vt_symbol)
                self.sltp_triggered.pop(vt_symbol, None)
                continue

            pnl_pct = (bar.close_price - cost) / cost
            if pnl_pct <= -self.stop_loss_pct:
                sltp_sell_set.add(vt_symbol)
                self.write_log(f"止损: {vt_symbol} 成本={cost:.2f} 现价={bar.close_price:.2f} 亏损={pnl_pct:.1%}")
            elif pnl_pct >= self.take_profit_pct:
                sltp_sell_set.add(vt_symbol)
                self.write_log(f"止盈: {vt_symbol} 成本={cost:.2f} 现价={bar.close_price:.2f} 盈利={pnl_pct:.1%}")

        # ── 4. 信号选股 ──────────────────────────────────────
        signal = self.get_signal()
        long_set: set[str] = set()
        n_long = 1
        if not signal.is_empty():
            signal_sorted = signal.sort("signal", descending=True)
            n_long = max(1, int(len(signal_sorted) * LONG_TOP_N_RATIO))
            long_set = set(signal_sorted.head(n_long)["vt_symbol"].to_list())

        # ── 5. 计算目标持仓 ──────────────────────────────────
        effective_long = long_set - sltp_sell_set
        n_effective = max(1, len(effective_long))
        capital_per_stock = portfolio_value / n_effective

        for vt_symbol, bar in bars.items():
            if vt_symbol in sltp_sell_set:
                self.set_target(vt_symbol, 0)
            elif vt_symbol in effective_long and bar.close_price > 0:
                target = int(capital_per_stock / bar.close_price)
                self.set_target(vt_symbol, target)
            else:
                self.set_target(vt_symbol, 0)

        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade: TradeData) -> None:
        if trade.direction == Direction.LONG:
            vt_symbol = trade.vt_symbol
            old_pos = self.get_pos(vt_symbol) - trade.volume
            old_cost = self.cost_basis.get(vt_symbol, 0)
            if old_pos <= 0:
                self.cost_basis[vt_symbol] = trade.price
            else:
                total = old_pos * old_cost + trade.volume * trade.price
                self.cost_basis[vt_symbol] = total / (old_pos + trade.volume)
        elif trade.direction == Direction.SHORT:
            vt_symbol = trade.vt_symbol
            remaining = self.get_pos(vt_symbol)
            if remaining <= 0:
                self.cost_basis.pop(vt_symbol, None)


# ═══════════════════════════════════════════════════════════════
# 4. 对比策略（无熔断，仅个股止损止盈）
# ═══════════════════════════════════════════════════════════════

class SLTPOnlyStrategy(AlphaStrategy):
    """ML 信号选股 + 个股止损止盈（无组合熔断）"""

    price_add_pct = 0.003
    stop_loss_pct = STOP_LOSS_PCT
    take_profit_pct = TAKE_PROFIT_PCT

    def on_init(self) -> None:
        self.cost_basis: dict[str, float] = {}
        self.sltp_triggered: dict[str, bool] = defaultdict(bool)
        self.write_log(f"止损止盈策略初始化 SL={self.stop_loss_pct:.0%} TP={self.take_profit_pct:.0%}")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        signal = self.get_signal()

        sltp_sell_set: set[str] = set()
        for vt_symbol, bar in bars.items():
            pos = self.get_pos(vt_symbol)
            if pos <= 0:
                self.cost_basis.pop(vt_symbol, None)
                self.sltp_triggered.pop(vt_symbol, None)
                continue
            cost = self.cost_basis.get(vt_symbol, 0)
            if cost <= 0:
                continue
            if self.sltp_triggered.get(vt_symbol, False):
                sltp_sell_set.add(vt_symbol)
                self.sltp_triggered.pop(vt_symbol, None)
                continue
            pnl_pct = (bar.close_price - cost) / cost
            if pnl_pct <= -self.stop_loss_pct:
                sltp_sell_set.add(vt_symbol)
            elif pnl_pct >= self.take_profit_pct:
                sltp_sell_set.add(vt_symbol)

        long_set: set[str] = set()
        n_long = 1
        if not signal.is_empty():
            signal_sorted = signal.sort("signal", descending=True)
            n_long = max(1, int(len(signal_sorted) * LONG_TOP_N_RATIO))
            long_set = set(signal_sorted.head(n_long)["vt_symbol"].to_list())

        effective_long = long_set - sltp_sell_set
        n_effective = max(1, len(effective_long))
        portfolio_value = self.get_portfolio_value()
        capital_per_stock = portfolio_value / n_effective

        for vt_symbol, bar in bars.items():
            if vt_symbol in sltp_sell_set:
                self.set_target(vt_symbol, 0)
            elif vt_symbol in effective_long and bar.close_price > 0:
                target = int(capital_per_stock / bar.close_price)
                self.set_target(vt_symbol, target)
            else:
                self.set_target(vt_symbol, 0)

        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade: TradeData) -> None:
        if trade.direction == Direction.LONG:
            vt_symbol = trade.vt_symbol
            old_pos = self.get_pos(vt_symbol) - trade.volume
            old_cost = self.cost_basis.get(vt_symbol, 0)
            if old_pos <= 0:
                self.cost_basis[vt_symbol] = trade.price
            else:
                total = old_pos * old_cost + trade.volume * trade.price
                self.cost_basis[vt_symbol] = total / (old_pos + trade.volume)
        elif trade.direction == Direction.SHORT:
            vt_symbol = trade.vt_symbol
            if self.get_pos(vt_symbol) <= 0:
                self.cost_basis.pop(vt_symbol, None)


# ═══════════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  多因子 Alpha 一年回测 + 回撤清仓熔断")
    print(f"  回测期: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"  个股止损={STOP_LOSS_PCT:.0%} 止盈={TAKE_PROFIT_PCT:.0%}")
    print(f"  组合回撤清仓={PORTFOLIO_DD_PCT:.0%} 冷却={COOLDOWN_DAYS}天")
    print("=" * 60)

    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]

    # ── Step 1: 更新数据 ─────────────────────────────────────
    print("\n[1/4] 检查/下载最新日线数据...")
    download_daily_data(lab, STOCK_LIST, TRAIN_START)

    # ── Step 2: 构建因子 + 训练模型 ──────────────────────────
    print("\n[2/4] 构建因子数据集 + 训练 LightGBM...")
    df = build_dataset_df(lab, vt_symbols, TRAIN_START, BACKTEST_END, LOOKBACK_DAYS)
    print(f"  数据范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"  行数: {len(df)}, 股票数: {df['vt_symbol'].n_unique()}")

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
    print(f"  信号记录数: {len(signal_df)}")

    # ── Step 3: 无熔断基准回测 ──────────────────────────────
    print("\n[3/4] 基准回测 (个股止损止盈，无组合熔断)...")
    engine_base = BacktestingEngine(lab)
    engine_base.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(BACKTEST_START, "%Y-%m-%d"),
        end=datetime.strptime(BACKTEST_END, "%Y-%m-%d"),
        capital=INITIAL_CAPITAL,
        risk_free=0.0,
        annual_days=ANNUAL_TRADING_DAYS,
    )
    engine_base.add_strategy(SLTPOnlyStrategy, {}, signal_df)
    engine_base.load_data()
    engine_base.run_backtesting()
    engine_base.calculate_result()
    stats_base = engine_base.calculate_statistics()

    # ── Step 4: 熔断策略回测 ────────────────────────────────
    print("\n[4/4] 熔断策略回测 (个股止损止盈 + 组合回撤清仓)...")
    engine_cb = BacktestingEngine(lab)
    engine_cb.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(BACKTEST_START, "%Y-%m-%d"),
        end=datetime.strptime(BACKTEST_END, "%Y-%m-%d"),
        capital=INITIAL_CAPITAL,
        risk_free=0.0,
        annual_days=ANNUAL_TRADING_DAYS,
    )
    engine_cb.add_strategy(CircuitBreakerStrategy, {}, signal_df)
    engine_cb.load_data()
    engine_cb.run_backtesting()
    engine_cb.calculate_result()
    stats_cb = engine_cb.calculate_statistics()

    # 提取熔断事件信息
    cb_strategy = engine_cb.strategy
    circuit_count = cb_strategy.circuit_trigger_count
    circuit_dates = cb_strategy.circuit_trigger_dates

    # ── 输出对比 ─────────────────────────────────────────────
    keys = [
        "start_date", "end_date", "total_days", "profit_days", "loss_days",
        "capital", "end_balance", "max_drawdown", "max_ddpercent",
        "max_drawdown_duration", "total_net_pnl", "total_commission",
        "total_trade_count", "total_return", "annual_return",
        "daily_return", "return_std", "sharpe_ratio", "return_drawdown_ratio",
    ]

    print("\n" + "=" * 70)
    print("  回测结果对比")
    print("=" * 70)
    print(f"  {'指标':30s}  {'无熔断(基准)':>16s}  {'回撤清仓熔断':>16s}")
    print(f"  {'-'*30}  {'-'*16}  {'-'*16}")
    for k in keys:
        v_base = stats_base.get(k, 0)
        v_cb = stats_cb.get(k, 0)
        if isinstance(v_base, float) or isinstance(v_cb, float):
            print(f"  {k:30s}  {v_base:>16.2f}  {v_cb:>16.2f}")
        else:
            print(f"  {k:30s}  {str(v_base):>16s}  {str(v_cb):>16s}")

    # 改善幅度
    ret_diff = stats_cb.get("annual_return", 0) - stats_base.get("annual_return", 0)
    dd_diff = stats_cb.get("max_ddpercent", 0) - stats_base.get("max_ddpercent", 0)
    sharpe_diff = stats_cb.get("sharpe_ratio", 0) - stats_base.get("sharpe_ratio", 0)
    rdd_diff = stats_cb.get("return_drawdown_ratio", 0) - stats_base.get("return_drawdown_ratio", 0)

    print(f"\n  {'改善幅度':30s}  {'年化收益':>16s}  {'最大回撤':>16s}  {'夏普':>10s}  {'收益回撤比':>10s}")
    print(f"  {'':30s}  {ret_diff:>+16.2f}  {dd_diff:>+16.2f}  {sharpe_diff:>+10.2f}  {rdd_diff:>+10.2f}")

    # 熔断事件
    print(f"\n  熔断触发次数: {circuit_count}")
    if circuit_dates:
        print(f"  熔断触发日期: {', '.join(circuit_dates)}")

    # 保存结果
    result = {
        "params": {
            "stop_loss": STOP_LOSS_PCT,
            "take_profit": TAKE_PROFIT_PCT,
            "portfolio_dd": PORTFOLIO_DD_PCT,
            "cooldown_days": COOLDOWN_DAYS,
        },
        "baseline": stats_base,
        "circuit_breaker": stats_cb,
        "circuit_events": {
            "count": circuit_count,
            "dates": circuit_dates,
        },
    }
    result_path = os.path.join(ALPHA_LAB_PATH, "circuit_breaker_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  结果已保存: {result_path}")
    print("\n✓ 回测完成!")


if __name__ == "__main__":
    main()
