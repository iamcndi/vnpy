"""
多因子 Alpha 一年回测 + 移动止盈 + 回撤清仓熔断

策略逻辑：
  1. 个股止损 7%：相对成本价亏损达 7% → 卖出该股
  2. 移动止盈：持仓最高价相对成本先盈利 A% 后，从最高点回撤 B% → 卖出
     （不再使用固定上限止盈）
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

from stock_universe import STOCK_LIST
from datafeed import download_daily_data


# ═══════════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════════

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")


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
STOP_LOSS_PCT = 0.07       # 个股止损 7%（相对成本）
PORTFOLIO_DD_PCT = 0.10    # 组合回撤清仓阈值 10%
COOLDOWN_DAYS = 5          # 熔断后冷却交易日数

# 移动止盈网格：先盈利 A%，再从持仓最高点回撤 B%
TRAIL_ACTIVATE_GRID = [0.10, 0.15, 0.20, 0.25]   # A
TRAIL_PULLBACK_GRID = [0.05, 0.08, 0.12]         # B


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


def setup_contracts(lab: AlphaLab) -> None:
    for code, exchange_str, _name in STOCK_LIST:
        lab.add_contract_setting(
            f"{code}.{exchange_str}",
            long_rate=COMMISSION_RATE_BUY,
            short_rate=COMMISSION_RATE_SELL,
            size=1,
            pricetick=0.01,
        )


def check_stop_and_trail(strategy, bars: dict[str, BarData]) -> set[str]:
    """个股止损 + 移动止盈，返回当日应清仓的股票"""
    sell_set: set[str] = set()
    for vt_symbol, bar in bars.items():
        pos = strategy.get_pos(vt_symbol)
        if pos <= 0:
            strategy.cost_basis.pop(vt_symbol, None)
            strategy.high_price.pop(vt_symbol, None)
            strategy.trail_armed.pop(vt_symbol, None)
            continue

        cost = strategy.cost_basis.get(vt_symbol, 0)
        if cost <= 0:
            continue

        peak = max(strategy.high_price.get(vt_symbol, bar.high_price), bar.high_price)
        strategy.high_price[vt_symbol] = peak

        pnl_pct = (bar.close_price - cost) / cost
        if pnl_pct <= -strategy.stop_loss_pct:
            sell_set.add(vt_symbol)
            continue

        if (peak - cost) / cost >= strategy.trail_activate_pct:
            strategy.trail_armed[vt_symbol] = True

        if strategy.trail_armed.get(vt_symbol) and peak > 0:
            pullback = (peak - bar.close_price) / peak
            if pullback >= strategy.trail_pullback_pct:
                sell_set.add(vt_symbol)
    return sell_set


def update_cost_and_high(strategy, trade: TradeData) -> None:
    vt_symbol = trade.vt_symbol
    if trade.direction == Direction.LONG:
        old_pos = strategy.get_pos(vt_symbol) - trade.volume
        old_cost = strategy.cost_basis.get(vt_symbol, 0)
        if old_pos <= 0:
            strategy.cost_basis[vt_symbol] = trade.price
            strategy.high_price[vt_symbol] = trade.price
            strategy.trail_armed[vt_symbol] = False
        else:
            total = old_pos * old_cost + trade.volume * trade.price
            strategy.cost_basis[vt_symbol] = total / (old_pos + trade.volume)
            strategy.high_price[vt_symbol] = max(
                strategy.high_price.get(vt_symbol, trade.price), trade.price
            )
    elif trade.direction == Direction.SHORT:
        if strategy.get_pos(vt_symbol) <= 0:
            strategy.cost_basis.pop(vt_symbol, None)
            strategy.high_price.pop(vt_symbol, None)
            strategy.trail_armed.pop(vt_symbol, None)


# ═══════════════════════════════════════════════════════════════
# 3. 带熔断的策略
# ═══════════════════════════════════════════════════════════════

class CircuitBreakerStrategy(AlphaStrategy):
    """ML 信号选股 + 个股止损 + 移动止盈 + 组合回撤清仓熔断"""

    price_add_pct = 0.003
    stop_loss_pct = STOP_LOSS_PCT
    trail_activate_pct = 0.15
    trail_pullback_pct = 0.08
    portfolio_dd_pct = PORTFOLIO_DD_PCT
    cooldown_days = COOLDOWN_DAYS

    def on_init(self) -> None:
        self.cost_basis: dict[str, float] = {}
        self.high_price: dict[str, float] = {}
        self.trail_armed: dict[str, bool] = {}
        self.high_watermark: float = 0.0
        self.circuit_triggered: bool = False
        self.circuit_day_count: int = 0
        self.circuit_trigger_count: int = 0
        self.circuit_trigger_dates: list[str] = []
        self.write_log(
            f"熔断策略初始化 SL={self.stop_loss_pct:.0%} "
            f"移动止盈 A={self.trail_activate_pct:.0%} B={self.trail_pullback_pct:.0%} "
            f"DD={self.portfolio_dd_pct:.0%} 冷却={self.cooldown_days}天"
        )

    def on_bars(self, bars: dict[str, BarData]) -> None:
        portfolio_value = self.get_portfolio_value()

        if portfolio_value > self.high_watermark:
            self.high_watermark = portfolio_value

        if self.circuit_triggered:
            self.circuit_day_count += 1
            for vt_symbol in bars:
                self.set_target(vt_symbol, 0)
            self.execute_trading(bars, price_add=self.price_add_pct)
            if self.circuit_day_count >= self.cooldown_days:
                self.circuit_triggered = False
                self.circuit_day_count = 0
                self.high_watermark = self.get_portfolio_value()
                self.write_log("熔断冷却期结束，恢复交易，重置峰值基准")
            return

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
                for vt_symbol in bars:
                    self.set_target(vt_symbol, 0)
                self.execute_trading(bars, price_add=self.price_add_pct)
                return

        sltp_sell_set = check_stop_and_trail(self, bars)

        signal = self.get_signal()
        long_set: set[str] = set()
        if not signal.is_empty():
            signal_sorted = signal.sort("signal", descending=True)
            n_long = max(1, int(len(signal_sorted) * LONG_TOP_N_RATIO))
            long_set = set(signal_sorted.head(n_long)["vt_symbol"].to_list())

        effective_long = long_set - sltp_sell_set
        n_effective = max(1, len(effective_long))
        capital_per_stock = portfolio_value / n_effective

        for vt_symbol, bar in bars.items():
            if vt_symbol in sltp_sell_set:
                self.set_target(vt_symbol, 0)
            elif vt_symbol in effective_long and bar.close_price > 0:
                self.set_target(vt_symbol, int(capital_per_stock / bar.close_price))
            else:
                self.set_target(vt_symbol, 0)

        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade: TradeData) -> None:
        update_cost_and_high(self, trade)


class SLTPOnlyStrategy(AlphaStrategy):
    """ML 信号选股 + 个股止损 + 移动止盈（无组合熔断）"""

    price_add_pct = 0.003
    stop_loss_pct = STOP_LOSS_PCT
    trail_activate_pct = 0.15
    trail_pullback_pct = 0.08

    def on_init(self) -> None:
        self.cost_basis: dict[str, float] = {}
        self.high_price: dict[str, float] = {}
        self.trail_armed: dict[str, bool] = {}
        self.write_log(
            f"移动止盈策略初始化 SL={self.stop_loss_pct:.0%} "
            f"A={self.trail_activate_pct:.0%} B={self.trail_pullback_pct:.0%}"
        )

    def on_bars(self, bars: dict[str, BarData]) -> None:
        sltp_sell_set = check_stop_and_trail(self, bars)

        long_set: set[str] = set()
        signal = self.get_signal()
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
                self.set_target(vt_symbol, int(capital_per_stock / bar.close_price))
            else:
                self.set_target(vt_symbol, 0)

        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade: TradeData) -> None:
        update_cost_and_high(self, trade)


def run_one(lab, vt_symbols, signal_df, strategy_class, setting: dict) -> tuple[dict, object]:
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
    engine.add_strategy(strategy_class, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    try:
        stats = engine.calculate_statistics()
    except AttributeError:
        stats = {}
    return stats, engine


def main() -> None:
    print("=" * 70)
    print("  多因子 Alpha 一年回测 + 移动止盈网格 + 回撤清仓熔断")
    print(f"  回测期: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"  个股止损={STOP_LOSS_PCT:.0%}  组合回撤清仓={PORTFOLIO_DD_PCT:.0%}  冷却={COOLDOWN_DAYS}天")
    print(f"  移动止盈 A={TRAIL_ACTIVATE_GRID}  B={TRAIL_PULLBACK_GRID}")
    print("=" * 70)

    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]
    setup_contracts(lab)

    print("\n[1/3] 检查/下载最新日线数据...")
    download_daily_data(lab, STOCK_LIST, TRAIN_START)

    print("\n[2/3] 构建因子数据集 + 训练 LightGBM...")
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

    print("\n[3/3] 网格回测 (熔断固定，扫描 A/B)...")
    rows: list[dict] = []
    total = len(TRAIL_ACTIVATE_GRID) * len(TRAIL_PULLBACK_GRID)
    i = 0
    for a in TRAIL_ACTIVATE_GRID:
        for b in TRAIL_PULLBACK_GRID:
            i += 1
            setting = {
                "stop_loss_pct": STOP_LOSS_PCT,
                "trail_activate_pct": a,
                "trail_pullback_pct": b,
                "portfolio_dd_pct": PORTFOLIO_DD_PCT,
                "cooldown_days": COOLDOWN_DAYS,
            }
            print(f"  ({i}/{total}) A={a:.0%} B={b:.0%} ...", flush=True)
            stats, engine = run_one(lab, vt_symbols, signal_df, CircuitBreakerStrategy, setting)
            strategy = engine.strategy
            row = {
                "A": a,
                "B": b,
                "annual_return": stats.get("annual_return", float("nan")),
                "total_return": stats.get("total_return", float("nan")),
                "sharpe_ratio": stats.get("sharpe_ratio", float("nan")),
                "max_ddpercent": stats.get("max_ddpercent", float("nan")),
                "return_drawdown_ratio": stats.get("return_drawdown_ratio", float("nan")),
                "end_balance": stats.get("end_balance", float("nan")),
                "total_trade_count": stats.get("total_trade_count", 0),
                "circuit_count": getattr(strategy, "circuit_trigger_count", 0),
                "circuit_dates": getattr(strategy, "circuit_trigger_dates", []),
            }
            rows.append(row)
            print(
                f"      年化={row['annual_return']:.2f}%  夏普={row['sharpe_ratio']:.2f}  "
                f"回撤={row['max_ddpercent']:.2f}%  熔断={row['circuit_count']}次"
            )

    rows_sorted = sorted(
        rows,
        key=lambda r: (r["sharpe_ratio"] if r["sharpe_ratio"] == r["sharpe_ratio"] else -999),
        reverse=True,
    )

    print("\n" + "=" * 88)
    print("  移动止盈网格结果（按夏普排序）")
    print("=" * 88)
    print(
        f"  {'A':>6s}  {'B':>6s}  {'年化%':>10s}  {'总收益%':>10s}  {'夏普':>8s}  "
        f"{'最大回撤%':>10s}  {'收益回撤比':>10s}  {'成交':>6s}  {'熔断':>4s}"
    )
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*4}")
    for r in rows_sorted:
        print(
            f"  {r['A']:>5.0%}  {r['B']:>5.0%}  {r['annual_return']:>10.2f}  {r['total_return']:>10.2f}  "
            f"{r['sharpe_ratio']:>8.2f}  {r['max_ddpercent']:>10.2f}  {r['return_drawdown_ratio']:>10.2f}  "
            f"{r['total_trade_count']:>6}  {r['circuit_count']:>4}"
        )

    best = rows_sorted[0]
    print(
        f"\n  夏普最高: A={best['A']:.0%} B={best['B']:.0%}  "
        f"年化={best['annual_return']:.2f}%  夏普={best['sharpe_ratio']:.2f}"
    )
    if best["circuit_dates"]:
        print(f"  该组熔断日期: {', '.join(best['circuit_dates'])}")

    result_path = os.path.join(ALPHA_LAB_PATH, "circuit_breaker_result.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "params": {
                    "stop_loss": STOP_LOSS_PCT,
                    "portfolio_dd": PORTFOLIO_DD_PCT,
                    "cooldown_days": COOLDOWN_DAYS,
                    "trail_activate_grid": TRAIL_ACTIVATE_GRID,
                    "trail_pullback_grid": TRAIL_PULLBACK_GRID,
                },
                "grid": rows_sorted,
                "best_by_sharpe": best,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  结果已保存: {result_path}")
    print("\n✓ 回测完成!")


if __name__ == "__main__":
    main()
