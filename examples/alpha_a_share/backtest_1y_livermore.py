"""
利弗莫尔趋势管仓一年回测（小资金版：4万 + Top5 + 100股整手）

同区间强制对比：
  基准 A: ML 等权 Top5
  基准 B: ML 等权 Top5 + 移动止盈（A=15% / B=8%，止损 7%）
  本策略: ML Top5 候选 + 利弗莫尔管仓（加仓间距 X 网格）

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/backtest_1y_livermore.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import polars as pl

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from vnpy.alpha import AlphaLab, AlphaStrategy, BacktestingEngine
from vnpy.alpha.dataset import Segment
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.lgb_model import LGBAlphaModel
from vnpy.trader.constant import Direction, Interval
from vnpy.trader.object import BarData, TradeData

from datafeed import download_daily_data
from stock_universe import STOCK_LIST


ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")

BACKTEST_START = "2025-06-29"
BACKTEST_END = "2026-06-29"
TRAIN_START = "2023-01-01"
TRAIN_END = "2025-06-28"

LOOKBACK_DAYS = 90
INITIAL_CAPITAL = 40_000          # 小资金实盘约束
ANNUAL_TRADING_DAYS = 250
TOP_N = 5                         # 只做信号最强的 N 只
MIN_LOT = 100                     # A 股整手

COMMISSION_RATE_BUY = 0.00025
COMMISSION_RATE_SELL = 0.00125

BREAKOUT_WINDOW = 20
# 小资金下金字塔缩为两档，避免加仓金额远小于 1 手
PYRAMID_FRACTIONS = (0.60, 0.40)
ADD_SPACING_GRID = [0.05, 0.08, 0.10]

STOP_LOSS_PCT = 0.07
TRAIL_ACTIVATE_PCT = 0.15
TRAIL_PULLBACK_PCT = 0.08
# 早期盈利区 (+3%~+15%)：峰值浮盈达 arm 后、移动止盈激活前，从高点回撤 X% → 清仓
EARLY_ZONE_ARM_PCT = 0.03
EARLY_ZONE_MAX_PCT = TRAIL_ACTIVATE_PCT  # 与移动止盈激活线一致，之上交给 15%/8% 规则
EARLY_ZONE_PULLBACK_GRID = [0.02, 0.03, 0.05, 0.08, 0.10]
# 盈利锁利（固定回落线）：曾浮盈 ≥3% 后回撤至 ≤1% → 清仓（旧测试）
PROFIT_LOCK_ARM_PCT = 0.03
PROFIT_LOCK_EXIT_PCT = 0.01


def round_lot(shares: float | int, lot: int = MIN_LOT) -> int:
    """向下取整到整手；不足一手则为 0"""
    s = int(shares)
    if s < lot:
        return 0
    return s // lot * lot


def select_top_n(signal: pl.DataFrame, n: int = TOP_N) -> tuple[set[str], int]:
    if signal.is_empty():
        return set(), 1
    signal = signal.sort("signal", descending=True)
    n_long = max(1, min(n, len(signal)))
    return set(signal.head(n_long)["vt_symbol"].to_list()), n_long


def build_dataset_df(lab, vt_symbols, start, end, lookback_days):
    start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=lookback_days)
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    records = []
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


def setup_contracts(lab: AlphaLab) -> None:
    for code, exchange_str, _name in STOCK_LIST:
        lab.add_contract_setting(
            f"{code}.{exchange_str}",
            long_rate=COMMISSION_RATE_BUY,
            short_rate=COMMISSION_RATE_SELL,
            size=1,
            pricetick=0.01,
        )


def update_cost_and_high(strategy, trade: TradeData) -> None:
    vt_symbol = trade.vt_symbol
    if trade.direction == Direction.LONG:
        old_pos = strategy.get_pos(vt_symbol) - trade.volume
        old_cost = strategy.cost_basis.get(vt_symbol, 0.0)
        if old_pos <= 0:
            strategy.cost_basis[vt_symbol] = trade.price
            strategy.high_price[vt_symbol] = trade.price
        else:
            total = old_pos * old_cost + trade.volume * trade.price
            strategy.cost_basis[vt_symbol] = total / (old_pos + trade.volume)
            strategy.high_price[vt_symbol] = max(
                strategy.high_price.get(vt_symbol, trade.price), trade.price
            )
        if hasattr(strategy, "last_buy_price"):
            strategy.last_buy_price[vt_symbol] = trade.price
    elif trade.direction == Direction.SHORT:
        if strategy.get_pos(vt_symbol) <= 0:
            strategy.cost_basis.pop(vt_symbol, None)
            strategy.high_price.pop(vt_symbol, None)
            for attr in ("last_buy_price", "pyramid_stage", "trail_armed", "halved", "profit_lock_armed"):
                mapping = getattr(strategy, attr, None)
                if isinstance(mapping, dict):
                    mapping.pop(vt_symbol, None)


class EqualWeightStrategy(AlphaStrategy):
    """基准 A：ML 等权 TopN + 整手"""

    price_add_pct = 0.003

    def on_init(self) -> None:
        self.write_log(f"基准A 等权策略初始化 Top{TOP_N} 整手{MIN_LOT}")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        signal = self.get_signal()
        if signal.is_empty():
            return
        long_set, n_long = select_top_n(signal)
        capital_per_stock = self.get_portfolio_value() / n_long
        for vt_symbol, bar in bars.items():
            if vt_symbol in long_set and bar.close_price > 0:
                self.set_target(vt_symbol, round_lot(capital_per_stock / bar.close_price))
            else:
                self.set_target(vt_symbol, 0)
        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade: TradeData) -> None:
        pass


class TrailEqualWeightStrategy(AlphaStrategy):
    """基准 B：ML 等权 TopN + 移动止盈全清（无减半）+ 整手"""

    price_add_pct = 0.003
    stop_loss_pct = STOP_LOSS_PCT
    trail_activate_pct = TRAIL_ACTIVATE_PCT
    trail_pullback_pct = TRAIL_PULLBACK_PCT

    def on_init(self) -> None:
        self.cost_basis: dict[str, float] = {}
        self.high_price: dict[str, float] = {}
        self.last_buy_price: dict[str, float] = {}
        self.trail_armed: dict[str, bool] = {}
        self.write_log(
            f"基准B 移动止盈初始化 Top{TOP_N} SL={self.stop_loss_pct:.0%} "
            f"A={self.trail_activate_pct:.0%} B={self.trail_pullback_pct:.0%}"
        )

    def on_bars(self, bars: dict[str, BarData]) -> None:
        sell_set: set[str] = set()
        for vt_symbol, bar in bars.items():
            pos = self.get_pos(vt_symbol)
            if pos <= 0:
                self.cost_basis.pop(vt_symbol, None)
                self.high_price.pop(vt_symbol, None)
                self.trail_armed.pop(vt_symbol, None)
                continue
            cost = self.cost_basis.get(vt_symbol, 0.0)
            if cost <= 0:
                continue
            peak = max(self.high_price.get(vt_symbol, bar.high_price), bar.high_price)
            self.high_price[vt_symbol] = peak
            pnl = (bar.close_price - cost) / cost
            if pnl <= -self.stop_loss_pct:
                sell_set.add(vt_symbol)
                continue
            if (peak - cost) / cost >= self.trail_activate_pct:
                self.trail_armed[vt_symbol] = True
            if self.trail_armed.get(vt_symbol) and peak > 0:
                if (peak - bar.close_price) / peak >= self.trail_pullback_pct:
                    sell_set.add(vt_symbol)

        signal = self.get_signal()
        long_set, n_long = select_top_n(signal)
        effective = long_set - sell_set
        n_eff = max(1, len(effective)) if effective else n_long
        capital_per_stock = self.get_portfolio_value() / n_eff
        for vt_symbol, bar in bars.items():
            if vt_symbol in sell_set:
                self.set_target(vt_symbol, 0)
            elif vt_symbol in effective and bar.close_price > 0:
                self.set_target(vt_symbol, round_lot(capital_per_stock / bar.close_price))
            else:
                self.set_target(vt_symbol, 0)
        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade: TradeData) -> None:
        update_cost_and_high(self, trade)


class LivermoreStrategy(AlphaStrategy):
    """ML 候选 + 利弗莫尔：突破建仓、金字塔加仓、止损清仓、回撤先减半再清"""

    price_add_pct = 0.003
    breakout_window = BREAKOUT_WINDOW
    add_spacing_pct = 0.08
    stop_loss_pct = STOP_LOSS_PCT
    trail_activate_pct = TRAIL_ACTIVATE_PCT
    trail_pullback_pct = TRAIL_PULLBACK_PCT
    pyramid_fractions = PYRAMID_FRACTIONS
    profit_lock_arm_pct = 0.0   # >0 启用：峰值浮盈达此比例后武装
    profit_lock_exit_pct = 0.0  # 武装后浮盈回落至此比例 → 清仓（固定线模式）
    early_zone_pullback_pct = 0.0  # >0 启用：+arm~+trail 区间从高点回撤此比例 → 清仓
    early_zone_giveback_pct = 0.0  # >0 启用：早期区吐回峰值利润的此比例 → 清仓
    trail_giveback_pct = 0.0       # >0 启用：+15% 后吐回峰值利润的此比例 → 减半/清

    def on_init(self) -> None:
        self.cost_basis: dict[str, float] = {}
        self.high_price: dict[str, float] = {}
        self.last_buy_price: dict[str, float] = {}
        self.pyramid_stage: dict[str, int] = {}
        self.trail_armed: dict[str, bool] = {}
        self.halved: dict[str, bool] = {}
        self.profit_lock_armed: dict[str, bool] = {}
        self.close_history: dict[str, list[float]] = defaultdict(list)
        self.entry_count = 0
        self.add_count = 0
        self.half_count = 0
        self.stop_count = 0
        self.trail_clear_count = 0
        self.profit_lock_count = 0
        self.early_zone_exit_count = 0
        lock_msg = ""
        if self.early_zone_giveback_pct > 0:
            lock_msg = (
                f" 早期区={self.profit_lock_arm_pct:.0%}~{self.trail_activate_pct:.0%}"
                f" 吐回利润{self.early_zone_giveback_pct:.0%}清"
            )
        elif self.early_zone_pullback_pct > 0:
            lock_msg = (
                f" 早期区={self.profit_lock_arm_pct:.0%}~{self.trail_activate_pct:.0%}"
                f" 回撤{self.early_zone_pullback_pct:.0%}清"
            )
        elif self.profit_lock_arm_pct > 0 and self.profit_lock_exit_pct > 0:
            lock_msg = (
                f" 盈利锁利={self.profit_lock_arm_pct:.0%}→{self.profit_lock_exit_pct:.0%}"
            )
        trail_msg = (
            f" 移动吐回{self.trail_giveback_pct:.0%}"
            if self.trail_giveback_pct > 0
            else f" 移动A/B={self.trail_activate_pct:.0%}/{self.trail_pullback_pct:.0%}"
        )
        self.write_log(
            f"利弗莫尔策略初始化 N={self.breakout_window} X={self.add_spacing_pct:.0%} "
            f"金字塔={self.pyramid_fractions} SL={self.stop_loss_pct:.0%}"
            f"{trail_msg}{lock_msg}"
        )

    def _reset_symbol(self, vt_symbol: str) -> None:
        self.cost_basis.pop(vt_symbol, None)
        self.high_price.pop(vt_symbol, None)
        self.last_buy_price.pop(vt_symbol, None)
        self.pyramid_stage.pop(vt_symbol, None)
        self.trail_armed.pop(vt_symbol, None)
        self.halved.pop(vt_symbol, None)
        self.profit_lock_armed.pop(vt_symbol, None)

    def _is_breakout(self, vt_symbol: str, close: float) -> bool:
        hist = self.close_history[vt_symbol]
        window = hist[-self.breakout_window:]
        if len(window) < self.breakout_window:
            return False
        return close >= max(window) - 1e-12

    def _stage_target_shares(self, stage: int, stock_budget: float, price: float) -> int:
        if stage <= 0 or price <= 0:
            return 0
        frac = sum(self.pyramid_fractions[:stage])
        return round_lot(stock_budget * frac / price)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        for vt_symbol, bar in bars.items():
            if bar.close_price > 0:
                self.close_history[vt_symbol].append(bar.close_price)
                if len(self.close_history[vt_symbol]) > self.breakout_window + 5:
                    self.close_history[vt_symbol] = self.close_history[vt_symbol][
                        -(self.breakout_window + 5):
                    ]

        signal = self.get_signal()
        candidates, n_slots = select_top_n(signal)

        portfolio_value = self.get_portfolio_value()
        stock_budget = portfolio_value / n_slots

        targets: dict[str, int] = {vt: 0 for vt in bars}

        for vt_symbol, bar in bars.items():
            pos = self.get_pos(vt_symbol)
            price = bar.close_price
            if price <= 0:
                continue

            if pos <= 0:
                self._reset_symbol(vt_symbol)
                if vt_symbol in candidates and self._is_breakout(vt_symbol, price):
                    stage = 1
                    self.pyramid_stage[vt_symbol] = stage
                    targets[vt_symbol] = self._stage_target_shares(stage, stock_budget, price)
                continue

            cost = self.cost_basis.get(vt_symbol, 0.0)
            if cost <= 0:
                continue

            peak = max(self.high_price.get(vt_symbol, bar.high_price), bar.high_price)
            self.high_price[vt_symbol] = peak

            pnl = (price - cost) / cost
            if pnl <= -self.stop_loss_pct:
                targets[vt_symbol] = 0
                self.stop_count += 1
                continue

            peak_pnl = (peak - cost) / cost
            in_early_zone = (
                self.profit_lock_arm_pct <= peak_pnl < self.trail_activate_pct
            )

            if in_early_zone and peak > cost:
                if self.early_zone_giveback_pct > 0:
                    # 吐回峰值利润的 α：清仓价 = 高点 - (高点-成本)×α
                    if (peak - price) >= (peak - cost) * self.early_zone_giveback_pct:
                        targets[vt_symbol] = 0
                        self.early_zone_exit_count += 1
                        continue
                elif self.early_zone_pullback_pct > 0 and peak > 0:
                    pullback = (peak - price) / peak
                    if pullback >= self.early_zone_pullback_pct:
                        targets[vt_symbol] = 0
                        self.early_zone_exit_count += 1
                        continue

            if (
                self.profit_lock_arm_pct > 0
                and self.profit_lock_exit_pct > 0
                and self.early_zone_pullback_pct <= 0
                and self.early_zone_giveback_pct <= 0
            ):
                if peak_pnl >= self.profit_lock_arm_pct:
                    self.profit_lock_armed[vt_symbol] = True
                if self.profit_lock_armed.get(vt_symbol) and pnl <= self.profit_lock_exit_pct:
                    targets[vt_symbol] = 0
                    self.profit_lock_count += 1
                    continue

            if peak_pnl >= self.trail_activate_pct:
                self.trail_armed[vt_symbol] = True

            if self.trail_armed.get(vt_symbol) and peak > cost:
                if self.trail_giveback_pct > 0:
                    hit = (peak - price) >= (peak - cost) * self.trail_giveback_pct
                else:
                    hit = peak > 0 and (peak - price) / peak >= self.trail_pullback_pct
                if hit:
                    if not self.halved.get(vt_symbol, False):
                        half = round_lot(pos // 2)
                        # 不足一手则直接清仓
                        targets[vt_symbol] = half
                        self.halved[vt_symbol] = True
                        self.half_count += 1
                        if half <= 0:
                            self.trail_clear_count += 1
                    else:
                        targets[vt_symbol] = 0
                        self.trail_clear_count += 1
                    continue

            stage = self.pyramid_stage.get(vt_symbol, 1)
            if (
                not self.halved.get(vt_symbol, False)
                and stage < len(self.pyramid_fractions)
                and vt_symbol in candidates
            ):
                last_buy = self.last_buy_price.get(vt_symbol, cost)
                if last_buy > 0 and price >= last_buy * (1.0 + self.add_spacing_pct):
                    stage += 1
                    self.pyramid_stage[vt_symbol] = stage

            targets[vt_symbol] = self._stage_target_shares(stage, stock_budget, price)
            if self.halved.get(vt_symbol, False):
                targets[vt_symbol] = round_lot(pos)

        for vt_symbol, bar in bars.items():
            self.set_target(vt_symbol, targets.get(vt_symbol, 0))
        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade: TradeData) -> None:
        vt_symbol = trade.vt_symbol
        if trade.direction == Direction.LONG:
            old_pos = self.get_pos(vt_symbol) - trade.volume
            if old_pos <= 0:
                self.entry_count += 1
                self.pyramid_stage[vt_symbol] = max(self.pyramid_stage.get(vt_symbol, 1), 1)
                self.trail_armed[vt_symbol] = False
                self.halved[vt_symbol] = False
                self.profit_lock_armed[vt_symbol] = False
            else:
                self.add_count += 1
        update_cost_and_high(self, trade)


def run_one(lab, vt_symbols, signal_df, strategy_class, setting: dict | None = None):
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
    engine.add_strategy(strategy_class, setting or {}, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    try:
        stats = engine.calculate_statistics()
    except AttributeError:
        stats = {}
    return stats, engine


def pick_stats(stats: dict) -> dict:
    keys = [
        "annual_return", "total_return", "sharpe_ratio", "max_ddpercent",
        "return_drawdown_ratio", "end_balance", "total_trade_count",
    ]
    return {k: stats.get(k, float("nan")) for k in keys}


def main() -> None:
    print("=" * 78)
    print("  利弗莫尔趋势管仓一年回测（小资金 4万 / Top5 / 整手100）")
    print(f"  回测期: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"  资金={INITIAL_CAPITAL:,}  TopN={TOP_N}  整手={MIN_LOT}")
    print(f"  突破N={BREAKOUT_WINDOW}  金字塔={PYRAMID_FRACTIONS}")
    print(f"  止损={STOP_LOSS_PCT:.0%}  移动A/B={TRAIL_ACTIVATE_PCT:.0%}/{TRAIL_PULLBACK_PCT:.0%}")
    print(f"  加仓间距网格 X={ADD_SPACING_GRID}")
    print("=" * 78)

    lab = AlphaLab(ALPHA_LAB_PATH)
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]
    setup_contracts(lab)

    print("\n[1/4] 检查/下载日线...")
    download_daily_data(lab, STOCK_LIST, TRAIN_START)

    print("\n[2/4] 构建因子 + 训练 LightGBM...")
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

    print("\n[3/4] 基准 A/B 回测...")
    stats_a, _ = run_one(lab, vt_symbols, signal_df, EqualWeightStrategy)
    print(
        f"  基准A 等权: 年化={stats_a.get('annual_return', float('nan')):.2f}%  "
        f"夏普={stats_a.get('sharpe_ratio', float('nan')):.2f}  "
        f"回撤={stats_a.get('max_ddpercent', float('nan')):.2f}%"
    )
    stats_b, _ = run_one(lab, vt_symbols, signal_df, TrailEqualWeightStrategy)
    print(
        f"  基准B 移动止盈: 年化={stats_b.get('annual_return', float('nan')):.2f}%  "
        f"夏普={stats_b.get('sharpe_ratio', float('nan')):.2f}  "
        f"回撤={stats_b.get('max_ddpercent', float('nan')):.2f}%"
    )

    print("\n[4/4] 利弗莫尔网格回测...")
    liver_rows = []
    for i, x in enumerate(ADD_SPACING_GRID, 1):
        print(f"  ({i}/{len(ADD_SPACING_GRID)}) X={x:.0%} ...", flush=True)
        stats, engine = run_one(
            lab, vt_symbols, signal_df, LivermoreStrategy,
            {"add_spacing_pct": x},
        )
        st = engine.strategy
        row = {
            "X": x,
            **pick_stats(stats),
            "entry_count": getattr(st, "entry_count", 0),
            "add_count": getattr(st, "add_count", 0),
            "half_count": getattr(st, "half_count", 0),
            "stop_count": getattr(st, "stop_count", 0),
            "trail_clear_count": getattr(st, "trail_clear_count", 0),
            "delta_annual_vs_B": stats.get("annual_return", 0) - stats_b.get("annual_return", 0),
            "delta_sharpe_vs_B": stats.get("sharpe_ratio", 0) - stats_b.get("sharpe_ratio", 0),
            "delta_dd_vs_B": stats.get("max_ddpercent", 0) - stats_b.get("max_ddpercent", 0),
        }
        liver_rows.append(row)
        print(
            f"      年化={row['annual_return']:.2f}%  夏普={row['sharpe_ratio']:.2f}  "
            f"回撤={row['max_ddpercent']:.2f}%  "
            f"建仓={row['entry_count']} 加仓={row['add_count']} "
            f"减半={row['half_count']} 止损={row['stop_count']}"
        )

    liver_rows.sort(
        key=lambda r: r["sharpe_ratio"] if r["sharpe_ratio"] == r["sharpe_ratio"] else -999,
        reverse=True,
    )
    best = liver_rows[0]

    print("\n" + "=" * 98)
    print("  同区间对比（回测期相同）")
    print("=" * 98)
    print(
        f"  {'策略':22s}  {'年化%':>10s}  {'总收益%':>10s}  {'夏普':>8s}  "
        f"{'最大回撤%':>10s}  {'收益回撤比':>10s}  {'成交':>8s}"
    )
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")

    def _line(name: str, s: dict) -> None:
        print(
            f"  {name:22s}  {s.get('annual_return', float('nan')):>10.2f}  "
            f"{s.get('total_return', float('nan')):>10.2f}  "
            f"{s.get('sharpe_ratio', float('nan')):>8.2f}  "
            f"{s.get('max_ddpercent', float('nan')):>10.2f}  "
            f"{s.get('return_drawdown_ratio', float('nan')):>10.2f}  "
            f"{s.get('total_trade_count', 0):>8}"
        )

    _line("基准A ML等权", stats_a)
    _line("基准B ML+移动止盈", stats_b)
    for r in liver_rows:
        _line(f"利弗莫尔 X={r['X']:.0%}", r)

    print("\n  利弗莫尔相对基准B（按夏普最优）:")
    print(
        f"  最优 X={best['X']:.0%}  "
        f"Δ年化={best['delta_annual_vs_B']:+.2f}%  "
        f"Δ夏普={best['delta_sharpe_vs_B']:+.2f}  "
        f"Δ回撤={best['delta_dd_vs_B']:+.2f}%"
    )
    print(
        f"  事件: 建仓={best['entry_count']} 加仓={best['add_count']} "
        f"减半={best['half_count']} 止损清仓={best['stop_count']} "
        f"回撤清仓={best['trail_clear_count']}"
    )

    best_x = best["X"]
    print(f"\n[5/5] 盈利锁利规则对比 (X={best_x:.0%})...")
    print(
        f"  规则: 峰值浮盈 ≥{PROFIT_LOCK_ARM_PCT:.0%} 后，"
        f"回落到 ≤{PROFIT_LOCK_EXIT_PCT:.0%} → 清仓"
    )
    print(
        "  原策略: 此区间不会卖，除非 -7% 止损 或 +15% 后回撤 8% 移动止盈"
    )
    base_setting = {"add_spacing_pct": best_x}
    lock_setting = {
        "add_spacing_pct": best_x,
        "profit_lock_arm_pct": PROFIT_LOCK_ARM_PCT,
        "profit_lock_exit_pct": PROFIT_LOCK_EXIT_PCT,
    }
    stats_lm_base, eng_base = run_one(lab, vt_symbols, signal_df, LivermoreStrategy, base_setting)
    stats_lm_lock, eng_lock = run_one(lab, vt_symbols, signal_df, LivermoreStrategy, lock_setting)
    st_base = eng_base.strategy
    st_lock = eng_lock.strategy
    profit_lock_compare = {
        "X": best_x,
        "profit_lock_arm": PROFIT_LOCK_ARM_PCT,
        "profit_lock_exit": PROFIT_LOCK_EXIT_PCT,
        "original": {
            **pick_stats(stats_lm_base),
            "stop_count": getattr(st_base, "stop_count", 0),
            "half_count": getattr(st_base, "half_count", 0),
            "trail_clear_count": getattr(st_base, "trail_clear_count", 0),
            "profit_lock_count": 0,
        },
        "with_profit_lock": {
            **pick_stats(stats_lm_lock),
            "stop_count": getattr(st_lock, "stop_count", 0),
            "half_count": getattr(st_lock, "half_count", 0),
            "trail_clear_count": getattr(st_lock, "trail_clear_count", 0),
            "profit_lock_count": getattr(st_lock, "profit_lock_count", 0),
        },
        "delta": {
            "annual_return": stats_lm_lock.get("annual_return", 0) - stats_lm_base.get("annual_return", 0),
            "sharpe_ratio": stats_lm_lock.get("sharpe_ratio", 0) - stats_lm_base.get("sharpe_ratio", 0),
            "max_ddpercent": stats_lm_lock.get("max_ddpercent", 0) - stats_lm_base.get("max_ddpercent", 0),
            "total_trade_count": stats_lm_lock.get("total_trade_count", 0) - stats_lm_base.get("total_trade_count", 0),
        },
    }

    print("\n" + "=" * 98)
    print(f"  盈利锁利 vs 原利弗莫尔 (X={best_x:.0%})")
    print("=" * 98)
    _line(f"原利弗莫尔 X={best_x:.0%}", stats_lm_base)
    _line(
        f"利弗莫尔+锁利 {PROFIT_LOCK_ARM_PCT:.0%}→{PROFIT_LOCK_EXIT_PCT:.0%}",
        stats_lm_lock,
    )
    print(
        f"\n  原策略退出: 止损={profit_lock_compare['original']['stop_count']} "
        f"减半={profit_lock_compare['original']['half_count']} "
        f"回撤清={profit_lock_compare['original']['trail_clear_count']}"
    )
    print(
        f"  加锁利后: 止损={profit_lock_compare['with_profit_lock']['stop_count']} "
        f"减半={profit_lock_compare['with_profit_lock']['half_count']} "
        f"回撤清={profit_lock_compare['with_profit_lock']['trail_clear_count']} "
        f"锁利清={profit_lock_compare['with_profit_lock']['profit_lock_count']}"
    )
    print(
        f"  Δ年化={profit_lock_compare['delta']['annual_return']:+.2f}%  "
        f"Δ夏普={profit_lock_compare['delta']['sharpe_ratio']:+.2f}  "
        f"Δ回撤={profit_lock_compare['delta']['max_ddpercent']:+.2f}%  "
        f"Δ成交={profit_lock_compare['delta']['total_trade_count']:+}"
    )

    result_path = os.path.join(ALPHA_LAB_PATH, "livermore_result_40k.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "period": {"train": [TRAIN_START, TRAIN_END], "backtest": [BACKTEST_START, BACKTEST_END]},
                "params": {
                    "initial_capital": INITIAL_CAPITAL,
                    "top_n": TOP_N,
                    "min_lot": MIN_LOT,
                    "breakout_window": BREAKOUT_WINDOW,
                    "pyramid": list(PYRAMID_FRACTIONS),
                    "stop_loss": STOP_LOSS_PCT,
                    "trail_activate": TRAIL_ACTIVATE_PCT,
                    "trail_pullback": TRAIL_PULLBACK_PCT,
                    "add_spacing_grid": ADD_SPACING_GRID,
                    "profit_lock_arm": PROFIT_LOCK_ARM_PCT,
                    "profit_lock_exit": PROFIT_LOCK_EXIT_PCT,
                },
                "baseline_A_equal_weight": pick_stats(stats_a),
                "baseline_B_trail": pick_stats(stats_b),
                "livermore_grid": liver_rows,
                "best_by_sharpe": best,
                "profit_lock_compare": profit_lock_compare,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  结果已保存: {result_path}")
    print("\n✓ 回测完成!")


if __name__ == "__main__":
    main()
