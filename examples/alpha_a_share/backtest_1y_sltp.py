"""
多因子 Alpha 一年回测 + 止损止盈优化

基于 predict_daily.py 的多因子逻辑（Alpha158 + LightGBM），
网格搜索最优止损/止盈参数，然后用最优参数回测最近一年。

止损止盈逻辑：
  - 每只持仓股票跟踪成本价
  - 当日收盘价跌破 成本 × (1 - stop_loss) → 次日卖出（止损）
  - 当日收盘价涨破 成本 × (1 + take_profit) → 次日卖出（止盈）
  - 止损/止盈卖出不受信号排名影响，强制清仓

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/backtest_1y_sltp.py
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

# 止损止盈搜索范围
STOP_LOSS_GRID = [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]
TAKE_PROFIT_GRID = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


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
# 3. 带止损止盈的策略
# ═══════════════════════════════════════════════════════════════

class SLTPStrategy(AlphaStrategy):
    """ML 信号选股 + 止损止盈"""

    price_add_pct = 0.003
    stop_loss_pct = 0.08      # 止损百分比（成本价下跌 X%）
    take_profit_pct = 0.15    # 止盈百分比（成本价上涨 X%）

    def on_init(self) -> None:
        # cost_basis[vt_symbol] = 持仓成本价
        self.cost_basis: dict[str, float] = {}
        # sltp_triggered[vt_symbol] = True 表示触发了止损/止盈，次日强制卖出
        self.sltp_triggered: dict[str, bool] = defaultdict(bool)
        self.write_log(f"止损止盈策略初始化 SL={self.stop_loss_pct:.0%} TP={self.take_profit_pct:.0%}")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        signal = self.get_signal()

        # ── 1. 检查止损止盈 ──────────────────────────────────
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

            # 检查是否触发止损或止盈
            pnl_pct = (bar.close_price - cost) / cost
            if pnl_pct <= -self.stop_loss_pct:
                sltp_sell_set.add(vt_symbol)
                self.write_log(f"止损: {vt_symbol} 成本={cost:.2f} 现价={bar.close_price:.2f} 亏损={pnl_pct:.1%}")
            elif pnl_pct >= self.take_profit_pct:
                sltp_sell_set.add(vt_symbol)
                self.write_log(f"止盈: {vt_symbol} 成本={cost:.2f} 现价={bar.close_price:.2f} 盈利={pnl_pct:.1%}")

        # ── 2. 信号选股 ──────────────────────────────────────
        long_set: set[str] = set()
        n_long = 1
        if not signal.is_empty():
            signal_sorted = signal.sort("signal", descending=True)
            n_long = max(1, int(len(signal_sorted) * LONG_TOP_N_RATIO))
            long_set = set(signal_sorted.head(n_long)["vt_symbol"].to_list())

        # ── 3. 计算目标持仓 ──────────────────────────────────
        # 止损止盈的股票不参与本期持仓
        effective_long = long_set - sltp_sell_set
        n_effective = max(1, len(effective_long))

        portfolio_value = self.get_portfolio_value()
        capital_per_stock = portfolio_value / n_effective

        for vt_symbol, bar in bars.items():
            if vt_symbol in sltp_sell_set:
                # 强制清仓
                self.set_target(vt_symbol, 0)
            elif vt_symbol in effective_long and bar.close_price > 0:
                target = int(capital_per_stock / bar.close_price)
                self.set_target(vt_symbol, target)
            else:
                self.set_target(vt_symbol, 0)

        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade: TradeData) -> None:
        # 更新成本价（买入时记录，卖出时清除）
        if trade.direction == Direction.LONG:
            old_pos = self.get_pos(vt_symbol := trade.vt_symbol) - trade.volume
            old_cost = self.cost_basis.get(vt_symbol, 0)
            if old_pos <= 0:
                # 新开仓
                self.cost_basis[vt_symbol] = trade.price
            else:
                # 加仓，加权平均成本
                total = old_pos * old_cost + trade.volume * trade.price
                self.cost_basis[vt_symbol] = total / (old_pos + trade.volume)
        elif trade.direction == Direction.SHORT:
            vt_symbol = trade.vt_symbol
            remaining = self.get_pos(vt_symbol)
            if remaining <= 0:
                self.cost_basis.pop(vt_symbol, None)


# ═══════════════════════════════════════════════════════════════
# 4. 回测运行器
# ═══════════════════════════════════════════════════════════════

def run_backtest(lab, vt_symbols, signal_df, stop_loss, take_profit):
    """运行单次回测，返回统计指标字典"""
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
    setting = {
        "stop_loss_pct": stop_loss,
        "take_profit_pct": take_profit,
    }
    engine.add_strategy(SLTPStrategy, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    return engine.calculate_statistics(), engine


# ═══════════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  多因子 Alpha 一年回测 + 止损止盈优化")
    print(f"  回测期: {BACKTEST_START} ~ {BACKTEST_END}")
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

    if model.model:
        importance = sorted(
            zip(model.feature_cols, model.model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        print(f"\n  因子重要性 Top10:")
        for name, imp in importance[:10]:
            print(f"    {name:15s}: {imp}")

    # ── Step 3: 网格搜索最优止损止盈 ─────────────────────────
    print(f"\n[3/4] 网格搜索最优止损止盈...")
    print(f"  止损范围: {[f'{x:.0%}' for x in STOP_LOSS_GRID]}")
    print(f"  止盈范围: {[f'{x:.0%}' for x in TAKE_PROFIT_GRID]}")

    results: list[dict] = []
    total = len(STOP_LOSS_GRID) * len(TAKE_PROFIT_GRID)
    count = 0

    for sl in STOP_LOSS_GRID:
        for tp in TAKE_PROFIT_GRID:
            count += 1
            stats, _ = run_backtest(lab, vt_symbols, signal_df, sl, tp)
            annual_ret = stats.get("annual_return", 0)
            sharpe = stats.get("sharpe_ratio", 0)
            max_dd = stats.get("max_ddpercent", 0)
            ret_dd_ratio = stats.get("return_drawdown_ratio", 0)

            results.append({
                "stop_loss": sl,
                "take_profit": tp,
                "annual_return": annual_ret,
                "sharpe_ratio": sharpe,
                "max_ddpercent": max_dd,
                "return_drawdown_ratio": ret_dd_ratio,
                "total_return": stats.get("total_return", 0),
            })
            print(f"  [{count}/{total}] SL={sl:.0%} TP={tp:.0%} → "
                  f"年化={annual_ret:.1f}% 夏普={sharpe:.2f} 回撤={max_dd:.1f}% 收益回撤比={ret_dd_ratio:.2f}")

    # 按 sharpe_ratio 排序找最优
    results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    print(f"\n  {'排名':>4s}  {'止损':>6s}  {'止盈':>6s}  {'年化收益':>8s}  {'夏普':>6s}  {'最大回撤':>8s}  {'收益回撤比':>8s}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i:>4d}  {r['stop_loss']:>6.0%}  {r['take_profit']:>6.0%}  "
              f"{r['annual_return']:>7.1f}%  {r['sharpe_ratio']:>6.2f}  "
              f"{r['max_ddpercent']:>7.1f}%  {r['return_drawdown_ratio']:>8.2f}")

    best = results[0]
    best_sl = best["stop_loss"]
    best_tp = best["take_profit"]
    print(f"\n  ★ 最优参数: 止损={best_sl:.0%} 止盈={best_tp:.0%}")
    print(f"    年化收益={best['annual_return']:.1f}% 夏普={best['sharpe_ratio']:.2f} "
          f"回撤={best['max_ddpercent']:.1f}% 收益回撤比={best['return_drawdown_ratio']:.2f}")

    # ── Step 4: 最优参数回测 ─────────────────────────────────
    print(f"\n[4/4] 用最优参数 (SL={best_sl:.0%}, TP={best_tp:.0%}) 回测...")
    stats, engine = run_backtest(lab, vt_symbols, signal_df, best_sl, best_tp)

    print("\n" + "=" * 60)
    print(f"  最终回测结果 (止损={best_sl:.0%}, 止盈={best_tp:.0%})")
    print("=" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>18.2f}")
        else:
            print(f"  {k:30s}: {v}")
    print("-" * 60)
    print(f"  交易笔数:     {engine.trade_count:>15d}")

    # 保存搜索结果
    search_path = os.path.join(ALPHA_LAB_PATH, "sltp_search_results.json")
    with open(search_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  搜索结果已保存: {search_path}")
    print("\n✓ 回测完成!")


if __name__ == "__main__":
    main()
