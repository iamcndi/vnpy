"""
A股 Alpha 因子回测系统

功能：
 1. 通过 akshare / baostock 下载 A 股日线数据（前复权）
 2. 配置合约信息（佣金、最小变动等）
 3. 预计算信号（20日动量因子）
 4. 运行组合回测（等权买入前1/3股票）
 5. 输出统计指标（年化收益、夏普比率、最大回撤等）

使用方法：
   cd /Users/chendi/project/vnpy
   .venv/bin/python examples/alpha_a_share/run.py
"""

import os
import sys
from datetime import datetime, date
from collections import defaultdict

import polars as pl
import numpy as np

# ── 项目路径 ─────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from vnpy.alpha import AlphaLab, BacktestingEngine, AlphaStrategy, logger
from vnpy.trader.constant import Interval, Exchange, Direction
from vnpy.trader.object import BarData

from datafeed import download_daily_data, data_end_date


# ═══════════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════════

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")

# 持仓股票列表（来自持有.csv）
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

# 回测时间范围
BACKTEST_START = "2023-01-01"

# 回测参数
INITIAL_CAPITAL = 1_000_000           # 起始资金 100 万
ANNUAL_TRADING_DAYS = 250             # 年交易天数
MOMENTUM_LOOKBACK = 20                # 动量回溯窗口（交易日）
LONG_TOP_N_RATIO = 1.0 / 3.0         # 做多前 N% 股票
COMMISSION_RATE_BUY  = 0.00025        # 买入佣金（万2.5）
COMMISSION_RATE_SELL = 0.00125        # 卖出佣金（万2.5）+ 印花税（万10）


# ═══════════════════════════════════════════════════════════════
# 2. 合约配置
# ═══════════════════════════════════════════════════════════════

def setup_contracts(lab: AlphaLab) -> None:
    """配置 A 股交易参数"""
    for code, exchange_str, name in STOCK_LIST:
        vt_symbol = f"{code}.{exchange_str}"
        lab.add_contract_setting(
            vt_symbol,
            long_rate=COMMISSION_RATE_BUY,
            short_rate=COMMISSION_RATE_SELL,
            size=1,           # A 股 1 股 = 1 单位
            pricetick=0.01,   # 最小变动 0.01 元
        )
    print(f"  ✓ 已配置 {len(STOCK_LIST)} 个合约信息")


# ═══════════════════════════════════════════════════════════════
# 4. 预计算信号（20日动量）
# ═══════════════════════════════════════════════════════════════

def compute_momentum_signals(lab: AlphaLab) -> pl.DataFrame:
    """计算 20 日动量因子"""
    signal_data: list[dict] = []
    start_dt = datetime.strptime(BACKTEST_START, "%Y-%m-%d")
    end_dt   = datetime.strptime(BACKTEST_END, "%Y-%m-%d")

    for code, exchange_str, name in STOCK_LIST:
        vt_symbol = f"{code}.{exchange_str}"
        bars = lab.load_bar_data(vt_symbol, Interval.DAILY, start_dt, end_dt)
        if len(bars) < MOMENTUM_LOOKBACK + 1:
            continue

        for i in range(MOMENTUM_LOOKBACK, len(bars)):
            ret = bars[i].close_price / bars[i - MOMENTUM_LOOKBACK].close_price - 1
            signal_data.append({
                "datetime": bars[i].datetime,
                "vt_symbol": vt_symbol,
                "signal": ret,
            })

    df = pl.DataFrame(signal_data)
    print(f"  ✓ 计算 {len(df)} 条动量信号记录")
    return df


# ═══════════════════════════════════════════════════════════════
# 5. 动量策略
# ═══════════════════════════════════════════════════════════════

class MomentumTopNStrategy(AlphaStrategy):
    """动量选股策略：每天选择动量前1/3的股票等权买入"""

    price_add_pct = 0.003  # 以收盘价*1.003 发出买入限价单

    def on_init(self) -> None:
        self.write_log("动量选股策略初始化完成")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """每日调仓"""
        signal = self.get_signal()
        if signal.is_empty():
            return

        # 按信号从高到低排序
        signal = signal.sort("signal", descending=True)
        n_long = max(1, int(len(signal) * LONG_TOP_N_RATIO))

        long_set = set(signal.head(n_long)["vt_symbol"].to_list())

        # 按等权重分配资金
        portfolio_value = self.get_portfolio_value()
        capital_per_stock = portfolio_value / n_long

        for vt_symbol, bar in bars.items():
            if vt_symbol in long_set and bar.close_price > 0:
                target = int(capital_per_stock / bar.close_price)
            else:
                target = 0
            self.set_target(vt_symbol, target)

        self.execute_trading(bars, price_add=self.price_add_pct)

    def on_trade(self, trade):
        pass


# ═══════════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    download_only = len(sys.argv) > 1 and sys.argv[1] == "--download-only"

    print("=" * 60)
    print("  A 股 Alpha 因子回测系统")
    print("=" * 60)

    # ── 初始化 AlphaLab ───────────────────────────────────────
    lab = AlphaLab(ALPHA_LAB_PATH)
    end_date = data_end_date()
    start_dt = datetime.strptime(BACKTEST_START, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date, "%Y-%m-%d")

    # ── 下载/更新日线数据 ───────────────────────────────────────
    print("\n[1/5] 检查/下载日线数据...")
    download_daily_data(lab, STOCK_LIST, BACKTEST_START)

    if download_only:
        print(f"\n✓ 数据已更新至 {end_date}")
        return

    # ── 配置合约 ───────────────────────────────────────────────
    print("\n[2/5] 配置合约参数...")
    setup_contracts(lab)

    # ── 预计算信号 ─────────────────────────────────────────────
    print("\n[3/5] 预计算动量信号...")
    signal_df = compute_momentum_signals(lab)
    if signal_df.is_empty():
        print("✗ 信号数据为空，无法执行回测")
        return

    # ── 初始化回测引擎 ──────────────────────────────────────────
    print("\n[4/5] 运行回测...")
    vt_symbols = [f"{code}.{exch}" for code, exch, _ in STOCK_LIST]

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

    engine.add_strategy(MomentumTopNStrategy, {}, signal_df)
    engine.load_data()

    # ── 执行回测 ───────────────────────────────────────────────
    engine.run_backtesting()
    engine.calculate_result()
    statistics = engine.calculate_statistics()

    # ── 输出结果 ───────────────────────────────────────────────
    print("\n[5/5] 回测结果")
    print("-" * 60)
    for key, value in statistics.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:>16.2f}")
        else:
            print(f"  {key:30s}: {value}")
    print("-" * 60)
    print(f"  日志条数: {len(engine.logs)}")
    print(f"  交易笔数: {engine.trade_count}")

    # ── 显示图表（交互式环境可用） ─────────────────────────────
    print("\n提示: 如需查看图表，请在 Jupyter 中运行:")
    print("  engine.show_chart()")
    print("  engine.show_performance('000300.SSE')  # 与沪深300对比")
    print("\n✓ 回测完成!")


if __name__ == "__main__":
    main()
