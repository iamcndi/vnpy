"""
当日盈亏：按账户持仓 + 本地日线最新价汇总

用法：
  .venv/bin/python examples/alpha_a_share/daily_pnl.py
  .venv/bin/python examples/alpha_a_share/daily_pnl.py --account 中信6700
  .venv/bin/python examples/alpha_a_share/daily_pnl.py --update
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from vnpy.alpha import AlphaLab

from datafeed import download_daily_data
from livermore_positions_store import all_prediction_accounts, load_positions
from stock_universe import STOCK_LIST, stock_name

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")
DAILY_DIR = Path(ALPHA_LAB_PATH) / "daily"


def _last_two_closes(vt: str) -> tuple[float | None, float | None, object | None]:
    path = DAILY_DIR / f"{vt}.parquet"
    if not path.exists():
        return None, None, None
    df = pl.read_parquet(path).sort("datetime")
    if df.height < 1:
        return None, None, None
    px = float(df["close"][-1])
    dt = df["datetime"][-1]
    prev = float(df["close"][-2]) if df.height >= 2 else None
    return px, prev, dt


def _held_stock_list(accounts: list[str | None]) -> list[tuple[str, str, str]]:
    """持仓代码 → STOCK_LIST 条目（仅用于增量下载）"""
    wanted: set[str] = set()
    for acc in accounts:
        for vt, p in (load_positions(account=acc).get("positions") or {}).items():
            if int(p.get("shares") or 0) > 0:
                wanted.add(vt)
    out: list[tuple[str, str, str]] = []
    for code, exch, name in STOCK_LIST:
        if f"{code}.{exch}" in wanted:
            out.append((code, exch, name))
            wanted.discard(f"{code}.{exch}")
    for vt in sorted(wanted):
        code, _, exch = vt.partition(".")
        out.append((code, exch, code))
    return out


def report_account(account: str | None) -> tuple[float, float, float, float]:
    label = account or "默认"
    book = load_positions(account=account)
    cash = float(book.get("cash") or 0)
    positions = book.get("positions") or {}

    print("=" * 72)
    print(f"账户: {label}  | cash={cash:,.2f}  | updated={book.get('updated', '')}")
    print(
        f"{'代码':<14} {'名称':<8} {'股数':>6} {'成本':>8} "
        f"{'昨收':>8} {'最新':>8} {'当日盈亏':>10} {'持仓盈亏':>10}"
    )

    day_sum = cost_sum = mv = 0.0
    asof = None
    for vt, pos in positions.items():
        shares = int(pos.get("shares") or 0)
        if shares <= 0:
            continue
        cost = float(pos.get("cost") or 0)
        px, prev, dt = _last_two_closes(vt)
        asof = dt or asof
        try:
            name = stock_name(vt)
        except KeyError:
            name = vt.split(".", 1)[0]
        if px is None:
            print(f"{vt:<14} {name:<8} {shares:>6} {cost:>8.3f} {'—':>8} {'缺数据':>8}")
            continue
        day_pnl = (px - prev) * shares if prev is not None else 0.0
        cost_pnl = (px - cost) * shares
        mv += px * shares
        day_sum += day_pnl
        cost_sum += cost_pnl
        prev_s = f"{prev:>8.2f}" if prev is not None else f"{'—':>8}"
        print(
            f"{vt:<14} {name:<8} {shares:>6} {cost:>8.3f} "
            f"{prev_s} {px:>8.2f} {day_pnl:>+10.2f} {cost_pnl:>+10.2f}"
        )

    equity = cash + mv
    print("-" * 72)
    print(
        f"市值={mv:,.2f}  权益={equity:,.2f}  "
        f"当日盈亏合计={day_sum:+,.2f}  持仓盈亏合计={cost_sum:+,.2f}"
        + (f"  | 价源截止 {asof}" if asof else "")
    )
    return day_sum, cost_sum, mv, cash


def main() -> None:
    parser = argparse.ArgumentParser(description="持仓当日盈亏汇总")
    parser.add_argument("--account", default=None, help="账户名（默认=全部账户）")
    parser.add_argument("--update", action="store_true", help="先增量更新持仓标的日线")
    args = parser.parse_args()

    accounts = [args.account] if args.account else all_prediction_accounts()
    if args.account is not None and not args.account.strip():
        accounts = [None]

    if args.update:
        held = _held_stock_list(accounts)
        if held:
            print("更新持仓标的日线...")
            lab = AlphaLab(ALPHA_LAB_PATH)
            download_daily_data(lab, held)
        else:
            print("无持仓，跳过下载")

    grand_day = grand_cost = grand_mv = grand_cash = 0.0
    for acc in accounts:
        day, cost, mv, cash = report_account(acc)
        grand_day += day
        grand_cost += cost
        grand_mv += mv
        grand_cash += cash

    if len(accounts) > 1:
        print("=" * 72)
        print(
            f"全部账户合计  市值={grand_mv:,.2f}  现金={grand_cash:,.2f}  "
            f"权益={grand_mv + grand_cash:,.2f}  "
            f"当日盈亏={grand_day:+,.2f}  持仓盈亏={grand_cost:+,.2f}"
        )
    print(
        "说明: 当日盈亏=(最新收盘-昨收)×股数；"
        "持仓盈亏=(最新收盘-成本)×股数；价格取本地日线。"
    )


if __name__ == "__main__":
    main()
