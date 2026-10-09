"""
当日盈亏 + 历史区间盈亏：按账户持仓 / 流水 + 本地日线汇总

用法：
  .venv/bin/python examples/alpha_a_share/daily_pnl.py
  .venv/bin/python examples/alpha_a_share/daily_pnl.py --account 中信6700
  .venv/bin/python examples/alpha_a_share/daily_pnl.py --history
  .venv/bin/python examples/alpha_a_share/daily_pnl.py --account 兴业 --history --asof 2026-09-28
  .venv/bin/python examples/alpha_a_share/daily_pnl.py --update
  .venv/bin/python examples/alpha_a_share/daily_pnl.py --no-update
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

from account_equity import account_label, close_on_or_before, compute_equity_period
from datafeed import download_daily_data
from livermore_positions_store import all_prediction_accounts, load_positions
from stock_universe import STOCK_LIST, stock_name

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")
DAILY_DIR = Path(ALPHA_LAB_PATH) / "daily"


def _last_bar_date(vt: str) -> str | None:
    path = DAILY_DIR / f"{vt}.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path).sort("datetime")
    if df.height < 1:
        return None
    return df["datetime"][-1].strftime("%Y-%m-%d")


def _held_vt_symbols(accounts: list[str | None]) -> list[str]:
    wanted: set[str] = set()
    for acc in accounts:
        for vt, p in (load_positions(account=acc).get("positions") or {}).items():
            if int(p.get("shares") or 0) > 0:
                wanted.add(vt)
    return sorted(wanted)


def _held_stock_list(accounts: list[str | None]) -> list[tuple[str, str, str]]:
    """持仓代码 → STOCK_LIST 条目（仅用于增量下载）"""
    wanted = set(_held_vt_symbols(accounts))
    out: list[tuple[str, str, str]] = []
    for code, exch, name in STOCK_LIST:
        if f"{code}.{exch}" in wanted:
            out.append((code, exch, name))
            wanted.discard(f"{code}.{exch}")
    for vt in sorted(wanted):
        code, _, exch = vt.partition(".")
        out.append((code, exch, code))
    return out


def resolve_price_asof(
    accounts: list[str | None],
    *,
    override: str = "",
) -> str | None:
    """
    统一估值日：优先用 --price-asof；
    否则取各持仓本地日线最新日期的最大值（更新后应对齐到同一交易日）。
    """
    override = (override or "").strip()
    if override:
        return override
    dates = [_last_bar_date(vt) for vt in _held_vt_symbols(accounts)]
    dates = [d for d in dates if d]
    return max(dates) if dates else None


def closes_on_asof(vt: str, asof: str) -> tuple[float | None, float | None, str | None]:
    """
    取估值日收盘与昨收。
    若该票最新日线早于 asof，返回 (None, None, last_date) 表示缺当日。
    """
    last = _last_bar_date(vt)
    if last is None:
        return None, None, None
    if last < asof:
        return None, None, last
    px, prev = close_on_or_before(vt, asof)
    return px, prev, last


def report_account(
    account: str | None,
    price_asof: str,
) -> tuple[float, float, float, float]:
    label = account_label(account)
    book = load_positions(account=account)
    cash = float(book.get("cash") or 0)
    positions = book.get("positions") or {}

    print("=" * 72)
    print(
        f"账户: {label}  | cash={cash:,.2f}  | updated={book.get('updated', '')}"
        f"  | 价源估值日={price_asof}"
    )
    print(
        f"{'代码':<14} {'名称':<8} {'股数':>6} {'成本':>8} "
        f"{'昨收':>8} {'最新':>8} {'当日盈亏':>10} {'持仓盈亏':>10}"
    )

    day_sum = cost_sum = mv = 0.0
    missing = 0
    for vt, pos in positions.items():
        shares = int(pos.get("shares") or 0)
        if shares <= 0:
            continue
        cost = float(pos.get("cost") or 0)
        px, prev, last = closes_on_asof(vt, price_asof)
        try:
            name = stock_name(vt)
        except KeyError:
            name = vt.split(".", 1)[0]
        if px is None:
            missing += 1
            tip = f"缺{price_asof}" + (f"(本地至{last})" if last else "")
            print(f"{vt:<14} {name:<8} {shares:>6} {cost:>8.3f} {'—':>8} {tip:>8}")
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
    extra = f"  | 价源估值日 {price_asof}"
    if missing:
        extra += f"  | {missing} 只缺该日数据未计入"
    print(
        f"市值={mv:,.2f}  权益={equity:,.2f}  "
        f"当日盈亏合计={day_sum:+,.2f}  持仓盈亏合计={cost_sum:+,.2f}"
        + extra
    )
    return day_sum, cost_sum, mv, cash


def report_history(account: str | None, asof: str | None = None) -> float:
    """历史区间总盈亏（起始权益→估值日权益）。"""
    try:
        period = compute_equity_period(account, asof)
    except ValueError as e:
        print(f"  历史盈亏跳过: {e}")
        return 0.0
    print("-" * 72)
    print(period.summary)
    print(
        "  （股票口径：区间盈亏=Δ浮盈+区间估算已实现；"
        "已实现由快照股数减少估算，不含银证转入转出）"
    )
    return period.pnl


def main() -> None:
    parser = argparse.ArgumentParser(description="持仓当日盈亏 / 历史区间盈亏汇总")
    parser.add_argument("--account", default=None, help="账户名（默认=全部账户）")
    parser.add_argument(
        "--update",
        dest="update",
        action="store_true",
        default=False,
        help="先增量更新持仓标的日线",
    )
    parser.add_argument(
        "--no-update",
        dest="update",
        action="store_false",
        help="跳过日线更新，只用本地已有数据（默认）",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="额外输出历史区间盈亏（账户起始→估值日）",
    )
    parser.add_argument(
        "--asof",
        default="",
        help="历史盈亏估值日 YYYY-MM-DD（默认今天；需配合 --history）",
    )
    parser.add_argument(
        "--price-asof",
        default="",
        help="当日盈亏统一估值日 YYYY-MM-DD（默认=持仓日线最新交易日）",
    )
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

    price_asof = resolve_price_asof(accounts, override=args.price_asof)
    if not price_asof:
        print("无持仓日线，无法计算当日盈亏")
        return
    print(f"当日盈亏统一估值日: {price_asof}")

    stale = []
    for vt in _held_vt_symbols(accounts):
        last = _last_bar_date(vt)
        if last and last < price_asof:
            stale.append(f"{vt}(至{last})")
    if stale:
        print("⚠ 以下持仓尚无估值日日线，当日盈亏将跳过: " + ", ".join(stale))

    grand_day = grand_cost = grand_mv = grand_cash = grand_hist = 0.0
    for acc in accounts:
        day, cost, mv, cash = report_account(acc, price_asof)
        grand_day += day
        grand_cost += cost
        grand_mv += mv
        grand_cash += cash
        if args.history:
            grand_hist += report_history(acc, args.asof or None)

    if len(accounts) > 1:
        print("=" * 72)
        print(
            f"全部账户合计  市值={grand_mv:,.2f}  现金={grand_cash:,.2f}  "
            f"权益={grand_mv + grand_cash:,.2f}  "
            f"当日盈亏={grand_day:+,.2f}  持仓盈亏={grand_cost:+,.2f}"
            f"  | 价源估值日 {price_asof}"
        )
        if args.history:
            print(f"全部账户历史区间总收益合计={grand_hist:+,.2f}")
    print(
        "说明: 当日盈亏=(估值日收盘-昨收)×股数；"
        "持仓盈亏=(估值日收盘-成本)×股数；"
        "默认不更新日线（菜单选 Y 或加 --update）；统一估值日计算当日盈亏。"
    )


if __name__ == "__main__":
    main()
