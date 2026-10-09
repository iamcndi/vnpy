"""账户区间权益 / 历史盈亏（实盘与模拟共用）。

默认股票口径：只看持仓市值 + 估算已实现，忽略银证转入/转出对现金的影响。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from livermore_positions_store import (
    history_on_or_before,
    list_history,
    load_positions,
    resolve_paths,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DAILY_DIR = PROJECT_DIR / "alpha_data" / "daily"


def account_label(account: str | None) -> str:
    return account or "默认"


def close_on_or_before(vt: str, asof: str) -> tuple[float | None, float | None]:
    path = DAILY_DIR / f"{vt}.parquet"
    if not path.exists():
        return None, None
    df = pl.read_parquet(path).sort("datetime")
    if df.height < 1:
        return None, None
    dates = [d.strftime("%Y-%m-%d") for d in df["datetime"].to_list()]
    idx = None
    for i, d in enumerate(dates):
        if d <= asof:
            idx = i
    if idx is None:
        return None, None
    px = float(df["close"][idx])
    prev = float(df["close"][idx - 1]) if idx > 0 else None
    return px, prev


def book_on_or_before(account: str | None, asof: str) -> dict:
    """估值日及之前最近账本快照；无历史则用当前账本。"""
    _, hist_path = resolve_paths(account)
    snap = history_on_or_before(asof, path=hist_path)
    if snap is not None:
        return {
            "updated": snap.get("updated", ""),
            "cash": float(snap.get("cash") or 0),
            "positions": dict(snap.get("positions") or {}),
        }
    return load_positions(account=account)


def account_start_date(account: str | None) -> str:
    _, hist_path = resolve_paths(account)
    rows = list_history(path=hist_path)
    for r in rows:
        u = (r.get("updated") or "").strip()
        if u:
            return u
    book = load_positions(account=account)
    return (book.get("updated") or "").strip() or date.today().isoformat()


def symbols_from_history(account: str | None) -> set[str]:
    _, hist_path = resolve_paths(account)
    symbols: set[str] = set()
    for r in list_history(path=hist_path):
        for vt, pos in (r.get("positions") or {}).items():
            if int(pos.get("shares") or 0) > 0:
                symbols.add(vt)
    book = load_positions(account=account)
    for vt, pos in (book.get("positions") or {}).items():
        if int(pos.get("shares") or 0) > 0:
            symbols.add(vt)
    return symbols


def trading_days_in_range(symbols: set[str], start: str, end: str) -> list[str]:
    days: set[str] = set()
    for vt in symbols:
        path = DAILY_DIR / f"{vt}.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path).sort("datetime")
        for d in df["datetime"].to_list():
            ds = d.strftime("%Y-%m-%d")
            if start <= ds <= end:
                days.add(ds)
    if not days:
        return [start] if start == end else [start, end]
    return sorted(days)


def stock_market_value(book: dict, asof: str) -> float:
    mv = 0.0
    for vt, pos in (book.get("positions") or {}).items():
        shares = int(pos.get("shares") or 0)
        if shares <= 0:
            continue
        px, _ = close_on_or_before(vt, asof)
        if px is not None:
            mv += px * shares
        else:
            mv += float(pos.get("cost") or 0) * shares
    return mv


def stock_cost_basis(book: dict) -> float:
    total = 0.0
    for pos in (book.get("positions") or {}).values():
        shares = int(pos.get("shares") or 0)
        if shares <= 0:
            continue
        total += float(pos.get("cost") or 0) * shares
    return total


def stock_unrealized(book: dict, asof: str) -> float:
    return stock_market_value(book, asof) - stock_cost_basis(book)


def equity_on_date(book: dict, asof: str) -> float:
    """兼容旧接口：现金 + 持仓市值（含转入转出影响）。"""
    return float(book.get("cash") or 0) + stock_market_value(book, asof)


def _history_rows(account: str | None) -> list[dict]:
    _, hist_path = resolve_paths(account)
    return list_history(path=hist_path)


def _snap_positions(row: dict) -> dict[str, tuple[int, float]]:
    out: dict[str, tuple[int, float]] = {}
    for vt, pos in (row.get("positions") or {}).items():
        shares = int(pos.get("shares") or 0)
        if shares <= 0:
            continue
        out[vt] = (shares, float(pos.get("cost") or 0))
    return out


def realized_pnl_up_to(account: str | None, asof: str) -> float:
    """
    估算截至 asof 的已实现盈亏：相邻历史快照间股数减少时，
    按 (估值日收盘或成本 − 原成本) × 卖出股数 累加。
    """
    rows = _history_rows(account)
    prev: dict[str, tuple[int, float]] = {}
    realized = 0.0
    for row in rows:
        u = (row.get("updated") or "").strip()
        if not u or u > asof:
            continue
        cur = _snap_positions(row)
        for vt, (old_shares, old_cost) in prev.items():
            new_shares = cur.get(vt, (0, 0.0))[0]
            if new_shares >= old_shares:
                continue
            sold = old_shares - new_shares
            px, _ = close_on_or_before(vt, u)
            if px is None or px <= 0:
                px = old_cost
            realized += (px - old_cost) * sold
        prev = cur
    return realized


@dataclass
class EquityPeriod:
    account: str | None
    start: str
    end: str
    dates: list[str]
    equities: list[float]  # 权益曲线：现金 + 市值（用于绘图与回撤）
    start_equity: float  # 期初持仓市值（股票口径）
    end_equity: float  # 期末持仓市值（股票口径）
    pnl: float  # 区间股票盈亏
    ret: float
    max_drawdown: float  # 按现金+市值权益计算
    realized: float = 0.0
    start_cost: float = 0.0
    end_cost: float = 0.0
    cash_end: float = 0.0
    ret_base: float = 0.0
    start_nav: float = 0.0
    end_nav: float = 0.0

    @property
    def summary(self) -> str:
        label = account_label(self.account)
        ret_s = f"{self.ret * 100:+.2f}%" if self.ret_base > 1e-6 else "n/a"
        return (
            f"账户={label}  区间={self.start} → {self.end}  样本日={len(self.dates)}\n"
            f"【股票口径·盈亏】期初市值={self.start_equity:,.2f}  期末市值={self.end_equity:,.2f}  "
            f"估算已实现={self.realized:+,.2f}  "
            f"区间股票盈亏={self.pnl:+,.2f}  收益率={ret_s}"
            f"（分母=期初权益 {self.ret_base:,.2f}）\n"
            f"【权益口径·回撤】期初权益={self.start_nav:,.2f}  期末权益={self.end_nav:,.2f}  "
            f"最大回撤={self.max_drawdown * 100:.2f}%  "
            f"（期末现金={self.cash_end:,.2f}）"
        )


def compute_equity_period(
    account: str | None,
    asof: str | None = None,
    *,
    start: str | None = None,
) -> EquityPeriod:
    """
    区间分析：
    - 盈亏：股票口径（市值浮盈 + 估算已实现）
    - 收益率：区间股票盈亏 / 期初权益（现金+市值），避免周转后用成本分母夸大
    - 曲线/最大回撤：权益口径（每日现金 + 市值）
    """
    asof = (asof or "").strip() or date.today().isoformat()
    start = (start or "").strip() or account_start_date(account)
    if asof < start:
        raise ValueError(f"估值日 {asof} 早于起始日 {start}")

    symbols = symbols_from_history(account)
    days = trading_days_in_range(symbols, start, asof)
    if not days:
        days = [start] if start == asof else [start, asof]

    dates: list[str] = []
    navs: list[float] = []
    for d in days:
        book = book_on_or_before(account, d)
        mv = stock_market_value(book, d)
        cash = float(book.get("cash") or 0)
        dates.append(d)
        navs.append(cash + mv)

    book0 = book_on_or_before(account, start)
    book1 = book_on_or_before(account, asof)
    start_mv = stock_market_value(book0, start)
    end_mv = stock_market_value(book1, asof)
    start_cash = float(book0.get("cash") or 0)
    end_cash = float(book1.get("cash") or 0)
    start_nav = start_cash + start_mv
    end_nav = end_cash + end_mv
    start_cost = stock_cost_basis(book0)
    end_cost = stock_cost_basis(book1)
    real0 = realized_pnl_up_to(account, start)
    real1 = realized_pnl_up_to(account, asof)
    # 区间内新增已实现
    realized_period = real1 - real0
    pnl = (stock_unrealized(book1, asof) + real1) - (stock_unrealized(book0, start) + real0)

    # 收益率分母：期初权益；若期初几乎无权益则回退峰值成本
    base = start_nav
    if base <= 1e-6:
        peak_cost = 0.0
        for d in days:
            peak_cost = max(peak_cost, stock_cost_basis(book_on_or_before(account, d)))
        base = peak_cost
    ret = (pnl / base) if base > 1e-6 else 0.0

    peak = navs[0] if navs else 0.0
    max_dd = 0.0
    for nav in navs:
        if nav > peak:
            peak = nav
        if peak > 1e-6:
            dd = (nav - peak) / peak
            if dd < max_dd:
                max_dd = dd

    return EquityPeriod(
        account=account,
        start=start,
        end=asof,
        dates=dates,
        equities=navs,
        start_equity=start_mv,
        end_equity=end_mv,
        pnl=pnl,
        ret=ret,
        max_drawdown=max_dd,
        realized=realized_period,
        start_cost=start_cost,
        end_cost=end_cost,
        cash_end=end_cash,
        ret_base=base,
        start_nav=start_nav,
        end_nav=end_nav,
    )
