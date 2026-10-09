"""A 股涨跌停幅度与判定（供预测 / 模拟下单共用）。"""

from __future__ import annotations


def _resolve_name(vt_symbol: str, name: str = "") -> str:
    if name:
        return name
    try:
        from stock_universe import stock_name

        return stock_name(vt_symbol)
    except Exception:
        return ""


def a_share_daily_limit_pct(vt_symbol: str, name: str = "") -> float | None:
    """A 股日涨跌停幅度；港股等无涨跌停板返回 None。"""
    code, _, exch = vt_symbol.partition(".")
    if not code or exch == "SEHK":
        return None
    nm = _resolve_name(vt_symbol, name)
    compact = nm.replace("*", "").replace(" ", "").upper()
    if "ST" in compact:
        return 0.05
    if code.startswith(("300", "301", "688")):
        return 0.20
    if exch == "BSE" or code.startswith(("8", "4", "92")):
        return 0.30
    return 0.10


# 兼容旧名
a_share_daily_limit_up_pct = a_share_daily_limit_pct


def a_share_lot_size(vt_symbol: str) -> int:
    """A 股最小交易单位：科创板 200，其余 100。"""
    code, _, _ = vt_symbol.partition(".")
    if code.startswith("688"):
        return 200
    return 100


def limit_up_price(pre_close: float, ratio: float) -> float:
    return round(pre_close * (1.0 + ratio) + 1e-9, 2)


def limit_down_price(pre_close: float, ratio: float) -> float:
    return round(pre_close * (1.0 - ratio) + 1e-9, 2)


def a_share_limit_prices(
    vt_symbol: str,
    pre_close: float,
    name: str = "",
) -> tuple[float, float] | None:
    """返回 (跌停价, 涨停价)；无法判定时返回 None。"""
    ratio = a_share_daily_limit_pct(vt_symbol, name)
    if ratio is None or pre_close <= 0:
        return None
    return limit_down_price(pre_close, ratio), limit_up_price(pre_close, ratio)


def is_a_share_limit_up(
    vt_symbol: str,
    price: float,
    pre_close: float,
    name: str = "",
) -> bool:
    """现价是否触及日涨停价（按昨收×涨幅四舍五入到分）。"""
    bounds = a_share_limit_prices(vt_symbol, pre_close, name)
    if bounds is None or price <= 0:
        return False
    _, up = bounds
    return price + 1e-9 >= up


def is_a_share_limit_down(
    vt_symbol: str,
    price: float,
    pre_close: float,
    name: str = "",
) -> bool:
    """现价是否触及日跌停价。"""
    bounds = a_share_limit_prices(vt_symbol, pre_close, name)
    if bounds is None or price <= 0:
        return False
    down, _ = bounds
    return price - 1e-9 <= down
