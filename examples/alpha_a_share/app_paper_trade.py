"""
模拟操盘 Web UI（Gradio）

启动：
  .venv/bin/python examples/alpha_a_share/app_paper_trade.py
"""

from __future__ import annotations

import os
import sys
from datetime import date

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import polars as pl

from account_equity import (
    account_label,
    book_on_or_before,
    close_on_or_before,
    compute_equity_period,
)
from a_share_limits import a_share_lot_size
from livermore_positions_store import list_history, load_positions, resolve_paths
from paper_trade import (
    SIM_PREFIX,
    check_order,
    create_sim_account,
    execute_order,
    list_sim_accounts,
    list_sim_session_choices,
    list_sim_sessions,
    load_sim_session,
    reset_sim_account,
    resolve_trade_price,
    resolve_vt_symbol,
    save_sim_session,
)
from stock_universe import STOCK_LIST, stock_name

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")
DAILY_DIR = os.path.join(ALPHA_LAB_PATH, "daily")
TREND_LOOKBACK_BARS = 60


def _setup_chinese_font() -> None:
    candidates = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


_setup_chinese_font()


def _refresh_accounts(selected: str | None = None):
    rows = list_sim_accounts()
    value = selected if selected in rows else (rows[0] if rows else None)
    return gr.update(choices=rows, value=value)


def _create_account(suffix: str, cash: float, updated: str):
    name = (suffix or "").strip()
    if not name:
        return "请填写账户名后缀（将自动加前缀 模拟_）", _refresh_accounts()
    try:
        cash_f = float(cash)
    except (TypeError, ValueError):
        return "初始资金无效", _refresh_accounts()
    updated = (updated or "").strip() or date.today().isoformat()
    try:
        create_sim_account(name, cash_f, updated=updated)
    except FileExistsError:
        full = name if name.startswith(SIM_PREFIX) else f"{SIM_PREFIX}{name}"
        return f"账户已存在: {full}", _refresh_accounts()
    except Exception as e:
        return f"创建失败: {e}", _refresh_accounts()
    full = name if name.startswith(SIM_PREFIX) else f"{SIM_PREFIX}{name}"
    return f"已创建 {full}，初始资金 {cash_f:,.2f}", _refresh_accounts(full)


def _reset_account(account: str, cash: float, updated: str, save_first: bool, asof: str, note: str):
    if not account:
        return "请先选择要重置的模拟账户", _refresh_accounts()
    try:
        cash_f = float(cash)
    except (TypeError, ValueError):
        return "重置资金无效", _refresh_accounts(account)
    updated = (updated or "").strip() or date.today().isoformat()
    parts: list[str] = []
    if save_first:
        try:
            path = save_sim_session(account, asof=asof or updated, note=note or "重置前自动保存")
            parts.append(f"已保存本次结果 → {path}")
        except Exception as e:
            return f"重置已取消：保存失败（{e}）。请先手动保存或取消勾选「重置前先保存」", _refresh_accounts(account)
    try:
        reset_sim_account(account, cash_f, updated=updated)
    except FileNotFoundError as e:
        return str(e), _refresh_accounts()
    except Exception as e:
        return f"重置失败: {e}", _refresh_accounts(account)
    parts.append(f"已重置 {account}：空仓，资金={cash_f:,.2f}，起始日={updated}")
    return "\n".join(parts), _refresh_accounts(account)


def _save_session(account: str, asof: str, note: str):
    if not account:
        return "请先选择模拟账户"
    try:
        path = save_sim_session(account, asof=asof, note=note)
    except Exception as e:
        return f"保存失败: {e}"
    rows = list_sim_sessions(account, limit=5)
    recent = "\n".join(f"  - {p.name}" for p in rows)
    return f"已保存本次结果 → {path}\n最近存档:\n{recent}"


def _stock_choices() -> list[str]:
    return [f"{code}.{exch} {name}" for code, exch, name in STOCK_LIST]


def parse_code_input(raw: str) -> str:
    """下拉「代码.交易所 名称」或手输代码 → 可解析片段。"""
    s = (raw or "").strip()
    if not s:
        return ""
    return s.split()[0]


def _resolve_order_code(pick: str, manual: str) -> str:
    """手动输入优先，否则用下拉选择。"""
    manual = (manual or "").strip()
    if manual:
        return parse_code_input(manual)
    return parse_code_input(pick)


def _empty_trend_fig(message: str = ""):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.text(0.5, 0.5, message or "填写代码与交易日后显示走势", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _load_daily_series(
    vt: str,
) -> tuple[list[str], list[float], list[float], list[float], list[float], list[float]] | None:
    """返回 (dates, opens, highs, lows, closes, volumes)。缺 OHLC 时用 close 填充。"""
    path = os.path.join(DAILY_DIR, f"{vt}.parquet")
    if not os.path.exists(path):
        return None
    df = pl.read_parquet(path).sort("datetime")
    if df.height < 1:
        return None
    dates = [d.strftime("%Y-%m-%d") for d in df["datetime"].to_list()]
    closes = [float(c) for c in df["close"].to_list()]
    if "open" in df.columns:
        opens = [float(v) for v in df["open"].to_list()]
    else:
        opens = list(closes)
    if "high" in df.columns:
        highs = [float(v) for v in df["high"].to_list()]
    else:
        highs = list(closes)
    if "low" in df.columns:
        lows = [float(v) for v in df["low"].to_list()]
    else:
        lows = list(closes)
    if "volume" in df.columns:
        volumes = [float(v or 0) for v in df["volume"].to_list()]
    else:
        volumes = [0.0] * len(dates)
    return dates, opens, highs, lows, closes, volumes


def _end_idx_on_or_before(dates: list[str], trade_date: str) -> int | None:
    end_idx = None
    for i, d in enumerate(dates):
        if d <= trade_date:
            end_idx = i
    return end_idx


def _draw_candles(
    ax,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> None:
    """A 股习惯：红涨绿跌。"""
    n = len(closes)
    if n <= 0:
        return
    width = 0.6
    # 十字星最小实体高度（相对价格）
    eps = max((max(highs) - min(lows)) * 0.002, 1e-6)
    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        up = c >= o
        color = "#ef4444" if up else "#22c55e"
        ax.vlines(i, l, h, color=color, linewidth=1.0, zorder=2)
        body_low = min(o, c)
        body_h = max(abs(c - o), eps)
        ax.bar(
            i,
            body_h,
            bottom=body_low,
            width=width,
            color=color,
            edgecolor=color,
            linewidth=0.5,
            zorder=3,
            align="center",
        )


def _trend_chart(code: str, trade_date: str):
    """截至所选交易日的近 N 日蜡烛图 + 成交量。"""
    code = (code or "").strip()
    trade_date = (trade_date or "").strip()
    if not code or not trade_date:
        return _empty_trend_fig()
    try:
        vt = resolve_vt_symbol(code)
    except Exception as e:
        return _empty_trend_fig(str(e))

    series = _load_daily_series(vt)
    if series is None:
        return _empty_trend_fig(f"{vt} 无本地日线")
    dates, opens, highs, lows, closes, volumes = series
    end_idx = _end_idx_on_or_before(dates, trade_date)
    if end_idx is None:
        return _empty_trend_fig(f"{vt} 在 {trade_date} 之前无数据")

    start_idx = max(0, end_idx - TREND_LOOKBACK_BARS + 1)
    xs = dates[start_idx : end_idx + 1]
    os_ = opens[start_idx : end_idx + 1]
    hs = highs[start_idx : end_idx + 1]
    ls = lows[start_idx : end_idx + 1]
    cs = closes[start_idx : end_idx + 1]
    vs = volumes[start_idx : end_idx + 1]
    try:
        name = stock_name(vt)
    except KeyError:
        name = vt.split(".", 1)[0]

    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(8, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )
    x_idx = list(range(len(xs)))
    _draw_candles(ax_price, os_, hs, ls, cs)
    ax_price.scatter(
        [x_idx[-1]],
        [cs[-1]],
        color="#f97316",
        s=36,
        zorder=4,
        label=f"{xs[-1]} 收 {cs[-1]:.2f}",
    )
    ax_price.axvline(x_idx[-1], color="#f97316", linestyle="--", linewidth=1, alpha=0.7)
    ax_price.set_title(f"{vt} {name}（截至 {xs[-1]}，近 {len(xs)} 日蜡烛）")
    ax_price.set_ylabel("价格")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left", fontsize=8)

    vol_colors = [
        "#ef4444" if cs[i] >= os_[i] else "#22c55e" for i in range(len(cs))
    ]
    ax_vol.bar(x_idx, vs, color=vol_colors, width=0.8, alpha=0.85)
    ax_vol.axvline(x_idx[-1], color="#f97316", linestyle="--", linewidth=1, alpha=0.7)
    ax_vol.set_ylabel("成交量")
    ax_vol.grid(True, axis="y", alpha=0.25)
    _sparse_date_ticks(ax_vol, xs, max_ticks=8)

    fig.tight_layout()
    return fig


def _shift_trade_date(code: str, trade_date: str, delta: int) -> tuple[str, str]:
    """
    按该股日线交易日前后移动。
    返回 (新交易日, 提示)；无法移动时交易日不变并带提示。
    """
    code = (code or "").strip()
    trade_date = (trade_date or "").strip()
    if not code:
        return trade_date, "请先填写代码"
    if not trade_date:
        return trade_date, "请先填写交易日"
    try:
        vt = resolve_vt_symbol(code)
    except Exception as e:
        return trade_date, str(e)
    series = _load_daily_series(vt)
    if series is None:
        return trade_date, f"{vt} 无本地日线"
    dates = series[0]
    if trade_date in dates:
        idx = dates.index(trade_date)
    else:
        idx = _end_idx_on_or_before(dates, trade_date)
        if idx is None:
            return trade_date, f"{vt} 在 {trade_date} 之前无交易日"
        # 当前不是交易日：向后取下一根，向前取已对齐的上一有效日
        if delta > 0:
            if idx + 1 >= len(dates):
                return dates[idx], f"已是最后交易日 {dates[idx]}"
            return dates[idx + 1], ""
        return dates[idx], f"已对齐到最近交易日 {dates[idx]}"

    new_idx = idx + delta
    if new_idx < 0:
        return dates[0], f"已是最早交易日 {dates[0]}"
    if new_idx >= len(dates):
        return dates[-1], f"已是最后交易日 {dates[-1]}"
    return dates[new_idx], ""


def _default_price(code: str, trade_date: str) -> tuple[str, str]:
    code = (code or "").strip()
    trade_date = (trade_date or "").strip()
    if not code or not trade_date:
        return "", "请先填代码与交易日"
    try:
        vt = resolve_vt_symbol(code)
        px = resolve_trade_price(vt, trade_date)
        try:
            name = stock_name(vt)
        except KeyError:
            name = vt
        _, prev = close_on_or_before(vt, trade_date)
        # close_on_or_before 的 prev 已是昨收；再算涨跌幅
        chg = ""
        if prev is not None and prev > 1e-9:
            pct = (px / prev - 1.0) * 100
            chg = f"  涨跌幅 {pct:+.2f}%"
        return f"{px:.2f}", f"{vt} {name} @ {trade_date} 收盘 {px:.2f}{chg}"
    except Exception as e:
        return "", str(e)


def _shares_choices_update(code: str, current: int | float | None = None):
    """按板块返回数量下拉：主板 100/200/500/1000，科创 200/400/1000/2000。"""
    lot = 100
    code = (code or "").strip()
    if code:
        try:
            lot = a_share_lot_size(resolve_vt_symbol(code))
        except Exception:
            lot = 100
    choices = [lot, lot * 2, lot * 5, lot * 10]
    try:
        cur = int(current) if current is not None else None
    except (TypeError, ValueError):
        cur = None
    value = cur if cur in choices else lot
    return gr.update(choices=choices, value=value)


def _resolve_shares(pick, manual) -> int | None:
    manual = (manual or "").strip() if manual is not None else ""
    if manual:
        try:
            return int(float(manual))
        except (TypeError, ValueError):
            return None
    if pick is None or pick == "":
        return None
    try:
        return int(float(pick))
    except (TypeError, ValueError):
        return None


def _refresh_order_inputs(code: str, trade_date: str, current_shares=None):
    price, hint = _default_price(code, trade_date)
    return price, hint, _trend_chart(code, trade_date), _shares_choices_update(code, current_shares)


def _step_trade_date(code: str, trade_date: str, delta: int, current_shares=None):
    new_date, step_hint = _shift_trade_date(code, trade_date, delta)
    price, hint = _default_price(code, new_date)
    if step_hint:
        hint = f"{step_hint}；{hint}" if hint else step_hint
    return (
        new_date,
        price,
        hint,
        _trend_chart(code, new_date),
        _shares_choices_update(code, current_shares),
    )


def _evaluate_order(
    account: str,
    side: str,
    code_pick: str,
    code_manual: str,
    shares_pick,
    shares_manual: str,
    trade_date: str,
    price: str,
) -> str:
    code = _resolve_order_code(code_pick, code_manual)
    if not account:
        return "请选择模拟账户"
    if not code:
        return "请选择或输入股票代码"
    shares_i = _resolve_shares(shares_pick, shares_manual)
    if shares_i is None:
        return "请选择或输入买入/卖出数量"
    price_f = None
    price = (price or "").strip()
    if price:
        try:
            price_f = float(price)
        except ValueError:
            return "价格无效"
    return check_order(
        account,
        side or "买入",
        code,
        shares_i,
        trade_date or "",
        price=price_f,
    ).message


def _submit_order(
    account: str,
    side: str,
    code_pick: str,
    code_manual: str,
    shares_pick,
    shares_manual: str,
    trade_date: str,
    price: str,
    update_data: bool,
) -> str:
    if not account:
        return "请选择模拟账户"
    code = _resolve_order_code(code_pick, code_manual)
    if not code:
        return "请选择或输入股票代码"
    shares_i = _resolve_shares(shares_pick, shares_manual)
    if shares_i is None:
        return "请选择或输入数量"
    price_f = None
    price = (price or "").strip()
    if price:
        try:
            price_f = float(price)
        except ValueError:
            return "价格无效"
    chk = check_order(account, side, code, shares_i, trade_date or "", price=price_f)
    if not chk.can_submit:
        return chk.message
    result = execute_order(
        account,
        side,
        code,
        shares_i,
        trade_date or "",
        price=price_f,
        update_data=bool(update_data),
    )
    if result.ok and chk.level == "warn":
        return result.message + "\n" + chk.message
    return result.message


def _refresh_from_symbols(
    code_pick: str,
    code_manual: str,
    trade_date: str,
    shares_pick=None,
    shares_manual: str = "",
):
    code = _resolve_order_code(code_pick, code_manual)
    cur = _resolve_shares(shares_pick, shares_manual)
    return _refresh_order_inputs(code, trade_date, cur)


def _step_from_symbols(
    code_pick: str,
    code_manual: str,
    trade_date: str,
    delta: int,
    shares_pick=None,
    shares_manual: str = "",
):
    code = _resolve_order_code(code_pick, code_manual)
    cur = _resolve_shares(shares_pick, shares_manual)
    return _step_trade_date(code, trade_date, delta, cur)


def _on_stock_pick(pick: str, trade_date: str, shares_pick=None, shares_manual: str = ""):
    """下拉选股 → 同步手输框为纯代码，并刷新价格/图/数量选项。"""
    code = parse_code_input(pick)
    cur = _resolve_shares(shares_pick, shares_manual)
    price, hint, fig, shares_u = _refresh_order_inputs(code, trade_date, cur)
    manual_shares = str(cur) if cur else ""
    return code, price, hint, fig, shares_u, manual_shares


def _on_shares_pick(pick, shares_manual: str):
    """数量下拉 → 同步手输。"""
    if pick is None or pick == "":
        return shares_manual or ""
    return str(int(pick))


def _close_on_or_before(vt: str, asof: str) -> tuple[float | None, float | None]:
    return close_on_or_before(vt, asof)


def _book_on_or_before(account: str, asof: str) -> dict:
    return book_on_or_before(account, asof)


def _empty_equity_fig(message: str = ""):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.text(0.5, 0.5, message or "选择模拟账户与估值日后显示权益曲线", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _sparse_date_ticks(ax, dates: list[str], *, max_ticks: int = 8) -> None:
    """横轴按等距抽稀标注日期，避免长区间挤成一团。"""
    n = len(dates)
    if n <= 0:
        ax.set_xticks([])
        return
    tick_n = min(max_ticks, n)
    if tick_n == 1:
        positions = [0]
    else:
        positions = [int(i * (n - 1) / (tick_n - 1)) for i in range(tick_n)]
    # 跨度大时用 YYYY-MM，否则 YYYY-MM-DD
    span_days = n
    try:
        from datetime import date as _date

        span_days = (_date.fromisoformat(dates[-1]) - _date.fromisoformat(dates[0])).days
    except ValueError:
        pass

    def _fmt(d: str) -> str:
        if span_days >= 120 and len(d) >= 7:
            return d[:7]
        return d

    ax.set_xticks(positions)
    ax.set_xticklabels([_fmt(dates[i]) for i in positions], rotation=20, ha="right")


def _equity_analysis(account: str, asof: str) -> tuple[object, str]:
    """从模拟起始日到估值日的权益曲线 + 区间汇总。"""
    if not account:
        return _empty_equity_fig("请选择模拟账户"), "请选择模拟账户"
    asof = (asof or "").strip() or date.today().isoformat()
    try:
        period = compute_equity_period(account, asof)
    except ValueError as e:
        return _empty_equity_fig(str(e)), str(e)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    x_idx = list(range(len(period.dates)))
    ax.plot(x_idx, period.equities, color="#2563eb", linewidth=1.6)
    ax.scatter([x_idx[0]], [period.equities[0]], color="#64748b", s=28, zorder=3)
    ax.scatter([x_idx[-1]], [period.equities[-1]], color="#dc2626", s=36, zorder=3)
    _sparse_date_ticks(ax, period.dates)
    ax.set_title(f"{account_label(account)} 权益（现金+市值，{period.start} → {period.end}）")
    ax.set_ylabel("现金+市值")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, period.summary


def _positions_table(account: str, asof: str) -> tuple[list[list], str]:
    if not account:
        return [], "请选择模拟账户"
    asof = (asof or "").strip() or date.today().isoformat()
    book = _book_on_or_before(account, asof)
    cash = float(book.get("cash") or 0)
    rows: list[list] = []
    mv = day_sum = cost_sum = 0.0
    for vt, pos in (book.get("positions") or {}).items():
        shares = int(pos.get("shares") or 0)
        if shares <= 0:
            continue
        cost = float(pos.get("cost") or 0)
        px, prev = _close_on_or_before(vt, asof)
        try:
            name = stock_name(vt)
        except KeyError:
            name = vt.split(".", 1)[0]
        if px is None:
            rows.append([vt, name, shares, f"{cost:.3f}", "—", "缺数据", "—", "—", "—", "—"])
            continue
        chg_pct = ((px / prev) - 1.0) * 100 if prev is not None and prev > 1e-9 else None
        buy_day = str(pos.get("last_buy_date") or "").strip()
        same_day_buy = bool(buy_day and buy_day == asof)
        # 买入当日不计当日盈亏（仍显示涨跌幅）
        if same_day_buy or prev is None:
            day_pnl_s = "—"
        else:
            day_pnl = (px - prev) * shares
            day_sum += day_pnl
            day_pnl_s = f"{day_pnl:+.2f}"
        cost_pnl = (px - cost) * shares
        pnl_pct = ((px / cost) - 1.0) * 100 if cost > 1e-9 else 0.0
        mv += px * shares
        cost_sum += cost_pnl
        rows.append(
            [
                vt,
                name,
                shares,
                f"{cost:.3f}",
                f"{prev:.2f}" if prev is not None else "—",
                f"{px:.2f}",
                f"{chg_pct:+.2f}%" if chg_pct is not None else "—",
                day_pnl_s,
                f"{cost_pnl:+.2f}",
                f"{pnl_pct:+.2f}%",
            ]
        )
    equity = cash + mv
    cost_basis = sum(
        float(p.get("cost") or 0) * int(p.get("shares") or 0)
        for p in (book.get("positions") or {}).values()
        if int(p.get("shares") or 0) > 0
    )
    total_pct = (cost_sum / cost_basis * 100) if cost_basis > 1e-9 else 0.0
    pos_line = (
        f"期末持仓（估值日={asof}，快照updated={book.get('updated', '')}）\n"
        f"现金={cash:,.2f}  市值={mv:,.2f}  权益={equity:,.2f}  "
        f"当日盈亏={day_sum:+,.2f}（买入日不计）\n"
        f"当前持仓盈亏={cost_sum:+,.2f}  当前持仓盈亏比={total_pct:+.2f}%"
        f"（不含已卖出轮次；再买入后按新成本）"
    )
    return rows, pos_line


def _refresh_positions(account: str, asof: str):
    rows, pos_line = _positions_table(account, asof)
    fig, period = _equity_analysis(account, asof)
    summary = f"{period}\n{pos_line}"
    return rows, summary, fig


def _history_table(account: str, trade_date: str, limit: float) -> list[list]:
    if not account:
        return []
    try:
        n = int(limit) if limit else 50
    except (TypeError, ValueError):
        n = 50
    _, hist_path = resolve_paths(account)
    rows = list_history(path=hist_path)
    trade_date = (trade_date or "").strip()
    if trade_date:
        rows = [
            r
            for r in rows
            if (r.get("updated") or "") == trade_date
            or (r.get("note") or "").find(f"交易日={trade_date}") >= 0
        ]
    rows = rows[-n:]
    out: list[list] = []
    for r in reversed(rows):
        note = r.get("note") or ""
        # 从 note 粗解析：模拟买入 000333 200@82.47 交易日=...
        side = code = shares = price = ""
        if "模拟买入" in note:
            side = "买入"
        elif "模拟卖出" in note:
            side = "卖出"
        parts = note.replace("模拟买入", "").replace("模拟卖出", "").strip().split()
        if parts:
            code = parts[0]
        if len(parts) >= 2 and "@" in parts[1]:
            sh, _, pr = parts[1].partition("@")
            shares, price = sh, pr
        out.append(
            [
                r.get("saved_at", ""),
                r.get("updated", ""),
                side or r.get("source", ""),
                code,
                shares,
                price,
                f"{float(r.get('cash') or 0):,.2f}",
                note,
            ]
        )
    return out


def _session_dropdown(account: str):
    choices = list_sim_session_choices(account, limit=30) if account else []
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def _session_detail(path: str):
    empty_pos: list[list] = []
    empty_hist: list[list] = []
    if not path:
        return "请选择一次练习存档", empty_pos, empty_hist, None
    try:
        data = load_sim_session(path)
    except Exception as e:
        return f"读取失败: {e}", empty_pos, empty_hist, None

    summary = data.get("summary") or {}
    period = data.get("period") or {}
    lines = [
        f"账户: {data.get('account', '')}",
        f"保存时间: {data.get('saved_at', '')}",
        f"估值日: {data.get('asof', '')}",
        f"备注: {data.get('note') or '（无）'}",
        f"现金={float(summary.get('cash') or 0):,.2f}  "
        f"市值={float(summary.get('market_value') or 0):,.2f}  "
        f"权益={float(summary.get('equity') or 0):,.2f}",
        f"持仓数={summary.get('open_positions', 0)}  "
        f"流水条数={summary.get('history_rows', 0)}",
    ]
    if period.get("error"):
        lines.append(f"区间汇总失败: {period['error']}")
    elif period:
        if period.get("text"):
            lines.append(str(period["text"]))
        else:
            lines.append(
                f"区间 {period.get('start')}→{period.get('end')}  "
                f"盈亏={period.get('pnl')}  收益率={period.get('ret')}  "
                f"最大回撤={period.get('max_drawdown')}"
            )
    summary_text = "\n".join(lines)

    pos_rows: list[list] = []
    for p in data.get("positions") or []:
        vt = p.get("vt_symbol") or ""
        try:
            name = stock_name(vt)
        except Exception:
            name = vt
        pnl = p.get("pnl")
        pct = p.get("pnl_pct")
        pos_rows.append(
            [
                vt,
                name,
                p.get("shares"),
                f"{float(p.get('cost') or 0):.2f}",
                f"{float(p['close']):.2f}" if p.get("close") is not None else "",
                f"{pnl:,.2f}" if isinstance(pnl, (int, float)) else "",
                f"{pct:.2f}%" if isinstance(pct, (int, float)) else "",
            ]
        )

    hist_rows: list[list] = []
    for r in reversed(data.get("history") or []):
        hist_rows.append(
            [
                r.get("saved_at", ""),
                r.get("updated", ""),
                r.get("source", ""),
                f"{float(r.get('cash') or 0):,.2f}",
                len(r.get("positions") or {}),
                r.get("note") or "",
            ]
        )

    fig = None
    curve = data.get("equity_curve") or []
    if curve:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        xs = [str(c.get("date") or "") for c in curve]
        ys = [c.get("wealth") for c in curve]
        x_idx = list(range(len(xs)))
        ax.plot(x_idx, ys, color="#1f77b4", linewidth=1.5)
        ax.set_title("存档时权益曲线（现金+市值）")
        _sparse_date_ticks(ax, xs)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    return summary_text, pos_rows, hist_rows, fig


def build_app() -> gr.Blocks:
    with gr.Blocks(title="模拟操盘") as demo:
        gr.Markdown("# 模拟操盘训练\n仅操作名称前缀为 `模拟_` 的账户；成交默认用所选交易日收盘价。")

        with gr.Tab("下单"):
            with gr.Row():
                account_dd = gr.Dropdown(
                    choices=list_sim_accounts(),
                    label="模拟账户",
                    interactive=True,
                )
                refresh_btn = gr.Button("刷新账户列表", scale=0)

            with gr.Accordion("创建 / 重置 / 保存练习结果", open=False):
                with gr.Row():
                    new_name = gr.Textbox(label="新账户名（可只填后缀）", placeholder="练习1")
                    new_cash = gr.Number(label="初始/重置资金", value=100000)
                    new_updated = gr.Textbox(
                        label="初始交易日",
                        value=date.today().isoformat(),
                    )
                with gr.Row():
                    session_note = gr.Textbox(
                        label="本次备注（保存用）",
                        placeholder="如：一周突破练习",
                    )
                    session_asof = gr.Textbox(
                        label="汇总估值日",
                        value=date.today().isoformat(),
                    )
                with gr.Row():
                    create_btn = gr.Button("创建账户")
                    save_session_btn = gr.Button("保存本次结果", variant="primary")
                    reset_btn = gr.Button("重置当前练习账户", variant="stop")
                save_before_reset = gr.Checkbox(
                    label="重置前先保存当次信息（历史流水 + 整体汇总）",
                    value=True,
                )
                create_msg = gr.Textbox(label="账户操作结果", interactive=False, lines=4)

            with gr.Row():
                side = gr.Radio(choices=["买入", "卖出"], value="买入", label="方向")
                code_pick = gr.Dropdown(
                    choices=_stock_choices(),
                    value=None,
                    label="选股（下拉）",
                    filterable=True,
                    allow_custom_value=True,
                )
                code_manual = gr.Textbox(
                    label="手动输入代码（优先）",
                    placeholder="如 000333 或 000333.SZSE",
                )
            with gr.Row():
                shares_pick = gr.Dropdown(
                    choices=[100, 200, 500, 1000],
                    value=100,
                    label="数量（下拉，随板块变化）",
                    allow_custom_value=True,
                )
                shares_manual = gr.Textbox(
                    label="手动输入数量（优先）",
                    placeholder="如 300",
                    value="100",
                )
            with gr.Row():
                trade_date = gr.Textbox(
                    label="交易日",
                    value=date.today().isoformat(),
                )
                prev_day_btn = gr.Button("上一日", scale=0)
                next_day_btn = gr.Button("下一日", scale=0)
                price = gr.Textbox(label="价格（默认收盘）", placeholder="留空=当日收盘")
                fill_price_btn = gr.Button("填入收盘价", scale=0)
            update_data = gr.Checkbox(label="下单前更新该股日线", value=False)
            price_hint = gr.Textbox(label="价格提示", interactive=False)
            order_check = gr.Textbox(label="买卖合理性判断", interactive=False, lines=2)
            trend_plot = gr.Plot(label="蜡烛图+成交量（截至所选交易日，近 60 日）")
            submit_btn = gr.Button("提交订单", variant="primary")
            order_msg = gr.Textbox(label="下单结果", interactive=False, lines=3)

        with gr.Tab("持仓"):
            with gr.Row():
                pos_account = gr.Dropdown(choices=list_sim_accounts(), label="模拟账户")
                pos_asof = gr.Textbox(label="估值日（当前模拟时间）", value=date.today().isoformat())
                pos_refresh = gr.Button("刷新分析", scale=0)
            pos_summary = gr.Textbox(
                label="区间汇总 + 期末持仓",
                interactive=False,
                lines=8,
                max_lines=12,
            )
            equity_plot = gr.Plot(label="权益曲线（现金+市值；回撤按此计算）")
            pos_table = gr.Dataframe(
                headers=["代码", "名称", "股数", "成本", "昨收", "收盘", "涨跌幅", "当日盈亏", "持仓盈亏", "当前盈亏比"],
                label="期末持仓明细（盈亏比=相对当前成本；卖出后再买按新成本）",
                interactive=False,
            )

        with gr.Tab("历史"):
            with gr.Row():
                hist_account = gr.Dropdown(choices=list_sim_accounts(), label="模拟账户")
            with gr.Tab("流水"):
                with gr.Row():
                    hist_date = gr.Textbox(label="筛选交易日（可空）", placeholder="YYYY-MM-DD")
                    hist_limit = gr.Number(label="最近 N 条", value=50, precision=0)
                    hist_refresh = gr.Button("刷新流水", scale=0)
                hist_table = gr.Dataframe(
                    headers=["记录时间", "交易日", "动作", "代码", "数量", "价格", "成交后现金", "备注"],
                    label="当前账本流水",
                    interactive=False,
                )
            with gr.Tab("练习存档"):
                gr.Markdown(
                    "查看「保存本次结果」写入的存档（`accounts/<模拟_>/sessions/`）。"
                )
                with gr.Row():
                    session_pick = gr.Dropdown(
                        choices=[],
                        label="存档列表",
                        interactive=True,
                    )
                    session_refresh = gr.Button("刷新存档列表", scale=0)
                session_summary = gr.Textbox(
                    label="存档汇总",
                    interactive=False,
                    lines=6,
                )
                session_equity_plot = gr.Plot(label="存档财富曲线")
                session_pos_table = gr.Dataframe(
                    headers=["代码", "名称", "股数", "成本", "收盘", "持仓盈亏", "盈亏比"],
                    label="存档时持仓",
                    interactive=False,
                )
                session_hist_table = gr.Dataframe(
                    headers=["记录时间", "交易日", "来源", "现金", "持仓数", "备注"],
                    label="存档时完整流水",
                    interactive=False,
                )

        def _sync_accounts():
            dd = _refresh_accounts()
            return dd, dd, dd

        def _on_hist_account(account: str):
            return _session_dropdown(account)

        refresh_btn.click(_sync_accounts, outputs=[account_dd, pos_account, hist_account])
        create_btn.click(
            _create_account,
            inputs=[new_name, new_cash, new_updated],
            outputs=[create_msg, account_dd],
        ).then(_sync_accounts, outputs=[account_dd, pos_account, hist_account])
        save_session_btn.click(
            _save_session,
            inputs=[account_dd, session_asof, session_note],
            outputs=[create_msg],
        ).then(
            _on_hist_account,
            inputs=[account_dd],
            outputs=[session_pick],
        )
        reset_btn.click(
            _reset_account,
            inputs=[
                account_dd,
                new_cash,
                new_updated,
                save_before_reset,
                session_asof,
                session_note,
            ],
            outputs=[create_msg, account_dd],
        ).then(_sync_accounts, outputs=[account_dd, pos_account, hist_account]).then(
            _on_hist_account,
            inputs=[account_dd],
            outputs=[session_pick],
        )

        # 交易日联动保存估值日
        trade_date.change(
            lambda d: d,
            inputs=[trade_date],
            outputs=[session_asof],
        )

        check_inputs = [
            account_dd,
            side,
            code_pick,
            code_manual,
            shares_pick,
            shares_manual,
            trade_date,
            price,
        ]

        fill_price_btn.click(
            _refresh_from_symbols,
            inputs=[code_pick, code_manual, trade_date, shares_pick, shares_manual],
            outputs=[price, price_hint, trend_plot, shares_pick],
        ).then(_evaluate_order, inputs=check_inputs, outputs=[order_check])
        prev_day_btn.click(
            lambda p, m, d, sp, sm: _step_from_symbols(p, m, d, -1, sp, sm),
            inputs=[code_pick, code_manual, trade_date, shares_pick, shares_manual],
            outputs=[trade_date, price, price_hint, trend_plot, shares_pick],
        )
        next_day_btn.click(
            lambda p, m, d, sp, sm: _step_from_symbols(p, m, d, 1, sp, sm),
            inputs=[code_pick, code_manual, trade_date, shares_pick, shares_manual],
            outputs=[trade_date, price, price_hint, trend_plot, shares_pick],
        )
        trade_date.change(
            _refresh_from_symbols,
            inputs=[code_pick, code_manual, trade_date, shares_pick, shares_manual],
            outputs=[price, price_hint, trend_plot, shares_pick],
        ).then(_evaluate_order, inputs=check_inputs, outputs=[order_check])
        code_pick.change(
            _on_stock_pick,
            inputs=[code_pick, trade_date, shares_pick, shares_manual],
            outputs=[code_manual, price, price_hint, trend_plot, shares_pick, shares_manual],
        ).then(_evaluate_order, inputs=check_inputs, outputs=[order_check])
        code_manual.blur(
            _refresh_from_symbols,
            inputs=[code_pick, code_manual, trade_date, shares_pick, shares_manual],
            outputs=[price, price_hint, trend_plot, shares_pick],
        ).then(_evaluate_order, inputs=check_inputs, outputs=[order_check])
        code_manual.change(
            _refresh_from_symbols,
            inputs=[code_pick, code_manual, trade_date, shares_pick, shares_manual],
            outputs=[price, price_hint, trend_plot, shares_pick],
        ).then(_evaluate_order, inputs=check_inputs, outputs=[order_check])
        shares_pick.change(
            _on_shares_pick,
            inputs=[shares_pick, shares_manual],
            outputs=[shares_manual],
        ).then(_evaluate_order, inputs=check_inputs, outputs=[order_check])
        for _comp in (side, shares_manual, price, account_dd):
            _comp.change(_evaluate_order, inputs=check_inputs, outputs=[order_check])

        def _after_order(*args):
            msg = _submit_order(*args)
            rows, summary, fig = _refresh_positions(args[0], args[6])
            hist = _history_table(args[0], "", 50)
            chk = _evaluate_order(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]
            )
            return msg, rows, summary, fig, hist, chk

        submit_btn.click(
            _after_order,
            inputs=[
                account_dd,
                side,
                code_pick,
                code_manual,
                shares_pick,
                shares_manual,
                trade_date,
                price,
                update_data,
            ],
            outputs=[
                order_msg,
                pos_table,
                pos_summary,
                equity_plot,
                hist_table,
                order_check,
            ],
        )

        pos_refresh.click(
            _refresh_positions,
            inputs=[pos_account, pos_asof],
            outputs=[pos_table, pos_summary, equity_plot],
        )
        pos_account.change(
            _refresh_positions,
            inputs=[pos_account, pos_asof],
            outputs=[pos_table, pos_summary, equity_plot],
        )
        pos_asof.change(
            _refresh_positions,
            inputs=[pos_account, pos_asof],
            outputs=[pos_table, pos_summary, equity_plot],
        )
        hist_refresh.click(
            _history_table,
            inputs=[hist_account, hist_date, hist_limit],
            outputs=[hist_table],
        )
        hist_account.change(
            _history_table,
            inputs=[hist_account, hist_date, hist_limit],
            outputs=[hist_table],
        ).then(
            _on_hist_account,
            inputs=[hist_account],
            outputs=[session_pick],
        )
        session_refresh.click(
            _on_hist_account,
            inputs=[hist_account],
            outputs=[session_pick],
        ).then(
            _session_detail,
            inputs=[session_pick],
            outputs=[
                session_summary,
                session_pos_table,
                session_hist_table,
                session_equity_plot,
            ],
        )
        session_pick.change(
            _session_detail,
            inputs=[session_pick],
            outputs=[
                session_summary,
                session_pos_table,
                session_hist_table,
                session_equity_plot,
            ],
        )

        # 账户下拉联动
        account_dd.change(
            lambda a, d: (a, d or date.today().isoformat()),
            inputs=[account_dd, trade_date],
            outputs=[pos_account, pos_asof],
        ).then(
            lambda a: a,
            inputs=[account_dd],
            outputs=[hist_account],
        )

        # 下单交易日同步为持仓估值日
        trade_date.change(
            lambda d: d,
            inputs=[trade_date],
            outputs=[pos_asof],
        )

    return demo


def main() -> None:
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
