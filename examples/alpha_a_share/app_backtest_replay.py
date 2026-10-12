"""
回测交易回放 Web UI（Gradio）

依赖：先跑 backtest_1y_livermore.py（默认 --save-replay）生成
  alpha_data/backtest_replay/<run_id>/

启动：
  .venv/bin/python examples/alpha_a_share/app_backtest_replay.py
  → http://127.0.0.1:7861
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from account_equity import close_on_or_before
from backtest_replay_store import list_replay_runs, load_replay
from paper_trade import resolve_vt_symbol
from stock_universe import stock_name

# 复用模拟操盘蜡烛绘制
from app_paper_trade import (  # noqa: E402
    TREND_LOOKBACK_BARS,
    _draw_candles,
    _empty_trend_fig,
    _end_idx_on_or_before,
    _load_daily_series,
)

POS_HEADERS = [
    "代码",
    "名称",
    "股数",
    "成本",
    "昨收",
    "收盘",
    "涨跌幅",
    "当日盈亏",
    "持仓盈亏",
    "盈亏比",
    "市值",
]

TRADE_HEADERS = ["时间", "代码", "名称", "方向", "动作", "数量", "价格", "成交额", "盈亏比", "理由"]

ALPHA_LAB_PATH = os.path.join(PROJECT_DIR, "alpha_data")


def _empty_equity():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, "无权益数据", ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _date_str(d) -> str:
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return str(d)[:10]


def _refresh_runs():
    choices = list_replay_runs(ALPHA_LAB_PATH)
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def _run_dates(daily: pl.DataFrame, trades: pl.DataFrame) -> list[str]:
    if not daily.is_empty() and "date" in daily.columns:
        return [_date_str(x) for x in daily["date"].to_list()]
    if not trades.is_empty() and "date" in trades.columns:
        return sorted({_date_str(x) for x in trades["date"].to_list()})
    return []


def _filter_day(df: pl.DataFrame, asof: str) -> pl.DataFrame:
    if df.is_empty() or not asof or "date" not in df.columns:
        return pl.DataFrame()
    return df.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("d")).filter(
        pl.col("d") == asof
    )


def _day_symbols(trades: pl.DataFrame, positions: pl.DataFrame, asof: str | None) -> list[str]:
    """优先当日成交标的（按成交时间），否则当日持仓。"""
    if not asof:
        return []
    day_tr = _filter_day(trades, asof)
    if not day_tr.is_empty():
        seen: list[str] = []
        for vt in day_tr["vt_symbol"].to_list():
            if vt not in seen:
                seen.append(vt)
        return seen
    day_pos = _filter_day(positions, asof)
    if not day_pos.is_empty():
        return day_pos.sort("market_value", descending=True)["vt_symbol"].to_list()
    return []


def _panels_for_day(data: dict, asof: str | None, symbol: str | None):
    eq = _equity_fig(data["daily"], asof)
    candle = (
        _symbol_candle(symbol, asof, data["trades"])
        if symbol and asof
        else _empty_trend_fig("当日无成交/持仓标的" if asof else "")
    )
    txt, tr, pos = _day_panels(data, asof)
    return eq, candle, txt, tr, pos


def _load_run(run_id: str):
    empty_dates = gr.update(choices=[], value=None)
    empty_sym = gr.update(choices=[], value=None)
    empty_state: list[str] = []
    if not run_id:
        return (
            "请选择回放 run",
            empty_dates,
            empty_sym,
            _empty_equity(),
            _empty_trend_fig(),
            "",
            [],
            [],
            empty_state,
        )
    try:
        data = load_replay(ALPHA_LAB_PATH, run_id)
    except Exception as e:
        return (
            f"加载失败: {e}",
            empty_dates,
            empty_sym,
            _empty_equity(),
            _empty_trend_fig(),
            "",
            [],
            [],
            empty_state,
        )

    meta = data["meta"]
    daily: pl.DataFrame = data["daily"]
    trades: pl.DataFrame = data["trades"]
    positions: pl.DataFrame = data["positions"]

    stats = meta.get("stats") or {}
    fee_txt = ""
    if meta.get("long_rate") is not None:
        fee_txt = (
            f"费率: 买={meta.get('long_rate')} 卖={meta.get('short_rate')}  "
            f"佣金={meta.get('commission_rate')} 印花税={meta.get('stamp_tax')} "
            f"（不免5）\n"
        )
    summary = (
        f"run={meta.get('run_id')}\n"
        f"策略={meta.get('strategy')}  信号={meta.get('signal_source')}  "
        f"X={meta.get('add_spacing_pct')}\n"
        f"区间={meta.get('start')} ~ {meta.get('end')}  资金={meta.get('capital')}\n"
        f"年化={stats.get('annual_return', float('nan')):.2f}%  "
        f"夏普={stats.get('sharpe_ratio', float('nan')):.2f}  "
        f"回撤={stats.get('max_ddpercent', float('nan')):.2f}%  "
        f"成交={meta.get('trade_count')}\n"
        f"{fee_txt}"
        f"目录={data['run_dir']}"
    )

    dates = _run_dates(daily, trades)
    date_value = dates[-1] if dates else None
    symbols = _day_symbols(trades, positions, date_value)
    sym_value = symbols[0] if symbols else None
    eq_fig, candle, day_txt, day_trades, day_pos = _panels_for_day(data, date_value, sym_value)

    return (
        summary,
        gr.update(choices=dates, value=date_value),
        gr.update(choices=symbols, value=sym_value),
        eq_fig,
        candle,
        day_txt,
        day_trades,
        day_pos,
        dates,
    )


def _equity_fig(daily: pl.DataFrame, asof: str | None):
    if daily.is_empty() or "balance" not in daily.columns:
        return _empty_equity()
    dates = [_date_str(x) for x in daily["date"].to_list()]
    bal = daily["balance"].to_list()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(range(len(dates)), bal, color="#2563eb", linewidth=1.5)
    if asof and asof in dates:
        i = dates.index(asof)
        ax.axvline(i, color="#f97316", linestyle="--", linewidth=1)
        ax.scatter([i], [bal[i]], color="#f97316", zorder=3)
        ax.set_title(f"权益曲线（标记日 {asof} 权益={bal[i]:,.2f}）")
    else:
        ax.set_title("权益曲线")
    ax.set_ylabel("权益")
    ax.grid(True, alpha=0.25)
    step = max(1, len(dates) // 8)
    ax.set_xticks(list(range(0, len(dates), step)))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def _symbol_candle(vt_symbol: str | None, asof: str | None, trades: pl.DataFrame):
    if not vt_symbol or not asof:
        return _empty_trend_fig()
    try:
        vt = resolve_vt_symbol(vt_symbol.split(".", 1)[0] if "." in vt_symbol else vt_symbol)
        # 若已是完整 vt 则保留
        if "." in vt_symbol:
            vt = vt_symbol
    except Exception:
        vt = vt_symbol

    series = _load_daily_series(vt)
    if series is None:
        return _empty_trend_fig(f"{vt} 无本地日线")
    dates, opens, highs, lows, closes, volumes = series
    end_idx = _end_idx_on_or_before(dates, asof)
    if end_idx is None:
        return _empty_trend_fig(f"{vt} 在 {asof} 前无数据")
    start_idx = max(0, end_idx - TREND_LOOKBACK_BARS + 1)
    xs = dates[start_idx : end_idx + 1]
    os_ = opens[start_idx : end_idx + 1]
    hs = highs[start_idx : end_idx + 1]
    ls = lows[start_idx : end_idx + 1]
    cs = closes[start_idx : end_idx + 1]
    vs = volumes[start_idx : end_idx + 1]
    try:
        name = stock_name(vt)
    except Exception:
        name = vt.split(".", 1)[0]

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(8, 4.8), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0]}
    )
    x_idx = list(range(len(xs)))
    _draw_candles(ax_price, os_, hs, ls, cs)

    # 买卖点：窗口内该标的成交
    if not trades.is_empty():
        day_map = {d: i for i, d in enumerate(xs)}
        tdf = trades.filter(pl.col("vt_symbol") == vt)
        for row in tdf.iter_rows(named=True):
            d = _date_str(row["date"])
            if d not in day_map:
                continue
            i = day_map[d]
            px = float(row["price"])
            if row["side"] == "买入":
                ax_price.scatter([i], [px], marker="^", s=70, color="#ef4444", zorder=5, label="买")
            else:
                ax_price.scatter([i], [px], marker="v", s=70, color="#22c55e", zorder=5, label="卖")

    # 去重图例
    handles, labels = ax_price.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax_price.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=8)

    ax_price.axvline(x_idx[-1], color="#f97316", linestyle="--", linewidth=1, alpha=0.7)
    ax_price.set_title(f"{vt} {name}（截至 {xs[-1]}，近 {len(xs)} 日）")
    ax_price.set_ylabel("价格")
    ax_price.grid(True, alpha=0.25)

    colors = ["#ef4444" if cs[i] >= os_[i] else "#22c55e" for i in range(len(cs))]
    ax_vol.bar(x_idx, vs, color=colors, width=0.6, alpha=0.8)
    ax_vol.set_ylabel("量")
    ax_vol.grid(True, alpha=0.2)
    step = max(1, len(xs) // 6)
    ax_vol.set_xticks(list(range(0, len(xs), step)))
    ax_vol.set_xticklabels([xs[i] for i in range(0, len(xs), step)], rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def _apply_trade_to_books(books: dict[str, dict], row: dict) -> float | None:
    """按一笔成交更新均价账本；若为卖出，返回卖出前成本（用于盈亏比）。"""
    vt = row.get("vt_symbol")
    if not vt:
        return None
    vol = float(row.get("volume") or 0)
    px = float(row.get("price") or 0)
    side = row.get("side") or ""
    d = _date_str(row.get("date"))
    book = books.setdefault(vt, {"shares": 0.0, "cost": 0.0, "last_buy_date": ""})
    sh = float(book["shares"])
    cost = float(book["cost"])
    sell_cost: float | None = None
    if side == "买入":
        if sh <= 1e-9:
            book["cost"] = px
            book["shares"] = vol
        else:
            book["cost"] = (sh * cost + vol * px) / (sh + vol)
            book["shares"] = sh + vol
        book["last_buy_date"] = d
    else:
        sell_cost = cost if cost > 1e-9 else None
        new_sh = sh - vol
        if new_sh <= 1e-9:
            books.pop(vt, None)
        else:
            book["shares"] = new_sh
    return sell_cost


def _cost_books_asof(trades: pl.DataFrame, asof: str) -> dict[str, dict]:
    """截至 asof（含当日）用成交还原均价成本；买入日记 last_buy_date。"""
    books: dict[str, dict] = {}
    if trades.is_empty():
        return books
    tt = trades.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("d"))
    tt = tt.filter(pl.col("d") <= asof).sort(["datetime", "tradeid"])
    for r in tt.iter_rows(named=True):
        _apply_trade_to_books(books, r)
    return books


def _trade_action_label(side: str, shares_before: float, volume: float) -> str:
    """买入→建仓/加仓；卖出→清仓/减仓。"""
    vol = float(volume or 0)
    sh0 = float(shares_before or 0)
    if side == "买入":
        return "建仓" if sh0 <= 1e-9 else "加仓"
    if side == "卖出":
        return "清仓" if sh0 - vol <= 1e-9 else "减仓"
    return side or "—"


def _enrich_trade_rows(trades: pl.DataFrame, asof: str) -> list[list]:
    """当日成交；含名称、建仓/加仓/减仓/清仓、策略理由；卖出附相对成本盈亏比。"""
    if trades.is_empty():
        return []
    tt = trades.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("d"))
    # 先回放到前一日，再逐笔吃当日成交，卖出用卖前成本算盈亏比
    books: dict[str, dict] = {}
    prior = tt.filter(pl.col("d") < asof).sort(["datetime", "tradeid"])
    for r in prior.iter_rows(named=True):
        _apply_trade_to_books(books, r)

    rows: list[list] = []
    day = tt.filter(pl.col("d") == asof).sort(["datetime", "tradeid"])
    for r in day.iter_rows(named=True):
        vt = r.get("vt_symbol") or ""
        side = r.get("side") or ""
        vol = float(r.get("volume") or 0)
        px = float(r.get("price") or 0)
        shares_before = float((books.get(vt) or {}).get("shares") or 0)
        action = _trade_action_label(side, shares_before, vol)
        try:
            name = stock_name(vt)
        except Exception:
            name = vt.split(".", 1)[0] if vt else ""
        sell_cost = _apply_trade_to_books(books, r)
        if side == "卖出" and sell_cost is not None and sell_cost > 1e-9:
            pnl_pct_s = f"{(px / sell_cost - 1.0) * 100:+.2f}%"
        else:
            pnl_pct_s = "—"
        reason = (r.get("reason") or "").strip() or "—"
        rows.append(
            [
                str(r.get("datetime", ""))[:19],
                vt,
                name,
                side,
                action,
                f"{vol:.0f}",
                f"{px:.3f}",
                f"{float(r.get('turnover') or 0):,.2f}",
                pnl_pct_s,
                reason,
            ]
        )
    return rows


def _enrich_pos_rows(positions: pl.DataFrame, trades: pl.DataFrame, asof: str) -> list[list]:
    """对齐模拟操盘：成本 / 昨收 / 涨跌 / 当日盈亏 / 持仓盈亏 / 盈亏比。"""
    if positions.is_empty():
        return []
    books = _cost_books_asof(trades, asof)
    pp = positions.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("d"))
    rows: list[list] = []
    for r in pp.filter(pl.col("d") == asof).sort("market_value", descending=True).iter_rows(named=True):
        vt = r.get("vt_symbol") or ""
        shares = float(r.get("shares") or 0)
        close = float(r.get("close") or 0)
        mv = float(r.get("market_value") or shares * close)
        book = books.get(vt) or {}
        cost = float(book.get("cost") or 0)
        last_buy = str(book.get("last_buy_date") or "")
        try:
            name = stock_name(vt)
        except Exception:
            name = vt.split(".", 1)[0] if vt else ""
        px, prev = close_on_or_before(vt, asof)
        if px is None:
            px = close if close > 0 else None
        if px is None:
            rows.append(
                [vt, name, f"{shares:.0f}", f"{cost:.3f}" if cost else "—", "—", "缺数据", "—", "—", "—", "—", f"{mv:,.2f}"]
            )
            continue
        chg_pct = ((px / prev) - 1.0) * 100 if prev is not None and prev > 1e-9 else None
        same_day_buy = bool(last_buy and last_buy == asof)
        if same_day_buy or prev is None:
            day_pnl_s = "—"
        else:
            day_pnl_s = f"{(px - prev) * shares:+.2f}"
        if cost > 1e-9:
            cost_pnl = (px - cost) * shares
            pnl_pct = (px / cost - 1.0) * 100
            cost_s = f"{cost:.3f}"
            cost_pnl_s = f"{cost_pnl:+.2f}"
            pnl_pct_s = f"{pnl_pct:+.2f}%"
        else:
            cost_s = cost_pnl_s = pnl_pct_s = "—"
        rows.append(
            [
                vt,
                name,
                f"{shares:.0f}",
                cost_s,
                f"{prev:.3f}" if prev is not None else "—",
                f"{px:.3f}",
                f"{chg_pct:+.2f}%" if chg_pct is not None else "—",
                day_pnl_s,
                cost_pnl_s,
                pnl_pct_s,
                f"{mv:,.2f}",
            ]
        )
    return rows


def _day_panels(data: dict, asof: str | None):
    if not asof:
        return "请选择日期", [], []
    daily: pl.DataFrame = data["daily"]
    trades: pl.DataFrame = data["trades"]
    positions: pl.DataFrame = data["positions"]
    meta = data["meta"]
    capital = float(meta.get("capital") or 0)

    day_row = None
    if not daily.is_empty():
        # date 列可能是 Date
        dd = daily.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("d"))
        hit = dd.filter(pl.col("d") == asof)
        if not hit.is_empty():
            day_row = hit.row(0, named=True)

    if day_row:
        bal = float(day_row.get("balance") or 0)
        ret_pct = (bal / capital - 1) * 100 if capital > 0 else float("nan")
        txt = (
            f"日期 {asof}\n"
            f"权益={bal:,.2f}  累计盈亏={bal - capital:+,.2f}（{ret_pct:+.2f}%）\n"
            f"当日净盈亏={float(day_row.get('net_pnl') or 0):+,.2f}  "
            f"持仓盈亏={float(day_row.get('holding_pnl') or 0):+,.2f}  "
            f"交易盈亏={float(day_row.get('trading_pnl') or 0):+,.2f}\n"
            f"成交笔数={day_row.get('trade_count')}  手续费={float(day_row.get('commission') or 0):,.2f}  "
            f"回撤%={float(day_row.get('ddpercent') or 0):+.2f}"
        )
    else:
        txt = f"日期 {asof}\n（无当日权益行）"

    trade_rows = _enrich_trade_rows(trades, asof)
    pos_rows = _enrich_pos_rows(positions, trades, asof)
    return txt, trade_rows, pos_rows


def _on_date_change(run_id: str, asof: str):
    """切交易日：刷新当日标的列表并自动选中当日第一笔成交。"""
    empty_sym = gr.update(choices=[], value=None)
    if not run_id or not asof:
        return empty_sym, _empty_equity(), _empty_trend_fig(), "请选择 run / 交易日", [], []
    data = load_replay(ALPHA_LAB_PATH, run_id)
    symbols = _day_symbols(data["trades"], data["positions"], asof)
    sym = symbols[0] if symbols else None
    eq, candle, txt, tr, pos = _panels_for_day(data, asof, sym)
    return gr.update(choices=symbols, value=sym), eq, candle, txt, tr, pos


def _on_symbol_change(run_id: str, asof: str, symbol: str):
    if not run_id or not asof or not symbol:
        return _empty_trend_fig()
    data = load_replay(ALPHA_LAB_PATH, run_id)
    return _symbol_candle(symbol, asof, data["trades"])


def _row_vt_from_table(table, row_idx: int, code_col: int) -> str | None:
    """从 Gradio Dataframe 取值中解析 vt_symbol。"""
    if table is None or row_idx is None or row_idx < 0:
        return None
    try:
        if hasattr(table, "iloc"):
            if row_idx >= len(table):
                return None
            val = table.iloc[row_idx, code_col]
        else:
            rows = table if isinstance(table, list) else list(table)
            if row_idx >= len(rows):
                return None
            row = rows[row_idx]
            val = row[code_col] if isinstance(row, (list, tuple)) else row
        return str(val).strip() if val is not None else None
    except Exception:
        return None


def _on_trade_select(run_id: str, asof: str, table, evt: gr.SelectData):
    """点击当日成交行 → 切换单股图。"""
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    # TRADE_HEADERS: 时间, 代码, ...
    vt = _row_vt_from_table(table, int(row), 1)
    if not vt:
        return gr.update(), _empty_trend_fig()
    fig = _on_symbol_change(run_id, asof, vt)
    return gr.update(value=vt), fig


def _on_pos_select(run_id: str, asof: str, table, evt: gr.SelectData):
    """点击当日持仓行 → 切换单股图。"""
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    # POS_HEADERS: 代码, 名称, ...
    vt = _row_vt_from_table(table, int(row), 0)
    if not vt:
        return gr.update(), _empty_trend_fig()
    fig = _on_symbol_change(run_id, asof, vt)
    return gr.update(value=vt), fig


def _shift_day(run_id: str, asof: str, dates: list[str], delta: int):
    if not dates:
        return gr.update()
    if asof in dates:
        i = dates.index(asof)
    else:
        i = len(dates) - 1
    j = max(0, min(len(dates) - 1, i + delta))
    return gr.update(value=dates[j])


def build_app() -> gr.Blocks:
    with gr.Blocks(title="回测交易回放") as demo:
        dates_state = gr.State([])
        gr.Markdown(
            "## 回测交易回放\n"
            "数据来自 `backtest_1y_livermore.py` 落盘的 `alpha_data/backtest_replay/`。\n"
            "布局：整体趋势 → 当日明细 → 单股图；**点击成交/持仓行**可切换单股。"
        )
        with gr.Row():
            run_dd = gr.Dropdown(choices=[], label="回放 run", interactive=True, scale=4)
            refresh_btn = gr.Button("刷新列表", scale=0)
        meta_box = gr.Textbox(label="run 摘要", interactive=False, lines=4)
        with gr.Row():
            prev_btn = gr.Button("上一日", scale=0)
            date_dd = gr.Dropdown(choices=[], label="交易日", interactive=True, scale=3)
            next_btn = gr.Button("下一日", scale=0)

        gr.Markdown("### 整体趋势")
        equity_plot = gr.Plot(label="权益曲线")

        gr.Markdown("### 当日明细")
        day_box = gr.Textbox(label="当日盈亏 / 权益", interactive=False, lines=5)
        trade_table = gr.Dataframe(
            headers=TRADE_HEADERS,
            label="当日成交（点击行切换下方单股图）",
            interactive=True,
            type="pandas",
        )
        pos_table = gr.Dataframe(
            headers=POS_HEADERS,
            label="当日持仓（点击行切换下方单股图）",
            interactive=True,
            type="pandas",
        )

        gr.Markdown("### 单股走势")
        with gr.Row():
            symbol_dd = gr.Dropdown(
                choices=[],
                label="当前标的（也可下拉切换）",
                interactive=True,
            )
        candle_plot = gr.Plot(label="标的走势 + 买卖点")

        demo.load(_refresh_runs, outputs=[run_dd])
        refresh_btn.click(_refresh_runs, outputs=[run_dd])
        run_dd.change(
            _load_run,
            inputs=[run_dd],
            outputs=[
                meta_box,
                date_dd,
                symbol_dd,
                equity_plot,
                candle_plot,
                day_box,
                trade_table,
                pos_table,
                dates_state,
            ],
        )
        date_dd.change(
            _on_date_change,
            inputs=[run_dd, date_dd],
            outputs=[symbol_dd, equity_plot, candle_plot, day_box, trade_table, pos_table],
        )
        symbol_dd.change(
            _on_symbol_change,
            inputs=[run_dd, date_dd, symbol_dd],
            outputs=[candle_plot],
        )
        trade_table.select(
            _on_trade_select,
            inputs=[run_dd, date_dd, trade_table],
            outputs=[symbol_dd, candle_plot],
        )
        pos_table.select(
            _on_pos_select,
            inputs=[run_dd, date_dd, pos_table],
            outputs=[symbol_dd, candle_plot],
        )
        prev_btn.click(
            lambda rid, asof, dates: _shift_day(rid, asof, dates, -1),
            inputs=[run_dd, date_dd, dates_state],
            outputs=[date_dd],
        )
        next_btn.click(
            lambda rid, asof, dates: _shift_day(rid, asof, dates, 1),
            inputs=[run_dd, date_dd, dates_state],
            outputs=[date_dd],
        )
    return demo


def main() -> None:
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True)


if __name__ == "__main__":
    main()
