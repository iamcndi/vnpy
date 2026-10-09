"""模拟操盘成交引擎：按交易日收盘价买卖，更新模拟账本。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXAMPLE_DIR)

from a_share_limits import (
    a_share_limit_prices,
    a_share_lot_size,
    is_a_share_limit_down,
    is_a_share_limit_up,
)
from livermore_positions_store import (
    SIM_PREFIX as STORE_SIM_PREFIX,
    create_account,
    list_accounts,
    list_history,
    load_positions,
    merge_broker_position,
    normalize_account,
    resolve_paths,
    save_positions,
)
from stock_universe import HK_STOCK_LIST, STOCK_LIST, stock_name

ALPHA_LAB_PATH = Path(PROJECT_DIR) / "alpha_data"
DAILY_DIR = ALPHA_LAB_PATH / "daily"

SIM_PREFIX = STORE_SIM_PREFIX
COMMISSION_RATE_BUY = 0.00025
COMMISSION_RATE_SELL = 0.00125


@dataclass
class OrderResult:
    ok: bool
    message: str
    book: dict | None = None


@dataclass
class OrderCheck:
    """下单前合理性检查：reject 不可下；warn 可下但提示；ok 正常。"""

    level: str  # ok | warn | reject
    message: str

    @property
    def can_submit(self) -> bool:
        return self.level != "reject"


def ensure_sim_account_name(name: str) -> str:
    """强制账户名以「模拟_」开头。"""
    acc = normalize_account(name)
    if acc is None:
        raise ValueError("账户名不能为空")
    if not acc.startswith(SIM_PREFIX):
        acc = f"{SIM_PREFIX}{acc}"
    return acc


def list_sim_accounts() -> list[str]:
    return [a for a in list_accounts(include_sim=True) if a.startswith(SIM_PREFIX)]


def create_sim_account(
    name: str,
    cash: float,
    *,
    updated: str = "",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    acc = ensure_sim_account_name(name)
    if cash < 0:
        raise ValueError("初始资金不能为负")
    if not updated:
        updated = date.today().isoformat()
    return create_account(acc, cash=float(cash), updated=updated, overwrite=overwrite)


def reset_sim_account(
    name: str,
    cash: float,
    *,
    updated: str = "",
) -> tuple[Path, Path]:
    """重置模拟账户为空仓 + 指定初始资金（覆盖已有持仓与流水目录内快照）。"""
    acc = ensure_sim_account_name(name)
    if acc not in list_accounts():
        raise FileNotFoundError(f"模拟账户不存在: {acc}")
    if cash < 0:
        raise ValueError("初始资金不能为负")
    if not updated:
        updated = date.today().isoformat()
    # overwrite 重建空仓快照；另追加一条重置说明
    pos_path, hist_path = create_account(
        acc, cash=float(cash), updated=updated, overwrite=True
    )
    book = load_positions(account=acc)
    save_positions(
        book,
        note=f"重置练习账户 初始资金={cash:.2f} 起始日={updated}",
        source="paper_trade_reset",
        account=acc,
        force_history=True,
    )
    return pos_path, hist_path


def sessions_dir(account: str) -> Path:
    acc = ensure_sim_account_name(account)
    pos_path, _ = resolve_paths(acc)
    d = pos_path.parent / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_sim_session(
    account: str,
    *,
    asof: str = "",
    note: str = "",
) -> Path:
    """
    保存当次练习结果：当前账本快照、完整历史流水、股票口径区间汇总。
    写入 alpha_data/accounts/<模拟_xxx>/sessions/session_*.json
    """
    import json
    from datetime import datetime

    from account_equity import (
        close_on_or_before,
        compute_equity_period,
        stock_cost_basis,
        stock_market_value,
    )

    acc = ensure_sim_account_name(account)
    if acc not in list_accounts():
        raise FileNotFoundError(f"模拟账户不存在: {acc}")

    asof = (asof or "").strip() or date.today().isoformat()
    book = load_positions(account=acc)
    _, hist_path = resolve_paths(acc)
    history = list_history(path=hist_path)

    try:
        period = compute_equity_period(acc, asof)
        equity_curve = [
            {"date": d, "wealth": w} for d, w in zip(period.dates, period.equities)
        ]
        period_summary = {
            "start": period.start,
            "end": period.end,
            "sample_days": len(period.dates),
            "start_mv": period.start_equity,
            "end_mv": period.end_equity,
            "start_nav": period.start_nav,
            "end_nav": period.end_nav,
            "realized": period.realized,
            "pnl": period.pnl,
            "ret": period.ret,
            "ret_base": period.ret_base,
            "max_drawdown": period.max_drawdown,
            "cash_end": period.cash_end,
            "text": period.summary,
        }
    except Exception as e:
        period_summary = {"error": str(e)}
        equity_curve = []

    positions_detail = []
    for vt, pos in (book.get("positions") or {}).items():
        shares = int(pos.get("shares") or 0)
        if shares <= 0:
            continue
        cost = float(pos.get("cost") or 0)
        px, _ = close_on_or_before(vt, asof)
        row = {
            "vt_symbol": vt,
            "shares": shares,
            "cost": cost,
            "close": px,
        }
        if px is not None and cost > 0:
            row["pnl"] = (px - cost) * shares
            row["pnl_pct"] = (px / cost - 1.0) * 100
        positions_detail.append(row)

    mv = stock_market_value(book, asof)
    cash = float(book.get("cash") or 0)
    payload = {
        "account": acc,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof": asof,
        "note": note or "",
        "snapshot": book,
        "summary": {
            "cash": cash,
            "market_value": mv,
            "equity": cash + mv,
            "cost_basis": stock_cost_basis(book),
            "open_positions": len(positions_detail),
            "history_rows": len(history),
        },
        "positions": positions_detail,
        "period": period_summary,
        "equity_curve": equity_curve,
        "history": history,
    }

    out_dir = sessions_dir(acc)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"session_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def list_sim_sessions(account: str, limit: int = 20) -> list[Path]:
    acc = ensure_sim_account_name(account)
    d = sessions_dir(acc)
    rows = sorted(d.glob("session_*.json"), reverse=True)
    return rows[:limit] if limit else rows


def load_sim_session(path: str | Path) -> dict:
    """读取一次练习存档 JSON。"""
    import json

    p = Path(path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def session_choice_label(path: Path, payload: dict | None = None) -> str:
    """下拉显示：文件名 | 保存时间 | 备注 | 区间盈亏。"""
    data = payload
    if data is None:
        try:
            data = load_sim_session(path)
        except Exception:
            return path.name
    saved = data.get("saved_at") or ""
    note = (data.get("note") or "").strip() or "无备注"
    period = data.get("period") or {}
    pnl = period.get("pnl")
    pnl_s = f"盈亏={pnl:+,.2f}" if isinstance(pnl, (int, float)) else ""
    asof = data.get("asof") or ""
    parts = [path.name, saved, f"估值={asof}", note]
    if pnl_s:
        parts.append(pnl_s)
    return " | ".join(parts)


def list_sim_session_choices(account: str, limit: int = 30) -> list[tuple[str, str]]:
    """[(显示名, 文件路径), ...]，供 Gradio Dropdown。"""
    out: list[tuple[str, str]] = []
    for p in list_sim_sessions(account, limit=limit):
        try:
            data = load_sim_session(p)
            label = session_choice_label(p, data)
        except Exception as e:
            label = f"{p.name}（读取失败: {e}）"
        out.append((label, str(p)))
    return out


@dataclass
class RoundStats:
    """本轮练习统计：自上次重置起，卖出平仓算一笔。"""

    since_label: str
    closed: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0  # 正数：平均单笔亏损金额
    payoff: float | None = None  # 平均盈 / 平均亏
    total_realized: float = 0.0
    open_symbols: int = 0

    @property
    def win_rate(self) -> float | None:
        decided = self.wins + self.losses
        if decided <= 0:
            return None
        return self.wins / decided

    @property
    def summary(self) -> str:
        wr = f"{self.win_rate * 100:.1f}%" if self.win_rate is not None else "n/a"
        if self.payoff is None:
            pay = "n/a（无亏损笔）" if self.wins and not self.losses else "n/a"
        else:
            pay = f"{self.payoff:.2f}"
        return (
            f"【本轮练习统计】自 {self.since_label}（重置起算）\n"
            f"已平仓={self.closed} 笔  胜={self.wins}  负={self.losses}"
            + (f"  平={self.breakeven}" if self.breakeven else "")
            + f"  胜率={wr}\n"
            f"平均盈利={self.avg_win:+,.2f}  平均亏损={-self.avg_loss:,.2f}  "
            f"盈亏比(均盈/均亏)={pay}\n"
            f"已实现合计={self.total_realized:+,.2f}  未平仓标的={self.open_symbols}"
        )


def _parse_paper_trade_note(note: str) -> tuple[str, str, int, float] | None:
    """解析『模拟买入/卖出 代码 数量@价格』→ (side, code, shares, price)。"""
    note = (note or "").strip()
    if "模拟买入" in note:
        side = "buy"
        body = note.replace("模拟买入", "", 1).strip()
    elif "模拟卖出" in note:
        side = "sell"
        body = note.replace("模拟卖出", "", 1).strip()
    else:
        return None
    parts = body.split()
    if len(parts) < 2 or "@" not in parts[1]:
        return None
    code = parts[0]
    sh_s, _, px_s = parts[1].partition("@")
    try:
        shares = int(float(sh_s))
        price = float(px_s)
    except ValueError:
        return None
    if shares <= 0 or price <= 0:
        return None
    return side, code, shares, price


def _snap_share_map(row: dict) -> dict[str, tuple[int, float]]:
    out: dict[str, tuple[int, float]] = {}
    for vt, pos in (row.get("positions") or {}).items():
        sh = int(pos.get("shares") or 0)
        if sh <= 0:
            continue
        out[vt] = (sh, float(pos.get("cost") or 0))
    return out


def compute_round_stats(account: str) -> RoundStats:
    """
    自最近一次 paper_trade_reset 起：
    - 某标的从有仓到清仓算一笔已平仓
    - 盈亏比 = 平均盈利金额 / 平均亏损金额
    """
    from account_equity import close_on_or_before

    acc = ensure_sim_account_name(account)
    _, hist_path = resolve_paths(acc)
    rows = list_history(path=hist_path)
    if not rows:
        return RoundStats(since_label="（暂无流水）")

    reset_i: int | None = None
    since_label = "账户起始"
    for i, r in enumerate(rows):
        src = r.get("source") or ""
        note = r.get("note") or ""
        if src == "paper_trade_reset" or "重置练习账户" in note:
            reset_i = i
            since_label = (r.get("saved_at") or r.get("updated") or "最近重置").strip()

    if reset_i is not None:
        prev_map = _snap_share_map(rows[reset_i])
        segment = rows[reset_i + 1 :]
    else:
        prev_map = {}
        segment = rows

    # vt -> shares, cost, accrued_realized in current open round
    open_lots: dict[str, tuple[int, float, float]] = {}
    closed_pnls: list[float] = []

    for r in segment:
        cur_map = _snap_share_map(r)
        parsed = _parse_paper_trade_note(r.get("note") or "")
        trade_code = parsed[1] if parsed else ""
        trade_side = parsed[0] if parsed else ""
        trade_px = parsed[3] if parsed else 0.0

        for vt in set(prev_map) | set(cur_map):
            old_sh, old_cost = prev_map.get(vt, (0, 0.0))
            new_sh, new_cost = cur_map.get(vt, (0, 0.0))
            if new_sh == old_sh:
                if new_sh > 0:
                    acc_r = open_lots.get(vt, (0, 0.0, 0.0))[2]
                    open_lots[vt] = (new_sh, new_cost, acc_r)
                continue

            code_key = vt.split(".", 1)[0]
            use_note = bool(parsed and trade_code == code_key)

            if new_sh > old_sh:
                if old_sh <= 0:
                    px = trade_px if use_note and trade_side == "buy" else new_cost
                    open_lots[vt] = (new_sh, float(px), 0.0)
                else:
                    _, _, acc_r = open_lots.get(vt, (old_sh, old_cost, 0.0))
                    open_lots[vt] = (new_sh, new_cost, acc_r)
            else:
                sold = old_sh - new_sh
                _, lot_cost, acc_r = open_lots.get(vt, (old_sh, old_cost, 0.0))
                if use_note and trade_side == "sell":
                    px = trade_px
                else:
                    u = (r.get("updated") or "").strip()
                    close_px, _ = close_on_or_before(vt, u) if u else (None, None)
                    px = float(close_px) if close_px else lot_cost
                acc_r += (px - lot_cost) * sold
                if new_sh <= 0:
                    closed_pnls.append(acc_r)
                    open_lots.pop(vt, None)
                else:
                    open_lots[vt] = (new_sh, lot_cost, acc_r)

        prev_map = cur_map

    wins = [p for p in closed_pnls if p > 1e-9]
    losses = [p for p in closed_pnls if p < -1e-9]
    be = len(closed_pnls) - len(wins) - len(losses)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = (-sum(losses) / len(losses)) if losses else 0.0
    payoff = (avg_win / avg_loss) if losses and avg_loss > 1e-9 else None

    return RoundStats(
        since_label=since_label or "账户起始",
        closed=len(closed_pnls),
        wins=len(wins),
        losses=len(losses),
        breakeven=be,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff=payoff,
        total_realized=sum(closed_pnls),
        open_symbols=len(open_lots),
    )


def resolve_vt_symbol(raw: str) -> str:
    """解析代码为 vt_symbol；池外若本地已有日线也可。"""
    s = raw.strip().upper()
    if not s:
        raise ValueError("代码不能为空")
    if "." in s:
        vt = s
        if (DAILY_DIR / f"{vt}.parquet").exists():
            return vt
        try:
            stock_name(vt)
            return vt
        except KeyError as exc:
            raise ValueError(f"未找到代码或日线: {raw}") from exc

    for code, exch, _ in STOCK_LIST + HK_STOCK_LIST:
        if code == s:
            return f"{code}.{exch}"

    matches = sorted(DAILY_DIR.glob(f"{s}.*.parquet"))
    if len(matches) == 1:
        return matches[0].stem
    if len(matches) > 1:
        raise ValueError(f"代码 {raw} 对应多个市场: {[p.stem for p in matches]}")
    raise ValueError(f"未在股票池找到代码，且无本地日线: {raw}")


def _symbol_name(vt_symbol: str) -> str:
    try:
        return stock_name(vt_symbol)
    except KeyError:
        return vt_symbol.split(".", 1)[0]


def _load_daily_df(vt_symbol: str) -> pl.DataFrame | None:
    path = DAILY_DIR / f"{vt_symbol}.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path).sort("datetime")
    if df.height < 1:
        return None
    return df


def _date_str(dt) -> str:
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d")
    return str(dt)[:10]


def resolve_trade_price(vt_symbol: str, trade_date: str) -> float:
    """返回该股在交易日的日线收盘价。"""
    close, _, _ = _bar_on_date(vt_symbol, trade_date)
    return close


def _bar_on_date(
    vt_symbol: str,
    trade_date: str,
) -> tuple[float, float | None, str]:
    """
    返回 (当日收盘, 昨收, 实际交易日)。
    无当日日线则抛 ValueError。
    """
    df = _load_daily_df(vt_symbol)
    if df is None:
        raise ValueError(f"{vt_symbol} 无本地日线，请先下载")

    dates = [_date_str(d) for d in df["datetime"].to_list()]
    if trade_date not in dates:
        raise ValueError(f"{vt_symbol} 在 {trade_date} 无日线（非交易日或未下载）")

    idx = dates.index(trade_date)
    close = float(df["close"][idx])
    pre_close = float(df["close"][idx - 1]) if idx > 0 else None
    return close, pre_close, trade_date


def _maybe_update_daily(vt_symbol: str) -> None:
    from vnpy.alpha import AlphaLab
    from datafeed import download_daily_data

    code, _, exch = vt_symbol.partition(".")
    name = _symbol_name(vt_symbol)
    lab = AlphaLab(str(ALPHA_LAB_PATH))
    download_daily_data(lab, [(code, exch, name)])


def _normalize_side(side: str) -> str | None:
    side_key = side.strip().lower()
    if side_key in ("买入", "buy", "b"):
        return "buy"
    if side_key in ("卖出", "sell", "s"):
        return "sell"
    return None


def check_order(
    account: str,
    side: str,
    code: str,
    shares: int | str,
    trade_date: str,
    price: float | None = None,
) -> OrderCheck:
    """判断买入/卖出是否合理（不改账本）。"""
    try:
        acc = ensure_sim_account_name(account)
    except ValueError as e:
        return OrderCheck("reject", str(e))
    if acc not in list_accounts():
        return OrderCheck("reject", f"模拟账户不存在: {acc}")

    side_key = _normalize_side(side)
    if side_key is None:
        return OrderCheck("reject", f"无效方向: {side}")

    try:
        shares_i = int(float(shares))
    except (TypeError, ValueError):
        return OrderCheck("reject", "数量须为正整数")
    if shares_i <= 0:
        return OrderCheck("reject", "数量须 > 0")

    trade_date = (trade_date or "").strip()
    if len(trade_date) != 10:
        return OrderCheck("reject", "交易日格式须为 YYYY-MM-DD")

    try:
        vt = resolve_vt_symbol(code)
    except ValueError as e:
        return OrderCheck("reject", str(e))

    lot = a_share_lot_size(vt)
    if shares_i < lot or shares_i % lot != 0:
        return OrderCheck("reject", f"数量须为 {lot} 的整数倍（该标的整手={lot}）")

    try:
        day_close, pre_close, _ = _bar_on_date(vt, trade_date)
    except ValueError as e:
        return OrderCheck("reject", str(e))

    px = float(price) if price is not None else day_close
    if px <= 0:
        return OrderCheck("reject", "价格须 > 0")

    name = _symbol_name(vt)
    notes: list[str] = []
    level = "ok"

    if pre_close is None or pre_close <= 0:
        notes.append("无昨收，未做涨跌停检查")
        level = "warn"
    else:
        bounds = a_share_limit_prices(vt, pre_close, name)
        if bounds is not None:
            down, up = bounds
            if px < down - 1e-9 or px > up + 1e-9:
                return OrderCheck(
                    "reject",
                    f"成交价 {px:.2f} 超出涨跌停板 [{down:.2f}, {up:.2f}]",
                )
            if side_key == "buy" and is_a_share_limit_up(vt, day_close, pre_close, name):
                return OrderCheck(
                    "reject",
                    f"涨停不可买（{trade_date} 收盘 {day_close:.2f}）",
                )
            if side_key == "sell" and is_a_share_limit_down(vt, day_close, pre_close, name):
                return OrderCheck(
                    "reject",
                    f"跌停不可卖（{trade_date} 收盘 {day_close:.2f}）",
                )

    if abs(px - day_close) / day_close > 0.02:
        notes.append(f"价格偏离收盘 {((px / day_close) - 1) * 100:+.2f}%")
        level = "warn"

    book = load_positions(account=acc)
    cash = float(book.get("cash") or 0)
    existing = (book.get("positions") or {}).get(vt)
    held = int((existing or {}).get("shares") or 0)
    cost = float((existing or {}).get("cost") or 0)
    notional = px * shares_i

    if side_key == "buy":
        fee = notional * COMMISSION_RATE_BUY
        need = notional + fee
        if cash + 1e-9 < need:
            return OrderCheck(
                "reject",
                f"现金不足: 需要 {need:,.2f}（含费），可用 {cash:,.2f}",
            )
        notes.append(f"买入 {shares_i} 股约 {need:,.2f}（含费），剩余现金约 {cash - need:,.2f}")
        if held > 0:
            notes.append(f"加仓：原持有 {held} 股，成本 {cost:.3f}")
            level = "warn"
        if cash > 0 and need / cash > 0.5:
            notes.append(f"将占用现金 {need / cash * 100:.0f}%（>50%）")
            level = "warn"
        mv_other = 0.0
        for k, p in (book.get("positions") or {}).items():
            if k == vt:
                continue
            sh = int(p.get("shares") or 0)
            if sh <= 0:
                continue
            try:
                c, _, _ = _bar_on_date(k, trade_date)
                mv_other += c * sh
            except ValueError:
                mv_other += float(p.get("cost") or 0) * sh
        equity_after = cash - need + mv_other + notional
        if equity_after > 0 and notional / equity_after > 0.4:
            notes.append(f"买入后该股约占权益 {notional / equity_after * 100:.0f}%（>40%）")
            level = "warn"
    else:
        if held <= 0:
            return OrderCheck("reject", f"无持仓可卖: {vt}")
        if held < shares_i:
            return OrderCheck(
                "reject",
                f"持仓不足: 持有 {held}，卖出 {shares_i}",
            )
        fee = notional * COMMISSION_RATE_SELL
        proceeds = notional - fee
        notes.append(f"卖出 {shares_i}/{held} 股，约回笼 {proceeds:,.2f}（扣费后）")
        if shares_i == held:
            notes.append("将清仓该标的")
            level = "warn"
        if cost > 0:
            pnl_pct = (px / cost - 1.0) * 100
            notes.append(f"相对成本 {cost:.3f} 浮盈 {pnl_pct:+.2f}%")
            if pnl_pct < -7:
                notes.append("浮亏已超约 7%（对照止损阈值）")
                level = "warn"

    head = {"ok": "✅ 合理可下", "warn": "⚠️ 可下但请注意", "reject": "❌ 不合理/不可下"}[level]
    return OrderCheck(level, head + "：" + "；".join(notes))


def execute_order(
    account: str,
    side: str,
    code: str,
    shares: int,
    trade_date: str,
    price: float | None = None,
    *,
    update_data: bool = False,
) -> OrderResult:
    """
    模拟买入/卖出。
    side: buy / sell（或 买入 / 卖出）
    price: 默认用交易日收盘；可手工改价。
    """
    try:
        acc = ensure_sim_account_name(account)
    except ValueError as e:
        return OrderResult(False, str(e))

    if acc not in list_accounts():
        return OrderResult(False, f"模拟账户不存在: {acc}（请先创建）")

    side_key = side.strip().lower()
    if side_key in ("买入", "buy", "b"):
        side_key = "buy"
    elif side_key in ("卖出", "sell", "s"):
        side_key = "sell"
    else:
        return OrderResult(False, f"无效方向: {side}")

    try:
        shares = int(shares)
    except (TypeError, ValueError):
        return OrderResult(False, "数量须为正整数")
    if shares <= 0:
        return OrderResult(False, "数量须 > 0")

    trade_date = trade_date.strip()
    if len(trade_date) != 10:
        return OrderResult(False, "交易日格式须为 YYYY-MM-DD")

    try:
        vt = resolve_vt_symbol(code)
    except ValueError as e:
        return OrderResult(False, str(e))

    lot = a_share_lot_size(vt)
    if shares < lot or shares % lot != 0:
        return OrderResult(False, f"数量须为 {lot} 的整数倍（该标的整手={lot}）")

    if update_data:
        try:
            _maybe_update_daily(vt)
        except Exception as e:
            return OrderResult(False, f"更新日线失败: {e}")

    try:
        day_close, pre_close, _ = _bar_on_date(vt, trade_date)
    except ValueError as e:
        return OrderResult(False, str(e))

    px = float(price) if price is not None else day_close
    if px <= 0:
        return OrderResult(False, "价格须 > 0")

    name = _symbol_name(vt)
    warnings: list[str] = []

    if pre_close is None or pre_close <= 0:
        warnings.append("无昨收，未做涨跌停检查")
    else:
        bounds = a_share_limit_prices(vt, pre_close, name)
        if bounds is not None:
            down, up = bounds
            if px < down - 1e-9 or px > up + 1e-9:
                return OrderResult(
                    False,
                    f"成交价 {px:.2f} 超出涨跌停板 [{down:.2f}, {up:.2f}]",
                )
            if side_key == "buy" and is_a_share_limit_up(vt, day_close, pre_close, name):
                return OrderResult(False, f"涨停不可买（{trade_date} 收盘 {day_close:.2f}）")
            if side_key == "sell" and is_a_share_limit_down(vt, day_close, pre_close, name):
                return OrderResult(False, f"跌停不可卖（{trade_date} 收盘 {day_close:.2f}）")

    book = load_positions(account=acc)
    cash = float(book.get("cash") or 0)
    positions = dict(book.get("positions") or {})
    existing = positions.get(vt)

    if side_key == "buy":
        notional = px * shares
        fee = notional * COMMISSION_RATE_BUY
        need = notional + fee
        if cash + 1e-9 < need:
            return OrderResult(
                False,
                f"现金不足: 需要 {need:,.2f}（含费），可用 {cash:,.2f}",
            )
        old_shares = int((existing or {}).get("shares") or 0)
        old_cost = float((existing or {}).get("cost") or 0)
        new_shares = old_shares + shares
        if old_shares > 0:
            new_cost = (old_shares * old_cost + shares * px) / new_shares
        else:
            new_cost = px
        positions[vt] = merge_broker_position(
            existing,
            shares=new_shares,
            cost=new_cost,
            latest_price=px,
            last_buy=px,
            last_buy_date=trade_date,
        )
        cash -= need
        note = f"模拟买入 {vt.split('.', 1)[0]} {shares}@{px:.2f} 交易日={trade_date}"
    else:
        old_shares = int((existing or {}).get("shares") or 0)
        if old_shares < shares:
            return OrderResult(
                False,
                f"持仓不足: 持有 {old_shares}，卖出 {shares}",
            )
        notional = px * shares
        fee = notional * COMMISSION_RATE_SELL
        proceeds = notional - fee
        new_shares = old_shares - shares
        if new_shares <= 0:
            positions.pop(vt, None)
        else:
            positions[vt] = merge_broker_position(
                existing,
                shares=new_shares,
                cost=float((existing or {}).get("cost") or px),
                latest_price=px,
                last_buy=(existing or {}).get("last_buy"),
            )
        cash += proceeds
        note = f"模拟卖出 {vt.split('.', 1)[0]} {shares}@{px:.2f} 交易日={trade_date}"

    book = {
        "updated": trade_date,
        "cash": round(cash, 2),
        "positions": positions,
    }
    save_positions(
        book,
        note=note,
        source="paper_trade",
        account=acc,
        force_history=True,
    )
    msg = f"成功: {note}；现金={book['cash']:,.2f}"
    if warnings:
        msg += "；" + "；".join(warnings)
    return OrderResult(True, msg, book=book)
