"""
回测回放落盘 / 读取

目录：alpha_data/backtest_replay/<run_id>/
  meta.json
  trades.parquet
  daily.parquet
  positions_daily.parquet
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from vnpy.trader.constant import Direction


def replay_root(alpha_lab_path: str | Path) -> Path:
    return Path(alpha_lab_path) / "backtest_replay"


def list_replay_runs(alpha_lab_path: str | Path) -> list[tuple[str, str]]:
    """返回 [(label, run_id), ...]，新的在前。"""
    root = replay_root(alpha_lab_path)
    if not root.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for p in sorted(root.iterdir(), reverse=True):
        meta_path = p / "meta.json"
        if not p.is_dir() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        label = (
            f"{meta.get('saved_at', p.name)[:19]}  "
            f"{meta.get('strategy', '?')}  "
            f"src={meta.get('signal_source', '?')}  "
            f"X={meta.get('add_spacing_pct', '?')}"
        )
        out.append((label, p.name))
    return out


def load_replay(alpha_lab_path: str | Path, run_id: str) -> dict:
    run_dir = replay_root(alpha_lab_path) / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"回放目录不存在: {run_dir}")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    trades = pl.read_parquet(run_dir / "trades.parquet") if (run_dir / "trades.parquet").is_file() else pl.DataFrame()
    daily = pl.read_parquet(run_dir / "daily.parquet") if (run_dir / "daily.parquet").is_file() else pl.DataFrame()
    positions = (
        pl.read_parquet(run_dir / "positions_daily.parquet")
        if (run_dir / "positions_daily.parquet").is_file()
        else pl.DataFrame()
    )
    return {"meta": meta, "trades": trades, "daily": daily, "positions": positions, "run_dir": str(run_dir)}


def save_engine_replay(
    engine,
    alpha_lab_path: str | Path,
    *,
    strategy_name: str,
    signal_source: str,
    add_spacing_pct: float | None,
    stats: dict | None = None,
    extra_meta: dict | None = None,
) -> Path:
    """从已 calculate_result + calculate_statistics 的引擎落盘。"""
    root = replay_root(alpha_lab_path)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = (signal_source or "vnpy").strip().lower()
    x_tag = f"_x{int(add_spacing_pct * 100)}" if add_spacing_pct is not None else ""
    run_id = f"{strategy_name}_{src}{x_tag}_{ts}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # trades
    trade_rows = []
    for t in engine.trades.values():
        if not t.datetime:
            continue
        side = "买入" if t.direction == Direction.LONG else "卖出"
        trade_rows.append(
            {
                "datetime": t.datetime,
                "date": t.datetime.date(),
                "vt_symbol": t.vt_symbol,
                "side": side,
                "direction": t.direction.value if t.direction else "",
                "offset": t.offset.value if t.offset else "",
                "price": float(t.price),
                "volume": float(t.volume),
                "turnover": float(t.price) * float(t.volume),
                "tradeid": t.tradeid,
            }
        )
    trades_df = pl.DataFrame(trade_rows) if trade_rows else pl.DataFrame(
        schema={
            "datetime": pl.Datetime("us"),
            "date": pl.Date,
            "vt_symbol": pl.Utf8,
            "side": pl.Utf8,
            "direction": pl.Utf8,
            "offset": pl.Utf8,
            "price": pl.Float64,
            "volume": pl.Float64,
            "turnover": pl.Float64,
            "tradeid": pl.Utf8,
        }
    )
    trades_df.write_parquet(run_dir / "trades.parquet")

    # daily equity
    daily_df = getattr(engine, "daily_df", None)
    if daily_df is not None and not daily_df.is_empty():
        daily_df.write_parquet(run_dir / "daily.parquet")
    else:
        pl.DataFrame().write_parquet(run_dir / "daily.parquet")

    # positions snapshot each day
    pos_rows = []
    for d, dr in sorted(engine.daily_results.items(), key=lambda x: x[0]):
        closes = getattr(dr, "close_prices", {}) or {}
        poses = getattr(dr, "end_poses", {}) or {}
        for vt, shares in poses.items():
            sh = float(shares or 0)
            if abs(sh) < 1e-9:
                continue
            px = float(closes.get(vt) or 0)
            pos_rows.append(
                {
                    "date": d,
                    "vt_symbol": vt,
                    "shares": sh,
                    "close": px,
                    "market_value": sh * px,
                }
            )
    pos_df = pl.DataFrame(pos_rows) if pos_rows else pl.DataFrame(
        schema={
            "date": pl.Date,
            "vt_symbol": pl.Utf8,
            "shares": pl.Float64,
            "close": pl.Float64,
            "market_value": pl.Float64,
        }
    )
    pos_df.write_parquet(run_dir / "positions_daily.parquet")

    meta = {
        "run_id": run_id,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": strategy_name,
        "signal_source": src,
        "add_spacing_pct": add_spacing_pct,
        "capital": getattr(engine, "capital", None),
        "start": getattr(engine, "start", None).isoformat(timespec="seconds") if getattr(engine, "start", None) else None,
        "end": getattr(engine, "end", None).isoformat(timespec="seconds") if getattr(engine, "end", None) else None,
        "trade_count": len(trade_rows),
        "stats": stats or {},
    }
    if extra_meta:
        meta.update(extra_meta)
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # index append
    index_path = root / "index.jsonl"
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "saved_at": meta["saved_at"], "strategy": strategy_name, "signal_source": src}, ensure_ascii=False) + "\n")

    return run_dir
