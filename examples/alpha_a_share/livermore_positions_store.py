"""利弗莫尔持仓：当前快照 + 历史流水（复盘用）。"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
ALPHA_LAB_PATH = PROJECT_DIR / "alpha_data"
ACCOUNTS_DIR = ALPHA_LAB_PATH / "accounts"
POSITIONS_PATH = ALPHA_LAB_PATH / "livermore_positions.json"
HISTORY_PATH = ALPHA_LAB_PATH / "livermore_positions_history.jsonl"


def normalize_account(account: str | None) -> str | None:
    """账户名去空白；空串视为默认账本。"""
    if account is None:
        return None
    acc = account.strip()
    if not acc:
        return None
    if acc in (".", "..") or "/" in acc or "\\" in acc:
        raise ValueError(f"invalid account name: {account!r}")
    return acc


def resolve_paths(account: str | None = None) -> tuple[Path, Path]:
    """
    返回 (持仓快照, 历史流水) 路径。
    未指定 account 时用默认单文件；指定后各账户独立目录：
    alpha_data/accounts/<account>/
    """
    acc = normalize_account(account)
    if acc is None:
        return POSITIONS_PATH, HISTORY_PATH
    base = ACCOUNTS_DIR / acc
    return (
        base / "livermore_positions.json",
        base / "livermore_positions_history.jsonl",
    )


def history_path_for(positions_path: Path | str) -> Path:
    """与给定持仓文件同目录的历史流水路径。"""
    p = Path(positions_path)
    if p == POSITIONS_PATH:
        return HISTORY_PATH
    return p.parent / "livermore_positions_history.jsonl"


def list_accounts() -> list[str]:
    """已创建的命名账户（不含默认账本）。"""
    if not ACCOUNTS_DIR.exists():
        return []
    return sorted(d.name for d in ACCOUNTS_DIR.iterdir() if d.is_dir())


def all_prediction_accounts() -> list[str | None]:
    """每日预测管仓账户列表：有默认文件或无命名账户时含默认账本，再加全部命名账户。"""
    named = list_accounts()
    accounts: list[str | None] = []
    if POSITIONS_PATH.exists() or not named:
        accounts.append(None)
    accounts.extend(named)
    return accounts


def create_account(
    account: str,
    *,
    cash: float = 0.0,
    updated: str = "",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """创建命名账户及初始空仓快照；已存在则报错（除非 overwrite）。"""
    acc = normalize_account(account)
    if acc is None:
        raise ValueError("account name required")
    pos_path, hist_path = resolve_paths(acc)
    if pos_path.exists() and not overwrite:
        raise FileExistsError(acc)
    data = normalize_positions({"updated": updated, "cash": cash, "positions": {}})
    save_positions(
        data,
        note="创建账户",
        source="create",
        path=pos_path,
        record_history=True,
        force_history=True,
    )
    return pos_path, hist_path


def delete_account(account: str) -> Path:
    """删除命名账户目录（持仓 + 历史）；不可用于默认账本。"""
    import shutil

    acc = normalize_account(account)
    if acc is None:
        raise ValueError("account name required（默认账本请手动删 livermore_positions*.json）")
    base = ACCOUNTS_DIR / acc
    if not base.is_dir():
        raise FileNotFoundError(acc)
    shutil.rmtree(base)
    return base


def rename_account(old_name: str, new_name: str) -> tuple[Path, Path]:
    """重命名命名账户目录；返回 (旧路径, 新路径)。不可用于默认账本。"""
    old = normalize_account(old_name)
    new = normalize_account(new_name)
    if old is None or new is None:
        raise ValueError("新旧账户名均不能为空（默认账本不支持改名）")
    if old == new:
        raise ValueError("新旧账户名相同")
    old_base = ACCOUNTS_DIR / old
    new_base = ACCOUNTS_DIR / new
    if not old_base.is_dir():
        raise FileNotFoundError(old)
    if new_base.exists():
        raise FileExistsError(new)
    ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
    old_base.rename(new_base)
    return old_base, new_base


def normalize_positions(data: dict) -> dict:
    out = deepcopy(data)
    out.setdefault("updated", "")
    out.setdefault("cash", 0.0)
    out.setdefault("positions", {})
    return out


def load_positions(
    path: Path | str | None = None,
    *,
    account: str | None = None,
) -> dict:
    """读取当前持仓快照；不存在则返回空仓模板。"""
    if path is None:
        path, _ = resolve_paths(account)
    default: dict = {"updated": "", "cash": 0.0, "positions": {}}
    p = Path(path)
    if not p.exists():
        return default
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return default
    return normalize_positions(data)


def _fingerprint(data: dict) -> str:
    payload = {
        "updated": data.get("updated", ""),
        "cash": round(float(data.get("cash", 0) or 0), 2),
        "positions": data.get("positions", {}),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def list_history(limit: int | None = None, path: Path | str = HISTORY_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows


def append_history(
    data: dict,
    *,
    note: str = "",
    source: str = "manual",
    path: Path | str = HISTORY_PATH,
    force: bool = False,
) -> bool:
    """追加一条历史记录；与上一条内容相同则跳过（除非 force）。"""
    data = normalize_positions(data)
    record = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated": data.get("updated", ""),
        "source": source,
        "note": note,
        "cash": data.get("cash", 0.0),
        "positions": data.get("positions", {}),
    }
    history = list_history(path=path)
    if history and not force:
        last = history[-1]
        last_fp = _fingerprint(
            {"updated": last.get("updated"), "cash": last.get("cash"), "positions": last.get("positions")}
        )
        if _fingerprint(data) == last_fp:
            return False

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def merge_broker_position(
    existing: dict | None,
    *,
    shares: int,
    cost: float,
    latest_price: float | None = None,
    last_buy: float | None = None,
) -> dict:
    """
    用券商截图字段合并持仓；high 只升不降（保留 predict_daily 建议或手动改过的更高值）。
    """
    old = existing or {}
    old_high = float(old.get("high", 0) or 0)
    high_candidates = [old_high]
    if latest_price and latest_price > 0:
        high_candidates.append(float(latest_price))
    high = max(high_candidates)

    out = {
        "shares": int(shares),
        "cost": float(cost),
        "high": high,
        "last_buy": float(last_buy if last_buy is not None else old.get("last_buy", cost) or cost),
        "stage": int(old.get("stage", 1) or 1),
        "halved": bool(old.get("halved", False)),
    }
    if int(shares) <= 0:
        return out
    return out


def save_positions(
    data: dict,
    *,
    note: str = "",
    source: str = "manual",
    record_history: bool = True,
    force_history: bool = False,
    path: Path | str | None = None,
    account: str | None = None,
) -> None:
    """写入当前快照，并可选追加历史。"""
    if path is None:
        path, hist_path = resolve_paths(account)
    else:
        hist_path = history_path_for(path)
    data = normalize_positions(data)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if record_history:
        append_history(
            data,
            note=note,
            source=source,
            force=force_history,
            path=hist_path,
        )


def snapshot_current(
    *,
    note: str = "",
    source: str = "manual",
    account: str | None = None,
) -> bool:
    """把当前持仓快照记入历史（不改文件内容）。"""
    pos_path, hist_path = resolve_paths(account)
    return append_history(
        load_positions(path=pos_path),
        note=note,
        source=source,
        path=hist_path,
    )


def history_on_or_before(date: str, path: Path | str = HISTORY_PATH) -> dict | None:
    """取某日及之前最近一条历史（date 格式 YYYY-MM-DD）。"""
    rows = list_history(path=path)
    picked: dict | None = None
    for row in rows:
        key = row.get("updated") or row.get("saved_at", "")[:10]
        if key <= date:
            picked = row
    return picked
