"""
持仓历史：保存快照 / 查看流水（复盘用）

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/snapshot_positions.py --note "收盘更新"
  .venv/bin/python examples/alpha_a_share/snapshot_positions.py --list
  .venv/bin/python examples/alpha_a_share/snapshot_positions.py --list 10
  .venv/bin/python examples/alpha_a_share/snapshot_positions.py --account 本人 --note "收盘更新"
  .venv/bin/python examples/alpha_a_share/snapshot_positions.py --list-accounts
  .venv/bin/python examples/alpha_a_share/snapshot_positions.py --create-account 本人 --cash 100000
  .venv/bin/python examples/alpha_a_share/snapshot_positions.py --delete-account 本人
  .venv/bin/python examples/alpha_a_share/snapshot_positions.py --rename-account 本人 --to 家人
"""

from __future__ import annotations

import argparse

from livermore_positions_store import (
    SIM_PREFIX,
    create_account,
    delete_account,
    is_sim_account,
    list_history,
    list_live_accounts,
    load_positions,
    rename_account,
    resolve_paths,
    save_positions,
    snapshot_current,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="利弗莫尔持仓历史快照")
    parser.add_argument("--account", default=None, help="持仓账户名（独立账本）")
    parser.add_argument("--note", default="", help="本次变更说明")
    parser.add_argument("--list", type=int, nargs="?", const=20, help="列出最近 N 条历史")
    parser.add_argument("--list-accounts", action="store_true", help="列出已创建的命名账户")
    parser.add_argument("--create-account", metavar="NAME", help="创建命名账户（初始空仓）")
    parser.add_argument("--delete-account", metavar="NAME", help="删除命名账户目录（不可恢复）")
    parser.add_argument("--rename-account", metavar="OLD", help="重命名命名账户（配合 --to）")
    parser.add_argument("--to", metavar="NEW", help="新账户名（配合 --rename-account）")
    parser.add_argument("--cash", type=float, default=0.0, help="创建账户时的初始可用资金")
    parser.add_argument("--updated", default="", help="创建账户时的 updated 日期 (YYYY-MM-DD)")
    parser.add_argument("--overwrite", action="store_true", help="账户已存在时重置为空仓")
    parser.add_argument("--force", action="store_true", help="即使与上条相同也写入历史")
    parser.add_argument("--yes", action="store_true", help="删除账户时跳过确认")
    args = parser.parse_args()

    if args.rename_account:
        if not args.to:
            print("改名需同时指定 --to <新名>")
            raise SystemExit(1)
        if is_sim_account(args.rename_account) or is_sim_account(args.to):
            print(f"「{SIM_PREFIX}」练习账户请在模拟操盘中管理，不在此改名")
            raise SystemExit(1)
        try:
            old_base, new_base = rename_account(args.rename_account, args.to)
        except (ValueError, FileNotFoundError, FileExistsError, OSError) as e:
            print(f"改名失败: {e}")
            raise SystemExit(1) from e
        print(f"已改名「{args.rename_account}」→「{args.to}」")
        print(f"  {old_base} → {new_base}")
        return

    if args.delete_account:
        name = args.delete_account
        if is_sim_account(name):
            print(f"「{SIM_PREFIX}」练习账户请在模拟操盘中管理，不在此删除")
            raise SystemExit(1)
        if name not in list_live_accounts():
            print(f"未找到账户「{name}」")
            raise SystemExit(1)
        pos_path, _ = resolve_paths(name)
        if not args.yes:
            print(f"将删除: {pos_path.parent}")
            print("请加 --yes 确认删除（不可恢复）")
            raise SystemExit(1)
        try:
            base = delete_account(name)
        except (ValueError, FileNotFoundError) as e:
            print(f"删除失败: {e}")
            raise SystemExit(1) from e
        print(f"已删除账户「{name}」→ {base}")
        return

    if args.create_account:
        if is_sim_account(args.create_account):
            print(f"「{SIM_PREFIX}」练习账户请在「模拟操盘 Web」中创建")
            raise SystemExit(1)
        try:
            pos_path, hist_path = create_account(
                args.create_account,
                cash=args.cash,
                updated=args.updated,
                overwrite=args.overwrite,
            )
        except (ValueError, FileExistsError) as e:
            print(f"创建失败: {e}")
            raise SystemExit(1) from e
        print(f"已创建账户「{args.create_account}」")
        print(f"  持仓: {pos_path}")
        print(f"  历史: {hist_path}")
        print(f"  初始资金: {args.cash:,.2f}")
        return

    if args.list_accounts:
        rows = list_live_accounts()
        print("实盘命名账户（alpha_data/accounts/，不含「模拟_」练习账户）：")
        if not rows:
            print("  （暂无；用 --account <名称> 创建；练习账户见模拟操盘 Web）")
        else:
            for name in rows:
                pos_path, _ = resolve_paths(name)
                print(f"  {name}  →  {pos_path}")
        return

    if is_sim_account(args.account):
        print(f"「{SIM_PREFIX}」练习账户请在模拟操盘中操作，不在此快照")
        raise SystemExit(1)

    positions_path, history_path = resolve_paths(args.account)

    if args.list is not None:
        rows = list_history(limit=args.list, path=history_path)
        acct_label = args.account or "默认"
        print(f"账户: {acct_label}")
        print(f"历史文件: {history_path}")
        if not rows:
            print("  （暂无记录）")
            return
        for i, row in enumerate(rows, 1):
            n_pos = len(row.get("positions") or {})
            print(
                f"  {i:>2}. {row.get('saved_at', '')}  "
                f"updated={row.get('updated', '')}  "
                f"cash={row.get('cash', 0)}  "
                f"持仓={n_pos}  "
                f"[{row.get('source', '')}] {row.get('note', '')}"
            )
        return

    book = load_positions(path=positions_path)
    note = args.note or "手动快照"
    written, msg = snapshot_current(
        book,
        note=note,
        source="manual",
        force=args.force,
        history_path=history_path,
    )
    print(f"账户: {args.account or '默认'}")
    print(f"持仓文件: {positions_path}")
    print(msg if written else f"未写入: {msg}")
    if written:
        print(f"历史文件: {history_path}")


if __name__ == "__main__":
    main()
