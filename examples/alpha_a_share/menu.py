"""
A 股 Alpha / 利弗莫尔 — 交互式菜单

用法：
  cd /Users/chendi/project/vnpy
  .venv/bin/python examples/alpha_a_share/menu.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"
EXAMPLE_JSON = PROJECT_DIR / "alpha_data" / "livermore_positions.example.json"

sys.path.insert(0, str(EXAMPLE_DIR))
from livermore_positions_store import (  # noqa: E402
    create_account,
    delete_account,
    list_accounts,
    normalize_account,
    rename_account,
    resolve_paths,
)


def clear_screen() -> None:
    print("\n" * 2)


def pause() -> None:
    input("\n按 Enter 继续...")


def read_choice(prompt: str, valid: set[str]) -> str:
    while True:
        choice = input(prompt).strip()
        if choice in valid:
            return choice
        print(f"  请输入 {', '.join(sorted(valid, key=lambda x: (x != '0', x)))}")


def read_line(prompt: str, *, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    line = input(f"{prompt}{suffix}: ").strip()
    return line if line else default


def read_yes_no(prompt: str, *, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{prompt} ({hint}): ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes", "是"):
            return True
        if ans in ("n", "no", "否"):
            return False
        print("  请输入 y 或 n")


def prompt_account() -> str | None:
    acc = read_line("账户名（直接回车=默认账本）")
    return acc or None


def run_script(script: str, *args: str) -> int:
    cmd = [str(PYTHON), str(EXAMPLE_DIR / script), *args]
    print("\n>>> " + " ".join(cmd) + "\n")
    try:
        return subprocess.call(cmd, cwd=PROJECT_DIR)
    except KeyboardInterrupt:
        print("\n（已中断）")
        return 130


def show_positions_help() -> None:
    account = prompt_account()
    pos_path, hist_path = resolve_paths(account)
    acct_label = account or "默认"

    print(f"\n{'=' * 60}")
    print(f"  更新持仓 — 账户: {acct_label}")
    print(f"{'=' * 60}")
    print(f"\n  持仓文件:\n    {pos_path}")
    print(f"\n  历史流水:\n    {hist_path}")
    print(f"\n  格式示例:\n    {EXAMPLE_JSON}")

    if EXAMPLE_JSON.exists():
        print("\n  --- JSON 结构 ---")
        print(EXAMPLE_JSON.read_text(encoding="utf-8").rstrip())

    print("\n  --- 字段说明 ---")
    print("  updated   更新日期 (YYYY-MM-DD)")
    print("  cash      可用资金")
    print("  positions 持仓字典，键为 vt_symbol（如 600487.SSE）")
    print("    shares    持股数")
    print("    cost      加权成本")
    print("    high      持仓以来最高价（只升不降）")
    print("    last_buy  上次买入价（加仓判断）")
    print("    stage     金字塔档位：1=首仓 2=加满")
    print("    halved    是否已移动止盈减半")
    print("\n  改完后回到菜单 → 每日实盘 → 保存持仓快照，记入历史。")
    print(f"{'=' * 60}")


def action_predict_daily() -> None:
    account = read_line("账户名（直接回车=全部账户）")
    args: list[str] = []
    if account:
        args.extend(["--account", account])
    code = run_script("predict_daily.py", *args)
    if code != 0:
        print(f"\n  退出码: {code}")


def action_snapshot() -> None:
    account = prompt_account()
    note = read_line("备注说明", default="收盘更新")
    args: list[str] = []
    if account:
        args.extend(["--account", account])
    if note:
        args.extend(["--note", note])
    code = run_script("snapshot_positions.py", *args)
    if code != 0:
        print(f"\n  退出码: {code}")


def action_snapshot_list() -> None:
    account = prompt_account()
    limit = read_line("最近 N 条", default="20")
    args: list[str] = ["--list"]
    if limit.isdigit():
        args.append(limit)
    if account:
        args.extend(["--account", account])
    run_script("snapshot_positions.py", *args)


def action_create_account() -> None:
    from datetime import date

    name = read_line("新账户名称")
    if not name:
        print("  已取消")
        return
    try:
        normalize_account(name)
    except ValueError as e:
        print(f"  无效账户名: {e}")
        return

    overwrite = False
    if name in list_accounts():
        if not read_yes_no(f"账户「{name}」已存在，是否重置为空仓?", default=False):
            print("  已取消")
            return
        overwrite = True

    cash_str = read_line("初始可用资金", default="0")
    try:
        cash = float(cash_str) if cash_str else 0.0
    except ValueError:
        print("  无效金额，已取消")
        return

    updated = read_line("更新日期 (YYYY-MM-DD)", default=date.today().isoformat())
    try:
        pos_path, hist_path = create_account(
            name, cash=cash, updated=updated, overwrite=overwrite
        )
    except (ValueError, FileExistsError) as e:
        print(f"  错误: {e}")
        return

    print(f"\n  已创建账户「{name}」")
    print(f"  持仓文件: {pos_path}")
    print(f"  历史流水: {hist_path}")
    print(f"  初始资金: {cash:,.2f}")
    print("\n  下一步：菜单 → 更新持仓，或直接编辑 JSON → 保存持仓快照")


def action_delete_account() -> None:
    rows = list_accounts()
    if not rows:
        print("  （暂无命名账户可删）")
        return

    print("  已有命名账户:")
    for i, name in enumerate(rows, 1):
        print(f"    {i}. {name}")
    print("  （默认账本不在此列，需手动删 alpha_data/livermore_positions*.json）")

    name = read_line("要删除的账户名")
    if not name:
        print("  已取消")
        return
    try:
        acc = normalize_account(name)
    except ValueError as e:
        print(f"  无效账户名: {e}")
        return
    if acc is None or acc not in rows:
        print(f"  未找到账户「{name}」")
        return

    pos_path, hist_path = resolve_paths(acc)
    print(f"\n  将永久删除目录及文件:")
    print(f"    {pos_path.parent}")
    print(f"    {pos_path.name}")
    print(f"    {hist_path.name}")
    if not read_yes_no(f"确认删除账户「{acc}」?", default=False):
        print("  已取消")
        return
    confirm = read_line(f"请再次输入账户名「{acc}」以确认")
    if confirm != acc:
        print("  名称不一致，已取消")
        return

    try:
        base = delete_account(acc)
    except (ValueError, FileNotFoundError) as e:
        print(f"  删除失败: {e}")
        return
    print(f"\n  已删除账户「{acc}」→ {base}")


def action_rename_account() -> None:
    rows = list_accounts()
    if not rows:
        print("  （暂无命名账户可改名）")
        return

    print("  已有命名账户:")
    for i, name in enumerate(rows, 1):
        print(f"    {i}. {name}")
    print("  （默认账本不支持改名）")

    old = read_line("当前账户名")
    if not old:
        print("  已取消")
        return
    try:
        old_acc = normalize_account(old)
    except ValueError as e:
        print(f"  无效账户名: {e}")
        return
    if old_acc is None or old_acc not in rows:
        print(f"  未找到账户「{old}」")
        return

    new = read_line("新账户名")
    if not new:
        print("  已取消")
        return
    try:
        new_acc = normalize_account(new)
    except ValueError as e:
        print(f"  无效账户名: {e}")
        return
    if new_acc is None:
        print("  新账户名不能为空")
        return
    if new_acc == old_acc:
        print("  新旧名称相同，已取消")
        return
    if new_acc in rows:
        print(f"  账户「{new_acc}」已存在")
        return

    old_path, _ = resolve_paths(old_acc)
    new_path, _ = resolve_paths(new_acc)
    print(f"\n  {old_path.parent}")
    print(f"  → {new_path.parent}")
    if not read_yes_no(f"确认将「{old_acc}」改名为「{new_acc}」?", default=False):
        print("  已取消")
        return

    try:
        old_base, new_base = rename_account(old_acc, new_acc)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as e:
        print(f"  改名失败: {e}")
        return
    print(f"\n  已改名「{old_acc}」→「{new_acc}」")
    print(f"  {old_base} → {new_base}")


def action_list_accounts() -> None:
    run_script("snapshot_positions.py", "--list-accounts")
    rows = list_accounts()
    if not rows:
        print("\n  （暂无命名账户；可用菜单「创建账户」或 predict_daily --account 创建）")


def action_intraday_add() -> None:
    args: list[str] = []
    if read_yes_no("扫描前先更新日线?", default=False):
        args.append("--update")
    account = read_line("账户名（直接回车=全部账户）")
    if account:
        args.extend(["--account", account])
    while True:
        px = read_line("盘中价 代码:价格（直接回车结束）")
        if not px:
            break
        args.extend(["--price", px])
    run_script("intraday_add.py", *args)


def action_check_trend() -> None:
    symbol = read_line("股票代码（如 600487 或 01024）")
    if not symbol:
        print("  已取消")
        return
    account = prompt_account()
    args = [symbol]
    if read_yes_no("检查前先更新日线?", default=True):
        args.append("--update")
    price = read_line("盘中临时价（直接回车=用日线收盘）")
    if price:
        args.extend(["--price", price])
    if account:
        args.extend(["--account", account])
    run_script("check_trend.py", *args)


def action_daily_pnl() -> None:
    args: list[str] = []
    if read_yes_no("先更新持仓标的日线?", default=False):
        args.append("--update")
    account = read_line("账户名（直接回车=全部账户）")
    if account:
        args.extend(["--account", account])
    run_script("daily_pnl.py", *args)


MENU_DAILY = [
    ("1", "每日预测", action_predict_daily),
    ("2", "盘中开仓/加仓扫描", action_intraday_add),
    ("3", "当日盈亏", action_daily_pnl),
    ("4", "更新持仓（JSON 路径与格式说明）", show_positions_help),
    ("5", "保存持仓快照", action_snapshot),
    ("6", "单票趋势检查", action_check_trend),
    ("7", "查看历史流水", action_snapshot_list),
    ("8", "创建账户", action_create_account),
    ("9", "修改账户名", action_rename_account),
    ("10", "删除账户", action_delete_account),
    ("11", "列出所有账户", action_list_accounts),
]

MENU_TRAIN = [
    ("1", "训练 LightGBM (1/3/5 日)", lambda: run_script("run_ml.py")),
    ("2", "评估历史预测", lambda: run_script("evaluate_predictions.py")),
    ("3", "优化选股参数", lambda: run_script("optimize_prediction.py")),
]

MENU_BACKTEST = [
    ("1", "ML 等权 Top 1/3（基准）", lambda: run_script("backtest_1y.py")),
    ("2", "ML + 固定止损止盈", lambda: run_script("backtest_1y_sltp.py")),
    ("3", "ML + 移动止盈 + 组合回撤熔断", lambda: run_script("backtest_1y_circuit.py")),
    ("4", "利弗莫尔（4 万 / Top5 / 整手）", lambda: run_script("backtest_1y_livermore.py")),
]

MENU_OTHER = [
    ("1", "20 日动量因子（非 ML）", lambda: run_script("run.py")),
    ("2", "Alpha158 多因子合成", lambda: run_script("run_multifactor.py")),
    ("3", "早期盈利区回撤（实验）", lambda: run_script("backtest_early_zone_pullback.py")),
    ("4", "早期区利润吐回α（实验）", lambda: run_script("backtest_early_zone_giveback.py")),
    ("5", "早期+移动止盈吐回（实验）", lambda: run_script("backtest_trail_giveback.py")),
    ("6", "早期盈利区扩展（实验）", lambda: run_script("backtest_early_zone_extended.py")),
    ("7", "止盈策略对比（实验）", lambda: run_script("backtest_profit_lock_compare.py")),
]

MENU_ROOT = [
    ("1", "每日实盘", MENU_DAILY),
    ("2", "模型训练与优化", MENU_TRAIN),
    ("3", "一年回测", MENU_BACKTEST),
    ("4", "其他可选脚本", MENU_OTHER),
]


def run_menu(title: str, items: list, *, is_root: bool = False) -> None:
    actions = {key: (label, target) for key, label, target in items}
    valid = set(actions) | {"0"}

    while True:
        clear_screen()
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)
        for key, label, _ in items:
            print(f"  {key}. {label}")
        print(f"  0. {'退出' if is_root else '返回上级'}")
        print()

        choice = read_choice(f"请选择 [0-{max(valid, key=lambda x: int(x) if x.isdigit() else 0)}]: ", valid)
        if choice == "0":
            return

        _, target = actions[choice]
        if isinstance(target, list):
            run_menu(actions[choice][0], target)
        else:
            clear_screen()
            print(f"--- {actions[choice][0]} ---\n")
            try:
                target()
            except KeyboardInterrupt:
                print("\n（已中断）")
            pause()


def main() -> None:
    if not PYTHON.exists():
        print(f"未找到 Python: {PYTHON}")
        print("请先在项目根目录创建 .venv")
        raise SystemExit(1)

    print("\n  A 股 Alpha / 利弗莫尔 — 交互式菜单")
    print(f"  项目目录: {PROJECT_DIR}\n")
    try:
        run_menu("A 股 Alpha / 利弗莫尔", MENU_ROOT, is_root=True)
    except KeyboardInterrupt:
        print("\n再见。")
    print()


if __name__ == "__main__":
    main()
