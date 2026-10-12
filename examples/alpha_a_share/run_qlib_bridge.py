"""
调用独立 Qlib 仓脚本（subprocess，不 import qlib）。

用法（在 vnpy 根目录）：
  .venv/bin/python examples/alpha_a_share/run_qlib_bridge.py convert
  .venv/bin/python examples/alpha_a_share/run_qlib_bridge.py train
  .venv/bin/python examples/alpha_a_share/run_qlib_bridge.py predict [--asof YYYY-MM-DD]
  .venv/bin/python examples/alpha_a_share/run_qlib_bridge.py all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_QLIB_ROOT = Path("/Users/chendi/project/qlib")
BRIDGE = "scripts/a_share_bridge"


def _qlib_python(qlib_root: Path) -> Path:
    py = qlib_root / ".venv" / "bin" / "python"
    if not py.is_file():
        raise SystemExit(
            f"未找到 Qlib venv: {py}\n"
            f"请先: cd {qlib_root} && python3 -m venv .venv && .venv/bin/pip install -e ."
        )
    return py


def _run(qlib_root: Path, script: str, extra: list[str] | None = None) -> None:
    py = _qlib_python(qlib_root)
    cmd = [
        str(py),
        str(qlib_root / BRIDGE / script),
        "--vnpy-root",
        str(PROJECT_DIR),
    ]
    if extra:
        cmd.extend(extra)
    print(">>", " ".join(cmd))
    env = os.environ.copy()
    # 保证 bridge 内 from common import 可用
    env["PYTHONPATH"] = str(qlib_root / BRIDGE) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(qlib_root), env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="subprocess 调用 Qlib a_share_bridge")
    ap.add_argument(
        "step",
        choices=["instruments", "convert", "train", "predict", "all"],
        help="执行步骤",
    )
    ap.add_argument("--qlib-root", default=str(DEFAULT_QLIB_ROOT))
    ap.add_argument("--asof", default=None, help="仅 predict：信号日 YYYY-MM-DD")
    ap.add_argument(
        "--only-asof",
        action="store_true",
        help="predict 时只写当天；默认回填 pred 段全部交易日",
    )
    args = ap.parse_args()

    qlib_root = Path(args.qlib_root).expanduser().resolve()
    if not qlib_root.is_dir():
        raise SystemExit(f"qlib root 不存在: {qlib_root}")

    if args.step in ("instruments", "all"):
        _run(qlib_root, "export_instruments.py")
    if args.step in ("convert", "all"):
        _run(qlib_root, "convert_daily.py")
    if args.step in ("train", "all"):
        # train 不需要 --vnpy-root，但带上也无害；train 脚本若未知参数会失败
        py = _qlib_python(qlib_root)
        cmd = [str(py), str(qlib_root / BRIDGE / "train_lgb_alpha158.py")]
        print(">>", " ".join(cmd))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(qlib_root / BRIDGE) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(cmd, check=True, cwd=str(qlib_root), env=env)
    if args.step in ("predict", "all"):
        extra: list[str] = []
        if args.asof:
            extra.extend(["--asof", args.asof])
        if args.only_asof:
            extra.append("--only-asof")
        _run(qlib_root, "predict_export.py", extra=extra or None)

    print("✓ done:", args.step)


if __name__ == "__main__":
    main()
