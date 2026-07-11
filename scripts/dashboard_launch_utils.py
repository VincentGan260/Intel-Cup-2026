"""Small helpers for launching the RiderGuardian Dashboard modes."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_dashboard(args: Iterable[str], *, env_updates: dict[str, str] | None = None) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "run_dashboard.py"), *args]
    env = os.environ.copy()
    if env_updates:
        env.update({k: v for k, v in env_updates.items() if v})
    print("$ " + " ".join(str(x) for x in cmd))
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\n[launcher] stopping dashboard...")
        proc.send_signal(signal.SIGINT)
        try:
            return proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            return proc.wait(timeout=10)


def newest_session(record_root: Path, started_at: float) -> Path | None:
    if not record_root.exists():
        return None
    candidates = [
        p for p in record_root.iterdir()
        if p.is_dir() and p.name.endswith("_dashboard") and p.stat().st_mtime >= started_at - 2
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def copy_session(session_dir: Path, export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / session_dir.name
    if target.exists():
        suffix = time.strftime("%H%M%S")
        target = export_dir / f"{session_dir.name}_{suffix}"
    shutil.copytree(session_dir, target)
    return target


def zip_session(session_dir: Path, export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    archive_base = export_dir / session_dir.name
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=session_dir.parent, base_dir=session_dir.name)
    return Path(archive)
