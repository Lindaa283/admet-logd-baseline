"""Capture the environment a result was produced in.

A reported metric without the environment that produced it is not a
reproducible result. Every results JSON in this repo embeds this block.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "rdkit",
    "joblib",
    "matplotlib",
)


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not-installed"


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_commit() -> str | None:
    return _run(["git", "rev-parse", "HEAD"])


def git_dirty() -> bool | None:
    status = _run(["git", "status", "--porcelain"])
    if status is None:
        return None
    return bool(status.strip())


def gpu_info() -> str | None:
    return _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )


def snapshot() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "packages": {p: _version(p) for p in TRACKED_PACKAGES},
        "gpu_detected": gpu_info(),
        "gpu_used_for_training": False,
        "git_commit": git_commit(),
        "git_working_tree_dirty": git_dirty(),
        "monetary_cost_usd": 0.0,
        "cost_note": (
            "All computation local on the author's laptop. No cloud compute, no "
            "paid API, no licensed software. Public dataset downloaded once."
        ),
    }


def write_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
