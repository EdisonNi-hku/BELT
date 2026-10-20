from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True,
                                check=False, cwd=REPO_ROOT, timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _record(path: Path, digest: str) -> dict:
    commit_out = _git("rev-parse", "HEAD")
    commit = commit_out.strip() if commit_out is not None else None

    status_out = _git("status", "--porcelain")
    dirty = bool(status_out.strip()) if status_out is not None else None

    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "commit": commit,
        "dirty": dirty,
    }


def source_record(path: str | Path) -> dict:
    path = Path(path)
    return _record(path, hashlib.sha256(path.read_bytes()).hexdigest())


def parse_and_record(path: str | Path,
                     parser: Callable[[Iterable[str]], Any]) -> tuple[Any, dict]:
    from rl_controller.iclr.banked_baselines import _HashingJsonlLines

    path = Path(path)
    with path.open("rb") as handle:
        lines = _HashingJsonlLines(handle)
        parsed = parser(lines)
        digest = lines.hexdigest()
    return parsed, _record(path, digest)
