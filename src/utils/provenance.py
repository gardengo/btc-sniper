"""Run provenance: what code and which libraries produced an artifact.

MODEL_SPEC.md section 10 and VALIDATION_SPEC.md section 11 both require every
trained model and every evaluation to record the code version, the library
versions and the seed. Without that a "reproducible" model is only reproducible
until someone upgrades pandas.

Everything here degrades gracefully. A missing git binary or a shallow checkout
makes the commit unknown, which is worth recording as ``"unknown"`` rather than
crashing a training run.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from functools import lru_cache
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

UNKNOWN: str = "unknown"

# Libraries whose version can change a fitted model or a metric value.
TRACKED_PACKAGES: tuple[str, ...] = ("numpy", "pandas", "scipy", "lightgbm")


@lru_cache(maxsize=1)
def code_commit(repo_root: Path | None = None) -> str:
    """Short git commit of the working tree, with ``-dirty`` when uncommitted.

    A dirty marker matters: a model trained from uncommitted code cannot be
    reproduced from the recorded commit, and silently recording the parent
    commit would be a lie.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("could not read the git commit: %s", exc)
        return UNKNOWN
    return f"{commit}-dirty" if status else commit


@lru_cache(maxsize=1)
def library_versions() -> dict[str, str]:
    """Versions of the libraries that can change a result."""
    versions = {"python": platform.python_version()}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = UNKNOWN
    return versions


def describe_environment() -> str:
    """One-line provenance summary for logs."""
    versions = library_versions()
    listed = " ".join(f"{name}={value}" for name, value in versions.items())
    return f"commit={code_commit()} {listed}"
