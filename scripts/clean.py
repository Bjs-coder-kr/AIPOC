"""Cleanup caches from the repository."""

from __future__ import annotations

import shutil
from pathlib import Path


TARGET_DIRS = {"__pycache__", ".pytest_cache"}


def remove_dirs(root: Path) -> list[Path]:
    removed: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name in TARGET_DIRS:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    removed = remove_dirs(repo_root)
    print(f"Removed {len(removed)} cache directories.")
    for path in removed:
        print(f"- {path}")


if __name__ == "__main__":
    main()
