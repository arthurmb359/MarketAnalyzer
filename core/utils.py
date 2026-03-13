from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def normalize_lines(lines: Iterable[str]) -> list[str]:
    return [str(line) for line in lines]


__all__ = ["ensure_path", "normalize_lines"]
