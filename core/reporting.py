from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime

from core.utils import normalize_lines


@dataclass(frozen=True)
class ResultSection:
    lines: list[str] = field(default_factory=list)
    title: str | None = None


@dataclass(frozen=True)
class BacktestResult:
    title: str
    sections: list[ResultSection] = field(default_factory=list)


def section(lines: Iterable[str], title: str | None = None) -> ResultSection:
    return ResultSection(title=title, lines=list(lines))


def result(title: str, *sections: ResultSection) -> BacktestResult:
    return BacktestResult(title=title, sections=list(sections))


def render_lines(lines: Iterable[str]) -> str:
    return "\n".join(normalize_lines(lines))


def render_result(result: BacktestResult) -> str:
    lines: list[str] = [f"=== {result.title} ==="]

    for section in result.sections:
        lines.append("")
        if section.title:
            lines.append(f"=== {section.title} ===")
        lines.extend(section.lines)

    return render_lines(lines)


def format_value(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}%"


def format_date(value: date | datetime) -> str:
    return value.strftime("%d/%m/%Y")


__all__ = [
    "BacktestResult",
    "ResultSection",
    "format_date",
    "format_pct",
    "format_value",
    "render_lines",
    "render_result",
    "result",
    "section",
]
