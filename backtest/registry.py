from __future__ import annotations

from typing import Callable

BacktestFn = Callable[[], str]


class BacktestRegistry:
    def __init__(self) -> None:
        self._algorithms: dict[str, BacktestFn] = {}

    def register(self, name: str, fn: BacktestFn) -> None:
        self._algorithms[name] = fn

    def names(self) -> list[str]:
        return list(self._algorithms.keys())

    def get(self, name: str) -> BacktestFn:
        return self._algorithms[name]