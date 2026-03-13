from __future__ import annotations

from typing import Callable

from markets.tesouro_ipca.backtests import (
    backtest_optimize_entry_threshold_fine,
    backtest_realrate_state_of_art,
)
from markets.tesouro_ipca.signals import backtest_ipca_entry_signal
from markets.macro_system.backtests import (
    backtest_fx_regime_event_sensitivity,
    backtest_realrate_signal_validity_by_fx_regime,
    backtest_realrate_trade_by_fx_regime,
    backtest_realrate_trade_fx_regime_detail,
)

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


def create_backtest_registry() -> BacktestRegistry:
    registry = BacktestRegistry()

    registry.register("IPCA+ Entry Signal", backtest_ipca_entry_signal)
    registry.register(
        "Optimize Entry Threshold Fine",
        backtest_optimize_entry_threshold_fine,
    )
    registry.register("IPCA+ State of Art", backtest_realrate_state_of_art)
    registry.register("FX Regime Event Sensitivity", backtest_fx_regime_event_sensitivity)
    registry.register("Real Rate Trade by FX Regime", backtest_realrate_trade_by_fx_regime)
    registry.register(
        "Real Rate Trade FX Regime Detail",
        backtest_realrate_trade_fx_regime_detail,
    )
    registry.register(
        "Real Rate Signal Validity by FX Regime",
        backtest_realrate_signal_validity_by_fx_regime,
    )

    return registry


__all__ = ["BacktestFn", "BacktestRegistry", "create_backtest_registry"]
