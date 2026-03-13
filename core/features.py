from __future__ import annotations

import pandas as pd


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, int(window * 0.25))

    rolling_mean = series.rolling(
        window=window,
        min_periods=min_periods,
    ).mean()

    rolling_std = series.rolling(
        window=window,
        min_periods=min_periods,
    ).std()

    return (series - rolling_mean) / rolling_std


def rolling_quantile(
    series: pd.Series,
    window: int,
    quantile: float,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, int(window * 0.25))

    return series.rolling(
        window=window,
        min_periods=min_periods,
    ).quantile(quantile)


__all__ = ["rolling_quantile", "rolling_zscore"]
