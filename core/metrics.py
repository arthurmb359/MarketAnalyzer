from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def safe_mean(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype="float64").dropna()
    return float("nan") if series.empty else float(series.mean())


def safe_median(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype="float64").dropna()
    return float("nan") if series.empty else float(series.median())


def win_rate_pct(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype="float64").dropna()
    return float("nan") if series.empty else float((series > 0).mean() * 100.0)


__all__ = ["safe_mean", "safe_median", "win_rate_pct"]
