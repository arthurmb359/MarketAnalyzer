import pandas as pd


def mark_signal_events(
    df: pd.DataFrame,
    signal_col: str,
    threshold: float,
) -> tuple[pd.DataFrame, str]:
    above = df[signal_col] >= threshold
    prev = above.shift(1, fill_value=False)
    event_col = f"{signal_col}_event"
    work = df.copy()
    work[event_col] = above & (~prev)
    return work, event_col


__all__ = ["mark_signal_events"]
