import pandas as pd

from markets.usdbrl.loader import load_usdbrl_frame


def build_usdbrl_macro_base_frame() -> pd.DataFrame:
    df = load_usdbrl_frame().copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "usdbrl" not in df.columns:
        raise ValueError(
            f"load_usdbrl_frame() deveria retornar 'usdbrl'. Colunas: {list(df.columns)}"
        )

    df = df.sort_values("data").reset_index(drop=True)
    df["usd_ret_1d"] = df["usdbrl"].pct_change()
    df["usd_trend_21d"] = df["usdbrl"] / df["usdbrl"].shift(21) - 1.0
    df["usd_trend_63d"] = df["usdbrl"] / df["usdbrl"].shift(63) - 1.0
    df["usd_trend_126d"] = df["usdbrl"] / df["usdbrl"].shift(126) - 1.0
    df["usd_vol_21d"] = df["usd_ret_1d"].rolling(21, min_periods=10).std()
    df["usd_vol_p80"] = df["usd_vol_21d"].rolling(252, min_periods=60).quantile(0.80)
    df["usd_trend_63d_p80"] = df["usd_trend_63d"].rolling(252, min_periods=60).quantile(0.80)
    df["usd_trend_accel"] = (
        (df["usd_trend_21d"] > 0)
        & (df["usd_trend_63d"] > 0)
        & (df["usd_trend_126d"] > 0)
        & (df["usd_trend_21d"] > df["usd_trend_63d"])
        & (df["usd_trend_63d"] > df["usd_trend_126d"])
    )
    return df


__all__ = ["build_usdbrl_macro_base_frame"]
