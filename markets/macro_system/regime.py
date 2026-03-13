import pandas as pd

from markets.usdbrl.series import build_usdbrl_macro_base_frame


def build_fx_macro_regime_frame() -> pd.DataFrame:
    df = build_usdbrl_macro_base_frame().copy()

    def classify_fx_regime(row: pd.Series) -> str:
        if (
            pd.isna(row["usd_vol_21d"])
            or pd.isna(row["usd_vol_p80"])
            or pd.isna(row["usd_trend_63d"])
            or pd.isna(row["usd_trend_63d_p80"])
        ):
            return "indefinido"

        accel = bool(row["usd_trend_accel"])
        vol_high = row["usd_vol_21d"] > row["usd_vol_p80"]
        trend_high = row["usd_trend_63d"] > row["usd_trend_63d_p80"]

        if (trend_high and vol_high) or accel:
            return "stress"
        if trend_high:
            return "alerta"
        if vol_high:
            return "turbulencia"
        return "normal"

    df["fx_macro_regime_raw"] = df.apply(classify_fx_regime, axis=1)

    smoothed = []
    values = df["fx_macro_regime_raw"].tolist()

    for i in range(len(values)):
        start = max(0, i - 4)
        window = values[start:i + 1]
        counts = pd.Series(window).value_counts()
        smoothed.append(counts.index[0])

    df["fx_macro_regime"] = smoothed
    return df


__all__ = ["build_fx_macro_regime_frame"]
