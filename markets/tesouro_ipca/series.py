import pandas as pd


def build_daily_ipca_long_series(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(["Data Base", "Data Vencimento"])

    daily = (
        df_sorted
        .groupby("Data Base")
        .tail(1)
        .copy()
    )

    daily = daily.rename(columns={"Taxa Compra Manha": "taxa_media"})
    daily = daily[["Data Base", "taxa_media"]]
    daily = daily.sort_values("Data Base").reset_index(drop=True)

    mean_value = daily["taxa_media"].mean()
    std_value = daily["taxa_media"].std()

    daily["media_historica"] = mean_value
    daily["desvio_padrao"] = std_value
    daily["zscore"] = (daily["taxa_media"] - mean_value) / std_value

    percentis = []
    values = daily["taxa_media"].tolist()

    for i, value in enumerate(values):
        history = values[: i + 1]
        percentil = sum(v <= value for v in history) / len(history)
        percentis.append(percentil)

    daily["percentil_historico"] = percentis

    daily["mm_252"] = daily["taxa_media"].rolling(252, min_periods=30).mean()
    daily["mm_1260"] = daily["taxa_media"].rolling(1260, min_periods=60).mean()

    daily["banda_1dp_sup"] = mean_value + std_value
    daily["banda_1dp_inf"] = mean_value - std_value
    daily["banda_2dp_sup"] = mean_value + 2 * std_value
    daily["banda_2dp_inf"] = mean_value - 2 * std_value

    rolling_window = min(252, len(daily))
    min_periods = min(60, rolling_window)

    daily["media_rolling_252d"] = daily["taxa_media"].rolling(
        window=rolling_window,
        min_periods=min_periods,
    ).mean()

    daily["desvio_rolling_252d"] = daily["taxa_media"].rolling(
        window=rolling_window,
        min_periods=min_periods,
    ).std()

    daily["zscore_rolling_252d"] = (
        (daily["taxa_media"] - daily["media_rolling_252d"])
        / daily["desvio_rolling_252d"]
    )

    return daily


__all__ = ["build_daily_ipca_long_series"]
