import math

import pandas as pd

from core.features import rolling_zscore
from core.metrics import safe_mean, safe_median
from core.reporting import format_date, format_value, render_result, result as build_result, section
from markets.tesouro_ipca.loader import load_ipca_long_research_frame


def backtest_ipca_entry_signal() -> str:
    """
    Sinal operacional atual do IPCA+.

    Mostra:
    - z atual
    - threshold de entrada
    - regime atual do z (subida / queda / neutro)
    - decisao de entrada
    - estimativa historica de tempo ate o proximo z >= 2.0
      condicionada ao regime atual
    """
    df = load_ipca_long_research_frame(duration_minima=0.0).copy()
    df = (
        df.dropna(subset=["data", "taxa_media"])
        .sort_values("data")
        .reset_index(drop=True)
    )

    entry_threshold = 2.0
    trend_lookback = 21
    neutral_band = 0.10

    df["z_252"] = rolling_zscore(df["taxa_media"], window=252, min_periods=60)
    df["z_1260"] = rolling_zscore(df["taxa_media"], window=1260, min_periods=315)
    df["z_trend_21d"] = df["z_252"] - df["z_252"].shift(trend_lookback)

    def classify_regime(z_trend: float) -> str:
        if pd.isna(z_trend):
            return "indefinido"
        if z_trend > neutral_band:
            return "subida"
        if z_trend < -neutral_band:
            return "queda"
        return "neutro"

    df["z_regime"] = df["z_trend_21d"].apply(classify_regime)

    next_entry_days = []
    z_values = df["z_252"].tolist()

    for i in range(len(df)):
        days_to_next = math.nan
        for j in range(i + 1, len(df)):
            z_next = z_values[j]
            if pd.notna(z_next) and z_next >= entry_threshold:
                days_to_next = j - i
                break
        next_entry_days.append(days_to_next)

    df["days_to_next_entry"] = next_entry_days

    valid = df.dropna(subset=["z_252", "days_to_next_entry"]).copy()

    regime_stats = {}
    for regime in ["subida", "queda", "neutro"]:
        subset = valid[valid["z_regime"] == regime].copy()
        if subset.empty:
            regime_stats[regime] = None
            continue

        avg_days = safe_mean(subset["days_to_next_entry"])
        median_days = safe_median(subset["days_to_next_entry"])

        regime_stats[regime] = {
            "count": int(len(subset)),
            "avg_days": avg_days,
            "median_days": median_days,
            "avg_years": avg_days / 252.0,
            "median_years": median_days / 252.0,
        }

    current = df.dropna(subset=["z_252"]).iloc[-1]

    current_date = current["data"]
    current_rate = float(current["taxa_media"])
    current_z252 = float(current["z_252"])
    current_z1260 = float(current["z_1260"]) if pd.notna(current["z_1260"]) else math.nan
    current_trend = float(current["z_trend_21d"]) if pd.notna(current["z_trend_21d"]) else math.nan
    current_regime = current["z_regime"]

    enter_now = current_z252 >= entry_threshold
    distance_to_entry = entry_threshold - current_z252

    stats = regime_stats.get(current_regime)
    if stats is not None:
        est_avg_years = stats["avg_years"]
        est_median_years = stats["median_years"]
        est_avg_days = stats["avg_days"]
        est_median_days = stats["median_days"]
        stats_count = stats["count"]
    else:
        est_avg_years = math.nan
        est_median_years = math.nan
        est_avg_days = math.nan
        est_median_days = math.nan
        stats_count = 0

    intro_lines = [
        f"Data atual da serie: {format_date(current_date)}",
        f"Taxa real atual: {format_value(current_rate)}",
        f"z_252 atual: {format_value(current_z252)}",
        (
            f"z_1260 atual: {format_value(current_z1260)}"
            if not math.isnan(current_z1260)
            else "z_1260 atual: nan"
        ),
        f"threshold de entrada: {format_value(entry_threshold)}",
        f"distancia ate entrada: {distance_to_entry:+.2f}",
        "",
        (
            f"tendencia do z (21d): {current_trend:+.2f}"
            if not math.isnan(current_trend)
            else "tendencia do z (21d): nan"
        ),
        f"regime atual: {current_regime}",
    ]

    decisao_lines: list[str] = []
    if enter_now:
        decisao_lines.append("SINAL: ENTRA")
        decisao_lines.append("Motivo: z_252 atual ja esta em ou acima do threshold operacional.")
    else:
        decisao_lines.append("SINAL: NAO ENTRA")
        decisao_lines.append("Motivo: z_252 atual ainda esta abaixo do threshold operacional.")

    estimativa_lines: list[str] = []
    if stats_count > 0:
        estimativa_lines.append(
            f"Condicional ao regime atual ({current_regime}), usando {stats_count} observacoes historicas:"
        )
        estimativa_lines.append(f"media: {est_avg_days:.1f} dias uteis (~{est_avg_years:.2f} anos)")
        estimativa_lines.append(f"mediana: {est_median_days:.1f} dias uteis (~{est_median_years:.2f} anos)")
    else:
        estimativa_lines.append("Sem observacoes suficientes para estimar o tempo ate a proxima entrada neste regime.")

    resumo_lines: list[str] = []
    for regime in ["subida", "neutro", "queda"]:
        regime_stat = regime_stats.get(regime)
        if regime_stat is None:
            resumo_lines.append(f"{regime}: sem dados")
        else:
            resumo_lines.append(
                f"{regime}: n={regime_stat['count']} | "
                f"media={regime_stat['avg_days']:.1f} dias (~{regime_stat['avg_years']:.2f} anos) | "
                f"mediana={regime_stat['median_days']:.1f} dias (~{regime_stat['median_years']:.2f} anos)"
            )

    report = build_result(
        "IPCA+ ENTRY SIGNAL",
        section(intro_lines),
        section(decisao_lines, title="DECISAO OPERACIONAL"),
        section(estimativa_lines, title="ESTIMATIVA HISTORICA ATE PROXIMA ENTRADA"),
        section(resumo_lines, title="RESUMO POR REGIME"),
    )
    return render_result(report)


__all__ = ["backtest_ipca_entry_signal"]
