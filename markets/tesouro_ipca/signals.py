import math
import math

import pandas as pd

from core.features import rolling_zscore
from core.reporting import format_date, format_value, render_result, result as build_result, section
from markets.tesouro_ipca.loader import load_ipca_long_research_frame


def backtest_ipca_entry_signal() -> str:
    """
    Sinal operacional atual do IPCA+.

    Mostra:
    - z_2412 atual
    - threshold de entrada
    - nivel de alocacao sugerido
    - condicao de saida
    - decisao operacional atual
    """
    df = load_ipca_long_research_frame(duration_minima=0.0).copy()
    df = (
        df.dropna(subset=["data", "taxa_media"])
        .sort_values("data")
        .reset_index(drop=True)
    )

    entry_threshold = 1.2
    add_threshold_mid = 1.6
    add_threshold_high = 2.0
    exit_threshold = -2.0

    df["z_2412"] = rolling_zscore(df["taxa_media"], window=2412, min_periods=603)

    current = df.dropna(subset=["z_2412"]).iloc[-1]

    current_date = current["data"]
    current_rate = float(current["taxa_media"])
    current_z2412 = float(current["z_2412"])

    enter_now = current_z2412 >= entry_threshold
    exit_now = current_z2412 <= exit_threshold

    if current_z2412 >= add_threshold_high:
        target_weight = 4.0
        allocation_status = "Escalonamento completo"
    elif current_z2412 >= add_threshold_mid:
        target_weight = 2.0
        allocation_status = "Primeiro escalonamento ativo"
    elif current_z2412 >= entry_threshold:
        target_weight = 1.0
        allocation_status = "Entrada base ativa"
    else:
        target_weight = 0.0
        allocation_status = "Fora da faixa de entrada"

    distance_to_entry = entry_threshold - current_z2412
    distance_to_exit = current_z2412 - exit_threshold

    intro_lines = [
        f"Data atual da serie: {format_date(current_date)}",
        f"Taxa real atual: {format_value(current_rate)}",
        f"z_2412 atual: {format_value(current_z2412)}",
        f"threshold de entrada: {format_value(entry_threshold)}",
        f"threshold de saida: {format_value(exit_threshold)}",
        f"distancia ate entrada: {distance_to_entry:+.2f}",
        f"distancia ate saida: {distance_to_exit:+.2f}",
        "",
        f"alocacao alvo: {target_weight:.1f}x",
        f"status da alocacao: {allocation_status}",
    ]

    decisao_lines: list[str] = []
    if exit_now:
        decisao_lines.append("SINAL: SAIDA / ZERA")
        decisao_lines.append("Motivo: z_2412 atual ja esta em ou abaixo do threshold de saida.")
    elif enter_now:
        decisao_lines.append("SINAL: ENTRA / MANTEM")
        decisao_lines.append("Motivo: z_2412 atual ja esta em ou acima do threshold operacional.")
    else:
        decisao_lines.append("SINAL: NAO ENTRA")
        decisao_lines.append("Motivo: z_2412 atual ainda esta abaixo do threshold operacional.")

    estrategia_lines = [
        f"Entrada base: z_2412 >= {entry_threshold:.1f} -> 1.0x",
        f"Escalonamento 1: z_2412 >= {add_threshold_mid:.1f} -> +1.0x",
        f"Escalonamento 2: z_2412 >= {add_threshold_high:.1f} -> +2.0x",
        "Peso maximo: 4.0x",
        f"Saida: z_2412 <= {exit_threshold:.1f}",
    ]

    report = build_result(
        "IPCA+ ENTRY SIGNAL",
        section(intro_lines),
        section(decisao_lines, title="DECISAO OPERACIONAL"),
        section(estrategia_lines, title="REGRAS DA ESTRATEGIA"),
    )
    return render_result(report)


__all__ = ["backtest_ipca_entry_signal"]
