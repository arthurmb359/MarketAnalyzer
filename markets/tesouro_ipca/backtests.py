import math

import pandas as pd

from core.features import rolling_zscore
from core.metrics import safe_mean, safe_median, win_rate_pct
from core.reporting import render_lines, render_result, result as build_result, section
from markets.tesouro_ipca.loader import load_ipca_long_research_frame


def _float_range(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start

    while current <= (stop + 1e-9):
        values.append(round(current, 10))
        current += step

    return values


def _int_range(start: int, stop: int, step: int) -> list[int]:
    return list(range(start, stop + 1, step))


def _approx_duration_from_prazo(prazo_anos: float) -> float:
    if pd.isna(prazo_anos):
        return 0.0
    return max(0.0, float(prazo_anos))


def _mark_to_market_return_pct(
    entry_rate_pct: float,
    exit_rate_pct: float,
    duration: float,
) -> float:
    delta_rate_pp = exit_rate_pct - entry_rate_pct
    return -duration * delta_rate_pp


def backtest_optimize_zscore_grid() -> str:
    entry_thresholds = _float_range(1.0, 3.0, 0.2)
    rolling_windows = _int_range(252, 1242, 90)
    exit_thresholds = [-value for value in _float_range(1.0, 2.0, 0.2)]
    duration_minima = 15
    base_notional = 100.0

    df = load_ipca_long_research_frame(duration_minima=0.0).copy()
    df = (
        df.dropna(subset=["data", "taxa_media"])
        .sort_values("data")
        .reset_index(drop=True)
    )
    dates = df["data"].tolist()
    rates = df["taxa_media"].astype(float).tolist()
    prazos = df["prazo_anos"].astype(float).tolist()

    intro_lines = [
        "entry_z grid: " + ", ".join(f"{value:.1f}" for value in entry_thresholds),
        "z_rolling grid: " + ", ".join(str(value) for value in rolling_windows),
        "exit_z grid: " + ", ".join(f"{value:.1f}" for value in exit_thresholds),
        f"duration_minima fixa: {duration_minima}",
        "logica operacional: primeiro cruzamento do threshold de entrada e saida por reversao do zscore",
        "pnl: aproximacao mark-to-market de NTN-B usando duration ~= prazo_anos na entrada",
        "nao inclui carry real, IPCA, cupom, custos ou convexidade",
        "criterio principal: score_total",
    ]

    results: list[dict[str, float | int]] = []

    for window in rolling_windows:
        zscores = rolling_zscore(
            df["taxa_media"],
            window=window,
        ).tolist()

        for entry_threshold in entry_thresholds:
            for exit_threshold in exit_thresholds:
                trades: list[dict[str, float | int | object]] = []

                in_trade = False
                entry_idx: int | None = None
                entry_date = None
                entry_rate: float | None = None
                entry_prazo: float | None = None
                entry_zscore: float | None = None
                prev_signal = False

                for i, zscore in enumerate(zscores):
                    rate = rates[i]
                    prazo = prazos[i]
                    dt = dates[i]
                    signal = pd.notna(zscore) and zscore >= entry_threshold
                    entry_event = signal and (not prev_signal)
                    prev_signal = bool(signal)

                    if not in_trade:
                        if entry_event:
                            in_trade = True
                            entry_idx = i
                            entry_date = dt
                            entry_rate = rate
                            entry_prazo = prazo
                            entry_zscore = float(zscore)
                    else:
                        holding_days = i - entry_idx

                        if (
                            pd.notna(zscore)
                            and holding_days >= duration_minima
                            and zscore <= exit_threshold
                        ):
                            exit_rate = rate
                            rate_move = entry_rate - exit_rate
                            duration = _approx_duration_from_prazo(entry_prazo)
                            return_pct = _mark_to_market_return_pct(
                                entry_rate_pct=entry_rate,
                                exit_rate_pct=exit_rate,
                                duration=duration,
                            )
                            score = base_notional * (return_pct / 100.0)

                            trades.append(
                                {
                                    "entry_date": entry_date,
                                    "exit_date": dt,
                                    "entry_zscore": entry_zscore,
                                    "exit_zscore": float(zscore),
                                    "entry_prazo": float(entry_prazo),
                                    "duration": float(duration),
                                    "holding_days": int(holding_days),
                                    "rate_move": float(rate_move),
                                    "score": float(score),
                                    "return_pct": float(return_pct),
                                }
                            )

                            in_trade = False
                            entry_idx = None
                            entry_date = None
                            entry_rate = None
                            entry_prazo = None
                            entry_zscore = None

                trades_df = pd.DataFrame(trades)

                if trades_df.empty:
                    result = {
                        "z_rolling": window,
                        "z_entry_threshold": entry_threshold,
                        "z_exit_threshold": exit_threshold,
                        "trades": 0,
                        "avg_return": float("nan"),
                        "median_return": float("nan"),
                        "win_rate": float("nan"),
                        "score_total": float("nan"),
                        "score_mean": float("nan"),
                        "holding_mean": float("nan"),
                    }
                else:
                    result = {
                        "z_rolling": window,
                        "z_entry_threshold": entry_threshold,
                        "z_exit_threshold": exit_threshold,
                        "trades": int(len(trades_df)),
                        "avg_return": safe_mean(trades_df["return_pct"]),
                        "median_return": safe_median(trades_df["return_pct"]),
                        "win_rate": win_rate_pct(trades_df["score"]),
                        "score_total": float(trades_df["score"].sum()),
                        "score_mean": safe_mean(trades_df["score"]),
                        "holding_mean": safe_mean(trades_df["holding_days"]),
                    }

                results.append(result)

    valid_results = [r for r in results if not math.isnan(r["score_total"])]
    sorted_results = sorted(
        valid_results,
        key=lambda item: (
            item["score_total"],
            item["avg_return"],
            item["win_rate"],
        ),
        reverse=True,
    )

    if not valid_results:
        report = build_result(
            "BACKTEST: OTIMIZACAO GRID ZSCORE",
            section(intro_lines),
            section(
                [
                    f"combos_testados={len(results)}",
                    "Nenhum resultado valido encontrado.",
                ],
                title="RESUMO",
            ),
        )
        return render_result(report)

    top_lines: list[str] = []
    for window in rolling_windows:
        window_results = [
            row for row in sorted_results if row["z_rolling"] == window
        ]
        if not window_results:
            continue

        top_lines.append(f"z_rolling={window}")
        for idx, row in enumerate(window_results[:20], start=1):
            top_lines.append(
                f"{idx:02d}. "
                f"entry>={row['z_entry_threshold']:.1f} | "
                f"exit<={row['z_exit_threshold']:.1f} | "
                f"trades={row['trades']} | "
                f"score_total={row['score_total']:.2f} | "
                f"score_medio={row['score_mean']:.2f} | "
                f"ret_medio={row['avg_return']:.2f}% | "
                f"mediana={row['median_return']:.2f}% | "
                f"win={row['win_rate']:.1f}% | "
                f"holding={row['holding_mean']:.1f}d"
            )
        top_lines.append("")

    best_total = sorted_results[0]
    best_avg_return = max(valid_results, key=lambda item: item["avg_return"])
    best_median = max(valid_results, key=lambda item: item["median_return"])
    best_win_rate = max(valid_results, key=lambda item: item["win_rate"])

    summary_lines = [
        f"combos_testados={len(results)}",
        f"combos_validos={len(valid_results)}",
        "top_por_z_rolling=20",
        f"janelas_com_resultado={len({row['z_rolling'] for row in valid_results})}",
    ]

    best_total_lines = [
        f"z_roll={best_total['z_rolling']} | "
        f"entry>={best_total['z_entry_threshold']:.1f} | "
        f"exit<={best_total['z_exit_threshold']:.1f} | "
        f"score_total={best_total['score_total']:.2f} | "
        f"score_medio={best_total['score_mean']:.2f} | "
        f"ret_medio={best_total['avg_return']:.2f}% | "
        f"win={best_total['win_rate']:.1f}% | "
        f"trades={best_total['trades']}"
    ]

    best_return_lines = [
        f"z_roll={best_avg_return['z_rolling']} | "
        f"entry>={best_avg_return['z_entry_threshold']:.1f} | "
        f"exit<={best_avg_return['z_exit_threshold']:.1f} | "
        f"ret_medio={best_avg_return['avg_return']:.2f}% | "
        f"score_total={best_avg_return['score_total']:.2f} | "
        f"win={best_avg_return['win_rate']:.1f}% | "
        f"trades={best_avg_return['trades']}"
    ]

    best_median_lines = [
        f"z_roll={best_median['z_rolling']} | "
        f"entry>={best_median['z_entry_threshold']:.1f} | "
        f"exit<={best_median['z_exit_threshold']:.1f} | "
        f"mediana={best_median['median_return']:.2f}% | "
        f"score_total={best_median['score_total']:.2f} | "
        f"win={best_median['win_rate']:.1f}% | "
        f"trades={best_median['trades']}"
    ]

    best_win_lines = [
        f"z_roll={best_win_rate['z_rolling']} | "
        f"entry>={best_win_rate['z_entry_threshold']:.1f} | "
        f"exit<={best_win_rate['z_exit_threshold']:.1f} | "
        f"win={best_win_rate['win_rate']:.1f}% | "
        f"score_total={best_win_rate['score_total']:.2f} | "
        f"ret_medio={best_win_rate['avg_return']:.2f}% | "
        f"trades={best_win_rate['trades']}"
    ]

    report = build_result(
        "BACKTEST: OTIMIZACAO GRID ZSCORE",
        section(intro_lines),
        section(summary_lines, title="RESUMO"),
        section(best_total_lines, title="MELHOR POR SCORE TOTAL"),
        section(best_return_lines, title="MELHOR POR RETORNO MEDIO"),
        section(best_median_lines, title="MELHOR POR MEDIANA"),
        section(best_win_lines, title="MELHOR POR WIN RATE"),
        section(top_lines, title="TOP 20 POR Z_ROLLING"),
    )
    return render_result(report)


def backtest_realrate_state_of_art() -> str:
    """
    Estado da arte atual do sistema Real Rate / IPCA+.

    Modo operacional:
    - entrada no INÍCIO do stress
    - evento = primeiro cruzamento de z_252 >= 2.0

    Peso inicial:
    - 1.0x se z_1260 < 1.2
    - 1.5x se z_1260 >= 1.2

    Escalonamento tático agressivo:
    - z_252 >= 2.5 -> +1.0x
    - z_252 >= 3.0 -> +1.5x
    - z_252 >= 3.5 -> +2.5x

    Peso máximo efetivo:
    - 6.5x

    Saída:
    - z_252 <= -2.0
    - duration_minima = 15

    Métricas de stress:
    - maxDD_MTM: drawdown mark-to-market simplificado
    - carry_proxy_anual: proxy conservador de carry anual
    - anos_para_recuperar_dd: quantos anos de carry seriam necessários
      para compensar o pior drawdown
    """

    import math
    import pandas as pd

    df = load_ipca_long_research_frame(duration_minima=0.0).copy()
    df = (
        df.dropna(subset=["data", "taxa_media"])
        .sort_values("data")
        .reset_index(drop=True)
    )

    entry_threshold_252 = 2.0
    entry_threshold_1260_overlay = 1.2
    exit_threshold = -2.0
    duration_minima = 15
    base_notional = 100.0

    # fração conservadora da taxa real usada como carry aproveitável
    carry_proxy_fraction = 0.30

    def base_entry_weight(z1260: float) -> float:
        return 1.5 if (pd.notna(z1260) and z1260 >= entry_threshold_1260_overlay) else 1.0

    df["z_252"] = rolling_zscore(df["taxa_media"], window=252, min_periods=60)
    df["z_1260"] = rolling_zscore(df["taxa_media"], window=1260, min_periods=315)

    # entrada = primeiro cruzamento do stress
    df["entry_signal_raw"] = df["z_252"] >= entry_threshold_252
    prev_signal = df["entry_signal_raw"].shift(1, fill_value=False)
    df["entry_event"] = df["entry_signal_raw"] & (~prev_signal)

    trades = []

    in_trade = False
    entry_idx = None
    entry_date = None
    entry_rate = None
    entry_z252 = None
    entry_z1260 = None
    entry_weight = None

    current_weight = None
    worst_mtm_pnl = None
    worst_mtm_date = None
    worst_mtm_rate = None
    weight_at_dd = None

    add_25_done = False
    add_30_done = False
    add_35_done = False

    for i, row in df.iterrows():
        z252 = row["z_252"]
        z1260 = row["z_1260"]
        rate = float(row["taxa_media"])
        dt = row["data"]

        if not in_trade:
            if bool(row["entry_event"]):
                weight = base_entry_weight(z1260)

                in_trade = True
                entry_idx = i
                entry_date = dt
                entry_rate = rate
                entry_z252 = float(z252)
                entry_z1260 = float(z1260) if pd.notna(z1260) else None
                entry_weight = weight

                current_weight = weight
                add_25_done = False
                add_30_done = False
                add_35_done = False

                worst_mtm_pnl = 0.0
                worst_mtm_date = None
                worst_mtm_rate = entry_rate
                weight_at_dd = current_weight

        else:
            holding_days = i - entry_idx

            # escalonamento progressivo agressivo
            new_weight = current_weight

            if (not add_25_done) and pd.notna(z252) and z252 >= 2.5:
                new_weight += 1.0
                add_25_done = True

            if (not add_30_done) and pd.notna(z252) and z252 >= 3.0:
                new_weight += 1.5
                add_30_done = True

            if (not add_35_done) and pd.notna(z252) and z252 >= 3.5:
                new_weight += 2.5
                add_35_done = True

            current_weight = min(new_weight, 6.5)

            # MTM simplificado usando peso corrente
            mtm_pnl = base_notional * current_weight * (entry_rate - rate)

            if mtm_pnl < worst_mtm_pnl:
                worst_mtm_pnl = mtm_pnl
                worst_mtm_date = dt
                worst_mtm_rate = rate
                weight_at_dd = current_weight

            if pd.notna(z252) and holding_days >= duration_minima and z252 <= exit_threshold:
                exit_rate = rate
                rate_move = entry_rate - exit_rate
                score = base_notional * current_weight * rate_move

                # carry proxy anual no ponto do DD
                carry_proxy_anual = (
                    base_notional
                    * (weight_at_dd if weight_at_dd is not None else current_weight)
                    * entry_rate
                    * carry_proxy_fraction
                )

                if carry_proxy_anual > 0:
                    anos_para_recuperar_dd = abs(worst_mtm_pnl) / carry_proxy_anual
                else:
                    anos_para_recuperar_dd = math.nan

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": dt,
                        "entry_rate": entry_rate,
                        "exit_rate": exit_rate,
                        "entry_z252": entry_z252,
                        "entry_z1260": entry_z1260,
                        "entry_weight": entry_weight,
                        "exit_weight": current_weight,
                        "exit_z252": float(z252),
                        "holding_days": int(holding_days),
                        "rate_move": float(rate_move),
                        "score": float(score),
                        "max_drawdown_score": float(worst_mtm_pnl),
                        "max_drawdown_date": worst_mtm_date,
                        "max_drawdown_rate": float(worst_mtm_rate - entry_rate) if worst_mtm_date is not None else 0.0,
                        "weight_at_dd": float(weight_at_dd) if weight_at_dd is not None else 0.0,
                        "carry_proxy_anual": float(carry_proxy_anual),
                        "anos_para_recuperar_dd": float(anos_para_recuperar_dd) if not math.isnan(anos_para_recuperar_dd) else None,
                        "hit_25": add_25_done,
                        "hit_30": add_30_done,
                        "hit_35": add_35_done,
                    }
                )

                in_trade = False
                entry_idx = None
                entry_date = None
                entry_rate = None
                entry_z252 = None
                entry_z1260 = None
                entry_weight = None
                current_weight = None
                worst_mtm_pnl = None
                worst_mtm_date = None
                worst_mtm_rate = None
                weight_at_dd = None
                add_25_done = False
                add_30_done = False
                add_35_done = False

    intro_lines = [
        "Regras do sistema:",
        f"Entrada: primeiro cruzamento de z_252 >= {entry_threshold_252:.1f}",
        f"Peso inicial: 1.0x se z_1260 < {entry_threshold_1260_overlay:.1f}",
        f"Peso inicial: 1.5x se z_1260 >= {entry_threshold_1260_overlay:.1f}",
        "Escalonamento tático agressivo:",
        "z_252 >= 2.5 -> +1.0x",
        "z_252 >= 3.0 -> +1.5x",
        "z_252 >= 3.5 -> +2.5x",
        "Peso máximo: 6.5x",
        f"Saída: z_252 <= {exit_threshold:.1f}",
        f"Duração mínima: {duration_minima} dias",
        "",
        "Stress econômico aproximado:",
        f"carry_proxy_anual = {carry_proxy_fraction:.0%} da taxa real de entrada",
        "anos_para_recuperar_dd = abs(maxDD_MTM) / carry_proxy_anual",
    ]

    if not trades:
        report = build_result(
            "REAL RATE STATE OF ART",
            section(intro_lines + ["", "Nenhum trade encontrado."]),
        )
        return render_result(report)

    trades_df = pd.DataFrame(trades)

    win_rate = win_rate_pct(trades_df["score"])
    rate_move_mean = safe_mean(trades_df["rate_move"])
    rate_move_median = safe_median(trades_df["rate_move"])
    score_total = trades_df["score"].sum()
    score_mean = safe_mean(trades_df["score"])
    holding_mean = safe_mean(trades_df["holding_days"])
    entry_weight_mean = safe_mean(trades_df["entry_weight"])
    exit_weight_mean = safe_mean(trades_df["exit_weight"])

    dd_mean = safe_mean(trades_df["max_drawdown_score"])
    dd_worst = trades_df["max_drawdown_score"].min()

    carry_proxy_mean = safe_mean(trades_df["carry_proxy_anual"])
    recovery_years_mean = safe_mean(trades_df["anos_para_recuperar_dd"])
    recovery_years_worst = trades_df["anos_para_recuperar_dd"].dropna().max()

    resumo_lines = [
        f"trades={len(trades_df)}",
        f"win={win_rate:.1f}%",
        f"rate_move_médio={rate_move_mean:.4f}",
        f"rate_move_mediana={rate_move_median:.4f}",
        f"score_médio={score_mean:.2f}",
        f"score_total={score_total:.2f}",
        f"holding_médio={holding_mean:.1f}d",
        f"peso_inicial_médio={entry_weight_mean:.2f}x",
        f"peso_final_médio={exit_weight_mean:.2f}x",
        f"drawdown_médio_mark_to_market={dd_mean:.2f}",
        f"pior_drawdown_mark_to_market={dd_worst:.2f}",
        f"carry_proxy_anual_médio={carry_proxy_mean:.2f}",
        f"anos_médios_para_recuperar_dd={recovery_years_mean:.2f}",
        f"pior_caso_anos_para_recuperar_dd={recovery_years_worst:.2f}",
    ]

    detalhe_lines: list[str] = []
    for _, row in trades_df.iterrows():
        z1260_txt = "nan" if pd.isna(row["entry_z1260"]) else f"{row['entry_z1260']:.2f}"
        dd_date_txt = (
            row["max_drawdown_date"].strftime("%d/%m/%Y")
            if pd.notna(row["max_drawdown_date"])
            else "n/a"
        )
        years_txt = (
            f"{row['anos_para_recuperar_dd']:.2f}"
            if pd.notna(row["anos_para_recuperar_dd"])
            else "n/a"
        )

        detalhe_lines.append(
            f"{row['entry_date'].strftime('%d/%m/%Y')} -> {row['exit_date'].strftime('%d/%m/%Y')} | "
            f"z252={row['entry_z252']:.2f} | "
            f"z1260={z1260_txt} | "
            f"peso_in={row['entry_weight']:.2f}x | "
            f"peso_out={row['exit_weight']:.2f}x | "
            f"hit2.5={'Y' if row['hit_25'] else 'N'} | "
            f"hit3.0={'Y' if row['hit_30'] else 'N'} | "
            f"hit3.5={'Y' if row['hit_35'] else 'N'} | "
            f"rate_move={row['rate_move']:+.4f} | "
            f"score={row['score']:.2f} | "
            f"maxDD_MTM={row['max_drawdown_score']:.2f} | "
            f"carry_proxy_anual={row['carry_proxy_anual']:.2f} | "
            f"anos_recuperar_dd={years_txt} | "
            f"DD_day={dd_date_txt} | "
            f"holding={row['holding_days']}d"
        )

    report = build_result(
        "REAL RATE STATE OF ART",
        section(intro_lines),
        section(resumo_lines, title="RESUMO"),
        section(detalhe_lines, title="DETALHE DOS TRADES"),
    )
    return render_result(report)


__all__ = [
    "backtest_optimize_entry_threshold_fine",
    "backtest_optimize_zscore_grid",
    "backtest_realrate_state_of_art",
]
