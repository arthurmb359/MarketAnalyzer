import math

import pandas as pd

from core.features import rolling_zscore
from core.metrics import safe_mean, safe_median, win_rate_pct
from core.reporting import render_lines, render_result, result as build_result, section
from markets.tesouro_ipca.loader import load_ipca_long_research_frame


def backtest_optimize_entry_threshold_fine() -> str:
    z_values = [
        1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40,
        2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.10, 3.20, 3.30, 3.40,
        3.50,
    ]
    duration_minima = 15
    exit_threshold = -2.0
    base_notional = 100.0
    entry_threshold_1260_overlay = 1.2

    df = load_ipca_long_research_frame(duration_minima=0.0).copy()
    df = (
        df.dropna(subset=["data", "taxa_media"])
        .sort_values("data")
        .reset_index(drop=True)
    )

    df["z_252"] = rolling_zscore(df["taxa_media"], window=252, min_periods=60)
    df["z_1260"] = rolling_zscore(df["taxa_media"], window=1260, min_periods=315)

    intro_lines = [
        "z_entry_threshold grid: " + ", ".join(f"{z:.2f}" for z in z_values),
        f"duration_minima fixa: {duration_minima}",
        f"saída fixa: zscore_rolling_252d <= {exit_threshold}",
        "lógica operacional: início do stress + sizing defensivo",
        "critério principal: ganho absoluto total",
    ]

    results = []

    for z_entry in z_values:
        try:
            work = df.copy()

            work["entry_signal_raw"] = work["z_252"] >= z_entry
            prev_signal = work["entry_signal_raw"].shift(1, fill_value=False)
            work["entry_event"] = work["entry_signal_raw"] & (~prev_signal)

            trades = []

            in_trade = False
            entry_idx = None
            entry_date = None
            entry_rate = None
            entry_z252 = None
            entry_z1260 = None
            entry_weight = None
            current_weight = None

            add_20_done = False
            add_25_done = False
            add_30_done = False

            for i, row in work.iterrows():
                z252 = row["z_252"]
                z1260 = row["z_1260"]
                rate = float(row["taxa_media"])
                dt = row["data"]

                if not in_trade:
                    if bool(row["entry_event"]):
                        base_weight = (
                            1.5
                            if (pd.notna(z1260) and z1260 >= entry_threshold_1260_overlay)
                            else 1.0
                        )

                        in_trade = True
                        entry_idx = i
                        entry_date = dt
                        entry_rate = rate
                        entry_z252 = float(z252)
                        entry_z1260 = float(z1260) if pd.notna(z1260) else None
                        entry_weight = base_weight
                        current_weight = base_weight

                        add_20_done = False
                        add_25_done = False
                        add_30_done = False

                else:
                    holding_days = i - entry_idx

                    new_weight = current_weight

                    if (not add_20_done) and pd.notna(z252) and z252 >= 2.0:
                        new_weight += 0.5
                        add_20_done = True

                    if (not add_25_done) and pd.notna(z252) and z252 >= 2.5:
                        new_weight += 1.0
                        add_25_done = True

                    if (not add_30_done) and pd.notna(z252) and z252 >= 3.0:
                        new_weight += 1.5
                        add_30_done = True

                    current_weight = min(new_weight, 4.5)

                    if pd.notna(z252) and holding_days >= duration_minima and z252 <= exit_threshold:
                        exit_rate = rate
                        rate_move = entry_rate - exit_rate
                        score = base_notional * current_weight * rate_move

                        return_pct = (score / (base_notional * entry_weight)) * 100.0

                        trades.append(
                            {
                                "entry_date": entry_date,
                                "exit_date": dt,
                                "entry_z252": entry_z252,
                                "entry_z1260": entry_z1260,
                                "entry_weight": entry_weight,
                                "exit_weight": current_weight,
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
                        entry_z252 = None
                        entry_z1260 = None
                        entry_weight = None
                        current_weight = None
                        add_20_done = False
                        add_25_done = False
                        add_30_done = False

            trades_df = pd.DataFrame(trades)

            if trades_df.empty:
                result = {
                    "z_entry_threshold": z_entry,
                    "duration_minima": duration_minima,
                    "trades": 0,
                    "discarded": 0,
                    "avg_return": float("nan"),
                    "median_return": float("nan"),
                    "win_rate": float("nan"),
                    "avg_absolute_pnl_units": float("nan"),
                    "median_absolute_pnl_units": float("nan"),
                    "total_absolute_pnl_units": float("nan"),
                    "avg_holding": float("nan"),
                    "avg_weight": float("nan"),
                }
            else:
                result = {
                    "z_entry_threshold": z_entry,
                    "duration_minima": duration_minima,
                    "trades": int(len(trades_df)),
                    "discarded": 0,
                    "avg_return": safe_mean(trades_df["return_pct"]),
                    "median_return": safe_median(trades_df["return_pct"]),
                    "win_rate": win_rate_pct(trades_df["score"]),
                    "avg_absolute_pnl_units": safe_mean(trades_df["score"]),
                    "median_absolute_pnl_units": safe_median(trades_df["score"]),
                    "total_absolute_pnl_units": float(trades_df["score"].sum()),
                    "avg_holding": safe_mean(trades_df["holding_days"]),
                    "avg_weight": safe_mean(trades_df["exit_weight"]),
                }

            results.append(result)

        except Exception as exc:
            results.append({
                "z_entry_threshold": z_entry,
                "duration_minima": duration_minima,
                "trades": 0,
                "discarded": 0,
                "avg_return": float("nan"),
                "median_return": float("nan"),
                "win_rate": float("nan"),
                "avg_absolute_pnl_units": float("nan"),
                "median_absolute_pnl_units": float("nan"),
                "total_absolute_pnl_units": float("nan"),
                "avg_holding": float("nan"),
                "avg_weight": float("nan"),
                "error": str(exc),
            })

    valid_results = [
        r for r in results if not math.isnan(r["total_absolute_pnl_units"])
    ]

    grid_lines: list[str] = []
    for r in results:
        grid_lines.append(
            f"z={r['z_entry_threshold']:.2f} | "
            f"trades={r['trades']} | "
            f"desc={r['discarded']} | "
            f"ret_médio={r['avg_return'] if not math.isnan(r['avg_return']) else float('nan'):.2f}% | "
            f"mediana={r['median_return'] if not math.isnan(r['median_return']) else float('nan'):.2f}% | "
            f"win={r['win_rate'] if not math.isnan(r['win_rate']) else float('nan'):.1f}% | "
            f"abs_médio={r['avg_absolute_pnl_units'] if not math.isnan(r['avg_absolute_pnl_units']) else float('nan'):.2f} | "
            f"abs_total={r['total_absolute_pnl_units'] if not math.isnan(r['total_absolute_pnl_units']) else float('nan'):.2f} | "
            f"holding={r['avg_holding'] if not math.isnan(r['avg_holding']) else float('nan'):.1f} | "
            f"peso={r['avg_weight'] if not math.isnan(r['avg_weight']) else float('nan'):.2f}x"
        )
        if "error" in r:
            grid_lines.append(f"  erro: {r['error']}")

    if not valid_results:
        report = build_result(
            "BACKTEST: OTIMIZAÇÃO FINA DO THRESHOLD DE ENTRADA",
            section(intro_lines),
            section(grid_lines + ["", "Nenhum resultado válido encontrado."], title="RESULTADOS DO GRID"),
        )
        return render_result(report)

    best_abs_total = max(valid_results, key=lambda x: x["total_absolute_pnl_units"])
    best_abs_avg = max(valid_results, key=lambda x: x["avg_absolute_pnl_units"])
    best_return_avg = max(valid_results, key=lambda x: x["avg_return"])
    best_median = max(valid_results, key=lambda x: x["median_return"])

    best_total_lines = [
        f"z={best_abs_total['z_entry_threshold']:.2f} | "
        f"abs_total={best_abs_total['total_absolute_pnl_units']:.2f} | "
        f"abs_médio={best_abs_total['avg_absolute_pnl_units']:.2f} | "
        f"ret_médio={best_abs_total['avg_return']:.2f}% | "
        f"mediana={best_abs_total['median_return']:.2f}% | "
        f"win={best_abs_total['win_rate']:.1f}% | "
        f"trades={best_abs_total['trades']}"
    ]

    best_avg_lines = [
        f"z={best_abs_avg['z_entry_threshold']:.2f} | "
        f"abs_médio={best_abs_avg['avg_absolute_pnl_units']:.2f} | "
        f"abs_total={best_abs_avg['total_absolute_pnl_units']:.2f} | "
        f"ret_médio={best_abs_avg['avg_return']:.2f}% | "
        f"mediana={best_abs_avg['median_return']:.2f}% | "
        f"win={best_abs_avg['win_rate']:.1f}% | "
        f"trades={best_abs_avg['trades']}"
    ]

    best_return_lines = [
        f"z={best_return_avg['z_entry_threshold']:.2f} | "
        f"ret_médio={best_return_avg['avg_return']:.2f}% | "
        f"mediana={best_return_avg['median_return']:.2f}% | "
        f"abs_total={best_return_avg['total_absolute_pnl_units']:.2f} | "
        f"win={best_return_avg['win_rate']:.1f}% | "
        f"trades={best_return_avg['trades']}"
    ]

    best_median_lines = [
        f"z={best_median['z_entry_threshold']:.2f} | "
        f"mediana={best_median['median_return']:.2f}% | "
        f"ret_médio={best_median['avg_return']:.2f}% | "
        f"abs_total={best_median['total_absolute_pnl_units']:.2f} | "
        f"win={best_median['win_rate']:.1f}% | "
        f"trades={best_median['trades']}"
    ]

    report = build_result(
        "BACKTEST: OTIMIZAÇÃO FINA DO THRESHOLD DE ENTRADA",
        section(intro_lines),
        section(grid_lines, title="RESULTADOS DO GRID"),
        section(best_total_lines, title="MELHOR POR GANHO ABSOLUTO TOTAL"),
        section(best_avg_lines, title="MELHOR POR GANHO ABSOLUTO MÉDIO"),
        section(best_return_lines, title="MELHOR POR RETORNO MÉDIO"),
        section(best_median_lines, title="MELHOR POR MEDIANA"),
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
    "backtest_realrate_state_of_art",
]

