import math

import pandas as pd

from core.features import rolling_zscore
from core.metrics import safe_mean, safe_median, win_rate_pct
from core.reporting import render_lines, render_result, result as build_result, section
from markets.tesouro_ipca.loader import load_ipca_long_research_frame, load_tesouro_ipca_frame
from markets.tesouro_ipca.series import build_daily_ipca_duration_bucket_series


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


def _parse_br_numeric(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(".", "").replace(",", ".").strip())


def _prepare_trade_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in ["PU Compra Manha", "PU Venda Manha"]:
        prepared[column] = prepared[column].map(_parse_br_numeric)
    return prepared.dropna(subset=["PU Compra Manha", "PU Venda Manha"]).copy()


def _pu_return_pct(entry_pu_venda: float, exit_pu_compra: float) -> float:
    if entry_pu_venda <= 0:
        return float("nan")
    return ((exit_pu_compra / entry_pu_venda) - 1.0) * 100.0


def backtest_optimize_zscore_grid() -> str:
    entry_thresholds = _float_range(1.0, 3.0, 0.2)
    rolling_windows = _int_range(1852, 7300, 730)
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

    Usa as faixas media e longa com parametros aprendidos no otimizador por
    duration. O sinal vem da serie da faixa, mas a marcacao e venda usam o
    mesmo vencimento comprado na entrada.
    """

    import math
    import pandas as pd

    add_threshold_2412_mid = 1.6
    add_threshold_2412_high = 2.0
    holding_minimo_dias = 15
    base_notional = 100.0
    carry_proxy_fraction = 0.30

    configs = [
        {
            "bucket": "Media",
            "min_prazo": 8.0,
            "max_prazo": 14.0,
            "include_max": False,
            "entry_z": 1.0,
            "exit_z": -1.6,
        },
        {
            "bucket": "Grande",
            "min_prazo": 14.0,
            "max_prazo": 20.0,
            "include_max": True,
            "entry_z": 1.2,
            "exit_z": -2.0,
        },
    ]

    raw_df = load_tesouro_ipca_frame()
    if "Tipo Titulo" in raw_df.columns:
        raw_df = raw_df[raw_df["Tipo Titulo"].astype(str).str.strip() == "Tesouro IPCA+"].copy()
    raw_df = _prepare_trade_price_columns(raw_df)

    quote_history_by_maturity = {
        maturity: group.sort_values("Data Base").reset_index(drop=True)
        for maturity, group in raw_df.groupby("Data Vencimento")
    }

    def quote_same_title_on_or_before(maturity: object, dt: object) -> tuple[float, float, float, object, bool] | None:
        history = quote_history_by_maturity.get(maturity)
        if history is None or history.empty:
            return None

        valid_quotes = history[history["Data Base"] <= dt]
        if valid_quotes.empty:
            return None

        quote = valid_quotes.iloc[-1]
        last_quote = history.iloc[-1]
        is_last_available_quote = (
            quote["Data Base"] == last_quote["Data Base"]
            and quote["Data Base"] < dt
        )
        return (
            float(quote["Taxa Compra Manha"]),
            float(quote["PU Compra Manha"]),
            float(quote["PU Venda Manha"]),
            quote["Data Base"],
            bool(is_last_available_quote),
        )

    all_trades: list[dict[str, float | int | object]] = []
    summary_lines: list[str] = []
    detail_lines: list[str] = []

    for config in configs:
        bucket_name = str(config["bucket"])
        entry_threshold = float(config["entry_z"])
        exit_threshold = float(config["exit_z"])
        df = build_daily_ipca_duration_bucket_series(
            raw_df,
            min_prazo_anos=float(config["min_prazo"]),
            max_prazo_anos=float(config["max_prazo"]),
            include_max=bool(config["include_max"]),
        )
        df = (
            df.dropna(subset=["data", "taxa_media", "prazo_anos"])
            .sort_values("data")
            .reset_index(drop=True)
        )

        if df.empty:
            summary_lines.append(f"{bucket_name}: sem dados na faixa")
            continue

        df["z_2412"] = rolling_zscore(df["taxa_media"], window=2412, min_periods=603)

        trades: list[dict[str, float | int | object]] = []
        open_position: dict[str, object] | None = None
        prev_signal = False

        for i, row in df.iterrows():
            z2412 = row["z_2412"]
            bucket_rate = float(row["taxa_media"])
            dt = row["data"]

            signal = pd.notna(z2412) and z2412 >= entry_threshold
            entry_event = bool(signal) and (not prev_signal)
            prev_signal = bool(signal)

            if entry_event and open_position is None:
                open_position = {
                    "entry_idx": i,
                    "entry_date": dt,
                    "entry_rate": bucket_rate,
                    "entry_z2412": float(z2412),
                    "entry_prazo": float(row["prazo_anos"]),
                    "entry_vencimento": row["data_vencimento"],
                    "entry_weight": 1.0,
                    "current_weight": 1.0,
                    "worst_mtm_score": 0.0,
                    "worst_mtm_date": None,
                    "weight_at_dd": 1.0,
                    "hit_16": False,
                    "hit_20": False,
                }

            if open_position is None:
                continue

            entry_idx = int(open_position["entry_idx"])
            entry_rate = float(open_position["entry_rate"])
            entry_prazo = float(open_position["entry_prazo"])
            entry_vencimento = open_position["entry_vencimento"]
            current_quote = quote_same_title_on_or_before(entry_vencimento, dt)
            if current_quote is None:
                continue

            current_rate, current_pu_compra, _current_pu_venda, quote_date, is_last_available_quote = current_quote
            holding_days = i - entry_idx
            current_weight = float(open_position["current_weight"])

            if "entry_pu_venda" not in open_position:
                entry_quote = quote_same_title_on_or_before(entry_vencimento, open_position["entry_date"])
                if entry_quote is None:
                    continue
                _entry_rate_quote, _entry_pu_compra, entry_pu_venda, _entry_quote_date, _forced = entry_quote
                open_position["entry_pu_venda"] = entry_pu_venda

            new_weight = current_weight
            if (
                not bool(open_position["hit_16"])
                and pd.notna(z2412)
                and z2412 >= add_threshold_2412_mid
            ):
                new_weight += 1.0
                open_position["hit_16"] = True

            if (
                not bool(open_position["hit_20"])
                and pd.notna(z2412)
                and z2412 >= add_threshold_2412_high
            ):
                new_weight += 2.0
                open_position["hit_20"] = True

            current_weight = min(new_weight, 4.0)
            open_position["current_weight"] = current_weight

            mtm_return_pct = _pu_return_pct(
                float(open_position["entry_pu_venda"]),
                current_pu_compra,
            )
            mtm_score = base_notional * current_weight * (mtm_return_pct / 100.0)

            if mtm_score < float(open_position["worst_mtm_score"]):
                open_position["worst_mtm_score"] = mtm_score
                open_position["worst_mtm_date"] = dt
                open_position["weight_at_dd"] = current_weight

            should_exit = (
                pd.notna(z2412)
                and holding_days >= holding_minimo_dias
                and z2412 <= exit_threshold
            )
            is_last_row = i == len(df) - 1
            forced_exit = bool(is_last_available_quote)

            if should_exit or forced_exit or is_last_row:
                exit_date = quote_date if forced_exit else dt
                exit_rate = current_rate
                rate_move = entry_rate - exit_rate
                return_pct = _pu_return_pct(
                    float(open_position["entry_pu_venda"]),
                    current_pu_compra,
                )
                score = base_notional * current_weight * (return_pct / 100.0)

                carry_proxy_anual = (
                    base_notional
                    * float(open_position["weight_at_dd"])
                    * entry_rate
                    * carry_proxy_fraction
                )
                if carry_proxy_anual > 0:
                    anos_para_recuperar_dd = abs(float(open_position["worst_mtm_score"])) / carry_proxy_anual
                else:
                    anos_para_recuperar_dd = math.nan

                trades.append(
                    {
                        "bucket": bucket_name,
                        "entry_date": open_position["entry_date"],
                        "exit_date": exit_date,
                        "entry_rate": entry_rate,
                        "exit_rate": exit_rate,
                        "entry_z2412": float(open_position["entry_z2412"]),
                        "exit_z2412": float(z2412) if pd.notna(z2412) else None,
                        "entry_prazo": entry_prazo,
                        "entry_vencimento": open_position["entry_vencimento"],
                        "entry_pu_venda": float(open_position["entry_pu_venda"]),
                        "exit_pu_compra": float(current_pu_compra),
                        "entry_weight": float(open_position["entry_weight"]),
                        "max_weight": current_weight,
                        "holding_days": int(holding_days),
                        "rate_move": float(rate_move),
                        "return_pct": float(return_pct),
                        "score": float(score),
                        "max_drawdown_score": float(open_position["worst_mtm_score"]),
                        "max_drawdown_date": open_position["worst_mtm_date"],
                        "weight_at_dd": float(open_position["weight_at_dd"]),
                        "carry_proxy_anual": float(carry_proxy_anual),
                        "anos_para_recuperar_dd": float(anos_para_recuperar_dd) if not math.isnan(anos_para_recuperar_dd) else None,
                        "is_open": bool(is_last_row and not should_exit and not forced_exit),
                    }
                )
                open_position = None

        all_trades.extend(trades)
        trades_df = pd.DataFrame(trades)
        start_date = df["data"].min().strftime("%d/%m/%Y")
        end_date = df["data"].max().strftime("%d/%m/%Y")

        if trades_df.empty:
            summary_lines.append(
                f"{bucket_name}: entry_z={entry_threshold:.1f} | exit_z={exit_threshold:.1f} | "
                f"periodo={start_date}->{end_date} | trades=0"
            )
            continue

        closed_df = trades_df[~trades_df["is_open"].fillna(False)].copy()
        closed_score = float(closed_df["score"].sum()) if not closed_df.empty else float("nan")
        closed_win = win_rate_pct(closed_df["score"]) if not closed_df.empty else float("nan")

        summary_lines.append(
            f"{bucket_name}: entry_z={entry_threshold:.1f} | exit_z={exit_threshold:.1f} | "
            f"trades={len(trades_df)} | abertos={int(trades_df['is_open'].fillna(False).sum())} | "
            f"score_total={trades_df['score'].sum():.2f} | "
            f"score_fechado={closed_score:.2f} | "
            f"win_fechado={closed_win:.1f}% | "
            f"ret_pu_medio={safe_mean(trades_df['return_pct']):.2f}% | "
            f"pior_dd={trades_df['max_drawdown_score'].min():.2f} | "
            f"periodo={start_date}->{end_date}"
        )

        detail_lines.append(f"[{bucket_name}]")
        for _, trade in trades_df.iterrows():
            z_txt = "nan" if pd.isna(trade["entry_z2412"]) else f"{trade['entry_z2412']:.2f}"
            dd_date_txt = (
                trade["max_drawdown_date"].strftime("%d/%m/%Y")
                if pd.notna(trade["max_drawdown_date"])
                else "n/a"
            )
            venc_txt = trade["entry_vencimento"].strftime("%Y-%m-%d")
            detail_lines.append(
                f"{trade['entry_date'].strftime('%d/%m/%Y')} -> {trade['exit_date'].strftime('%d/%m/%Y')} | "
                f"status={'aberto' if bool(trade['is_open']) else 'fechado'} | "
                f"taxa_compra={trade['entry_rate']:.2f} | venc={venc_txt} | prazo={trade['entry_prazo']:.2f} | "
                f"pu_entrada={trade['entry_pu_venda']:.2f} | pu_saida={trade['exit_pu_compra']:.2f} | "
                f"z2412={z_txt} | peso_max={trade['max_weight']:.2f}x | "
                f"rate_move={trade['rate_move']:+.4f} | ret_pu={trade['return_pct']:.2f}% | "
                f"score={trade['score']:.2f} | maxDD_PU={trade['max_drawdown_score']:.2f} | "
                f"DD_day={dd_date_txt} | holding={trade['holding_days']}d"
            )
        detail_lines.append("")

    intro_lines = [
        "Regras do sistema:",
        "Universo: series media e longa de Tesouro IPCA+",
        "Media: prazo >= 8 e < 14 anos | entry_z=1.0 | exit_z=-1.6",
        "Grande: prazo >= 14 e <= 20 anos | entry_z=1.2 | exit_z=-2.0",
        "Sinal: z_2412 da serie da faixa, com min_periods=603",
        "Entrada: primeiro cruzamento do entry_z da faixa",
        "Peso inicial: 1.0x",
        f"Escalonamento: z_2412 >= {add_threshold_2412_mid:.1f} -> +1.0x; z_2412 >= {add_threshold_2412_high:.1f} -> +2.0x",
        "Peso maximo: 4.0x",
        f"Holding minimo: {holding_minimo_dias} dias",
        "Marcacao e venda: mesmo vencimento comprado na entrada",
        "PnL: compra pelo PU Venda Manha e venda pelo PU Compra Manha",
        "",
        "Stress econômico aproximado:",
        f"carry_proxy_anual = {carry_proxy_fraction:.0%} da taxa real de entrada",
        "anos_para_recuperar_dd = abs(maxDD_MTM) / carry_proxy_anual",
    ]

    if not all_trades:
        report = build_result(
            "REAL RATE STATE OF ART",
            section(intro_lines + ["", "Nenhum trade encontrado."]),
        )
        return render_result(report)

    trades_df = pd.DataFrame(all_trades)
    closed_df = trades_df[~trades_df["is_open"].fillna(False)].copy()

    win_rate = win_rate_pct(trades_df["score"])
    score_total = trades_df["score"].sum()
    score_fechado = closed_df["score"].sum() if not closed_df.empty else float("nan")
    win_fechado = win_rate_pct(closed_df["score"]) if not closed_df.empty else float("nan")

    resumo_lines = [
        f"trades={len(trades_df)}",
        f"trades_abertos={int(trades_df['is_open'].fillna(False).sum())}",
        f"trades_fechados={len(closed_df)}",
        f"score_total={score_total:.2f}",
        f"score_fechado={score_fechado:.2f}",
        f"win_total={win_rate:.1f}%",
        f"win_fechado={win_fechado:.1f}%",
        f"ret_pu_medio={safe_mean(trades_df['return_pct']):.2f}%",
        f"holding_medio={safe_mean(trades_df['holding_days']):.1f}d",
        f"pior_drawdown_pu={trades_df['max_drawdown_score'].min():.2f}",
    ]

    report = build_result(
        "REAL RATE STATE OF ART",
        section(intro_lines),
        section(resumo_lines, title="RESUMO"),
        section(summary_lines, title="RESUMO POR SERIE"),
        section(detail_lines, title="DETALHE DOS TRADES"),
    )
    return render_result(report)


def backtest_optimize_realrate_state_of_art_by_duration() -> str:
    """
    Otimiza z de entrada e saida do setup State of Art por faixa de prazo.
    """

    import pandas as pd

    entry_thresholds = _float_range(1.0, 3.0, 0.2)
    exit_thresholds = [-value for value in _float_range(1.0, 2.0, 0.2)]
    add_threshold_2412_mid = 1.6
    add_threshold_2412_high = 2.0
    holding_minimo_dias = 15
    base_notional = 100.0

    buckets = [
        ("Pequena", 5.0, 8.0, False),
        ("Media", 8.0, 14.0, False),
        ("Grande", 14.0, 20.0, True),
    ]

    raw_df = load_tesouro_ipca_frame()
    if "Tipo Titulo" in raw_df.columns:
        raw_df = raw_df[raw_df["Tipo Titulo"].astype(str).str.strip() == "Tesouro IPCA+"].copy()
    raw_df = _prepare_trade_price_columns(raw_df)

    quote_history_by_maturity = {
        maturity: group.sort_values("Data Base").reset_index(drop=True)
        for maturity, group in raw_df.groupby("Data Vencimento")
    }

    def quote_same_title_on_or_before(maturity: object, dt: object) -> tuple[float, float, float, object, bool] | None:
        history = quote_history_by_maturity.get(maturity)
        if history is None or history.empty:
            return None

        valid_quotes = history[history["Data Base"] <= dt]
        if valid_quotes.empty:
            return None

        quote = valid_quotes.iloc[-1]
        last_quote = history.iloc[-1]
        is_last_available_quote = (
            quote["Data Base"] == last_quote["Data Base"]
            and quote["Data Base"] < dt
        )
        return (
            float(quote["Taxa Compra Manha"]),
            float(quote["PU Compra Manha"]),
            float(quote["PU Venda Manha"]),
            quote["Data Base"],
            bool(is_last_available_quote),
        )

    intro_lines = [
        "Objetivo: achar o melhor par entry_z / exit_z para cada faixa de prazo.",
        "zscore: z_2412 com min_periods=603",
        "entry_z grid: " + ", ".join(f"{value:.1f}" for value in entry_thresholds),
        "exit_z grid: " + ", ".join(f"{value:.1f}" for value in exit_thresholds),
        f"Holding minimo: {holding_minimo_dias} dias",
        "Escalonamento fixo: z_2412 >= 1.6 -> +1.0x; z_2412 >= 2.0 -> +2.0x",
        "Peso maximo: 4.0x",
        "Criterio principal: score_total por PU real",
        "A marcacao e a venda usam o mesmo vencimento comprado na entrada.",
        "PnL: compra pelo PU Venda Manha e venda pelo PU Compra Manha.",
        "",
        "Faixas:",
        "Pequena: prazo >= 5 e < 8 anos",
        "Media: prazo >= 8 e < 14 anos",
        "Grande: prazo >= 14 e <= 20 anos",
    ]

    best_all_lines: list[str] = []
    best_closed_lines: list[str] = []
    top_all_lines: list[str] = []
    top_closed_lines: list[str] = []

    for bucket_name, min_prazo, max_prazo, include_max in buckets:
        df = build_daily_ipca_duration_bucket_series(
            raw_df,
            min_prazo_anos=min_prazo,
            max_prazo_anos=max_prazo,
            include_max=include_max,
        )
        df = (
            df.dropna(subset=["data", "taxa_media", "prazo_anos"])
            .sort_values("data")
            .reset_index(drop=True)
        )

        if df.empty:
            best_all_lines.append(f"{bucket_name}: sem dados na faixa")
            best_closed_lines.append(f"{bucket_name}: sem dados na faixa")
            continue

        df["z_2412"] = rolling_zscore(df["taxa_media"], window=2412, min_periods=603)

        dates = df["data"].tolist()
        rates = df["taxa_media"].astype(float).tolist()
        prazos = df["prazo_anos"].astype(float).tolist()
        vencimentos = df["data_vencimento"].tolist()
        zscores = df["z_2412"].tolist()

        bucket_results: list[dict[str, float | int]] = []

        for entry_threshold in entry_thresholds:
            for exit_threshold in exit_thresholds:
                trades: list[dict[str, float | int | object]] = []
                open_position: dict[str, object] | None = None
                prev_signal = False

                for i, z2412 in enumerate(zscores):
                    bucket_rate = rates[i]
                    prazo = prazos[i]
                    dt = dates[i]
                    vencimento = vencimentos[i]

                    signal = pd.notna(z2412) and z2412 >= entry_threshold
                    entry_event = bool(signal) and (not prev_signal)
                    prev_signal = bool(signal)

                    if entry_event and open_position is None:
                        open_position = {
                            "entry_idx": i,
                            "entry_date": dt,
                            "entry_rate": bucket_rate,
                            "entry_z2412": float(z2412),
                            "entry_prazo": prazo,
                            "entry_vencimento": vencimento,
                            "current_weight": 1.0,
                            "worst_mtm_score": 0.0,
                            "hit_16": False,
                            "hit_20": False,
                        }

                    if open_position is None:
                        continue

                    entry_idx = int(open_position["entry_idx"])
                    entry_rate = float(open_position["entry_rate"])
                    entry_prazo = float(open_position["entry_prazo"])
                    entry_vencimento = open_position["entry_vencimento"]
                    current_quote = quote_same_title_on_or_before(entry_vencimento, dt)
                    if current_quote is None:
                        continue

                    current_rate, current_pu_compra, _current_pu_venda, quote_date, is_last_available_quote = current_quote
                    holding_days = i - entry_idx
                    current_weight = float(open_position["current_weight"])

                    if "entry_pu_venda" not in open_position:
                        entry_quote = quote_same_title_on_or_before(entry_vencimento, open_position["entry_date"])
                        if entry_quote is None:
                            continue
                        _entry_rate_quote, _entry_pu_compra, entry_pu_venda, _entry_quote_date, _forced = entry_quote
                        open_position["entry_pu_venda"] = entry_pu_venda

                    new_weight = current_weight
                    if (
                        not bool(open_position["hit_16"])
                        and pd.notna(z2412)
                        and z2412 >= add_threshold_2412_mid
                    ):
                        new_weight += 1.0
                        open_position["hit_16"] = True

                    if (
                        not bool(open_position["hit_20"])
                        and pd.notna(z2412)
                        and z2412 >= add_threshold_2412_high
                    ):
                        new_weight += 2.0
                        open_position["hit_20"] = True

                    current_weight = min(new_weight, 4.0)
                    open_position["current_weight"] = current_weight

                    mtm_return_pct = _pu_return_pct(
                        float(open_position["entry_pu_venda"]),
                        current_pu_compra,
                    )
                    mtm_score = base_notional * current_weight * (mtm_return_pct / 100.0)
                    if mtm_score < float(open_position["worst_mtm_score"]):
                        open_position["worst_mtm_score"] = mtm_score

                    should_exit = (
                        pd.notna(z2412)
                        and holding_days >= holding_minimo_dias
                        and z2412 <= exit_threshold
                    )
                    is_last_row = i == len(df) - 1
                    forced_exit = bool(is_last_available_quote)

                    if should_exit or forced_exit or is_last_row:
                        exit_date = quote_date if forced_exit else dt
                        return_pct = _pu_return_pct(
                            float(open_position["entry_pu_venda"]),
                            current_pu_compra,
                        )
                        score = base_notional * current_weight * (return_pct / 100.0)

                        trades.append(
                            {
                                "entry_date": open_position["entry_date"],
                                "exit_date": exit_date,
                                "entry_z2412": float(open_position["entry_z2412"]),
                                "entry_prazo": entry_prazo,
                                "entry_vencimento": open_position["entry_vencimento"],
                                "entry_pu_venda": float(open_position["entry_pu_venda"]),
                                "exit_pu_compra": float(current_pu_compra),
                                "max_weight": current_weight,
                                "holding_days": int(holding_days),
                                "rate_move": float(entry_rate - current_rate),
                                "return_pct": float(return_pct),
                                "score": float(score),
                                "max_drawdown_score": float(open_position["worst_mtm_score"]),
                                "is_open": bool(is_last_row and not should_exit and not forced_exit),
                            }
                        )
                        open_position = None

                trades_df = pd.DataFrame(trades)
                if trades_df.empty:
                    bucket_results.append(
                        {
                            "entry_z": entry_threshold,
                            "exit_z": exit_threshold,
                            "trades": 0,
                            "open_trades": 0,
                            "score_total": float("nan"),
                            "score_mean": float("nan"),
                            "avg_return": float("nan"),
                            "median_return": float("nan"),
                            "win_rate": float("nan"),
                            "worst_dd": float("nan"),
                            "holding_mean": float("nan"),
                            "closed_trades": 0,
                            "closed_score_total": float("nan"),
                            "closed_avg_return": float("nan"),
                            "closed_median_return": float("nan"),
                            "closed_win_rate": float("nan"),
                            "closed_worst_dd": float("nan"),
                            "closed_holding_mean": float("nan"),
                        }
                    )
                    continue

                closed_trades_df = trades_df[~trades_df["is_open"].fillna(False)].copy()
                if closed_trades_df.empty:
                    closed_summary = {
                        "closed_trades": 0,
                        "closed_score_total": float("nan"),
                        "closed_avg_return": float("nan"),
                        "closed_median_return": float("nan"),
                        "closed_win_rate": float("nan"),
                        "closed_worst_dd": float("nan"),
                        "closed_holding_mean": float("nan"),
                    }
                else:
                    closed_summary = {
                        "closed_trades": int(len(closed_trades_df)),
                        "closed_score_total": float(closed_trades_df["score"].sum()),
                        "closed_avg_return": safe_mean(closed_trades_df["return_pct"]),
                        "closed_median_return": safe_median(closed_trades_df["return_pct"]),
                        "closed_win_rate": win_rate_pct(closed_trades_df["score"]),
                        "closed_worst_dd": float(closed_trades_df["max_drawdown_score"].min()),
                        "closed_holding_mean": safe_mean(closed_trades_df["holding_days"]),
                    }

                bucket_results.append(
                    {
                        "entry_z": entry_threshold,
                        "exit_z": exit_threshold,
                        "trades": int(len(trades_df)),
                        "open_trades": int(trades_df["is_open"].fillna(False).sum()),
                        "score_total": float(trades_df["score"].sum()),
                        "score_mean": safe_mean(trades_df["score"]),
                        "avg_return": safe_mean(trades_df["return_pct"]),
                        "median_return": safe_median(trades_df["return_pct"]),
                        "win_rate": win_rate_pct(trades_df["score"]),
                        "worst_dd": float(trades_df["max_drawdown_score"].min()),
                        "holding_mean": safe_mean(trades_df["holding_days"]),
                        **closed_summary,
                    }
                )

        valid_results = [
            row
            for row in bucket_results
            if not pd.isna(row["score_total"]) and int(row["trades"]) > 0
        ]

        start_date = df["data"].min().strftime("%d/%m/%Y")
        end_date = df["data"].max().strftime("%d/%m/%Y")

        if not valid_results:
            best_all_lines.append(
                f"{bucket_name}: nenhum resultado valido | "
                f"linhas={len(df)} | periodo={start_date}->{end_date}"
            )
            best_closed_lines.append(
                f"{bucket_name}: nenhum resultado valido | "
                f"linhas={len(df)} | periodo={start_date}->{end_date}"
            )
            continue

        sorted_all_results = sorted(
            valid_results,
            key=lambda row: (
                float(row["score_total"]),
                float(row["avg_return"]),
                float(row["win_rate"]),
            ),
            reverse=True,
        )
        best_all = sorted_all_results[0]

        best_all_lines.append(
            f"{bucket_name}: entry_z={best_all['entry_z']:.1f} | exit_z={best_all['exit_z']:.1f} | "
            f"score_total={best_all['score_total']:.2f} | trades={best_all['trades']} | "
            f"abertos={best_all['open_trades']} | win={best_all['win_rate']:.1f}% | "
            f"ret_pu_medio={best_all['avg_return']:.2f}% | ret_pu_mediana={best_all['median_return']:.2f}% | "
            f"pior_dd={best_all['worst_dd']:.2f} | holding={best_all['holding_mean']:.1f}d | "
            f"periodo={start_date}->{end_date}"
        )

        valid_closed_results = [
            row
            for row in valid_results
            if not pd.isna(row["closed_score_total"]) and int(row["closed_trades"]) > 0
        ]

        if valid_closed_results:
            sorted_closed_results = sorted(
                valid_closed_results,
                key=lambda row: (
                    float(row["closed_score_total"]),
                    float(row["closed_avg_return"]),
                    float(row["closed_win_rate"]),
                ),
                reverse=True,
            )
            best_closed = sorted_closed_results[0]
            best_closed_lines.append(
                f"{bucket_name}: entry_z={best_closed['entry_z']:.1f} | exit_z={best_closed['exit_z']:.1f} | "
                f"score_total_fechado={best_closed['closed_score_total']:.2f} | "
                f"trades_fechados={best_closed['closed_trades']} | "
                f"win={best_closed['closed_win_rate']:.1f}% | "
                f"ret_pu_medio={best_closed['closed_avg_return']:.2f}% | "
                f"ret_pu_mediana={best_closed['closed_median_return']:.2f}% | "
                f"pior_dd={best_closed['closed_worst_dd']:.2f} | "
                f"holding={best_closed['closed_holding_mean']:.1f}d | "
                f"periodo={start_date}->{end_date}"
            )
        else:
            sorted_closed_results = []
            best_closed_lines.append(
                f"{bucket_name}: nenhum trade fechado | periodo={start_date}->{end_date}"
            )

        top_all_lines.append(f"[{bucket_name}]")
        for idx, row in enumerate(sorted_all_results[:10], start=1):
            top_all_lines.append(
                f"{idx:02d}. entry_z={row['entry_z']:.1f} | exit_z={row['exit_z']:.1f} | "
                f"score_total={row['score_total']:.2f} | trades={row['trades']} | "
                f"abertos={row['open_trades']} | win={row['win_rate']:.1f}% | "
                f"ret_pu_medio={row['avg_return']:.2f}% | pior_dd={row['worst_dd']:.2f}"
            )
        top_all_lines.append("")

        top_closed_lines.append(f"[{bucket_name}]")
        if sorted_closed_results:
            for idx, row in enumerate(sorted_closed_results[:10], start=1):
                top_closed_lines.append(
                    f"{idx:02d}. entry_z={row['entry_z']:.1f} | exit_z={row['exit_z']:.1f} | "
                    f"score_total_fechado={row['closed_score_total']:.2f} | "
                    f"trades_fechados={row['closed_trades']} | "
                    f"win={row['closed_win_rate']:.1f}% | "
                    f"ret_pu_medio={row['closed_avg_return']:.2f}% | "
                    f"pior_dd={row['closed_worst_dd']:.2f}"
                )
        else:
            top_closed_lines.append("Nenhum trade fechado.")
        top_closed_lines.append("")

    report = build_result(
        "OTIMIZACAO REAL RATE STATE OF ART POR DURACAO",
        section(intro_lines),
        section(best_all_lines, title="MELHOR PARAMETRO INCLUINDO ABERTOS"),
        section(best_closed_lines, title="MELHOR PARAMETRO APENAS FECHADOS"),
        section(top_all_lines, title="TOP 10 INCLUINDO ABERTOS"),
        section(top_closed_lines, title="TOP 10 APENAS FECHADOS"),
    )
    return render_result(report)


__all__ = [
    "backtest_optimize_entry_threshold_fine",
    "backtest_optimize_realrate_state_of_art_by_duration",
    "backtest_optimize_zscore_grid",
    "backtest_realrate_state_of_art",
]
