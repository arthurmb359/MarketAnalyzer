from __future__ import annotations

from matplotlib import lines
import pandas as pd

from macro.macro_features import (
    build_daily_macro_frame,
    add_macro_features,
    add_macro_regime_score,
)


def _load_macro_frame() -> pd.DataFrame:
    df = build_daily_macro_frame()
    df = add_macro_features(df)
    df = add_macro_regime_score(df)
    
    # manter apenas período moderno 2000 em diante, para evitar distorções por dados antigos e esparsos
    df = df[df["data"] >= pd.Timestamp("2000-01-01")].copy()
    df = df.sort_values("data").reset_index(drop=True)
    return df


def _find_stress_periods(df: pd.DataFrame) -> list[dict]:
    """
    Detecta blocos contínuos de regime 'stress' usando macro_regime_label.
    """
    work = df.dropna(subset=["macro_regime_label"]).copy()
    work = work.sort_values("data").reset_index(drop=True)

    is_stress = work["macro_regime_label"] == "stress"
    prev = is_stress.shift(1, fill_value=False)

    starts = work.index[is_stress & (~prev)].tolist()

    periods: list[dict] = []

    for start_idx in starts:
        start_date = work.loc[start_idx, "data"]

        end_idx = start_idx
        while (
            end_idx + 1 < len(work)
            and work.loc[end_idx + 1, "macro_regime_label"] == "stress"
        ):
            end_idx += 1

        end_date = work.loc[end_idx, "data"]
        duration_days = (end_date - start_date).days + 1

        periods.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_date": start_date,
            "end_date": end_date,
            "duration_days": duration_days,
        })

    return periods

def _build_research_frame(duration_minima: float = 0.0):
    from macro.macro_features import (
        build_daily_macro_frame,
        add_macro_features,
        add_macro_regime_score,
    )
    from backtest.ipca_series import load_ipca_long_series
    import pandas as pd

    price_df = load_ipca_long_series(duration_minima=duration_minima)
    price_df = price_df.sort_values("data").reset_index(drop=True)

    macro_df = build_daily_macro_frame()
    macro_df = add_macro_features(macro_df)
    macro_df = add_macro_regime_score(macro_df)
    macro_df = macro_df.sort_values("data").reset_index(drop=True)

    # merge temporal robusto
    df = pd.merge_asof(
        price_df,
        macro_df[["data", "macro_regime_label", "macro_score_smooth"]],
        on="data",
        direction="backward",
    )

    # normaliza faltantes
    df["macro_regime_label"] = df["macro_regime_label"].fillna("indefinido")

    return df

def _classify_macro_regime_from_thresholds(
    df,
    low_thr: float,
    high_thr: float,
):
    work = df.copy()

    def classify(x):
        if pd.isna(x):
            return "indefinido"
        if x < low_thr:
            return "normal"
        if x < high_thr:
            return "alerta"
        return "stress"

    work["macro_regime_test"] = work["macro_score_smooth"].apply(classify)
    return work


def _detect_regime_starts(
    df,
    regime_col: str = "macro_regime_test",
):
    work = df.copy().sort_values("data").reset_index(drop=True)

    prev = work[regime_col].shift(1)

    starts = work[
        (work[regime_col].isin(["alerta", "stress"]))
        & (work[regime_col] != prev)
    ][["data", regime_col]].copy()

    starts = starts.rename(columns={regime_col: "regime"})
    return starts.reset_index(drop=True)


def _compute_macro_threshold_score(
    df,
    low_thr: float,
    high_thr: float,
    known_event_dates,
):
    import numpy as np

    work = _classify_macro_regime_from_thresholds(df, low_thr, high_thr)
    starts = _detect_regime_starts(work, regime_col="macro_regime_test")

    score = 0.0
    matched_events = 0
    stress_hits = 0
    alert_hits = 0

    known_dates = pd.to_datetime(known_event_dates)

    # 1) recompensa por encaixe com eventos históricos
    for event_date in known_dates:
        if starts.empty:
            score -= 6.0
            continue

        starts["delta_days"] = (starts["data"] - event_date).dt.days
        nearby = starts[starts["delta_days"].between(-60, 60)].copy()

        if nearby.empty:
            # perdeu um evento importante
            score -= 6.0
            continue

        matched_events += 1

        stress_nearby = nearby[nearby["regime"] == "stress"]
        alert_nearby = nearby[nearby["regime"] == "alerta"]

        if not stress_nearby.empty:
            best = stress_nearby.iloc[(stress_nearby["delta_days"].abs()).argmin()]
            score += 12.0
            stress_hits += 1
        elif not alert_nearby.empty:
            best = alert_nearby.iloc[(alert_nearby["delta_days"].abs()).argmin()]
            score += 7.0
            alert_hits += 1
        else:
            best = nearby.iloc[(nearby["delta_days"].abs()).argmin()]
            score += 5.0

        # bônus se antecipou o evento
        if best["delta_days"] < 0:
            score += 3.0

        # bônus extra se antecipou por até 30 dias
        if -30 <= best["delta_days"] < 0:
            score += 1.5

        # penalidade se veio atrasado
        if best["delta_days"] > 30:
            score -= 3.0

    # 2) penalidade por excesso de alarmes sem evento próximo
    false_alerts = 0
    false_stress = 0

    for _, row in starts.iterrows():
        deltas = (known_dates - row["data"]).days

        if not any(abs(d) <= 60 for d in deltas):
            if row["regime"] == "stress":
                false_stress += 1
                score -= 3.0
            elif row["regime"] == "alerta":
                false_alerts += 1
                score -= 1.5

    # 3) penalidade por regimes muito curtos
    durations = []
    short_alerts = 0
    short_stress = 0

    work = work.sort_values("data").reset_index(drop=True)
    current_regime = None
    start_date = None

    for i in range(len(work)):
        regime = work.loc[i, "macro_regime_test"]

        if regime != current_regime:
            if current_regime in ["alerta", "stress"] and start_date is not None:
                end_date = work.loc[i - 1, "data"]
                duration = (end_date - start_date).days + 1
                durations.append(duration)

                if duration < 20:
                    if current_regime == "alerta":
                        short_alerts += 1
                        score -= 2.0
                    elif current_regime == "stress":
                        short_stress += 1
                        score -= 4.0

                elif duration < 45:
                    if current_regime == "alerta":
                        score -= 0.8
                    elif current_regime == "stress":
                        score -= 1.5

            if regime in ["alerta", "stress"]:
                start_date = work.loc[i, "data"]
            else:
                start_date = None

            current_regime = regime

    if current_regime in ["alerta", "stress"] and start_date is not None:
        end_date = work.loc[len(work) - 1, "data"]
        duration = (end_date - start_date).days + 1
        durations.append(duration)

        if duration < 20:
            if current_regime == "alerta":
                short_alerts += 1
                score -= 2.0
            elif current_regime == "stress":
                short_stress += 1
                score -= 4.0

        elif duration < 45:
            if current_regime == "alerta":
                score -= 0.8
            elif current_regime == "stress":
                score -= 1.5

    avg_duration = float(np.mean(durations)) if durations else float("nan")

    # 4) penalidade por troca excessiva de regime
    transitions = (work["macro_regime_test"] != work["macro_regime_test"].shift(1)).sum()

    # penalidade mais dura
    score -= max(0, transitions - 20) * 0.7

    # 5) bônus por duração média saudável
    if not np.isnan(avg_duration):
        if 90 <= avg_duration <= 250:
            score += 4.0
        elif 60 <= avg_duration < 90 or 250 < avg_duration <= 320:
            score += 1.5
        elif avg_duration < 45:
            score -= 3.0
        elif avg_duration > 400:
            score -= 2.0

    return {
        "low_thr": low_thr,
        "high_thr": high_thr,
        "score": score,
        "matched_events": matched_events,
        "stress_hits": stress_hits,
        "alert_hits": alert_hits,
        "false_alerts": false_alerts,
        "false_stress": false_stress,
        "short_alerts": short_alerts,
        "short_stress": short_stress,
        "avg_duration": avg_duration,
        "transitions": int(transitions),
    }


def backtest_optimize_macro_regime_thresholds() -> str:
    df = _load_macro_frame()
    df = df[df["data"] >= pd.Timestamp("2000-01-01")].copy()
    df = df.sort_values("data").reset_index(drop=True)

    known_event_dates = [
        "2002-07-01",
        "2008-09-15",
        "2013-05-22",
        "2015-07-01",
        "2018-05-20",
        "2020-03-15",
        "2021-10-01",
        "2024-11-01",
    ]

    low_grid = [0.60, 0.70, 0.80, 0.90, 1.00]
    high_grid = [1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80]

    results = []

    for low_thr in low_grid:
        for high_thr in high_grid:
            if high_thr <= low_thr:
                continue

            result = _compute_macro_threshold_score(
                df=df,
                low_thr=low_thr,
                high_thr=high_thr,
                known_event_dates=known_event_dates,
            )
            results.append(result)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    lines: list[str] = []
    lines.append("=== OTIMIZAÇÃO DOS THRESHOLDS DO MACRO REGIME ===")
    lines.append("")
    lines.append("Objetivo: encaixar os regimes macro nas datas históricas conhecidas")
    lines.append("")
    lines.append("Eventos históricos usados:")
    for d in known_event_dates:
        lines.append(f"- {d}")
    lines.append("")

    lines.append("=== TOP 10 COMBINAÇÕES ===")
    for r in results[:10]:
        lines.append(
            f"low={r['low_thr']:.2f} | "
            f"high={r['high_thr']:.2f} | "
            f"score={r['score']:.2f} | "
            f"matched={r['matched_events']} | "
            f"stress_hits={r['stress_hits']} | "
            f"alert_hits={r['alert_hits']} | "
            f"false_alerts={r['false_alerts']} | "
            f"false_stress={r['false_stress']} | "
            f"short_alerts={r['short_alerts']} | "
            f"short_stress={r['short_stress']} | "
            f"avg_duration={r['avg_duration']:.1f} | "
            f"transitions={r['transitions']}"
        )

    best = results[0]

    lines.append("")
    lines.append("=== MELHOR COMBINAÇÃO ===")
    lines.append(f"normal < {best['low_thr']:.2f}")
    lines.append(f"alerta >= {best['low_thr']:.2f} e < {best['high_thr']:.2f}")
    lines.append(f"stress >= {best['high_thr']:.2f}")
    lines.append(f"score total = {best['score']:.2f}")
    lines.append(f"eventos encaixados = {best['matched_events']}")
    lines.append(f"stress hits = {best['stress_hits']}")
    lines.append(f"alert hits = {best['alert_hits']}")
    lines.append(f"false alerts = {best['false_alerts']}")
    lines.append(f"false stress = {best['false_stress']}")
    lines.append(f"duração média dos regimes = {best['avg_duration']:.1f} dias")
    lines.append(f"número de transições = {best['transitions']}")
    lines.append(f"short alerts = {best['short_alerts']}")
    lines.append(f"short stress = {best['short_stress']}")

    return "\n".join(lines)

def _run_real_rate_scaled_once(
    z_entry_threshold: float,
    duration_minima: float,
    exit_threshold: float = -2.0,
    tiers: list[tuple[float, float]] | None = None,
) -> dict:
    import numpy as np
    import pandas as pd

    df = _build_research_frame(duration_minima=duration_minima)
    df = df.sort_values("data").reset_index(drop=True)

    zscore_col = "zscore_rolling_5a"

    if tiers is None:
        tiers = [
            (z_entry_threshold, 1.0),
            (z_entry_threshold + 0.5, 1.5),
            (z_entry_threshold + 1.0, 2.0),
            (z_entry_threshold + 1.5, 3.0),
        ]

    completed_trades = []

    in_regime = False
    activated_tiers = set()
    tranches = []

    for i in range(len(df)):
        z = df.loc[i, zscore_col]

        if pd.isna(z):
            continue

        triggered_any = False
        for threshold, weight in tiers:
            if z >= threshold and threshold not in activated_tiers:
                activated_tiers.add(threshold)
                triggered_any = True
                entry_macro = df.loc[i, "macro_regime_label"]

                if entry_macro is None or pd.isna(entry_macro):
                    entry_macro = "indefinido"
                else:
                    entry_macro = str(entry_macro).strip().lower()

                tranches.append({
                    "entry_idx": i,
                    "entry_date": df.loc[i, "data"],
                    "entry_rate": df.loc[i, "taxa_media"],
                    "entry_prazo_anos": df.loc[i, "prazo_anos"],
                    "entry_macro_regime": entry_macro,
                    "weight": weight,
                    "threshold": threshold,
                })

        if triggered_any:
            in_regime = True

        if in_regime and z <= exit_threshold and len(tranches) > 0:
            exit_idx = i
            exit_date = df.loc[exit_idx, "data"]
            exit_rate = df.loc[exit_idx, "taxa_media"]

            tranche_returns = []
            tranche_mtm = []
            tranche_carry = []
            tranche_weights = []
            holding_bars_all = []
            tranche_durations = []

            for tr in tranches:
                holding_bars = exit_idx - tr["entry_idx"]
                duration_model = _approx_duration_from_prazo(tr["entry_prazo_anos"])

                total_pct, mtm_pct, carry_pct = _total_real_return_components(
                    entry_rate_pct=tr["entry_rate"],
                    exit_rate_pct=exit_rate,
                    duration=duration_model,
                    holding_bars=holding_bars,
                    bars_per_year=252,
                )

                tranche_returns.append(total_pct)
                tranche_mtm.append(mtm_pct)
                tranche_carry.append(carry_pct)
                tranche_weights.append(tr["weight"])
                holding_bars_all.append(holding_bars)
                tranche_durations.append(duration_model)

            weights = np.array(tranche_weights, dtype=float)
            total_returns = np.array(tranche_returns, dtype=float)
            mtm_returns = np.array(tranche_mtm, dtype=float)
            carry_returns = np.array(tranche_carry, dtype=float)
            holding_bars_all = np.array(holding_bars_all, dtype=float)
            tranche_durations = np.array(tranche_durations, dtype=float)
            absolute_pnl_units = np.sum(weights * total_returns)

            completed_trades.append({
                "entry_date": tranches[0]["entry_date"],
                "exit_date": exit_date,
                "entry_macro_regime": tranches[0]["entry_macro_regime"],
                "num_tranches": len(tranches),
                "max_tier": max(t["threshold"] for t in tranches),
                "total_weight": weights.sum(),
                "return_pct": np.average(total_returns, weights=weights),
                "absolute_pnl_units": absolute_pnl_units,
                "mtm_pct": np.average(mtm_returns, weights=weights),
                "carry_pct": np.average(carry_returns, weights=weights),
                "holding_bars": np.average(holding_bars_all, weights=weights),
                "avg_duration_model": np.average(tranche_durations, weights=weights),
            })

            in_regime = False
            activated_tiers = set()
            tranches = []

    discarded = 1 if len(tranches) > 0 else 0

    if len(completed_trades) == 0:
        return {
            "z_entry_threshold": z_entry_threshold,
            "duration_minima": duration_minima,
            "trades": 0,
            "discarded": discarded,
            "avg_return": float("nan"),
            "median_return": float("nan"),
            "win_rate": float("nan"),
            "avg_absolute_pnl_units": float("nan"),
            "median_absolute_pnl_units": float("nan"),
            "total_absolute_pnl_units": float("nan"),
            "avg_mtm": float("nan"),
            "avg_carry": float("nan"),
            "avg_holding": float("nan"),
            "avg_weight": float("nan"),
            "avg_duration_model": float("nan"),
            "completed_trades": [],
        }

    returns = np.array([t["return_pct"] for t in completed_trades], dtype=float)
    absolute_pnls = np.array([t["absolute_pnl_units"] for t in completed_trades], dtype=float)
    mtm = np.array([t["mtm_pct"] for t in completed_trades], dtype=float)
    carry = np.array([t["carry_pct"] for t in completed_trades], dtype=float)
    holding = np.array([t["holding_bars"] for t in completed_trades], dtype=float)
    total_weights = np.array([t["total_weight"] for t in completed_trades], dtype=float)
    avg_durations = np.array([t["avg_duration_model"] for t in completed_trades], dtype=float)

    return {
        "z_entry_threshold": z_entry_threshold,
        "duration_minima": duration_minima,
        "trades": len(completed_trades),
        "discarded": discarded,
        "avg_return": returns.mean(),
        "median_return": np.median(returns),
        "win_rate": (returns > 0).mean() * 100,
        "avg_absolute_pnl_units": absolute_pnls.mean(),
        "median_absolute_pnl_units": np.median(absolute_pnls),
        "total_absolute_pnl_units": absolute_pnls.sum(),
        "avg_mtm": mtm.mean(),
        "avg_carry": carry.mean(),
        "avg_holding": holding.mean(),
        "avg_weight": total_weights.mean(),
        "avg_duration_model": avg_durations.mean(),
        "completed_trades": completed_trades,
    }

def backtest_optimize_entry_threshold_fine() -> str:
    import math
    import numpy as np

    # grid fino
    z_values = [1.15, 1.2, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.0]
    duration_minima = 15
    exit_threshold = -2.0

    lines: list[str] = []
    lines.append("=== BACKTEST: OTIMIZAÇÃO FINA DO THRESHOLD DE ENTRADA ===")
    lines.append("")
    lines.append(
        "z_entry_threshold grid: "
        + ", ".join(f"{z:.2f}" for z in z_values)
    )
    lines.append(f"duration_minima fixa: {duration_minima}")
    lines.append(f"saída fixa: zscore_rolling_5a <= {exit_threshold}")
    lines.append("critério principal: ganho absoluto total")
    lines.append("")

    results = []

    for z in z_values:
        # estratégia: entra em z e depois escala a cada +0.5
        tiers = [
            (z, 1.0),
            (z + 0.5, 1.5),
            (z + 1.0, 2.0),
            (z + 1.5, 3.0),
        ]

        try:
            result = _run_real_rate_scaled_once(
                z_entry_threshold=z,
                duration_minima=duration_minima,
                exit_threshold=exit_threshold,
                tiers=tiers,
            )
            results.append(result)
        except Exception as exc:
            results.append({
                "z_entry_threshold": z,
                "duration_minima": duration_minima,
                "trades": 0,
                "discarded": 0,
                "avg_return": float("nan"),
                "median_return": float("nan"),
                "win_rate": float("nan"),
                "avg_absolute_pnl_units": float("nan"),
                "median_absolute_pnl_units": float("nan"),
                "total_absolute_pnl_units": float("nan"),
                "avg_mtm": float("nan"),
                "avg_carry": float("nan"),
                "avg_holding": float("nan"),
                "avg_weight": float("nan"),
                "avg_duration_model": float("nan"),
                "error": str(exc),
            })

    valid_results = [
        r for r in results
        if not math.isnan(r["total_absolute_pnl_units"])
    ]

    lines.append("=== RESULTADOS DO GRID ===")
    for r in results:
        lines.append(
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
            lines.append(f"  erro: {r['error']}")

    lines.append("")

    if not valid_results:
        lines.append("Nenhum resultado válido encontrado.")
        return "\n".join(lines)

    best_abs_total = max(valid_results, key=lambda x: x["total_absolute_pnl_units"])
    best_abs_avg = max(valid_results, key=lambda x: x["avg_absolute_pnl_units"])
    best_return_avg = max(valid_results, key=lambda x: x["avg_return"])
    best_median = max(valid_results, key=lambda x: x["median_return"])

    lines.append("=== MELHOR POR GANHO ABSOLUTO TOTAL ===")
    lines.append(
        f"z={best_abs_total['z_entry_threshold']:.1f} | "
        f"abs_total={best_abs_total['total_absolute_pnl_units']:.2f} | "
        f"abs_médio={best_abs_total['avg_absolute_pnl_units']:.2f} | "
        f"ret_médio={best_abs_total['avg_return']:.2f}% | "
        f"mediana={best_abs_total['median_return']:.2f}% | "
        f"win={best_abs_total['win_rate']:.1f}% | "
        f"trades={best_abs_total['trades']}"
    )
    lines.append("")

    lines.append("=== MELHOR POR GANHO ABSOLUTO MÉDIO ===")
    lines.append(
        f"z={best_abs_avg['z_entry_threshold']:.1f} | "
        f"abs_médio={best_abs_avg['avg_absolute_pnl_units']:.2f} | "
        f"abs_total={best_abs_avg['total_absolute_pnl_units']:.2f} | "
        f"ret_médio={best_abs_avg['avg_return']:.2f}% | "
        f"mediana={best_abs_avg['median_return']:.2f}% | "
        f"win={best_abs_avg['win_rate']:.1f}% | "
        f"trades={best_abs_avg['trades']}"
    )
    lines.append("")

    lines.append("=== MELHOR POR RETORNO MÉDIO ===")
    lines.append(
        f"z={best_return_avg['z_entry_threshold']:.1f} | "
        f"ret_médio={best_return_avg['avg_return']:.2f}% | "
        f"mediana={best_return_avg['median_return']:.2f}% | "
        f"abs_total={best_return_avg['total_absolute_pnl_units']:.2f} | "
        f"win={best_return_avg['win_rate']:.1f}% | "
        f"trades={best_return_avg['trades']}"
    )
    lines.append("")

    lines.append("=== MELHOR POR MEDIANA ===")
    lines.append(
        f"z={best_median['z_entry_threshold']:.1f} | "
        f"mediana={best_median['median_return']:.2f}% | "
        f"ret_médio={best_median['avg_return']:.2f}% | "
        f"abs_total={best_median['total_absolute_pnl_units']:.2f} | "
        f"win={best_median['win_rate']:.1f}% | "
        f"trades={best_median['trades']}"
    )

    return "\n".join(lines)

def backtest_macro_entry_impact() -> str:
    import numpy as np
    from collections import defaultdict

    z_entry = 1.5
    duration_minima = 15
    exit_threshold = -2.0

    tiers = [
        (1.5, 1.0),
        (2.0, 1.5),
        (2.5, 2.0),
        (3.0, 3.0),
    ]

    result = _run_real_rate_scaled_once(
        z_entry_threshold=z_entry,
        duration_minima=duration_minima,
        exit_threshold=exit_threshold,
        tiers=tiers,
    )

    trades = result.get("completed_trades", [])

    print("DEBUG regimes nos trades:")
    for t in trades:
        print(
            t.get("entry_date"),
            t.get("entry_macro_regime"),
            t.get("return_pct")
        )

    lines = []
    lines.append("=== IMPACTO DO REGIME MACRO NA ENTRADA ===")
    lines.append("")
    lines.append(f"Entrada: z >= {z_entry}")
    lines.append(f"Saída: z <= {exit_threshold}")
    lines.append(f"duration_minima: {duration_minima}")
    lines.append("Escalonamento: 1.0 / 1.5 / 2.0 / 3.0")
    lines.append("")

    if not trades:
        lines.append("Nenhum trade encontrado.")
        return "\n".join(lines)

    grouped = defaultdict(list)

    for t in trades:
        regime = t.get("entry_macro_regime", "unknown")
        grouped[regime].append(t)

    lines.append("=== RESULTADOS POR REGIME ===")

    total_abs = 0

    for regime in ["normal", "alerta", "stress"]:

        g = grouped.get(regime, [])

        if not g:
            lines.append(f"\n[{regime}] sem trades")
            continue

        returns = np.array([t["return_pct"] for t in g])
        abs_pnl = np.array([t["absolute_pnl_units"] for t in g])
        mtm = np.array([t["mtm_pct"] for t in g])
        carry = np.array([t["carry_pct"] for t in g])
        holding = np.array([t["holding_bars"] for t in g])
        weights = np.array([t["total_weight"] for t in g])

        total_abs += abs_pnl.sum()

        lines.append(f"\n[{regime}]")
        lines.append(f"Trades: {len(g)}")
        lines.append(f"Ganho absoluto total: {abs_pnl.sum():.2f}")
        lines.append(f"Ganho absoluto médio: {abs_pnl.mean():.2f}")
        lines.append(f"Retorno médio: {returns.mean():.2f}%")
        lines.append(f"Mediana: {np.median(returns):.2f}%")
        lines.append(f"Win rate: {(returns > 0).mean()*100:.1f}%")
        lines.append(f"MTM médio: {mtm.mean():.2f}%")
        lines.append(f"Carry médio: {carry.mean():.2f}%")
        lines.append(f"Holding médio: {holding.mean():.1f} dias")
        lines.append(f"Peso médio: {weights.mean():.2f}x")

    lines.append("")
    lines.append("=== PARTICIPAÇÃO NO PnL TOTAL ===")

    for regime in ["normal", "alerta", "stress"]:

        g = grouped.get(regime, [])

        if not g:
            continue

        abs_pnl = np.array([t["absolute_pnl_units"] for t in g])
        share = abs_pnl.sum() / total_abs * 100

        lines.append(f"{regime}: {share:.1f}% do lucro total")

    return "\n".join(lines)


def _approx_duration_from_prazo(prazo_anos: float) -> float:
    """
    Proxy simples de duration a partir do prazo remanescente.

    Nesta versão inicial usamos a própria maturidade remanescente como proxy.
    Não é a modified duration exata, mas já é melhor que duration fixa.
    """
    if pd.isna(prazo_anos):
        return 0.0
    return max(0.0, float(prazo_anos))

def _mark_to_market_return_decimal(
    entry_rate_pct: float,
    exit_rate_pct: float,
    duration: float,
) -> float:
    """
    Retorno aproximado de marcação a mercado em decimal.
    Ex:
    delta de taxa = -1.00 p.p., duration=10
    => retorno ≈ +10% = 0.10
    """
    delta_rate_pp = exit_rate_pct - entry_rate_pct
    return -duration * (delta_rate_pp / 100.0)


def _real_carry_return_decimal(
    entry_real_rate_pct: float,
    holding_bars: int,
    bars_per_year: int = 252,
) -> float:
    """
    Carry real composto em decimal.
    Usa a taxa real anualizada da entrada e compõe pelo tempo de holding.
    """
    years = holding_bars / bars_per_year
    return (1.0 + entry_real_rate_pct / 100.0) ** years - 1.0


def _total_real_return_components(
    entry_rate_pct: float,
    exit_rate_pct: float,
    duration: float,
    holding_bars: int,
    bars_per_year: int = 252,
) -> tuple[float, float, float]:
    """
    Retorna:
    - retorno total (%)
    - componente MTM (%)
    - componente carry (%)
    """
    mtm_dec = _mark_to_market_return_decimal(
        entry_rate_pct=entry_rate_pct,
        exit_rate_pct=exit_rate_pct,
        duration=duration,
    )

    carry_dec = _real_carry_return_decimal(
        entry_real_rate_pct=entry_rate_pct,
        holding_bars=holding_bars,
        bars_per_year=bars_per_year,
    )

    total_dec = (1.0 + mtm_dec) * (1.0 + carry_dec) - 1.0

    return total_dec * 100.0, mtm_dec * 100.0, carry_dec * 100.0

def backtest_macro_variable_event_sensitivity() -> str:
    import numpy as np
    import pandas as pd

    df = _load_macro_frame()
    df = df[df["data"] >= pd.Timestamp("2000-01-01")].copy()
    df = df.sort_values("data").reset_index(drop=True)

    known_event_dates = pd.to_datetime([
        "2002-07-01",
        "2008-09-15",
        "2013-05-22",
        "2015-07-01",
        "2018-05-20",
        "2020-03-15",
        "2021-10-01",
        "2024-11-01",
    ])

    # variáveis que queremos testar
    variable_specs = {
        "ipca_12m": {
            "label": "IPCA 12m",
            "mode": "diff",   # diferença absoluta
            "expected_direction": "up",
        },
        "dbgg_12m_diff": {
            "label": "DBGG 12m diff",
            "mode": "diff",
            "expected_direction": "up",
        },
        "usdbrl_stress": {
            "label": "USD/BRL stress",
            "mode": "diff",
            "expected_direction": "up",
        },
        "usdbrl_63d_change": {
            "label": "USD/BRL 63d change",
            "mode": "diff",
            "expected_direction": "up",
        },
    }

    windows = [30, 60, 90, 180]

    lines: list[str] = []
    lines.append("=== SENSIBILIDADE DAS VARIÁVEIS MACRO ANTES DOS EVENTOS ===")
    lines.append("")
    lines.append("Objetivo: verificar se as variáveis já se movem ANTES dos eventos de regime.")
    lines.append("")

    all_results = {}

    for var, spec in variable_specs.items():
        if var not in df.columns:
            lines.append(f"[ERRO] Coluna ausente: {var}")
            lines.append("")
            continue

        lines.append(f"=== VARIÁVEL: {spec['label']} ({var}) ===")

        per_window_results = {}

        for window in windows:
            changes = []
            directional_hits = 0
            usable_events = 0

            lines.append(f"-- Janela pré-evento: {window} dias --")

            for event_date in known_event_dates:
                event_rows = df[df["data"] <= event_date]
                if event_rows.empty:
                    lines.append(f"{event_date.date()} | sem dados até a data do evento")
                    continue

                event_idx = event_rows.index[-1]
                start_idx = event_idx - window

                if start_idx < 0:
                    lines.append(f"{event_date.date()} | janela insuficiente")
                    continue

                start_value = df.loc[start_idx, var]
                end_value = df.loc[event_idx, var]

                if pd.isna(start_value) or pd.isna(end_value):
                    lines.append(f"{event_date.date()} | valores ausentes")
                    continue

                change = end_value - start_value
                changes.append(change)
                usable_events += 1

                if spec["expected_direction"] == "up" and change > 0:
                    directional_hits += 1
                elif spec["expected_direction"] == "down" and change < 0:
                    directional_hits += 1

                lines.append(
                    f"{event_date.date()} | início={start_value:.4f} | "
                    f"fim={end_value:.4f} | delta={change:.4f}"
                )

            if usable_events > 0:
                arr = np.array(changes, dtype=float)
                per_window_results[window] = {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "hit_rate": directional_hits / usable_events * 100.0,
                    "usable_events": usable_events,
                }

                lines.append(
                    f"Resumo {window}d | "
                    f"média={arr.mean():.4f} | "
                    f"mediana={np.median(arr):.4f} | "
                    f"acertos direção={directional_hits}/{usable_events} "
                    f"({directional_hits / usable_events * 100.0:.1f}%)"
                )
            else:
                per_window_results[window] = None
                lines.append(f"Resumo {window}d | sem eventos utilizáveis")

            lines.append("")

        all_results[var] = {
            "label": spec["label"],
            "per_window": per_window_results,
        }

    # ranking simples por poder preditivo pré-evento
    lines.append("=== RANKING SIMPLES DAS VARIÁVEIS ===")
    lines.append("Critério: melhor hit rate médio nas janelas pré-evento.")
    ranking = []

    for var, res in all_results.items():
        hit_rates = []
        for window in windows:
            row = res["per_window"].get(window)
            if row is not None:
                hit_rates.append(row["hit_rate"])

        if hit_rates:
            ranking.append((var, res["label"], float(np.mean(hit_rates))))

    ranking = sorted(ranking, key=lambda x: x[2], reverse=True)

    for var, label, avg_hit in ranking:
        lines.append(f"{label} ({var}) | hit rate médio = {avg_hit:.1f}%")

    lines.append("")
    lines.append("=== MATRIZ DE CORRELAÇÃO ===")

    corr_cols = [c for c in variable_specs.keys() if c in df.columns]
    corr_df = df[corr_cols].copy().dropna()

    if len(corr_df) >= 10:
        corr = corr_df.corr()

        for row_var in corr.index:
            row_text = [row_var]
            for col_var in corr.columns:
                row_text.append(f"{corr.loc[row_var, col_var]:.2f}")
            lines.append(" | ".join(row_text))

        lines.append("")
        lines.append("=== ALERTAS DE REDUNDÂNCIA ===")
        threshold = 0.80
        found_redundancy = False

        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if j <= i:
                    continue
                value = corr.loc[c1, c2]
                if abs(value) >= threshold:
                    found_redundancy = True
                    lines.append(
                        f"{c1} vs {c2} | correlação={value:.2f} "
                        f"| possível redundância"
                    )

        if not found_redundancy:
            lines.append("Nenhuma redundância forte detectada com |corr| >= 0.80.")
    else:
        lines.append("Dados insuficientes para correlação.")

    return "\n".join(lines)

def _build_fx_regime_frame():
    import pandas as pd

    df = _load_macro_frame().copy()
    df = df.sort_values("data").reset_index(drop=True)

    # retorno diário
    df["usd_ret_1d"] = df["usdbrl"].pct_change()

    # tendências em 3 escalas
    df["usd_trend_21d"] = df["usdbrl"] / df["usdbrl"].shift(21) - 1.0
    df["usd_trend_63d"] = df["usdbrl"] / df["usdbrl"].shift(63) - 1.0
    df["usd_trend_126d"] = df["usdbrl"] / df["usdbrl"].shift(126) - 1.0

    # volatilidade de curto prazo
    df["usd_vol_21d"] = df["usd_ret_1d"].rolling(21, min_periods=10).std()

    # thresholds rolling por percentil
    df["usd_vol_p80"] = df["usd_vol_21d"].rolling(252, min_periods=60).quantile(0.80)
    df["usd_trend_63d_p80"] = df["usd_trend_63d"].rolling(252, min_periods=60).quantile(0.80)

    # aceleração cambial:
    # curto > médio > longo e todos positivos
    df["usd_trend_accel"] = (
        (df["usd_trend_21d"] > 0) &
        (df["usd_trend_63d"] > 0) &
        (df["usd_trend_126d"] > 0) &
        (df["usd_trend_21d"] > df["usd_trend_63d"]) &
        (df["usd_trend_63d"] > df["usd_trend_126d"])
    )

    def classify_fx_regime(row):
        if (
            pd.isna(row["usd_vol_21d"]) or
            pd.isna(row["usd_vol_p80"]) or
            pd.isna(row["usd_trend_63d"]) or
            pd.isna(row["usd_trend_63d_p80"])
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

    # suavização simples por maioria móvel de 5 dias
    smoothed = []
    values = df["fx_macro_regime_raw"].tolist()

    for i in range(len(values)):
        start = max(0, i - 4)
        window = values[start:i + 1]
        counts = pd.Series(window).value_counts()
        smoothed.append(counts.index[0])

    df["fx_macro_regime"] = smoothed

    return df

def backtest_fx_trend_vol_regime() -> str:
    df = _build_fx_regime_frame()

    lines: list[str] = []
    lines.append("=== FX TREND + VOLATILITY REGIME MODEL ===")
    lines.append("")

    last = df.dropna(
        subset=[
            "usd_trend_21d",
            "usd_trend_63d",
            "usd_trend_126d",
            "usd_vol_21d",
            "usd_vol_p80",
        ]
    ).iloc[-1]

    lines.append(f"Última data: {last['data'].strftime('%d/%m/%Y')}")
    lines.append(f"USD/BRL: {last['usdbrl']:.4f}")
    lines.append(f"Trend 21d: {last['usd_trend_21d']:.4f}")
    lines.append(f"Trend 63d: {last['usd_trend_63d']:.4f}")
    lines.append(f"Trend 126d: {last['usd_trend_126d']:.4f}")
    lines.append(f"Vol 21d: {last['usd_vol_21d']:.4f}")
    lines.append(f"Vol p80: {last['usd_vol_p80']:.4f}")
    lines.append(f"Aceleração cambial: {bool(last['usd_trend_accel'])}")
    lines.append(f"Regime atual: {last['fx_macro_regime']}")
    lines.append("")

    lines.append("=== DISTRIBUIÇÃO DOS REGIMES ===")
    counts = df["fx_macro_regime"].value_counts(dropna=False)
    total = len(df)

    for regime, count in counts.items():
        pct = count / total * 100
        lines.append(f"{regime}: {count} dias ({pct:.2f}%)")

    lines.append("")

    work = df.copy()
    prev = work["fx_macro_regime"].shift(1)

    starts = work[
        (work["fx_macro_regime"].isin(["alerta", "turbulencia", "stress"])) &
        (work["fx_macro_regime"] != prev)
    ][[
        "data",
        "fx_macro_regime",
        "usd_trend_21d",
        "usd_trend_63d",
        "usd_trend_126d",
        "usd_vol_21d",
        "usd_vol_p80",
        "usd_trend_accel",
    ]].copy()

    lines.append("=== INÍCIOS DE REGIME RELEVANTE (últimos 20) ===")

    if starts.empty:
        lines.append("Nenhum início detectado.")
    else:
        for _, row in starts.tail(20).iterrows():
            lines.append(
                f"{row['data'].strftime('%d/%m/%Y')} | "
                f"{row['fx_macro_regime']} | "
                f"t21={row['usd_trend_21d']:.4f} | "
                f"t63={row['usd_trend_63d']:.4f} | "
                f"t126={row['usd_trend_126d']:.4f} | "
                f"vol={row['usd_vol_21d']:.4f} / p80={row['usd_vol_p80']:.4f} | "
                f"accel={bool(row['usd_trend_accel'])}"
            )

    return "\n".join(lines)

def backtest_fx_regime_event_sensitivity() -> str:
    import pandas as pd

    df = _build_fx_regime_frame()
    df = df[df["data"] >= pd.Timestamp("2000-01-01")].copy()
    df = df.sort_values("data").reset_index(drop=True)

    known_event_dates = pd.to_datetime([
        "2002-07-01",
        "2008-09-15",
        "2013-05-22",
        "2015-07-01",
        "2018-05-20",
        "2020-03-15",
        "2021-10-01",
        "2024-11-01",
    ])

    work = df.copy()
    prev = work["fx_macro_regime"].shift(1)

    starts = work[
        (work["fx_macro_regime"].isin(["alerta", "turbulencia", "stress"])) &
        (work["fx_macro_regime"] != prev)
    ][[
        "data",
        "fx_macro_regime",
        "usd_trend_21d",
        "usd_trend_63d",
        "usd_trend_126d",
        "usd_vol_21d",
        "usd_vol_p80",
        "usd_trend_accel",
    ]].copy()

    lines: list[str] = []
    lines.append("=== FX REGIME EVENT SENSITIVITY ===")
    lines.append("")
    lines.append("Objetivo: verificar se o regime FX antecipa eventos históricos.")
    lines.append("")

    matched = 0

    for event_date in known_event_dates:
        nearby = starts[
            (starts["data"] >= event_date - pd.Timedelta(days=60)) &
            (starts["data"] <= event_date + pd.Timedelta(days=60))
        ].copy()

        lines.append(f"Evento: {event_date.date()}")

        if nearby.empty:
            lines.append("  Nenhum regime FX detectado na janela [-60, +60].")
            continue

        nearby["delta_days"] = (nearby["data"] - event_date).dt.days
        best = nearby.iloc[(nearby["delta_days"].abs()).argmin()]
        matched += 1

        lines.append(
            f"  Mais próximo: {best['data'].date()} | "
            f"regime={best['fx_macro_regime']} | "
            f"delta={int(best['delta_days'])} dias | "
            f"t21={best['usd_trend_21d']:.4f} | "
            f"t63={best['usd_trend_63d']:.4f} | "
            f"t126={best['usd_trend_126d']:.4f} | "
            f"vol={best['usd_vol_21d']:.4f} / p80={best['usd_vol_p80']:.4f} | "
            f"accel={bool(best['usd_trend_accel'])}"
        )

    lines.append("")
    lines.append(f"Eventos com algum encaixe na janela: {matched}/{len(known_event_dates)}")

    return "\n".join(lines)

def backtest_fx_forward_returns() -> str:
    import pandas as pd

    df = _build_fx_regime_frame().copy()
    df = df.sort_values("data").reset_index(drop=True)

    horizons = [30, 60, 120, 252]

    for h in horizons:
        df[f"ret_{h}d"] = df["usdbrl"].shift(-h) / df["usdbrl"] - 1.0

    # detecta apenas o INÍCIO de cada regime
    prev = df["fx_macro_regime"].shift(1)

    starts = df[
        (df["fx_macro_regime"].isin(["alerta", "turbulencia", "stress"])) &
        (df["fx_macro_regime"] != prev)
    ].copy()

    lines: list[str] = []
    lines.append("=== FX REGIME FORWARD RETURNS FROM REGIME START ===")
    lines.append("")
    lines.append("Objetivo: medir edge real usando apenas o início do regime.")
    lines.append("")

    if starts.empty:
        lines.append("Nenhum início de regime detectado.")
        return "\n".join(lines)

    lines.append(f"Inícios de regime detectados: {len(starts)}")
    lines.append("")

    regimes = ["alerta", "turbulencia", "stress"]

    for regime in regimes:
        subset = starts[starts["fx_macro_regime"] == regime].copy()

        lines.append(f"=== REGIME: {regime} ===")

        if subset.empty:
            lines.append("sem dados")
            lines.append("")
            continue

        lines.append(f"Inícios detectados: {len(subset)}")

        for h in horizons:
            r = subset[f"ret_{h}d"].dropna()

            if len(r) == 0:
                continue

            mean = r.mean()
            median = r.median()
            winrate = (r > 0).mean()
            best = r.max()
            worst = r.min()

            lines.append(
                f"+{h}d | média={mean:.4%} | "
                f"mediana={median:.4%} | "
                f"winrate={winrate:.1%} | "
                f"melhor={best:.4%} | "
                f"pior={worst:.4%} | "
                f"n={len(r)}"
            )

        lines.append("")

    # últimos sinais para inspeção
    lines.append("=== ÚLTIMOS 15 INÍCIOS DE REGIME ===")
    preview_cols = ["data", "fx_macro_regime"]

    if "usd_trend_21d" in starts.columns:
        preview_cols.append("usd_trend_21d")
    if "usd_trend_63d" in starts.columns:
        preview_cols.append("usd_trend_63d")
    if "usd_trend_126d" in starts.columns:
        preview_cols.append("usd_trend_126d")
    if "usd_vol_21d" in starts.columns:
        preview_cols.append("usd_vol_21d")

    for _, row in starts[preview_cols].tail(15).iterrows():
        text = f"{row['data'].strftime('%d/%m/%Y')} | {row['fx_macro_regime']}"
        if "usd_trend_21d" in row:
            text += f" | t21={row['usd_trend_21d']:.4f}"
        if "usd_trend_63d" in row:
            text += f" | t63={row['usd_trend_63d']:.4f}"
        if "usd_trend_126d" in row:
            text += f" | t126={row['usd_trend_126d']:.4f}"
        if "usd_vol_21d" in row:
            text += f" | vol21={row['usd_vol_21d']:.4f}"
        lines.append(text)

    return "\n".join(lines)

def backtest_realrate_trade_by_fx_regime() -> str:
    import pandas as pd
    import numpy as np

    result = _run_real_rate_scaled_once(
        z_entry_threshold=1.5,
        duration_minima=15,
        exit_threshold=-2.0,
        tiers=[
            (1.5, 1.0),
            (2.0, 1.5),
            (2.5, 2.0),
            (3.0, 3.0),
        ],
    )

    trades = pd.DataFrame(result.get("completed_trades", []))

    lines: list[str] = []
    lines.append("=== REAL RATE TRADE POR REGIME FX ===")
    lines.append("")

    if trades.empty:
        lines.append("Nenhum trade encontrado.")
        return "\n".join(lines)

    fx = _build_fx_regime_frame()[["data", "fx_macro_regime"]].copy()

    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    fx["data"] = pd.to_datetime(fx["data"], errors="coerce")

    trades = trades.merge(
        fx,
        left_on="entry_date",
        right_on="data",
        how="left",
    )

    # ajuste os nomes abaixo se no seu trade estiverem diferentes
    ret_col = "return_pct"
    pnl_col = "absolute_pnl_units"

    regimes = ["normal", "alerta", "turbulencia", "stress"]

    for regime in regimes:
        subset = trades[trades["fx_macro_regime"] == regime].copy()

        lines.append(f"=== REGIME: {regime} ===")

        if subset.empty:
            lines.append("sem trades")
            lines.append("")
            continue

        ret = subset[ret_col].dropna()
        pnl = subset[pnl_col].dropna()

        lines.append(f"trades: {len(subset)}")
        lines.append(f"retorno médio: {ret.mean():.2f}%")
        lines.append(f"mediana: {ret.median():.2f}%")
        lines.append(f"win rate: {(ret > 0).mean() * 100:.1f}%")
        lines.append(f"PnL médio: {pnl.mean():.2f}")
        lines.append(f"PnL total: {pnl.sum():.2f}")
        lines.append("")

    return "\n".join(lines)

def backtest_realrate_trade_fx_regime_detail() -> str:
    import pandas as pd

    result = _run_real_rate_scaled_once(
        z_entry_threshold=1.5,
        duration_minima=15,
        exit_threshold=-2.0,
        tiers=[
            (1.5, 1.0),
            (2.0, 1.5),
            (2.5, 2.0),
            (3.0, 3.0),
        ],
    )

    trades = pd.DataFrame(result.get("completed_trades", []))

    lines = []
    lines.append("=== REAL RATE TRADES COM REGIME FX ===")
    lines.append("")

    if trades.empty:
        lines.append("Nenhum trade encontrado.")
        return "\n".join(lines)

    fx = _build_fx_regime_frame()[["data", "fx_macro_regime"]].copy()

    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")

    fx["data"] = pd.to_datetime(fx["data"], errors="coerce")

    trades = trades.merge(
        fx,
        left_on="entry_date",
        right_on="data",
        how="left"
    )

    trades = trades.sort_values("entry_date")

    for _, row in trades.iterrows():

        entry = row["entry_date"].strftime("%d/%m/%Y")
        exit = row["exit_date"].strftime("%d/%m/%Y")

        regime = row.get("fx_macro_regime", "unknown")

        ret = row.get("return_pct", None)
        pnl = row.get("absolute_pnl_units", None)

        holding = row.get("holding_days", None)

        lines.append(
            f"{entry} → {exit} | "
            f"FX={regime} | "
            f"ret={ret:.2f}% | "
            f"PnL={pnl:.2f} | "
            f"holding={holding}d"
        )

    lines.append("")
    lines.append(f"Total trades: {len(trades)}")

    return "\n".join(lines)

def _mark_signal_events(df, signal_col: str, threshold: float):
    above = df[signal_col] >= threshold
    prev = above.shift(1, fill_value=False)
    event_col = f"{signal_col}_event"
    work = df.copy()
    work[event_col] = above & (~prev)
    return work, event_col

def backtest_realrate_signal_validity_by_fx_regime(
    z_threshold: float = 1.7,
    start_date: str = "2000-01-01",
    horizons: list[int] | None = None,
) -> str:
    import pandas as pd

    if horizons is None:
        horizons = [21, 63, 126, 252]

    # 1) base principal já pronta do projeto
    df = _build_research_frame(duration_minima=0.0).copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    # 2) colunas reais do repo
    z_col = "zscore_rolling_5a"
    rr_col = "taxa_media"

    df = (
        df.dropna(subset=["data", z_col, rr_col])
          .sort_values("data")
          .reset_index(drop=True)
    )
    df = df[df["data"] >= pd.Timestamp(start_date)].copy()

    # 3) regime FX já existente no repo
    fx = _build_fx_regime_frame()[["data", "fx_macro_regime"]].copy()
    fx["data"] = pd.to_datetime(fx["data"], errors="coerce")

    # merge temporal robusto, igual ao estilo do projeto
    df = pd.merge_asof(
        df.sort_values("data"),
        fx.sort_values("data"),
        on="data",
        direction="backward",
    )

    df["fx_macro_regime"] = df["fx_macro_regime"].fillna("indefinido")

    # 4) helper real do repo retorna (df, nome_da_coluna_evento)
    df, event_col = _mark_signal_events(
        df,
        signal_col=z_col,
        threshold=z_threshold,
    )

    # 5) forward moves do real rate
    for h in horizons:
        df[f"forward_{h}d"] = df[rr_col].shift(-h) - df[rr_col]

    events = df[df[event_col]].copy()
    events = events[events["fx_macro_regime"] != "indefinido"].copy()

    lines: list[str] = []
    lines.append("=== REAL RATE SIGNAL VALIDITY BY FX REGIME ===")
    lines.append("")
    lines.append(f"Período: {pd.Timestamp(start_date).strftime('%d/%m/%Y')} em diante")
    lines.append(f"Sinal: {z_col} >= {z_threshold:.2f}")
    lines.append(f"Eventos únicos: {len(events)}")
    lines.append("")

    if events.empty:
        lines.append("Nenhum evento encontrado.")
        return "\n".join(lines)

    lines.append("=== DISTRIBUIÇÃO POR REGIME FX ===")
    counts = events["fx_macro_regime"].value_counts()
    for regime, count in counts.items():
        pct = 100.0 * count / len(events)
        lines.append(f"{regime}: {count} eventos ({pct:.1f}%)")
    lines.append("")

    lines.append("=== DETALHE DOS EVENTOS ===")
    for _, row in events.iterrows():
        parts = [
            row["data"].strftime("%d/%m/%Y"),
            f"z={row[z_col]:.2f}",
            f"FX={row['fx_macro_regime']}",
        ]
        for h in horizons:
            val = row[f"forward_{h}d"]
            if pd.notna(val):
                parts.append(f"{h}d={val:+.2f}")
        lines.append(" | ".join(parts))
    lines.append("")

    lines.append("=== RESUMO POR REGIME FX ===")
    for regime in ["normal", "alerta", "turbulencia", "stress"]:
        subset = events[events["fx_macro_regime"] == regime].copy()

        if subset.empty:
            continue

        lines.append("")
        lines.append(f"[{regime}]")
        lines.append(f"Eventos: {len(subset)}")

        for h in horizons:
            col = f"forward_{h}d"
            valid = subset[col].dropna()

            if valid.empty:
                lines.append(f"{h}d | sem dados suficientes")
                continue

            mean_move = valid.mean()
            median_move = valid.median()
            reversion_hit = (valid < 0).mean() * 100.0

            lines.append(
                f"{h:>3}d | média: {mean_move:+.3f} | "
                f"mediana: {median_move:+.3f} | "
                f"reversão (queda da taxa): {reversion_hit:5.1f}%"
            )

    return "\n".join(lines)