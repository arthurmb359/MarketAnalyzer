from __future__ import annotations

import pandas as pd

from core.metrics import safe_mean, safe_median, win_rate_pct
from core.reporting import (
    format_date,
    format_pct,
    format_value,
    render_result,
    result as build_result,
    section,
)
from markets.macro_system.regime import build_fx_macro_regime_frame
from markets.macro_system.signals import mark_signal_events
from markets.tesouro_ipca.loader import load_ipca_long_research_frame


def _build_research_frame(duration_minima: float = 0.0) -> pd.DataFrame:
    df = load_ipca_long_research_frame(duration_minima=duration_minima)
    df = df.sort_values("data").reset_index(drop=True)
    return df


def _approx_duration_from_prazo(prazo_anos: float) -> float:
    if pd.isna(prazo_anos):
        return 0.0
    return max(0.0, float(prazo_anos))


def _mark_to_market_return_decimal(
    entry_rate_pct: float,
    exit_rate_pct: float,
    duration: float,
) -> float:
    delta_rate_pp = exit_rate_pct - entry_rate_pct
    return -duration * (delta_rate_pp / 100.0)


def _real_carry_return_decimal(
    entry_real_rate_pct: float,
    holding_bars: int,
    bars_per_year: int = 252,
) -> float:
    years = holding_bars / bars_per_year
    return (1.0 + entry_real_rate_pct / 100.0) ** years - 1.0


def _total_real_return_components(
    entry_rate_pct: float,
    exit_rate_pct: float,
    duration: float,
    holding_bars: int,
    bars_per_year: int = 252,
) -> tuple[float, float, float]:
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


def _run_real_rate_scaled_once(
    z_entry_threshold: float,
    duration_minima: float,
    exit_threshold: float = -2.0,
    tiers: list[tuple[float, float]] | None = None,
) -> dict:
    import numpy as np

    df = _build_research_frame(duration_minima=duration_minima)
    df = df.sort_values("data").reset_index(drop=True)

    zscore_col = "zscore_rolling_252d"

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
                entry_macro = (
                    df.loc[i, "macro_regime_label"]
                    if "macro_regime_label" in df.columns
                    else "indefinido"
                )

                if entry_macro is None or pd.isna(entry_macro):
                    entry_macro = "indefinido"
                else:
                    entry_macro = str(entry_macro).strip().lower()

                tranches.append(
                    {
                        "entry_idx": i,
                        "entry_date": df.loc[i, "data"],
                        "entry_rate": df.loc[i, "taxa_media"],
                        "entry_prazo_anos": df.loc[i, "prazo_anos"],
                        "entry_macro_regime": entry_macro,
                        "weight": weight,
                        "threshold": threshold,
                    }
                )

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

            completed_trades.append(
                {
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
                    "holding_days": np.average(holding_bars_all, weights=weights),
                    "avg_duration_model": np.average(tranche_durations, weights=weights),
                }
            )

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
        "avg_return": safe_mean(returns),
        "median_return": safe_median(returns),
        "win_rate": win_rate_pct(returns),
        "avg_absolute_pnl_units": safe_mean(absolute_pnls),
        "median_absolute_pnl_units": safe_median(absolute_pnls),
        "total_absolute_pnl_units": absolute_pnls.sum(),
        "avg_mtm": safe_mean(mtm),
        "avg_carry": safe_mean(carry),
        "avg_holding": safe_mean(holding),
        "avg_weight": safe_mean(total_weights),
        "avg_duration_model": safe_mean(avg_durations),
        "completed_trades": completed_trades,
    }


def backtest_fx_regime_event_sensitivity() -> str:
    df = build_fx_macro_regime_frame()
    df = df[df["data"] >= pd.Timestamp("2000-01-01")].copy()
    df = df.sort_values("data").reset_index(drop=True)

    known_event_dates = pd.to_datetime(
        [
            "2002-07-01",
            "2008-09-15",
            "2013-05-22",
            "2015-07-01",
            "2018-05-20",
            "2020-03-15",
            "2021-10-01",
            "2024-11-01",
        ]
    )

    work = df.copy()
    prev = work["fx_macro_regime"].shift(1)

    starts = work[
        (work["fx_macro_regime"].isin(["alerta", "turbulencia", "stress"]))
        & (work["fx_macro_regime"] != prev)
    ][
        [
            "data",
            "fx_macro_regime",
            "usd_trend_21d",
            "usd_trend_63d",
            "usd_trend_126d",
            "usd_vol_21d",
            "usd_vol_p80",
            "usd_trend_accel",
        ]
    ].copy()

    intro_lines = ["Objetivo: verificar se o regime FX antecipa eventos historicos."]
    event_lines: list[str] = []

    matched = 0

    for event_date in known_event_dates:
        nearby = starts[
            (starts["data"] >= event_date - pd.Timedelta(days=60))
            & (starts["data"] <= event_date + pd.Timedelta(days=60))
        ].copy()

        event_lines.append(f"Evento: {format_date(event_date)}")

        if nearby.empty:
            event_lines.append("  Nenhum regime FX detectado na janela [-60, +60].")
            continue

        nearby["delta_days"] = (nearby["data"] - event_date).dt.days
        best = nearby.iloc[(nearby["delta_days"].abs()).argmin()]
        matched += 1

        event_lines.append(
            f"  Mais proximo: {format_date(best['data'])} | "
            f"regime={best['fx_macro_regime']} | "
            f"delta={int(best['delta_days'])} dias | "
            f"t21={best['usd_trend_21d']:.4f} | "
            f"t63={best['usd_trend_63d']:.4f} | "
            f"t126={best['usd_trend_126d']:.4f} | "
            f"vol={best['usd_vol_21d']:.4f} / p80={best['usd_vol_p80']:.4f} | "
            f"accel={bool(best['usd_trend_accel'])}"
        )

    report = build_result(
        "FX REGIME EVENT SENSITIVITY",
        section(intro_lines),
        section(event_lines, title="EVENTOS"),
        section(
            [f"Eventos com algum encaixe na janela: {matched}/{len(known_event_dates)}"],
            title="RESUMO",
        ),
    )
    return render_result(report)

def backtest_realrate_trade_by_fx_regime() -> str:
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

    regime_lines: list[str] = []

    if trades.empty:
        report = build_result(
            "REAL RATE TRADE POR REGIME FX",
            section(["Nenhum trade encontrado."]),
        )
        return render_result(report)

    fx = build_fx_macro_regime_frame()[["data", "fx_macro_regime"]].copy()

    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    fx["data"] = pd.to_datetime(fx["data"], errors="coerce")

    trades = trades.merge(
        fx,
        left_on="entry_date",
        right_on="data",
        how="left",
    )

    ret_col = "return_pct"
    pnl_col = "absolute_pnl_units"
    regimes = ["normal", "alerta", "turbulencia", "stress"]

    for regime in regimes:
        subset = trades[trades["fx_macro_regime"] == regime].copy()

        regime_lines.append(f"=== REGIME: {regime} ===")

        if subset.empty:
            regime_lines.append("sem trades")
            regime_lines.append("")
            continue

        ret = subset[ret_col].dropna()
        pnl = subset[pnl_col].dropna()

        regime_lines.append(f"trades: {len(subset)}")
        regime_lines.append(f"retorno medio: {format_pct(safe_mean(ret), 2)}")
        regime_lines.append(f"mediana: {format_pct(safe_median(ret), 2)}")
        regime_lines.append(f"win rate: {format_pct(win_rate_pct(ret), 1)}")
        regime_lines.append(f"PnL medio: {format_value(safe_mean(pnl), 2)}")
        regime_lines.append(f"PnL total: {format_value(float(pnl.sum()), 2)}")
        regime_lines.append("")

    report = build_result(
        "REAL RATE TRADE POR REGIME FX",
        section(regime_lines, title="REGIMES"),
    )
    return render_result(report)


def backtest_realrate_trade_fx_regime_detail() -> str:
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

    detail_lines: list[str] = []

    if trades.empty:
        report = build_result(
            "REAL RATE TRADES COM REGIME FX",
            section(["Nenhum trade encontrado."]),
        )
        return render_result(report)

    fx = build_fx_macro_regime_frame()[["data", "fx_macro_regime"]].copy()

    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    fx["data"] = pd.to_datetime(fx["data"], errors="coerce")

    trades = trades.merge(
        fx,
        left_on="entry_date",
        right_on="data",
        how="left",
    )

    trades = trades.sort_values("entry_date")

    for _, row in trades.iterrows():
        entry = format_date(row["entry_date"])
        exit = format_date(row["exit_date"])
        regime = row.get("fx_macro_regime", "unknown")
        ret = row.get("return_pct", None)
        pnl = row.get("absolute_pnl_units", None)
        holding = row.get("holding_days", None)

        detail_lines.append(
            f"{entry} -> {exit} | "
            f"FX={regime} | "
            f"ret={format_pct(ret, 2)} | "
            f"PnL={format_value(pnl, 2)} | "
            f"holding={holding}d"
        )

    report = build_result(
        "REAL RATE TRADES COM REGIME FX",
        section(detail_lines, title="TRADES"),
        section([f"Total trades: {len(trades)}"], title="RESUMO"),
    )
    return render_result(report)


def backtest_realrate_signal_validity_by_fx_regime(
    z_threshold: float = 1.7,
    start_date: str = "2000-01-01",
    horizons: list[int] | None = None,
) -> str:
    if horizons is None:
        horizons = [21, 63, 126, 252]

    df = _build_research_frame(duration_minima=0.0).copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    z_col = "zscore_rolling_252d"
    rr_col = "taxa_media"

    df = (
        df.dropna(subset=["data", z_col, rr_col])
        .sort_values("data")
        .reset_index(drop=True)
    )
    df = df[df["data"] >= pd.Timestamp(start_date)].copy()

    fx = build_fx_macro_regime_frame()[["data", "fx_macro_regime"]].copy()
    fx["data"] = pd.to_datetime(fx["data"], errors="coerce")

    df = pd.merge_asof(
        df.sort_values("data"),
        fx.sort_values("data"),
        on="data",
        direction="backward",
    )

    df["fx_macro_regime"] = df["fx_macro_regime"].fillna("indefinido")

    df, event_col = mark_signal_events(
        df,
        signal_col=z_col,
        threshold=z_threshold,
    )

    for h in horizons:
        df[f"forward_{h}d"] = df[rr_col].shift(-h) - df[rr_col]

    events = df[df[event_col]].copy()
    events = events[events["fx_macro_regime"] != "indefinido"].copy()

    intro_lines = [
        f"Periodo: {format_date(pd.Timestamp(start_date))} em diante",
        f"Sinal: {z_col} >= {z_threshold:.2f}",
        f"Eventos unicos: {len(events)}",
    ]

    if events.empty:
        report = build_result(
            "REAL RATE SIGNAL VALIDITY BY FX REGIME",
            section(intro_lines + ["", "Nenhum evento encontrado."]),
        )
        return render_result(report)

    distribuicao_lines: list[str] = []
    counts = events["fx_macro_regime"].value_counts()
    for regime, count in counts.items():
        pct = 100.0 * count / len(events)
        distribuicao_lines.append(f"{regime}: {count} eventos ({format_pct(pct, 1)})")

    detalhe_lines: list[str] = []
    for _, row in events.iterrows():
        parts = [
            format_date(row["data"]),
            f"z={row[z_col]:.2f}",
            f"FX={row['fx_macro_regime']}",
        ]
        for h in horizons:
            val = row[f"forward_{h}d"]
            if pd.notna(val):
                parts.append(f"{h}d={val:+.2f}")
        detalhe_lines.append(" | ".join(parts))

    regime_lines: list[str] = []
    for regime in ["normal", "alerta", "turbulencia", "stress"]:
        subset = events[events["fx_macro_regime"] == regime].copy()

        if subset.empty:
            continue

        regime_lines.append(f"[{regime}]")
        regime_lines.append(f"Eventos: {len(subset)}")

        for h in horizons:
            col = f"forward_{h}d"
            valid = subset[col].dropna()

            if valid.empty:
                regime_lines.append(f"{h}d | sem dados suficientes")
                continue

            mean_move = safe_mean(valid)
            median_move = safe_median(valid)
            reversion_hit = (valid < 0).mean() * 100.0

            regime_lines.append(
                f"{h:>3}d | media: {mean_move:+.3f} | "
                f"mediana: {median_move:+.3f} | "
                f"reversao (queda da taxa): {reversion_hit:5.1f}%"
            )
        regime_lines.append("")

    report = build_result(
        "REAL RATE SIGNAL VALIDITY BY FX REGIME",
        section(intro_lines),
        section(distribuicao_lines, title="DISTRIBUICAO POR REGIME FX"),
        section(detalhe_lines, title="DETALHE DOS EVENTOS"),
        section(regime_lines, title="RESUMO POR REGIME FX"),
    )
    return render_result(report)


__all__ = [
    "backtest_fx_regime_event_sensitivity",
    "backtest_realrate_signal_validity_by_fx_regime",
    "backtest_realrate_trade_by_fx_regime",
    "backtest_realrate_trade_fx_regime_detail",
]
