from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backtest.algorithms import _build_research_frame, _build_fx_regime_frame


# ============================================================
# CONFIG
# ============================================================

DEFAULT_HORIZON_DAYS = 120
DEFAULT_DURATION_PROXY = 12.0
DEFAULT_TRAIN_YEARS = 5
DEFAULT_TEST_YEARS = 1
DEFAULT_RIDGE_ALPHA = 1.0

DEFAULT_Z_ENTRY = 1.7
DEFAULT_Z_EXIT = -2.0
DEFAULT_DURATION_MINIMA = 15
DEFAULT_ML_THRESHOLD = 0.0
DEFAULT_BASE_NOTIONAL = 100.0


# ============================================================
# HELPERS
# ============================================================

@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _safe_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], ctx: str = "") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Colunas ausentes em {ctx}: {missing}" if ctx else f"Colunas ausentes: {missing}"
        raise ValueError(msg)


def _cumulative_entry_weight(z_value: float) -> float:
    """
    Escalonamento cumulativo correto:
    1.7 -> 1.0x
    2.0 -> +1.5x
    2.5 -> +2.0x
    3.0 -> +3.0x

    Total máximo = 6.5x
    """
    weight = 0.0

    if z_value >= 1.7:
        weight += 1.0
    if z_value >= 2.0:
        weight += 1.5
    if z_value >= 2.5:
        weight += 2.0
    if z_value >= 3.0:
        weight += 3.0

    return min(weight, 6.5)

def _format_pct(x: float) -> str:
    return f"{x:.2f}%"


def _format_num(x: float) -> str:
    return f"{x:.2f}"


# ============================================================
# DATASET SUPERVISIONADO
# ============================================================

def _build_supervised_realrate_dataset(
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    duration_proxy: float = DEFAULT_DURATION_PROXY,
    start_date: str = "2000-01-01",
) -> pd.DataFrame:
    """
    Dataset supervisionado diário para prever retorno forward 120d.
    Target:
        target_return_120d = -duration_proxy * (taxa_t+120 - taxa_t)

    Reaproveita:
    - _build_research_frame()
    - _build_fx_regime_frame()
    """
    base = _build_research_frame(duration_minima=0.0).copy()
    fx = _build_fx_regime_frame().copy()

    base = _safe_datetime(base, "data")
    fx = _safe_datetime(fx, "data")

    _ensure_columns(
        base,
        ["data", "taxa_media", "zscore_rolling_252d"],
        ctx="_build_research_frame",
    )
    _ensure_columns(
        fx,
        ["data", "usd_trend_21d", "usd_trend_63d", "usd_trend_126d", "usd_vol_21d", "fx_macro_regime"],
        ctx="_build_fx_regime_frame",
    )

    base = base[base["data"] >= pd.Timestamp(start_date)].copy()
    base = base.sort_values("data").reset_index(drop=True)
    fx = fx.sort_values("data").reset_index(drop=True)

    df = pd.merge_asof(
        base,
        fx[["data", "usd_trend_21d", "usd_trend_63d", "usd_trend_126d", "usd_vol_21d", "fx_macro_regime"]],
        on="data",
        direction="backward",
    )

    # Aliases centrais
    df["z"] = df["zscore_rolling_252d"]
    df["rate"] = df["taxa_media"]

    # Lags / deltas do z-score
    df["z_lag_1"] = df["z"].shift(1)
    df["z_lag_5"] = df["z"].shift(5)
    df["z_lag_21"] = df["z"].shift(21)
    df["z_delta_1"] = df["z"] - df["z"].shift(1)
    df["z_delta_5"] = df["z"] - df["z"].shift(5)
    df["z_delta_21"] = df["z"] - df["z"].shift(21)
    df["z_abs"] = df["z"].abs()
    df["z_sq"] = df["z"] ** 2

    # Lags / deltas da taxa
    df["rate_lag_1"] = df["rate"].shift(1)
    df["rate_lag_5"] = df["rate"].shift(5)
    df["rate_lag_21"] = df["rate"].shift(21)
    df["rate_delta_1"] = df["rate"] - df["rate"].shift(1)
    df["rate_delta_5"] = df["rate"] - df["rate"].shift(5)
    df["rate_delta_21"] = df["rate"] - df["rate"].shift(21)

    # Target
    df["forward_rate_120d"] = df["rate"].shift(-horizon_days) - df["rate"]
    df["target_return_120d"] = -duration_proxy * df["forward_rate_120d"]

    # Dummies do regime FX
    df["fx_macro_regime"] = df["fx_macro_regime"].fillna("indefinido")
    fx_dummies = pd.get_dummies(df["fx_macro_regime"], prefix="fx", dtype=float)
    df = pd.concat([df, fx_dummies], axis=1)

    return df


def _get_ridge_feature_columns(df: pd.DataFrame) -> list[str]:
    base_cols = [
        "z",
        "z_lag_1",
        "z_lag_5",
        "z_lag_21",
        "z_delta_1",
        "z_delta_5",
        "z_delta_21",
        "z_abs",
        "z_sq",
        "rate",
        "rate_lag_1",
        "rate_lag_5",
        "rate_lag_21",
        "rate_delta_1",
        "rate_delta_5",
        "rate_delta_21",
        "usd_trend_21d",
        "usd_trend_63d",
        "usd_trend_126d",
        "usd_vol_21d",
    ]

    numeric_fx_cols = []
    for c in df.columns:
        if c.startswith("fx_") and c != "fx_macro_regime":
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_fx_cols.append(c)

    return [c for c in base_cols + sorted(numeric_fx_cols) if c in df.columns]


def _generate_walk_forward_splits(
    df: pd.DataFrame,
    train_years: int = DEFAULT_TRAIN_YEARS,
    test_years: int = DEFAULT_TEST_YEARS,
) -> list[WalkForwardSplit]:
    work = df.dropna(subset=["data"]).copy()
    work = work.sort_values("data").reset_index(drop=True)

    years = sorted(work["data"].dt.year.unique().tolist())
    splits: list[WalkForwardSplit] = []

    for i in range(train_years, len(years) - test_years + 1):
        train_start_year = years[i - train_years]
        train_end_year = years[i - 1]
        test_start_year = years[i]
        test_end_year = years[i + test_years - 1]

        splits.append(
            WalkForwardSplit(
                train_start=pd.Timestamp(f"{train_start_year}-01-01"),
                train_end=pd.Timestamp(f"{train_end_year}-12-31"),
                test_start=pd.Timestamp(f"{test_start_year}-01-01"),
                test_end=pd.Timestamp(f"{test_end_year}-12-31"),
            )
        )

    return splits


# ============================================================
# MODELAGEM RIDGE
# ============================================================

def run_ridge_walk_forward_120d(
    alpha: float = DEFAULT_RIDGE_ALPHA,
    train_years: int = DEFAULT_TRAIN_YEARS,
    test_years: int = DEFAULT_TEST_YEARS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    duration_proxy: float = DEFAULT_DURATION_PROXY,
) -> pd.DataFrame:
    df = _build_supervised_realrate_dataset(
        horizon_days=horizon_days,
        duration_proxy=duration_proxy,
    ).copy()

    feature_cols = _get_ridge_feature_columns(df)
    required_cols = ["data", "target_return_120d"] + feature_cols
    work = df.dropna(subset=required_cols).copy()
    work = work.sort_values("data").reset_index(drop=True)

    splits = _generate_walk_forward_splits(
        work,
        train_years=train_years,
        test_years=test_years,
    )

    predictions: list[pd.DataFrame] = []

    for split in splits:
        train_mask = (work["data"] >= split.train_start) & (work["data"] <= split.train_end)
        test_mask = (work["data"] >= split.test_start) & (work["data"] <= split.test_end)

        train_df = work.loc[train_mask].copy()
        test_df = work.loc[test_mask].copy()

        if len(train_df) < 200 or len(test_df) < 20:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["target_return_120d"]
        X_test = test_df[feature_cols]
        y_test = test_df["target_return_120d"]

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold = test_df[
            [
                "data",
                "taxa_media",
                "zscore_rolling_252d",
                "fx_macro_regime",
            ]
        ].copy()

        fold["y_true"] = y_test.values
        fold["y_pred"] = y_pred
        fold["train_start"] = split.train_start
        fold["train_end"] = split.train_end
        fold["test_start"] = split.test_start
        fold["test_end"] = split.test_end

        predictions.append(fold)

    if not predictions:
        raise ValueError("Nenhum split walk-forward válido foi gerado para a Ridge.")

    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df = pred_df.sort_values("data").reset_index(drop=True)
    return pred_df


def backtest_ridge_forward_return_120d() -> str:
    pred_df = run_ridge_walk_forward_120d()

    mse = mean_squared_error(pred_df["y_true"], pred_df["y_pred"])
    rmse = mse ** 0.5
    r2 = r2_score(pred_df["y_true"], pred_df["y_pred"])
    corr = pred_df["y_true"].corr(pred_df["y_pred"])
    hit = (pred_df["y_true"] * pred_df["y_pred"] > 0).mean() * 100.0

    lines: list[str] = []
    lines.append("=== RIDGE REGRESSION - FORWARD RETURN 120D ===")
    lines.append("")
    lines.append(f"Observações OOS: {len(pred_df)}")
    lines.append(f"RMSE: {rmse:.4f}")
    lines.append(f"R²: {r2:.4f}")
    lines.append(f"Correlação y_true vs y_pred: {corr:.4f}")
    lines.append(f"Hit rate direcional: {hit:.1f}%")
    lines.append("")
    lines.append("Amostra das predições:")
    preview = pred_df.tail(10)

    for _, row in preview.iterrows():
        lines.append(
            f"{row['data'].strftime('%d/%m/%Y')} | "
            f"z={row['zscore_rolling_252d']:.2f} | "
            f"FX={row['fx_macro_regime']} | "
            f"y_true={row['y_true']:+.3f} | "
            f"y_pred={row['y_pred']:+.3f}"
        )

    return "\n".join(lines)

def _mark_threshold_cross_events(
    df: pd.DataFrame,
    signal_col: str,
    threshold: float,
    event_col: str = "signal_event",
) -> pd.DataFrame:
    """
    Marca apenas o primeiro dia em que o sinal cruza para cima do limiar.
    Evita contar vários dias seguidos do mesmo episódio.
    """
    work = df.copy()
    above = work[signal_col] >= threshold
    above_prev = above.shift(1, fill_value=False)
    work[event_col] = above & (~above_prev)
    return work

def _run_event_trade_engine(
    df: pd.DataFrame,
    event_flag_col: str,
    z_col: str = "zscore_rolling_252d",
    rate_col: str = "taxa_media",
    exit_threshold: float = DEFAULT_Z_EXIT,
    duration_minima: int = DEFAULT_DURATION_MINIMA,
    base_notional: float = DEFAULT_BASE_NOTIONAL,
) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    work = work.sort_values("data").reset_index(drop=True)

    _ensure_columns(
        work,
        ["data", event_flag_col, z_col, rate_col],
        ctx="_run_event_trade_engine",
    )

    trades: list[dict] = []

    in_position = False
    entry_idx = None
    entry_date = None
    entry_rate = None
    entry_z = None
    entry_weight = None

    for i, row in work.iterrows():
        current_date = row["data"]
        current_z = row[z_col]
        current_rate = row[rate_col]

        if not in_position:
            if bool(row[event_flag_col]):
                weight = _cumulative_entry_weight(float(current_z))
                if weight <= 0:
                    continue

                in_position = True
                entry_idx = i
                entry_date = current_date
                entry_rate = float(current_rate)
                entry_z = float(current_z)
                entry_weight = float(weight)
                continue

        else:
            holding_days = i - entry_idx

            if holding_days >= duration_minima and current_z <= exit_threshold:
                exit_rate = float(current_rate)
                rate_move = _simple_rate_trade_score(
                    entry_rate=entry_rate,
                    exit_rate=exit_rate,
                )
                pnl_score = base_notional * entry_weight * rate_move

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_z": entry_z,
                        "exit_z": float(current_z),
                        "entry_rate": entry_rate,
                        "exit_rate": exit_rate,
                        "holding_days": int(holding_days),
                        "weight": entry_weight,
                        "rate_move": float(rate_move),
                        "pnl_score": float(pnl_score),
                    }
                )

                in_position = False
                entry_idx = None
                entry_date = None
                entry_rate = None
                entry_z = None
                entry_weight = None

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        stats = {
            "trades": 0,
            "win_rate": 0.0,
            "rate_move_mean": 0.0,
            "rate_move_median": 0.0,
            "score_total": 0.0,
            "score_mean": 0.0,
            "holding": 0.0,
            "peso": 0.0,
        }
        return trades_df, stats

    wins = (trades_df["rate_move"] > 0).mean() * 100.0

    stats = {
        "trades": int(len(trades_df)),
        "win_rate": float(wins),
        "rate_move_mean": float(trades_df["rate_move"].mean()),
        "rate_move_median": float(trades_df["rate_move"].median()),
        "score_total": float(trades_df["pnl_score"].sum()),
        "score_mean": float(trades_df["pnl_score"].mean()),
        "holding": float(trades_df["holding_days"].mean()),
        "peso": float(trades_df["weight"].mean()),
    }

    return trades_df, stats


def backtest_ridge_feature_importance() -> str:
    df = _build_supervised_realrate_dataset().copy()
    feature_cols = _get_ridge_feature_columns(df)

    work = df.dropna(subset=["data", "target_return_120d"] + feature_cols).copy()
    work = work.sort_values("data").reset_index(drop=True)

    splits = _generate_walk_forward_splits(work)
    coef_frames: list[pd.DataFrame] = []

    for split in splits:
        train_mask = (work["data"] >= split.train_start) & (work["data"] <= split.train_end)
        train_df = work.loc[train_mask].copy()

        if len(train_df) < 200:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["target_return_120d"]

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)

        model = Ridge(alpha=DEFAULT_RIDGE_ALPHA)
        model.fit(Xs, y_train)

        coef_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "coef": model.coef_,
                "train_end": split.train_end,
            }
        )
        coef_frames.append(coef_df)

    if not coef_frames:
        raise ValueError("Não foi possível calcular coeficientes médios da Ridge.")

    coef_all = pd.concat(coef_frames, ignore_index=True)
    summary = (
        coef_all.groupby("feature")["coef"]
        .agg(["mean", "median", "std"])
        .sort_values("mean", key=lambda s: s.abs(), ascending=False)
    )

    lines: list[str] = []
    lines.append("=== RIDGE FEATURE IMPORTANCE ===")
    lines.append("")

    for feature, row in summary.iterrows():
        lines.append(
            f"{feature}: mean={row['mean']:+.4f} | "
            f"median={row['median']:+.4f} | "
            f"std={row['std']:.4f}"
        )

    return "\n".join(lines)


# ============================================================
# ENGINE DE TRADE PARA COMPARAÇÃO BASELINE VS ML
# ============================================================

def _simple_rate_trade_score(
    entry_rate: float,
    exit_rate: float,
) -> float:
    """
    Score simples do trade:
    queda da taxa = positivo
    alta da taxa = negativo

    Não tenta converter para retorno percentual de bond.
    Serve apenas para comparação robusta entre estratégias.
    """
    return entry_rate - exit_rate


def _run_entry_flag_trade_engine(
    df: pd.DataFrame,
    entry_flag_col: str,
    z_col: str = "zscore_rolling_252d",
    rate_col: str = "taxa_media",
    exit_threshold: float = DEFAULT_Z_EXIT,
    duration_minima: int = DEFAULT_DURATION_MINIMA,
    base_notional: float = DEFAULT_BASE_NOTIONAL,
) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    work = work.sort_values("data").reset_index(drop=True)

    _ensure_columns(work, ["data", entry_flag_col, z_col, rate_col], ctx="_run_entry_flag_trade_engine")

    trades: list[dict] = []

    in_position = False
    entry_idx = None
    entry_date = None
    entry_rate = None
    entry_z = None
    entry_weight = None

    for i, row in work.iterrows():
        current_date = row["data"]
        current_z = row[z_col]
        current_rate = row[rate_col]

        if not in_position:
            if bool(row[entry_flag_col]):
                weight = _cumulative_entry_weight(float(current_z))
                if weight <= 0:
                    continue

                in_position = True
                entry_idx = i
                entry_date = current_date
                entry_rate = float(current_rate)
                entry_z = float(current_z)
                entry_weight = float(weight)
                continue

        else:
            holding_days = i - entry_idx

            if holding_days >= duration_minima and current_z <= exit_threshold:
                exit_rate = float(current_rate)
                score = _simple_rate_trade_score(
                    entry_rate=entry_rate,
                    exit_rate=exit_rate,
                )
                pnl_score = base_notional * entry_weight * score

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_z": entry_z,
                        "exit_z": float(current_z),
                        "entry_rate": entry_rate,
                        "exit_rate": exit_rate,
                        "holding_days": int(holding_days),
                        "weight": entry_weight,
                        "rate_move": float(score),
                        "pnl_score": float(pnl_score),
                    }
                )

                in_position = False
                entry_idx = None
                entry_date = None
                entry_rate = None
                entry_z = None
                entry_weight = None

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        stats = {
            "trades": 0,
            "win_rate": 0.0,
            "rate_move_mean": 0.0,
            "rate_move_median": 0.0,
            "score_total": 0.0,
            "score_mean": 0.0,
            "holding": 0.0,
            "peso": 0.0,
        }
        return trades_df, stats

    wins = (trades_df["rate_move"] > 0).mean() * 100.0

    stats = {
        "trades": int(len(trades_df)),
        "win_rate": float(wins),
        "rate_move_mean": float(trades_df["rate_move"].mean()),
        "rate_move_median": float(trades_df["rate_move"].median()),
        "score_total": float(trades_df["pnl_score"].sum()),
        "score_mean": float(trades_df["pnl_score"].mean()),
        "holding": float(trades_df["holding_days"].mean()),
        "peso": float(trades_df["weight"].mean()),
    }

    return trades_df, stats

def _stats_to_lines(label: str, stats: dict) -> list[str]:
    lines: list[str] = []
    lines.append(f"[{label}]")
    lines.append(f"trades={stats['trades']}")
    lines.append(f"rate_move_médio={stats['rate_move_mean']:.4f}")
    lines.append(f"rate_move_mediana={stats['rate_move_median']:.4f}")
    lines.append(f"win={stats['win_rate']:.1f}%")
    lines.append(f"score_médio={stats['score_mean']:.2f}")
    lines.append(f"score_total={stats['score_total']:.2f}")
    lines.append(f"holding={stats['holding']:.1f}")
    lines.append(f"peso={stats['peso']:.2f}x")
    return lines


def _attach_oos_ml_predictions_to_research_frame() -> pd.DataFrame:
    base = _build_research_frame(duration_minima=0.0).copy()
    pred = run_ridge_walk_forward_120d().copy()

    base = _safe_datetime(base, "data")
    pred = _safe_datetime(pred, "data")

    out = base.merge(pred[["data", "y_pred"]], on="data", how="left")
    out = out.sort_values("data").reset_index(drop=True)
    return out


def backtest_compare_baseline_vs_ridge_filter(
    z_threshold: float = DEFAULT_Z_ENTRY,
    ml_threshold: float = DEFAULT_ML_THRESHOLD,
    duration_minima: int = DEFAULT_DURATION_MINIMA,
    z_exit_threshold: float = DEFAULT_Z_EXIT,
) -> str:
    df = _attach_oos_ml_predictions_to_research_frame().copy()

    _ensure_columns(
        df,
        ["data", "zscore_rolling_252d", "taxa_media", "y_pred"],
        ctx="backtest_compare_baseline_vs_ridge_filter",
    )

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values("data").reset_index(drop=True)

    # =========================================================
    # OOS ONLY:
    # baseline e ML só podem atuar onde existe predição OOS válida
    # =========================================================
    oos_df = df[df["y_pred"].notna()].copy()
    oos_df = oos_df.sort_values("data").reset_index(drop=True)

    # =========================================================
    # EVENTOS:
    # marcar apenas o primeiro cruzamento de cada episódio
    # =========================================================
    oos_df = _mark_threshold_cross_events(
        oos_df,
        signal_col="zscore_rolling_252d",
        threshold=z_threshold,
        event_col="z_event",
    )

    # baseline entra em todo evento OOS
    oos_df["baseline_event"] = oos_df["z_event"]

    # ML entra apenas nos eventos OOS em que a predição supera o corte
    oos_df["ml_event"] = oos_df["z_event"] & (oos_df["y_pred"] > ml_threshold)

    baseline_trades, baseline_stats = _run_event_trade_engine(
        df=oos_df,
        event_flag_col="baseline_event",
        z_col="zscore_rolling_252d",
        rate_col="taxa_media",
        exit_threshold=z_exit_threshold,
        duration_minima=duration_minima,
    )

    ml_trades, ml_stats = _run_event_trade_engine(
        df=oos_df,
        event_flag_col="ml_event",
        z_col="zscore_rolling_252d",
        rate_col="taxa_media",
        exit_threshold=z_exit_threshold,
        duration_minima=duration_minima,
    )

    lines: list[str] = []
    lines.append("=== BASELINE OOS VS RIDGE FILTER OOS (EVENT-BASED) ===")
    lines.append("")
    lines.append(f"z_entry_threshold = {z_threshold:.2f}")
    lines.append(f"ml_threshold = {ml_threshold:.4f}")
    lines.append(f"z_exit_threshold = {z_exit_threshold:.2f}")
    lines.append(f"duration_minima = {duration_minima}")
    lines.append(f"dias OOS válidos = {len(oos_df)}")
    lines.append(f"eventos OOS = {int(oos_df['z_event'].sum())}")
    lines.append("")

    lines.extend(_stats_to_lines("Baseline OOS", baseline_stats))
    lines.append("")
    lines.extend(_stats_to_lines("ML Filter OOS", ml_stats))
    lines.append("")

    baseline_total = baseline_stats["score_total"]
    ml_total = ml_stats["score_total"]

    delta_abs = ml_total - baseline_total
    delta_pct = (delta_abs / baseline_total * 100.0) if baseline_total != 0 else np.nan

    lines.append("=== DELTA ML VS BASELINE ===")
    lines.append(f"delta_score_total = {delta_abs:+.2f}")
    if pd.notna(delta_pct):
        lines.append(f"delta_score_total_pct = {delta_pct:+.2f}%")
    else:
        lines.append("delta_score_total_pct = n/a")
    lines.append("")

    if not baseline_trades.empty:
        lines.append("=== DETALHE DOS TRADES BASELINE OOS ===")
        for _, row in baseline_trades.iterrows():
            lines.append(
                f"{row['entry_date'].strftime('%d/%m/%Y')} -> {row['exit_date'].strftime('%d/%m/%Y')} | "
                f"z_in={row['entry_z']:.2f} | "
                f"rate_move={row['rate_move']:+.4f} | "
                f"score={row['pnl_score']:.2f} | "
                f"holding={row['holding_days']}d | "
                f"peso={row['weight']:.2f}x"
            )
        lines.append("")

    if not ml_trades.empty:
        lines.append("=== DETALHE DOS TRADES ML OOS ===")
        for _, row in ml_trades.iterrows():
            lines.append(
                f"{row['entry_date'].strftime('%d/%m/%Y')} -> {row['exit_date'].strftime('%d/%m/%Y')} | "
                f"z_in={row['entry_z']:.2f} | "
                f"rate_move={row['rate_move']:+.4f} | "
                f"score={row['pnl_score']:.2f} | "
                f"holding={row['holding_days']}d | "
                f"peso={row['weight']:.2f}x"
            )

    return "\n".join(lines)


# ============================================================
# REGISTRY-FRIENDLY WRAPPERS
# ============================================================

def backtest_build_supervised_dataset_preview() -> str:
    df = _build_supervised_realrate_dataset().copy()
    feature_cols = _get_ridge_feature_columns(df)

    lines: list[str] = []
    lines.append("=== SUPERVISED DATASET PREVIEW ===")
    lines.append("")
    lines.append(f"Linhas totais: {len(df)}")
    lines.append(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    lines.append("")
    lines.append("Últimas 10 linhas válidas:")

    preview_cols = ["data", "zscore_rolling_252d", "taxa_media", "fx_macro_regime", "target_return_120d"] + feature_cols[:8]
    preview_cols = [c for c in preview_cols if c in df.columns]

    preview = df.dropna(subset=["target_return_120d"]).tail(10)[preview_cols]

    for _, row in preview.iterrows():
        parts = [row["data"].strftime("%d/%m/%Y")]
        for col in preview_cols[1:]:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                parts.append(f"{col}={val:+.3f}")
            else:
                parts.append(f"{col}={val}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)


def backtest_optimize_ridge_filter_threshold(
    z_threshold: float = DEFAULT_Z_ENTRY,
    duration_minima: int = DEFAULT_DURATION_MINIMA,
    z_exit_threshold: float = DEFAULT_Z_EXIT,
    ml_threshold_grid: list[float] | None = None,
) -> str:
    """
    Testa múltiplos thresholds para o filtro da Ridge, sempre em:
    - janela OOS válida
    - lógica event-based
    - comparando contra o baseline OOS

    Regra:
    baseline_event = primeiro cruzamento de z >= z_threshold
    ml_event = baseline_event AND y_pred > ml_threshold
    """
    if ml_threshold_grid is None:
        ml_threshold_grid = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    df = _attach_oos_ml_predictions_to_research_frame().copy()

    _ensure_columns(
        df,
        ["data", "zscore_rolling_252d", "taxa_media", "y_pred"],
        ctx="backtest_optimize_ridge_filter_threshold",
    )

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values("data").reset_index(drop=True)

    # OOS only
    oos_df = df[df["y_pred"].notna()].copy()
    oos_df = oos_df.sort_values("data").reset_index(drop=True)

    # Eventos por primeiro cruzamento
    oos_df = _mark_threshold_cross_events(
        oos_df,
        signal_col="zscore_rolling_252d",
        threshold=z_threshold,
        event_col="z_event",
    )

    # Baseline OOS
    oos_df["baseline_event"] = oos_df["z_event"]

    baseline_trades, baseline_stats = _run_event_trade_engine(
        df=oos_df,
        event_flag_col="baseline_event",
        z_col="zscore_rolling_252d",
        rate_col="taxa_media",
        exit_threshold=z_exit_threshold,
        duration_minima=duration_minima,
    )

    results: list[dict] = []

    for ml_threshold in ml_threshold_grid:
        work = oos_df.copy()
        work["ml_event"] = work["z_event"] & (work["y_pred"] > ml_threshold)

        ml_trades, ml_stats = _run_event_trade_engine(
            df=work,
            event_flag_col="ml_event",
            z_col="zscore_rolling_252d",
            rate_col="taxa_media",
            exit_threshold=z_exit_threshold,
            duration_minima=duration_minima,
        )

        baseline_total = baseline_stats["score_total"]
        ml_total = ml_stats["score_total"]
        delta_abs = ml_total - baseline_total
        delta_pct = (delta_abs / baseline_total * 100.0) if baseline_total != 0 else np.nan

        results.append(
            {
                "ml_threshold": float(ml_threshold),
                "trades": int(ml_stats["trades"]),
                "win_rate": float(ml_stats["win_rate"]),
                "rate_move_mean": float(ml_stats["rate_move_mean"]),
                "rate_move_median": float(ml_stats["rate_move_median"]),
                "score_mean": float(ml_stats["score_mean"]),
                "score_total": float(ml_total),
                "holding": float(ml_stats["holding"]),
                "peso": float(ml_stats["peso"]),
                "delta_score_total": float(delta_abs),
                "delta_score_total_pct": float(delta_pct) if pd.notna(delta_pct) else np.nan,
            }
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return "Nenhum resultado gerado para Optimize Ridge Filter Threshold."

    # critério principal: score_total
    best_total = results_df.sort_values(
        ["score_total", "score_mean", "win_rate"],
        ascending=[False, False, False],
    ).iloc[0]

    # critério secundário: score_mean
    best_mean = results_df.sort_values(
        ["score_mean", "score_total", "win_rate"],
        ascending=[False, False, False],
    ).iloc[0]

    # critério de delta vs baseline
    best_delta = results_df.sort_values(
        ["delta_score_total", "score_total"],
        ascending=[False, False],
    ).iloc[0]

    lines: list[str] = []
    lines.append("=== OPTIMIZE RIDGE FILTER THRESHOLD ===")
    lines.append("")
    lines.append(f"z_entry_threshold = {z_threshold:.2f}")
    lines.append(f"z_exit_threshold = {z_exit_threshold:.2f}")
    lines.append(f"duration_minima = {duration_minima}")
    lines.append(f"dias OOS válidos = {len(oos_df)}")
    lines.append(f"eventos OOS = {int(oos_df['z_event'].sum())}")
    lines.append("")
    lines.append("[Baseline OOS]")
    lines.append(f"trades={baseline_stats['trades']}")
    lines.append(f"win={baseline_stats['win_rate']:.1f}%")
    lines.append(f"rate_move_médio={baseline_stats['rate_move_mean']:.4f}")
    lines.append(f"rate_move_mediana={baseline_stats['rate_move_median']:.4f}")
    lines.append(f"score_médio={baseline_stats['score_mean']:.2f}")
    lines.append(f"score_total={baseline_stats['score_total']:.2f}")
    lines.append(f"holding={baseline_stats['holding']:.1f}")
    lines.append(f"peso={baseline_stats['peso']:.2f}x")
    lines.append("")
    lines.append("=== RESULTADOS DO GRID ===")

    for _, row in results_df.sort_values("ml_threshold").iterrows():
        delta_pct_str = "n/a" if pd.isna(row["delta_score_total_pct"]) else f"{row['delta_score_total_pct']:+.2f}%"
        lines.append(
            f"ml>{row['ml_threshold']:+.2f} | "
            f"trades={int(row['trades'])} | "
            f"win={row['win_rate']:.1f}% | "
            f"rate_médio={row['rate_move_mean']:.4f} | "
            f"mediana={row['rate_move_median']:.4f} | "
            f"score_médio={row['score_mean']:.2f} | "
            f"score_total={row['score_total']:.2f} | "
            f"delta={row['delta_score_total']:+.2f} ({delta_pct_str}) | "
            f"holding={row['holding']:.1f} | "
            f"peso={row['peso']:.2f}x"
        )

    lines.append("")
    lines.append("=== MELHOR POR SCORE TOTAL ===")
    lines.append(
        f"ml>{best_total['ml_threshold']:+.2f} | "
        f"score_total={best_total['score_total']:.2f} | "
        f"score_médio={best_total['score_mean']:.2f} | "
        f"win={best_total['win_rate']:.1f}% | "
        f"trades={int(best_total['trades'])} | "
        f"delta={best_total['delta_score_total']:+.2f}"
    )

    lines.append("")
    lines.append("=== MELHOR POR SCORE MÉDIO ===")
    lines.append(
        f"ml>{best_mean['ml_threshold']:+.2f} | "
        f"score_médio={best_mean['score_mean']:.2f} | "
        f"score_total={best_mean['score_total']:.2f} | "
        f"win={best_mean['win_rate']:.1f}% | "
        f"trades={int(best_mean['trades'])} | "
        f"delta={best_mean['delta_score_total']:+.2f}"
    )

    lines.append("")
    lines.append("=== MELHOR DELTA VS BASELINE ===")
    delta_pct_str = "n/a" if pd.isna(best_delta["delta_score_total_pct"]) else f"{best_delta['delta_score_total_pct']:+.2f}%"
    lines.append(
        f"ml>{best_delta['ml_threshold']:+.2f} | "
        f"delta={best_delta['delta_score_total']:+.2f} ({delta_pct_str}) | "
        f"score_total={best_delta['score_total']:.2f} | "
        f"score_médio={best_delta['score_mean']:.2f} | "
        f"win={best_delta['win_rate']:.1f}% | "
        f"trades={int(best_delta['trades'])}"
    )

    return "\n".join(lines)

def _ridge_position_multiplier(y_pred: float) -> float:
    """
    Converte previsão da Ridge em multiplicador de posição.
    """
    if y_pred <= 0:
        return 0.0
    if y_pred <= 0.5:
        return 0.5
    if y_pred <= 1.0:
        return 1.0
    if y_pred <= 2.0:
        return 1.5
    return 2.0

def _run_ridge_sizing_trade_engine(
    df: pd.DataFrame,
    event_flag_col: str,
    z_col: str = "zscore_rolling_252d",
    rate_col: str = "taxa_media",
    pred_col: str = "y_pred",
    exit_threshold: float = DEFAULT_Z_EXIT,
    duration_minima: int = DEFAULT_DURATION_MINIMA,
    base_notional: float = DEFAULT_BASE_NOTIONAL,
):
    work = df.copy()
    work = work.sort_values("data").reset_index(drop=True)

    _ensure_columns(
        work,
        ["data", event_flag_col, z_col, rate_col, pred_col],
        ctx="_run_ridge_sizing_trade_engine",
    )

    trades = []

    in_position = False
    entry_idx = None
    entry_date = None
    entry_rate = None
    entry_z = None
    entry_weight = None
    entry_pred = None

    for i, row in work.iterrows():
        current_date = row["data"]
        current_z = float(row[z_col])
        current_rate = float(row[rate_col])

        if not in_position:
            if bool(row[event_flag_col]):
                pred = float(row[pred_col])

                weight_z = _cumulative_entry_weight(current_z)
                multiplier = _ridge_position_multiplier(pred)
                final_weight = weight_z * multiplier

                if final_weight <= 0:
                    continue

                in_position = True
                entry_idx = i
                entry_date = current_date
                entry_rate = current_rate
                entry_z = current_z
                entry_weight = float(final_weight)
                entry_pred = pred
                continue

        else:
            holding_days = i - entry_idx

            if holding_days >= duration_minima and current_z <= exit_threshold:
                exit_rate = current_rate
                rate_move = _simple_rate_trade_score(entry_rate, exit_rate)
                pnl_score = base_notional * entry_weight * rate_move

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_z": entry_z,
                        "entry_pred": entry_pred,
                        "entry_rate": entry_rate,
                        "exit_rate": exit_rate,
                        "weight": entry_weight,
                        "rate_move": float(rate_move),
                        "pnl_score": float(pnl_score),
                        "holding_days": int(holding_days),
                    }
                )

                in_position = False
                entry_idx = None
                entry_date = None
                entry_rate = None
                entry_z = None
                entry_weight = None
                entry_pred = None

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        stats = {
            "trades": 0,
            "win_rate": 0.0,
            "rate_move_mean": 0.0,
            "rate_move_median": 0.0,
            "score_total": 0.0,
            "score_mean": 0.0,
            "holding": 0.0,
            "peso": 0.0,
        }
        return trades_df, stats

    stats = {
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["rate_move"] > 0).mean() * 100.0),
        "rate_move_mean": float(trades_df["rate_move"].mean()),
        "rate_move_median": float(trades_df["rate_move"].median()),
        "score_total": float(trades_df["pnl_score"].sum()),
        "score_mean": float(trades_df["pnl_score"].mean()),
        "holding": float(trades_df["holding_days"].mean()),
        "peso": float(trades_df["weight"].mean()),
    }

    return trades_df, stats

def backtest_ridge_position_sizing_overlay() -> str:
    df = _attach_oos_ml_predictions_to_research_frame().copy()

    _ensure_columns(
        df,
        ["data", "zscore_rolling_252d", "taxa_media", "y_pred"],
        ctx="backtest_ridge_position_sizing_overlay",
    )

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df[df["y_pred"].notna()].copy()
    df = df.sort_values("data").reset_index(drop=True)

    df = _mark_threshold_cross_events(
        df,
        signal_col="zscore_rolling_252d",
        threshold=DEFAULT_Z_ENTRY,
        event_col="z_event",
    )

    df["baseline_event"] = df["z_event"]

    baseline_trades, baseline_stats = _run_event_trade_engine(
        df=df,
        event_flag_col="baseline_event",
        z_col="zscore_rolling_252d",
        rate_col="taxa_media",
        exit_threshold=DEFAULT_Z_EXIT,
        duration_minima=DEFAULT_DURATION_MINIMA,
    )

    ridge_trades, ridge_stats = _run_ridge_sizing_trade_engine(
        df=df,
        event_flag_col="z_event",
        z_col="zscore_rolling_252d",
        rate_col="taxa_media",
        pred_col="y_pred",
        exit_threshold=DEFAULT_Z_EXIT,
        duration_minima=DEFAULT_DURATION_MINIMA,
    )

    lines = []
    lines.append("=== RIDGE POSITION SIZING OVERLAY ===")
    lines.append("")
    lines.extend(_stats_to_lines("Baseline OOS", baseline_stats))
    lines.append("")
    lines.extend(_stats_to_lines("Ridge Sizing", ridge_stats))
    lines.append("")

    baseline_total = baseline_stats["score_total"]
    ridge_total = ridge_stats["score_total"]
    delta_abs = ridge_total - baseline_total
    delta_pct = (delta_abs / baseline_total * 100.0) if baseline_total != 0 else np.nan

    lines.append("=== DELTA RIDGE SIZING VS BASELINE ===")
    lines.append(f"delta_score_total = {delta_abs:+.2f}")
    if pd.notna(delta_pct):
        lines.append(f"delta_score_total_pct = {delta_pct:+.2f}%")
    else:
        lines.append("delta_score_total_pct = n/a")
    lines.append("")

    if not ridge_trades.empty:
        lines.append("=== DETALHE DOS TRADES RIDGE SIZING ===")
        for _, row in ridge_trades.iterrows():
            lines.append(
                f"{row['entry_date'].strftime('%d/%m/%Y')} -> {row['exit_date'].strftime('%d/%m/%Y')} | "
                f"z_in={row['entry_z']:.2f} | "
                f"pred={row['entry_pred']:+.3f} | "
                f"rate_move={row['rate_move']:+.4f} | "
                f"score={row['pnl_score']:.2f} | "
                f"holding={row['holding_days']}d | "
                f"peso={row['weight']:.2f}x"
            )

    return "\n".join(lines)