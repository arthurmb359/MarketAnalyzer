from __future__ import annotations

from backtest.registry import BacktestRegistry
from backtest.ui import MarketAnalyzerWindow
from pathlib import Path
import sys
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_string_dtype
from backtest.algorithms import *
from backtest.algorithms_ml import *
from data_updater.tesouro_updater import *
from data_updater.update_config import *


CSV_PATH = Path("data/tesouro_ipca.csv")

def main() -> int:
    try:
        bootstrap_tesouro_updates()
        df = load_data(CSV_PATH)
        daily = build_daily_series(df)
        app_registry = create_backtest_registry()

        app = MarketAnalyzerWindow(daily=daily, registry=app_registry)
        app.run()

        return 0

    except Exception as exc:
        print(f"Erro ao executar analise: {exc}", file=sys.stderr)
        return 1
    
def bootstrap_tesouro_updates() -> None:
    series_name = "tesouro_ipca"
    raw_csv = "data/precotaxatesourodireto.csv"
    tesouro_ipca_csv = "data/tesouro_ipca.csv"

    print("=== Atualização Tesouro Direto ===")

    if was_updated_today(series_name):
        print("[SKIP] atualização do Tesouro já foi tentada hoje")

        if not Path(tesouro_ipca_csv).exists():
            rebuilt = rebuild_tesouro_ipca(raw_csv, tesouro_ipca_csv)
            print(
                f"[OK] tesouro_ipca.csv regenerado com {rebuilt['rows']} linhas "
                f"({rebuilt['start_date']} -> {rebuilt['end_date']})"
            )
        else:
            print("[SKIP] tesouro_ipca.csv já existe")
        return

    try:
        result = update_tesouro_csv_if_needed(raw_csv)

        if result["updated"]:
            print(
                f"[OK] bruto atualizado de {result.get('old_last_date')} "
                f"para {result['last_date']}"
            )
        else:
            print(
                f"[SKIP] bruto já está no snapshot esperado "
                f"(last={result['last_date']}, target={result['target_date']}, reason={result['reason']})"
            )

        if result["updated"] or (not Path(tesouro_ipca_csv).exists()):
            rebuilt = rebuild_tesouro_ipca(raw_csv, tesouro_ipca_csv)
            print(
                f"[OK] tesouro_ipca.csv regenerado com {rebuilt['rows']} linhas "
                f"({rebuilt['start_date']} -> {rebuilt['end_date']})"
            )
        else:
            print("[SKIP] tesouro_ipca.csv já estava consistente com o bruto")

    finally:
        # marca que a tentativa foi feita hoje, mesmo se não houve update novo
        mark_updated_today(series_name)


def create_backtest_registry() -> BacktestRegistry:
    app_registry = BacktestRegistry()

    app_registry.register(
        "Optimize Entry Threshold Fine",
        backtest_optimize_entry_threshold_fine,
    )
    # app_registry.register("FX Trend Vol Regime", backtest_fx_trend_vol_regime)
    # app_registry.register("FX Regime Event Sensitivity",backtest_fx_regime_event_sensitivity)
    # app_registry.register("FX Forward Returns", backtest_fx_forward_returns)
    # app_registry.register("Real Rate Trade by FX Regime",backtest_realrate_trade_by_fx_regime)
    # app_registry.register("Real Rate Trade by FX Regime Detail", backtest_realrate_trade_fx_regime_detail)
    # app_registry.register("Real Rate Signal Validity by FX Regime", backtest_realrate_signal_validity_by_fx_regime)
    app_registry.register(
        "Real Rate Non-Optimal Entry Plan",
        backtest_realrate_non_optimal_entry_plan,
    )
    app_registry.register("IPCA+ State of Art",backtest_realrate_state_of_art)

    #Algoritmos de ML, por enquanto pausados.
    #app_registry.register("Ridge Position Sizing Overlay",backtest_ridge_position_sizing_overlay)
    #app_registry.register("Supervised Dataset Preview", backtest_build_supervised_dataset_preview)
    # app_registry.register("Ridge Forward Return 120d", backtest_ridge_forward_return_120d)
    # app_registry.register("Ridge Feature Importance", backtest_ridge_feature_importance)
    # app_registry.register("Baseline vs Ridge Filter", backtest_compare_baseline_vs_ridge_filter)
    # app_registry.register("Optimize Ridge Filter Threshold", backtest_optimize_ridge_filter_threshold)
    return app_registry

def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {csv_path.resolve()}\n"
            "Coloque o CSV na raiz do projeto ou ajuste a constante CSV_PATH."
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "Tipo Titulo",
        "Data Vencimento",
        "Data Base",
        "Taxa Compra Manha",
        "Prazo_anos",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"O CSV não possui as colunas esperadas. Faltando: {sorted(missing)}"
        )

    df["Data Base"] = pd.to_datetime(df["Data Base"], errors="coerce")
    df["Data Vencimento"] = pd.to_datetime(df["Data Vencimento"], errors="coerce")

    # A coluna pode vir como string com vírgula decimal dependendo da exportação.
    if is_string_dtype(df["Taxa Compra Manha"]):
        df["Taxa Compra Manha"] = (
            df["Taxa Compra Manha"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )

    df["Taxa Compra Manha"] = pd.to_numeric(
        df["Taxa Compra Manha"],
        errors="coerce",
    )

    df["Prazo_anos"] = pd.to_numeric(df["Prazo_anos"], errors="coerce")

    df = df.dropna(
        subset=[
            "Data Base",
            "Data Vencimento",
            "Taxa Compra Manha",
            "Prazo_anos",
        ]
    ).copy()

    if df.empty:
        raise ValueError(
            "Nenhuma linha valida apos limpeza do CSV. "
            "Verifique formato de datas, separador decimal e valores ausentes."
        )

    df = df.sort_values(["Data Base", "Data Vencimento"]).reset_index(drop=True)
    return df

def estimate_bond_return(
    rate_change: float,
    duration: float,
) -> float:
    """
    Aproximação linear simples:
    retorno (%) ≈ - duration × variação da taxa

    Exemplo:
    taxa cai 1.0 ponto percentual (-1.0)
    duration = 12
    retorno ≈ 12%
    """
    return -duration * rate_change


def analyze_expected_bond_returns(
    daily: pd.DataFrame,
    z_thresholds: list[float] | None = None,
    horizons: list[int] | None = None,
    durations: list[float] | None = None,
) -> None:
    if z_thresholds is None:
        z_thresholds = [0.5, 1.0, 1.5, 2.0]

    if horizons is None:
        horizons = [90, 180, 252]

    if durations is None:
        durations = [8.0, 10.0, 12.0, 15.0]

    print("\n" + "=" * 60)
    print("RETORNO ESPERADO APROXIMADO DA NTN-B LONGA")
    print("=" * 60)
    print(
        "Leitura:\n"
        "- usamos apenas eventos únicos (primeiro cruzamento do limiar)\n"
        "- retorno aproximado do título = -duration × variação da taxa\n"
        "- aproximação linear, sem convexidade e sem cupom\n"
    )

    for z_threshold in z_thresholds:
        df = mark_threshold_cross_events(daily, z_threshold)
        event_col = f"event_z_gt_{str(z_threshold).replace('.', '_')}"
        subset = df[df[event_col]].copy()

        print(f"\n=== Z-SCORE > {z_threshold:.2f} ===")
        print(f"Eventos únicos: {len(subset)}")

        if len(subset) == 0:
            print("Nenhum evento encontrado.")
            continue

        for h in horizons:
            col = f"forward_{h}d"
            valid = subset[col].dropna()

            if len(valid) == 0:
                print(f"{h}d -> sem dados suficientes")
                continue

            mean_rate_change = valid.mean()

            print(
                f"\nHorizonte {h} dias | "
                f"variação média da taxa: {mean_rate_change:+.3f} p.p."
            )

            for d in durations:
                est_return = estimate_bond_return(mean_rate_change, d)
                print(
                    f"  Duration {d:>4.1f} -> retorno aprox.: {est_return:+.2f}%"
                )


def analyze_expected_bond_returns_generic(
    daily: pd.DataFrame,
    signal_col: str,
    thresholds: list[float] | None = None,
    horizons: list[int] | None = None,
    durations: list[float] | None = None,
) -> None:
    if thresholds is None:
        thresholds = [0.5, 1.0, 1.5, 2.0]

    if horizons is None:
        horizons = [90, 180, 252]

    if durations is None:
        durations = [8.0, 10.0, 12.0, 15.0]

    print("\n" + "=" * 60)
    print(f"RETORNO ESPERADO APROXIMADO DA NTN-B LONGA - {signal_col}")
    print("=" * 60)
    print(
        "Leitura:\n"
        "- usamos apenas eventos únicos (primeiro cruzamento do limiar)\n"
        "- retorno aproximado do título = -duration × variação da taxa\n"
        "- aproximação linear, sem convexidade e sem cupom\n"
    )

    for threshold in thresholds:
        df = mark_threshold_cross_events_generic(daily, signal_col, threshold)
        event_col = f"event_{signal_col}_gt_{str(threshold).replace('.', '_')}"
        subset = df[df[event_col]].copy()

        print(f"\n=== {signal_col} > {threshold:.2f} ===")
        print(f"Eventos únicos: {len(subset)}")

        if len(subset) == 0:
            print("Nenhum evento encontrado.")
            continue

        for h in horizons:
            col = f"forward_{h}d"
            valid = subset[col].dropna()

            if len(valid) == 0:
                print(f"{h}d -> sem dados suficientes")
                continue

            mean_rate_change = valid.mean()

            print(
                f"\nHorizonte {h} dias | "
                f"variação média da taxa: {mean_rate_change:+.3f} p.p."
            )

            for d in durations:
                est_return = estimate_bond_return(mean_rate_change, d)
                print(f"  Duration {d:>4.1f} -> retorno aprox.: {est_return:+.2f}%")


def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria série diária usando o título com maior vencimento do dia.
    Também calcula estatísticas históricas e rolling z-score de 252 dias.
    """

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

    # Estatísticas de histórico completo
    mean_value = daily["taxa_media"].mean()
    std_value = daily["taxa_media"].std()

    daily["media_historica"] = mean_value
    daily["desvio_padrao"] = std_value
    daily["zscore"] = (daily["taxa_media"] - mean_value) / std_value

    # Percentil histórico expanding
    percentis = []
    values = daily["taxa_media"].tolist()

    for i, value in enumerate(values):
        history = values[: i + 1]
        percentil = sum(v <= value for v in history) / len(history)
        percentis.append(percentil)

    daily["percentil_historico"] = percentis

    # Médias móveis
    daily["mm_252"] = daily["taxa_media"].rolling(252, min_periods=30).mean()
    daily["mm_1260"] = daily["taxa_media"].rolling(1260, min_periods=60).mean()

    # Bandas históricas fixas
    daily["banda_1dp_sup"] = mean_value + std_value
    daily["banda_1dp_inf"] = mean_value - std_value
    daily["banda_2dp_sup"] = mean_value + 2 * std_value
    daily["banda_2dp_inf"] = mean_value - 2 * std_value

    # Rolling z-score de 252 dias, alinhado ao backtest
    rolling_window = min(252, len(daily))
    min_periods = min(60, rolling_window)

    daily["media_rolling_252d"] = daily["taxa_media"].rolling(
        window=rolling_window,
        min_periods=min_periods
    ).mean()

    daily["desvio_rolling_252d"] = daily["taxa_media"].rolling(
        window=rolling_window,
        min_periods=min_periods
    ).std()

    daily["zscore_rolling_252d"] = (
        (daily["taxa_media"] - daily["media_rolling_252d"]) /
        daily["desvio_rolling_252d"]
    )

    return daily


def classify_stretch(zscore: float) -> str:
    abs_z = abs(zscore)

    if abs_z < 1:
        return "normal"
    if abs_z < 2:
        return "esticado"
    if abs_z < 3:
        return "extremo"
    return "raríssimo"

def mark_threshold_cross_events(
    daily: pd.DataFrame,
    z_threshold: float,
) -> pd.DataFrame:
    """
    Marca apenas o primeiro dia em que o z-score cruza acima do limiar.
    Isso remove eventos consecutivos do mesmo regime.
    """
    df = daily.copy()

    above = df["zscore"] > z_threshold
    above_prev = above.shift(1, fill_value=False)

    # Evento ocorre quando hoje está acima do limiar
    # e ontem não estava.
    event_col = f"event_z_gt_{str(z_threshold).replace('.', '_')}"
    df[event_col] = above & (~above_prev)

    return df


def analyze_threshold(
    daily: pd.DataFrame,
    z_threshold: float,
    horizons: list[int] | None = None,
) -> None:
    if horizons is None:
        horizons = [30, 90, 180, 252]

    df = mark_threshold_cross_events(daily, z_threshold)

    event_col = f"event_z_gt_{str(z_threshold).replace('.', '_')}"
    subset = df[df[event_col]].copy()

    print(f"\n=== EVENT STUDY: z-score > {z_threshold:.2f} ===")
    print(f"Quantidade de eventos únicos: {len(subset)}")

    if len(subset) == 0:
        print("Nenhum evento encontrado.")
        return

    for h in horizons:
        col = f"forward_{h}d"
        valid = subset[col].dropna()

        if len(valid) == 0:
            print(f"{h}d -> sem dados suficientes")
            continue

        mean_move = valid.mean()
        median_move = valid.median()
        hit_ratio = (valid < 0).mean() * 100

        print(
            f"{h:>3}d | média: {mean_move:+.3f} | "
            f"mediana: {median_move:+.3f} | "
            f"taxa caiu depois: {hit_ratio:5.1f}%"
        )

def analyze_threshold_generic(
    daily: pd.DataFrame,
    signal_col: str,
    threshold: float,
    horizons: list[int] | None = None,
) -> None:
    if horizons is None:
        horizons = [30, 90, 180, 252]

    df = mark_threshold_cross_events_generic(daily, signal_col, threshold)
    event_col = f"event_{signal_col}_gt_{str(threshold).replace('.', '_')}"
    subset = df[df[event_col]].copy()

    print(f"\n=== EVENT STUDY: {signal_col} > {threshold:.2f} ===")
    print(f"Quantidade de eventos únicos: {len(subset)}")

    if len(subset) == 0:
        print("Nenhum evento encontrado.")
        return

    for h in horizons:
        col = f"forward_{h}d"
        valid = subset[col].dropna()

        if len(valid) == 0:
            print(f"{h}d -> sem dados suficientes")
            continue

        mean_move = valid.mean()
        median_move = valid.median()
        hit_ratio = (valid < 0).mean() * 100

        print(
            f"{h:>3}d | média: {mean_move:+.3f} | "
            f"mediana: {median_move:+.3f} | "
            f"taxa caiu depois: {hit_ratio:5.1f}%"
        )

def mark_threshold_cross_events_generic(
    daily: pd.DataFrame,
    signal_col: str,
    threshold: float,
) -> pd.DataFrame:
    """
    Marca apenas o primeiro dia em que a coluna cruza acima do limiar.
    Funciona para zscore histórico ou rolling.
    """
    df = daily.copy()

    above = df[signal_col] > threshold
    above_prev = above.shift(1, fill_value=False)

    event_col = f"event_{signal_col}_gt_{str(threshold).replace('.', '_')}"
    df[event_col] = above & (~above_prev)

    return df

def run_event_studies(daily: pd.DataFrame) -> None:
    """
    Roda estudos para alguns limiares de z-score.
    """
    thresholds = [0.5, 1.0, 1.5, 2.0]

    print("\n" + "=" * 60)
    print("ESTUDO DE REVERSÃO APÓS ESTICAMENTO")
    print("=" * 60)
    print(
        "Interpretação:\n"
        "- média negativa  => taxa caiu depois => sinal favorável à compressão\n"
        "- média positiva  => taxa subiu depois => sinal desfavorável\n"
    )

    for threshold in thresholds:
        analyze_threshold(daily, threshold)

def run_event_studies_rolling(daily: pd.DataFrame) -> None:
    thresholds = [0.5, 1.0, 1.5, 2.0]

    print("\n" + "=" * 60)
    print("ESTUDO DE REVERSÃO APÓS ESTICAMENTO - ROLLING Z-SCORE 252D")
    print("=" * 60)
    print(
        "Interpretação:\n"
        "- média negativa  => taxa caiu depois => sinal favorável à compressão\n"
        "- média positiva  => taxa subiu depois => sinal desfavorável\n"
    )

    for threshold in thresholds:
        analyze_threshold_generic(
            daily=daily,
            signal_col="zscore_rolling_252d",
            threshold=threshold,
        )

def print_summary(daily: pd.DataFrame) -> None:
    last = daily.iloc[-1]

    current_rate = last["taxa_media"]
    mean_value = last["media_historica"]
    std_value = last["desvio_padrao"]
    zscore = last["zscore"]
    rolling_z = last["zscore_rolling_252d"]
    percentil = last["percentil_historico"] * 100
    stretch = classify_stretch(zscore)

    print("\n=== RESUMO ATUAL DA SÉRIE IPCA+ LONGO ===")
    print(f"Última data:           {last['Data Base'].date()}")
    print(f"Taxa média atual:      {current_rate:.2f}%")
    print(f"Média histórica:       {mean_value:.2f}%")
    print(f"Desvio padrão:         {std_value:.2f}")
    print(f"Z-score histórico:     {zscore:.2f}")
    print(f"Z-score rolling 252d:  {rolling_z:.2f}")
    print(f"Percentil histórico:   {percentil:.2f}%")
    print(f"Classificação:         {stretch}")

    if pd.notna(rolling_z):
        if rolling_z > 1.5:
            print("\nLeitura rolling: taxa bem acima do regime recente.")
        elif rolling_z > 1.0:
            print("\nLeitura rolling: taxa esticada em relação aos últimos 252 dias úteis.")
        elif rolling_z > 0.5:
            print("\nLeitura rolling: taxa levemente acima do regime recente.")
        else:
            print("\nLeitura rolling: taxa dentro de faixa normal do regime recente.")


def plot_series(daily: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    views = ["taxa", "zscore"]
    state = {"index": 0}

    fig, ax = plt.subplots(figsize=(16, 8))

    def draw_current_view() -> None:
        ax.clear()
        date_format = mdates.DateFormatter("%d/%m/%Y")
        ax.xaxis.set_major_formatter(date_format)

        current_view = views[state["index"]]

        if current_view == "taxa":
            ax.plot(
                daily["Data Base"],
                daily["taxa_media"],
                label="Taxa média diária IPCA+",
                linewidth=1.5,
            )
            ax.plot(
                daily["Data Base"],
                daily["media_historica"],
                label="Média histórica",
                linestyle="--",
                linewidth=1.2,
            )
            ax.plot(
                daily["Data Base"],
                daily["media_rolling_252d"],
                label="Média rolling 252d",
                linewidth=1.2,
            )
            ax.plot(
                daily["Data Base"],
                daily["mm_252"],
                label="Média móvel rolling 252d",
                linewidth=1.2,
            )
            ax.plot(
                daily["Data Base"],
                daily["mm_1260"],
                label="Média móvel 1260d",
                linewidth=1.2,
            )
            ax.plot(
                daily["Data Base"],
                daily["banda_1dp_sup"],
                label="+1 desvio",
                linestyle=":",
                linewidth=1.0,
            )
            ax.plot(
                daily["Data Base"],
                daily["banda_1dp_inf"],
                label="-1 desvio",
                linestyle=":",
                linewidth=1.0,
            )
            ax.plot(
                daily["Data Base"],
                daily["banda_2dp_sup"],
                label="+2 desvios",
                linestyle=":",
                linewidth=1.0,
            )
            ax.plot(
                daily["Data Base"],
                daily["banda_2dp_inf"],
                label="-2 desvios",
                linestyle=":",
                linewidth=1.0,
            )

            ax.set_title("IPCA+ Longo - Série Histórica da Taxa Média")
            ax.set_xlabel("Data")
            ax.set_ylabel("Taxa Real (%)")

        elif current_view == "zscore":
            ax.plot(
                daily["Data Base"],
                daily["zscore"],
                label="Z-score histórico",
                linewidth=1.2,
            )
            ax.plot(
                daily["Data Base"],
                daily["zscore_rolling_252d"],
                label="Z-score rolling 252d",
                linewidth=1.2,
            )
            ax.axhline(0, linestyle="--", linewidth=1.0)
            ax.axhline(1, linestyle=":", linewidth=1.0)
            ax.axhline(2, linestyle=":", linewidth=1.0)
            ax.axhline(-1, linestyle=":", linewidth=1.0)
            ax.axhline(-2, linestyle=":", linewidth=1.0)

            ax.set_title("IPCA+ Longo - Z-score Histórico vs Rolling 252d")
            ax.set_xlabel("Data")
            ax.set_ylabel("Z-score")

        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key == "right":
            state["index"] = (state["index"] + 1) % len(views)
            draw_current_view()
        elif event.key == "left":
            state["index"] = (state["index"] - 1) % len(views)
            draw_current_view()

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Use as setas ← e → para alternar entre as views.")
    draw_current_view()
    plt.show()

if __name__ == "__main__":
    raise SystemExit(main())
