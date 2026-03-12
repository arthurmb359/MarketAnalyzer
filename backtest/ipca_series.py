from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/tesouro_ipca.csv")


def load_ipca_long_series(
    csv_path: Path = CSV_PATH,
    duration_minima: float = 0.0,
) -> pd.DataFrame:
    """
    Carrega a série diária do Tesouro IPCA+ longo.

    Regras:
    - usa apenas o CSV já filtrado de Tesouro IPCA+
    - aplica filtro de duration_minima em Prazo_anos
    - dentro de cada Data Base, mantém o título de maior prazo
    - calcula média/desvio rolling 252d e z-score rolling 252d
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {csv_path.resolve()}\n"
            "Gere/atualize o tesouro_ipca.csv antes de rodar."
        )

    try:
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, sep=",", encoding="latin1")

    df.columns = [str(c).strip() for c in df.columns]

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

    # -----------------------------
    # Parsing vetorizado
    # -----------------------------
    df["Data Base"] = pd.to_datetime(df["Data Base"], errors="coerce")
    df["Data Vencimento"] = pd.to_datetime(df["Data Vencimento"], errors="coerce")

    df["Taxa Compra Manha"] = (
        df["Taxa Compra Manha"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace('"', "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    df["Prazo_anos"] = (
        df["Prazo_anos"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace('"', "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    df["Tipo Titulo"] = df["Tipo Titulo"].astype(str).str.strip()

    # -----------------------------
    # Limpeza
    # -----------------------------
    df = df.dropna(
        subset=["Data Base", "Data Vencimento", "Taxa Compra Manha", "Prazo_anos"]
    ).copy()

    if df.empty:
        raise ValueError("O dataframe ficou vazio após parsing do CSV filtrado.")

    # segurança extra: mantém só Tesouro IPCA+
    df = df[df["Tipo Titulo"] == "Tesouro IPCA+"].copy()

    if df.empty:
        raise ValueError("Nenhuma linha de 'Tesouro IPCA+' encontrada no CSV filtrado.")

    # filtro por duration
    if duration_minima > 0:
        df = df[df["Prazo_anos"] >= duration_minima].copy()

    if df.empty:
        raise ValueError(
            f"Nenhum título restou após aplicar duration_minima={duration_minima}."
        )

    # -----------------------------
    # Seleção eficiente do título mais longo por dia
    # -----------------------------
    # Em vez de sort + groupby.tail(1), usamos idxmax por Data Base.
    idx = df.groupby("Data Base")["Prazo_anos"].idxmax()
    daily = df.loc[idx].copy()

    # desempate estável por vencimento, se quiser consistência visual
    daily = daily.sort_values(["Data Base", "Data Vencimento"]).copy()

    # como idxmax já escolheu 1 linha por Data Base, agora só ordenamos por data
    daily = daily.sort_values("Data Base").reset_index(drop=True)

    daily = daily.rename(
        columns={
            "Data Base": "data",
            "Taxa Compra Manha": "taxa_media",
            "Prazo_anos": "prazo_anos",
            "Data Vencimento": "data_vencimento",
        }
    )

    daily = daily[
        ["data", "taxa_media", "prazo_anos", "data_vencimento"]
    ].copy()

    if daily.empty:
        raise ValueError("A série diária ficou vazia após selecionar o maior prazo por dia.")

    # -----------------------------
    # Estatísticas rolling 252d
    # -----------------------------
    rolling_window = min(252, len(daily))
    min_periods = min(60, rolling_window)

    if rolling_window <= 0:
        raise ValueError("rolling_window ficou inválido; verifique a série diária.")

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