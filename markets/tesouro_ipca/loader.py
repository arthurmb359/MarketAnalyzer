from pathlib import Path

import pandas as pd
from pandas.api.types import is_string_dtype


def load_tesouro_ipca_frame(csv_path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path("data/tesouro_ipca.csv") if csv_path is None else Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {csv_path.resolve()}\n"
            "Gere ou atualize o CSV canonico em data/tesouro_ipca.csv."
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
            f"O CSV nao possui as colunas esperadas. Faltando: {sorted(missing)}"
        )

    df["Data Base"] = pd.to_datetime(df["Data Base"], errors="coerce")
    df["Data Vencimento"] = pd.to_datetime(df["Data Vencimento"], errors="coerce")

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


def load_ipca_long_research_frame(
    csv_path: str | Path | None = None,
    duration_minima: float = 0.0,
) -> pd.DataFrame:
    df = load_tesouro_ipca_frame(csv_path).copy()

    if "Tipo Titulo" in df.columns:
        df["Tipo Titulo"] = df["Tipo Titulo"].astype(str).str.strip()
        df = df[df["Tipo Titulo"] == "Tesouro IPCA+"].copy()

    if df.empty:
        raise ValueError("Nenhuma linha de 'Tesouro IPCA+' encontrada no CSV canonico.")

    if duration_minima > 0:
        df = df[df["Prazo_anos"] >= duration_minima].copy()

    if df.empty:
        raise ValueError(
            f"Nenhum titulo restou apos aplicar duration_minima={duration_minima}."
        )

    idx = df.groupby("Data Base")["Prazo_anos"].idxmax()
    daily = df.loc[idx].copy()
    daily = daily.sort_values(["Data Base", "Data Vencimento"]).copy()
    daily = daily.sort_values("Data Base").reset_index(drop=True)

    daily = daily.rename(
        columns={
            "Data Base": "data",
            "Taxa Compra Manha": "taxa_media",
            "Prazo_anos": "prazo_anos",
            "Data Vencimento": "data_vencimento",
        }
    )

    daily = daily[["data", "taxa_media", "prazo_anos", "data_vencimento"]].copy()

    if daily.empty:
        raise ValueError("A serie diaria ficou vazia apos selecionar o maior prazo por dia.")

    rolling_window = min(252, len(daily))
    min_periods = min(60, rolling_window)

    if rolling_window <= 0:
        raise ValueError("rolling_window ficou invalido; verifique a serie diaria.")

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


__all__ = ["load_ipca_long_research_frame", "load_tesouro_ipca_frame"]
