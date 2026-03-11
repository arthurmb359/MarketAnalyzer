from pathlib import Path
import pandas as pd


CSV_PATH = Path("data/ipca_principal.csv")


def load_ipca_long_series(
    csv_path: Path = CSV_PATH,
    duration_minima: float = 0.0,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {csv_path.resolve()}\n"
            "Coloque o CSV em /data ou ajuste CSV_PATH em ipca_series.py."
        )

    try:
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, sep=",", encoding="latin1")

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

    df["Data Base"] = df["Data Base"].astype(str).str.strip()
    df["Data Vencimento"] = df["Data Vencimento"].astype(str).str.strip()
    df["Taxa Compra Manha"] = df["Taxa Compra Manha"].astype(str).str.strip()
    df["Prazo_anos"] = df["Prazo_anos"].astype(str).str.strip()

    data_base = pd.to_datetime(df["Data Base"], format="%Y-%m-%d", errors="coerce")
    if data_base.isna().all():
        data_base = pd.to_datetime(df["Data Base"], errors="coerce")

    data_venc = pd.to_datetime(df["Data Vencimento"], format="%Y-%m-%d", errors="coerce")
    if data_venc.isna().all():
        data_venc = pd.to_datetime(df["Data Vencimento"], errors="coerce")

    df["Data Base"] = data_base
    df["Data Vencimento"] = data_venc

    taxa = (
        df["Taxa Compra Manha"]
        .str.replace(",", ".", regex=False)
        .str.replace('"', "", regex=False)
    )
    df["Taxa Compra Manha"] = pd.to_numeric(taxa, errors="coerce")

    prazo = (
        df["Prazo_anos"]
        .str.replace(",", ".", regex=False)
        .str.replace('"', "", regex=False)
    )
    df["Prazo_anos"] = pd.to_numeric(prazo, errors="coerce")

    df = df.dropna(
        subset=["Data Base", "Data Vencimento", "Taxa Compra Manha", "Prazo_anos"]
    ).copy()

    if df.empty:
        raise ValueError(
            "O dataframe ficou vazio após parsing do CSV de IPCA principal."
        )

    # filtra pela duration mínima
    df = df[df["Prazo_anos"] >= duration_minima].copy()

    if df.empty:
        raise ValueError(
            f"Nenhum título restou após aplicar duration_minima={duration_minima}."
        )

    # dentro de cada dia, pega o título de maior prazo
    df = df.sort_values(["Data Base", "Prazo_anos", "Data Vencimento"])
    daily = df.groupby("Data Base").tail(1).copy()

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
    ].sort_values("data").reset_index(drop=True)

    if daily.empty:
        raise ValueError(
            "A série diária ficou vazia após selecionar o maior vencimento por dia."
        )

    rolling_window = min(252, len(daily))
    min_periods = min(60, rolling_window)

    if rolling_window == 0:
        raise ValueError(
            "rolling_window ficou 0; verifique se a série diária foi carregada corretamente."
        )

    daily["media_rolling_5a"] = daily["taxa_media"].rolling(
        window=rolling_window,
        min_periods=min_periods,
    ).mean()

    daily["desvio_rolling_5a"] = daily["taxa_media"].rolling(
        window=rolling_window,
        min_periods=min_periods,
    ).std()

    daily["zscore_rolling_5a"] = (
        (daily["taxa_media"] - daily["media_rolling_5a"]) /
        daily["desvio_rolling_5a"]
    )

    return daily