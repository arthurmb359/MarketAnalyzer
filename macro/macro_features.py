from __future__ import annotations

from pathlib import Path
import pandas as pd

MACRO_DIR = Path("data")


def load_series(csv_path: Path, value_name: str) -> pd.DataFrame:
    # tenta ; primeiro, depois ,
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
        if df.shape[1] < 2:
            df = pd.read_csv(csv_path, sep=",", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, sep=";", encoding="latin1")
        if df.shape[1] < 2:
            df = pd.read_csv(csv_path, sep=",", encoding="latin1")

    df.columns = [str(c).strip() for c in df.columns]

    if len(df.columns) < 2:
        raise ValueError(
            f"{csv_path} precisa ter pelo menos 2 colunas. "
            f"Colunas encontradas: {list(df.columns)}"
        )

    date_col = df.columns[0]
    value_col = df.columns[1]

    df = df[[date_col, value_col]].copy()
    df = df.rename(columns={date_col: "data", value_col: value_name})

    # limpa strings
    df["data"] = df["data"].astype(str).str.strip()
    df[value_name] = df[value_name].astype(str).str.strip()

    # tenta alguns formatos comuns
    parsed = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")

    if parsed.isna().all():
        parsed = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="coerce")

    if parsed.isna().all():
        parsed = pd.to_datetime(df["data"], format="%m/%Y", errors="coerce")

    if parsed.isna().all():
        parsed = pd.to_datetime(df["data"], errors="coerce")

    df["data"] = parsed

    df[value_name] = (
        df[value_name]
        .str.replace(",", ".", regex=False)
    )
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")

    df = df.dropna(subset=["data", value_name]).copy()
    df = df.sort_values("data").reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"{csv_path} ficou vazio após parsing. "
            "Verifique formato da data, separador e cabeçalho."
        )

    return df


def load_macro_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ipca = load_series(MACRO_DIR / "ipca.csv", "ipca_mensal")
    dbgg = load_series(MACRO_DIR / "dbgg.csv", "dbgg")
    usdbrl = load_series(MACRO_DIR / "usdbrl.csv", "usdbrl")

    return ipca, dbgg, usdbrl


def build_daily_macro_frame(
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ipca, dbgg, usdbrl = load_macro_series()

    if ipca.empty:
        raise ValueError("Série IPCA vazia após parsing.")
    if dbgg.empty:
        raise ValueError("Série DBGG vazia após parsing.")
    if usdbrl.empty:
        raise ValueError("Série USD/BRL vazia após parsing.")

    inferred_start = min(ipca["data"].min(), dbgg["data"].min(), usdbrl["data"].min())
    inferred_end = max(ipca["data"].max(), dbgg["data"].max(), usdbrl["data"].max())

    start_date = inferred_start if start_date is None else start_date
    end_date = inferred_end if end_date is None else end_date

    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError(
            f"start_date ou end_date inválido. "
            f"IPCA: {ipca['data'].min()} -> {ipca['data'].max()} | "
            f"DBGG: {dbgg['data'].min()} -> {dbgg['data'].max()} | "
            f"USD/BRL: {usdbrl['data'].min()} -> {usdbrl['data'].max()}"
        )

    daily_index = pd.DataFrame({
        "data": pd.date_range(start=start_date, end=end_date, freq="D")
    })

    df = daily_index.merge(usdbrl, on="data", how="left")
    df = df.merge(ipca, on="data", how="left")
    df = df.merge(dbgg, on="data", how="left")

    df[["usdbrl", "ipca_mensal", "dbgg"]] = df[["usdbrl", "ipca_mensal", "dbgg"]].ffill()

    return df

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # IPCA acumulado em 12 meses
    out["ipca_12m"] = out["ipca_mensal"].rolling(window=12, min_periods=12).sum()

    # Tendência recente de inflação: média 3m menos média 12m/4
    out["ipca_mm3"] = out["ipca_mensal"].rolling(window=3, min_periods=3).mean()
    out["ipca_tendencia"] = out["ipca_mm3"] - (out["ipca_12m"] / 12)

    # Variação anual da dívida/PIB
    out["dbgg_12m_diff"] = out["dbgg"] - out["dbgg"].shift(12)

    # Médias móveis do câmbio
    out["usdbrl_mm50"] = out["usdbrl"].rolling(window=50, min_periods=50).mean()
    out["usdbrl_mm200"] = out["usdbrl"].rolling(window=200, min_periods=200).mean()

    # Distância do câmbio para a MM200
    out["usdbrl_stress"] = (out["usdbrl"] / out["usdbrl_mm200"]) - 1.0

    # Variação do câmbio em 3 meses (~63 dias úteis aproximados em série diária simples)
    out["usdbrl_63d_change"] = out["usdbrl"] / out["usdbrl"].shift(63) - 1.0

    return out

def add_macro_regime_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    score = pd.Series(0, index=out.index, dtype=float)

    # Fiscal
    score += (out["dbgg_12m_diff"] > 2.0).astype(int)
    score += (out["dbgg_12m_diff"] > 4.0).astype(int)

    # Inflação
    score += (out["ipca_12m"] > 5.0).astype(int)
    score += (out["ipca_tendencia"] > 0).astype(int)

    # Câmbio
    score += (out["usdbrl_stress"] > 0.05).astype(int)
    score += (out["usdbrl_63d_change"] > 0.10).astype(int)

    out["macro_regime_score"] = score

    # suavização de 90 dias
    out["macro_score_smooth"] = (
        out["macro_regime_score"]
        .rolling(window=90, min_periods=30)
        .mean()
    )

    # regime baseado no score suavizado
    def classify_smooth(score_value: float) -> str:
        if pd.isna(score_value):
            return "indefinido"
        if score_value < 0.8:
            return "normal"
        if score_value < 1.5:
            return "alerta"
        return "stress"

    out["macro_regime_label"] = out["macro_score_smooth"].apply(classify_smooth)

    return out

    def classify(score_value: float) -> str:
        if score_value <= 1:
            return "normal"
        if score_value <= 3:
            return "alerta"
        return "stress"

    out["macro_regime_label"] = out["macro_regime_score"].apply(classify)

    return out