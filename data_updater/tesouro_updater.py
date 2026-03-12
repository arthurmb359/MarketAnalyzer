from __future__ import annotations

from pathlib import Path
from datetime import datetime, date, timedelta
import requests
import pandas as pd

TESOURO_CSV_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/"
    "precotaxatesourodireto.csv"
)


def _today_brazil() -> date:
    return datetime.now().date()


def _last_business_day(ref: date | None = None) -> date:
    ref = ref or _today_brazil()
    d = ref
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _read_tesouro_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, sep=";", encoding="latin1")

    df.columns = [str(c).strip() for c in df.columns]

    if "Data Base" not in df.columns:
        raise ValueError(f"{csv_path} não possui coluna 'Data Base'. Colunas: {list(df.columns)}")

    df["Data Base"] = pd.to_datetime(df["Data Base"], format="%d/%m/%Y", errors="coerce")
    if df["Data Base"].isna().all():
        df["Data Base"] = pd.to_datetime(df["Data Base"], errors="coerce", dayfirst=True)

    df = df.dropna(subset=["Data Base"]).copy()
    return df


def update_tesouro_csv_if_needed(csv_path: str | Path) -> dict:
    """
    Se o CSV bruto do Tesouro não estiver atualizado até o último dia útil,
    baixa a versão mais nova inteira e sobrescreve o arquivo local.
    """
    csv_path = Path(csv_path)
    target_date = _last_business_day()

    existing = _read_tesouro_csv(csv_path)
    if not existing.empty:
        last_date = existing["Data Base"].max().date()
        if last_date >= target_date:
            return {
                "updated": False,
                "path": str(csv_path),
                "last_date": str(last_date),
                "target_date": str(target_date),
            }

    response = requests.get(TESOURO_CSV_URL, timeout=60)
    response.raise_for_status()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(response.content)

    refreshed = _read_tesouro_csv(csv_path)
    if refreshed.empty:
        raise ValueError("CSV do Tesouro foi baixado, mas ficou vazio após leitura.")

    return {
        "updated": True,
        "path": str(csv_path),
        "last_date": str(refreshed["Data Base"].max().date()),
        "target_date": str(target_date),
    }


def rebuild_tesouro_ipca(raw_csv_path: str | Path, tesouro_ipca_csv_path: str | Path) -> dict:
    raw_csv_path = Path(raw_csv_path)
    tesouro_ipca_csv_path = Path(tesouro_ipca_csv_path)

    try:
        df = pd.read_csv(raw_csv_path, sep=";", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(raw_csv_path, sep=";", encoding="latin1")

    df.columns = [c.strip() for c in df.columns]

    required = {"Tipo Titulo", "Data Base", "Data Vencimento"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV bruto não possui colunas esperadas. Faltando: {sorted(missing)}"
        )

    df["Tipo Titulo"] = df["Tipo Titulo"].astype(str).str.strip()
    df["Data Base"] = pd.to_datetime(df["Data Base"], format="%d/%m/%Y", errors="coerce")
    df["Data Vencimento"] = pd.to_datetime(
        df["Data Vencimento"], format="%d/%m/%Y", errors="coerce"
    )

    df = df.dropna(subset=["Tipo Titulo", "Data Base", "Data Vencimento"]).copy()

    filtered = df[df["Tipo Titulo"] == "Tesouro IPCA+"].copy()

    filtered["Prazo_anos"] = (
        (filtered["Data Vencimento"] - filtered["Data Base"]).dt.days / 365.25
    )

    filtered = filtered.sort_values(
        ["Data Base", "Data Vencimento"]
    ).reset_index(drop=True)

    tesouro_ipca_csv_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(
        tesouro_ipca_csv_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    return {
        "path": str(tesouro_ipca_csv_path),
        "rows": int(len(filtered)),
        "start_date": filtered["Data Base"].min().strftime("%Y-%m-%d"),
        "end_date": filtered["Data Base"].max().strftime("%Y-%m-%d"),
    }