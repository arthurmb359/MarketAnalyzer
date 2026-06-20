from __future__ import annotations

from pathlib import Path
from datetime import datetime, date, timedelta

import pandas as pd
import requests

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


def _read_csv_flexible(csv_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(csv_path, sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Nao foi possivel ler {csv_path}: {last_error}")


def _parse_tesouro_date(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%d/%m/%Y", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
    return parsed


def _read_tesouro_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()

    df = _read_csv_flexible(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "Data Base" not in df.columns:
        raise ValueError(f"{csv_path} nao possui coluna 'Data Base'. Colunas: {list(df.columns)}")

    df["Data Base"] = _parse_tesouro_date(df["Data Base"])

    df = df.dropna(subset=["Data Base"]).copy()
    return df


def get_tesouro_csv_last_date(csv_path: str | Path) -> date | None:
    df = _read_tesouro_csv(Path(csv_path))
    if df.empty:
        return None

    return df["Data Base"].max().date()


def update_tesouro_csv_if_needed(csv_path: str | Path) -> dict:
    """
    Se o CSV bruto do Tesouro nao estiver atualizado ate o ultimo dia util,
    baixa a versao mais nova inteira e sobrescreve o arquivo local.
    """
    csv_path = Path(csv_path)
    target_date = _last_business_day()

    existing = _read_tesouro_csv(csv_path)
    old_last_date = None
    if not existing.empty:
        last_date = existing["Data Base"].max().date()
        old_last_date = str(last_date)
        if last_date >= target_date:
            return {
                "updated": False,
                "path": str(csv_path),
                "last_date": str(last_date),
                "target_date": str(target_date),
                "reason": "up-to-date",
            }

    response = requests.get(TESOURO_CSV_URL, timeout=60)
    response.raise_for_status()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(response.content)

    refreshed = _read_tesouro_csv(csv_path)
    if refreshed.empty:
        raise ValueError("CSV do Tesouro foi baixado, mas ficou vazio apos leitura.")

    return {
        "updated": True,
        "path": str(csv_path),
        "old_last_date": old_last_date,
        "last_date": str(refreshed["Data Base"].max().date()),
        "target_date": str(target_date),
    }


def rebuild_tesouro_ipca(raw_csv_path: str | Path, tesouro_ipca_csv_path: str | Path) -> dict:
    raw_csv_path = Path(raw_csv_path)
    tesouro_ipca_csv_path = Path(tesouro_ipca_csv_path)

    df = _read_csv_flexible(raw_csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"Tipo Titulo", "Data Base", "Data Vencimento"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV bruto nao possui colunas esperadas. Faltando: {sorted(missing)}"
        )

    df["Tipo Titulo"] = df["Tipo Titulo"].astype(str).str.strip()
    df["Data Base"] = _parse_tesouro_date(df["Data Base"])
    df["Data Vencimento"] = _parse_tesouro_date(df["Data Vencimento"])

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
