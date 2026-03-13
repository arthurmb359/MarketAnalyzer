from pathlib import Path

import pandas as pd


def _normalize_usdbrl_value(value: object) -> float:
    text = str(value).strip()

    if not text:
        return float("nan")

    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        text = text.replace(".", "").replace(",", ".")
    elif has_comma:
        text = text.replace(",", ".")

    return pd.to_numeric(text, errors="coerce")


def load_usdbrl_frame(csv_path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path("data/usdbrl.csv") if csv_path is None else Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo FX nao encontrado: {csv_path.resolve()}")

    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(csv_path, sep=";", encoding="latin1")
        except Exception:
            df = pd.read_csv(csv_path, sep=",", encoding="utf-8")

    df.columns = [str(c).strip() for c in df.columns]

    if "data" not in df.columns and "data;valor" in df.columns:
        split_df = df["data;valor"].astype(str).str.split(";", n=1, expand=True)
        split_df.columns = ["data", "usdbrl"]
        df = split_df

    if "data" not in df.columns:
        raise ValueError(f"Base FX sem coluna 'data'. Colunas: {list(df.columns)}")

    if "usdbrl" not in df.columns:
        if "valor" in df.columns:
            df = df.rename(columns={"valor": "usdbrl"})
        else:
            raise ValueError(
                f"Base FX sem coluna 'usdbrl' nem 'valor'. Colunas: {list(df.columns)}"
            )

    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
    if df["data"].isna().all():
        df["data"] = pd.to_datetime(df["data"], errors="coerce", dayfirst=True)

    df["usdbrl"] = df["usdbrl"].apply(_normalize_usdbrl_value)

    df = df.dropna(subset=["data", "usdbrl"]).copy()
    df = df.sort_values("data").drop_duplicates(subset=["data"], keep="last")
    df = df.reset_index(drop=True)

    return df[["data", "usdbrl"]]


__all__ = ["load_usdbrl_frame"]
