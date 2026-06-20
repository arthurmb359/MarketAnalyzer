from pathlib import Path
import io
import re

import pandas as pd
import requests

from core.features import rolling_zscore
from core.reporting import format_date, format_value, render_result, result as build_result, section
from markets.tesouro_ipca.loader import load_tesouro_ipca_frame
from markets.tesouro_ipca.series import build_daily_ipca_duration_bucket_series


INVESTIR_CSV_URL = (
    "https://www.tesourodireto.com.br/documents/d/guest/"
    "rendimento-investir-csv?download=true"
)
TESOURO_SALES_CSV_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "f0468ecc-ae97-4287-89c2-6d8139fb4343/resource/"
    "e5f90e3a-8f8d-4895-9c56-4bb2f7877920/download/"
    "vendastesourodireto.csv"
)
LOCAL_INVESTIR_CSV_PATH = Path("data/rendimento-investir.csv")


def _parse_br_numeric(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(".", "").replace(",", ".").strip())


def _prepare_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in ["Taxa Compra Manha", "Taxa Venda Manha", "PU Compra Manha", "PU Venda Manha"]:
        prepared[column] = prepared[column].map(_parse_br_numeric)
    return prepared.dropna(
        subset=["Taxa Compra Manha", "Taxa Venda Manha", "PU Compra Manha", "PU Venda Manha"]
    ).copy()


def _latest_quote_for_maturity(raw_df: pd.DataFrame, maturity: object, date: object) -> pd.Series | None:
    quotes = raw_df[
        (raw_df["Data Vencimento"] == maturity)
        & (raw_df["Data Base"] <= date)
    ].sort_values("Data Base")

    if quotes.empty:
        return None

    return quotes.iloc[-1]


def _extract_investable_ipca_years(text: str) -> set[int]:
    years: set[int] = set()
    for match in re.finditer(r"Tesouro\s+IPCA\+\s+(\d{4})", text, flags=re.IGNORECASE):
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        context = text[start:end].lower()
        if "juros semestrais" in context:
            continue
        years.add(int(match.group(1)))
    return years


def _load_recently_sold_ipca_years() -> tuple[set[int], str, str | None]:
    try:
        response = requests.get(TESOURO_SALES_CSV_URL, timeout=60)
        response.raise_for_status()
        sales = pd.read_csv(io.BytesIO(response.content), sep=";", encoding="utf-8-sig")
    except Exception as exc:
        return set(), TESOURO_SALES_CSV_URL, f"falha ao baixar Vendas do Tesouro Direto: {exc}"

    required = {"Tipo Titulo", "Vencimento do Titulo", "Data Venda"}
    missing = required - set(sales.columns)
    if missing:
        return set(), TESOURO_SALES_CSV_URL, f"CSV de vendas sem colunas esperadas: {sorted(missing)}"

    sales["Tipo Titulo"] = sales["Tipo Titulo"].astype(str).str.strip()
    sales["Data Venda"] = pd.to_datetime(sales["Data Venda"], format="%d/%m/%Y", errors="coerce")
    sales["Vencimento do Titulo"] = pd.to_datetime(
        sales["Vencimento do Titulo"], format="%d/%m/%Y", errors="coerce"
    )

    sales = sales.dropna(subset=["Data Venda", "Vencimento do Titulo"]).copy()
    if sales.empty:
        return set(), TESOURO_SALES_CSV_URL, "CSV de vendas ficou vazio apos limpeza."

    latest_sale_date = sales["Data Venda"].max()
    latest_ipca_sales = sales[
        (sales["Data Venda"] == latest_sale_date)
        & (sales["Tipo Titulo"] == "Tesouro IPCA+")
    ].copy()

    years = set(latest_ipca_sales["Vencimento do Titulo"].dt.year.astype(int).tolist())
    if not years:
        return (
            set(),
            TESOURO_SALES_CSV_URL,
            f"nenhum Tesouro IPCA+ sem cupom vendido em {latest_sale_date.strftime('%d/%m/%Y')}.",
        )

    source = (
        f"{TESOURO_SALES_CSV_URL} "
        f"(vendas em {latest_sale_date.strftime('%d/%m/%Y')})"
    )
    return years, source, None


def _load_investable_ipca_years() -> tuple[set[int], str, str | None]:
    years, source, warning = _load_recently_sold_ipca_years()
    if years:
        return years, source, None

    if LOCAL_INVESTIR_CSV_PATH.exists():
        text = LOCAL_INVESTIR_CSV_PATH.read_text(encoding="utf-8-sig", errors="ignore")
        years = _extract_investable_ipca_years(text)
        if years:
            return years, str(LOCAL_INVESTIR_CSV_PATH), None

    try:
        response = requests.get(INVESTIR_CSV_URL, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        return set(), INVESTIR_CSV_URL, warning or f"nao foi possivel baixar CSV de investir: {exc}"

    text = response.text
    years = _extract_investable_ipca_years(text)
    if not years:
        return set(), INVESTIR_CSV_URL, "CSV de investir baixado, mas nenhum Tesouro IPCA+ sem cupom foi identificado."

    return years, INVESTIR_CSV_URL, None


def backtest_ipca_entry_signal() -> str:
    """
    Sinal operacional atual do IPCA+.

    Usa as mesmas series media e grande do State of Art. O z-score vem da taxa
    da faixa, e o preco de entrada indicado e o PU Venda Manha do titulo atual.
    """

    raw_df = load_tesouro_ipca_frame()
    if "Tipo Titulo" in raw_df.columns:
        raw_df = raw_df[raw_df["Tipo Titulo"].astype(str).str.strip() == "Tesouro IPCA+"].copy()
    raw_df = _prepare_price_columns(raw_df)

    investable_years, investable_source, investable_warning = _load_investable_ipca_years()
    signal_df = raw_df[
        raw_df["Data Vencimento"].dt.year.isin(investable_years)
    ].copy()

    configs = [
        {
            "bucket": "Media",
            "min_prazo": 8.0,
            "max_prazo": 14.0,
            "include_max": False,
            "entry_z": 1.0,
            "exit_z": -1.6,
        },
        {
            "bucket": "Grande",
            "min_prazo": 14.0,
            "max_prazo": 20.0,
            "include_max": True,
            "entry_z": 1.2,
            "exit_z": -2.0,
        },
    ]

    add_threshold_mid = 1.6
    add_threshold_high = 2.0

    summary_lines: list[str] = []
    decision_lines: list[str] = []
    strategy_lines: list[str] = [
        "Sinal: z_2412 da serie da faixa, calculado sobre Taxa Compra Manha.",
        "Universo operacional: apenas Tesouro IPCA+ sem cupom confirmados como disponiveis para investir hoje.",
        "Entrada indicada: comprar o titulo atual pelo PU Venda Manha.",
        "Venda/backtest: vender o mesmo vencimento pelo PU Compra Manha.",
        "Escalonamento comum: z_2412 >= 1.6 -> +1.0x; z_2412 >= 2.0 -> +2.0x",
        "Peso maximo: 4.0x",
        f"Fonte do universo atual: {investable_source}",
    ]

    if investable_warning:
        summary_lines.extend(
            [
                "Universo atual de investimento nao confirmado.",
                investable_warning,
                "Fallback manual disponivel: salve data/rendimento-investir.csv.",
                "Sem essa confirmacao, o sinal nao sugere nova entrada.",
                "",
            ]
        )

    for config in configs:
        bucket_name = str(config["bucket"])
        entry_threshold = float(config["entry_z"])
        exit_threshold = float(config["exit_z"])

        df = build_daily_ipca_duration_bucket_series(
            raw_df,
            min_prazo_anos=float(config["min_prazo"]),
            max_prazo_anos=float(config["max_prazo"]),
            include_max=bool(config["include_max"]),
        )
        df = (
            df.dropna(subset=["data", "taxa_media", "prazo_anos", "data_vencimento"])
            .sort_values("data")
            .reset_index(drop=True)
        )

        if df.empty:
            summary_lines.append(
                f"{bucket_name}: nenhum Tesouro IPCA+ disponivel para investir nesta faixa."
            )
            decision_lines.append(
                f"{bucket_name}: SEM SINAL | nenhum titulo compravel confirmado na faixa."
            )
            continue

        df["z_2412"] = rolling_zscore(df["taxa_media"], window=2412, min_periods=603)
        valid = df.dropna(subset=["z_2412"])
        if valid.empty:
            summary_lines.append(f"{bucket_name}: z_2412 ainda indisponivel.")
            decision_lines.append(f"{bucket_name}: SEM SINAL | historico insuficiente para z_2412.")
            continue

        current = valid.iloc[-1]
        current_date = current["data"]
        z_rate = float(current["taxa_media"])
        current_z2412 = float(current["z_2412"])
        if bool(config["include_max"]):
            current_candidates = signal_df[
                (signal_df["Data Base"] == current_date)
                & (signal_df["Prazo_anos"] >= float(config["min_prazo"]))
                & (signal_df["Prazo_anos"] <= float(config["max_prazo"]))
            ].copy()
        else:
            current_candidates = signal_df[
                (signal_df["Data Base"] == current_date)
                & (signal_df["Prazo_anos"] >= float(config["min_prazo"]))
                & (signal_df["Prazo_anos"] < float(config["max_prazo"]))
            ].copy()

        if current_candidates.empty:
            summary_lines.append(
                f"{bucket_name}: nenhum Tesouro IPCA+ compravel confirmado na data {format_date(current_date)}."
            )
            decision_lines.append(
                f"{bucket_name}: SEM SINAL | nenhum titulo compravel confirmado na faixa."
            )
            continue

        selected = current_candidates.sort_values("Prazo_anos").iloc[-1]
        current_rate = float(selected["Taxa Compra Manha"])
        current_prazo = float(selected["Prazo_anos"])
        current_maturity = selected["Data Vencimento"]
        quote = _latest_quote_for_maturity(raw_df, current_maturity, current_date)

        if quote is None:
            summary_lines.append(
                f"{bucket_name}: sem cotacao de PU para {format_date(current_date)}."
            )
            continue

        pu_venda = float(quote["PU Venda Manha"])
        pu_compra = float(quote["PU Compra Manha"])
        taxa_venda = float(quote["Taxa Venda Manha"])
        taxa_compra = float(quote["Taxa Compra Manha"])
        quote_date = quote["Data Base"]

        if current_z2412 >= add_threshold_high:
            target_weight = 4.0
            allocation_status = "Escalonamento completo"
        elif current_z2412 >= add_threshold_mid:
            target_weight = 2.0
            allocation_status = "Primeiro escalonamento ativo"
        elif current_z2412 >= entry_threshold:
            target_weight = 1.0
            allocation_status = "Entrada base ativa"
        else:
            target_weight = 0.0
            allocation_status = "Fora da faixa de entrada"

        enter_now = current_z2412 >= entry_threshold
        exit_now = current_z2412 <= exit_threshold
        distance_to_entry = entry_threshold - current_z2412
        distance_to_exit = current_z2412 - exit_threshold
        title_txt = f"IPCA+ {current_maturity.year}"

        summary_lines.extend(
            [
                f"{bucket_name}:",
                f"Data da serie: {format_date(current_date)} | Cotacao PU: {format_date(quote_date)}",
                f"Titulo indicado: {title_txt} | venc={current_maturity.strftime('%Y-%m-%d')}",
                f"Prazo atual: {current_prazo:.2f} anos",
                f"Taxa Venda Manha (entrada): {format_value(taxa_venda)}",
                f"PU Venda Manha (entrada): {pu_venda:.2f}",
                f"Taxa Compra Manha (resgate): {format_value(taxa_compra)}",
                f"PU Compra Manha (resgate): {pu_compra:.2f}",
                f"Taxa usada no z_2412: {format_value(z_rate)}",
                f"z_2412 atual: {format_value(current_z2412)}",
                f"threshold entrada: {entry_threshold:.1f} | threshold saida: {exit_threshold:.1f}",
                f"distancia ate entrada: {distance_to_entry:+.2f}",
                f"distancia ate saida: {distance_to_exit:+.2f}",
                f"alocacao alvo: {target_weight:.1f}x | status: {allocation_status}",
                "",
            ]
        )

        if exit_now:
            decision_lines.append(
                f"{bucket_name}: SINAL DE SAIDA / ZERA | z_2412={current_z2412:.2f} <= {exit_threshold:.1f}"
            )
        elif enter_now:
            decision_lines.append(
                f"{bucket_name}: SINAL DE ENTRADA / MANTEM | comprar {title_txt} a PU Venda {pu_venda:.2f}"
            )
        else:
            decision_lines.append(
                f"{bucket_name}: NAO ENTRA | z_2412={current_z2412:.2f} abaixo de {entry_threshold:.1f}"
            )

    report = build_result(
        "IPCA+ ENTRY SIGNAL",
        section(summary_lines),
        section(decision_lines, title="DECISAO OPERACIONAL"),
        section(strategy_lines, title="REGRAS DA ESTRATEGIA"),
    )
    return render_result(report)


__all__ = ["backtest_ipca_entry_signal"]
