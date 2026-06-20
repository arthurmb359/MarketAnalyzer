from pathlib import Path

from data_updater.tesouro_updater import (
    _last_business_day,
    get_tesouro_csv_last_date,
    rebuild_tesouro_ipca,
    update_tesouro_csv_if_needed,
)
from data_updater.update_config import mark_updated_today, was_updated_today

TESOURO_RAW_CSV_PATH = Path("data/precotaxatesourodireto.csv")
TESOURO_SERIES_NAME = "tesouro_ipca"


def bootstrap_tesouro_updates() -> None:
    raw_csv = TESOURO_RAW_CSV_PATH
    tesouro_ipca_csv = Path("data/tesouro_ipca.csv")
    target_date = _last_business_day()

    print("=== Atualizacao Tesouro Direto ===")

    if was_updated_today(TESOURO_SERIES_NAME):
        raw_last_date = get_tesouro_csv_last_date(raw_csv)
        ipca_last_date = (
            get_tesouro_csv_last_date(tesouro_ipca_csv)
            if tesouro_ipca_csv.exists()
            else None
        )

        already_consistent = (
            raw_last_date is not None
            and raw_last_date >= target_date
            and ipca_last_date == raw_last_date
        )

        if already_consistent:
            print(
                f"[SKIP] Tesouro ja atualizado hoje "
                f"(last={raw_last_date}, target={target_date})"
            )
            return

        print(
            f"[WARN] flag diario encontrado, mas arquivos estao defasados "
            f"(raw={raw_last_date}, ipca={ipca_last_date}, target={target_date}); "
            "tentando recuperar"
        )

    result = update_tesouro_csv_if_needed(raw_csv)

    if result["updated"]:
        print(
            f"[OK] bruto atualizado de {result.get('old_last_date')} "
            f"para {result['last_date']}"
        )
    else:
        print(
            f"[SKIP] bruto ja esta no snapshot esperado "
            f"(last={result['last_date']}, target={result['target_date']}, "
            f"reason={result.get('reason', 'up-to-date')})"
        )

    ipca_last_date = (
        get_tesouro_csv_last_date(tesouro_ipca_csv)
        if tesouro_ipca_csv.exists()
        else None
    )
    raw_last_date = get_tesouro_csv_last_date(raw_csv)

    if result["updated"] or ipca_last_date != raw_last_date:
        rebuilt = rebuild_tesouro_ipca(raw_csv, tesouro_ipca_csv)
        print(
            f"[OK] tesouro_ipca.csv regenerado com {rebuilt['rows']} linhas "
            f"({rebuilt['start_date']} -> {rebuilt['end_date']})"
        )
    else:
        print("[SKIP] tesouro_ipca.csv ja estava consistente com o bruto")

    mark_updated_today(TESOURO_SERIES_NAME)


__all__ = ["TESOURO_RAW_CSV_PATH", "TESOURO_SERIES_NAME", "bootstrap_tesouro_updates"]
