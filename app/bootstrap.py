from pathlib import Path

from data_updater.tesouro_updater import rebuild_tesouro_ipca, update_tesouro_csv_if_needed
from data_updater.update_config import mark_updated_today, was_updated_today

TESOURO_RAW_CSV_PATH = Path("data/precotaxatesourodireto.csv")
TESOURO_SERIES_NAME = "tesouro_ipca"


def bootstrap_tesouro_updates() -> None:
    raw_csv = TESOURO_RAW_CSV_PATH
    tesouro_ipca_csv = Path("data/tesouro_ipca.csv")

    print("=== Atualizacao Tesouro Direto ===")

    if was_updated_today(TESOURO_SERIES_NAME):
        print("[SKIP] atualizacao do Tesouro ja foi tentada hoje")

        if not tesouro_ipca_csv.exists():
            rebuilt = rebuild_tesouro_ipca(raw_csv, tesouro_ipca_csv)
            print(
                f"[OK] tesouro_ipca.csv regenerado com {rebuilt['rows']} linhas "
                f"({rebuilt['start_date']} -> {rebuilt['end_date']})"
            )
        else:
            print("[SKIP] tesouro_ipca.csv ja existe")
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
                f"[SKIP] bruto ja esta no snapshot esperado "
                f"(last={result['last_date']}, target={result['target_date']}, "
                f"reason={result.get('reason', 'up-to-date')})"
            )

        if result["updated"] or (not tesouro_ipca_csv.exists()):
            rebuilt = rebuild_tesouro_ipca(raw_csv, tesouro_ipca_csv)
            print(
                f"[OK] tesouro_ipca.csv regenerado com {rebuilt['rows']} linhas "
                f"({rebuilt['start_date']} -> {rebuilt['end_date']})"
            )
        else:
            print("[SKIP] tesouro_ipca.csv ja estava consistente com o bruto")

    finally:
        mark_updated_today(TESOURO_SERIES_NAME)


__all__ = ["TESOURO_RAW_CSV_PATH", "TESOURO_SERIES_NAME", "bootstrap_tesouro_updates"]
