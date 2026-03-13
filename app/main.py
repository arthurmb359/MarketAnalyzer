from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bootstrap import bootstrap_tesouro_updates
from app.registry import create_backtest_registry
from app.ui import MarketAnalyzerWindow
from markets.tesouro_ipca import build_daily_ipca_long_series, load_tesouro_ipca_frame


def main() -> int:
    try:
        bootstrap_tesouro_updates()
        df = load_tesouro_ipca_frame()
        daily = build_daily_ipca_long_series(df)
        registry = create_backtest_registry()

        app = MarketAnalyzerWindow(daily=daily, registry=registry)
        app.run()
        return 0
    except Exception as exc:
        print(f"Erro ao executar analise: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
