from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime


CONFIG_PATH = Path("data_updater/update_config.json")


def _today_str() -> str:
    return datetime.now().date().isoformat()


def load_update_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_update_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def was_updated_today(series_name: str) -> bool:
    config = load_update_config()
    key = f"{series_name}_last_update"
    return config.get(key) == _today_str()


def mark_updated_today(series_name: str) -> None:
    config = load_update_config()
    key = f"{series_name}_last_update"
    config[key] = _today_str()
    save_update_config(config)