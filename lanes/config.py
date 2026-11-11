"""Читання конфігурації проєкту. Власник: Фояк Віталій."""

import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Читає ``config.json`` і повертає словник усіх параметрів.

    Ключі описано в ``docs/pipelines.md`` і ``docs/metrics.md``.
    """
    with open(path, encoding="utf-8") as file:
        return json.load(file)
