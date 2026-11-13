"""Читання конфігурації проєкту. Власник: Фояк Віталій."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Читає ``config.json`` і повертає словник усіх параметрів.

    Ключі описано в ``docs/pipelines.md`` і ``docs/metrics.md``.
    """
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def project_path(path: Path | str) -> Path:
    """Відносний шлях з конфігурації рахуємо від кореня репозиторію.

    Так ``data/sample`` знаходиться, з якої б теки не запустили програму.
    Абсолютний шлях повертаємо як є.
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
