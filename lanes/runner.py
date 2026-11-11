"""Обробка вибірки: стадії, маски, метрики. Задача F11, власник: Фояк Віталій.

Потік даних — ``ARCHITECTURE.md``, розділ 4; формат ``metrics.csv`` —
``docs/metrics.md``.
"""

from pathlib import Path


def run_pipelines(config: dict, names: list[str], limit: int | None = None) -> Path:
    """Запускає пайплайни ``names`` на фото з ``data/sample.csv``.

    ``limit`` — обробити лише перші N фото (пілот). Для кожного фото
    зберігає ``results/<пайплайн>/<група>/<фото>/NN_<стадія>.png``,
    ``mask.png`` і ``overlay.png``, рахує метрики проти розмітки й час.
    Пайплайн, що впав, і битий файл не зупиняють прогін: рядок з порожніми
    метриками і повідомлення в консоль. Повертає шлях до ``metrics.csv``.
    """
    raise NotImplementedError("F11")
