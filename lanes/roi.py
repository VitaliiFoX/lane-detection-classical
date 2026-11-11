"""Трапеція області інтересу перед автомобілем. Задача M1, власник: Сивопляс Максим.

Вершини задано частками ширини й висоти (``config.json → roi.polygon``), тож
та сама трапеція працює для фото CULane і для малого синтетичного кадру.
"""

import numpy as np


def roi_polygon(shape: tuple[int, int], fractions: list[list[float]]) -> list[tuple[float, float]]:
    """Переводить вершини з часток ``(x/W, y/H)`` у пікселі ``(x, y)``.

    ``shape`` — ``(H, W)``. Частка 1.0 відповідає останньому пікселю
    (``W − 1`` чи ``H − 1``).
    """
    raise NotImplementedError("M1")


def roi_mask(shape: tuple[int, int], fractions: list[list[float]]) -> np.ndarray:
    """Бінарна маска ``(H, W)``: ``True`` всередині трапеції.

    Трапецію малюємо заповненим многокутником через ``ImageDraw.polygon``.
    """
    raise NotImplementedError("M1")


def roi_top(shape: tuple[int, int], fractions: list[list[float]]) -> int:
    """Рядок верхньої межі трапеції — до нього Хаф продовжує знайдені лінії."""
    raise NotImplementedError("M1")
