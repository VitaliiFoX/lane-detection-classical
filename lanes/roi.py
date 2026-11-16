"""Трапеція області інтересу перед автомобілем. Задача M1, власник: Сивопляс Максим.

Вершини задано частками ширини й висоти (``config.json → roi.polygon``), тож
та сама трапеція працює для фото CULane і для малого синтетичного кадру.
"""

import numpy as np
from PIL import Image, ImageDraw


def roi_polygon(shape: tuple[int, int], fractions: list[list[float]]) -> list[tuple[float, float]]:
    """Переводить вершини з часток ``(x/W, y/H)`` у пікселі ``(x, y)``.

    ``shape`` — ``(H, W)``. Частка 1.0 відповідає останньому пікселю
    (``W − 1`` чи ``H − 1``), тому ``x = fx · (W − 1)``, ``y = fy · (H − 1)``.
    """
    height, width = shape
    return [(fx * (width - 1), fy * (height - 1)) for fx, fy in fractions]


def roi_mask(shape: tuple[int, int], fractions: list[list[float]]) -> np.ndarray:
    """Бінарна маска ``(H, W)``: ``True`` всередині трапеції.

    Трапецію малюємо заповненим многокутником через ``ImageDraw.polygon``.
    """
    height, width = shape
    # Чорне полотно розміру фото; Pillow задає розмір як (ширина, висота).
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(roi_polygon(shape, fractions), fill=255)
    # Зафарбовані пікселі мають значення 255, решта 0 — порівняння дає bool-маску.
    return np.array(canvas) > 0


def roi_top(shape: tuple[int, int], fractions: list[list[float]]) -> int:
    """Рядок верхньої межі трапеції — до нього Хаф продовжує знайдені лінії.

    Це найменший ``y`` серед вершин, округлений до цілого рядка.
    """
    return int(round(min(y for _, y in roi_polygon(shape, fractions))))
