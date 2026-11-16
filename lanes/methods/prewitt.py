"""Фільтр Прюітта. Задача M5, власник: Сивопляс Максим.

Категорія — фільтрація (стадія 2 p3).
"""

import numpy as np

from lanes.filters import apply_kernel

# Різниця «праворуч мінус ліворуч», усі три рядки з однаковою вагою 1.
# У Собеля центральний рядок має вагу 2 — це єдина відмінність.
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
PREWITT_Y = PREWITT_X.T


def prewitt_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Градієнти ``(gx, gy)`` через ``filters.apply_kernel`` з ядрами Прюітта.

    ``gx`` — зміна яскравості по горизонталі (додатна, коли праворуч
    світліше), ``gy`` — по вертикалі (додатна, коли нижче світліше).
    """
    return apply_kernel(gray, PREWITT_X), apply_kernel(gray, PREWITT_Y)


def prewitt_magnitude(gray: np.ndarray) -> np.ndarray:
    """Модуль градієнта ``sqrt(gx² + gy²)``, ``np.float64 (H, W)``.

    Великий там, де яскравість різко змінюється в будь-якому напрямку, —
    тобто на контурах.
    """
    gx, gy = prewitt_gradients(gray)
    return np.hypot(gx, gy)
