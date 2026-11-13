"""Фільтр Собеля. Задача F7, власник: Фояк Віталій.

Категорія — контурний аналіз (стадія 2 p1). Градієнти також бере Кенні
(ADR-0010).
"""

import numpy as np

from lanes.filters import apply_kernel

# Різниця «праворуч мінус ліворуч», де центральний рядок має вагу 2:
# це водночас похідна по x і згладжування по y.
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Градієнти ``(gx, gy)`` через ``filters.apply_kernel`` з ядрами Собеля.

    ``gx`` додатний, коли праворуч світліше, ``gy`` — коли нижче світліше.
    """
    return apply_kernel(gray, SOBEL_X), apply_kernel(gray, SOBEL_Y)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Модуль градієнта ``sqrt(gx² + gy²)``, ``np.float64 (H, W)``."""
    gx, gy = sobel_gradients(gray)
    return np.hypot(gx, gy)
