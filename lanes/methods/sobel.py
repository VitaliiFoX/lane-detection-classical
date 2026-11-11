"""Фільтр Собеля. Задача F7, власник: Фояк Віталій.

Категорія — контурний аналіз (стадія 2 p1). Градієнти також бере Кенні
(ADR-0010).
"""

import numpy as np

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Градієнти ``(gx, gy)`` через ``filters.apply_kernel`` з ядрами Собеля.

    ``gx`` додатний, коли праворуч світліше, ``gy`` — коли нижче світліше.
    """
    raise NotImplementedError("F7")


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Модуль градієнта ``sqrt(gx² + gy²)``, ``np.float64 (H, W)``."""
    raise NotImplementedError("F7")
