"""Фільтр Прюітта. Задача M5, власник: Сивопляс Максим.

Категорія — фільтрація (стадія 2 p3).
"""

import numpy as np

PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
PREWITT_Y = PREWITT_X.T


def prewitt_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Градієнти ``(gx, gy)`` через ``filters.apply_kernel`` з ядрами Прюітта."""
    raise NotImplementedError("M5")


def prewitt_magnitude(gray: np.ndarray) -> np.ndarray:
    """Модуль градієнта ``sqrt(gx² + gy²)``, ``np.float64 (H, W)``."""
    raise NotImplementedError("M5")
