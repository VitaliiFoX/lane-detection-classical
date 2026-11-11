"""Пошук смуг на виді зверху: гістограма стовпців і ковзні вікна.

Задача M6, власник: Сивопляс Максим. Категорія — гістограмний аналіз
(стадія 3 p3). Кроки — ``docs/pipelines.md``, розділ p3.
"""

import numpy as np


def column_histogram(edges: np.ndarray) -> np.ndarray:
    """Кількість білих пікселів у кожному стовпці нижньої половини маски."""
    raise NotImplementedError("M6")


def find_base_peaks(histogram: np.ndarray, max_lanes: int, min_distance: int,
                    min_height: int) -> list[int]:
    """До ``max_lanes`` стовпців-піків, віддалених не менше ніж на ``min_distance``.

    Пік нижчий за ``min_height`` не враховуємо. Повертає ``x`` зліва направо.
    """
    raise NotImplementedError("M6")


def track_lane(edges: np.ndarray, x_base: int, n_windows: int, margin: int,
               min_pixels: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    """Ковзні вікна знизу вгору від стовпця ``x_base``.

    Повертає ``(xs, ys, windows)``: координати зібраних білих пікселів і
    прямокутники вікон ``(x0, y0, x1, y1)`` для малюнка.
    """
    raise NotImplementedError("M6")


def detect_lanes_top_view(edges: np.ndarray,
                          params: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """Усі кроки на виді зверху: піки → вікна → парабола ``x = a·y² + b·y + c``.

    ``params`` — секція ``p3`` з ``config.json``. Повертає ``(lanes, debug)``:
    смуги в координатах виду зверху (``(n_windows + 1, 2)`` точок кожна) і
    кольоровий малюнок ``np.uint8 (H, W, 3)`` з вікнами й кривими.
    """
    raise NotImplementedError("M6")
