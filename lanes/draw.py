"""Маски смуг для метрик і фото з намальованими лініями.

Задача F4, власник: Фояк Віталій. Малювання — через Pillow ``ImageDraw``:
методичка дозволяє бібліотеки для візуалізації.
"""

import numpy as np


def lanes_mask(shape: tuple[int, int], lanes: list[np.ndarray], thickness: int) -> np.ndarray:
    """Малює смуги ламаними товщини ``thickness`` на чорному полі.

    ``shape`` — ``(H, W)``. Повертає ``np.uint8`` ``(H, W)`` зі значеннями 0
    і 255. Порожній список смуг — повністю чорна маска.
    """
    raise NotImplementedError("F4")


def overlay(image: np.ndarray, lanes: list[np.ndarray],
            gt_lanes: list[np.ndarray] | None = None) -> np.ndarray:
    """Повертає копію фото з лініями поверх.

    Знайдені смуги — червоні товщиною 6 px, еталонні (якщо передано) —
    зелені товщиною 3 px. Вхідне фото не змінює.
    """
    raise NotImplementedError("F4")
