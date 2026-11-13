"""Маски смуг для метрик і фото з намальованими лініями.

Задача F4, власник: Фояк Віталій. Малювання — через Pillow ``ImageDraw``:
методичка дозволяє бібліотеки для візуалізації.
"""

import numpy as np
from PIL import Image, ImageDraw

# Оформлення картинки overlay.png — це лише візуалізація, на метрики не впливає.
PREDICTED_COLOR, PREDICTED_WIDTH = (255, 0, 0), 6
GROUND_TRUTH_COLOR, GROUND_TRUTH_WIDTH = (0, 255, 0), 3


def _points(lane: np.ndarray) -> list[tuple[float, float]]:
    """Точки смуги у вигляді, який розуміє ``ImageDraw.line``."""
    return [(float(x), float(y)) for x, y in lane]


def lanes_mask(shape: tuple[int, int], lanes: list[np.ndarray], thickness: int) -> np.ndarray:
    """Малює смуги ламаними товщини ``thickness`` на чорному полі.

    ``shape`` — ``(H, W)``. Повертає ``np.uint8`` ``(H, W)`` зі значеннями 0
    і 255. Порожній список смуг — повністю чорна маска.
    """
    height, width = shape
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for lane in lanes:
        if len(lane) >= 2:
            # joint="curve" заокруглює стики ламаної, щоб на згинах не було щілин.
            draw.line(_points(lane), fill=255, width=thickness, joint="curve")
    return np.array(canvas, dtype=np.uint8)


def overlay(image: np.ndarray, lanes: list[np.ndarray],
            gt_lanes: list[np.ndarray] | None = None) -> np.ndarray:
    """Повертає копію фото з лініями поверх.

    Знайдені смуги — червоні товщиною 6 px, еталонні (якщо передано) —
    зелені товщиною 3 px поверх червоних, щоб було видно обидві. Вхідне фото
    не змінює.
    """
    canvas = Image.fromarray(np.array(image, dtype=np.uint8))
    draw = ImageDraw.Draw(canvas)
    for lane in lanes:
        if len(lane) >= 2:
            draw.line(_points(lane), fill=PREDICTED_COLOR, width=PREDICTED_WIDTH)
    for lane in gt_lanes or []:
        if len(lane) >= 2:
            draw.line(_points(lane), fill=GROUND_TRUTH_COLOR, width=GROUND_TRUTH_WIDTH)
    return np.array(canvas, dtype=np.uint8)
