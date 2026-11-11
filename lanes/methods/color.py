"""Колірні перетворення: сіре, HSV, маска білого й жовтого.

Задача F6, власник: Фояк Віталій. Категорія — колірна корекція (стадія 1 p1).
Формули — ``docs/pipelines.md``, розділ p1.
"""

import numpy as np


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Яскравість ``Y = 0.299·R + 0.587·G + 0.114·B``.

    Вхід ``np.uint8 (H, W, 3)``, вихід ``np.float64 (H, W)`` у 0…255.
    """
    raise NotImplementedError("F6")


def rgb_to_hsv(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Переводить RGB у HSV вручну.

    Повертає ``(h, s, v)``, кожен ``np.float64 (H, W)``: ``h`` у градусах
    0…360, ``s`` і ``v`` у 0…1. Для сірих пікселів (max = min) ``h = 0``,
    для чорних ``s = 0``.
    """
    raise NotImplementedError("F6")


def white_yellow_mask(image: np.ndarray, white_s_max: float, white_v_min: float,
                      yellow_h_min: float, yellow_h_max: float,
                      yellow_s_min: float, yellow_v_min: float) -> np.ndarray:
    """Маска пікселів, схожих на білу або жовту розмітку.

    Білий: ``s ≤ white_s_max`` і ``v ≥ white_v_min``. Жовтий:
    ``yellow_h_min ≤ h ≤ yellow_h_max``, ``s ≥ yellow_s_min``,
    ``v ≥ yellow_v_min``. Повертає ``np.bool_ (H, W)``.
    """
    raise NotImplementedError("F6")
