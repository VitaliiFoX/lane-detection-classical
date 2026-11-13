"""Колірні перетворення: сіре, HSV, маска білого й жовтого.

Задача F6, власник: Фояк Віталій. Категорія — колірна корекція (стадія 1 p1).
Формули — ``docs/pipelines.md``, розділ p1.
"""

import numpy as np


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Яскравість ``Y = 0.299·R + 0.587·G + 0.114·B``.

    Вхід ``np.uint8 (H, W, 3)``, вихід ``np.float64 (H, W)`` у 0…255.
    Коефіцієнти — стандарт ITU-R BT.601: око найчутливіше до зеленого.
    """
    rgb = image.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def rgb_to_hsv(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Переводить RGB у HSV вручну.

    Повертає ``(h, s, v)``, кожен ``np.float64 (H, W)``: ``h`` у градусах
    0…360, ``s`` і ``v`` у 0…1. Для сірих пікселів (max = min) ``h = 0``,
    для чорних ``s = 0``.

    ``V = max``, ``S = (max − min) / max``, а відтінок залежить від того,
    який канал найбільший (``δ = max − min``):
    R → ``60·((G − B)/δ mod 6)``, G → ``60·((B − R)/δ + 2)``,
    B → ``60·((R − G)/δ + 4)``.
    """
    rgb = image.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    c_max = rgb.max(axis=2)
    c_min = rgb.min(axis=2)
    delta = c_max - c_min

    v = c_max
    # Ділимо лише там, де дільник не нуль; деінде підставляємо 1, а результат обнуляємо.
    s = np.where(c_max > 0, delta / np.where(c_max > 0, c_max, 1.0), 0.0)

    h = np.zeros_like(c_max)
    safe_delta = np.where(delta > 0, delta, 1.0)
    red_max = (delta > 0) & (c_max == r)
    green_max = (delta > 0) & (c_max == g) & ~red_max
    blue_max = (delta > 0) & ~red_max & ~green_max
    h[red_max] = (60 * (g - b) / safe_delta % 360)[red_max]
    h[green_max] = (60 * (b - r) / safe_delta + 120)[green_max]
    h[blue_max] = (60 * (r - g) / safe_delta + 240)[blue_max]
    return h, s, v


def white_yellow_mask(image: np.ndarray, white_s_max: float, white_v_min: float,
                      yellow_h_min: float, yellow_h_max: float,
                      yellow_s_min: float, yellow_v_min: float) -> np.ndarray:
    """Маска пікселів, схожих на білу або жовту розмітку.

    Білий: ``s ≤ white_s_max`` і ``v ≥ white_v_min`` (майже без кольору й
    світлий). Жовтий: ``yellow_h_min ≤ h ≤ yellow_h_max``,
    ``s ≥ yellow_s_min``, ``v ≥ yellow_v_min``. Повертає ``np.bool_ (H, W)``.
    """
    h, s, v = rgb_to_hsv(image)
    white = (s <= white_s_max) & (v >= white_v_min)
    yellow = ((h >= yellow_h_min) & (h <= yellow_h_max)
              & (s >= yellow_s_min) & (v >= yellow_v_min))
    return white | yellow
