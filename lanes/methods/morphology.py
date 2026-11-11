"""Морфологія сірого зображення і top-hat. Задача M4, власник: Сивопляс Максим.

Категорія — морфологія (стадія 1 p4). Структурний елемент — прямокутник
``height × width`` з центром посередині (розміри непарні).
"""

import numpy as np


def erode(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    """Ерозія: мінімум у вікні ``height × width`` навколо кожного пікселя.

    Краї доповнюємо повтором крайніх пікселів; вікна —
    ``sliding_window_view``. Повертає ``np.float64 (H, W)``.
    """
    raise NotImplementedError("M4")


def dilate(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    """Дилатація: максимум у вікні ``height × width``."""
    raise NotImplementedError("M4")


def opening(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    """Морфологічне відкриття: ``dilate(erode(gray))``."""
    raise NotImplementedError("M4")


def top_hat(gray: np.ndarray, width: int) -> np.ndarray:
    """Білий top-hat з горизонтальним елементом ``1 × width``: ``gray − opening``.

    Лишає світлі деталі, вужчі за ``width``, — смуги розмітки.
    """
    raise NotImplementedError("M4")
