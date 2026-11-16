"""Морфологія сірого зображення і top-hat. Задача M4, власник: Сивопляс Максим.

Категорія — морфологія (стадія 1 p4). Структурний елемент — прямокутник
``height × width`` з центром посередині (розміри непарні).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _windows(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    """Вікна ``height × width`` навколо кожного пікселя: масив ``(H, W, height, width)``.

    Краї доповнюємо повтором крайніх пікселів, тому вікно є в кожного
    пікселя, а результат морфології має розмір входу.
    """
    if height < 1 or width < 1 or height % 2 == 0 or width % 2 == 0:
        raise ValueError("Розміри структурного елемента мають бути додатними й непарними")
    gray = np.asarray(gray, dtype=np.float64)
    # Рамка в половину елемента: зверху й знизу height // 2, зліва й справа width // 2.
    padded = np.pad(gray, ((height // 2, height // 2), (width // 2, width // 2)), mode="edge")
    # sliding_window_view не копіює дані: windows[y, x] — це зріз
    # padded[y:y + height, x:x + width], тобто окіл пікселя (y, x).
    return sliding_window_view(padded, (height, width))


def erode(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    """Ерозія: мінімум у вікні ``height × width`` навколо кожного пікселя.

    Краї доповнюємо повтором крайніх пікселів; вікна —
    ``sliding_window_view``. Повертає ``np.float64 (H, W)``.

    Світлі деталі, вужчі за елемент, зникають: у їхньому вікні завжди є
    темний сусід, і мінімум бере його.
    """
    # Мінімум по двох останніх осях — по всіх пікселях кожного вікна.
    return _windows(gray, height, width).min(axis=(2, 3))


def dilate(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    """Дилатація: максимум у вікні ``height × width``.

    Світлі ділянки розростаються на половину елемента в кожен бік.
    """
    return _windows(gray, height, width).max(axis=(2, 3))


def opening(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    """Морфологічне відкриття: ``dilate(erode(gray))``.

    Ерозія стирає вузькі світлі деталі й звужує широкі; дилатація повертає
    широким попередній розмір, а стертих уже не відновлює.
    """
    return dilate(erode(gray, height, width), height, width)


def top_hat(gray: np.ndarray, width: int) -> np.ndarray:
    """Білий top-hat з горизонтальним елементом ``1 × width``: ``gray − opening``.

    Лишає світлі деталі, вужчі за ``width``, — смуги розмітки. Відкриття
    ніколи не перевищує оригінал, тож результат невід'ємний.
    """
    gray = np.asarray(gray, dtype=np.float64)
    return gray - opening(gray, 1, width)
