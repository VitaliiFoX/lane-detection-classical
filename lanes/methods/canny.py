"""Метод Кенні. Задача M7, власник: Сивопляс Максим.

Категорія — контурний аналіз (стадія 2 p4). Гаусове згладжування тут —
внутрішній крок методу, погоджено з викладачкою. Градієнти — Собель з
``sobel.py`` (ADR-0010). Кроки — ``docs/pipelines.md``, розділ p4.
"""

import numpy as np


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Гаусове ядро ``size × size`` (непарний розмір), нормоване до суми 1."""
    raise NotImplementedError("M7")


def non_max_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Потоншення контурів до одного пікселя.

    Напрям градієнта (радіани) округлюємо до 0°, 45°, 90°, 135°; піксель
    лишаємо, якщо він не менший за обох сусідів уздовж градієнта, інакше 0.
    """
    raise NotImplementedError("M7")


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Гістерезис: слабкий піксель стає контуром, якщо з'єднаний із сильним.

    Поширюємо сильні пікселі на 8 сусідів серед слабких, доки маска
    змінюється. Повертає ``np.bool_ (H, W)``.
    """
    raise NotImplementedError("M7")


def canny(gray: np.ndarray, kernel_size: int, sigma: float,
          low: float, high: float) -> np.ndarray:
    """Повний Кенні: згладжування → Собель → потоншення → подвійний поріг → гістерезис.

    ``low`` і ``high`` — частки максимуму модуля градієнта після
    потоншення. Однотонне зображення — порожня маска.
    """
    raise NotImplementedError("M7")
