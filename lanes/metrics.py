"""Метрики якості між еталонною і знайденою масками смуг.

Задача F5, власник: Фояк Віталій. Формули — ``docs/metrics.md``. Маски —
``np.uint8`` однакової форми зі значеннями 0 і 255.
"""

import numpy as np


def mse(reference: np.ndarray, result: np.ndarray) -> float:
    """Середньоквадратична помилка ``(1/N) · Σ (I − K)²``."""
    raise NotImplementedError("F5")


def psnr(reference: np.ndarray, result: np.ndarray, cap: float = 100.0) -> float:
    """Пікове відношення сигнал/шум ``10 · log10(255² / MSE)`` у дБ.

    Якщо маски збігаються (MSE = 0), повертає ``cap``.
    """
    raise NotImplementedError("F5")


def ssim(reference: np.ndarray, result: np.ndarray, window: int = 7,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Індекс структурної подібності, усереднений по вікнах ``window × window``.

    У кожному вікні (``sliding_window_view``) рахуємо середні, дисперсії й
    коваріацію і формулу з ``docs/metrics.md`` з ``C1 = (k1·255)²``,
    ``C2 = (k2·255)²``. Зображення, менше за вікно, — одне вікно на все
    зображення.
    """
    raise NotImplementedError("F5")


def iou(reference: np.ndarray, result: np.ndarray) -> float:
    """Перетин на об'єднання білих пікселів; обидві маски порожні → 1.0."""
    raise NotImplementedError("F5")
