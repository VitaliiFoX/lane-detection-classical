"""Метрики якості між еталонною і знайденою масками смуг.

Задача F5, власник: Фояк Віталій. Формули — ``docs/metrics.md``. Маски —
``np.uint8`` однакової форми зі значеннями 0 і 255.
"""

import numpy as np

MAX_VALUE = 255.0


def mse(reference: np.ndarray, result: np.ndarray) -> float:
    """Середньоквадратична помилка ``(1/N) · Σ (I − K)²``."""
    difference = reference.astype(np.float64) - result.astype(np.float64)
    return float(np.mean(difference ** 2))


def psnr(reference: np.ndarray, result: np.ndarray, cap: float = 100.0) -> float:
    """Пікове відношення сигнал/шум ``10 · log10(255² / MSE)`` у дБ.

    Якщо маски збігаються (MSE = 0), повертає ``cap``; більше за ``cap``
    теж не повертає, щоб середні в таблицях були порівнянні.
    """
    error = mse(reference, result)
    if error == 0:
        return cap
    return float(min(cap, 10 * np.log10(MAX_VALUE ** 2 / error)))


def _window_means(image: np.ndarray, window: int) -> np.ndarray:
    """Середнє в кожному вікні ``window × window`` через інтегральне зображення.

    ``s[y, x]`` — сума всіх пікселів вище й лівіше; суму у вікні дають чотири
    його кути: ``s[низ, право] − s[верх, право] − s[низ, ліво] + s[верх, ліво]``.
    Так рахуємо всі вікна за один прохід, а не складаємо 49 пікселів у кожному.
    Результат ``(H − window + 1, W − window + 1)``.
    """
    sums = np.pad(image, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    totals = (sums[window:, window:] - sums[:-window, window:]
              - sums[window:, :-window] + sums[:-window, :-window])
    return totals / (window * window)


def ssim(reference: np.ndarray, result: np.ndarray, window: int = 7,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Індекс структурної подібності, усереднений по вікнах ``window × window``.

    У кожному вікні рахуємо середні ``μx``, ``μy``, дисперсії ``σx²``,
    ``σy²`` і коваріацію ``σxy`` через ``E[x²] − E[x]²`` та
    ``E[xy] − E[x]·E[y]``, далі формулу з ``docs/metrics.md`` з
    ``C1 = (k1·255)²``, ``C2 = (k2·255)²``. Зображення, менше за вікно, —
    одне вікно на все зображення.
    """
    x = reference.astype(np.float64)
    y = result.astype(np.float64)
    c1 = (k1 * MAX_VALUE) ** 2
    c2 = (k2 * MAX_VALUE) ** 2

    height, width = x.shape
    if height < window or width < window:
        # Одне вікно на все зображення: середні — числа, а не масиви.
        mu_x, mu_y = x.mean(), y.mean()
        var_x, var_y = x.var(), y.var()
        cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    else:
        mu_x, mu_y = _window_means(x, window), _window_means(y, window)
        var_x = _window_means(x * x, window) - mu_x ** 2
        var_y = _window_means(y * y, window) - mu_y ** 2
        cov_xy = _window_means(x * y, window) - mu_x * mu_y

    index = (((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2))
             / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)))
    return float(np.mean(index))


def iou(reference: np.ndarray, result: np.ndarray) -> float:
    """Перетин на об'єднання білих пікселів; обидві маски порожні → 1.0."""
    a = reference > 0
    b = result > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)
