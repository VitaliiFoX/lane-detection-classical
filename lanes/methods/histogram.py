"""Вирівнювання гістограми. Задача F9, власник: Фояк Віталій.

Категорія — гістограмна корекція (стадія 1 p2).
"""

import numpy as np


def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Глобальне вирівнювання гістограми сірого зображення.

    Округлюємо до 256 рівнів, гістограма ``np.bincount``, кумулятивна
    ``cdf``, нове значення ``round((cdf(v) − cdf_min) / (N − cdf_min) · 255)``,
    де ``cdf_min`` — перше ненульове значення ``cdf``, ``N`` — кількість
    пікселів. Однотонне зображення повертає без змін.

    Вхід і вихід — ``np.float64 (H, W)`` у 0…255.
    """
    levels = np.clip(np.round(gray), 0, 255).astype(np.int64)
    # Скільки пікселів має кожен із 256 рівнів яскравості.
    histogram = np.bincount(levels.ravel(), minlength=256)
    # cdf[v] — скільки пікселів мають яскравість ≤ v.
    cdf = np.cumsum(histogram)
    total = levels.size
    cdf_min = cdf[cdf > 0][0]
    if cdf_min == total:
        # Усі пікселі одного рівня: розтягувати нічого, а формула ділила б на нуль.
        return np.array(gray, dtype=np.float64)

    # Таблиця «старий рівень → новий»: частка пікселів, темніших або рівних v,
    # розтягнута на весь діапазон 0…255. Часті рівні отримують ширший діапазон.
    lookup = np.round((cdf - cdf_min) / (total - cdf_min) * 255)
    return lookup[levels].astype(np.float64)
