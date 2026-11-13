"""Поріг Оцу. Задача F10, власник: Фояк Віталій.

Категорія — сегментація (стадія 2 p2). Чому ітеративно — ``docs/pipelines.md``.
"""

import numpy as np


def otsu_threshold(values: np.ndarray) -> int:
    """Поріг Оцу для набору яскравостей 0…255.

    Для кожного ``t`` міжкласова дисперсія
    ``σ²_B(t) = ω₀(t) · ω₁(t) · (μ₀(t) − μ₁(t))²``; повертаємо ``t`` з
    максимумом (при рівних — найменший). Пікселі ``> t`` — світлий клас.
    Якщо всі значення однакові, повертає це значення.
    """
    levels = np.clip(np.round(np.ravel(values)), 0, 255).astype(np.int64)
    if levels.size == 0:
        return 0
    if levels.min() == levels.max():
        return int(levels[0])

    # Імовірність кожного рівня яскравості.
    probability = np.bincount(levels, minlength=256) / levels.size
    # Для порогу t темний клас — рівні 0…t:
    # ω₀(t) — його частка, cumulative_mean(t) — Σ i·p(i) до t включно.
    omega0 = np.cumsum(probability)
    omega1 = 1 - omega0
    cumulative_mean = np.cumsum(probability * np.arange(256))
    total_mean = cumulative_mean[-1]

    between = np.zeros(256)
    # Поріг, за якого один із класів порожній, не ділить зображення.
    valid = (omega0 > 0) & (omega1 > 0)
    mu0 = cumulative_mean[valid] / omega0[valid]
    mu1 = (total_mean - cumulative_mean[valid]) / omega1[valid]
    between[valid] = omega0[valid] * omega1[valid] * (mu0 - mu1) ** 2
    # argmax повертає перший максимум, тобто найменший t серед рівних.
    return int(np.argmax(between))


def otsu_binarize(gray: np.ndarray, mask: np.ndarray | None = None,
                  iterations: int = 1) -> tuple[np.ndarray, list[int]]:
    """Бінаризація порогом Оцу, рахованим лише по пікселях ``mask``.

    Кожна наступна ітерація рахує поріг лише по пікселях, світлішим за
    попередній поріг. Повертає ``(binary, thresholds)``: маску
    ``np.bool_ (H, W)`` (пікселі поза ``mask`` — ``False``) і пороги всіх
    ітерацій. Якщо пікселів для наступної ітерації не лишилося — зупиняється.
    """
    # Порівнюємо з округленими значеннями, як і рахував otsu_threshold.
    levels = np.round(gray)
    candidates = np.ones(gray.shape, dtype=bool) if mask is None else mask.astype(bool)
    binary = np.zeros(gray.shape, dtype=bool)
    thresholds = []
    for _ in range(iterations):
        values = levels[candidates]
        if values.size == 0:
            break
        threshold = otsu_threshold(values)
        thresholds.append(threshold)
        binary = candidates & (levels > threshold)
        if not binary.any():
            break
        # Наступна ітерація ділить уже тільки світлий клас.
        candidates = binary
    return binary, thresholds
