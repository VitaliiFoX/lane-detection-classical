"""Перетворення Хафа для прямих. Задача F8, власник: Фояк Віталій.

Спільний модуль для p1, p2, p4 (ADR-0006). Алгоритм крок за кроком —
``docs/pipelines.md``, розділ «Перетворення Хафа».
"""

import numpy as np


def accumulator(edges: np.ndarray, rho_step: float, theta_step_deg: float,
                min_angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Голосування білих пікселів у просторі ``(ρ, θ)``.

    ``ρ = x·cos θ + y·sin θ``; лишаємо лише θ, для яких кут лінії до
    горизонталі ``90° − |θ − 90°| ≥ min_angle_deg``. Повертає
    ``(acc, rhos, thetas_deg)``: акумулятор ``(n_rho, n_theta)`` цілих
    голосів і значення осей.
    """
    raise NotImplementedError("F8")


def find_peaks(acc: np.ndarray, max_peaks: int, min_votes: int,
               suppress_rho_bins: int, suppress_theta_bins: int) -> list[tuple[int, int]]:
    """До ``max_peaks`` найсильніших піків з придушенням околиці.

    Повторюємо: максимум; менше ``min_votes`` — стоп; інакше запам'ятовуємо
    індекси ``(i_rho, i_theta)`` і обнуляємо прямокутник околиці в копії
    акумулятора.
    """
    raise NotImplementedError("F8")


def peaks_to_lanes(peaks: list[tuple[int, int]], rhos: np.ndarray, thetas_deg: np.ndarray,
                   height: int, y_top: int) -> list[np.ndarray]:
    """Перетворює піки на відрізки від ``y = height − 1`` до ``y = y_top``.

    ``x = (ρ − y·sin θ) / cos θ``. Кожна смуга — ``np.ndarray (2, 2)``.
    """
    raise NotImplementedError("F8")


def detect_lanes(edges: np.ndarray, y_top: int,
                 params: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """Повний Хаф: акумулятор → піки → відрізки.

    ``params`` — секція ``hough`` з ``config.json``. Повертає
    ``(lanes, acc_image)``, де ``acc_image`` — акумулятор, нормований до
    0…255, для збереження як стадія ``hough_space``. Порожня маска — ``([], …)``.
    """
    raise NotImplementedError("F8")
