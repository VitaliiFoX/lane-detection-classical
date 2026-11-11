"""Еталонні сценарії для критерію C1. Задача M10, власник: Сивопляс Максим.

Сценарії й пороги — ``docs/metrics.md``. Результат — ``results/c1_check.csv``.
"""

import numpy as np


def mean_lane_offset(gt_lanes: list[np.ndarray], pred_lanes: list[np.ndarray],
                     ys: np.ndarray) -> float:
    """Середнє горизонтальне відхилення знайдених смуг від еталонних, px.

    Для кожної еталонної смуги шукаємо найближчу знайдену за середнім
    ``|Δx|`` на рівнях ``ys``; ``x`` на рівні ``y`` — лінійна інтерполяція
    по точках смуги. Знайдених менше, ніж еталонних, або смуга не перетинає
    якийсь рівень — ``inf``.
    """
    raise NotImplementedError("M10")


def run_checks(config: dict) -> list[dict]:
    """Проганяє p1–p4 по чотирьох сценаріях і пише ``c1_check.csv``.

    Три синтетичні сценарії будує ``synthetic.make_road`` розміру
    ``checks.synthetic_size``; ``normal_group`` бере середні з
    ``results/metrics.csv`` і порівнює ще й з ``p0``. Повертає рядки таблиці
    ``{pipeline, scenario, value, threshold, passed}`` і друкує підсумок
    «пройдено k з 16».
    """
    raise NotImplementedError("M10")
