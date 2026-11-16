"""p2 «Оцу»: вирівнювання гістограми → поріг Оцу → Хаф. Задача F13, власник: Фояк Віталій.

Специфікація — ``docs/pipelines.md``, розділ p2.
"""

import numpy as np

from lanes.methods.color import rgb_to_gray
from lanes.methods.histogram import equalize_histogram
from lanes.methods.hough import detect_lanes
from lanes.methods.threshold import otsu_binarize
from lanes.pipelines.base import PipelineResult
from lanes.roi import roi_mask, roi_top


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Стадії: ``equalized`` → ``otsu`` → ``hough_space``."""
    params = config["p2"]
    polygon = config["roi"]["polygon"]
    shape = image.shape[:2]

    # Стадія 1. Сіре й вирівнювання гістограми: темний нічний кадр розтягується
    # на весь діапазон 0…255.
    equalized = equalize_histogram(rgb_to_gray(image))

    # Стадія 2. Поріг Оцу лише по пікселях ROI (небо й капот не впливають на
    # поріг). Друга ітерація ділить уже світлий клас — так лишається розмітка,
    # а не вся світліша половина дороги. Пікселі поза ROI otsu_binarize
    # повертає як False.
    otsu, _ = otsu_binarize(equalized, roi_mask(shape, polygon), params["otsu_iterations"])

    # Стадія 3. Прямі Хафа від нижнього краю фото до верху ROI.
    lanes, hough_space = detect_lanes(otsu, roi_top(shape, polygon), config["hough"])

    return PipelineResult(stages=[("equalized", equalized), ("otsu", otsu),
                                  ("hough_space", hough_space)],
                          lanes=lanes)
