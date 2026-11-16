"""p4 «Top-hat»: морфологічний top-hat → Кенні → Хаф. Задача M9, власник: Сивопляс Максим.

Специфікація — ``docs/pipelines.md``, розділ p4.
"""

import numpy as np

from lanes.methods.canny import canny
from lanes.methods.color import rgb_to_gray
from lanes.methods.hough import detect_lanes
from lanes.methods.morphology import top_hat
from lanes.pipelines.base import PipelineResult
from lanes.roi import roi_mask, roi_top


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Стадії: ``tophat`` → ``canny``, ``edges_roi`` → ``hough_space``."""
    params = config["p4"]
    polygon = config["roi"]["polygon"]
    gray = rgb_to_gray(image)

    # Стадія 1. Top-hat лишає світлі деталі, вужчі за структурний елемент.
    tophat = top_hat(gray, params["tophat_width"])

    # Стадія 2. Контури Кенні; ROI накладаємо після них, щоб межа трапеції
    # сама не стала контуром.
    edges = canny(tophat, params["canny_kernel"], params["canny_sigma"],
                  params["canny_low"], params["canny_high"])
    edges_roi = edges & roi_mask(gray.shape, polygon)

    # Стадія 3. Прямі Хафа від нижнього краю фото до верху ROI.
    lanes, hough_space = detect_lanes(edges_roi, roi_top(gray.shape, polygon), config["hough"])

    return PipelineResult(stages=[("tophat", tophat), ("canny", edges),
                                  ("edges_roi", edges_roi), ("hough_space", hough_space)],
                          lanes=lanes)
