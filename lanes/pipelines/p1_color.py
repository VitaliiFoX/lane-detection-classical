"""p1 «Колір»: HSV-маска → Собель → Хаф. Задача F12, власник: Фояк Віталій.

Специфікація — ``docs/pipelines.md``, розділ p1.
"""

import numpy as np

from lanes.methods.color import white_yellow_mask
from lanes.methods.hough import detect_lanes
from lanes.methods.sobel import sobel_magnitude
from lanes.pipelines.base import PipelineResult
from lanes.roi import roi_mask, roi_top


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Стадії: ``hsv_mask`` → ``sobel_magnitude``, ``edges_roi`` → ``hough_space``."""
    params = config["p1"]
    polygon = config["roi"]["polygon"]
    shape = image.shape[:2]

    # Стадія 1. Пікселі, схожі на білу або жовту фарбу розмітки.
    mask = white_yellow_mask(image, params["white_s_max"], params["white_v_min"],
                             params["yellow_h_min"], params["yellow_h_max"],
                             params["yellow_s_min"], params["yellow_v_min"])

    # Стадія 2. Собель по масці як по сірому зображенню 0/255: контури — межі
    # плям фарби. Лишаємо сильні, потім накладаємо ROI, щоб межа трапеції сама
    # не стала контуром.
    magnitude = sobel_magnitude(mask * 255.0)
    edges = magnitude > params["sobel_threshold"] * magnitude.max()
    edges_roi = edges & roi_mask(shape, polygon)

    # Стадія 3. Прямі Хафа від нижнього краю фото до верху ROI.
    lanes, hough_space = detect_lanes(edges_roi, roi_top(shape, polygon), config["hough"])

    return PipelineResult(stages=[("hsv_mask", mask), ("sobel_magnitude", magnitude),
                                  ("edges_roi", edges_roi), ("hough_space", hough_space)],
                          lanes=lanes)
