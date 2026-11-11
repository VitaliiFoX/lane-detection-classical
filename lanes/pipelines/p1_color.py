"""p1 «Колір»: HSV-маска → Собель → Хаф. Задача F12, власник: Фояк Віталій.

Специфікація — ``docs/pipelines.md``, розділ p1.
"""

import numpy as np

from lanes.pipelines.base import PipelineResult


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Стадії: ``hsv_mask`` → ``sobel_magnitude``, ``edges_roi`` → ``hough_space``."""
    raise NotImplementedError("F12")
