"""p2 «Оцу»: вирівнювання гістограми → поріг Оцу → Хаф. Задача F13, власник: Фояк Віталій.

Специфікація — ``docs/pipelines.md``, розділ p2.
"""

import numpy as np

from lanes.pipelines.base import PipelineResult


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Стадії: ``equalized`` → ``otsu`` → ``hough_space``."""
    raise NotImplementedError("F13")
