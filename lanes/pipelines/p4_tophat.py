"""p4 «Top-hat»: морфологічний top-hat → Кенні → Хаф. Задача M9, власник: Сивопляс Максим.

Специфікація — ``docs/pipelines.md``, розділ p4.
"""

import numpy as np

from lanes.pipelines.base import PipelineResult


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Стадії: ``tophat`` → ``canny``, ``edges_roi`` → ``hough_space``."""
    raise NotImplementedError("M9")
