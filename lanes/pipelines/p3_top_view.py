"""p3 «Вид зверху»: перспектива → Прюітт → ковзні вікна. Задача M8, власник: Сивопляс Максим.

Специфікація — ``docs/pipelines.md``, розділ p3.
"""

import numpy as np

from lanes.pipelines.base import PipelineResult


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Стадії: ``top_view`` → ``prewitt_magnitude``, ``edges`` → ``windows``.

    Смуги, знайдені на виді зверху, повертає в координатах фото.
    """
    raise NotImplementedError("M8")
