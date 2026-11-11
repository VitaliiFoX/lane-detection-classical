"""p0 — базова лінія: не знаходить жодної смуги. Власник: Фояк Віталій.

Показує, які метрики дає порожня маска (``docs/metrics.md``, «Пастка»).
"""

import numpy as np

from lanes.pipelines.base import PipelineResult


def run(_image: np.ndarray, _config: dict) -> PipelineResult:
    """Повертає результат без стадій і без смуг; аргументи — лише за контрактом."""
    return PipelineResult()
