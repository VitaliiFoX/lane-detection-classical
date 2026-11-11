"""Спільний контракт пайплайнів (ADR-0005). Власник: Фояк Віталій."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class PipelineResult:
    """Що повертає пайплайн.

    ``stages`` — проміжні результати ``(назва, масив)`` у порядку стадій;
    раннер збереже їх як ``01_<назва>.png``, ``02_…``.
    ``lanes`` — знайдені смуги, кожна ``np.ndarray (N, 2)`` точок ``(x, y)``
    у координатах вхідного фото.
    """

    stages: list[tuple[str, np.ndarray]] = field(default_factory=list)
    lanes: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineInfo:
    """Запис реєстру: ідентифікатор, назва, автор і функція ``run(image, config)``."""

    name: str
    title: str
    owner: str
    run: Callable[[np.ndarray, dict], PipelineResult]
