"""Реєстр пайплайнів. Раннер, звіт і перевірки беруть список звідси.

Новий пайплайн — файл ``pN_<назва>.py`` з функцією ``run`` і рядок нижче.
"""

from lanes.pipelines import p0_baseline, p1_color, p2_otsu, p3_top_view, p4_tophat
from lanes.pipelines.base import PipelineInfo, PipelineResult

REGISTRY: dict[str, PipelineInfo] = {
    info.name: info
    for info in (
        PipelineInfo("p0", "Базова лінія (порожня маска)", "Фояк Віталій", p0_baseline.run),
        PipelineInfo("p1", "Колір", "Фояк Віталій", p1_color.run),
        PipelineInfo("p2", "Оцу", "Фояк Віталій", p2_otsu.run),
        PipelineInfo("p3", "Вид зверху", "Сивопляс Максим", p3_top_view.run),
        PipelineInfo("p4", "Top-hat", "Сивопляс Максим", p4_tophat.run),
    )
}

__all__ = ["REGISTRY", "PipelineInfo", "PipelineResult"]
