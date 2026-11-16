"""p3 «Вид зверху»: перспектива → Прюітт → ковзні вікна. Задача M8, власник: Сивопляс Максим.

Специфікація — ``docs/pipelines.md``, розділ p3.
"""

import numpy as np

from lanes.methods.color import rgb_to_gray
from lanes.methods.morphology import erode
from lanes.methods.perspective import homography, top_view_rectangle, transform_points, warp
from lanes.methods.prewitt import PREWITT_X, prewitt_magnitude
from lanes.methods.sliding_windows import detect_lanes_top_view
from lanes.pipelines.base import PipelineResult
from lanes.roi import roi_polygon


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Стадії: ``top_view`` → ``prewitt_magnitude``, ``edges`` → ``windows``.

    Смуги, знайдені на виді зверху, повертає в координатах фото.
    """
    params = config["p3"]
    out_size = (params["warp_size"][0], params["warp_size"][1])
    view_shape = (out_size[1], out_size[0])
    gray = rgb_to_gray(image)

    # Стадія 1. Трапеція дороги на фото (p3.src) стає прямокутником виду зверху.
    photo_points = np.array(roi_polygon(gray.shape, params["src"]))
    view_points = top_view_rectangle(out_size, params["warp_margin"])
    try:
        to_top = homography(photo_points, view_points)
        to_photo = homography(view_points, photo_points)
    except np.linalg.LinAlgError:
        # Фото 1×1: точки трапеції збіглися, виду зверху не існує.
        empty = np.zeros(view_shape)
        return PipelineResult(stages=[("top_view", empty), ("prewitt_magnitude", empty),
                                      ("edges", empty > 0),
                                      ("windows", np.zeros(view_shape + (3,), dtype=np.uint8))])
    top_view = warp(gray, to_top, out_size)

    # Стадія 2. Контури Прюітта, сильніші за частку найбільшого модуля.
    magnitude = prewitt_magnitude(top_view)
    edges = magnitude > params["prewitt_threshold"] * magnitude.max()
    # Кути виду зверху лежать поза фото й заповнені нулями; межа «фото —
    # порожнеча» дала б хибний контур. Деформуємо так само фото з одиниць —
    # це маска «тут є фото», звужуємо її ерозією на розмір ядра Прюітта
    # й лишаємо тільки контури всередині.
    kernel_size = PREWITT_X.shape[0]
    inside_photo = erode(warp(np.ones(gray.shape), to_top, out_size), kernel_size, kernel_size)
    edges = edges & (inside_photo > 0)

    # Стадія 3. Смуги на виді зверху й повернення їх у координати фото.
    top_lanes, windows = detect_lanes_top_view(edges, params)
    lanes = [transform_points(lane, to_photo) for lane in top_lanes]

    return PipelineResult(stages=[("top_view", top_view), ("prewitt_magnitude", magnitude),
                                  ("edges", edges), ("windows", windows)],
                          lanes=lanes)
