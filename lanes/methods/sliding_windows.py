"""Пошук смуг на виді зверху: гістограма стовпців і ковзні вікна.

Задача M6, власник: Сивопляс Максим. Категорія — гістограмний аналіз
(стадія 3 p3). Кроки — ``docs/pipelines.md``, розділ p3.

На виді зверху смуги майже вертикальні, тому стовпець, через який проходить
смуга, містить багато білих пікселів контуру. Піки гістограми стовпців дають
початок кожної смуги, а ковзні вікна простежують її знизу вгору.
"""

import numpy as np
from PIL import Image, ImageDraw

# Оформлення малюнка windows.png — лише візуалізація, на результат не впливає.
WINDOW_COLOR = (0, 255, 0)
CURVE_COLOR, CURVE_WIDTH = (255, 0, 0), 3


def column_histogram(edges: np.ndarray) -> np.ndarray:
    """Кількість білих пікселів у кожному стовпці нижньої половини маски.

    Беремо нижню половину, бо там смуги найближчі до авто й найчіткіші.
    Повертає цілі числа форми ``(W,)``.
    """
    height = edges.shape[0]
    # Сума True по рядках нижньої половини: True рахується як 1.
    return edges[height // 2:].sum(axis=0).astype(np.int64)


def find_base_peaks(histogram: np.ndarray, max_lanes: int, min_distance: int,
                    min_height: int) -> list[int]:
    """До ``max_lanes`` стовпців-піків, віддалених не менше ніж на ``min_distance``.

    Пік нижчий за ``min_height`` не враховуємо. Повертає ``x`` зліва направо.

    Повторюємо: найвищий стовпець; нижчий за ``min_height`` — стоп; інакше
    запам'ятовуємо його й обнуляємо в копії гістограми всі стовпці, ближчі
    за ``min_distance``, — два краї однієї смуги не стануть двома піками.
    """
    work = np.array(histogram, dtype=np.int64)
    peaks = []
    if work.size == 0:
        return peaks
    for _ in range(max_lanes):
        x = int(np.argmax(work))
        if work[x] < min_height or work[x] == 0:
            break
        peaks.append(x)
        work[max(0, x - min_distance + 1):x + min_distance] = 0
    return sorted(peaks)


def track_lane(edges: np.ndarray, x_base: int, n_windows: int, margin: int,
               min_pixels: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    """Ковзні вікна знизу вгору від стовпця ``x_base``.

    Повертає ``(xs, ys, windows)``: координати зібраних білих пікселів і
    прямокутники вікон ``(x0, y0, x1, y1)`` для малюнка (``x1``, ``y1`` — не
    включно).

    Висота вікна ``H // n_windows``, ширина ``2 · margin``. Якщо у вікні
    більше за ``min_pixels`` білих пікселів, центр наступного вікна — їхнє
    середнє ``x``: так вікна «повзуть» за вигином смуги.
    """
    height, width = edges.shape
    window_height = height // n_windows
    x_center = int(x_base)
    xs_parts, ys_parts, windows = [], [], []
    for index in range(n_windows):
        # Вікно номер index рахуємо від нижнього краю маски.
        y1 = height - index * window_height
        y0 = y1 - window_height
        x0 = max(0, x_center - margin)
        x1 = min(width, x_center + margin)
        windows.append((x0, y0, x1, y1))

        # Координати білих пікселів усередині вікна; nonzero рахує їх від
        # кута вікна, тому додаємо зсув (x0, y0).
        rows, cols = np.nonzero(edges[y0:y1, x0:x1])
        xs_parts.append(cols + x0)
        ys_parts.append(rows + y0)
        if cols.size > min_pixels:
            x_center = int(round(float(np.mean(cols + x0))))
    return np.concatenate(xs_parts), np.concatenate(ys_parts), windows


def _draw_debug(edges: np.ndarray, lanes: list[np.ndarray],
                windows: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Контури (білі на чорному) з рамками вікон і знайденими кривими."""
    gray = edges.astype(np.uint8) * 255
    canvas = Image.fromarray(np.stack([gray, gray, gray], axis=2))
    draw = ImageDraw.Draw(canvas)
    for x0, y0, x1, y1 in windows:
        if x1 > x0 and y1 > y0:
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=WINDOW_COLOR)
    for lane in lanes:
        draw.line([(float(x), float(y)) for x, y in lane], fill=CURVE_COLOR, width=CURVE_WIDTH)
    return np.array(canvas, dtype=np.uint8)


def detect_lanes_top_view(edges: np.ndarray,
                          params: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """Усі кроки на виді зверху: піки → вікна → парабола ``x = a·y² + b·y + c``.

    ``params`` — секція ``p3`` з ``config.json``. Повертає ``(lanes, debug)``:
    смуги в координатах виду зверху (``(n_windows + 1, 2)`` точок кожна) і
    кольоровий малюнок ``np.uint8 (H, W, 3)`` з вікнами й кривими.

    Криву шукаємо як ``x`` від ``y``, а не навпаки: смуга майже вертикальна,
    і для одного ``y`` у неї рівно один ``x``.
    """
    height = edges.shape[0]
    n_windows = params["n_windows"]
    degree = params["poly_degree"]
    peaks = find_base_peaks(column_histogram(edges), params["max_lanes"],
                            params["peak_min_distance"], params["min_pixels"])
    # Рівні y, на яких беремо точки кривої: межі вікон від верху до низу.
    levels = np.linspace(0, height - 1, n_windows + 1)

    lanes, all_windows = [], []
    for x_base in peaks:
        xs, ys, windows = track_lane(edges, x_base, n_windows, params["window_margin"],
                                     params["min_pixels"])
        all_windows.extend(windows)
        # Многочлен ступеня degree визначають щонайменше degree + 1 точок з
        # різними y; інакше підгонка не має єдиного розв'язку.
        if np.unique(ys).size <= degree:
            continue
        # Метод найменших квадратів: коефіцієнти, за яких сума квадратів
        # відхилень x пікселів від кривої найменша.
        coefficients = np.polyfit(ys, xs, degree)
        lanes.append(np.column_stack([np.polyval(coefficients, levels), levels]))
    return lanes, _draw_debug(edges, lanes, all_windows)
