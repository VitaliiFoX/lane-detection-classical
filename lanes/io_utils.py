"""Читання фото й розмітки CULane, збереження проміжних результатів.

Задача F2, власник: Фояк Віталій. Це єдиний модуль, що кидає помилку на
поганий файл — методи й пайплайни працюють лише з масивами.
"""

from pathlib import Path

import numpy as np
from PIL import Image


class ImageLoadError(Exception):
    """Фото не вдалося прочитати: файлу немає, він порожній або пошкоджений."""


def load_rgb(path: Path | str) -> np.ndarray:
    """Читає фото будь-якого режиму Pillow і повертає його в RGB.

    Повертає ``np.uint8`` форми ``(H, W, 3)``. Режими L, RGBA, P тощо
    перетворює на RGB через ``Image.convert("RGB")``.

    Кидає ``ImageLoadError``, якщо файлу немає, він порожній, пошкоджений
    або має нульовий розмір.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Файл не знайдено: {path}")
    if path.stat().st_size == 0:
        raise ImageLoadError(f"Порожній файл: {path}")
    try:
        with Image.open(path) as image:
            # convert() змушує Pillow прочитати весь файл, тож обрізаний JPEG
            # дасть помилку тут, а не пізніше в пайплайні.
            rgb = image.convert("RGB")
    except (OSError, ValueError) as error:
        raise ImageLoadError(f"Не вдалося прочитати фото {path}: {error}") from error

    array = np.asarray(rgb, dtype=np.uint8)
    if array.size == 0:
        raise ImageLoadError(f"Фото нульового розміру: {path}")
    return array


def read_lanes(path: Path | str) -> list[np.ndarray]:
    """Читає розмітку CULane ``.lines.txt``.

    Кожен рядок файлу — одна смуга ``x1 y1 x2 y2 …``. Повертає список масивів
    ``np.float64`` форми ``(N, 2)``. Порожній файл, файл без смуг або
    відсутній файл — порожній список. Рядок з непарною кількістю чисел, менш
    ніж двома точками чи не числами пропускає.
    """
    path = Path(path)
    if not path.is_file():
        return []

    lanes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        # Смуга — пари (x, y), і щоб провести лінію, потрібно хоча б 2 точки.
        if len(values) < 4 or len(values) % 2 != 0:
            continue
        lanes.append(np.array(values, dtype=np.float64).reshape(-1, 2))
    return lanes


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Переводить масив у ``np.uint8`` для збереження чи показу.

    ``bool`` → 0/255; ``uint8`` — без змін; інші числа нормує лінійно:
    мінімум → 0, максимум → 255. Однотонний масив стає чорним.
    """
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if array.dtype == np.uint8:
        return array
    values = array.astype(np.float64)
    low, high = values.min(), values.max()
    if high == low:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.round((values - low) / (high - low) * 255).astype(np.uint8)


def save_png(path: Path | str, array: np.ndarray) -> None:
    """Зберігає масив як PNG, створюючи теки на шляху.

    Перетворення в ``uint8`` — ``to_uint8``. Підтримує форми ``(H, W)`` і
    ``(H, W, 3)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(array)).save(path)
