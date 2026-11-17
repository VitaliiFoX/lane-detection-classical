"""Перспективне перетворення у «вид зверху». Задача M3, власник: Сивопляс Максим.

Категорія — геометричні перетворення (стадія 1 p3).

Перспективне перетворення (гомографія) задається матрицею 3×3 в однорідних
координатах: точка ``(x, y)`` стає вектором ``(x, y, 1)``, множиться на
матрицю, а результат ``(u', v', w)`` ділиться на ``w``:
``u = u' / w``, ``v = v' / w``. Саме ділення на ``w`` робить далекі об'єкти
меншими — це й є перспектива.
"""

import numpy as np

# Найменший модуль w, на який ще ділимо: точка з w = 0 лежить «на нескінченності».
MIN_W = 1e-12


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Матриця 3×3, що переводить 4 точки ``src`` у 4 точки ``dst``.

    Для кожної пари точок два рівняння; 8 невідомих при ``h33 = 1``
    розв'язуємо ``np.linalg.solve``. ``src``, ``dst`` — ``(4, 2)``.

    З ``u = (h11·x + h12·y + h13) / (h31·x + h32·y + 1)`` після множення на
    знаменник: ``h11·x + h12·y + h13 − u·x·h31 − u·y·h32 = u``; так само для
    ``v`` з другим рядком матриці. Якщо три точки лежать на одній прямій чи
    збігаються, розв'язку немає — ``np.linalg.LinAlgError``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    # Невідомі в порядку h11, h12, h13, h21, h22, h23, h31, h32.
    system = np.zeros((8, 8))
    right = np.zeros(8)
    for k in range(4):
        x, y = src[k]
        u, v = dst[k]
        system[2 * k] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        right[2 * k] = u
        system[2 * k + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        right[2 * k + 1] = v
    solution = np.linalg.solve(system, right)
    # Дев'ятий елемент h33 = 1 дописуємо в кінець і складаємо матрицю 3×3.
    return np.append(solution, 1.0).reshape(3, 3)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Застосовує матрицю до точок ``(N, 2)`` в однорідних координатах.

    ``(u', v', w) = matrix · (x, y, 1)``, результат ``(u'/w, v'/w)``,
    ``np.float64 (N, 2)``.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    # Дописуємо стовпець одиниць: (N, 2) → (N, 3).
    homogeneous = np.column_stack([points, np.ones(len(points))])
    # Кожен рядок множимо на матрицю: (matrix · pᵀ)ᵀ = p · matrixᵀ.
    mapped = homogeneous @ matrix.T
    w = mapped[:, 2]
    safe_w = np.where(np.abs(w) > MIN_W, w, 1.0)
    return mapped[:, :2] / safe_w[:, np.newaxis]


def top_view_rectangle(out_size: tuple[int, int], margin: float) -> np.ndarray:
    """Чотири точки ``dst`` — прямокутник виду зверху з відступами з боків.

    ``out_size`` — ``(ширина, висота)``; ``margin`` — частка ширини, яку
    лишаємо зліва й справа, щоб було видно сусідні смуги. Порядок точок, як у
    ``p3.src``: верх-ліво, верх-право, низ-право, низ-ліво.
    """
    out_width, out_height = out_size
    left = margin * (out_width - 1)
    right = (out_width - 1) - left
    bottom = out_height - 1
    return np.array([[left, 0], [right, 0], [right, bottom], [left, bottom]], dtype=np.float64)


def extend_to_row(lane: np.ndarray, y_bottom: float) -> np.ndarray:
    """Продовжує смугу вниз до рядка ``y_bottom`` по прямій останнього відрізка.

    Вид зверху закінчується на лінії капота, а розмітку CULane проведено до
    нижнього краю фото. Хаф продовжує свої прямі до низу — робимо те саме для
    кривих p3: через дві найнижчі точки ``(x1, y1)``, ``(x2, y2)`` проводимо
    пряму й беремо її точку на рядку ``y_bottom``:
    ``x = x2 + (x2 − x1) · (y_bottom − y2) / (y2 − y1)``.

    Повертає точки, впорядковані згори вниз, з доданою нижньою точкою. Смугу
    з менш ніж двох точок або таку, що вже сягає ``y_bottom``, не змінює.
    """
    lane = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
    if len(lane) < 2:
        return lane
    # Впорядковуємо точки за y: останні дві — найближчі до авто.
    lane = lane[np.argsort(lane[:, 1])]
    (x1, y1), (x2, y2) = lane[-2], lane[-1]
    # Горизонтальний останній відрізок (y2 = y1) продовжити вниз неможливо.
    if y2 >= y_bottom or y2 == y1:
        return lane
    x_bottom = x2 + (x2 - x1) * (y_bottom - y2) / (y2 - y1)
    return np.vstack([lane, [x_bottom, y_bottom]])


def warp(image: np.ndarray, matrix: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """Деформує зображення зворотним відображенням.

    ``matrix`` переводить фото у результат; ``out_size`` — ``(ширина,
    висота)`` результату. Для кожного пікселя результату обчислюємо точку у
    фото через обернену матрицю й беремо найближчий піксель; поза фото — 0.
    Працює для сірих ``(H, W)`` і кольорових ``(H, W, 3)`` масивів.

    Чому зворотне: якщо йти від пікселів фото до результату, частина
    пікселів результату лишиться порожньою («дірки»). Ідучи від результату,
    кожен його піксель гарантовано отримує значення.
    """
    out_width, out_height = out_size
    height, width = image.shape[:2]

    # Координати всіх пікселів результату: xs і ys форми (out_height, out_width).
    xs, ys = np.meshgrid(np.arange(out_width), np.arange(out_height))
    # Однорідні координати стовпцями: (3, N), де N — кількість пікселів результату.
    target = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    # Обернене відображення: шукаємо p з рівняння matrix · p = target.
    # np.linalg.solve розв'язує його одразу для всіх N стовпців.
    source = np.linalg.solve(matrix, target)
    w = source[2]
    valid_w = np.abs(w) > MIN_W
    safe_w = np.where(valid_w, w, 1.0)
    # Найближчий піксель фото — округлення координат до цілих.
    source_x = np.round(source[0] / safe_w)
    source_y = np.round(source[1] / safe_w)

    inside = (valid_w & (source_x >= 0) & (source_x < width)
              & (source_y >= 0) & (source_y < height))
    result = np.zeros((out_height * out_width,) + image.shape[2:], dtype=image.dtype)
    # Копіюємо лише ті пікселі, чия точка потрапила в межі фото; решта лишається 0.
    result[inside] = image[source_y[inside].astype(np.int64), source_x[inside].astype(np.int64)]
    return result.reshape((out_height, out_width) + image.shape[2:])
