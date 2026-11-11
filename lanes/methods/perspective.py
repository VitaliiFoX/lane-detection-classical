"""Перспективне перетворення у «вид зверху». Задача M3, власник: Сивопляс Максим.

Категорія — геометричні перетворення (стадія 1 p3).
"""

import numpy as np


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Матриця 3×3, що переводить 4 точки ``src`` у 4 точки ``dst``.

    Для кожної пари точок два рівняння; 8 невідомих при ``h33 = 1``
    розв'язуємо ``np.linalg.solve``. ``src``, ``dst`` — ``(4, 2)``.
    """
    raise NotImplementedError("M3")


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Застосовує матрицю до точок ``(N, 2)`` в однорідних координатах."""
    raise NotImplementedError("M3")


def warp(image: np.ndarray, matrix: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """Деформує зображення зворотним відображенням.

    ``matrix`` переводить фото у результат; ``out_size`` — ``(ширина,
    висота)`` результату. Для кожного пікселя результату обчислюємо точку у
    фото через обернену матрицю й беремо найближчий піксель; поза фото — 0.
    Працює для сірих ``(H, W)`` і кольорових ``(H, W, 3)`` масивів.
    """
    raise NotImplementedError("M3")
