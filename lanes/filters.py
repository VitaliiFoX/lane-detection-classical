"""Застосування ядра до сірого зображення. Задача F1, власник: Фояк Віталій.

Спільний примітив для Собеля, Прюітта й гаусового згладжування в Кенні.
"""

import numpy as np


def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Кореляція сірого зображення з ядром непарного розміру.

    ``result[y, x] = Σ kernel[i, j] · image[y + i − r, x + j − r]``, де
    ``r = kernel_size // 2``. Ядро не перевертаємо (як у більшості
    підручників з обробки зображень); для симетричних ядер це те саме, що
    згортка. Краї доповнюємо повтором крайніх пікселів
    (``np.pad(mode="edge")``), тому результат того ж розміру, що й вхід.

    Повертає ``np.float64`` форми ``(H, W)``; вхід не змінює.
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    size = kernel.shape[0]
    if kernel.ndim != 2 or kernel.shape[1] != size or size % 2 == 0:
        raise ValueError("Ядро має бути квадратним і непарного розміру")

    r = size // 2
    height, width = image.shape
    # Рамка шириною r з повтором крайніх пікселів: тоді вікно є в кожного пікселя.
    padded = np.pad(image, r, mode="edge")

    result = np.zeros((height, width))
    # Замість циклу по пікселях — цикл по елементах ядра. Зріз
    # padded[i:i+H, j:j+W] — це все зображення, зсунуте на (i − r, j − r),
    # тобто для кожного пікселя одразу його сусід image[y + i − r, x + j − r].
    # Додаємо зсунуті копії з вагами kernel[i, j] — отримуємо ту саму суму з формули.
    for i in range(size):
        for j in range(size):
            result += kernel[i, j] * padded[i:i + height, j:j + width]
    return result
