"""Метод Кенні. Задача M7, власник: Сивопляс Максим.

Категорія — контурний аналіз (стадія 2 p4). Гаусове згладжування тут —
внутрішній крок методу, погоджено з викладачкою. Градієнти — Собель з
``sobel.py`` (ADR-0010). Кроки — ``docs/pipelines.md``, розділ p4.
"""

import numpy as np

from lanes.filters import apply_kernel
from lanes.methods.sobel import sobel_gradients

# Зсуви (dy, dx) до сусіда вздовж градієнта для чотирьох округлених напрямів.
# Вісь y спрямована вниз, тому напрям 45° — це «праворуч і вниз».
DIRECTION_STEPS = {0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1)}

# Усі 8 сусідів пікселя: зсуви (dy, dx).
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Захист від похибки округлення дійсних чисел. Після згладжування однотонного
# фото сусідні пікселі можуть різнитися на ~1e-14, і Собель дає «градієнт»
# такого ж порядку. Градієнт, менший за цю частку яскравості фото, — не контур.
ROUNDING_TOLERANCE = 1e-9


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Гаусове ядро ``size × size`` (непарний розмір), нормоване до суми 1.

    ``g(x, y) = exp(−(x² + y²) / (2σ²))``, де ``x``, ``y`` — відстань від
    центру ядра. Ділимо на суму всіх елементів, щоб згладжування не
    змінювало середню яскравість.
    """
    r = size // 2
    offsets = np.arange(-r, r + 1, dtype=np.float64)
    # x змінюється вздовж рядка, y — вздовж стовпця: таблиця відстаней від центру.
    x, y = np.meshgrid(offsets, offsets)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _shifted(padded: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Масив сусідів: для пікселя ``(y, x)`` значення в ``(y + dy, x + dx)``.

    ``padded`` — вихідний масив з рамкою в 1 піксель, тому сусід є й у
    крайніх пікселів.
    """
    height, width = padded.shape[0] - 2, padded.shape[1] - 2
    return padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]


def non_max_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Потоншення контурів до одного пікселя.

    Напрям градієнта (радіани) округлюємо до 0°, 45°, 90°, 135°; піксель
    лишаємо, якщо він не менший за обох сусідів уздовж градієнта, інакше 0.

    Градієнт перпендикулярний до контуру, тож уздовж нього модуль спершу
    росте, а потім спадає; лишаємо тільки вершину цього «горба».
    """
    # Протилежні напрями (θ і θ + 180°) дають тих самих двох сусідів, тому
    # зводимо кут до 0…180°, а потім округлюємо до кроку 45°: 0, 45, 90, 135.
    degrees = np.rad2deg(angle) % 180
    sector = (np.round(degrees / 45).astype(np.int64) % 4) * 45

    # Рамка з нулів: за межами зображення контуру немає.
    padded = np.pad(magnitude, 1, mode="constant")
    keep = np.zeros(magnitude.shape, dtype=np.bool_)
    for direction, (dy, dx) in DIRECTION_STEPS.items():
        ahead = _shifted(padded, dy, dx)
        behind = _shifted(padded, -dy, -dx)
        is_peak = (magnitude >= ahead) & (magnitude >= behind)
        keep |= (sector == direction) & is_peak
    return np.where(keep, magnitude, 0.0)


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Гістерезис: слабкий піксель стає контуром, якщо з'єднаний із сильним.

    Поширюємо сильні пікселі на 8 сусідів серед слабких, доки маска
    змінюється. Повертає ``np.bool_ (H, W)``.

    Сильні пікселі — напевно контур. Слабкий може бути і продовженням
    контуру, і шумом; шум зазвичай ізольований, а продовження торкається
    сильного пікселя прямо або через ланцюжок інших слабких.
    """
    result = np.array(strong, dtype=np.bool_)
    weak = np.asarray(weak, dtype=np.bool_)
    while True:
        # near[y, x] — чи є серед 8 сусідів пікселя вже знайдений контур.
        padded = np.pad(result, 1, mode="constant")
        near = np.zeros(result.shape, dtype=np.bool_)
        for dy, dx in NEIGHBOURS:
            near |= _shifted(padded, dy, dx)
        grown = result | (weak & near)
        # Нових пікселів не додалося — поширення закінчено.
        if np.array_equal(grown, result):
            return result
        result = grown


def canny(gray: np.ndarray, kernel_size: int, sigma: float,
          low: float, high: float) -> np.ndarray:
    """Повний Кенні: згладжування → Собель → потоншення → подвійний поріг → гістерезис.

    ``low`` і ``high`` — частки максимуму модуля градієнта після
    потоншення. Однотонне зображення — порожня маска.
    """
    # 1. Згладжування прибирає дрібний шум, який інакше дав би хибні контури.
    smooth = apply_kernel(gray, gaussian_kernel(kernel_size, sigma))
    # 2. Градієнти Собеля: модуль — сила контуру, кут — напрям градієнта.
    gx, gy = sobel_gradients(smooth)
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)
    # 3. Потоншення: лишаємо тільки вершини модуля вздовж градієнта.
    thin = non_max_suppression(magnitude, angle)

    peak = thin.max() if thin.size else 0.0
    # Найсильніший «контур» — лише похибка округлення: фото однотонне, контурів немає.
    if peak <= ROUNDING_TOLERANCE * np.abs(smooth).max(initial=0.0):
        return np.zeros(thin.shape, dtype=np.bool_)
    # 4. Подвійний поріг відносно найсильнішого контуру.
    strong = thin >= high * peak
    weak = thin >= low * peak
    # 5. Слабкі пікселі лишаються лише поруч із сильними.
    return hysteresis(strong, weak)
