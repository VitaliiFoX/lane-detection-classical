"""Перетворення Хафа для прямих. Задача F8, власник: Фояк Віталій.

Спільний модуль для p1, p2, p4 (ADR-0006). Алгоритм крок за кроком —
``docs/pipelines.md``, розділ «Перетворення Хафа».

Пряму задаємо нормаллю: ``ρ = x·cos θ + y·sin θ``, де θ — кут нормалі до
осі x, а ρ — відстань від початку координат. Кожен білий піксель «голосує»
за всі прямі, що через нього проходять; пряма, за яку проголосувало багато
пікселів, — знайдена лінія.
"""

import numpy as np


def accumulator(edges: np.ndarray, rho_step: float, theta_step_deg: float,
                min_angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Голосування білих пікселів у просторі ``(ρ, θ)``.

    ``ρ = x·cos θ + y·sin θ``, θ від −90° до 90°; лишаємо лише θ, для яких
    кут лінії до горизонталі ``90° − |θ| ≥ min_angle_deg``. Повертає
    ``(acc, rhos, thetas_deg)``: акумулятор ``(n_rho, n_theta)`` цілих
    голосів і значення осей.
    """
    height, width = edges.shape
    # θ = 0° — вертикальна лінія, θ = ±90° — горизонтальна. Беремо −90°…90°, а
    # не 0°…180°: тоді майже вертикальна смуга має один запис біля θ = 0°, а не
    # два (біля 0° і біля 180° з протилежним ρ), і не дає подвійних піків.
    thetas_deg = np.arange(-90, 90, theta_step_deg, dtype=np.float64)
    # Смуги на фото круті, тож майже горизонтальні лінії (тіні, бампери,
    # стоп-лінії) відкидаємо одразу.
    angle_to_horizontal = 90 - np.abs(thetas_deg)
    thetas_deg = thetas_deg[angle_to_horizontal >= min_angle_deg]

    # ρ лежить у межах ±діагоналі зображення.
    diagonal = int(np.ceil(np.hypot(width, height)))
    n_rho = int(np.ceil(2 * diagonal / rho_step)) + 1
    rhos = np.arange(n_rho) * rho_step - diagonal

    ys, xs = np.nonzero(edges)
    if xs.size == 0 or thetas_deg.size == 0:
        return np.zeros((n_rho, thetas_deg.size), dtype=np.int64), rhos, thetas_deg

    thetas = np.deg2rad(thetas_deg)
    # Таблиця «піксель × кут»: рядок — білий піксель, стовпець — θ, значення — ρ.
    rho_values = np.outer(xs, np.cos(thetas)) + np.outer(ys, np.sin(thetas))
    rho_index = np.round((rho_values + diagonal) / rho_step).astype(np.int64)
    theta_index = np.arange(thetas.size)
    # Кожна пара (ρ, θ) — одна комірка акумулятора. Нумеруємо комірки одним
    # числом i_rho · n_theta + i_theta і рахуємо голоси np.bincount.
    cells = rho_index * thetas.size + theta_index
    votes = np.bincount(cells.ravel(), minlength=n_rho * thetas.size)
    return votes.reshape(n_rho, thetas.size), rhos, thetas_deg


def find_peaks(acc: np.ndarray, max_peaks: int, min_votes: int,
               suppress_rho_bins: int, suppress_theta_bins: int) -> list[tuple[int, int]]:
    """До ``max_peaks`` найсильніших піків з придушенням околиці.

    Повторюємо: максимум; менше ``min_votes`` — стоп; інакше запам'ятовуємо
    індекси ``(i_rho, i_theta)`` і обнуляємо прямокутник околиці в копії
    акумулятора. Придушення потрібне, бо одна товста смуга дає цілу «пляму»
    сильних комірок навколо свого піка.
    """
    work = acc.copy()
    peaks = []
    if work.size == 0:
        return peaks
    for _ in range(max_peaks):
        # argmax дає номер у «розгорнутому» масиві; divmod повертає рядок і стовпець.
        i_rho, i_theta = divmod(int(np.argmax(work)), work.shape[1])
        if work[i_rho, i_theta] < min_votes:
            break
        peaks.append((int(i_rho), int(i_theta)))
        work[max(0, i_rho - suppress_rho_bins):i_rho + suppress_rho_bins + 1,
             max(0, i_theta - suppress_theta_bins):i_theta + suppress_theta_bins + 1] = 0
    return peaks


def peaks_to_lanes(peaks: list[tuple[int, int]], rhos: np.ndarray, thetas_deg: np.ndarray,
                   height: int, y_top: int) -> list[np.ndarray]:
    """Перетворює піки на відрізки від ``y = height − 1`` до ``y = y_top``.

    З ``ρ = x·cos θ + y·sin θ`` маємо ``x = (ρ − y·sin θ) / cos θ``. Кожна
    смуга — ``np.ndarray (2, 2)``: нижня й верхня точки.
    """
    lanes = []
    for i_rho, i_theta in peaks:
        rho = rhos[i_rho]
        theta = np.deg2rad(thetas_deg[i_theta])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Горизонтальну лінію (cos θ = 0) не можна записати як x(y); такі θ
        # accumulator уже відкинув, це лише захист від ділення на нуль.
        if abs(cos_t) < 1e-9:
            continue
        points = [((rho - y * sin_t) / cos_t, float(y)) for y in (height - 1, y_top)]
        lanes.append(np.array(points, dtype=np.float64))
    return lanes


def detect_lanes(edges: np.ndarray, y_top: int,
                 params: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """Повний Хаф: акумулятор → піки → відрізки.

    ``params`` — секція ``hough`` з ``config.json``. Повертає
    ``(lanes, acc_image)``, де ``acc_image`` — акумулятор, нормований до
    0…255, для збереження як стадія ``hough_space``. Порожня маска — ``([], …)``.
    """
    acc, rhos, thetas_deg = accumulator(edges, params["rho_step"], params["theta_step_deg"],
                                        params["min_angle_deg"])
    peaks = find_peaks(acc, params["max_lanes"], params["min_votes"],
                       suppress_rho_bins=round(params["suppress_rho"] / params["rho_step"]),
                       suppress_theta_bins=round(params["suppress_theta_deg"]
                                                 / params["theta_step_deg"]))
    lanes = peaks_to_lanes(peaks, rhos, thetas_deg, edges.shape[0], y_top)

    acc_image = np.zeros(acc.shape, dtype=np.uint8)
    if acc.size and acc.max() > 0:
        acc_image = np.round(acc / acc.max() * 255).astype(np.uint8)
    return lanes, acc_image
