"""Синтетичний кадр дороги з точно відомими смугами. Задача M2, власник: Сивопляс Максим.

Потрібен для автотестів пайплайнів і еталонних сценаріїв C1
(``docs/metrics.md``): на ньому ми знаємо правильну відповідь.
"""

import numpy as np
from PIL import Image, ImageDraw

# Смуги задано частками ширини й висоти: від нижнього краю до горизонту.
# Геометрія узгоджена з трапеціями roi.polygon і p3.src у config.json.
LEFT_LANE = ((0.20, 1.0), (0.46, 0.5))
RIGHT_LANE = ((0.80, 1.0), (0.54, 0.5))

# Оформлення кадру. Це не параметри методів обробки, а опис тестової сцени,
# тому вони сталі й лежать тут, а не в config.json.
HORIZON = 0.5                 # частка висоти, де небо переходить в асфальт
SKY_COLOR = (135, 206, 235)   # світло-блакитне небо
ROAD_COLOR = (80, 80, 80)     # сірий асфальт
LANE_COLOR = (255, 255, 255)  # біла розмітка
LANE_WIDTH = 0.0125           # товщина смуги, частка ширини кадру (4 px з 320)


def _to_pixels(lane: tuple, width: int, height: int) -> np.ndarray:
    """Точки смуги з часток у пікселі: ``x = fx · (W − 1)``, ``y = fy · (H − 1)``."""
    return np.array([(fx * (width - 1), fy * (height - 1)) for fx, fy in lane],
                    dtype=np.float64)


def make_road(width: int, height: int, brightness: float = 1.0,
              noise_sigma: float = 0.0, seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Малює кадр: світле «небо» над горизонтом, сірий асфальт, дві білі смуги.

    ``brightness`` множить усі яскравості (сценарій «темрява»),
    ``noise_sigma`` додає гаусів шум з генератора ``np.random.default_rng(seed)``
    (сценарій «шум»); значення обрізаємо до 0…255.

    Повертає ``(image, lanes)``: фото ``np.uint8 (height, width, 3)`` і
    еталонні смуги — масиви ``(2, 2)`` точок ``(x, y)`` у пікселях.
    """
    lanes = [_to_pixels(LEFT_LANE, width, height), _to_pixels(RIGHT_LANE, width, height)]

    # Усе полотно — асфальт, верхня частина до горизонту — небо.
    canvas = Image.new("RGB", (width, height), ROAD_COLOR)
    draw = ImageDraw.Draw(canvas)
    horizon_row = round(HORIZON * (height - 1))
    draw.rectangle([0, 0, width - 1, horizon_row], fill=SKY_COLOR)

    # Смуги — відрізки від нижнього краю до горизонту, товщиною щонайменше 1 px.
    line_width = max(1, round(LANE_WIDTH * width))
    for lane in lanes:
        draw.line([(float(x), float(y)) for x, y in lane], fill=LANE_COLOR, width=line_width)

    # Далі рахуємо в дійсних числах, щоб множення й шум не переповнили uint8.
    pixels = np.array(canvas, dtype=np.float64) * brightness
    if noise_sigma > 0:
        # Той самий seed дає той самий шум — тести й сценарії C1 відтворювані.
        rng = np.random.default_rng(seed)
        pixels = pixels + rng.normal(0.0, noise_sigma, size=pixels.shape)
    image = np.clip(np.round(pixels), 0, 255).astype(np.uint8)
    return image, lanes
