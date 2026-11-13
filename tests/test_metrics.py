"""Тести metrics (F5): MSE, PSNR, SSIM, IoU."""

import numpy as np
import pytest

from lanes.metrics import iou, mse, psnr, ssim

# Еталон 3×3: вертикальна смуга в середньому стовпці.
REFERENCE = np.array([[0, 255, 0],
                      [0, 255, 0],
                      [0, 255, 0]], dtype=np.uint8)
EMPTY = np.zeros((3, 3), dtype=np.uint8)


def test_mse_by_hand():
    # Не збігаються 3 пікселі з 9, у кожному різниця 255:
    # MSE = 3 · 255² / 9 = 65025 / 3 = 21675.
    assert mse(REFERENCE, EMPTY) == pytest.approx(21675)


def test_psnr_by_hand():
    # PSNR = 10 · log10(255² / 21675) = 10 · log10(3) ≈ 4.771 дБ.
    assert psnr(REFERENCE, EMPTY) == pytest.approx(10 * np.log10(3))


def test_identical_masks():
    assert mse(REFERENCE, REFERENCE) == 0
    assert psnr(REFERENCE, REFERENCE, cap=100.0) == 100.0
    assert ssim(REFERENCE, REFERENCE) == pytest.approx(1.0)
    assert iou(REFERENCE, REFERENCE) == 1.0


def test_ssim_single_window_by_hand():
    # Усе 0 проти усе 255, зображення менше за вікно → одне вікно:
    # μx = 0, μy = 255, σ = 0, тож SSIM = C1 · C2 / ((255² + C1) · C2) = C1 / (255² + C1),
    # C1 = (0.01 · 255)² = 6.5025 → SSIM ≈ 1.0e-4.
    black = np.zeros((3, 3))
    white = np.full((3, 3), 255.0)
    c1 = (0.01 * 255) ** 2
    assert ssim(black, white) == pytest.approx(c1 / (255 ** 2 + c1))


def test_ssim_windowed_matches_single_window():
    # Зображення 7×7: з вікном 7 є рівно одне вікно (гілка інтегрального
    # зображення), з вікном 8 — гілка «одне вікно на все». Результат має збігтися.
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (7, 7)).astype(np.float64)
    y = rng.integers(0, 256, (7, 7)).astype(np.float64)
    assert ssim(x, y, window=7) == pytest.approx(ssim(x, y, window=8))


def test_iou_by_hand():
    # Еталон — стовпець з 3 пікселів, знайдено 2 з них і ще 1 зайвий:
    # перетин 2, об'єднання 4 → IoU = 0.5.
    found = np.array([[0, 255, 0],
                      [0, 255, 0],
                      [255, 0, 0]], dtype=np.uint8)
    assert iou(REFERENCE, found) == 0.5


def test_iou_empty():
    assert iou(EMPTY, EMPTY) == 1.0
    assert iou(REFERENCE, EMPTY) == 0.0
