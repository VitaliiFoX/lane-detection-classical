"""Тести io_utils (F2): читання фото, розмітки й збереження PNG."""

import numpy as np
import pytest
from PIL import Image

from lanes.io_utils import ImageLoadError, load_rgb, read_lanes, save_png, to_uint8


def test_rgb_round_trip(tmp_path):
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[1, 2] = (10, 200, 30)
    save_png(tmp_path / "a.png", image)
    np.testing.assert_array_equal(load_rgb(tmp_path / "a.png"), image)


def test_gray_image_becomes_three_equal_channels(tmp_path):
    Image.fromarray(np.full((3, 5), 77, dtype=np.uint8), mode="L").save(tmp_path / "g.png")
    image = load_rgb(tmp_path / "g.png")
    assert image.shape == (3, 5, 3)
    assert (image == 77).all()


def test_rgba_drops_alpha(tmp_path):
    Image.new("RGBA", (2, 2), (1, 2, 3, 0)).save(tmp_path / "t.png")
    image = load_rgb(tmp_path / "t.png")
    assert image.shape == (2, 2, 3)
    assert tuple(image[0, 0]) == (1, 2, 3)


def test_missing_file(tmp_path):
    with pytest.raises(ImageLoadError):
        load_rgb(tmp_path / "немає.jpg")


def test_empty_file(tmp_path):
    (tmp_path / "empty.jpg").write_bytes(b"")
    with pytest.raises(ImageLoadError):
        load_rgb(tmp_path / "empty.jpg")


def test_corrupted_file(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"\xff\xd8\xff not a real jpeg")
    with pytest.raises(ImageLoadError):
        load_rgb(tmp_path / "bad.jpg")


def test_truncated_jpeg(tmp_path):
    Image.new("RGB", (64, 64), (200, 10, 10)).save(tmp_path / "full.jpg")
    data = (tmp_path / "full.jpg").read_bytes()
    (tmp_path / "cut.jpg").write_bytes(data[: len(data) // 2])
    with pytest.raises(ImageLoadError):
        load_rgb(tmp_path / "cut.jpg")


def test_read_lanes(tmp_path):
    path = tmp_path / "f.lines.txt"
    path.write_text("10 590 20.5 400 \n100 590 110 400 120 300\n", encoding="utf-8")
    lanes = read_lanes(path)
    assert len(lanes) == 2
    np.testing.assert_allclose(lanes[0], [[10, 590], [20.5, 400]])
    assert lanes[1].shape == (3, 2)


def test_read_lanes_skips_bad_lines(tmp_path):
    path = tmp_path / "f.lines.txt"
    path.write_text("1 2 3\nx y z w\n5 6\n\n1 2 3 4\n", encoding="utf-8")
    lanes = read_lanes(path)
    assert len(lanes) == 1
    np.testing.assert_allclose(lanes[0], [[1, 2], [3, 4]])


def test_read_lanes_empty_and_missing(tmp_path):
    (tmp_path / "e.lines.txt").write_text("", encoding="utf-8")
    assert not read_lanes(tmp_path / "e.lines.txt")
    assert not read_lanes(tmp_path / "немає.lines.txt")


def test_to_uint8_conversions():
    np.testing.assert_array_equal(to_uint8(np.array([True, False])), [255, 0])
    # 0 → 0, 5 → 5/10·255 = 127.5 → 128, 10 → 255
    np.testing.assert_array_equal(to_uint8(np.array([0.0, 5.0, 10.0])), [0, 128, 255])
    np.testing.assert_array_equal(to_uint8(np.full(3, 4.2)), [0, 0, 0])


def test_save_png_creates_folders(tmp_path):
    save_png(tmp_path / "a" / "b" / "mask.png", np.eye(3, dtype=bool))
    saved = np.asarray(Image.open(tmp_path / "a" / "b" / "mask.png"))
    np.testing.assert_array_equal(saved, np.eye(3) * 255)
