import numpy as np
import pytest

from src.preprocessing.image_utils import (
    adjust_brightness,
    crop_region,
    pixel_histogram,
    simple_threshold,
    to_grayscale,
)


# =============================================================================
# helpers - reusable synthetic images
# =============================================================================

def _solid_rgb(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _gradient_gray(h: int, w: int) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


# =============================================================================
# to_grayscale
# =============================================================================

class TestToGrayscale:
    """Test for to_grayscale()."""

    # ---- shape & d-type ----
    def test_returns_2d_from_3d_input(self) -> None:
        img = _solid_rgb(10, 10, (100, 150, 50))
        gray = to_grayscale(img)
        assert gray.ndim == 2

    def test_output_shape_matches_spatial_dimensions(self) -> None:
        img = _solid_rgb(13, 27, (0, 0, 0))
        gray = to_grayscale(img)
        assert gray.shape == (13, 27)

    def test_output_dtype_is_uint8(self) -> None:
        img = _solid_rgb(5, 5, (128, 128, 128))
        assert to_grayscale(img).dtype == np.uint8

    # ---- happy-path values ----
    def test_known_rgb_produces_correct_luminance(self) -> None:
        img = _solid_rgb(1, 1, (100, 150, 50))
        expected = int(round(100 * 0.2989 + 150 * 0.5870 + 50 * 0.1140))
        assert to_grayscale(img)[0, 0] == expected

    def test_pure_channels_respect_luminance_weights(self) -> None:
        """Green should map brightest, red mid, blue dimmest."""
        r = to_grayscale(_solid_rgb(1, 1, (255, 0, 0)))[0, 0]
        g = to_grayscale(_solid_rgb(1, 1, (0, 255, 0)))[0, 0]
        b = to_grayscale(_solid_rgb(1, 1, (0, 0, 255)))[0, 0]

        assert g > r > b

    # ---- edge cases ----

    def test_all_black_image(self) -> None:
        gray = to_grayscale(_solid_rgb(8, 8, (0, 0, 0)))
        assert np.all(gray == 0)

    def test_all_white_image(self) -> None:
        gray = to_grayscale(_solid_rgb(8, 8, (255, 255, 255)))
        assert np.all(gray == 255)

    def test_single_pixel_image(self) -> None:
        gray = to_grayscale(_solid_rgb(1, 1, (200, 200, 200)))
        assert gray.shape == (1, 1)
        assert gray.dtype == np.uint8

    def test_already_grayscale_passthrough(self) -> None:
        original = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        result = to_grayscale(original)
        np.testing.assert_array_equal(result, original)

    def test_rgba_ignores_alpha(self) -> None:
        """4-channel RGBA should ignore alpha channel."""
        rgba = np.zeros((3, 3, 4), dtype=np.uint8)
        rgba[..., :3] = [100, 150, 50]
        rgba[..., 3] = 255
        expected = to_grayscale(rgba[..., :3])
        np.testing.assert_array_equal(to_grayscale(rgba), expected)

    def test_invalid_shape_raises(self) -> None:
        bad = np.zeros((5, 5, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected RGB"):
            to_grayscale(bad)


# =============================================================================
# adjust_brightness
# =============================================================================

class TestAdjustBrightness:

    # ---- shape & d-type ----

    def test_output_dtype_is_uint8(self) -> None:
        img = _solid_rgb(5, 5, (100, 100, 100))
        assert adjust_brightness(img, 1.5).dtype == np.uint8

    def test_output_shape_matches_input_rgb(self) -> None:
        img = _solid_rgb(7, 11, (50, 50, 50))
        assert adjust_brightness(img, 2.0).shape == img.shape

    def test_output_shape_matches_input_gray(self) -> None:
        gray = np.full((7, 11), 100, dtype=np.uint8)
        assert adjust_brightness(gray, 2.0).shape == gray.shape

    # ---- happy path ----

    def test_factor_one_is_identity(self) -> None:
        img = _solid_rgb(4, 4, (123, 45, 67))
        np.testing.assert_array_equal(adjust_brightness(img, 1.0), img)

    def test_factor_two_doubles_values(self) -> None:
        img = _solid_rgb(3, 3, (50, 60, 70))
        result = adjust_brightness(img, 2.0)
        np.testing.assert_array_equal(result[0, 0], [100, 120, 140])

    def test_factor_half_halves_values(self) -> None:
        img = _solid_rgb(2, 2, (100, 200, 60))
        result = adjust_brightness(img, 0.5)
        np.testing.assert_array_equal(result[0, 0], [50, 100, 30])

    # ---- the critical overflow test ----

    def test_no_uint8_wraparound_with_float_factor(self) -> None:
        """200 * 2.0 must clamp to 255"""
        img = np.full((2, 2, 3), 200, dtype=np.uint8)
        result = adjust_brightness(img, 2.0)
        assert np.all(result == 255)

    def test_no_uint8_wraparound_with_int_factor(self) -> None:
        """Even when factor is a bare int (2), the fix must prevent wrap."""
        img = np.full((2, 2, 3), 200, dtype=np.uint8)
        result = adjust_brightness(img, 2)  # int, not float
        assert np.all(result == 255)

    # ---- edge cases ----

    def test_factor_zero_gives_all_black(self) -> None:
        img = _solid_rgb(4, 4, (255, 128, 64))
        result = adjust_brightness(img, 0.0)
        assert np.all(result == 0)

    def test_extreme_factor_clips_to_255(self) -> None:
        img = _solid_rgb(3, 3, (1, 1, 1))
        result = adjust_brightness(img, 10_000.0)
        assert np.all(result == 255)

    def test_negative_factor_clips_to_zero(self) -> None:
        img = _solid_rgb(3, 3, (100, 100, 100))
        result = adjust_brightness(img, -1.0)
        assert np.all(result == 0)

    def test_all_black_image_stays_black(self) -> None:
        img = _solid_rgb(3, 3, (0, 0, 0))
        result = adjust_brightness(img, 5.0)
        assert np.all(result == 0)

    def test_single_pixel(self) -> None:
        img = np.array([[[128, 64, 32]]], dtype=np.uint8)
        result = adjust_brightness(img, 2.0)
        assert result.shape == (1, 1, 3)
        np.testing.assert_array_equal(result[0, 0], [255, 128, 64])


# =============================================================================
# crop_region
# =============================================================================

class TestCropRegion:

    @pytest.fixture()
    def numbered_grid(self) -> np.ndarray:
        return np.arange(100, dtype=np.uint8).reshape(10, 10)

    # ---- shape & d-type ----

    def test_output_shape_matches_request_size(
            self, numbered_grid: np.ndarray
    ) -> None:
        cropped = crop_region(numbered_grid, x=2, y=3, w=5, h=4)
        assert cropped.shape == (4, 5)

    def test_output_shape_rgb(self) -> None:
        img = _solid_rgb(20, 30, (1, 2, 3))
        cropped = crop_region(img, x=5, y=5, w=10, h=8)
        assert cropped.shape == (8, 10, 3)

    def test_output_dtype_preserved(self, numbered_grid: np.ndarray) -> None:
        assert crop_region(numbered_grid, 0, 0, 5, 5).dtype == np.uint8

    # ---- happy path ----

    def test_crop_returns_correct_pixels(
            self, numbered_grid: np.ndarray
    ) -> None:
        cropped = crop_region(numbered_grid, x=1, y=2, w=3, h=2)
        # row 2, cols 1-3: [21, 22, 23]
        # row 3, cols 1-3: [31, 32, 33]
        expected = np.array([[21, 22, 23], [31, 32, 33]], dtype=np.uint8)
        np.testing.assert_array_equal(cropped, expected)

    def test_full_image_crop(self, numbered_grid: np.ndarray) -> None:
        cropped = crop_region(numbered_grid, x=0, y=0, w=10, h=10)
        np.testing.assert_array_equal(cropped, numbered_grid)

    # ---- edge cases ----

    def test_single_pixel_crop(self, numbered_grid: np.ndarray) -> None:
        cropped = crop_region(numbered_grid, x=7, y=3, w=1, h=1)
        assert cropped.shape == (1, 1)
        assert cropped[0, 0] == 37  # row 3, col 7

    def test_top_left_corner(self, numbered_grid: np.ndarray) -> None:
        cropped = crop_region(numbered_grid, x=0, y=0, w=1, h=1)
        assert cropped[0, 0] == 0

    def test_bottom_right_corner(self, numbered_grid: np.ndarray) -> None:
        cropped = crop_region(numbered_grid, x=9, y=9, w=1, h=1)
        assert cropped[0, 0] == 99

    # ---- out-of-bounds rejection ----

    def test_raises_when_x_plus_w_exceeds_width(
            self, numbered_grid: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            crop_region(numbered_grid, x=8, y=0, w=5, h=5)

    def test_raises_when_y_plus_h_exceeds_height(
            self, numbered_grid: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            crop_region(numbered_grid, x=0, y=8, w=5, h=5)

    def test_raises_on_negative_x(
            self, numbered_grid: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            crop_region(numbered_grid, x=-1, y=0, w=5, h=5)

    def test_raises_on_negative_y(
            self, numbered_grid: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            crop_region(numbered_grid, x=0, y=-1, w=5, h=5)

    def test_raises_when_completely_outside(
            self, numbered_grid: np.ndarray
    ) -> None:
        with pytest.raises(ValueError):
            crop_region(numbered_grid, x=50, y=50, w=10, h=10)


# =============================================================================
# simple_threshold
# =============================================================================

class TestSimpleThreshold:
    def temp(self):
        pass


# =============================================================================
# pixel_histogram
# =============================================================================

class TestPixelHistogram:
    def temp(self):
        pass
