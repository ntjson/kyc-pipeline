import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using ITU-R BT.601 luminance weights."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Expected RGB image (H,W,3), got shape {img.shape}")
    gray_float = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return np.round(gray_float).astype(np.uint8)


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel intensities by factor, clamping to [0, 255]."""
    brightened = img.astype(np.float32) * float(factor)
    return np.clip(brightened, 0, 255).astype(np.uint8)


def crop_region(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Extract a rectangular ROI. Raises on out-of-bounds."""
    img_h, img_w = img.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        raise ValueError(
            f"Crop region ({x},{y},{w},{h}) exceeds image bounds ({img_w},{img_h})"
        )
    return img[y: y + h, x: x + w]


def simple_threshold(gray: np.ndarray, thresh: int = 128) -> np.ndarray:
    """Binary threshold: pixel > thresh → 255, else → 0."""
    return np.where(gray > thresh, np.uint8(255), np.uint8(0))


def pixel_histogram(gray: np.ndarray) -> np.ndarray:
    """Return a 256-element array of per-intensity pixel counts."""
    if gray.dtype != np.uint8:
        raise ValueError(f"Expected uint8 grayscale, got {gray.dtype}")
    return np.bincount(gray.ravel(), minlength=256)
