__all__ = [
    "QualityAction",
    "QualityReport",
    "PreprocessResult",
    "PreprocessorConfig",
    "DocumentPreprocessor",
]

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class QualityAction(Enum):
    """Outcome of quality assessment."""
    PASS = "pass"
    ENHANCED = "enhanced"
    REJECT = "reject"


@dataclass
class QualityIssue:
    """Single quality issue found during assessment."""

    code: str
    message: str
    is_critical: bool = False


@dataclass
class QualityReport:
    """Aggregated quality assessment of a document image."""

    action: QualityAction
    blur_score: float
    brightness_mean: float
    glare_ratio: float
    resolution_ok: bool
    skew_angle: float | None
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def hints(self) -> list[str]:
        """Human-readable hints derived from issues."""
        return [issue.message for issue in self.issues]

    @property
    def issue_codes(self) -> set[str]:
        """Set of machine-readable issue codes for quick lookup."""
        return {issue.code for issue in self.issues}


@dataclass
class PreprocessResult:
    """Result returned by the preprocessing pipeline."""

    image: np.ndarray | None
    gray: np.ndarray | None
    quality: QualityReport


@dataclass
class PreprocessorConfig:
    """Centralised, documented thresholds for the preprocessing pipeline."""

    blur_threshold: float = 100.0
    blur_reject_threshold: float = 50.0
    min_width: int = 640
    min_height: int = 480
    glare_max_ratio: float = 0.15
    glare_pixel_value: int = 240
    brightness_low: float = 40.0
    brightness_high: float = 200.0
    max_skew_degrees: float = 15.0
    deskew_min_angle: float = 1.0
    denoise_h: int = 10
    quality_sample_size: tuple[int, int] = (640, 480)


class QualityCheck(ABC):
    """Interface for a single, pluggable quality check.

    Implement this to add new checks (e.g. colour-space validation)
    without modifying the existing assessment pipeline.
    """

    @abstractmethod
    def __call__(
            self,
            image: np.ndarray,
            gray: np.ndarray,
            config: PreprocessorConfig,
    ) -> list[QualityIssue]:
        """Run the check and return any issues found."""
        ...


class ResolutionCheck(QualityCheck):
    """Verify the image meets minimum pixel dimensions."""

    def __call__(
            self,
            image: np.ndarray,
            gray: np.ndarray,
            config: PreprocessorConfig,
    ) -> list[QualityIssue]:
        h, w = image.shape[:2]
        if h < config.min_height or w < config.min_width:
            return [
                QualityIssue(
                    code="low_resolution",
                    message=(
                        f"Image resolution too low ({w}x{h}). "
                        f"Minimum {config.min_width}x{config.min_height} pixels."
                    ),
                    is_critical=True,
                )
            ]
        return []


class BlurCheck(QualityCheck):
    """Measure sharpness via Laplacian variance."""

    def __call__(
            self,
            image: np.ndarray,
            gray: np.ndarray,
            config: PreprocessorConfig,
    ) -> list[QualityIssue]:
        sample = cv2.resize(gray, config.quality_sample_size)
        score = cv2.Laplacian(sample, cv2.CV_64F).var()

        if score < config.blur_threshold:
            return [
                QualityIssue(
                    code="blurry",
                    message=f"Image is blurry (score {score:.1f}, threshold {config.blur_threshold}).",
                    is_critical=(score < config.blur_reject_threshold),
                )
            ]
        return []


class BrightnessCheck(QualityCheck):
    """Check for under- or over-exposure."""

    def __call__(
            self,
            image: np.ndarray,
            gray: np.ndarray,
            config: PreprocessorConfig,
    ) -> list[QualityIssue]:
        mean = float(gray.mean())
        if mean < config.brightness_low:
            return [
                QualityIssue(code="dark_image", message=f"Image too dark (mean {mean:.0f}).")
            ]
        if mean > config.brightness_high:
            return [
                QualityIssue(code="overexposed", message=f"Image too bright (mean {mean:.0f}).")
            ]
        return []


class GlareCheck(QualityCheck):
    """Detect specular glare (high ratio of near-white pixels)."""

    def __call__(
            self,
            image: np.ndarray,
            gray: np.ndarray,
            config: PreprocessorConfig,
    ) -> list[QualityIssue]:
        glare_ratio = float(np.mean(gray > config.glare_pixel_value))
        if glare_ratio > config.glare_max_ratio:
            return [
                QualityIssue(
                    code="glare",
                    message=f"Specular glare detected ({glare_ratio:.1%} bright pixels).",
                )
            ]
        return []


DEFAULT_QUALITY_CHECKS: list[QualityCheck] = [
    ResolutionCheck(),
    BlurCheck(),
    BrightnessCheck(),
    GlareCheck(),
]


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four 2-D points as top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of shape (4, 2).

    Returns:
        Ordered array of shape (4, 2), dtype float32.

    Raises:
        ValueError: If *pts* does not have the expected shape.
    """
    if pts.shape != (4, 2):
        raise ValueError(f"Expected shape (4, 2), got {pts.shape}")

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Return a guaranteed 3-channel BGR image."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """Return a guaranteed single-channel grayscale image."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Perspective-corrected BGR image, or the original on failure."""
    bgr = _ensure_bgr(image)
    gray = _ensure_gray(image)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_corners = None
    for contour in contours[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            document_corners = approx.reshape((4, 2))
            break

    if document_corners is None:
        return bgr

    corners = order_corners(document_corners)

    width = max(
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[3] - corners[2]),
    )
    height = max(
        np.linalg.norm(corners[0] - corners[3]),
        np.linalg.norm(corners[1] - corners[2]),
    )

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    corrected = cv2.warpPerspective(bgr, matrix, (int(width), int(height)))
    return corrected


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle"""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE adaptive histogram equalisation."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def estimate_skew(
        gray: np.ndarray,
        max_skew_degrees: float = 15.0,
) -> float | None:
    """Estimate document skew angle via Hough line detection."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is None:
        return None

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_skew_degrees:
            angles.append(float(angle))

    if not angles:
        logger.debug("No Hough lines within ±%.1f°; skew undetermined.", max_skew_degrees)
        return None

    return float(np.median(angles))


def load_image(source: str | np.ndarray) -> np.ndarray:
    """Load or validate a BGR image from a file path or ndarray. """
    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise ValueError("Received an empty ndarray.")
        return _ensure_bgr(source)

    if not os.path.isfile(source):
        raise FileNotFoundError(f"Image file not found: {source}")

    image = cv2.imread(source)
    if image is None:
        raise ValueError(
            f"Could not decode image (corrupt or unsupported format): {source}"
        )

    if image.size == 0:
        raise ValueError(f"Decoded image has zero dimensions: {source}")

    return image


class DocumentPreprocessor:
    """Orchestrates quality assessment and image correction for KYC documents."""

    def __init__(
            self,
            config: PreprocessorConfig | None = None,
            quality_checks: Sequence[QualityCheck] | None = None,
    ) -> None:
        self.config = config or PreprocessorConfig()
        self.checks = list(quality_checks or DEFAULT_QUALITY_CHECKS)

    def process(self, image_source: str | np.ndarray) -> PreprocessResult:
        """Run the full preprocessing pipeline.

        Steps:
            1. Load / validate image.
            2. Assess quality — reject if critical issues are found.
            3. Perspective correction.
            4. Deskew.
            5. Contrast enhancement (if needed).
            6. Denoising (if needed).
        """
        image = load_image(image_source)
        gray = _ensure_gray(image)

        quality = self.assess_quality(image, gray)

        if quality.action == QualityAction.REJECT:
            return PreprocessResult(image=None, gray=None, quality=quality)

        corrected = self._try_perspective_correction(image)
        corrected_gray = _ensure_gray(corrected)

        if quality.skew_angle is not None and abs(quality.skew_angle) > self.config.deskew_min_angle:
            corrected = deskew(corrected, quality.skew_angle)
            corrected_gray = deskew(corrected_gray, quality.skew_angle)

        brightness = quality.brightness_mean
        if brightness < self.config.brightness_low or brightness > self.config.brightness_high:
            corrected_gray = enhance_contrast(corrected_gray)

        if quality.blur_score < self.config.blur_threshold * 2:
            corrected_gray = cv2.fastNlMeansDenoising(
                corrected_gray, h=self.config.denoise_h
            )

        return PreprocessResult(
            image=corrected,
            gray=corrected_gray,
            quality=quality,
        )

    def assess_quality(
            self,
            image: np.ndarray,
            gray: np.ndarray,
    ) -> QualityReport:
        """Run all registered quality checks and produce a report."""
        all_issues: list[QualityIssue] = []
        for check in self.checks:
            all_issues.extend(check(image, gray, self.config))

        sample = cv2.resize(gray, self.config.quality_sample_size)
        blur_score = float(cv2.Laplacian(sample, cv2.CV_64F).var())
        brightness_mean = float(gray.mean())
        glare_ratio = float(np.mean(gray > self.config.glare_pixel_value))
        h, w = image.shape[:2]
        resolution_ok = h >= self.config.min_height and w >= self.config.min_width
        skew_angle = estimate_skew(gray, self.config.max_skew_degrees)

        if skew_angle is None:
            all_issues.append(
                QualityIssue(
                    code="skew_undetermined",
                    message="Could not estimate skew angle.",
                    is_critical=False,
                )
            )

        has_critical = any(issue.is_critical for issue in all_issues)
        action: QualityAction
        if has_critical:
            action = QualityAction.REJECT
        elif all_issues:
            action = QualityAction.ENHANCED
        else:
            action = QualityAction.PASS

        return QualityReport(
            action=action,
            blur_score=blur_score,
            brightness_mean=brightness_mean,
            glare_ratio=glare_ratio,
            resolution_ok=resolution_ok,
            skew_angle=skew_angle,
            issues=all_issues,
        )

    def _try_perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """Attempt perspective correction; return the original on failure."""
        try:
            return correct_perspective(image)
        except Exception:
            logger.warning("Perspective correction failed.", exc_info=True)
            return image
