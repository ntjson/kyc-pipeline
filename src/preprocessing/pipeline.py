import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class QualityAction(Enum):
    PASS = "pass"
    ENHANCED = "enhanced"
    REJECT = "reject"


@dataclass
class QualityReport:
    action: QualityAction
    blur_score: float
    brightness_mean: float
    glare_ratio: float
    resolution_ok: bool
    skew_angle: float | None
    issues: list = field(default_factory=list)
    hints: list = field(default_factory=list)


@dataclass
class PreprocessResult:
    image: np.ndarray | None
    gray: np.ndarray | None
    quality: QualityReport


def order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def correct_perspective(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_corners = None
    for contour in contours[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            document_corners = approx.reshape((4, 2))
            break

    if document_corners is None:
        return image

    corners = order_corners(document_corners)

    width = max(
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[3] - corners[2]),
    )

    height = max(
        np.linalg.norm(corners[0] - corners[3]),
        np.linalg.norm(corners[1] - corners[2]),
    )

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],

    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    corrected = cv2.warpPerspective(image, matrix, (int(width), int(height)))

    return corrected


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def try_perspective_correction(image: np.ndarray) -> np.ndarray:
    try:
        return correct_perspective(image)
    except Exception as e:
        return image


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


class DocumentPreprocessor:
    BLUR_THRESHOLD = 100.0
    MIN_WIDTH = 640
    MIN_HEIGHT = 480
    GLARE_MAX_RATIO = 0.15
    MAX_SKEW_DEGREES = 15.0

    def process(self, image_path: str) -> "PreprocessResult":
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        quality = self.assess_quality(image, gray)

        if quality.action == QualityAction.REJECT:
            return PreprocessResult(
                image=None,
                gray=None,
                quality=quality,
            )

        corrected = image.copy()
        corrected_gray = gray.copy()

        corrected = try_perspective_correction(corrected)
        corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

        if quality.skew_angle and abs(quality.skew_angle) > 1.0:
            corrected_gray = deskew(corrected_gray, quality.skew_angle)

        if quality.brightness_mean < 80 or quality.brightness_mean > 200:
            corrected_gray = enhance_contrast(corrected_gray)

        corrected_gray = cv2.fastNlMeansDenoising(corrected_gray, h=10)

        return PreprocessResult(
            image=corrected,
            gray=corrected_gray,
            quality=quality,
        )

    def assess_quality(self, image: np.ndarray, gray: np.ndarray):
        issues = []
        hints = []

        # check resolution
        h, w = image.shape[:2]
        resolution_ok = h > self.MIN_HEIGHT and w > self.MIN_WIDTH

        if not resolution_ok:
            issues.append("low resolution")
            hints.append(
                f"Image resolution too low ({w}x{h})"
                f"Minimum {self.MIN_WIDTH} x {self.MIN_HEIGHT} pixels."
            )

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.BLUR_THRESHOLD:
            issues.append("blurry")
        hints.append(
            f"blurry image ({blur_score})"
        )

        brightness_mean = float(gray.mean())
        if brightness_mean < 40:
            issues.append("dark image")
            hints.append("too dark")
        elif brightness_mean > 200:
            issues.append("overexposed image")
            hints.append("too bright")

        glare_ratio = float(np.mean(gray > 240))
        if glare_ratio < self.GLARE_MAX_RATIO:
            issues.append("glare")
            hints.append("glare")

        skew_angle = self.estimate_skew(gray)

        critical_issues = {"low resolution", "blurry"} & set(issues)
        if critical_issues and blur_score < 50:
            action = QualityAction.REJECT
        elif issues:
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
            issues=issues,
            hints=hints
        )

    def estimate_skew(self, gray: np.ndarray) -> float | None:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return None

        angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < self.MAX_SKEW_DEGREES:
                angles.append(angle)

        return float(np.median(angles)) if angles else None
