__all__ = [
    "OCRConfig",
    "OCRWord",
    "OCRLine",
    "OCRResult",
    "DocumentOCR",
]

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Centralised configuration for the OCR engine."""

    lang: str = "vi"
    use_gpu: bool = False
    min_confidence: float = 0.5
    line_tolerance: float = 15.0
    min_image_side: int = 16


@dataclass
class OCRWord:
    """A single word detected by OCR."""

    text: str
    confidence: float
    bbox: list[list[float]]

    @property
    def center_y(self) -> float:
        """Vertical centre of the bounding polygon."""
        return sum(pt[1] for pt in self.bbox) / len(self.bbox)

    @property
    def left_x(self) -> float:
        """Left-most x-coordinate of the bounding polygon."""
        return min(pt[0] for pt in self.bbox)


@dataclass
class OCRLine:
    """A horizontal line of text composed of one or more OCRWords."""

    words: list[OCRWord]
    text: str
    confidence: float


@dataclass
class OCRResult:
    """Aggregated result of OCR on a single image."""
    lines: list[OCRLine] = field(default_factory=list)
    raw_text: str = ""
    mean_confidence: float = 0.0
    word_count: int = 0
    filtered_count: int = 0


class DocumentOCR:

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()

        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(
            use_textline_orientation=True,
            lang=self.config.lang,
            device="gpu" if self.config.use_gpu else "cpu",
        )

    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Run OCR and return structured lines of text."""
        self._validate_image(image)

        raw_results = self._engine.predict(image)

        words, filtered_count = self._parse_results(raw_results)

        if not words:
            if filtered_count > 0:
                logger.info(
                    "OCR detected %d word(s) but all fell below "
                    "min_confidence=%.2f.",
                    filtered_count,
                    self.config.min_confidence,
                )
            return OCRResult(filtered_count=filtered_count)

        lines = self._group_into_lines(words)

        raw_text = "\n".join(line.text for line in lines)
        all_confs = [w.confidence for w in words]
        mean_conf = sum(all_confs) / len(all_confs)

        return OCRResult(
            lines=lines,
            raw_text=raw_text,
            mean_confidence=mean_conf,
            word_count=len(words),
            filtered_count=filtered_count,
        )

    def _validate_image(self, image: np.ndarray) -> None:
        """Raise ValueError if the image is unsuitable for OCR."""
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Expected a numpy ndarray, got %s." % type(image))

        if image.size == 0:
            raise ValueError("Received an empty image (size 0).")

        h, w = image.shape[:2]
        min_side = self.config.min_image_side
        if h < min_side or w < min_side:
            raise ValueError(
                f"Image too small for OCR ({w}x{h}). "
                f"Minimum side length is {min_side}px."
            )

    def _parse_results(
            self,
            raw_results: list,
    ) -> tuple[list[OCRWord], int]:
        """Convert raw PaddleOCR output into OCRWord objects."""
        if not raw_results:
            return [], 0

        result = raw_results[0]

        try:
            texts: list[str] = result["rec_texts"]
            scores: list[float] = result["rec_scores"]
            polys: list[np.ndarray] = result["rec_polys"]
        except (KeyError, TypeError) as exc:
            logger.error(
                "Malformed PaddleOCR result — expected keys "
                "'rec_texts', 'rec_scores', 'rec_polys'. Got: %s",
                exc,
            )
            return [], 0

        words: list[OCRWord] = []
        filtered_count = 0

        for text, conf, bbox in zip(texts, scores, polys):
            stripped = text.strip()
            if not stripped:
                continue
            if conf < self.config.min_confidence:
                filtered_count += 1
                continue

            words.append(
                OCRWord(
                    text=stripped,
                    confidence=float(conf),
                    bbox=bbox.tolist() if isinstance(bbox, np.ndarray) else bbox,
                )
            )

        return words, filtered_count

    def _group_into_lines(
            self,
            words: list[OCRWord],
    ) -> list[OCRLine]:
        """Group words into horizontal lines by vertical proximity."""
        if not words:
            return []
        words_sorted = sorted(words, key=lambda w: w.center_y)

        lines: list[OCRLine] = []
        current_line_words: list[OCRWord] = [words_sorted[0]]
        anchor_y: float = words_sorted[0].center_y

        for word in words_sorted[1:]:
            if abs(word.center_y - anchor_y) <= self.config.line_tolerance:
                current_line_words.append(word)
            else:
                lines.append(self._make_line(current_line_words))
                current_line_words = [word]
                anchor_y = word.center_y

        lines.append(self._make_line(current_line_words))

        return lines

    @staticmethod
    def _make_line(words: list[OCRWord]) -> OCRLine:
        """Build an OCRLine from a group of words on the same line."""
        words_left_to_right = sorted(words, key=lambda w: w.left_x)

        return OCRLine(
            words=words_left_to_right,
            text=" ".join(w.text for w in words_left_to_right),
            confidence=sum(w.confidence for w in words_left_to_right) / len(words),
        )


def main() -> None:
    """Process a sample image through preprocessing → OCR."""
    from src.preprocessing.pipeline import DocumentPreprocessor

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    preprocessor = DocumentPreprocessor()
    ocr = DocumentOCR()

    sample = r"C:\Users\nts\PycharmProjects\kyc-pipeline\data\sample_images\id_card3.jpg"
    result = preprocessor.process(sample)

    print(f"Action : {result.quality.action.value}")
    print(f"Issues : {result.quality.hints}")

    if result.image is not None:
        ocr_result = ocr.extract_text(result.image)

        print(f"\n=== OCR Results ({ocr_result.word_count} words, "
              f"{ocr_result.filtered_count} filtered) ===")
        for i, line in enumerate(ocr_result.lines):
            print(f"  Line {i}: [{line.confidence:.2f}] {line.text}")


if __name__ == "__main__":
    main()
