"""EasyOCR inference and comparison of preprocessed image variants."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import easyocr
import numpy as np
from PIL import Image

from ocr.layout import TextDetection, reconstruct_text
from ocr.preprocessing import OCRImage, make_ocr_images

# Add languages here as needed. EasyOCR language codes are listed in its documentation.
SUPPORTED_LANGUAGES: dict[str, list[str]] = {"English": ["en"]}
LOW_CONFIDENCE_LIMIT = 0.50


@dataclass
class OCRResult:
    """OCR output and useful metadata for the Streamlit interface."""

    detections: list[TextDetection]
    plain_text: str
    structured_text: str
    average_confidence: float
    uncertain_count: int
    selected_preprocessing: str
    processed_preview: Image.Image


@lru_cache(maxsize=8)
def _get_reader(language_codes: tuple[str, ...]) -> easyocr.Reader:
    """Create a reader once per language selection; use CPU for local portability."""
    return easyocr.Reader(list(language_codes), gpu=False)


def _convert_detections(
    raw_detections: list[tuple], candidate: OCRImage, source_size: tuple[int, int]
) -> list[TextDetection]:
    """Keep each detected phrase, map its box back to source-image coordinates."""
    source_width, source_height = source_size
    converted = []
    for polygon, raw_text, raw_confidence in raw_detections:
        text = str(raw_text).strip()
        if not text:
            continue
        confidence = float(raw_confidence)
        if not np.isfinite(confidence):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        points = np.asarray(polygon, dtype=float)
        original_points = tuple(
            (
                min(source_width, max(0.0, float(point[0]) / candidate.scale_x)),
                min(source_height, max(0.0, float(point[1]) / candidate.scale_y)),
            )
            for point in points
        )
        converted.append(
            TextDetection(
                text=text,
                box=original_points,
                confidence=confidence,
                uncertain=confidence < LOW_CONFIDENCE_LIMIT,
            )
        )
    return converted


def _candidate_quality(detections: list[TextDetection]) -> float:
    """Compare passes using confidence, amount of recovered text, and uncertainty."""
    if not detections:
        return 0.0
    mean_confidence = sum(item.confidence for item in detections) / len(detections)
    uncertain_fraction = sum(item.uncertain for item in detections) / len(detections)
    text_coverage = min(sum(len(item.text) for item in detections) / 160, 1.0)
    return 0.72 * mean_confidence + 0.28 * text_coverage - 0.12 * uncertain_fraction


def _as_preview(candidate: OCRImage) -> Image.Image:
    """Convert grayscale or color OCR pixels into a displayable RGB image."""
    preview = Image.fromarray(candidate.pixels)
    return preview.convert("RGB")


def extract_document(image: Image.Image, language_codes: Sequence[str] = ("en",)) -> OCRResult:
    """Run OCR on multiple variants and return the strongest coordinate-aware result."""
    if image.width <= 0 or image.height <= 0:
        raise ValueError("The image has no pixels.")

    reader = _get_reader(tuple(language_codes))
    candidates = make_ocr_images(image)
    scored_candidates = []
    for candidate in candidates:
        # detail=1 retains the quadrilateral and confidence for every recognized region.
        raw_detections = reader.readtext(candidate.pixels, detail=1, paragraph=False)
        detections = _convert_detections(raw_detections, candidate, image.size)
        scored_candidates.append((_candidate_quality(detections), detections, candidate))

    # Stable ordering favors the original image when scores are equal.
    _, detections, selected_image = max(scored_candidates, key=lambda item: item[0])
    confidence = (
        sum(item.confidence for item in detections) / len(detections) if detections else 0.0
    )
    return OCRResult(
        detections=detections,
        plain_text=reconstruct_text(detections, structured=False),
        structured_text=reconstruct_text(detections, structured=True),
        average_confidence=confidence,
        uncertain_count=sum(item.uncertain for item in detections),
        selected_preprocessing=selected_image.name,
        processed_preview=_as_preview(selected_image),
    )
