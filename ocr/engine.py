"""EasyOCR inference and comparison of preprocessed image variants."""

from dataclasses import dataclass
from functools import lru_cache
import re
from time import perf_counter
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
    candidate_scores: dict[str, float]
    elapsed_seconds: float
    preprocessing_seconds: float
    inference_seconds: float

    @property
    def raw_detections(self) -> list[dict]:
        return [{"text": d.text, "confidence": d.confidence,
                 "bbox": [list(point) for point in d.box]} for d in self.detections]


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
        mapped_points = []
        for point in points:
            x, y = float(point[0]), float(point[1])
            if candidate.inverse_rotation is not None:
                x, y, _ = candidate.inverse_rotation @ np.asarray([x, y, 1.0])
            mapped_points.append((min(source_width, max(0.0, x / candidate.scale_x)),
                                  min(source_height, max(0.0, y / candidate.scale_y))))
        original_points = tuple(mapped_points)
        converted.append(
            TextDetection(
                text=text,
                box=original_points,
                confidence=confidence,
                uncertain=confidence < LOW_CONFIDENCE_LIMIT,
            )
        )
    return converted


def score_ocr_result(detections: list[TextDetection]) -> float:
    """Score OCR quality with receipt fields weighted above generic text volume."""
    if not detections:
        return 0.0
    text = "\n".join(d.text for d in detections)
    mean_confidence = sum(item.confidence for item in detections) / len(detections)
    uncertain_fraction = sum(item.uncertain for item in detections) / len(detections)
    meaningful_lines = sum(bool(re.search(r"[A-Za-z0-9]", line.text)) for line in detections)
    total = bool(re.search(r"\b(grand\s+total|total|amount\s+due|balance\s+due|net\s+total)\b", text, re.I))
    amount = bool(re.search(r"(?:\$\s*)?\d[\d,]*(?:\.\d{1,2})?\s*(?:៛|KHR)?", text, re.I))
    currency = bool(re.search(r"\$|\bUSD\b|\bKHR\b|៛|\bRiel\b", text, re.I))
    date_found = bool(re.search(r"\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b", text))
    garbage_fraction = sum(not re.search(r"[A-Za-z0-9$៛]", d.text) for d in detections) / len(detections)
    important = [d.confidence for d in detections if re.search(
        r"\b(total|amount\s+due|balance\s+due|net\s+total|subtotal)\b|\$|៛|\bUSD\b|\bKHR\b|\d{1,4}[-/]\d{1,2}[-/]\d{2,4}", d.text, re.I)]
    important_confidence = sum(important) / len(important) if important else mean_confidence
    return (0.32 * mean_confidence + 0.18 * important_confidence + 0.08 * min(meaningful_lines / 8, 1)
            + 0.16 * total + 0.08 * amount + 0.08 * currency + 0.07 * date_found
            - 0.10 * uncertain_fraction - 0.05 * garbage_fraction)


def _as_preview(candidate: OCRImage) -> Image.Image:
    """Convert grayscale or color OCR pixels into a displayable RGB image."""
    preview = Image.fromarray(candidate.pixels)
    return preview.convert("RGB")


def extract_document(image: Image.Image, language_codes: Sequence[str] = ("en",)) -> OCRResult:
    """Run OCR on multiple variants and return the strongest coordinate-aware result."""
    if image.width <= 0 or image.height <= 0:
        raise ValueError("The image has no pixels.")

    started = perf_counter()
    reader = _get_reader(tuple(language_codes))
    preprocess_started = perf_counter()
    candidates = make_ocr_images(image)
    preprocessing_seconds = perf_counter() - preprocess_started
    inference_started = perf_counter()
    scored_candidates = []
    candidate_scores = {}
    for candidate in candidates:
        # detail=1 retains the quadrilateral and confidence for every recognized region.
        raw_detections = reader.readtext(candidate.pixels, detail=1, paragraph=False)
        detections = _convert_detections(raw_detections, candidate, image.size)
        score = score_ocr_result(detections)
        candidate_scores[candidate.name] = score
        scored_candidates.append((score, detections, candidate))

    # Stable ordering favors the original image when scores are equal.
    _, detections, selected_image = max(scored_candidates, key=lambda item: item[0])
    inference_seconds = perf_counter() - inference_started
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
        candidate_scores=candidate_scores,
        elapsed_seconds=perf_counter() - started,
        preprocessing_seconds=preprocessing_seconds,
        inference_seconds=inference_seconds,
    )
