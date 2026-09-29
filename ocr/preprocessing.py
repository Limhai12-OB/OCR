"""Prepare a few conservative image variants for OCR comparison."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class OCRImage:
    """An OCR-ready image and the scale needed to map boxes back to the source."""

    name: str
    pixels: np.ndarray
    scale_x: float
    scale_y: float


def _resize_for_ocr(image: Image.Image, max_side: int = 2400, min_side: int = 1200) -> tuple[np.ndarray, float, float]:
    """Resize very large images down and small images up while retaining aspect ratio."""
    width, height = image.size
    longest_side = max(width, height)
    if longest_side > max_side:
        scale = max_side / longest_side
    elif longest_side < min_side:
        scale = min_side / longest_side
    else:
        scale = 1.0

    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.convert("RGB").resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    return np.asarray(resized), resized_width / width, resized_height / height


def make_ocr_images(image: Image.Image) -> list[OCRImage]:
    """Create original, enhanced grayscale, and thresholded candidates."""
    rgb, scale_x, scale_y = _resize_for_ocr(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE evens out lighting; denoising is deliberately mild to retain thin strokes.
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrast, None, h=7, templateWindowSize=7, searchWindowSize=21)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return [
        OCRImage("Original", rgb, scale_x, scale_y),
        OCRImage("Enhanced grayscale", denoised, scale_x, scale_y),
        OCRImage("Thresholded", thresholded, scale_x, scale_y),
    ]
