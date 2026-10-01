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
    inverse_rotation: np.ndarray | None = None


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


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correct modest receipt rotation from the dominant text-line angle."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=max(40, gray.shape[1] // 5), maxLineGap=20)
    if lines is None:
        return gray, np.eye(3)
    # OpenCV versions/platforms return either (N, 1, 4) or (N, 4).
    segments = np.asarray(lines).reshape(-1, 4)
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1))
              for x1, y1, x2, y2 in segments]
    angles = [angle for angle in angles if abs(angle) <= 12]
    if not angles:
        return gray, np.eye(3)
    angle = float(np.median(angles))
    if abs(angle) < 0.4:
        return gray, np.eye(3)
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    inverse = cv2.invertAffineTransform(matrix)
    inverse_3x3 = np.vstack([inverse, [0.0, 0.0, 1.0]])
    return (cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE), inverse_3x3)


def make_ocr_images(image: Image.Image) -> list[OCRImage]:
    """Create conservative receipt-oriented variants for OCR comparison."""
    rgb, scale_x, scale_y = _resize_for_ocr(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray, inverse_rotation = _deskew(gray)

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
    otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    sharpened = cv2.filter2D(denoised, -1,
                             np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))

    return [
        OCRImage("Original", rgb, scale_x, scale_y),
        OCRImage("Grayscale", gray, scale_x, scale_y, inverse_rotation),
        OCRImage("Contrast enhanced", denoised, scale_x, scale_y, inverse_rotation),
        OCRImage("Thresholded", thresholded, scale_x, scale_y, inverse_rotation),
        OCRImage("Otsu threshold", otsu, scale_x, scale_y, inverse_rotation),
        OCRImage("Sharpened", sharpened, scale_x, scale_y, inverse_rotation),
    ]
