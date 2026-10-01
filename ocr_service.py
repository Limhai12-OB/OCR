"""Small adapter around the project's existing EasyOCR implementation."""

from pathlib import Path
from config import OCR_ENGINE

from PIL import Image

from ocr.engine import OCRResult, extract_document


def extract(image: Image.Image) -> OCRResult:
    """OCR interface kept independent from receipt parsing and model choice."""
    if OCR_ENGINE != "easyocr":
        raise ValueError(f"Unsupported OCR_ENGINE={OCR_ENGINE!r}; only easyocr is installed")
    return extract_document(image.convert("RGB"), ("en",))


def extract_text(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Receipt image not found: {path}")
    with Image.open(path) as image:
        result = extract(image)
    return result.plain_text
