"""Small adapter around the project's existing EasyOCR implementation."""

from pathlib import Path

from PIL import Image

from ocr.engine import extract_document


def extract_text(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Receipt image not found: {path}")
    with Image.open(path) as image:
        result = extract_document(image.convert("RGB"), ("en",))
    return result.plain_text

