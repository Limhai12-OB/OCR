"""Helpers for loading and validating uploaded image files."""

from io import BytesIO

from PIL import Image, UnidentifiedImageError


def validate_image(image_bytes: bytes) -> Image.Image:
    """Open an uploaded image safely and return a detached RGB image."""
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        # Reopen after verify(), because verify() invalidates the original image object.
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The file is damaged or is not a supported image.") from error
