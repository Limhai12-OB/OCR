"""Run the OCR and deterministic receipt parser across an image directory."""

import argparse
from pathlib import Path

from PIL import Image

from ocr_service import extract
from receipt.rules import parse_rules


def benchmark(directory: Path) -> int:
    images = sorted(path for path in directory.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not images:
        print(f"No receipt images found in {directory}")
        return 1
    for path in images:
        with Image.open(path) as image:
            result = extract(image.convert("RGB"))
        fields = parse_rules(result.structured_text)
        errors = []
        if not fields["merchant"]:
            errors.append("Merchant is missing")
        if not fields["date"]:
            errors.append("Transaction date is missing" if fields["date_source"] == "missing" else "Transaction date is invalid")
        if not fields["amount"]:
            errors.append("Total amount is missing")
        if fields["currency"] not in {"USD", "KHR"}:
            errors.append("Currency could not be detected")
        print(f"\n=== {path.name} ===")
        print(f"OCR engine/pass: EasyOCR / {result.selected_preprocessing}")
        print(f"OCR text:\n{result.structured_text}")
        print(f"OCR average confidence: {result.average_confidence:.3f}")
        print(f"Merchant: {fields['merchant']}")
        print(f"Date: {fields['date']}")
        print(f"Amount: {fields['amount']}")
        print(f"Currency: {fields['currency']}")
        print(f"Validation: {'PASS' if not errors else 'FAIL'} {errors}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing receipt images")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")
    return benchmark(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
