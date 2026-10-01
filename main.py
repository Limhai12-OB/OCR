"""Run receipt OCR, Ollama parsing, validation, and display."""

import argparse
import json
import sys
from time import perf_counter
from PIL import Image

from llm_service import parse_receipt
from ocr_service import extract
from receipt.validator import validate_receipt

EXAMPLES = {
    "usd": """ANGKOR MARKET\n30/09/2026\nCoffee Beans $35\nMilk $12\nSugar $8\nTOTAL $55""",
    "khr": """SOKHA MART\n30/09/2026\nRice\nWater\nTOTAL 50,000៛""",
    "low-confidence": """UNKNOWN SHOP\n$25""",
    "subtotal": """ABC MART\nSubtotal $40\nTax $4\nTOTAL $44""",
    "bad-ocr": """ANGK0R MARKET\n3O/O9/2O26\nT0TAL $55""",
}


def run(text: str, model: str, ocr_result=None, started=None) -> int:
    started = started or perf_counter()
    if ocr_result:
        print("OCR RAW DETECTIONS:")
        print(json.dumps(ocr_result.raw_detections, indent=2, ensure_ascii=False))
    print("OCR RAW TEXT:")
    print(text or "(no text detected)")
    ocr_confidence = ocr_result.average_confidence if ocr_result else 0.0
    print(f"\nOCR average confidence: {ocr_confidence:.2f}")
    receipt = parse_receipt(text, model, ocr_confidence)
    validation = validate_receipt(receipt)
    total_seconds = perf_counter() - started
    timings = {
        "preprocessing_seconds": ocr_result.preprocessing_seconds if ocr_result else 0.0,
        "ocr_seconds": ocr_result.inference_seconds if ocr_result else 0.0,
        "llm_seconds": receipt.llm_seconds,
        "total_seconds": total_seconds,
    }
    print("\nAI RESULT:")
    print(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False))
    if receipt.llm_error:
        print(f"\nLLM FALLBACK: {receipt.llm_error}")
    print("\nVALIDATION:")
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    print("\nFINAL:")
    print(json.dumps({"receipt": receipt.to_dict(), **validation, "timings": timings}, indent=2, ensure_ascii=False))
    return 0 if validation["valid"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype receipt OCR and structured extraction")
    parser.add_argument("image", nargs="?", help="Receipt image path (PNG/JPG)")
    parser.add_argument("--demo", choices=EXAMPLES, help="Run a built-in OCR-text example without image OCR")
    parser.add_argument("--model", default=None, help="Ollama model name (defaults to OLLAMA_MODEL or llama3.2:3b)")
    args = parser.parse_args()
    if bool(args.image) == bool(args.demo):
        parser.error("provide an image path or --demo {usd,khr,low-confidence}")
    started = perf_counter()
    try:
        if args.demo:
            text, result = EXAMPLES[args.demo], None
        else:
            with Image.open(args.image) as image:
                result = extract(image.convert("RGB"))
            text = result.structured_text
    except Exception as error:
        print(f"OCR ERROR: {error}", file=sys.stderr)
        return 2
    return run(text, args.model, result, started)


if __name__ == "__main__":
    raise SystemExit(main())
