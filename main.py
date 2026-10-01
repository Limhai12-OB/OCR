"""Run receipt OCR, Ollama parsing, validation, and display."""

import argparse
import json
import sys

from llm_service import parse_receipt
from ocr_service import extract_text
from validator import validate_receipt

EXAMPLES = {
    "usd": """ANGKOR MARKET\n30/09/2026\nCoffee Beans $35\nMilk $12\nSugar $8\nTOTAL $55""",
    "khr": """SOKHA MART\n30/09/2026\nRice\nWater\nTOTAL 50,000៛""",
    "low-confidence": """UNKNOWN SHOP\n$25""",
}


def run(text: str, model: str) -> int:
    print("OCR RAW TEXT:")
    print(text or "(no text detected)")
    try:
        receipt = parse_receipt(text, model)
    except (RuntimeError, ValueError) as error:
        print(f"\nAI ERROR: {error}", file=sys.stderr)
        return 2
    validation = validate_receipt(receipt)
    print("\nAI RESULT:")
    print(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False))
    print("\nVALIDATION:")
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    print("\nFINAL:")
    print(json.dumps({**receipt.to_dict(), **validation}, indent=2, ensure_ascii=False))
    return 0 if validation["valid"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype receipt OCR and structured extraction")
    parser.add_argument("image", nargs="?", help="Receipt image path (PNG/JPG)")
    parser.add_argument("--demo", choices=EXAMPLES, help="Run a built-in OCR-text example without image OCR")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    args = parser.parse_args()
    if bool(args.image) == bool(args.demo):
        parser.error("provide an image path or --demo {usd,khr,low-confidence}")
    try:
        text = EXAMPLES[args.demo] if args.demo else extract_text(args.image)
    except Exception as error:
        print(f"OCR ERROR: {error}", file=sys.stderr)
        return 2
    return run(text, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
