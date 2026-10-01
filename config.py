"""Central local prototype configuration."""

import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OCR_ENGINE = os.getenv("OCR_ENGINE", "easyocr").lower()
SUPPORTED_CURRENCIES = ("USD", "KHR")
CONFIDENCE_REVIEW_THRESHOLD = 0.70
