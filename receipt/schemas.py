"""Structured receipt schema helpers."""

from models import CATEGORIES

RECEIPT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "merchant": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
        "amount": {"type": ["number", "null"]},
        "currency": {"enum": ["USD", "KHR", None]},
        "category": {"enum": list(CATEGORIES)},
        "description": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "amount": {"type": ["number", "null"]}},
            "required": ["name", "amount"]}},
        "extraction_confidence": {"type": "object"},
    },
    "required": ["merchant", "date", "amount", "currency", "category", "description", "items", "extraction_confidence"],
}
