"""Data structures and supported values for receipt extraction."""

from dataclasses import asdict, dataclass
from typing import Any


CATEGORIES = (
    "Ingredients", "Rent", "Utilities", "Salary", "Marketing", "Equipment",
    "Software", "Transportation", "Other",
)


@dataclass
class FieldValue:
    value: Any
    confidence: float
    needs_review: bool = False


@dataclass
class Receipt:
    merchant: FieldValue
    date: FieldValue
    amount: FieldValue
    currency: FieldValue
    category: FieldValue
    description: FieldValue
    items: list[dict[str, Any]]
    ocr_confidence: float = 0.0
    extraction_confidence: float = 0.0
    llm_raw_json: str | None = None
    llm_error: str | None = None
    llm_seconds: float = 0.0
    field_sources: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self),
                "extraction_confidence": self.extraction_confidence,
                "ocr_confidence": self.ocr_confidence}
