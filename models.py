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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

