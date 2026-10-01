"""Validation for normalized receipt values."""

from datetime import date

from config import CONFIDENCE_REVIEW_THRESHOLD
from models import CATEGORIES, Receipt


def validate_receipt(receipt: Receipt) -> dict:
    errors = []
    if not receipt.merchant.value:
        errors.append("Merchant is missing")
    if not receipt.date.value:
        if receipt.field_sources and receipt.field_sources.get("date") == "invalid":
            errors.append("Transaction date is invalid")
        else:
            errors.append("Transaction date is missing")
    else:
        try:
            date.fromisoformat(receipt.date.value)
        except (TypeError, ValueError):
            errors.append("Transaction date is invalid")
    if receipt.amount.value is None:
        errors.append("Total amount is missing")
    elif not isinstance(receipt.amount.value, (int, float)) or receipt.amount.value <= 0:
        errors.append("Total amount must be numeric and greater than zero")
    if receipt.currency.value not in ("USD", "KHR"):
        errors.append("Currency could not be detected as USD or KHR")
    if receipt.category.value not in CATEGORIES:
        errors.append("Category is missing or unsupported")

    fields = (receipt.merchant, receipt.date, receipt.amount,
              receipt.currency, receipt.category, receipt.description)
    for field in fields:
        field.needs_review = field.confidence < CONFIDENCE_REVIEW_THRESHOLD
    needs_review = bool(errors) or any(field.needs_review for field in fields)
    return {"valid": not errors, "needs_review": needs_review, "errors": errors}
