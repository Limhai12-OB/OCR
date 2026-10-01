"""Receipt validation and low-confidence review marking."""

from datetime import date

from models import Receipt


def validate_receipt(receipt: Receipt) -> dict:
    errors = []
    if not receipt.merchant.value:
        errors.append("Merchant is missing")
    if not receipt.date.value:
        errors.append("Date is missing")
    else:
        try:
            date.fromisoformat(receipt.date.value)
        except (TypeError, ValueError):
            errors.append("Date is invalid")
    if receipt.amount.value is None:
        errors.append("Amount is missing")
    elif receipt.amount.value <= 0:
        errors.append("Amount must be greater than zero")
    if receipt.currency.value not in ("USD", "KHR"):
        errors.append("Currency could not be detected")
    if not receipt.category.value:
        errors.append("Category is missing")
    for field in (receipt.merchant, receipt.date, receipt.amount,
                  receipt.currency, receipt.category, receipt.description):
        field.needs_review = field.confidence < 0.70
    return {"valid": not errors, "errors": errors}

