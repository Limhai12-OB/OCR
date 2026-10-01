"""Hybrid deterministic receipt rules plus local LLM interpretation."""

from models import CATEGORIES, FieldValue, Receipt
from llm.ollama_client import extract_with_ollama
from receipt.rules import extract_date, extract_merchant, extract_total, parse_rules
import re


def _confidence(data: dict, field: str, fallback: float) -> float:
    raw = data.get("extraction_confidence", {}).get(field, fallback)
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return fallback


def _amount_is_in_ocr(amount: float, text: str) -> bool:
    rendered = f"{amount:g}"
    return bool(re.search(rf"(?<!\d){re.escape(rendered)}(?!\d)", text.replace(",", "")))


def parse_receipt(text: str, model: str | None = None, ocr_confidence: float = 0.0) -> Receipt:
    """Parse with rules first; Ollama failures leave a usable rule-based result."""
    rules = parse_rules(text)
    llm_data, raw_json, llm_error, llm_seconds = {}, None, None, 0.0
    try:
        llm_data, raw_json, llm_seconds = extract_with_ollama(text, model) if model else extract_with_ollama(text)
    except (RuntimeError, ValueError, OSError) as error:
        llm_error = str(error)

    # Explicit OCR rule evidence always wins for financial fields.
    merchant = rules["merchant"]
    merchant_conf = rules["merchant_confidence"]
    if not merchant and isinstance(llm_data.get("merchant"), str):
        merchant = llm_data["merchant"].strip() or None
        merchant_conf = _confidence(llm_data, "merchant", 0.55)
    elif merchant and isinstance(llm_data.get("merchant"), str):
        # Accept only casing/spacing normalization of the OCR-supported name.
        proposed = " ".join(llm_data["merchant"].split())
        if proposed.casefold() == " ".join(merchant.split()).casefold():
            merchant = proposed if proposed != proposed.upper() else proposed.title()
        elif proposed.casefold().replace("0", "o") == " ".join(merchant.split()).casefold().replace("0", "o"):
            merchant = proposed
            merchant_conf = min(_confidence(llm_data, "merchant", 0.68), 0.68)

    transaction_date = rules["date"]
    date_conf = rules["date_confidence"]
    amount = rules["amount"]
    amount_conf = rules["amount_confidence"]
    currency = rules["currency"]
    currency_conf = rules["currency_confidence"]

    # LLM date/amount guesses are accepted only when deterministic text rules found them.
    # This prevents silently fabricated financial values on ambiguous OCR.
    category = llm_data.get("category")
    category = category if category in CATEGORIES else "Other"
    category_conf = _confidence(llm_data, "category", 0.62 if category == "Other" else 0.75)
    if rules["category"] and category == "Other":
        category = rules["category"]
        category_conf = rules["category_confidence"]
    raw_items = llm_data.get("items")
    items = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if (isinstance(item, dict) and isinstance(item.get("name"), str)
                    and item["name"].strip() and item["name"].strip().casefold() in text.casefold()):
                item_amount = item.get("amount")
                if isinstance(item_amount, (int, float)) and item_amount > 0:
                    grounded_amount = float(item_amount) if _amount_is_in_ocr(float(item_amount), text) else None
                    items.append({"name": item["name"].strip(), "amount": grounded_amount})
                elif item_amount is None:
                    items.append({"name": item["name"].strip(), "amount": None})
    description = llm_data.get("description")
    if items:
        description = ", ".join(item["name"] for item in items)
    elif not isinstance(description, str) or not description.strip():
        description = ""
    description = description.strip() or None

    # Keep source and rule findings visible for debugging/validation.
    receipt = Receipt(
        merchant=FieldValue(merchant, merchant_conf),
        date=FieldValue(transaction_date, date_conf),
        amount=FieldValue(amount, amount_conf),
        currency=FieldValue(currency, currency_conf),
        category=FieldValue(category, category_conf),
        description=FieldValue(description, 0.85 if items else _confidence(llm_data, "description", 0.65)),
        items=items,
        ocr_confidence=max(0.0, min(1.0, ocr_confidence)),
        extraction_confidence=sum((merchant_conf, date_conf, amount_conf, currency_conf, category_conf)) / 5,
        llm_raw_json=raw_json,
        llm_error=llm_error,
        llm_seconds=llm_seconds,
        field_sources={
            "merchant": ("model" if merchant and rules["merchant"] and merchant.casefold() != rules["merchant"].casefold()
                         else rules["merchant_source"] if rules["merchant"] else ("model" if merchant else "missing")),
            "date": rules["date_source"], "amount": rules["amount_source"],
            "currency": rules["currency_source"],
            "category": "rules" if rules["category"] else ("model" if llm_data else "fallback"),
            "description": "rules" if items else ("model" if description else "missing"),
        },
    )
    return receipt
