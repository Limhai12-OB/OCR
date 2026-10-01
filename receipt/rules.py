"""High-precision receipt rules for total, date, and merchant candidates."""

import re
from datetime import date

from receipt.currency import detect_currency

TOTAL_LABEL = re.compile(r"\b(grand\s+total|total|amount\s+due|balance\s+due|net\s+total)\b", re.I)
DATE_PATTERNS = (
    (re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"), "dmy"),
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b"), "mdy_name"),
    (re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b"), "dmy_name"),
)
MONTHS = {name.lower(): number for number, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), 1)}
MONTHS.update({key[:3]: value for key, value in list(MONTHS.items())})


def _repair_date_ocr(value: str) -> str:
    """Repair O/I confusions only inside a date-shaped candidate."""
    if re.fullmatch(r"[0-9OIl]{1,4}[-/][0-9OIl]{1,2}[-/][0-9OIl]{2,4}", value, re.I):
        return value.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    return value


def _keyword_text(text: str) -> str:
    return re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "O", text).upper()


def extract_date(text: str) -> tuple[str | None, float, str]:
    invalid_candidate = False
    for line in text.splitlines():
        repaired = _repair_date_ocr(line.strip())
        recovered = repaired != line.strip()
        for pattern, kind in DATE_PATTERNS:
            for match in pattern.finditer(repaired):
                invalid_candidate = True
                try:
                    values = match.groups()
                    if kind == "dmy":
                        parsed = date(int(values[2]), int(values[1]), int(values[0]))
                    elif kind == "ymd":
                        parsed = date(int(values[0]), int(values[1]), int(values[2]))
                    elif kind == "mdy_name":
                        parsed = date(int(values[2]), MONTHS[values[0].lower()], int(values[1]))
                    else:
                        parsed = date(int(values[2]), MONTHS[values[1].lower()], int(values[0]))
                    return parsed.isoformat(), (0.66 if recovered else 0.96), ("inferred" if recovered else "explicit")
                except (ValueError, KeyError):
                    continue
    if invalid_candidate:
        return None, 0.2, "invalid"
    return None, 0.0, "missing"


def _amount_on_line(line: str) -> float | None:
    without_label = TOTAL_LABEL.sub(" ", line)
    without_label = re.sub(r"(?<![A-Za-z])S(?=\s*\d+[.,]\d{2}(?!\d))", " ", without_label, flags=re.I)
    matches = re.findall(r"(?<![A-Za-z])(?:\$\s*)?([0-9][0-9,]*(?:\.[0-9]{1,2})?)(?:\s*(?:៛|KHR|USD))?", without_label, re.I)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def extract_total(text: str) -> tuple[float | None, float, str]:
    lines = text.splitlines()
    def repaired_label(line: str) -> bool:
        normalized = _keyword_text(line)
        return bool(TOTAL_LABEL.search(normalized))
    payable = [line for line in lines if re.search(r"\b(grand\s+total|amount\s+due|balance\s+due|net\s+total)\b", _keyword_text(line), re.I)]
    totals = [line for line in lines if repaired_label(line)]
    for line in reversed(payable or totals):
        amount = _amount_on_line(line)
        if amount is not None and amount > 0:
            recovered = not bool(TOTAL_LABEL.search(line))
            return amount, (0.72 if recovered else 0.99), ("inferred" if recovered else "explicit")
    # Last labeled total wins; an unlabeled largest number is never assumed to be the total.
    return None, 0.0, "missing"


def extract_merchant(text: str) -> tuple[str | None, float, str]:
    excluded = re.compile(r"\b(receipt|invoice|tax\s*id|vat|phone|tel|address|date|cashier)\b|\d{7,}", re.I)
    for line in text.splitlines()[:8]:
        candidate = re.sub(r"\s+", " ", line).strip(" -_:")
        if (len(candidate) < 3 or not re.search(r"[A-Za-z]", candidate)
                or excluded.search(candidate) or extract_date(candidate)[2] != "missing"):
            continue
        return candidate, 0.78, "explicit"
    return None, 0.0, "missing"


def extract_category(text: str) -> tuple[str | None, float]:
    normalized = _keyword_text(text)
    rules = (
        ("Rent", r"\b(rent|lease)\b"),
        ("Utilities", r"\b(utility|electricity|internet bill|water bill)\b"),
        ("Salary", r"\b(salary|wage|payroll)\b"),
        ("Marketing", r"\b(marketing|advertising|promotion|ads)\b"),
        ("Equipment", r"\b(equipment|computer|printer|furniture|machine)\b"),
        ("Software", r"\b(software|subscription|license)\b"),
        ("Transportation", r"\b(transport|taxi|fuel|gasoline|bus fare|parking)\b"),
        ("Ingredients", r"\b(coffee|milk|sugar|rice|water|beans|food|grocery|flour|meat|vegetable)\b"),
    )
    for category, pattern in rules:
        if re.search(pattern, normalized, re.I):
            return category, 0.86
    return None, 0.0


def parse_rules(text: str) -> dict:
    merchant, merchant_conf, merchant_source = extract_merchant(text)
    transaction_date, date_conf, date_source = extract_date(text)
    amount, amount_conf, amount_source = extract_total(text)
    currency, currency_conf, currency_source = detect_currency(text)
    category, category_conf = extract_category(text)
    return {
        "merchant": merchant, "merchant_confidence": merchant_conf, "merchant_source": merchant_source,
        "date": transaction_date, "date_confidence": date_conf, "date_source": date_source,
        "amount": amount, "amount_confidence": amount_conf, "amount_source": amount_source,
        "currency": currency, "currency_confidence": currency_conf, "currency_source": currency_source,
        "category": category, "category_confidence": category_conf,
    }
