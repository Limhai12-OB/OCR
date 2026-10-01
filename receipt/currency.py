"""Deterministic currency detection for supported LedgerVision currencies."""

import re


def detect_currency(text: str) -> tuple[str | None, float, str]:
    has_khr = bool(re.search(r"\bKHR\b|៛|\bCAMBODIAN\s+RIEL\b|\bRIELS?\b", text, re.I))
    has_usd = bool(re.search(r"\bUSD\b|\bUS\s+DOLLARS?\b|\$|\bDOLLARS?\b", text, re.I))
    # Small receipt print can make '$' look like 'S'. Treat it as inferred evidence
    # only when it is immediately followed by a conventional two-decimal price.
    has_s_before_price = bool(re.search(r"(?<![A-Za-z])S\s*\d+[.,]\d{2}(?!\d)", text, re.I))
    if has_khr and (has_usd or has_s_before_price):
        return None, 0.0, "ambiguous"
    if has_khr:
        return "KHR", 0.99, "explicit"
    if re.search(r"\bUSD\b|\bUS\s+DOLLARS?\b", text, re.I):
        return "USD", 0.99, "explicit"
    if re.search(r"\$", text):
        return "USD", 0.90, "explicit"
    if has_s_before_price:
        return "USD", 0.66, "inferred"
    if re.search(r"\bDOLLARS?\b", text, re.I):
        return "USD", 0.80, "explicit"
    return None, 0.0, "missing"
