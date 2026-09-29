"""Conservative OCR text cleanup that leaves recognized values intact."""

import re


def clean_recognized_text(text: str) -> str:
    """Normalize whitespace only; never rewrite digits, punctuation, or currency."""
    cleaned_lines = []
    for line in text.splitlines():
        # OCR can insert tabs and repeated spaces inside a text box. Keep a single
        # separator within each recognized box; layout spacing is added separately.
        cleaned_lines.append(re.sub(r"[\t\u00a0]+", " ", line).rstrip())
    return "\n".join(cleaned_lines).strip()
