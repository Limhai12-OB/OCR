import unittest
from unittest.mock import patch

from llm_service import parse_receipt
from receipt.validator import validate_receipt


class ReceiptValidatorTests(unittest.TestCase):
    def test_missing_date_is_invalid_and_needs_review(self):
        with patch("llm_service.extract_with_ollama", side_effect=RuntimeError("offline")):
            receipt = parse_receipt("UNKNOWN SHOP\nTOTAL $25")
        result = validate_receipt(receipt)
        self.assertFalse(result["valid"])
        self.assertTrue(result["needs_review"])
        self.assertIn("Transaction date is missing", result["errors"])

    def test_nonpositive_total_is_invalid(self):
        with patch("llm_service.extract_with_ollama", return_value=({"category": "Other"}, "{}", 0.01)):
            receipt = parse_receipt("SHOP\n30/09/2026\nTOTAL $0")
        self.assertIsNone(receipt.amount.value)
        self.assertIn("Total amount is missing", validate_receipt(receipt)["errors"])

    def test_invalid_date_is_reported_as_invalid(self):
        with patch("llm_service.extract_with_ollama", side_effect=RuntimeError("offline")):
            receipt = parse_receipt("SHOP\n31/02/2026\nTOTAL $10")
        self.assertIn("Transaction date is invalid", validate_receipt(receipt)["errors"])


if __name__ == "__main__":
    unittest.main()
