import unittest
from unittest.mock import patch

from llm_service import parse_receipt
from receipt.rules import extract_date, extract_total, parse_rules
from llm.ollama_client import extract_with_ollama


class ReceiptParserTests(unittest.TestCase):
    def test_usd_receipt(self):
        text = "ANGKOR MARKET\n30/09/2026\nCoffee Beans $35\nMilk $12\nSugar $8\nTOTAL $55"
        rules = parse_rules(text)
        self.assertEqual(rules["merchant"], "ANGKOR MARKET")
        self.assertEqual(rules["date"], "2026-09-30")
        self.assertEqual(rules["amount"], 55)
        self.assertEqual(rules["currency"], "USD")

    def test_khr_receipt(self):
        rules = parse_rules("SOKHA MART\n30/09/2026\nRice\nWater\nTOTAL 50,000៛")
        self.assertEqual(rules["amount"], 50000)
        self.assertEqual(rules["currency"], "KHR")

    def test_total_beats_subtotal(self):
        self.assertEqual(extract_total("ABC MART\nSubtotal $40\nTax $4\nTOTAL $44")[0], 44)

    def test_dollar_misread_as_s_is_recovered_for_total(self):
        rules = parse_rules("SHOP\nTOTAL S5.00")
        self.assertEqual(rules["amount"], 5.0)
        self.assertEqual(rules["currency"], "USD")
        self.assertEqual(rules["currency_source"], "inferred")
        self.assertLess(rules["currency_confidence"], 0.70)

    def test_missing_date_is_not_invented_and_llm_failure_falls_back(self):
        with patch("llm_service.extract_with_ollama", side_effect=RuntimeError("offline")):
            receipt = parse_receipt("UNKNOWN SHOP\nTOTAL $25")
        self.assertIsNone(receipt.date.value)
        self.assertEqual(receipt.amount.value, 25)
        self.assertEqual(receipt.currency.value, "USD")
        self.assertIn("offline", receipt.llm_error)

    def test_common_date_formats_and_invalid_dates(self):
        self.assertEqual(extract_date("Date: 2026-09-30")[0], "2026-09-30")
        self.assertEqual(extract_date("Sep 30 2026")[0], "2026-09-30")
        self.assertIsNone(extract_date("31/02/2026")[0])

    def test_common_ocr_confusions_are_recovered_with_lower_confidence(self):
        rules = parse_rules("ANGK0R MARKET\n3O/O9/2O26\nT0TAL $55")
        self.assertEqual(rules["date"], "2026-09-30")
        self.assertEqual(rules["date_source"], "inferred")
        self.assertLess(rules["date_confidence"], 0.70)
        self.assertEqual(rules["amount"], 55)
        self.assertEqual(rules["amount_source"], "inferred")
        self.assertIsNone(rules["category"])

    def test_item_text_maps_to_ingredients(self):
        self.assertEqual(parse_rules("SHOP\nCoffee Beans $35\nMilk $12")["category"], "Ingredients")

    def test_rule_category_overrides_model_other_and_description_uses_grounded_items(self):
        response = ({"category": "Other", "description": "Invented description",
                     "items": [{"name": "Coffee Beans", "amount": 999},
                               {"name": "Milk", "amount": 12}],
                     "extraction_confidence": {}}, "{}", 0.01)
        with patch("llm_service.extract_with_ollama", return_value=response):
            receipt = parse_receipt("SHOP\nCoffee Beans $35\nMilk $12\nTOTAL $47")
        self.assertEqual(receipt.category.value, "Ingredients")
        self.assertEqual(receipt.description.value, "Coffee Beans, Milk")
        self.assertIsNone(receipt.items[0]["amount"])
        self.assertEqual(receipt.items[1]["amount"], 12.0)

    def test_malformed_ollama_json_is_reported_and_falls_back(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return None
            def read(self):
                return b'{"response":"not json"}'

        with patch("llm.ollama_client.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "malformed JSON"):
                extract_with_ollama("SHOP\nTOTAL $4")
        with patch("llm_service.extract_with_ollama", side_effect=ValueError("bad JSON")):
            receipt = parse_receipt("SHOP\nTOTAL $4")
        self.assertEqual(receipt.amount.value, 4)
        self.assertEqual(receipt.currency.value, "USD")


if __name__ == "__main__":
    unittest.main()
