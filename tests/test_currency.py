import unittest

from receipt.currency import detect_currency


class CurrencyTests(unittest.TestCase):
    def test_supported_markers(self):
        self.assertEqual(detect_currency("TOTAL $12.50")[0], "USD")
        self.assertEqual(detect_currency("TOTAL USD 12.50")[0], "USD")
        self.assertEqual(detect_currency("50,000៛")[0], "KHR")
        self.assertEqual(detect_currency("KHR 50000")[0], "KHR")

    def test_unsupported_or_missing_currency_is_not_guessed(self):
        self.assertIsNone(detect_currency("Total 25 EUR")[0])
        self.assertIsNone(detect_currency("Total 25")[0])

    def test_conflicting_supported_markers_are_ambiguous(self):
        self.assertIsNone(detect_currency("$10 and 5,000៛")[0])

    def test_s_before_decimal_price_is_only_low_confidence_usd_evidence(self):
        currency, confidence, source = detect_currency("TOTAL S5.00")
        self.assertEqual(currency, "USD")
        self.assertLess(confidence, 0.70)
        self.assertEqual(source, "inferred")


if __name__ == "__main__":
    unittest.main()
