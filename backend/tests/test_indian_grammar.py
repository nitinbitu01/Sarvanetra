import unittest
from backend.scripts.indian_plate_grammar import decode_plate, fix_plate


class TestIndianPlateGrammar(unittest.TestCase):
    def test_bh_series_protection(self):
        # BH format: DD BH DDDD LL -> 22BH1234AB
        # Crucial: digits at 0-1 must NOT be mutated to ZZ
        raw = "22BH1234AB"
        res = decode_plate(raw)
        self.assertEqual(res["format"], "bharat")
        self.assertEqual(res["plate"], "22BH1234AB")
        self.assertGreaterEqual(res["score"], 0.70)

    def test_standard_gujarat_plate(self):
        raw = "GJ01AB1234"
        res = decode_plate(raw, local_state="GJ")
        self.assertEqual(res["format"], "standard")
        self.assertEqual(res["plate"], "GJ01AB1234")
        self.assertEqual(res["state"], "GJ")
        self.assertGreaterEqual(res["score"], 0.75)

    def test_positional_glyph_disambiguation(self):
        # OCR confuses 'O' with '0' in digit position
        raw_confused = "GJO1AB1234"  # 'O' at pos 2 should become '0'
        res = decode_plate(raw_confused, local_state="GJ")
        self.assertEqual(res["plate"], "GJ01AB1234")

        # OCR confuses '0' with 'O' in letter position
        raw_confused2 = "0J01AB1234"  # '0' at pos 0 should become 'O' or 'GJ' via state match
        fixed = fix_plate(raw_confused2, local_state="GJ")
        self.assertTrue(fixed.startswith("GJ"))

    def test_out_of_state_plate_not_dropped(self):
        # Maharashtra plate should decode cleanly, not be dropped
        raw = "MH12DE5678"
        res = decode_plate(raw, local_state="GJ")
        self.assertEqual(res["plate"], "MH12DE5678")
        self.assertEqual(res["state"], "MH")
        self.assertGreaterEqual(res["score"], 0.60)

    def test_old_eight_char_format(self):
        raw = "GJ011234"
        res = decode_plate(raw, local_state="GJ")
        self.assertEqual(res["format"], "standard")
        self.assertEqual(res["plate"], "GJ011234")


if __name__ == "__main__":
    unittest.main()
