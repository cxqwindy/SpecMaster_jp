"""The synthetic fixture is small, local, and shaped like an NMR match case."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_nmr_case.json"


class SyntheticFixtureTests(unittest.TestCase):
    def test_shape_and_size(self) -> None:
        self.assertLess(FIXTURE.stat().st_size, 4096)
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(data["candidate_smiles"], "CCO")
        self.assertGreaterEqual(len(data["c_nmr_peaks"]), 1)
        self.assertGreaterEqual(len(data["h_nmr_peaks"]), 1)
        self.assertIn("delta (ppm)", data["c_nmr_peaks"][0])
        self.assertIn("delta", data["h_nmr_peaks"][0])


if __name__ == "__main__":
    unittest.main()
