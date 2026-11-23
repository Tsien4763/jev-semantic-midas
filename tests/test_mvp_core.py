"""Core unit tests for Jev-Semantic MIDAS MVP."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from semantic_midas.calibration import calibrate, cp_lower, fit_scale  # noqa: E402


class JevSemanticMIDASTests(unittest.TestCase):
    def test_fit_scale_and_cp_lower(self) -> None:
        b = np.ones((20, 3))
        r = 0.5 * b
        c, U, V = fit_scale(r, b, cap=2.0)
        self.assertAlmostEqual(c, 0.5, places=6)
        self.assertAlmostEqual(U / V, 0.5, places=6)
        self.assertGreater(cp_lower(18, 20), 0.70)
        self.assertEqual(cp_lower(0, 10), 0.0)

    def test_gate_passes_for_informative_signal(self) -> None:
        rng = np.random.default_rng(7)
        s = rng.choice([-1.0, 1.0], size=(50, 1))
        b = np.repeat(s, 3, axis=1)
        r = 0.8 * b + rng.normal(0.0, 0.1, size=(50, 3))
        calib = calibrate(r, b, b, s[:, 0], s[:, 0], r, b, reps=300, seed=7)
        self.assertGreater(calib.gate_scale, 0.0)
        self.assertGreater(calib.gate_gain_lower, 0.0)


if __name__ == "__main__":
    unittest.main()
