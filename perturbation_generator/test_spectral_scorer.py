"""Regression tests for globally optimal spectral peak matching."""

import unittest

from spectral_scorer import _optimal_peak_match, spectral_similarity


class OptimalPeakMatchTest(unittest.TestCase):
    def test_weak_early_match_cannot_steal_exact_later_match(self):
        # A sorted greedy traversal matches exp=9 to pred=10 first, then loses
        # the exact exp=10 -> pred=10 edge.  The optimum leaves exp=9 unmatched.
        score = _optimal_peak_match([0.0, 10.0], [9.0, 10.0], sigma=1.0)
        self.assertAlmostEqual(score, 0.5, places=12)

    def test_perfect_match_scores_one(self):
        self.assertAlmostEqual(
            _optimal_peak_match([1.0, 2.0], [2.0, 1.0], sigma=0.3),
            1.0,
            places=12,
        )

    def test_excess_prediction_keeps_symmetric_coverage_penalty(self):
        score = _optimal_peak_match([1.0, 2.0, 100.0], [1.0, 2.0], sigma=0.1)
        self.assertAlmostEqual(score, 2.0 / 3.0, places=12)

    def test_invalid_sigma_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sigma must be positive"):
            _optimal_peak_match([1.0], [1.0], sigma=0.0)

    def test_spectral_similarity_still_weights_by_experimental_signal_count(self):
        # Carbon is a perfect 1/1 match; both hydrogen peaks are too distant.
        # The nucleus weighting is therefore 1 C signal / 3 total signals.
        score = spectral_similarity(
            pred_c_peaks=[10.0],
            pred_h_peaks=[0.0, 1.0],
            exp_c_peaks=[{"delta (ppm)": 10.0}],
            exp_h_peaks=[{"delta": 10.0}, {"delta": 11.0}],
            sigma_c=5.0,
            sigma_h=0.3,
        )
        self.assertAlmostEqual(score, 1.0 / 3.0, places=12)


if __name__ == "__main__":
    unittest.main()
