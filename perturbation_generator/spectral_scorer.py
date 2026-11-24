"""Spectral similarity scoring: predicted NMR peaks vs experimental peaks.

Globally optimal one-to-one matching with Gaussian decay computes how similar
predicted and experimental spectra are while allowing peaks to remain unmatched.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from scipy.optimize import linear_sum_assignment


_MIN_MATCH_SCORE = 0.01


def _optimal_peak_match(
    pred_peaks: List[float],
    exp_peaks: List[float],
    sigma: float,
) -> float:
    """Return the maximum-weight one-to-one peak matching similarity.

    Real edges have Gaussian weight ``exp(-Δ²/(2σ²))`` when that weight is
    greater than the historical 0.01 acceptance threshold.  One zero-weight
    dummy column per experimental peak lets the assignment leave any
    experimental peak unmatched instead of consuming a prediction in a weak
    match.  Predictions may likewise remain unused because the assignment
    matrix has more columns than rows.

    The previous ``average_match * coverage`` score simplifies algebraically to
    ``sum(matched_weights) / max(n_pred, n_exp)``.  Maximising the weight sum
    therefore fixes greedy peak stealing without changing the threshold,
    symmetric coverage penalty, or output scale.
    """
    if not pred_peaks or not exp_peaks:
        return 0.0
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    pred = [float(p) for p in pred_peaks]
    exp = [float(e) for e in exp_peaks]

    # Add one dummy column per experimental peak.  Negating weights lets us use
    # scipy's minimum-cost assignment API while remaining compatible with scipy
    # versions that predate the ``maximize`` argument.
    weights = []
    for e in exp:
        real_weights = []
        for p in pred:
            score = math.exp(-((p - e) ** 2) / (2 * sigma ** 2))
            real_weights.append(score if score > _MIN_MATCH_SCORE else 0.0)
        weights.append(real_weights + [0.0] * len(exp))

    row_indices, col_indices = linear_sum_assignment(
        [[-weight for weight in row] for row in weights]
    )
    matched_weight = sum(
        weights[row][col]
        for row, col in zip(row_indices, col_indices)
        if col < len(pred)
    )
    return matched_weight / max(len(pred), len(exp))


def _greedy_peak_match(
    pred_peaks: List[float],
    exp_peaks: List[float],
    sigma: float,
) -> float:
    """Backward-compatible private alias for the now-global matcher."""
    return _optimal_peak_match(pred_peaks, exp_peaks, sigma)


def spectral_similarity(
    pred_c_peaks: List[float],
    pred_h_peaks: List[float],
    exp_c_peaks: Optional[List[Dict[str, Any]]],
    exp_h_peaks: Optional[List[Dict[str, Any]]],
    *,
    w_c: float = 0.5,
    w_h: float = 0.5,
    sigma_c: float = 5.0,
    sigma_h: float = 0.3,
) -> float:
    """Compute spectral similarity between predicted and experimental peaks.

    Parameters
    ----------
    pred_c_peaks : list of float
        Predicted ¹³C chemical shifts (ppm).
    pred_h_peaks : list of float
        Predicted ¹H chemical shifts (ppm).
    exp_c_peaks : list of dict
        Experimental ¹³C peaks, each with key "delta (ppm)".
    exp_h_peaks : list of dict
        Experimental ¹H peaks, each with key "centroid" or "delta".
    w_c, w_h : float
        Weights for ¹³C and ¹H similarity.
    sigma_c, sigma_h : float
        Gaussian width for peak matching (ppm).

    Returns
    -------
    float
        Similarity score in [0, 1]. Higher = more similar.
    """
    # Extract experimental peak values
    exp_c_vals = []
    if exp_c_peaks:
        for p in exp_c_peaks:
            v = p.get("delta (ppm)")
            if v is not None:
                exp_c_vals.append(float(v))

    exp_h_vals = []
    if exp_h_peaks:
        for p in exp_h_peaks:
            v = p.get("centroid") or p.get("delta")
            if v is not None:
                exp_h_vals.append(float(v))

    # Compute per-nucleus similarity
    have_c = bool(pred_c_peaks) and bool(exp_c_vals)
    have_h = bool(pred_h_peaks) and bool(exp_h_vals)

    if not have_c and not have_h:
        return 0.0

    c_sim = _optimal_peak_match(pred_c_peaks, exp_c_vals, sigma_c) if have_c else 0.0
    h_sim = _optimal_peak_match(pred_h_peaks, exp_h_vals, sigma_h) if have_h else 0.0

    # Weight by number of experimental signals so the more-populated nucleus
    # dominates.  Fixed 50:50 was wrong: adding a low-quality 1H prediction
    # could drag a good 13C score from 0.80 down to 0.55.
    if have_c and have_h:
        n_c = len(exp_c_vals)
        n_h = len(exp_h_vals)
        total = n_c + n_h
        return (n_c / total) * c_sim + (n_h / total) * h_sim
    elif have_c:
        return c_sim
    else:
        return h_sim
