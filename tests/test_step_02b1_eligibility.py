"""Synthetic eligibility checks. They do not read LMDB or choose a dev split."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

from step_02b1_eligibility import (  # noqa: E402
    check_cross_split,
    exposure_reasons,
    exposure_remainder,
    quality_reasons,
    reason_family_counts,
    shift_only_diagnostic,
    spectrum_difference_labels,
)


def test_multiple_exposure_reasons_remove_a_row_once():
    confirmed = {"structure": {"canonical:A"}, "spectrum": set(), "pair": set()}
    inferred = {"structure": set(), "spectrum": set(), "pair": set()}
    evaluation = {"structure": set()}
    row = {
        "candidate_token": "canonical:A",
        "gt_token": "canonical:A",
        "spectrum_norm_hash": "spec",
        "pair_token": "pair",
        "shift_sequence_hash": "shift",
    }
    reasons = exposure_reasons(
        row,
        confirmed=confirmed,
        inferred=inferred,
        evaluation=evaluation,
        test_gt=set(),
        test_candidate={"canonical:A"},
        unknown=set(),
    )
    assert reasons == ["confirmed_structure", "test_candidate_structure"]
    rows = [
        {"exposure_reasons": reasons},
        {"exposure_reasons": []},
        {"exposure_reasons": ["unknown_possible_structure"], "quality_reasons": ["unparsed_structure"]},
    ]
    assert exposure_remainder(rows) == 1
    assert len(rows) == 3


def test_negative_candidate_cannot_cross_splits_and_errors_stay_counted():
    result = check_cross_split([
        {"record_id": "t", "source_gt": "G-test", "candidate": "G-test", "split": "test", "kind": "positive"},
        {"record_id": "neg", "source_gt": "G-other", "candidate": "G-test", "split": "search-dev", "kind": "negative"},
        {"record_id": "a", "source_gt": "G-shared", "candidate": "C1", "split": "search-dev", "kind": "positive"},
        {"record_id": "b", "source_gt": "G-shared", "candidate": "C2", "split": "gate-dev", "kind": "negative"},
    ], frozen_groups={"G-frozen": "future-train"})
    assert result["leaked_structures"] == []
    reasons = {item["record_id"]: item["reason"] for item in result["rejected"]}
    assert reasons["neg"] == "cross_split_structure"
    assert reasons["a"] == "source_gt_derivatives_conflict"
    assert reasons["b"] == "source_gt_derivatives_conflict"
    frozen = check_cross_split([
        {"record_id": "keep", "source_gt": "G-frozen", "candidate": "C3", "split": "future-train", "kind": "positive"},
        {"record_id": "move", "source_gt": "G-frozen", "candidate": "C4", "split": "test", "kind": "negative"},
    ], frozen_groups={"G-frozen": "future-train"})
    assert [item["record_id"] for item in frozen["accepted"]] == ["keep"]
    assert frozen["rejected"][0]["reason"] == "frozen_source_group"
    flags = quality_reasons({
        "unparsed_structure": True,
        "missing_nH": True,
        "single_nucleus_missing": True,
        "frequency_unknown": True,
    })
    assert flags == ["unparsed_structure"]


def test_shift_sequence_is_not_a_full_spectrum_and_labels_can_coexist():
    assert shift_only_diagnostic(
        {"shift_sequence_hash": "shift", "spectrum_norm_hash": "other"},
        {"shift"},
        {"exact"},
    )
    assert not shift_only_diagnostic(
        {"shift_sequence_hash": "shift", "spectrum_norm_hash": "exact"},
        {"shift"},
        {"exact"},
    )
    labels = spectrum_difference_labels(
        {
            "c_nmr_peaks": [],
            "h_nmr_peaks": [{"delta": 7.455, "centroid": 7.455, "nH": 2}],
        },
        {
            "c_nmr_peaks": None,
            "h_nmr_peaks": [{
                "category": "m",
                "centroid": 7.455,
                "delta": 7.455,
                "j_values": "7.3_2.0_",
                "nH": 2,
                "rangeMax": 7.51,
                "rangeMin": 7.4,
            }],
        },
    )
    assert "c_empty_list_vs_null" in labels
    assert "h_category" in labels
    assert "h_j" in labels
    assert "h_range" in labels


def test_overlapping_reasons_are_not_added_twice():
    rows = [
        {"quality_reasons": ["gold_unresolved"], "structure_reasons": ["test_gt_structure"], "spectrum_reasons": []},
        {"quality_reasons": [], "structure_reasons": [], "spectrum_reasons": ["confirmed_exact_spectrum"]},
        {"quality_reasons": [], "structure_reasons": [], "spectrum_reasons": []},
    ]
    counts = reason_family_counts(rows)
    assert counts["rows"] == 3
    assert counts["union_removed"] == 2
    assert counts["quality_independent"] == 1
    assert counts["structure_independent"] == 1
    assert counts["spectrum_independent"] == 1
    assert counts["quality_and_structure"] == 1
