"""Synthetic checks for the step-02A fingerprint and overlap audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

from step_02a_fingerprints import spectrum_views, structure_keys  # noqa: E402
from step_02a_overlap import summarize_records  # noqa: E402
from step_02a_records import build_index_record  # noqa: E402


def _record(candidate, gt, label, spectrum, negative_type="neg/random"):
    return build_index_record(
        asset="synthetic",
        role="test",
        locator="memory",
        candidate=candidate,
        gt=gt,
        label=label,
        negative_type=negative_type,
        spectrum=spectrum,
    )


SPECTRUM_A = {
    "c_nmr_peaks": [{"delta (ppm)": 57.8}, {"delta (ppm)": 18.2}],
    "h_nmr_peaks": [{"delta": 1.18, "centroid": 1.18, "nH": 3, "j_values": "7.1_"}],
}
SPECTRUM_J = {
    "c_nmr_peaks": [{"delta (ppm)": 57.8}, {"delta (ppm)": 18.2}],
    "h_nmr_peaks": [{"delta": 1.18, "centroid": 1.18, "nH": 3, "j_values": "7.2_"}],
}
SPECTRUM_ORDER = {
    "c_nmr_peaks": [{"delta (ppm)": 18.2}, {"delta (ppm)": 57.8}],
    "h_nmr_peaks": [{"delta": 1.18, "centroid": 1.18, "nH": 3, "j_values": "7.1_"}],
}


def test_duplicate_rows_differ_from_same_gt_different_candidates():
    rows = [
        _record("CCO", "CCO", 1, SPECTRUM_A, "positive"),
        _record("CCO", "CCO", 1, SPECTRUM_A, "positive"),
        _record("CC", "CCO", 0, SPECTRUM_A, "neg/random"),
    ]
    summary = summarize_records(rows)
    assert summary["rows"] == 3
    assert summary["unique_candidate_tokens"] == 2
    assert summary["unique_gt_tokens"] == 1
    assert summary["duplicate_groups"] == 1
    assert summary["duplicate_excess_rows"] == 1
    assert summary["gt_groups_with_multiple_candidates"] == 1
    assert summary["contradictory_pair_groups"] == 0


def test_unparsed_smiles_remain_in_the_row_denominator():
    rows = [_record("not a smiles", "CCO", 0, SPECTRUM_A)]
    summary = summarize_records(rows)
    assert summary["rows"] == 1
    assert summary["candidate_status_rows"]["unparsed"] == 1
    assert summary["unique_candidate_tokens"] == 1
    assert summary["unique_gt_tokens"] == 1
    keys = structure_keys("not a smiles")
    assert keys["status"] == "unparsed"
    assert keys["canonical_isomeric"] is None


def test_contradictory_labels_on_the_same_pair_are_reported():
    rows = [
        _record("CCO", "CC", 1, SPECTRUM_A, "positive"),
        _record("CCO", "CCC", 0, SPECTRUM_A, "neg/same_formula"),
    ]
    summary = summarize_records(rows)
    assert summary["contradictory_pair_groups"] == 1
    assert summary["rows"] == 2


def test_missing_spectrum_is_not_a_contradictory_pair():
    rows = [
        _record("CCO", "CCO", 1, None, "positive"),
        _record("CCO", "CC", 0, None, "neg/random"),
    ]
    summary = summarize_records(rows)
    assert summary["rows"] == 2
    assert summary["contradictory_pair_groups"] == 0


def test_physical_spectrum_differences_do_not_collapse():
    base = spectrum_views(SPECTRUM_A)
    changed_j = spectrum_views(SPECTRUM_J)
    reordered = spectrum_views(SPECTRUM_ORDER)
    null_integral = spectrum_views({
        "h_nmr_peaks": [{"delta": 1.18, "nH": None}],
    })
    zero_integral = spectrum_views({
        "h_nmr_peaks": [{"delta": 1.18, "nH": 0}],
    })
    missing_integral = spectrum_views({
        "h_nmr_peaks": [{"delta": 1.18}],
    })
    same_float = spectrum_views({
        "c_nmr_peaks": [{"delta (ppm)": 57.80}, {"delta (ppm)": 18.20}],
        "h_nmr_peaks": SPECTRUM_A["h_nmr_peaks"],
    })
    tenth = spectrum_views({
        "c_nmr_peaks": [{"delta (ppm)": 57.9}, {"delta (ppm)": 18.2}],
        "h_nmr_peaks": SPECTRUM_A["h_nmr_peaks"],
    })
    assert base["normalized_hash"] != changed_j["normalized_hash"]
    assert base["normalized_hash"] != reordered["normalized_hash"]
    assert null_integral["normalized_hash"] != zero_integral["normalized_hash"]
    assert null_integral["normalized_hash"] != missing_integral["normalized_hash"]
    assert base["normalized_hash"] == same_float["normalized_hash"]
    assert base["normalized_hash"] != tenth["normalized_hash"]


def test_reading_a_source_does_not_change_its_bytes(tmp_path):
    source = tmp_path / "rows.json"
    payload = [_record("CCO", "CCO", 1, SPECTRUM_A, "positive")]
    source.write_text(json.dumps(payload), encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    mtime = source.stat().st_mtime_ns
    loaded = json.loads(source.read_text(encoding="utf-8"))
    summarize_records(loaded)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert source.stat().st_mtime_ns == mtime


def test_shift_sequence_overlap_does_not_remove_a_source_row():
    from step_02a_refine import classify_source_row

    confirmed = {
        "candidate": {"canonical:CCO"},
        "gt": set(),
        "spectrum": set(),
        "pair": set(),
        "shift_sequence": {"shift-a"},
    }
    empty = {"candidate": set(), "gt": set(), "spectrum": set(), "pair": set(), "shift_sequence": set()}
    unknown = {"candidate": {"unparsed:abc"}, "gt": set()}
    row = {
        "candidate_token": "unparsed:abc",
        "gt_token": "missing",
        "spectrum_norm_hash": "spectrum",
        "pair_token": "pair",
        "shift_sequence_hash": "shift-a",
    }
    flags = classify_source_row(row, confirmed, empty, empty, empty)
    assert flags["confirmed_shift_sequence"] is True
    assert flags["excluded_from_upper_bound"] is False
    seen = classify_source_row(
        {"candidate_token": "canonical:CCO", "gt_token": "missing"},
        confirmed, empty, empty, empty,
    )
    assert seen["excluded_from_upper_bound"] is True
    conservative = classify_source_row(row, confirmed, empty, empty, unknown)
    assert conservative["excluded_from_upper_bound"] is False
    assert conservative["excluded_from_conservative"] is True


def test_modules_compile_without_writing_bytecode():
    for name in (
        "step_02a_fingerprints.py",
        "step_02a_overlap.py",
        "step_02a_records.py",
        "step_02a_audit.py",
        "step_02a_refine.py",
    ):
        path = REPO / "scripts" / "rsi" / name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
