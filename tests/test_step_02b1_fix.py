"""Synthetic checks for the 02B-1 repair.

The historical expected values come from an independent witness. Corrupting the
parser output must change the audit status.
"""

from __future__ import annotations

import copy
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import step_02b1_audit as audit  # noqa: E402
from step_02b1_eligibility import check_cross_split  # noqa: E402

FIXTURE = (
    "Candidate SMILES: CCO\n"
    "13C NMR (100 MHz, CDCl3): 57.8\n"
    "1H NMR (400 MHz, CDCl3): 1.20 (s, J=7.1, 12H)\n"
)


def _ids(rows):
    return [row.get("record_id") for row in rows]


def _decision(result):
    return {
        "accepted": _ids(result["accepted"]),
        "rejected": [(row["record_id"], row["reason"]) for row in result["rejected"]],
        "owner": result["structure_owner"],
        "leaked": result["leaked_structures"],
    }


def test_r1_tampered_parser_output_is_changed_against_the_witness(monkeypatch):
    bucket: dict[str, Counter] = defaultdict(Counter)
    honest = audit._text_fields(FIXTURE, bucket)
    assert honest["nH"] == "kept"
    assert honest["j"] == "kept"
    assert honest["category"] == "kept"
    assert bucket["nH"]["denominator"] == 1
    assert bucket["j"]["denominator"] == 1
    assert bucket["category"]["denominator"] == 1

    real_parse = audit.parse_contract_text
    parsed, diagnostics = real_parse(FIXTURE)
    tampered = copy.deepcopy(parsed)
    tampered["h_nmr_peaks"][0]["nH"] = 2
    tampered["h_nmr_peaks"][0]["j_values"] = "7.1, 1"

    def fake_parse(_text):
        return copy.deepcopy(tampered), diagnostics

    def audited_parser_must_not_supply_the_expected(*_args, **_kwargs):
        raise AssertionError("the audited parser supplied the expected nH or J")

    monkeypatch.setattr(audit, "parse_contract_text", fake_parse)
    monkeypatch.setattr(
        "recipe.nmr_matching_agent.task_input.parse_h_peak_list",
        audited_parser_must_not_supply_the_expected,
    )
    scored = audit._text_fields(FIXTURE, bucket)
    assert scored["nH"] == "changed"
    assert scored["j"] == "changed"
    assert scored["category"] == "kept"
    assert bucket["nH"]["changed"] == 1
    assert bucket["nH"]["kept"] == 1
    assert bucket["nH"]["denominator"] == 2
    assert scored["parse"] == "kept"


def test_r1_structured_range_stays_kept_until_its_value_changes():
    source = [{
        "category": "m",
        "centroid": 7.2,
        "delta": 7.2,
        "j_values": "7.1",
        "nH": 12,
        "rangeMin": 7.1,
        "rangeMax": 7.4,
    }]
    totals = defaultdict(Counter)
    assert audit._explicit_range_status(source, source) == "kept"
    changed = copy.deepcopy(source)
    changed[0]["rangeMax"] = 7.9
    assert audit._explicit_range_status(source, changed) == "changed"
    invented = [{"rangeMin": 7.1, "rangeMax": 7.4, "delta": 7.2, "centroid": 7.2}]
    text_only = [{"rangeMin": 7.1, "rangeMax": 7.4}]
    assert audit._explicit_range_status(text_only, invented) == "changed"
    task = {
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": source,
    }
    fields = audit._structured_fields(task, totals)
    assert fields["ranges"] == "kept"
    assert fields["nH"] == "kept"
    assert fields["j"] == "kept"
    assert totals["ranges"]["denominator"] == 1


def test_r3_frozen_ownership_does_not_depend_on_order():
    frozen = {"CCO": "test"}
    negative = {
        "record_id": "dev_neg",
        "source_gt": "CCC",
        "candidate": "CCO",
        "split": "search-dev",
        "kind": "negative",
    }
    positive = {
        "record_id": "test_pos",
        "source_gt": "CCO",
        "candidate": "CCO",
        "split": "test",
        "kind": "positive",
    }
    forward = _decision(check_cross_split([negative, positive], frozen))
    reverse = _decision(check_cross_split([positive, negative], frozen))
    assert forward == reverse
    assert forward["accepted"] == ["test_pos"]
    assert forward["rejected"] == [("dev_neg", "cross_split_structure")]
    assert forward["owner"]["CCO"] == "test"
    assert forward["leaked"] == []

    alone = check_cross_split([negative], frozen)
    assert _ids(alone["accepted"]) == []
    assert alone["rejected"][0]["reason"] == "cross_split_structure"
    assert alone["structure_owner"]["CCO"] == "test"
    assert alone["leaked_structures"] == []
    assert len(alone["accepted"]) + len(alone["rejected"]) == 1


def test_r3_multiple_frozen_splits_and_derivatives_are_order_independent():
    frozen = {"CCO": "test", "CCC": "gate-dev"}
    rows = [
        {"record_id": "neg-cco", "source_gt": "AAA", "candidate": "CCO", "split": "search-dev", "kind": "negative"},
        {"record_id": "pos-cco", "source_gt": "CCO", "candidate": "CCO", "split": "test", "kind": "positive"},
        {"record_id": "pos-ccc", "source_gt": "CCC", "candidate": "CCC", "split": "gate-dev", "kind": "positive"},
        {"record_id": "neg-ccc", "source_gt": "BBB", "candidate": "CCC", "split": "future-train", "kind": "negative"},
        {"record_id": "same-1", "source_gt": "ZZZ", "candidate": "ZZZ", "split": "future-train", "kind": "positive"},
        {"record_id": "same-2", "source_gt": "ZZZ", "candidate": "QQQ", "split": "future-train", "kind": "negative"},
        {"record_id": "split-a", "source_gt": "SHARE", "candidate": "SHARE", "split": "search-dev", "kind": "positive"},
        {"record_id": "split-b", "source_gt": "SHARE", "candidate": "OTHER", "split": "gate-dev", "kind": "negative"},
    ]
    forward = _decision(check_cross_split(rows, frozen))
    reverse = _decision(check_cross_split(list(reversed(rows)), frozen))
    assert set(forward["accepted"]) == set(reverse["accepted"])
    assert sorted(forward["rejected"]) == sorted(reverse["rejected"])
    assert forward["owner"] == reverse["owner"]
    assert forward["leaked"] == []
    assert set(forward["accepted"]) == {"pos-cco", "pos-ccc", "same-1", "same-2"}
    reasons = dict(forward["rejected"])
    assert reasons["neg-cco"] == "cross_split_structure"
    assert reasons["neg-ccc"] == "cross_split_structure"
    assert reasons["split-a"] == "source_gt_derivatives_conflict"
    assert reasons["split-b"] == "source_gt_derivatives_conflict"
    assert forward["owner"]["CCO"] == "test"
    assert forward["owner"]["CCC"] == "gate-dev"
    assert forward["owner"]["ZZZ"] == "future-train"


def test_r3_unfrozen_candidate_tie_break_is_lexicographic_and_reproducible():
    rows = [
        {"record_id": "from-test", "source_gt": "AAA", "candidate": "ZZZ", "split": "test", "kind": "negative"},
        {"record_id": "from-dev", "source_gt": "BBB", "candidate": "ZZZ", "split": "search-dev", "kind": "negative"},
    ]
    forward = check_cross_split(rows, {})
    reverse = check_cross_split(list(reversed(rows)), {})
    assert _ids(forward["accepted"]) == _ids(reverse["accepted"]) == ["from-dev"]
    assert forward["rejected"][0]["reason"] == "unfrozen_structure_tie_break"
    assert forward["rejected"][0]["kept_split"] == "search-dev"
    assert forward["structure_owner"]["ZZZ"] == "search-dev"
    assert forward["tie_break"].startswith("lexicographic")
    assert len(forward["accepted"]) + len(forward["rejected"]) == 2

    owned = [
        {"record_id": "owner", "source_gt": "ZZZ", "candidate": "ZZZ", "split": "test", "kind": "positive"},
        {"record_id": "claim", "source_gt": "AAA", "candidate": "ZZZ", "split": "search-dev", "kind": "negative"},
    ]
    result = check_cross_split(owned, {})
    assert _ids(result["accepted"]) == ["owner"]
    assert result["rejected"][0]["reason"] == "cross_split_structure"
    assert result["structure_owner"]["ZZZ"] == "test"


def test_r3_schema_errors_cover_every_row_and_do_not_merge_missing_ids():
    rows = [
        {"source_gt": "CCO", "candidate": "CCO", "split": "test"},
        {"record_id": None, "source_gt": "CCO", "candidate": "CCO", "split": "test"},
        {"record_id": "", "source_gt": "CCO", "candidate": "CCO", "split": "test"},
        {"record_id": "no-gt", "source_gt": None, "candidate": "CCO", "split": "test"},
        {"record_id": "no-candidate", "source_gt": "CCO", "split": "test"},
        {"record_id": "bad-split", "source_gt": "CCO", "candidate": "CCO", "split": "train"},
        {"record_id": "dup", "source_gt": "AAA", "candidate": "AAA", "split": "test"},
        {"record_id": "dup", "source_gt": "BBB", "candidate": "BBB", "split": "gate-dev"},
        {"record_id": "ok", "source_gt": "CCC", "candidate": "CCC", "split": "future-train"},
    ]
    result = check_cross_split(rows, {"CCO": "test"})
    assert len(result["accepted"]) + len(result["rejected"]) == len(rows)
    assert _ids(result["accepted"]) == ["ok"]
    assert result["structure_owner"]["CCO"] == "test"
    reasons = [row["reason"] for row in result["rejected"]]
    assert reasons.count("missing_record_id") == 3
    assert "missing_source_gt" in reasons
    assert "missing_candidate" in reasons
    assert "illegal_split" in reasons
    assert reasons.count("duplicate_record_id") == 2
    assert "ok" not in _ids(result["rejected"])
    covered = _ids(result["accepted"]) + _ids(result["rejected"])
    assert covered.count("dup") == 2
    assert covered.count(None) == 2
    assert covered.count("") == 1

    illegal = check_cross_split(
        [{"record_id": "x", "source_gt": "CCO", "candidate": "CCO", "split": "test"}],
        {"CCO": "not-a-split"},
    )
    assert illegal["accepted"] == []
    assert illegal["rejected"][0]["reason"] == "illegal_frozen_assignment"
    assert illegal["structure_owner"] == {}
