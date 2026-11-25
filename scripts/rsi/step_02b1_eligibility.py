"""Source-eligibility rules for step 02B-1.

Exposure exclusion starts from the step-02A conservative rule and also removes
test_2000 candidate structures. A shift-sequence hit is recorded and is not
treated as an identical spectrum. Quality failures stay in the denominator.
This module does not choose a development split or call a predictor.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

PROTOCOL_VERSION = "step_02b1.v1"
SPLITS = ("search-dev", "gate-dev", "future-train", "test")

EXPOSURE_REASONS = (
    "confirmed_structure",
    "inferred_structure",
    "confirmed_exact_spectrum",
    "inferred_exact_spectrum",
    "confirmed_pair",
    "inferred_pair",
    "known_eval_structure",
    "test_gt_structure",
    "test_candidate_structure",
    "unknown_possible_structure",
)


def _present(value: Any) -> bool:
    return value not in (None, "", "missing")


def _hit(value: Any, bucket: set[str] | None) -> bool:
    return bool(bucket) and _present(value) and value in bucket


def _structure_hit(row: Mapping[str, Any], tokens: set[str] | None) -> bool:
    return _hit(row.get("candidate_token"), tokens) or _hit(row.get("gt_token"), tokens)


def exposure_reasons(
    row: Mapping[str, Any],
    *,
    confirmed: Mapping[str, set[str]],
    inferred: Mapping[str, set[str]],
    evaluation: Mapping[str, set[str]],
    test_gt: set[str],
    test_candidate: set[str],
    unknown: set[str],
) -> list[str]:
    """Reasons that remove a row from the conservative pool. Shift-only is absent."""
    reasons: list[str] = []
    if _structure_hit(row, confirmed.get("structure")):
        reasons.append("confirmed_structure")
    if _structure_hit(row, inferred.get("structure")):
        reasons.append("inferred_structure")
    spectrum = row.get("spectrum_norm_hash")
    if _hit(spectrum, confirmed.get("spectrum")):
        reasons.append("confirmed_exact_spectrum")
    if _hit(spectrum, inferred.get("spectrum")):
        reasons.append("inferred_exact_spectrum")
    pair = row.get("pair_token")
    if _hit(pair, confirmed.get("pair")):
        reasons.append("confirmed_pair")
    if _hit(pair, inferred.get("pair")):
        reasons.append("inferred_pair")
    if _structure_hit(row, evaluation.get("structure")):
        reasons.append("known_eval_structure")
    if _structure_hit(row, test_gt):
        reasons.append("test_gt_structure")
    if _structure_hit(row, test_candidate):
        reasons.append("test_candidate_structure")
    if _structure_hit(row, unknown):
        reasons.append("unknown_possible_structure")
    return reasons


def shift_only_diagnostic(row: Mapping[str, Any], shift_tokens: set[str], exact_tokens: set[str]) -> bool:
    """A shared shift list is not a full-spectrum match."""
    shift = row.get("shift_sequence_hash")
    spectrum = row.get("spectrum_norm_hash")
    return _hit(shift, shift_tokens) and not _hit(spectrum, exact_tokens)


def quality_reasons(flags: Mapping[str, bool]) -> list[str]:
    """Hard quality failures. Missing nH, one nucleus, or unknown frequency are not failures."""
    reasons = []
    for key in (
        "unparsed_structure",
        "missing_structure",
        "no_valid_spectrum",
        "non_finite_value",
        "gold_unresolved",
    ):
        if flags.get(key):
            reasons.append(key)
    return reasons


def in_auto_positive_pool(exposure: Iterable[str], quality: Iterable[str]) -> bool:
    return not list(exposure) and not list(quality)


def exposure_remainder(rows: Iterable[Mapping[str, Any]]) -> int:
    """Each exposed row leaves the remainder once, however many reasons it has."""
    return sum(1 for row in rows if not row.get("exposure_reasons"))


def reason_family_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Independent, overlap, and union counts for quality, structure, and spectrum."""
    families = Counter()
    for row in rows:
        quality = bool(row.get("quality_reasons"))
        structure = bool(row.get("structure_reasons"))
        spectrum = bool(row.get("spectrum_reasons"))
        key = ("quality" if quality else "") + ("+structure" if structure else "") + ("+spectrum" if spectrum else "")
        key = key.lstrip("+") or "none"
        families[key] += 1
    quality_only = families["quality"]
    structure_only = families["structure"]
    spectrum_only = families["spectrum"]
    quality_structure = families["quality+structure"]
    quality_spectrum = families["quality+spectrum"]
    structure_spectrum = families["structure+spectrum"]
    all_three = families["quality+structure+spectrum"]
    return {
        "rows": sum(families.values()),
        "none": families["none"],
        "quality_independent": quality_only + quality_structure + quality_spectrum + all_three,
        "structure_independent": structure_only + quality_structure + structure_spectrum + all_three,
        "spectrum_independent": spectrum_only + quality_spectrum + structure_spectrum + all_three,
        "quality_and_structure": quality_structure + all_three,
        "quality_and_spectrum": quality_spectrum + all_three,
        "structure_and_spectrum": structure_spectrum + all_three,
        "all_three": all_three,
        "union_removed": sum(families.values()) - families["none"],
        "combinations": dict(families),
    }


def spectrum_difference_labels(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[str, ...]:
    """Every applicable difference. Empty list and null are not the same label as a missing field."""
    labels: list[str] = []

    def empty_versus_null(field: str, prefix: str) -> None:
        a = left.get(field) if field in left else None
        b = right.get(field) if field in right else None
        a_present = field in left
        b_present = field in right
        if a_present and b_present and a == [] and b is None:
            labels.append(f"{prefix}_empty_list_vs_null")
        elif a_present and b_present and a is None and b == []:
            labels.append(f"{prefix}_null_vs_empty_list")

    empty_versus_null("c_nmr_peaks", "c")
    empty_versus_null("h_nmr_peaks", "h")
    left_h = left.get("h_nmr_peaks")
    right_h = right.get("h_nmr_peaks")
    if isinstance(left_h, list) and isinstance(right_h, list) and len(left_h) == len(right_h):
        flags = set()
        for peak_a, peak_b in zip(left_h, right_h):
            if not isinstance(peak_a, dict) or not isinstance(peak_b, dict):
                flags.add("h_not_object")
                continue
            if peak_a.get("category") != peak_b.get("category"):
                flags.add("h_category")
            if peak_a.get("j_values") != peak_b.get("j_values"):
                flags.add("h_j")
            if peak_a.get("nH") != peak_b.get("nH"):
                flags.add("h_nH")
            if peak_a.get("rangeMin") != peak_b.get("rangeMin") or peak_a.get("rangeMax") != peak_b.get("rangeMax"):
                flags.add("h_range")
        labels.extend(sorted(flags))
    return tuple(labels)


def check_cross_split(
    records: Iterable[Mapping[str, Any]],
    frozen_groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Keep every derivative of one source GT in one split, including negative candidates.

    ``records`` contain ``source_gt``, ``candidate``, ``split``, and ``kind``.
    A structure held by one split cannot appear as a GT or a negative candidate
    in another split. ``frozen_groups`` maps a source GT to the split it must keep.
    """
    frozen = dict(frozen_groups or {})
    items = [dict(record) for record in records]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in items:
        grouped.setdefault(str(record.get("source_gt")), []).append(record)
    group_split: dict[str, str] = {}
    rejected: list[dict[str, Any]] = []
    blocked: set[tuple[str, str]] = set()
    for source_gt, group in grouped.items():
        requested = {str(record.get("split")) for record in group}
        if source_gt in frozen:
            required = frozen[source_gt]
            group_split[source_gt] = required
            for record in group:
                if str(record.get("split")) != required:
                    rejected.append({
                        **record,
                        "reason": "frozen_source_group",
                        "required_split": required,
                    })
                    blocked.add((str(record.get("record_id")), str(record.get("split"))))
            continue
        if len(requested) > 1:
            for record in group:
                rejected.append({
                    **record,
                    "reason": "source_gt_derivatives_conflict",
                    "requested_splits": sorted(requested),
                })
                blocked.add((str(record.get("record_id")), str(record.get("split"))))
            continue
        group_split[source_gt] = next(iter(requested))

    accepted: list[dict[str, Any]] = []
    owner: dict[str, str] = {}
    for record in items:
        source_gt = str(record.get("source_gt"))
        split = str(record.get("split"))
        if (str(record.get("record_id")), split) in blocked or source_gt not in group_split:
            continue
        if split != group_split[source_gt]:
            continue
        structures = [value for value in (record.get("source_gt"), record.get("candidate")) if _present(value)]
        conflict = next((value for value in structures if value in owner and owner[value] != split), None)
        if conflict is not None:
            rejected.append({
                **record,
                "reason": "cross_split_structure",
                "structure": conflict,
                "owner_split": owner[conflict],
            })
            continue
        for value in structures:
            owner.setdefault(str(value), split)
        accepted.append({**record, "assigned_split": split})
    leaked = []
    seen: dict[str, str] = {}
    for record in accepted:
        for value in (record.get("source_gt"), record.get("candidate")):
            if not _present(value):
                continue
            if value in seen and seen[str(value)] != record["assigned_split"]:
                leaked.append(str(value))
            seen[str(value)] = str(record["assigned_split"])
    return {
        "accepted": accepted,
        "rejected": rejected,
        "structure_owner": owner,
        "leaked_structures": leaked,
        "splits": list(SPLITS),
    }
