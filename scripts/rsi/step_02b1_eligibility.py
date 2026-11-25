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


def _record_problem(record: Mapping[str, Any]) -> str | None:
    record_id = record.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        return "missing_record_id"
    source_gt = record.get("source_gt")
    if not isinstance(source_gt, str) or not source_gt:
        return "missing_source_gt"
    candidate = record.get("candidate")
    if not isinstance(candidate, str) or not candidate:
        return "missing_candidate"
    if record.get("split") not in SPLITS:
        return "illegal_split"
    return None


def check_cross_split(
    records: Iterable[Mapping[str, Any]],
    frozen_groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply frozen structure ownership before any candidate is accepted.

    ``frozen_groups`` owns its structures even when this batch has no positive
    row for them. A negative candidate cannot take a frozen test or gate
    structure, and input order does not change that decision. Every input row
    is returned once, in ``accepted`` or ``rejected``.

    An unfrozen structure that is only a candidate, and is requested by more
    than one split, keeps the lexicographically smallest split name. The other
    rows are rejected with ``unfrozen_structure_tie_break``. A structure that
    is a source GT is owned by that group's split before candidate claims are
    considered, so this tie-break does not replace a source-GT or frozen owner.
    """
    frozen = dict(frozen_groups or {})
    items = [dict(record) for record in records]
    rejected: list[dict[str, Any]] = []
    if any(
        not isinstance(structure, str) or not structure or split not in SPLITS
        for structure, split in frozen.items()
    ):
        for record in items:
            rejected.append({**record, "reason": "illegal_frozen_assignment"})
        return {
            "accepted": [],
            "rejected": rejected,
            "structure_owner": {},
            "leaked_structures": [],
            "splits": list(SPLITS),
            "tie_break": "lexicographic split name for unfrozen candidate-only claims",
        }

    valid: list[dict[str, Any]] = []
    for record in items:
        problem = _record_problem(record)
        if problem:
            rejected.append({**record, "reason": problem})
        else:
            valid.append(record)
    id_counts = Counter(record["record_id"] for record in valid)
    unique: list[dict[str, Any]] = []
    for record in valid:
        if id_counts[record["record_id"]] > 1:
            rejected.append({**record, "reason": "duplicate_record_id"})
        else:
            unique.append(record)

    provisional = dict(frozen)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in unique:
        grouped.setdefault(record["source_gt"], []).append(record)
    pending: list[dict[str, Any]] = []
    for source_gt in sorted(grouped):
        group = grouped[source_gt]
        requested = sorted({record["split"] for record in group})
        if source_gt in frozen and any(record["split"] != frozen[source_gt] for record in group):
            required = frozen[source_gt]
            for record in group:
                if record["split"] != required:
                    rejected.append({
                        **record,
                        "reason": "frozen_source_group",
                        "required_split": required,
                    })
                else:
                    pending.append(record)
            continue
        if len(requested) > 1:
            for record in group:
                rejected.append({
                    **record,
                    "reason": "source_gt_derivatives_conflict",
                    "requested_splits": requested,
                })
            continue
        proposed = frozen.get(source_gt, requested[0])
        if source_gt in provisional and provisional[source_gt] != proposed:
            for record in group:
                rejected.append({
                    **record,
                    "reason": "cross_split_structure",
                    "structure": source_gt,
                    "owner_split": provisional[source_gt],
                })
            continue
        provisional.setdefault(source_gt, proposed)
        pending.extend(group)

    claim_splits: dict[str, set[str]] = {}
    for record in pending:
        claim_splits.setdefault(record["candidate"], set()).add(record["split"])
    accepted: list[dict[str, Any]] = []
    for record in pending:
        candidate = record["candidate"]
        split = record["split"]
        if candidate in provisional and provisional[candidate] != split:
            rejected.append({
                **record,
                "reason": "cross_split_structure",
                "structure": candidate,
                "owner_split": provisional[candidate],
            })
            continue
        claims = claim_splits.get(candidate, {split})
        if candidate not in provisional and len(claims) > 1 and split != min(claims):
            rejected.append({
                **record,
                "reason": "unfrozen_structure_tie_break",
                "structure": candidate,
                "kept_split": min(claims),
            })
            continue
        accepted.append({**record, "assigned_split": split})

    owner = dict(frozen)
    for record in accepted:
        for value in (record["source_gt"], record["candidate"]):
            current = owner.get(value)
            if current is not None and current != record["assigned_split"]:
                raise RuntimeError(f"ownership conflict for {value}")
            owner[value] = record["assigned_split"]
    leaked = []
    seen: dict[str, str] = {}
    for record in accepted:
        for value in (record["source_gt"], record["candidate"]):
            if value in seen and seen[value] != record["assigned_split"]:
                leaked.append(value)
            seen[value] = record["assigned_split"]
    covered = [item.get("record_id") for item in accepted] + [item.get("record_id") for item in rejected]
    if len(covered) != len(items):
        raise RuntimeError("accepted and rejected do not cover every input record")
    return {
        "accepted": accepted,
        "rejected": rejected,
        "structure_owner": owner,
        "leaked_structures": leaked,
        "splits": list(SPLITS),
        "tie_break": "lexicographic split name for unfrozen candidate-only claims",
    }
