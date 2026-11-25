"""Aggregate row, molecule, and contradiction counts from fingerprint records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def _add(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count rows separately from unique molecules and source groups.

    ``records`` items use the compact index fields written by the audit:
    candidate_token, gt_token, candidate_status, gt_status, label,
    spectrum_norm_hash, pair_token, scaffold, connectivity.
    """
    rows = 0
    candidate_status: dict[str, int] = {}
    gt_status: dict[str, int] = {}
    label_rows: dict[str, int] = {}
    candidates: set[str] = set()
    gts: set[str] = set()
    connectivities: set[str] = set()
    scaffolds: set[str] = set()
    spectra: set[str] = set()
    pairs: set[str] = set()
    duplicate_key_rows: dict[tuple[Any, ...], int] = defaultdict(int)
    gt_to_candidates: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[str, set[str]] = defaultdict(set)
    spectrum_labels: dict[str, set[str]] = defaultdict(set)

    for record in records:
        rows += 1
        _add(candidate_status, str(record.get("candidate_status") or "missing"))
        _add(gt_status, str(record.get("gt_status") or "missing"))
        label = record.get("label")
        _add(label_rows, "null" if label is None else str(label))
        candidate = record.get("candidate_token")
        gt = record.get("gt_token")
        spectrum = record.get("spectrum_norm_hash")
        pair = record.get("pair_token")
        if candidate and candidate != "missing":
            candidates.add(candidate)
        if gt and gt != "missing":
            gts.add(gt)
        if record.get("connectivity"):
            connectivities.add(record["connectivity"])
        if record.get("scaffold"):
            scaffolds.add(record["scaffold"])
        if spectrum:
            spectra.add(spectrum)
            if label is not None:
                spectrum_labels[spectrum].add(str(label))
        if pair:
            pairs.add(pair)
            # A missing spectrum is not one shared physical spectrum. Counting
            # those rows would call the same candidate a contradiction merely
            # because it was paired with different unstored spectra.
            if spectrum and label is not None:
                pair_labels[pair].add(str(label))
        if gt and gt != "missing" and candidate and candidate != "missing":
            gt_to_candidates[gt].add(candidate)
        duplicate_key_rows[(
            candidate,
            gt,
            spectrum,
            None if label is None else str(label),
            record.get("negative_type"),
        )] += 1

    duplicate_groups = sum(1 for count in duplicate_key_rows.values() if count > 1)
    duplicate_excess = sum(count - 1 for count in duplicate_key_rows.values() if count > 1)
    same_gt_multi = {
        gt: sorted(cands)
        for gt, cands in gt_to_candidates.items()
        if len(cands) > 1
    }
    contradictory_pairs = {
        pair: sorted(labels)
        for pair, labels in pair_labels.items()
        if len(labels) > 1
    }
    return {
        "rows": rows,
        "candidate_status_rows": candidate_status,
        "gt_status_rows": gt_status,
        "label_rows": label_rows,
        "unique_candidate_tokens": len(candidates),
        "unique_gt_tokens": len(gts),
        "unique_connectivity": len(connectivities),
        "unique_scaffold": len(scaffolds),
        "unique_spectrum_norm": len(spectra),
        "unique_candidate_spectrum_pairs": len(pairs),
        "duplicate_groups": duplicate_groups,
        "duplicate_excess_rows": duplicate_excess,
        "gt_groups_with_multiple_candidates": len(same_gt_multi),
        "contradictory_pair_groups": len(contradictory_pairs),
        "contradictory_pairs": contradictory_pairs,
        "same_gt_multiple_candidates": {
            gt: cands for gt, cands in same_gt_multi.items()
        },
    }


def intersection_count(left: set[str], right: set[str]) -> dict[str, int]:
    both = left & right
    return {
        "left": len(left),
        "right": len(right),
        "intersection": len(both),
        "left_only": len(left - right),
        "right_only": len(right - left),
    }
