"""Build compact audit index records from already-parsed fields."""

from __future__ import annotations

from typing import Any, Mapping

from step_02a_fingerprints import (
    identity_token,
    pair_token,
    shift_sequence_hash,
    spectrum_views,
    structure_keys,
)


def build_index_record(
    *,
    asset: str,
    role: str,
    locator: str,
    candidate: str | None,
    gt: str | None,
    label: Any,
    negative_type: str | None,
    spectrum: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate_keys = structure_keys(candidate)
    gt_keys = structure_keys(gt)
    viewed = spectrum_views(spectrum)
    spectrum_hash = viewed.get("normalized_hash")
    return {
        "asset": asset,
        "role": role,
        "locator": locator,
        "candidate_raw": candidate,
        "candidate_status": candidate_keys["status"],
        "candidate_token": identity_token(candidate_keys),
        "canonical": candidate_keys["canonical_isomeric"],
        "connectivity": candidate_keys["connectivity"],
        "scaffold": candidate_keys["scaffold"],
        "gt_raw": gt,
        "gt_status": gt_keys["status"],
        "gt_token": identity_token(gt_keys),
        "gt_canonical": gt_keys["canonical_isomeric"],
        "gt_connectivity": gt_keys["connectivity"],
        "gt_scaffold": gt_keys["scaffold"],
        "label": label,
        "negative_type": negative_type,
        "spectrum_status": viewed["status"],
        "spectrum_raw_hash": viewed["raw_content_hash"],
        "spectrum_norm_hash": spectrum_hash,
        "pair_token": pair_token(candidate_keys, spectrum_hash),
        "shift_sequence_hash": shift_sequence_hash(spectrum),
    }
