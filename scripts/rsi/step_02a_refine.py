"""Second pass over step-02A indexes.

The LMDB scan writes compact rows. This module recomputes isolation from those
rows and corrects pair-contradiction counts so a missing spectrum is not treated
as one shared spectrum. It does not read weight tensors or rewrite sources.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _present(value: Any) -> bool:
    return value not in (None, "", "missing")


def _in(value: Any, bucket: set[str] | None) -> bool:
    return bool(bucket) and _present(value) and value in bucket


def classify_source_row(
    row: Mapping[str, Any],
    confirmed: Mapping[str, set[str]],
    evaluation: Mapping[str, set[str]],
    test: Mapping[str, set[str]],
    unknown: Mapping[str, set[str]],
) -> dict[str, bool]:
    """Classify one source molecule without treating a shift list as identity."""
    candidate = row.get("candidate_token")
    gt = row.get("gt_token")
    structure = (
        _in(candidate, confirmed.get("candidate"))
        or _in(candidate, confirmed.get("gt"))
        or _in(gt, confirmed.get("gt"))
        or _in(gt, confirmed.get("candidate"))
    )
    spectrum = _in(row.get("spectrum_norm_hash"), confirmed.get("spectrum"))
    pair = _in(row.get("pair_token"), confirmed.get("pair"))
    shift = _in(row.get("shift_sequence_hash"), confirmed.get("shift_sequence"))
    eval_structure = (
        _in(candidate, evaluation.get("candidate"))
        or _in(candidate, evaluation.get("gt"))
        or _in(gt, evaluation.get("gt"))
        or _in(gt, evaluation.get("candidate"))
    )
    test_source = _in(candidate, test.get("gt")) or _in(gt, test.get("gt"))
    test_candidate = _in(candidate, test.get("candidate")) or _in(gt, test.get("candidate"))
    unknown_structure = (
        _in(candidate, unknown.get("candidate"))
        or _in(candidate, unknown.get("gt"))
        or _in(gt, unknown.get("gt"))
        or _in(gt, unknown.get("candidate"))
    )
    upper_excluded = structure or spectrum or pair or eval_structure or test_source
    return {
        "confirmed_structure": structure,
        "confirmed_exact_spectrum": spectrum,
        "confirmed_pair": pair,
        "confirmed_shift_sequence": shift,
        "shift_sequence_without_exact_spectrum": shift and not spectrum,
        "known_eval_structure": eval_structure,
        "test_source_group": test_source,
        "test_candidate_overlap_not_used_for_exclusion": test_candidate,
        "unknown_possible_structure": unknown_structure,
        "excluded_from_upper_bound": upper_excluded,
        "excluded_from_conservative": upper_excluded or unknown_structure,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _role_sets(records: Iterable[Mapping[str, Any]], roles: set[str]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    fields = (
        "candidate_token",
        "gt_token",
        "spectrum_norm_hash",
        "shift_sequence_hash",
        "pair_token",
        "connectivity",
        "scaffold",
        "gt_connectivity",
        "gt_scaffold",
    )
    for record in records:
        if record.get("role") not in roles:
            continue
        for field in fields:
            value = record.get(field)
            if _present(value):
                found[field].add(value)
    return {
        "candidate": found["candidate_token"],
        "gt": found["gt_token"],
        "spectrum": found["spectrum_norm_hash"],
        "shift_sequence": found["shift_sequence_hash"],
        "pair": found["pair_token"],
        "connectivity": found["connectivity"] | found["gt_connectivity"],
        "scaffold": found["scaffold"] | found["gt_scaffold"],
    }


def _human(row: Mapping[str, Any]) -> str | None:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == "human":
            value = turn.get("value")
            return value if isinstance(value, str) else None
    return None


def _quantify_alpaca(source: Path, roles: list[str]) -> dict[str, Any]:
    """Project v6.4 user JSON through the current Alpaca formatter and parser."""
    import sys

    spec = Path("/fs_mol/cxq/SpecMaster_jp")
    oa = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
    rsi = Path(__file__).resolve().parent
    for entry in (str(rsi), str(oa), str(spec / "scripts")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from build_test_2000 import format_alpaca
    from recipe.nmr_matching_agent.standalone_infer_matching import parse_alpaca_input
    from step_02a_audit import iter_jsonl, user_value

    rows = list(iter_jsonl(source))
    if len(rows) != len(roles):
        raise RuntimeError(f"{source} has {len(rows)} rows but {len(roles)} roles")
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    for (_line_number, row), role in zip(rows, roles):
        counter = by_role[role]
        counter["rows"] += 1
        task = json.loads(user_value(row) or "")
        rendered = format_alpaca({
            "candidate_smiles": task.get("candidate_smiles"),
            "c_nmr_peaks": task.get("c_nmr_peaks") or [],
            "h_nmr_peaks": task.get("h_nmr_peaks") or [],
            "nmr_frequency": task.get("nmr_frequency") or "",
            "nmr_solvent": task.get("nmr_solvent") or "",
            "label": 1,
            "gt_smiles": "",
        })["input"]
        recovered = parse_alpaca_input(rendered)
        frequency = task.get("nmr_frequency")
        if frequency:
            counter["rows_with_frequency"] += 1
            counter["frequency_occurrences_in_text"] += rendered.count(str(frequency))
            if rendered.count(str(frequency)) >= 2:
                counter["frequency_stamped_on_both_headers"] += 1
        if "nmr_solvent" not in recovered:
            counter["parser_omits_solvent"] += 1
        if "nmr_frequency" not in recovered:
            counter["parser_omits_frequency"] += 1
        if "experimental_total_nH" not in task:
            counter["user_json_lacks_experimental_total_nH"] += 1
        original_c = []
        for peak in task.get("c_nmr_peaks") or []:
            if isinstance(peak, dict) and peak.get("delta (ppm)") is not None:
                original_c.append(float(peak["delta (ppm)"]))
        parsed_c = [float(peak["delta (ppm)"]) for peak in recovered.get("c_nmr_peaks") or []]
        if len(original_c) != len(parsed_c):
            counter["c_count_changed"] += 1
        for left, right in zip(original_c, parsed_c):
            counter["c_peaks_compared"] += 1
            if abs(left - right) > 1e-6:
                counter["c_peaks_value_changed_by_rounding"] += 1
        parsed_h = recovered.get("h_nmr_peaks") or []
        original_h = [peak for peak in task.get("h_nmr_peaks") or [] if isinstance(peak, dict)]
        kept_h = []
        kept_nh = []
        for peak in original_h:
            shown = peak.get("centroid") or peak.get("delta")
            if peak.get("centroid") in (0, 0.0):
                counter["h_zero_centroid_falls_through_in_formatter"] += 1
            if peak.get("centroid") != peak.get("delta"):
                counter["h_centroid_differs_from_delta"] += 1
            j_values = peak.get("j_values")
            if j_values not in (None, "", "_"):
                counter["h_peaks_with_explicit_j"] += 1
                if "J=" not in rendered:
                    counter["explicit_j_missing_from_text"] += 1
            if "rangeMin" in peak or "rangeMax" in peak:
                counter["h_peaks_with_range"] += 1
                if "rangeMin" not in rendered and "rangeMax" not in rendered:
                    counter["range_not_written_by_formatter"] += 1
            if shown is None:
                counter["h_peaks_without_shift_skipped_by_formatter"] += 1
                continue
            kept_h.append(float(shown))
            kept_nh.append(peak.get("nH"))
        if len(kept_h) != len(parsed_h):
            counter["h_count_changed"] += 1
        for left, original_nh, peak in zip(kept_h, kept_nh, parsed_h):
            counter["h_peaks_compared"] += 1
            if abs(left - float(peak.get("centroid"))) > 1e-6:
                counter["h_peaks_value_changed_by_rounding"] += 1
            if original_nh != peak.get("nH"):
                counter["h_nH_changed"] += 1
        for peak in parsed_h:
            if "j_values" not in peak:
                counter["parser_h_peaks_without_j"] += 1
            if "category" not in peak:
                counter["parser_h_peaks_without_category"] += 1
            if "rangeMin" not in peak and "rangeMax" not in peak:
                counter["parser_h_peaks_without_range"] += 1
    return {role: dict(counter) for role, counter in sorted(by_role.items())}


def _registered_match(train_users: list[str], val_users: list[str]) -> dict[str, Any]:
    data = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data")

    def humans(path: Path) -> list[str]:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [_human(row) or "" for row in rows]

    train = humans(data / "nmr_matching_tool_sft_v6_4.json")
    validation = humans(data / "nmr_matching_tool_sft_v6_4_val.json")
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_human_equals_replayed_user_in_order": train == train_users,
        "validation_human_equals_replayed_user_in_order": validation == val_users,
        "train_set_equal": set(train) == set(train_users),
        "validation_set_equal": set(validation) == set(val_users),
    }


def _join_sibling(sharegpt: Path, trajectories: list[Path]) -> dict[str, Any]:
    from step_02a_audit import iter_jsonl, user_value

    meta_by_user: dict[str, set[tuple[Any, Any]]] = defaultdict(set)
    for path in trajectories:
        if not path.exists():
            continue
        for _line, row in iter_jsonl(path):
            text = user_value(row)
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            if text:
                meta_by_user[text].add((meta.get("gt_smiles") or None, meta.get("label")))
    rows = json.loads(sharegpt.read_text(encoding="utf-8"))
    joined = ambiguous = unmatched = 0
    for row in rows:
        text = _human(row)
        matches = meta_by_user.get(text or "", set())
        if len(matches) == 1:
            joined += 1
        elif len(matches) > 1:
            ambiguous += 1
        else:
            unmatched += 1
    return {
        "sharegpt": str(sharegpt),
        "trajectory_files": [str(path) for path in trajectories if path.exists()],
        "rows": len(rows),
        "unique_gt_joined": joined,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
        "note": "Sibling runs are not added to the confirmed exposure set.",
    }


def _recount_lmdb(out: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    from step_02a_audit import sha256_file_always

    confirmed = _role_sets(records, {"weight_train"})
    evaluation = _role_sets(records, {"weight_eval"})
    test = _role_sets(records, {"test"})
    unknown = _role_sets(records, {"unknown_ancestor"})
    sibling = _role_sets(records, {"sibling_train"})
    pools = _role_sets(records, {"generation_pool"})
    files = {}
    for name in ("all.lmdb", "train.lmdb", "valid.lmdb", "test.lmdb"):
        index = out / f"lmdb_{name}.index.jsonl"
        if not index.exists():
            files[name] = {"exists": False}
            continue
        stats: Counter[str] = Counter()
        upper_scaffolds: set[str] = set()
        conservative_scaffolds: set[str] = set()
        upper_tokens: set[str] = set()
        conservative_tokens: set[str] = set()
        with index.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                stats["rows"] += 1
                flags = classify_source_row(row, confirmed, evaluation, test, unknown)
                for key, value in flags.items():
                    if value:
                        stats[key] += 1
                connectivity = row.get("connectivity")
                if _present(connectivity) and _in(connectivity, confirmed.get("connectivity")):
                    stats["connectivity_overlaps_confirmed_train"] += 1
                if flags["excluded_from_upper_bound"]:
                    continue
                stats["remaining_upper_bound"] += 1
                token = row.get("candidate_token")
                scaffold = row.get("scaffold")
                if _present(token):
                    upper_tokens.add(token)
                if _present(scaffold):
                    upper_scaffolds.add(scaffold)
                if _in(token, sibling.get("candidate")) or _in(row.get("gt_token"), sibling.get("candidate")):
                    stats["upper_bound_also_in_sibling_candidate"] += 1
                if _in(token, pools.get("gt")) or _in(row.get("gt_token"), pools.get("gt")):
                    stats["upper_bound_also_in_generation_pool_gt"] += 1
                if flags["excluded_from_conservative"]:
                    continue
                stats["remaining_conservative"] += 1
                if _present(token):
                    conservative_tokens.add(token)
                if _present(scaffold):
                    conservative_scaffolds.add(scaffold)
        files[name] = {
            "rows": stats["rows"],
            "counts": dict(stats),
            "remaining_upper_bound_unique_structures": len(upper_tokens),
            "remaining_upper_bound_scaffolds": len(upper_scaffolds),
            "remaining_conservative_unique_structures": len(conservative_tokens),
            "remaining_conservative_scaffolds": len(conservative_scaffolds),
            "index_sha256": sha256_file_always(index),
            "index_bytes": index.stat().st_size,
            "label": (
                "Not present in the known exposure index. This is not proof that no "
                "upstream model, including Qwen3-32B pretraining, has seen the molecule."
            ),
        }
    return {
        "definition": {
            "remaining_upper_bound": (
                "Row is outside confirmed weight-train structure, exact normalized spectrum, "
                "and candidate-spectrum pair, and outside known eval and test_2000 structures. "
                "Shift-sequence overlap is reported and does not remove the row."
            ),
            "remaining_conservative": "Upper bound, and also outside the unresolved cot candidate file.",
            "not_additive": "The four LMDB files overlap. Do not add their row counts.",
        },
        "files": files,
    }


def _lmdb_split_relations(out: Path) -> dict[str, Any]:
    """Measure whether the official LMDB splits are disjoint after canonicalization."""
    token_sets: dict[str, set[str]] = {}
    disagreement = {}
    for name in ("train.lmdb", "valid.lmdb", "test.lmdb"):
        tokens: set[str] = set()
        rows = token_ne_gt = unparsed = 0
        path = out / f"lmdb_{name}.index.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                token = row.get("candidate_token")
                if _present(token):
                    tokens.add(token)
                gt = row.get("gt_token")
                if _present(gt) and gt != token:
                    token_ne_gt += 1
                if row.get("candidate_status") == "unparsed":
                    unparsed += 1
        token_sets[name] = tokens
        disagreement[name] = {
            "rows": rows,
            "unique_candidate_tokens": len(tokens),
            "candidate_token_differs_from_declared_gt_token": token_ne_gt,
            "unparsed_smiles": unparsed,
        }
    names = list(token_sets)
    intersections = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            both = token_sets[left] & token_sets[right]
            intersections[f"{left}_vs_{right}"] = {
                "left": len(token_sets[left]),
                "right": len(token_sets[right]),
                "intersection": len(both),
            }
    return {
        "within_file": disagreement,
        "split_candidate_intersections": intersections,
        "note": (
            "String inequality of smiles and smiles_canonical is not a different molecule. "
            "candidate_token_differs_from_declared_gt_token is the count after RDKit canonicalization. "
            "Equal row sums do not by themselves prove the splits are disjoint; the intersections do."
        ),
    }


def _extra_intersections(records: list[dict[str, Any]]) -> dict[str, Any]:
    from step_02a_audit import _select
    from step_02a_overlap import intersection_count

    def one(left_asset, left_role, left_field, right_asset, right_role, right_field):
        return intersection_count(
            _select(records, left_asset, left_role, left_field),
            _select(records, right_asset, right_role, right_field),
        )

    pairs = {}
    for asset, role in (
        ("v64_source_split", "weight_train"),
        ("v61_train", "weight_train"),
        ("v5_train", "weight_train"),
    ):
        pairs[f"{asset}_gt_vs_test_gt"] = one(asset, role, "gt_token", "test_2000", "test", "gt_token")
        pairs[f"{asset}_candidate_vs_test_candidate"] = one(
            asset, role, "candidate_token", "test_2000", "test", "candidate_token"
        )
        pairs[f"{asset}_gt_connectivity_vs_test_gt_connectivity"] = one(
            asset, role, "gt_connectivity", "test_2000", "test", "gt_connectivity"
        )
    pairs["v64_train_spectrum_vs_v61_train_spectrum"] = one(
        "v64_source_split", "weight_train", "spectrum_norm_hash", "v61_train", "weight_train", "spectrum_norm_hash"
    )
    pairs["v64_train_spectrum_vs_v5_train_spectrum"] = one(
        "v64_source_split", "weight_train", "spectrum_norm_hash", "v5_train", "weight_train", "spectrum_norm_hash"
    )
    pairs["sibling_v6_candidate_vs_test_gt"] = one(
        "v6_train_sibling", "sibling_train", "candidate_token", "test_2000", "test", "gt_token"
    )
    pairs["cot_train_candidate_vs_test_gt"] = one(
        "cot_train_unknown_ancestor", "unknown_ancestor", "candidate_token", "test_2000", "test", "gt_token"
    )
    return pairs


def _parquet_notes() -> list[dict[str, Any]]:
    paths = [
        Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/data/rl_train_expanded_agent.parquet"),
        Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching/data/rl_train_expanded.parquet"),
        Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching/data/rl_train.parquet"),
    ]
    notes = []
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        parquet = None
    for path in paths:
        if not path.exists():
            notes.append({"path": str(path), "exists": False, "role": "configured_rl_not_in_v64_weight_chain"})
            continue
        item = {
            "path": str(path),
            "exists": True,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "hash_policy": "metadata_only",
            "role": "configured_rl_not_in_v64_weight_chain",
        }
        if parquet is not None:
            meta = parquet.ParquetFile(path).metadata
            item["rows"] = meta.num_rows
            item["columns"] = meta.num_columns
        notes.append(item)
    return notes


def refine_run(out: Path) -> None:
    """Update reports in an existing step-02A directory."""
    from step_02a_audit import (
        compare_observation_files,
        iter_jsonl,
        sha256_file_always,
        user_value,
        v64_roles,
    )
    from step_02a_overlap import summarize_records

    out = Path(out)
    records = _load_jsonl(out / "labeled_index.jsonl")
    overlap_path = out / "overlap_report.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_asset[record["asset"]].append(record)
        by_role[f"{record['asset']}:{record['role']}"].append(record)

    def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
        summary = summarize_records(rows)
        summary["contradictory_pairs"] = dict(list(summary["contradictory_pairs"].items())[:20])
        summary["same_gt_multiple_candidates"] = dict(list(summary["same_gt_multiple_candidates"].items())[:20])
        return summary

    overlap["contradiction_rule"] = (
        "contradictory_pair_groups counts a pair only when spectrum_norm_hash is present "
        "and that same pair token carries more than one label. A text row with no spectrum "
        "is not evidence that two labels refer to one spectrum."
    )
    overlap["by_asset"] = {asset: compact(rows) for asset, rows in by_asset.items()}
    overlap["by_asset_and_role"] = {key: compact(rows) for key, rows in by_role.items()}
    overlap["intersections_by_component"] = _extra_intersections(records)
    overlap_path.write_text(json.dumps(overlap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    roles = v64_roles()
    source = Path("/fs_mol/cxq/SpecMaster_jp/distill_out/trajectories_v6_3_filtered_balanced_observation_fixed.jsonl")
    users = [user_value(row) or "" for _line, row in iter_jsonl(source)]
    if len(users) != len(roles):
        raise RuntimeError("role replay does not match trajectory rows")
    train_users = [text for text, role in zip(users, roles) if role == "weight_train"]
    val_users = [text for text, role in zip(users, roles) if role == "weight_eval"]
    preservation_path = out / "input_preservation_report.json"
    preservation = json.loads(preservation_path.read_text(encoding="utf-8"))
    preservation["alpaca_projection_and_parser"] = _quantify_alpaca(source, roles)
    preservation["registered_v6_4_files"] = _registered_match(train_users, val_users)
    preservation["sample_rule"]["historical_sample"]["rule"] = (
        "Rows are grouped by label, or by 'unknown' when no label is present. "
        "Each bucket is shuffled with random.Random(20261125) and at most 200 indexes are kept. "
        "The sample describes format coverage only."
    )
    preservation["training_user_json_note"] = (
        "The v6.4 training user turn is structured JSON. It already contains J, nH, ranges, "
        "solvent, and one nmr_frequency string. experimental_total_nH is absent from that object. "
        "Loss figures in alpaca_projection_and_parser describe the text formatter and "
        "parse_alpaca_input, not fields missing from the training JSON."
    )
    preservation_path.write_text(json.dumps(preservation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pool_path = out / "candidate_pool_report.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    pool["authoritative_isolation"] = _recount_lmdb(out, records)
    pool["lmdb_split_relations"] = _lmdb_split_relations(out)
    pool["sibling_gt_recovery"] = [
        _join_sibling(
            Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v6_3.json"),
            [
                Path("/fs_mol/cxq/SpecMaster_jp/distill_out/trajectories_v6_3_filtered_balanced_observation_fixed.jsonl"),
                Path("/fs_mol/cxq/SpecMaster_jp/distill_out/trajectories_v6_3.jsonl"),
            ],
        ),
        _join_sibling(
            Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v6_2.json"),
            [Path("/fs_mol/cxq/SpecMaster_jp/distill_out/trajectories_v6_2.jsonl")],
        ),
    ]
    pool["rl_assets_not_in_weight_chain"] = _parquet_notes()
    pool_path.write_text(json.dumps(pool, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lineage_path = out / "lineage_report.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["observation_fixed_comparison"] = compare_observation_files()
    lineage_path.write_text(json.dumps(lineage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_path = out / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_hashes"] = {
        "labeled_index.jsonl": sha256_file_always(out / "labeled_index.jsonl"),
        "lmdb_all.lmdb.index.jsonl": sha256_file_always(out / "lmdb_all.lmdb.index.jsonl"),
        "lmdb_train.lmdb.index.jsonl": sha256_file_always(out / "lmdb_train.lmdb.index.jsonl"),
        "lmdb_valid.lmdb.index.jsonl": sha256_file_always(out / "lmdb_valid.lmdb.index.jsonl"),
        "lmdb_test.lmdb.index.jsonl": sha256_file_always(out / "lmdb_test.lmdb.index.jsonl"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pool["first_pass_note"] = (
        "lmdb.files is the counter written while scanning. authoritative_isolation is the "
        "definition used for the conclusion: exact structure, spectrum, and pair hits are "
        "exclusions, a shift-sequence hit is not, and test_2000 exclusion uses source-group "
        "GT rather than every negative candidate."
    )
    pool_path.write_text(json.dumps(pool, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    contract_path = out / "input_contract_draft.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["formatter_note"] = (
            "scripts/build_test_2000.py format_alpaca writes one nmr_frequency string into both "
            "the 13C and 1H headers and rounds carbon shifts to 0.1 ppm and proton shifts to 0.01 ppm. "
            "parse_alpaca_input does not return solvent, frequency, J, category, or ranges."
        )
        contract["interfaces"] = {
            "training_user_json": "v6.4 trajectory user content: candidate_smiles, c_nmr_peaks, h_nmr_peaks, nmr_solvent, nmr_frequency",
            "api": "recipe/nmr_matching_agent/api_service.py MatchRequest",
            "v5_vllm": "recipe/nmr_matching_agent/call_v5_vllm.py run_case omits experimental_total_nH; parse_h_peaks_arg sets nH to None when the token has no parenthesis",
            "task_object": "recipe/nmr_matching_agent/task_input.py TASK_FIELDS",
            "rl_parquet": "recipe/nmr_matching_agent/build_agent_parquet.py parses Alpaca text, then may overwrite candidate_smiles from reward ground truth. Not in the v6.4 weight chain.",
        }
        contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[refine] wrote authoritative isolation and input projection", flush=True)


def _peak_reasons(train_spec: Mapping[str, Any], lmdb_spec: Mapping[str, Any]) -> tuple[str, ...]:
    reasons = []
    if train_spec.get("nmr_solvent") != lmdb_spec.get("nmr_solvent"):
        reasons.append("solvent")
    if train_spec.get("nmr_frequency") != lmdb_spec.get("nmr_frequency"):
        reasons.append("frequency")
    train_c = train_spec.get("c_nmr_peaks") or []
    lmdb_c = lmdb_spec.get("c_nmr_peaks") or []
    if train_c != lmdb_c:
        reasons.append("c_peaks")
    train_h = train_spec.get("h_nmr_peaks") or []
    lmdb_h = lmdb_spec.get("h_nmr_peaks") or []
    if train_h != lmdb_h:
        reasons.append("h_peaks")
        if len(train_h) != len(lmdb_h):
            reasons.append("h_length")
        else:
            flags = set()
            for left, right in zip(train_h, lmdb_h):
                if not isinstance(left, dict) or not isinstance(right, dict):
                    flags.add("h_not_object")
                    continue
                if left.get("j_values") != right.get("j_values"):
                    flags.add("h_j")
                if left.get("nH") != right.get("nH"):
                    flags.add("h_nH")
                if left.get("category") != right.get("category"):
                    flags.add("h_category")
                if left.get("rangeMin") != right.get("rangeMin") or left.get("rangeMax") != right.get("rangeMax"):
                    flags.add("h_range")
                if left.get("centroid") != right.get("centroid") or left.get("delta") != right.get("delta"):
                    flags.add("h_shift_field")
            reasons.extend(sorted(flags))
    return tuple(reasons)


def explain_spectrum_gap(out: Path) -> None:
    """Compare confirmed training spectra with LMDB test records that share a shift list."""
    import pickle

    import lmdb

    from step_02a_audit import human_value, iter_jsonl, load_json, parse_task_text, user_value, v64_roles
    from step_02a_fingerprints import shift_sequence_hash, spectrum_views

    by_shift: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact = set()

    def remember(spectrum: Mapping[str, Any] | None) -> None:
        if not spectrum:
            return
        viewed = spectrum_views(spectrum)
        shift = shift_sequence_hash(spectrum)
        if not shift or not viewed.get("normalized_hash"):
            return
        exact.add(viewed["normalized_hash"])
        by_shift[shift].append({
            "c_nmr_peaks": spectrum.get("c_nmr_peaks"),
            "h_nmr_peaks": spectrum.get("h_nmr_peaks"),
            "nmr_solvent": spectrum.get("nmr_solvent"),
            "nmr_frequency": spectrum.get("nmr_frequency"),
        })

    source = Path("/fs_mol/cxq/SpecMaster_jp/distill_out/trajectories_v6_3_filtered_balanced_observation_fixed.jsonl")
    for (_line, row), role in zip(iter_jsonl(source), v64_roles()):
        if role == "weight_train":
            remember(parse_task_text(user_value(row))["spectrum"])
    factory = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data")
    for name in ("nmr_matching_tool_sft_v6_1.json", "nmr_matching_tool_sft_v5.json"):
        for row in load_json(factory / name):
            remember(parse_task_text(human_value(row))["spectrum"])

    reasons: Counter[tuple[str, ...]] = Counter()
    exact_rows = shift_rows = unexplained = 0
    path = Path("/fs_mol/fjxu/proj_data/Spectra/NMR/si_0820/lmdb_dataset/test.lmdb")
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(buffers=True) as txn:
        for _key, value in txn.cursor():
            raw = pickle.loads(bytes(value))
            spectrum = {
                field: raw.get(field)
                for field in ("c_nmr_peaks", "h_nmr_peaks", "nmr_solvent", "nmr_frequency")
                if field in raw
            }
            viewed = spectrum_views(spectrum)
            shift = shift_sequence_hash(spectrum)
            if viewed.get("normalized_hash") in exact:
                exact_rows += 1
            if shift not in by_shift:
                continue
            shift_rows += 1
            if viewed.get("normalized_hash") in exact:
                continue
            matched = False
            for train_spec in by_shift[shift]:
                reason = _peak_reasons(train_spec, spectrum)
                if reason:
                    reasons[reason] += 1
                    matched = True
                    break
            if matched:
                continue
            kind = []
            train_spec = by_shift[shift][0]
            for field in ("c_nmr_peaks", "h_nmr_peaks"):
                left = train_spec.get(field)
                right = spectrum.get(field)
                if left == [] and right is None:
                    kind.append(f"{field}:training_empty_list_vs_lmdb_null")
                elif left is None and right == []:
                    kind.append(f"{field}:training_null_vs_lmdb_empty_list")
            if kind:
                reasons[tuple(kind)] += 1
            else:
                unexplained += 1
    env.close()
    payload = {
        "lmdb": str(path),
        "training_exact_spectra": len(exact),
        "training_shift_sequences": len(by_shift),
        "test_lmdb_rows_exact_spectrum": exact_rows,
        "test_lmdb_rows_shift_sequence": shift_rows,
        "shift_only_reason_counts": {",".join(key): value for key, value in reasons.most_common()},
        "shift_only_without_a_field_difference": unexplained,
        "note": (
            "A shift-sequence hit is not an identical spectrum. h_category/h_j/h_range means "
            "the stored training peak object lacks category, J, or range values that the LMDB "
            "record still has. training_empty_list_vs_lmdb_null means one side has an empty "
            "peak list and the other has null; those are not treated as the same spectrum."
        ),
    }
    (Path(out) / "spectrum_gap.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[gap] wrote spectrum_gap.json", flush=True)
