#!/usr/bin/env python3
"""Build the step-02B-1 exposure and source-eligibility reports.

Reuses the step-02A compact indexes after checking their SHA-256 values.
It does not canonicalize the LMDB with RDKit again, does not overwrite the
02A directory, and does not write a replacement training set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

RSI = Path(__file__).resolve().parent
SPEC = Path("/fs_mol/cxq/SpecMaster_jp")
OA = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
LF = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory")
AUDIT_02A = SPEC / "rsi_runs/step_02a_20261125T130404+0800"
sys.path.insert(0, str(RSI))
sys.path.insert(0, str(OA))

from nmr_lossless_task import (  # noqa: E402
    SCHEMA_VERSION,
    normalize_task_input,
    parse_task_text as parse_contract_text,
    task_to_lossless_text,
)
from recipe.nmr_matching_agent.task_input import (  # noqa: E402
    _CAND_RE,
    _HEADER_RE,
    _split_header_parenthetical,
    parse_c_peak_list,
    parse_h_peak_list,
)
from step_02a_audit import (  # noqa: E402
    LMDB_ROOT,
    human_value,
    iter_jsonl,
    load_json,
    parse_task_text as parse_stored_text,
    user_value,
    v64_roles,
)
from step_02a_fingerprints import identity_token, shift_sequence_hash, spectrum_views, structure_keys  # noqa: E402
from step_02a_refine import _role_sets, classify_source_row  # noqa: E402
from step_02b1_eligibility import (  # noqa: E402
    PROTOCOL_VERSION,
    exposure_reasons,
    quality_reasons,
    reason_family_counts,
    spectrum_difference_labels,
)

EXPECTED_BASELINE = {"train.lmdb": 917357, "valid.lmdb": 48292, "test.lmdb": 104796}
EXPECTED_NEW = {"train.lmdb": 820, "valid.lmdb": 49, "test.lmdb": 88}
EXPECTED_REMAINING = {"train.lmdb": 916537, "valid.lmdb": 48243, "test.lmdb": 104708}
SPLIT_FILES = ("train.lmdb", "valid.lmdb", "test.lmdb")
PRESERVED = [
    "recipe/demystify/grpo_tcr_qwen2_7b.sh",
    "recipe/demystify/qwen2_7b_sft.sh",
    "verl/experimental/agent_loop/agent_loop.py",
    "verl/experimental/agent_loop/tool_agent_loop.py",
    "verl/model_merger/base_model_merger.py",
    "verl/model_merger/fsdp_model_merger.py",
    "verl/protocol.py",
    "verl/trainer/config/_generated_ppo_trainer.yaml",
    "verl/trainer/config/rollout/rollout.yaml",
    "verl/trainer/main_ppo.py",
    "verl/trainer/ppo/core_algos.py",
    "verl/trainer/ppo/metric_utils.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/utils/dataset/multiturn_sft_dataset.py",
    "verl/workers/config/rollout.py",
    "verl/workers/fsdp_workers.py",
    "verl/workers/reward_manager/__init__.py",
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_indexes() -> dict[str, Any]:
    manifest = json.loads((AUDIT_02A / "asset_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["index_hashes"]
    checked = {}
    mismatches = []
    for name, digest in expected.items():
        path = AUDIT_02A / name
        actual = sha256_file(path)
        checked[name] = {"expected": digest, "actual": actual, "match": actual == digest, "bytes": path.stat().st_size}
        if actual != digest:
            mismatches.append(name)
    return {"source": str(AUDIT_02A), "hashes": checked, "mismatches": mismatches}


def _tokens(records: Iterable[Mapping[str, Any]], predicate) -> dict[str, set[str]]:
    selected = [record for record in records if predicate(record)]
    packed = _role_sets(selected, {record.get("role") for record in selected} or {"__none__"})
    return packed


def _combined(packed: Mapping[str, set[str]]) -> set[str]:
    return set(packed.get("candidate") or set()) | set(packed.get("gt") or set())


def build_exposure(records: list[dict[str, Any]]) -> dict[str, Any]:
    confirmed_rows = [row for row in records if row.get("asset") == "v64_source_split" and row.get("role") == "weight_train"]
    inferred_rows = [row for row in records if row.get("asset") in {"v61_train", "v5_train"} and row.get("role") == "weight_train"]
    other_rows = [
        row for row in records
        if row.get("role") == "weight_train" and row.get("asset") not in {"v64_source_split", "v61_train", "v5_train"}
    ]
    union_rows = [row for row in records if row.get("role") == "weight_train"]
    eval_rows = [row for row in records if row.get("role") == "weight_eval"]
    test_rows = [row for row in records if row.get("asset") == "test_2000" and row.get("role") == "test"]
    unknown_rows = [row for row in records if row.get("role") == "unknown_ancestor"]
    confirmed = _role_sets(confirmed_rows, {"weight_train"})
    inferred = _role_sets(inferred_rows, {"weight_train"})
    other = _role_sets(other_rows, {"weight_train"})
    union = _role_sets(union_rows, {"weight_train"})
    evaluation = _role_sets(eval_rows, {"weight_eval"})
    test = _role_sets(test_rows, {"test"})
    unknown = _role_sets(unknown_rows, {"unknown_ancestor"})
    return {
        "records": {
            "confirmed_rows": len(confirmed_rows),
            "inferred_rows": len(inferred_rows),
            "other_weight_train_rows": len(other_rows),
            "other_weight_train_assets": sorted({row.get("asset") for row in other_rows}),
            "union_rows": len(union_rows),
            "eval_rows": len(eval_rows),
            "test_rows": len(test_rows),
            "unknown_rows": len(unknown_rows),
        },
        "packed": {
            "confirmed": confirmed,
            "inferred": inferred,
            "other": other,
            "union": union,
            "evaluation": evaluation,
            "test": test,
            "unknown": unknown,
        },
    }


def fingerprint_test_keys() -> dict[str, Any]:
    path = SPEC / "data/nmr_matching_test_2000.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    text_tokens: set[str] = set()
    declared_tokens: set[str] = set()
    gt_tokens: set[str] = set()
    text_declared_differ = 0
    for row in rows:
        text = row.get("input") if isinstance(row.get("input"), str) else ""
        match = re.search(r"Candidate SMILES:\s*(\S+)", text or "")
        text_value = match.group(1) if match else None
        declared = row.get("_candidate_smiles") if isinstance(row.get("_candidate_smiles"), str) else None
        canonical = row.get("smiles_canonical") if isinstance(row.get("smiles_canonical"), str) else None
        gt = row.get("_gt_smiles") if isinstance(row.get("_gt_smiles"), str) else None
        text_token = identity_token(structure_keys(text_value)) if text_value else None
        declared_token = identity_token(structure_keys(declared)) if declared else None
        canonical_token = identity_token(structure_keys(canonical)) if canonical else None
        gt_token = identity_token(structure_keys(gt)) if gt else None
        if text_token:
            text_tokens.add(text_token)
        if declared_token:
            declared_tokens.add(declared_token)
        if canonical_token:
            declared_tokens.add(canonical_token)
        if gt_token:
            gt_tokens.add(gt_token)
        if text_token and declared_token and text_token != declared_token:
            text_declared_differ += 1
    return {
        "rows": len(rows),
        "text_candidate_tokens": text_tokens,
        "declared_candidate_tokens": declared_tokens,
        "gt_tokens": gt_tokens,
        "text_and_declared_token_disagreements": text_declared_differ,
        "declared_canonical_field_present": any(isinstance(row.get("smiles_canonical"), str) for row in rows),
    }


def _view(packed: Mapping[str, set[str]]) -> dict[str, set[str]]:
    return {
        "structure": _combined(packed),
        "spectrum": set(packed.get("spectrum") or set()),
        "pair": set(packed.get("pair") or set()),
        "shift": set(packed.get("shift_sequence") or set()),
    }


def prepare_views(exposure: Mapping[str, Any], test_keys: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the exposure sets once. The LMDB scan must not rebuild them per row."""
    packed = exposure["packed"]
    return {
        "confirmed": _view(packed["confirmed"]),
        "inferred": _view(packed["inferred"]),
        "other": _view(packed["other"]),
        "evaluation": _view(packed["evaluation"]),
        "union": packed["union"],
        "evaluation_packed": packed["evaluation"],
        "test_packed": packed["test"],
        "unknown_packed": packed["unknown"],
        "test_gt": set(packed["test"].get("gt") or set()) | set(test_keys["gt_tokens"]),
        "test_candidate": (
            set(packed["test"].get("candidate") or set())
            | set(test_keys["text_candidate_tokens"])
            | set(test_keys["declared_candidate_tokens"])
        ),
        "unknown": _combined(packed["unknown"]),
    }


def classify_index_row(row: Mapping[str, Any], views: Mapping[str, Any]) -> dict[str, Any]:
    reasons = exposure_reasons(
        row,
        confirmed=views["confirmed"],
        inferred=views["inferred"],
        evaluation=views["evaluation"],
        test_gt=views["test_gt"],
        test_candidate=views["test_candidate"],
        unknown=views["unknown"],
    )
    other = views["other"]
    if _structure_extra(row, other["structure"]):
        reasons.append("other_weight_train_structure")
    if row.get("spectrum_norm_hash") in other["spectrum"]:
        reasons.append("other_weight_train_spectrum")
    if row.get("pair_token") in other["pair"]:
        reasons.append("other_weight_train_pair")
    flags = classify_source_row(
        row, views["union"], views["evaluation_packed"], views["test_packed"], views["unknown_packed"]
    )
    candidate_token = str(row.get("candidate_token"))
    gt_token = str(row.get("gt_token"))
    index_quality = []
    if candidate_token.startswith("unparsed:") or row.get("candidate_status") == "unparsed":
        index_quality.append("unparsed_structure")
    if candidate_token in {"missing", "None", ""} or row.get("candidate_status") == "missing":
        index_quality.append("missing_structure")
    if candidate_token.startswith("canonical:") and gt_token.startswith("canonical:") and candidate_token != gt_token:
        index_quality.append("gold_unresolved")
    return {"reasons": reasons, "baseline": flags, "index_quality": index_quality}


def _structure_extra(row: Mapping[str, Any], tokens: set[str]) -> bool:
    return bool(tokens) and (
        row.get("candidate_token") in tokens or row.get("gt_token") in tokens
    )


def sequential_bucket(reasons: list[str]) -> str:
    if any(reason.startswith("confirmed_") for reason in reasons):
        return "confirmed"
    if any(reason.startswith("inferred_") or reason.startswith("other_weight_train_") for reason in reasons):
        return "inferred_or_other_weight_train"
    if "known_eval_structure" in reasons:
        return "known_eval"
    if "test_gt_structure" in reasons:
        return "test_gt"
    if "unknown_possible_structure" in reasons:
        return "unknown"
    if "test_candidate_structure" in reasons:
        return "test_candidate"
    return "remaining"


def inspect_quality(raw: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    c_peaks = raw.get("c_nmr_peaks") if "c_nmr_peaks" in raw else None
    h_peaks = raw.get("h_nmr_peaks") if "h_nmr_peaks" in raw else None

    def finite(value: Any) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, list):
            return all(finite(item) for item in value)
        return True

    def signal(peaks: Any) -> bool:
        if not isinstance(peaks, list):
            return False
        for peak in peaks:
            if isinstance(peak, bool):
                continue
            if isinstance(peak, (int, float)):
                return not isinstance(peak, float) or math.isfinite(peak)
            if isinstance(peak, dict):
                for key in ("delta (ppm)", "delta", "centroid", "rangeMin", "rangeMax"):
                    number = peak.get(key)
                    if isinstance(number, bool):
                        continue
                    if isinstance(number, (int, float)) and (not isinstance(number, float) or math.isfinite(number)):
                        return True
        return False

    missing_nh = null_nh = known_nh = 0
    if isinstance(h_peaks, list):
        for peak in h_peaks:
            if not isinstance(peak, dict) or "nH" not in peak:
                missing_nh += 1
            elif peak.get("nH") is None:
                null_nh += 1
            else:
                known_nh += 1
    c_signal = signal(c_peaks)
    h_signal = signal(h_peaks)
    candidate_status = row.get("candidate_status")
    gt_token = row.get("gt_token")
    candidate_token = row.get("candidate_token")
    flags = {
        "unparsed_structure": candidate_status == "unparsed" or str(candidate_token).startswith("unparsed:"),
        "missing_structure": candidate_status == "missing" or candidate_token in (None, "", "missing"),
        "no_valid_spectrum": not c_signal and not h_signal,
        "non_finite_value": not finite({"c": c_peaks, "h": h_peaks, "frequency": raw.get("nmr_frequency")}),
        "gold_unresolved": (
            str(candidate_token).startswith("canonical:")
            and str(gt_token).startswith("canonical:")
            and candidate_token != gt_token
        ),
    }
    return {
        "reasons": quality_reasons(flags),
        "coverage": {
            "c_signal": c_signal,
            "h_signal": h_signal,
            "both_nuclei": c_signal and h_signal,
            "one_nucleus_only": c_signal != h_signal,
            "missing_nH_peaks": missing_nh,
            "null_nH_peaks": null_nh,
            "known_nH_peaks": known_nh,
            "frequency_present": isinstance(raw.get("nmr_frequency"), str) and bool(raw.get("nmr_frequency")),
            "frequency_unknown_assignment": True,
        },
    }


def scan_splits(exposure: Mapping[str, Any], test_keys: Mapping[str, Any], out: Path, write_index: bool, with_quality: bool) -> dict[str, Any]:
    import pickle
    import lmdb

    training = load_training_spectra() if with_quality else None
    views = exposure["views"]
    index_path = out / "source_eligibility_index.jsonl"
    handle = index_path.open("w", encoding="utf-8") if write_index else None
    per_file = {}
    family = Counter()
    spectrum_first: dict[str, str] = {}
    shift_first: dict[str, str] = {}
    spectrum_cross = 0
    shift_cross = 0
    try:
        for name in SPLIT_FILES:
            started = time.time()
            stats: Counter[str] = Counter()
            reason_counts: Counter[str] = Counter()
            coverage: Counter[str] = Counter()
            gap = Counter()
            env = None
            txn = None
            cursor = None
            if with_quality:
                env = lmdb.open(str(LMDB_ROOT / name), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
                txn = env.begin(buffers=True)
                cursor = txn.cursor()
                cursor_iter = iter(cursor)
            else:
                cursor_iter = None
            with (AUDIT_02A / f"lmdb_{name}.index.jsonl").open(encoding="utf-8") as index:
                for line in index:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    stats["rows"] += 1
                    classified = classify_index_row(row, views)
                    reasons = classified["reasons"]
                    for quality_name in classified["index_quality"]:
                        stats[f"index_quality_{quality_name}"] += 1
                    bucket = sequential_bucket(reasons)
                    stats[f"bucket_{bucket}"] += 1
                    for reason in reasons:
                        reason_counts[reason] += 1
                    baseline = classified["baseline"]["excluded_from_conservative"]
                    my_baseline = bucket not in {"remaining", "test_candidate"}
                    if baseline != my_baseline:
                        stats["baseline_mismatch"] += 1
                    if baseline:
                        stats["baseline_excluded"] += 1
                    else:
                        stats["baseline_remaining"] += 1
                    quality = {"reasons": [], "coverage": {}}
                    raw = None
                    if with_quality and cursor_iter is not None:
                        try:
                            key, value = next(cursor_iter)
                        except StopIteration as exc:
                            raise RuntimeError(f"{name} cursor ended before the index") from exc
                        locator_key = row["locator"].split(":", 1)[1]
                        if bytes(key).decode("utf-8", "replace") != locator_key:
                            raise RuntimeError(f"{name} index order does not match the LMDB cursor at {row['locator']}")
                        raw = pickle.loads(bytes(value))
                        quality = inspect_quality(raw, row)
                        for reason in quality["reasons"]:
                            stats[f"quality_{reason}"] += 1
                        cov = quality["coverage"]
                        if cov.get("both_nuclei"):
                            coverage["both_nuclei"] += 1
                        if cov.get("one_nucleus_only"):
                            coverage["one_nucleus_only"] += 1
                        if cov.get("frequency_present"):
                            coverage["frequency_present"] += 1
                        coverage["missing_nH_peaks"] += cov.get("missing_nH_peaks", 0)
                        coverage["null_nH_peaks"] += cov.get("null_nH_peaks", 0)
                        if name == "test.lmdb" and training is not None:
                            _accumulate_gap(raw, training, gap)
                    structure_reasons = [reason for reason in reasons if reason.endswith("_structure")]
                    spectrum_reasons = [reason for reason in reasons if reason not in structure_reasons]
                    if bucket == "remaining" and not quality["reasons"]:
                        stats["auto_positive_pool"] += 1
                    quality_on = bool(quality["reasons"])
                    structure_on = bool(structure_reasons)
                    spectrum_on = bool(spectrum_reasons)
                    family_key = (
                        ("quality" if quality_on else "")
                        + ("+structure" if structure_on else "")
                        + ("+spectrum" if spectrum_on else "")
                    ).lstrip("+") or "none"
                    family[family_key] += 1
                    spectrum = row.get("spectrum_norm_hash")
                    shift = row.get("shift_sequence_hash")
                    if spectrum:
                        previous = spectrum_first.get(spectrum)
                        if previous is None:
                            spectrum_first[spectrum] = name
                        elif previous != name and previous != "multi":
                            spectrum_cross += 1
                            spectrum_first[spectrum] = "multi"
                    if shift:
                        previous_shift = shift_first.get(shift)
                        if previous_shift is None:
                            shift_first[shift] = name
                        elif previous_shift != name and previous_shift != "multi":
                            shift_cross += 1
                            shift_first[shift] = "multi"
                    if handle is not None:
                        source_name, source_key = row["locator"].split(":", 1)
                        handle.write(json.dumps({
                            "protocol_version": PROTOCOL_VERSION,
                            "source_lmdb": source_name,
                            "source_key": source_key,
                            "locator": row["locator"],
                            "candidate_token": row.get("candidate_token"),
                            "gt_token": row.get("gt_token"),
                            "spectrum_norm_hash": spectrum,
                            "shift_sequence_hash": shift,
                            "exposure_reasons": reasons,
                            "evidence_bucket": bucket,
                            "quality_reasons": quality["reasons"],
                            "quality_status": "rejected" if quality["reasons"] else "not_rejected",
                            "in_auto_positive_pool": bucket == "remaining" and not quality["reasons"],
                            "source_index": str(AUDIT_02A / f"lmdb_{name}.index.jsonl"),
                            "source_index_sha256": exposure["index_hashes"][f"lmdb_{name}.index.jsonl"],
                        }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if stats["rows"] % 100000 == 0:
                        print(f"[scan] {name} {stats['rows']}", flush=True)
            if env is not None:
                env.close()
            per_file[name] = {
                "rows": stats["rows"],
                "buckets": {key.removeprefix("bucket_"): value for key, value in stats.items() if key.startswith("bucket_")},
                "reason_counts": dict(reason_counts),
                "baseline_remaining": stats["baseline_remaining"],
                "baseline_excluded": stats["baseline_excluded"],
                "baseline_mismatch": stats["baseline_mismatch"],
                "auto_positive_pool": stats["auto_positive_pool"],
                "quality": {key.removeprefix("quality_"): value for key, value in stats.items() if key.startswith("quality_")},
                "index_quality": {key.removeprefix("index_quality_"): value for key, value in stats.items() if key.startswith("index_quality_")},
                "coverage": dict(coverage),
                "elapsed_sec": round(time.time() - started, 1),
            }
            if name == "test.lmdb" and with_quality:
                per_file[name]["shift_only_gap"] = _gap_summary(gap)
            print(f"[scan] finished {name} rows={stats['rows']} remaining={stats['bucket_remaining']} new={stats['bucket_test_candidate']}", flush=True)
    finally:
        if handle is not None:
            handle.close()
    summary = {
        "files": per_file,
        "reason_families": _family_from_counter(family),
        "wording": "未出现在所列已知/推断暴露与评测索引中。这不是从未见过，也不覆盖上游预训练语料。",
        "index_path": str(index_path) if write_index else None,
        "cross_split_exact_spectra": spectrum_cross,
        "cross_split_shift_sequences": shift_cross,
        "cross_split_note": "A shift-sequence shared by two splits is not counted as an identical spectrum.",
    }
    if write_index:
        summary["index_sha256"] = sha256_file(index_path)
        summary["index_bytes"] = index_path.stat().st_size
    return summary


def _family_from_counter(family: Counter) -> dict[str, Any]:
    quality_only = family["quality"]
    structure_only = family["structure"]
    spectrum_only = family["spectrum"]
    quality_structure = family["quality+structure"]
    quality_spectrum = family["quality+spectrum"]
    structure_spectrum = family["structure+spectrum"]
    all_three = family["quality+structure+spectrum"]
    total = sum(family.values())
    return {
        "rows": total,
        "none": family["none"],
        "quality_independent": quality_only + quality_structure + quality_spectrum + all_three,
        "structure_independent": structure_only + quality_structure + structure_spectrum + all_three,
        "spectrum_independent": spectrum_only + quality_spectrum + structure_spectrum + all_three,
        "quality_and_structure": quality_structure + all_three,
        "quality_and_spectrum": quality_spectrum + all_three,
        "structure_and_spectrum": structure_spectrum + all_three,
        "all_three": all_three,
        "union_removed": total - family["none"],
        "combinations": dict(family),
    }


def _structure_hit_view(row, view) -> bool:
    return False


def load_training_spectra() -> dict[str, list[dict[str, Any]]]:
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
            "c_nmr_peaks": spectrum.get("c_nmr_peaks") if "c_nmr_peaks" in spectrum else None,
            "h_nmr_peaks": spectrum.get("h_nmr_peaks") if "h_nmr_peaks" in spectrum else None,
            "nmr_solvent": spectrum.get("nmr_solvent") if "nmr_solvent" in spectrum else None,
            "nmr_frequency": spectrum.get("nmr_frequency") if "nmr_frequency" in spectrum else None,
            "_present": {
                "c_nmr_peaks": "c_nmr_peaks" in spectrum,
                "h_nmr_peaks": "h_nmr_peaks" in spectrum,
            },
        })

    source = SPEC / "distill_out/trajectories_v6_3_filtered_balanced_observation_fixed.jsonl"
    for (_line, row), role in zip(iter_jsonl(source), v64_roles()):
        if role == "weight_train":
            remember(parse_stored_text(user_value(row)).get("spectrum"))
    for name in ("nmr_matching_tool_sft_v6_1.json", "nmr_matching_tool_sft_v5.json"):
        for row in load_json(LF / "data" / name):
            remember(parse_stored_text(human_value(row)).get("spectrum"))
    return {"by_shift": by_shift, "exact": exact}


def _labels_for_stored(train_spec: Mapping[str, Any], raw: Mapping[str, Any]) -> tuple[str, ...]:
    left = {}
    right = {}
    if train_spec["_present"]["c_nmr_peaks"]:
        left["c_nmr_peaks"] = train_spec.get("c_nmr_peaks")
    if train_spec["_present"]["h_nmr_peaks"]:
        left["h_nmr_peaks"] = train_spec.get("h_nmr_peaks")
    if "c_nmr_peaks" in raw:
        right["c_nmr_peaks"] = raw.get("c_nmr_peaks")
    if "h_nmr_peaks" in raw:
        right["h_nmr_peaks"] = raw.get("h_nmr_peaks")
    return spectrum_difference_labels(left, right)


def _accumulate_gap(raw: Mapping[str, Any], training: Mapping[str, Any], gap: Counter) -> None:
    spectrum = {
        field: raw.get(field)
        for field in ("c_nmr_peaks", "h_nmr_peaks", "nmr_solvent", "nmr_frequency")
        if field in raw
    }
    viewed = spectrum_views(spectrum)
    shift = shift_sequence_hash(spectrum)
    if viewed.get("normalized_hash") in training["exact"]:
        gap["exact_rows"] += 1
        return
    specs = training["by_shift"].get(shift or "")
    if not specs:
        return
    gap["shift_only_rows"] += 1
    labels: set[str] = set()
    for spec in specs:
        labels.update(_labels_for_stored(spec, raw))
    gap[("labels", tuple(sorted(labels)))] += 1


def _gap_summary(gap: Counter) -> dict[str, Any]:
    combinations = {}
    for key, value in gap.items():
        if isinstance(key, tuple) and key and key[0] == "labels":
            combinations[",".join(key[1]) or "no_label"] = value
    return {
        "exact_rows_skipped": gap["exact_rows"],
        "shift_only_rows": gap["shift_only_rows"],
        "multi_label_combinations": dict(sorted(combinations.items(), key=lambda item: -item[1])),
        "note": (
            "Labels may co-occur. A shift-only row is not an identical spectrum. "
            "The earlier 946 figure combined hydrogen-field gaps that also had an empty-list versus null carbon list."
        ),
    }


def _field_status(source_present: bool, equal: bool, failed: bool) -> str:
    if failed:
        return "fail"
    if not source_present:
        return "missing"
    return "kept" if equal else "changed"


def historical_samples() -> dict[str, Any]:
    files = [
        (LF / "data/nmr_matching_cot_train.json", "json"),
        (LF / "data/nmr_matching_tool_sft_v6.json", "json"),
        (SPEC / "distill_out/trajectories_v6_1.jsonl", "jsonl"),
    ]
    report = {}
    for path, kind in files:
        if not path.exists():
            report[str(path)] = {"exists": False}
            continue
        rows = []
        if kind == "jsonl":
            for line_number, row in iter_jsonl(path):
                rows.append((line_number, row))
        else:
            loaded = load_json(path)
            rows = list(enumerate(loaded, 1))
        buckets: dict[str, list[int]] = defaultdict(list)
        for ordinal, (_line, row) in enumerate(rows, 1):
            label = None
            if isinstance(row, dict):
                meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
                label = row.get("_label", row.get("label", meta.get("label")))
            buckets["unknown" if label is None else str(label)].append(ordinal)
        rng = random.Random(20261125)
        chosen = []
        for indexes in buckets.values():
            rng.shuffle(indexes)
            chosen.extend(sorted(indexes[:200]))
        chosen_set = set(chosen)
        samples = []
        totals: dict[str, Counter] = defaultdict(Counter)
        for ordinal, (line_number, row) in enumerate(rows, 1):
            if ordinal not in chosen_set:
                continue
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
            text = user_value(row) or human_value(row) or (row.get("input") if isinstance(row, dict) else None)
            stored = parse_stored_text(text if isinstance(text, str) else None)
            sample = {
                "ordinal": ordinal,
                "source_line": line_number,
                "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                "stored_format": stored["format"],
            }
            if stored["format"] == "structured_json" and isinstance(stored.get("task"), dict):
                sample["fields"] = _structured_fields(stored["task"], totals)
            elif isinstance(text, str):
                sample["fields"] = _text_fields(text, totals)
            else:
                sample["fields"] = {"parse": "fail"}
                totals["parse"]["fail"] += 1
            samples.append(sample)
        report[str(path)] = {
            "rows": len(rows),
            "sampled": len(samples),
            "buckets": {key: len(value) for key, value in buckets.items()},
            "rule": "random.Random(20261125); group by label or unknown; at most 200 per bucket; ordinal is the non-empty row number used by the 02A sampler",
            "field_counts": {key: dict(value) for key, value in totals.items()},
            "samples": samples,
        }
    return report


def _add(totals: dict[str, Counter], field: str, status: str) -> str:
    totals[field][status] += 1
    totals[field]["denominator"] += 1
    return status


def _structured_fields(task: Mapping[str, Any], totals: dict[str, Counter]) -> dict[str, str]:
    try:
        normalized = normalize_task_input(task)
        again, _diagnostics = parse_contract_text(task_to_lossless_text(task))
        failed = False
    except Exception as exc:
        return {"parse": _add(totals, "parse", "fail"), "error": str(exc)}
    result = {"parse": _add(totals, "parse", "kept")}
    same = again == normalized
    result["candidate"] = _add(totals, "candidate", _field_status(True, normalized["candidate_smiles"] == again["candidate_smiles"], failed))
    result["c_peaks"] = _add(totals, "c_peaks", _field_status(True, normalized["c_nmr_peaks"] == again["c_nmr_peaks"], failed))
    result["h_peaks"] = _add(totals, "h_peaks", _field_status(True, normalized["h_nmr_peaks"] == again["h_nmr_peaks"], failed))
    result["nH"] = _add(totals, "nH", _peak_key_status(normalized["h_nmr_peaks"], "nH", same))
    result["category"] = _add(totals, "category", _peak_key_status(normalized["h_nmr_peaks"], "category", same))
    result["j"] = _add(totals, "j", _peak_key_status(normalized["h_nmr_peaks"], "j_values", same))
    result["ranges"] = _add(totals, "ranges", _peak_key_status(normalized["h_nmr_peaks"], "rangeMin", same))
    for field in ("nmr_solvent", "nmr_frequency", "experimental_total_nH"):
        present = field in task and task.get(field) is not None
        result[field] = _add(totals, field, _field_status(present, normalized.get(field) == again.get(field), failed))
    result["frequency_nucleus"] = _add(
        totals,
        "frequency_nucleus",
        "kept" if normalized.get("nmr_frequency_nucleus", "unknown") == again.get("nmr_frequency_nucleus", "unknown") else "changed",
    )
    return result


def _peak_key_status(peaks: Any, key: str, roundtrip_same: bool) -> str:
    present = False
    if isinstance(peaks, list):
        for peak in peaks:
            if isinstance(peak, dict) and peak.get(key) is not None:
                present = True
                break
    if not present:
        return "missing"
    return "kept" if roundtrip_same else "changed"


def _text_fields(text: str, totals: dict[str, Counter]) -> dict[str, str]:
    legacy = _legacy_parse(text)
    try:
        parsed, diagnostics = parse_contract_text(text)
    except Exception as exc:
        totals["parse"]["fail"] += 1
        totals["parse"]["denominator"] += 1
        return {"parse": "fail", "error": str(exc), "legacy_candidate": legacy.get("candidate_smiles")}
    result = {"parse": _add(totals, "parse", "kept"), "format": diagnostics.get("format")}
    sections = _alpaca_sections(text)
    result["candidate"] = _add(totals, "candidate", _explicit_match(
        sections["candidate"] is not None,
        parsed.get("candidate_smiles") == sections["candidate"],
    ))
    result["c_peaks"] = _add(totals, "c_peaks", _list_status(sections["carbon_body"], parsed.get("c_nmr_peaks"), parse_c_peak_list, "13C"))
    result["h_peaks"] = _add(totals, "h_peaks", _list_status(sections["proton_body"], parsed.get("h_nmr_peaks"), parse_h_peak_list, "1H"))
    expected_h = [] if sections["proton_body"] is None else parse_h_peak_list(sections["proton_body"], "1H")
    result["nH"] = _add(totals, "nH", _explicit_peak_key(expected_h, parsed.get("h_nmr_peaks"), "nH"))
    result["category"] = _add(totals, "category", _explicit_peak_key(expected_h, parsed.get("h_nmr_peaks"), "category"))
    result["j"] = _add(totals, "j", _explicit_peak_key(expected_h, parsed.get("h_nmr_peaks"), "j_values"))
    result["ranges"] = _add(totals, "ranges", _explicit_range_status(expected_h, parsed.get("h_nmr_peaks")))
    result["solvent"] = _add(totals, "solvent", _header_value_status(
        sections["carbon_solvent"], sections["proton_solvent"], parsed.get("nmr_solvent"), diagnostics, "nmr_solvent",
    ))
    result["frequency"] = _add(totals, "frequency", _header_value_status(
        sections["carbon_frequency"], sections["proton_frequency"], parsed.get("nmr_frequency"), diagnostics, "nmr_frequency",
    ))
    result["frequency_nucleus"] = _add(
        totals, "frequency_nucleus",
        "kept" if parsed.get("nmr_frequency_nucleus", "unknown") == "unknown" else "changed",
    )
    result["experimental_total_nH"] = _add(
        totals, "experimental_total_nH",
        "changed" if "experimental_total_nH" in parsed else "missing",
    )
    result["gt"] = _add(totals, "gt", "missing")
    return result


def _alpaca_sections(text: str) -> dict[str, Any]:
    found = {
        "candidate": None,
        "carbon_body": None,
        "proton_body": None,
        "carbon_frequency": None,
        "proton_frequency": None,
        "carbon_solvent": None,
        "proton_solvent": None,
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        candidate_match = _CAND_RE.match(line)
        if candidate_match:
            found["candidate"] = candidate_match.group(1).strip()
            continue
        header = _HEADER_RE.match(line)
        if header is None:
            continue
        frequency, solvent = _split_header_parenthetical(header.group(2))
        section = header.group(1).upper()
        prefix = "carbon" if section == "13C" else "proton"
        found[f"{prefix}_body"] = header.group(3).strip()
        found[f"{prefix}_frequency"] = frequency
        found[f"{prefix}_solvent"] = solvent
    return found


def _explicit_match(present: bool, equal: bool) -> str:
    if not present:
        return "missing"
    return "kept" if equal else "changed"


def _list_status(body: str | None, parsed_peaks: Any, parser, where: str) -> str:
    if body is None:
        return "missing"
    expected = parser(body, where)
    return "kept" if expected == parsed_peaks else "changed"


def _explicit_peak_key(expected: list, parsed_peaks: Any, key: str) -> str:
    present = any(isinstance(peak, dict) and peak.get(key) is not None for peak in expected)
    if not present:
        invented = isinstance(parsed_peaks, list) and any(
            isinstance(peak, dict) and peak.get(key) is not None for peak in parsed_peaks
        )
        return "changed" if invented else "missing"
    got = [
        peak.get(key) if isinstance(peak, dict) else None
        for peak in (parsed_peaks or [])
    ]
    wanted = [peak.get(key) for peak in expected]
    return "kept" if got == wanted else "changed"


def _explicit_range_status(expected: list, parsed_peaks: Any) -> str:
    present = any(isinstance(peak, dict) and peak.get("rangeMin") is not None for peak in expected)
    if not present:
        invented = isinstance(parsed_peaks, list) and any(
            isinstance(peak, dict) and ("rangeMin" in peak or "rangeMax" in peak) for peak in parsed_peaks
        )
        return "changed" if invented else "missing"
    if not isinstance(parsed_peaks, list) or len(parsed_peaks) != len(expected):
        return "changed"
    for left, right in zip(expected, parsed_peaks):
        if not isinstance(right, dict):
            return "changed"
        if left.get("rangeMin") != right.get("rangeMin") or left.get("rangeMax") != right.get("rangeMax"):
            return "changed"
        if left.get("rangeMin") is not None and ("delta" in right or "centroid" in right):
            return "changed"
    return "kept"


def _header_value_status(left, right, selected, diagnostics: Mapping[str, Any], field: str) -> str:
    present = [value for value in (left, right) if value]
    if not present:
        return "changed" if selected not in (None, "") else "missing"
    unique = set(present)
    conflicts = diagnostics.get("conflicts") or []
    conflict = any(item.get("code") == "header_values_differ" and item.get("field") == field for item in conflicts)
    if len(unique) > 1:
        return "kept" if conflict and selected in (None, "") else "changed"
    return "kept" if selected == present[0] else "changed"


def _legacy_parse(text: str) -> dict[str, Any]:
    """Frozen copy of parse_alpaca_input as it behaved before step 02B-1."""
    candidate_match = re.search(r"Candidate SMILES:\s*(\S+)", text, re.IGNORECASE)
    candidate = candidate_match.group(1).strip() if candidate_match else ""
    carbon_match = re.search(r"13C NMR[^:]*:\s*([^\n]+)", text, re.IGNORECASE)
    c_peaks = []
    if carbon_match and "N/A" not in carbon_match.group(1):
        for token in carbon_match.group(1).split(","):
            try:
                c_peaks.append({"delta (ppm)": float(token.strip())})
            except ValueError:
                continue
    proton_match = re.search(r"1H NMR[^:]*:\s*([^\n]+)", text, re.IGNORECASE)
    h_peaks = []
    if proton_match and "N/A" not in proton_match.group(1):
        for delta_s, inner in re.findall(r"([0-9]+\.?[0-9]*)\s*\(([^)]*)\)", proton_match.group(1)):
            try:
                delta = float(delta_s)
            except ValueError:
                continue
            nh_match = re.search(r"(\d+)\s*H", inner)
            h_peaks.append({"delta": delta, "centroid": delta, "nH": int(nh_match.group(1)) if nh_match else None})
    return {"candidate_smiles": candidate, "c_nmr_peaks": c_peaks, "h_nmr_peaks": h_peaks}


def v64_roundtrip() -> dict[str, Any]:
    report = {}
    for name in ("nmr_matching_tool_sft_v6_4.json", "nmr_matching_tool_sft_v6_4_val.json"):
        path = LF / "data" / name
        rows = load_json(path)
        counts: Counter[str] = Counter()
        failures = []
        for index, row in enumerate(rows, 1):
            counts["rows"] += 1
            human = None
            for turn in row.get("conversations") or []:
                if isinstance(turn, dict) and turn.get("from") == "human":
                    human = turn.get("value")
                    break
            if not isinstance(human, str):
                counts["missing_human"] += 1
                failures.append({"row": index, "error": "missing human turn"})
                continue
            stripped = human.lstrip()
            if not stripped.startswith("{"):
                counts["not_json"] += 1
                failures.append({"row": index, "error": "human turn is not a JSON object", "head": stripped[:120]})
                continue
            try:
                normalized = normalize_task_input(json.loads(stripped))
                again, _diagnostics = parse_contract_text(task_to_lossless_text(normalized))
            except Exception as exc:
                counts["fail"] += 1
                failures.append({"row": index, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if again != normalized:
                counts["roundtrip_changed"] += 1
                failures.append({"row": index, "error": "lossless roundtrip changed the normalized task"})
            else:
                counts["roundtrip_kept"] += 1
            if normalized.get("nmr_frequency_nucleus") != "unknown":
                counts["nucleus_not_unknown"] += 1
            if "experimental_total_nH" in json.loads(stripped):
                counts["source_has_total"] += 1
        report[name] = {"counts": dict(counts), "failures": failures, "rewritten_training_file": False}
    return report


def before_after(out: Path) -> dict[str, Any]:
    before_path = out / "before_behavior.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else {"missing": True}
    text = (
        "Candidate SMILES: CCO\n"
        "13C NMR (400 MHz, CDCl3): 57.812, 18.2\n"
        "1H NMR (100 MHz, DMSO): 1.184 (t, J=7.1, 3H), 3.70 (q, J=7.1, 2.0, 2H)\n"
    )
    parsed, diagnostics = parse_contract_text(text)
    from recipe.nmr_matching_agent.call_v5_vllm import user_record_from_case
    from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime
    import asyncio
    runtime = NmrMatchingRuntime({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"delta": 1.2, "nH": 3}, {"delta": 3.7, "nH": None}],
    })
    observed = json.loads(asyncio.run(runtime.run_tool_call_async(
        {"function": {"name": "count_atoms", "arguments": {"element": "H"}}}
    ))["content"])
    record = user_record_from_case({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"delta": 1.2, "nH": 3}],
        "experimental_total_nH": 6,
        "nmr_frequency": "400 MHz",
    })
    return {
        "before": before,
        "after": {
            "alpaca_parse": parsed,
            "diagnostics": diagnostics,
            "partial_nH_observation": observed,
            "vllm_record_keeps_experimental_total_nH": record.get("experimental_total_nH") == 6,
            "vllm_frequency_nucleus": record.get("nmr_frequency_nucleus"),
        },
        "note": "The before object was captured from the unmodified functions. The 02A report was not rewritten.",
    }


def preservation() -> dict[str, Any]:
    rows = []
    for relative in PRESERVED:
        path = OA / relative
        digest = sha256_file(path) if path.exists() else None
        rows.append({"path": relative, "sha256": digest, "exists": path.exists()})
    return {"files": rows, "note": "These pre-existing dirty framework files were not part of this step."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--counts-only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checked = verify_indexes()
    (out / "index_hash_check.json").write_text(json.dumps(checked, indent=2) + "\n", encoding="utf-8")
    if checked["mismatches"]:
        raise SystemExit(f"02A index hash mismatch: {checked['mismatches']}")
    print("[02b1] indexes match", flush=True)
    records = []
    with (AUDIT_02A / "labeled_index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    exposure = build_exposure(records)
    exposure["index_hashes"] = json.loads((AUDIT_02A / "asset_manifest.json").read_text(encoding="utf-8"))["index_hashes"]
    test_keys = fingerprint_test_keys()
    exposure["views"] = prepare_views(exposure, test_keys)
    print("[02b1] exposure sets ready", flush=True)
    scan = scan_splits(exposure, test_keys, out, write_index=not args.counts_only, with_quality=not args.counts_only)
    files = scan["files"]
    comparison = {}
    for name in SPLIT_FILES:
        comparison[name] = {
            "rows": files[name]["rows"],
            "baseline_remaining": files[name]["baseline_remaining"],
            "expected_baseline_remaining": EXPECTED_BASELINE[name],
            "baseline_match": files[name]["baseline_remaining"] == EXPECTED_BASELINE[name],
            "new_test_candidate_exclusion": files[name]["buckets"].get("test_candidate", 0),
            "expected_new_exclusion": EXPECTED_NEW[name],
            "new_exclusion_match": files[name]["buckets"].get("test_candidate", 0) == EXPECTED_NEW[name],
            "remaining_after_new_rule": files[name]["buckets"].get("remaining", 0),
            "expected_remaining": EXPECTED_REMAINING[name],
            "remaining_match": files[name]["buckets"].get("remaining", 0) == EXPECTED_REMAINING[name],
            "baseline_mismatch_rows": files[name]["baseline_mismatch"],
        }
    all_rows = sum(item["rows"] for item in comparison.values())
    all_remaining = sum(item["remaining_after_new_rule"] for item in comparison.values())
    all_new = sum(item["new_test_candidate_exclusion"] for item in comparison.values())
    eligibility = {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "wording": scan["wording"],
        "splits_are_not_summed_with_all_lmdb": (
            "train, valid, and test are the partition of all.lmdb established by the 02A token intersections. "
            "all counts below are the sum of those three files, not a fourth scan."
        ),
        "all": {"rows": all_rows, "new_test_candidate_exclusion": all_new, "remaining_after_new_rule": all_remaining},
        "expected_all": {"rows": 1076955, "new_test_candidate_exclusion": 957, "remaining_after_new_rule": 1069488},
        "files": comparison,
        "exposure_set_sizes": {
            name: {key: len(value) for key, value in packed.items() if isinstance(value, set)}
            for name, packed in exposure["packed"].items()
        },
        "exposure_rows": exposure["records"],
        "test_key_audit": {
            "rows": test_keys["rows"],
            "text_candidate_tokens": len(test_keys["text_candidate_tokens"]),
            "declared_candidate_tokens": len(test_keys["declared_candidate_tokens"]),
            "gt_tokens": len(test_keys["gt_tokens"]),
            "text_and_declared_token_disagreements": test_keys["text_and_declared_token_disagreements"],
            "declared_canonical_field_present": test_keys["declared_canonical_field_present"],
        },
        "per_file": files,
        "reason_families": scan["reason_families"],
        "cross_split_exact_spectra": scan.get("cross_split_exact_spectra"),
        "cross_split_shift_sequences": scan.get("cross_split_shift_sequences"),
        "cross_split_note": scan.get("cross_split_note"),
        "index": {"path": scan.get("index_path"), "sha256": scan.get("index_sha256"), "bytes": scan.get("index_bytes")},
        "not_selected": "No search-dev or gate-dev sample was drawn. Size, ratio, difficulty, and thresholds remain unset.",
    }
    (out / "source_eligibility_report.json").write_text(json.dumps(eligibility, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    certainty = {
        "confirmed": "v6.4 checkpoint-63 registered train rows, asset v64_source_split role weight_train",
        "inferred": "v6.1 and v5 train files. Their logs do not name those files. They stay excluded.",
        "unknown": "checkpoint-200 dataset is unknown. cot train and val are possible sources and stay excluded.",
        "union": "confirmed plus inferred. The 02A field weight_train used this union and must not be read as pure confirmed.",
        "rows": exposure["records"],
        "unique_tokens": eligibility["exposure_set_sizes"],
        "sequential_buckets": {name: files[name]["buckets"] for name in SPLIT_FILES},
        "do_not_upgrade_without_a_new_log": True,
    }
    (out / "exposure_certainty_report.json").write_text(json.dumps(certainty, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.counts_only:
        print("[02b1] historical and v6.4", flush=True)
        (out / "historical_sample_fields.json").write_text(json.dumps(historical_samples(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (out / "v64_roundtrip_report.json").write_text(json.dumps(v64_roundtrip(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (out / "before_after_repro.json").write_text(json.dumps(before_after(out), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "worktree_preservation.json").write_text(json.dumps(preservation(), indent=2) + "\n", encoding="utf-8")
    elapsed = round(time.time() - started, 1)
    print(f"[02b1] done in {elapsed}s", flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
