#!/usr/bin/env python3
"""Read-only step-02A audit of NMR matching data lineage, overlap, and inputs.

The script reads existing JSON, JSONL, YAML, trainer state, and LMDB files.
It writes only under the directory given by --out. It does not train, call a
predictor, or select a frozen development split.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

RSI_DIR = Path(__file__).resolve().parent
if str(RSI_DIR) not in sys.path:
    sys.path.insert(0, str(RSI_DIR))

from step_02a_fingerprints import RDKIT_VERSION, STRUCTURE_OPTIONS, spectrum_views  # noqa: E402
from step_02a_overlap import intersection_count, summarize_records  # noqa: E402
from step_02a_records import build_index_record  # noqa: E402

SPEC = Path("/fs_mol/cxq/SpecMaster_jp")
OA = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
LF = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory")
LMDB_ROOT = Path("/fs_mol/fjxu/proj_data/Spectra/NMR/si_0820/lmdb_dataset")
HASH_LIMIT = 80_000_000
CANDIDATE_RE = re.compile(r"Candidate SMILES:\s*(\S+)")


def sha256_file_always(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    if path.stat().st_size > HASH_LIMIT:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    digest = sha256_file(path)
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest if digest is not None else None,
        "hash_policy": "sha256" if digest is not None else "metadata_only",
    }


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def snippet(path: Path, needle: str) -> dict[str, Any]:
    """Find a unique source snippet and return its line number."""
    if not path.exists():
        return {"path": str(path), "found": False, "needle": needle}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if needle in line:
                return {
                    "path": str(path),
                    "found": True,
                    "line": line_number,
                    "text": line.strip()[:240],
                }
    return {"path": str(path), "found": False, "needle": needle}


def trainer_brief(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    state = load_json(path)
    history = state.get("log_history") or []
    return {
        "path": str(path),
        "exists": True,
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "eval_events": sum(1 for row in history if "eval_loss" in row),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
    }


def human_value(row: Mapping[str, Any]) -> str | None:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == "human":
            value = turn.get("value")
            return value if isinstance(value, str) else None
    return None


def user_value(row: Mapping[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            value = message.get("content")
            return value if isinstance(value, str) else None
    return None


def parse_task_text(text: str | None) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {"format": "empty", "candidate": None, "spectrum": None, "task": None}
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            task = json.loads(stripped)
        except json.JSONDecodeError:
            return {"format": "json_error", "candidate": None, "spectrum": None, "task": None}
        if not isinstance(task, dict):
            return {"format": "json_not_object", "candidate": None, "spectrum": None, "task": None}
        candidate = task.get("candidate_smiles") or task.get("candidate")
        return {"format": "structured_json", "candidate": candidate, "spectrum": task, "task": task}
    match = CANDIDATE_RE.search(text)
    return {
        "format": "alpaca_text",
        "candidate": match.group(1) if match else None,
        "spectrum": None,
        "task": None,
        "text": text,
    }


def append_record(bucket: list[dict[str, Any]], **kwargs: Any) -> None:
    bucket.append(build_index_record(**kwargs))


def index_trajectory(
    path: Path,
    asset: str,
    role: str,
    records: list[dict[str, Any]],
    role_sequence: list[str] | None = None,
) -> dict[str, Any]:
    count = 0
    meta_gt = 0
    for sequence, (line_number, row) in enumerate(iter_jsonl(path)):
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        parsed = parse_task_text(user_value(row))
        label = meta.get("label")
        gt = meta.get("gt_smiles")
        if gt:
            meta_gt += 1
        item_role = role_sequence[sequence] if role_sequence else role
        append_record(
            records,
            asset=asset,
            role=item_role,
            locator=f"{path}:{line_number}",
            candidate=parsed["candidate"] or meta.get("candidate_smiles"),
            gt=gt if gt else None,
            label=label if isinstance(label, int) and not isinstance(label, bool) else label,
            negative_type=meta.get("negative_type"),
            spectrum=parsed["spectrum"],
        )
        count += 1
    if role_sequence is not None and count != len(role_sequence):
        raise RuntimeError(f"{path} row count {count} does not match split length {len(role_sequence)}")
    return {"rows": count, "meta_gt_present": meta_gt}


def index_sharegpt(
    path: Path,
    asset: str,
    role: str,
    records: list[dict[str, Any]],
    meta_by_user: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    rows = load_json(path)
    joined = ambiguous = unmatched = 0
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a JSON list")
    for index, row in enumerate(rows, 1):
        text = human_value(row) if isinstance(row, dict) else None
        parsed = parse_task_text(text)
        gt = None
        label = None
        negative = None
        gt_source = "absent"
        matches = meta_by_user.get(text or "", []) if meta_by_user and text else []
        gts = {(item.get("gt_smiles"), item.get("label"), item.get("negative_type")) for item in matches}
        if len(gts) == 1:
            gt, label, negative = next(iter(gts))
            gt_source = "joined_source_meta"
            joined += 1
        elif len(gts) > 1:
            gt_source = "ambiguous_source_meta"
            ambiguous += 1
        else:
            unmatched += 1
        record = build_index_record(
            asset=asset,
            role=role,
            locator=f"{path}:{index}",
            candidate=parsed["candidate"],
            gt=gt if gt else None,
            label=label,
            negative_type=negative,
            spectrum=parsed["spectrum"],
        )
        record["gt_recovery"] = gt_source
        records.append(record)
    return {"rows": len(rows), "joined": joined, "ambiguous": ambiguous, "unmatched": unmatched}


def index_alpaca(path: Path, asset: str, role: str, records: list[dict[str, Any]]) -> int:
    rows = load_json(path)
    for index, row in enumerate(rows, 1):
        parsed = parse_task_text(row.get("input") if isinstance(row, dict) else None)
        append_record(
            records,
            asset=asset,
            role=role,
            locator=f"{path}:{index}",
            candidate=row.get("_candidate_smiles") or parsed["candidate"],
            gt=row.get("_gt_smiles") or None,
            label=row.get("_label"),
            negative_type=row.get("_negative_type"),
            spectrum=parsed["spectrum"],
        )
    return len(rows)


def index_pool(path: Path, asset: str, role: str, records: list[dict[str, Any]]) -> int:
    count = 0
    for line_number, row in iter_jsonl(path):
        nested = row.get("input") if isinstance(row.get("input"), dict) else row
        candidate = nested.get("candidate_smiles") or row.get("candidate_smiles")
        spectrum = {
            key: nested.get(key)
            for key in ("c_nmr_peaks", "h_nmr_peaks", "nmr_solvent", "nmr_frequency", "experimental_total_nH")
            if key in nested
        }
        append_record(
            records,
            asset=asset,
            role=role,
            locator=f"{path}:{line_number}",
            candidate=candidate,
            gt=row.get("gt_smiles") or None,
            label=row.get("label"),
            negative_type=row.get("negative_type"),
            spectrum=spectrum or None,
        )
        count += 1
    return count


def meta_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    if not path.exists():
        return found
    for _line, row in iter_jsonl(path):
        text = user_value(row)
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        if text:
            found[text].append(meta)
    return found


def v64_roles() -> list[str]:
    """Replay the production grouped split without writing its outputs."""
    source = SPEC / "distill_out/trajectories_v6_3_filtered_balanced_observation_fixed.jsonl"
    script = SPEC / "scripts/prepare_v6_4_sft_data.py"
    spec = importlib.util.spec_from_file_location("prepare_v6_4_sft_data_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.read_jsonl(source)
    train, validation, _report = module.split_grouped_stratified(rows, 0.03, 42)
    train_ids = {id(row) for row in train}
    val_ids = {id(row) for row in validation}
    roles: list[str] = []
    for row in rows:
        if id(row) in val_ids:
            roles.append("weight_eval")
        elif id(row) in train_ids:
            roles.append("weight_train")
        else:
            roles.append("excluded_from_v64_outputs")
    return roles


def tokens(records: Iterable[Mapping[str, Any]], role_set: set[str], field: str) -> set[str]:
    return {
        record[field]
        for record in records
        if record.get("role") in role_set and record.get(field) not in (None, "", "missing")
    }


def compare_observation_files() -> dict[str, Any]:
    left = SPEC / "distill_out/trajectories_v6_3_filtered_balanced.jsonl"
    right = SPEC / "distill_out/trajectories_v6_3_filtered_balanced_observation_fixed.jsonl"
    if not left.exists() or not right.exists():
        return {"status": "unknown", "reason": "one of the trajectory files is absent"}
    meta_differ = user_differ = final_differ = both_meta_and_final = 0
    provenance_cleared = other_meta_fields = 0
    rows = 0
    left_rows = right_rows = 0
    for (ln_a, a), (ln_b, b) in zip(iter_jsonl(left), iter_jsonl(right)):
        rows += 1
        meta_changed = a.get("meta") != b.get("meta")
        if meta_changed:
            meta_differ += 1
            am = a.get("meta") if isinstance(a.get("meta"), dict) else {}
            bm = b.get("meta") if isinstance(b.get("meta"), dict) else {}
            if am.get("provenance_violations") and bm.get("provenance_violations") == []:
                provenance_cleared += 1
            if {k: v for k, v in am.items() if k != "provenance_violations"} != {
                k: v for k, v in bm.items() if k != "provenance_violations"
            }:
                other_meta_fields += 1
        if user_value(a) != user_value(b):
            user_differ += 1
        ma, mb = a.get("messages") or [], b.get("messages") or []
        fa = ma[-1]["content"] if ma else None
        fb = mb[-1]["content"] if mb else None
        final_changed = fa != fb
        if final_changed:
            final_differ += 1
        if meta_changed and final_changed:
            both_meta_and_final += 1
    left_rows = sum(1 for _ in iter_jsonl(left))
    right_rows = sum(1 for _ in iter_jsonl(right))
    return {
        "status": "inferred",
        "rows_compared": rows,
        "left_rows": left_rows,
        "right_rows": right_rows,
        "meta_differ": meta_differ,
        "provenance_violations_cleared": provenance_cleared,
        "other_meta_fields_differ": other_meta_fields,
        "user_json_differ": user_differ,
        "final_assistant_differ": final_differ,
        "meta_and_final_assistant_both_differ": both_meta_and_final,
        "basis": (
            "Both files have the same number of JSONL rows. User JSON is unchanged. "
            "The rows whose meta differs also change the final assistant turn, and the "
            "meta change is provenance_violations becoming an empty list. That is the "
            "behavior coded in scripts/rewrite_final_evidence_from_tools.py, but no log "
            "of the command that wrote observation_fixed was found, so the producer stays inferred."
        ),
    }


def build_lineage() -> dict[str, Any]:
    yaml_dir = LF / "examples/train_full"
    saves = LF / "saves"
    items = [
        {
            "id": "v6_4_checkpoint_63",
            "status": "confirmed",
            "claim": "checkpoint-63 was trained for one epoch on nmr_matching_tool_sft_v6_4, initialized from full-v6_1.",
            "evidence": [
                snippet(yaml_dir / "qwen3_32b_nmr_matching_tool_v6_4.yaml", "model_name_or_path:"),
                snippet(yaml_dir / "qwen3_32b_nmr_matching_tool_v6_4.yaml", "dataset:"),
                snippet(SPEC / "v6_4_train.log", "Loading dataset nmr_matching_tool_sft_v6_4.json"),
                snippet(SPEC / "v6_4_train.log", "full-v6_1/config.json"),
                snippet(SPEC / "v6_4_train.log", "checkpoint-63"),
                trainer_brief(saves / "qwen3-32b-nmr-matching-tool/full-v6_4/checkpoint-63/trainer_state.json"),
            ],
        },
        {
            "id": "v6_4_rows",
            "status": "confirmed",
            "claim": "The registered v6.4 files contain 4009 train rows and 125 validation rows from the observation-fixed trajectory file.",
            "evidence": [
                file_record(LF / "data/nmr_matching_tool_sft_v6_4_split_report.json"),
                snippet(SPEC / "scripts/prepare_v6_4_sft_data.py", "trajectories_v6_3_filtered_balanced_observation_fixed.jsonl"),
            ],
        },
        {
            "id": "v6_1_parent",
            "status": "inferred",
            "claim": "full-v6_1 is the v6.1 SFT run: its yaml names dataset v6_1 and init full-v5, and trainer_state reaches epoch 1. The training log that would name the dataset file was not found.",
            "evidence": [
                snippet(yaml_dir / "qwen3_32b_nmr_matching_tool_v6_1.yaml", "model_name_or_path:"),
                snippet(yaml_dir / "qwen3_32b_nmr_matching_tool_v6_1.yaml", "dataset:"),
                trainer_brief(saves / "qwen3-32b-nmr-matching-tool/full-v6_1/checkpoint-61/trainer_state.json"),
            ],
        },
        {
            "id": "v5_parent",
            "status": "inferred",
            "claim": "full-v5 is the v5 tool SFT run: its yaml names dataset v5, two epochs, and init no-tool checkpoint-200. trainer_state at that directory reaches epoch 2. No separate log naming the dataset was found.",
            "evidence": [
                snippet(yaml_dir / "qwen3_32b_nmr_matching_tool_v5.yaml", "model_name_or_path:"),
                snippet(yaml_dir / "qwen3_32b_nmr_matching_tool_v5.yaml", "dataset:"),
                snippet(yaml_dir / "qwen3_32b_nmr_matching_tool_v5.yaml", "num_train_epochs:"),
                trainer_brief(saves / "qwen3-32b-nmr-matching-tool/full-v5/trainer_state.json"),
            ],
        },
        {
            "id": "checkpoint_200_dataset",
            "status": "unknown",
            "claim": "checkpoint-200 exists and is the configured init of v5, but no yaml in the tree sets output_dir to qwen3-32b-nmr-matching/full-exp1. Comments call it no-tool SFT-v4. nmr_matching_cot_train.json is a possible source, not a confirmed one.",
            "evidence": [
                trainer_brief(saves / "qwen3-32b-nmr-matching/full-exp1/checkpoint-200/trainer_state.json"),
                snippet(SPEC / "scripts/prepare_cot_sft_data.py", "nmr_matching_cot_train.json"),
                snippet(yaml_dir / "qwen3_32b_nmr_matching_full.yaml", "output_dir:"),
            ],
        },
        {
            "id": "siblings_not_in_weight_chain",
            "status": "confirmed",
            "claim": "v6, v6.2, and v6.3 have trainer_state files, but their yamls do not sit between full-v5 and checkpoint-63. v6.4's yaml initializes from full-v6_1, not from v6.2 or v6.3.",
            "evidence": [
                trainer_brief(saves / "qwen3-32b-nmr-matching-tool/full-v6/checkpoint-780/trainer_state.json"),
                trainer_brief(saves / "qwen3-32b-nmr-matching-tool/full-v6_2/checkpoint-62/trainer_state.json"),
                trainer_brief(saves / "qwen3-32b-nmr-matching-tool/full-v6_3/checkpoint-64/trainer_state.json"),
                snippet(yaml_dir / "qwen3_32b_nmr_matching_tool_v6_4.yaml", "full-v6_1"),
            ],
        },
        {
            "id": "spectrum_and_labels",
            "status": "confirmed",
            "claim": "Negative-engine samples copy the query record's peak lists and store the query smiles as gt_smiles. Hard-negative mining uses a forward predictor only to rank perturbations; the stored peaks remain the query peaks.",
            "evidence": [
                snippet(SPEC / "perturbation_generator/negative_engine.py", "pairing *neg_smiles* with query's NMR"),
                snippet(SPEC / "perturbation_generator/negative_engine.py", "c_nmr_peaks=query_record.get"),
                snippet(SPEC / "perturbation_generator/hard_negative_cli.py", "exp_c_peaks=sample.c_nmr_peaks"),
                snippet(SPEC / "perturbation_generator/hard_negative_cli.py", "experimental NMR"),
            ],
        },
        {
            "id": "lmdb_experimental_claim",
            "status": "inferred",
            "claim": "Generator help text calls the LMDB spectra experimental. No dataset card in this audit proves that claim or names the spectrometer nucleus for nmr_frequency.",
            "evidence": [
                snippet(SPEC / "perturbation_generator/lmdb_reader.py", "lmdb_dataset"),
                snippet(SPEC / "perturbation_generator/hard_negative_cli.py", "experimental NMR"),
            ],
        },
        {
            "id": "rl_not_ancestor",
            "status": "confirmed",
            "claim": "The agent GRPO script defaults to tool full-exp1 checkpoint-200, which is not the v6.4 weight path. Configuring that script does not show that a finished RL run updated checkpoint-63.",
            "evidence": [
                snippet(OA / "recipe/nmr_matching_agent/grpo_qwen3_32b_agent_8gpu.sh", "model_path="),
            ],
        },
    ]
    return {
        "weight_chain": [
            "Qwen3-32B base weights",
            "unknown no-tool run that produced full-exp1/checkpoint-200",
            "inferred v5 tool SFT at full-v5",
            "inferred v6.1 tool SFT at full-v6_1",
            "confirmed v6.4 tool SFT checkpoint-63",
        ],
        "items": items,
        "observation_fixed_comparison": compare_observation_files(),
        "rdkit": {"version": RDKIT_VERSION, "options": STRUCTURE_OPTIONS},
    }


def field_counter(task: Mapping[str, Any] | None, counter: dict[str, int]) -> None:
    if not isinstance(task, dict):
        counter["rows_without_structured_task"] += 1
        return
    counter["structured_rows"] += 1
    for key in (
        "candidate_smiles", "candidate", "c_nmr_peaks", "h_nmr_peaks",
        "nmr_solvent", "solvent", "nmr_frequency", "nmr_freq", "experimental_total_nH",
    ):
        if key in task:
            counter[f"key_present:{key}"] += 1
            if task.get(key) in (None, "", [], {}):
                counter[f"key_empty:{key}"] += 1
    peaks = task.get("h_nmr_peaks")
    if isinstance(peaks, list):
        for peak in peaks:
            counter["h_peaks"] += 1
            if not isinstance(peak, dict):
                counter["h_peak_not_object"] += 1
                continue
            for key in ("category", "centroid", "delta", "j_values", "nH", "rangeMax", "rangeMin"):
                if key not in peak:
                    counter[f"h_missing:{key}"] += 1
                elif peak.get(key) is None:
                    counter[f"h_null:{key}"] += 1
                else:
                    counter[f"h_value:{key}"] += 1
            if peak.get("nH") == 0:
                counter["h_zero_nH"] += 1
    c_peaks = task.get("c_nmr_peaks")
    if isinstance(c_peaks, list):
        counter["c_peaks"] += len(c_peaks)
        for peak in c_peaks:
            if isinstance(peak, dict) and "delta (ppm)" in peak:
                counter["c_delta_key"] += 1


def input_audit(records_source: Path) -> dict[str, Any]:
    """Full-scan the v6.4 trajectory user JSON and compare it with the current parser."""
    sys.path.insert(0, str(OA))
    sys.path.insert(0, str(SPEC / "scripts"))
    from build_test_2000 import format_alpaca
    from recipe.nmr_matching_agent.standalone_infer_matching import (
        parse_alpaca_input,
        prepare_eval_record,
    )

    counter: dict[str, int] = collections.Counter()
    loss = collections.Counter()
    for line_number, row in iter_jsonl(records_source):
        parsed = parse_task_text(user_value(row))
        task = parsed["task"]
        field_counter(task, counter)
        if not isinstance(task, dict) or not task.get("candidate_smiles"):
            loss["skipped_no_structured_candidate"] += 1
            continue
        rendered = format_alpaca({
            "candidate_smiles": task.get("candidate_smiles"),
            "c_nmr_peaks": task.get("c_nmr_peaks") or [],
            "h_nmr_peaks": task.get("h_nmr_peaks") or [],
            "nmr_frequency": task.get("nmr_frequency") or "",
            "nmr_solvent": task.get("nmr_solvent") or "",
            "label": 1,
            "gt_smiles": "",
        })
        try:
            recovered = parse_alpaca_input(rendered["input"])
        except Exception as exc:
            loss[f"parser_error:{type(exc).__name__}"] += 1
            continue
        original_c = task.get("c_nmr_peaks") or []
        original_h = task.get("h_nmr_peaks") or []
        if len(recovered.get("c_nmr_peaks") or []) != len(original_c):
            loss["c_count_changed"] += 1
        if len(recovered.get("h_nmr_peaks") or []) != len(original_h):
            loss["h_count_changed"] += 1
        for peak in original_h:
            if not isinstance(peak, dict):
                continue
            if peak.get("j_values") not in (None, "", "_") and "J=" not in rendered["input"]:
                loss["j_not_explicit_in_text"] += 1
            if "nH" not in peak and re.search(r"\d+H", rendered["input"]):
                loss["missing_nH_rendered_as_number"] += 1
            if peak.get("nH") is None and re.search(r"\d+H", rendered["input"]):
                loss["null_nH_rendered_as_number"] += 1
        if task.get("nmr_solvent") and task.get("nmr_solvent") not in rendered["input"]:
            loss["solvent_dropped_from_text"] += 1
        if task.get("nmr_frequency") and str(task.get("nmr_frequency")) not in rendered["input"]:
            loss["frequency_dropped_from_text"] += 1
        loss["roundtrip_rows"] += 1

    text = (
        "Candidate SMILES: CCO\n"
        "13C NMR (400 MHz, CDCl3): 57.8, 18.2\n"
        "1H NMR (400 MHz, CDCl3): 1.18 (t, J=7.1, 3H)\n"
    )
    mixed_cases = [
        ("text_only", {"input": text, "_label": 1}),
        ("text_same_candidate", {"input": text, "candidate_smiles": "CCO", "_label": 1}),
        ("text_conflicting_candidate", {"input": text, "candidate_smiles": "CCC", "_label": 0, "_gt_smiles": "CCO"}),
        ("structured_null_integral", {
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}],
            "h_nmr_peaks": [{"delta": 1.18, "centroid": 1.18, "nH": None}],
            "_label": 1,
        }),
        ("structured_zero_integral", {
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}],
            "h_nmr_peaks": [{"delta": 1.18, "centroid": 1.18, "nH": 0}],
            "_label": 1,
        }),
        ("empty_peaks", {"candidate_smiles": "CCO", "c_nmr_peaks": [], "h_nmr_peaks": [], "_label": 1}),
        ("duplicate_peaks", {
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}, {"delta (ppm)": 57.8}],
            "h_nmr_peaks": [],
            "_label": 1,
        }),
        ("reordered_peaks", {
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 18.2}, {"delta (ppm)": 57.8}],
            "h_nmr_peaks": [],
            "_label": 1,
        }),
    ]
    mixed = []
    for name, record in mixed_cases:
        try:
            prepared = prepare_eval_record(record, 0)
            task = prepared["inference_input"]
            viewed = spectrum_views(task)
            mixed.append({
                "name": name,
                "status": "accepted",
                "candidate": task.get("candidate_smiles"),
                "c_count": len(task.get("c_nmr_peaks") or []),
                "h_count": len(task.get("h_nmr_peaks") or []),
                "h_nH": [peak.get("nH") for peak in task.get("h_nmr_peaks") or []],
                "spectrum_norm_hash": viewed["normalized_hash"],
            })
        except Exception as exc:
            mixed.append({"name": name, "status": "rejected", "error_type": type(exc).__name__, "error": str(exc)})
    return {
        "source": str(records_source),
        "field_counts": dict(counter),
        "alpaca_roundtrip_loss": dict(loss),
        "current_mixed_input_behavior": mixed,
        "frequency_note": (
            "nmr_frequency strings such as 300MHz, 150 MHz, and 101 MHz occur in the LMDB. "
            "Those magnitudes are consistent with a mixture of nominal spectrometer frequencies "
            "and per-nucleus frequencies. This audit does not assign a nucleus."
        ),
    }


def stratified_sample_rule() -> dict[str, Any]:
    return {
        "full_scan": [
            "v6.4 observation-fixed trajectories",
            "v6.4 train and validation JSON",
            "test_2000",
            "ancestor trajectory or source JSONL files when present",
        ],
        "historical_sample": {
            "seed": 20261125,
            "rule": "For each file, rows are grouped by label when a label is present, otherwise by the single bucket 'unknown'. Within each bucket the row indexes are shuffled with random.Random(20261125) and at most 200 indexes are kept. The sample describes format coverage only and is not used to choose a model change.",
        },
    }


def sample_historical(path: Path, kind: str) -> dict[str, Any]:
    rng_seed = 20261125
    import random
    random.seed(rng_seed)
    buckets: dict[str, list[int]] = collections.defaultdict(list)
    rows = 0
    if kind == "jsonl":
        iterable = (row for _ln, row in iter_jsonl(path))
    else:
        loaded = load_json(path)
        iterable = loaded if isinstance(loaded, list) else []
    formats = collections.Counter()
    for index, row in enumerate(iterable, 1):
        rows += 1
        if not isinstance(row, dict):
            continue
        label = None
        if isinstance(row.get("meta"), dict):
            label = row["meta"].get("label")
        label = row.get("_label", row.get("label", label))
        key = "unknown" if label is None else str(label)
        buckets[key].append(index)
        text = user_value(row) or human_value(row) or row.get("input")
        formats[parse_task_text(text if isinstance(text, str) else None)["format"]] += 1
    chosen = []
    for indexes in buckets.values():
        random.shuffle(indexes)
        chosen.extend(sorted(indexes[:200]))
    return {"rows": rows, "formats": dict(formats), "sampled_indexes": len(chosen), "buckets": {k: len(v) for k, v in buckets.items()}}


def sets_from(records: list[dict[str, Any]], roles: set[str]) -> dict[str, set[str]]:
    return {
        "candidate": tokens(records, roles, "candidate_token"),
        "gt": tokens(records, roles, "gt_token"),
        "spectrum": tokens(records, roles, "spectrum_norm_hash"),
        "shift_sequence": tokens(records, roles, "shift_sequence_hash"),
        "pair": tokens(records, roles, "pair_token"),
        "connectivity": tokens(records, roles, "connectivity"),
        "scaffold": tokens(records, roles, "scaffold"),
    }


def lmdb_audit(
    out_dir: Path,
    exposures: dict[str, dict[str, set[str]]],
    limit: int | None,
) -> dict[str, Any]:
    import lmdb

    files = ["all.lmdb", "train.lmdb", "valid.lmdb", "test.lmdb"]
    per_file = {}
    confirmed = exposures["confirmed_train"]
    evaluation = exposures["known_eval_and_test"]
    unknown = exposures["unknown_possible"]
    for name in files:
        path = LMDB_ROOT / name
        started = time.time()
        env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        index_path = out_dir / f"lmdb_{name}.index.jsonl"
        progress_path = out_dir / f"lmdb_{name}.progress.json"
        stats = collections.Counter()
        frequencies = collections.Counter()
        remaining_scaffolds: set[str] = set()
        canonicals: set[str] = set()
        spectra: set[str] = set()
        with env.begin(buffers=True) as txn, index_path.open("w", encoding="utf-8") as index:
            for key, value in txn.cursor():
                stats["cursor_rows"] += 1
                raw = pickle.loads(bytes(value))
                smiles = raw.get("smiles") if isinstance(raw.get("smiles"), str) else None
                declared = raw.get("smiles_canonical") if isinstance(raw.get("smiles_canonical"), str) else None
                if smiles and declared and smiles != declared:
                    stats["raw_and_declared_smiles_differ"] += 1
                identity = smiles or declared
                spectrum = {
                    field: raw.get(field)
                    for field in ("c_nmr_peaks", "h_nmr_peaks", "nmr_solvent", "nmr_frequency")
                    if field in raw
                }
                record = build_index_record(
                    asset=f"lmdb:{name}",
                    role="source_pool",
                    locator=f"{name}:{bytes(key).decode('utf-8', 'replace')}",
                    candidate=identity,
                    gt=declared or None,
                    label=None,
                    negative_type=None,
                    spectrum=spectrum,
                )
                if not raw.get("molecular_formula"):
                    stats["missing_formula"] += 1
                freq = raw.get("nmr_frequency")
                if isinstance(freq, str):
                    frequencies[freq] += 1
                canonicals.add(record["candidate_token"])
                if record["spectrum_norm_hash"]:
                    spectra.add(record["spectrum_norm_hash"])
                structure_hit = record["candidate_token"] in confirmed["candidate"] or record["candidate_token"] in confirmed["gt"] or record["gt_token"] in confirmed["gt"] or record["gt_token"] in confirmed["candidate"]
                spectrum_hit = record["spectrum_norm_hash"] in confirmed["spectrum"]
                pair_hit = record["pair_token"] in confirmed["pair"]
                shift_hit = record.get("shift_sequence_hash") in confirmed["shift_sequence"]
                eval_structure = record["candidate_token"] in evaluation["candidate"] or record["candidate_token"] in evaluation["gt"] or record["gt_token"] in evaluation["gt"]
                unknown_structure = record["candidate_token"] in unknown["candidate"] or record["candidate_token"] in unknown["gt"]
                if structure_hit:
                    stats["hit_confirmed_structure"] += 1
                if spectrum_hit:
                    stats["hit_confirmed_spectrum"] += 1
                if pair_hit:
                    stats["hit_confirmed_pair"] += 1
                if shift_hit:
                    stats["hit_confirmed_shift_sequence"] += 1
                if shift_hit and not spectrum_hit:
                    stats["shift_sequence_hit_without_exact_spectrum"] += 1
                if eval_structure:
                    stats["hit_known_eval_or_test_structure"] += 1
                if unknown_structure:
                    stats["hit_unknown_possible_structure"] += 1
                confirmed_excluded = structure_hit or spectrum_hit or pair_hit or eval_structure
                if shift_hit and not confirmed_excluded:
                    stats["shift_sequence_only_not_used_as_exclusion"] += 1
                conservative_excluded = confirmed_excluded or unknown_structure
                if confirmed_excluded:
                    stats["excluded_confirmed_or_known_eval"] += 1
                else:
                    stats["remaining_upper_bound"] += 1
                    if record["scaffold"]:
                        remaining_scaffolds.add(record["scaffold"])
                if conservative_excluded:
                    stats["excluded_conservative"] += 1
                else:
                    stats["remaining_conservative"] += 1
                if record["candidate_status"] == "unparsed":
                    stats["unparsed_smiles"] += 1
                index.write(json.dumps({
                    "locator": record["locator"],
                    "candidate_status": record["candidate_status"],
                    "candidate_token": record["candidate_token"],
                    "gt_token": record["gt_token"],
                    "connectivity": record["connectivity"],
                    "scaffold": record["scaffold"],
                    "spectrum_norm_hash": record["spectrum_norm_hash"],
                    "shift_sequence_hash": record.get("shift_sequence_hash"),
                    "pair_token": record["pair_token"],
                }, ensure_ascii=False) + "\n")
                if stats["cursor_rows"] % 20000 == 0:
                    progress_path.write_text(json.dumps({
                        "rows": stats["cursor_rows"],
                        "elapsed_sec": round(time.time() - started, 1),
                    }), encoding="utf-8")
                    print(f"[lmdb] {name} {stats['cursor_rows']}", flush=True)
                if limit is not None and stats["cursor_rows"] >= limit:
                    break
        env.close()
        per_file[name] = {
            "source": file_record(path),
            "elapsed_sec": round(time.time() - started, 1),
            "counts": dict(stats),
            "unique_structure_tokens": len(canonicals),
            "unique_spectrum_hashes": len(spectra),
            "remaining_upper_bound_scaffolds": len(remaining_scaffolds),
            "frequency_top": frequencies.most_common(25),
            "index": str(index_path),
            "index_sha256": sha256_file_always(index_path),
            "index_hash_policy": "sha256",
            "interpretation": (
                "remaining_upper_bound excludes confirmed weight-training structures, exact "
                "spectra, pairs, and known test/eval structures. A shift-sequence hit is counted "
                "but does not by itself remove a row, because that key ignores J values and "
                "integrals. It is not proof the molecule was absent from every upstream model. "
                "remaining_conservative also excludes the unresolved checkpoint-200 candidate file."
            ),
        }
    return {"files": per_file, "not_additive": "all.lmdb and the split files are reported separately and must not be summed."}


def _select(records: list[dict[str, Any]], asset: str | None, role: str | None, field: str) -> set[str]:
    found = set()
    for record in records:
        if asset is not None and record.get("asset") != asset:
            continue
        if role is not None and record.get("role") != role:
            continue
        value = record.get(field)
        if value not in (None, "", "missing"):
            found.add(value)
    return found


def overlap_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_asset: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_asset[record["asset"]].append(record)
    summaries = {asset: summarize_records(rows) for asset, rows in by_asset.items()}
    for summary in summaries.values():
        summary["contradictory_pairs"] = dict(list(summary["contradictory_pairs"].items())[:20])
        summary["same_gt_multiple_candidates"] = dict(list(summary["same_gt_multiple_candidates"].items())[:20])
    def chosen(asset: str | None, role: str | None, field: str) -> set[str]:
        return _select(records, asset, role, field)

    pairs = {
        "v64_train_candidate_vs_test_2000_candidate": intersection_count(
            chosen("v64_source_split", "weight_train", "candidate_token"),
            chosen("test_2000", "test", "candidate_token"),
        ),
        "v64_train_gt_vs_test_2000_gt": intersection_count(
            chosen("v64_source_split", "weight_train", "gt_token"),
            chosen("test_2000", "test", "gt_token"),
        ),
        "v64_train_pair_vs_test_2000_pair": intersection_count(
            chosen("v64_source_split", "weight_train", "pair_token"),
            chosen("test_2000", "test", "pair_token"),
        ),
        "confirmed_ancestor_candidate_vs_test_gt": intersection_count(
            chosen(None, "weight_train", "candidate_token"),
            chosen("test_2000", "test", "gt_token"),
        ),
        "confirmed_ancestor_gt_vs_test_gt": intersection_count(
            chosen(None, "weight_train", "gt_token"),
            chosen("test_2000", "test", "gt_token"),
        ),
        "confirmed_ancestor_spectrum_vs_test_spectrum": intersection_count(
            chosen(None, "weight_train", "spectrum_norm_hash"),
            chosen("test_2000", "test", "spectrum_norm_hash"),
        ),
    }
    return {
        "fingerprint_definition": STRUCTURE_OPTIONS,
        "rdkit_version": RDKIT_VERSION,
        "by_asset": summaries,
        "intersections": pairs,
        "test_2000_note": (
            "test_2000 text rows have no structured spectrum in the file. Their spectrum "
            "intersection is therefore not a physical-spectrum comparison. GT overlap uses "
            "_gt_smiles and is the meaningful isolation check."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-lmdb", action="store_true")
    parser.add_argument("--lmdb-limit", type=int, default=None)
    args = parser.parse_args()
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("[audit] lineage", flush=True)
    lineage = build_lineage()
    (out / "lineage_report.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[audit] index labeled assets", flush=True)
    records: list[dict[str, Any]] = []
    asset_notes: dict[str, Any] = {}
    roles = v64_roles()
    source = SPEC / "distill_out/trajectories_v6_3_filtered_balanced_observation_fixed.jsonl"
    asset_notes["v64_source"] = index_trajectory(source, "v64_source_split", "weight_train", records, roles)
    for path, asset, role in (
        (LF / "data/nmr_matching_tool_sft_v6_1.json", "v61_train", "weight_train"),
        (LF / "data/nmr_matching_tool_sft_v6_1_val.json", "v61_val", "weight_eval"),
        (LF / "data/nmr_matching_tool_sft_v5.json", "v5_train", "weight_train"),
        (LF / "data/nmr_matching_tool_sft_v5_val.json", "v5_val", "weight_eval"),
        (LF / "data/nmr_matching_tool_sft_v6.json", "v6_train_sibling", "sibling_train"),
        (LF / "data/nmr_matching_tool_sft_v6_2.json", "v62_train_sibling", "sibling_train"),
        (LF / "data/nmr_matching_tool_sft_v6_3.json", "v63_train_sibling", "sibling_train"),
    ):
        if not path.exists():
            asset_notes[asset] = {"exists": False, "path": str(path)}
            continue
        meta_path = {
            "v61_train": SPEC / "distill_out/trajectories_v6_1.jsonl",
            "v61_val": SPEC / "distill_out/trajectories_v6_1.jsonl",
            "v5_train": OA / "recipe/nmr_matching_agent/data_v5/tool_sft_v5.jsonl",
            "v5_val": OA / "recipe/nmr_matching_agent/data_v5/tool_sft_v5_val.jsonl",
        }.get(asset)
        meta = meta_index(meta_path) if meta_path and meta_path.exists() else None
        asset_notes[asset] = index_sharegpt(path, asset, role, records, meta)
        asset_notes[asset]["path"] = str(path)
    for path, asset in (
        (LF / "data/nmr_matching_cot_train.json", "cot_train_unknown_ancestor"),
        (LF / "data/nmr_matching_cot_val.json", "cot_val_unknown_ancestor"),
    ):
        if not path.exists():
            asset_notes[asset] = {"exists": False, "path": str(path)}
            continue
        rows = load_json(path)
        kept = 0
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                continue
            text = row.get("input") if isinstance(row.get("input"), str) else None
            parsed = parse_task_text(text)
            append_record(
                records,
                asset=asset,
                role="unknown_ancestor",
                locator=f"{path}:{index}",
                candidate=row.get("_candidate_smiles") or parsed["candidate"],
                gt=row.get("_gt_smiles") or row.get("gt_smiles") or None,
                label=row.get("_label", row.get("label")),
                negative_type=row.get("_negative_type") or row.get("negative_type"),
                spectrum=parsed["spectrum"],
            )
            kept += 1
        asset_notes[asset] = {
            "rows": kept,
            "path": str(path),
            "gt_recovery": "only when the JSON object itself has gt_smiles or _gt_smiles",
        }
    if (SPEC / "data/nmr_matching_test_2000.json").exists():
        asset_notes["test_2000"] = {"rows": index_alpaca(SPEC / "data/nmr_matching_test_2000.json", "test_2000", "test", records)}
    for path, asset in (
        (SPEC / "distill_out/distill_pool.jsonl", "v5_distill_pool"),
        (SPEC / "distill_out/v6_pool.jsonl", "v6_generation_pool"),
        (SPEC / "distill_out/v6_1_pool.jsonl", "v61_generation_pool"),
        (SPEC / "negative_output_1000/matching_1000_matching.jsonl", "pool_negative_1000"),
        (SPEC / "hard_negative_output_1000/hard_neg_1000_matching.jsonl", "pool_hard_negative_1000"),
        (SPEC / "perturbation_output_1000/perturbed_1000_matching_benchmark.jsonl", "pool_perturbation_1000"),
    ):
        if path.exists():
            asset_notes[asset] = {"rows": index_pool(path, asset, "generation_pool", records), "path": str(path)}
    index_path = out / "labeled_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[audit] labeled rows {len(records)}", flush=True)
    overlap = overlap_report(records)
    overlap["asset_notes"] = asset_notes
    (out / "overlap_report.json").write_text(json.dumps(overlap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    exposures = {
        "confirmed_train": sets_from(records, {"weight_train"}),
        "known_eval_and_test": sets_from(records, {"weight_eval", "test"}),
        "unknown_possible": sets_from(records, {"unknown_ancestor"}),
        "sibling": sets_from(records, {"sibling_train"}),
    }
    blind = {
        "unknown_ancestor_rows": sum(1 for record in records if record["role"] == "unknown_ancestor"),
        "weight_train_rows_missing_gt": sum(
            1 for record in records if record["role"] == "weight_train" and record["gt_status"] == "missing"
        ),
        "weight_train_rows_unparsed_candidate": sum(
            1 for record in records if record["role"] == "weight_train" and record["candidate_status"] == "unparsed"
        ),
        "minimum_additional_information": [
            "The dataset file and row membership that produced full-exp1/checkpoint-200.",
            "Parent GT for any weight-training row whose source meta could not be joined.",
            "A data card stating whether LMDB peaks are experimental and which nucleus nmr_frequency refers to.",
        ],
    }
    print("[audit] inputs", flush=True)
    preservation = input_audit(source)
    preservation["historical_format_sample"] = {}
    for path, kind in (
        (LF / "data/nmr_matching_cot_train.json", "json"),
        (LF / "data/nmr_matching_tool_sft_v6.json", "json"),
        (SPEC / "distill_out/trajectories_v6_1.jsonl", "jsonl"),
    ):
        if path.exists():
            preservation["historical_format_sample"][str(path)] = sample_historical(path, kind)
    preservation["sample_rule"] = stratified_sample_rule()
    (out / "input_preservation_report.json").write_text(
        json.dumps(preservation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    contract = {
        "status": "draft_not_implemented",
        "fields": [
            "candidate_smiles", "c_nmr_peaks", "h_nmr_peaks", "nmr_solvent",
            "nmr_frequency", "experimental_total_nH",
        ],
        "aliases_currently_implemented": {
            "candidate": "candidate_smiles",
            "experimental_13c_peaks": "c_nmr_peaks",
            "experimental_1h_peaks": "h_nmr_peaks",
            "solvent": "nmr_solvent",
            "nmr_freq": "nmr_frequency",
        },
        "h_peak_keys_seen_in_training_schema": [
            "category", "centroid", "delta", "j_values", "nH", "rangeMax", "rangeMin",
        ],
        "c_peak_key": "delta (ppm)",
        "pending": [
            "Whether each nmr_frequency string is the spectrometer's nominal 1H frequency or that nucleus's frequency.",
            "Whether a missing nH may be rendered as 1 by an Alpaca formatter. The current text builder does this; the audit does not change it.",
            "Promotion thresholds, ambiguity flags, and the positive/negative ratio of a future development set.",
        ],
        "suggested_conflict_behavior": "Keep the current step-01 behavior: one-sided fields are retained, equal repeats are retained, and explicit disagreements raise. Do not fill a null integral with zero.",
        "evidence": "input_preservation_report.json current_mixed_input_behavior and alpaca_roundtrip_loss",
    }
    (out / "input_contract_draft.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pool = {"lmdb": "skipped"}
    if not args.skip_lmdb:
        print("[audit] lmdb", flush=True)
        pool = lmdb_audit(out, exposures, args.lmdb_limit)
    (out / "candidate_pool_report.json").write_text(json.dumps({
        "exposures_are_counts_of_tokens": {name: {key: len(value) for key, value in sets.items()} for name, sets in exposures.items()},
        "blind_spot": blind,
        "lmdb": pool,
        "division_not_frozen": (
            "No search-dev or gate-dev split is selected. A later step can hold out two "
            "source-group disjoint subsets from remaining_conservative. Sizes, class balance, "
            "and ambiguity rules are pending."
        ),
        "generation_cost": (
            "Unpriced. A positive row can reuse a source spectrum. A negative row needs the "
            "existing generator; hard negatives additionally need the forward predictor. "
            "This audit does not choose how many negatives to build."
        ),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_paths = [
        source,
        SPEC / "distill_out/trajectories_v6_3_filtered_balanced.jsonl",
        SPEC / "distill_out/trajectories_v6_1.jsonl",
        SPEC / "data/nmr_matching_test_2000.json",
        LF / "data/nmr_matching_tool_sft_v6_4.json",
        LF / "data/nmr_matching_tool_sft_v6_4_val.json",
        LF / "data/nmr_matching_tool_sft_v6_4_split_report.json",
        LF / "data/nmr_matching_tool_sft_v6_1.json",
        LF / "data/nmr_matching_tool_sft_v5.json",
        LF / "data/nmr_matching_cot_train.json",
        LF / "examples/train_full/qwen3_32b_nmr_matching_tool_v6_4.yaml",
        SPEC / "v6_4_train.log",
        LMDB_ROOT / "all.lmdb",
        LMDB_ROOT / "train.lmdb",
        LMDB_ROOT / "valid.lmdb",
        LMDB_ROOT / "test.lmdb",
        index_path,
    ]
    manifest = {"files": [file_record(path) for path in manifest_paths], "labeled_index_rows": len(records)}
    (out / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.skip_lmdb:
        from step_02a_refine import explain_spectrum_gap, refine_run
        refine_run(out)
        explain_spectrum_gap(out)
    elapsed = round(time.time() - started, 1)
    (out / "audit_checks.txt").write_text(
        "\n".join([
            f"elapsed_sec {elapsed}",
            f"labeled_rows {len(records)}",
            f"rdkit {RDKIT_VERSION}",
            f"skip_lmdb {args.skip_lmdb}",
            "source files were opened read-only; this script writes only under --out",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"[audit] done {elapsed}s", flush=True)


if __name__ == "__main__":
    main()
