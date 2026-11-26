"""Offline NMR matching evaluator for step 02B-2.

Gold is the only label. A valid Yes/No terminal is the only prediction that
enters the primary denominator. Invalid terminals stay in the coverage counts
and out of balanced accuracy. This module does not load a model, a predictor,
verl, or transformers, and it does not enable promotion.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from step_02a_fingerprints import identity_token, structure_keys

from nmr_lossless_task import SCHEMA_VERSION as TASK_SCHEMA_VERSION
from nmr_lossless_task import normalize_task_input, task_to_lossless_text

EVALUATOR_VERSION = "step_02b2.v1"
DATASET_SCHEMA = "nmr-eval-dataset-v1"
GOLD_SCHEMA = "nmr-eval-gold-v1"
RUN_SCHEMA = "nmr-eval-run-v1"
PREDICTION_SCHEMA = "nmr-eval-prediction-v1"
COMPARISON_SCHEMA = "nmr-eval-comparison-v1"
CONFIG_SCHEMA = "step_02b2.config.v1"
SOURCE_GROUP_ID = "parent_gt_rdkit_isomeric_v1"
SOURCE_GROUP_VERSION = "step_02a.identity_token"
PRIMARY_METRIC = "balanced_accuracy_valid"
INVALID_POLICY = "exclude_from_primary"
TERMINAL_STATES = frozenset({"answer", "timeout", "max_turns", "empty_answer", "parse_error"})
STRICT_PROFILES = frozenset({"canonical", "standalone", "api"})
LEGACY_PROFILE = "legacy_diagnostic"
PROFILES = STRICT_PROFILES | {LEGACY_PROFILE}
TASK_ALIASES = frozenset({
    "candidate",
    "experimental_13c_peaks",
    "experimental_1h_peaks",
    "solvent",
    "nmr_freq",
})
GOLD_TASK_KEYS = frozenset({
    "_label",
    "label",
    "gold_label",
    "gt",
    "ground_truth",
    "_gt_smiles",
    "gt_smiles",
    "parent_gt_smiles",
    "source_group_id",
    "pair_id",
    "_pair_id",
    "run_id",
    "_harness_run_id",
    "_harness_pair_id",
})
ANSWER_BLOCK_RE = re.compile(r"<answer\b[^>]*>([\s\S]*?)</answer>", re.IGNORECASE)
PAIR_KEY_RE = re.compile(r"^(0|[1-9][0-9]*)$")
COST_FIELDS = (
    "tool_calls",
    "tool_cache_hits",
    "tool_network_calls",
    "tokens",
    "elapsed_seconds",
)
COST_UNITS = {
    "tool_calls": "calls",
    "tool_cache_hits": "calls",
    "tool_network_calls": "calls",
    "tokens": "tokens",
    "elapsed_seconds": "seconds",
}
COST_DEFINITIONS = {
    "tool_calls": "Each trace event with type tool counts once. An answer event is not a tool call.",
    "tool_cache_hits": "Tool events whose cache field is hit. Unknown cache status is not zero.",
    "tool_network_calls": "Tool events whose cache field is network. Tool name alone does not assign a cache class.",
    "tokens": "A recorded token count. String length is not a token count.",
    "elapsed_seconds": "A recorded elapsed time in seconds.",
}


class EvaluatorError(Exception):
    """A configuration or promotion refusal. Integrity failures are reports."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class PromotionRefused(EvaluatorError):
    def __init__(self, detail: str):
        super().__init__("promotion_refused", detail)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pack_fraction(value: Fraction | None, reason: str | None = None) -> dict[str, Any]:
    if value is None:
        return {"numerator": None, "denominator": None, "fraction": None, "reason": reason}
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "fraction": f"{value.numerator}/{value.denominator}",
        "reason": None,
    }


def _error(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _is_yes_no_text(value: Any) -> tuple[str, int | None, str | None]:
    """Return (status, label, reason). Only a Yes/No string is a valid label."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return "invalid", None, "non_binary_answer"
    if value is None:
        return "invalid", None, "empty_answer"
    if not isinstance(value, str):
        return "invalid", None, "non_binary_answer"
    token = value.strip().casefold()
    if token == "":
        return "invalid", None, "empty_answer"
    if token == "yes":
        return "valid", 1, None
    if token == "no":
        return "valid", 0, None
    return "invalid", None, "non_binary_answer"


def _json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None, "parse_error"
    if not isinstance(loaded, dict):
        return None, "unsupported_format"
    return loaded, None


def _answer_blocks(text: str) -> list[str]:
    return [match.group(1).strip() for match in ANSWER_BLOCK_RE.finditer(text)]


def _object_final(obj: Mapping[str, Any]) -> tuple[str, int | None, str | None]:
    if "final_answer" not in obj:
        return "invalid", None, "empty_answer"
    return _is_yes_no_text(obj.get("final_answer"))


def parse_answer_container(value: Any) -> tuple[str, int | None, str | None]:
    """Parse one explicit answer container.

    A dict, a complete JSON object string, or one complete answer block is
    accepted. Truncated JSON, a second answer block, and prose Yes/No are not
    repaired or searched.
    """
    if isinstance(value, dict):
        return _object_final(value)
    if isinstance(value, bool) or isinstance(value, (int, float)) or value is None:
        return _is_yes_no_text(value)
    if not isinstance(value, str):
        return "invalid", None, "unsupported_format"
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        obj, reason = _json_object(stripped)
        if obj is None:
            return "invalid", None, reason or "parse_error"
        return _object_final(obj)
    blocks = _answer_blocks(stripped)
    if len(blocks) > 1:
        return "invalid", None, "multiple_answers"
    if len(blocks) == 1:
        obj, reason = _json_object(blocks[0])
        if obj is None:
            return "invalid", None, "parse_error" if reason == "parse_error" else "unsupported_format"
        return _object_final(obj)
    if "<answer" in stripped.casefold() or stripped.startswith("{"):
        return "invalid", None, "parse_error"
    return "invalid", None, "unsupported_format"


def _terminal_from_state(state: Any, final_answer: Any, answer_present: bool) -> dict[str, Any]:
    if not isinstance(state, str) or state not in TERMINAL_STATES:
        return {"valid": False, "pred_label": None, "invalid_reason": "unsupported_format", "diagnostic": {}}
    status, label, reason = _is_yes_no_text(final_answer) if answer_present else ("invalid", None, "empty_answer")
    legal_answer = status == "valid"
    if state == "answer":
        if legal_answer:
            return {"valid": True, "pred_label": label, "invalid_reason": None, "diagnostic": {}}
        return {"valid": False, "pred_label": None, "invalid_reason": reason or "empty_answer", "diagnostic": {}}
    if legal_answer:
        return {
            "valid": False,
            "pred_label": None,
            "invalid_reason": "terminal_conflict",
            "diagnostic": {"claimed_failure": state, "claimed_answer": final_answer},
        }
    return {"valid": False, "pred_label": None, "invalid_reason": state, "diagnostic": {}}


def parse_canonical_terminal(terminal: Any) -> dict[str, Any]:
    if isinstance(terminal, str):
        obj, reason = _json_object(terminal.strip())
        if obj is None:
            return {"valid": False, "pred_label": None, "invalid_reason": reason or "parse_error", "diagnostic": {}}
        terminal = obj
    if not isinstance(terminal, dict):
        return {"valid": False, "pred_label": None, "invalid_reason": "unsupported_format", "diagnostic": {}}
    return _terminal_from_state(terminal.get("state"), terminal.get("final_answer"), "final_answer" in terminal)


def parse_standalone_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Read answer and fail_reason. Transcript prose is not an answer."""
    fail_reason = row.get("fail_reason")
    answer_present = "answer" in row and row.get("answer") not in (None, "")
    status, label, reason = parse_answer_container(row.get("answer")) if answer_present else ("invalid", None, "empty_answer")
    legal = status == "valid"
    if fail_reason not in (None, ""):
        if not isinstance(fail_reason, str):
            return {"valid": False, "pred_label": None, "invalid_reason": "unsupported_format", "diagnostic": {}}
        if legal:
            return {
                "valid": False,
                "pred_label": None,
                "invalid_reason": "terminal_conflict",
                "diagnostic": {"fail_reason": fail_reason},
            }
        mapped = fail_reason if fail_reason in TERMINAL_STATES or reason == "empty_answer" else fail_reason
        if not answer_present:
            mapped = fail_reason
        return {"valid": False, "pred_label": None, "invalid_reason": mapped, "diagnostic": {}}
    if legal:
        diagnostic = {}
        trace = row.get("trace")
        if isinstance(trace, list) and any(isinstance(item, dict) and item.get("error") for item in trace):
            diagnostic["tool_error_before_answer"] = True
        return {"valid": True, "pred_label": label, "invalid_reason": None, "diagnostic": diagnostic}
    return {"valid": False, "pred_label": None, "invalid_reason": reason or "empty_answer", "diagnostic": {}}


def parse_api_response(response: Any) -> dict[str, Any]:
    if isinstance(response, str):
        obj, reason = _json_object(response.strip())
        if obj is None:
            return {"valid": False, "pred_label": None, "invalid_reason": reason or "parse_error", "diagnostic": {}}
        response = obj
    if not isinstance(response, dict):
        return {"valid": False, "pred_label": None, "invalid_reason": "unsupported_format", "diagnostic": {}}
    if "final_answer" not in response:
        return {"valid": False, "pred_label": None, "invalid_reason": "empty_answer", "diagnostic": {}}
    status, label, reason = _is_yes_no_text(response.get("final_answer"))
    trajectory = response.get("trajectory") or []
    others = []
    if isinstance(trajectory, list):
        for turn in trajectory:
            if isinstance(turn, dict) and "final_answer" in turn and turn.get("final_answer") is not None:
                others.append(turn.get("final_answer"))
    if status != "valid":
        return {"valid": False, "pred_label": None, "invalid_reason": reason or "empty_answer", "diagnostic": {}}
    for other in others:
        other_status, other_label, _other_reason = _is_yes_no_text(other)
        if other_status != "valid" or other_label != label:
            return {"valid": False, "pred_label": None, "invalid_reason": "terminal_conflict", "diagnostic": {}}
    return {"valid": True, "pred_label": label, "invalid_reason": None, "diagnostic": {}}


def _pair_key(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value < 0:
            return None
        return str(value)
    if isinstance(value, str) and PAIR_KEY_RE.fullmatch(value):
        return str(int(value))
    return None


def _record_id(value: Any) -> str | None:
    if isinstance(value, str) and value.strip() and value == value.strip():
        return value
    return None


def _gold_label(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        return None
    return value


def source_group_for_parent(smiles: str) -> str:
    """Reuse the audited 02A identity token. This does not strip salts or stereo."""
    return identity_token(structure_keys(smiles))


def _task_key_errors(task: Mapping[str, Any]) -> list[dict[str, str]]:
    errors = []
    for key in task:
        if key in GOLD_TASK_KEYS:
            errors.append(_error("gold_in_task", key))
    return errors


def validate_task(task: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None, list[dict[str, str]]]:
    errors = _task_key_errors(task)
    if errors:
        return None, None, errors
    try:
        normalized = normalize_task_input(task)
        digest = sha256_text(task_to_lossless_text(normalized))
    except Exception as exc:
        return None, None, [_error("illegal_task", f"{type(exc).__name__}: {exc}")]
    return normalized, digest, []


def _definition_ok(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("id") == SOURCE_GROUP_ID
        and value.get("version") == SOURCE_GROUP_VERSION
    )


def _semantic_record(record: Mapping[str, Any], task_sha: str) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "task_sha256": task_sha,
        "gold_label": record["gold_label"],
        "source_group_id": record["source_group_id"],
        "parent_gt_smiles": record["parent_gt_smiles"],
        "difficulty": record.get("difficulty"),
        "ambiguity": record.get("ambiguity"),
    }


def dataset_semantic_body(manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "dataset_id": manifest["dataset_id"],
        "task_schema_version": TASK_SCHEMA_VERSION,
        "source_group_definition": {
            "id": SOURCE_GROUP_ID,
            "version": SOURCE_GROUP_VERSION,
        },
        "records": list(records),
    }


def stamp_hash(body: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(body))


def _load_task(record: Mapping[str, Any], manifest_path: Path) -> tuple[Mapping[str, Any] | None, list[dict[str, str]]]:
    if "task" in record and "task_ref" in record:
        return None, [_error("illegal_task", f"{record.get('record_id')}: both task and task_ref")]
    if "task" in record:
        task = record.get("task")
        if not isinstance(task, dict):
            return None, [_error("illegal_task", f"{record.get('record_id')}: task is not an object")]
        return task, []
    ref = record.get("task_ref")
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not isinstance(ref.get("sha256"), str):
        return None, [_error("illegal_task", f"{record.get('record_id')}: missing task")]
    path = (manifest_path.parent / ref["path"]).resolve()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, [_error("illegal_task", f"{record.get('record_id')}: {exc}")]
    file_hash = hashlib.sha256(raw).hexdigest()
    if file_hash != ref["sha256"]:
        return None, [_error(
            "dataset_hash_mismatch",
            f"{record.get('record_id')}: task_ref bytes hash {file_hash} != {ref['sha256']}",
        )]
    try:
        task = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        return None, [_error("illegal_task", f"{record.get('record_id')}: {exc}")]
    if not isinstance(task, dict):
        return None, [_error("illegal_task", f"{record.get('record_id')}: task_ref is not an object")]
    return task, []


def prepare_dataset(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    verify_hash: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if manifest.get("schema_version") != DATASET_SCHEMA:
        errors.append(_error("schema_mismatch", f"dataset schema {manifest.get('schema_version')!r}"))
    if not isinstance(manifest.get("dataset_id"), str) or not manifest.get("dataset_id"):
        errors.append(_error("schema_mismatch", "dataset_id is missing"))
    if not _definition_ok(manifest.get("source_group_definition")):
        errors.append(_error("unsupported_source_group_definition", json.dumps(manifest.get("source_group_definition"), default=str)))
    rows = manifest.get("records")
    if not isinstance(rows, list):
        return None, errors + [_error("schema_mismatch", "records is not a list")]
    if len(rows) == 0:
        errors.append(_error("empty_evaluation_set", "dataset has no records"))
        return None, errors
    semantic = []
    seen: set[str] = set()
    prepared_rows = []
    for index, record in enumerate(rows):
        if not isinstance(record, dict):
            errors.append(_error("schema_mismatch", f"record {index} is not an object"))
            continue
        record_id = _record_id(record.get("record_id"))
        if record_id is None:
            errors.append(_error("schema_mismatch", f"record {index} has no stable record_id"))
            continue
        if record_id in seen:
            errors.append(_error("duplicate_record_id", record_id))
            continue
        seen.add(record_id)
        gold = _gold_label(record.get("gold_label"))
        if gold is None:
            errors.append(_error("illegal_gold", f"{record_id}: {record.get('gold_label')!r}"))
        parent = record.get("parent_gt_smiles")
        if not isinstance(parent, str) or parent == "":
            errors.append(_error("missing_source_group", record_id))
            expected_group = None
        else:
            expected_group = source_group_for_parent(parent)
            if record.get("source_group_id") != expected_group:
                errors.append(_error(
                    "source_group_mismatch",
                    f"{record_id}: {record.get('source_group_id')!r} != {expected_group}",
                ))
        task, task_errors = _load_task(record, manifest_path)
        errors.extend(task_errors)
        task_sha = None
        if task is not None:
            _normalized, task_sha, task_value_errors = validate_task(task)
            errors.extend(_error(item["code"], f"{record_id}: {item['detail']}") for item in task_value_errors)
            if isinstance(record.get("task_sha256"), str) and task_sha is not None and record["task_sha256"] != task_sha:
                errors.append(_error("dataset_hash_mismatch", f"{record_id}: stored task_sha256 does not match the normalized task"))
        if gold is None or expected_group is None or task_sha is None:
            continue
        item = {
            "record_id": record_id,
            "gold_label": gold,
            "source_group_id": expected_group,
            "parent_gt_smiles": parent,
            "difficulty": record.get("difficulty"),
            "ambiguity": record.get("ambiguity"),
            "task_sha256": task_sha,
            "task_hash_source": "dataset_manifest.task normalized by " + TASK_SCHEMA_VERSION,
        }
        prepared_rows.append(item)
        semantic.append(_semantic_record(item, task_sha))
    if errors:
        return None, errors
    semantic.sort(key=lambda row: row["record_id"])
    body = dataset_semantic_body(manifest, semantic)
    digest = stamp_hash(body)
    if verify_hash and manifest.get("content_sha256") != digest:
        return None, [_error("dataset_hash_mismatch", f"recomputed {digest}")]
    return {
        "dataset_id": manifest["dataset_id"],
        "content_sha256": digest,
        "records": prepared_rows,
        "body": body,
    }, []


def _gold_body(sidecar: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": GOLD_SCHEMA,
        "dataset_id": sidecar["dataset_id"],
        "records": list(records),
    }


def check_gold_sidecar(
    sidecar: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> tuple[str | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if sidecar.get("schema_version") != GOLD_SCHEMA:
        errors.append(_error("schema_mismatch", f"gold schema {sidecar.get('schema_version')!r}"))
    if sidecar.get("dataset_id") != dataset["dataset_id"]:
        errors.append(_error("gold_conflict", "gold dataset_id does not match the dataset manifest"))
    rows = sidecar.get("records")
    if not isinstance(rows, list):
        return None, errors + [_error("schema_mismatch", "gold records is not a list")]
    semantic = []
    seen: set[str] = set()
    by_dataset = {row["record_id"]: row for row in dataset["records"]}
    for record in rows:
        if not isinstance(record, dict):
            errors.append(_error("schema_mismatch", "gold record is not an object"))
            continue
        record_id = _record_id(record.get("record_id"))
        if record_id is None:
            errors.append(_error("schema_mismatch", "gold record_id is missing"))
            continue
        if record_id in seen:
            errors.append(_error("duplicate_record_id", f"gold {record_id}"))
            continue
        seen.add(record_id)
        gold = _gold_label(record.get("gold_label"))
        if gold is None:
            errors.append(_error("illegal_gold", f"sidecar {record_id}: {record.get('gold_label')!r}"))
            continue
        parent = record.get("parent_gt_smiles")
        if not isinstance(parent, str) or parent == "":
            errors.append(_error("missing_source_group", f"sidecar {record_id}"))
            continue
        group = source_group_for_parent(parent)
        if record.get("source_group_id") != group:
            errors.append(_error("source_group_mismatch", f"sidecar {record_id}"))
        dataset_row = by_dataset.get(record_id)
        if dataset_row is None:
            errors.append(_error("extra_prediction", f"gold record {record_id} is not in the dataset"))
            continue
        if (
            dataset_row["gold_label"] != gold
            or dataset_row["source_group_id"] != group
            or dataset_row["parent_gt_smiles"] != parent
        ):
            errors.append(_error("gold_conflict", f"{record_id}: sidecar does not match the dataset manifest"))
        semantic.append({
            "record_id": record_id,
            "gold_label": gold,
            "source_group_id": group,
            "parent_gt_smiles": parent,
        })
    missing = sorted(set(by_dataset) - seen)
    for record_id in missing:
        errors.append(_error("illegal_gold", f"sidecar missing {record_id}"))
    if errors:
        return None, errors
    semantic.sort(key=lambda row: row["record_id"])
    digest = stamp_hash(_gold_body(sidecar, semantic))
    if verify_hash and sidecar.get("content_sha256") != digest:
        return None, [_error("gold_hash_mismatch", f"recomputed {digest}")]
    return digest, []


def load_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise EvaluatorError("schema_mismatch", f"config schema {config.get('schema_version')!r}")
    if config.get("evaluator_version") != EVALUATOR_VERSION:
        raise EvaluatorError("evaluator_version_mismatch", str(config.get("evaluator_version")))
    if config.get("invalid_policy") != INVALID_POLICY:
        raise EvaluatorError("invalid_policy_rejected", str(config.get("invalid_policy")))
    if config.get("primary_metric") != PRIMARY_METRIC:
        raise EvaluatorError("primary_metric_rejected", str(config.get("primary_metric")))
    promotion = config.get("promotion")
    if not isinstance(promotion, dict) or "enabled" not in promotion:
        raise EvaluatorError("promotion_config_missing", "promotion.enabled is required and must be false")
    if promotion.get("enabled") is not False:
        raise PromotionRefused("promotion.enabled must stay false; this step has no promotion protocol")
    statistics = config.get("statistics", {"enabled": False})
    if not isinstance(statistics, dict):
        raise EvaluatorError("statistics_config_invalid", "statistics must be an object")
    if statistics.get("enabled") is False:
        parsed_statistics: dict[str, Any] = {"enabled": False}
    elif statistics.get("enabled") is True:
        method = statistics.get("method")
        replicates = statistics.get("bootstrap_replicates")
        level = statistics.get("confidence_level")
        seed = statistics.get("seed")
        if method != "paired_cluster_percentile":
            raise EvaluatorError("statistics_config_invalid", f"method {method!r} is not implemented")
        if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
            raise EvaluatorError("statistics_config_invalid", "bootstrap_replicates must be an integer >= 1")
        if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0 < float(level) < 1:
            raise EvaluatorError("statistics_config_invalid", "confidence_level must be explicitly between 0 and 1")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvaluatorError("statistics_config_invalid", "seed must be an integer")
        parsed_statistics = {
            "enabled": True,
            "method": method,
            "bootstrap_replicates": replicates,
            "confidence_level": float(level),
            "seed": seed,
            "protocol_status": "synthetic_not_a_formal_threshold",
        }
    else:
        raise EvaluatorError("statistics_config_invalid", "statistics.enabled must be boolean")
    return {
        "schema_version": CONFIG_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "invalid_policy": INVALID_POLICY,
        "primary_metric": PRIMARY_METRIC,
        "promotion": {"enabled": False},
        "statistics": parsed_statistics,
        "example_status": config.get("example_status"),
        "content_sha256": stamp_hash({
            "schema_version": CONFIG_SCHEMA,
            "evaluator_version": EVALUATOR_VERSION,
            "invalid_policy": INVALID_POLICY,
            "primary_metric": PRIMARY_METRIC,
            "promotion": {"enabled": False},
            "statistics": parsed_statistics,
            "example_status": config.get("example_status"),
        }),
    }


def _prediction_identity(row: Mapping[str, Any], envelope_run: str | None) -> tuple[str | None, str | None, str | None, list[dict[str, str]]]:
    errors = []
    run_id = row.get("run_id", envelope_run)
    if not isinstance(run_id, str) or not run_id:
        errors.append(_error("unbound_run_id", "prediction has no run_id"))
        run_id = None
    pair_id = _pair_key(row.get("pair_id", row.get("_pair_id")))
    if pair_id is None:
        errors.append(_error("unbound_pair_id", f"pair_id {row.get('pair_id', row.get('_pair_id'))!r}"))
    record_id = row.get("record_id")
    if record_id is not None:
        record_id = _record_id(record_id)
        if record_id is None:
            errors.append(_error("schema_mismatch", "prediction record_id is not a stable string"))
    return run_id, pair_id, record_id, errors


def _residual_gold(row: Mapping[str, Any], gold: int) -> list[dict[str, str]]:
    errors = []
    for key in ("_label", "gold_label", "gt"):
        if key in row:
            if _gold_label(row.get(key)) != gold:
                errors.append(_error("prediction_gold_conflict", f"{key}={row.get(key)!r} disagrees with sidecar gold {gold}"))
    return errors


def _nonnegative_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value >= 0 and value == value and value not in (float("inf"), float("-inf")):
            return value
    return None


def cost_from_trace(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, list):
        return {"tool_calls": None, "tool_cache_hits": None, "tool_network_calls": None, "trace_conflict": True}
    tool_calls = 0
    hits = 0
    network = 0
    unknown_cache = False
    for event in trace:
        if not isinstance(event, dict):
            return {"tool_calls": None, "tool_cache_hits": None, "tool_network_calls": None, "trace_conflict": True}
        kind = event.get("type")
        if kind == "answer":
            continue
        if kind != "tool":
            return {"tool_calls": None, "tool_cache_hits": None, "tool_network_calls": None, "trace_conflict": True}
        tool_calls += 1
        cache = event.get("cache")
        if cache == "hit":
            hits += 1
        elif cache == "network":
            network += 1
        else:
            unknown_cache = True
    return {
        "tool_calls": tool_calls,
        "tool_cache_hits": None if unknown_cache else hits,
        "tool_network_calls": None if unknown_cache else network,
        "trace_conflict": False,
    }


def _row_cost(row: Mapping[str, Any]) -> dict[str, Any]:
    explicit = row.get("cost") if isinstance(row.get("cost"), dict) else {}
    traced = cost_from_trace(row.get("trace")) if "trace" in row else None
    cost: dict[str, Any] = {}
    for field in COST_FIELDS:
        if field in ("tool_calls", "tool_cache_hits", "tool_network_calls") and traced is not None:
            traced_value = traced.get(field)
            if field in explicit:
                explicit_value = _nonnegative_number(explicit.get(field))
                if explicit_value is None or traced.get("trace_conflict") or explicit_value != traced_value:
                    cost[field] = None
                    cost[field + "_availability"] = "conflict"
                    continue
            if traced.get("trace_conflict"):
                cost[field] = None
                cost[field + "_availability"] = "conflict"
                continue
            cost[field] = traced_value
            cost[field + "_availability"] = "recorded" if traced_value is not None else "not_recorded"
            continue
        if field not in explicit:
            cost[field] = None
            cost[field + "_availability"] = "not_recorded"
            continue
        parsed = _nonnegative_number(explicit.get(field))
        if parsed is None:
            cost[field] = None
            cost[field + "_availability"] = "conflict"
        else:
            cost[field] = parsed
            cost[field + "_availability"] = "recorded"
    return cost


def parse_prediction_row(row: Mapping[str, Any], profile: str) -> dict[str, Any]:
    if profile == "canonical":
        parsed = parse_canonical_terminal(row.get("terminal"))
    elif profile == "standalone":
        parsed = parse_standalone_row(row)
    elif profile == "api":
        parsed = parse_api_response(row.get("response"))
    else:
        parsed = parse_canonical_terminal(row.get("terminal"))
    parsed["cost"] = _row_cost(row)
    if isinstance(row.get("observed_task_sha256"), str):
        parsed["observed_task_sha256"] = row["observed_task_sha256"]
    else:
        parsed["observed_task_sha256"] = None
    return parsed


def _envelope(document: Any) -> tuple[str | None, str | None, list[Any], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if isinstance(document, list):
        return None, None, document, [_error("profile_missing", "prediction document has no profile")]
    if not isinstance(document, dict):
        return None, None, [], [_error("schema_mismatch", "predictions are not an object")]
    profile = document.get("profile")
    if profile not in PROFILES:
        errors.append(_error("profile_unsupported" if profile is not None else "profile_missing", str(profile)))
    rows = document.get("predictions")
    if not isinstance(rows, list):
        errors.append(_error("schema_mismatch", "predictions is not a list"))
        rows = []
    run_id = document.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        errors.append(_error("unbound_run_id", "envelope run_id is not a string"))
        run_id = None
    return profile if isinstance(profile, str) else None, run_id, rows, errors


def align_predictions(
    dataset: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    document: Any,
    *,
    legacy: bool,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if run_manifest.get("schema_version") != RUN_SCHEMA:
        errors.append(_error("schema_mismatch", f"run schema {run_manifest.get('schema_version')!r}"))
    run_id = run_manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        errors.append(_error("unbound_run_id", "run manifest run_id is missing"))
    if run_manifest.get("dataset_id") != dataset["dataset_id"]:
        errors.append(_error("run_dataset_hash_mismatch", "run dataset_id does not match"))
    if run_manifest.get("dataset_content_sha256") != dataset["content_sha256"]:
        errors.append(_error("run_dataset_hash_mismatch", "run dataset hash does not match the dataset manifest"))
    if run_manifest.get("evaluator_version") != EVALUATOR_VERSION:
        errors.append(_error("evaluator_version_mismatch", str(run_manifest.get("evaluator_version"))))
    profile, envelope_run, rows, envelope_errors = _envelope(document)
    errors.extend(envelope_errors)
    if legacy:
        if profile != LEGACY_PROFILE:
            errors.append(_error("profile_unsupported", "legacy diagnostic requires profile legacy_diagnostic"))
    elif profile == LEGACY_PROFILE:
        errors.append(_error("legacy_not_formal", "legacy diagnostic cannot publish a formal primary score"))
    elif profile not in STRICT_PROFILES:
        errors.append(_error("profile_unsupported", str(profile)))
    by_id = {row["record_id"]: row for row in dataset["records"]}
    if legacy:
        if errors:
            return None, None, errors
        if len(rows) != len(dataset["records"]):
            return None, None, [_error("missing_prediction", f"legacy length {len(rows)} != {len(dataset['records'])}")]
        aligned = []
        file_order = [row["record_id"] for row in dataset["records"]]
        for record, prediction in zip(dataset["records"], rows):
            if not isinstance(prediction, dict):
                errors.append(_error("schema_mismatch", f"legacy row {record['record_id']} is not an object"))
                continue
            parsed = parse_prediction_row(prediction, "canonical")
            aligned.append({**record, **parsed, "pair_id": None, "run_id": run_id, "pairing_verified": False})
        if errors:
            return None, None, errors
        return aligned, {"profile": LEGACY_PROFILE, "pairing_verified": False, "file_order_record_ids": file_order}, []
    pair_map_raw = run_manifest.get("pair_to_record")
    pair_map: dict[str, str] = {}
    if not isinstance(pair_map_raw, dict):
        errors.append(_error("pair_map_incomplete", "pair_to_record is missing"))
    else:
        for key, record_id in pair_map_raw.items():
            pair_id = _pair_key(key)
            record_key = _record_id(record_id)
            if pair_id is None or record_key is None or record_key not in by_id:
                errors.append(_error("pair_map_incomplete", f"{key!r} -> {record_id!r}"))
                continue
            if pair_id in pair_map:
                errors.append(_error("duplicate_pair_id", pair_id))
                continue
            if record_key in pair_map.values():
                errors.append(_error("duplicate_record_id", f"pair map repeats {record_key}"))
                continue
            pair_map[pair_id] = record_key
        missing_ids = sorted(set(by_id) - set(pair_map.values()))
        for record_id in missing_ids:
            errors.append(_error("pair_map_incomplete", f"missing record {record_id}"))
    if errors:
        return None, None, errors
    aligned = []
    seen_records: set[str] = set()
    seen_pairs: set[str] = set()
    for prediction in rows:
        if not isinstance(prediction, dict):
            errors.append(_error("schema_mismatch", "prediction is not an object"))
            continue
        row_run, pair_id, stated_record, identity_errors = _prediction_identity(prediction, envelope_run if isinstance(envelope_run, str) else None)
        errors.extend(identity_errors)
        if row_run is not None and row_run != run_id:
            errors.append(_error("mixed_run_id", f"{row_run} != {run_id}"))
        if pair_id is None:
            continue
        if pair_id in seen_pairs:
            errors.append(_error("duplicate_prediction", f"pair {pair_id}"))
            continue
        seen_pairs.add(pair_id)
        mapped = pair_map.get(pair_id)
        if mapped is None:
            errors.append(_error("extra_prediction", f"pair {pair_id}"))
            continue
        if stated_record is not None and stated_record != mapped:
            errors.append(_error("pair_record_mismatch", f"pair {pair_id}: {stated_record} != {mapped}"))
            continue
        if mapped in seen_records:
            errors.append(_error("duplicate_prediction", mapped))
            continue
        seen_records.add(mapped)
        record = by_id[mapped]
        errors.extend(_residual_gold(prediction, record["gold_label"]))
        parsed = parse_prediction_row(prediction, profile or "")
        observed = parsed.get("observed_task_sha256")
        if observed is not None and observed != record["task_sha256"]:
            errors.append(_error("observed_task_hash_mismatch", f"{mapped}: observation does not match the manifest task hash"))
        aligned.append({
            **record,
            **parsed,
            "pair_id": pair_id,
            "run_id": run_id,
            "pairing_verified": True,
        })
    for record_id in sorted(set(by_id) - seen_records):
        errors.append(_error("missing_prediction", record_id))
    if errors:
        return None, None, errors
    return aligned, {"profile": profile, "pairing_verified": True, "run_id": run_id}, []


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, int]]:
    counts = {0: {"N": 0, "V": 0, "I": 0}, 1: {"N": 0, "V": 0, "I": 0}}
    for row in rows:
        gold = int(row["gold_label"])
        counts[gold]["N"] += 1
        if row["valid"]:
            counts[gold]["V"] += 1
        else:
            counts[gold]["I"] += 1
    return counts


def classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one multiset of candidate rows. Group weight is already expanded."""
    counts = _label_counts(rows)
    n = len(rows)
    v = sum(1 for row in rows if row["valid"])
    invalid = n - v
    matrix = [[0, 0], [0, 0]]
    outcome = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        gold = int(row["gold_label"])
        if not row["valid"]:
            outcome[gold][2] += 1
            continue
        pred = int(row["pred_label"])
        matrix[gold][pred] += 1
        outcome[gold][pred] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    if v == 0:
        empty = "no_valid_predictions"
        recall_no = recall_yes = accuracy = f1_no = f1_yes = macro = ba = None
        reasons = {
            "recall_no_valid": empty,
            "recall_yes_valid": empty,
            "balanced_accuracy_valid": empty,
            "accuracy_valid": empty,
            "f1_no_valid": empty,
            "f1_yes_valid": empty,
            "macro_f1_valid": empty,
        }
        status = empty
    else:
        reasons = {}
        recall_no = Fraction(tn, tn + fp) if counts[0]["V"] else None
        recall_yes = Fraction(tp, tp + fn) if counts[1]["V"] else None
        if counts[0]["V"] == 0:
            reasons["recall_no_valid"] = "zero_valid_class"
        if counts[1]["V"] == 0:
            reasons["recall_yes_valid"] = "zero_valid_class"
        accuracy = Fraction(tn + tp, v)
        if recall_no is None or recall_yes is None:
            ba = None
            macro = None
            reasons["balanced_accuracy_valid"] = "missing_valid_class"
            reasons["macro_f1_valid"] = "missing_valid_class"
            f1_no = Fraction(2 * tn, 2 * tn + fp + fn) if (2 * tn + fp + fn) else None
            f1_yes = Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else None
            if f1_no is None:
                reasons["f1_no_valid"] = "zero_valid_class"
            if f1_yes is None:
                reasons["f1_yes_valid"] = "zero_valid_class"
            status = "missing_valid_class"
        else:
            ba = (recall_yes + recall_no) / 2
            f1_yes = Fraction(2 * tp, 2 * tp + fp + fn)
            f1_no = Fraction(2 * tn, 2 * tn + fp + fn)
            macro = (f1_yes + f1_no) / 2
            status = "scored"
    def pack(name: str, value: Fraction | None) -> dict[str, Any]:
        return pack_fraction(value, reasons.get(name))

    return {
        "N": n,
        "V": v,
        "I": invalid,
        "coverage": pack_fraction(Fraction(v, n) if n else None, None if n else "empty_evaluation_set"),
        "invalid_rate": pack_fraction(Fraction(invalid, n) if n else None, None if n else "empty_evaluation_set"),
        "by_label": {
            "0": {**counts[0], "coverage": pack_fraction(Fraction(counts[0]["V"], counts[0]["N"]) if counts[0]["N"] else None, None if counts[0]["N"] else "zero_gold_class")},
            "1": {**counts[1], "coverage": pack_fraction(Fraction(counts[1]["V"], counts[1]["N"]) if counts[1]["N"] else None, None if counts[1]["N"] else "zero_gold_class")},
        },
        "confusion_matrix_valid": matrix,
        "confusion_matrix_valid_sum": v,
        "gold_by_outcome_columns": ["No", "Yes", "invalid"],
        "gold_by_outcome": outcome,
        "gold_by_outcome_sum": n,
        "recall_no_valid": pack("recall_no_valid", recall_no),
        "recall_yes_valid": pack("recall_yes_valid", recall_yes),
        "balanced_accuracy_valid": pack("balanced_accuracy_valid", ba),
        "accuracy_valid": pack("accuracy_valid", accuracy),
        "f1_no_valid": pack("f1_no_valid", f1_no),
        "f1_yes_valid": pack("f1_yes_valid", f1_yes),
        "macro_f1_valid": pack("macro_f1_valid", macro),
        "status": status,
    }


def _fraction_from_pack(pack: Mapping[str, Any]) -> Fraction | None:
    if pack.get("numerator") is None or pack.get("denominator") in (None, 0):
        return None
    return Fraction(int(pack["numerator"]), int(pack["denominator"]))


def _bucket(value: Any, kind: str) -> str:
    if value is None or value == "":
        return "unknown" if kind == "difficulty" else "unannotated"
    return str(value)


def grouped_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {"difficulty": {}, "ambiguity": {}}
    for row in rows:
        for kind in ("difficulty", "ambiguity"):
            key = _bucket(row.get(kind), kind)
            grouped[kind].setdefault(key, []).append(row)
    return {
        kind: {key: classification_metrics(members) for key, members in sorted(buckets.items())}
        for kind, buckets in grouped.items()
    }


def aggregate_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    report = {}
    total = len(rows)
    for field in COST_FIELDS:
        known = []
        conflict = 0
        for row in rows:
            availability = row["cost"].get(field + "_availability")
            if availability == "conflict":
                conflict += 1
                continue
            value = row["cost"].get(field)
            if value is not None:
                known.append(value)
        known_count = len(known)
        complete = conflict == 0 and known_count == total and total > 0
        if conflict:
            availability = "conflict"
            known_sum = None
        elif known_count == 0:
            availability = "not_recorded"
            known_sum = None
        elif complete:
            availability = "complete"
            known_sum = sum(known)
        else:
            availability = "partial"
            known_sum = sum(known)
        report[field] = {
            "known": known_count,
            "unknown": total - known_count - conflict,
            "conflict": conflict,
            "total_attempts": total,
            "unit": COST_UNITS[field],
            "definition": COST_DEFINITIONS[field],
            "known_sum": known_sum,
            "complete": complete,
            "availability": availability,
            "total_cost": known_sum if complete else None,
            "denominator": "all_attempted_records_including_invalid",
        }
    return report


def expand_group_draw(rows: Sequence[Mapping[str, Any]], draw: Sequence[str]) -> tuple[list[dict[str, Any]] | None, list[dict[str, str]]]:
    known = {row["source_group_id"] for row in rows}
    errors = [_error("missing_source_group", group_id) for group_id in draw if group_id not in known]
    if errors:
        return None, errors
    weights = Counter(draw)
    expanded: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["record_id"]):
        copies = weights[row["source_group_id"]]
        expanded.extend([dict(row) for _ in range(copies)])
    return expanded, []


def group_draw_indices(n_groups: int, replicates: int, seed: int) -> list[list[int]]:
    """Draw whole-group indices with replacement.

    The generator is random.Random(seed). The group at index i is the i-th
    lexicographic source_group_id. Row order does not enter this function.
    """
    rng = random.Random(seed)
    return [[rng.randrange(n_groups) for _ in range(n_groups)] for _ in range(replicates)]


def _percentile(values: Sequence[Fraction], probability: float) -> Fraction:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = Fraction(str(probability)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _interval(values: Sequence[Fraction], level: float) -> dict[str, Any]:
    alpha = (1 - level) / 2
    low = _percentile(values, alpha)
    high = _percentile(values, 1 - alpha)
    constant_zero = all(value == 0 for value in values)
    return {
        "low": pack_fraction(low),
        "high": pack_fraction(high),
        "constant_zero": constant_zero,
        "percentile_method": "inclusive_linear",
        "inference_claim": "not_a_promotion_criterion",
    }


def _group_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    groups = {row["source_group_id"] for row in rows}
    valid_0 = {row["source_group_id"] for row in rows if row["valid"] and row["gold_label"] == 0}
    valid_1 = {row["source_group_id"] for row in rows if row["valid"] and row["gold_label"] == 1}
    return {
        "n_groups": len(groups),
        "groups_with_valid_gold_0": len(valid_0),
        "groups_with_valid_gold_1": len(valid_1),
    }


def bootstrap_metric(
    rows: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    *,
    other_rows: Sequence[Mapping[str, Any]] | None = None,
    metric: str,
) -> dict[str, Any]:
    coverage = _group_coverage(rows if other_rows is None else list(rows) + list(other_rows))
    base = {
        "enabled": True,
        "method": statistics["method"],
        "seed": statistics["seed"],
        "confidence_level": statistics["confidence_level"],
        "bootstrap_replicates_requested": statistics["bootstrap_replicates"],
        "protocol_status": statistics["protocol_status"],
        "formal_threshold": False,
        "sample_set_sha256": stamp_hash([
            {"record_id": row["record_id"], "gold_label": row["gold_label"], "source_group_id": row["source_group_id"]}
            for row in sorted(rows, key=lambda item: item["record_id"])
        ]),
        **coverage,
    }
    point_rows = rows
    point = _fraction_from_pack(classification_metrics(point_rows)["balanced_accuracy_valid"])
    if metric != "balanced_accuracy_valid":
        point = None
    groups = sorted({row["source_group_id"] for row in rows})
    if len(groups) < 2:
        return {
            **base,
            "replicates_defined": 0,
            "replicates_undefined": 0,
            "group_index_draws": [],
            "interval": None,
            "reason": "fewer_than_two_groups",
        }
    if metric == "balanced_accuracy_valid" and point is None:
        return {
            **base,
            "replicates_defined": 0,
            "replicates_undefined": 0,
            "group_index_draws": [],
            "interval": None,
            "reason": "point_estimate_undefined",
        }
    draws = group_draw_indices(len(groups), statistics["bootstrap_replicates"], statistics["seed"])
    defined: list[Fraction] = []
    undefined = 0
    other_by_id = None if other_rows is None else {row["record_id"]: row for row in other_rows}
    for indices in draws:
        draw = [groups[index] for index in indices]
        expanded, expand_errors = expand_group_draw(rows, draw)
        if expand_errors or expanded is None:
            undefined += 1
            continue
        if metric == "balanced_accuracy_valid":
            value = _fraction_from_pack(classification_metrics(expanded)["balanced_accuracy_valid"])
        else:
            assert other_by_id is not None
            other_expanded = []
            for row in expanded:
                other_expanded.append(other_by_id[row["record_id"]])
            if metric == "delta_ba_own_valid":
                left = _fraction_from_pack(classification_metrics(other_expanded)["balanced_accuracy_valid"])
                right = _fraction_from_pack(classification_metrics(expanded)["balanced_accuracy_valid"])
            else:
                common_ids = [
                    row["record_id"]
                    for row in expanded
                    if row["valid"] and other_by_id[row["record_id"]]["valid"]
                ]
                # Preserve multiplicity: expanded may repeat a record.
                left_rows = []
                right_rows = []
                for row in expanded:
                    other = other_by_id[row["record_id"]]
                    if row["valid"] and other["valid"]:
                        right_rows.append(row)
                        left_rows.append(other)
                left = _fraction_from_pack(classification_metrics(left_rows)["balanced_accuracy_valid"]) if left_rows else None
                right = _fraction_from_pack(classification_metrics(right_rows)["balanced_accuracy_valid"]) if right_rows else None
            value = None if left is None or right is None else right - left
        if value is None:
            undefined += 1
        else:
            defined.append(value)
    if undefined or not defined:
        return {
            **base,
            "replicates_defined": len(defined),
            "replicates_undefined": undefined,
            "group_index_draws": draws,
            "interval": None,
            "reason": "undefined_replicate" if undefined else "point_estimate_undefined",
        }
    return {
        **base,
        "replicates_defined": len(defined),
        "replicates_undefined": 0,
        "group_index_draws": draws,
        "interval": _interval(defined, statistics["confidence_level"]),
        "reason": None,
    }


def _promotion_block() -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "disabled_pending_protocol",
        "model_pointer_changed": False,
    }


def _row_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    audit = []
    for row in sorted(rows, key=lambda item: item["record_id"]):
        audit.append({
            "record_id": row["record_id"],
            "pair_id": row.get("pair_id"),
            "run_id": row.get("run_id"),
            "source_group_id": row["source_group_id"],
            "gold_label": row["gold_label"],
            "task_sha256": row["task_sha256"],
            "task_hash_source": row["task_hash_source"],
            "model_observed_task_sha256": row.get("observed_task_sha256"),
            "model_observation_compared": row.get("observed_task_sha256") is not None,
            "prediction_status": "valid" if row["valid"] else "invalid",
            "pred_label": row.get("pred_label"),
            "invalid_reason": row.get("invalid_reason"),
            "diagnostic": row.get("diagnostic") or {},
        })
    return audit


def _failed_report(errors: list[dict[str, str]], config: Mapping[str, Any] | None, *, dataset_hash: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": "nmr-eval-report-v1",
        "evaluator_version": EVALUATOR_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "invalid_policy": INVALID_POLICY,
        "primary_value": None,
        "metrics": None,
        "formal_primary_published": False,
        "integrity": {"status": "failed", "errors": errors},
        "pairing_verified": False,
        "dataset_content_sha256": dataset_hash,
        "config_sha256": None if config is None else config.get("content_sha256"),
        "promotion": _promotion_block(),
        "example_status": None if config is None else config.get("example_status"),
    }


def evaluate_aligned(
    aligned: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    gold_sha256: str,
    run_manifest: Mapping[str, Any],
    explicit_group_draw: Sequence[str] | None = None,
) -> dict[str, Any]:
    formal = bool(meta.get("pairing_verified"))
    metrics = classification_metrics(aligned) if formal else None
    diagnostic = None if formal else classification_metrics(aligned)
    explicit = None
    explicit_errors: list[dict[str, str]] = []
    if explicit_group_draw is not None:
        expanded, explicit_errors = expand_group_draw(aligned, explicit_group_draw)
        explicit = {
            "draw": list(explicit_group_draw),
            "weights": dict(Counter(explicit_group_draw)),
            "metrics": classification_metrics(expanded) if expanded is not None and formal and not explicit_errors else None,
            "errors": explicit_errors,
            "role": "synthetic_group_weight_check",
        }
    statistics = {"enabled": False}
    if formal and config["statistics"]["enabled"]:
        statistics = {
            "balanced_accuracy_valid": bootstrap_metric(aligned, config["statistics"], metric="balanced_accuracy_valid"),
        }
    published = formal
    return {
        "schema_version": "nmr-eval-report-v1",
        "evaluator_version": EVALUATOR_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "source_group_definition": {"id": SOURCE_GROUP_ID, "version": SOURCE_GROUP_VERSION},
        "dataset_id": dataset["dataset_id"],
        "dataset_content_sha256": dataset["content_sha256"],
        "gold_sidecar_sha256": gold_sha256,
        "run_id": run_manifest.get("run_id"),
        "evaluated_version": run_manifest.get("evaluated_version"),
        "profile": meta.get("profile"),
        "primary_metric": PRIMARY_METRIC,
        "invalid_policy": INVALID_POLICY,
        "primary_value": None if metrics is None else metrics["balanced_accuracy_valid"],
        "metrics_status": None if metrics is None else metrics["status"],
        "metrics": metrics,
        "diagnostic_metrics": diagnostic,
        "formal_primary_published": published and metrics is not None,
        "groups": grouped_metrics(aligned) if formal else None,
        "cost": aggregate_cost(aligned) if formal else None,
        "rows": _row_audit(aligned),
        "explicit_group_draw": explicit,
        "statistics": statistics,
        "integrity": {"status": "passed" if formal else "failed", "errors": []},
        "pairing_verified": formal,
        "config_sha256": config["content_sha256"],
        "promotion": _promotion_block(),
        "example_status": config.get("example_status"),
        "input_hash_policy": "A manifest task hash describes the evaluation task. It is not copied into model_observed_task_sha256.",
    }


def evaluate_documents(
    dataset_manifest: Mapping[str, Any],
    dataset_path: Path,
    gold_sidecar: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    predictions: Any,
    config_document: Mapping[str, Any],
    *,
    explicit_group_draw: Sequence[str] | None = None,
    legacy: bool = False,
) -> dict[str, Any]:
    config = load_config(config_document)
    if isinstance(run_manifest.get("config_sha256"), str) and run_manifest["config_sha256"] != config["content_sha256"]:
        return _failed_report([_error("config_hash_mismatch", "run manifest config hash does not match the loaded config")], config)
    dataset, errors = prepare_dataset(dataset_manifest, dataset_path)
    if dataset is None:
        return _failed_report(errors, config)
    gold_sha, gold_errors = check_gold_sidecar(gold_sidecar, dataset)
    if gold_sha is None:
        return _failed_report(gold_errors, config, dataset_hash=dataset["content_sha256"])
    if run_manifest.get("gold_sidecar_sha256") != gold_sha:
        return _failed_report(
            [_error("gold_hash_mismatch", "run manifest gold hash does not match the sidecar")],
            config,
            dataset_hash=dataset["content_sha256"],
        )
    aligned, meta, align_errors = align_predictions(dataset, run_manifest, predictions, legacy=legacy)
    if aligned is None or meta is None:
        return _failed_report(align_errors, config, dataset_hash=dataset["content_sha256"])
    if align_errors and not meta.get("pairing_verified"):
        report = evaluate_aligned(aligned, meta, dataset, config, gold_sha256=gold_sha, run_manifest=run_manifest)
        report["formal_primary_published"] = False
        report["metrics"] = None
        report["primary_value"] = None
        report["integrity"] = {"status": "failed", "errors": align_errors}
        return report
    report = evaluate_aligned(
        aligned,
        meta,
        dataset,
        config,
        gold_sha256=gold_sha,
        run_manifest=run_manifest,
        explicit_group_draw=explicit_group_draw,
    )
    if not meta.get("pairing_verified"):
        report["formal_primary_published"] = False
        report["primary_value"] = None
        report["metrics"] = None
        report["integrity"] = {"status": "legacy_diagnostic", "errors": []}
        report["pairing_verified"] = False
    return report


def _delta(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None, reason: str) -> dict[str, Any]:
    if left is None or right is None:
        return pack_fraction(None, reason)
    return pack_fraction(_fraction_from_pack(left) - _fraction_from_pack(right) if _fraction_from_pack(left) is not None and _fraction_from_pack(right) is not None else None, None if _fraction_from_pack(left) is not None and _fraction_from_pack(right) is not None else reason)


def compare_aligned(
    aligned_a: Sequence[Mapping[str, Any]],
    aligned_b: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
) -> dict[str, Any]:
    by_a = {row["record_id"]: row for row in aligned_a}
    by_b = {row["record_id"]: row for row in aligned_b}
    if set(by_a) != set(by_b):
        return _failed_report([_error("gold_conflict", "comparison record ids differ")], config, dataset_hash=dataset["content_sha256"])
    for record_id, row_a in by_a.items():
        row_b = by_b[record_id]
        if row_a["gold_label"] != row_b["gold_label"] or row_a["source_group_id"] != row_b["source_group_id"]:
            return _failed_report([_error("gold_conflict", f"{record_id}: gold or source group differs")], config, dataset_hash=dataset["content_sha256"])
    metrics_a = classification_metrics(aligned_a)
    metrics_b = classification_metrics(aligned_b)
    both = a_only = b_only = neither = 0
    common_a = []
    common_b = []
    both_right = both_wrong = a_wrong_b_right = a_right_b_wrong = 0
    for record_id in sorted(by_a):
        row_a = by_a[record_id]
        row_b = by_b[record_id]
        if row_a["valid"] and row_b["valid"]:
            both += 1
            common_a.append(row_a)
            common_b.append(row_b)
            correct_a = row_a["pred_label"] == row_a["gold_label"]
            correct_b = row_b["pred_label"] == row_b["gold_label"]
            if correct_a and correct_b:
                both_right += 1
            elif not correct_a and not correct_b:
                both_wrong += 1
            elif correct_a and not correct_b:
                a_right_b_wrong += 1
            else:
                a_wrong_b_right += 1
        elif row_a["valid"]:
            a_only += 1
        elif row_b["valid"]:
            b_only += 1
        else:
            neither += 1
    if not common_a:
        common_block: dict[str, Any] = {
            "record_ids": [],
            "N": 0,
            "V": 0,
            "by_label": {"0": {"N": 0, "V": 0}, "1": {"N": 0, "V": 0}},
            "A_balanced_accuracy_valid": pack_fraction(None, "empty_common_valid"),
            "B_balanced_accuracy_valid": pack_fraction(None, "empty_common_valid"),
            "delta_ba_common_valid": pack_fraction(None, "empty_common_valid"),
            "paired": {"both_right": 0, "both_wrong": 0, "a_wrong_b_right": 0, "a_right_b_wrong": 0},
            "role": "auxiliary_not_a_replacement_for_own_valid",
        }
        common_reason = "empty_common_valid"
    else:
        common_metrics_a = classification_metrics(common_a)
        common_metrics_b = classification_metrics(common_b)
        common_reason = None if common_metrics_a["status"] == "scored" and common_metrics_b["status"] == "scored" else "missing_valid_class"
        common_block = {
            "record_ids": [row["record_id"] for row in common_a],
            "N": len(common_a),
            "V": len(common_a),
            "by_label": {
                "0": {"N": common_metrics_a["by_label"]["0"]["N"], "V": common_metrics_a["by_label"]["0"]["V"]},
                "1": {"N": common_metrics_a["by_label"]["1"]["N"], "V": common_metrics_a["by_label"]["1"]["V"]},
            },
            "A_balanced_accuracy_valid": common_metrics_a["balanced_accuracy_valid"],
            "B_balanced_accuracy_valid": common_metrics_b["balanced_accuracy_valid"],
            "A_accuracy_valid": common_metrics_a["accuracy_valid"],
            "B_accuracy_valid": common_metrics_b["accuracy_valid"],
            "delta_ba_common_valid": _delta(
                common_metrics_b["balanced_accuracy_valid"],
                common_metrics_a["balanced_accuracy_valid"],
                common_reason or "missing_valid_class",
            ),
            "paired": {
                "both_right": both_right,
                "both_wrong": both_wrong,
                "a_wrong_b_right": a_wrong_b_right,
                "a_right_b_wrong": a_right_b_wrong,
            },
            "role": "auxiliary_not_a_replacement_for_own_valid",
        }
    statistics: dict[str, Any] = {"enabled": False}
    if config["statistics"]["enabled"]:
        draws = None
        own = bootstrap_metric(aligned_b, config["statistics"], other_rows=aligned_a, metric="delta_ba_own_valid")
        # Reuse the same index draws for the common-valid interval.
        common_stat = bootstrap_metric(aligned_b, config["statistics"], other_rows=aligned_a, metric="delta_ba_common_valid")
        if own.get("group_index_draws") != common_stat.get("group_index_draws") and own.get("group_index_draws") and common_stat.get("group_index_draws"):
            common_stat["interval"] = None
            common_stat["reason"] = "draw_mismatch"
        draws = own.get("group_index_draws")
        statistics = {
            "enabled": True,
            "shared_group_index_draws": draws,
            "delta_ba_own_valid": {key: value for key, value in own.items() if key != "group_index_draws"},
            "delta_ba_common_valid": {key: value for key, value in common_stat.items() if key != "group_index_draws"},
        }
    def side(metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "balanced_accuracy_valid": metrics["balanced_accuracy_valid"],
            "accuracy_valid": metrics["accuracy_valid"],
            "coverage": metrics["coverage"],
            "coverage_0": metrics["by_label"]["0"]["coverage"],
            "coverage_1": metrics["by_label"]["1"]["coverage"],
            "N": metrics["N"],
            "V": metrics["V"],
            "I": metrics["I"],
            "status": metrics["status"],
        }

    return {
        "schema_version": COMPARISON_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "dataset_id": dataset["dataset_id"],
        "dataset_content_sha256": dataset["content_sha256"],
        "run_id_a": run_a.get("run_id"),
        "run_id_b": run_b.get("run_id"),
        "evaluated_version_a": run_a.get("evaluated_version"),
        "evaluated_version_b": run_b.get("evaluated_version"),
        "same_dataset": True,
        "primary_metric": PRIMARY_METRIC,
        "invalid_policy": INVALID_POLICY,
        "own_valid": {
            "A": side(metrics_a),
            "B": side(metrics_b),
            "delta_ba_own_valid": _delta(metrics_b["balanced_accuracy_valid"], metrics_a["balanced_accuracy_valid"], metrics_a["status"] if metrics_a["status"] != "scored" or metrics_b["status"] != "scored" else "missing_valid_class"),
            "delta_coverage": _delta(metrics_b["coverage"], metrics_a["coverage"], "empty_evaluation_set"),
            "delta_coverage_0": _delta(metrics_b["by_label"]["0"]["coverage"], metrics_a["by_label"]["0"]["coverage"], "zero_gold_class"),
            "delta_coverage_1": _delta(metrics_b["by_label"]["1"]["coverage"], metrics_a["by_label"]["1"]["coverage"], "zero_gold_class"),
            "populations_may_differ": True,
        },
        "common_valid": common_block,
        "coverage_table": {
            "both_valid": both,
            "A_only_valid": a_only,
            "B_only_valid": b_only,
            "neither_valid": neither,
            "sum": both + a_only + b_only + neither,
        },
        "statistics": statistics,
        "integrity": {"status": "passed", "errors": []},
        "promotion": _promotion_block(),
        "conclusion": "comparison_only_promotion_disabled",
        "example_status": config.get("example_status"),
    }


def compare_documents(
    dataset_manifest: Mapping[str, Any],
    dataset_path: Path,
    gold_sidecar: Mapping[str, Any],
    run_a: Mapping[str, Any],
    predictions_a: Any,
    run_b: Mapping[str, Any],
    predictions_b: Any,
    config_document: Mapping[str, Any],
) -> dict[str, Any]:
    config = load_config(config_document)
    dataset, errors = prepare_dataset(dataset_manifest, dataset_path)
    if dataset is None:
        return _failed_report(errors, config)
    gold_sha, gold_errors = check_gold_sidecar(gold_sidecar, dataset)
    if gold_sha is None:
        return _failed_report(gold_errors, config, dataset_hash=dataset["content_sha256"])
    aligned_a, meta_a, errors_a = align_predictions(dataset, run_a, predictions_a, legacy=False)
    aligned_b, meta_b, errors_b = align_predictions(dataset, run_b, predictions_b, legacy=False)
    if aligned_a is None or meta_a is None or not meta_a.get("pairing_verified"):
        return _failed_report(errors_a or [_error("legacy_not_formal", "run A")], config, dataset_hash=dataset["content_sha256"])
    if aligned_b is None or meta_b is None or not meta_b.get("pairing_verified"):
        return _failed_report(errors_b or [_error("legacy_not_formal", "run B")], config, dataset_hash=dataset["content_sha256"])
    if run_a.get("gold_sidecar_sha256") != gold_sha or run_b.get("gold_sidecar_sha256") != gold_sha:
        return _failed_report([_error("gold_hash_mismatch", "run gold hash does not match the sidecar")], config, dataset_hash=dataset["content_sha256"])
    return compare_aligned(aligned_a, aligned_b, dataset, config, run_a, run_b)
