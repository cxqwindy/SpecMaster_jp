"""Write synthetic evaluation inputs.

The files are inputs. Expected scores live in the hand-calculated JSON and are
not produced here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from matching_evaluator import (
    CONFIG_SCHEMA,
    DATASET_SCHEMA,
    EVALUATOR_VERSION,
    GOLD_SCHEMA,
    INVALID_POLICY,
    PRIMARY_METRIC,
    RUN_SCHEMA,
    SOURCE_GROUP_ID,
    SOURCE_GROUP_VERSION,
    check_gold_sidecar,
    dump_json,
    load_config,
    prepare_dataset,
    source_group_for_parent,
)


def scoring_config(statistics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "invalid_policy": INVALID_POLICY,
        "primary_metric": PRIMARY_METRIC,
        "promotion": {"enabled": False},
        "statistics": dict(statistics or {"enabled": False}),
        "example_status": "synthetic",
    }


def _task(candidate: str) -> dict[str, Any]:
    return {
        "candidate_smiles": candidate,
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"delta": 1.2, "centroid": 1.2, "nH": 3, "category": "t"}],
    }


def write_bundle(
    directory: Path,
    *,
    dataset_id: str,
    records: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a dataset, gold sidecar, config, and one or more run/prediction files.

    ``records`` contain record_id, parent_gt_smiles, gold_label, and
    candidate_smiles. ``runs`` contain run_id, evaluated_version, profile, and
    predictions in the same record order. Pair ids are assigned from the
    record order and are not treated as molecule identities.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for record in records:
        parent = str(record["parent_gt_smiles"])
        manifest_records.append({
            "record_id": str(record["record_id"]),
            "task": _task(str(record["candidate_smiles"])),
            "gold_label": record["gold_label"],
            "source_group_id": source_group_for_parent(parent),
            "parent_gt_smiles": parent,
            "difficulty": record.get("difficulty"),
            "ambiguity": record.get("ambiguity"),
        })
    manifest = {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": dataset_id,
        "source_group_definition": {"id": SOURCE_GROUP_ID, "version": SOURCE_GROUP_VERSION},
        "records": manifest_records,
    }
    dataset_path = directory / "dataset_manifest.json"
    dump_json(dataset_path, manifest)
    prepared, errors = prepare_dataset(manifest, dataset_path, verify_hash=False)
    if prepared is None:
        raise RuntimeError(json.dumps(errors, ensure_ascii=False))
    manifest["content_sha256"] = prepared["content_sha256"]
    for raw, stamped in zip(manifest["records"], prepared["records"]):
        raw["task_sha256"] = stamped["task_sha256"]
    dump_json(dataset_path, manifest)
    gold = {
        "schema_version": GOLD_SCHEMA,
        "dataset_id": dataset_id,
        "records": [
            {
                "record_id": row["record_id"],
                "gold_label": row["gold_label"],
                "source_group_id": row["source_group_id"],
                "parent_gt_smiles": row["parent_gt_smiles"],
            }
            for row in prepared["records"]
        ],
    }
    gold_path = directory / "gold_sidecar.json"
    dump_json(gold_path, gold)
    gold_sha, gold_errors = check_gold_sidecar(gold, prepared, verify_hash=False)
    if gold_sha is None:
        raise RuntimeError(json.dumps(gold_errors, ensure_ascii=False))
    gold["content_sha256"] = gold_sha
    dump_json(gold_path, gold)
    config_document = dict(config or scoring_config())
    config_path = directory / "config.json"
    dump_json(config_path, config_document)
    normalized = load_config(config_document)
    written_runs = []
    for index, run in enumerate(runs):
        predictions = []
        pair_to_record = {}
        for record, prediction in zip(prepared["records"], run["predictions"]):
            pair_id = prediction.get("pair_id", len(pair_to_record) + 1)
            pair_key = str(int(pair_id))
            pair_to_record[pair_key] = record["record_id"]
            row = dict(prediction)
            row["pair_id"] = int(pair_key)
            row.setdefault("run_id", run["run_id"])
            predictions.append(row)
        stem = run.get("stem") or f"run_{index}"
        prediction_path = directory / f"{stem}_predictions.json"
        run_path = directory / f"{stem}_run_manifest.json"
        dump_json(prediction_path, {
            "schema_version": "nmr-eval-prediction-v1",
            "profile": run["profile"],
            "run_id": run["run_id"],
            "predictions": predictions,
        })
        dump_json(run_path, {
            "schema_version": RUN_SCHEMA,
            "run_id": run["run_id"],
            "dataset_id": dataset_id,
            "dataset_content_sha256": prepared["content_sha256"],
            "gold_sidecar_sha256": gold_sha,
            "evaluated_version": run["evaluated_version"],
            "evaluator_version": EVALUATOR_VERSION,
            "config_sha256": normalized["content_sha256"],
            "pair_to_record": pair_to_record,
        })
        written_runs.append({"run": str(run_path), "predictions": str(prediction_path), "run_id": run["run_id"]})
    aliases = {}
    for record in prepared["records"]:
        alias = next(
            (item.get("alias") for item in records if str(item["record_id"]) == record["record_id"]),
            None,
        )
        if alias:
            aliases[alias] = record["source_group_id"]
    dump_json(directory / "group_alias.json", aliases)
    return {
        "dataset": str(dataset_path),
        "gold": str(gold_path),
        "config": str(config_path),
        "config_sha256": normalized["content_sha256"],
        "dataset_content_sha256": prepared["content_sha256"],
        "gold_sha256": gold_sha,
        "runs": written_runs,
        "aliases": aliases,
        "records": prepared["records"],
    }


HAND_PARENTS = {
    "gA": "CCO",
    "gB": "CCCO",
    "gC": "CCCCO",
    "gD": "CC(C)O",
}

# Gold 0 on gA uses gB's parent SMILES as the candidate. The group must stay gA.
HAND_SPEC = [
    ("1", "gA", 1, "Yes", "answer"),
    ("2", "gA", 1, "No", "answer"),
    ("3", "gA", 0, "Yes", "answer"),
    ("4", "gB", 1, "Yes", "answer"),
    ("5", "gB", 0, "No", "answer"),
    ("6", "gB", 0, None, "empty_answer"),
    ("7", "gC", 1, "Yes", "answer"),
    ("8", "gC", 0, "Yes", "answer"),
    ("9", "gC", 0, "No", "answer"),
    ("10", "gD", 1, None, "timeout"),
    ("11", "gD", 1, None, "parse_error"),
    ("12", "gD", 0, "No", "answer"),
]


def _candidate(record_id: str, alias: str, gold: int) -> str:
    if record_id == "3":
        return HAND_PARENTS["gB"]
    if gold == 1:
        return HAND_PARENTS[alias]
    return "c1ccccc1"


def _terminal(answer: str | None, state: str, *, cost: dict[str, Any] | None = None) -> dict[str, Any]:
    terminal: dict[str, Any] = {"state": state, "final_answer": answer}
    row: dict[str, Any] = {"terminal": terminal}
    if cost is not None:
        row["cost"] = cost
    return row


def hand_records() -> list[dict[str, Any]]:
    rows = []
    for record_id, alias, gold, _answer, _state in HAND_SPEC:
        rows.append({
            "record_id": record_id,
            "alias": alias,
            "parent_gt_smiles": HAND_PARENTS[alias],
            "gold_label": gold,
            "candidate_smiles": _candidate(record_id, alias, gold),
            "difficulty": None,
            "ambiguity": None,
        })
    return rows


def hand_predictions_a() -> list[dict[str, Any]]:
    rows = []
    for record_id, _alias, _gold, answer, state in HAND_SPEC:
        cost = None
        if record_id == "1":
            cost = {"tool_calls": 1}
        elif record_id == "6":
            cost = {"tool_calls": 2}
        rows.append(_terminal(answer, state, cost=cost))
    return rows


def hand_predictions_b() -> list[dict[str, Any]]:
    rows = []
    for record_id, _alias, _gold, _answer, _state in HAND_SPEC:
        if record_id == "1":
            rows.append(_terminal("Yes", "answer"))
        elif record_id == "5":
            rows.append(_terminal("No", "answer"))
        else:
            rows.append(_terminal(None, "timeout"))
    return rows


def write_hand_fixture(directory: Path) -> dict[str, Any]:
    return write_bundle(
        directory,
        dataset_id="synthetic-hand-12",
        records=hand_records(),
        runs=[
            {
                "stem": "model_a",
                "run_id": "synthetic-run-a",
                "evaluated_version": "synthetic-model-a",
                "profile": "canonical",
                "predictions": hand_predictions_a(),
            },
            {
                "stem": "model_b",
                "run_id": "synthetic-run-b",
                "evaluated_version": "synthetic-model-b",
                "profile": "canonical",
                "predictions": hand_predictions_b(),
            },
        ],
    )
