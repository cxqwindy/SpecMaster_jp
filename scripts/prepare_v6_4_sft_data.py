#!/usr/bin/env python3
"""Prepare deterministic, group-isolated v6.4 LLaMA-Factory SFT splits.

Validation is stratified by label. One representative per selected semantic group is kept
in validation and every other copy of that group is excluded, preventing oversampled
positive trajectories from leaking between train and validation.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable

DEFAULT_INPUT = Path(
    "/fs_mol/cxq/SpecMaster_jp/distill_out/"
    "trajectories_v6_3_filtered_balanced_observation_fixed.jsonl"
)
DEFAULT_CONVERTER = Path(
    "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/"
    "trajectory_to_llamafactory.py"
)
DEFAULT_DATA_DIR = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row.get("messages"), list) or not isinstance(row.get("meta"), dict):
                raise ValueError(f"{path}:{line_number}: expected messages list and meta object")
            if row["meta"].get("provenance_violations"):
                raise ValueError(f"{path}:{line_number}: provenance violations are not empty")
            rows.append(row)
    return rows


def semantic_key(row: dict[str, Any]) -> tuple[Any, ...]:
    meta = row["meta"]
    return (
        meta.get("candidate_smiles"),
        meta.get("gt_smiles"),
        int(meta["label"]),
        meta.get("negative_type"),
    )


def label_quotas(rows: list[dict[str, Any]], n_val: int) -> dict[int, int]:
    counts = collections.Counter(int(row["meta"]["label"]) for row in rows)
    raw = {label: n_val * count / len(rows) for label, count in counts.items()}
    quotas = {label: math.floor(value) for label, value in raw.items()}
    remainder = n_val - sum(quotas.values())
    order = sorted(quotas, key=lambda label: (raw[label] - quotas[label], label), reverse=True)
    for label in order[:remainder]:
        quotas[label] += 1
    return quotas


def split_grouped_stratified(
    rows: list[dict[str, Any]], val_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    n_val = int(len(rows) * val_frac)
    if n_val <= 0:
        raise ValueError("val_frac produces an empty validation split")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[semantic_key(row)].append(row)

    quotas = label_quotas(rows, n_val)
    rng = random.Random(seed)
    selected: set[tuple[Any, ...]] = set()
    for label, quota in sorted(quotas.items()):
        keys = [key for key in groups if key[2] == label]
        rng.shuffle(keys)
        if len(keys) < quota:
            raise ValueError(f"label={label} has only {len(keys)} groups for quota={quota}")
        selected.update(keys[:quota])

    validation = [groups[key][0] for key in selected]
    validation.sort(key=lambda row: semantic_key(row))
    train = [row for row in rows if semantic_key(row) not in selected]
    removed_rows = sum(len(groups[key]) for key in selected)
    dropped_duplicates = removed_rows - len(validation)

    train_keys = {semantic_key(row) for row in train}
    val_keys = {semantic_key(row) for row in validation}
    if train_keys & val_keys:
        raise AssertionError("semantic group leakage between train and validation")
    if len(validation) != n_val:
        raise AssertionError(f"validation size mismatch: {len(validation)} != {n_val}")

    report = {
        "source_rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "excluded_oversample_copies": dropped_duplicates,
        "seed": seed,
        "val_frac": val_frac,
        "validation_label_quota": {str(key): value for key, value in sorted(quotas.items())},
        "train_label_counts": dict(collections.Counter(int(r["meta"]["label"]) for r in train)),
        "validation_label_counts": dict(
            collections.Counter(int(r["meta"]["label"]) for r in validation)
        ),
        "semantic_overlap": 0,
    }
    return train, validation, report


def load_converter(path: Path) -> tuple[Callable[[list[dict[str, Any]]], list[dict[str, str]]], str]:
    spec = importlib.util.spec_from_file_location("trajectory_to_llamafactory_v6_4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_messages, module.TOOLS_JSON


def convert_rows(
    rows: list[dict[str, Any]],
    convert_messages: Callable[[list[dict[str, Any]]], list[dict[str, str]]],
    tools_json: str,
) -> list[dict[str, Any]]:
    return [
        {"conversations": convert_messages(row["messages"]), "tools": tools_json}
        for row in rows
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_converted(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if set(row) != {"conversations", "tools"}:
            raise ValueError(f"converted row {index}: unexpected keys {set(row)}")
        json.loads(row["tools"])
        conversations = row["conversations"]
        roles = [turn.get("from") for turn in conversations]
        if roles[:2] != ["system", "human"] or roles[-1:] != ["gpt"]:
            raise ValueError(f"converted row {index}: invalid boundary roles {roles}")
        middle = roles[2:-1]
        expected = ["function_call", "observation"] * (len(middle) // 2)
        if len(middle) % 2 or middle != expected:
            raise ValueError(f"converted row {index}: non-alternating tool roles {roles}")
        final = conversations[-1]["value"]
        if "<answer>" not in final or '"final_answer"' not in final:
            raise ValueError(f"converted row {index}: missing final answer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT))
    parser.add_argument("--converter", default=str(DEFAULT_CONVERTER))
    parser.add_argument("--train-out", default=str(DEFAULT_DATA_DIR / "nmr_matching_tool_sft_v6_4.json"))
    parser.add_argument(
        "--val-out", default=str(DEFAULT_DATA_DIR / "nmr_matching_tool_sft_v6_4_val.json")
    )
    parser.add_argument(
        "--report", default=str(DEFAULT_DATA_DIR / "nmr_matching_tool_sft_v6_4_split_report.json")
    )
    parser.add_argument("--val-frac", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    train_out, val_out, report_out = Path(args.train_out), Path(args.val_out), Path(args.report)
    paths = [input_path.resolve(), train_out.resolve(), val_out.resolve(), report_out.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("input and output paths must be distinct")
    existing = [str(path) for path in (train_out, val_out, report_out) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"refusing to overwrite without --force: {existing}")

    actual_sha = sha256(input_path)
    if args.expected_sha256 and actual_sha != args.expected_sha256:
        raise ValueError(f"input SHA-256 mismatch: {actual_sha} != {args.expected_sha256}")

    source = read_jsonl(input_path)
    train_source, val_source, report = split_grouped_stratified(
        source, args.val_frac, args.seed
    )
    convert_messages, tools_json = load_converter(Path(args.converter))
    train = convert_rows(train_source, convert_messages, tools_json)
    validation = convert_rows(val_source, convert_messages, tools_json)
    validate_converted(train)
    validate_converted(validation)

    report.update({
        "input": str(input_path),
        "input_sha256": actual_sha,
        "train_output": str(train_out),
        "validation_output": str(val_out),
        "converter": str(args.converter),
    })
    atomic_json(train_out, train)
    atomic_json(val_out, validation)
    atomic_json(report_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
