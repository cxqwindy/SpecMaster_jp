#!/usr/bin/env python3
"""Merge the v6.4 NMR dataset entries into a LLaMA-Factory dataset_info.json.

The live training registry is not modified unless ``--write`` is passed.
Existing unrelated keys are kept. A key that already matches the registry is
left byte-for-byte untouched. A key with different contents is not overwritten
unless ``--force`` is passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "llamafactory" / "dataset_info_nmr_v6_4.json"


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a JSON object")
    return data


def plan_merge(current: dict, registry: dict) -> tuple[dict, list[str], list[str]]:
    merged = dict(current)
    added: list[str] = []
    conflicts: list[str] = []
    for key, value in registry.items():
        if key not in merged:
            merged[key] = value
            added.append(key)
        elif merged[key] != value:
            conflicts.append(key)
        # Equal keys stay as they are.
    return merged, added, conflicts


def apply_registry(dataset_info: Path, registry_path: Path, write: bool, force: bool) -> dict:
    current_text = dataset_info.read_text(encoding="utf-8")
    current = load_json(dataset_info)
    registry = load_json(registry_path)
    merged, added, conflicts = plan_merge(current, registry)
    if conflicts and not force:
        return {
            "changed": False,
            "wrote": False,
            "added": added,
            "conflicts": conflicts,
            "status": "conflict",
        }
    if conflicts and force:
        for key in conflicts:
            merged[key] = registry[key]
    changed = merged != current
    wrote = False
    if changed and write:
        dataset_info.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        wrote = True
    elif not changed:
        # Identical registration: do not rewrite formatting or mtime.
        if dataset_info.read_text(encoding="utf-8") != current_text:
            raise RuntimeError("unexpected dataset_info rewrite while unchanged")
    return {
        "changed": changed,
        "wrote": wrote,
        "added": added,
        "conflicts": conflicts,
        "status": "ok" if not conflicts or force else "conflict",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--write", action="store_true", help="write only when a change is required")
    parser.add_argument("--force", action="store_true", help="overwrite conflicting keys")
    args = parser.parse_args()
    result = apply_registry(args.dataset_info, args.registry, args.write, args.force)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "conflict":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
