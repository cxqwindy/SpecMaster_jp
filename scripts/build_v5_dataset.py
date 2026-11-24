#!/usr/bin/env python3
"""Merge v4-clean (Path B) + Path A teacher-distilled Hermes trajectories into v5, then hand off
to trajectory_to_llamafactory.py to produce the LLaMA-Factory sharegpt data files.

Inputs:
  - Path B: recipe/nmr_matching_agent/data_v5/tool_sft_v4clean.jsonl  (Hermes messages format)
  - Path A: recipe/nmr_matching_agent/data_v5/tool_sft_teacher.jsonl  (Hermes messages format)

Steps:
  1) load both files
  2) canonical-SMILES dedup — for any row shared by both, keep the Path-A (teacher) copy
     (higher reasoning quality)
  3) build stratified 100-row validation split by _negative_type / label
  4) write merged jsonl to recipe/nmr_matching_agent/data_v5/tool_sft_v5.jsonl
     and a separate _val.jsonl
  5) call trajectory_to_llamafactory.py to produce the LF-consumable json files
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def canonical(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else smi
    except Exception:
        return smi


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        print(f"[warn] missing {path}", flush=True)
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def sample_class_key(row: Dict[str, Any]) -> str:
    meta = row.get("meta", {})
    label = meta.get("label")
    nt = meta.get("negative_type") or "positive"
    if label == 1 or nt == "positive":
        return "positive"
    return nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_b",
                    default="/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/data_v5/tool_sft_v4clean.jsonl")
    ap.add_argument("--path_a",
                    default="/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/data_v5/tool_sft_teacher.jsonl")
    ap.add_argument("--out_dir",
                    default="/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/data_v5")
    ap.add_argument("--lf_out",
                    default="/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v5.json")
    ap.add_argument("--lf_val_out",
                    default="/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v5_val.json")
    ap.add_argument("--val_n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=45)
    args = ap.parse_args()

    b = load_jsonl(args.path_b)
    a = load_jsonl(args.path_a)
    print(f"[load] path_B(v4-clean) = {len(b)}   path_A(teacher) = {len(a)}", flush=True)

    # Dedup: prefer teacher copy on collisions.
    by_canon: Dict[str, Dict[str, Any]] = {}
    for row in b:
        cs = canonical(row.get("meta", {}).get("candidate_smiles", "") or "")
        if cs:
            row["meta"]["_source"] = row["meta"].get("source", "path_B_v4clean")
            by_canon[cs] = row
    n_overlap = 0
    for row in a:
        cs = canonical(row.get("meta", {}).get("candidate_smiles", "") or "")
        if not cs:
            continue
        if cs in by_canon:
            n_overlap += 1
        row["meta"]["_source"] = row["meta"].get("source", "path_A_distill")
        by_canon[cs] = row  # teacher wins
    rows = list(by_canon.values())
    print(f"[merge] overlap (teacher overwrote): {n_overlap}   final unique = {len(rows)}",
          flush=True)

    # Class distribution
    counter = Counter(sample_class_key(r) for r in rows)
    print("\nClass distribution:")
    for k, v in counter.most_common():
        print(f"  {k:60s} {v}")

    # Val split: stratified by class
    rng = random.Random(args.seed)
    by_class: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_class[sample_class_key(r)].append(i)
    val_idx: Set[int] = set()
    per_cls = max(1, args.val_n // len(by_class))
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        take = min(per_cls, len(idxs) // 4)  # never take more than 25% of the class
        val_idx.update(idxs[:take])
    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    print(f"\n[split] train={len(train_rows)}  val={len(val_rows)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "tool_sft_v5.jsonl"
    val_path = out_dir / "tool_sft_v5_val.jsonl"
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {train_path}  {val_path}", flush=True)

    # LLaMA-Factory sharegpt conversion
    conv_script = "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/trajectory_to_llamafactory.py"
    Path(args.lf_out).parent.mkdir(parents=True, exist_ok=True)
    for src, dst in [(train_path, args.lf_out), (val_path, args.lf_val_out)]:
        print(f"\n[to_lf] {src} -> {dst}", flush=True)
        subprocess.run(
            [sys.executable, conv_script,
             "--in", str(src), "--out", str(dst)],
            check=True,
        )


if __name__ == "__main__":
    main()
