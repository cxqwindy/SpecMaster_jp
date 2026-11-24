#!/usr/bin/env python3
"""Build nmr_matching_test_2000.json: existing 1000 rows verbatim + 1000 new
rows drawn from the 1000-scale pools, mirroring the exact `_negative_type`
histogram of test_1000 (so per-type accuracy stays comparable).

Guarantees:
  * First 1000 rows byte-identical to nmr_matching_test_1000.json.
  * New rows are pair-disjoint from the existing 1000 by (candidate, gt).
  * Same Alpaca schema and same INSTRUCTION as test_1000.

Usage:
    python scripts/build_test_2000.py
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

TEST_1000 = "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json"
POOLS = [
    "/fs_mol/cxq/SpecMaster_jp/negative_output_1000/matching_1000_matching.jsonl",
    "/fs_mol/cxq/SpecMaster_jp/hard_negative_output_1000/hard_neg_1000_matching.jsonl",
    "/fs_mol/cxq/SpecMaster_jp/perturbation_output_1000/perturbed_1000_matching_benchmark.jsonl",
]

INSTRUCTION = (
    "Given NMR spectral data and a candidate molecular structure (SMILES), "
    "determine whether the candidate matches the NMR spectrum. "
    "Provide your reasoning as a structured JSON object."
)


def format_alpaca(rec):
    """Reformat a pool row into the same Alpaca schema as test_1000. Mirrors
    scripts/prepare_cot_sft_data.py::build_test_set (lines 200-242)."""
    candidate = rec["candidate_smiles"]

    c_peaks = rec.get("c_nmr_peaks") or []
    c_vals = []
    for p in c_peaks:
        v = p.get("delta (ppm)")
        if v is not None:
            c_vals.append(f"{float(v):.1f}")
    c_str = ", ".join(c_vals) if c_vals else "N/A"

    h_peaks = rec.get("h_nmr_peaks") or []
    h_parts = []
    for p in h_peaks:
        delta = p.get("centroid") or p.get("delta")
        if delta is None:
            continue
        cat = p.get("category", "")
        nH = p.get("nH", 1)
        j_vals = p.get("j_values", "")
        if j_vals and j_vals != "_":
            j_str = j_vals.replace("_", ", ").strip(", ")
            h_parts.append(f"{float(delta):.2f} ({cat}, J={j_str}, {nH}H)")
        else:
            h_parts.append(f"{float(delta):.2f} ({cat}, {nH}H)")
    h_str = ", ".join(h_parts) if h_parts else "N/A"

    freq = rec.get("nmr_frequency", "")
    solvent = rec.get("nmr_solvent", "")
    header = f" ({freq}, {solvent})" if freq else ""

    return {
        "instruction": INSTRUCTION,
        "input": (
            f"Candidate SMILES: {candidate}\n\n"
            f"13C NMR{header}: {c_str}\n"
            f"1H NMR{header}: {h_str}"
        ),
        "output": "Yes" if rec["label"] == 1 else "No",
        "_label": int(rec["label"]),
        "_negative_type": rec.get("negative_type") or "positive",
        "_candidate_smiles": candidate,
        "_gt_smiles": rec.get("gt_smiles", ""),
    }


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json")
    ap.add_argument("--manifest", default="/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000_manifest.json")
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    old_rows = json.load(open(TEST_1000))
    assert len(old_rows) == 1000, f"expected 1000 old rows, got {len(old_rows)}"
    used_pairs = {(r["_candidate_smiles"], r["_gt_smiles"]) for r in old_rows}
    used_pos_gts = {r["_gt_smiles"] for r in old_rows if r["_label"] == 1}

    old_hist = Counter(r["_negative_type"] for r in old_rows)
    # target histogram for the NEW 1000 = exact mirror of test_1000
    quotas = dict(old_hist)
    assert sum(quotas.values()) == 1000

    # bucket pool rows by _negative_type. Positives live under "positive".
    buckets = defaultdict(list)
    for path in POOLS:
        for r in read_jsonl(path):
            key = "positive" if r["label"] == 1 else (r.get("negative_type") or "")
            if not key:
                continue
            if r["label"] == 1:
                # skip positives whose GT already appears in test_1000 positives (rare, but keeps
                # the new-1000 slice disjoint from the old-1000 slice in molecule identity too).
                if r["candidate_smiles"] in used_pos_gts:
                    continue
            buckets[key].append(r)

    rng = random.Random(args.seed)
    new_rows_pool = []
    for nt, q in quotas.items():
        pool = buckets.get(nt, [])
        rng.shuffle(pool)
        picked = []
        seen_local = set()
        for r in pool:
            pair = (r["candidate_smiles"], r.get("gt_smiles", ""))
            if pair in used_pairs or pair in seen_local:
                continue
            picked.append(r)
            seen_local.add(pair)
            if len(picked) == q:
                break
        if len(picked) < q:
            raise RuntimeError(
                f"insufficient rows for bucket {nt!r}: need {q}, got {len(picked)}"
            )
        new_rows_pool.extend(picked)

    assert len(new_rows_pool) == 1000

    new_rows = [format_alpaca(r) for r in new_rows_pool]

    # Sanity: all new pairs unique across the full 2000
    all_pairs = list(used_pairs)
    all_pairs.extend((r["_candidate_smiles"], r["_gt_smiles"]) for r in new_rows)
    if len(set(all_pairs)) != 2000:
        dups = Counter(all_pairs).most_common(5)
        raise RuntimeError(f"pair collision after concat, sample dups: {dups}")

    all_rows = old_rows + new_rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    # First-1000 identity check + manifest
    old_serialized = json.dumps(all_rows[:1000], ensure_ascii=False, sort_keys=False)
    old_ref = json.dumps(old_rows, ensure_ascii=False, sort_keys=False)
    assert old_serialized == old_ref, "first 1000 rows drifted from test_1000"

    def sha256(s):
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    new_hist = Counter(r["_negative_type"] for r in new_rows)
    manifest = {
        "test_file": str(out_path),
        "total_rows": len(all_rows),
        "old_rows_source": TEST_1000,
        "old_rows_sha256": sha256(old_ref),
        "seed": args.seed,
        "pool_files": POOLS,
        "old_histogram": dict(sorted(old_hist.items())),
        "new_histogram": dict(sorted(new_hist.items())),
        "combined_histogram": dict(sorted(Counter(r["_negative_type"] for r in all_rows).items())),
        "used_pos_gts_dropped": None,  # not tracked in detail; count is stable across runs due to fixed seed
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {out_path} rows={len(all_rows)}")
    print(f"wrote {args.manifest}")
    print("new-rows histogram:")
    for k, v in sorted(new_hist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
