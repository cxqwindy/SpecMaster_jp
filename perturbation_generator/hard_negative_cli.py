"""Hard Negative Mining CLI.

Generates hard negatives by:
1. Perturbing correct SMILES (PerturbationEngine)
2. Predicting NMR spectra of perturbed structures (NMR API)
3. Scoring predicted vs experimental spectral similarity
4. Selecting top-K most similar (hardest) negatives per query

Usage::

    python -m perturbation_generator.hard_negative_cli \\
        --query_lmdb /fs_mol/fjxu/.../test_100sample.lmdb \\
        --output_dir /fs_mol/cxq/SpecMaster_jp/hard_negative_output/ \\
        --top_k 5 --batch_size 50 --format both
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .base import PerturbedSample
from .engine import PerturbationEngine
from .formatter import write_matching_both, write_matching_jsonl, write_matching_alpaca_json
from .lmdb_reader import read_lmdb
from .nmr_predictor import predict_nmr_batch
from .spectral_scorer import spectral_similarity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate hard negatives via NMR forward prediction + "
                    "spectral similarity filtering",
    )
    p.add_argument("--query_lmdb", type=str, required=True,
                   help="LMDB with correct molecules + experimental NMR")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory")
    p.add_argument("--prefix", type=str, default="hard_neg",
                   help="Filename prefix (default: hard_neg)")
    p.add_argument("--format", type=str, default="both",
                   choices=["alpaca", "jsonl", "both"])

    # Perturbation params
    p.add_argument("--max_per_operator", type=int, default=5,
                   help="Max perturbation sites per operator (default: 5)")
    p.add_argument("--max_edit_lb", type=int, default=3,
                   help="Max edit lower-bound (default: 3)")

    # NMR prediction params
    p.add_argument("--batch_size", type=int, default=50,
                   help="Batch size for NMR API calls (default: 50)")

    # Filtering params
    p.add_argument("--top_k", type=int, default=5,
                   help="Top-K hard negatives per query (default: 5)")
    p.add_argument("--min_similarity", type=float, default=0.0,
                   help="Minimum spectral similarity to include (default: 0.0)")

    # Query params
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--require_nmr", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("hard_negative_mining")

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load query set
    # ------------------------------------------------------------------
    log.info("Loading query set: %s", args.query_lmdb)
    queries = read_lmdb(args.query_lmdb, require_nmr=True,
                        max_records=args.max_queries)
    log.info("Loaded %d query records", len(queries))
    if not queries:
        log.error("No valid query records. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Generate perturbations
    # ------------------------------------------------------------------
    log.info("Generating perturbations (max_per_operator=%d, max_edit_lb=%d)...",
             args.max_per_operator, args.max_edit_lb)
    engine = PerturbationEngine(
        max_per_operator=args.max_per_operator,
        max_edit_lb=args.max_edit_lb,
        seed=args.seed,
    )
    all_samples = engine.generate_batch(queries, progress=True)
    log.info("Generated %d perturbations from %d queries",
             len(all_samples), len(queries))

    if not all_samples:
        log.error("No perturbations generated. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Batch NMR prediction
    # ------------------------------------------------------------------
    unique_smiles = list(set(s.input_smiles for s in all_samples))
    log.info("Predicting NMR for %d unique perturbed SMILES (batch_size=%d)...",
             len(unique_smiles), args.batch_size)

    t_pred = time.time()
    predictions = predict_nmr_batch(unique_smiles, batch_size=args.batch_size)
    pred_map: Dict[str, Dict] = {}
    n_ok = 0
    for smi, pred in zip(unique_smiles, predictions):
        pred_map[smi] = pred
        if pred["ok"]:
            n_ok += 1
    log.info("NMR prediction done in %.1fs: %d/%d successful",
             time.time() - t_pred, n_ok, len(unique_smiles))

    # ------------------------------------------------------------------
    # 4. Compute spectral similarity
    # ------------------------------------------------------------------
    log.info("Computing spectral similarity...")
    for sample in all_samples:
        pred = pred_map.get(sample.input_smiles, {})
        if not pred.get("ok"):
            sample.spectral_similarity = None
            continue

        sim = spectral_similarity(
            pred_c_peaks=pred["c_peaks"],
            pred_h_peaks=pred["h_peaks"],
            exp_c_peaks=sample.c_nmr_peaks,
            exp_h_peaks=sample.h_nmr_peaks,
        )
        sample.spectral_similarity = sim

    # ------------------------------------------------------------------
    # 5. Select top-K per query
    # ------------------------------------------------------------------
    # Group samples by gt_smiles (query)
    by_query: Dict[str, List[PerturbedSample]] = defaultdict(list)
    for s in all_samples:
        if s.spectral_similarity is not None:
            by_query[s.gt_smiles].append(s)

    hard_negatives: List[PerturbedSample] = []
    for gt_smi, candidates in by_query.items():
        # Sort by spectral similarity descending (most similar = hardest)
        candidates.sort(key=lambda x: x.spectral_similarity or 0.0, reverse=True)
        # Filter by min similarity
        filtered = [c for c in candidates
                    if (c.spectral_similarity or 0.0) >= args.min_similarity]
        # Take top-K
        top_k = filtered[:args.top_k]
        # Tag the negative type
        for s in top_k:
            s.perturbation_type = f"neg/hard_nmr/{s.perturbation_type}"
            s.source_tool = "hard_negative_mining"
        hard_negatives.extend(top_k)

    log.info("Selected %d hard negatives from %d queries (top_k=%d)",
             len(hard_negatives), len(by_query), args.top_k)

    # Stats
    if hard_negatives:
        sims = [s.spectral_similarity for s in hard_negatives if s.spectral_similarity]
        log.info("  Spectral similarity: mean=%.3f, min=%.3f, max=%.3f",
                 sum(sims) / len(sims), min(sims), max(sims))

    # ------------------------------------------------------------------
    # 6. Write output
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "alpaca":
        outfile = str(output_dir / f"{args.prefix}_alpaca.json")
        counts = write_matching_alpaca_json(hard_negatives, outfile,
                                            include_positives=True)
        log.info("Wrote %d records to %s", counts["total"], outfile)
    elif args.format == "jsonl":
        outfile = str(output_dir / f"{args.prefix}_matching.jsonl")
        counts = write_matching_jsonl(hard_negatives, outfile,
                                      include_positives=True)
        log.info("Wrote %d records to %s", counts["total"], outfile)
    else:
        result = write_matching_both(hard_negatives, str(output_dir),
                                     prefix=args.prefix,
                                     include_positives=True)
        log.info("Wrote Alpaca: %d pos + %d neg = %d",
                 result["alpaca"]["positive"], result["alpaca"]["negative"],
                 result["alpaca"]["total"])

    # Stats file
    stats = {
        "total_queries": len(queries),
        "total_perturbations": len(all_samples),
        "nmr_prediction_success": n_ok,
        "hard_negatives_selected": len(hard_negatives),
        "top_k": args.top_k,
        "spectral_similarity": {
            "mean": sum(sims) / len(sims) if sims else 0,
            "min": min(sims) if sims else 0,
            "max": max(sims) if sims else 0,
        } if hard_negatives else {},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    stats_path = output_dir / f"{args.prefix}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Stats: %s", stats_path)
    log.info("Done in %.1fs.", time.time() - t0)


if __name__ == "__main__":
    main()
