"""CLI entry point for negative sample generation.

Usage::

    python -m perturbation_generator.negative_cli \\
        --query_lmdb /fs_mol/fjxu/.../test_100sample.lmdb \\
        --pool_lmdb /fs_mol/fjxu/.../si_0820/all.lmdb \\
        --output_dir /fs_mol/cxq/SpecMaster_jp/negative_output/ \\
        --cache_dir /fs_mol/cxq/SpecMaster_jp/.pool_cache/ \\
        --n_random 5 --n_same_formula 10 --n_similar 5 \\
        --format both --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .formatter import (
    write_matching_alpaca_json,
    write_matching_jsonl,
    write_matching_both,
)
from .lmdb_reader import read_lmdb
from .negative_engine import NegativeEngine
from .pool_index import PoolIndex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate random / same-formula / similar-structure "
                    "negative samples from an LMDB pool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    p.add_argument("--query_lmdb", type=str, required=True,
                   help="LMDB with query molecules (correct SMILES + NMR)")
    p.add_argument("--pool_lmdb", type=str, required=True,
                   help="LMDB pool to draw negatives from (can be the same)")

    # Output
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory")
    p.add_argument("--prefix", type=str, default="negatives",
                   help="Filename prefix (default: negatives)")
    p.add_argument("--format", type=str, default="both",
                   choices=["alpaca", "jsonl", "both"],
                   help="Output format (default: both)")

    # Pool index
    p.add_argument("--cache_dir", type=str,
                   default="/fs_mol/cxq/SpecMaster_jp/.pool_cache/",
                   help="Directory for pool index cache")
    p.add_argument("--pool_max_records", type=int, default=None,
                   help="Cap pool scan (for testing)")

    # Negative counts
    p.add_argument("--n_random", type=int, default=5,
                   help="Random negatives per query (default: 5)")
    p.add_argument("--n_same_formula", type=int, default=10,
                   help="Max same-formula negatives per query (default: 10)")
    p.add_argument("--n_similar", type=int, default=5,
                   help="Similar-structure negatives per query (default: 5)")
    p.add_argument("--min_similarity", type=float, default=0.3,
                   help="Min Tanimoto for similar negatives (default: 0.3)")

    # Query set
    p.add_argument("--max_queries", type=int, default=None,
                   help="Cap query records (for testing)")
    p.add_argument("--require_nmr", action="store_true",
                   help="Only use queries with NMR data")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no_nmr_in_output", action="store_true",
                   help="Exclude NMR from Alpaca instruction")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose logging")

    return p.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("negative_sample_generator")

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load query set
    # ------------------------------------------------------------------
    log.info("Loading query set: %s", args.query_lmdb)
    queries = read_lmdb(
        args.query_lmdb,
        require_nmr=args.require_nmr,
        max_records=args.max_queries,
    )
    log.info("Loaded %d query records", len(queries))
    if not queries:
        log.error("No valid query records. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Build / load pool index
    # ------------------------------------------------------------------
    log.info("Building pool index from: %s", args.pool_lmdb)
    pool = PoolIndex.get_or_build(
        args.pool_lmdb,
        args.cache_dir,
        max_records=args.pool_max_records,
        progress=True,
    )
    log.info("Pool index ready: %d molecules, %d unique formulas",
             pool.size, len(pool.formula_to_indices))

    # ------------------------------------------------------------------
    # 3. Generate negatives
    # ------------------------------------------------------------------
    engine = NegativeEngine(
        pool,
        n_random=args.n_random,
        n_same_formula=args.n_same_formula,
        n_similar=args.n_similar,
        min_similarity=args.min_similarity,
        seed=args.seed,
    )

    log.info("Generating negatives (random=%d, same_formula=%d, similar=%d)...",
             args.n_random, args.n_same_formula, args.n_similar)
    samples = engine.generate_batch(queries, progress=True)
    stats = engine.summary(samples)

    log.info("Generated %d negative samples from %d queries",
             stats["total"], len(queries))
    log.info("  By type:")
    for ntype, count in stats["by_negative_type"].items():
        log.info("    %s: %d", ntype, count)
    if stats["similar_structure_tanimoto"]:
        sim = stats["similar_structure_tanimoto"]
        log.info("  Similar-structure Tanimoto: mean=%.3f, min=%.3f, max=%.3f",
                 sim["mean"], sim["min"], sim["max"])
    log.info("  Queries with same-formula negatives: %d / %d",
             stats["queries_with_same_formula_negatives"], len(queries))

    # ------------------------------------------------------------------
    # 4. Write output (spectrum-matching format: NMR + candidate → Yes/No)
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "alpaca":
        outfile = str(output_dir / f"{args.prefix}_alpaca.json")
        counts = write_matching_alpaca_json(samples, outfile,
                                            include_positives=True)
        log.info("Wrote %d records (%d pos + %d neg) to %s",
                 counts["total"], counts["positive"], counts["negative"], outfile)
    elif args.format == "jsonl":
        outfile = str(output_dir / f"{args.prefix}_matching.jsonl")
        counts = write_matching_jsonl(samples, outfile,
                                      include_positives=True)
        log.info("Wrote %d records (%d pos + %d neg) to %s",
                 counts["total"], counts["positive"], counts["negative"], outfile)
    else:
        result = write_matching_both(samples, str(output_dir),
                                     prefix=args.prefix,
                                     include_positives=True)
        log.info("Wrote Alpaca: %d pos + %d neg = %d total",
                 result["alpaca"]["positive"], result["alpaca"]["negative"],
                 result["alpaca"]["total"])
        log.info("Wrote JSONL:  %d pos + %d neg = %d total",
                 result["jsonl"]["positive"], result["jsonl"]["negative"],
                 result["jsonl"]["total"])

    # Write stats
    stats_path = output_dir / f"{args.prefix}_stats.json"
    stats["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    log.info("Stats written to %s", stats_path)

    elapsed = time.time() - t0
    log.info("Done in %.1f seconds. Total: %d negative samples.", elapsed, len(samples))


if __name__ == "__main__":
    main()
