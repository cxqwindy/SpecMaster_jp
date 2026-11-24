"""CLI entry point for the perturbation sample generator.

Usage::

    python -m perturbation_generator.cli \\
        --lmdb_path /fs_mol/fjxu/.../test_100sample.lmdb \\
        --output_dir ./output \\
        --max_edit_lb 3 \\
        --max_per_operator 3 \\
        --verify \\
        --workers 4

Or with a plain SMILES file::

    python -m perturbation_generator.cli \\
        --smiles_file correct_smiles.txt \\
        --output_dir ./output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from .base import PerturbedSample
from .engine import PerturbationEngine
from .formatter import write_alpaca_json, write_benchmark_jsonl, write_both
from .operators import build_all_perturbation_operators


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate perturbed SMILES samples for repair_tool training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input source (one of --lmdb_path or --smiles_file)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--lmdb_path", type=str,
        help="Path to an LMDB file with (SMILES, formula, NMR) records",
    )
    input_group.add_argument(
        "--smiles_file", type=str,
        help="Path to a text file with one correct SMILES per line",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--prefix", type=str, default="perturbed",
        help="Filename prefix for output files (default: perturbed)",
    )
    parser.add_argument(
        "--format", type=str, default="both",
        choices=["alpaca", "jsonl", "both"],
        help="Output format (default: both)",
    )

    # Engine parameters
    parser.add_argument(
        "--max_edit_lb", type=int, default=3,
        help="Maximum edit lower-bound for perturbations (default: 3)",
    )
    parser.add_argument(
        "--max_per_operator", type=int, default=3,
        help="Max perturbation sites per operator per molecule (default: 3)",
    )
    parser.add_argument(
        "--max_records", type=int, default=None,
        help="Cap the number of input records to process (for testing)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # Verification
    parser.add_argument(
        "--verify", action="store_true",
        help="Run repair_tool verification on generated samples",
    )
    parser.add_argument(
        "--keep_mode", type=str, default="formula_exact",
        choices=["all", "formula_exact", "structurally_correct"],
        help="Which verified samples to keep (default: formula_exact)",
    )
    parser.add_argument(
        "--beam_size", type=int, default=64,
        help="Beam size for repair verification (default: 64)",
    )
    parser.add_argument(
        "--max_rounds", type=int, default=2,
        help="Max beam search rounds for verification (default: 2)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel workers for verification (0=sequential, default: 0)",
    )

    # NMR
    parser.add_argument(
        "--require_nmr", action="store_true",
        help="Only use LMDB records that have NMR data",
    )
    parser.add_argument(
        "--no_nmr_in_output", action="store_true",
        help="Exclude NMR data from Alpaca instruction",
    )

    # Misc
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_smiles_file(path: str) -> List[Dict]:
    """Load a text file with one SMILES per line as records."""
    from .base import mol_from_smiles, molecule_formula

    records = []
    with open(path, "r") as f:
        for line in f:
            smi = line.strip()
            if not smi or smi.startswith("#"):
                continue
            formula = molecule_formula(smi)
            if not formula:
                continue
            records.append({
                "smiles_canonical": smi,
                "molecular_formula": formula,
            })
    return records


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("perturbation_generator")

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load input data
    # ------------------------------------------------------------------
    if args.lmdb_path:
        from .lmdb_reader import read_lmdb
        log.info("Reading LMDB: %s", args.lmdb_path)
        records = read_lmdb(
            args.lmdb_path,
            require_nmr=args.require_nmr,
            max_records=args.max_records,
        )
    else:
        log.info("Reading SMILES file: %s", args.smiles_file)
        records = load_smiles_file(args.smiles_file)
        if args.max_records:
            records = records[:args.max_records]

    log.info("Loaded %d valid records", len(records))
    if not records:
        log.error("No valid records found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Generate perturbations
    # ------------------------------------------------------------------
    engine = PerturbationEngine(
        max_per_operator=args.max_per_operator,
        max_edit_lb=args.max_edit_lb,
        seed=args.seed,
    )

    log.info("Generating perturbations (max_per_operator=%d, max_edit_lb=%d)...",
             args.max_per_operator, args.max_edit_lb)

    samples = engine.generate_batch(records, progress=True)
    gen_stats = engine.summary(samples)

    log.info("Generated %d raw samples from %d records",
             gen_stats["total"], len(records))
    log.info("  By perturbation type:")
    for ptype, count in gen_stats["by_perturbation_type"].items():
        log.info("    %s: %d", ptype, count)
    log.info("  By edit_lb:")
    for lb, count in sorted(gen_stats["by_edit_lb"].items()):
        log.info("    edit_lb=%d: %d", lb, count)

    # ------------------------------------------------------------------
    # 3. Verification (optional)
    # ------------------------------------------------------------------
    if args.verify:
        from .verify import VerificationPipeline

        log.info("Running repair_tool verification (workers=%d)...",
                 args.workers)
        pipeline = VerificationPipeline(
            beam_size=args.beam_size,
            max_rounds=args.max_rounds,
            workers=args.workers,
        )

        samples = pipeline.verify_batch(
            samples,
            keep_mode=args.keep_mode,
            progress=True,
        )

        v_stats = pipeline.summary(samples)
        log.info("Verification results:")
        log.info("  Total verified: %d", v_stats["total"])
        log.info("  Repair success: %d (%.1f%%)",
                 v_stats["repair_success"],
                 v_stats["repair_success_rate"] * 100)
        log.info("  Structurally correct: %d (%.1f%%)",
                 v_stats["structurally_correct"],
                 v_stats["structural_accuracy"] * 100)
        log.info("  Fallback distribution: %s",
                 v_stats["fallback_distribution"])

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_nmr = not args.no_nmr_in_output

    if args.format == "alpaca":
        outfile = str(output_dir / f"{args.prefix}_alpaca.json")
        n = write_alpaca_json(samples, outfile, include_nmr=include_nmr)
        log.info("Wrote %d Alpaca samples to %s", n, outfile)
    elif args.format == "jsonl":
        outfile = str(output_dir / f"{args.prefix}_benchmark.jsonl")
        n = write_benchmark_jsonl(samples, outfile)
        log.info("Wrote %d JSONL records to %s", n, outfile)
    else:  # both
        counts = write_both(
            samples, str(output_dir),
            prefix=args.prefix, include_nmr=include_nmr,
        )
        log.info("Wrote %d Alpaca + %d JSONL records to %s",
                 counts["alpaca"], counts["jsonl"], output_dir)

    # Write summary stats
    stats = {
        "generation": gen_stats,
        "output_count": len(samples),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if args.verify:
        stats["verification"] = v_stats

    stats_path = output_dir / f"{args.prefix}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    log.info("Stats written to %s", stats_path)

    elapsed = time.time() - t0
    log.info("Done in %.1f seconds. Total output: %d samples.",
             elapsed, len(samples))


if __name__ == "__main__":
    main()
