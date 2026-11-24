"""Perturbation sample generator for SMILES repair training data augmentation.

Given correct (SMILES, formula, NMR) triplets, generate perturbed SMILES that
can be repaired by the repair_tool pipeline. Output pairs:
  (perturbed_SMILES + correct_NMR + correct_formula) for SFT training.
"""

from .base import (
    BasePerturbation,
    PerturbationSite,
    PerturbedSample,
)
from .lmdb_reader import read_lmdb
from .operators import build_all_perturbation_operators
from .engine import PerturbationEngine
from .formatter import (
    write_alpaca_json,
    write_benchmark_jsonl,
    write_both,
)

__all__ = [
    "BasePerturbation",
    "PerturbationSite",
    "PerturbedSample",
    "PerturbationEngine",
    "build_all_perturbation_operators",
    "read_lmdb",
    "write_alpaca_json",
    "write_benchmark_jsonl",
    "write_both",
]
