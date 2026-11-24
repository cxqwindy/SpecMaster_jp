"""Negative sample engine: generate random / same-formula / similar-structure negatives.

Takes query records (from LMDB) and a :class:`PoolIndex`, produces
:class:`PerturbedSample` objects with ``perturbation_type`` set to the
negative type (``neg/random``, ``neg/same_formula``, ``neg/similar_structure``).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence

from .base import PerturbedSample, canonical_smiles, mol_from_smiles, molecule_formula
from .pool_index import PoolIndex

# Import formula utilities from repair_tool
import sys, os

_REPAIR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "Xmaster_AgentRL", "mol_repair")
)
if _REPAIR_ROOT not in sys.path:
    sys.path.insert(0, _REPAIR_ROOT)

from repair_tool.formula import parse_formula, delta_counts, counts_to_sig, edit_lb

logger = logging.getLogger(__name__)


class NegativeEngine:
    """Generate three types of negative samples from a pool index.

    Parameters
    ----------
    pool_index : PoolIndex
        Pre-built index over the negative sample pool.
    n_random : int
        Number of random negatives per query molecule.
    n_same_formula : int
        Max number of same-formula negatives per query molecule.
    n_similar : int
        Number of similar-structure negatives per query molecule.
    min_similarity : float
        Minimum Tanimoto similarity for similar-structure negatives.
    exclude_same_formula_from_similar : bool
        If True, exclude same-formula molecules from similar-structure results
        (to avoid overlap with same-formula negatives).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        pool_index: PoolIndex,
        *,
        n_random: int = 5,
        n_same_formula: int = 10,
        n_similar: int = 5,
        min_similarity: float = 0.3,
        exclude_same_formula_from_similar: bool = True,
        seed: Optional[int] = None,
    ):
        self.pool = pool_index
        self.n_random = n_random
        self.n_same_formula = n_same_formula
        self.n_similar = n_similar
        self.min_similarity = min_similarity
        self.exclude_same_formula_from_similar = exclude_same_formula_from_similar
        self._rng = random.Random(seed)
        self._case_counter = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_case_id(self, prefix: str) -> str:
        self._case_counter += 1
        return f"{prefix}_{self._case_counter:06d}"

    def _make_sample(
        self,
        neg_smiles: str,
        query_record: Dict[str, Any],
        *,
        negative_type: str,
        tanimoto_sim: Optional[float] = None,
    ) -> Optional[PerturbedSample]:
        """Build a PerturbedSample pairing *neg_smiles* with query's NMR/formula."""
        gt_smi = query_record.get("smiles_canonical", "")
        gt_formula = query_record.get("molecular_formula", "")

        # Compute neg formula
        neg_formula = molecule_formula(neg_smiles)
        if not neg_formula:
            return None

        # Compute formula delta (target - input)
        gt_counts = parse_formula(gt_formula)
        neg_counts = parse_formula(neg_formula)
        delta = delta_counts(gt_counts, neg_counts)
        delta_sig = counts_to_sig(delta) if delta else "ZERO"
        lb = edit_lb(delta) if delta else 0

        return PerturbedSample(
            input_smiles=neg_smiles,
            input_formula=neg_formula,
            target_formula=gt_formula,
            gt_smiles=gt_smi,
            formula_delta=delta_sig,
            edit_lb=lb,
            c_nmr_peaks=query_record.get("c_nmr_peaks"),
            h_nmr_peaks=query_record.get("h_nmr_peaks"),
            nmr_frequency=query_record.get("nmr_frequency"),
            nmr_solvent=query_record.get("nmr_solvent"),
            perturbation_type=negative_type,
            case_id=self._next_case_id(negative_type.replace("/", "_")),
            source_tool="negative_sample_generator",
            tanimoto_similarity=tanimoto_sim,
        )

    # ------------------------------------------------------------------
    # Per-type generators
    # ------------------------------------------------------------------

    def _gen_random(self, record: Dict[str, Any]) -> List[PerturbedSample]:
        """Generate random negatives for one query."""
        gt_smi = record.get("smiles_canonical", "")
        gt_formula = record.get("molecular_formula", "")
        if not gt_smi or not gt_formula:
            return []

        neg_list = self.pool.random_negatives(
            gt_smi, gt_formula, self.n_random,
            exclude_same_formula=True, rng=self._rng,
        )

        samples = []
        for neg_smi in neg_list:
            s = self._make_sample(neg_smi, record, negative_type="neg/random")
            if s is not None:
                samples.append(s)
        return samples

    def _gen_same_formula(self, record: Dict[str, Any]) -> List[PerturbedSample]:
        """Generate same-formula negatives for one query."""
        gt_smi = record.get("smiles_canonical", "")
        gt_formula = record.get("molecular_formula", "")
        if not gt_smi or not gt_formula:
            return []

        neg_list = self.pool.formula_negatives(
            gt_formula, gt_smi, self.n_same_formula, rng=self._rng,
        )

        samples = []
        for neg_smi in neg_list:
            s = self._make_sample(neg_smi, record, negative_type="neg/same_formula")
            if s is not None:
                samples.append(s)
        return samples

    def _gen_similar(self, record: Dict[str, Any]) -> List[PerturbedSample]:
        """Generate similar-structure negatives for one query."""
        gt_smi = record.get("smiles_canonical", "")
        gt_formula = record.get("molecular_formula", "")
        if not gt_smi or not gt_formula:
            return []

        results = self.pool.similar_negatives(
            gt_smi, gt_formula, self.n_similar,
            exclude_same_formula=self.exclude_same_formula_from_similar,
            min_similarity=self.min_similarity,
        )

        samples = []
        for neg_smi, sim in results:
            s = self._make_sample(
                neg_smi, record,
                negative_type="neg/similar_structure",
                tanimoto_sim=sim,
            )
            if s is not None:
                samples.append(s)
        return samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_record(self, record: Dict[str, Any]) -> List[PerturbedSample]:
        """Generate all three types of negatives for a single query record."""
        samples: List[PerturbedSample] = []

        if self.n_random > 0:
            samples.extend(self._gen_random(record))
        if self.n_same_formula > 0:
            samples.extend(self._gen_same_formula(record))
        if self.n_similar > 0:
            samples.extend(self._gen_similar(record))

        return samples

    def generate_batch(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        progress: bool = False,
    ) -> List[PerturbedSample]:
        """Generate negatives for a batch of query records."""
        all_samples: List[PerturbedSample] = []

        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(records, desc="Generating negatives")
            except ImportError:
                iterator = records
        else:
            iterator = records

        for rec in iterator:
            all_samples.extend(self.generate_from_record(rec))

        return all_samples

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def summary(self, samples: Sequence[PerturbedSample]) -> Dict[str, Any]:
        """Compute summary statistics for generated negative samples."""
        from collections import Counter

        by_type = Counter(s.perturbation_type for s in samples)
        by_edit_lb = Counter(s.edit_lb for s in samples)

        # Tanimoto stats for similar-structure negatives
        sim_samples = [s for s in samples
                       if s.tanimoto_similarity is not None]
        sim_values = [s.tanimoto_similarity for s in sim_samples]

        sim_stats = {}
        if sim_values:
            sim_stats = {
                "count": len(sim_values),
                "mean": sum(sim_values) / len(sim_values),
                "min": min(sim_values),
                "max": max(sim_values),
            }

        # Same-formula coverage
        same_formula_queries = set()
        for s in samples:
            if s.perturbation_type == "neg/same_formula":
                same_formula_queries.add(s.gt_smiles)

        return {
            "total": len(samples),
            "by_negative_type": dict(by_type.most_common()),
            "by_edit_lb": dict(sorted(by_edit_lb.items())),
            "unique_gt_smiles": len(set(s.gt_smiles for s in samples)),
            "unique_neg_smiles": len(set(s.input_smiles for s in samples)),
            "similar_structure_tanimoto": sim_stats,
            "queries_with_same_formula_negatives": len(same_formula_queries),
        }
