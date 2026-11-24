"""Perturbation engine: iterate molecules × operators → PerturbedSample list.

The engine takes input records (from LMDB or plain SMILES lists), applies every
perturbation operator, validates the output, computes formula deltas and edit
lower bounds, deduplicates, and returns a list of :class:`PerturbedSample`.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .base import (
    BasePerturbation,
    PerturbedSample,
    canonical_smiles,
    mol_from_smiles,
    molecule_formula,
)
from .operators import build_all_perturbation_operators

# Import formula helpers from repair_tool (for delta calculation and edit_lb)
import sys, os
_REPAIR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "Xmaster_AgentRL", "mol_repair")
)
if _REPAIR_ROOT not in sys.path:
    sys.path.insert(0, _REPAIR_ROOT)

from repair_tool.formula import (
    parse_formula,
    delta_counts,
    counts_to_sig,
    edit_lb,
    counts_equal,
)

logger = logging.getLogger(__name__)


class PerturbationEngine:
    """Generate perturbed SMILES samples from correct (SMILES, formula, NMR) records.

    Parameters
    ----------
    operators : list, optional
        Perturbation operators to use. Defaults to all 4 families.
    max_per_operator : int
        Maximum number of perturbation sites to sample per operator per molecule.
    max_edit_lb : int
        Maximum edit lower-bound; perturbations with higher edit_lb are dropped.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        operators: Optional[List] = None,
        *,
        max_per_operator: int = 3,
        max_edit_lb: int = 3,
        seed: Optional[int] = None,
    ):
        self.operators = operators or build_all_perturbation_operators()
        self.max_per_operator = max_per_operator
        self.max_edit_lb = max_edit_lb
        self._rng = random.Random(seed)
        self._case_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_record(self, record: Dict[str, Any]) -> List[PerturbedSample]:
        """Generate perturbations from a single LMDB record.

        The record must have at least ``smiles_canonical`` and
        ``molecular_formula``.  NMR fields (``c_nmr_peaks``, ``h_nmr_peaks``,
        ``nmr_frequency``, ``nmr_solvent``) are optional and will be carried
        through to the output samples.
        """
        smi = record.get("smiles_canonical") or record.get("smiles", "")
        formula = record.get("molecular_formula", "")
        if not smi or not formula:
            return []

        mol = mol_from_smiles(smi)
        if mol is None:
            return []

        gt_canonical = canonical_smiles(mol)
        if not gt_canonical:
            return []

        # Recompute formula from RDKit to ensure consistency
        gt_formula = molecule_formula(mol)
        if not gt_formula:
            return []
        gt_counts = parse_formula(gt_formula)

        # NMR data (carry through from the record)
        c_nmr = record.get("c_nmr_peaks", [])
        h_nmr = record.get("h_nmr_peaks", [])
        nmr_freq = record.get("nmr_frequency", "")
        nmr_solvent = record.get("nmr_solvent", "")

        samples: List[PerturbedSample] = []
        seen_canonical: set = set()

        for op in self.operators:
            try:
                sites = op.detect(mol)
            except Exception:
                logger.debug("Operator %s detection failed for %s",
                             op.name, smi, exc_info=True)
                continue

            # Sample at most max_per_operator sites
            if len(sites) > self.max_per_operator:
                sites = self._rng.sample(sites, self.max_per_operator)

            for site in sites:
                try:
                    perturbed_smi = op.apply(mol, site)
                except Exception:
                    logger.debug("Operator %s apply failed for site %s",
                                 op.name, site, exc_info=True)
                    continue

                if not perturbed_smi:
                    continue

                # Validate the perturbed SMILES
                perturbed_mol = mol_from_smiles(perturbed_smi)
                if perturbed_mol is None:
                    continue

                perturbed_canonical = canonical_smiles(perturbed_mol)
                if not perturbed_canonical:
                    continue

                # Must be different from original
                if perturbed_canonical == gt_canonical:
                    continue

                # Deduplicate
                if perturbed_canonical in seen_canonical:
                    continue
                seen_canonical.add(perturbed_canonical)

                # Compute formula of perturbed molecule
                perturbed_formula = molecule_formula(perturbed_mol)
                if not perturbed_formula:
                    continue

                perturbed_counts = parse_formula(perturbed_formula)

                # Must have different formula (repair_tool needs formula mismatch)
                if counts_equal(perturbed_counts, gt_counts):
                    continue

                # Compute delta: target(correct) - current(perturbed)
                # This is what the repair_tool would see
                delta = delta_counts(gt_counts, perturbed_counts)
                lb = edit_lb(delta)

                # Filter by max edit_lb
                if lb > self.max_edit_lb:
                    continue

                # Assign case_id
                self._case_counter += 1
                case_id = f"perturb_{self._case_counter:06d}"

                # Determine difficulty
                if lb <= 1:
                    difficulty = "easy"
                elif lb == 2:
                    difficulty = "medium"
                else:
                    difficulty = "hard"

                # Generate reasoning
                try:
                    reasoning = op.reasoning(site, gt_canonical, perturbed_canonical)
                except Exception:
                    reasoning = f"Repair by reversing {op.name}/{site.sub_variant}"

                sample = PerturbedSample(
                    input_smiles=perturbed_canonical,
                    input_formula=perturbed_formula,
                    target_formula=gt_formula,
                    gt_smiles=gt_canonical,
                    formula_delta=counts_to_sig(delta),
                    edit_lb=lb,
                    c_nmr_peaks=c_nmr if c_nmr else None,
                    h_nmr_peaks=h_nmr if h_nmr else None,
                    nmr_frequency=nmr_freq or None,
                    nmr_solvent=nmr_solvent or None,
                    perturbation_type=f"{op.name}/{site.sub_variant}",
                    case_id=case_id,
                )

                samples.append(sample)

        return samples

    def generate_from_smiles(
        self,
        correct_smiles: str,
        correct_formula: Optional[str] = None,
    ) -> List[PerturbedSample]:
        """Generate perturbations from a single correct SMILES string.

        Convenience method when no NMR data is available.
        """
        record = {"smiles_canonical": correct_smiles}
        if correct_formula:
            record["molecular_formula"] = correct_formula
        else:
            f = molecule_formula(correct_smiles)
            if f:
                record["molecular_formula"] = f
        return self.generate_from_record(record)

    def generate_batch(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        progress: bool = False,
    ) -> List[PerturbedSample]:
        """Generate perturbations for a batch of records.

        Parameters
        ----------
        records : list of dict
            Each record should have ``smiles_canonical`` and ``molecular_formula``.
        progress : bool
            If True, use tqdm for a progress bar.

        Returns
        -------
        list of PerturbedSample
        """
        all_samples: List[PerturbedSample] = []

        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(records, desc="Generating perturbations")
            except ImportError:
                iterator = records
        else:
            iterator = records

        for rec in iterator:
            samples = self.generate_from_record(rec)
            all_samples.extend(samples)

        return all_samples

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def summary(self, samples: Sequence[PerturbedSample]) -> Dict[str, Any]:
        """Compute summary statistics over a set of generated samples."""
        from collections import Counter

        by_type = Counter(s.perturbation_type for s in samples)
        by_edit_lb = Counter(s.edit_lb for s in samples)
        by_difficulty = Counter(
            ("easy" if s.edit_lb <= 1 else "medium" if s.edit_lb == 2 else "hard")
            for s in samples
        )

        return {
            "total": len(samples),
            "by_perturbation_type": dict(by_type.most_common()),
            "by_edit_lb": dict(sorted(by_edit_lb.items())),
            "by_difficulty": dict(by_difficulty),
            "unique_gt_smiles": len(set(s.gt_smiles for s in samples)),
            "unique_input_smiles": len(set(s.input_smiles for s in samples)),
        }
