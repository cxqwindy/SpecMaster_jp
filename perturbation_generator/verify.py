"""Verification pipeline: run repair_tool on generated samples.

For each :class:`PerturbedSample`, call ``repair_tool.api.repair()`` and
record whether the repair succeeded, which candidate was produced, and
whether it structurally matches the ground truth.
"""

from __future__ import annotations

import logging
import multiprocessing
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import PerturbedSample, canonical_smiles

# Import repair_tool
import sys, os
_REPAIR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "Xmaster_AgentRL", "mol_repair")
)
if _REPAIR_ROOT not in sys.path:
    sys.path.insert(0, _REPAIR_ROOT)

logger = logging.getLogger(__name__)


def _verify_single(args: Tuple[PerturbedSample, Dict[str, Any]]) -> PerturbedSample:
    """Verify one sample (worker function for multiprocessing)."""
    sample, config = args

    # Lazy imports inside worker to avoid pickling issues
    from repair_tool.api import repair
    from repair_tool.operators.library import build_default_operators

    beam_size = config.get("beam_size", 64)
    max_rounds = config.get("max_rounds", 2)
    topk = config.get("topk_output", 5)
    enable_fallback = config.get("enable_fallback", True)

    operators = build_default_operators()

    try:
        result = repair(
            candidates=[sample.input_smiles],
            target_formula=sample.target_formula,
            operators=operators,
            beam_size=beam_size,
            max_rounds=max_rounds,
            topk_output=topk,
            fallback_enabled=enable_fallback,
        )
    except Exception as exc:
        logger.debug("repair() failed for %s: %s", sample.input_smiles, exc)
        return replace(
            sample,
            repair_verified=False,
            repair_pred_smiles=None,
            structurally_correct=False,
            fallback_used=None,
        )

    is_success = result.is_success
    pred_smi = None
    struct_correct = False

    if is_success and result.ranked_exact_candidates:
        pred_smi = result.ranked_exact_candidates[0].smiles
        gt_canon = canonical_smiles(sample.gt_smiles)
        pred_canon = canonical_smiles(pred_smi)
        struct_correct = (gt_canon is not None
                          and pred_canon is not None
                          and gt_canon == pred_canon)

    return replace(
        sample,
        repair_verified=is_success,
        repair_pred_smiles=pred_smi,
        structurally_correct=struct_correct,
        fallback_used=result.fallback_used,
    )


class VerificationPipeline:
    """Run repair_tool on generated samples and filter by success.

    Parameters
    ----------
    beam_size : int
        Beam size for repair_tool.
    max_rounds : int
        Max beam search rounds.
    topk_output : int
        Number of top candidates to return from repair.
    enable_fallback : bool
        Whether to enable the 3-tier fallback in repair_tool.
    workers : int
        Number of parallel workers (0 = single-process).
    """

    def __init__(
        self,
        *,
        beam_size: int = 64,
        max_rounds: int = 2,
        topk_output: int = 5,
        enable_fallback: bool = True,
        workers: int = 0,
    ):
        self.config = {
            "beam_size": beam_size,
            "max_rounds": max_rounds,
            "topk_output": topk_output,
            "enable_fallback": enable_fallback,
        }
        self.workers = workers

    def verify_batch(
        self,
        samples: Sequence[PerturbedSample],
        *,
        keep_mode: str = "formula_exact",
        progress: bool = False,
    ) -> List[PerturbedSample]:
        """Verify a batch of samples and optionally filter.

        Parameters
        ----------
        samples : list of PerturbedSample
        keep_mode : str
            - ``"all"``: return all samples with verification metadata
            - ``"formula_exact"``: only keep formula-exact repairs
            - ``"structurally_correct"``: only keep structurally correct repairs
        progress : bool
            Show a tqdm progress bar.

        Returns
        -------
        list of PerturbedSample
            Samples with verification fields populated.
        """
        work = [(s, self.config) for s in samples]

        if self.workers > 0:
            verified = self._run_parallel(work, progress)
        else:
            verified = self._run_sequential(work, progress)

        # Filter by keep_mode
        if keep_mode == "all":
            return verified
        elif keep_mode == "formula_exact":
            return [s for s in verified if s.repair_verified]
        elif keep_mode == "structurally_correct":
            return [s for s in verified if s.structurally_correct]
        else:
            raise ValueError(f"Unknown keep_mode: {keep_mode!r}")

    def _run_sequential(
        self,
        work: List[Tuple[PerturbedSample, Dict]],
        progress: bool,
    ) -> List[PerturbedSample]:
        results = []
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(work, desc="Verifying repairs")
            except ImportError:
                iterator = work
        else:
            iterator = work

        for item in iterator:
            results.append(_verify_single(item))
        return results

    def _run_parallel(
        self,
        work: List[Tuple[PerturbedSample, Dict]],
        progress: bool,
    ) -> List[PerturbedSample]:
        results = []
        with multiprocessing.Pool(self.workers) as pool:
            if progress:
                try:
                    from tqdm import tqdm
                    for res in tqdm(
                        pool.imap(_verify_single, work),
                        total=len(work),
                        desc="Verifying repairs",
                    ):
                        results.append(res)
                except ImportError:
                    results = pool.map(_verify_single, work)
            else:
                results = pool.map(_verify_single, work)
        return results

    def summary(
        self,
        verified: Sequence[PerturbedSample],
    ) -> Dict[str, Any]:
        """Compute verification statistics."""
        from collections import Counter

        total = len(verified)
        n_success = sum(1 for s in verified if s.repair_verified)
        n_struct = sum(1 for s in verified if s.structurally_correct)

        by_type = {}
        for s in verified:
            ptype = s.perturbation_type
            if ptype not in by_type:
                by_type[ptype] = {"total": 0, "repair_success": 0,
                                  "struct_correct": 0}
            by_type[ptype]["total"] += 1
            if s.repair_verified:
                by_type[ptype]["repair_success"] += 1
            if s.structurally_correct:
                by_type[ptype]["struct_correct"] += 1

        fallback_counts = Counter(
            s.fallback_used for s in verified if s.repair_verified
        )

        return {
            "total": total,
            "repair_success": n_success,
            "repair_success_rate": n_success / total if total else 0.0,
            "structurally_correct": n_struct,
            "structural_accuracy": n_struct / total if total else 0.0,
            "by_perturbation_type": by_type,
            "fallback_distribution": dict(fallback_counts),
        }
