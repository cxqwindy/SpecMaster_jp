"""Pool index for negative sample generation.

Scans a large LMDB pool of molecules and builds:
  1. Formula → canonical SMILES index (for same-formula negatives)
  2. Packed Morgan fingerprint matrix (for similarity search)
  3. Parallel arrays for random sampling

Supports disk caching to avoid re-scanning on subsequent runs.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .base import mol_from_smiles, canonical_smiles

logger = logging.getLogger(__name__)


class PoolIndex:
    """Pre-built index over a large LMDB molecule pool.

    Attributes
    ----------
    all_smiles : list of str
        Canonical SMILES for every valid molecule, in scan order.
    all_formulas : list of str
        Molecular formula aligned with ``all_smiles``.
    formula_to_indices : dict[str, list[int]]
        Formula string → list of indices into ``all_smiles``.
    fp_matrix : np.ndarray, shape (N, 256), dtype uint8
        Packed Morgan fingerprints (2048 bits → 256 bytes per molecule).
    """

    def __init__(self):
        self.all_smiles: List[str] = []
        self.all_formulas: List[str] = []
        self.formula_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.fp_matrix: Optional[np.ndarray] = None
        self._smiles_set: Optional[set] = None  # lazily built for O(1) lookup

    @property
    def size(self) -> int:
        return len(self.all_smiles)

    def _ensure_smiles_set(self):
        if self._smiles_set is None:
            self._smiles_set = set(self.all_smiles)

    # ------------------------------------------------------------------
    # Build from LMDB
    # ------------------------------------------------------------------

    @classmethod
    def build_from_lmdb(
        cls,
        lmdb_path: str,
        *,
        max_records: Optional[int] = None,
        progress: bool = True,
    ) -> "PoolIndex":
        """Scan an LMDB file and build the full index.

        Parameters
        ----------
        lmdb_path : str
            Path to the ``.lmdb`` file.
        max_records : int, optional
            Cap for testing.
        progress : bool
            Show a tqdm progress bar.
        """
        import lmdb as _lmdb
        from rdkit.Chem import rdMolDescriptors

        idx = cls()
        fp_rows: List[np.ndarray] = []

        env = _lmdb.open(str(lmdb_path), subdir=False, readonly=True,
                         lock=False, readahead=True, meminit=False)
        n_entries = env.stat()["entries"]
        logger.info("Scanning LMDB %s (%d entries)...", lmdb_path, n_entries)

        if progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=min(n_entries, max_records or n_entries),
                            desc="Building pool index")
            except ImportError:
                pbar = None
        else:
            pbar = None

        count = 0
        skipped = 0
        with env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key_buf, val_buf in cursor:
                raw = pickle.loads(bytes(val_buf))

                smi = raw.get("smiles_canonical") or raw.get("smiles")
                formula = raw.get("molecular_formula")
                if not smi or not formula:
                    skipped += 1
                    continue

                mol = mol_from_smiles(str(smi))
                if mol is None:
                    skipped += 1
                    continue

                canon = canonical_smiles(mol)
                if not canon:
                    skipped += 1
                    continue

                # Recompute formula from RDKit for consistency
                try:
                    formula_rdk = rdMolDescriptors.CalcMolFormula(mol)
                except Exception:
                    formula_rdk = str(formula)

                # Compute Morgan fingerprint
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=2, nBits=2048)
                    fp_arr = np.zeros(2048, dtype=np.uint8)
                    for bit in fp.GetOnBits():
                        fp_arr[bit] = 1
                    fp_packed = np.packbits(fp_arr)  # (256,) uint8
                except Exception:
                    skipped += 1
                    continue

                pool_idx = len(idx.all_smiles)
                idx.all_smiles.append(canon)
                idx.all_formulas.append(formula_rdk)
                idx.formula_to_indices[formula_rdk].append(pool_idx)
                fp_rows.append(fp_packed)

                count += 1
                if pbar is not None:
                    pbar.update(1)

                if max_records is not None and count >= max_records:
                    break

        if pbar is not None:
            pbar.close()
        env.close()

        idx.fp_matrix = np.array(fp_rows, dtype=np.uint8)  # (N, 256)
        logger.info("Pool index built: %d molecules, %d formulas, %d skipped",
                    idx.size, len(idx.formula_to_indices), skipped)
        return idx

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(lmdb_path: str) -> str:
        return hashlib.md5(str(lmdb_path).encode()).hexdigest()[:12]

    def save_cache(self, cache_dir: str, lmdb_path: str) -> None:
        """Save index to disk for fast reload."""
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(lmdb_path)

        with open(d / f"{key}_smiles.pkl", "wb") as f:
            pickle.dump(self.all_smiles, f)
        with open(d / f"{key}_formulas.pkl", "wb") as f:
            pickle.dump(self.all_formulas, f)
        with open(d / f"{key}_formula_index.pkl", "wb") as f:
            pickle.dump(dict(self.formula_to_indices), f)
        np.save(str(d / f"{key}_fp_matrix.npy"), self.fp_matrix)
        logger.info("Cache saved to %s (key=%s)", cache_dir, key)

    @classmethod
    def load_cache(cls, cache_dir: str, lmdb_path: str) -> Optional["PoolIndex"]:
        """Load index from disk cache, or return None if not found."""
        d = Path(cache_dir)
        key = cls._cache_key(lmdb_path)

        files = [
            d / f"{key}_smiles.pkl",
            d / f"{key}_formulas.pkl",
            d / f"{key}_formula_index.pkl",
            d / f"{key}_fp_matrix.npy",
        ]
        if not all(f.exists() for f in files):
            return None

        logger.info("Loading pool index from cache (key=%s)...", key)
        idx = cls()
        with open(files[0], "rb") as f:
            idx.all_smiles = pickle.load(f)
        with open(files[1], "rb") as f:
            idx.all_formulas = pickle.load(f)
        with open(files[2], "rb") as f:
            idx.formula_to_indices = defaultdict(list, pickle.load(f))
        idx.fp_matrix = np.load(str(files[3]))
        logger.info("Cache loaded: %d molecules, %d formulas",
                    idx.size, len(idx.formula_to_indices))
        return idx

    @classmethod
    def get_or_build(
        cls,
        lmdb_path: str,
        cache_dir: str,
        *,
        max_records: Optional[int] = None,
        progress: bool = True,
    ) -> "PoolIndex":
        """Load from cache if available, otherwise build and cache."""
        cached = cls.load_cache(cache_dir, lmdb_path)
        if cached is not None:
            return cached
        idx = cls.build_from_lmdb(lmdb_path, max_records=max_records,
                                  progress=progress)
        idx.save_cache(cache_dir, lmdb_path)
        return idx

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def random_negatives(
        self,
        query_smiles: str,
        query_formula: str,
        n: int,
        *,
        exclude_same_formula: bool = True,
        rng: Optional[random.Random] = None,
    ) -> List[str]:
        """Randomly sample *n* SMILES different from *query_smiles*.

        Parameters
        ----------
        query_smiles : str
            Canonical SMILES to exclude.
        query_formula : str
            Formula of query (excluded if *exclude_same_formula*).
        n : int
            Number of negatives.
        exclude_same_formula : bool
            If True, skip molecules with the same formula as query.
        rng : random.Random, optional
            Random state.
        """
        if rng is None:
            rng = random.Random()

        self._ensure_smiles_set()
        pool_size = self.size
        results: List[str] = []
        attempts = 0
        max_attempts = n * 20

        while len(results) < n and attempts < max_attempts:
            attempts += 1
            idx = rng.randint(0, pool_size - 1)
            smi = self.all_smiles[idx]
            if smi == query_smiles:
                continue
            if exclude_same_formula and self.all_formulas[idx] == query_formula:
                continue
            if smi not in [r for r in results]:  # avoid duplicates
                results.append(smi)

        return results

    def formula_negatives(
        self,
        formula: str,
        exclude_smiles: str,
        max_n: int = 10,
        *,
        rng: Optional[random.Random] = None,
    ) -> List[str]:
        """Find molecules with the same formula (constitutional isomers).

        Returns at most *max_n* canonical SMILES, excluding *exclude_smiles*.
        """
        indices = self.formula_to_indices.get(formula, [])
        candidates = [
            self.all_smiles[i] for i in indices
            if self.all_smiles[i] != exclude_smiles
        ]
        # Deduplicate (there may be SMILES duplicates across records)
        candidates = list(dict.fromkeys(candidates))

        if len(candidates) <= max_n:
            return candidates

        if rng is None:
            rng = random.Random()
        return rng.sample(candidates, max_n)

    def similar_negatives(
        self,
        query_smiles: str,
        query_formula: str,
        top_k: int = 5,
        *,
        exclude_same_formula: bool = True,
        min_similarity: float = 0.0,
        chunk_size: int = 100_000,
    ) -> List[Tuple[str, float]]:
        """Find the top-K most similar molecules by Tanimoto on Morgan FP.

        Returns list of ``(canonical_smiles, tanimoto_similarity)`` sorted
        by similarity descending.

        Uses chunked computation to manage memory when the pool is large.
        """
        if self.fp_matrix is None or self.size == 0:
            return []

        # Compute query fingerprint
        mol = mol_from_smiles(query_smiles)
        if mol is None:
            return []
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            query_arr = np.zeros(2048, dtype=np.uint8)
            for bit in fp.GetOnBits():
                query_arr[bit] = 1
            query_packed = np.packbits(query_arr)  # (256,)
        except Exception:
            return []

        # Compute Tanimoto in chunks to save memory
        n = self.size
        all_tanimoto = np.zeros(n, dtype=np.float32)

        query_bits = np.unpackbits(query_packed).astype(np.float32)  # (2048,)
        query_popcount = query_bits.sum()

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = self.fp_matrix[start:end]  # (chunk, 256)
            pool_bits = np.unpackbits(chunk, axis=1).astype(np.float32)  # (chunk, 2048)

            intersection = pool_bits.dot(query_bits)  # (chunk,)
            pool_popcount = pool_bits.sum(axis=1)  # (chunk,)
            union = pool_popcount + query_popcount - intersection
            tanimoto = np.where(union > 0, intersection / union, 0.0)
            all_tanimoto[start:end] = tanimoto

        # Get top-K indices (excluding self and optionally same-formula)
        sorted_indices = np.argsort(all_tanimoto)[::-1]

        results: List[Tuple[str, float]] = []
        for idx in sorted_indices:
            idx = int(idx)
            smi = self.all_smiles[idx]
            sim = float(all_tanimoto[idx])

            if sim < min_similarity:
                break  # sorted descending, so no more above threshold
            if smi == query_smiles:
                continue
            if exclude_same_formula and self.all_formulas[idx] == query_formula:
                continue

            results.append((smi, sim))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def formula_distribution_stats(self, top_n: int = 20) -> Dict:
        """Return stats about formula distribution in the pool."""
        sizes = {f: len(idxs) for f, idxs in self.formula_to_indices.items()}
        sorted_formulas = sorted(sizes.items(), key=lambda x: -x[1])
        return {
            "total_molecules": self.size,
            "unique_formulas": len(sizes),
            "avg_molecules_per_formula": self.size / max(len(sizes), 1),
            "top_formulas": sorted_formulas[:top_n],
            "singletons": sum(1 for v in sizes.values() if v == 1),
        }
