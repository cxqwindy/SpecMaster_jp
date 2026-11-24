"""LMDB reader for NMR spectral datasets.

Reads records from the LMDB files under
``/fs_mol/fjxu/proj_data/Spectra/NMR/si_0820/lmdb_dataset/``.

Each record is a pickle'd dict with keys:
    smiles, smiles_canonical, nmr_frequency, nmr_solvent,
    molecular_formula, c_nmr_peaks, h_nmr_peaks, atoms, coordinates
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import lmdb

from .base import mol_from_smiles


def read_lmdb(
    path: str,
    *,
    require_nmr: bool = False,
    max_records: Optional[int] = None,
) -> List[Dict]:
    """Read an LMDB file and return a list of valid records.

    Parameters
    ----------
    path : str
        Path to the ``.lmdb`` file (single file, not a directory).
    require_nmr : bool
        If True, skip records where both ``c_nmr_peaks`` and ``h_nmr_peaks``
        are missing or empty.
    max_records : int or None
        Cap the number of records returned (for quick testing).

    Returns
    -------
    list of dict
        Each dict contains at minimum:
        ``smiles_canonical``, ``molecular_formula``,
        ``c_nmr_peaks``, ``h_nmr_peaks``,
        ``nmr_frequency``, ``nmr_solvent``.
    """
    path = str(Path(path).resolve())
    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    records: List[Dict] = []
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        for key_buf, val_buf in cursor:
            raw = pickle.loads(bytes(val_buf))

            # Must have SMILES and formula
            smi = raw.get("smiles_canonical") or raw.get("smiles")
            formula = raw.get("molecular_formula")
            if not smi or not formula:
                continue

            # Validate SMILES
            if mol_from_smiles(smi) is None:
                continue

            c_peaks = raw.get("c_nmr_peaks")
            h_peaks = raw.get("h_nmr_peaks")

            if require_nmr:
                has_c = c_peaks and len(c_peaks) > 0
                has_h = h_peaks and len(h_peaks) > 0
                if not has_c and not has_h:
                    continue

            record = {
                "smiles_canonical": str(smi),
                "molecular_formula": str(formula),
                "c_nmr_peaks": c_peaks if c_peaks else [],
                "h_nmr_peaks": h_peaks if h_peaks else [],
                "nmr_frequency": raw.get("nmr_frequency", ""),
                "nmr_solvent": raw.get("nmr_solvent", ""),
            }
            records.append(record)

            if max_records is not None and len(records) >= max_records:
                break

    env.close()
    return records


def iter_lmdb(
    path: str,
    *,
    require_nmr: bool = False,
) -> Iterator[Dict]:
    """Iterate over an LMDB file yielding one record at a time.

    Same filtering as :func:`read_lmdb` but doesn't load everything
    into memory at once.  Useful for very large datasets.
    """
    path = str(Path(path).resolve())
    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=True, meminit=False)
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        for key_buf, val_buf in cursor:
            raw = pickle.loads(bytes(val_buf))

            smi = raw.get("smiles_canonical") or raw.get("smiles")
            formula = raw.get("molecular_formula")
            if not smi or not formula:
                continue
            if mol_from_smiles(smi) is None:
                continue

            c_peaks = raw.get("c_nmr_peaks")
            h_peaks = raw.get("h_nmr_peaks")

            if require_nmr:
                has_c = c_peaks and len(c_peaks) > 0
                has_h = h_peaks and len(h_peaks) > 0
                if not has_c and not has_h:
                    continue

            yield {
                "smiles_canonical": str(smi),
                "molecular_formula": str(formula),
                "c_nmr_peaks": c_peaks if c_peaks else [],
                "h_nmr_peaks": h_peaks if h_peaks else [],
                "nmr_frequency": raw.get("nmr_frequency", ""),
                "nmr_solvent": raw.get("nmr_solvent", ""),
            }
    env.close()
