"""Base classes and data structures for the perturbation generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PerturbationSite:
    """A detected site on a molecule where a perturbation can be applied.

    Attributes:
        atom_indices: Atom indices involved in this site (e.g. a single atom
            for heteroatom swap, or a bond pair for CH2 insertion).
        sub_variant: Name of the specific sub-variant (e.g. "O_to_S").
        metadata: Arbitrary extra info (SMARTS match, bond index, etc.).
    """
    atom_indices: tuple
    sub_variant: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbedSample:
    """A single perturbed sample with full provenance.

    Pairs a *wrong* SMILES with the *correct* NMR data, formula, and
    ground-truth SMILES from the original LMDB record.
    """
    # --- perturbed (wrong) side ---
    input_smiles: str           # perturbed SMILES (the "wrong" structure)
    input_formula: str          # molecular formula of the perturbed SMILES

    # --- correct (ground truth) side ---
    target_formula: str         # correct molecular formula
    gt_smiles: str              # correct SMILES (original)

    # --- formula delta ---
    formula_delta: str          # human-readable delta, e.g. "O:-1,S:+1"
    edit_lb: int                # edit lower-bound (how many operator steps)

    # --- NMR spectral data (correct, from LMDB) ---
    c_nmr_peaks: Optional[List[Dict]] = None
    h_nmr_peaks: Optional[List[Dict]] = None
    nmr_frequency: Optional[str] = None
    nmr_solvent: Optional[str] = None

    # --- provenance ---
    perturbation_type: str = ""     # e.g. "heteroatom_O_to_S"
    case_id: str = ""               # unique identifier
    source_tool: str = "perturbation_generator"

    # --- similarity (filled by negative_engine for similar_structure) ---
    tanimoto_similarity: Optional[float] = None

    # --- spectral similarity (filled by hard_negative_cli) ---
    spectral_similarity: Optional[float] = None

    # --- verification (filled by verify.py) ---
    repair_verified: Optional[bool] = None
    repair_pred_smiles: Optional[str] = None
    structurally_correct: Optional[bool] = None
    fallback_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Protocol for perturbation operators
# ---------------------------------------------------------------------------

class BasePerturbation(Protocol):
    """Protocol that every perturbation operator must satisfy."""

    name: str
    """Short name, e.g. 'heteroatom_replacement'."""

    formula_changing: bool
    """True if this perturbation always changes the molecular formula."""

    def detect(self, mol: Chem.Mol) -> List[PerturbationSite]:
        """Find all applicable perturbation sites on *mol*.

        Returns a list of :class:`PerturbationSite` objects, each describing
        one possible edit.  The engine will pick a subset (up to
        ``max_per_operator``) and call :meth:`apply` for each.
        """
        ...

    def apply(self, mol: Chem.Mol, site: PerturbationSite) -> Optional[str]:
        """Apply the perturbation at *site* and return canonical SMILES.

        Returns ``None`` if the perturbation produces an invalid molecule
        (sanitization failure, empty product, etc.).
        """
        ...

    def reasoning(self, site: PerturbationSite,
                  original_smiles: str, perturbed_smiles: str) -> str:
        """Auto-generate a repair-reasoning string for SFT training output.

        Describes what correction steps are needed to go from the perturbed
        structure back to the original.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    """Parse SMILES into a sanitized RDKit Mol, or None on failure."""
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonical_smiles(smi_or_mol) -> Optional[str]:
    """Return canonical SMILES string, or None on failure."""
    try:
        if isinstance(smi_or_mol, str):
            mol = Chem.MolFromSmiles(smi_or_mol)
        else:
            mol = smi_or_mol
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def molecule_formula(smi_or_mol) -> Optional[str]:
    """Compute Hill-style molecular formula, or None on failure."""
    try:
        if isinstance(smi_or_mol, str):
            mol = Chem.MolFromSmiles(smi_or_mol)
        else:
            mol = smi_or_mol
        if mol is None:
            return None
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None
