"""Perturbation operators for generating wrong-SMILES samples.

Four operator families, each with multiple sub-variants:
  1. HeteroatomReplacementPerturbation  (4 sub-variants)
  2. FunctionalGroupReplacementPerturbation  (6 sub-variants)
  3. AlkylChainShiftPerturbation  (3 sub-variants)
  4. ProtectingGroupReplacementPerturbation  (3 sub-variants)

Every operator implements ``detect(mol) -> List[PerturbationSite]`` and
``apply(mol, site) -> Optional[str]``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from .base import PerturbationSite, canonical_smiles, mol_from_smiles


# ===================================================================
# Shared helpers
# ===================================================================

def _swap_atom_num(mol: Chem.Mol, atom_idx: int,
                   new_atomic_num: int) -> Optional[str]:
    """Swap an atom's element and return canonical SMILES, or None."""
    rw = Chem.RWMol(mol)
    try:
        atom = rw.GetAtomWithIdx(atom_idx)
    except Exception:
        return None
    atom.SetAtomicNum(new_atomic_num)
    atom.SetFormalCharge(0)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    try:
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _remove_atom(mol: Chem.Mol, atom_idx: int) -> Optional[str]:
    """Remove an atom by index and return canonical SMILES, or None."""
    rw = Chem.RWMol(mol)
    try:
        rw.RemoveAtom(atom_idx)
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _add_substituent(mol: Chem.Mol, anchor_idx: int,
                     substituent_smiles: str) -> Optional[str]:
    """Add a fragment to *anchor_idx* and return canonical SMILES."""
    frag = Chem.MolFromSmiles(substituent_smiles)
    if frag is None or frag.GetNumAtoms() == 0:
        return None
    try:
        combo = Chem.CombineMols(mol, frag)
        rw = Chem.RWMol(combo)
        frag_start = mol.GetNumAtoms()
        rw.AddBond(anchor_idx, frag_start, Chem.rdchem.BondType.SINGLE)
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _insert_ch2_in_bond(mol: Chem.Mol, begin_idx: int,
                        end_idx: int) -> Optional[str]:
    """Insert a CH2 group into the bond between *begin_idx* and *end_idx*."""
    rw = Chem.RWMol(mol)
    try:
        bond = rw.GetBondBetweenAtoms(begin_idx, end_idx)
        if bond is None:
            return None
        bond_type = bond.GetBondType()
        if bond_type != Chem.rdchem.BondType.SINGLE:
            return None
        rw.RemoveBond(begin_idx, end_idx)
        new_idx = rw.AddAtom(Chem.Atom(6))  # Carbon
        rw.AddBond(begin_idx, new_idx, Chem.rdchem.BondType.SINGLE)
        rw.AddBond(new_idx, end_idx, Chem.rdchem.BondType.SINGLE)
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


# ===================================================================
# 1. Heteroatom Replacement  (4 sub-variants)
# ===================================================================

_HETEROATOM_VARIANTS = [
    {
        "name": "O_to_S",
        # Non-ring, degree-2 oxygen, not carbonyl/nitro
        "smarts": "[#8;D2;!$([OD1]);!$([O]=[*]);!$([O][N+]);!R]",
        "new_atomic_num": 16,
        "description": "Replace oxygen with sulfur (ether/hydroxyl → thioether/thiol)",
        "repair_hint": "The input has sulfur (S) where the correct structure has "
                       "oxygen (O). Replace S with O to match the target formula.",
    },
    {
        "name": "S_to_O",
        "smarts": "[#16;D2;!$([S]=[*]);!R]",
        "new_atomic_num": 8,
        "description": "Replace sulfur with oxygen",
        "repair_hint": "The input has oxygen (O) where the correct structure has "
                       "sulfur (S). Replace O with S to match the target formula.",
    },
    {
        "name": "ring_O_to_NH",
        # Aromatic ring oxygen (furan-like)
        "smarts": "[o;R1]",
        "new_atomic_num": 7,
        "description": "Replace ring oxygen with NH (furan → pyrrole)",
        "repair_hint": "The input has a pyrrole-type NH where the correct structure "
                       "has a furan-type O in the aromatic ring. Replace ring N with O.",
    },
    {
        "name": "ring_NH_to_O",
        # Aromatic ring NH (pyrrole-like)
        "smarts": "[nH1;R1]",
        "new_atomic_num": 8,
        "description": "Replace ring NH with O (pyrrole → furan)",
        "repair_hint": "The input has a furan-type O where the correct structure has "
                       "a pyrrole-type NH in the aromatic ring. Replace ring O with N.",
    },
]


class HeteroatomReplacementPerturbation:
    """Swap one heteroatom for another, changing the molecular formula."""

    name = "heteroatom_replacement"
    formula_changing = True

    def detect(self, mol: Chem.Mol) -> List[PerturbationSite]:
        sites: List[PerturbationSite] = []
        for var in _HETEROATOM_VARIANTS:
            pat = Chem.MolFromSmarts(var["smarts"])
            if pat is None:
                continue
            matches = mol.GetSubstructMatches(pat)
            for match in matches:
                sites.append(PerturbationSite(
                    atom_indices=match,
                    sub_variant=var["name"],
                    metadata={"new_atomic_num": var["new_atomic_num"],
                              "repair_hint": var["repair_hint"]},
                ))
        return sites

    def apply(self, mol: Chem.Mol, site: PerturbationSite) -> Optional[str]:
        idx = site.atom_indices[0]
        new_num = site.metadata["new_atomic_num"]
        return _swap_atom_num(mol, idx, new_num)

    def reasoning(self, site: PerturbationSite,
                  original_smiles: str, perturbed_smiles: str) -> str:
        return site.metadata["repair_hint"]


# ===================================================================
# 2. Functional Group Replacement  (6 sub-variants)
# ===================================================================

class FunctionalGroupReplacementPerturbation:
    """Replace one functional group with a similar one."""

    name = "functional_group_replacement"
    formula_changing = True

    # Sub-variant definitions
    _VARIANTS = [
        "ester_to_amide",
        "amide_to_ester",
        "aldehyde_to_ketone",
        "ketone_to_aldehyde",
        "alcohol_to_thiol",
        "thiol_to_alcohol",
    ]

    def detect(self, mol: Chem.Mol) -> List[PerturbationSite]:
        sites: List[PerturbationSite] = []
        self._detect_ester(mol, sites)
        self._detect_amide(mol, sites)
        self._detect_aldehyde(mol, sites)
        self._detect_ketone(mol, sites)
        self._detect_alcohol(mol, sites)
        self._detect_thiol(mol, sites)
        return sites

    # --- ester → amide ---
    def _detect_ester(self, mol: Chem.Mol,
                      sites: List[PerturbationSite]) -> None:
        # [C](=O)-O-[C,c] : the bridging O in an ester
        pat = Chem.MolFromSmarts("[C;!R](=O)-[O;D2;!R]-[C,c]")
        if pat is None:
            return
        for match in mol.GetSubstructMatches(pat):
            # match = (carbonyl_C, O_double, bridging_O, other_C)
            bridging_o_idx = match[2]
            sites.append(PerturbationSite(
                atom_indices=(bridging_o_idx,),
                sub_variant="ester_to_amide",
                metadata={
                    "repair_hint": "The input has an amide bond (C(=O)-NH) where "
                                   "the correct structure has an ester (C(=O)-O). "
                                   "Replace the nitrogen with oxygen.",
                },
            ))

    # --- amide → ester ---
    def _detect_amide(self, mol: Chem.Mol,
                      sites: List[PerturbationSite]) -> None:
        pat = Chem.MolFromSmarts("[C;!R](=O)-[NH;D2;!R]-[C,c]")
        if pat is None:
            return
        for match in mol.GetSubstructMatches(pat):
            nh_idx = match[2]
            sites.append(PerturbationSite(
                atom_indices=(nh_idx,),
                sub_variant="amide_to_ester",
                metadata={
                    "repair_hint": "The input has an ester (C(=O)-O) where the "
                                   "correct structure has an amide (C(=O)-NH). "
                                   "Replace the oxygen with nitrogen.",
                },
            ))

    # --- aldehyde → ketone ---
    def _detect_aldehyde(self, mol: Chem.Mol,
                         sites: List[PerturbationSite]) -> None:
        pat = Chem.MolFromSmarts("[CH1;!R](=O)")
        if pat is None:
            return
        for match in mol.GetSubstructMatches(pat):
            c_idx = match[0]
            sites.append(PerturbationSite(
                atom_indices=(c_idx,),
                sub_variant="aldehyde_to_ketone",
                metadata={
                    "repair_hint": "The input has a methyl ketone (C(=O)CH3) where "
                                   "the correct structure has an aldehyde (C(=O)H). "
                                   "Remove the extra methyl group from the carbonyl.",
                },
            ))

    # --- ketone → aldehyde ---
    def _detect_ketone(self, mol: Chem.Mol,
                       sites: List[PerturbationSite]) -> None:
        # Methyl ketone: C(=O) bonded to a CH3 and another carbon
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C" or atom.IsInRing():
                continue
            # Check if it's a carbonyl carbon
            has_double_o = False
            ch3_idx = None
            other_c = False
            for bond in atom.GetBonds():
                nbr = bond.GetOtherAtom(atom)
                if (nbr.GetSymbol() == "O"
                        and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE):
                    has_double_o = True
                elif (nbr.GetSymbol() == "C" and nbr.GetDegree() == 1
                      and nbr.GetTotalNumHs() == 3):
                    ch3_idx = nbr.GetIdx()
                elif nbr.GetSymbol() in ("C", "N", "O", "S"):
                    other_c = True
            if has_double_o and ch3_idx is not None and other_c:
                sites.append(PerturbationSite(
                    atom_indices=(atom.GetIdx(), ch3_idx),
                    sub_variant="ketone_to_aldehyde",
                    metadata={
                        "ch3_idx": ch3_idx,
                        "repair_hint": "The input has an aldehyde (C(=O)H) where "
                                       "the correct structure has a methyl ketone "
                                       "(C(=O)CH3). Add a methyl group to the "
                                       "carbonyl carbon.",
                    },
                ))

    # --- alcohol → thiol ---
    def _detect_alcohol(self, mol: Chem.Mol,
                        sites: List[PerturbationSite]) -> None:
        pat = Chem.MolFromSmarts("[OH1;!R]")
        if pat is None:
            return
        for match in mol.GetSubstructMatches(pat):
            sites.append(PerturbationSite(
                atom_indices=match,
                sub_variant="alcohol_to_thiol",
                metadata={
                    "repair_hint": "The input has a thiol group (-SH) where the "
                                   "correct structure has a hydroxyl (-OH). "
                                   "Replace sulfur with oxygen.",
                },
            ))

    # --- thiol → alcohol ---
    def _detect_thiol(self, mol: Chem.Mol,
                      sites: List[PerturbationSite]) -> None:
        pat = Chem.MolFromSmarts("[SH1;!R]")
        if pat is None:
            return
        for match in mol.GetSubstructMatches(pat):
            sites.append(PerturbationSite(
                atom_indices=match,
                sub_variant="thiol_to_alcohol",
                metadata={
                    "repair_hint": "The input has a hydroxyl (-OH) where the "
                                   "correct structure has a thiol (-SH). "
                                   "Replace oxygen with sulfur.",
                },
            ))

    def apply(self, mol: Chem.Mol, site: PerturbationSite) -> Optional[str]:
        v = site.sub_variant
        if v == "ester_to_amide":
            # Swap bridging O → N
            return _swap_atom_num(mol, site.atom_indices[0], 7)
        elif v == "amide_to_ester":
            # Swap bridging N → O
            return _swap_atom_num(mol, site.atom_indices[0], 8)
        elif v == "aldehyde_to_ketone":
            # Add methyl to the aldehyde carbon
            return _add_substituent(mol, site.atom_indices[0], "C")
        elif v == "ketone_to_aldehyde":
            # Remove the CH3 from the methyl ketone
            ch3_idx = site.metadata["ch3_idx"]
            return _remove_atom(mol, ch3_idx)
        elif v == "alcohol_to_thiol":
            return _swap_atom_num(mol, site.atom_indices[0], 16)
        elif v == "thiol_to_alcohol":
            return _swap_atom_num(mol, site.atom_indices[0], 8)
        return None

    def reasoning(self, site: PerturbationSite,
                  original_smiles: str, perturbed_smiles: str) -> str:
        return site.metadata["repair_hint"]


# ===================================================================
# 3. Alkyl Chain Shift  (3 sub-variants)
# ===================================================================

class AlkylChainShiftPerturbation:
    """Add/remove/insert CH2/CH3 groups on alkyl chains."""

    name = "alkyl_chain_shift"
    formula_changing = True

    def detect(self, mol: Chem.Mol) -> List[PerturbationSite]:
        sites: List[PerturbationSite] = []
        self._detect_add_methyl(mol, sites)
        self._detect_remove_methyl(mol, sites)
        self._detect_insert_ch2(mol, sites)
        return sites

    def _detect_add_methyl(self, mol: Chem.Mol,
                           sites: List[PerturbationSite]) -> None:
        """Find acyclic CH2 groups that can accept a methyl branch."""
        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == "C"
                    and not atom.IsInRing()
                    and not atom.GetIsAromatic()
                    and atom.GetDegree() == 2
                    and atom.GetTotalNumHs() >= 1):
                # Only pick CH2 or CH with at least one spare H
                sites.append(PerturbationSite(
                    atom_indices=(atom.GetIdx(),),
                    sub_variant="add_methyl_branch",
                    metadata={
                        "repair_hint": "The input has an extra methyl branch. "
                                       "Remove the methyl group to match the "
                                       "target formula (C:-1, H:-2).",
                    },
                ))

    def _detect_remove_methyl(self, mol: Chem.Mol,
                              sites: List[PerturbationSite]) -> None:
        """Find terminal CH3 groups that can be removed."""
        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == "C"
                    and not atom.IsInRing()
                    and atom.GetDegree() == 1
                    and atom.GetTotalNumHs() == 3):
                # It's a terminal CH3; check its parent is not aromatic C
                nbr = atom.GetNeighbors()[0]
                if nbr.GetSymbol() == "C" and not nbr.GetIsAromatic():
                    # Don't remove the only non-aromatic substituent
                    if nbr.GetDegree() >= 2:
                        sites.append(PerturbationSite(
                            atom_indices=(atom.GetIdx(),),
                            sub_variant="remove_terminal_methyl",
                            metadata={
                                "repair_hint": "The input is missing a methyl "
                                               "group. Add a CH3 to match the "
                                               "target formula (C:+1, H:+2).",
                            },
                        ))

    def _detect_insert_ch2(self, mol: Chem.Mol,
                           sites: List[PerturbationSite]) -> None:
        """Find acyclic single bonds where a CH2 can be inserted."""
        for bond in mol.GetBonds():
            if bond.IsInRing():
                continue
            if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                continue
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            # Both must be heavy atoms (not H), at least one non-aromatic C
            if a1.GetAtomicNum() < 6 or a2.GetAtomicNum() < 6:
                continue
            if a1.GetIsAromatic() and a2.GetIsAromatic():
                continue
            sites.append(PerturbationSite(
                atom_indices=(a1.GetIdx(), a2.GetIdx()),
                sub_variant="insert_ch2",
                metadata={
                    "repair_hint": "The input has an extra CH2 group inserted "
                                   "in the carbon chain. Remove one CH2 to "
                                   "match the target formula (C:-1, H:-2).",
                },
            ))

    def apply(self, mol: Chem.Mol, site: PerturbationSite) -> Optional[str]:
        v = site.sub_variant
        if v == "add_methyl_branch":
            return _add_substituent(mol, site.atom_indices[0], "C")
        elif v == "remove_terminal_methyl":
            return _remove_atom(mol, site.atom_indices[0])
        elif v == "insert_ch2":
            return _insert_ch2_in_bond(mol, site.atom_indices[0],
                                       site.atom_indices[1])
        return None

    def reasoning(self, site: PerturbationSite,
                  original_smiles: str, perturbed_smiles: str) -> str:
        return site.metadata["repair_hint"]


# ===================================================================
# 4. Protecting Group Replacement  (3 sub-variants)
# ===================================================================

class ProtectingGroupReplacementPerturbation:
    """Swap common protecting groups (TBS↔TMS, Ac removal)."""

    name = "protecting_group_replacement"
    formula_changing = True

    def detect(self, mol: Chem.Mol) -> List[PerturbationSite]:
        sites: List[PerturbationSite] = []
        self._detect_tbs_to_tms(mol, sites)
        self._detect_tms_to_tbs(mol, sites)
        self._detect_acetyl_removal(mol, sites)
        return sites

    def _detect_tbs_to_tms(self, mol: Chem.Mol,
                           sites: List[PerturbationSite]) -> None:
        """Find TBS groups: Si bonded to a tert-butyl (C(C)(C)C)."""
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "Si":
                continue
            # Look for a C neighbor that has 3 CH3 children (t-Bu central C)
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() != "C":
                    continue
                ch3_children = []
                for nbr2 in nbr.GetNeighbors():
                    if nbr2.GetIdx() == atom.GetIdx():
                        continue
                    if (nbr2.GetSymbol() == "C"
                            and nbr2.GetDegree() == 1
                            and nbr2.GetTotalNumHs() == 3):
                        ch3_children.append(nbr2.GetIdx())
                if len(ch3_children) == 3:
                    sites.append(PerturbationSite(
                        atom_indices=(atom.GetIdx(), nbr.GetIdx(),
                                      *ch3_children),
                        sub_variant="TBS_to_TMS",
                        metadata={
                            "si_idx": atom.GetIdx(),
                            "tbu_c_idx": nbr.GetIdx(),
                            "ch3_indices": ch3_children,
                            "repair_hint": "The input has a TMS (trimethylsilyl) "
                                           "protecting group where the correct "
                                           "structure has TBS (tert-butyldimethyl"
                                           "silyl). Add a tert-butyl group to the "
                                           "silicon atom.",
                        },
                    ))

    def _detect_tms_to_tbs(self, mol: Chem.Mol,
                           sites: List[PerturbationSite]) -> None:
        """Find TMS groups: Si with exactly 3 methyls and one O/N."""
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "Si":
                continue
            ch3_indices = []
            has_o_or_n = False
            for nbr in atom.GetNeighbors():
                if (nbr.GetSymbol() == "C"
                        and nbr.GetDegree() == 1
                        and nbr.GetTotalNumHs() == 3):
                    ch3_indices.append(nbr.GetIdx())
                elif nbr.GetSymbol() in ("O", "N"):
                    has_o_or_n = True
            if len(ch3_indices) == 3 and has_o_or_n:
                # This is a TMS group; we can upgrade one CH3 to C(C)(C)C
                sites.append(PerturbationSite(
                    atom_indices=(atom.GetIdx(), ch3_indices[0]),
                    sub_variant="TMS_to_TBS",
                    metadata={
                        "si_idx": atom.GetIdx(),
                        "ch3_to_replace": ch3_indices[0],
                        "repair_hint": "The input has a TBS (tert-butyldimethyl"
                                       "silyl) group where the correct structure "
                                       "has TMS (trimethylsilyl). Remove the "
                                       "tert-butyl group from the silicon atom.",
                    },
                ))

    def _detect_acetyl_removal(self, mol: Chem.Mol,
                               sites: List[PerturbationSite]) -> None:
        """Find acetyl (Ac) protecting groups: CH3-C(=O)-O-[*]."""
        pat = Chem.MolFromSmarts("[CH3;!R]-[C;!R](=O)-[O;D2;!R]")
        if pat is None:
            return
        for match in mol.GetSubstructMatches(pat):
            ch3_idx, carbonyl_c_idx, _o_double_idx, bridging_o_idx = (
                match[0], match[1], match[1], match[2]
            )
            # We will remove the acetyl (CH3CO) leaving a free OH
            sites.append(PerturbationSite(
                atom_indices=(ch3_idx, carbonyl_c_idx),
                sub_variant="remove_acetyl",
                metadata={
                    "ch3_idx": ch3_idx,
                    "carbonyl_c_idx": carbonyl_c_idx,
                    "repair_hint": "The input is missing an acetyl (Ac) protecting "
                                   "group. The -OH should be protected as "
                                   "-OC(=O)CH3. Add the acetyl group back.",
                },
            ))

    def apply(self, mol: Chem.Mol, site: PerturbationSite) -> Optional[str]:
        v = site.sub_variant
        if v == "TBS_to_TMS":
            return self._apply_tbs_to_tms(mol, site)
        elif v == "TMS_to_TBS":
            return self._apply_tms_to_tbs(mol, site)
        elif v == "remove_acetyl":
            return self._apply_remove_acetyl(mol, site)
        return None

    def _apply_tbs_to_tms(self, mol: Chem.Mol,
                          site: PerturbationSite) -> Optional[str]:
        """Remove the t-Bu from TBS, leaving TMS."""
        ch3_indices = site.metadata["ch3_indices"]
        tbu_c_idx = site.metadata["tbu_c_idx"]
        atoms_to_remove = sorted([tbu_c_idx] + ch3_indices, reverse=True)
        rw = Chem.RWMol(mol)
        try:
            for idx in atoms_to_remove:
                rw.RemoveAtom(idx)
            new_mol = rw.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol, canonical=True,
                                    isomericSmiles=True)
        except Exception:
            return None

    def _apply_tms_to_tbs(self, mol: Chem.Mol,
                          site: PerturbationSite) -> Optional[str]:
        """Upgrade one CH3 on TMS to C(C)(C)C, making TBS."""
        ch3_idx = site.metadata["ch3_to_replace"]
        si_idx = site.metadata["si_idx"]
        rw = Chem.RWMol(mol)
        try:
            # Remove the CH3 atom, then add C(C)(C)C fragment
            rw.RemoveAtom(ch3_idx)
            # After removal, indices shift. Find the new Si index.
            new_si_idx = si_idx if ch3_idx > si_idx else si_idx - 1
            # Add the t-Bu fragment: C(C)(C)C as a single carbon with 3 methyls
            new_mol = rw.GetMol()
            tbu_frag = Chem.MolFromSmiles("CC(C)C")
            combo = Chem.CombineMols(new_mol, tbu_frag)
            rw2 = Chem.RWMol(combo)
            tbu_start = new_mol.GetNumAtoms()
            rw2.AddBond(new_si_idx, tbu_start, Chem.rdchem.BondType.SINGLE)
            final = rw2.GetMol()
            Chem.SanitizeMol(final)
            return Chem.MolToSmiles(final, canonical=True, isomericSmiles=True)
        except Exception:
            return None

    def _apply_remove_acetyl(self, mol: Chem.Mol,
                             site: PerturbationSite) -> Optional[str]:
        """Remove CH3CO from an acetyl ester, leaving free OH."""
        ch3_idx = site.metadata["ch3_idx"]
        carbonyl_c_idx = site.metadata["carbonyl_c_idx"]
        # Remove CH3 first (higher index first to avoid shift issues)
        atoms_to_remove = sorted([ch3_idx, carbonyl_c_idx], reverse=True)

        rw = Chem.RWMol(mol)
        try:
            # We also need to remove the carbonyl O (=O double bond partner)
            carbonyl_c = rw.GetAtomWithIdx(carbonyl_c_idx)
            double_o_idx = None
            for bond in carbonyl_c.GetBonds():
                nbr = bond.GetOtherAtom(carbonyl_c)
                if (nbr.GetSymbol() == "O"
                        and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE):
                    double_o_idx = nbr.GetIdx()
                    break

            to_remove = sorted(
                [ch3_idx, carbonyl_c_idx]
                + ([double_o_idx] if double_o_idx is not None else []),
                reverse=True,
            )
            for idx in to_remove:
                rw.RemoveAtom(idx)
            new_mol = rw.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol, canonical=True,
                                    isomericSmiles=True)
        except Exception:
            return None

    def reasoning(self, site: PerturbationSite,
                  original_smiles: str, perturbed_smiles: str) -> str:
        return site.metadata["repair_hint"]


# ===================================================================
# Registry
# ===================================================================

def build_all_perturbation_operators() -> List:
    """Return one instance of each perturbation operator family."""
    return [
        HeteroatomReplacementPerturbation(),
        FunctionalGroupReplacementPerturbation(),
        AlkylChainShiftPerturbation(),
        ProtectingGroupReplacementPerturbation(),
    ]
