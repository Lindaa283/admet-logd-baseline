"""Structure handling: standardisation, identity keys, and descriptors.

Two jobs that matter for the honesty of the result:

1. **Identity.** Two rows that describe the same compound must be recognised as
   such, otherwise the same molecule can sit in both train and test and the
   reported error is optimistic.  We key on the InChIKey, and separately on its
   first block (the connectivity layer), which ignores stereochemistry and
   protonation state.  logD is a property of the compound in solution; a pair
   of enantiomers has the same logD, so connectivity-level identity is the
   right granularity for leakage checks.

2. **Descriptors.** A small, interpretable set, chosen for this endpoint rather
   than scraped wholesale: logD7.4 is lipophilicity *after* accounting for
   ionisation at pH 7.4, so alongside the usual size/polarity terms we count
   acidic and basic groups explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

# RDKit is chatty about every unparseable SMILES; we count failures ourselves.
RDLogger.DisableLog("rdApp.*")


# --- ionisable-group patterns -------------------------------------------------
# Deliberately simple and readable.  These are *approximations*: they capture
# the common acidic and basic motifs in a drug-like set, not every edge case.
# Their purpose is to give the linear baselines a chemically meaningful
# ionisation term, not to replace a real pKa predictor.
_ACID_SMARTS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "tetrazole": "c1nnn[nH]1",
    "sulfonamide_NH": "[SX4](=O)(=O)[NX3H1,NX3H2]",
    "acyl_sulfonamide": "[CX3](=O)[NX3H1][SX4](=O)=O",
    "phosphonic_acid": "[PX4](=O)([OX2H1])",
}
_BASE_SMARTS = {
    # Aliphatic amine not attached to a carbonyl/sulfonyl (i.e. not an amide
    # or sulfonamide, which are not basic at pH 7.4).
    "aliphatic_amine": (
        "[NX3;H2,H1,H0;!$(N[CX3]=[OX1,SX1,NX2]);!$(N[SX4](=O)=O);"
        "!$(N[CX3]=[CX3]);!$(Nc);!$(N#*);!$(N=*);!$(N[OX1,OX2])]"
    ),
    "amidine_guanidine": "[NX3][CX3]=[NX2]",
    "basic_aromatic_N": "[nX2;$(n1ccccc1),$(n1ccnc1)]",
}
_PHENOL_SMARTS = "[OX2H1]c"

_COMPILED_ACIDS = {k: Chem.MolFromSmarts(v) for k, v in _ACID_SMARTS.items()}
_COMPILED_BASES = {k: Chem.MolFromSmarts(v) for k, v in _BASE_SMARTS.items()}
_COMPILED_PHENOL = Chem.MolFromSmarts(_PHENOL_SMARTS)

# Order is fixed and explicit so that the feature matrix is stable across runs
# and the coefficients in the linear baselines stay interpretable.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",           # Wildman-Crippen cLogP
    "MolMR",             # molar refractivity
    "TPSA",
    "LabuteASA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "NOCount",
    "NHOHCount",
    "HallKierAlpha",
    "BertzCT",
    "n_halogens",
    "n_acidic_groups",
    "n_basic_groups",
    "n_phenol",
    "formal_charge",
    "n_fragments",
)


@dataclass(frozen=True)
class Standardised:
    """A row's structure after standardisation, plus its identity keys."""

    input_smiles: str
    canonical_smiles: str       # largest fragment, canonical, stereo kept
    inchikey: str               # full key: distinguishes stereoisomers
    connectivity_key: str       # first block: ignores stereo/protonation
    n_fragments: int            # >1 means the input was a salt or mixture
    heavy_atoms: int


def standardise(smiles: str) -> Standardised | None:
    """Parse, keep the largest covalent fragment, and compute identity keys.

    Returns ``None`` if RDKit cannot parse or sanitise the input, so callers
    can count and report failures rather than crash on one bad row.

    Keeping the largest fragment removes counter-ions (``[Na+]``, ``Cl-``,
    maleate, …).  The measured logD belongs to the parent compound, and the
    counter-ion would otherwise change both the fingerprint and the
    descriptors for what is chemically the same molecule.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return None
    n_fragments = len(fragments)
    parent = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
    if parent.GetNumHeavyAtoms() == 0:
        return None

    try:
        canonical = Chem.MolToSmiles(parent)
        inchikey = Chem.MolToInchiKey(parent)
    except Exception:
        return None
    if not inchikey:
        return None

    return Standardised(
        input_smiles=smiles,
        canonical_smiles=canonical,
        inchikey=inchikey,
        connectivity_key=inchikey.split("-")[0],
        n_fragments=n_fragments,
        heavy_atoms=parent.GetNumHeavyAtoms(),
    )


def mol_from_canonical(canonical_smiles: str) -> Chem.Mol | None:
    """Re-parse a canonical SMILES produced by :func:`standardise`."""
    return Chem.MolFromSmiles(canonical_smiles)


def _count_matches(mol: Chem.Mol, patterns: dict[str, Chem.Mol]) -> int:
    total = 0
    for pattern in patterns.values():
        if pattern is None:
            continue
        total += len(mol.GetSubstructMatches(pattern))
    return total


def descriptors(mol: Chem.Mol, n_fragments: int = 1) -> dict[str, float]:
    """The fixed descriptor set for one molecule, as a name -> value mapping."""
    halogens = sum(
        1 for atom in mol.GetAtoms() if atom.GetSymbol() in ("F", "Cl", "Br", "I")
    )
    values = {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "MolMR": Crippen.MolMR(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "NOCount": Lipinski.NOCount(mol),
        "NHOHCount": Lipinski.NHOHCount(mol),
        "HallKierAlpha": rdMolDescriptors.CalcHallKierAlpha(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "n_halogens": halogens,
        "n_acidic_groups": _count_matches(mol, _COMPILED_ACIDS),
        "n_basic_groups": _count_matches(mol, _COMPILED_BASES),
        "n_phenol": len(mol.GetSubstructMatches(_COMPILED_PHENOL)),
        "formal_charge": Chem.GetFormalCharge(mol),
        "n_fragments": n_fragments,
    }
    # Guard against a descriptor silently disappearing after an RDKit upgrade.
    missing = set(DESCRIPTOR_NAMES) - set(values)
    if missing:
        raise RuntimeError(f"descriptor(s) not computed: {sorted(missing)}")
    return values


def clogp(mol: Chem.Mol) -> float:
    """Wildman-Crippen cLogP, used on its own as the simplest chemistry baseline."""
    return Crippen.MolLogP(mol)
