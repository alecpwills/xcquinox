"""Dick & Fernandez-Serra 2021 SI §II training pool.

Citation: Dick, S.; Fernandez-Serra, M. *Phys. Rev. B* **104**, L161109 (2021).
SI §II "Training Data" page 1 (verbatim member lists transcribed below).

Pool composition (28 distinct training points):
- 21 atomization-energy (AE) entries from G2/97 (10 linear closed-shell +
  3 linear open-shell + 8 non-linear)
- 3 BH76 reaction barriers
- 2 IP13 ionization potentials
- 2 atomic-density references (H, Li)

All published reference values (AE in kcal/mol for the 21 AE molecules;
e_rxn_ref for BH76; ip_ref for IP13) come from authoritative benchmarks
and are attached to the build_dick_pool() output via Atoms.info /
reaction-spec dict entries — never fabricated.  See DICK_AE_DATA for AE
sources (W4-11 anchors via step-6 for H2O+C2H2; Haunschild & Klopper
J. Chem. Phys. 136, 164102 (2012) for the other 19).
"""
from __future__ import annotations

from pathlib import Path
from ase import Atoms
from ase.io import read

# Dick 2021 SI §II AE molecule list, in ASE Hill formula, with
# authoritative non-relativistic, zero-point-exclusive (TAE_e) reference
# atomization energies in kcal/mol.
#
# Sources:
#
# - "step6":
#       H2O = 232.974 and C2H2 = 405.525 kcal/mol are the W4-11 reference
#       values anchored by the step-6 notebook
#       (`notebooks/_build_step6_notebook.py`,
#        constants H2O_AE_REF_KCALMOL / C2H2_AE_REF_KCALMOL).
#       These are the canonical anchors for cross-notebook
#       reproducibility (step 5 / step 6 / step 7 must all use the same
#       AE reference for a given molecule).  See step-6 design spec
#       §17.1 and §17.3 (Δ from W4-17 < 0.5 kcal/mol — well within the
#       1 kJ/mol confidence interval of the W4 family).
#
# - "Haunschild2012":
#       Haunschild & Klopper, "New accurate reference energies for the
#       G2/97 test set", J. Chem. Phys. **136**, 164102 (2012),
#       DOI 10.1063/1.4704796.  Table I, column "E_ref,non-rel"
#       (frozen-core, all-electron-corrected, non-relativistic
#       atomization energies obtained from CCSD(T)(F12)/cc-pVQZ-F12
#       with higher-excitation and core/core-valence corrections;
#       reported in kJ/mol — converted here via 1 kcal = 4.184 kJ).
#       This dataset has a 0.1 kJ/mol per valence electron error
#       budget (see Haunschild 2012 §III) and a published mean
#       deviation of −0.75 kJ/mol vs ATcT.
#       Local copy of the paper:
#       scripts/script_data/haunschild_g2/haunschild2012.pdf
#       Local CSV (kJ/mol; column E):
#       scripts/script_data/haunschild_g2/g2_97.csv
#
# Cross-checked against W4-17 (Karton, Sylvetsky, Martin J. Comp. Chem.
# 38, 2063 (2017), DOI 10.1002/jcc.24854) Table S2, TAE_e column,
# for first/second-row species; agreement is sub-0.5 kcal/mol on every
# Dick-pool molecule that appears in W4-17.  Li/Na species (LiF, LiH,
# Na2) are not in W4-17 and are sourced exclusively from Haunschild2012.
#
# Notes:
#   - CH2 in g2_97.traj contains both singlet (idx 105) and triplet
#     (idx 106) entries with identical Hill formula "CH2"; build_dick_pool
#     selects the triplet (last in iteration), matching W4-17 ch2-trip
#     (TAE_e = 190.53) and Haunschild "Triplet carbene" (797.23 kJ/mol →
#     190.541 kcal/mol).
#   - Open-shell molecules (NO, CH, OH, NO2, NH, CH3, CH2-triplet) carry
#     their published reference AEs at the appropriate ground-state
#     spin; spin/multiplicity is set by the step-7 driver via
#     `at.info['spin']`.
DICK_AE_DATA = [
    # --- 10 linear closed-shell ---
    {"hill": "H2",   "name": "Dihydrogen",
     "ae_kcalmol": 457.73 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 457.73 kJ/mol"},
    {"hill": "N2",   "name": "Dinitrogen",
     "ae_kcalmol": 955.82 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 955.82 kJ/mol"},
    {"hill": "FLi",  "name": "Lithium fluoride",
     "ae_kcalmol": 583.99 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 583.99 kJ/mol"},
    {"hill": "CHN",  "name": "Hydrogen cyanide",
     "ae_kcalmol": 1310.97 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1310.97 kJ/mol"},
    {"hill": "CO2",  "name": "Carbon dioxide",
     "ae_kcalmol": 1633.95 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1633.95 kJ/mol"},
    {"hill": "F2",   "name": "Difluorine",
     "ae_kcalmol": 162.31 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 162.31 kJ/mol"},
    {"hill": "C2H2", "name": "Acetylene",
     "ae_kcalmol": 405.525,  # step-6 anchor (W4-11)
     "source": "step6 (W4-11 anchor; H2O+C2H2 must match step-6)"},
    {"hill": "CO",   "name": "Carbon monoxide",
     "ae_kcalmol": 1087.57 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1087.57 kJ/mol"},
    {"hill": "HLi",  "name": "Lithium hydride",
     "ae_kcalmol": 242.27 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 242.27 kJ/mol"},
    {"hill": "Na2",  "name": "Disodium",
     "ae_kcalmol": 71.78 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 71.78 kJ/mol"},
    # --- 3 linear open-shell ---
    {"hill": "NO",   "name": "Nitric oxide",
     "ae_kcalmol": 639.28 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 639.28 kJ/mol"},
    {"hill": "CH",   "name": "Methylidyne radical",
     "ae_kcalmol": 351.60 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 351.60 kJ/mol"},
    {"hill": "HO",   "name": "Hydroxyl radical",
     "ae_kcalmol": 448.30 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 448.30 kJ/mol"},
    # --- 8 non-linear (Dick labels closed-shell; several open-shell) ---
    {"hill": "NO2",  "name": "Nitrogen dioxide",
     "ae_kcalmol": 954.10 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 954.10 kJ/mol"},
    {"hill": "HN",   "name": "Imidogen",
     "ae_kcalmol": 347.02 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 347.02 kJ/mol"},
    {"hill": "O3",   "name": "Ozone",
     "ae_kcalmol": 615.78 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 615.78 kJ/mol"},
    {"hill": "N2O",  "name": "Nitrous oxide",
     "ae_kcalmol": 1133.70 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1133.70 kJ/mol"},
    {"hill": "CH3",  "name": "Methyl radical",
     "ae_kcalmol": 1287.21 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1287.21 kJ/mol"},
    {"hill": "CH2",  "name": "Methylene (triplet, X 3B1)",
     "ae_kcalmol": 797.23 / 4.184,  # triplet wins in g2_97.traj
     "source": "Haunschild2012 Table I, E_ref,non-rel = 797.23 kJ/mol (Triplet carbene)"},
    {"hill": "H2O",  "name": "Water",
     "ae_kcalmol": 232.974,  # step-6 anchor (W4-11)
     "source": "step6 (W4-11 anchor; H2O+C2H2 must match step-6)"},
    {"hill": "H3N",  "name": "Ammonia",
     "ae_kcalmol": 1245.99 / 4.184,
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1245.99 kJ/mol"},
]

# Backward-compatible Hill list (used by select_subset / step-7 driver
# that already iterates pool["ae_molecules"]).
DICK_AE_HILL = [d["hill"] for d in DICK_AE_DATA]

# Quick-lookup map by Hill formula → AE in kcal/mol.
DICK_AE_KCALMOL = {d["hill"]: d["ae_kcalmol"] for d in DICK_AE_DATA}

# 3 BH76 reactions per Dick SI §II:
#   OH + N2 → H + N2O,   OH + CH3 → O + CH4,   HF + F → H + F2
#
# Reference forward-barrier heights (Vf) in kcal/mol come from the
# Truhlar Minnesota-database BH76 subset entries:
#   - NHTBH38/08 (heavy-atom transfer / non-H-transfer barriers):
#       https://comp.chem.umn.edu/db/dbs/nhtbh38.html
#   - HTBH38/08  (hydrogen-transfer barriers):
#       https://comp.chem.umn.edu/db/dbs/htbh38.html
# These are the values that Goerigk & Grimme (PCCP 19, 32184, 2017)
# adopt verbatim for the GMTKN55-BH76 subset. We use REF1 (the value
# directly comparable to non-relativistic calculations) for each.
DICK_BH76_REACTIONS = [
    {
        "name": "OH+N2_to_H+N2O",
        "reactants": ["HO", "N2"],
        "products": ["H", "N2O"],
        "coeffs": [-1.0, -1.0, +1.0, +1.0],
        # Forward barrier of OH+N2→H+N2O = REVERSE barrier of NHTBH38
        # entry #1 (H+N2O → OH+N2, Vf=17.13, Vr=82.27 kcal/mol REF1).
        "e_rxn_ref": 82.27,  # kcal/mol
        "source": (
            "NHTBH38/08 entry 1 (H+N2O → OH+N2), Vr (REF1) = 82.27 kcal/mol; "
            "Zheng, Zhao, Truhlar JCTC 5, 808 (2009); also GMTKN55-BH76."
        ),
    },
    {
        "name": "OH+CH3_to_O+CH4",
        "reactants": ["HO", "CH3"],
        "products": ["O", "CH4"],
        "coeffs": [-1.0, -1.0, +1.0, +1.0],
        # Forward barrier of OH+CH3→O+CH4 = REVERSE barrier of HTBH38
        # entry 19/20 (O+CH4 → OH+CH3, Vf=13.47, Vr=7.90 kcal/mol REF1).
        "e_rxn_ref": 7.90,  # kcal/mol
        "source": (
            "HTBH38/08 entry 19-20 (O+CH4 → OH+CH3), Vr (REF1) = 7.90 kcal/mol; "
            "Zheng, Zhao, Truhlar JCTC 5, 808 (2009); also GMTKN55-BH76."
        ),
    },
    {
        "name": "HF+F_to_H+F2",
        "reactants": ["FH", "F"],
        "products": ["H", "F2"],
        "coeffs": [-1.0, -1.0, +1.0, +1.0],
        # Forward barrier of HF+F→H+F2 = REVERSE barrier of NHTBH38
        # entry #5 (H+F2 → HF+F, Vf=2.27, Vr=105.80 kcal/mol REF1).
        "e_rxn_ref": 105.80,  # kcal/mol
        "source": (
            "NHTBH38/08 entry 5 (H+F2 → HF+F), Vr (REF1) = 105.80 kcal/mol; "
            "Zheng, Zhao, Truhlar JCTC 5, 808 (2009); also GMTKN55-BH76."
        ),
    },
]

# 2 IP13 ionization potentials: Li → Li⁺,  C → C⁺
#
# Reference IPs are NIST experimental atomic ionization energies, which
# are the canonical values used by IP13 (Lynch, Truhlar JPCA 107, 3898,
# 2003) and IP23 (Goerigk-Grimme GMTKN55, 2017). Conversion factor:
# 1 eV = 23.0605 kcal/mol (CODATA).
#   - Li I → Li II: 5.391719 eV (43487.150 cm⁻¹) → 124.335736 kcal/mol
#       NIST atomic spectra database, Atomic Data for Lithium,
#       https://www.physics.nist.gov/PhysRefData/Handbook/Tables/lithiumtable1.htm
#   - C  I → C  II: 11.26030 eV (90820.45 cm⁻¹) → 259.668148 kcal/mol
#       NIST atomic spectra database, Atomic Data for Carbon,
#       https://www.physics.nist.gov/PhysRefData/Handbook/Tables/carbontable1.htm
DICK_IP13_PAIRS = [
    {
        "name": "Li_IP",
        "neutral": "Li",
        "cation": "Li",
        "cation_charge": 1,
        # 5.391719 eV * 23.0605 kcal/(mol*eV) = 124.335736 kcal/mol
        "ip_ref": 124.335736,  # kcal/mol
        "source": (
            "NIST atomic spectra database, Li I IE_1 = 5.391719 eV "
            "(43487.150 cm^-1); converted via 1 eV = 23.0605 kcal/mol (CODATA)."
        ),
    },
    {
        "name": "C_IP",
        "neutral": "C",
        "cation": "C",
        "cation_charge": 1,
        # 11.26030 eV * 23.0605 kcal/(mol*eV) = 259.668148 kcal/mol
        "ip_ref": 259.668148,  # kcal/mol
        "source": (
            "NIST atomic spectra database, C I IE_1 = 11.26030 eV "
            "(90820.45 cm^-1); converted via 1 eV = 23.0605 kcal/mol (CODATA)."
        ),
    },
]

# 2 atomic-density references
DICK_ATOM_REFS = ["H", "Li"]


def _g297_traj_path() -> Path:
    """Authoritative G2/97 ASE-trajectory file."""
    return Path(__file__).resolve().parents[2] / "scripts" / "script_data" / \
        "haunschild_g2" / "g2_97.traj"


def build_dick_pool() -> dict:
    """Assemble the 28-entry Dick 2021 training pool.

    Returns dict with keys:
      ae_molecules       : 21 ASE Atoms (the AE-residual targets;
                            this is the SELECTION POOL for select_subset).
                           Each Atoms carries the following info-dict
                           entries used by step-7's training driver:
                             - "dick_hill"   : Hill formula key (str)
                             - "ae_kcalmol"  : AE reference (float, kcal/mol)
                             - "ae_source"   : citation string (str)
                             - "ae_name"     : human-readable name (str)
                           See module-level DICK_AE_DATA for sources.
      bh76_reactions     : 3 reaction-spec dicts
      ip13_pairs         : 2 IP-spec dicts
      atom_refs          : 2 ASE Atoms (H, Li)
      n_total            : 28
    """
    traj_path = _g297_traj_path()
    traj = read(str(traj_path), ":")

    by_hill: dict = {a.get_chemical_formula(): a for a in traj}

    ae_atoms: list = []
    missing: list = []
    for entry in DICK_AE_DATA:
        hill = entry["hill"]
        if hill in by_hill:
            a = by_hill[hill].copy()
            a.info["dick_hill"] = hill
            # Attach the published AE reference (kcal/mol) plus the
            # human-readable name and source citation.  Step-7's loss
            # driver reads `at.info["ae_kcalmol"]`; the source string is
            # for downstream provenance / sanity-check tests.
            a.info["ae_kcalmol"] = float(entry["ae_kcalmol"])
            a.info["ae_source"] = entry["source"]
            a.info["ae_name"] = entry["name"]
            ae_atoms.append(a)
        else:
            missing.append(hill)
    if missing:
        raise RuntimeError(
            f"Dick AE pool: {len(missing)} formulas missing from g2_97.traj: {missing}.\n"
            f"Available Hill formulas in g2_97.traj: {sorted(by_hill.keys())}"
        )

    atom_refs: list = []
    for sym in DICK_ATOM_REFS:
        if sym in by_hill:
            atom_refs.append(by_hill[sym].copy())
        else:
            atom_refs.append(Atoms(sym, positions=[(0.0, 0.0, 0.0)]))

    return {
        "ae_molecules": ae_atoms,
        "bh76_reactions": DICK_BH76_REACTIONS,
        "ip13_pairs": DICK_IP13_PAIRS,
        "atom_refs": atom_refs,
        "n_total": (
            len(ae_atoms) + len(DICK_BH76_REACTIONS)
            + len(DICK_IP13_PAIRS) + len(atom_refs)
        ),
    }
