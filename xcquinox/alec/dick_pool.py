"""Dick & Fernandez-Serra 2021 SI §II training pool.

Citation: Dick, S.; Fernandez-Serra, M. *Phys. Rev. B* **104**, L161109 (2021).
SI §II "Training Data" page 1 (verbatim member lists transcribed below).

Pool composition (28 distinct training points):
- 21 atomization-energy (AE) entries from G2/97 (10 linear closed-shell +
  3 linear open-shell + 8 non-linear)
- 3 BH76 reaction barriers
- 2 IP13 ionization potentials
- 2 atomic-density references (H, Li)
"""
from __future__ import annotations

from pathlib import Path
from ase import Atoms
from ase.io import read

# Dick 2021 SI §II AE molecule list, in ASE Hill formula. Names are
# Dick's verbatim (e.g. CNH) but ASE Hill formulas may be different
# (CHN). The list below uses the ASE Hill formula form so that
# `at.get_chemical_formula()` matches.
DICK_AE_HILL = [
    # 10 linear closed-shell:
    # H2, N2, LiF, CNH (=HCN), CO2, F2, C2H2, OC (=CO), LiH, Na2
    "H2", "N2", "FLi", "CHN", "CO2", "F2", "C2H2", "CO", "HLi", "Na2",
    # 3 linear open-shell:
    # NO, CH, OH
    "NO", "CH", "HO",
    # 8 non-linear (Dick labels these "closed-shell"; several actually
    # open-shell at multireference level — transcribed faithfully):
    # NO2, NH, O3, N2O, CH3, CH2, H2O, NH3
    "NO2", "HN", "O3", "N2O", "CH3", "CH2", "H2O", "H3N",
]

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
                            this is the SELECTION POOL for select_subset)
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
    for hill in DICK_AE_HILL:
        if hill in by_hill:
            a = by_hill[hill].copy()
            a.info["dick_hill"] = hill
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
