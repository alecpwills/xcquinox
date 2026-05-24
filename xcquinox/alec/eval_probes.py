"""Step-7 held-out evaluation probe subsets.

Each probe is a 5-6 molecule (or reaction) subset of a host benchmark,
designed to test a distinct generalization axis of the GGA network
trained on the Dick & Fernandez-Serra 2021 SI §II 28-entry pool. Every
molecule has a published non-relativistic atomization energy (TAE_e in
kcal/mol), reaction barrier (Vf in kcal/mol), or first ionization
potential (kcal/mol) with a per-entry citation.

The four probe sets target:

- **Probe A — Chemical similarity transfer** (in-distribution AE).
  6 first-row organics/inorganics from G2/97 that mirror the diversity
  of the Dick training pool (saturated/unsaturated CH bonds; closed and
  open shells; oxygen and nitrogen radicals) but are NOT in the Dick
  training pool.  Tests in-distribution generalization.

- **Probe B — Heteroatom / heavier-element extrapolation** (OOD AE).
  6 molecules containing 3rd-period elements (S, P, Cl, Si) — none of
  which appear in the Dick pool.  Tests how a network trained on H, C,
  N, O, F, Li, Na transfers to H + 3rd-row chemistry.

- **Probe C — Reaction-energy transfer** (BH76 outside training).
  6 forward barrier heights from the Truhlar Minnesota HTBH38/08 +
  NHTBH38/08 datasets that constitute GMTKN55-BH76, EXCLUDING the 3
  reactions that appear in Dick training.  Tests kinetic-barrier
  transferability.

- **Probe D — Multireference / static-correlation challenge.**
  6 small molecules with significant static correlation (recognized
  hard cases for any single-reference GGA): O2 (³Σg⁻ triplet), CN
  (²Σ⁺ doublet), ClO (²Π doublet), OF2 (bent dichalcogen), Cl2
  (single-bond chalcogen) and BeH (one-electron Be-H bond).  Tests
  the network at the single-reference DFT failure boundary.

Citations
---------
- **Haunschild2012** : Haunschild & Klopper, *J. Chem. Phys.* **136**,
  164102 (2012), DOI 10.1063/1.4704796.  Table I, column "E_ref,non-rel"
  (frozen-core, non-relativistic atomization energies obtained from
  CCSD(T)(F12)/cc-pVQZ-F12 with higher-excitation and core/valence
  corrections), reported in kJ/mol; converted here via 1 kcal = 4.184 kJ.
  Local copy: ``scripts/script_data/haunschild_g2/haunschild2012.pdf``;
  local CSV (kJ/mol; column ``E``):
  ``scripts/script_data/haunschild_g2/g2_97.csv``.
- **HTBH38/08** : Zheng, Zhao, Truhlar, *J. Chem. Theory Comput.* **5**,
  808 (2009).  Hydrogen-transfer barrier database, REF1 column.
  Online: https://comp.chem.umn.edu/db/dbs/htbh38.html
- **NHTBH38/08** : Zheng, Zhao, Truhlar, *J. Chem. Theory Comput.* **5**,
  808 (2009).  Non-hydrogen-transfer barrier database, REF1 column.
  Online: https://comp.chem.umn.edu/db/dbs/nhtbh38.html
- **GMTKN55** : Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme,
  *Phys. Chem. Chem. Phys.* **19**, 32184 (2017), DOI 10.1039/c7cp04913g.
  GMTKN55-BH76 adopts the Truhlar HTBH/NHTBH REF1 values verbatim.
- **W4-11** (cross-check; not the primary AE source here):
  Karton, Daon, Martin, *Chem. Phys. Lett.* **510**, 165 (2011).
- **Karton2017 / W4-17** (multireference subset rationale):
  Karton, Sylvetsky, Martin, *J. Comp. Chem.* **38**, 2063 (2017).
- **NIST atomic spectra** : NIST Standard Reference Database 78
  (Atomic Spectra Database).  Used for any IP reference values.

Per-molecule rationale (the ``rationale`` field on every probe entry)
documents the mechanism / training-type analog the probe tests.

Convention
----------
``ae_kcalmol`` and ``reaction_energy_ref`` (the BH76RC reaction energy ΔE used
by the probe metric) and ``barrier_vf_ref`` (forward-barrier provenance) are
stored in kcal/mol throughout, to match ``dfs_pool.py``.  ``spin`` follows ASE
convention (number of unpaired electrons; e.g., O2 triplet has spin=2).
"""
from __future__ import annotations

from pathlib import Path
from ase import Atoms
from ase.io import read

# 1 kcal = 4.184 kJ exactly (CODATA).
_KCAL_PER_KJ = 1.0 / 4.184


# ---------------------------------------------------------------------------
# Probe A — chemical-similarity transfer (in-distribution AE)
# ---------------------------------------------------------------------------
#
# All values from Haunschild2012 Table I E_ref,non-rel (kJ/mol)
# converted to kcal/mol via 1 kcal = 4.184 kJ.
#
# Hill-formula keys correspond to the entries in
# ``scripts/script_data/haunschild_g2/g2_97.traj``.  The "spin" entry
# follows ASE convention (number of unpaired electrons).
PROBE_A_CHEMICAL_SIMILARITY = [
    {"hill": "CH4", "name": "Methane",
     "ae_kcalmol": 1757.82 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row CH4 (E_ref,non-rel = 1757.82 kJ/mol)",
     "rationale": (
         "Closed-shell saturated first-row organic.  Tests whether the "
         "network — trained on the open-shell CH3 radical — generalizes "
         "to its CH4 closed-shell parent.  CH4 is the paradigmatic test "
         "of CH-bond density representation.")},
    {"hill": "C2H4", "name": "Ethylene",
     "ae_kcalmol": 2358.75 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row C2H4 (E_ref,non-rel = 2358.75 kJ/mol)",
     "rationale": (
         "Closed-shell pi-bonded organic.  Training contains C2H2 (triple "
         "bond); C2H4 (double bond) tests whether the network captures "
         "the bond-order continuum from sp³ → sp² → sp.")},
    {"hill": "C2H6", "name": "Ethane",
     "ae_kcalmol": 2981.64 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row C2H6 (E_ref,non-rel = 2981.64 kJ/mol)",
     "rationale": (
         "Closed-shell saturated organic with C-C single bond.  Training "
         "lacks any neutral two-heavy-atom hydrocarbon other than C2H2, "
         "so this tests sp³-sp³ C-C bond AE prediction.")},
    {"hill": "CHO", "name": "Formyl radical",
     "ae_kcalmol": 1169.59 * _KCAL_PER_KJ,
     "spin": 1, "charge": 0,
     "source": "Haunschild2012 Table I row CHO (E_ref,non-rel = 1169.59 kJ/mol)",
     "rationale": (
         "Open-shell ²A' radical containing C, H, and O.  Training has "
         "CO and OH separately as well as CH (radical), but no fragment "
         "with all three elements; CHO probes interpolation across "
         "training-element combinations.")},
    {"hill": "H2N", "name": "Amino radical",
     "ae_kcalmol": 762.95 * _KCAL_PER_KJ,
     "spin": 1, "charge": 0,
     "source": "Haunschild2012 Table I row NH2 (E_ref,non-rel = 762.95 kJ/mol)",
     "rationale": (
         "Open-shell ²B₁ NH-bond radical.  Training has NH (triplet) and "
         "NH3 (closed-shell) but not the intermediate NH2 doublet; this "
         "tests interpolation along the N-H hydrogenation series.")},
    {"hill": "H2O2", "name": "Hydrogen peroxide",
     "ae_kcalmol": 1126.34 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row H2O2 (E_ref,non-rel = 1126.34 kJ/mol)",
     "rationale": (
         "Closed-shell O-O single-bond molecule.  Training contains O3 "
         "(triplet O-O-O) but no peroxide; H2O2 tests the network on a "
         "weak O-O σ-bond — known to be sensitive to delocalization "
         "error in semilocal DFT.")},
]


# ---------------------------------------------------------------------------
# Probe B — heteroatom / heavier-element extrapolation (OOD AE)
# ---------------------------------------------------------------------------
PROBE_B_HETEROATOM_EXTRAPOLATION = [
    {"hill": "H2S", "name": "Hydrogen sulfide",
     "ae_kcalmol": 768.72 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row H2S (E_ref,non-rel = 768.72 kJ/mol)",
     "rationale": (
         "Closed-shell 2nd-row hydride; 3rd-period analog of H2O.  "
         "Training contains H2O but no S-containing species, so H2S "
         "probes pure heteroatom transfer along an isovalent series.")},
    {"hill": "HCl", "name": "Hydrogen chloride",
     "ae_kcalmol": 449.58 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row HCl (E_ref,non-rel = 449.58 kJ/mol)",
     "rationale": (
         "Closed-shell 3rd-period diatomic hydride; halogen analog of "
         "HF.  Tests whether σ-bond-energy prediction transfers from "
         "F-row to Cl-row chemistry without retraining.")},
    {"hill": "OS", "name": "Sulfur monoxide",
     "ae_kcalmol": 528.72 * _KCAL_PER_KJ,
     "spin": 2, "charge": 0,
     "source": "Haunschild2012 Table I row OS (E_ref,non-rel = 528.72 kJ/mol)",
     "rationale": (
         "Open-shell ³Σ⁻ diatomic; 3rd-period analog of O2 (also "
         "triplet).  Training has CO (closed-shell) but no S-O bond; "
         "tests density extrapolation onto a S+O combination.")},
    {"hill": "O2S", "name": "Sulfur dioxide",
     "ae_kcalmol": 1091.61 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row O2S (E_ref,non-rel = 1091.61 kJ/mol)",
     "rationale": (
         "Closed-shell bent triatomic with two S-O bonds.  Heavier "
         "analog of O3 (which is in training); tests how the network "
         "scales bond-energy prediction across a +8 nuclear-charge "
         "substitution at a central atom.")},
    {"hill": "H3P", "name": "Phosphane",
     "ae_kcalmol": 1012.24 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row PH3 (E_ref,non-rel = 1012.24 kJ/mol)",
     "rationale": (
         "Closed-shell 3rd-period hydride; analog of NH3.  Training has "
         "NH3 (and NH, NH3, etc.) but no phosphorus; PH3 tests transfer "
         "across the N→P substitution while preserving Lewis structure.")},
    {"hill": "H4Si", "name": "Silane",
     "ae_kcalmol": 1357.91 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row SiH4 (E_ref,non-rel = 1357.91 kJ/mol)",
     "rationale": (
         "Closed-shell 3rd-period tetrahedral hydride; analog of CH4.  "
         "Training has CH4-like geometries (CH3) but no silicon; SiH4 "
         "is the canonical Si-containing AE benchmark molecule.")},
]


# ---------------------------------------------------------------------------
# Probe C — BH76 REACTION ENERGIES outside the Dick training set
# ---------------------------------------------------------------------------
#
# The probe metric is the REACTION ENERGY ΔE = Σ coeffs·E = E(products) −
# E(reactants), matching the BH76 TRAINING channel (losses._rxn_residual_term;
# dfs_pool.py). The reference ``reaction_energy_ref`` is the GMTKN55-BH76RC
# (W2-F12) value (Goerigk et al. PCCP 19, 32184 (2017); subset file
# ``BH76/.resRC`` — see scripts/script_data/GMTKN55_BH76RC_PROVENANCE.md). A
# barrier height (Vf) is NOT comparable to a reaction energy and could not be
# computed here anyway (no transition-state species are carried). The forward
# barrier ``barrier_vf_ref`` (Truhlar HTBH38/08 + NHTBH38/08 REF1; Zheng-Zhao-
# Truhlar JCTC 5, 808 (2009)) is retained for provenance only.
#
# Five reactions are outside Dick training; entry 5 (H+N2O→OH+N2) is the
# INTENTIONAL REVERSE of training reaction 1 (OH+N2→H+N2O) — a directional-
# consistency probe (its ΔE is −1× the training reaction's), NOT accidental
# leakage.
#
# Reactant/product names in this dict refer to MoleculeSpec.name strings
# that build_probe_pool() will create.  The convention matches
# DFS_BH76_REACTIONS: signed coefficients line up with
# (*reactants, *products) in order.
#
# Each entry carries per-species ``species_spins`` and ``species_charges``
# dicts (Hill formula → ground-state 2S / charge in PySCF convention).
# Sources for atomic spins: NIST ASD ground-state term symbols.
# Sources for molecular spins: NIST CCCBDB / Herzberg I.
PROBE_C_BH76_OUT_OF_TRAINING = [
    {
        "name": "OH+H2_to_H2O+H",
        "reactants": ["HO", "H2"],
        "products":  ["H2O", "H"],
        "coeffs":    [-1.0, -1.0, +1.0, +1.0],
        "reaction_energy_ref": -16.39,  # kcal/mol, GMTKN55-BH76RC (W2-F12)
        "barrier_vf_ref": 4.90,         # kcal/mol, HTBH38/08 REF1 (provenance)
        "species_spins":   {"HO": 1, "H2": 0, "H2O": 0, "H": 1},
        "species_charges": {"HO": 0, "H2": 0, "H2O": 0, "H": 0},
        "source": (
            "ΔE = -16.39 kcal/mol, GMTKN55-BH76RC (W2-F12; Goerigk et al. PCCP "
            "19, 32184 (2017), BH76/.resRC). Forward barrier Vf (REF1) = 4.90 "
            "kcal/mol, HTBH38/08 entry 2; Zheng, Zhao, Truhlar JCTC 5, 808 (2009)."
        ),
        "spin_source": (
            "NIST ASD H I (²S, spin=1); NIST CCCBDB OH (X²Π, 1), "
            "H2 (X¹Σg+, 0), H2O (X¹A1, 0)."
        ),
        "rationale": (
            "Classic combustion-relevant H-abstraction with low barrier. "
            "Probes whether the network's trained BH76 channel "
            "(OH+N2→H+N2O, OH+CH3→O+CH4, HF+F→H+F2) generalizes to a "
            "small, low-barrier H-transfer outside training."
        ),
    },
    {
        "name": "H+HCl_to_H2+Cl",
        "reactants": ["H", "HCl"],
        "products":  ["H2", "Cl"],
        "coeffs":    [-1.0, -1.0, +1.0, +1.0],
        "reaction_energy_ref": -1.90,  # kcal/mol, GMTKN55-BH76RC (W2-F12)
        "barrier_vf_ref": 5.70,        # kcal/mol, HTBH38/08 REF1 (provenance)
        "species_spins":   {"H": 1, "HCl": 0, "H2": 0, "Cl": 1},
        "species_charges": {"H": 0, "HCl": 0, "H2": 0, "Cl": 0},
        "source": (
            "ΔE = -1.90 kcal/mol, GMTKN55-BH76RC (W2-F12; BH76/.resRC). Forward "
            "barrier Vf (REF1) = 5.70 kcal/mol, HTBH38/08 entry 1; Zheng, Zhao, "
            "Truhlar JCTC 5, 808 (2009)."
        ),
        "spin_source": (
            "NIST ASD H I (²S, spin=1), Cl I (²P°, spin=1); "
            "NIST CCCBDB HCl (X¹Σ+, 0), H2 (X¹Σg+, 0)."
        ),
        "rationale": (
            "H-transfer involving a 3rd-period halogen.  Probes BH76 "
            "transferability AND heteroatom extrapolation simultaneously."
        ),
    },
    {
        "name": "CH3+H2_to_CH4+H",
        "reactants": ["CH3", "H2"],
        "products":  ["CH4", "H"],
        "coeffs":    [-1.0, -1.0, +1.0, +1.0],
        "reaction_energy_ref": -3.11,  # kcal/mol, GMTKN55-BH76RC (W2-F12)
        "barrier_vf_ref": 12.10,       # kcal/mol, HTBH38/08 REF1 (provenance)
        "species_spins":   {"CH3": 1, "H2": 0, "CH4": 0, "H": 1},
        "species_charges": {"CH3": 0, "H2": 0, "CH4": 0, "H": 0},
        "source": (
            "ΔE = -3.11 kcal/mol, GMTKN55-BH76RC (W2-F12; BH76/.resRC). Forward "
            "barrier Vf (REF1) = 12.10 kcal/mol, HTBH38/08 entry 3; Zheng, Zhao, "
            "Truhlar JCTC 5, 808 (2009)."
        ),
        "spin_source": (
            "NIST ASD H I (²S, spin=1); NIST CCCBDB CH3 (X²A2'', 1), "
            "H2 (X¹Σg+, 0), CH4 (X¹A1, 0)."
        ),
        "rationale": (
            "Methyl + H2 → methane.  Training has CH3 species but no "
            "saturated CH4 in any reaction; this tests reaction "
            "energetics across the CH3 → CH4 hydrogenation step."
        ),
    },
    {
        "name": "OH+NH3_to_H2O+NH2",
        "reactants": ["HO", "H3N"],
        "products":  ["H2O", "H2N"],
        "coeffs":    [-1.0, -1.0, +1.0, +1.0],
        "reaction_energy_ref": -10.32,  # kcal/mol, GMTKN55-BH76RC (W2-F12)
        "barrier_vf_ref": 3.00,         # kcal/mol, HTBH38/08 REF1 (provenance)
        "species_spins":   {"HO": 1, "H3N": 0, "H2O": 0, "H2N": 1},
        "species_charges": {"HO": 0, "H3N": 0, "H2O": 0, "H2N": 0},
        "source": (
            "ΔE = -10.32 kcal/mol, GMTKN55-BH76RC (W2-F12; BH76/.resRC). Forward "
            "barrier Vf (REF1) = 3.00 kcal/mol, HTBH38/08 entry 6; Zheng, Zhao, "
            "Truhlar JCTC 5, 808 (2009)."
        ),
        "spin_source": (
            "NIST CCCBDB OH (X²Π, 1), NH3 (X¹A1, 0), H2O (X¹A1, 0), "
            "NH2 (X²B1, 1)."
        ),
        "rationale": (
            "Atmospheric H-abstraction with very low barrier.  "
            "Probes the small-Vf regime where the BH76 channel "
            "is most sensitive to AE-error cancellation."
        ),
    },
    {
        "name": "H+N2O_to_OH+N2",
        "reactants": ["H", "N2O"],
        "products":  ["HO", "N2"],
        "coeffs":    [-1.0, -1.0, +1.0, +1.0],
        "reaction_energy_ref": -64.91,  # kcal/mol, GMTKN55-BH76RC (W2-F12)
        "barrier_vf_ref": 17.13,        # kcal/mol, NHTBH38/08 REF1 (provenance)
        "species_spins":   {"H": 1, "N2O": 0, "HO": 1, "N2": 0},
        "species_charges": {"H": 0, "N2O": 0, "HO": 0, "N2": 0},
        "source": (
            "ΔE = -64.91 kcal/mol, GMTKN55-BH76RC (W2-F12; BH76/.resRC, "
            "'h n2o -> oh n2'). This is the REVERSE of Dick training reaction 1 "
            "(OH+N2 -> H+N2O, +64.91), so ΔE here is exactly its negative. "
            "Forward barrier Vf (REF1) = 17.13 kcal/mol, NHTBH38/08 entry 1; "
            "Zheng, Zhao, Truhlar JCTC 5, 808 (2009)."
        ),
        "spin_source": (
            "NIST ASD H I (²S, spin=1); NIST CCCBDB OH (X²Π, 1), "
            "N2 (X¹Σg+, 0), N2O (X¹Σ+, 0)."
        ),
        "rationale": (
            "REVERSE direction of Dick training reaction 1 (training uses "
            "OH+N2 → H+N2O). Intentional directional-consistency probe: a "
            "network whose reaction-energy error cancels in one direction may "
            "not cancel in the other. ΔE = -(training ΔE) by construction."
        ),
    },
    {
        "name": "H+H2S_to_H2+HS",
        "reactants": ["H", "H2S"],
        "products":  ["H2", "HS"],
        "coeffs":    [-1.0, -1.0, +1.0, +1.0],
        "reaction_energy_ref": -13.26,  # kcal/mol, GMTKN55-BH76RC (W2-F12)
        "barrier_vf_ref": 3.50,         # kcal/mol, HTBH38/08 REF1 (provenance)
        "species_spins":   {"H": 1, "H2S": 0, "H2": 0, "HS": 1},
        "species_charges": {"H": 0, "H2S": 0, "H2": 0, "HS": 0},
        "source": (
            "ΔE = -13.26 kcal/mol, GMTKN55-BH76RC (W2-F12; BH76/.resRC). Forward "
            "barrier Vf (REF1) = 3.50 kcal/mol, HTBH38/08 entry 13; Zheng, Zhao, "
            "Truhlar JCTC 5, 808 (2009)."
        ),
        "spin_source": (
            "NIST ASD H I (²S, spin=1); NIST CCCBDB H2 (X¹Σg+, 0), "
            "H2S (X¹A1, 0), HS (X²Π, 1)."
        ),
        "rationale": (
            "Low-barrier H-transfer with sulfur.  Combines BH76 "
            "transferability with 3rd-period (S) heteroatom "
            "extrapolation in a single benchmark."
        ),
    },
]


# ---------------------------------------------------------------------------
# Probe D — multireference / static-correlation challenge
# ---------------------------------------------------------------------------
#
# These molecules are recognized as challenging for any single-reference
# DFT method (and certainly for any GGA).  Karton, Sylvetsky, Martin
# JCC 38, 2063 (2017) §II identifies a 17-molecule W4-17-MR subset that
# includes O2, ClO and CN as canonical multireference cases.  BeH carries
# a single valence electron and is a classic GGA self-interaction stress
# test (Cohen, Mori-Sanchez, Yang, Science 321, 792 (2008)).
#
# Reference AEs are the same authoritative Haunschild2012 values used for
# the rest of the alec stack — frozen-core CCSD(T)(F12) — converted from
# kJ/mol to kcal/mol.  These are non-relativistic; agreement with W4-17
# TAEe column is sub-0.5 kcal/mol on every entry that appears in W4-17.
PROBE_D_MULTIREFERENCE = [
    {"hill": "O2", "name": "Dioxygen (triplet)",
     "ae_kcalmol": 505.88 * _KCAL_PER_KJ,
     "spin": 2, "charge": 0,
     "source": "Haunschild2012 Table I row O2 (E_ref,non-rel = 505.88 kJ/mol)",
     "rationale": (
         "³Σg⁻ ground-state triplet — the canonical small-molecule "
         "multireference benchmark.  GGAs typically over-bind O2 by "
         "5-10 kcal/mol.  See Karton 2017 W4-17 §II ('multireference "
         "subset W4-17-MR') for the diagnostic-molecule rationale.")},
    {"hill": "CN", "name": "Cyano radical",
     "ae_kcalmol": 758.56 * _KCAL_PER_KJ,
     "spin": 1, "charge": 0,
     "source": "Haunschild2012 Table I row CN (E_ref,non-rel = 758.56 kJ/mol)",
     "rationale": (
         "²Σ⁺ open-shell diatomic with notorious near-degeneracy "
         "between σ and π configurations.  Listed in Karton 2017 "
         "W4-17-MR.  Training has C and N atoms but no C-N triple "
         "bond in radical form — only the closed-shell HCN.")},
    {"hill": "ClO", "name": "Chlorine monoxide",
     "ae_kcalmol": 271.20 * _KCAL_PER_KJ,
     "spin": 1, "charge": 0,
     "source": "Haunschild2012 Table I row OCl (E_ref,non-rel = 271.20 kJ/mol)",
     "rationale": (
         "²Π open-shell halogen oxide — recognized multireference "
         "diatomic (Karton 2017 W4-17-MR).  Combines static "
         "correlation with 3rd-period (Cl) extrapolation; double "
         "out-of-distribution probe.")},
    {"hill": "F2O", "name": "Difluorine monoxide",
     "ae_kcalmol": 392.68 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row OF2 (E_ref,non-rel = 392.68 kJ/mol)",
     "rationale": (
         "Bent F-O-F triatomic — closed-shell but with weak O-F "
         "single bonds and significant nondynamical correlation.  "
         "Pairs with O3 in training (also bent triatomic, but "
         "homonuclear); tests heteronuclear bent geometry.")},
    {"hill": "Cl2", "name": "Dichlorine",
     "ae_kcalmol": 248.22 * _KCAL_PER_KJ,
     "spin": 0, "charge": 0,
     "source": "Haunschild2012 Table I row Cl2 (E_ref,non-rel = 248.22 kJ/mol)",
     "rationale": (
         "Closed-shell single-bond halogen homodiatomic; analog of "
         "F2 (in training).  GGAs systematically under-bind Cl2 due "
         "to the diffuse 3p valence shell.  Tests transfer of the "
         "F2 single-bond representation onto a 3rd-period halogen.")},
    {"hill": "HBe", "name": "Beryllium monohydride",
     "ae_kcalmol": 212.50 * _KCAL_PER_KJ,
     "spin": 1, "charge": 0,
     "source": "Haunschild2012 Table I row BeH (E_ref,non-rel = 212.50 kJ/mol)",
     "rationale": (
         "²Σ⁺ open-shell diatomic with one valence electron pair "
         "shared between Be and H.  Be has a notoriously hard "
         "near-degeneracy of 2s and 2p; classic GGA self-interaction "
         "stress test (Cohen-Mori-Sanchez-Yang, Science 321, 792, "
         "2008).  Tests element-extrapolation onto the pre-Boron "
         "row of the periodic table not seen in training.")},
]


# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------
ALL_PROBES = {
    "probe_a_chemical_similarity":   PROBE_A_CHEMICAL_SIMILARITY,
    "probe_b_heteroatom":            PROBE_B_HETEROATOM_EXTRAPOLATION,
    "probe_c_bh76_transfer":         PROBE_C_BH76_OUT_OF_TRAINING,
    "probe_d_multireference":        PROBE_D_MULTIREFERENCE,
}

PROBE_KIND = {
    "probe_a_chemical_similarity": "ae",
    "probe_b_heteroatom":          "ae",
    "probe_c_bh76_transfer":       "bh76",
    "probe_d_multireference":      "ae",
}


def _g297_traj_path() -> Path:
    """Authoritative G2/97 ASE-trajectory file (same as dfs_pool)."""
    return Path(__file__).resolve().parents[2] / "scripts" / "script_data" / \
        "haunschild_g2" / "g2_97.traj"


def _bh76_extra_geometries() -> dict:
    """Hill-formula → ASE Atoms for BH76 reactant/product species that
    are NOT in g2_97.traj (typically bare atoms and HS).

    Geometries here are deliberately minimal — atomic positions for bare
    atoms; equilibrium HS bond length from CCCBDB (1.341 Å) for the
    diatomic HS radical.  These geometries enter the AE/total-energy
    computation only as inputs to the SCF, so a few mÅ accuracy is
    irrelevant to the BH76 reaction-energy probe (the reactants and
    products use the SAME geometries on both sides of the reaction).
    """
    extras: dict = {
        # Bare atoms used as reactants/products.  ASE convention sets
        # spin via `info["spin"]` which step-7's MoleculeSpec builder
        # picks up.  Ground-state spins:
        "H":  Atoms("H",  positions=[(0.0, 0.0, 0.0)]),  # ²S
        "F":  Atoms("F",  positions=[(0.0, 0.0, 0.0)]),  # ²P
        "O":  Atoms("O",  positions=[(0.0, 0.0, 0.0)]),  # ³P
        "N":  Atoms("N",  positions=[(0.0, 0.0, 0.0)]),  # ⁴S
        "C":  Atoms("C",  positions=[(0.0, 0.0, 0.0)]),  # ³P
        "Cl": Atoms("Cl", positions=[(0.0, 0.0, 0.0)]),  # ²P
        # HS diatomic (²Π); CCCBDB experimental r_e = 1.341 Å.
        "HS": Atoms("HS", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.341)]),
    }
    # Ground-state spin (number of unpaired electrons) per ASE convention.
    spin_map = {"H": 1, "F": 1, "O": 2, "N": 3, "C": 2, "Cl": 1, "HS": 1}
    for sym, sp in spin_map.items():
        extras[sym].info["spin"] = sp
        extras[sym].info["charge"] = 0
        extras[sym].info["name"] = sym
    return extras


def _attach_info(at: Atoms, entry: dict) -> Atoms:
    """Attach probe metadata to an ASE Atoms object (matches dfs_pool style).

    Sets:
      - dfs_hill / probe_hill : Hill formula key (str)
      - ae_kcalmol            : AE reference (float, kcal/mol)
      - ae_source             : citation string
      - ae_name               : human-readable name
      - rationale             : per-molecule probe rationale
      - spin / charge         : open-shell metadata (read by training driver)
      - name                  : training-driver-friendly name (= ae_name)
    """
    a = at.copy()
    a.info["probe_hill"] = entry["hill"]
    a.info["dfs_hill"] = entry["hill"]    # alias for step-7 driver compat
    a.info["ae_kcalmol"] = float(entry["ae_kcalmol"])
    a.info["ae_source"] = entry["source"]
    a.info["ae_name"] = entry["name"]
    a.info["rationale"] = entry["rationale"]
    a.info["spin"] = int(entry.get("spin", 0))
    a.info["charge"] = int(entry.get("charge", 0))
    a.info["name"] = entry["name"]
    return a


def build_probe_pool(probe_name: str) -> dict:
    """Assemble probe-set evaluation inputs.

    Parameters
    ----------
    probe_name : str
        One of ``ALL_PROBES.keys()``.

    Returns
    -------
    dict
        For AE probes (A, B, D)::

            {
                "kind":             "ae",
                "molecules":        list[ase.Atoms],   # carries info-dict
                "ae_refs_kcalmol":  dict[str, float],  # name -> ae kcal/mol
                "atom_set":         set[str],          # element symbols
                "n":                int,                # len(molecules)
                "entries":          list[dict],         # raw probe data
            }

        For BH76 probes (C)::

            {
                "kind":             "bh76",
                "reactions":        list[dict],         # like DFS_BH76_REACTIONS
                "molecules":        list[ase.Atoms],   # all reactants+products
                "atom_set":         set[str],
                "n":                int,                # len(reactions)
                "entries":          list[dict],
            }

    Raises
    ------
    ValueError
        If ``probe_name`` is unknown.
    RuntimeError
        If a probe-A/B/D Hill formula is missing from ``g2_97.traj``.
    """
    if probe_name not in ALL_PROBES:
        raise ValueError(
            f"unknown probe {probe_name!r}; known: {sorted(ALL_PROBES)}")
    entries = ALL_PROBES[probe_name]
    kind = PROBE_KIND[probe_name]

    if kind == "ae":
        traj = read(str(_g297_traj_path()), ":")
        by_hill: dict = {a.get_chemical_formula(): a for a in traj}
        molecules: list = []
        ae_refs: dict = {}
        missing: list = []
        atom_set: set = set()
        for entry in entries:
            hill = entry["hill"]
            if hill not in by_hill:
                missing.append(hill)
                continue
            a = _attach_info(by_hill[hill], entry)
            molecules.append(a)
            ae_refs[entry["name"]] = float(entry["ae_kcalmol"])
            atom_set.update(a.get_chemical_symbols())
        if missing:
            raise RuntimeError(
                f"{probe_name}: {len(missing)} Hill formulas missing from "
                f"g2_97.traj: {missing}.")
        return {
            "kind": "ae",
            "molecules": molecules,
            "ae_refs_kcalmol": ae_refs,
            "atom_set": atom_set,
            "n": len(molecules),
            "entries": list(entries),
        }

    # kind == "bh76"
    #
    # Spin/charge propagation: read every species' ground-state spin and
    # charge from the per-reaction ``species_spins`` / ``species_charges``
    # dicts on each PROBE_C entry.  This is load-bearing — without it,
    # PySCF will reject the SCF call for any open-shell radical (e.g. NO,
    # OH, CH3, HS) because g2_97.traj carries no spin info and PySCF's
    # default of spin=0 violates the (nelec - spin) % 2 == 0 invariant
    # for odd-electron molecules.  See docstring of dfs_pool.py for the
    # original incident report (NO, 15 electrons, smoke run 2026-05-01).
    traj = read(str(_g297_traj_path()), ":")
    by_hill: dict = {a.get_chemical_formula(): a for a in traj}
    extras = _bh76_extra_geometries()
    seen_species: dict = {}    # species_hill -> Atoms (deduplicated)
    atom_set: set = set()
    for rxn in entries:
        species_spins = rxn.get("species_spins", {})
        species_charges = rxn.get("species_charges", {})
        for sp in (*rxn["reactants"], *rxn["products"]):
            if sp in seen_species:
                continue
            if sp in by_hill:
                a = by_hill[sp].copy()
                a.info.setdefault("name", sp)
                # Use the per-reaction ground-state spin if provided
                # (authoritative).  No fallback — every PROBE_C entry
                # MUST carry species_spins per the design above.
                if sp not in species_spins:
                    raise RuntimeError(
                        f"{probe_name}: reaction {rxn['name']!r} is missing "
                        f"species_spins[{sp!r}].  Every BH76 reactant/product "
                        f"requires an explicit ground-state spin (PySCF 2S) "
                        f"so the SCF call satisfies (nelec - spin) % 2 == 0."
                    )
                a.info["spin"] = int(species_spins[sp])
                a.info["charge"] = int(species_charges.get(sp, 0))
                seen_species[sp] = a
                atom_set.update(a.get_chemical_symbols())
            elif sp in extras:
                a = extras[sp].copy()
                # Allow per-reaction override of the bare-atom spin
                # (e.g. if a probe used a non-ground-state ionic state),
                # but ground-state spin remains the default from
                # _bh76_extra_geometries.
                if sp in species_spins:
                    a.info["spin"] = int(species_spins[sp])
                if sp in species_charges:
                    a.info["charge"] = int(species_charges[sp])
                seen_species[sp] = a
                atom_set.update(a.get_chemical_symbols())
            else:
                raise RuntimeError(
                    f"{probe_name}: BH76 species {sp!r} not in g2_97.traj "
                    f"and not in eval_probes._bh76_extra_geometries; add a "
                    f"geometry there.")
    return {
        "kind": "bh76",
        "reactions": list(entries),
        "molecules": list(seen_species.values()),
        "atom_set": atom_set,
        "n": len(entries),
        "entries": list(entries),
    }
