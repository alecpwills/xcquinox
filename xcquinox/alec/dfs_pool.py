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
and are attached to the build_dfs_pool() output via Atoms.info /
reaction-spec dict entries — never fabricated.  See DFS_AE_DATA for AE
sources (W4-11 anchors via step-6 for H2O+C2H2; Haunschild & Klopper
J. Chem. Phys. 136, 164102 (2012) for the other 19).

Spin / charge metadata
----------------------
Every species in this pool — molecular AE entries, BH76 reactant/product
species, IP13 neutral/cation pairs, and atomic refs — carries an
explicit ground-state ``spin`` (= 2S = N_α − N_β, PySCF convention) and
``charge`` (default 0; +1 for cations).  Sources for every value:
  - Atomic spins: NIST Atomic Spectra Database (ASD; Standard Reference
    Database 78), Hund's-rule ground-state term symbols.
  - Molecular spins: NIST CCCBDB / Herzberg, *Molecular Spectra and
    Molecular Structure*; per-entry comments cite the term symbol.

Special cases (verified against published spectroscopy):
  - **NH** (X³Σ⁻): triplet ground state — spin=2.  Herzberg
    "Molecular Spectra and Molecular Structure I" §VI; NIST CCCBDB.
  - **CH₂** (X³B₁): triplet ground state — spin=2.  ¹A₁ singlet is a
    low-lying excited state ~9 kcal/mol higher (the "methylene paradox";
    Bunker & Sears 1985; NIST CCCBDB).  G2/97 traj entry 106 ("Triplet
    carbene") carries the triplet TAE_e — must be run as triplet.
  - **O₃** (X¹A₁): closed-shell singlet ground state despite
    multireference character (Borden 1996; W4-17 SI).  spin=0.
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
#     (idx 106) entries with identical Hill formula "CH2"; build_dfs_pool
#     selects the triplet (last in iteration), matching W4-17 ch2-trip
#     (TAE_e = 190.53) and Haunschild "Triplet carbene" (797.23 kJ/mol →
#     190.541 kcal/mol).  Spin metadata for this entry MUST be 2.
#   - Open-shell molecules (NO, CH, OH, NO2, NH, CH3, CH2-triplet) carry
#     their published reference AEs at the appropriate ground-state
#     spin; spin/multiplicity is set on every Atoms via at.info['spin']
#     by build_dfs_pool() and read by _ase_atoms_to_pyscf_mol +
#     MoleculeSpec construction in the step-7 notebook builder.
#
# Spin field convention (per entry below): PySCF spin = 2S = N_α − N_β.
# Closed-shell singlets: spin=0.  Doublets: spin=1.  Triplets: spin=2.
# Quartets: spin=3.  Sources for each non-trivial spin are cited inline.
DFS_AE_DATA = [
    # --- 10 linear closed-shell ---
    {"hill": "H2",   "name": "Dihydrogen",
     "ae_kcalmol": 457.73 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σg+ closed-shell (Herzberg I, NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 457.73 kJ/mol"},
    {"hill": "N2",   "name": "Dinitrogen",
     "ae_kcalmol": 955.82 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σg+ closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 955.82 kJ/mol"},
    {"hill": "FLi",  "name": "Lithium fluoride",
     "ae_kcalmol": 583.99 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σ+ closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 583.99 kJ/mol"},
    {"hill": "CHN",  "name": "Hydrogen cyanide",
     "ae_kcalmol": 1310.97 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σ+ closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1310.97 kJ/mol"},
    {"hill": "CO2",  "name": "Carbon dioxide",
     "ae_kcalmol": 1633.95 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σg+ closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1633.95 kJ/mol"},
    {"hill": "F2",   "name": "Difluorine",
     "ae_kcalmol": 162.31 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σg+ closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 162.31 kJ/mol"},
    {"hill": "C2H2", "name": "Acetylene",
     "ae_kcalmol": 405.525,  # step-6 anchor (W4-11)
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σg+ closed-shell (NIST CCCBDB)",
     "source": "step6 (W4-11 anchor; H2O+C2H2 must match step-6)"},
    {"hill": "CO",   "name": "Carbon monoxide",
     "ae_kcalmol": 1087.57 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σ+ closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1087.57 kJ/mol"},
    {"hill": "HLi",  "name": "Lithium hydride",
     "ae_kcalmol": 242.27 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σ+ closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 242.27 kJ/mol"},
    {"hill": "Na2",  "name": "Disodium",
     "ae_kcalmol": 71.78 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σg+ closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 71.78 kJ/mol"},
    # --- 3 linear open-shell ---
    {"hill": "NO",   "name": "Nitric oxide",
     "ae_kcalmol": 639.28 / 4.184,
     "spin": 1, "charge": 0,
     "spin_source": "X²Π doublet ground state (NIST CCCBDB; Herzberg I)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 639.28 kJ/mol"},
    {"hill": "CH",   "name": "Methylidyne radical",
     "ae_kcalmol": 351.60 / 4.184,
     "spin": 1, "charge": 0,
     "spin_source": "X²Π doublet ground state (NIST CCCBDB; Herzberg I)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 351.60 kJ/mol"},
    {"hill": "HO",   "name": "Hydroxyl radical",
     "ae_kcalmol": 448.30 / 4.184,
     "spin": 1, "charge": 0,
     "spin_source": "X²Π doublet ground state (NIST CCCBDB; Herzberg I)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 448.30 kJ/mol"},
    # --- 8 non-linear (Dick labels closed-shell; several open-shell) ---
    {"hill": "NO2",  "name": "Nitrogen dioxide",
     "ae_kcalmol": 954.10 / 4.184,
     "spin": 1, "charge": 0,
     "spin_source": "X²A1 doublet ground state (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 954.10 kJ/mol"},
    {"hill": "HN",   "name": "Imidogen",
     # NH ground state is X³Σ⁻ triplet — analogous to O2.
     # Herzberg, *Molecular Spectra and Molecular Structure I*, §VI;
     # NIST CCCBDB row "NH (Imidogen)".
     "ae_kcalmol": 347.02 / 4.184,
     "spin": 2, "charge": 0,
     "spin_source": "X³Σ⁻ triplet ground state (Herzberg I §VI; NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 347.02 kJ/mol"},
    {"hill": "O3",   "name": "Ozone",
     # O3 ground state: X¹A1 closed-shell singlet (despite multireference
     # character; Borden 1996; W4-17 SI).  Restricted-singlet is the
     # canonical reference state.
     "ae_kcalmol": 615.78 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹A1 closed-shell singlet (Borden 1996; W4-17 SI; NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 615.78 kJ/mol"},
    {"hill": "N2O",  "name": "Nitrous oxide",
     "ae_kcalmol": 1133.70 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹Σ+ closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1133.70 kJ/mol"},
    {"hill": "CH3",  "name": "Methyl radical",
     "ae_kcalmol": 1287.21 / 4.184,
     "spin": 1, "charge": 0,
     "spin_source": "X²A2'' doublet ground state (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1287.21 kJ/mol"},
    {"hill": "CH2",  "name": "Methylene (triplet, X 3B1)",
     # CH2 ground state is X³B1 triplet (the methylene paradox; ¹A1
     # singlet ~9 kcal/mol higher).  Bunker & Sears, JCP 83, 4866 (1985);
     # NIST CCCBDB "CH2 triplet".  G2/97 traj entry 106 carries the
     # triplet TAE_e (797.23 kJ/mol).
     "ae_kcalmol": 797.23 / 4.184,  # triplet wins in g2_97.traj
     "spin": 2, "charge": 0,
     "spin_source": "X³B1 triplet ground state (Bunker & Sears 1985; NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 797.23 kJ/mol (Triplet carbene)"},
    {"hill": "H2O",  "name": "Water",
     "ae_kcalmol": 232.974,  # step-6 anchor (W4-11)
     "spin": 0, "charge": 0,
     "spin_source": "X¹A1 closed-shell (NIST CCCBDB)",
     "source": "step6 (W4-11 anchor; H2O+C2H2 must match step-6)"},
    {"hill": "H3N",  "name": "Ammonia",
     "ae_kcalmol": 1245.99 / 4.184,
     "spin": 0, "charge": 0,
     "spin_source": "X¹A1 closed-shell (NIST CCCBDB)",
     "source": "Haunschild2012 Table I, E_ref,non-rel = 1245.99 kJ/mol"},
]

# Backward-compatible Hill list (used by select_subset / step-7 driver
# that already iterates pool["ae_molecules"]).
DFS_AE_HILL = [d["hill"] for d in DFS_AE_DATA]

# Quick-lookup map by Hill formula → AE in kcal/mol.
DFS_AE_KCALMOL = {d["hill"]: d["ae_kcalmol"] for d in DFS_AE_DATA}

# Quick-lookup map by Hill formula → ground-state spin (2S, PySCF
# convention).  Used by build_dfs_pool() to attach at.info["spin"] for
# every Atoms returned, and re-exported as a separate map so callers
# (notebook builders, post-processing scripts) can validate spin
# coverage without rebuilding the pool.
DFS_AE_SPIN = {d["hill"]: d["spin"] for d in DFS_AE_DATA}

# 3 BH76 reactions per Dick SI §II:
#   OH + N2 → H + N2O,   OH + CH3 → O + CH4,   HF + F → H + F2
#
# Reference forward-barrier heights (Vf) and reverse-barrier heights (Vr)
# in kcal/mol come from the Truhlar Minnesota-database BH76 subset entries:
#   - NHTBH38/08 (heavy-atom transfer / non-H-transfer barriers):
#       https://comp.chem.umn.edu/db/dbs/nhtbh38.html
#   - HTBH38/08  (hydrogen-transfer barriers):
#       https://comp.chem.umn.edu/db/dbs/htbh38.html
# These are the values that Goerigk & Grimme (PCCP 19, 32184, 2017)
# adopt verbatim for the GMTKN55-BH76 subset. We use REF1 (the value
# directly comparable to non-relativistic calculations) for each.
#
# bh76_mode toggle (added 2026-05-19)
# -----------------------------------
# The loss term ``_rxn_residual_term`` (losses.py) computes
# ``e_rxn = Σ(coeffs · e_nn) = E(products) − E(reactants)`` — a true
# *reaction energy* ΔE, NOT a barrier height. Dick & Fernandez-Serra
# 2021 trained against reaction energies (their training set had no
# transition-state geometries; SI §II). Therefore each entry below
# carries BOTH numbers:
#   - ``barrier_ref``        — the forward barrier height (kept for
#                              provenance and for the opt-in
#                              ``bh76_mode="barrier_height"`` path).
#   - ``reaction_energy_ref``— the true reaction energy ΔE of the
#                              reactant→product direction below.
#   - ``e_rxn_ref``          — kept for backward compatibility; equals
#                              ``barrier_ref`` here. The mode-aware
#                              builder in training_points.py selects the
#                              correct value per ``bh76_mode``.
# For an elementary reaction ΔE = Vr − Vf exactly. Using the in-code
# Vf/Vr below, the reaction energies for the pool's reaction directions
# (GMTKN55-BH76RC) are:
#   - OH+N2 → H+N2O:  Vr 82.27 − Vf 17.13 =  +65.14 kcal/mol
#   - OH+CH3 → O+CH4: Vr  7.90 − Vf 13.47 =   −5.57 kcal/mol
#   - HF+F → H+F2:    Vr 105.80 − Vf 2.27 = +103.53 kcal/mol
#
# Per-reaction ``species_spins`` and ``species_charges`` dicts hold the
# ground-state spin (2S, PySCF convention) and charge (default 0) for
# each Hill formula appearing in reactants/products.  Sources:
#   - Atomic ground states from NIST ASD (Hund's rule term symbols):
#       H (²S, spin=1), F (²P°, spin=1), O (³P, spin=2)
#   - Molecular ground states from NIST CCCBDB / Herzberg I:
#       OH (X²Π, 1), N2 (X¹Σg+, 0), N2O (X¹Σ+, 0), CH3 (X²A2'', 1),
#       CH4 (X¹A1, 0), HF (X¹Σ+, 0), F2 (X¹Σg+, 0)
#
# ``ts_species`` is an optional transition-state-species slot (default
# None). It is required ONLY for ``bh76_mode="barrier_height"``: the
# barrier-height path needs a TS geometry so that
# ``Σ coeffs·E = E(TS) − E(reactants)`` is a true forward barrier. The
# 3 BH76 transition-state geometries are NOT yet staged in this repo;
# until they are, ``bh76_mode="barrier_height"`` raises a clear error
# (see training_points.build_dfs_pool_points).
#
# Valid bh76_mode values, exported for validation by the builder.
BH76_MODES: tuple[str, ...] = ("reaction_energy", "barrier_height")

DFS_BH76_REACTIONS = [
    {
        "name": "OH+N2_to_H+N2O",
        "reactants": ["HO", "N2"],
        "products": ["H", "N2O"],
        "coeffs": [-1.0, -1.0, +1.0, +1.0],
        # Forward barrier of OH+N2→H+N2O = REVERSE barrier of NHTBH38
        # entry #1 (H+N2O → OH+N2, Vf=17.13, Vr=82.27 kcal/mol REF1).
        "barrier_ref": 82.27,  # kcal/mol — forward barrier (Vr of NHTBH38 #1)
        # Reaction energy ΔE of OH+N2 → H+N2O (GMTKN55-BH76RC):
        #   ΔE = Vr − Vf = 82.27 − 17.13 = +65.14 kcal/mol.
        "reaction_energy_ref": 65.14,  # kcal/mol
        # Backward-compat alias (equals barrier_ref); the mode-aware
        # builder selects barrier_ref vs reaction_energy_ref.
        "e_rxn_ref": 82.27,  # kcal/mol
        # Optional TS geometry for bh76_mode="barrier_height" (not staged).
        "ts_species": None,
        "species_spins":   {"HO": 1, "N2": 0, "H": 1, "N2O": 0},
        "species_charges": {"HO": 0, "N2": 0, "H": 0, "N2O": 0},
        "source": (
            "NHTBH38/08 entry 1 (H+N2O → OH+N2), Vr (REF1) = 82.27 kcal/mol; "
            "Zheng, Zhao, Truhlar JCTC 5, 808 (2009); also GMTKN55-BH76."
        ),
        "spin_source": (
            "NIST ASD H I (²S, spin=1); NIST CCCBDB OH (X²Π, 1), "
            "N2 (X¹Σg+, 0), N2O (X¹Σ+, 0)."
        ),
    },
    {
        "name": "OH+CH3_to_O+CH4",
        "reactants": ["HO", "CH3"],
        "products": ["O", "CH4"],
        "coeffs": [-1.0, -1.0, +1.0, +1.0],
        # Forward barrier of OH+CH3→O+CH4 = REVERSE barrier of HTBH38
        # entry 19/20 (O+CH4 → OH+CH3, Vf=13.47, Vr=7.90 kcal/mol REF1).
        "barrier_ref": 7.90,  # kcal/mol — forward barrier (Vr of HTBH38 19-20)
        # Reaction energy ΔE of OH+CH3 → O+CH4 (GMTKN55-BH76RC):
        #   ΔE = Vr − Vf = 7.90 − 13.47 = −5.57 kcal/mol.
        "reaction_energy_ref": -5.57,  # kcal/mol
        # Backward-compat alias (equals barrier_ref).
        "e_rxn_ref": 7.90,  # kcal/mol
        # Optional TS geometry for bh76_mode="barrier_height" (not staged).
        "ts_species": None,
        "species_spins":   {"HO": 1, "CH3": 1, "O": 2, "CH4": 0},
        "species_charges": {"HO": 0, "CH3": 0, "O": 0, "CH4": 0},
        "source": (
            "HTBH38/08 entry 19-20 (O+CH4 → OH+CH3), Vr (REF1) = 7.90 kcal/mol; "
            "Zheng, Zhao, Truhlar JCTC 5, 808 (2009); also GMTKN55-BH76."
        ),
        "spin_source": (
            "NIST ASD O I (³P, spin=2); NIST CCCBDB OH (X²Π, 1), "
            "CH3 (X²A2'', 1), CH4 (X¹A1, 0)."
        ),
    },
    {
        "name": "HF+F_to_H+F2",
        # NOTE: ASE's get_chemical_formula() / Atoms.get_chemical_formula()
        # returns "HF" for H-F (not "FH") despite Hill ordering.  The
        # MoleculeSpec.name set by the step-7 notebook builder uses Hill
        # formula consistently, so reactant species are keyed as "HF".
        "reactants": ["HF", "F"],
        "products": ["H", "F2"],
        "coeffs": [-1.0, -1.0, +1.0, +1.0],
        # Forward barrier of HF+F→H+F2 = REVERSE barrier of NHTBH38
        # entry #5 (H+F2 → HF+F, Vf=2.27, Vr=105.80 kcal/mol REF1).
        "barrier_ref": 105.80,  # kcal/mol — forward barrier (Vr of NHTBH38 #5)
        # Reaction energy ΔE of HF+F → H+F2 (GMTKN55-BH76RC):
        #   ΔE = Vr − Vf = 105.80 − 2.27 = +103.53 kcal/mol.
        "reaction_energy_ref": 103.53,  # kcal/mol
        # Backward-compat alias (equals barrier_ref).
        "e_rxn_ref": 105.80,  # kcal/mol
        # Optional TS geometry for bh76_mode="barrier_height" (not staged).
        "ts_species": None,
        "species_spins":   {"HF": 0, "F": 1, "H": 1, "F2": 0},
        "species_charges": {"HF": 0, "F": 0, "H": 0, "F2": 0},
        "source": (
            "NHTBH38/08 entry 5 (H+F2 → HF+F), Vr (REF1) = 105.80 kcal/mol; "
            "Zheng, Zhao, Truhlar JCTC 5, 808 (2009); also GMTKN55-BH76."
        ),
        "spin_source": (
            "NIST ASD H I (²S, spin=1), F I (²P°, spin=1); "
            "NIST CCCBDB HF (X¹Σ+, 0), F2 (X¹Σg+, 0)."
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
#
# Atomic ground-state spins (PySCF 2S convention) from NIST ASD:
#   - Li I  (3 e⁻): ²S       → spin=1
#   - Li II (2 e⁻): ¹S       → spin=0
#   - C  I  (6 e⁻): ³P       → spin=2
#   - C  II (5 e⁻): ²P°      → spin=1
# Naming convention: the IP13 loss channel looks up neutral and cation
# total energies by separate MoleculeSpec.name keys (see
# xcquinox/alec/losses.py::L5GradnormVxcStep7._ip13_channel which reads
# ``E_nn[name_to_idx[neutral]]`` and ``E_nn[name_to_idx[cation]]``
# independently).  Therefore neutral and cation must be DIFFERENT
# MoleculeSpec.name strings.  The training-pool builder constructs the
# cation MoleculeSpec as a separate single-atom entry with a "+" suffix
# and charge=+1 / cation_spin.  The neutral keeps the bare element
# symbol ("Li", "C") which matches the bare-atom MoleculeSpec used as
# atom_energies anchor.
DFS_IP13_PAIRS = [
    {
        "name": "Li_IP",
        "neutral": "Li",      # MoleculeSpec.name (Hill formula, charge=0)
        "cation":  "Li+",     # MoleculeSpec.name (charge=+1, separate spec)
        "neutral_spin": 1,    # NIST ASD Li I (²S)
        "cation_spin":  0,    # NIST ASD Li II (¹S)
        "neutral_charge": 0,
        "cation_charge": 1,
        # 5.391719 eV * 23.0605 kcal/(mol*eV) = 124.335736 kcal/mol
        "ip_ref": 124.335736,  # kcal/mol
        "source": (
            "NIST atomic spectra database, Li I IE_1 = 5.391719 eV "
            "(43487.150 cm^-1); converted via 1 eV = 23.0605 kcal/mol (CODATA)."
        ),
        "spin_source": (
            "NIST ASD Li I (²S, spin=1) ground state; "
            "Li II (¹S, spin=0) ground state."
        ),
    },
    {
        "name": "C_IP",
        "neutral": "C",       # MoleculeSpec.name (Hill formula, charge=0)
        "cation":  "C+",      # MoleculeSpec.name (charge=+1, separate spec)
        "neutral_spin": 2,    # NIST ASD C I (³P)
        "cation_spin":  1,    # NIST ASD C II (²P°)
        "neutral_charge": 0,
        "cation_charge": 1,
        # 11.26030 eV * 23.0605 kcal/(mol*eV) = 259.668148 kcal/mol
        "ip_ref": 259.668148,  # kcal/mol
        "source": (
            "NIST atomic spectra database, C I IE_1 = 11.26030 eV "
            "(90820.45 cm^-1); converted via 1 eV = 23.0605 kcal/mol (CODATA)."
        ),
        "spin_source": (
            "NIST ASD C I (³P, spin=2) ground state; "
            "C II (²P°, spin=1) ground state."
        ),
    },
]

# 2 atomic-density references: H (²S, spin=1), Li (²S, spin=1).
# Each entry now carries explicit ground-state spin/charge metadata
# (NIST ASD ground-state term symbols) so build_dfs_pool() can attach
# at.info["spin"]/["charge"] for every Atoms returned.
DFS_ATOM_REFS = [
    {"sym": "H",  "spin": 1, "charge": 0,
     "spin_source": "NIST ASD H I (²S) ground state"},
    {"sym": "Li", "spin": 1, "charge": 0,
     "spin_source": "NIST ASD Li I (²S) ground state"},
]


# Atomic ground-state spins (PySCF 2S = N_α − N_β convention) for every
# element that appears in the DFS training pool, BH76 reactions, IP13
# pairs, AND the held-out probe sets.  Sources:
#   - NIST Atomic Spectra Database, ground-state term symbols
#     (https://physics.nist.gov/PhysRefData/ASD/levels_form.html)
#   - Hund's rule for half-filled subshells (e.g., N ⁴S°, P ⁴S°)
# Used by the step-7 notebook subset-generation cell to attach
# at.info["spin"] when constructing on-the-fly atom MoleculeSpecs for
# elements not in DFS_ATOM_REFS (which only covers H and Li).  Without
# this lookup, atomic Atoms enter SCF with spin=0 and PySCF rejects with
# "Electron number N and spin 0 are not consistent" for any open-shell
# element (C, N, O, F, P, S, Cl, ...).
ATOMIC_GROUND_STATE_SPIN: dict[str, int] = {
    "H":  1,   # ²S    NIST ASD H I
    "He": 0,   # ¹S    NIST ASD He I
    "Li": 1,   # ²S    NIST ASD Li I
    "Be": 0,   # ¹S    NIST ASD Be I
    "B":  1,   # ²P°   NIST ASD B I
    "C":  2,   # ³P    NIST ASD C I
    "N":  3,   # ⁴S°   NIST ASD N I (Hund: half-filled 2p³)
    "O":  2,   # ³P    NIST ASD O I
    "F":  1,   # ²P°   NIST ASD F I
    "Ne": 0,   # ¹S    NIST ASD Ne I
    "Na": 1,   # ²S    NIST ASD Na I
    "Mg": 0,   # ¹S    NIST ASD Mg I
    "Al": 1,   # ²P°   NIST ASD Al I
    "Si": 2,   # ³P    NIST ASD Si I
    "P":  3,   # ⁴S°   NIST ASD P I (Hund: half-filled 3p³)
    "S":  2,   # ³P    NIST ASD S I
    "Cl": 1,   # ²P°   NIST ASD Cl I
    "Ar": 0,   # ¹S    NIST ASD Ar I
}


def make_atom_atoms(sym: str, *, charge: int = 0, spin: int | None = None) -> Atoms:
    """Build an ASE Atoms object for a single atom at the origin with
    the correct ground-state spin attached to ``info``.

    If ``spin`` is None, looks up the neutral atom's NIST ASD ground-state
    2S from ATOMIC_GROUND_STATE_SPIN. Caller must pass an explicit spin
    for cations or non-neutral charges (which have a different occupation).

    Returned Atoms has ``info`` populated with ``spin``, ``charge``, and
    ``name`` so downstream PySCF builders can read them verbatim.
    """
    if spin is None:
        try:
            spin = ATOMIC_GROUND_STATE_SPIN[sym]
        except KeyError:
            raise KeyError(
                f"Atomic ground-state spin for {sym!r} is not in "
                f"ATOMIC_GROUND_STATE_SPIN. Add it (with NIST ASD "
                f"citation) before constructing this atom."
            )
    a = Atoms(sym, positions=[(0.0, 0.0, 0.0)])
    a.info["spin"] = int(spin)
    a.info["charge"] = int(charge)
    a.info["name"] = sym if charge == 0 else f"{sym}{'+' * charge if charge > 0 else '-' * (-charge)}"
    return a


def _g297_traj_path() -> Path:
    """Authoritative G2/97 ASE-trajectory file."""
    return Path(__file__).resolve().parents[2] / "scripts" / "script_data" / \
        "haunschild_g2" / "g2_97.traj"


def build_dfs_pool() -> dict:
    """Assemble the 28-entry Dick 2021 training pool.

    Returns dict with keys:
      ae_molecules       : 21 ASE Atoms (the AE-residual targets;
                            this is the SELECTION POOL for select_subset).
                           Each Atoms carries the following info-dict
                           entries used by step-7's training driver:
                             - "dfs_hill"   : Hill formula key (str)
                             - "ae_kcalmol"  : AE reference (float, kcal/mol)
                             - "ae_source"   : citation string (str)
                             - "ae_name"     : human-readable name (str)
                             - "spin"        : ground-state 2S (PySCF convention)
                             - "charge"      : net charge (default 0)
                             - "spin_source" : citation for spin assignment
                          See module-level DFS_AE_DATA for sources.
      bh76_reactions     : 3 reaction-spec dicts (each carries
                           ``species_spins`` / ``species_charges`` maps)
      ip13_pairs         : 2 IP-spec dicts (each with neutral/cation
                           spin + charge)
      atom_refs          : 2 ASE Atoms (H, Li) carrying at.info["spin"]
                           and ["charge"]
      n_total            : 28
    """
    traj_path = _g297_traj_path()
    traj = read(str(traj_path), ":")

    by_hill: dict = {a.get_chemical_formula(): a for a in traj}

    ae_atoms: list = []
    missing: list = []
    for entry in DFS_AE_DATA:
        hill = entry["hill"]
        if hill in by_hill:
            a = by_hill[hill].copy()
            a.info["dfs_hill"] = hill
            # Attach the published AE reference (kcal/mol) plus the
            # human-readable name and source citation.  Step-7's loss
            # driver reads `at.info["ae_kcalmol"]`; the source string is
            # for downstream provenance / sanity-check tests.
            a.info["ae_kcalmol"] = float(entry["ae_kcalmol"])
            a.info["ae_source"] = entry["source"]
            a.info["ae_name"] = entry["name"]
            # Spin / charge metadata (PySCF convention: spin = 2S = N_α − N_β).
            # Required by every SCF call downstream (see
            # subset_selection._ase_atoms_to_pyscf_mol which reads
            # at.info["spin"] / ["charge"] verbatim).
            a.info["spin"] = int(entry["spin"])
            a.info["charge"] = int(entry.get("charge", 0))
            a.info["spin_source"] = entry.get("spin_source", "")
            ae_atoms.append(a)
        else:
            missing.append(hill)
    if missing:
        raise RuntimeError(
            f"Dick AE pool: {len(missing)} formulas missing from g2_97.traj: {missing}.\n"
            f"Available Hill formulas in g2_97.traj: {sorted(by_hill.keys())}"
        )

    atom_refs: list = []
    for ref in DFS_ATOM_REFS:
        sym = ref["sym"]
        if sym in by_hill:
            a = by_hill[sym].copy()
        else:
            a = Atoms(sym, positions=[(0.0, 0.0, 0.0)])
        # Always set explicit spin/charge from the canonical ref entry
        # (NIST ASD ground-state).  Even when g2_97.traj happens to carry
        # an Atoms for the bare atom, it has no spin info, so we set it
        # here unconditionally.
        a.info["spin"] = int(ref["spin"])
        a.info["charge"] = int(ref.get("charge", 0))
        a.info["spin_source"] = ref.get("spin_source", "")
        a.info["name"] = sym
        atom_refs.append(a)

    return {
        "ae_molecules": ae_atoms,
        "bh76_reactions": DFS_BH76_REACTIONS,
        "ip13_pairs": DFS_IP13_PAIRS,
        "atom_refs": atom_refs,
        "n_total": (
            len(ae_atoms) + len(DFS_BH76_REACTIONS)
            + len(DFS_IP13_PAIRS) + len(atom_refs)
        ),
    }
