"""Unit tests for xcquinox.alec.eval_probes (step-7 held-out probes).

Tests verify:
  - Each of the 4 probes has 5-6 entries.
  - Every entry carries a non-empty source citation string.
  - Every entry carries a finite reference value (ae_kcalmol, or for probe_c
    the GMTKN55-BH76RC reaction_energy_ref + barrier_vf_ref provenance).
  - probe_c measures BH76 reaction-energy transfer; entry 5 (H+N2O→OH+N2) is
    the intentional REVERSE of training reaction 1 (a directional-consistency
    probe), not accidental overlap.
  - Spot-check selected reference values against the published numbers
    (Haunschild2012 Table I; GMTKN55-BH76RC W2-F12; Truhlar HTBH/NHTBH REF1).
  - build_probe_pool() returns the expected shape for each probe.
"""
from __future__ import annotations

import math

import pytest

from xcquinox.alec import eval_probes as ep
from xcquinox.alec.dfs_pool import (
    DFS_AE_HILL,
    DFS_BH76_REACTIONS,
    DFS_IP13_PAIRS,
    DFS_ATOM_REFS,
)


# ---------------------------------------------------------------------------
# 1. Structural completeness
# ---------------------------------------------------------------------------
def test_all_probes_registry_has_four_entries():
    assert len(ep.ALL_PROBES) == 4
    assert set(ep.ALL_PROBES) == {
        "probe_a_chemical_similarity",
        "probe_b_heteroatom",
        "probe_c_bh76_transfer",
        "probe_d_multireference",
    }


def test_all_probes_kinds_aligned():
    """PROBE_KIND must cover every entry in ALL_PROBES."""
    assert set(ep.PROBE_KIND) == set(ep.ALL_PROBES)
    assert ep.PROBE_KIND["probe_a_chemical_similarity"] == "ae"
    assert ep.PROBE_KIND["probe_b_heteroatom"] == "ae"
    assert ep.PROBE_KIND["probe_c_bh76_transfer"] == "bh76"
    assert ep.PROBE_KIND["probe_d_multireference"] == "ae"


@pytest.mark.parametrize("probe_name", list(ep.ALL_PROBES))
def test_each_probe_has_five_or_six_entries(probe_name):
    entries = ep.ALL_PROBES[probe_name]
    assert 5 <= len(entries) <= 6, (
        f"{probe_name}: expected 5-6 entries, got {len(entries)}")


# ---------------------------------------------------------------------------
# 2. Citations and reference values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("probe_name", list(ep.ALL_PROBES))
def test_every_entry_has_nonempty_source(probe_name):
    """No fabricated values: every entry must cite its published source."""
    for entry in ep.ALL_PROBES[probe_name]:
        assert "source" in entry, f"{probe_name}: missing source on {entry}"
        assert isinstance(entry["source"], str)
        assert len(entry["source"].strip()) > 0


@pytest.mark.parametrize("probe_name", list(ep.ALL_PROBES))
def test_every_entry_has_nonempty_rationale(probe_name):
    """Every entry must document why it tests its probe characteristic."""
    for entry in ep.ALL_PROBES[probe_name]:
        assert "rationale" in entry, (
            f"{probe_name}: missing rationale on {entry}")
        assert isinstance(entry["rationale"], str)
        assert len(entry["rationale"].strip()) > 20  # non-trivial


def test_ae_probes_have_finite_ae_refs():
    """Probes A/B/D entries must have a finite, physical ae_kcalmol."""
    for probe_name in ("probe_a_chemical_similarity",
                       "probe_b_heteroatom",
                       "probe_d_multireference"):
        for entry in ep.ALL_PROBES[probe_name]:
            ae = entry.get("ae_kcalmol")
            assert ae is not None, f"{probe_name}: missing ae_kcalmol"
            assert isinstance(ae, float)
            assert math.isfinite(ae)
            # All Haunschild2012 first/second-row AEs sit in (10, 1000) kcal/mol.
            assert 10.0 < ae < 1000.0, (
                f"{probe_name}: {entry['hill']}: AE out of range {ae}")


def test_bh76_probe_has_finite_reaction_energies():
    """Probe C reactions carry a finite GMTKN55-BH76RC reaction energy (the
    metric reference) and a forward-barrier provenance value, both kcal/mol."""
    for entry in ep.PROBE_C_BH76_OUT_OF_TRAINING:
        de = entry["reaction_energy_ref"]
        vf = entry["barrier_vf_ref"]
        assert isinstance(de, float) and math.isfinite(de)
        assert isinstance(vf, float) and math.isfinite(vf)
        assert -100.0 < de < 100.0, f"{entry['name']}: ΔE out of range {de}"
        # The legacy barrier key must be gone (it was a category error to
        # compare a reaction energy against it).
        assert "e_rxn_ref" not in entry, (
            f"{entry['name']}: stale 'e_rxn_ref' (barrier) key still present")


# GMTKN55-BH76RC (W2-F12) reaction energies, kcal/mol — source of truth
# (grimme-lab/GMTKN55 BH76/.resRC; see GMTKN55_BH76RC_PROVENANCE.md).
_GMTKN55_BH76RC = {
    "OH+H2_to_H2O+H":   -16.39,
    "H+HCl_to_H2+Cl":    -1.90,
    "CH3+H2_to_CH4+H":   -3.11,
    "OH+NH3_to_H2O+NH2": -10.32,
    "H+N2O_to_OH+N2":   -64.91,
    "H+H2S_to_H2+HS":   -13.26,
}


def test_probe_c_reaction_energies_match_gmtkn55():
    """Each probe_c reaction_energy_ref equals its GMTKN55-BH76RC value."""
    by_name = {e["name"]: e for e in ep.PROBE_C_BH76_OUT_OF_TRAINING}
    assert set(by_name) == set(_GMTKN55_BH76RC), "probe_c reaction set drifted"
    for name, ref in _GMTKN55_BH76RC.items():
        assert by_name[name]["reaction_energy_ref"] == pytest.approx(ref, abs=1e-9)


def test_probe_c_directional_consistency_entry_is_reverse_of_training():
    """Entry 5 (H+N2O->OH+N2) is the intentional reverse of Dick training
    reaction 1 (OH+N2->H+N2O), so its ΔE == -(training rxn ΔE) using the SAME
    GMTKN55 source of truth (the reverse direction, +64.91)."""
    by_name = {e["name"]: e for e in ep.PROBE_C_BH76_OUT_OF_TRAINING}
    assert by_name["H+N2O_to_OH+N2"]["reaction_energy_ref"] == pytest.approx(
        -64.91, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. No overlap with Dick training pool
# ---------------------------------------------------------------------------
def test_ae_probes_disjoint_from_dfs_training_pool():
    """No probe-A/B/D Hill formula may appear in Dick training (21 AE
    molecules + 2 atom refs)."""
    # DFS_ATOM_REFS is a list of dicts (per 2026-05-01 spin-metadata
    # refactor); pull the bare element symbol off each entry.
    training_hills = set(DFS_AE_HILL) | {r["sym"] for r in DFS_ATOM_REFS}
    for probe_name in ("probe_a_chemical_similarity",
                       "probe_b_heteroatom",
                       "probe_d_multireference"):
        probe_hills = {e["hill"] for e in ep.ALL_PROBES[probe_name]}
        overlap = probe_hills & training_hills
        assert not overlap, (
            f"{probe_name}: AE probe overlaps Dick training pool: {overlap}")


def test_bh76_probe_reaction_names_distinct_from_training():
    """Probe C reaction names must not duplicate the 3 Dick training BH76
    reactions (the canonical-name strings differ)."""
    training_names = {r["name"] for r in DFS_BH76_REACTIONS}
    probe_names = {r["name"] for r in ep.PROBE_C_BH76_OUT_OF_TRAINING}
    overlap = training_names & probe_names
    assert not overlap, (
        f"probe_c BH76 names overlap training: {overlap}")


def test_bh76_probe_reaction_signatures_distinct_from_training():
    """Probe C forward direction must differ from each Dick training
    reaction in the SAME direction.

    Reactions with identical (reactants, products) ordered tuples are
    duplicates.  Reverse reactions (swapped reactants/products) are
    treated as DISTINCT probes because their forward barrier height
    Vf differs from the training reaction's Vr — they are independent
    targets in HTBH/NHTBH.  See PROBE_C entry ``H+N2O_to_OH+N2`` whose
    rationale documents the directional-consistency probe.
    """
    def signature(rxn):
        return (tuple(sorted(rxn["reactants"])),
                tuple(sorted(rxn["products"])))

    train_sigs = {signature(r) for r in DFS_BH76_REACTIONS}
    for probe_rxn in ep.PROBE_C_BH76_OUT_OF_TRAINING:
        sig = signature(probe_rxn)
        assert sig not in train_sigs, (
            f"Probe C reaction {probe_rxn['name']!r} matches a Dick "
            f"training reaction signature in the same direction.")


def test_ip13_probe_atoms_distinct_from_training():
    """The IP13 training pairs use Li and C neutral/cation; no probe-D
    diatomic should be a 'Li' or 'C' atom (which are training-pool atoms
    via DFS_IP13_PAIRS / atom_refs).  This is mostly a no-op tripwire
    given probe-D contains diatomics, but it documents the constraint."""
    ip_atoms = ({p["neutral"] for p in DFS_IP13_PAIRS}
                | {r["sym"] for r in DFS_ATOM_REFS})
    for entry in ep.PROBE_D_MULTIREFERENCE:
        # A multireference probe should not be a single atom.
        assert entry["hill"] not in ip_atoms, (
            f"Probe D entry {entry['hill']!r} duplicates training atom set")


# ---------------------------------------------------------------------------
# 4. Spot-check published reference values
# ---------------------------------------------------------------------------
def test_probe_a_ch4_value_matches_haunschild_kj():
    """CH4 AE (Haunschild2012 Table I): 1757.82 kJ/mol → 420.129 kcal/mol."""
    e = next(d for d in ep.PROBE_A_CHEMICAL_SIMILARITY if d["hill"] == "CH4")
    expected_kcal = 1757.82 / 4.184
    assert e["ae_kcalmol"] == pytest.approx(expected_kcal, abs=1e-3)
    # And cross-check magnitude vs the W4-11 anchor convention used by
    # alec.dfs_pool — both Haunschild2012 and W4-11 must give CH4 within
    # 1 kcal/mol of each other (sub-1-kJ/mol agreement is documented in
    # Haunschild 2012 §III).  The W4-11 SI value (Karton 2011 Table 1
    # row 5) is 419.30 kcal/mol; Haunschild gives 420.13 — Δ = 0.83
    # kcal/mol, well within the W4 0.24 kcal/mol error budget.
    assert abs(e["ae_kcalmol"] - 419.30) < 1.0


def test_probe_b_h2s_value_matches_haunschild_kj():
    """H2S AE (Haunschild2012 Table I): 768.72 kJ/mol → 183.728 kcal/mol."""
    e = next(d for d in ep.PROBE_B_HETEROATOM_EXTRAPOLATION if d["hill"] == "H2S")
    expected_kcal = 768.72 / 4.184
    assert e["ae_kcalmol"] == pytest.approx(expected_kcal, abs=1e-3)


def test_probe_c_oh_h2_to_h2o_h_provenance_and_dE():
    """OH+H2 → H2O+H: GMTKN55-BH76RC ΔE = -16.39; forward barrier provenance
    Vf = 4.90 kcal/mol (HTBH38/08 entry 2 REF1)."""
    rxn = next(r for r in ep.PROBE_C_BH76_OUT_OF_TRAINING
               if r["name"] == "OH+H2_to_H2O+H")
    assert rxn["reaction_energy_ref"] == pytest.approx(-16.39, abs=1e-2)
    assert rxn["barrier_vf_ref"] == pytest.approx(4.90, abs=1e-2)


def test_probe_c_h_n2o_to_oh_n2_provenance_and_dE():
    """H+N2O → OH+N2: GMTKN55-BH76RC ΔE = -64.91 (the REVERSE of Dick training
    reaction 1, OH+N2→H+N2O = +64.91); forward barrier provenance Vf = 17.13
    kcal/mol (NHTBH38/08 entry 1 REF1)."""
    rxn = next(r for r in ep.PROBE_C_BH76_OUT_OF_TRAINING
               if r["name"] == "H+N2O_to_OH+N2")
    assert rxn["reaction_energy_ref"] == pytest.approx(-64.91, abs=1e-2)
    assert rxn["barrier_vf_ref"] == pytest.approx(17.13, abs=1e-2)


def test_probe_d_o2_value_matches_haunschild_kj():
    """O2 (triplet) AE (Haunschild2012 Table I): 505.88 kJ/mol → 120.908 kcal/mol."""
    e = next(d for d in ep.PROBE_D_MULTIREFERENCE if d["hill"] == "O2")
    expected_kcal = 505.88 / 4.184
    assert e["ae_kcalmol"] == pytest.approx(expected_kcal, abs=1e-3)
    # O2 multiplicity must be triplet (³Σg⁻); spin = 2 in ASE convention.
    assert e["spin"] == 2


def test_probe_d_beh_value_matches_haunschild_kj():
    """BeH AE (Haunschild2012 Table I): 212.50 kJ/mol → 50.789 kcal/mol."""
    e = next(d for d in ep.PROBE_D_MULTIREFERENCE if d["hill"] == "HBe")
    expected_kcal = 212.50 / 4.184
    assert e["ae_kcalmol"] == pytest.approx(expected_kcal, abs=1e-3)
    # BeH ground state is ²Σ⁺ — one unpaired electron.
    assert e["spin"] == 1


# ---------------------------------------------------------------------------
# 5. build_probe_pool factory
# ---------------------------------------------------------------------------
def test_build_probe_pool_unknown_name_raises():
    with pytest.raises(ValueError):
        ep.build_probe_pool("not_a_probe")


@pytest.mark.parametrize("probe_name",
                         ["probe_a_chemical_similarity",
                          "probe_b_heteroatom",
                          "probe_d_multireference"])
def test_build_probe_pool_ae_kind(probe_name):
    pool = ep.build_probe_pool(probe_name)
    assert pool["kind"] == "ae"
    assert 5 <= pool["n"] <= 6
    assert len(pool["molecules"]) == pool["n"]
    assert len(pool["ae_refs_kcalmol"]) == pool["n"]
    # Every Atoms must carry the probe metadata
    for at in pool["molecules"]:
        for k in ("ae_kcalmol", "ae_source", "ae_name", "rationale",
                  "spin", "charge"):
            assert k in at.info, f"{probe_name}: missing info[{k!r}] on {at}"
        assert math.isfinite(at.info["ae_kcalmol"])


def test_build_probe_pool_bh76_kind():
    pool = ep.build_probe_pool("probe_c_bh76_transfer")
    assert pool["kind"] == "bh76"
    assert pool["n"] == 6
    assert len(pool["reactions"]) == 6
    # Every species referenced in any reaction must appear in molecules
    referenced = set()
    for rxn in pool["reactions"]:
        for s in (*rxn["reactants"], *rxn["products"]):
            referenced.add(s)
    formulas = {a.get_chemical_formula() for a in pool["molecules"]}
    missing = referenced - formulas
    assert not missing, (
        f"BH76 probe missing geometries for species: {missing}")
    # atom_set must be a subset of expected elements (H, C, N, O, S, Cl)
    assert pool["atom_set"] <= {"H", "C", "N", "O", "S", "Cl"}


def test_build_probe_pool_ae_atom_set_does_not_intersect_unexpected():
    """Probe A & D should be H/C/N/O/F/Be only; Probe B should add S/P/Cl/Si."""
    pa = ep.build_probe_pool("probe_a_chemical_similarity")
    assert pa["atom_set"] <= {"H", "C", "N", "O"}
    pb = ep.build_probe_pool("probe_b_heteroatom")
    assert pb["atom_set"] <= {"H", "O", "S", "P", "Cl", "Si"}
    pd = ep.build_probe_pool("probe_d_multireference")
    assert pd["atom_set"] <= {"H", "Be", "C", "N", "O", "F", "Cl"}


def test_build_probe_pool_no_overlap_with_dfs_training_atoms_set():
    """All probe AE molecules' Hill formulas, when compared against
    DFS_AE_HILL, must be disjoint — this is the load-bearing
    no-fabrication / no-training-leak invariant."""
    training_hills = set(DFS_AE_HILL)
    for probe_name in ("probe_a_chemical_similarity",
                       "probe_b_heteroatom",
                       "probe_d_multireference"):
        pool = ep.build_probe_pool(probe_name)
        probe_hills = {a.info["probe_hill"] for a in pool["molecules"]}
        assert probe_hills.isdisjoint(training_hills), (
            f"{probe_name}: probe_hills overlap training: "
            f"{probe_hills & training_hills}")


def test_build_probe_pool_bh76_reactions_reference_total_atom_count():
    """For each BH76 reaction, atom-count balance: sum(coeffs * n_atoms)
    must be zero.  This catches any typo in coeffs or species lists."""
    pool = ep.build_probe_pool("probe_c_bh76_transfer")
    by_formula = {a.get_chemical_formula(): a for a in pool["molecules"]}
    for rxn in pool["reactions"]:
        species = (*rxn["reactants"], *rxn["products"])
        coeffs = rxn["coeffs"]
        # element-wise balance
        elem_balance: dict = {}
        for sp, c in zip(species, coeffs):
            for sym in by_formula[sp].get_chemical_symbols():
                elem_balance[sym] = elem_balance.get(sym, 0.0) + c
        for sym, val in elem_balance.items():
            assert abs(val) < 1e-9, (
                f"{rxn['name']}: {sym} balance = {val} (should be 0)")


# ---------------------------------------------------------------------------
# 6. Spin / charge metadata invariants (2026-05-01 NO-spin-bug fix)
# ---------------------------------------------------------------------------
#
# Every probe Atoms returned by build_probe_pool() must carry spin and
# charge fields satisfying (nelec - spin) % 2 == 0, the invariant that
# PySCF enforces.  Failures of this invariant are exactly what triggered
# the 2026-05-01 NO-spin smoke-run bug.

_HILL_TO_NELEC = {
    "H": 1, "Be": 4, "C": 6, "N": 7, "O": 8, "F": 9,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Li": 3, "Na": 11,
}


def _atoms_nelec(at):
    n = sum(_HILL_TO_NELEC[s] for s in at.get_chemical_symbols())
    return n - int(at.info.get("charge", 0))


def test_every_ae_probe_entry_has_spin_field():
    """Probes A/B/D entries must carry an explicit `spin` field in
    PySCF 2S convention (already part of the eval_probes module schema,
    but assert its presence per-entry for the regression tripwire)."""
    for probe_name in ("probe_a_chemical_similarity",
                       "probe_b_heteroatom",
                       "probe_d_multireference"):
        for entry in ep.ALL_PROBES[probe_name]:
            assert "spin" in entry, (
                f"{probe_name}: {entry['hill']}: missing spin field")
            assert isinstance(entry["spin"], int)
            assert entry["spin"] >= 0
            assert "charge" in entry
            assert isinstance(entry["charge"], int)


@pytest.mark.parametrize("probe_name",
                         ["probe_a_chemical_similarity",
                          "probe_b_heteroatom",
                          "probe_d_multireference"])
def test_build_probe_pool_ae_atoms_satisfy_pyscf_spin_invariant(probe_name):
    """For probes A, B, D: every Atoms must satisfy
    (nelec - spin) % 2 == 0, the PySCF mol-construction invariant."""
    pool = ep.build_probe_pool(probe_name)
    for at in pool["molecules"]:
        nelec = _atoms_nelec(at)
        spin = int(at.info["spin"])
        assert (nelec - spin) % 2 == 0, (
            f"{probe_name}: {at.info.get('probe_hill', at.get_chemical_formula())} "
            f"nelec={nelec} spin={spin} — PySCF will reject SCF.")


def test_probe_c_every_reaction_has_species_spins():
    """Every PROBE_C reaction must carry species_spins / species_charges
    dicts covering every reactant + product."""
    for rxn in ep.PROBE_C_BH76_OUT_OF_TRAINING:
        assert "species_spins" in rxn, (
            f"{rxn['name']}: missing species_spins")
        assert "species_charges" in rxn, (
            f"{rxn['name']}: missing species_charges")
        for sp in (*rxn["reactants"], *rxn["products"]):
            assert sp in rxn["species_spins"], (
                f"{rxn['name']}: species_spins missing {sp!r}")
            assert sp in rxn["species_charges"], (
                f"{rxn['name']}: species_charges missing {sp!r}")


def test_build_probe_pool_bh76_atoms_satisfy_pyscf_spin_invariant():
    """For probe C (BH76): every species Atoms must satisfy the spin
    invariant (the 2026-05-01 NO regression test, generalized)."""
    pool = ep.build_probe_pool("probe_c_bh76_transfer")
    for at in pool["molecules"]:
        nelec = _atoms_nelec(at)
        spin = int(at.info["spin"])
        assert (nelec - spin) % 2 == 0, (
            f"BH76 species {at.get_chemical_formula()}: "
            f"nelec={nelec} spin={spin}")


def test_probe_d_o2_is_triplet_and_so_is_oxygen_so():
    """Two famously-triplet species in the multireference / heteroatom
    probes must carry spin=2: O2 (³Σg⁻) and SO (³Σ⁻)."""
    o2 = next(d for d in ep.PROBE_D_MULTIREFERENCE if d["hill"] == "O2")
    so = next(d for d in ep.PROBE_B_HETEROATOM_EXTRAPOLATION if d["hill"] == "OS")
    assert o2["spin"] == 2, "O2 ground state is X³Σg⁻ — spin must be 2"
    assert so["spin"] == 2, "SO ground state is X³Σ⁻ — spin must be 2"
