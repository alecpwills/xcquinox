"""Unit tests for xcquinox.pipeline.subset_selection."""
from __future__ import annotations


import numpy as np
import pytest

from xcquinox.pipeline import subset_selection as ss


def test_compute_descriptor_triple_uniform_gas_returns_alpha_one():
    """For a uniform electron gas: ∇ρ = 0 -> τ_W = 0; τ = τ_unif by construction -> α = 1.

    Reference: Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015), eq. (5).
    """
    n_grid = 10**3
    rho = np.full(n_grid, 0.1)
    sigma = np.zeros(n_grid)  # |∇ρ|² = 0 (uniform)
    tau_unif = (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    tau = tau_unif.copy()
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    assert desc["rho_third"].shape == (n_grid,)
    assert desc["s"].shape == (n_grid,)
    assert desc["alpha"].shape == (n_grid,)
    assert np.allclose(desc["alpha"], 1.0, atol=1e-6), \
        f"α for uniform gas should be 1.0; got {desc['alpha'][:5]}"
    assert np.allclose(desc["s"], 0.0, atol=1e-12)


def test_compute_descriptor_triple_iso_orbital_returns_alpha_zero():
    """For a single-orbital iso-orbital region, τ = τ_W -> α = 0."""
    n_grid = 100
    rho = np.linspace(0.05, 0.5, n_grid)
    sigma = np.linspace(0.001, 0.01, n_grid)  # |∇ρ|²
    tau_W = sigma / (8.0 * rho)
    tau = tau_W.copy()  # τ = τ_W -> iso-orbital
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    assert np.allclose(desc["alpha"], 0.0, atol=1e-12), \
        f"α should be 0 in iso-orbital region; got max |α|={np.abs(desc['alpha']).max()}"


def test_compute_descriptor_triple_s_formula_matches_pbe1996():
    """s = |∇ρ| / [2 (3π²)^{1/3} ρ^{4/3}], PBE 1996 eq. block before eq. (12)."""
    rho = np.array([0.5, 1.0, 2.0])
    sigma = np.array([1.0, 4.0, 9.0])  # |∇ρ|² -> |∇ρ| = sqrt(σ) = [1, 2, 3]
    tau = np.zeros_like(rho)  # don't care about α here
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    grad_rho = np.sqrt(sigma)
    expected_s = grad_rho / (2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0) * rho ** (4.0 / 3.0))
    np.testing.assert_allclose(desc["s"], expected_s, rtol=1e-12)


def _mock_three_histograms(seed=0):
    rng = np.random.default_rng(seed)
    h1 = rng.uniform(size=ss.NBINS)
    h2 = rng.uniform(size=ss.NBINS)
    h3 = rng.uniform(size=ss.NBINS)
    return {
        "rho_third": h1 / h1.sum(),
        "s": h2 / h2.sum(),
        "alpha": h3 / h3.sum(),
    }


def test_metric_l2_self_zero():
    h = _mock_three_histograms(seed=42)
    assert ss.metric_l2(h, h) == pytest.approx(0.0, abs=1e-15)


def test_metric_jsd_self_zero():
    h = _mock_three_histograms(seed=42)
    assert ss.metric_jsd(h, h) == pytest.approx(0.0, abs=1e-12)


def test_metric_jsd_symmetric():
    p = _mock_three_histograms(seed=1)
    q = _mock_three_histograms(seed=2)
    assert ss.metric_jsd(p, q) == pytest.approx(ss.metric_jsd(q, p), abs=1e-15)


def test_metric_jsd_uses_natural_log():
    """JSD with natural log: max value per marginal is ln(2)."""
    p = {"rho_third": np.zeros(ss.NBINS), "s": np.zeros(ss.NBINS), "alpha": np.zeros(ss.NBINS)}
    q = {"rho_third": np.zeros(ss.NBINS), "s": np.zeros(ss.NBINS), "alpha": np.zeros(ss.NBINS)}
    p["rho_third"][0] = 1.0
    q["rho_third"][1] = 1.0
    p["s"][0] = 1.0
    q["s"][1] = 1.0
    p["alpha"][0] = 1.0
    q["alpha"][1] = 1.0
    err = ss.metric_jsd(p, q)
    assert err == pytest.approx(3.0 * np.log(2.0), rel=1e-6)


# ----------------------------------------------------------------------
# DEFECT SUBSET-01: JSD must operate on PMFs (sum=1), be bounded by ln 2
# per marginal, and not upper-clip legitimate density peaks > 1.
# DEFECT SUBSET-05: an empty-in-range candidate must be maximally
# divergent (never selected), not a moderate ~0.5*ln2 score.
# ----------------------------------------------------------------------


def test_metric_jsd_disjoint_support_hits_ln2_bound_per_marginal():
    """Two distributions with disjoint support have maximal JSD = ln 2 per
    marginal (Lin 1991). Feed UNNORMALIZED density-like spikes (mass != 1,
    peak value > 1) on disjoint bins and check the total saturates at
    3*ln2: proving internal PMF normalization AND the ln2 bound."""
    p = {k: np.zeros(ss.NBINS) for k in ("rho_third", "s", "alpha")}
    q = {k: np.zeros(ss.NBINS) for k in ("rho_third", "s", "alpha")}
    for k in ("rho_third", "s", "alpha"):
        # Unnormalized density-like mass: two bins each, peaks well above 1,
        # disjoint support between p and q.
        p[k][0] = 7.0
        p[k][1] = 3.0
        q[k][2] = 11.0
        q[k][3] = 2.0
    err = ss.metric_jsd(p, q)
    assert err == pytest.approx(3.0 * np.log(2.0), rel=1e-9)


def test_metric_jsd_normalizes_density_input_matches_pmf_input():
    """Scaling a histogram by a constant (e.g. a different bin width) must
    NOT change the JSD, because each input is normalized to a PMF
    internally. The pre-fix code consumed densities directly, so a uniform
    rescale of one input changed the result."""
    rng = np.random.default_rng(31)
    p = {k: rng.uniform(0.1, 1.0, size=ss.NBINS) for k in ("rho_third", "s", "alpha")}
    q = {k: rng.uniform(0.1, 1.0, size=ss.NBINS) for k in ("rho_third", "s", "alpha")}
    base = ss.metric_jsd(p, q)
    # Rescale every input by an arbitrary positive constant: PMF identical.
    p_scaled = {k: 13.7 * v for k, v in p.items()}
    q_scaled = {k: 0.021 * v for k, v in q.items()}
    scaled = ss.metric_jsd(p_scaled, q_scaled)
    assert scaled == pytest.approx(base, rel=1e-9)


def _toy_descriptor_arrays(seed):
    rng = np.random.default_rng(seed)
    n = 5000
    return {
        "rho_third": np.abs(rng.normal(loc=0.5, scale=0.2, size=n)) + 1e-6,
        "s": np.abs(rng.normal(loc=1.0, scale=0.5, size=n)) + 1e-6,
        "alpha": np.abs(rng.normal(loc=1.0, scale=0.3, size=n)) + 1e-6,
        "weights": np.ones(n) / n,
    }


def test_bin_descriptors_returns_three_normalized_marginals():
    arrs = _toy_descriptor_arrays(seed=0)
    hist = ss.bin_descriptors(arrs)
    for k in ("rho_third", "s", "alpha"):
        assert hist[k].shape == (ss.NBINS,)
        assert hist[k].min() >= 0.0
        assert hist[k].sum() > 0.0


def _build_toy_pool(npool=8, seed=0):
    """Build a synthetic 8-entry pool with consistent log10 edges."""
    pool = [_toy_descriptor_arrays(seed=seed + i) for i in range(npool)]
    h_ref, edges = ss.build_reference_histograms(pool)
    return pool, h_ref, edges


def test_select_subset_recovers_pool_when_r_eq_n_jsd():
    pool, h_ref, edges = _build_toy_pool(npool=5)
    chosen, val = ss.select_subset(pool, edges, h_ref, r=5, metric="jsd")
    assert sorted(chosen) == [0, 1, 2, 3, 4]
    # JSD is roughly quadratic near zero, so its roundoff is bounded by
    # the square of the L2 case's roundoff; 1e-15 in practice.
    assert val == pytest.approx(0.0, abs=1e-12)


def test_select_subset_exhaustive_for_small_r():
    pool, h_ref, edges = _build_toy_pool(npool=6)
    chosen, val = ss.select_subset(pool, edges, h_ref, r=2, metric="l2")
    assert len(chosen) == 2
    from itertools import combinations as _C
    best_val, best_pair = float("inf"), None
    for pair in _C(range(6), 2):
        cat = {k: np.concatenate([pool[i][k] for i in pair]) for k in ss._DESCRIPTOR_KEYS}
        cat["weights"] = np.concatenate([pool[i].get("weights", np.ones_like(pool[i]["rho_third"])) for i in pair])
        h_cand = ss._bin_with_edges(cat, edges)
        v = ss.metric_l2(h_ref, h_cand)
        if v < best_val:
            best_val, best_pair = v, pair
    assert sorted(chosen) == sorted(best_pair)
    # Slow path bins concatenated arrays; fast path bins per pool entry then
    # sums.  Equivalent up to summation-order roundoff (~NBINS * float64-eps).
    assert val == pytest.approx(best_val, rel=1e-9, abs=1e-12)


def test_select_subset_fast_matches_slow_per_combo():
    """Pin the prebin-then-batch fast path against an explicit slow
    concatenate-then-bin recompute for EVERY combo on a moderately
    large toy pool (C(10, 4) = 210 combos).  Catches regressions where
    the in-range-weight normalization or batching logic drift away from
    the original ``_bin_with_edges`` semantics."""
    from itertools import combinations as _C
    pool, h_ref, edges = _build_toy_pool(npool=10, seed=42)
    # Fast path with return_all=True so we can compare every combo's value.
    _, _, vals_fast, idx_fast = ss.select_subset(
        pool, edges, h_ref, r=4, metric="l2",
        return_all=True, progress=False,
    )
    # Slow path: explicit concatenate-then-bin, in iteration order.
    vals_slow = np.empty_like(vals_fast)
    idx_slow = np.empty_like(idx_fast)
    for k, combo in enumerate(_C(range(10), 4)):
        cat = {key: np.concatenate([pool[i][key] for i in combo])
               for key in ss._DESCRIPTOR_KEYS}
        cat["weights"] = np.concatenate(
            [pool[i].get("weights", np.ones_like(pool[i]["rho_third"]))
             for i in combo]
        )
        h_cand = ss._bin_with_edges(cat, edges)
        vals_slow[k] = ss.metric_l2(h_ref, h_cand)
        idx_slow[k, :] = combo
    np.testing.assert_array_equal(idx_fast, idx_slow)
    # Element-wise comparison: fast and slow paths agree up to summation-
    # order float roundoff bounded by ~NBINS * float64-eps.
    np.testing.assert_allclose(vals_fast, vals_slow, rtol=1e-9, atol=1e-12)


def test_compute_atom_set_for_simple_subset():
    from ase import Atoms
    a1 = Atoms("H2O", positions=[(0,0,0),(1,0,0),(0,1,0)])
    a2 = Atoms("LiF", positions=[(0,0,0),(1,0,0)])
    atom_set = ss.compute_atom_set([a1, a2])
    assert atom_set == {"H", "O", "Li", "F"}


def test_augment_with_hbpt_water_adds_two_entries():
    from ase import Atoms
    a = Atoms("H2O", positions=[(0,0,0),(1,0,0),(0,1,0)])
    refs = [Atoms("H", positions=[(0,0,0)]), Atoms("O", positions=[(0,0,0)])]
    out = ss.augment_with_hbpt([a], refs, with_hbpt=True)
    assert len(out) == 5
    names = [at.info.get("name") for at in out]
    assert names.count("HBWD") == 1
    assert names.count("PTWD") == 1


def test_extract_descriptors_caches_to_disk(tmp_path):
    """Second call for the same species hits the cache (no SCF re-run)."""
    from ase import Atoms
    a = Atoms("H2", positions=[(0,0,0),(0.74,0,0)])
    a.info["species"] = "H2"
    cache_dir = tmp_path / "subset_descriptors"
    arrs1 = ss.extract_descriptors(a, idx=0, cache_dir=cache_dir)
    assert (cache_dir / "0_H2.npz").exists()
    arrs2 = ss.extract_descriptors(a, idx=0, cache_dir=cache_dir)
    for k in ("rho_third", "s", "alpha", "weights"):
        np.testing.assert_array_equal(arrs1[k], arrs2[k])


def test_dfs_pool_has_28_distinct_training_points():
    """Per Dick 2021 SI §II: 21 AE + 3 BH76 + 2 IP13 + 2 atom = 28."""
    from xcquinox.pipeline.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    assert pool["n_total"] == 28
    assert len(pool["ae_molecules"]) == 21
    assert len(pool["bh76_reactions"]) == 3
    assert len(pool["ip13_pairs"]) == 2
    assert len(pool["atom_refs"]) == 2


def test_dfs_pool_ae_molecule_set_matches_si_section_ii():
    """Hill-formula equality with Dick SI §II text."""
    from xcquinox.pipeline.dfs_pool import build_dfs_pool, DFS_AE_HILL
    pool = build_dfs_pool()
    found = {a.get_chemical_formula() for a in pool["ae_molecules"]}
    assert found == set(DFS_AE_HILL)


def test_dfs_ip13_li_matches_nist():
    """Li IE_1 = 5.391719 eV from NIST -> 124.336 kcal/mol."""
    from xcquinox.pipeline.dfs_pool import DFS_IP13_PAIRS
    li = next(p for p in DFS_IP13_PAIRS if p["name"] == "Li_IP")
    expected = 5.391719 * 23.0605  # NIST eV × CODATA conversion
    assert li["ip_ref"] == pytest.approx(expected, abs=0.01)


def test_dfs_bh76_oh_n2_to_h_n2o_value():
    """OH+N2 -> H+N2O: forward barrier 82.6 (GMTKN55 BH76/.res 'oh n2
    n2ohts'; Minnesota REF1 provenance 82.27), reaction energy
    ΔE = +64.91 kcal/mol (GMTKN55-BH76RC W2-F12)."""
    from xcquinox.pipeline.dfs_pool import DFS_BH76_REACTIONS
    rxn = next(r for r in DFS_BH76_REACTIONS if r["name"] == "OH+N2_to_H+N2O")
    assert rxn["barrier_ref"] == pytest.approx(82.6, abs=0.01)
    assert rxn["reaction_energy_ref"] == pytest.approx(64.91, abs=0.01)


# ----------------------------------------------------------------------
# DFS_AE_DATA / build_dfs_pool() AE-reference attachment tests
# ----------------------------------------------------------------------


def test_dfs_pool_ae_anchor_w411_provenance():
    """H2O and C2H2 carry the W4-11 geometry+AE (a documented deviation from
    the 19 G2/97 + Haunschild anchors). Their AE reference is the GMTKN55-W4-11
    zero-point-exclusive nonrelativistic atomization energy (Karton, Daon &
    Martin, Chem. Phys. Lett. 510, 165 (2011) = DFS ref [29]). Verify the
    dfs_pool anchor against the W4-11 pool file directly -- an INDEPENDENT
    source (GMTKN55-W4-11/.res), not the step-6 notebook the value was
    previously only cross-checked against, so the provenance is no longer
    circular."""
    import json
    from pathlib import Path
    from xcquinox.pipeline.dfs_pool import build_dfs_pool

    w411 = json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "w411_full_pool.json")
        .read_text()
    )

    def w411_atomization_ref(mol):
        hits = [r for r in w411["reactions"] if r.get("reactants") == [mol]]
        assert len(hits) == 1, f"expected one W4-11 atomization for {mol!r}"
        return hits[0]["reaction_energy_ref"]

    pool = build_dfs_pool()
    by_hill = {a.info["dfs_hill"]: a for a in pool["ae_molecules"]}
    assert by_hill["H2O"].info["ae_kcalmol"] == pytest.approx(
        w411_atomization_ref("h2o"), abs=1e-3)
    assert by_hill["C2H2"].info["ae_kcalmol"] == pytest.approx(
        w411_atomization_ref("c2h2"), abs=1e-3)


# ----------------------------------------------------------------------
# Spin / charge metadata invariants (2026-05-01 NO-spin-bug fix)
# ----------------------------------------------------------------------
#
# Background: a step-7 smoke run (2026-05-01) failed on entry #10 (NO,
# 15 electrons) with PySCF "Electron number 15 and spin 0 are not
# consistent".  Root cause: ASE Atoms loaded from g2_97.traj have no
# spin/charge in info{}, so _ase_atoms_to_pyscf_mol defaulted spin=0 for
# every species, wrong for the 7 open-shell molecules in the AE pool
# (NO, CH, OH, NO2, NH, CH3, CH2-triplet) and the atomic refs (H, Li).
#
# These tests enforce the (nelec - spin) % 2 == 0 invariant PySCF
# requires for every Atoms returned by build_dfs_pool().

_HILL_TO_NELEC = {
    # Atomic numbers used to compute electron counts.  Only the elements
    # that appear in the Dick pool need to be listed here.
    "H": 1, "Li": 3, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11,
}


def _atoms_nelec(at):
    """Compute total electron count from chemical symbols + at.info['charge']."""
    n = sum(_HILL_TO_NELEC[s] for s in at.get_chemical_symbols())
    return n - int(at.info.get("charge", 0))


def test_dfs_ae_data_open_shell_spins_match_published_ground_states():
    """Spot-check the published ground-state spins for the 7 open-shell
    AE molecules + the special triplet-singlet cases (NH, CH2)."""
    from xcquinox.pipeline.dfs_pool import DFS_AE_SPIN
    expected = {
        "NO":  1,  # X²Π doublet
        "CH":  1,  # X²Π doublet
        "HO":  1,  # X²Π doublet
        "NO2": 1,  # X²A1 doublet
        "HN":  2,  # X³Σ⁻ TRIPLET (Herzberg I §VI; load-bearing)
        "CH3": 1,  # X²A2'' doublet
        "CH2": 2,  # X³B1 TRIPLET (Bunker & Sears 1985; load-bearing)
        "O3":  0,  # X¹A1 closed-shell singlet (despite multireference)
        "H2":  0, "N2": 0, "FLi": 0, "CHN": 0, "CO2": 0, "F2": 0,
        "C2H2": 0, "CO": 0, "HLi": 0, "Na2": 0, "N2O": 0, "H2O": 0,
        "H3N": 0,
    }
    for hill, exp_spin in expected.items():
        assert DFS_AE_SPIN[hill] == exp_spin, (
            f"{hill}: spin mismatch (got {DFS_AE_SPIN[hill]}, "
            f"expected {exp_spin})")


def test_dfs_pool_every_ae_atoms_satisfies_pyscf_spin_invariant():
    """Every Atoms in pool['ae_molecules'] must satisfy
    (nelec - spin) % 2 == 0, the invariant PySCF enforces.  This is
    the regression test for the 2026-05-01 NO smoke-run failure."""
    from xcquinox.pipeline.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    for at in pool["ae_molecules"]:
        nelec = _atoms_nelec(at)
        spin = int(at.info["spin"])
        assert (nelec - spin) % 2 == 0, (
            f"{at.info['dfs_hill']}: nelec={nelec}, spin={spin}, "
            f"(nelec - spin) is odd; PySCF will reject this SCF.")


def test_select_subset_return_all_returns_full_distribution():
    """return_all=True returns vals array with C(n, r) entries."""
    import math
    import numpy as np
    from xcquinox.pipeline.subset_selection import (
        build_reference_histograms, select_subset,
    )
    rng = np.random.default_rng(0)
    pool = []
    for _ in range(6):
        pool.append({
            "rho_third": rng.uniform(0.1, 1.0, size=(50,)),
            "s": rng.uniform(0.0, 2.0, size=(50,)),
            "alpha": rng.uniform(0.0, 5.0, size=(50,)),
            "weights": np.ones(50),
        })
    h_ref, edges = build_reference_histograms(pool)
    chosen, best_val, vals, idx_array = select_subset(
        pool, edges, h_ref, r=3, metric="l2",
        progress=False, return_all=True,
    )
    n_combos = math.comb(6, 3)
    assert vals.shape == (n_combos,)
    assert idx_array.shape == (n_combos, 3)
    assert vals.min() == best_val
    assert vals.dtype == np.float64


# per-descriptor selection weights (default equal, opt-in down-weighting)


