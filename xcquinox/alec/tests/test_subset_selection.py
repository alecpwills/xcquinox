"""Unit tests for xcquinox.alec.subset_selection."""
from __future__ import annotations

import math

import numpy as np
import pytest

from xcquinox.alec import subset_selection as ss


def test_compute_descriptor_triple_uniform_gas_returns_alpha_one():
    """For a uniform electron gas: ∇ρ = 0 → τ_W = 0; τ = τ_unif by construction → α = 1.

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
    """For a single-orbital iso-orbital region, τ = τ_W → α = 0."""
    n_grid = 100
    rho = np.linspace(0.05, 0.5, n_grid)
    sigma = np.linspace(0.001, 0.01, n_grid)  # |∇ρ|²
    tau_W = sigma / (8.0 * rho)
    tau = tau_W.copy()  # τ = τ_W → iso-orbital
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    assert np.allclose(desc["alpha"], 0.0, atol=1e-12), \
        f"α should be 0 in iso-orbital region; got max |α|={np.abs(desc['alpha']).max()}"


def test_compute_descriptor_triple_s_formula_matches_pbe1996():
    """s = |∇ρ| / [2 (3π²)^{1/3} ρ^{4/3}], PBE 1996 eq. block before eq. (12)."""
    rho = np.array([0.5, 1.0, 2.0])
    sigma = np.array([1.0, 4.0, 9.0])  # |∇ρ|² → |∇ρ| = sqrt(σ) = [1, 2, 3]
    tau = np.zeros_like(rho)  # don't care about α here
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    grad_rho = np.sqrt(sigma)
    expected_s = grad_rho / (2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0) * rho ** (4.0 / 3.0))
    np.testing.assert_allclose(desc["s"], expected_s, rtol=1e-12)


def test_compute_descriptor_triple_no_negative_alpha_under_clip():
    """α can in principle be negative due to grid noise; values are clipped at 0
    (matches data_binning2.ipynb cell 17 implicit behavior of histogramming
    only positive values via log10)."""
    rho = np.full(10, 0.1)
    sigma = np.full(10, 1.0)  # |∇ρ|² = 1
    tau_W = sigma / (8.0 * rho)
    # τ < τ_W → α < 0; we expect the clip
    tau = 0.5 * tau_W
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    assert (desc["alpha"] >= 0.0).all(), "α must be clipped to non-negative"


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


def test_metric_l2_three_histogram_sum():
    """The L2 error sums sqrt over all 3 histograms per bin (matching
    data_binning2.ipynb cell 17). Constructed input where only h1 differs
    in a single bin should yield exactly the displacement magnitude."""
    p = _mock_three_histograms(seed=3)
    q = {k: v.copy() for k, v in p.items()}
    q["rho_third"] = q["rho_third"].copy()
    q["rho_third"][0] += 0.5
    err = ss.metric_l2(p, q)
    # Only one bin differs; sqrt(0.25 + 0 + 0) = 0.5; sum over 200 bins of
    # sqrt of the per-bin sum-of-squared-diffs => 0.5 + sum(sqrt(0)) = 0.5
    assert err == pytest.approx(0.5, abs=1e-12)


def test_metric_jsd_three_histogram_sum():
    """JSD totals over the 3 marginals."""
    p = _mock_three_histograms(seed=4)
    q = _mock_three_histograms(seed=5)
    err = ss.metric_jsd(p, q)
    assert err > 0.0
    assert err <= 3.0 * np.log(2.0) + 1e-12


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


def test_build_reference_histograms_concats_pool():
    pool = [_toy_descriptor_arrays(seed=i) for i in range(3)]
    h_ref, edges = ss.build_reference_histograms(pool)
    assert set(h_ref.keys()) == {"rho_third", "s", "alpha"}
    assert set(edges.keys()) == {"rho_third", "s", "alpha"}
    for k in ("rho_third", "s", "alpha"):
        assert h_ref[k].shape == (ss.NBINS,)
        assert edges[k].shape == (ss.NBINS + 1,)


def _build_toy_pool(npool=8, seed=0):
    """Build a synthetic 8-entry pool with consistent log10 edges."""
    pool = [_toy_descriptor_arrays(seed=seed + i) for i in range(npool)]
    h_ref, edges = ss.build_reference_histograms(pool)
    return pool, h_ref, edges


def test_select_subset_recovers_pool_when_r_eq_n_l2():
    pool, h_ref, edges = _build_toy_pool(npool=5)
    chosen, val = ss.select_subset(pool, edges, h_ref, r=5, metric="l2")
    assert sorted(chosen) == [0, 1, 2, 3, 4]
    # The vectorized prebin-then-sum path uses a different summation order
    # than the concatenate-then-bin path used to construct h_ref, so the
    # r=npool case is algorithmically identical but not bit-identical;
    # roundoff is bounded by ~NBINS * float64-eps per descriptor.
    assert val == pytest.approx(0.0, abs=1e-9)


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


def test_compute_atom_set_single_molecule():
    from ase import Atoms
    a = Atoms("NH3", positions=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)])
    atom_set = ss.compute_atom_set([a])
    assert atom_set == {"N", "H"}


def test_compute_atom_set_full_dfs_pool_yields_seven():
    """For the full 21-AE-molecule Dick pool, the union of elements is
    {H, Li, C, N, O, F, Na} — 7 atomic refs."""
    from ase import Atoms
    formulas = ["H2", "N2", "LiF", "CHN", "CO2", "F2", "C2H2", "CO", "LiH", "Na2",
                "NO", "CH", "OH",
                "NO2", "HN", "O3", "N2O", "CH3", "CH2", "H2O", "H3N"]
    mocks = []
    for f in formulas:
        a = Atoms(f, positions=np.zeros((len(Atoms(f)), 3)))
        mocks.append(a)
    atom_set = ss.compute_atom_set(mocks)
    assert atom_set == {"H", "Li", "C", "N", "O", "F", "Na"}


def test_make_hb_atoms_geometry():
    from ase import Atoms
    hb = ss._make_hb_atoms()
    assert isinstance(hb, Atoms)
    assert hb.get_chemical_formula() == "H4O2"
    assert hb.info["name"] == "HBWD"


def test_make_pt_atoms_geometry():
    from ase import Atoms
    pt = ss._make_pt_atoms()
    assert isinstance(pt, Atoms)
    assert pt.get_chemical_formula() == "H4O2"
    assert pt.info["name"] == "PTWD"


def test_augment_with_hbpt_no_water_omits_hbpt():
    from ase import Atoms
    a = Atoms("H2O", positions=[(0,0,0),(1,0,0),(0,1,0)])
    refs = [Atoms("H", positions=[(0,0,0)]), Atoms("O", positions=[(0,0,0)])]
    out = ss.augment_with_hbpt([a], refs, with_hbpt=False)
    assert len(out) == 1 + 2
    assert all(at.info.get("name") not in ("HBWD", "PTWD") for at in out)


def test_augment_with_hbpt_water_adds_two_entries():
    from ase import Atoms
    a = Atoms("H2O", positions=[(0,0,0),(1,0,0),(0,1,0)])
    refs = [Atoms("H", positions=[(0,0,0)]), Atoms("O", positions=[(0,0,0)])]
    out = ss.augment_with_hbpt([a], refs, with_hbpt=True)
    assert len(out) == 5
    names = [at.info.get("name") for at in out]
    assert names.count("HBWD") == 1
    assert names.count("PTWD") == 1


def test_augment_with_hbpt_idempotent():
    from ase import Atoms
    a = Atoms("H2O", positions=[(0,0,0),(1,0,0),(0,1,0)])
    refs = [Atoms("H", positions=[(0,0,0)]), Atoms("O", positions=[(0,0,0)])]
    out1 = ss.augment_with_hbpt([a], refs, with_hbpt=True)
    out2 = ss.augment_with_hbpt([a], refs, with_hbpt=True)
    assert len(out1) == len(out2)
    for a1, a2 in zip(out1, out2):
        assert a1.get_chemical_formula() == a2.get_chemical_formula()


def test_atom_set_regularizer_independent_of_HBPT_variant():
    """spec §5c: HBPT augmentation only adds H/O which are present in
    practical subsets; atom-set is invariant of with_hbpt."""
    from ase import Atoms
    a = Atoms("H2O", positions=[(0,0,0),(1,0,0),(0,1,0)])
    s1 = ss.compute_atom_set([a])
    a_hb = ss._make_hb_atoms()
    a_pt = ss._make_pt_atoms()
    s2 = ss.compute_atom_set([a, a_hb, a_pt])
    assert s1 == s2 == {"H", "O"}


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


def test_extract_descriptors_returns_finite_arrays(tmp_path):
    from ase import Atoms
    a = Atoms("H2", positions=[(0,0,0),(0.74,0,0)])
    a.info["species"] = "H2"
    arrs = ss.extract_descriptors(a, idx=0, cache_dir=tmp_path)
    for k in ("rho_third", "s", "alpha", "weights"):
        assert np.isfinite(arrs[k]).all()
        assert arrs[k].size > 0


def test_dfs_pool_has_28_distinct_training_points():
    """Per Dick 2021 SI §II: 21 AE + 3 BH76 + 2 IP13 + 2 atom = 28."""
    from xcquinox.alec.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    assert pool["n_total"] == 28
    assert len(pool["ae_molecules"]) == 21
    assert len(pool["bh76_reactions"]) == 3
    assert len(pool["ip13_pairs"]) == 2
    assert len(pool["atom_refs"]) == 2


def test_dfs_pool_ae_molecule_set_matches_si_section_ii():
    """Hill-formula equality with Dick SI §II text."""
    from xcquinox.alec.dfs_pool import build_dfs_pool, DFS_AE_HILL
    pool = build_dfs_pool()
    found = {a.get_chemical_formula() for a in pool["ae_molecules"]}
    assert found == set(DFS_AE_HILL)


def test_dfs_pool_atom_refs_are_h_and_li():
    from xcquinox.alec.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    formulas = {a.get_chemical_formula() for a in pool["atom_refs"]}
    assert formulas == {"H", "Li"}


def test_dfs_bh76_references_present():
    """All 3 BH76 reactions must have a finite e_rxn_ref in kcal/mol."""
    from xcquinox.alec.dfs_pool import DFS_BH76_REACTIONS
    assert len(DFS_BH76_REACTIONS) == 3
    for rxn in DFS_BH76_REACTIONS:
        assert "e_rxn_ref" in rxn
        assert isinstance(rxn["e_rxn_ref"], float)
        assert math.isfinite(rxn["e_rxn_ref"])
        # Barrier heights are positive and < 200 kcal/mol typically
        assert 0.0 < rxn["e_rxn_ref"] < 200.0
        assert "source" in rxn


def test_dfs_ip13_references_present():
    """Both IP13 pairs must have a finite ip_ref in kcal/mol."""
    from xcquinox.alec.dfs_pool import DFS_IP13_PAIRS
    assert len(DFS_IP13_PAIRS) == 2
    for ip in DFS_IP13_PAIRS:
        assert "ip_ref" in ip
        assert isinstance(ip["ip_ref"], float)
        assert math.isfinite(ip["ip_ref"])
        # IP1 of Li~124, C~260, He~570 kcal/mol
        assert 50.0 < ip["ip_ref"] < 600.0
        assert "source" in ip


def test_dfs_ip13_li_matches_nist():
    """Li IE_1 = 5.391719 eV from NIST → 124.336 kcal/mol."""
    from xcquinox.alec.dfs_pool import DFS_IP13_PAIRS
    li = next(p for p in DFS_IP13_PAIRS if p["name"] == "Li_IP")
    expected = 5.391719 * 23.0605  # NIST eV × CODATA conversion
    assert li["ip_ref"] == pytest.approx(expected, abs=0.01)


def test_dfs_ip13_c_matches_nist():
    """C IE_1 = 11.26030 eV from NIST → 259.668 kcal/mol."""
    from xcquinox.alec.dfs_pool import DFS_IP13_PAIRS
    c = next(p for p in DFS_IP13_PAIRS if p["name"] == "C_IP")
    expected = 11.26030 * 23.0605
    assert c["ip_ref"] == pytest.approx(expected, abs=0.05)


def test_dfs_bh76_oh_n2_to_h_n2o_value():
    """OH+N2 → H+N2O barrier is the reverse of NHTBH38 #1: 82.27 kcal/mol REF1."""
    from xcquinox.alec.dfs_pool import DFS_BH76_REACTIONS
    rxn = next(r for r in DFS_BH76_REACTIONS if r["name"] == "OH+N2_to_H+N2O")
    assert rxn["e_rxn_ref"] == pytest.approx(82.27, abs=0.01)


def test_dfs_bh76_oh_ch3_to_o_ch4_value():
    """OH+CH3 → O+CH4 barrier is the reverse of HTBH38 #19-20: 7.90 kcal/mol REF1."""
    from xcquinox.alec.dfs_pool import DFS_BH76_REACTIONS
    rxn = next(r for r in DFS_BH76_REACTIONS if r["name"] == "OH+CH3_to_O+CH4")
    assert rxn["e_rxn_ref"] == pytest.approx(7.90, abs=0.01)


def test_dfs_bh76_hf_f_to_h_f2_value():
    """HF+F → H+F2 barrier is the reverse of NHTBH38 #5: 105.80 kcal/mol REF1."""
    from xcquinox.alec.dfs_pool import DFS_BH76_REACTIONS
    rxn = next(r for r in DFS_BH76_REACTIONS if r["name"] == "HF+F_to_H+F2")
    assert rxn["e_rxn_ref"] == pytest.approx(105.80, abs=0.01)


# ----------------------------------------------------------------------
# DFS_AE_DATA / build_dfs_pool() AE-reference attachment tests
# ----------------------------------------------------------------------

def test_dfs_ae_data_complete_21_molecules():
    """DFS_AE_DATA must list exactly 21 molecules with finite AE refs."""
    from xcquinox.alec.dfs_pool import DFS_AE_DATA, DFS_AE_HILL
    assert len(DFS_AE_DATA) == 21
    # DFS_AE_HILL is now derived from DFS_AE_DATA — must agree.
    assert [d["hill"] for d in DFS_AE_DATA] == DFS_AE_HILL
    seen_hills = set()
    for d in DFS_AE_DATA:
        for key in ("hill", "name", "ae_kcalmol", "source"):
            assert key in d, f"DFS_AE_DATA entry missing {key}: {d}"
        ae = d["ae_kcalmol"]
        assert isinstance(ae, float)
        assert math.isfinite(ae)
        # All 21 Dick molecules sit between Na2 (~17 kcal/mol) and
        # C2H2 (~406 kcal/mol); pad generously to avoid false alarms
        # if a future re-citation shifts a value by < 1 kcal/mol.
        assert 0.0 < ae < 2000.0, f"{d['hill']}: AE out of range {ae}"
        assert isinstance(d["source"], str) and len(d["source"]) > 0
        # Each Hill formula must appear exactly once.
        assert d["hill"] not in seen_hills
        seen_hills.add(d["hill"])


def test_dfs_pool_ae_references_complete():
    """Every AE molecule built by build_dfs_pool must have ae_kcalmol."""
    from xcquinox.alec.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    assert len(pool["ae_molecules"]) == 21
    for a in pool["ae_molecules"]:
        hill = a.info.get("dfs_hill")
        assert "ae_kcalmol" in a.info, f"{hill}: missing ae_kcalmol"
        ae = a.info["ae_kcalmol"]
        assert isinstance(ae, float)
        assert math.isfinite(ae)
        assert 0.0 < ae < 2000.0, f"{hill}: AE out of physical range {ae}"
        # provenance fields
        assert "ae_source" in a.info
        assert "ae_name" in a.info


def test_dfs_pool_ae_anchor_consistency_with_step6():
    """H2O and C2H2 AE refs must match step-6's published anchor values
    (W4-11; tested in xcquinox/alec/tests/test_step6_notebook.py at the
    string level — here we enforce the numeric equality)."""
    from xcquinox.alec.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    by_hill = {a.info["dfs_hill"]: a for a in pool["ae_molecules"]}
    assert by_hill["H2O"].info["ae_kcalmol"] == pytest.approx(232.974, abs=1e-3)
    assert by_hill["C2H2"].info["ae_kcalmol"] == pytest.approx(405.525, abs=1e-3)


def test_dfs_pool_ae_haunschild_lif_lih_na2():
    """LiF, LiH, Na2 are not in W4-17 — Haunschild Table I (kJ/mol/4.184)
    is the authoritative non-relativistic source."""
    from xcquinox.alec.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    by_hill = {a.info["dfs_hill"]: a for a in pool["ae_molecules"]}
    # Haunschild 2012 Table I "E_ref,non-rel" (kJ/mol)
    expected = {
        "FLi": 583.99 / 4.184,   # LiF
        "HLi": 242.27 / 4.184,   # LiH
        "Na2":  71.78 / 4.184,
    }
    for hill, ae_expected in expected.items():
        assert by_hill[hill].info["ae_kcalmol"] == pytest.approx(
            ae_expected, abs=1e-3
        ), f"{hill}: expected {ae_expected:.3f}"


def test_dfs_pool_ae_h2_consistent_with_haunschild():
    """H2 spot-check: 457.73 kJ/mol → 109.401 kcal/mol (Haunschild 2012)."""
    from xcquinox.alec.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    by_hill = {a.info["dfs_hill"]: a for a in pool["ae_molecules"]}
    assert by_hill["H2"].info["ae_kcalmol"] == pytest.approx(
        457.73 / 4.184, abs=1e-3
    )


def test_dfs_pool_ae_kcalmol_lookup_matches_data():
    """DFS_AE_KCALMOL must mirror DFS_AE_DATA exactly."""
    from xcquinox.alec.dfs_pool import DFS_AE_DATA, DFS_AE_KCALMOL
    assert set(DFS_AE_KCALMOL.keys()) == {d["hill"] for d in DFS_AE_DATA}
    for d in DFS_AE_DATA:
        assert DFS_AE_KCALMOL[d["hill"]] == d["ae_kcalmol"]


# ----------------------------------------------------------------------
# Spin / charge metadata invariants (2026-05-01 NO-spin-bug fix)
# ----------------------------------------------------------------------
#
# Background: a step-7 smoke run (2026-05-01) failed on entry #10 (NO,
# 15 electrons) with PySCF "Electron number 15 and spin 0 are not
# consistent".  Root cause: ASE Atoms loaded from g2_97.traj have no
# spin/charge in info{}, so _ase_atoms_to_pyscf_mol defaulted spin=0 for
# every species — wrong for the 7 open-shell molecules in the AE pool
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


def test_dfs_ae_data_every_entry_has_spin_field():
    """Every DFS_AE_DATA entry must carry an explicit `spin` field
    (PySCF 2S convention) plus a `spin_source` citation."""
    from xcquinox.alec.dfs_pool import DFS_AE_DATA
    for d in DFS_AE_DATA:
        assert "spin" in d, f"{d['hill']}: missing spin field"
        assert isinstance(d["spin"], int)
        assert d["spin"] >= 0
        assert "spin_source" in d
        assert isinstance(d["spin_source"], str)
        assert len(d["spin_source"].strip()) > 0


def test_dfs_ae_data_open_shell_spins_match_published_ground_states():
    """Spot-check the published ground-state spins for the 7 open-shell
    AE molecules + the special triplet-singlet cases (NH, CH2)."""
    from xcquinox.alec.dfs_pool import DFS_AE_SPIN
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
    (nelec - spin) % 2 == 0 — the invariant PySCF enforces.  This is
    the regression test for the 2026-05-01 NO smoke-run failure."""
    from xcquinox.alec.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    for at in pool["ae_molecules"]:
        nelec = _atoms_nelec(at)
        spin = int(at.info["spin"])
        assert (nelec - spin) % 2 == 0, (
            f"{at.info['dfs_hill']}: nelec={nelec}, spin={spin} — "
            f"(nelec - spin) is odd; PySCF will reject this SCF.")


def test_dfs_pool_every_atom_ref_satisfies_pyscf_spin_invariant():
    """Every Atoms in pool['atom_refs'] (H, Li) must satisfy the
    spin/electron-count invariant and carry the NIST ASD ground-state
    spin (²S → spin=1)."""
    from xcquinox.alec.dfs_pool import build_dfs_pool
    pool = build_dfs_pool()
    by_sym = {a.get_chemical_formula(): a for a in pool["atom_refs"]}
    for sym in ("H", "Li"):
        a = by_sym[sym]
        # NIST ASD: H I and Li I are both ²S — spin=1 (one unpaired e⁻).
        assert a.info["spin"] == 1, (
            f"{sym}: atom-ref spin must be 1 (²S ground state, NIST ASD)")
        assert a.info["charge"] == 0
        nelec = _atoms_nelec(a)
        assert (nelec - a.info["spin"]) % 2 == 0


def test_dfs_bh76_every_reaction_has_species_spins():
    """Every BH76 reaction must carry species_spins / species_charges
    dicts covering every reactant + product."""
    from xcquinox.alec.dfs_pool import DFS_BH76_REACTIONS
    for rxn in DFS_BH76_REACTIONS:
        assert "species_spins" in rxn, (
            f"{rxn['name']}: missing species_spins dict")
        assert "species_charges" in rxn, (
            f"{rxn['name']}: missing species_charges dict")
        for sp in (*rxn["reactants"], *rxn["products"]):
            assert sp in rxn["species_spins"], (
                f"{rxn['name']}: species_spins missing {sp!r}")
            assert sp in rxn["species_charges"], (
                f"{rxn['name']}: species_charges missing {sp!r}")
            assert isinstance(rxn["species_spins"][sp], int)
            assert isinstance(rxn["species_charges"][sp], int)


def test_dfs_ip13_every_pair_has_neutral_and_cation_spin():
    """Every IP13 pair must carry neutral_spin/cation_spin and
    cation_charge=+1.  Spot-check Li (²S→¹S, spins 1→0) and C (³P→²P°,
    spins 2→1) from NIST ASD."""
    from xcquinox.alec.dfs_pool import DFS_IP13_PAIRS
    expected = {
        "Li_IP": {"neutral_spin": 1, "cation_spin": 0},
        "C_IP":  {"neutral_spin": 2, "cation_spin": 1},
    }
    for pair in DFS_IP13_PAIRS:
        for k in ("neutral_spin", "cation_spin", "cation_charge"):
            assert k in pair, f"{pair['name']}: missing {k}"
        assert pair["cation_charge"] == 1
        assert pair["neutral_charge"] == 0
        exp = expected[pair["name"]]
        assert pair["neutral_spin"] == exp["neutral_spin"]
        assert pair["cation_spin"] == exp["cation_spin"]


def test_dfs_atom_refs_carry_spin_metadata():
    """DFS_ATOM_REFS is a list of dicts with sym/spin/charge."""
    from xcquinox.alec.dfs_pool import DFS_ATOM_REFS
    syms = [r["sym"] for r in DFS_ATOM_REFS]
    assert syms == ["H", "Li"]
    for r in DFS_ATOM_REFS:
        assert r["spin"] == 1   # NIST ASD ²S ground state
        assert r["charge"] == 0
        assert isinstance(r.get("spin_source", ""), str)


def test_select_subset_return_all_returns_full_distribution():
    """return_all=True returns vals array with C(n, r) entries."""
    import math
    import numpy as np
    from xcquinox.alec.subset_selection import (
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


def test_select_subset_return_all_distribution_path(tmp_path):
    """return_all=True with distribution_path persists vals + indices to npz."""
    import math
    import numpy as np
    from xcquinox.alec.subset_selection import (
        build_reference_histograms, select_subset,
    )
    rng = np.random.default_rng(1)
    pool = [{
        "rho_third": rng.uniform(0.1, 1.0, size=(40,)),
        "s": rng.uniform(0.0, 2.0, size=(40,)),
        "alpha": rng.uniform(0.0, 5.0, size=(40,)),
        "weights": np.ones(40),
    } for _ in range(5)]
    h_ref, edges = build_reference_histograms(pool)
    out_npz = tmp_path / "dist.npz"
    select_subset(
        pool, edges, h_ref, r=2, metric="jsd",
        progress=False, return_all=True,
        distribution_path=str(out_npz),
    )
    assert out_npz.is_file()
    npz_safe = np.load(out_npz, allow_pickle=False)
    assert "vals" in npz_safe.files
    assert "indices" in npz_safe.files
    assert npz_safe["vals"].size == math.comb(5, 2)
    npz_safe.close()


def test_select_subset_return_all_distribution_path_hits_cache_on_second_call(tmp_path):
    """Second call with the same distribution_path returns immediately
    from the cached .npz without re-enumerating C(npool, r) combinations.
    Verified by tampering with the cached vals: if cache-read-back works,
    the tampered values are returned verbatim. If the function were
    re-running, fresh (untampered) values would come back instead."""
    import math
    import numpy as np
    from xcquinox.alec.subset_selection import (
        build_reference_histograms, select_subset,
    )
    rng = np.random.default_rng(7)
    pool = [{
        "rho_third": rng.uniform(0.1, 1.0, size=(40,)),
        "s": rng.uniform(0.0, 2.0, size=(40,)),
        "alpha": rng.uniform(0.0, 5.0, size=(40,)),
        "weights": np.ones(40),
    } for _ in range(5)]
    h_ref, edges = build_reference_histograms(pool)
    out_npz = tmp_path / "dist.npz"

    # First call: populates the cache.
    select_subset(
        pool, edges, h_ref, r=2, metric="jsd",
        progress=False, return_all=True,
        distribution_path=str(out_npz),
    )
    assert out_npz.is_file()

    # Overwrite the cache with sentinel values that cannot be the output
    # of the real enumeration. If cache-read-back works, these values
    # come back verbatim on the second call.
    n_combos = math.comb(5, 2)
    sentinel_vals = np.full(n_combos, -42.0, dtype=np.float64)
    sentinel_idx = np.zeros((n_combos, 2), dtype=np.int64)
    np.savez_compressed(
        out_npz,
        vals=sentinel_vals,
        indices=sentinel_idx,
        best_combo=np.array([99, 99]),
        best_val=np.array(-42.0),
    )

    # Second call: must read from cache and return the sentinel values.
    best2, val2, vals2, idx2 = select_subset(
        pool, edges, h_ref, r=2, metric="jsd",
        progress=False, return_all=True,
        distribution_path=str(out_npz),
    )
    np.testing.assert_array_equal(vals2, sentinel_vals)
    np.testing.assert_array_equal(idx2, sentinel_idx)
    assert best2 == (99, 99)
    assert val2 == -42.0


def test_select_subset_return_all_cache_shape_mismatch_raises(tmp_path):
    """Cached .npz from r=2 cannot be reused for an r=3 call — the
    shape sanity check raises ValueError pointing to the cache path."""
    import numpy as np
    import pytest
    from xcquinox.alec.subset_selection import (
        build_reference_histograms, select_subset,
    )
    rng = np.random.default_rng(9)
    pool = [{
        "rho_third": rng.uniform(0.1, 1.0, size=(40,)),
        "s": rng.uniform(0.0, 2.0, size=(40,)),
        "alpha": rng.uniform(0.0, 5.0, size=(40,)),
        "weights": np.ones(40),
    } for _ in range(5)]
    h_ref, edges = build_reference_histograms(pool)
    out_npz = tmp_path / "dist.npz"

    # Populate the cache with r=2 data:
    select_subset(
        pool, edges, h_ref, r=2, metric="jsd",
        progress=False, return_all=True,
        distribution_path=str(out_npz),
    )
    # Re-invoke with r=3; shape mismatch (n_combos differs, idx width differs):
    with pytest.raises(ValueError, match="delete the cache file"):
        select_subset(
            pool, edges, h_ref, r=3, metric="jsd",
            progress=False, return_all=True,
            distribution_path=str(out_npz),
        )


def test_extract_descriptors_write_is_atomic_no_tmp_leftover(tmp_path):
    """After extract_descriptors completes, cache_dir contains exactly
    the cache file — no .tmp leftovers from the atomic-write tempfile.
    Pins the tempfile + os.replace pattern."""
    import os
    from ase import Atoms
    from xcquinox.alec import subset_selection as ss
    a = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])
    ss.extract_descriptors(a, idx=0, cache_dir=tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    # exactly one file: 0_H2.npz; no tempfile-mkstemp leftover with
    # `tmp` prefix or `.npz` suffix.
    assert any(name == "0_H2.npz" for name in files), files
    assert not any(name.startswith("tmp") for name in files), files


def test_select_subset_distribution_write_is_atomic_no_tmp_leftover(tmp_path):
    """After select_subset(return_all=True, distribution_path) completes,
    only the destination .npz exists (no tempfile leftover)."""
    import math
    import numpy as np
    from xcquinox.alec.subset_selection import (
        build_reference_histograms, select_subset,
    )
    rng = np.random.default_rng(11)
    pool = [{
        "rho_third": rng.uniform(0.1, 1.0, size=(40,)),
        "s": rng.uniform(0.0, 2.0, size=(40,)),
        "alpha": rng.uniform(0.0, 5.0, size=(40,)),
        "weights": np.ones(40),
    } for _ in range(5)]
    h_ref, edges = build_reference_histograms(pool)
    out_npz = tmp_path / "dist.npz"
    select_subset(
        pool, edges, h_ref, r=2, metric="jsd",
        progress=False, return_all=True,
        distribution_path=str(out_npz),
    )
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "dist.npz" in files, files
    assert not any(n.startswith("tmp") and n.endswith(".npz")
                    for n in files if n != "dist.npz"), files


def test_concatenate_point_descriptors_concats_all_species():
    """concatenate_point_descriptors stacks per-species descriptors across
    every species in a TrainingPoint (design choice "a" — full union)."""
    import numpy as np
    from ase import Atoms
    from xcquinox.alec.subset_selection import concatenate_point_descriptors
    from xcquinox.alec.training_points import TrainingPoint
    # Synthetic descriptor dicts for two species:
    desc_h2 = {"rho_third": np.array([1., 2., 3.]),
               "s":         np.array([0.1, 0.2, 0.3]),
               "alpha":     np.array([0.5, 0.6, 0.7]),
               "weights":   np.array([1., 1., 1.])}
    desc_h = {"rho_third": np.array([4., 5.]),
              "s":         np.array([0.4, 0.5]),
              "alpha":     np.array([0.8, 0.9]),
              "weights":   np.array([1., 1.])}
    species_desc = {("H2", 0, 0): desc_h2, ("H", 0, 1): desc_h}
    h2_atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)],
                     info={"name": "H2", "charge": 0, "spin": 0})
    h_atoms = Atoms("H", positions=[(0, 0, 0)],
                    info={"name": "H", "charge": 0, "spin": 1})
    point = TrainingPoint(kind="ae", name="H2", species=(h2_atoms, h_atoms))
    out = concatenate_point_descriptors([point], species_desc)
    assert len(out) == 1
    np.testing.assert_array_equal(
        out[0]["rho_third"], np.array([1., 2., 3., 4., 5.]),
    )
    np.testing.assert_array_equal(
        out[0]["s"], np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    )
    np.testing.assert_array_equal(out[0]["weights"], np.ones(5))


def test_concatenate_point_descriptors_dedupe_by_key_lookup():
    """When two points share an atom anchor (e.g. AE 'CH4' and BH76 OH+CH3),
    the species_descriptors dict has it once, but each point still gets the
    H atom's descriptors stacked into its own concatenation. Verifies the
    helper looks up by (name, charge, spin) — no accidental sharing of
    array refs across points."""
    import numpy as np
    from ase import Atoms
    from xcquinox.alec.subset_selection import concatenate_point_descriptors
    from xcquinox.alec.training_points import TrainingPoint
    desc_a = {"rho_third": np.array([1., 2.]),
              "s": np.array([0.1, 0.2]), "alpha": np.array([0.5, 0.6]),
              "weights": np.array([1., 1.])}
    desc_h = {"rho_third": np.array([10., 20.]),
              "s": np.array([0.9, 1.0]), "alpha": np.array([1.5, 1.6]),
              "weights": np.array([2., 2.])}
    species_desc = {("CompA", 0, 0): desc_a, ("H", 0, 1): desc_h}
    a_atoms = Atoms("H", positions=[(0, 0, 0)],
                    info={"name": "CompA", "charge": 0, "spin": 0})
    h_atoms = Atoms("H", positions=[(0, 0, 0)],
                    info={"name": "H", "charge": 0, "spin": 1})
    p1 = TrainingPoint(kind="ae", name="P1", species=(a_atoms, h_atoms))
    p2 = TrainingPoint(kind="bh76", name="P2", species=(h_atoms,))
    out = concatenate_point_descriptors([p1, p2], species_desc)
    # P1 = CompA + H; P2 = H only:
    np.testing.assert_array_equal(out[0]["rho_third"], np.array([1., 2., 10., 20.]))
    np.testing.assert_array_equal(out[1]["rho_third"], np.array([10., 20.]))
    # weights propagate (H has weight=2):
    np.testing.assert_array_equal(out[0]["weights"], np.array([1., 1., 2., 2.]))


def test_concatenate_point_descriptors_missing_species_raises():
    """If a point references a species not in the cache dict, raise
    KeyError pointing at the missing key."""
    import numpy as np
    import pytest
    from ase import Atoms
    from xcquinox.alec.subset_selection import concatenate_point_descriptors
    from xcquinox.alec.training_points import TrainingPoint
    species_desc = {}   # empty cache
    h_atoms = Atoms("H", positions=[(0, 0, 0)],
                    info={"name": "H", "charge": 0, "spin": 1})
    point = TrainingPoint(kind="ae", name="just_H", species=(h_atoms,))
    with pytest.raises(KeyError, match="just_H"):
        concatenate_point_descriptors([point], species_desc)
