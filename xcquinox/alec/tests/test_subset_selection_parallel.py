"""Parallel exhaustive subset selector: correctness vs the serial reference,
partition completeness, batched-JSD equivalence, and the MoleculeSpec/reaction
descriptor helpers feeding it."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from xcquinox.alec import subset_selection as ss
from xcquinox.alec import subset_selection_parallel as ssp


def _synth_pool(n, seed, ngrid=60):
    """A small synthetic descriptor pool with distinct per-point data (so the
    JSD argmin is unique → no tie-break ambiguity in the parallel-vs-serial
    comparison)."""
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        pool.append({
            "rho_third": np.abs(rng.standard_normal(ngrid)) + 0.05 * (i + 1),
            "s": np.abs(rng.standard_normal(ngrid)) + 0.02 * (i + 1),
            "alpha": np.abs(rng.standard_normal(ngrid)),
            "weights": np.abs(rng.standard_normal(ngrid)) + 0.01,
        })
    return pool


# ---------------------------------------------------------------------------
# Parallel == serial select_subset (the correctness contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r", [2, 3, 4])
def test_parallel_matches_serial_jsd(r):
    pool = _synth_pool(8, seed=r)
    h_ref, edges = ss.build_reference_histograms(pool)
    s_combo, s_val = ss.select_subset(
        pool, edges, h_ref, r=r, metric="jsd", progress=False)
    p_combo, p_val = ssp.select_subset_parallel(
        pool, edges, h_ref, r=r, metric="jsd", n_jobs=2)
    assert tuple(s_combo) == tuple(p_combo)
    assert p_val == pytest.approx(s_val, rel=1e-9, abs=1e-12)


def test_parallel_single_job_matches_serial():
    pool = _synth_pool(7, seed=11)
    h_ref, edges = ss.build_reference_histograms(pool)
    s_combo, s_val = ss.select_subset(
        pool, edges, h_ref, r=3, metric="jsd", progress=False)
    p_combo, p_val = ssp.select_subset_parallel(
        pool, edges, h_ref, r=3, metric="jsd", n_jobs=1)
    assert tuple(s_combo) == tuple(p_combo)
    assert p_val == pytest.approx(s_val, rel=1e-9, abs=1e-12)


def test_parallel_rejects_non_jsd_metric():
    pool = _synth_pool(5, seed=1)
    h_ref, edges = ss.build_reference_histograms(pool)
    with pytest.raises(ValueError, match="jsd"):
        ssp.select_subset_parallel(pool, edges, h_ref, r=2, metric="l2")


# ---------------------------------------------------------------------------
# Partition completeness — every C(n, r) combo enumerated exactly once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r", [1, 2, 3, 4])
def test_partition_covers_all_combos_exactly_once(r):
    n = 9
    got = []
    for prefix in ssp._partition_prefixes(n, r):
        p = len(prefix)
        start = prefix[-1] + 1 if prefix else 0
        for suf in itertools.combinations(range(start, n), r - p):
            got.append(tuple(prefix) + suf)
    expected = list(itertools.combinations(range(n), r))
    assert sorted(got) == expected            # same set, in order
    assert len(got) == len(set(got))          # no duplicates


# ---------------------------------------------------------------------------
# Batched index-sum JSD == subset_selection.metric_jsd
# ---------------------------------------------------------------------------

def test_index_sum_jsd_matches_metric_jsd():
    pool = _synth_pool(6, seed=7)
    h_ref, edges = ss.build_reference_histograms(pool)
    counts, _ = ss._prebin_pool(pool, edges)
    ref_pmf = {k: ss._to_pmf(h_ref[k]) for k in ss._DESCRIPTOR_KEYS}
    ssp._init_worker(counts, ref_pmf, {k: 1.0 for k in ss._DESCRIPTOR_KEYS}, 1024)

    combos = [(0, 1, 2), (1, 3, 5), (0, 4, 5)]
    vals = ssp._jsd_batch(np.asarray(combos, dtype=np.int64))
    for combo, v in zip(combos, vals):
        cat = {k: np.concatenate([pool[i][k] for i in combo])
               for k in ss._DESCRIPTOR_KEYS}
        cat["weights"] = np.concatenate([pool[i]["weights"] for i in combo])
        h_cand = ss._bin_with_edges(cat, edges)
        assert v == pytest.approx(ss.metric_jsd(h_ref, h_cand),
                                  rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# MoleculeSpec descriptors + reaction concatenation
# ---------------------------------------------------------------------------

def test_extract_descriptors_for_molspecs_tiny(tmp_path):
    from xcquinox.alec.config import MoleculeSpec
    specs = [
        MoleculeSpec(name="He", atom="He 0 0 0", basis="sto-3g",
                     charge=0, spin=0),
        MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                     charge=0, spin=0),
    ]
    out = ss.extract_descriptors_for_molspecs(
        specs, basis="sto-3g", grid_level=1, cache_dir=tmp_path)
    assert set(out) == {("He", 0, 0), ("H2", 0, 0)}
    for d in out.values():
        n = len(d["rho_third"])
        assert n > 0
        assert len(d["s"]) == len(d["alpha"]) == len(d["weights"]) == n
    # second call hits the cache (no error, same keys)
    out2 = ss.extract_descriptors_for_molspecs(
        specs, basis="sto-3g", grid_level=1, cache_dir=tmp_path)
    assert set(out2) == set(out)


def test_extract_descriptors_for_molspecs_parallel_matches_sequential(tmp_path):
    """n_jobs>1 (spawn pool, 1 thread/worker) yields the same descriptors as the
    sequential path (each species' SCF is independent)."""
    from xcquinox.alec.config import MoleculeSpec
    specs = [
        MoleculeSpec(name="He", atom="He 0 0 0", basis="sto-3g", charge=0, spin=0),
        MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                     charge=0, spin=0),
        MoleculeSpec(name="Li", atom="Li 0 0 0", basis="sto-3g", charge=0, spin=1),
    ]
    seq = ss.extract_descriptors_for_molspecs(
        specs, basis="sto-3g", grid_level=1, cache_dir=tmp_path / "seq", n_jobs=1)
    par = ss.extract_descriptors_for_molspecs(
        specs, basis="sto-3g", grid_level=1, cache_dir=tmp_path / "par", n_jobs=2)
    assert set(seq) == set(par)
    for key in seq:
        # rho^{1/3}, s and the grid weights are reproducible to machine
        # precision (the residual is BLAS thread-count nondeterminism).
        for col in ("rho_third", "s", "weights"):
            np.testing.assert_allclose(par[key][col], seq[key][col],
                                       rtol=1e-6, atol=1e-8)
        # alpha = (tau - tau_W)/tau_unif is a ratio of small quantities in the
        # low-density tail, so the same machine-eps density difference amplifies
        # there (both are valid SCF solutions; selection bins these). A loose
        # tail tolerance is the physically meaningful equivalence check.
        np.testing.assert_allclose(par[key]["alpha"], seq[key]["alpha"],
                                   rtol=1e-3, atol=1e-2)


def test_concatenate_reaction_descriptors_unions_species():
    specs_by_name = {
        "a": SimpleNamespace(name="a", charge=0, spin=0),
        "b": SimpleNamespace(name="b", charge=0, spin=1),
    }
    sd = {
        ("a", 0, 0): {"rho_third": np.array([1.0, 2.0]), "s": np.array([0.1, 0.2]),
                      "alpha": np.array([0.3, 0.4]), "weights": np.array([1.0, 1.0])},
        ("b", 0, 1): {"rho_third": np.array([3.0]), "s": np.array([0.5]),
                      "alpha": np.array([0.6]), "weights": np.array([2.0])},
    }
    rxn = {"reactants": ["a"], "products": ["b"]}
    out = ss.concatenate_reaction_descriptors([rxn], sd, specs_by_name)
    assert len(out) == 1
    assert out[0]["rho_third"].tolist() == [1.0, 2.0, 3.0]
    assert out[0]["weights"].tolist() == [1.0, 1.0, 2.0]
