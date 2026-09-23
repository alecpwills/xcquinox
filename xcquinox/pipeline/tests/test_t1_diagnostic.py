"""The CCSD T1 diagnostic stored with the benchmark reference densities.

The diagnostic is the model-free multireference flag the outlier-free figure
variant keys on, so it has to travel with the reference it describes: computed
beside the CCSD 1-RDM, cached in the intermediates npz, written into the final
per-species npz, and collected into one run-level ``t1_diagnostics.json``.

No quantum chemistry runs here: the HF stage and the coupled-cluster classes
are replaced by recorders, and only pyscf's pure-numpy
``cc.ccsd.get_t1_diagnostic`` is evaluated, as the closed-shell oracle.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
# the closed-shell oracle, captured before any test stubs the pyscf.cc.ccsd
# module away
from pyscf.cc.ccsd import get_t1_diagnostic as _library_t1_diagnostic

import xcquinox.pipeline.benchmark_refs as br
import xcquinox.pipeline.external_refs as er


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------

class _FakeCC:
    """Stand-in for a converged ``mycc``: amplitudes, the convergence flag and
    the RCCSD accessor, nothing else."""

    def __init__(self, t1, counter=None):
        self.t1 = t1
        self.frozen = 0
        self.converged = True
        self._counter = counter if counter is not None else {}

    def kernel(self):
        self._counter["kernel"] = self._counter.get("kernel", 0) + 1

    def get_t1_diagnostic(self, t1=None):
        return _library_t1_diagnostic(self.t1 if t1 is None else t1)

    def make_rdm1(self, ao_repr=True):
        return np.eye(2) * 0.5


def test_uccsd_t1_diagnostic_matches_the_hand_value():
    """Kills the mutation ``the T1 denominator nocc_a + nocc_b instead of
    twice the electron count``: the unrestricted diagnostic is the
    spin-orbital amplitude norm over TWICE the number of correlated electrons,
    sqrt((|t1a|^2 + |t1b|^2) / (2 (nocc_a + nocc_b))), the normalization under
    which a closed shell reproduces pyscf's restricted value (whose spatial
    amplitudes count each spin once), so one 0.02 threshold reads both legs.

    t1a = [[0.1, 0.2]] (1 correlated alpha electron), t1b = [[0.3, 0.4],
    [0.5, 0.6]] (2 correlated beta electrons):
        sum of squares = 0.91, denominator = 2 x 3 = 6, T1 = sqrt(0.91/6).
    """
    t1a = np.array([[0.1, 0.2]])
    t1b = np.array([[0.3, 0.4], [0.5, 0.6]])
    got = er._t1_diagnostic(_FakeCC((t1a, t1b)))
    assert got == pytest.approx(math.sqrt(0.91 / 6.0), rel=1e-12)
    # neither the electron count alone nor either channel's nocc
    assert got != pytest.approx(math.sqrt(0.91 / 3.0), rel=1e-6)
    assert got != pytest.approx(math.sqrt(0.91 / 1.0), rel=1e-6)
    assert got != pytest.approx(math.sqrt(0.91 / 2.0), rel=1e-6)
    # a closed shell (t1a == t1b == the spatial amplitudes) reproduces the
    # restricted diagnostic exactly
    from pyscf.cc import ccsd
    t = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert er._t1_diagnostic(_FakeCC((t, t))) == pytest.approx(
        ccsd.get_t1_diagnostic(t), rel=1e-12)


def test_rccsd_t1_diagnostic_returns_the_library_value():
    """The closed-shell value equals pyscf's own diagnostic (ccsd.py
    get_t1_diagnostic) to the last bit: the seam evaluates the same
    Lee-Taylor formula on the spatial-orbital amplitude array, so it does
    not depend on which coupled-cluster class produced t1."""
    from pyscf.cc import ccsd
    t1 = np.array([[0.1, 0.2, 0.05], [0.3, 0.4, 0.02]])
    assert er._t1_diagnostic(_FakeCC(t1)) == pytest.approx(
        ccsd.get_t1_diagnostic(t1), rel=1e-12)


# ---------------------------------------------------------------------------
# The reference pipeline
# ---------------------------------------------------------------------------

class _FakeAtoms:
    def get_positions(self):
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    def get_chemical_symbols(self):
        return ["H", "H"]


_T1_ARRAY = np.array([[0.1, 0.2], [0.3, 0.4]])


def _ccsd_stubs(monkeypatch, counter=None):
    """Replace the HF stage and the RCCSD class so ``run_ccsd_with_cache``
    runs its cache/write logic with no correlated calculation. The molecule
    build and the AO evaluation are left to pyscf (both are cheap and pure)."""
    import types as _types

    import pyscf.cc
    monkeypatch.setattr(er, "_prepare_converged_hf",
                        lambda mol, **kw: object())
    fake_ccsd = _types.SimpleNamespace(
        RCCSD=lambda mf: _FakeCC(_T1_ARRAY, counter))
    monkeypatch.setattr(pyscf.cc, "ccsd", fake_ccsd)


def _ccsd_kwargs(cache_dir):
    return dict(
        scf_payload={
            "spin_unrestricted": False,
            "dm": np.eye(2) * 0.5,
            "grid_coords": np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.3],
                                     [0.0, 0.0, 0.5]]),
            "grid_weights": np.array([1.0, 1.0, 1.0]),
        },
        cache_dir=str(cache_dir), basis="sto-3g", grid_level=1,
        density_fit=False, auxbasis=None, orientation_lock_strength=0.0)


def _spec():
    return er.SpeciesEntry(name="h2", charge=0, spin=0, source="benchmark")


def _cache_path(cache_dir):
    from pathlib import Path
    return (Path(cache_dir) / "_intermediates"
            / er._intermediate_cache_name("h2", grid_level=1, basis="sto-3g",
                                          density_fit=False, kind="ccsd",
                                          orientation_lock_strength=0.0))


def _expected_t1():
    return float(_library_t1_diagnostic(_T1_ARRAY))


def test_run_ccsd_with_cache_stores_the_t1_diagnostic(tmp_path, monkeypatch):
    """The diagnostic is computed where the amplitudes still exist (the npz
    carries no amplitudes) and is returned in the payload as well as cached."""
    _ccsd_stubs(monkeypatch)
    out = er.run_ccsd_with_cache(_spec(), _FakeAtoms(),
                                 **_ccsd_kwargs(tmp_path))
    assert out["t1_diagnostic"] == pytest.approx(_expected_t1(), rel=1e-12)
    with np.load(_cache_path(tmp_path), allow_pickle=False) as z:
        assert "t1_diagnostic" in z.files
        assert float(z["t1_diagnostic"]) == pytest.approx(_expected_t1(),
                                                          rel=1e-12)


def test_run_ccsd_with_cache_returns_a_cached_t1_without_recomputing(
        tmp_path, monkeypatch):
    """A cache that already carries the key is served from disk: the backfill
    must not re-run coupled cluster for a species that has the value."""
    counter = {}
    _ccsd_stubs(monkeypatch, counter)
    er.run_ccsd_with_cache(_spec(), _FakeAtoms(), **_ccsd_kwargs(tmp_path))
    assert counter["kernel"] == 1
    out = er.run_ccsd_with_cache(_spec(), _FakeAtoms(),
                                 **_ccsd_kwargs(tmp_path))
    assert counter["kernel"] == 1                      # served from cache
    assert out["t1_diagnostic"] == pytest.approx(_expected_t1(), rel=1e-12)


def test_require_t1_recomputes_a_cache_lacking_the_key(tmp_path, monkeypatch):
    """Existing caches predate the diagnostic. Under ``require_t1`` such a
    cache is recomputed and rewritten with the key added and every other array
    byte-identical -- the rewrite must not move a reference density."""
    counter = {}
    _ccsd_stubs(monkeypatch, counter)
    er.run_ccsd_with_cache(_spec(), _FakeAtoms(), **_ccsd_kwargs(tmp_path))
    path = _cache_path(tmp_path)
    with np.load(path, allow_pickle=False) as z:
        before = {k: np.array(z[k]) for k in z.files}
    stripped = {k: v for k, v in before.items() if k != "t1_diagnostic"}
    np.savez_compressed(path, **stripped)

    out = er.run_ccsd_with_cache(_spec(), _FakeAtoms(), require_t1=True,
                                 **_ccsd_kwargs(tmp_path))
    assert counter["kernel"] == 2                       # recomputed once
    assert out["t1_diagnostic"] == pytest.approx(_expected_t1(), rel=1e-12)
    with np.load(path, allow_pickle=False) as z:
        after = {k: np.array(z[k]) for k in z.files}
    assert "t1_diagnostic" in after
    assert set(after) - {"t1_diagnostic"} == set(stripped)
    for k, v in stripped.items():
        assert after[k].dtype == v.dtype, k
        assert after[k].tobytes() == v.tobytes(), k


# ---------------------------------------------------------------------------
# benchmark_refs: the final npz, the backfill and the run-level JSON
# ---------------------------------------------------------------------------

def _mol_spec():
    from xcquinox.pipeline.config import MoleculeSpec
    return MoleculeSpec(name="h2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                        charge=0, spin=0, atom_composition=(("H", 2),))


def _stage_stubs(monkeypatch, t1=0.0173):
    """Both reference stages canned, so generate_one exercises only its own
    write path."""
    monkeypatch.setattr(br, "_mol_spec_to_atoms", lambda ms: _FakeAtoms())
    monkeypatch.setattr(br, "run_scf_with_cache", lambda *a, **k: {
        "dm": np.eye(2) * 0.5})
    cc = {
        "dm_ao": np.eye(2) * 0.5,
        "rho_ref_grid": np.array([0.4, 0.3, 0.2]),
        "grid_weights": np.array([1.0, 1.0, 1.0]),
        "ao_grid": np.array([[0.5, 0.1], [0.4, 0.2], [0.3, 0.3]]),
        "t1_diagnostic": np.array(float(t1)),
    }
    monkeypatch.setattr(br, "run_ccsd_with_cache", lambda *a, **k: dict(cc))
    return cc


def test_generate_one_writes_the_t1_key(tmp_path, monkeypatch):
    """The final per-species npz carries the diagnostic beside the density it
    describes, so the collector needs no second pass over the intermediates."""
    _stage_stubs(monkeypatch, t1=0.0173)
    assert br.generate_one(_mol_spec(), out_dir=tmp_path, basis="sto-3g",
                           grid_level=1) == "OK"
    with np.load(tmp_path / "h2.npz", allow_pickle=False) as z:
        assert "t1_diagnostic" in z.files
        assert float(z["t1_diagnostic"]) == pytest.approx(0.0173)


