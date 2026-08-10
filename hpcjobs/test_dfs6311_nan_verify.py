"""Plumbing tests for the production-scale V_xc verification job.

The physics legs run on the cluster; what is pinned here is everything that
could silently invalidate the job's OUTPUT: loading a random net where a
checkpoint was expected, the prior-potential emulation not actually switching
(or not switching back), and the exit-code / report contract the sbatch
wrapper and the downstream reader rely on. All stubs, no SCF -- milliseconds.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "dfs6311_nan_verify", _HERE / "dfs6311_nan_verify.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["dfs6311_nan_verify"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


nv = _load()


def test_missing_checkpoint_raises_rather_than_running_a_random_net(tmp_path):
    with pytest.raises(FileNotFoundError, match="pretrain checkpoint missing"):
        nv.load_pretrained(str(tmp_path), "deep_mgga_3x16")


def test_prior_potential_swaps_and_restores_the_predicate():
    import xcquinox.alec as alec
    import xcquinox.alec.oneshot as oneshot

    model = alec.AlecGGAModel.from_arch(
        alec.get_architecture("deep_mgga_3x16"), seed=0)
    assert oneshot.has_dm_dependent_descriptor(model) is True
    orig = oneshot.has_dm_dependent_descriptor
    with nv.prior_potential():
        assert oneshot.has_dm_dependent_descriptor(model) is False
    assert oneshot.has_dm_dependent_descriptor is orig
    assert oneshot.has_dm_dependent_descriptor(model) is True
    # restored even when the body raises -- a leaked swap would silently turn
    # every later "corrected" run into a "prior" run.
    with pytest.raises(RuntimeError):
        with nv.prior_potential():
            raise RuntimeError("boom")
    assert oneshot.has_dm_dependent_descriptor is orig


def test_species_table_is_wellformed():
    for name, atom, spin, comp in nv.SPECIES:
        n_from_comp = sum(int(n) for _s, n in comp)
        n_from_atom = len([a for a in atom.split(";") if a.strip()])
        assert n_from_comp == n_from_atom, (name, comp, atom)
        assert spin in (0, 1, 2)
    assert set(nv.ARCHS) == {"deep_mgga_3x16", "deep_rung35_mgga_3x16",
                             "deep_rung35_3x16", "deep_rung35_attn_3x16"}


def _stubbed(monkeypatch, finite=True):
    """Stub the heavy pieces so main() exercises only its own plumbing."""
    import equinox as eqx
    import jax.numpy as jnp
    import xcquinox.alec as alec

    model = alec.AlecGGAModel.from_arch(
        alec.get_architecture("deep_mgga_3x16"), seed=0)
    monkeypatch.setattr(nv, "load_pretrained", lambda run, arch: model)
    monkeypatch.setattr(nv, "mol_data_for",
                        lambda *a, **k: {"rho_grid": jnp.ones(4)})

    class _R(types.SimpleNamespace):
        pass

    def fake_run_scf(cfg, m, md):
        return _R(total_energy=jnp.asarray(-1.0),
                  cycles_run=jnp.asarray(cfg.max_cycles),
                  converged=jnp.asarray(False))
    monkeypatch.setattr(nv, "run_scf", fake_run_scf)

    bad = jnp.asarray(float("nan")) if not finite else None

    def fake_vag(fn):
        def inner(m):
            params = eqx.filter(m, eqx.is_inexact_array)
            if bad is not None:
                params = jax.tree_util.tree_map(
                    lambda a: a * float("nan"), params)
            return jnp.asarray(-1.0), params
        return inner
    import jax
    monkeypatch.setattr(nv.eqx, "filter_value_and_grad", fake_vag)


def test_main_report_contract_and_exit_codes(tmp_path, monkeypatch):
    _stubbed(monkeypatch, finite=True)
    out = tmp_path / "report.json"
    rc = nv.main([str(tmp_path), "--grid-level", "1", "--cycles", "2",
                  "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["basis"] == nv.BASIS
    assert len(rep["leg1"]) == len(nv.ARCHS) * len(nv.SPECIES)
    assert len(rep["leg2"]) == len(nv.ARCHS) * len(nv.SPECIES)
    # leg 3 carries one row per (arch, species, variant)
    assert len(rep["leg3"]) == 2 * len(nv.ARCHS) * len(nv.SPECIES)
    assert all(r["grad_finite"] for r in rep["leg1"])
    assert {r["variant"] for r in rep["leg3"]} == {"corrected", "prior"}


def test_main_exit_code_flags_nonfinite_gradient(tmp_path, monkeypatch):
    _stubbed(monkeypatch, finite=False)
    out = tmp_path / "report.json"
    rc = nv.main([str(tmp_path), "--grid-level", "1", "--cycles", "2",
                  "--out", str(out)])
    assert rc == 1, "a non-finite leg-1 gradient must fail the job"
    rep = json.loads(out.read_text())
    assert not all(r["grad_finite"] for r in rep["leg1"])
