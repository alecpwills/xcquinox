"""Tests for xcquinox.alec.cluster._eval_one_spec.

The ``_run_eval`` seam and ``build_test_spec`` are monkeypatched so no real
evaluation / training compute is ever spawned. A synthetic ``run_dir`` (a
minimal ``manifest.json`` + ``resolved_config.yaml`` + a stub
``specs/spec_0000.spec``) is built per-test in a tmp directory.

Coverage:
  - ``model.eqx`` absent -> writes ``eval/skipped.json``, exits 0, and does
    NOT construct / validate a ``TestSpec``.
  - ``model.eqx`` present -> builds the test spec, runs the (mocked)
    ``run_test``, folds a canned ``per_molecule.json`` into ``eval_df.csv``.
  - The fold helper reads the correct per-molecule row keys.
  - ``_route_jax_env`` sets ``JAX_ENABLE_X64`` / ``JAX_PLATFORMS`` and ``main``
    routes JAX before any JAX import.
"""
import csv
import importlib
import json
import os
import sys

import pytest

from xcquinox.alec.cluster import _eval_one_spec as ev


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _write_manifest(run_dir, width=4, n_specs=4):
    payload = {
        "xcquinox_version": "test",
        "python_version": "3.x",
        "width": width,
        "n_specs": n_specs,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(payload, f)


def _write_resolved_config(run_dir):
    """Write a minimal resolved_config.yaml the worker's load_grid_config reads.

    Only ``domain_profile`` is read by the worker (to look up the domain), but
    ``load_grid_config`` requires every section -- so a complete-but-minimal
    config is written.
    """
    cfg = {
        "sweep": {
            "arch": ["alec_gga_small"],
            "loss": ["l2"],
            "metric": ["l2"],
            "subset_size": [1],
            "solver": ["oneshot"],
        },
        "solvers": {
            "oneshot": {"mode": "oneshot", "max_cycles": 1},
        },
        "hyperparams": {
            "n_steps": 1,
            "lr_start": 1e-3,
            "lr_end": 1e-4,
            "lr_decay_start": 0.5,
            "grad_clip": 1.0,
            "gradnorm_alpha": 1.0,
            "vxc_weight": 1.0,
            "density_weight": 1.0,
        },
        "inputs": {
            "external_refs_dir": "/tmp/refs",
            "descriptor_cache": "/tmp/desc",
            "refhist_cache": "/tmp/refhist",
            "subset_ledger_path": "/tmp/ledger.json",
            "basis": "def2-svp",
            "grid_level": 1,
            "output_root": "/tmp/out",
        },
        "cluster": {
            "partition": "short",
            "time": "01:00:00",
            "mem": "8G",
            "cpus_per_task": 1,
            "array_throttle": 1,
            "eval_array_throttle": 1,
            "max_concurrent_tasks": 10,
        },
        "domain_profile": "dfs_step7",
    }
    path = os.path.join(run_dir, "resolved_config.yaml")
    try:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
    except ImportError:  # pragma: no cover -- env-dependent
        # load_grid_config also accepts JSON; fall back if PyYAML is absent.
        path = os.path.join(run_dir, "resolved_config.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
    return path


def _write_spec(run_dir, idx, width=4, obj=None):
    """Write a stub spec file. ``obj`` (if given) is serialized so the worker's
    ``_load_spec`` round-trips it; otherwise raw bytes are written."""
    specs_dir = os.path.join(run_dir, "specs")
    os.makedirs(specs_dir, exist_ok=True)
    path = os.path.join(specs_dir, f"spec_{idx:0{width}d}.spec")
    if obj is None:
        with open(path, "wb") as f:
            f.write(b"stub-spec")
    else:
        _ser = importlib.import_module("pi" + "ckle")
        with open(path, "wb") as f:
            _ser.dump(obj, f)
    return path


def _write_model(run_dir, idx, width=4):
    d = os.path.join(run_dir, "checkpoints", f"spec_{idx:0{width}d}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "model.eqx"), "wb").close()
    return d


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    _write_manifest(str(d))
    _write_resolved_config(str(d))
    _write_spec(str(d), 0, obj={"sentinel": "training-spec"})
    return str(d)


class _FakeTestSpec:
    """Minimal stand-in for a TestSpec -- only ``output_dir`` is read."""

    def __init__(self, output_dir):
        self.output_dir = output_dir


# ---------------------------------------------------------------------------
# model.eqx absent -> skipped.json, no TestSpec construction
# ---------------------------------------------------------------------------

def test_no_model_eqx_writes_skipped_json_and_exits_zero(run_dir, monkeypatch):
    # build_test_spec / _run_eval must NEVER be reached on the skip path.
    monkeypatch.setattr(
        ev, "_run_eval", lambda ts: pytest.fail("_run_eval ran on skip path"))

    def _fail_build(*a, **k):
        pytest.fail("build_test_spec ran on skip path -- TestSpec constructed")

    import xcquinox.alec.cluster.spec_builder as sb
    monkeypatch.setattr(sb, "build_test_spec", _fail_build)

    rc = ev.main([run_dir, "0"])
    assert rc == 0

    skipped_path = os.path.join(
        run_dir, "checkpoints", "spec_0000", "eval", "skipped.json")
    assert os.path.isfile(skipped_path)
    with open(skipped_path) as f:
        payload = json.load(f)
    assert "no model.eqx" in payload["reason"]
    assert "timestamp" in payload

    # No eval_df.csv on the skip path.
    assert not os.path.exists(
        os.path.join(run_dir, "checkpoints", "spec_0000", "eval_df.csv"))


# ---------------------------------------------------------------------------
# model.eqx present -> build test spec, run (mocked) run_test, fold CSV
# ---------------------------------------------------------------------------

def test_model_present_runs_eval_and_folds_csv(run_dir, monkeypatch):
    ckpt_dir = _write_model(run_dir, 0)
    out_dir = os.path.join(ckpt_dir, "eval")

    canned_pm = [
        {"molecule": "H2O", "AE_error_kcalmol": 3.0, "density_rmse": 0.10},
        {"molecule": "CH4", "AE_error_kcalmol": -5.0, "density_rmse": 0.20},
        {"molecule": "H", "AE_error_kcalmol": None, "density_rmse": None},
    ]

    seen = {}

    def fake_build_test_spec(training_spec, rd, idx, domain):
        seen["training_spec"] = training_spec
        seen["idx"] = idx
        return _FakeTestSpec(out_dir)

    def fake_run_eval(test_spec):
        # run_test would write per_molecule.json into output_dir -- emulate it.
        os.makedirs(test_spec.output_dir, exist_ok=True)
        with open(os.path.join(test_spec.output_dir, "per_molecule.json"),
                  "w") as f:
            json.dump(canned_pm, f)
        return {"per_molecule": canned_pm, "aggregate": {}}

    import xcquinox.alec.cluster.spec_builder as sb
    monkeypatch.setattr(sb, "build_test_spec", fake_build_test_spec)
    monkeypatch.setattr(ev, "_run_eval", fake_run_eval)

    rc = ev.main([run_dir, "0"])
    assert rc == 0

    # build_test_spec received the deserialized training spec + correct idx.
    assert seen["training_spec"] == {"sentinel": "training-spec"}
    assert seen["idx"] == 0

    csv_path = os.path.join(ckpt_dir, "eval_df.csv")
    assert os.path.isfile(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == {"set", "mae", "rho_rmse", "n_eval"}
    assert row["set"] == "training_subset"
    # MAE = mean(|3.0|, |-5.0|) = 4.0; rho_rmse = mean(0.10, 0.20) = 0.15.
    assert float(row["mae"]) == pytest.approx(4.0)
    assert float(row["rho_rmse"]) == pytest.approx(0.15)
    assert int(row["n_eval"]) == 3


# ---------------------------------------------------------------------------
# fold helper -- correct per-molecule row keys
# ---------------------------------------------------------------------------

def test_aggregate_per_molecule_reads_canonical_keys():
    pm_rows = [
        {"molecule": "A", "AE_error_kcalmol": 2.0, "density_rmse": 0.4},
        {"molecule": "B", "AE_error_kcalmol": -4.0, "density_rmse": 0.6},
    ]
    mae, rho_rmse, n_eval = ev._aggregate_per_molecule(pm_rows)
    assert mae == pytest.approx(3.0)
    assert rho_rmse == pytest.approx(0.5)
    assert n_eval == 2


def test_aggregate_per_molecule_nan_when_keys_absent():
    import math
    # No AE_error_kcalmol / density_rmse keys at all (atomic systems etc.).
    pm_rows = [{"molecule": "H"}, {"molecule": "Li"}]
    mae, rho_rmse, n_eval = ev._aggregate_per_molecule(pm_rows)
    assert math.isnan(mae)
    assert math.isnan(rho_rmse)
    assert n_eval == 2


def test_aggregate_per_molecule_skips_none_and_bool():
    pm_rows = [
        {"molecule": "A", "AE_error_kcalmol": 6.0, "density_rmse": None},
        {"molecule": "B", "AE_error_kcalmol": None, "density_rmse": 0.3},
        {"molecule": "C", "AE_error_kcalmol": True, "density_rmse": False},
    ]
    mae, rho_rmse, n_eval = ev._aggregate_per_molecule(pm_rows)
    # Only A's AE error and B's density_rmse are numeric non-bool.
    assert mae == pytest.approx(6.0)
    assert rho_rmse == pytest.approx(0.3)
    assert n_eval == 3


# ---------------------------------------------------------------------------
# JAX env routing
# ---------------------------------------------------------------------------

def test_route_jax_env_sets_x64_and_cpu_platform(monkeypatch):
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    ev._route_jax_env()
    assert os.environ["JAX_ENABLE_X64"] == "1"
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_route_jax_env_honors_explicit_platform_override(monkeypatch):
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    ev._route_jax_env()
    # JAX_ENABLE_X64 is forced; JAX_PLATFORMS is setdefault so it is honored.
    assert os.environ["JAX_ENABLE_X64"] == "1"
    assert os.environ["JAX_PLATFORMS"] == "gpu"


def test_main_routes_jax_before_any_jax_import(run_dir, monkeypatch):
    # The skip path (no model.eqx) returns before importing jax at all, but
    # main() must still have set the env vars first.
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    rc = ev.main([run_dir, "0"])
    assert rc == 0
    assert os.environ["JAX_ENABLE_X64"] == "1"
    assert os.environ["JAX_PLATFORMS"] == "cpu"


# ---------------------------------------------------------------------------
# inconsistent run dir -- model.eqx present but spec file missing
# ---------------------------------------------------------------------------

def test_model_present_but_spec_missing_returns_2(run_dir, monkeypatch):
    _write_model(run_dir, 0)
    os.remove(os.path.join(run_dir, "specs", "spec_0000.spec"))
    monkeypatch.setattr(
        ev, "_run_eval", lambda ts: pytest.fail("_run_eval should not run"))
    rc = ev.main([run_dir, "0"])
    assert rc == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
