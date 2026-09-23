"""Tests for xcquinox.pipeline.cluster._eval_one_spec.

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
import json
import os
import pickle  # noqa: S403 - round-trips this test's own in-process spec fixtures
import sys

import pytest

from xcquinox.pipeline.cluster import _eval_one_spec as ev


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
            "arch": ["pipeline_gga_small"],
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
            "subset_ledger_path": "/tmp/ledger.json",
            "basis": "def2-svp",
            "grid_level": 1,
            "output_root": "/tmp/out",
        },
        "pretrain": {
            "data_dir": "/tmp/pretrain_data",
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
        with open(path, "wb") as f:
            pickle.dump(obj, f)
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

    import xcquinox.pipeline.cluster.spec_builder as sb
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
# WS6: eval gating when training is INCOMPLETE (resume in progress)
# ---------------------------------------------------------------------------


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

    import xcquinox.pipeline.cluster.spec_builder as sb
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
    assert set(row.keys()) == {"set", "mae", "rho_rmse", "rho_rmse_pbe",
                               "n_eval"}
    assert row["set"] == "training_subset"
    # MAE = mean(|3.0|, |-5.0|) = 4.0; rho_rmse = mean(0.10, 0.20) = 0.15.
    assert float(row["mae"]) == pytest.approx(4.0)
    assert float(row["rho_rmse"]) == pytest.approx(0.15)
    # CODE-03: n_eval counts AE-CONTRIBUTING molecules (the MAE denominator),
    # not total rows. H2O + CH4 contribute; H (AE=None) does not.
    assert int(row["n_eval"]) == 2


# ---------------------------------------------------------------------------
# default double held-out eval: final (model.eqx) + best (model_best.eqx)
# ---------------------------------------------------------------------------

def _stub_insample(monkeypatch, out_dir):
    """Mock the in-sample eval so main() reaches the held-out section."""
    def fake_build_test_spec(training_spec, rd, idx, domain):
        return _FakeTestSpec(out_dir)

    def fake_run_eval(test_spec):
        os.makedirs(test_spec.output_dir, exist_ok=True)
        with open(os.path.join(test_spec.output_dir, "per_molecule.json"),
                  "w") as f:
            json.dump([{"molecule": "H2O", "AE_error_kcalmol": 3.0,
                        "density_rmse": 0.1}], f)
        return {}
    import xcquinox.pipeline.cluster.spec_builder as sb
    monkeypatch.setattr(sb, "build_test_spec", fake_build_test_spec)
    monkeypatch.setattr(ev, "_run_eval", fake_run_eval)


def test_main_runs_both_final_and_best_held_out_eval(run_dir, monkeypatch):
    # model_best.eqx present -> held-out eval runs TWICE by default: final ->
    # eval_holdout/, best -> eval_holdout_best/ (the "double the data" return).
    ckpt_dir = _write_model(run_dir, 0)
    open(os.path.join(ckpt_dir, "model_best.eqx"), "wb").close()
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda rd, idx, cfg, ck, mp, ts, holdout_subdir="eval_holdout":
            calls.append((os.path.basename(mp), holdout_subdir)))

    assert ev.main([run_dir, "0"]) == 0
    assert calls == [("model.eqx", "eval_holdout"),
                     ("model_best.eqx", "eval_holdout_best")]


# ---------------------------------------------------------------------------
# WS3 (2026-06-20): report only the held-out TEST slice; eval model_val_best.eqx
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# fold helper -- correct per-molecule row keys
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JAX env routing
# ---------------------------------------------------------------------------

def test_route_jax_env_sets_x64_and_cpu_platform(monkeypatch):
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    ev._route_jax_env()
    assert os.environ["JAX_ENABLE_X64"] == "1"
    assert os.environ["JAX_PLATFORMS"] == "cpu"


# ---------------------------------------------------------------------------
# inconsistent run dir -- model.eqx present but spec file missing
# ---------------------------------------------------------------------------


# per-molecule aggregation must exclude non-finite values


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# Cold-start channel: 4th held-out pass on the FINAL checkpoint
# ---------------------------------------------------------------------------


def _full_mode_spec():
    """A minimal REAL TrainingSpec with a FULL-mode solver, so the
    orchestrator-side dataclasses.replace has something genuine to act on."""
    import dataclasses as _dc

    import xcquinox.pipeline as pipeline
    from xcquinox.pipeline.config import MoleculeSpec, TrainingSpec
    from xcquinox.pipeline.solver import (SolverBackend, SolverConfig,
                                      SolverMode)
    mol = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74",
                       basis="sto-3g", charge=0, spin=0,
                       atom_composition=(("H", 2),))
    spec = TrainingSpec.from_dicts(
        arch=pipeline.get_architecture("deep_3x16"), molecules=(mol,),
        targets={"H2": -1.0}, atom_energies={"H": -0.5},
        loss_name="A_atomization", loss_kwargs={"vxc_weight": 0.0},
        update_scheme="per_molecule", require_atom_anchors=False,
        n_steps=1, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
        grad_clip=1.0, checkpoint_dir=None, seed=42)
    solver = SolverConfig(backend=SolverBackend.MANUAL,
                          mode=SolverMode.FULL, max_cycles=3,
                          scf_loss_use_tail=True)
    return _dc.replace(spec, solver_config=solver)


# ---------------------------------------------------------------------------
# Held-out species slice
# ---------------------------------------------------------------------------

def _slice_fixture(monkeypatch, run_dir):
    """Wire the held-out seams so _run_held_out_eval runs no SCF.

    The pool stub is the six-species matrix slice plus one species outside it,
    so a slice that is applied is distinguishable from one that is not.
    """
    from types import SimpleNamespace
    import xcquinox.pipeline.eval_holdout as eh
    import xcquinox.pipeline.full_benchmark_pools as fbp
    spec = _full_mode_spec()
    ckpt_dir = _write_model(run_dir, 0)
    cfg = SimpleNamespace(cluster=SimpleNamespace(eval_workers=1),
                          held_out_strict=False)
    pool = {n: f"spec_{n}" for n in
            ("h", "h2", "o", "oh", "n2o", "n2ohts", "c2h6")}
    rxns = [
        {"name": "w411_h2_atomization", "reactants": ["h2"], "products": ["h"]},
        {"name": "w411_c2h6_atomization", "reactants": ["c2h6"],
         "products": ["h", "c"]},
    ]
    seen = {}
    monkeypatch.setattr(eh, "load_trained_model", lambda ts, mp: "MODEL")
    monkeypatch.setattr(fbp, "load_full_held_out_pools",
                        lambda basis=None, grid_level=None: (dict(pool),
                                                             list(rxns)))
    monkeypatch.setattr(ev, "_held_out_basis_grid", lambda cfg: ("def2-svp", 1))

    def _capture(**kw):
        seen["mol_specs"] = dict(kw["mol_specs"])
        seen["reactions"] = list(kw["reactions"])
        return {"n_reactions": len(kw["reactions"]),
                "n_species": len(kw["mol_specs"]),
                "n_dropped_nan": 0, "n_dropped_overlap": 0}

    monkeypatch.setattr(eh, "run_full_holdout_eval", _capture)
    return spec, cfg, ckpt_dir, seen


def test_sliced_channel_is_marked_before_the_evaluation_runs(run_dir,
                                                             monkeypatch):
    """The mark must survive an evaluation that dies: it is written before the
    energies, so an interrupted sliced channel is still unmistakable."""
    from xcquinox.pipeline.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    import xcquinox.pipeline.eval_holdout as eh
    monkeypatch.setenv(HELDOUT_SPECIES_SLICE_ENV, "h,h2")
    spec, cfg, ckpt_dir, _seen = _slice_fixture(monkeypatch, run_dir)

    def _boom(**kw):
        raise RuntimeError("synthetic eval failure")

    monkeypatch.setattr(eh, "run_full_holdout_eval", _boom)
    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir,
                          os.path.join(ckpt_dir, "model.eqx"), spec)
    chan = os.path.join(ckpt_dir, "eval_holdout")
    assert os.path.isfile(os.path.join(chan, "failure.json"))
    assert not os.path.exists(os.path.join(chan, "eval_metadata.json"))
    with open(os.path.join(chan, "sliced_eval.json")) as f:
        mark = json.load(f)
    assert mark["species_slice"] == ["h", "h2"]
    assert mark["env_var"] == HELDOUT_SPECIES_SLICE_ENV


# ---------------------------------------------------------------------------
# An empty post-split reaction set is refused, not averaged
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A channel directory holds the output of its last pass only
# ---------------------------------------------------------------------------

