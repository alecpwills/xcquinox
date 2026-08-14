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
import json
import os
import pickle  # noqa: S403 - round-trips this test's own in-process spec fixtures
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
# WS6: eval gating when training is INCOMPLETE (resume in progress)
# ---------------------------------------------------------------------------

def test_no_model_eqx_with_resume_state_reports_incomplete(run_dir,
                                                           monkeypatch):
    """WS6: when a resume checkpoint is present (resume_state.pkl) but no
    model.eqx yet, eval was scheduled before training finished/resumed. The
    skip marker must CLEARLY say training is incomplete / resume in progress
    (not a confusing crash), still exit 0, and still NOT construct a TestSpec."""
    # resume_state.pkl present, model.eqx absent -> mid-run.
    ckpt_dir = os.path.join(run_dir, "checkpoints", "spec_0000")
    os.makedirs(ckpt_dir, exist_ok=True)
    open(os.path.join(ckpt_dir, "resume_state.pkl"), "wb").close()

    monkeypatch.setattr(
        ev, "_run_eval", lambda ts: pytest.fail("_run_eval ran on skip path"))
    import xcquinox.alec.cluster.spec_builder as sb
    monkeypatch.setattr(
        sb, "build_test_spec",
        lambda *a, **k: pytest.fail("build_test_spec ran on incomplete path"))

    rc = ev.main([run_dir, "0"])
    assert rc == 0

    skipped_path = os.path.join(ckpt_dir, "eval", "skipped.json")
    assert os.path.isfile(skipped_path)
    with open(skipped_path) as f:
        payload = json.load(f)
    reason = payload["reason"]
    # Backward compatible: still mentions the missing model.eqx ...
    assert "no model.eqx" in reason
    # ... but ALSO flags the in-progress resume so the operator does not mistake
    # it for a genuinely-never-trained spec.
    assert "incomplete" in reason.lower()
    assert "resume" in reason.lower()


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
    import xcquinox.alec.cluster.spec_builder as sb
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


def test_main_skips_best_held_out_eval_when_absent(run_dir, monkeypatch):
    # No model_best.eqx (older run) -> only the final-checkpoint eval runs; the
    # best pass no-ops silently. Backward compatible.
    ckpt_dir = _write_model(run_dir, 0)
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda rd, idx, cfg, ck, mp, ts, holdout_subdir="eval_holdout":
            calls.append((os.path.basename(mp), holdout_subdir)))

    assert ev.main([run_dir, "0"]) == 0
    assert calls == [("model.eqx", "eval_holdout")]


# ---------------------------------------------------------------------------
# WS3 (2026-06-20): report only the held-out TEST slice; eval model_val_best.eqx
# ---------------------------------------------------------------------------

def _val_reactions(n=40):
    return [{"name": f"r{i}", "reactants": [f"A{i}"], "products": [f"B{i}"],
             "coeffs": [-1.0, 1.0], "reaction_energy_ref": 0.0}
            for i in range(n)]


def test_test_slice_reactions_filters_to_test_when_validation_enabled():
    """When the spec GENUINELY validated (validate_every>0 AND non-empty
    validation_molecules AND a validation_reactions_path) the reported held-out
    eval is the TEST slice ONLY (the val slice drove early-stop and must not leak
    into the reported metric). split_held_out is deterministic, so the kept set is
    the exact complement of the val slice for the same val_frac."""
    from xcquinox.alec.cluster._eval_one_spec import _test_slice_reactions
    from xcquinox.alec.eval_holdout import split_held_out

    class _Spec:
        validate_every = 2
        val_frac = 0.2
        validation_molecules = ("A", "B")   # non-empty -> validation ran
        validation_reactions_path = "/run/validation/val_reactions.json"
    reactions = _val_reactions(40)
    val, test = split_held_out(reactions, val_frac=0.2)
    kept = _test_slice_reactions(reactions, _Spec())
    kept_names = {r["name"] for r in kept}
    assert kept_names == {r["name"] for r in test}
    # disjoint from the val slice -> no leakage into the reported metric.
    assert kept_names.isdisjoint({r["name"] for r in val})
    assert len(kept) < len(reactions)        # the val slice was removed


def test_test_slice_reactions_prefers_recorded_val_slice(tmp_path):
    """A staged validation/val_reactions.json is what training's early-stop
    actually consumed: the reported test slice must be its complement BY
    PHYSICAL IDENTITY -- permuted-reactant twins of a val barrier are excluded
    with it -- regardless of what the current split hash would compute."""
    import json as _json
    from xcquinox.alec.cluster._eval_one_spec import _test_slice_reactions
    val_file = tmp_path / "val_reactions.json"
    val_file.write_text(_json.dumps([
        {"name": "bh76_hf_h_to_hfhts", "reactants": ["hf", "h"],
         "products": ["hfhts"], "reaction_energy_ref": 17.7}]))

    class _Spec:
        validate_every = 2
        val_frac = 0.2
        validation_molecules = ("A", "B")
        validation_reactions_path = str(val_file)
    reactions = [
        {"name": "bh76_h_hf_to_hfhts", "reactants": ["h", "hf"],
         "products": ["hfhts"], "reaction_energy_ref": 17.7},   # twin of val
        {"name": "w411_x_atomization", "reactants": ["x"],
         "products": ["a", "b"], "reaction_energy_ref": 100.0},
    ]
    kept = _test_slice_reactions(reactions, _Spec())
    assert [r["name"] for r in kept] == ["w411_x_atomization"]


def test_test_slice_reactions_noop_when_validation_disabled():
    """validate_every==0 (no validation) -> the full held-out set is reported,
    byte-identical to the pre-WS3 behavior (no silent metric shrink)."""
    from xcquinox.alec.cluster._eval_one_spec import _test_slice_reactions

    class _Spec:
        validate_every = 0
        val_frac = 0.2
        validation_molecules = ()
        validation_reactions_path = None
    reactions = _val_reactions(40)
    kept = _test_slice_reactions(reactions, _Spec())
    assert kept == reactions                 # unchanged, same object content


def test_test_slice_reactions_noop_when_validate_every_set_but_no_val_molecules():
    """FIX 1(a): validate_every>0 but EMPTY validation_molecules (a partial /
    misconfigured spec, or update_scheme='batched' which has NO validation hook)
    -> validation never ran, so the FULL held-out set is reported (no silent,
    non-comparable ~20% shrink). The eval gate must match the train-side
    ACTIVATION conditions, not validate_every alone."""
    from xcquinox.alec.cluster._eval_one_spec import _test_slice_reactions

    class _SpecNoMols:
        validate_every = 2
        val_frac = 0.2
        validation_molecules = ()                  # nothing staged
        validation_reactions_path = "/run/validation/val_reactions.json"

    class _SpecNoPath:
        validate_every = 2
        val_frac = 0.2
        validation_molecules = ("A", "B")
        validation_reactions_path = None           # no reactions wired

    reactions = _val_reactions(40)
    assert _test_slice_reactions(reactions, _SpecNoMols()) == reactions
    assert _test_slice_reactions(reactions, _SpecNoPath()) == reactions


def test_main_runs_val_best_held_out_eval_when_present(run_dir, monkeypatch):
    """model_val_best.eqx present -> a THIRD held-out pass runs on it into
    eval_holdout_val_best/ (mirrors the model_best.eqx -> eval_holdout_best/
    pass). Final + best + val_best = 3 passes."""
    ckpt_dir = _write_model(run_dir, 0)
    open(os.path.join(ckpt_dir, "model_best.eqx"), "wb").close()
    open(os.path.join(ckpt_dir, "model_val_best.eqx"), "wb").close()
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda rd, idx, cfg, ck, mp, ts, holdout_subdir="eval_holdout":
            calls.append((os.path.basename(mp), holdout_subdir)))

    assert ev.main([run_dir, "0"]) == 0
    assert calls == [("model.eqx", "eval_holdout"),
                     ("model_best.eqx", "eval_holdout_best"),
                     ("model_val_best.eqx", "eval_holdout_val_best")]


def test_main_skips_val_best_held_out_eval_when_absent(run_dir, monkeypatch):
    """No model_val_best.eqx (validation disabled / older run) -> the val_best
    pass no-ops; final (+ best if present) only."""
    ckpt_dir = _write_model(run_dir, 0)
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda rd, idx, cfg, ck, mp, ts, holdout_subdir="eval_holdout":
            calls.append((os.path.basename(mp), holdout_subdir)))

    assert ev.main([run_dir, "0"]) == 0
    assert ("model_val_best.eqx", "eval_holdout_val_best") not in calls


# ---------------------------------------------------------------------------
# fold helper -- correct per-molecule row keys
# ---------------------------------------------------------------------------

def test_aggregate_per_molecule_reads_canonical_keys():
    pm_rows = [
        {"molecule": "A", "AE_error_kcalmol": 2.0, "density_rmse": 0.4},
        {"molecule": "B", "AE_error_kcalmol": -4.0, "density_rmse": 0.6},
    ]
    mae, rho_rmse, n_eval, rho_rmse_pbe = ev._aggregate_per_molecule(pm_rows)
    assert mae == pytest.approx(3.0)
    assert rho_rmse == pytest.approx(0.5)
    assert n_eval == 2
    # no density_rmse_pbe key on any row (historical schema) -> nan
    import math
    assert math.isnan(rho_rmse_pbe)


def test_aggregate_per_molecule_pbe_density_channel():
    import math
    pm_rows = [
        {"molecule": "A", "AE_error_kcalmol": 1.0, "density_rmse": 0.4,
         "density_rmse_pbe": 0.8},
        {"molecule": "B", "AE_error_kcalmol": 2.0, "density_rmse": 0.6,
         "density_rmse_pbe": None},
        {"molecule": "C", "AE_error_kcalmol": 3.0, "density_rmse": 0.2,
         "density_rmse_pbe": float("nan")},
    ]
    mae, rho_rmse, n_eval, rho_rmse_pbe = ev._aggregate_per_molecule(pm_rows)
    # only A's finite numeric value contributes to the PBE baseline mean
    assert rho_rmse_pbe == pytest.approx(0.8)
    assert not math.isnan(rho_rmse)


def test_aggregate_per_molecule_nan_when_keys_absent():
    import math
    # No AE_error_kcalmol / density_rmse keys at all (atomic systems etc.).
    pm_rows = [{"molecule": "H"}, {"molecule": "Li"}]
    mae, rho_rmse, n_eval, rho_rmse_pbe = ev._aggregate_per_molecule(pm_rows)
    assert math.isnan(mae)
    assert math.isnan(rho_rmse)
    # CODE-03: n_eval is the AE-contributing count (MAE denominator); with no
    # AE_error_kcalmol on any row, no molecule contributes -> 0 (not total rows).
    assert n_eval == 0


def test_aggregate_per_molecule_skips_none_and_bool():
    pm_rows = [
        {"molecule": "A", "AE_error_kcalmol": 6.0, "density_rmse": None},
        {"molecule": "B", "AE_error_kcalmol": None, "density_rmse": 0.3},
        {"molecule": "C", "AE_error_kcalmol": True, "density_rmse": False},
    ]
    mae, rho_rmse, n_eval, rho_rmse_pbe = ev._aggregate_per_molecule(pm_rows)
    # Only A's AE error and B's density_rmse are numeric non-bool.
    assert mae == pytest.approx(6.0)
    assert rho_rmse == pytest.approx(0.3)
    # CODE-03: n_eval = AE-contributing count = 1 (only A has a numeric AE),
    # not total rows (3).
    assert n_eval == 1


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


# per-molecule aggregation must exclude non-finite values
def test_aggregate_per_molecule_excludes_nonfinite():
    rows = [
        {"AE_error_kcalmol": 1.0, "density_rmse": 0.01},
        {"AE_error_kcalmol": float("nan"), "density_rmse": float("nan")},
        {"AE_error_kcalmol": float("inf"), "density_rmse": 0.05},
        {"AE_error_kcalmol": -3.0, "density_rmse": 0.03},
    ]
    mae, rho_rmse, n_eval, rho_rmse_pbe = ev._aggregate_per_molecule(rows)
    assert n_eval == 2                      # NaN + inf excluded
    assert abs(mae - 2.0) < 1e-12           # (|1| + |-3|) / 2, finite only
    # density_rmse: mean(0.01, 0.05, 0.03) = 0.03; the NaN value is dropped.
    assert abs(rho_rmse - 0.03) < 1e-12


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# Cold-start channel: 4th held-out pass on the FINAL checkpoint
# ---------------------------------------------------------------------------

def _enable_coldstart(run_dir):
    import yaml
    path = os.path.join(run_dir, "resolved_config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["eval_coldstart"] = True
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def _full_mode_spec():
    """A minimal REAL TrainingSpec with a FULL-mode solver, so the
    orchestrator-side dataclasses.replace has something genuine to act on."""
    import dataclasses as _dc

    import xcquinox.alec as alec
    from xcquinox.alec.config import MoleculeSpec, TrainingSpec
    from xcquinox.alec.solver import (SolverBackend, SolverConfig,
                                      SolverMode)
    mol = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74",
                       basis="sto-3g", charge=0, spin=0,
                       atom_composition=(("H", 2),))
    spec = TrainingSpec.from_dicts(
        arch=alec.get_architecture("deep_3x16"), molecules=(mol,),
        targets={"H2": -1.0}, atom_energies={"H": -0.5},
        loss_name="A_atomization", loss_kwargs={"vxc_weight": 0.0},
        update_scheme="per_molecule", require_atom_anchors=False,
        n_steps=1, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
        grad_clip=1.0, checkpoint_dir=None, seed=42)
    solver = SolverConfig(backend=SolverBackend.MANUAL,
                          mode=SolverMode.FULL, max_cycles=3,
                          scf_loss_use_tail=True)
    return _dc.replace(spec, solver_config=solver)


def test_main_runs_coldstart_pass_when_enabled(run_dir, monkeypatch):
    """eval_coldstart: true + FULL-mode spec -> a 4th pass on model.eqx into
    eval_holdout_coldstart/ with the spec REPLACED orchestrator-side
    (minao seed, 25 cycles, conv_tol 1e-12, mode stays FULL) and the
    coldstart flag threaded toward the shard workers."""
    _enable_coldstart(run_dir)
    _write_spec(run_dir, 0, obj=_full_mode_spec())
    ckpt_dir = _write_model(run_dir, 0)
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda rd, idx, cfg, ck, mp, ts, holdout_subdir="eval_holdout",
        coldstart=False:
            calls.append((os.path.basename(mp), holdout_subdir, ts,
                          coldstart)))

    assert ev.main([run_dir, "0"]) == 0
    assert [(c[0], c[1]) for c in calls] == [
        ("model.eqx", "eval_holdout"),
        ("model.eqx", "eval_holdout_coldstart")]
    warm_ts = calls[0][2]
    cold_ts = calls[1][2]
    assert calls[0][3] is False and calls[1][3] is True
    # warm pass keeps the trained protocol
    assert warm_ts.solver_config.seed_source == "pbe"
    assert warm_ts.solver_config.max_cycles == 3
    # cold pass: the shared override, applied BEFORE dispatch so the serial
    # tiers inherit it too
    sc = cold_ts.solver_config
    assert sc.seed_source == "minao"
    assert sc.max_cycles == 25
    assert sc.conv_tol == 1e-12
    assert sc.mode.value == "full"
    # everything else preserved from the trained solver
    assert sc.scf_loss_use_tail is True


def test_main_coldstart_skips_specs_without_full_solver(run_dir, monkeypatch):
    """eval_coldstart: true with a spec that has no FULL-mode solver_config
    (legacy/sentinel) -> no 4th pass, no crash."""
    _enable_coldstart(run_dir)
    ckpt_dir = _write_model(run_dir, 0)
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda rd, idx, cfg, ck, mp, ts, holdout_subdir="eval_holdout",
        coldstart=False:
            calls.append(holdout_subdir))

    assert ev.main([run_dir, "0"]) == 0
    assert calls == ["eval_holdout"]


def test_main_no_coldstart_by_default(run_dir, monkeypatch):
    """Without the flag the channel set is byte-identical to before."""
    _write_spec(run_dir, 0, obj=_full_mode_spec())
    ckpt_dir = _write_model(run_dir, 0)
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda rd, idx, cfg, ck, mp, ts, holdout_subdir="eval_holdout",
        coldstart=False:
            calls.append(holdout_subdir))

    assert ev.main([run_dir, "0"]) == 0
    assert calls == ["eval_holdout"]
