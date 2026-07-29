"""Tests for the idempotent held-out re-eval driver ``reeval_holdout_fixed``.

The heavy compute (spec/model load, pool build, SCF eval) is injected as stubs
so these tests run instantly and assert the *orchestration*: discovery, the
stamp-based smart-skip, restriction to specific specs, and that a re-run is a
no-op until a new (unstamped) spec appears.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "reeval_holdout_fixed", _HERE / "reeval_holdout_fixed.py")
rh = importlib.util.module_from_spec(_spec)
sys.modules["reeval_holdout_fixed"] = rh
_spec.loader.exec_module(rh)


# ---------------------------------------------------------------------------
# Fixture + stubs
# ---------------------------------------------------------------------------

def _make_run(root: Path, trained=(0, 1, 2), untrained=(3,)) -> Path:
    run_dir = root / "run_20260529T165503Z"
    (run_dir / "specs").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"width": 4}))
    for i in list(trained) + list(untrained):
        sd = run_dir / "checkpoints" / f"spec_{i:04d}"
        sd.mkdir(parents=True)
        (run_dir / "specs" / f"spec_{i:04d}.spec").write_bytes(b"x")
        if i in trained:
            (sd / "model.eqx").write_bytes(b"x")
    return run_dir


def _stub_inject(calls: list, precompute_calls: Optional[list] = None):
    """Injected callables that record processed indices and write a real
    per_reaction.json so the artifacts look produced. ``precompute_calls``, if
    given, records each precompute invocation (to assert once-per-group)."""
    def pools_loader(basis, grid_level, refs_dir=None):
        return ({"h": object()}, [])

    def spec_loader(spec_path):
        return object()

    def model_loader(spec, model_path):
        return object()

    def precompute_fn(training_spec, mol_specs):
        if precompute_calls is not None:
            precompute_calls.append(1)
        return {"h": {"E_pbe": -0.5}}

    def eval_fn(spec, model, mol_specs, reactions, out_dir, mol_data):
        assert mol_data is not None, "eval_fn must receive shared mol_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "per_reaction.json").write_text("[]")
        calls.append(out_dir.parent.name)
        return {"n_reactions": 3, "n_species": 5, "n_dropped_nan": 0,
                "combined": (1.0, 2.0, 3, 0, 0)}

    return dict(pools_loader=pools_loader, spec_loader=spec_loader,
                model_loader=model_loader, precompute_fn=precompute_fn,
                eval_fn=eval_fn)


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------

def test_discover_trained_specs_skips_untrained(tmp_path):
    run = _make_run(tmp_path, trained=(0, 2), untrained=(1, 3))
    assert rh.discover_trained_specs(run) == [0, 2]


def test_needs_reeval_true_when_unstamped(tmp_path):
    run = _make_run(tmp_path, trained=(0,))
    sd = run / "checkpoints" / "spec_0000"
    assert rh.needs_reeval(sd) is True


def test_needs_reeval_false_when_stamped_current(tmp_path):
    run = _make_run(tmp_path, trained=(0,))
    sd = run / "checkpoints" / "spec_0000"
    rh.write_stamp(sd, {"n_reactions": 3})
    assert rh.read_stamp(sd) == rh.REEVAL_VERSION
    assert rh.needs_reeval(sd) is False
    assert rh.needs_reeval(sd, force=True) is True


def test_needs_reeval_false_without_model(tmp_path):
    run = _make_run(tmp_path, trained=(), untrained=(0,))
    sd = run / "checkpoints" / "spec_0000"
    assert rh.needs_reeval(sd) is False


# ---------------------------------------------------------------------------
# Driver orchestration (stubbed compute)
# ---------------------------------------------------------------------------

def test_run_processes_all_trained_then_skips_on_rerun(tmp_path):
    run = _make_run(tmp_path, trained=(0, 1, 2), untrained=(3,))
    calls: list = []
    inj = _stub_inject(calls)

    res1 = rh.run(run, clock=lambda: 0.0, **inj)
    assert sorted(res1["processed"]) == [0, 1, 2]
    assert res1["failed"] == []
    # Each processed spec now carries the stamp.
    for i in (0, 1, 2):
        assert rh.read_stamp(run / "checkpoints" / f"spec_{i:04d}") == \
            rh.REEVAL_VERSION

    # Second run is a no-op (all stamped); no new eval_fn calls.
    calls.clear()
    res2 = rh.run(run, clock=lambda: 0.0, **inj)
    assert res2["processed"] == []
    assert sorted(res2["skipped"]) == [0, 1, 2]
    assert calls == []


def test_run_picks_up_newly_arrived_spec(tmp_path):
    run = _make_run(tmp_path, trained=(0,), untrained=())
    inj = _stub_inject([])
    rh.run(run, clock=lambda: 0.0, **inj)  # process spec 0

    # A new spec finishes/downloads later.
    sd = run / "checkpoints" / "spec_0005"
    sd.mkdir(parents=True)
    (run / "specs" / "spec_0005.spec").write_bytes(b"x")
    (sd / "model.eqx").write_bytes(b"x")

    res = rh.run(run, clock=lambda: 0.0, **inj)
    assert res["processed"] == [5]
    assert 0 in res["skipped"]


def test_run_restricts_to_only_specs(tmp_path):
    run = _make_run(tmp_path, trained=(0, 1, 2), untrained=())
    res = rh.run(run, only_specs=[1], clock=lambda: 0.0, **_stub_inject([]))
    assert res["processed"] == [1]


def test_run_records_failure_and_continues(tmp_path):
    run = _make_run(tmp_path, trained=(0, 1), untrained=())
    inj = _stub_inject([])

    def boom(spec, model, mol_specs, reactions, out_dir, mol_data):
        if out_dir.parent.name.endswith("0000"):
            raise RuntimeError("kaboom")
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"n_reactions": 1, "combined": (1.0, 2.0, 1, 0, 0)}

    inj["eval_fn"] = boom
    res = rh.run(run, clock=lambda: 0.0, **inj)
    assert res["failed"] == [0]
    assert res["processed"] == [1]
    # Failed spec is NOT stamped, so a later run retries it.
    assert rh.needs_reeval(run / "checkpoints" / "spec_0000") is True


def test_precompute_runs_once_per_group(tmp_path):
    """The optimization: 3 same-signature specs share ONE precompute."""
    run = _make_run(tmp_path, trained=(0, 1, 2), untrained=())
    pre: list = []
    inj = _stub_inject([], precompute_calls=pre)
    res = rh.run(run, clock=lambda: 0.0, **inj)
    assert sorted(res["processed"]) == [0, 1, 2]
    # All three stubs share descriptor_signature -> exactly one precompute.
    assert len(pre) == 1, f"expected 1 precompute, got {len(pre)}"


def test_group_precompute_failure_fails_whole_group(tmp_path):
    run = _make_run(tmp_path, trained=(0, 1), untrained=())
    inj = _stub_inject([])

    def boom_pre(training_spec, mol_specs):
        raise RuntimeError("precompute OOM")

    inj["precompute_fn"] = boom_pre
    res = rh.run(run, clock=lambda: 0.0, **inj)
    assert sorted(res["failed"]) == [0, 1]
    assert res["processed"] == []
    # Neither spec is stamped, so a later run retries them.
    assert rh.needs_reeval(run / "checkpoints" / "spec_0000") is True


# ---------------------------------------------------------------------------
# checkpoint selection (model / model_best / model_val_best)
# ---------------------------------------------------------------------------

def test_eval_subdir_for_mapping():
    import pytest
    assert rh.eval_subdir_for("model") == "eval_holdout"
    assert rh.eval_subdir_for("model_best") == "eval_holdout_best"
    assert rh.eval_subdir_for("model_val_best") == "eval_holdout_val_best"
    with pytest.raises(ValueError):
        rh.eval_subdir_for("model_final")


def test_discover_trained_specs_checkpoint_param(tmp_path):
    run = _make_run(tmp_path, trained=(0, 1), untrained=())
    (run / "checkpoints" / "spec_0001" / "model_val_best.eqx").write_bytes(b"x")
    assert rh.discover_trained_specs(run) == [0, 1]
    assert rh.discover_trained_specs(run, checkpoint="model_val_best") == [1]


def test_needs_reeval_checkpoint_requires_that_checkpoint(tmp_path):
    run = _make_run(tmp_path, trained=(0,))
    sd = run / "checkpoints" / "spec_0000"
    # model.eqx exists but model_val_best.eqx does not
    assert rh.needs_reeval(sd, checkpoint="model_val_best") is False
    (sd / "model_val_best.eqx").write_bytes(b"x")
    assert rh.needs_reeval(sd, checkpoint="model_val_best") is True


def test_run_checkpoint_val_best_isolated_subdir_and_stamp(tmp_path):
    run = _make_run(tmp_path, trained=(0,), untrained=())
    sd = run / "checkpoints" / "spec_0000"
    (sd / "model_val_best.eqx").write_bytes(b"x")
    seen_paths: list = []
    out_dirs: list = []
    inj = _stub_inject([])

    def model_loader(spec, model_path):
        seen_paths.append(model_path.name)
        return object()

    def eval_fn(spec, model, mol_specs, reactions, out_dir, mol_data):
        out_dirs.append(out_dir.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "per_reaction.json").write_text("[]")
        return {"n_reactions": 1, "combined": (1.0, 2.0, 1, 0, 0)}

    inj["model_loader"] = model_loader
    inj["eval_fn"] = eval_fn
    res = rh.run(run, checkpoint="model_val_best", clock=lambda: 0.0, **inj)
    assert res["processed"] == [0]
    assert seen_paths == ["model_val_best.eqx"]
    assert out_dirs == ["eval_holdout_val_best"]
    # stamp lives in the val-best subdir; the default subdir has none
    assert (sd / "eval_holdout_val_best" / "reeval_meta.json").is_file()
    assert not (sd / "eval_holdout" / "reeval_meta.json").exists()
    # the default-checkpoint eval is INDEPENDENT: still needed afterwards
    assert rh.needs_reeval(sd) is True
    # val-best re-run is a no-op (idempotent under its own stamp)
    res2 = rh.run(run, checkpoint="model_val_best", clock=lambda: 0.0, **inj)
    assert res2["processed"] == [] and res2["skipped"] == [0]


def test_main_checkpoint_flag_threads(tmp_path, monkeypatch):
    import pytest
    run = _make_run(tmp_path, trained=(0,))
    seen: dict = {}

    def fake_run(run_dir, **kw):
        seen.update(kw)
        return {"processed": [], "skipped": [], "failed": []}

    monkeypatch.setattr(rh, "run", fake_run)
    rc = rh.main(["--run-dir", str(run), "--checkpoint", "model_val_best"])
    assert rc == 0 and seen["checkpoint"] == "model_val_best"
    with pytest.raises(SystemExit):
        rh.main(["--run-dir", str(run), "--checkpoint", "nope"])


# ---------------------------------------------------------------------------
# density refs + PBE-only mode
# ---------------------------------------------------------------------------

def test_density_stamp_v3_reprocesses_under_current_suffix(tmp_path):
    """Specs stamped +density_refs_v3 (pre-eps schema) must re-process under
    the current density stamp so per_molecule.json gains the DFS Eq. 20
    per-electron L1 columns."""
    run = _make_run(tmp_path, trained=(0,), untrained=())
    sd = run / "checkpoints" / "spec_0000"
    out = sd / "eval_holdout"
    out.mkdir(parents=True)
    (out / "reeval_meta.json").write_text(json.dumps(
        {"geom_units_fix": rh.REEVAL_VERSION + "+density_refs_v3"}))
    assert rh.needs_reeval(sd, version=rh.effective_version("/refs")) is True


def test_refs_free_rerun_does_not_churn_density_stamped_spec(tmp_path):
    """A density-stamped spec satisfies the refs-free base version: a re-run
    without --density-refs must skip it (re-processing would overwrite the
    density columns with the all-None schema and downgrade the stamp)."""
    run = _make_run(tmp_path, trained=(0,), untrained=())
    sd = run / "checkpoints" / "spec_0000"
    out = sd / "eval_holdout"
    out.mkdir(parents=True)
    (out / "reeval_meta.json").write_text(json.dumps(
        {"geom_units_fix": rh.effective_version("/refs")}))
    assert rh.needs_reeval(sd) is False
    # a genuinely NEW base version still re-evals (no prefix collision)
    assert rh.needs_reeval(sd, version=rh.REEVAL_VERSION + "9") is True
    # and force always overrides
    assert rh.needs_reeval(sd, force=True) is True
    # driver-level: a refs-free run skips the density-stamped spec entirely
    calls: list = []
    res = rh.run(run, clock=lambda: 0.0, **_stub_inject(calls))
    assert res["processed"] == [] and res["skipped"] == [0] and calls == []


def test_effective_version_switches_with_density_refs():
    assert rh.effective_version(None) == rh.REEVAL_VERSION
    v = rh.effective_version("/some/refs")
    assert v != rh.REEVAL_VERSION and v.startswith(rh.REEVAL_VERSION)


def test_density_refs_threads_into_pools_loader_and_restamps(tmp_path):
    run = _make_run(tmp_path, trained=(0,), untrained=())
    calls: list = []
    inj = _stub_inject(calls)
    seen_refs: list = []

    def pools_loader(basis, grid_level, refs_dir=None):
        seen_refs.append(refs_dir)
        return ({"h": object()}, [])

    inj["pools_loader"] = pools_loader
    # refs-free run stamps the base version
    rh.run(run, **inj)
    sd = run / "checkpoints" / "spec_0000"
    assert rh.read_stamp(sd) == rh.REEVAL_VERSION
    assert seen_refs == [None]
    # a refs run must RE-process the base-stamped spec and re-stamp with the
    # density version; the refs dir reaches the pools loader
    res = rh.run(run, density_refs="/refs/bench", **inj)
    assert res["processed"] == [0]
    assert seen_refs[-1] == "/refs/bench"
    assert rh.read_stamp(sd) == rh.effective_version("/refs/bench")
    # and a SECOND refs run is a no-op (idempotent under the new stamp)
    res2 = rh.run(run, density_refs="/refs/bench", **inj)
    assert res2["processed"] == [] and res2["skipped"] == [0]


def test_pbe_density_only_needs_no_model(tmp_path):
    """The PBE table must be computable on a summaries-profile pull with ZERO
    local model.eqx files -- and must never touch a model loader."""
    run = _make_run(tmp_path, trained=(), untrained=(0,))
    (run / "resolved_config.yaml").write_text("basis: def2-svp\ngrid_level: 2\n")

    class MS:
        def __init__(self, name, n_atoms, ext):
            self.name = name
            self.atom_composition = (("H", n_atoms),)
            self.external_data_path = ext

    pool = {
        "h": MS("h", 1, "/refs/h.npz"),          # atom -> skipped
        "h2o": MS("h2o", 3, "/refs/h2o.npz"),    # with ref -> computed
        "ch4": MS("ch4", 5, None),               # no ref -> not attempted
        "bad": MS("bad", 2, "/refs/bad.npz"),    # precompute raises -> failure
    }

    def pools_loader(basis, grid_level, refs_dir=None):
        assert refs_dir == "/refs"
        return (pool, [])

    def precompute_one(ms):
        if ms.name == "bad":
            raise RuntimeError("scf diverged")
        import numpy as np
        return {"rho_grid": np.array([1.5, 0.5]),
                "rho_ref_grid": np.array([2.0, 1.0]),
                "grid_weights": np.array([3.0, 1.0])}

    payload = rh.run_pbe_density_table(run, density_refs="/refs",
                                       pools_loader=pools_loader,
                                       precompute_one=precompute_one)
    assert set(payload["errors"]) == {"h2o"}
    assert payload["errors"]["h2o"]["density_rmse_pbe"] == 0.5
    assert payload["errors"]["h2o"]["density_l1_pbe"] == 0.5
    # DFS Eq. 20 per-electron L1: sum(w|drho|)/N_e = (3*0.5+1*0.5)/7 = 2/7,
    # distinct from the weight-mean l1 (2/4) because N_e = sum(w rho_ref) = 7
    import pytest
    assert payload["errors"]["h2o"]["density_eps_l1_pbe"] == \
        pytest.approx(2.0 / 7.0)
    assert payload["errors"]["h2o"]["n_electrons"] == pytest.approx(7.0)
    assert payload["errors"]["h2o"]["grid_weight_sum"] == pytest.approx(4.0)
    assert "bad" in payload["failures"]
    on_disk = json.loads((run / "pbe_density_errors.json").read_text())
    assert on_disk["errors"] == payload["errors"]
    assert on_disk["basis"] == "def2-svp" and on_disk["grid_level"] == 2


def test_main_pbe_density_only_requires_refs(tmp_path):
    import pytest
    run = _make_run(tmp_path, trained=(0,))
    with pytest.raises(SystemExit):
        rh.main(["--run-dir", str(run), "--pbe-density-only"])


def test_pbe_density_table_fast_path_uses_stored_pbe_grid(tmp_path):
    """Benchmark refs carrying rho_pbe_grid + grid_weights must be consumed
    by pure npz arithmetic -- precompute (PBE SCF) must NOT be invoked."""
    import numpy as np
    run = _make_run(tmp_path, trained=(), untrained=(0,))
    (run / "resolved_config.yaml").write_text("basis: def2-svp\ngrid_level: 2\n")
    refs = tmp_path / "refs"
    refs.mkdir()
    np.savez_compressed(refs / "h2o.npz",
                        rho_ref_grid=np.array([2.0, 1.0]),
                        rho_pbe_grid=np.array([1.5, 0.5]),
                        grid_weights=np.array([3.0, 1.0]),
                        ref_density_method=np.array("ccsd"))

    class MS:
        name = "h2o"
        atom_composition = (("H", 2), ("O", 1))
        external_data_path = str(refs / "h2o.npz")

    def pools_loader(basis, grid_level, refs_dir=None):
        return ({"h2o": MS()}, [])

    def precompute_one(ms):
        raise AssertionError("fast path must not run a PBE SCF")

    payload = rh.run_pbe_density_table(run, density_refs=str(refs),
                                       pools_loader=pools_loader,
                                       precompute_one=precompute_one)
    import pytest
    assert payload["errors"]["h2o"]["density_rmse_pbe"] == 0.5
    assert payload["errors"]["h2o"]["density_l1_pbe"] == 0.5
    assert payload["errors"]["h2o"]["density_eps_l1_pbe"] == \
        pytest.approx(2.0 / 7.0)
    assert payload["errors"]["h2o"]["n_electrons"] == pytest.approx(7.0)
    assert payload["errors"]["h2o"]["grid_weight_sum"] == pytest.approx(4.0)
    assert not payload["failures"]
