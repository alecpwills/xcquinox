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
    def pools_loader(basis, grid_level):
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
