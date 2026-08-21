"""Retroactive cold-start channel driver (cluster/coldstart_retro):
completed/done predicates, skip semantics, dry-run."""
import json
import os

import dataclasses

import pytest

from xcquinox.alec.cluster import coldstart_retro as cr


@dataclasses.dataclass(frozen=True)
class _PickleSpec:
    solver_config: object


def _mk_run(tmp_path, n=3, width=4):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    with open(run / "manifest.json", "w") as f:
        json.dump({"width": width}, f)
    dirs = []
    for i in range(n):
        d = run / "checkpoints" / f"spec_{i:0{width}d}"
        d.mkdir()
        dirs.append(d)
    return run, dirs


def test_spec_status_predicates(tmp_path):
    run, (pending, ready, done) = _mk_run(tmp_path)
    (ready / "model.eqx").write_bytes(b"x")
    (done / "model.eqx").write_bytes(b"x")
    (done / "eval_holdout_coldstart").mkdir()
    (done / "eval_holdout_coldstart" / "per_reaction.json").write_text("[]")
    assert cr.spec_status(str(pending)) == "pending"
    assert cr.spec_status(str(ready)) == "ready"
    assert cr.spec_status(str(done)) == "done"


def test_coldstart_one_spec_skips_pending_and_done(tmp_path, monkeypatch):
    run, (pending, ready, done) = _mk_run(tmp_path)
    (done / "model.eqx").write_bytes(b"x")
    (done / "eval_holdout_coldstart").mkdir()
    (done / "eval_holdout_coldstart" / "per_reaction.json").write_text("[]")
    import xcquinox.alec.cluster._eval_one_spec as ev
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda *a, **k: pytest.fail("eval ran on a skip path"))
    assert cr.coldstart_one_spec(str(run), 0) == "pending"
    assert cr.coldstart_one_spec(str(run), 2) == "done"


def test_coldstart_one_spec_runs_ready_with_override(tmp_path, monkeypatch):
    run, (ready, _b, _c) = _mk_run(tmp_path)
    (ready / "model.eqx").write_bytes(b"x")
    # minimal spec with a FULL solver + a loadable resolved config
    import pickle

    from xcquinox.alec.solver import SolverBackend, SolverConfig, SolverMode

    spec = _PickleSpec(solver_config=SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=3))
    (run / "specs").mkdir()
    with open(run / "specs" / "spec_0000.spec", "wb") as f:
        pickle.dump(spec, f)

    import xcquinox.alec.cluster.coldstart_retro as crmod
    import xcquinox.alec.cluster._eval_one_spec  # noqa: F401
    calls = {}

    def _fake_eval(rd, idx, cfg, ck, mp, ts, holdout_subdir=None,
                   coldstart=False):
        calls.update(subdir=holdout_subdir, coldstart=coldstart,
                     sc=ts.solver_config, model=os.path.basename(mp))

    monkeypatch.setattr(
        "xcquinox.alec.cluster._eval_one_spec._run_held_out_eval",
        _fake_eval)
    monkeypatch.setattr(
        "xcquinox.alec.cluster.grid_config.load_grid_config",
        lambda path: object())
    assert cr.coldstart_one_spec(str(run), 0) == "ran"
    assert calls["subdir"] == "eval_holdout_coldstart"
    assert calls["coldstart"] is True
    assert calls["model"] == "model.eqx"
    assert calls["sc"].seed_source == "minao"
    assert calls["sc"].max_cycles == 25


def test_main_dry_run_reports_without_running(tmp_path, monkeypatch, capsys):
    run, (pending, ready, _c) = _mk_run(tmp_path)
    (ready / "model.eqx").write_bytes(b"x")
    import xcquinox.alec.cluster._eval_one_spec as ev
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda *a, **k: pytest.fail("eval ran under --dry-run"))
    assert cr.main([str(run), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would run" in out
    assert "2 pending" in out and "1 ready" in out


def test_main_flags_non_run_dir(tmp_path, capsys):
    assert cr.main([str(tmp_path / "nope")]) == 1
    assert "not a run dir" in capsys.readouterr().out


_SLICE_MARK = json.dumps(
    {"species_slice": ["h", "h2", "o", "oh", "n2o", "n2ohts"],
     "n_species": 6, "n_reactions": 1,
     "env_var": "XCQUINOX_HELDOUT_SPECIES_SLICE"})


def test_spec_status_refuses_a_sliced_channel_reported_done(tmp_path):
    """A sliced cold-start channel must not read as "done": the retro pass
    would leave a six-species workflow slice standing as this spec's
    cold-start trajectory, and every later reader would take it for the
    pool's."""
    from xcquinox.alec.eval_holdout import SlicedChannelError
    run, (_pending, _ready, done) = _mk_run(tmp_path)
    (done / "model.eqx").write_bytes(b"x")
    chan = done / "eval_holdout_coldstart"
    chan.mkdir()
    (chan / "per_reaction.json").write_text("[]")
    (chan / "sliced_eval.json").write_text(_SLICE_MARK)
    with pytest.raises(SlicedChannelError) as exc:
        cr.spec_status(str(done))
    msg = str(exc.value)
    assert "spec_0002" in msg
    assert "eval_holdout_coldstart" in msg
    assert "'n2ohts'" in msg


def test_spec_status_refuses_an_interrupted_sliced_channel(tmp_path):
    """Marker present, energies never written: without the refusal the spec
    reads "ready" and the retro pass silently writes pool rows beside a
    marker that says the channel is a slice."""
    from xcquinox.alec.eval_holdout import SlicedChannelError
    run, (_pending, ready, _done) = _mk_run(tmp_path)
    (ready / "model.eqx").write_bytes(b"x")
    chan = ready / "eval_holdout_coldstart"
    chan.mkdir()
    (chan / "sliced_eval.json").write_text(_SLICE_MARK)
    with pytest.raises(SlicedChannelError):
        cr.spec_status(str(ready))
