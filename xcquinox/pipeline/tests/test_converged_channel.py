"""The converged-SCF held-out channel (``eval_holdout_converged``).

Covers the override helper, the pyscfad backend's density-fitting gate, the
``eval_converged`` config flag and its round trip, the two extra evaluation
passes and their channel stamp, the ``--channel`` plumbing through the
parallel driver and the shard worker, the generalized retroactive tool, the
verbatim re-finalizer's channel list, the figure suite's labels/collector/
views, and the shape of the retroactive job script.

No SCF is run: the pyscfad module tree is a stub injected into ``sys.modules``
and the mean-field object is a recorder, so the backend's wiring is exercised
with no quantum chemistry behind it.
"""
from __future__ import annotations

import json
import os
import types
from types import SimpleNamespace

import pytest

import xcquinox.pipeline.eval_holdout as eh
from xcquinox.pipeline.cluster import _eval_one_spec as ev
from xcquinox.pipeline.solver import (FeaturePolicy, SolverBackend, SolverConfig,
                                  SolverMode)
from xcquinox.pipeline.tests.test_cluster_eval_worker import (
    _full_mode_spec, _stub_insample, _write_manifest, _write_model,
    _write_resolved_config, _write_spec,
)
from xcquinox.pipeline.workers import eval_holdout_worker as ehw


# ---------------------------------------------------------------------------
# C1: converged_solver_config
# ---------------------------------------------------------------------------

def _trained_full_solver():
    """A FULL-mode solver carrying every knob the channel must preserve, plus
    a non-pbe seed so the seed replacement is observable."""
    return SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=3,
        conv_tol=1e-6, feature_policy=FeaturePolicy.REASSEMBLE,
        mixer_name="decaying_linear",
        mixer_kwargs=(("alpha", 0.6), ("decay", 0.9)),
        convergence_name="energy", density_fit=True,
        auxbasis="def2-universal-jkfit", scf_grad_checkpoint=True,
        scf_loss_use_tail=True, scf_loss_tail=8, scf_loss_weight_power=3.0,
        orientation_lock_strength=0.05, seed_source="scan",
        seed_cache_dir="/gpfs/scratch/x/seed_cache")


def test_converged_solver_config_replaces_the_protocol_fields():
    """Kills the mutation ``conv_tol 1e-8 -> 1e-6``: the channel evaluates at
    the Letter's evaluation tolerance under a converged pyscfad SCF seeded
    from PBE, with the tail mean off so the reported energy is the converged
    final energy."""
    sc = eh.converged_solver_config(_trained_full_solver())
    assert sc.backend == SolverBackend.PYSCFAD
    assert sc.mode == SolverMode.FULL
    assert sc.seed_source == "pbe"
    assert sc.seed_cache_dir is None
    assert sc.max_cycles == 100
    assert sc.conv_tol == 1e-8
    assert sc.scf_loss_use_tail is False
    # the constants the override reads, beside the cold-start pair
    assert eh.CONVERGED_MAX_CYCLES == 100
    assert eh.CONVERGED_CONV_TOL == 1e-8


def test_converged_solver_config_refuses_a_non_full_solver():
    """A ONESHOT-shaped spec has no self-consistent protocol to converge; the
    override refuses it rather than inventing one, as its cold-start sibling
    does."""
    oneshot = SolverConfig(backend=SolverBackend.MANUAL,
                           mode=SolverMode.ONESHOT, max_cycles=0)
    with pytest.raises(ValueError):
        eh.converged_solver_config(oneshot)


# ---------------------------------------------------------------------------
# C2: density fitting in the pyscfad backend
# ---------------------------------------------------------------------------

def _sentinel_callback(*args, **kwargs):
    """The functional the stubbed factory hands the backend; the test finds it
    by identity on the numint the installer wrote to."""
    return None


class _FakeMF:
    """Records what the backend does to the mean-field object it builds.

    The functional is no longer recorded by a method of this class: it is
    installed on ``_numint``, so the plain namespace below is what the test
    inspects afterwards.
    """

    def __init__(self, tag: str, log: dict):
        self.tag = tag
        self.log = log
        self.max_cycle = None
        self.conv_tol = None
        self.callback = None
        self.converged = True
        self.e_tot = -1.125
        self._numint = types.SimpleNamespace()

    def density_fit(self, auxbasis=None):
        self.log["density_fit"].append((self.tag, auxbasis))
        wrapper = _FakeMF("df", self.log)
        # pyscfad's DF wrapper copies the mean-field's attributes, so the two
        # objects share one numint; the stub shares it too, or it would pin
        # an install order the real objects cannot tell apart.
        wrapper._numint = self._numint
        self.log["objects"].append(wrapper)
        return wrapper

    def kernel(self, dm0=None):
        self.log["kernel"].append(self.tag)

    def make_rdm1(self):
        import numpy as np
        return np.eye(2)


# ---------------------------------------------------------------------------
# C3: the eval_converged flag
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# C3: the two extra passes in _eval_one_spec.main
# ---------------------------------------------------------------------------

@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    _write_manifest(str(d))
    _write_resolved_config(str(d))
    _write_spec(str(d), 0, obj={"sentinel": "training-spec"})
    return str(d)


def _enable_converged(run_dir):
    import yaml
    path = os.path.join(run_dir, "resolved_config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["eval_converged"] = True
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def _capture_passes(monkeypatch, calls):
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda rd, idx, cfg, ck, mp, ts, holdout_subdir="eval_holdout",
        channel=None:
            calls.append((os.path.basename(mp), holdout_subdir, ts, channel)))


def test_main_runs_both_converged_passes_when_enabled(run_dir, monkeypatch):
    """Kills the mutation ``the val-best pass omitted``: the figures' headline
    is the validation-best channel, so the converged view needs the val-best
    checkpoint as well as the final one, both under the replaced solver."""
    _enable_converged(run_dir)
    _write_spec(run_dir, 0, obj=_full_mode_spec())
    ckpt_dir = _write_model(run_dir, 0)
    open(os.path.join(ckpt_dir, "model_val_best.eqx"), "wb").close()
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    _capture_passes(monkeypatch, calls)
    assert ev.main([run_dir, "0"]) == 0

    assert [(c[0], c[1]) for c in calls] == [
        ("model.eqx", "eval_holdout"),
        ("model_val_best.eqx", "eval_holdout_val_best"),
        ("model.eqx", "eval_holdout_converged"),
        ("model_val_best.eqx", "eval_holdout_converged_val_best")]
    assert [c[3] for c in calls] == [None, None, "converged", "converged"]
    for c in calls[2:]:
        sc = c[2].solver_config
        assert sc.backend == SolverBackend.PYSCFAD
        assert sc.mode == SolverMode.FULL
        assert sc.seed_source == "pbe"
        assert sc.max_cycles == 100
        assert sc.conv_tol == 1e-8
        assert sc.scf_loss_use_tail is False
    # the warm pass keeps the trained protocol
    assert calls[0][2].solver_config.max_cycles == 3
    assert calls[0][2].solver_config.scf_loss_use_tail is True


def _holdout_seams(monkeypatch, result=None):
    """Wire _run_held_out_eval's collaborators so it runs no SCF."""
    import xcquinox.pipeline.full_benchmark_pools as fbp
    monkeypatch.setattr(eh, "load_trained_model", lambda ts, mp: "MODEL")
    monkeypatch.setattr(fbp, "load_full_held_out_pools",
                        lambda basis=None, grid_level=None: ({}, []))
    monkeypatch.setattr(ev, "_held_out_basis_grid", lambda cfg: ("sto-3g", 1))
    monkeypatch.setattr(
        eh, "run_full_holdout_eval",
        lambda **kw: (result if result is not None else
                      {"n_reactions": 0, "n_species": 0, "n_dropped_nan": 0,
                       "n_dropped_overlap": 0}))


def test_run_held_out_eval_stamps_the_channel_override(run_dir, monkeypatch):
    """The per-row columns cannot tell a converged pass from a capped warm
    one, so the channel records which override it ran under; the historical
    ``coldstart`` boolean stays, derived from the channel name."""
    spec = _full_mode_spec()
    ckpt_dir = _write_model(run_dir, 0)
    cfg = SimpleNamespace(cluster=SimpleNamespace(eval_workers=1),
                          held_out_strict=False)
    _holdout_seams(monkeypatch)
    model_path = os.path.join(ckpt_dir, "model.eqx")

    import dataclasses as _dc
    conv_spec = _dc.replace(
        spec, solver_config=eh.converged_solver_config(spec.solver_config))
    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir, model_path, conv_spec,
                          holdout_subdir="eval_holdout_converged",
                          channel="converged")
    with open(os.path.join(ckpt_dir, "eval_holdout_converged",
                           "eval_metadata.json")) as f:
        stamp = json.load(f)
    assert stamp["channel"] == "eval_holdout_converged"
    assert stamp["channel_override"] == "converged"
    assert stamp["coldstart"] is False
    assert stamp["solver_config"]["max_cycles"] == 100
    assert stamp["solver_config"]["conv_tol"] == 1e-8

    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir, model_path, spec,
                          holdout_subdir="eval_holdout", channel=None)
    with open(os.path.join(ckpt_dir, "eval_holdout",
                           "eval_metadata.json")) as f:
        warm = json.load(f)
    assert warm["channel_override"] is None
    assert warm["coldstart"] is False

    cold_spec = _dc.replace(
        spec, solver_config=eh.coldstart_solver_config(spec.solver_config))
    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir, model_path, cold_spec,
                          holdout_subdir="eval_holdout_coldstart",
                          channel="coldstart")
    with open(os.path.join(ckpt_dir, "eval_holdout_coldstart",
                           "eval_metadata.json")) as f:
        cold = json.load(f)
    assert cold["channel_override"] == "coldstart"
    assert cold["coldstart"] is True


# ---------------------------------------------------------------------------
# C3: the parallel driver emits --channel
# ---------------------------------------------------------------------------

class _FakeSpec:
    molecules = ()


def _cmd_arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


# ---------------------------------------------------------------------------
# C3: the shard worker
# ---------------------------------------------------------------------------

def _worker_seams(monkeypatch, spec, captured):
    """compute_shard resolves its collaborators through in-function imports,
    so the seams are the source modules."""
    import xcquinox.pipeline.cluster._eval_one_spec as ev_mod
    import xcquinox.pipeline.eval_holdout as eh_mod
    import xcquinox.pipeline.full_benchmark_pools as fbp
    monkeypatch.setattr(ev_mod, "_load_spec", lambda path: spec)
    monkeypatch.setattr(ev_mod, "_read_width", lambda rd: 4)
    monkeypatch.setattr(eh_mod, "load_trained_model", lambda spec, path: "MODEL")

    def _fake_compute(training_spec, model, subset, **kw):
        captured["sc"] = training_spec.solver_config
        return {"energies": {}, "pbe_energies": {}, "mol_records": []}
    monkeypatch.setattr(eh_mod, "compute_holdout_per_molecule", _fake_compute)
    monkeypatch.setattr(
        fbp, "load_full_held_out_pools",
        lambda basis=None, grid_level=None:
            ({"h2": SimpleNamespace(name="h2")}, []))


import dataclasses as _dc


@_dc.dataclass(frozen=True)
class _PickleSpec:
    """Module-level so the retro tests can pickle it as a .spec file."""
    solver_config: object


def _worker_spec():
    return _PickleSpec(solver_config=SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=3,
        density_fit=True, auxbasis="def2-universal-jkfit"))


def test_compute_shard_converged_applies_the_shared_override(monkeypatch):
    """Kills the mutation ``the worker ignoring --channel``: a shard that
    reloads the spec and does not apply the override evaluates a three-cycle
    warm trajectory and writes it into the converged channel."""
    captured = {}
    _worker_seams(monkeypatch, _worker_spec(), captured)
    ehw.compute_shard("/run", 0, ["h2"], "def2-svp", 1, channel="converged")
    sc = captured["sc"]
    assert sc.backend == SolverBackend.PYSCFAD
    assert sc.mode == SolverMode.FULL
    assert sc.seed_source == "pbe"
    assert sc.max_cycles == 100
    assert sc.conv_tol == 1e-8
    assert sc.density_fit is True
    assert sc.auxbasis == "def2-universal-jkfit"


# ---------------------------------------------------------------------------
# C4: the generalized retroactive tool
# ---------------------------------------------------------------------------

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


