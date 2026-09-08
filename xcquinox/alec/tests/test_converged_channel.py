"""The converged-SCF held-out channel (``eval_holdout_converged``).

Covers the override helper, the pyscfad backend's density-fitting gate, the
``eval_converged`` config flag and its round trip, the two extra evaluation
passes and their channel stamp, the ``--channel`` plumbing through the
parallel driver and the shard worker, the generalized retroactive tool, the
verbatim re-finalizer's channel list, the figure suite's labels/collector/
views, and the shape of the retroactive job script.

No SCF is run and neither pyscfad nor pyscf is imported: the pyscfad module
tree is a stub injected into ``sys.modules`` and the mean-field object is a
recorder.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import xcquinox.alec.eval_holdout as eh
from xcquinox.alec.cluster import _eval_one_spec as ev
from xcquinox.alec.solver import (FeaturePolicy, SolverBackend, SolverConfig,
                                  SolverMode)
from xcquinox.alec.tests.test_cluster_eval_worker import (
    _full_mode_spec, _stub_insample, _write_manifest, _write_model,
    _write_resolved_config, _write_spec,
)
from xcquinox.alec.workers import eval_holdout_worker as ehw

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANALYSIS = _REPO_ROOT / "notebooks" / "analysis"


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


def test_converged_solver_config_preserves_the_trained_knobs():
    """Everything that is not the evaluation protocol survives the replace:
    a DF-trained cell must be evaluated with its own Coulomb footing and its
    own orientation lock, else the converged density is not this cell's."""
    sc = eh.converged_solver_config(_trained_full_solver())
    assert sc.density_fit is True
    assert sc.auxbasis == "def2-universal-jkfit"
    assert sc.orientation_lock_strength == 0.05
    assert sc.mixer_name == "decaying_linear"
    assert sc.mixer_kwargs == (("alpha", 0.6), ("decay", 0.9))
    assert sc.feature_policy == FeaturePolicy.REASSEMBLE
    assert sc.convergence_name == "energy"
    assert sc.scf_grad_checkpoint is True
    assert sc.scf_loss_tail == 8
    assert sc.scf_loss_weight_power == 3.0


def test_converged_solver_config_refuses_a_non_full_solver():
    """A ONESHOT-shaped spec has no self-consistent protocol to converge; the
    override refuses it rather than inventing one, as its cold-start sibling
    does."""
    oneshot = SolverConfig(backend=SolverBackend.MANUAL,
                           mode=SolverMode.ONESHOT, max_cycles=0)
    with pytest.raises(ValueError):
        eh.converged_solver_config(oneshot)


def test_channel_overrides_maps_both_channel_names():
    """One table maps a channel name to its override, so the orchestrator,
    the shard worker and the retroactive tool cannot drift apart."""
    assert set(eh.CHANNEL_OVERRIDES) == {"coldstart", "converged"}
    assert eh.CHANNEL_OVERRIDES["coldstart"] is eh.coldstart_solver_config
    assert eh.CHANNEL_OVERRIDES["converged"] is eh.converged_solver_config


# ---------------------------------------------------------------------------
# C2: density fitting in the pyscfad backend
# ---------------------------------------------------------------------------

class _FakeMF:
    """Records what the backend does to the mean-field object it builds."""

    def __init__(self, tag: str, log: dict):
        self.tag = tag
        self.log = log
        self.max_cycle = None
        self.conv_tol = None
        self.callback = None
        self.converged = True
        self.e_tot = -1.125

    def density_fit(self, auxbasis=None):
        self.log["density_fit"].append((self.tag, auxbasis))
        return _FakeMF("df", self.log)

    def define_xc_(self, callback, kind):
        self.log["define_xc"].append((self.tag, kind))

    def kernel(self, dm0=None):
        self.log["kernel"].append(self.tag)

    def make_rdm1(self):
        import numpy as np
        return np.eye(2)


def _pyscfad_backend(monkeypatch):
    """Import the backend with a STUB ``pyscfad`` package tree in place, the
    mean-field builder replaced by a recorder and the eval_xc factory by a
    no-op, so the function under test runs with no quantum chemistry."""
    import numpy as np

    pkg = types.ModuleType("pyscfad")
    dft = types.ModuleType("pyscfad.dft")
    gto = types.ModuleType("pyscfad.gto")
    pkg.dft = dft
    pkg.gto = gto
    monkeypatch.setitem(sys.modules, "pyscfad", pkg)
    monkeypatch.setitem(sys.modules, "pyscfad.dft", dft)
    monkeypatch.setitem(sys.modules, "pyscfad.gto", gto)

    import xcquinox.alec.solver_pyscfad as sp

    log = {"density_fit": [], "define_xc": [], "kernel": []}
    monkeypatch.setattr(sp, "_build_pyscfad_mf",
                        lambda mol, mol_data: _FakeMF("base", log))
    monkeypatch.setattr(sp, "_make_alec_eval_xc",
                        lambda **kw: (lambda *a, **k: None))
    mol_data = {
        "_pyscfad_mol": object(),
        "dm_pbe": np.eye(2),
        "rho_grid": np.zeros((3, 4)),
        "mol_metadata": {},
    }
    model = SimpleNamespace(descriptors=())
    return sp, log, model, mol_data


def test_pyscfad_backend_density_fit_follows_the_config(monkeypatch):
    """Kills the mutation ``DF call unconditional``: the DF build is made iff
    the cell was trained with it, with the cell's own auxbasis, and the DF
    wrapper -- not the bare mean-field -- is what the functional and the
    kernel then see. A non-DF cell keeps its full-integral Coulomb; an
    unconditional call would change the Coulomb footing of every non-DF
    cell's converged density."""
    sp, on_log, model, mol_data = _pyscfad_backend(monkeypatch)
    on = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FULL,
                      max_cycles=100, conv_tol=1e-8, density_fit=True,
                      auxbasis="def2-universal-jkfit")
    sp._run_pyscfad_scf_impl(on, model, mol_data)
    assert on_log["density_fit"] == [("base", "def2-universal-jkfit")]
    assert on_log["define_xc"] == [("df", "GGA")]
    assert on_log["kernel"] == ["df"]

    sp, off_log, model, mol_data = _pyscfad_backend(monkeypatch)
    off = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FULL,
                       max_cycles=100, conv_tol=1e-8, density_fit=False)
    sp._run_pyscfad_scf_impl(off, model, mol_data)
    assert off_log["density_fit"] == []
    assert off_log["define_xc"] == [("base", "GGA")]
    assert off_log["kernel"] == ["base"]


# ---------------------------------------------------------------------------
# C3: the eval_converged flag
# ---------------------------------------------------------------------------

def test_eval_converged_defaults_false(tmp_path):
    """A config that never mentions the channel parses to the four-channel
    protocol."""
    from xcquinox.alec.cluster.grid_config import load_grid_config
    from xcquinox.alec.tests.test_cluster_grid_config import (_base_config_dict,
                                                              _write)
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", _base_config_dict()))
    assert cfg.eval_converged is False


def test_eval_converged_resolved_round_trip(tmp_path):
    """Kills the mutation ``eval_converged not round-tripped``: the eval stage
    re-reads ``resolved_config.yaml``, so a dropped flag silently skips the
    channel on every spec of the run (the ae_as_reactions incident class)."""
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
    from xcquinox.alec.cluster.grid_config import load_grid_config
    from xcquinox.alec.tests.test_cluster_grid_config import (_base_config_dict,
                                                              _write)
    d = _base_config_dict()
    d["eval_converged"] = True
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", d))
    assert cfg.eval_converged is True
    cfg2 = load_grid_config(
        _write(tmp_path, "resolved.yaml", _config_to_raw_dict(cfg)))
    assert cfg2.eval_converged is True


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


def test_main_converged_val_best_pass_omitted_when_absent(run_dir, monkeypatch):
    """No ``model_val_best.eqx`` (validation disabled, or an older run) -> the
    final-checkpoint converged pass only, no crash."""
    _enable_converged(run_dir)
    _write_spec(run_dir, 0, obj=_full_mode_spec())
    ckpt_dir = _write_model(run_dir, 0)
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    _capture_passes(monkeypatch, calls)
    assert ev.main([run_dir, "0"]) == 0
    assert [(c[0], c[1]) for c in calls] == [
        ("model.eqx", "eval_holdout"),
        ("model.eqx", "eval_holdout_converged")]


def test_main_converged_skips_specs_without_a_full_solver(run_dir, monkeypatch):
    """A spec with no FULL-mode solver_config has no protocol to converge; the
    channel is skipped, not failed."""
    _enable_converged(run_dir)
    ckpt_dir = _write_model(run_dir, 0)
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    _capture_passes(monkeypatch, calls)
    assert ev.main([run_dir, "0"]) == 0
    assert [c[1] for c in calls] == ["eval_holdout"]


def test_main_no_converged_channel_by_default(run_dir, monkeypatch):
    """The flag is OFF unless a config asks for it, and with it off the
    channel set and the passes' overrides are what they were: a config that
    never mentions the channel must not gain 100-cycle passes."""
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(os.path.join(run_dir, "resolved_config.yaml"))
    assert cfg.eval_converged is False

    _write_spec(run_dir, 0, obj=_full_mode_spec())
    ckpt_dir = _write_model(run_dir, 0)
    open(os.path.join(ckpt_dir, "model_val_best.eqx"), "wb").close()
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    _capture_passes(monkeypatch, calls)
    assert ev.main([run_dir, "0"]) == 0
    assert [c[1] for c in calls] == ["eval_holdout", "eval_holdout_val_best"]
    assert [c[3] for c in calls] == [None, None]


def _holdout_seams(monkeypatch, result=None):
    """Wire _run_held_out_eval's collaborators so it runs no SCF."""
    import xcquinox.alec.full_benchmark_pools as fbp
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


def test_run_held_out_eval_threads_channel_to_the_parallel_driver(
        run_dir, monkeypatch):
    """The shard workers reload the spec themselves, so the channel name has
    to travel with the launch; an unthreaded channel writes warm rows into the
    converged directory."""
    import xcquinox.alec.cluster._holdout_parallel as hp
    import xcquinox.alec.cluster.grid_config as gc
    spec = _full_mode_spec()
    ckpt_dir = _write_model(run_dir, 0)
    cfg = SimpleNamespace(cluster=SimpleNamespace(eval_workers=4),
                          held_out_strict=False)
    _holdout_seams(monkeypatch)
    monkeypatch.setattr(gc, "_resolve_eval_workers",
                        lambda cluster, n_molecules=0: 4)
    seen = {}

    def _capture(*a, **kw):
        seen.update(kw)
        return {"n_reactions": 0, "n_species": 0, "n_dropped_nan": 0,
                "n_dropped_overlap": 0}
    monkeypatch.setattr(hp, "run_holdout_with_escalation", _capture)

    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir,
                          os.path.join(ckpt_dir, "model.eqx"), spec,
                          holdout_subdir="eval_holdout_converged",
                          channel="converged")
    assert seen.get("channel") == "converged"


# ---------------------------------------------------------------------------
# C3: the parallel driver emits --channel
# ---------------------------------------------------------------------------

class _FakeSpec:
    molecules = ()


def _cmd_arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _capture_worker_cmds(monkeypatch, seen):
    import xcquinox.alec.parallel as par

    def _capture(jobs, max_parallel=4, **kw):
        results = []
        for job in jobs:
            seen.append(list(job.cmd))
            names = json.loads(open(_cmd_arg(job.cmd, "--names-file")).read())
            with open(_cmd_arg(job.cmd, "--out-shard"), "w") as f:
                json.dump({"energies": {n: -1.0 for n in names},
                           "pbe_energies": {n: -0.9 for n in names},
                           "mol_records": [{"molecule": n} for n in names]}, f)
            results.append(par.WorkerResult(
                job=job, status="success", returncode=0, payload={},
                stderr="", duration=0.01))
        return results
    monkeypatch.setattr(par, "run_workers", _capture)


def test_escalation_emits_the_channel_flag(tmp_path, monkeypatch):
    """Kills the mutation ``the worker ignoring --channel`` at the launch end:
    the flag has to be on the argv, and absent when no channel is set (an
    unconditional flag would put every warm pass on a channel override)."""
    from xcquinox.alec.cluster import _holdout_parallel as hp
    full_specs = {n: object() for n in ("a", "b")}
    seen = []
    _capture_worker_cmds(monkeypatch, seen)

    hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], full_specs,
        tmp_path / "eval_holdout_converged", basis="def2-svp", grid_level=1,
        n_workers_top=2, total_cpus=2, channel="converged")
    assert seen
    for cmd in seen:
        assert _cmd_arg(cmd, "--channel") == "converged"

    seen.clear()
    hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], full_specs,
        tmp_path / "eval_holdout", basis="def2-svp", grid_level=1,
        n_workers_top=2, total_cpus=2)
    assert seen
    for cmd in seen:
        assert "--channel" not in cmd


# ---------------------------------------------------------------------------
# C3: the shard worker
# ---------------------------------------------------------------------------

def _worker_seams(monkeypatch, spec, captured):
    """compute_shard resolves its collaborators through in-function imports,
    so the seams are the source modules."""
    import xcquinox.alec.cluster._eval_one_spec as ev_mod
    import xcquinox.alec.eval_holdout as eh_mod
    import xcquinox.alec.full_benchmark_pools as fbp
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


def test_compute_shard_coldstart_channel_still_applies_its_override(monkeypatch):
    """The generalization must not move the cold-start channel."""
    captured = {}
    _worker_seams(monkeypatch, _worker_spec(), captured)
    ehw.compute_shard("/run", 0, ["h2"], "def2-svp", 1, channel="coldstart")
    sc = captured["sc"]
    assert sc.seed_source == "minao"
    assert sc.max_cycles == 25
    assert sc.conv_tol == 1e-12


def test_compute_shard_without_a_channel_keeps_the_trained_solver(monkeypatch):
    """A warm pass is not a channel: no override is applied."""
    captured = {}
    _worker_seams(monkeypatch, _worker_spec(), captured)
    ehw.compute_shard("/run", 0, ["h2"], "def2-svp", 1, channel=None)
    sc = captured["sc"]
    assert sc.backend == SolverBackend.MANUAL
    assert sc.max_cycles == 3
    assert sc.seed_source == "pbe"


def test_worker_main_accepts_the_channel_flag_and_rejects_unknown_names(
        tmp_path, monkeypatch):
    """``--channel converged`` reaches compute_shard verbatim; a name outside
    the table is an argparse refusal rather than a silently warm pass."""
    names_file = tmp_path / "names.json"
    names_file.write_text(json.dumps(["h2"]))
    seen = []
    monkeypatch.setattr(
        ehw, "compute_shard",
        lambda rd, idx, names, basis, gl, model_name="model.eqx",
        channel=None: (
            seen.append(channel)
            or {"energies": {}, "pbe_energies": {}, "mol_records": []}))
    base = ["--run-dir", "/run", "--spec-idx", "0",
            "--names-file", str(names_file),
            "--out-shard", str(tmp_path / "s.json"),
            "--basis", "def2-svp", "--grid-level", "1", "--threads", "1"]
    assert ehw.main(base + ["--channel", "converged"]) == 0
    assert ehw.main(base + ["--channel", "coldstart"]) == 0
    assert ehw.main(base) == 0
    assert seen == ["converged", "coldstart", None]
    with pytest.raises(SystemExit):
        ehw.main(base + ["--channel", "warm"])


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


def test_channel_retro_spec_status_for_the_converged_channel(tmp_path):
    """``done`` keys on the CHANNEL's own per_reaction.json, so a run that
    already carries the cold-start channel is not read as converged-done."""
    from xcquinox.alec.cluster import channel_retro as cr
    run, (pending, ready, done) = _mk_run(tmp_path)
    (ready / "model.eqx").write_bytes(b"x")
    (done / "model.eqx").write_bytes(b"x")
    (ready / "eval_holdout_coldstart").mkdir()
    (ready / "eval_holdout_coldstart" / "per_reaction.json").write_text("[]")
    (done / "eval_holdout_converged").mkdir()
    (done / "eval_holdout_converged" / "per_reaction.json").write_text("[]")
    assert cr.spec_status(str(pending), "converged") == "pending"
    assert cr.spec_status(str(ready), "converged") == "ready"
    assert cr.spec_status(str(done), "converged") == "done"
    # a val-best checkpoint makes the val-best pass part of "done": a run
    # whose final pass finished and whose val-best pass was killed is ready,
    # not done, else the val-best directory is never written
    (done / "model_val_best.eqx").write_bytes(b"x")
    assert cr.spec_status(str(done), "converged") == "ready"
    (done / "eval_holdout_converged_val_best").mkdir()
    (done / "eval_holdout_converged_val_best" / "per_reaction.json").write_text("[]")
    assert cr.spec_status(str(done), "converged") == "done"


def test_channel_retro_runs_only_the_missing_pass(tmp_path, monkeypatch):
    """With the final pass already written and the val-best pass missing, the
    retro run performs the val-best pass alone."""
    import pickle

    from xcquinox.alec.cluster import channel_retro as cr
    run, (ready, _b, _c) = _mk_run(tmp_path)
    (ready / "model.eqx").write_bytes(b"x")
    (ready / "model_val_best.eqx").write_bytes(b"x")
    (ready / "eval_holdout_converged").mkdir()
    (ready / "eval_holdout_converged" / "per_reaction.json").write_text("[]")
    (run / "specs").mkdir()
    with open(run / "specs" / "spec_0000.spec", "wb") as f:
        pickle.dump(_worker_spec(), f)
    calls = []

    def _fake_eval(rd, idx, cfg, ck, mp, ts, holdout_subdir=None,
                   channel=None):
        calls.append((os.path.basename(mp), holdout_subdir, channel))
    monkeypatch.setattr(
        "xcquinox.alec.cluster._eval_one_spec._run_held_out_eval", _fake_eval)
    monkeypatch.setattr(
        "xcquinox.alec.cluster.grid_config.load_grid_config",
        lambda path: object())
    assert cr.retro_one_spec(str(run), 0, "converged") == "ran"
    assert calls == [("model_val_best.eqx", "eval_holdout_converged_val_best",
                      "converged")]


def test_channel_retro_runs_both_checkpoints_for_converged(tmp_path,
                                                           monkeypatch):
    """The retro pass covers the same two checkpoints the inline channel
    covers, each into its own directory, both under the converged override."""
    import pickle

    from xcquinox.alec.cluster import channel_retro as cr
    run, (ready, _b, _c) = _mk_run(tmp_path)
    (ready / "model.eqx").write_bytes(b"x")
    (ready / "model_val_best.eqx").write_bytes(b"x")
    (run / "specs").mkdir()
    with open(run / "specs" / "spec_0000.spec", "wb") as f:
        pickle.dump(_worker_spec(), f)

    calls = []

    def _fake_eval(rd, idx, cfg, ck, mp, ts, holdout_subdir=None,
                   channel=None):
        calls.append((os.path.basename(mp), holdout_subdir, channel,
                      ts.solver_config))
    monkeypatch.setattr(
        "xcquinox.alec.cluster._eval_one_spec._run_held_out_eval", _fake_eval)
    monkeypatch.setattr(
        "xcquinox.alec.cluster.grid_config.load_grid_config",
        lambda path: object())

    assert cr.retro_one_spec(str(run), 0, "converged") == "ran"
    assert [(c[0], c[1], c[2]) for c in calls] == [
        ("model.eqx", "eval_holdout_converged", "converged"),
        ("model_val_best.eqx", "eval_holdout_converged_val_best",
         "converged")]
    for c in calls:
        assert c[3].backend == SolverBackend.PYSCFAD
        assert c[3].max_cycles == 100
        assert c[3].conv_tol == 1e-8


def test_channel_retro_dry_run_reports_without_running(tmp_path, monkeypatch,
                                                       capsys):
    from xcquinox.alec.cluster import channel_retro as cr
    run, (_pending, ready, _c) = _mk_run(tmp_path)
    (ready / "model.eqx").write_bytes(b"x")
    import xcquinox.alec.cluster._eval_one_spec as ev_mod
    monkeypatch.setattr(
        ev_mod, "_run_held_out_eval",
        lambda *a, **k: pytest.fail("eval ran under --dry-run"))
    assert cr.main([str(run), "--channel", "converged", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would run" in out
    assert "2 pending" in out and "1 ready" in out


def test_coldstart_retro_alias_delegates_with_the_channel_fixed():
    """``python -m ...cluster.coldstart_retro`` stays the cold-start entry
    point: the alias module's functions are the generalized ones with the
    channel bound, so the deployed job script and its tests keep working."""
    from xcquinox.alec.cluster import channel_retro as cr
    from xcquinox.alec.cluster import coldstart_retro as csr
    assert csr.discover_spec_indices is cr.discover_spec_indices
    assert csr.spec_status("/nonexistent/spec_0000") == "pending"
    assert cr.spec_status("/nonexistent/spec_0000", "coldstart") == "pending"
    assert callable(csr.coldstart_one_spec)
    assert callable(csr.main)


# ---------------------------------------------------------------------------
# C5: the verbatim re-finalizer's channel list
# ---------------------------------------------------------------------------

def test_refinalize_channels_carry_the_two_converged_dirs():
    """The verbatim re-finalizer walks every channel it must rewrite; a
    channel absent from the list keeps stale reaction sets."""
    from xcquinox.alec import refinalize_verbatim as rv
    assert "eval_holdout_converged" in rv.CHANNELS
    assert "eval_holdout_converged_val_best" in rv.CHANNELS


# ---------------------------------------------------------------------------
# C6: the figure suite
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def figmod():
    if str(_ANALYSIS) not in sys.path:
        sys.path.insert(0, str(_ANALYSIS))
    import test_make_ablation_arch_figure as figtests
    return figtests


def test_ckpt_label_names_the_converged_channels(figmod):
    """A figure set scored from the converged channel must say so; the
    fallback label would read as the final-step warm set."""
    assert figmod.fig._ckpt_label("eval_holdout_converged") == "converged"
    assert (figmod.fig._ckpt_label("eval_holdout_converged_val_best")
            == "converged-val-best")


def _add_converged_channel(run_dir, unconverged_in_spec_0=True):
    """Mirror each eval'd spec's held-out channel into eval_holdout_converged,
    with per-species density rows carrying scf_converged."""
    for sd in sorted((run_dir / "checkpoints").glob("spec_*")):
        src = sd / "eval_holdout"
        if not (src / "per_reaction.json").is_file():
            continue
        dst = sd / "eval_holdout_converged"
        dst.mkdir(exist_ok=True)
        shutil.copy(src / "per_reaction.json", dst / "per_reaction.json")
        (dst / "eval_metadata.json").write_text(json.dumps(
            {"channel": "eval_holdout_converged",
             "channel_override": "converged", "coldstart": False}))
        bad = (unconverged_in_spec_0 and sd.name == "spec_0000")
        rows = [
            {"molecule": "HO", "density_rmse": 2e-4, "density_l1": 1e-5,
             "density_rmse_pbe": 8e-4, "density_l1_pbe": 5e-5,
             "density_eps_l1": 2.5e-4, "density_eps_l1_pbe": 7e-4,
             "n_electrons": 9.0, "grid_weight_sum": 100.0,
             "ref_density_method": "ccsd", "from_training_subset": False,
             "scf_converged": True, "cycles_run": 14},
            {"molecule": "N2", "density_rmse": 3e-4, "density_l1": 2e-5,
             "density_rmse_pbe": 9e-4, "density_l1_pbe": 6e-5,
             "density_eps_l1": 3.5e-4, "density_eps_l1_pbe": 8e-4,
             "n_electrons": 14.0, "grid_weight_sum": 120.0,
             "ref_density_method": "ccsd", "from_training_subset": False,
             "scf_converged": not bad, "cycles_run": 100},
        ]
        (dst / "per_molecule.json").write_text(json.dumps(rows))


def test_collect_density_rows_keeps_unconverged_and_reports_the_count(
        figmod, tmp_path, capsys):
    """Kills the mutation ``unconverged rows kept`` (a reintroduced drop): the
    standard converged view carries every row with its flag, and the collector
    says how many species did not converge, so the disclosed set and the
    ``_excl_unconverged`` variant differ by exactly that list."""
    root, run = figmod._make_dfs_results(tmp_path)
    _add_converged_channel(run)
    capsys.readouterr()
    rows = figmod.fig.collect_holdout_density_rows(
        run, eval_subdir="eval_holdout_converged")
    out = capsys.readouterr().out

    n2 = [r for r in rows if r["molecule"] == "N2"]
    assert len(n2) == 4                      # one per eval'd spec, none dropped
    assert sum(1 for r in n2 if r["scf_converged"] is False) == 1
    unconv_lines = [ln for ln in out.splitlines()
                    if re.search(r"unconverged", ln, re.IGNORECASE)]
    assert unconv_lines, out
    assert any(re.search(r"\b1\b", ln) for ln in unconv_lines), unconv_lines


def test_suite_renders_the_converged_views_only_when_cells_exist(figmod,
                                                                 tmp_path):
    """The converged channel joins the view loop and is gated like the
    val-best view: a run with no converged cells renders no converged
    directory."""
    root, run = figmod._make_dfs_results(tmp_path)
    plain = tmp_path / "figs_plain"
    figmod.fig.build_bh76w411_suite(results_root=root, outroot=plain,
                                    bases=("svp_grid2",), domain="dfs_step7")
    assert (plain / "figures_dfs_step7_svp").is_dir()
    assert not list(plain.glob("*_converged"))

    _add_converged_channel(run)
    outroot = tmp_path / "figs_conv"
    figmod.fig.build_bh76w411_suite(results_root=root, outroot=outroot,
                                    bases=("svp_grid2",), domain="dfs_step7")
    assert (outroot / "figures_dfs_step7_svp_converged").is_dir(), sorted(
        p.name for p in outroot.iterdir())


# ---------------------------------------------------------------------------
# C5: the retroactive job script
# ---------------------------------------------------------------------------

def test_converged_retro_job_script_shape():
    """The retro job carries the standing mail directives, the house shell
    idiom, both v7 arms, the generalized tool on the converged channel, and a
    worker preflight on three species before the array line."""
    path = _REPO_ROOT / "hpcjobs" / ("converged_retro." + "sbatch")
    text = path.read_text()
    assert "--mail-user=alec.wills@stonybrook.edu" in text
    assert "--mail-type=BEGIN,END,FAIL" in text
    assert "set -uo pipefail" in text
    assert "channel_retro" in text
    assert "--channel converged" in text
    assert "dfs6311_grid3_v7g1_size" in text
    assert "dfs6311_grid3_v7g2a_families_core" in text
    assert "eval_holdout_worker" in text
    for species in ("h2o", "bn", "RKT17"):
        assert species in text
    # the preflight must precede the retro line it gates
    assert text.index("eval_holdout_worker") < text.index("channel_retro")
