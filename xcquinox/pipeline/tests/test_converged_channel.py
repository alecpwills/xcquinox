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


def _enable_coldstart(run_dir):
    import yaml
    path = os.path.join(run_dir, "resolved_config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["eval_coldstart"] = True
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def test_main_runs_both_coldstart_passes_when_enabled(run_dir, monkeypatch,
                                                      capsys):
    """Kills the mutation ``the cold-start val-best pass omitted``: the
    reporting number is the cold-start protocol on the validation-best
    checkpoint, so the channel needs that checkpoint as well as the final one,
    both under the replaced solver.

    The warm passes are asserted untouched in the same capture: the override
    is applied to a copy of the spec, and a replacement applied in place would
    evaluate the trained-protocol channels at the cold-start budget while
    still naming them warm.
    """
    _enable_coldstart(run_dir)
    _write_spec(run_dir, 0, obj=_full_mode_spec())
    ckpt_dir = _write_model(run_dir, 0)
    val_best = os.path.join(ckpt_dir, "model_val_best.eqx")
    open(val_best, "wb").close()
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    _capture_passes(monkeypatch, calls)
    assert ev.main([run_dir, "0"]) == 0

    assert [(c[0], c[1]) for c in calls] == [
        ("model.eqx", "eval_holdout"),
        ("model_val_best.eqx", "eval_holdout_val_best"),
        ("model.eqx", "eval_holdout_coldstart"),
        ("model_val_best.eqx", "eval_holdout_coldstart_val_best")]
    assert [c[3] for c in calls] == [None, None, "coldstart", "coldstart"]
    for c in calls[2:]:
        sc = c[2].solver_config
        assert sc.mode == SolverMode.FULL
        assert sc.seed_source == "minao"
        assert sc.seed_cache_dir is None
        assert sc.max_cycles == eh.COLDSTART_MAX_CYCLES == 25
        assert sc.conv_tol == eh.COLDSTART_CONV_TOL == 1e-12
    # the two cold passes share one replaced spec object, not two replacements
    # that happen to agree
    assert calls[2][2] is calls[3][2]
    # the warm passes keep the trained protocol
    assert calls[0][2].solver_config.max_cycles == 3
    assert calls[0][2].solver_config.seed_source == "pbe"
    assert calls[1][2].solver_config.max_cycles == 3

    # without the validation-best checkpoint the channel is the final one
    # alone, and the stage says so rather than failing
    os.remove(val_best)
    calls.clear()
    capsys.readouterr()
    assert ev.main([run_dir, "0"]) == 0
    assert [(c[0], c[1]) for c in calls] == [
        ("model.eqx", "eval_holdout"),
        ("model.eqx", "eval_holdout_coldstart")]
    assert "model_val_best.eqx" in capsys.readouterr().out


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
    cfg = SimpleNamespace(cluster=SimpleNamespace(eval_workers=1))
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
        fbp, "load_held_out_pools",
        lambda names, basis=None, grid_level=None:
            ({"bh76@h2": SimpleNamespace(name="bh76@h2")}, []))


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
    ehw.compute_shard("/run", 0, ["bh76@h2"], "def2-svp", 1,
                      channel="converged")
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




# ---------------------------------------------------------------------------
# C3b: the configurable readout mode (eval_pool) -- the two-pass replace and
# the reference-file guard
# ---------------------------------------------------------------------------

def _readout_cfg(refs_dir, dump=True):
    """A readout cfg carrying ONLY the knobs: no inputs block at all, so
    every read in the readout path resolves through getattr-with-default
    (the slim16_eval.py SimpleNamespace cfg keeps working)."""
    return SimpleNamespace(
        eval_pool="slim16",
        eval_basis="def2-TZVP",
        eval_grid_level=4,
        eval_orientation_lock=0.0,
        eval_density_fit=True,
        eval_auxbasis="def2-tzvp-jkfit",
        eval_refs_dir=refs_dir,
        eval_dump_densities=dump,
        domain_profile="dfs_step7",
        eval_coldstart=True,
        eval_converged=True,
        cluster=SimpleNamespace(eval_workers=1),
    )


def _capture_readout_passes(monkeypatch, calls):
    monkeypatch.setattr(
        ev, "_run_held_out_eval",
        lambda rd, idx, cfg, ck, mp, ts, holdout_subdir="eval_holdout",
        channel=None, dump_densities=None:
            calls.append({
                "model": os.path.basename(mp), "subdir": holdout_subdir,
                "channel": channel, "spec": ts, "dump": dump_densities,
                "refs_env": os.environ.get("XCQUINOX_BENCH_REFS_DIR")}))


def test_main_readout_mode_runs_two_converged_passes_and_skips_the_rest(
        tmp_path, monkeypatch):
    """With eval_pool set the inline eval is REPLACED: no warm and no
    cold-start pass runs (eval_coldstart and eval_converged are inert in
    readout mode), exactly the final and the validation-best checkpoints are
    evaluated under the converged channel, each at the readout footing
    composed with the converged protocol, each with its own densities
    directory, and XCQUINOX_BENCH_REFS_DIR points at the readout references
    for the duration of the branch and is restored afterwards."""
    from xcquinox.pipeline.cluster import grid_config as gc

    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    refs = str(tmp_path / "refs")
    os.makedirs(refs)
    monkeypatch.setattr(gc, "load_grid_config",
                        lambda path: _readout_cfg(refs))
    _write_manifest(run_dir)
    _write_spec(run_dir, 0, obj=_full_mode_spec())
    ckpt_dir = _write_model(run_dir, 0)
    open(os.path.join(ckpt_dir, "model_val_best.eqx"), "wb").close()
    _stub_insample(monkeypatch, os.path.join(ckpt_dir, "eval"))

    calls = []
    _capture_readout_passes(monkeypatch, calls)
    prior = os.environ.get("XCQUINOX_BENCH_REFS_DIR")
    assert ev.main([run_dir, "0"]) == 0

    assert [(c["model"], c["subdir"], c["dump"]) for c in calls] == [
        ("model.eqx", "eval_holdout_converged",
         os.path.join(ckpt_dir, "eval_holdout_converged", "densities")),
        ("model_val_best.eqx", "eval_holdout_converged_val_best",
         os.path.join(ckpt_dir, "eval_holdout_converged_val_best",
                      "densities"))]
    assert [c["channel"] for c in calls] == ["converged", "converged"]
    for c in calls:
        sc = c["spec"].solver_config
        # the converged protocol
        assert sc.backend == SolverBackend.PYSCFAD
        assert sc.mode == SolverMode.FULL
        assert sc.seed_source == "pbe"
        assert sc.max_cycles == 100
        assert sc.conv_tol == 1e-8
        assert sc.scf_loss_use_tail is False
        # the readout footing: the training spec carries DF off and no
        # auxbasis, so both moves are observable
        assert sc.density_fit is True
        assert sc.auxbasis == "def2-tzvp-jkfit"
        assert sc.orientation_lock_strength == 0.0
        assert c["refs_env"] == refs
    assert os.environ.get("XCQUINOX_BENCH_REFS_DIR") == prior


def _write_ref_npz(path, *, basis, auxbasis=None):
    """A synthetic benchmark reference npz stamped for
    benchmark_refs._benchmark_npz_is_complete; auxbasis writes the
    auxbasis_used stamp, None leaves the legacy unstamped shape."""
    import numpy as np

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    arrays = dict(
        rho_ref_grid=np.zeros((3, 3)), ref_density_method="ccsd",
        basis_used=str(basis), grid_level_used=4,
        orientation_lock_strength=0.0, density_fit_used=True)
    if auxbasis is not None:
        arrays["auxbasis_used"] = str(auxbasis)
    np.savez(str(path), **arrays)


def test_readout_refs_guard_refuses_incomplete_or_mismatched_refs(
        tmp_path, monkeypatch):
    """With eval_refs_dir configured, every pool species needs a complete
    reference npz at the readout stamps: a missing file, a wrong basis stamp
    or a wrong auxbasis stamp stops the channel before any energy is
    computed (a RuntimeError in the channel's failure.json naming the count
    and the species), and a fully stamped directory lets the same call
    through. The eval-time loader checks only file existence, so this guard
    is the only completeness check on the read side."""
    import xcquinox.pipeline.full_benchmark_pools as fbp

    refs = str(tmp_path / "refs")
    os.makedirs(refs)
    cfg = _readout_cfg(refs)
    spec = SimpleNamespace(solver_config=None, validate_every=0)
    ckpt = str(tmp_path / "ck")
    run_dir = str(tmp_path / "run")
    model_path = os.path.join(ckpt, "model.eqx")

    monkeypatch.setattr(eh, "load_trained_model", lambda ts, mp: "MODEL")
    specs = {"slim16@a": "specA", "slim16@b": "specB",
             "slim16@c": "specC", "slim16@d": "specD"}
    rxns = [{"name": "r1", "reactants": ["slim16@a"],
             "products": ["slim16@b"], "coeffs": [-1.0, 1.0],
             "reaction_energy_ref": 1.0, "source_pool": "slim16"}]
    monkeypatch.setattr(
        fbp, "load_held_out_pools",
        lambda names, basis=None, grid_level=None: (specs, rxns))
    n_serial = {"n": 0}

    def _fake_serial(**kw):
        n_serial["n"] += 1
        return {"n_reactions": 0, "n_species": 0, "n_dropped_nan": 0,
                "n_dropped_overlap": 0}

    monkeypatch.setattr(eh, "run_full_holdout_eval", _fake_serial)

    # a: complete at the readout stamps; b: wrong basis stamp; c: missing;
    # d: complete but generated under another auxbasis
    _write_ref_npz(os.path.join(refs, "slim16@a.npz"), basis="def2-TZVP",
                   auxbasis="def2-tzvp-jkfit")
    _write_ref_npz(os.path.join(refs, "slim16@b.npz"), basis="def2-svp",
                   auxbasis="def2-tzvp-jkfit")
    _write_ref_npz(os.path.join(refs, "slim16@d.npz"), basis="def2-TZVP",
                   auxbasis="def2-svp-jkfit")

    ev._run_held_out_eval(run_dir, 0, cfg, ckpt, model_path, spec)
    failure = os.path.join(ckpt, "eval_holdout", "failure.json")
    assert os.path.isfile(failure)
    with open(failure) as f:
        payload = json.load(f)
    assert payload["exception_type"] == "RuntimeError"
    message = payload["exception_message"]
    for name in ("slim16@b", "slim16@c", "slim16@d"):
        assert name in message
    assert "3" in message
    assert n_serial["n"] == 0

    # a healed directory passes the same call through the guard
    for name in ("slim16@b", "slim16@c", "slim16@d"):
        _write_ref_npz(os.path.join(refs, f"{name}.npz"), basis="def2-TZVP",
                       auxbasis="def2-tzvp-jkfit")
    ev._run_held_out_eval(run_dir, 0, cfg, ckpt, model_path, spec)
    assert not os.path.isfile(failure)
    assert n_serial["n"] == 1
