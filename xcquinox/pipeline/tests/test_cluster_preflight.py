"""Tests for xcquinox.pipeline.cluster._preflight: the SLURM preflight entrypoint.

The re-scoped preflight is orchestration-only: the three heavy calls
(``prepare_inputs``, ``build_training_specs``, ``materialize_specs``) are bound
to module-level seams that these tests monkeypatch, so the whole preflight flow
runs without real CCSD / SCF / DFS-pool work.

Subset selection is a finished pre-process, the preflight consumes the
existing subset ledger read-only and does NOT run descriptor extraction,
reference histograms, ``select_subset``, or any ``regenerate``/``reuse`` mode
toggle. Those behaviours were removed and are no longer tested.

The materialization seam is left REAL in the happy-path tests (it only writes
small serialized stub specs) so the self-check exercises actual on-disk files;
the failure-injection tests stub it to drop a spec file.
"""
import json
import os
from dataclasses import dataclass, field

import pytest

from xcquinox.pipeline.cluster import _preflight
from xcquinox.pipeline.cluster._preflight import main
from xcquinox.pipeline.cluster.grid_config import GridCell


# ---------------------------------------------------------------------------
# Stubs: serializable so the real materialize_specs can write them
# ---------------------------------------------------------------------------

@dataclass
class _StubMol:
    """Minimal serializable stand-in for a MoleculeSpec (only ``name`` used)."""
    name: str


@dataclass
class _StubSpec:
    """Minimal serializable stand-in for a TrainingSpec.

    ``validate()`` creates ``checkpoint_dir`` (mirroring the real spec) and,
    when ``validate_error`` is set, raises it, the validation-failure test
    uses that to mimic the ``n_compounds >= 1`` rule firing.
    """
    checkpoint_dir: str
    molecules: tuple = ()
    pbe_anchor_sample: object = None
    validate_error: str = ""

    def validate(self):
        if self.validate_error:
            raise ValueError(self.validate_error)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


@dataclass
class _StagedStub:
    """Stand-in for inputs.StagedInputs."""
    points: list = field(default_factory=list)
    subset_ledger: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_resolved_config(run_dir, extra=None, cluster_extra=None):
    """Write a minimal valid ``resolved_config.yaml`` into ``run_dir``.

    The grid is metric=l2 x subset_size={2,3} -> 2 cells. ``extra`` merges
    top-level keys (e.g. ``on_precompute_failure``); ``cluster_extra`` merges
    keys into the nested ``cluster`` block (e.g. ``preflight_compile_smoke``),
    exercising the real ``_build_cluster`` round-trip.
    """
    cfg = {
        "sweep": {
            "arch": ["shallow"],
            "loss": ["L5_gradnorm_vxc_step7"],
            "metric": ["l2"],
            "subset_size": [2, 3],
            "solver": ["oneshot"],
        },
        "solvers": {"oneshot": {"mode": "oneshot", "max_cycles": 0}},
        "hyperparams": {
            "n_steps": 100,
            "lr_start": 1e-3,
            "lr_end": 1e-5,
            "lr_decay_start": 0.0,
            "grad_clip": 1.0,
            "gradnorm_alpha": 1.5,
            "vxc_weight": 0.01,
            "density_weight": 0.1,
        },
        "inputs": {
            "external_refs_dir": str(run_dir / "refs"),
            "subset_ledger_path": str(run_dir / "ledger.json"),
            "basis": "def2-svp",
            "grid_level": 1,
            "output_root": str(run_dir / "out"),
        },
        "pretrain": {
            "data_dir": str(run_dir / "data"),
        },
        "cluster": {
            "partition": "short",
            "time": "04:00:00",
            "mem": "16G",
            "cpus_per_task": 4,
            "array_throttle": 8,
            "eval_array_throttle": 4,
            "max_concurrent_tasks": 16,
        },
        "domain_profile": "dfs_step7",
    }
    if cluster_extra:
        cfg["cluster"].update(cluster_extra)
    if extra:
        cfg.update(extra)
    import yaml
    path = run_dir / "resolved_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    for arch in cfg["sweep"]["arch"]:
        _write_pass_certificate(run_dir, arch)
    return path


def _write_pass_certificate(run_dir, arch="shallow", verdict="PASS"):
    """Write a PASS fidelity certificate for ``arch`` under ``run_dir``.

    The preflight runs afterok on the pretrain array, so by preflight time
    every architecture already carries one; these fixtures describe that
    state, and the gate's own tests remove or downgrade it.
    """
    d = os.path.join(str(run_dir), "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    payload = {"verdict": verdict, "arch": arch,
               "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2}}
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump(payload, f)
    return d


def _two_cells():
    """The 2 GridCells the test config's grid expands to (sorted order)."""
    return [
        GridCell(arch="shallow", loss="L5_gradnorm_vxc_step7",
                 metric="l2", subset_size=2, solver="oneshot"),
        GridCell(arch="shallow", loss="L5_gradnorm_vxc_step7",
                 metric="l2", subset_size=3, solver="oneshot"),
    ]


def _make_specs(run_dir, n=2, validate_error="", molecules_per_spec=None):
    """Build ``n`` ``(cell, _StubSpec)`` pairs with checkpoint dirs under run_dir.

    ``molecules_per_spec``: optional list of per-spec molecule-name iterables;
    each name becomes a ``_StubMol``. Default: one ``"mol"`` molecule per spec.
    """
    cells = _two_cells()[:n]
    out = []
    for idx, cell in enumerate(cells):
        ckpt = os.path.join(str(run_dir), "checkpoints", f"spec_{idx:04d}")
        names = ("mol",) if molecules_per_spec is None else molecules_per_spec[idx]
        mols = tuple(_StubMol(name=nm) for nm in names)
        out.append((cell, _StubSpec(
            checkpoint_dir=ckpt,
            molecules=mols,
            validate_error=validate_error if idx == 0 else "",
        )))
    return out


@pytest.fixture
def patched(monkeypatch):
    """Monkeypatch the two upstream heavy seams with simple stubs.

    Returns a mutable dict so each test can install its own ``prepare_inputs``
    / ``build_training_specs`` behavior; ``materialize_specs`` stays real.
    """
    state = {}

    def fake_prepare_inputs(cfg, *, recompute_refs=True, run_dir=None):
        state["prepare_calls"] = state.get("prepare_calls", 0) + 1
        # Record the recompute_refs kwarg seen on EACH call so a test can
        # assert the re-stage (after a precompute failure) skipped the
        # precompute by passing recompute_refs=False.
        state.setdefault("recompute_refs_seen", []).append(recompute_refs)
        # WS3: record the run_dir forwarded so a test can assert the preflight
        # threads it (the val-slice staging writes val_reactions.json under it).
        state.setdefault("run_dir_seen", []).append(run_dir)
        hook = state.get("prepare_hook")
        if hook is not None:
            return hook(cfg, recompute_refs, state["prepare_calls"])
        return _StagedStub(points=["p0", "p1"], subset_ledger={"l2/2": {}})

    def fake_build_specs(points, ledger, cfg, domain, run_dir):
        builder = state.get("build_hook")
        if builder is not None:
            return builder(run_dir)
        return _make_specs(run_dir, n=2)

    monkeypatch.setattr(_preflight, "_prepare_inputs", fake_prepare_inputs)
    monkeypatch.setattr(_preflight, "_build_training_specs", fake_build_specs)
    return state


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_writes_specs_manifest_exit_0(tmp_path, patched):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)

    rc = main([str(run_dir)])

    assert rc == 0
    specs_dir = run_dir / "specs"
    assert (specs_dir / "spec_0000.spec").is_file()
    assert (specs_dir / "spec_0001.spec").is_file()
    manifest = run_dir / "manifest.json"
    assert manifest.is_file()
    payload = json.loads(manifest.read_text())
    assert payload["n_specs"] == 2
    assert len(payload["specs"]) == 2
    # prepare_inputs called once, with recompute_refs defaulting to True
    assert patched["prepare_calls"] == 1
    assert patched["recompute_refs_seen"] == [True]
    # WS3: the preflight forwards run_dir so prepare_inputs can stage the
    # val slice (val_reactions.json lives under run_dir).
    assert patched["run_dir_seen"] == [str(run_dir)]
    # a provenance copy of the consumed subset ledger was written
    assert (run_dir / "subset_ledger.json").is_file()


# ---------------------------------------------------------------------------
# prepare_inputs fail-fast on a missing ledger cell
# ---------------------------------------------------------------------------

def test_missing_ledger_cell_exit_1(tmp_path, patched, capsys):
    """prepare_inputs raises ValueError for a missing (metric, r) ledger cell
    -> the preflight catches it and exits 1."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)

    def raising(cfg, recompute_refs, call_n):
        raise ValueError(
            "subset ledger is missing entries for grid cells [('l2', 3)]"
        )

    patched["prepare_hook"] = raising
    rc = main([str(run_dir)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "input staging failed" in out


# ---------------------------------------------------------------------------
# Self-check failures
# ---------------------------------------------------------------------------


def test_self_check_fails_when_manifest_cell_count_wrong(tmp_path, patched,
                                                         monkeypatch):
    """manifest records the wrong n_specs -> self-check fails -> exit 1."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)

    def bad_manifest(cells, paths, out_dir):
        path = os.path.join(out_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump({"n_specs": 99, "width": 4, "specs": []}, f)
        return path

    monkeypatch.setattr(_preflight, "write_manifest", bad_manifest)
    assert main([str(run_dir)]) == 1


# ---------------------------------------------------------------------------
# precompute failure handling, on_precompute_failure policy
# ---------------------------------------------------------------------------

_PRECOMPUTE_ERR = (
    "Cell 0.5 pre-compute failed for 2 species: ['C+', 'O3']. "
    "Inspect _run_log_*.json for details."
)


def test_precompute_failure_drop_species_marks_specs_exit_0(tmp_path, patched):
    """on_precompute_failure='drop_failed_species': the first prepare_inputs
    raises, the re-stage (recompute_refs=False) succeeds; the spec whose
    molecule set references a failed species gets a
    ``precompute_failed_species`` failure.json; unaffected specs materialize;
    exit 0."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(
        run_dir, extra={"on_precompute_failure": "drop_failed_species"}
    )

    def hook(cfg, recompute_refs, call_n):
        if call_n == 1:
            raise RuntimeError(_PRECOMPUTE_ERR)
        return _StagedStub(points=["p0"], subset_ledger={"l2/2": {}})

    patched["prepare_hook"] = hook
    # spec 0 references the failed species 'O3'; spec 1 references only 'H2O'.
    patched["build_hook"] = lambda rd: _make_specs(
        rd, n=2, molecules_per_spec=[("O3", "O2"), ("H2O", "H2")]
    )

    rc = main([str(run_dir)])

    assert rc == 0
    # spec 0 was marked (its subset references the failed 'O3')
    fj0 = run_dir / "checkpoints" / "spec_0000" / "failure.json"
    assert fj0.is_file()
    payload = json.loads(fj0.read_text())
    assert payload["classification"] == "precompute_failed_species"
    assert payload["species"] == ["O3"]
    assert payload["failed_species"] == ["C+", "O3"]
    # spec 1 (no failed species) was NOT marked
    fj1 = run_dir / "checkpoints" / "spec_0001" / "failure.json"
    assert not fj1.exists()
    # both specs still materialized
    assert (run_dir / "specs" / "spec_0000.spec").is_file()
    assert (run_dir / "specs" / "spec_0001.spec").is_file()
    # both prepare_inputs calls happened: initial (recompute_refs=True) +
    # re-stage (recompute_refs=False, the failed precompute is NOT re-run)
    assert patched["prepare_calls"] == 2
    assert patched["recompute_refs_seen"] == [True, False]


# ---------------------------------------------------------------------------
# spec.validate() failure surfacing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# compile-smoke gate (cluster.preflight_compile_smoke)
# ---------------------------------------------------------------------------

def test_compile_smoke_gate_failure_blocks_exit_1(tmp_path, patched, monkeypatch):
    """cluster.preflight_compile_smoke=True and the heaviest-cell compile probe
    FAILS -> main() returns 1 so the train array's afterok dependency blocks."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir,
                           cluster_extra={"preflight_compile_smoke": True})

    monkeypatch.setattr(_preflight, "_compile_smoke",
                        lambda specs, paths, run_dir: False)
    assert main([str(run_dir)]) == 1


# ---------------------------------------------------------------------------
# cold-start convergence census (cluster.preflight_coldstart_census)
# ---------------------------------------------------------------------------

def _recording_census(calls, verdict):
    """A stand-in for the census seam that records its arguments and reports
    ``verdict``."""
    def _seam(specs, paths, run_dir):
        calls.append((len(specs), len(paths), run_dir))
        return verdict
    return _seam


def test_the_preflight_runs_the_census_and_never_blocks_on_it(
        tmp_path, patched, monkeypatch, capsys):
    """``cluster.preflight_coldstart_census`` runs the census seam once, over
    the staged specs and their materialized paths, and the run directory; the
    preflight exits 0 whichever verdict the census returns, and does not call
    the seam at all when the knob is absent.

    The census is a report on where the cold start converges, not a gate: a
    species that does not converge is information for the analysis, and
    blocking the array on it would cost the run. The never-blocks half is
    asserted against a seam that reports failure, which is the only state that
    separates a report from a gate; the knob-absent half holds every existing
    configuration's preflight unchanged.
    """
    calls = []

    # Knob on, census succeeds: called once with the staged grid, exit 0.
    run_dir = tmp_path / "on"
    run_dir.mkdir()
    _write_resolved_config(run_dir,
                           cluster_extra={"preflight_coldstart_census": True})
    monkeypatch.setattr(_preflight, "_coldstart_census",
                        _recording_census(calls, True))
    assert main([str(run_dir)]) == 0
    assert len(calls) == 1
    n_specs, n_paths, seen_dir = calls[0]
    assert n_specs == len(_two_cells()) and n_paths == n_specs
    assert seen_dir == str(run_dir)
    capsys.readouterr()          # drop this run's log before reading the next

    # Knob on, census reports a failure: still exit 0, and the log says so.
    calls.clear()
    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    _write_resolved_config(failed_dir,
                           cluster_extra={"preflight_coldstart_census": True})
    monkeypatch.setattr(_preflight, "_coldstart_census",
                        _recording_census(calls, False))
    assert main([str(failed_dir)]) == 0
    assert len(calls) == 1
    assert "FAILED" in capsys.readouterr().out

    # Knob absent: the seam is never reached.
    calls.clear()
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    _write_resolved_config(off_dir)
    monkeypatch.setattr(_preflight, "_coldstart_census",
                        _recording_census(calls, True))
    assert main([str(off_dir)]) == 0
    assert calls == []


def test_the_census_takes_the_first_full_cell_of_every_architecture():
    """``_census_cells`` picks, per architecture, the first cell whose solver
    is FULL (the only mode the solver accepts the atomic-guess seed in) and
    names the architectures with no such cell.

    The grid holds a one-shot cell before a FULL cell of the same
    architecture, an architecture with only a one-shot cell, and one with a
    FULL cell held in the ``solver_config`` field rather than in
    ``loss_kwargs``, so the selection is separated from "the first cell" and
    from a reader of one of the two places the solver config lives.
    """
    from types import SimpleNamespace
    from xcquinox.pipeline.solver import SolverConfig, SolverMode

    full = SolverConfig(mode=SolverMode.FULL, max_cycles=3)
    oneshot = SolverConfig(mode=SolverMode.ONESHOT, max_cycles=0)

    def _spec(sc, *, in_field=False):
        if in_field:
            return SimpleNamespace(loss_kwargs_dict={}, solver_config=sc)
        return SimpleNamespace(loss_kwargs_dict={"solver_config": sc},
                              solver_config=None)

    specs = [
        (_cell("a", 2), _spec(oneshot)),
        (_cell("a", 3), _spec(full)),
        (_cell("b", 2), _spec(oneshot)),
        (_cell("c", 2), _spec(full, in_field=True)),
        (_cell("c", 3), _spec(full)),
    ]
    chosen, skipped = _preflight._census_cells(specs)
    assert chosen == [(1, "a"), (3, "c")]
    assert skipped == ["b"]


def test_the_census_subprocess_writes_the_record_under_the_run_dir(tmp_path,
                                                                    capsys):
    """``_coldstart_census_impl`` runs the census as a subprocess over the
    selected cells and leaves ``coldstart_census.json`` and the subprocess
    output under the run directory, logging the cell's summary line.

    One FULL cell on a closed-shell H2 at sto-3g with a three-cycle solver,
    materialized as the preflight materializes specs, so the whole path the
    cluster job takes -- the spec file, the interpreter, the thread
    environment, the output persistence -- is exercised once. The oracle is
    the document the census module writes: one cell, one species, a row for
    it.
    """
    from xcquinox.pipeline.cluster.materialize import write_spec_atomic
    from xcquinox.pipeline.config import ArchitectureConfig, TrainingSpec
    from xcquinox.pipeline.solver import SolverConfig, SolverMode
    from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule

    arch = ArchitectureConfig(
        name="census_t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False)
    spec = TrainingSpec.from_dicts(
        arch=arch, molecules=(h2_molecule(),), targets={"H2": 0.2},
        atom_energies={"H": -0.5}, loss_name="L5_gradnorm_vxc_step7",
        loss_kwargs={"solver_config": SolverConfig(mode=SolverMode.FULL,
                                                   max_cycles=3)},
        n_steps=1, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
        grad_clip=1.0, checkpoint_dir=str(tmp_path / "ckpt"), seed=0,
        update_scheme="per_molecule", require_atom_anchors=False)
    run_dir = tmp_path / "run"
    (run_dir / "specs").mkdir(parents=True)
    path = str(run_dir / "specs" / "spec_0000.spec")
    write_spec_atomic(spec, path)

    ok = _preflight._coldstart_census_impl(
        [(_cell("census_t", 1), spec)], [path], str(run_dir))
    assert ok is True

    with open(run_dir / "coldstart_census.json") as fh:
        doc = json.load(fh)
    assert len(doc["cells"]) == 1
    cell = doc["cells"][0]
    assert cell["arch"] == "census_t"
    assert cell["error"] is None
    assert cell["n_species"] == 1
    assert [r["name"] for r in cell["rows"]] == ["H2"]
    assert os.path.isfile(run_dir / "logs" / "coldstart_census.out")
    out = capsys.readouterr().out
    assert "census census_t: " in out


# ---------------------------------------------------------------------------
# The per-architecture fidelity gate
# ---------------------------------------------------------------------------

def test_preflight_blocks_the_array_on_a_missing_certificate(tmp_path,
                                                             patched, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)
    os.remove(os.path.join(str(run_dir), "pretrain", "shallow",
                           "fidelity_certificate.json"))
    assert main([str(run_dir)]) == 1
    out = capsys.readouterr().out
    assert "fidelity gate FAILED" in out
    assert "shallow" in out


# ---------------------------------------------------------------------------
# The compile-smoke selector: which cell is "the heaviest attention cell"
# ---------------------------------------------------------------------------

def _cell(arch, subset_size):
    return GridCell(arch=arch, loss="L5_gradnorm_vxc_step7", metric="jsd",
                    subset_size=subset_size, solver="full_3")


