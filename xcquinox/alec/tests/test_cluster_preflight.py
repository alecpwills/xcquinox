"""Tests for xcquinox.alec.cluster._preflight — the SLURM preflight entrypoint.

The re-scoped preflight is orchestration-only: the three heavy calls
(``prepare_inputs``, ``build_training_specs``, ``materialize_specs``) are bound
to module-level seams that these tests monkeypatch, so the whole preflight flow
runs without real CCSD / SCF / DFS-pool work.

Subset selection is a *finished pre-process* — the preflight consumes the
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

from xcquinox.alec.cluster import _preflight
from xcquinox.alec.cluster._preflight import main
from xcquinox.alec.cluster.grid_config import GridCell


# ---------------------------------------------------------------------------
# Stubs — serializable so the real materialize_specs can write them
# ---------------------------------------------------------------------------

@dataclass
class _StubMol:
    """Minimal serializable stand-in for a MoleculeSpec (only ``name`` used)."""
    name: str


@dataclass
class _StubSpec:
    """Minimal serializable stand-in for a TrainingSpec.

    ``validate()`` creates ``checkpoint_dir`` (mirroring the real spec) and,
    when ``validate_error`` is set, raises it — the validation-failure test
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

def _write_resolved_config(run_dir, extra=None):
    """Write a minimal valid ``resolved_config.yaml`` into ``run_dir``.

    The grid is metric=l2 x subset_size={2,3} -> 2 cells. ``extra`` merges
    top-level keys (e.g. ``on_precompute_failure``).
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
            "pretrain_root": str(run_dir / "pretrain"),
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
    if extra:
        cfg.update(extra)
    import yaml
    path = run_dir / "resolved_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


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

    ``molecules_per_spec`` — optional list of per-spec molecule-name iterables;
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

    def fake_prepare_inputs(cfg, *, recompute_refs=True):
        state["prepare_calls"] = state.get("prepare_calls", 0) + 1
        # Record the recompute_refs kwarg seen on EACH call so a test can
        # assert the re-stage (after a precompute failure) skipped the
        # precompute by passing recompute_refs=False.
        state.setdefault("recompute_refs_seen", []).append(recompute_refs)
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
    # a provenance copy of the consumed subset ledger was written
    assert (run_dir / "subset_ledger.json").is_file()


def test_resolved_config_missing_exit_1(tmp_path, patched):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # no resolved_config.yaml written
    assert main([str(run_dir)]) == 1


def test_no_argv_exit_1(patched):
    assert main([]) == 1


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

def test_self_check_fails_when_spec_file_missing(tmp_path, patched, monkeypatch):
    """materialize writes only N-1 of N spec files -> self-check fails -> 1."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)

    real_materialize = _preflight._materialize_specs

    def short_materialize(specs, out_dir):
        paths = real_materialize(specs, out_dir)
        # delete the last written spec file to simulate an incomplete write
        os.unlink(paths[-1])
        return paths

    monkeypatch.setattr(_preflight, "_materialize_specs", short_materialize)
    assert main([str(run_dir)]) == 1


def test_self_check_fails_when_manifest_missing(tmp_path, patched, monkeypatch):
    """manifest never written -> self-check fails -> exit 1."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)

    monkeypatch.setattr(_preflight, "write_manifest",
                        lambda cells, paths, out_dir: "/nonexistent/manifest.json")
    assert main([str(run_dir)]) == 1


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
# precompute failure handling — on_precompute_failure policy
# ---------------------------------------------------------------------------

_PRECOMPUTE_ERR = (
    "Cell 0.5 pre-compute failed for 2 species: ['C+', 'O3']. "
    "Inspect _run_log_*.json for details."
)


def test_precompute_failure_abort_exit_1(tmp_path, patched):
    """on_precompute_failure='abort' (default): RuntimeError -> exit 1."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)  # default on_precompute_failure=abort

    def raising(cfg, recompute_refs, call_n):
        raise RuntimeError(_PRECOMPUTE_ERR)

    patched["prepare_hook"] = raising
    assert main([str(run_dir)]) == 1
    # abort never re-stages
    assert patched["prepare_calls"] == 1


def test_precompute_failure_abort_logs_failed_species(tmp_path, patched, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)

    def raising(cfg, recompute_refs, call_n):
        raise RuntimeError(_PRECOMPUTE_ERR)

    patched["prepare_hook"] = raising
    main([str(run_dir)])
    out = capsys.readouterr().out
    assert "C+" in out and "O3" in out
    assert "abort" in out


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
    # re-stage (recompute_refs=False — the failed precompute is NOT re-run)
    assert patched["prepare_calls"] == 2
    assert patched["recompute_refs_seen"] == [True, False]


def test_precompute_failure_drop_species_unparseable_aborts(tmp_path, patched):
    """drop_failed_species but the precompute error carries no parseable
    species list -> affected specs cannot be identified -> exit 1, no
    re-stage."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(
        run_dir, extra={"on_precompute_failure": "drop_failed_species"}
    )

    def raising(cfg, recompute_refs, call_n):
        # message with no "species: [" marker -> _failed_species_from_error
        # returns [] -> affected specs cannot be identified.
        raise RuntimeError("Cell 0.5 pre-compute failed catastrophically.")

    patched["prepare_hook"] = raising
    assert main([str(run_dir)]) == 1
    # only the initial call happened; no re-stage
    assert patched["prepare_calls"] == 1
    assert patched["recompute_refs_seen"] == [True]


# ---------------------------------------------------------------------------
# spec.validate() failure surfacing
# ---------------------------------------------------------------------------

def test_spec_validation_failure_names_cell_exit_1(tmp_path, patched, capsys):
    """A spec whose validate() fires the n_compounds rule -> exit 1, log names
    the failing cell."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)

    err = ("TrainingSpec requires at least one compound molecule "
           "(atom_composition summing to > 1); got only atomic molecules.")
    patched["build_hook"] = lambda rd: _make_specs(rd, n=2, validate_error=err)

    rc = main([str(run_dir)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "failed validation" in out
    # the failing cell is named — spec 0 is metric=l2, subset_size=2
    assert "subset_size=2" in out
    assert "compound molecule" in out
