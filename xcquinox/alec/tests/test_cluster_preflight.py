"""Tests for xcquinox.alec.cluster._preflight: the SLURM preflight entrypoint.

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
import subprocess
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from xcquinox.alec.cluster import _preflight
from xcquinox.alec.cluster._preflight import main
from xcquinox.alec.cluster.grid_config import GridCell


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
# precompute failure handling, on_precompute_failure policy
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
    # re-stage (recompute_refs=False, the failed precompute is NOT re-run)
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
    # the failing cell is named, spec 0 is metric=l2, subset_size=2
    assert "subset_size=2" in out
    assert "compound molecule" in out


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


def test_compile_smoke_gate_pass_exit_0(tmp_path, patched, monkeypatch):
    """cluster.preflight_compile_smoke=True and the probe PASSES -> exit 0, and
    the gate was actually invoked exactly once."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir,
                           cluster_extra={"preflight_compile_smoke": True})

    calls = {"n": 0}

    def fake_smoke(specs, paths, run_dir):
        calls["n"] += 1
        return True

    monkeypatch.setattr(_preflight, "_compile_smoke", fake_smoke)
    assert main([str(run_dir)]) == 0
    assert calls["n"] == 1


def test_compile_smoke_gate_off_by_default_not_called(tmp_path, patched,
                                                      monkeypatch):
    """With the flag OFF (default) the gate is NEVER invoked, so existing runs
    are byte-identical. A sentinel _compile_smoke that raises if called proves
    it, and main still returns 0."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)  # no preflight_compile_smoke -> default False

    def _boom(specs, paths, run_dir):
        raise AssertionError("_compile_smoke must NOT run when the flag is off")

    monkeypatch.setattr(_preflight, "_compile_smoke", _boom)
    assert main([str(run_dir)]) == 0


def test_compile_smoke_impl_classification(monkeypatch):
    """_compile_smoke_impl classifies the probe subprocess result. subprocess.run
    is stubbed (via the _preflight module) so no real worker runs:
      (a) host-OOM text + SIGABRT rc  -> False
      (b) clean completion (done, rc 0) -> True
      (c) done marker + benign teardown rc (-11) -> True
      (d) non-zero rc, no completion marker -> False
      (e) done marker + SIGABRT teardown (-6) with heap-corruption text but NO
          OOM marker -> the epoch finished, teardown crash is benign -> True
      (f) crash before completion with an OOM signal but no OOM text -> block -> False
    """
    cell = SimpleNamespace(arch="deep_attn_3x16", subset_size=26)
    specs = [(cell, object())]
    paths = ["/x"]

    def _cp(stdout, returncode, stderr=""):
        return subprocess.CompletedProcess(
            args=["stub"], returncode=returncode, stdout=stdout, stderr=stderr)

    # (a) host-allocator marker + SIGABRT exit -> OOM -> False
    monkeypatch.setattr(_preflight.subprocess, "run",
                        lambda *a, **k: _cp("Cannot allocate memory", -6))
    assert _preflight._compile_smoke_impl(specs, paths, "/run") is False

    # (b) clean completion (spaced done marker, rc 0) -> True
    monkeypatch.setattr(
        _preflight.subprocess, "run",
        lambda *a, **k: _cp('{"kind": "done", "elapsed_s": 1.0}', 0))
    assert _preflight._compile_smoke_impl(specs, paths, "/run") is True

    # (c) done marker (compact form) + benign C-extension teardown crash -> True
    monkeypatch.setattr(_preflight.subprocess, "run",
                        lambda *a, **k: _cp('{"kind":"done"}', -11))
    assert _preflight._compile_smoke_impl(specs, paths, "/run") is True

    # (d) non-zero rc, no completion marker -> False
    monkeypatch.setattr(_preflight.subprocess, "run",
                        lambda *a, **k: _cp("Traceback ... ValueError", 1))
    assert _preflight._compile_smoke_impl(specs, paths, "/run") is False

    # (e) done marker + SIGABRT teardown (-6/134) carrying the glibc
    # heap-corruption text but NO OOM marker -> the one epoch finished, so the
    # teardown crash is benign -> True. Regression: the old classifier read
    # rc=-6 as an OOM via _looks_like_gpu_oom and would have blocked the array.
    monkeypatch.setattr(
        _preflight.subprocess, "run",
        lambda *a, **k: _cp('{"kind": "done", "elapsed_s": 2.0}\n', -6,
                            stderr="corrupted size vs. prev_size while consolidating"))
    assert _preflight._compile_smoke_impl(specs, paths, "/run") is True

    # (f) crash BEFORE completion with an OOM-ish signal (SIGKILL) but no OOM
    # text and no done marker -> unknown non-completion -> block the array.
    monkeypatch.setattr(_preflight.subprocess, "run",
                        lambda *a, **k: _cp("", -9))
    assert _preflight._compile_smoke_impl(specs, paths, "/run") is False


def test_compile_smoke_probe_runs_in_production_env_and_persists_output(
        tmp_path, monkeypatch):
    """The probe MUST run in the production train-node compile env, not inherit
    the preflight shell's ``OMP=$SLURM_CPUS_PER_TASK`` (24) with no XLA trims --
    that env mismatch is what false-blocked the 030651Z train array
    (``pthread_create failed`` at 24 threads on the heaviest attention cell). And
    the FULL probe output (not just the 500-char tail) must be persisted so a gate
    failure is diagnosable off-cluster.
    """
    cell = SimpleNamespace(arch="deep_attn_3x16", subset_size=26)
    specs = [(cell, object())]
    paths = [str(tmp_path / "spec_0021.spec")]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # 96-core exclusive node -> production BLAS cap = 96 // 12 = 8.
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "96")
    # A hostile inherited value the probe must NOT propagate.
    monkeypatch.setenv("OMP_NUM_THREADS", "24")

    captured = {}
    # Output long enough that the HEAD marker (where a real pthread/OOM signature
    # would sit) falls OUTSIDE the 500-char tail the preflight log keeps -- so the
    # persisted file must capture strictly more than the tail.
    long_stdout = ("PROBE-HEAD-MARKER\n" + ("filler line\n" * 200)
                   + '{"kind": "done", "elapsed_s": 1.0}')

    def _capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout=long_stdout, stderr="PROBE-STDERR-TAIL")

    monkeypatch.setattr(_preflight.subprocess, "run", _capture_run)
    assert _preflight._compile_smoke_impl(
        specs, paths, str(run_dir)) is True
    # Precondition for the regression to be meaningful: the head marker is NOT in
    # the last 500 chars (what the preflight log would show).
    full_probe_text = long_stdout + "\nPROBE-STDERR-TAIL"
    assert "PROBE-HEAD-MARKER" not in full_probe_text[-500:]

    # --- the probe ran in the PRODUCTION compile env ---------------------------
    env = captured["kwargs"]["env"]
    assert env["OMP_NUM_THREADS"] == "8", (
        "probe inherited the preflight 24-thread env instead of the node-scaled "
        "production BLAS cap -- the exact bug that false-blocked 030651Z")
    assert env["MKL_NUM_THREADS"] == "8"
    assert env["OPENBLAS_NUM_THREADS"] == "8"
    assert "--xla_llvm_disable_expensive_passes=true" in env["XLA_FLAGS"]
    # os.environ is MERGED (not replaced): conda/PATH etc. survive.
    assert "PATH" in env

    # --- the FULL probe output is persisted, including the dropped head --------
    probe_out = run_dir / "logs" / "compile_smoke_probe.out"
    assert probe_out.is_file()
    body = probe_out.read_text()
    assert "PROBE-HEAD-MARKER" in body   # the head the 500-char tail would drop
    assert "PROBE-STDERR-TAIL" in body
    assert "blas_threads=8" in body


@pytest.mark.parametrize("cpus_on_node,expected", [("96", "8"), ("28", "2"),
                                                   ("4", "1")])
def test_compile_smoke_probe_blas_cap_scales_with_node(
        tmp_path, monkeypatch, cpus_on_node, expected):
    """Probe BLAS cap = max(1, SLURM_CPUS_ON_NODE // 12), matching the train
    template's node-scaled slice (96->8, 28->2, and the floor of 1 for tiny nodes).
    """
    cell = SimpleNamespace(arch="deep_attn_3x16", subset_size=26)
    specs = [(cell, object())]
    paths = [str(tmp_path / "s.spec")]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", cpus_on_node)

    captured = {}

    def _capture_run(cmd, **kwargs):
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"kind": "done"}', stderr="")

    monkeypatch.setattr(_preflight.subprocess, "run", _capture_run)
    assert _preflight._compile_smoke_impl(
        specs, paths, str(run_dir)) is True

    env = captured["kwargs"]["env"]
    assert env["OMP_NUM_THREADS"] == expected
    assert env["MKL_NUM_THREADS"] == expected
    assert env["OPENBLAS_NUM_THREADS"] == expected


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


def test_preflight_blocks_the_array_on_a_failed_certificate(tmp_path,
                                                            patched, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)
    _write_pass_certificate(run_dir, "shallow", verdict="FAIL")
    assert main([str(run_dir)]) == 1
    out = capsys.readouterr().out
    assert "fidelity gate FAILED" in out


def test_preflight_reports_the_gate_when_every_arch_certifies(tmp_path,
                                                              patched,
                                                              capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)
    assert main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "fidelity gate PASSED" in out
    assert "1/1 architecture certificate(s) released the gate" in out
    assert "preflight SUCCEEDED" in out


def test_preflight_releases_an_unenforced_failure(tmp_path, patched, capsys):
    """A workflow-verification run must reach its train array with the FAIL on
    record; the preflight log says the gate was not enforced."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)
    d = os.path.join(str(run_dir), "pretrain", "shallow")
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump({"verdict": "FAIL", "arch": "shallow", "enforced": False,
                   "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                                  "override_reason": "workflow matrix"},
                   "summary": {"max_atom_mHa": 13.7,
                               "max_dAE_kcalmol": 25.7}}, f)
    assert main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "enforcement is OFF" in out
    assert "preflight SUCCEEDED" in out


def test_preflight_checks_every_distinct_arch(tmp_path, patched, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = _write_resolved_config(run_dir)
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["sweep"]["arch"] = ["shallow", "medium"]
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    _write_pass_certificate(run_dir, "shallow")
    # "medium" has no certificate: the sweep must catch it.
    assert main([str(run_dir)]) == 1
    out = capsys.readouterr().out
    assert "medium" in out
