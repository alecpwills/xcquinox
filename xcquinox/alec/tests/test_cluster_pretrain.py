"""Tests for xcquinox.alec.cluster._pretrain.

The heavy ``run_pretrain`` call is monkeypatched at the ``_run_pretrain`` seam
so NO real pretraining / JAX compute is ever spawned. A synthetic ``run_dir``
(a minimal ``resolved_config.yaml``) is built per-test in a tmp directory.

Coverage:
  - ``_pretrain.main`` loads ``resolved_config.yaml``, selects the correct
    arch for a given ``arch_idx``, and builds a ``PretrainSpec`` with the
    right ``checkpoint_dir`` = ``<run_dir>/pretrain/<arch>/`` and every
    ``cfg.pretrain`` field threaded through.
  - out-of-range ``arch_idx`` fails fast (non-zero exit, clear message).
  - the silent-no-checkpoint guard: mocked ``_run_pretrain`` "succeeds" but
    writes nothing -> worker exits non-zero.
  - mocked ``_run_pretrain`` writes ``xnet.eqx`` + ``cnet.eqx`` -> exit 0.
  - the throttled progress callback emits at least one
    ``[harness pretrain arch=`` line over a multi-step run and does not crash
    on a zero-step run.
  - the ``pretrain.sbatch.tmpl`` template is a valid ``string.Template`` and
    renders with no leftover ``${...}``, and the rendered body invokes
    ``python -m xcquinox.alec.cluster._pretrain``.
"""
import importlib.resources
import json
import os
import sys
from string import Template

from xcquinox.alec.parallel import PYSCF_POOL_THREADS_MAX

import pytest

from xcquinox.alec.cluster import _pretrain as pt
from xcquinox.alec.config import PretrainSpec, get_architecture


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _config_dict(archs=("medium", "shallow", "deep"),
                  data_dir="/tmp/pretrain_data"):
    """A complete-but-minimal resolved config; ``load_grid_config`` needs every
    section, but only ``sweep.arch`` and ``pretrain`` are read by the worker."""
    return {
        "sweep": {
            "arch": list(archs),
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
            "data_dir": data_dir,
            "n_steps": 777,
            "lr_start": 5e-2,
            "lr_end": 3e-5,
            "lr_decay_start": 0.25,
            "grad_clip": 0.5,
            "seed": 17,
            "loss_weighting": "integration",
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


def _write_config(run_dir, cfg=None):
    """Write resolved_config.yaml (JSON fallback) and return its path."""
    cfg = cfg or _config_dict()
    path = os.path.join(run_dir, "resolved_config.yaml")
    try:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
    except ImportError:  # pragma: no cover -- env-dependent
        path = os.path.join(run_dir, "resolved_config.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
    return path


@pytest.fixture
def run_dir(tmp_path):
    """A run dir whose config points data_dir at a tmp-scoped path so checkpoint
    artifacts never leak between tests. Pretrain checkpoints land under the run
    dir itself (<run_dir>/pretrain/<arch>)."""
    d = tmp_path / "run"
    d.mkdir()
    data_dir = tmp_path / "pretrain_data"
    data_dir.mkdir()
    _write_config(
        str(d),
        _config_dict(data_dir=str(data_dir)),
    )
    return str(d)


def _stub_certificate_seam(monkeypatch, payload, *, write=True, record=None):
    """Replace the certificate seam with one that writes ``payload`` and
    returns it -- what :func:`fidelity.fidelity_certificate` itself does.

    The file matters: the worker's gate reads the certificate back through the
    shared predicate, so a stub that only returned a payload would describe a
    run the train task and the preflight could not reproduce. ``write=False``
    stubs a certificate call that returns a payload without leaving one on
    disk. ``record`` is an optional dict the call's arguments are stored in.
    """
    def fake_certificate(cfg, run_dir, arch):
        if record is not None:
            record["args"] = (run_dir, arch)
            record["tol"] = (cfg.fidelity.tol_AE, cfg.fidelity.tol_atom)
        if write:
            d = os.path.join(run_dir, "pretrain", arch)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
                json.dump(payload, f)
        return payload

    monkeypatch.setattr(pt, "_fidelity_certificate", fake_certificate)
    return fake_certificate


def _pass_payload(**overrides):
    payload = {
        "verdict": "PASS", "enforced": True,
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": None},
        "summary": {"max_atom_mHa": 0.12, "max_dAE_kcalmol": 0.34,
                    "n_systems": 40, "n_atoms": 16, "n_atomizations": 24,
                    "failure_reasons": []},
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def stub_certificate(request, monkeypatch):
    """Stub the fidelity certificate at its seam for every test in this file.

    The certificate loads the checkpoint and runs PySCF SCFs at the run's
    identity; the pretrain-worker tests are about worker orchestration, so
    they get a PASS payload for free. The tests that exercise the gate
    override this with their own seam. A test whose name ends in
    ``_unstubbed`` opts out entirely, which is how the seam-identity test can
    observe the real module-level binding.
    """
    if request.node.name.endswith("_unstubbed"):
        return
    _stub_certificate_seam(monkeypatch, _pass_payload())


# ---------------------------------------------------------------------------
# distinct-arch derivation + spec construction
# ---------------------------------------------------------------------------

def test_distinct_archs_is_sorted_and_deduped(run_dir):
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(
        os.path.join(run_dir, "resolved_config.yaml")
        if os.path.isfile(os.path.join(run_dir, "resolved_config.yaml"))
        else os.path.join(run_dir, "resolved_config.json")
    )
    # _config_dict declares ("medium", "shallow", "deep") -> sorted-deduped.
    assert pt._distinct_archs(cfg) == ["deep", "medium", "shallow"]


def test_main_builds_pretrain_spec_with_correct_checkpoint_dir(run_dir, monkeypatch):
    captured = {}

    def fake_run_pretrain(spec, progress_callback=None):
        captured["spec"] = spec
        # emulate run_pretrain writing both checkpoints.
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {"arch_name": spec.arch.name}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)

    # arch_idx 1 -> sorted distinct archs ["deep","medium","shallow"][1] = "medium".
    rc = pt.main([run_dir, "1"])
    assert rc == 0

    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(run_dir, "resolved_config.json")
    cfg = load_grid_config(cfg_path)

    spec = captured["spec"]
    assert isinstance(spec, PretrainSpec)
    assert spec.arch.name == "medium"
    # checkpoint_dir == <run_dir>/pretrain/<arch>/ (run-scoped so two runs
    # pretraining the same arch don't collide; run_dir is unique per submission).
    assert spec.checkpoint_dir == os.path.join(
        os.path.abspath(run_dir), "pretrain", "medium"
    )
    # every cfg.pretrain field threaded through.
    assert spec.data_dir == cfg.pretrain.data_dir
    assert spec.n_steps == 777
    assert spec.lr_start == 5e-2
    assert spec.lr_end == 3e-5
    assert spec.lr_decay_start == 0.25
    assert spec.grad_clip == 0.5
    assert spec.seed == 17
    assert spec.loss_weighting == "integration"


def test_main_selects_arch_by_index(run_dir, monkeypatch):
    seen = []

    def fake_run_pretrain(spec, progress_callback=None):
        seen.append(spec.arch.name)
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)

    for idx, expect in enumerate(["deep", "medium", "shallow"]):
        seen.clear()
        assert pt.main([run_dir, str(idx)]) == 0
        assert seen == [expect]


# ---------------------------------------------------------------------------
# out-of-range arch_idx
# ---------------------------------------------------------------------------

def test_out_of_range_arch_idx_fails_fast(run_dir, monkeypatch):
    monkeypatch.setattr(
        pt, "_run_pretrain",
        lambda *a, **k: pytest.fail("_run_pretrain ran for an out-of-range idx"))
    # 3 distinct archs -> valid indices 0..2; 3 is out of range.
    assert pt.main([run_dir, "3"]) != 0


def test_negative_arch_idx_fails_fast(run_dir, monkeypatch):
    monkeypatch.setattr(
        pt, "_run_pretrain",
        lambda *a, **k: pytest.fail("_run_pretrain ran for a negative idx"))
    assert pt.main([run_dir, "-1"]) != 0


# ---------------------------------------------------------------------------
# silent-no-checkpoint guard
# ---------------------------------------------------------------------------

def test_silent_no_checkpoint_exits_nonzero(run_dir, monkeypatch):
    # _run_pretrain "succeeds" (returns normally) but writes nothing.
    monkeypatch.setattr(pt, "_run_pretrain", lambda spec, progress_callback=None: {})
    assert pt.main([run_dir, "0"]) != 0


def test_partial_checkpoint_exits_nonzero(run_dir, monkeypatch):
    # Only xnet.eqx written -> still a failure (cnet.eqx missing).
    def fake_run_pretrain(spec, progress_callback=None):
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        return {}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)
    assert pt.main([run_dir, "0"]) != 0


def test_both_checkpoints_written_exits_zero(run_dir, monkeypatch):
    def fake_run_pretrain(spec, progress_callback=None):
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)
    assert pt.main([run_dir, "0"]) == 0


def test_run_pretrain_exception_exits_nonzero(run_dir, monkeypatch):
    def boom(spec, progress_callback=None):
        raise RuntimeError("pretrain blew up")

    monkeypatch.setattr(pt, "_run_pretrain", boom)
    assert pt.main([run_dir, "0"]) != 0


# ---------------------------------------------------------------------------
# missing config
# ---------------------------------------------------------------------------

def test_missing_resolved_config_exits_nonzero(tmp_path, monkeypatch):
    d = tmp_path / "emptyrun"
    d.mkdir()
    monkeypatch.setattr(
        pt, "_run_pretrain", lambda *a, **k: pytest.fail("ran"))
    assert pt.main([str(d), "0"]) != 0


# ---------------------------------------------------------------------------
# JAX env routing
# ---------------------------------------------------------------------------

def test_route_jax_env_sets_x64(monkeypatch):
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    pt._route_jax_env()
    assert os.environ["JAX_ENABLE_X64"] == "1"


# ---------------------------------------------------------------------------
# throttled progress callback
# ---------------------------------------------------------------------------

def test_progress_callback_emits_harness_line(capsys):
    cb = pt._make_progress_callback("medium")
    for step in range(1, 251):
        cb({"arch": "medium", "phase": "X", "step": step, "total": 250,
            "loss": 0.01, "timestamp": 0.0})
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines()
             if ln.startswith("[harness pretrain arch=medium]")]
    assert len(lines) >= 1
    assert all("phase=X" in ln and "loss=" in ln for ln in lines)


def test_progress_callback_zero_step_run_does_not_crash(capsys):
    cb = pt._make_progress_callback("shallow")
    # A zero-step run: callback is simply never invoked -- must not crash and
    # must produce no harness progress line.
    out = capsys.readouterr().out
    assert "[harness pretrain arch=" not in out
    # And invoking it once at step 0 (defensive) also must not crash.
    cb({"arch": "shallow", "phase": "X", "step": 0, "total": 0, "loss": 0.0,
        "timestamp": 0.0})


def test_progress_callback_emits_on_phase_switch(capsys):
    cb = pt._make_progress_callback("deep")
    cb({"phase": "X", "step": 1, "total": 10, "loss": 0.1, "timestamp": 0.0})
    cb({"phase": "C", "step": 1, "total": 10, "loss": 0.2, "timestamp": 0.0})
    out = capsys.readouterr().out
    assert "phase=X" in out
    assert "phase=C" in out


# ---------------------------------------------------------------------------
# pretrain.sbatch.tmpl template
# ---------------------------------------------------------------------------

def _template_text():
    res = (
        importlib.resources.files("xcquinox.alec.cluster")
        / "templates" / "pretrain.sbatch.tmpl"
    )
    return res.read_text(encoding="utf-8")


def test_pretrain_template_renders_with_no_leftover_placeholders():
    text = _template_text()
    mapping = {
        "JOB_NAME": "xcq_pretrain",
        "PARTITION": "short",
        "TIME": "04:00:00",
        "ALLOC_LINES": "#SBATCH --nodes=1\n#SBATCH --exclusive\n",
        "MEM_LINE": "",
        "CPUS_PER_TASK": 4,
        "ARRAY_MAX": 2,
        "THROTTLE": 3,
        "RUN_DIR": "/scratch/run",
        "CONDA_ACTIVATION": "conda activate xcq",
        "MAIL_USER_LINE": "",
        "MAIL_TYPE_LINE": "",
        "ACCOUNT_LINE": "",
        "PYSCF_POOL_THREADS_MAX": PYSCF_POOL_THREADS_MAX,
    }
    rendered = Template(text).substitute(mapping)
    # No leftover harness placeholder name survives. ``string.Template`` turns
    # ``$$`` into a literal ``$`` (so ``$${SLURM_ARRAY_TASK_ID}`` -> the bash
    # token ``${SLURM_ARRAY_TASK_ID}``); the meaningful check is that every
    # mapping KEY was substituted, leaving no ``${KEY}`` token behind.
    for key in mapping:
        assert "${" + key + "}" not in rendered
    # The rendered body invokes the pretrain worker module.
    assert "python -m xcquinox.alec.cluster._pretrain" in rendered
    # SLURM array directive present + per-array log path.
    assert "#SBATCH --array=0-2%3" in rendered
    assert "logs/pretrain_%A_%a.out" in rendered
    # SLURM's own ${SLURM_ARRAY_TASK_ID} survives string.Template ($$ -> $).
    assert "${SLURM_ARRAY_TASK_ID}" in rendered
    # The PySCF-serving pools are capped from the allocation
    # (parallel.pyscf_pool_threads), the shell default standing in for a
    # manual run outside SLURM.
    assert f"PYSCF_THREADS=${{SLURM_CPUS_PER_TASK:-{PYSCF_POOL_THREADS_MAX}}}" in rendered
    assert 'export OMP_NUM_THREADS="$PYSCF_THREADS"' in rendered


def test_pretrain_template_is_valid_string_template():
    # Template construction + identifier scan must not raise.
    tmpl = Template(_template_text())
    # Every identifier is a plain placeholder name (string.Template accepts it).
    assert isinstance(tmpl.template, str)


def test_pretrain_template_invokes_only_the_certifying_worker():
    """The certificate runs INSIDE ``_pretrain``, not as a second command.

    ``_pretrain.main`` certifies the checkpoint it has just written, on the
    node that holds it, at the run's identity, and folds the verdict into
    THIS job's exit code -- which is what the train array's ``afterok``
    dependency already reads. A second ``python -m`` line in this template
    would pay the JAX / PySCF import a second time, would need failure
    semantics of its own to make ``set -e`` block that dependency, and would
    still land on the same node against the same wall clock; a separate job
    kind would add a dependency edge, a submission record and a log family
    for one function call. The template therefore carries exactly one
    invocation, and this pins it.
    """
    text = _template_text()
    invocations = [ln.strip() for ln in text.splitlines()
                   if ln.strip().startswith("python -m")]
    assert invocations == [
        "python -m xcquinox.alec.cluster._pretrain "
        "${RUN_DIR} $${SLURM_ARRAY_TASK_ID}"
    ]


def test_pretrain_arch_polarized_when_flag_set(tmp_path, monkeypatch):
    """The pretrain stage rebuilds its arch spin-polarization-aware when the run
    config sets use_polarized_correlation, so the pretrained checkpoint matches
    the (polarized) training arch."""
    captured = {}

    def fake_run_pretrain(spec, progress_callback=None):
        captured["spec"] = spec
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {"arch_name": spec.arch.name}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)
    d = tmp_path / "run"
    d.mkdir()
    data_dir = tmp_path / "pretrain_data"
    data_dir.mkdir()
    cd = _config_dict(data_dir=str(data_dir))
    cd["use_polarized_correlation"] = True
    _write_config(str(d), cd)

    assert pt.main([str(d), "1"]) == 0
    assert captured["spec"].arch.use_polarized_correlation is True


# ---------------------------------------------------------------------------
# The on-node fidelity gate
# ---------------------------------------------------------------------------

def _stub_pretrain_writes_checkpoint(monkeypatch):
    def fake_run_pretrain(spec, progress_callback=None):
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {}
    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)


def _fail_payload(**overrides):
    payload = {
        "verdict": "FAIL", "enforced": True,
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": None},
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7,
                    "n_systems": 40, "n_atoms": 16, "n_atomizations": 24,
                    "failure_reasons": ["max |dE_xc| over free atoms 13.7000 "
                                        "mHa exceeds tol_atom 1.0 mHa"]},
    }
    payload.update(overrides)
    return payload


def test_pretrain_runs_the_certificate_for_its_own_arch(run_dir, monkeypatch):
    _stub_pretrain_writes_checkpoint(monkeypatch)
    seen = {}
    _stub_certificate_seam(monkeypatch, _pass_payload(), record=seen)
    assert pt.main([run_dir, "1"]) == 0
    assert seen["args"] == (os.path.abspath(run_dir), "medium")
    assert seen["tol"] == (1.0, 1.0)


def test_pretrain_exits_nonzero_on_a_failed_certificate(run_dir, monkeypatch,
                                                        capsys):
    _stub_pretrain_writes_checkpoint(monkeypatch)
    _stub_certificate_seam(monkeypatch, _fail_payload())
    assert pt.main([run_dir, "1"]) == 1
    out = capsys.readouterr().out
    assert "fidelity certificate FAILED" in out
    assert "13.7" in out and "25.7" in out
    assert "tol_atom" in out


def test_pretrain_continues_past_a_failure_when_enforcement_is_off(
        run_dir, monkeypatch, capsys):
    """Workflow-verification runs must reach the train stage with a FAIL on
    record; the worker says so in the log and exits 0."""
    _stub_pretrain_writes_checkpoint(monkeypatch)
    _stub_certificate_seam(monkeypatch, _fail_payload(
        enforced=False,
        tolerances={"tol_AE": 1.0, "tol_atom": 1.0,
                    "override_reason": "workflow matrix: 50-step pretrain"}))
    assert pt.main([run_dir, "1"]) == 0
    out = capsys.readouterr().out
    assert "fidelity certificate FAILED" in out
    assert "enforcement is OFF" in out
    assert "workflow matrix" in out
    assert "pretrain SUCCEEDED" in out


@pytest.mark.parametrize("reason", (None, "", "   ", False, 0))
def test_pretrain_refuses_a_waiver_that_states_no_reason(run_dir, monkeypatch,
                                                         capsys, reason):
    """Only a waiver with a written reason releases the worker's exit code.

    Disabling the on-node gates requires a non-empty prose
    ``fidelity.override_reason``. ``load_grid_config`` accepts a ``fidelity``
    block whose ``enforce`` is false with no reason -- ``main`` never calls
    ``validate_grid_semantics`` -- so the rule is imposed on the certificate
    itself, where a truthiness test on ``enforced`` alone would let a run with
    no reason on record report success. A boolean or a number is not a reason:
    ``str(False)`` is the non-empty string 'False'.
    """
    _stub_pretrain_writes_checkpoint(monkeypatch)
    _stub_certificate_seam(monkeypatch, _fail_payload(
        enforced=False,
        tolerances={"tol_AE": 1.0, "tol_atom": 1.0,
                    "override_reason": reason}))
    assert pt.main([run_dir, "1"]) == 1
    out = capsys.readouterr().out
    assert "override_reason" in out
    assert "pretrain SUCCEEDED" not in out


def test_pretrain_refuses_a_certificate_that_records_no_verdict(
        run_dir, monkeypatch, capsys):
    """A verdict-less certificate is UNREADABLE, and never waivable.

    FAIL is the one status a run can waive, so a file that states no
    recognised verdict must not be read as one: a truncated or schema-less
    certificate with ``enforced: false`` would otherwise release the stage.
    The stage names the status the record layer gives it.
    """
    _stub_pretrain_writes_checkpoint(monkeypatch)
    _stub_certificate_seam(monkeypatch, _fail_payload(
        verdict=None, enforced=False,
        tolerances={"tol_AE": 1.0, "tol_atom": 1.0,
                    "override_reason": "workflow matrix"}))
    assert pt.main([run_dir, "1"]) == 1
    out = capsys.readouterr().out
    assert "UNREADABLE" in out
    assert "pretrain SUCCEEDED" not in out


def test_pretrain_refuses_a_payload_that_is_not_a_certificate(run_dir,
                                                              monkeypatch,
                                                              capsys):
    """A certificate call that returns something other than a payload is a
    diagnosed refusal, not an exception out of ``main``."""
    _stub_pretrain_writes_checkpoint(monkeypatch)
    monkeypatch.setattr(pt, "_fidelity_certificate",
                        lambda cfg, rd, arch: ["not", "a", "certificate"])
    assert pt.main([run_dir, "1"]) == 1
    out = capsys.readouterr().out
    assert "fidelity certificate" in out
    assert "pretrain SUCCEEDED" not in out


def test_pretrain_refuses_when_no_certificate_reached_disk(run_dir,
                                                           monkeypatch,
                                                           capsys):
    """The gate reads the FILE, which is what the later stages read.

    The train task and the preflight decide from
    ``pretrain/<arch>/fidelity_certificate.json``; a worker that reported
    success on an in-memory payload that never landed would hand them a run
    they refuse, with the pretrain job recorded as successful.
    """
    _stub_pretrain_writes_checkpoint(monkeypatch)
    _stub_certificate_seam(monkeypatch, _pass_payload(), write=False)
    assert pt.main([run_dir, "1"]) == 1
    out = capsys.readouterr().out
    assert "pretrain SUCCEEDED" not in out


def test_pretrain_exits_nonzero_when_the_certificate_raises(run_dir,
                                                            monkeypatch,
                                                            capsys):
    _stub_pretrain_writes_checkpoint(monkeypatch)

    def boom(cfg, rd, arch):
        raise RuntimeError("libxc unavailable")

    monkeypatch.setattr(pt, "_fidelity_certificate", boom)
    assert pt.main([run_dir, "1"]) == 1
    out = capsys.readouterr().out
    assert "fidelity certificate RAISED" in out
    assert "libxc unavailable" in out


def test_pretrain_logs_the_passing_summary(run_dir, monkeypatch, capsys):
    _stub_pretrain_writes_checkpoint(monkeypatch)
    assert pt.main([run_dir, "1"]) == 0
    out = capsys.readouterr().out
    assert "fidelity certificate PASSED" in out
    assert "pretrain SUCCEEDED" in out
    # The certificate line precedes the SUCCEEDED line: the job only reports
    # success after the physics has been checked.
    assert out.index("fidelity certificate PASSED") < out.index(
        "pretrain SUCCEEDED")


def test_pretrain_reports_the_certificate_that_reached_disk(run_dir,
                                                            monkeypatch,
                                                            capsys):
    """The numbers in the log come from the FILE the gate acts on.

    The certificate call both writes the file and returns the payload, so the
    two agree in the ordinary case; they part company when the write did not
    land what the call returned -- an interrupted or refused rewrite leaving an
    older certificate in place. Reporting the returned payload while deciding
    on the file would put one document's numbers beside the other's verdict in
    the same line, and the numbers on record would not be the ones any later
    stage reads.
    """
    _stub_pretrain_writes_checkpoint(monkeypatch)
    on_disk = _fail_payload()

    def seam(cfg, rd, arch):
        d = os.path.join(rd, "pretrain", arch)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
            json.dump(on_disk, f)
        return _pass_payload()

    monkeypatch.setattr(pt, "_fidelity_certificate", seam)
    assert pt.main([run_dir, "1"]) == 1
    out = capsys.readouterr().out
    assert "fidelity certificate FAILED" in out
    assert "13.7" in out and "25.7" in out
    # The returned payload's numbers appear nowhere: one document is quoted.
    assert "0.12" not in out and "0.34" not in out
    assert "pretrain SUCCEEDED" not in out


def _serve_after_the_first_read(monkeypatch, document):
    """Serve ``document`` to every certificate READ after the first.

    The list returned collects one entry per read, so a caller can state how
    many parses its decision rested on. The first read is passed through to
    the file on disk and writes always are, so the seam still lands the
    certificate it wrote; only a LATER read sees the rewrite.
    """
    import builtins
    import io as _io
    real_open = builtins.open
    reads: list = []

    def fake_open(file, *args, **kwargs):
        path = str(file)
        mode = kwargs.get("mode", args[0] if args else "r")
        if path.endswith("fidelity_certificate.json") and "r" in mode:
            reads.append(path)
            if len(reads) > 1:
                return _io.StringIO(json.dumps(document))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return reads


def test_pretrain_gates_on_the_document_it_reports(run_dir, monkeypatch,
                                                   capsys):
    """The line, the verdict acted on and the exit code come from ONE parse.

    Reading the file for the summary and letting the gate open it again lets
    a certificate rewritten between the opens produce a report that states
    two documents at once: the numbers of the file as it was beside the
    verdict of the file as it became -- 'certificate FAILED ... / gate: PASS
    / SUCCEEDED' for a run whose recorded verdict is FAIL. Here every read
    after the first would find a PASS.
    """
    _stub_pretrain_writes_checkpoint(monkeypatch)
    _stub_certificate_seam(monkeypatch, _fail_payload())
    reads = _serve_after_the_first_read(monkeypatch, _pass_payload())
    rc = pt.main([run_dir, "1"])
    monkeypatch.undo()
    out = capsys.readouterr().out
    assert rc == 1, out
    assert len(reads) == 1, reads
    assert "fidelity certificate FAILED" in out
    assert "13.7" in out and "25.7" in out
    # The rewritten document's verdict reaches neither the gate nor the log.
    assert "fidelity certificate PASS" not in out
    assert "pretrain SUCCEEDED" not in out


def test_pretrain_does_not_certify_when_the_checkpoint_is_missing(
        run_dir, monkeypatch):
    """A worker that wrote no checkpoint fails at the existing guard; the
    certificate must not be attempted against an absent xnet.eqx."""
    monkeypatch.setattr(pt, "_run_pretrain", lambda spec, progress_callback=None: {})
    called = []
    monkeypatch.setattr(pt, "_fidelity_certificate",
                        lambda *a, **k: called.append(1))
    assert pt.main([run_dir, "1"]) == 1
    assert called == []


def test_fidelity_certificate_seam_is_the_library_function_unstubbed():
    """One implementation of the certificate, bound as a seam -- not a wrapper
    that could drift from what the library actually runs."""
    from xcquinox.alec.cluster import fidelity
    assert pt._fidelity_certificate is fidelity.fidelity_certificate


def test_pretrain_spec_carries_the_protocol_fields(tmp_path, monkeypatch):
    """A field the worker forgets to thread is a knob the YAML sets and the run
    silently ignores."""
    from xcquinox.alec.cluster import _pretrain as pretrain_mod
    d = tmp_path / "run"
    d.mkdir()
    data_dir = tmp_path / "pretrain_data"
    data_dir.mkdir()
    cfg = _config_dict(data_dir=str(data_dir))
    cfg["pretrain"].update({
        "parent_density": "auto", "energy_term_weight": 1.0,
        "validation_fraction": 0.2, "validation_seed": 11,
        "validate_every": 25, "patience": 8})
    _write_config(str(d), cfg)
    captured = {}

    def _fake(spec, progress_callback=None):
        captured["spec"] = spec
        return {}

    monkeypatch.setattr(pretrain_mod, "_run_pretrain", _fake)
    pretrain_mod.main([str(d), "0"])
    spec = captured["spec"]
    assert spec.parent_density == "auto"
    assert spec.energy_term_weight == 1.0
    assert spec.validation_fraction == 0.2
    assert spec.validation_seed == 11
    assert spec.validate_every == 25
    assert spec.patience == 8


def test_pretrain_log_line_states_every_protocol_knob(tmp_path, monkeypatch,
                                                      capsys):
    """The run record must show what the job trained with. A knob that is set
    in the YAML and absent from the log leaves no way to attribute a
    checkpoint to the configuration that produced it."""
    from xcquinox.alec.cluster import _pretrain as pretrain_mod
    d = tmp_path / "run"
    d.mkdir()
    data_dir = tmp_path / "pretrain_data"
    data_dir.mkdir()
    cfg = _config_dict(data_dir=str(data_dir))
    cfg["pretrain"].update({
        "parent_density": "scan", "energy_term_weight": 2.5,
        "validation_fraction": 0.2, "validation_seed": 11,
        "validate_every": 25, "patience": 8, "dfs_set": True,
        "pool_atoms": True, "exchange_footing": "spin_channel",
        "mesh_fraction": 0.25})
    _write_config(str(d), cfg)
    monkeypatch.setattr(pretrain_mod, "_run_pretrain",
                        lambda spec, progress_callback=None: {})
    pretrain_mod.main([str(d), "0"])
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if "running run_pretrain" in ln]
    assert len(line) == 1
    for stated in ("parent_density='scan'", "energy_term_weight=2.5",
                   "validation_fraction=0.2", "validation_seed=11",
                   "validate_every=25", "patience=8", "dfs_set=True",
                   "pool_atoms=True", "exchange_footing='spin_channel'",
                   "mesh_fraction=0.25"):
        assert stated in line[0], stated


# ---------------------------------------------------------------------------
# The exit status the array task hands SLURM
# ---------------------------------------------------------------------------

def _abort_injection(tmp_path):
    """A ``sitecustomize`` that registers ``os.abort`` as an atexit handler.

    This puts a real launch of the worker into the state the cluster produced
    without editing the worker: ``site`` imports ``sitecustomize`` at startup,
    so the handler is registered before the module runs -- exactly as JAX
    registers its backend cleanup when it is imported.
    """
    inject = tmp_path / "inject"
    inject.mkdir()
    (inject / "sitecustomize.py").write_text(
        "import atexit, os\natexit.register(os.abort)\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(inject)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.pop("XCQ_NO_HARD_EXIT", None)
    return env


def _launch_pretrain(env, *argv):
    """Launch the worker the way the rendered ``pretrain.sbatch`` does."""
    import subprocess
    return subprocess.run(
        [sys.executable, "-m", "xcquinox.alec.cluster._pretrain", *argv],
        capture_output=True, text=True, timeout=600, env=env)


@pytest.mark.parametrize("argv,expected_rc", [
    ((), 2),                              # argparse usage exit
    (("/nonexistent/run", "0"), 1),       # the worker's own refusal
])
def test_pretrain_exit_status_survives_an_abort_at_teardown(
        tmp_path, argv, expected_rc):
    """The defect measured on cluster job 2134455, node dn024.

    The pretrain worker completed both phases, wrote
    ``fidelity_certificate.json``, ``xnet.eqx`` and ``cnet.eqx``, logged
    ``pretrain SUCCEEDED``, and the interpreter then died in glibc's
    ``corrupted size vs. prev_size while consolidating`` during teardown -- so
    the array task was recorded as killed by SIGABRT and the train array's
    ``afterok`` dependency would never have fired. The worker's own verdict
    must reach the caller whatever teardown does; a return code the worker
    produced is asserted here rather than a successful pretrain, because the
    property under test is the EXIT, not the training.
    """
    proc = _launch_pretrain(_abort_injection(tmp_path), *argv)
    assert proc.returncode == expected_rc, (
        f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")


def test_red_pretrain_under_stock_teardown_is_reported_as_a_signal_death(
        tmp_path):
    """The same launch with ``XCQ_NO_HARD_EXIT=1``: the pre-change behaviour.

    -6 is SIGABRT as ``subprocess`` reports it, 134 as a shell does. Without
    this control the assertion above would not show that anything changed.
    """
    import resource
    env = _abort_injection(tmp_path)
    env["XCQ_NO_HARD_EXIT"] = "1"
    # The abort is deliberate; do not leave a core behind. The limit is
    # lowered on this process and inherited by the child, rather than being
    # set child-side, which would force subprocess down the fork() path that
    # the interpreter warns can deadlock a child of a JAX process.
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
    try:
        proc = _launch_pretrain(env, "/nonexistent/run", "0")
    finally:
        resource.setrlimit(resource.RLIMIT_CORE, (soft, hard))
    assert proc.returncode == -6, (
        f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")


def test_pretrain_log_reaches_the_caller_through_the_hard_exit(tmp_path):
    """``os._exit`` runs no finalizers, so the worker's log must be flushed.

    The SLURM log is the only record of why a stage refused; an exit that
    preserved the status but dropped the log would trade one defect for
    another.
    """
    proc = _launch_pretrain(_abort_injection(tmp_path), "/nonexistent/run", "0")
    assert "resolved_config.yaml not found" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# skip-if-complete: a pretrain task killed after success is not redone
# ---------------------------------------------------------------------------
# A pretrain array task has no resubmit path -- ``cmd_resubmit`` reduces the
# train and eval kinds only -- so the recovery for a dead pretrain stage is
# ``resubmit-preflight``, which re-submits the whole
# pretrain -> preflight -> train -> eval graph. Without the check below that
# recovery repeats every architecture's pretraining from scratch, including the
# ones whose networks and certificate are already on disk, which is the
# expensive half of the graph. The release rule for keeping one is the ON-NODE
# gate the stage already applies to its own verdict
# (``fidelity.gate_certificate_from_read``), so a kept architecture is one a
# later stage would accept: PASS, or FAIL under a recorded waiver that states
# its reason. Anything else -- a missing network, no certificate, an unreadable
# one, an enforced FAIL -- is redone. A released verdict is necessary and not
# sufficient: the certificate must also describe THESE networks at THIS run's
# identity, against this architecture's parent, which is the second block of
# cases at the end of this file.


def _completed_pretrain(run_dir, arch, payload, *, networks=("xnet.eqx",
                                                             "cnet.eqx")):
    """Write the artifacts a completed pretrain task leaves behind.

    A dict payload is completed with the facts the certificate writer records
    beside the verdict -- this run's identity, the architecture's parent and
    the SHA-256 digests of the networks just written -- unless it states them
    itself. The keep check compares those three against the run, so a document
    that omitted them would be refused for the omission and could say nothing
    about the release rule each case here is written for; the cases that
    perturb them are separate and use the writer's own output.
    """
    from xcquinox.alec.cluster import fidelity
    from xcquinox.alec.cluster.grid_config import load_grid_config
    from xcquinox.alec.cluster.materialize import _sha256_file

    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    for name in networks:
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"checkpoint-bytes-" + name.encode())
    if payload is not None:
        if isinstance(payload, dict):
            cfg = load_grid_config(
                os.path.join(run_dir, "resolved_config.yaml"))
            payload.setdefault("identity", fidelity.run_identity(cfg))
            payload.setdefault("parent", fidelity.resolve_parent(arch))
            digests = {
                key: _sha256_file(os.path.join(d, name))
                for name, key in fidelity.CHECKPOINT_DIGEST_KEYS
                if name in networks
            }
            payload.setdefault("checkpoint", digests)
        with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
            if isinstance(payload, str):
                f.write(payload)
            else:
                json.dump(payload, f)
    return d


def _waived_fail_payload():
    """A FAIL the on-node gate releases: enforcement off, reason recorded."""
    return _pass_payload(
        verdict="FAIL", enforced=False,
        tolerances={"tol_AE": 1.0, "tol_atom": 1.0,
                    "override_reason": "workflow matrix: wiring check"},
    )


def _forbid(monkeypatch, *names):
    """Bind each seam to a call that fails the test if it is reached."""
    for name in names:
        def _refuse(*a, _name=name, **kw):
            raise AssertionError(f"{_name} was called; the completed "
                                 "pretraining should have been kept")
        monkeypatch.setattr(pt, name, _refuse)


def _recorder(monkeypatch):
    """Stub ``_run_pretrain`` so it writes both networks and counts calls."""
    calls = []

    def fake_run_pretrain(spec, progress_callback=None):
        calls.append(spec.checkpoint_dir)
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        for name in ("xnet.eqx", "cnet.eqx"):
            with open(os.path.join(spec.checkpoint_dir, name), "wb") as f:
                f.write(b"redone")
        return {}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)
    return calls


@pytest.mark.parametrize("payload,label", [
    (_pass_payload(), "PASS"),
    (_waived_fail_payload(), "waived FAIL"),
])
def test_a_certified_pretrain_is_kept_and_not_redone(run_dir, monkeypatch,
                                                     capsys, payload, label):
    """Both networks plus a certificate the on-node gate releases -> exit 0.

    Neither the pretraining nor the certificate is recomputed: both seams are
    bound to a call that fails the test if it is reached.
    """
    _completed_pretrain(run_dir, "deep", payload)
    _forbid(monkeypatch, "_run_pretrain", "_fidelity_certificate")

    assert pt.main([run_dir, "0"]) == 0, label
    out = capsys.readouterr().out
    assert "KEPT" in out, out
    assert "arch=deep" in out, out


def test_the_kept_pretrain_leaves_every_artifact_byte_identical(run_dir,
                                                                monkeypatch):
    """Idempotent: a re-run of a completed task rewrites nothing."""
    payload = _pass_payload()
    d = _completed_pretrain(run_dir, "deep", payload)
    names = ("xnet.eqx", "cnet.eqx", "fidelity_certificate.json")
    before = {n: open(os.path.join(d, n), "rb").read() for n in names}
    _forbid(monkeypatch, "_run_pretrain", "_fidelity_certificate")

    assert pt.main([run_dir, "0"]) == 0
    after = {n: open(os.path.join(d, n), "rb").read() for n in names}
    assert after == before


@pytest.mark.parametrize("payload,networks,reason", [
    (None, ("xnet.eqx", "cnet.eqx"), "no certificate was written"),
    ("{not json", ("xnet.eqx", "cnet.eqx"), "the certificate is unreadable"),
    (_pass_payload(verdict=None), ("xnet.eqx", "cnet.eqx"),
     "the certificate records no verdict"),
    (_pass_payload(verdict="FAIL"), ("xnet.eqx", "cnet.eqx"),
     "the FAIL is enforced"),
    (_pass_payload(), ("xnet.eqx",), "cnet.eqx is missing"),
    (_pass_payload(), ("cnet.eqx",), "xnet.eqx is missing"),
    (_pass_payload(), (), "neither network is present"),
])
def test_an_incomplete_pretrain_is_redone(run_dir, monkeypatch, payload,
                                          networks, reason):
    """Anything the on-node gate would refuse is pretrained again.

    The certificate seam stays at the autouse PASS stub, so the redo lands the
    same exit 0 a first run does -- what is measured is that ``_run_pretrain``
    ran, i.e. nothing was kept on the strength of a partial or unverifiable
    record.
    """
    _completed_pretrain(run_dir, "deep", payload, networks=networks)
    calls = _recorder(monkeypatch)

    assert pt.main([run_dir, "0"]) == 0, reason
    assert len(calls) == 1, reason


def test_a_waiver_with_no_reason_does_not_keep_a_failing_pretrain(run_dir,
                                                                  monkeypatch):
    """``enforced: false`` alone is not a waiver -- the reason is required.

    The same rule ``fidelity.gate_certificate_from_read`` imposes everywhere,
    applied here so a hand-edited certificate on a compute node cannot keep an
    architecture no gate would release.
    """
    payload = _pass_payload(
        verdict="FAIL", enforced=False,
        tolerances={"tol_AE": 1.0, "tol_atom": 1.0, "override_reason": "  "})
    _completed_pretrain(run_dir, "deep", payload)
    calls = _recorder(monkeypatch)

    assert pt.main([run_dir, "0"]) == 0
    assert len(calls) == 1


def test_the_keep_check_runs_before_any_pretraining_work(run_dir, monkeypatch,
                                                         capsys):
    """The point of the check is the cost it avoids, so it precedes the work.

    ``_run_pretrain`` is the expensive seam and it is bound to a refusal here;
    a check placed after it would reach that refusal.
    """
    _completed_pretrain(run_dir, "deep", _pass_payload())
    _forbid(monkeypatch, "_run_pretrain", "_fidelity_certificate")

    assert pt.main([run_dir, "0"]) == 0
    out = capsys.readouterr().out
    assert "running run_pretrain" not in out, out


# ---------------------------------------------------------------------------
# the certificate must describe THESE networks at THIS run's identity
# ---------------------------------------------------------------------------
# A released verdict says the architecture reproduced its parent -- it does not
# say WHICH networks were measured, at which basis and grid, or against which
# parent. Those three facts are recorded in the certificate and are refused by
# ``validate_run`` (parent, identity over the union of both key sets, and the
# two SHA-256 digests), i.e. AFTER the whole train and eval graph has been
# spent. ``resubmit-preflight`` reloads and re-validates ``resolved_config.yaml``
# precisely because it can be edited between submissions, and none of its
# refusals covers an edited basis, grid level, density-fitting backend,
# auxiliary basis or orientation lock; a redo interrupted between the networks
# being written and the certificate being recomputed reaches the digest half
# with no edit at all. The keep check therefore compares the same three facts
# the later stage does, through the same helpers, and pretrains again on a
# disagreement.


def _real_certificate(run_dir, arch="deep"):
    """Real networks plus the certificate the certificate WRITER produces.

    The document is the certificate module's own rather than a hand-built
    dict, so the identity, the parent and the two digests are exactly the
    fields the later stages compare, and a test that perturbs one perturbs the
    real thing. The oracle set and the per-system evaluation are stubbed --
    what is under test is the bookkeeping beside the verdict, not the verdict
    -- which keeps the fixture at a fraction of a second rather than forty
    reference SCFs.
    """
    import equinox as eqx

    from xcquinox.alec.cluster import fidelity
    from xcquinox.alec.cluster.grid_config import (load_grid_config,
                                                   pretrain_checkpoint_dir)
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.networks import create_network_pair

    cfg = load_grid_config(os.path.join(run_dir, "resolved_config.yaml"))
    xnet, cnet = create_network_pair(get_architecture(arch), seed=cfg.pretrain.seed)
    d = pretrain_checkpoint_dir(run_dir, arch)
    os.makedirs(d, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(d, "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(os.path.join(d, "cnet.eqx"), cnet)

    def _evaluate(model, descriptors, mol_spec, *, parent, auxbasis=None,
                  orientation_lock_strength=0.0):
        return {"name": mol_spec.name, "spin": int(mol_spec.spin),
                "charge": int(mol_spec.charge),
                "is_atom": fidelity.is_atom_system(mol_spec), "n_grid": 10,
                "reference_xc": parent, "E_xc_nn": -1.0 + 0.5 / 1000.0,
                "E_xc_parent": -1.0, "E_xc_parent_numint": -1.0,
                "E_xc_parent_record": -1.0, "parent_grid_diff_Ha": 0.0,
                "parent_record_diff_Ha": 0.0, "dE_xc_mHa": 0.5,
                "duration_s": 0.0}

    basis = cfg.inputs.basis
    level = int(cfg.inputs.grid_level)
    oracle = (
        MoleculeSpec(name="atom_H", atom="H 0.0 0.0 0.0", basis=basis, spin=1,
                     atom_composition=(("H", 1),), grid_level=level),
        MoleculeSpec(name="H2", atom="H 0 0 0.371395; H 0 0 -0.371395",
                     basis=basis, spin=0, atom_composition=(("H", 2),),
                     grid_level=level),
    )
    payload = fidelity.fidelity_certificate(
        cfg, run_dir, arch, oracle_set=oracle, evaluate=_evaluate)
    assert payload["verdict"] == "PASS", payload["summary"]
    return cfg, payload


def _rewrite_certificate(run_dir, payload, arch="deep"):
    """Put an edited certificate back through the module's own writer."""
    from xcquinox.alec.cluster import fidelity

    fidelity._write_certificate_payload(
        payload, fidelity.certificate_path(run_dir, arch))
    return payload


def _edit_basis(payload):
    payload["identity"]["basis"] = "sto-3g"


def _edit_lock(payload):
    payload["identity"]["orientation_lock_strength"] = 0.0


def _edit_parent(payload):
    payload["parent"] = "scan"


def _edit_xnet_digest(payload):
    payload["checkpoint"]["xnet_sha256"] = "0" * 64


def _edit_cnet_digest(payload):
    payload["checkpoint"]["cnet_sha256"] = "0" * 64


@pytest.mark.parametrize("edit,named", [
    (_edit_basis, "basis"),
    (_edit_lock, "orientation_lock_strength"),
    (_edit_parent, "parent"),
    (_edit_xnet_digest, "xnet.eqx"),
    (_edit_cnet_digest, "cnet.eqx"),
])
def test_a_certificate_that_describes_another_run_is_not_kept(
        run_dir, monkeypatch, capsys, edit, named):
    """One perturbed fact per case, each of them one ``validate_run`` refuses.

    The verdict still reads PASS in every case, so nothing but the fact under
    test moves: a keep check reading the verdict alone keeps all five.
    """
    _cfg, payload = _real_certificate(run_dir)
    edit(payload)
    _rewrite_certificate(run_dir, payload)
    calls = _recorder(monkeypatch)

    assert pt.main([run_dir, "0"]) == 0
    assert len(calls) == 1, named
    out = capsys.readouterr().out
    assert "pretraining from scratch" in out, out
    assert named in out, out


def test_a_certificate_written_for_these_networks_is_kept(run_dir, monkeypatch,
                                                          capsys):
    """The unperturbed writer output at this run's identity is still kept.

    The discriminator for the five cases above: the comparison refuses a
    disagreement rather than everything.
    """
    _real_certificate(run_dir)
    _forbid(monkeypatch, "_run_pretrain", "_fidelity_certificate")

    assert pt.main([run_dir, "0"]) == 0
    out = capsys.readouterr().out
    assert "KEPT" in out, out


def test_the_keep_reason_names_every_fact_that_disagrees(run_dir):
    """The reason is the log line an operator reads, so it names the facts.

    All three kinds at once -- identity field, parent, digest -- since a
    check that stopped at the first would leave the rest to be discovered one
    resubmission at a time.
    """
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir

    cfg, payload = _real_certificate(run_dir)
    _edit_basis(payload)
    _edit_parent(payload)
    _edit_xnet_digest(payload)
    _rewrite_certificate(run_dir, payload)

    keep, reason = pt.completed_pretraining(
        pretrain_checkpoint_dir(run_dir, "deep"), cfg, "deep")
    assert keep is False
    assert "basis" in reason
    assert "scan" in reason
    assert "xnet.eqx" in reason


def test_the_keep_check_and_the_run_record_apply_one_comparison():
    """The keep check calls the record layer's own comparison helpers.

    Two implementations of "does this certificate describe what is on disk"
    would drift, and the cost of the drift is a train and eval graph: the
    pretrain stage keeps an architecture the record layer later refuses. The
    helpers live in the certificate module and both stages import them from
    there.
    """
    import inspect

    from xcquinox.alec.cluster import fidelity, validate_run

    source = inspect.getsource(pt.completed_pretraining)
    assert "certificate_describes_run" in source
    for name in ("identity_mismatches", "parent_mismatch",
                 "checkpoint_digest_findings"):
        assert hasattr(fidelity, name), name
        assert getattr(validate_run, name, None) is getattr(fidelity, name), \
            name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
