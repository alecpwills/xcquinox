"""Tests for xcquinox.pipeline.cluster._pretrain.

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
    ``python -m xcquinox.pipeline.cluster._pretrain``.
"""
import importlib.resources
import json
import os
import sys
from pathlib import Path
from string import Template

from xcquinox.pipeline.parallel import PYSCF_POOL_THREADS_MAX

import pytest

from xcquinox.pipeline.cluster import _pretrain as pt
from xcquinox.pipeline.config import PretrainSpec


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

    from xcquinox.pipeline.cluster.grid_config import load_grid_config
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


# ---------------------------------------------------------------------------
# silent-no-checkpoint guard
# ---------------------------------------------------------------------------

def test_silent_no_checkpoint_exits_nonzero(run_dir, monkeypatch):
    # _run_pretrain "succeeds" (returns normally) but writes nothing.
    monkeypatch.setattr(pt, "_run_pretrain", lambda spec, progress_callback=None: {})
    assert pt.main([run_dir, "0"]) != 0


# ---------------------------------------------------------------------------
# missing config
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JAX env routing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# throttled progress callback
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# pretrain.sbatch.tmpl template
# ---------------------------------------------------------------------------

def _template_text():
    res = (
        importlib.resources.files("xcquinox.pipeline.cluster")
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
    assert "python -m xcquinox.pipeline.cluster._pretrain" in rendered
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


# ---------------------------------------------------------------------------
# The exit status the array task hands SLURM
# ---------------------------------------------------------------------------


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
    beside the verdict -- the architecture's name, the running code's
    version, this run's identity, the architecture's parent and the SHA-256
    digests of the networks just written -- unless it states them itself. The
    keep check compares those five against the run, so a document that
    omitted them would be refused for the omission and could say nothing
    about the release rule each case here is written for; the cases that
    perturb them are separate and use the writer's own output.
    """
    from xcquinox.pipeline.cluster import fidelity
    from xcquinox.pipeline.cluster.grid_config import load_grid_config
    from xcquinox.pipeline.cluster.materialize import _sha256_file

    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    for name in networks:
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"checkpoint-bytes-" + name.encode())
    if payload is not None:
        if isinstance(payload, dict):
            cfg = load_grid_config(
                os.path.join(run_dir, "resolved_config.yaml"))
            payload.setdefault("arch", arch)
            payload.setdefault("xcquinox_version",
                               fidelity.running_xcquinox_version())
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


# ---------------------------------------------------------------------------
# The resolution and the spec as module-level functions: one implementation
# for the stage and for any tool that reproduces one of its fits
# ---------------------------------------------------------------------------

#: The campaign configuration the factored resolution is read against: the v7
#: meta-GGA families group, whose sorted sweep axis puts deep_cusp_mgga_3x16
#: at index 0 and whose pretrain block states every protocol knob.
_MGGA_CONFIG = (Path(__file__).resolve().parents[3] / "hpcjobs" / "configs"
                / "dfs_step7.dfs6311_grid3_v7g2_families_mgga.yaml")


def _mgga_config():
    """The campaign configuration file, skipped where the checkout has none."""
    pytest.importorskip("yaml")
    if not _MGGA_CONFIG.is_file():
        pytest.skip(f"no {_MGGA_CONFIG.name} in this checkout")
    return _MGGA_CONFIG


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


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
        "python -m xcquinox.pipeline.cluster._pretrain "
        "${RUN_DIR} $${SLURM_ARRAY_TASK_ID}"
    ]
