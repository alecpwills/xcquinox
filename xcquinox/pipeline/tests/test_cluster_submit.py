"""Tests for xcquinox.pipeline.cluster.submit: sbatch rendering + job-graph submit.

These tests NEVER shell out to a real SLURM controller: ``job_tracking._run_slurm``
is monkeypatched with canned behavior. A grid config is built from an in-memory
dict via ``load_grid_config`` (JSON, so no PyYAML dependency), and ``run_dir`` is
a tmp directory.
"""
import json
import os
import shutil
import subprocess

import pytest

from xcquinox.pipeline.cluster import job_tracking as jt
from xcquinox.pipeline.cluster.grid_config import load_grid_config
from xcquinox.pipeline.cluster.submit import render_sbatch, submit_jobs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_config_dict(*, device="cpu", gpus_per_task=0):
    """A complete, valid raw config dict. ``subset_size`` has 10 values and the
    other axes 1/2/2/.../2 so the grid expands to a controllable size."""
    return {
        "sweep": {
            "arch": ["medium"],
            "loss": ["delta_ae", "delta_de"],
            "metric": ["l2", "jsd"],
            "subset_size": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
            "solver": ["fast"],
        },
        "solvers": {
            "fast": {"mode": "fixed_density", "max_cycles": 1},
        },
        "hyperparams": {
            "n_steps": 200,
            "lr_start": 1e-3,
            "lr_end": 1e-5,
            "lr_decay_start": 0.2,
            "grad_clip": 1.0,
            "gradnorm_alpha": 1.5,
            "vxc_weight": 1.0,
            "density_weight": 0.5,
        },
        "inputs": {
            "external_refs_dir": "/shared/refs",
            "subset_ledger_path": "/shared/ledger.json",
            "basis": "def2-tzvp",
            "grid_level": 3,
            "output_root": "/shared/runs",
        },
        "pretrain": {
            "data_dir": "/shared/pretrain_data",
        },
        "cluster": {
            "partition": "long-40core",
            "time": "12:00:00",
            "mem": "32G",
            "cpus_per_task": 4,
            "array_throttle": 4,
            "eval_array_throttle": 8,
            "max_concurrent_tasks": 40,
            "device": device,
            "gpus_per_task": gpus_per_task,
            "conda_profile": "/opt/conda/etc/profile.d/conda.sh",
            "conda_env": "xcq",
            "mail_user": "user@example.com",
            "mail_type": "END,FAIL",
            "account": "xcq-acct",
        },
        "domain_profile": "gmtkn55_subset",
    }


def _make_cfg(tmp_path, **kw):
    """Build a GridConfig from a dict by round-tripping through a JSON file."""
    p = tmp_path / "grid.json"
    p.write_text(json.dumps(_base_config_dict(**kw)))
    return load_grid_config(str(p))


# The base config grid: arch(1) x loss(2) x metric(2) x subset_size(10) x
# solver(1) = 40 cells -> array indices 0..39.
_EXPECTED_N = 40
_EXPECTED_ARRAY_MAX = 39

# Template placeholders render_sbatch fills via string.Template.substitute.
_PLACEHOLDER_TOKENS = (
    "JOB_NAME", "PARTITION", "TIME", "ALLOC_LINES", "MEM_LINE",
    "CPUS_PER_TASK", "ARRAY_MAX",
    "THROTTLE", "RUN_DIR", "CONDA_ACTIVATION", "MAIL_USER_LINE",
    "MAIL_TYPE_LINE", "ACCOUNT_LINE", "SIGTERM_GRACE", "GPUS_PER_TASK",
)


class _FakeProc:
    """Minimal stand-in for subprocess.CompletedProcess (just ``stdout``)."""
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def _fake_slurm_factory(ids=None, fail_on_index=None):
    """Build a fake ``_run_slurm``.

    ``ids``: the sequence of array-job ids returned for successive ``sbatch``
    calls. ``fail_on_index``: if set, the Nth (0-based) ``sbatch`` call raises
    CalledProcessError instead of returning. ``scancel`` calls always succeed.
    The list of every cmd seen is recorded on ``.calls``.
    """
    ids = list(ids or ["1001", "1002", "1003", "1004", "1005"])
    state = {"sbatch_n": 0}
    calls = []

    def _fake(cmd, *, retries=3):
        calls.append(list(cmd))
        verb = os.path.basename(cmd[0])
        if verb == "sbatch":
            i = state["sbatch_n"]
            state["sbatch_n"] += 1
            if fail_on_index is not None and i == fail_on_index:
                raise subprocess.CalledProcessError(1, cmd, stderr="rejected")
            return _FakeProc(stdout=ids[i] + "\n")
        if verb == "scancel":
            return _FakeProc(stdout="")
        raise AssertionError(f"unexpected SLURM verb in test: {verb}")

    _fake.calls = calls
    return _fake


# ---------------------------------------------------------------------------
# render_sbatch: CPU vs GPU template selection
# ---------------------------------------------------------------------------


def test_render_train_cpu_has_xla_flags_no_gres(tmp_path):
    cfg = _make_cfg(tmp_path, device="cpu")
    text = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    assert "xla_cpu_multi_thread_eigen=true" in text
    assert "--xla_force_host_platform_device_count=1" in text
    assert "--gres=gpu" not in text
    assert "CUDA_VISIBLE_DEVICES" not in text
    assert "#SBATCH --signal=B:TERM@" in text


# ---------------------------------------------------------------------------
# Per-stage node-allocation mode (exclusive whole-node vs shared cpu/mem slice)
# ---------------------------------------------------------------------------


def test_render_thread_caps_present_every_template(tmp_path):
    """The JAX-dominated array stages hand both pools the allocation; the
    PySCF-heavy front stages (datagen, pretrain's certificate, preflight's
    reference build, the held-out reference build) cap each pool at
    ``parallel.PYSCF_POOL_THREADS_MAX`` -- at the allocation itself the two
    spin-waiting pools stall a reference build (job 2134488: about ten minutes
    per def2-svp molecule at 40 threads against 8 s at four)."""
    from xcquinox.pipeline.parallel import PYSCF_POOL_THREADS_MAX
    cfg = _make_cfg(tmp_path)
    # The train array is XLA's; its PySCF inputs come precomputed from the
    # preflight, so its pools keep the allocation.
    for kind, kw in (("train", {"array_max": 39}),):
        text = render_sbatch(kind, cfg, str(tmp_path / "run"), **kw)
        assert "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" in text, kind
        assert "export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK" in text, kind
        assert "export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" in text, kind
        assert "PYSCF_THREADS" not in text, kind
    d = _base_config_dict()
    d["inputs"]["benchmark_refs_dir"] = "/shared/bench_refs"
    p = tmp_path / "grid_bench.json"
    p.write_text(json.dumps(d))
    cfg_bench = load_grid_config(str(p))
    # The evaluation array runs PySCF in its own process on its final serial
    # tier and on the whole-serial fallback, so it is capped with the front
    # stages; its shard workers set their own pools.
    for kind, kw, c in (("pretrain", {"array_max": 0}, cfg),
                        ("preflight", {}, cfg), ("datagen", {}, cfg),
                        ("eval", {"array_max": 39}, cfg),
                        ("benchmark_refs", {}, cfg_bench)):
        text = render_sbatch(kind, c, str(tmp_path / "run"), **kw)
        cap = PYSCF_POOL_THREADS_MAX
        assert f"PYSCF_THREADS=${{SLURM_CPUS_PER_TASK:-{cap}}}" in text, kind
        assert (f'[ "$PYSCF_THREADS" -le {cap} ] || PYSCF_THREADS={cap}'
                in text), kind
        assert '[ "$PYSCF_THREADS" -ge 1 ] || PYSCF_THREADS=1' in text, kind
        assert "PYSCF_THREADS=$(( 10#$PYSCF_THREADS ))" in text, kind
        for pool in ("OMP", "MKL", "OPENBLAS"):
            assert f'export {pool}_NUM_THREADS="$PYSCF_THREADS"' in text, (
                kind, pool)
            assert f"export {pool}_NUM_THREADS=$SLURM_CPUS_PER_TASK" not in text
        assert "$$" not in text, kind


def test_the_benchmark_refs_job_renders_the_pools_and_the_size_cap(tmp_path):
    """The reference job covers the species of the pools the run evaluates, capped where
    the run states a cap. A job that always rendered ``--pool all`` would leave a wider
    run's species without references, and the evaluation reports those species' density
    leg as absent rather than as missing.

    Oracle: the rendered job script.
    """
    d = _base_config_dict()
    d["inputs"]["benchmark_refs_dir"] = "/shared/bench_refs"
    p = tmp_path / "grid_bench_default.json"
    p.write_text(json.dumps(d))
    text = render_sbatch("benchmark_refs", load_grid_config(str(p)),
                         str(tmp_path / "run"))
    assert "--pool bh76,w411" in text
    assert "--max-atoms" not in text

    d["inputs"]["held_out_pools"] = ["bh76", "w411", "diet150"]
    d["inputs"]["benchmark_refs_max_atoms"] = 8
    p2 = tmp_path / "grid_bench_wide.json"
    p2.write_text(json.dumps(d))
    text2 = render_sbatch("benchmark_refs", load_grid_config(str(p2)),
                          str(tmp_path / "run"))
    assert "--pool bh76,w411,diet150" in text2
    assert "--max-atoms 8" in text2
    assert "$$" not in text2

    # the cap survives density fitting, which every production run states
    d["inputs"]["density_fit"] = True
    p3 = tmp_path / "grid_bench_wide_df.json"
    p3.write_text(json.dumps(d))
    text3 = render_sbatch("benchmark_refs", load_grid_config(str(p3)),
                          str(tmp_path / "run"))
    assert "--density-fit" in text3
    assert "--max-atoms 8" in text3




def test_render_optional_directives_emitted_and_omitted(tmp_path):
    cfg = _make_cfg(tmp_path)
    text = render_sbatch("preflight", cfg, str(tmp_path / "run"))
    assert "#SBATCH --mail-user=user@example.com" in text
    assert "#SBATCH --account=xcq-acct" in text

    # With blank account/mail, no dangling #SBATCH directive should appear.
    d = _base_config_dict()
    d["cluster"]["account"] = ""
    d["cluster"]["mail_user"] = ""
    d["cluster"]["mail_type"] = ""
    p = tmp_path / "g2.json"
    p.write_text(json.dumps(d))
    cfg2 = load_grid_config(str(p))
    text2 = render_sbatch("preflight", cfg2, str(tmp_path / "run"))
    assert "--account=" not in text2
    assert "--mail-user=" not in text2


def _has_bare_source_line(text):
    """True iff the script has a ``source`` token with no argument after it."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "source" or stripped.startswith("source "):
            arg = stripped[len("source"):].strip()
            if not arg:
                return True
    return False


def test_render_conda_block_with_profile_sources_then_activates(tmp_path):
    # _base_config_dict sets conda_profile=/opt/conda/etc/profile.d/conda.sh.
    for kind, kw in (("preflight", {}), ("train", {"array_max": 39}),
                     ("eval", {"array_max": 39})):
        text = render_sbatch(kind, _make_cfg(tmp_path), str(tmp_path / "run"),
                             **kw)
        assert "source /opt/conda/etc/profile.d/conda.sh" in text, kind
        assert "conda activate xcq" in text, kind
        assert not _has_bare_source_line(text), kind
        # source must come before conda activate.
        assert (text.index("source /opt/conda/etc/profile.d/conda.sh")
                < text.index("conda activate xcq")), kind
        # ~/.local user-site isolation, exported AFTER activation (env parity).
        assert "export PYTHONNOUSERSITE=1" in text, kind
        assert (text.index("conda activate xcq")
                < text.index("export PYTHONNOUSERSITE=1")), kind


# ---------------------------------------------------------------------------
# submit_jobs: dry run
# ---------------------------------------------------------------------------

def test_dry_run_calls_no_sbatch_and_writes_no_jobs_json(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    fake = _fake_slurm_factory()
    monkeypatch.setattr(jt, "_run_slurm", fake)

    result = submit_jobs(cfg, run_dir, submit=False)

    assert result["dry_run"] is True
    assert result["n_specs"] == _EXPECTED_N
    assert result["array_max"] == _EXPECTED_ARRAY_MAX
    # The base config has a single distinct arch -> pretrain array 0-0.
    assert result["n_archs"] == 1
    assert result["pretrain_array_max"] == 0
    # No SLURM call whatsoever in a dry run.
    assert fake.calls == []
    # No jobs.json written.
    assert not os.path.exists(os.path.join(run_dir, "jobs.json"))
    # Scripts + the submit-commands record ARE written.
    for name in ("pretrain.sbatch", "preflight.sbatch", "train_array.sbatch",
                 "eval_array.sbatch"):
        assert os.path.exists(os.path.join(run_dir, "scripts", name))
    cmds_path = os.path.join(run_dir, "submit_commands.txt")
    assert os.path.exists(cmds_path)
    cmds_text = open(cmds_path).read()
    assert "[dry-run]" in cmds_text
    # The pretrain sbatch invocation is listed in the submit-commands record.
    assert "scripts/pretrain.sbatch" in cmds_text
    assert os.path.isdir(os.path.join(run_dir, "logs"))


# ---------------------------------------------------------------------------
# submit_jobs: real submission
# ---------------------------------------------------------------------------

def test_real_submit_dependency_directives(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    fake = _fake_slurm_factory(ids=["5000", "5001", "5002", "5003", "5004"])
    monkeypatch.setattr(jt, "_run_slurm", fake)

    result = submit_jobs(cfg, run_dir, submit=True)

    assert result["dry_run"] is False
    assert result["job_ids"] == {
        "datagen": "5000", "pretrain": "5001", "preflight": "5002",
        "train": "5003", "eval": "5004",
    }
    sbatch_calls = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch_calls) == 5
    joined = [" ".join(c) for c in sbatch_calls]
    # datagen: FIRST, no dependency.
    assert "--dependency" not in joined[0]
    assert joined[0].endswith("datagen.sbatch")
    # pretrain: afterok on the datagen id.
    assert "--dependency=afterok:5000" in joined[1]
    # preflight: afterok on the pretrain id.
    assert "--dependency=afterok:5001" in joined[2]
    # train: afterok on BOTH the pretrain and the preflight ids.
    assert "--dependency=afterok:5001:5002" in joined[3]
    # eval: aftercorr on the train array id.
    assert "--dependency=aftercorr:5003" in joined[4]

    # jobs.json now records all five stages.
    records = jt.read_job_records(run_dir)
    kinds = sorted(r["kind"] for r in records)
    assert kinds == ["datagen", "eval", "preflight", "pretrain", "train"]
    cmds = open(os.path.join(run_dir, "submit_commands.txt")).read()
    assert "[submit]" in cmds
    # Default (defer_eval off): no launcher script, no deferral flag.
    assert result.get("defer_eval") is False
    assert not os.path.exists(
        os.path.join(run_dir, "scripts", "eval_launcher.sbatch"))


@pytest.mark.parametrize(
    "fail_idx,expected_scancels",
    [
        (0, []),                                  # datagen rejected, nothing prior
        (1, ["9000"]),                            # pretrain rejected, cancel datagen
        (2, ["9000", "9001"]),                    # preflight rejected, cancel prior 2
        (3, ["9000", "9001", "9002"]),            # train rejected, cancel prior 3
        (4, ["9000", "9001", "9002", "9003"]),    # eval rejected, cancel all four
    ],
)
def test_rollback_scancels_on_midgraph_failure(tmp_path, monkeypatch,
                                               fail_idx, expected_scancels):
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    fake = _fake_slurm_factory(ids=["9000", "9001", "9002", "9003", "9004"],
                               fail_on_index=fail_idx)
    monkeypatch.setattr(jt, "_run_slurm", fake)

    with pytest.raises(RuntimeError, match="rolled back"):
        submit_jobs(cfg, run_dir, submit=True)

    scancels = [c for c in fake.calls if os.path.basename(c[0]) == "scancel"]
    assert scancels == [["scancel", j] for j in expected_scancels]
    # No partial records written.
    assert not os.path.exists(os.path.join(run_dir, "jobs.json"))


# ---------------------------------------------------------------------------
# shellcheck: lint the rendered scripts if shellcheck is available
# ---------------------------------------------------------------------------

def test_rendered_scripts_pass_shellcheck(tmp_path, monkeypatch):
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not on PATH")

    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm_factory())
    # defer_eval=True so the eval_launcher script is also rendered + linted.
    submit_jobs(cfg, run_dir, submit=False, defer_eval=True)

    for name in ("pretrain.sbatch", "preflight.sbatch", "train_array.sbatch",
                 "eval_array.sbatch", "eval_launcher.sbatch"):
        path = os.path.join(run_dir, "scripts", name)
        proc = subprocess.run(
            ["shellcheck", "--severity=warning", path],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"shellcheck flagged {name}:\n{proc.stdout}\n{proc.stderr}"
        )


# ---------------------------------------------------------------------------
# Train worker is exec'd so it receives the SLURM B:TERM grace signal
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# hold-out benchmark refs job (inputs.benchmark_refs_dir)
# ---------------------------------------------------------------------------


# --------------------------------------------------------------------------- #
# Seed-cache env wiring (fallback transport for retro/diagnostic runs)
# --------------------------------------------------------------------------- #


