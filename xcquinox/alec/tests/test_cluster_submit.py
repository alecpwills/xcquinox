"""Tests for xcquinox.alec.cluster.submit: sbatch rendering + job-graph submit.

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

from xcquinox.alec.cluster import job_tracking as jt
from xcquinox.alec.cluster.grid_config import load_grid_config
from xcquinox.alec.cluster.submit import render_sbatch, submit_jobs


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
            "mail_user": "alec@example.com",
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


def _assert_no_unrendered_placeholders(text):
    """Fail if any ${PLACEHOLDER} template token survived substitution.

    A legitimate rendered bash variable such as ``${SLURM_ARRAY_TASK_ID}`` is
    NOT a placeholder, only the tokens in _PLACEHOLDER_TOKENS are.
    """
    for tok in _PLACEHOLDER_TOKENS:
        assert ("${" + tok + "}") not in text, f"unrendered placeholder ${tok}"


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

def test_render_benchmark_refs_basis_is_shell_safe(tmp_path):
    """A basis containing shell metacharacters -- 6-311++G(3df,2pd) -- must render
    into a benchmark_refs sbatch script that is VALID bash. The unquoted
    ``--basis ${BASIS}`` broke here under ``set -euo pipefail`` (bash parsed the
    ``(3df,2pd)`` as a subshell -> syntax error before Python ran)."""
    d = _base_config_dict()
    d["inputs"]["basis"] = "6-311++G(3df,2pd)"
    d["inputs"]["benchmark_refs_dir"] = "/shared/bench_refs"
    p = tmp_path / "grid.json"
    p.write_text(json.dumps(d))
    cfg = load_grid_config(str(p))

    text = render_sbatch("benchmark_refs", cfg, str(tmp_path / "run"))
    script = tmp_path / "bench.sbatch"
    script.write_text(text)
    # bash -n = syntax check only (never executes); catches the unquoted-paren bug.
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, (
        "rendered benchmark_refs script is not valid bash:\n"
        f"{r.stderr}\n--- script ---\n{text}"
    )
    assert "6-311++G(3df,2pd)" in text


def test_render_benchmark_refs_passes_orientation_lock(tmp_path):
    """When inputs.orientation_lock_strength is set, the benchmark_refs command
    must carry --orientation-lock-strength (so held-out refs lock the same
    component as training) and stay valid bash."""
    d = _base_config_dict()
    d["inputs"]["basis"] = "6-311++G(3df,2pd)"
    d["inputs"]["benchmark_refs_dir"] = "/shared/bench_refs"
    d["inputs"]["orientation_lock_strength"] = 3e-5
    d["inputs"]["density_fit"] = True
    p = tmp_path / "grid.json"
    p.write_text(json.dumps(d))
    cfg = load_grid_config(str(p))

    text = render_sbatch("benchmark_refs", cfg, str(tmp_path / "run"))
    assert "--orientation-lock-strength 3e-05" in text
    assert "--density-fit" in text
    script = tmp_path / "bench.sbatch"
    script.write_text(text)
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    # off by default -> no flag
    d2 = _base_config_dict()
    d2["inputs"]["benchmark_refs_dir"] = "/shared/bench_refs"
    p2 = tmp_path / "grid2.json"
    p2.write_text(json.dumps(d2))
    text2 = render_sbatch("benchmark_refs", load_grid_config(str(p2)),
                          str(tmp_path / "run2"))
    assert "--orientation-lock-strength" not in text2


def test_render_train_cpu_has_xla_flags_no_gres(tmp_path):
    cfg = _make_cfg(tmp_path, device="cpu")
    text = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    assert "xla_cpu_multi_thread_eigen=true" in text
    assert "--xla_force_host_platform_device_count=1" in text
    assert "--gres=gpu" not in text
    assert "CUDA_VISIBLE_DEVICES" not in text
    assert "#SBATCH --signal=B:TERM@" in text


def test_render_inline_forwards_sigterm_to_worker(tmp_path):
    """Inline-eval mode must forward a wall-clock SIGTERM to the train worker
    (the batch shell survives to run eval, so B:TERM hits the shell, not the
    worker). Regression: the old `( exec python )` subshell never received the
    signal, so the grace handler's timeout failure.json was never written."""
    cfg = _make_cfg(tmp_path, device="cpu")
    text = render_sbatch("train_eval_inline", cfg, str(tmp_path / "run"),
                         array_max=39)
    # Worker runs backgrounded with its PID captured, and a TERM trap forwards
    # the signal to it.
    assert "_train_task" in text
    assert "train_pid=$!" in text
    assert "trap " in text and "kill -TERM ${train_pid}" in text
    assert "wait ${train_pid}" in text
    # The old subshell-exec form (which swallowed the signal) is gone.
    assert "exec python -m xcquinox.alec.cluster._train_task" not in text
    assert "#SBATCH --signal=B:TERM@" in text


def test_render_train_gpu_has_gres_no_xla_cpu(tmp_path):
    cfg = _make_cfg(tmp_path, device="gpu", gpus_per_task=2)
    text = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    assert "#SBATCH --gres=gpu:2" in text
    assert "xla_cpu_" not in text
    # SLURM's --gres binding sets CUDA_VISIBLE_DEVICES; the script must not.
    assert "CUDA_VISIBLE_DEVICES" not in text
    assert "#SBATCH --signal=B:TERM@" in text


def test_render_train_array_range_and_throttle(tmp_path):
    cfg = _make_cfg(tmp_path)  # array_throttle=4
    text = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    assert "#SBATCH --array=0-39%4" in text


def test_render_eval_is_cpu_only_with_eval_throttle(tmp_path):
    cfg = _make_cfg(tmp_path, device="gpu", gpus_per_task=2)
    text = render_sbatch("eval", cfg, str(tmp_path / "run"), array_max=39)
    # eval is always CPU even when the train device is gpu.
    assert "--gres=gpu" not in text
    assert "#SBATCH --array=0-39%8" in text  # eval_array_throttle=8
    assert "_eval_one_spec" in text


def test_render_pretrain_array_max_and_throttle(tmp_path):
    """pretrain: ARRAY_MAX = A-1; throttle defaults to the arch count (A);
    no leftover ${...} placeholder."""
    cfg = _make_cfg(tmp_path)  # 1 distinct arch -> array_max 0
    text = render_sbatch("pretrain", cfg, str(tmp_path / "run"), array_max=0)
    # 1 arch -> 0-0; default throttle = array_max + 1 = 1.
    assert "#SBATCH --array=0-0%1" in text
    assert "xcquinox.alec.cluster._pretrain" in text
    assert "pretrain_%A_%a.out" in text
    # No unsubstituted template placeholder survived (the rendered bash var
    # ${SLURM_ARRAY_TASK_ID} is legitimate, placeholders are the ones in
    # _PLACEHOLDER_TOKENS below).
    _assert_no_unrendered_placeholders(text)


def test_render_pretrain_explicit_throttle_and_multi_arch(tmp_path):
    """A multi-arch grid + explicit pretrain_throttle render the right range."""
    d = _base_config_dict()
    d["sweep"]["arch"] = ["medium", "shallow", "deep"]  # 3 distinct archs
    d["cluster"]["pretrain_throttle"] = 2
    p = tmp_path / "g_multi.json"
    p.write_text(json.dumps(d))
    cfg = load_grid_config(str(p))
    text = render_sbatch("pretrain", cfg, str(tmp_path / "run"), array_max=2)
    assert "#SBATCH --array=0-2%2" in text
    _assert_no_unrendered_placeholders(text)


def test_render_pretrain_resource_fallback(tmp_path):
    """Unset pretrain_* resource knobs fall back to the train-array values.

    Uses shared allocation so the inherited ``--mem`` is actually rendered
    (whole-node/exclusive stages emit ``--mem=0`` instead)."""
    d = _base_config_dict()
    d["cluster"]["pretrain_allocation"] = "shared"
    p = tmp_path / "g_shared.json"
    p.write_text(json.dumps(d))
    cfg = load_grid_config(str(p))  # no pretrain_* resource knobs set
    text = render_sbatch("pretrain", cfg, str(tmp_path / "run"), array_max=0)
    # train-array partition/time/mem are inherited.
    assert "#SBATCH --partition=long-40core" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "#SBATCH --mem=32G" in text


# ---------------------------------------------------------------------------
# Per-stage node-allocation mode (exclusive whole-node vs shared cpu/mem slice)
# ---------------------------------------------------------------------------

def test_render_exclusive_emits_node_lines_and_full_mem(tmp_path):
    """The default (exclusive) allocation books a whole node per task:
    ``--nodes=1 --exclusive`` plus ``--mem=0`` so the task claims all node RAM
    (an exclusive job that omits --mem is cgroup-capped at DefMemPerCPU*cpus)."""
    cfg = _make_cfg(tmp_path)  # all stages default to exclusive
    text = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --mem=0" in text


def test_render_shared_emits_mem_and_omits_node_lines(tmp_path):
    """A stage set to 'shared' requests a cpu/mem slice: ``--mem`` is emitted
    and no whole-node directives appear."""
    d = _base_config_dict()
    d["cluster"]["train_allocation"] = "shared"
    p = tmp_path / "g.json"
    p.write_text(json.dumps(d))
    cfg = load_grid_config(str(p))
    text = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    assert "#SBATCH --mem=32G" in text
    assert "#SBATCH --nodes=1" not in text
    assert "#SBATCH --exclusive" not in text


def test_render_shared_omits_mem_when_unset(tmp_path):
    """A shared stage with mem unset emits NO ``--mem`` line (SLURM applies the
    partition default-mem-per-cpu) and still no whole-node directives."""
    d = _base_config_dict()
    d["cluster"]["train_allocation"] = "shared"
    d["cluster"].pop("mem", None)
    p = tmp_path / "g.json"
    p.write_text(json.dumps(d))
    cfg = load_grid_config(str(p))
    text = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    assert "#SBATCH --mem=" not in text
    assert "#SBATCH --nodes=1" not in text
    assert "#SBATCH --exclusive" not in text


def test_render_per_stage_allocation_independent(tmp_path):
    """Each stage's allocation is independent: train whole-node, eval sliced."""
    d = _base_config_dict()
    d["cluster"]["train_allocation"] = "exclusive"
    d["cluster"]["eval_allocation"] = "shared"
    p = tmp_path / "g.json"
    p.write_text(json.dumps(d))
    cfg = load_grid_config(str(p))
    train = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    ev = render_sbatch("eval", cfg, str(tmp_path / "run"), array_max=39)
    assert "#SBATCH --exclusive" in train and "#SBATCH --mem=0" in train
    assert "#SBATCH --mem=32G" in ev and "#SBATCH --exclusive" not in ev


def test_render_thread_caps_present_every_template(tmp_path):
    cfg = _make_cfg(tmp_path)
    for kind, kw in (("pretrain", {"array_max": 0}), ("preflight", {}),
                     ("train", {"array_max": 39}),
                     ("eval", {"array_max": 39})):
        text = render_sbatch(kind, cfg, str(tmp_path / "run"), **kw)
        assert "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" in text, kind
        assert "export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK" in text, kind
        assert "export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" in text, kind


def test_render_inline_thread_cap_scales_from_node_cores(tmp_path):
    """The inline train template caps the idle BLAS/OMP pools from the whole-node
    core count (SLURM_CPUS_ON_NODE), not --cpus-per-task. Regression: sourcing the
    cap from SLURM_CPUS_PER_TASK (=24) mis-scaled it to 24/12=2 on a 96-core node
    instead of the intended 8."""
    cfg = _make_cfg(tmp_path, device="cpu")
    text = render_sbatch("train_eval_inline", cfg, str(tmp_path / "run"),
                         array_max=39)
    assert 'CORES="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"' in text
    assert "BLAS_THREADS=$(( CORES / 12 ))" in text
    assert 'export OMP_NUM_THREADS="$BLAS_THREADS"' in text
    # The old, mis-scaling source (--cpus-per-task) is gone from the CORES line.
    assert 'CORES="${SLURM_CPUS_PER_TASK' not in text


def test_render_preflight_has_no_array_directive(tmp_path):
    cfg = _make_cfg(tmp_path)
    text = render_sbatch("preflight", cfg, str(tmp_path / "run"))
    assert "#SBATCH --array" not in text
    assert "_preflight" in text
    assert "preflight_%j.out" in text


def test_render_preflight_raises_nproc_ceiling(tmp_path):
    """Preflight renders `ulimit -u unlimited` (mirrors the inline train template)
    so the compile-smoke probe's LLVM codegen has pthread_create headroom. Without
    it, the probe at the preflight 24-thread env hit `pthread_create failed` on the
    heaviest attention cell and false-blocked the 030651Z train array."""
    cfg = _make_cfg(tmp_path)
    text = render_sbatch("preflight", cfg, str(tmp_path / "run"))
    assert "ulimit -u unlimited 2>/dev/null || true" in text


def test_render_optional_directives_emitted_and_omitted(tmp_path):
    cfg = _make_cfg(tmp_path)
    text = render_sbatch("preflight", cfg, str(tmp_path / "run"))
    assert "#SBATCH --mail-user=alec@example.com" in text
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


def test_render_conda_block_empty_profile_no_bare_source(tmp_path):
    # An empty conda_profile must NEVER emit a bare ``source`` line, under
    # ``set -euo pipefail`` that is broken bash.
    d = _base_config_dict()
    d["cluster"]["conda_profile"] = ""
    p = tmp_path / "g_noprofile.json"
    p.write_text(json.dumps(d))
    cfg = load_grid_config(str(p))

    for kind, kw in (("preflight", {}), ("train", {"array_max": 39}),
                     ("eval", {"array_max": 39})):
        text = render_sbatch(kind, cfg, str(tmp_path / "run"), **kw)
        assert "conda activate xcq" in text, kind
        assert not _has_bare_source_line(text), kind
        assert "set -euo pipefail" in text, kind
        assert "export PYTHONNOUSERSITE=1" in text, kind


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


def test_dry_run_train_eval_array_ranges_identical(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm_factory())
    submit_jobs(cfg, run_dir, submit=False)

    train = open(os.path.join(run_dir, "scripts", "train_array.sbatch")).read()
    ev = open(os.path.join(run_dir, "scripts", "eval_array.sbatch")).read()

    def _range(txt):
        for line in txt.splitlines():
            if line.strip().startswith("#SBATCH --array="):
                return line.split("=", 1)[1].split("%", 1)[0]
        raise AssertionError("no --array directive")

    assert _range(train) == _range(ev) == "0-39"


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


def test_deferred_submit_launches_instead_of_eval_array(tmp_path, monkeypatch):
    """defer_eval=True: the post-train sbatch is the launcher (afterany), the
    eval array is NOT submitted, and only pretrain/preflight/train are recorded."""
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    fake = _fake_slurm_factory(ids=["5000", "5001", "5002", "5003", "5004"])
    monkeypatch.setattr(jt, "_run_slurm", fake)

    result = submit_jobs(cfg, run_dir, submit=True, defer_eval=True)

    assert result["defer_eval"] is True
    sbatch_calls = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch_calls) == 5
    joined = [" ".join(c) for c in sbatch_calls]
    # 5th sbatch is the launcher: afterany on the train id, launcher script.
    assert "--dependency=afterany:5003" in joined[4]
    assert joined[4].endswith("eval_launcher.sbatch")
    # The eval array itself was NOT submitted here.
    assert not any("eval_array.sbatch" in j for j in joined)
    # job_ids reports the launcher, not an eval id; manual fallback is surfaced.
    assert result["job_ids"] == {
        "datagen": "5000", "pretrain": "5001", "preflight": "5002",
        "train": "5003", "eval_launcher": "5004",
    }
    assert result["manual_eval_command"].startswith(
        "python -m xcquinox.alec.cluster submit-eval ")
    # Only datagen/pretrain/preflight/train recorded, eval is written later by
    # the launcher/manual step.
    kinds = sorted(r["kind"] for r in jt.read_job_records(run_dir))
    assert kinds == ["datagen", "preflight", "pretrain", "train"]
    # The launcher script was written and reuses the (also-written) eval script.
    assert os.path.exists(os.path.join(run_dir, "scripts", "eval_launcher.sbatch"))
    assert os.path.exists(os.path.join(run_dir, "scripts", "eval_array.sbatch"))


def test_deferred_dry_run_shows_launcher(tmp_path):
    """A deferred dry-run lists the launcher (afterany) instead of the eval
    array sbatch, and surfaces the manual fallback command."""
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    result = submit_jobs(cfg, run_dir, submit=False, defer_eval=True)
    assert result["dry_run"] is True
    cmds = "\n".join(result["commands"])
    assert "afterany:<TRAIN_ID>" in cmds
    assert "eval_launcher.sbatch" in cmds
    assert "manual_eval_command" in result


def test_render_eval_launcher_non_array(tmp_path):
    """The eval_launcher script is a single (non-array) job that invokes the
    deferred-eval worker, with conda activation and no leftover placeholders."""
    cfg = _make_cfg(tmp_path)
    text = render_sbatch("eval_launcher", cfg, str(tmp_path / "run"))
    assert "--array" not in text
    assert "python -m xcquinox.alec.cluster._submit_eval" in text
    assert "conda activate" in text
    _assert_no_unrendered_placeholders(text)


def test_inline_eval_submit_skips_eval_array_and_eval_record(
        tmp_path, monkeypatch):
    """inline_eval=True: only 3 sbatch calls (pretrain, preflight, train) get
    issued, NO 4th eval sbatch, NO ``eval`` record in jobs.json, and
    ``submit_jobs`` returns cleanly (regression test for the prior crash where
    ``append_job_record`` rejected an ``eval_id=None`` record under the
    ``if not defer`` guard that didn't account for inline mode).
    """
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    fake = _fake_slurm_factory(ids=["6000", "6001", "6002", "6003"])
    monkeypatch.setattr(jt, "_run_slurm", fake)

    result = submit_jobs(cfg, run_dir, submit=True, inline_eval=True)

    # Top-level flags reflect inline mode.
    assert result["inline_eval"] is True
    assert result["defer_eval"] is False
    # Only FOUR sbatch calls (datagen, pretrain, preflight, train), no eval array.
    sbatch_calls = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch_calls) == 4, [" ".join(c) for c in sbatch_calls]
    joined = [" ".join(c) for c in sbatch_calls]
    # The eval array and the deferred launcher are BOTH absent.
    assert not any("eval_array.sbatch" in j for j in joined), joined
    assert not any("eval_launcher.sbatch" in j for j in joined), joined
    # job_ids carries datagen/pretrain/preflight/train ONLY, no eval (which
    # would be None under inline) and no eval_launcher (which is defer-only).
    assert result["job_ids"] == {
        "datagen": "6000", "pretrain": "6001", "preflight": "6002",
        "train": "6003",
    }, result["job_ids"]
    # jobs.json carries datagen/pretrain/preflight/train ONLY, the absence of
    # an ``eval`` record is the point: nothing to recover via sacct because eval
    # ran inline as part of each train task.
    kinds = sorted(r["kind"] for r in jt.read_job_records(run_dir))
    assert kinds == ["datagen", "preflight", "pretrain", "train"]
    # The submit_commands record was written (proves the function returned
    # cleanly past the failure point at L627 in the buggy version).
    cmds_path = os.path.join(run_dir, "submit_commands.txt")
    assert os.path.exists(cmds_path)
    assert "[submit]" in open(cmds_path).read()


def test_double_submit_guard_requires_force(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    monkeypatch.setattr(jt, "_run_slurm",
                        _fake_slurm_factory(ids=["7000", "7001", "7002",
                                                 "7003", "7004"]))
    submit_jobs(cfg, run_dir, submit=True)

    # A second submit without force must be rejected.
    monkeypatch.setattr(jt, "_run_slurm",
                        _fake_slurm_factory(ids=["8000", "8001", "8002",
                                                 "8003", "8004"]))
    with pytest.raises(RuntimeError, match="force"):
        submit_jobs(cfg, run_dir, submit=True)

    # With force=True it goes through.
    result = submit_jobs(cfg, run_dir, submit=True, force=True)
    assert result["dry_run"] is False


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


def test_midgraph_failure_surfaces_sbatch_stderr(tmp_path, monkeypatch):
    """The rollback RuntimeError must include sbatch's captured stderr, the
    real SLURM rejection reason, not just CalledProcessError's opaque str()."""
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    # Fail on a mid-graph sbatch (index 2 = preflight); fake sets stderr="rejected".
    fake = _fake_slurm_factory(ids=["9000", "9001", "9002", "9003", "9004"],
                               fail_on_index=2)
    monkeypatch.setattr(jt, "_run_slurm", fake)

    with pytest.raises(RuntimeError, match="sbatch stderr: rejected"):
        submit_jobs(cfg, run_dir, submit=True)


def test_rollback_on_first_job_failure_no_scancel(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    # Fail on the very first sbatch: nothing to roll back.
    fake = _fake_slurm_factory(fail_on_index=0)
    monkeypatch.setattr(jt, "_run_slurm", fake)

    with pytest.raises(RuntimeError, match="rolled back"):
        submit_jobs(cfg, run_dir, submit=True)

    assert [c for c in fake.calls if os.path.basename(c[0]) == "scancel"] == []
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

def test_render_train_execs_worker_for_sigterm_delivery(tmp_path):
    """The train script must ``exec`` the worker. ``#SBATCH --signal=B:TERM``
    targets the batch-step PID; only by exec-ing does the Python worker (which
    installs the SIGTERM handler that records a timeout failure.json) become
    that PID and actually receive the grace signal."""
    cfg = _make_cfg(tmp_path)
    text = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    assert "exec python -m xcquinox.alec.cluster._train_task" in text


def test_render_train_gpu_execs_worker_for_sigterm_delivery(tmp_path):
    cfg = _make_cfg(tmp_path, device="gpu", gpus_per_task=1)
    text = render_sbatch("train", cfg, str(tmp_path / "run"), array_max=39)
    assert "exec python -m xcquinox.alec.cluster._train_task" in text


# ---------------------------------------------------------------------------
# hold-out benchmark refs job (inputs.benchmark_refs_dir)
# ---------------------------------------------------------------------------

def _make_cfg_bench(tmp_path, *, density_fit=False, auxbasis=None):
    d = _base_config_dict()
    d["inputs"]["benchmark_refs_dir"] = "/shared/bench_refs"
    if density_fit:
        d["inputs"]["density_fit"] = True
        if auxbasis:
            d["inputs"]["auxbasis"] = auxbasis
    d["cluster"]["benchmark_refs_time"] = "20:00:00"
    p = tmp_path / "grid_bench.json"
    p.write_text(json.dumps(d))
    return load_grid_config(str(p))


def test_render_benchmark_refs_single_job(tmp_path):
    cfg = _make_cfg_bench(tmp_path)
    text = render_sbatch("benchmark_refs", cfg, str(tmp_path / "run"))
    _assert_no_unrendered_placeholders(text)
    assert "${BENCH_REFS_DIR}" not in text and "${BENCH_DF_FLAGS}" not in text
    assert "--array" not in text                       # single job
    # basis + out-dir are shell-quoted (a basis with parens would otherwise break
    # bash under set -euo pipefail; see test_render_benchmark_refs_basis_is_shell_safe)
    assert '--out-dir "/shared/bench_refs"' in text
    assert '--basis "def2-tzvp"' in text
    assert "--grid-level 3" in text
    assert "--density-fit" not in text
    assert "#SBATCH --time=20:00:00" in text           # benchmark_refs_time
    assert "python -m xcquinox.alec.benchmark_refs" in text


def test_render_benchmark_refs_df_flags_and_fallback_time(tmp_path):
    cfg = _make_cfg_bench(tmp_path, density_fit=True,
                          auxbasis="def2-universal-jkfit")
    text = render_sbatch("benchmark_refs", cfg, str(tmp_path / "run"))
    assert "--density-fit --auxbasis def2-universal-jkfit" in text
    # no benchmark_refs_time / preflight_time -> falls back to cluster.time
    d = _base_config_dict()
    d["inputs"]["benchmark_refs_dir"] = "/shared/bench_refs"
    p = tmp_path / "grid_bench_nb.json"
    p.write_text(json.dumps(d))
    cfg2 = load_grid_config(str(p))
    text2 = render_sbatch("benchmark_refs", cfg2, str(tmp_path / "run"))
    assert "#SBATCH --time=12:00:00" in text2


def test_render_benchmark_refs_requires_dir(tmp_path):
    cfg = _make_cfg(tmp_path)                          # no benchmark_refs_dir
    with pytest.raises(ValueError, match="benchmark_refs_dir"):
        render_sbatch("benchmark_refs", cfg, str(tmp_path / "run"))


def test_eval_and_inline_export_bench_refs_env(tmp_path):
    cfg = _make_cfg_bench(tmp_path)
    ev = render_sbatch("eval", cfg, str(tmp_path / "run"), array_max=39)
    inl = render_sbatch("train_eval_inline", cfg, str(tmp_path / "run"),
                        array_max=39)
    for text in (ev, inl):
        assert "export XCQUINOX_BENCH_REFS_DIR=/shared/bench_refs" in text
    # off by default: no export line, no unrendered placeholder
    cfg_off = _make_cfg(tmp_path)
    ev_off = render_sbatch("eval", cfg_off, str(tmp_path / "run"),
                           array_max=39)
    assert "XCQUINOX_BENCH_REFS_DIR" not in ev_off
    assert "${BENCH_REFS_ENV_LINE}" not in ev_off


def test_dry_run_includes_bench_command_iff_configured(tmp_path):
    run_dir = str(tmp_path / "run")
    res_off = submit_jobs(_make_cfg(tmp_path), run_dir, submit=False)
    assert not any("benchmark_refs" in c for c in res_off["commands"])
    assert "benchmark_refs" not in res_off["scripts"]

    run_dir2 = str(tmp_path / "run2")
    res_on = submit_jobs(_make_cfg_bench(tmp_path), run_dir2, submit=False)
    bench_lines = [c for c in res_on["commands"] if "benchmark_refs" in c]
    assert len(bench_lines) == 1
    assert "--dependency=after:<TRAIN_ID>" in bench_lines[0]
    assert os.path.isfile(
        os.path.join(run_dir2, "scripts", "benchmark_refs.sbatch"))
    assert res_on["scripts"]["benchmark_refs"].endswith(
        "benchmark_refs.sbatch")


def test_real_submit_bench_after_train_and_recorded(tmp_path, monkeypatch):
    cfg = _make_cfg_bench(tmp_path)
    run_dir = str(tmp_path / "run")
    fake = _fake_slurm_factory(
        ids=["6000", "6001", "6002", "6003", "6004", "6005"])
    monkeypatch.setattr(jt, "_run_slurm", fake)

    result = submit_jobs(cfg, run_dir, submit=True)

    assert result["job_ids"]["benchmark_refs"] == "6005"
    joined = [" ".join(c) for c in fake.calls
              if os.path.basename(c[0]) == "sbatch"]
    assert len(joined) == 6
    # eligible once the train array (6003) has BEGUN -- 'after', not afterok
    assert "--dependency=after:6003" in joined[5]
    assert joined[5].endswith("benchmark_refs.sbatch")
    records = jt.read_job_records(run_dir)
    bench = [r for r in records if r["kind"] == "benchmark_refs"]
    assert len(bench) == 1
    assert bench[0]["array_job_id"] == "6005"
    assert bench[0]["indices"] == [0]
