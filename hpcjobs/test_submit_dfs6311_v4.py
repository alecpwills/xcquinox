"""Static and behavioral tests for the v4 submission script.

The script is operations code: it must never clobber the one-time provenance
backup, must gate the verification job on the sweep submission succeeding,
and must carry the resource parameters the measured requirements dictate.
Behavior is exercised through real bash with the cluster commands stubbed.
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "submit_dfs6311_v4.sh")


def _run(tmp_path, *, args=(), python_rc=0, data_dir=None):
    """Execute the script with python/sbatch stubbed; return (proc, calls)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "calls.log"
    py = bin_dir / "python"
    py.write_text(f"#!/bin/sh\necho python \"$@\" >> '{calls}'\n"
                  f"exit {python_rc}\n")
    py.chmod(0o755)
    sb = bin_dir / "sbatch"
    sb.write_text(f"#!/bin/sh\necho sbatch \"$@\" >> '{calls}'\nexit 0\n")
    sb.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    script = open(SCRIPT).read()
    # Neutralize the cluster-only conda activation for local behavioral
    # tests (no conda profile here); its presence and shape are pinned by
    # test_activates_parity_env_by_effect below.
    script = script.replace(
        "source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh",
        "true")
    script = script.replace('conda activate "$ENV_PREFIX" || true', "true")
    script = script.replace(
        '"$ENV_PREFIX"/*) echo "[submit-v4] python=$PYBIN" ;;',
        '*) echo "[submit-v4] python=$PYBIN" ;;')
    if data_dir is not None:
        script = script.replace(
            "DATA_DIR=/gpfs/scratch/awills/pretrain_data_dfs_6311ppg3df2pd_g3_allelem",
            f"DATA_DIR={data_dir}")
    patched = tmp_path / "script.sh"
    patched.write_text(script)
    proc = subprocess.run(["bash", str(patched), *args], env=env,
                          capture_output=True, text=True, cwd=tmp_path)
    return proc, (calls.read_text() if calls.exists() else "")


def test_shell_syntax():
    r = subprocess.run(["bash", "-n", SCRIPT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_backup_is_no_clobber_and_submit_order(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    npz = data / "pretrain_data_polarized.npz"
    npz.write_bytes(b"ORIGINAL-V3-BYTES")
    (data / "pretrain_data_polarized.npz.manifest.json").write_text("{}")

    proc, calls = _run(tmp_path, data_dir=str(data))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (data / "pretrain_data_polarized.npz.v3bak").read_bytes() \
        == b"ORIGINAL-V3-BYTES"
    # submit first, nan_verify second
    lines = [l for l in calls.splitlines() if l]
    assert "cluster submit" in lines[0] and "--submit" in lines[0]
    assert "long-96core" in lines[0] and "--max-nodes 3" in lines[0]
    assert "dfs_step7.dfs6311_grid3_v4.yaml" in lines[0]
    assert lines[1].startswith("sbatch") and "nan_verify" in lines[1]

    # Second run after the datagen "regenerated" the file: the backup must
    # keep the ORIGINAL bytes.
    npz.write_bytes(b"REGENERATED-BYTES")
    proc2, _ = _run(tmp_path, data_dir=str(data))
    assert proc2.returncode == 0
    assert (data / "pretrain_data_polarized.npz.v3bak").read_bytes() \
        == b"ORIGINAL-V3-BYTES", "re-run clobbered the provenance backup"


def test_failed_sweep_submission_blocks_nan_verify(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    proc, calls = _run(tmp_path, data_dir=str(data), python_rc=3)
    assert proc.returncode == 3, "sweep failure rc must propagate"
    assert "sbatch" not in calls, \
        "nan_verify submitted despite sweep submission failing"


def test_partition_override(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _proc, calls = _run(tmp_path, args=("long-40core",), data_dir=str(data))
    assert "--partition long-40core" in calls.splitlines()[0]


def test_activates_parity_env_by_effect():
    """The submit CLI imports xcquinox -> jax at module import, so the script
    must activate the parity env itself and verify by EFFECT (the resolved
    python path), never by conda's return code -- running under the login
    shell's base env produced ModuleNotFoundError: jax."""
    text = open(SCRIPT).read()
    assert "conda_envs/xcquinox_j070" in text
    assert 'conda activate "$ENV_PREFIX" || true' in text
    assert '"$ENV_PREFIX"/*)' in text, "activation-by-effect case missing"
    assert "parity env python not active" in text
    # The guard must run BEFORE the sweep submission.
    assert text.index("parity env python not active")         < text.index("cluster submit")
