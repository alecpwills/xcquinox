"""Donor-path resolution in ``merge_v4_arms.py``.

The merge tool refuses an arm whose registry architectures lack a PASS
pretraining-fidelity certificate, re-checks the PASS's records against the
files on disk, and links the certified pretrain directories into the view.
An arm architecture that warm-starts from a donor (another run's pretrain
product, stated in the arm's own resolved config) holds no run-local pretrain
directory at all, so every one of those reads must resolve through the config
the arm actually ran with, or the arm can never clear a gate its sweep never
had a product for.

Oracles: the tool's own functions on a synthetic arm whose one architecture
is donor-backed -- no run-local ``pretrain/`` tree exists, the donor carries
a PASS certificate naming real checkpoint files, and the arm's resolved
config states the donor.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


tool = _load("merge_v4_arms")

_ARCH = "deep_3x16"


def _write_arm_run(tmp_path: Path) -> Path:
    """A run dir whose resolved config warm-starts its one arch from a donor
    outside the run; the run has no ``pretrain/`` tree at all."""
    import yaml

    donor = tmp_path / "donor_run" / "pretrain" / _ARCH
    donor.mkdir(parents=True)
    blobs = {}
    for name in ("xnet.eqx", "cnet.eqx"):
        blob = f"{name}-donor-bytes".encode()
        (donor / name).write_bytes(blob)
        blobs[name] = hashlib.sha256(blob).hexdigest()
    payload = {
        "verdict": "PASS",
        "arch": _ARCH,
        "checkpoint": {"dir": str(donor),
                       "xnet_sha256": blobs["xnet.eqx"],
                       "cnet_sha256": blobs["cnet.eqx"]},
        "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2},
    }
    with open(donor / "fidelity_certificate.json", "w") as f:
        json.dump(payload, f)

    run = tmp_path / "arm_run"
    (run / "specs").mkdir(parents=True)
    cfg = {
        "sweep": {"arch": [_ARCH], "loss": ["delta_ae"], "metric": ["l2"],
                  "subset_size": [4], "solver": ["fast"]},
        "solvers": {"fast": {"mode": "fixed_density", "max_cycles": 1}},
        "hyperparams": {"n_steps": 200, "lr_start": 1e-3, "lr_end": 1e-5,
                        "lr_decay_start": 0.2, "grad_clip": 1.0,
                        "gradnorm_alpha": 1.5, "vxc_weight": 1.0,
                        "density_weight": 0.5},
        "inputs": {"external_refs_dir": "/refs", "subset_ledger_path":
                   "/ledger.json", "basis": "def2-tzvp", "grid_level": 3,
                   "output_root": "/out"},
        "pretrain": {"data_dir": "/pretrain_data",
                     "donor_checkpoints": {_ARCH: str(donor)}},
        "cluster": {"partition": "short", "time": "01:00:00", "mem": "8G",
                    "cpus_per_task": 1, "array_throttle": 1,
                    "eval_array_throttle": 1, "max_concurrent_tasks": 4},
        "domain_profile": "gmtkn55_subset",
    }
    with open(run / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    return run


def test_certificate_statuses_read_the_donor(tmp_path):
    """The status mapping reads the donor the config names: a PASS where no
    run-local pretrain tree exists, and a certificate path inside the donor."""
    from xcquinox.pipeline.cluster.fidelity import VERDICT_PASS

    run = _write_arm_run(tmp_path)
    statuses = tool._arm_certificate_statuses(run, [_ARCH])
    status, _reason, path, payload = statuses[_ARCH]
    assert status == VERDICT_PASS
    assert "donor_run" in str(path), path
    assert payload["verdict"] == "PASS"


def test_validate_certificates_accepts_a_donor_arch(tmp_path):
    """The record re-checks pass a donor-backed arch: the digests are hashed
    from the DONOR's files (a run-local path would refuse with a file that
    cannot be read), and the arm's config carries no run-local product."""
    statuses = tool._validate_arm_fidelity_certificates(
        _write_arm_run(tmp_path), [_ARCH], arm="a4")
    assert statuses[_ARCH][0] == "PASS"


def test_carry_certificates_links_the_donor_directory(tmp_path):
    """The view's per-arch pretrain slot links the DONOR directory, so the
    figure layer reading ``<view>/pretrain/<arch>`` sees the certificate the
    gate cleared."""
    run = _write_arm_run(tmp_path)
    view = tmp_path / "view"
    view.mkdir()
    tool._carry_arm_certificates(run, view, [_ARCH], "a4")
    linked = (view / "pretrain" / _ARCH).resolve()
    donor = run.parent / "donor_run" / "pretrain" / _ARCH
    assert linked == donor.resolve()
    assert (linked / "fidelity_certificate.json").is_file()
