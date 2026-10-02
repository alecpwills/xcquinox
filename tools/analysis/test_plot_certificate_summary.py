"""The donor-path resolution in ``plot_certificate_summary.py``.

The certificate bars plot, per (label, arch), the numbers each architecture
WARM-STARTED under. A run whose resolved config warm-starts an arch from a
donor has no run-local certificate for it, so the collector must resolve
through the config -- in both directions: the donor's certificate is the
arch's row, and a file left in the run-local slot of a donated arch is not
plotted beside the donor's as a second word on the same arch. The
run-local-slot direction is the discriminating one: a plain run-local glob
plots it, so each such case below fails against a run-local
implementation.

Oracle: ``collect_certificates`` on synthetic runs whose one architecture is
donor-backed, with the certificate placed in the donor directory, the
run-local slot, or neither.
"""
from __future__ import annotations

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


plot = _load("plot_certificate_summary")

_ARCH = "deep_3x16"


def _write_certificate(directory: Path, *, dAE: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": _ARCH,
                   "tolerances": {"tol_AE": 0.5},
                   "per_atomization": [{"name": "H2O", "dAE_kcalmol": dAE},
                                       {"name": "NH3", "dAE_kcalmol": 0.1}]},
                  f)


def _run_config(run: Path, donor: Path) -> None:
    import yaml
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


def _donor_backed_run(tmp_path: Path, *, donor_certificate: bool,
                      run_local_certificate: bool = False) -> Path:
    donor = tmp_path / "donor_run" / "pretrain" / _ARCH
    donor.mkdir(parents=True)
    if donor_certificate:
        _write_certificate(donor, dAE=0.2)
    run = tmp_path / "run"
    run.mkdir()
    if run_local_certificate:
        _write_certificate(run / "pretrain" / _ARCH, dAE=9.9)
    _run_config(run, donor)
    return run


def test_collect_plots_the_donor_certificate_of_a_donor_backed_arch(tmp_path):
    run = _donor_backed_run(tmp_path, donor_certificate=True)
    records = plot.collect_certificates([("v7", str(run))])
    assert [(label, arch) for label, arch, _ in records] == [("v7", _ARCH)]
    assert records[0][2]["max"] == 0.2


def test_collect_ignores_a_run_local_certificate_of_a_donor_backed_arch(
        tmp_path):
    """A leftover PASS in the run-local slot is not the donor's word on the
    arch: the donor carries nothing here, so the run is refused as holding
    no certificate -- a run-local glob would plot the leftover at 9.9."""
    run = _donor_backed_run(tmp_path, donor_certificate=False,
                            run_local_certificate=True)
    try:
        plot.collect_certificates([("v7", str(run))])
    except ValueError as exc:
        assert "no fidelity_certificate.json" in str(exc)
    else:
        raise AssertionError("a donor without a certificate was plotted")


def test_collect_keeps_the_run_local_glob_without_a_config(tmp_path):
    """A run directory with no loadable resolved config plots exactly as it
    did before the donor rule existed."""
    run = tmp_path / "run"
    run.mkdir()
    _write_certificate(run / "pretrain" / _ARCH, dAE=0.3)
    records = plot.collect_certificates([("v7", str(run))])
    assert [(label, arch) for label, arch, _ in records] == [("v7", _ARCH)]
    assert records[0][2]["max"] == 0.3
