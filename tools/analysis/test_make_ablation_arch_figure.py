"""The donor-path resolution in ``make_ablation_arch_figure.py``.

The figure annotates each architecture with its record-layer certificate
status and refuses to draw an uncertified one as certified. An architecture
that warm-starts from a donor (stated in the run's own resolved config) has
no certificate of its own under the run's ``pretrain/`` tree, so both the
status read and the footer's worst-number summary must resolve through the
config -- in BOTH directions: a donor's PASS is the arch's status, and a
file left in the run-local slot does not certify a donor-backed arch. The
run-local-slot direction is the discriminating one: a run-local read
reports it PASS, so each such case below fails against a run-local
implementation.

Oracle: ``_arch_certificate_status`` and ``fidelity_summary`` on a synthetic
run whose one architecture is donor-backed, with the certificate placed in
the donor directory, the run-local slot, or neither.
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


figure = _load("make_ablation_arch_figure")

_ARCH = "deep_3x16"


def _write_pass_certificate(directory: Path, *, atom: float, ae: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": _ARCH,
                   "summary": {"max_atom_mHa": atom, "max_dAE_kcalmol": ae}},
                  f)


def _donor_backed_run(tmp_path: Path, *, donor_certificate: bool,
                      run_local_certificate: bool = False) -> Path:
    """A run whose resolved config warm-starts its one arch from a donor
    outside the run; the run's own ``pretrain/`` tree exists only when a
    certificate is planted in it for the discriminating direction."""
    import yaml

    donor = tmp_path / "donor_run" / "pretrain" / _ARCH
    donor.mkdir(parents=True)
    if donor_certificate:
        _write_pass_certificate(donor, atom=0.1, ae=0.2)
    run = tmp_path / "run"
    run.mkdir()
    if run_local_certificate:
        # A different pair of numbers, so a summary read from the wrong slot
        # is distinguishable from one read from the donor by value.
        _write_pass_certificate(run / "pretrain" / _ARCH, atom=3.3, ae=4.4)
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


def test_arch_certificate_status_reads_the_donor(tmp_path):
    run = _donor_backed_run(tmp_path, donor_certificate=True)
    assert figure._arch_certificate_status(run, _ARCH) == "PASS"


def test_arch_certificate_status_ignores_a_run_local_certificate_of_a_donor_backed_arch(
        tmp_path):
    """A PASS in the run-local slot does not certify a donor-backed arch: the
    donor carries nothing, so the status is MISSING -- a run-local read
    reports PASS here and certifies the figure over a file the run's own
    gates never saw."""
    run = _donor_backed_run(tmp_path, donor_certificate=False,
                            run_local_certificate=True)
    assert figure._arch_certificate_status(run, _ARCH) == "MISSING"


def test_fidelity_summary_reads_the_donor_certificate_numbers(tmp_path):
    """The footer's worst numbers come from the donor's certificate, not the
    run-local slot: the two carry different values here, and the summary
    discloses the donor's."""
    run = _donor_backed_run(tmp_path, donor_certificate=True,
                            run_local_certificate=True)
    summary = figure.fidelity_summary(run, [_ARCH])
    assert summary is not None
    assert summary["n_archs"] == 1
    assert summary["max_atom_mHa"] == 0.1
    assert summary["max_dAE_kcalmol"] == 0.2


def test_fidelity_summary_counts_a_certificate_less_donor_as_unreadable(tmp_path):
    """A donor-backed arch whose donor carries no certificate is UNREADABLE
    to the summary even when the run-local slot holds a PASS: the footer then
    states the absence (n_archs_unreadable) rather than a bound no
    certificate of the loaded networks states."""
    run = _donor_backed_run(tmp_path, donor_certificate=False,
                            run_local_certificate=True)
    summary = figure.fidelity_summary(run, [_ARCH])
    assert summary is not None
    assert summary["n_archs"] == 0
    assert summary["n_archs_unreadable"] == 1
    assert summary["max_atom_mHa"] is None
    assert summary["max_dAE_kcalmol"] is None
