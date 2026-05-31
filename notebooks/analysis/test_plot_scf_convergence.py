"""Tests for ``plot_scf_convergence`` — per-molecule SCF-step ingest + render."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "plot_scf_convergence", _HERE / "plot_scf_convergence.py")
sc = importlib.util.module_from_spec(_spec)
sys.modules["plot_scf_convergence"] = sc
_spec.loader.exec_module(sc)


def test_molecule_scf_trace_extracts_ordered_steps():
    rec = {
        "molecule": "H2O",
        "scf_energy_step_0": -76.30, "scf_energy_residual_0": 0.10,
        "scf_energy_step_1": -76.38, "scf_energy_residual_1": 0.02,
        "scf_energy_step_2": -76.40, "scf_energy_residual_2": 0.0,
        "cycles_run": 3, "scf_converged": True,
    }
    tr = sc.molecule_scf_trace(rec)
    assert tr["steps"] == [0, 1, 2]
    assert tr["energies"] == [-76.30, -76.38, -76.40]
    assert tr["residuals"][0] == 0.10
    assert tr["cycles_run"] == 3 and tr["converged"] is True


def test_molecule_scf_trace_empty_without_steps():
    tr = sc.molecule_scf_trace({"molecule": "H", "cycles_run": 0})
    assert tr["steps"] == []


def _write_pm(run_dir: Path, idx: int, records):
    sd = run_dir / "checkpoints" / f"spec_{idx:04d}" / "eval_holdout"
    sd.mkdir(parents=True)
    (sd / "per_molecule.json").write_text(json.dumps(records))


def test_collect_spec_scf_traces_filters_to_molecules_with_traces(tmp_path):
    _write_pm(tmp_path, 0, [
        {"molecule": "H2O", "scf_energy_step_0": -76.3,
         "scf_energy_residual_0": 0.1, "scf_energy_step_1": -76.4,
         "scf_energy_residual_1": 0.0, "cycles_run": 2, "scf_converged": True},
        {"molecule": "H", "cycles_run": 0, "scf_converged": True},  # no trace
    ])
    traces = sc.collect_spec_scf_traces(tmp_path, 0)
    assert [t["molecule"] for t in traces] == ["H2O"]


def test_plot_spec_convergence_writes_png(tmp_path):
    traces = [
        {"molecule": "H2O", "steps": [0, 1, 2],
         "energies": [-76.3, -76.38, -76.4],
         "residuals": [0.1, 0.02, 0.0], "cycles_run": 3, "converged": True},
        {"molecule": "CH4", "steps": [0, 1, 2],
         "energies": [-40.4, -40.45, -40.46],
         "residuals": [0.06, 0.01, 0.0], "cycles_run": 3, "converged": False},
    ]
    out = sc.plot_spec_convergence(traces, tmp_path / "conv.png", title="t")
    assert out.is_file() and out.stat().st_size > 2000


def test_plot_spec_convergence_handles_empty(tmp_path):
    out = sc.plot_spec_convergence([], tmp_path / "empty.png", title="t")
    assert out.is_file()
