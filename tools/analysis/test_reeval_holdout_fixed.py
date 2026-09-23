"""Tests for ``reeval_holdout_fixed.py`` -- the run-level PBE density table.

The table is model-free: for each species it reads the reference file's
density, the reference calculation's own PBE density and the grid weights.
What is pinned: the closed-form numbers on a hand-written grid, that a
reference without its PBE twin and a reference without a density are each
recorded by name rather than dropped, that neither costs an SCF, and that
the loader path is judged by the same two keys.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


tool = _load("reeval_holdout_fixed")

#: A two-point grid: weights [3, 1], reference density [2, 1], PBE twin
#: [2.5, 0.5]. RMSE = sqrt((3*0.25 + 1*0.25)/4) = 0.5, L1 = (3*0.5 + 1*0.5)/4
#: = 0.5, eps = 2/N_e with N_e = 3*2 + 1*1 = 7, sum of weights 4.
_WEIGHTS = np.array([3.0, 1.0])
_RHO_REF = np.array([2.0, 1.0])
_RHO_PBE = np.array([2.5, 0.5])


def _species(name: str, path, atoms: int = 2):
    return SimpleNamespace(name=name, atom_composition=(("H", atoms),),
                           external_data_path=None if path is None else str(path))


def _assert_closed_form(row):
    assert row["density_rmse_pbe"] == pytest.approx(0.5)
    assert row["density_l1_pbe"] == pytest.approx(0.5)
    assert row["density_eps_l1_pbe"] == pytest.approx(2.0 / 7.0)
    assert row["n_electrons"] == pytest.approx(7.0)
    assert row["grid_weight_sum"] == pytest.approx(4.0)


def _run(tmp_path, specs, precompute_one):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return run_dir, tool.run_pbe_density_table(
        run_dir, density_refs=str(tmp_path / "refs"),
        pools_loader=lambda basis, grid_level, refs_dir: (specs, []),
        precompute_one=precompute_one)


def test_the_table_records_every_reference_by_its_outcome(tmp_path):
    """Three reference files: one with the twin, one without it, one without
    a reference density. The first is measured, the other two are listed by
    name, and the SCF fallback -- the sentinel here -- never runs, since the
    loader takes the baseline from the same file key whose absence put the
    species on that path. The atom is skipped before any file is read, and a
    species with no reference file is not in the pool the table reads.

    Oracle: the closed forms of the two-point grid above, and the written
    file, which equals the returned table.
    """
    refs = tmp_path / "refs"
    refs.mkdir()
    np.savez(refs / "with_twin.npz", rho_ref_grid=_RHO_REF,
             rho_pbe_grid=_RHO_PBE, grid_weights=_WEIGHTS)
    np.savez(refs / "no_twin.npz", rho_ref_grid=_RHO_REF,
             grid_weights=_WEIGHTS)
    np.savez(refs / "no_density.npz", grid_weights=_WEIGHTS,
             vxc_ref=np.zeros(2))
    specs = {
        "with_twin": _species("with_twin", refs / "with_twin.npz"),
        "no_twin": _species("no_twin", refs / "no_twin.npz"),
        "no_density": _species("no_density", refs / "no_density.npz"),
        "h": _species("h", refs / "absent.npz", atoms=1),
        "unreferenced": _species("unreferenced", None),
    }
    scf_calls = []

    def no_scf(ms):
        scf_calls.append(ms.name)
        raise AssertionError("the SCF fallback must not run")

    run_dir, payload = _run(tmp_path, specs, no_scf)

    assert scf_calls == []
    assert payload["failures"] == {}
    assert payload["no_pbe_twin"] == ["no_twin"]
    assert payload["no_reference"] == ["no_density"]
    assert sorted(payload["errors"]) == ["with_twin"]
    _assert_closed_form(payload["errors"]["with_twin"])
    written = json.loads((run_dir / "pbe_density_errors.json").read_text())
    assert written == payload


def test_a_reference_without_weights_takes_the_loader_path(tmp_path):
    """A readable file that carries both densities but no weights cannot be
    measured from the file alone and goes through the loader; the loader's
    record is judged by the same two keys, so a record carrying the twin is
    measured and one without it is listed, never measured against the
    locally recomputed density.

    Oracle: the closed forms again, from a loader that supplies the twin for
    one species and withholds it for the other.
    """
    refs = tmp_path / "refs"
    refs.mkdir()
    for name in ("loader_twin", "loader_no_twin"):
        np.savez(refs / f"{name}.npz", rho_ref_grid=_RHO_REF,
                 rho_pbe_grid=_RHO_PBE)
    specs = {name: _species(name, refs / f"{name}.npz")
             for name in ("loader_twin", "loader_no_twin")}
    loaded = []

    def loader(ms):
        loaded.append(ms.name)
        record = {"rho_ref_grid": _RHO_REF, "grid_weights": _WEIGHTS,
                  "rho_grid": np.zeros(2)}
        if ms.name == "loader_twin":
            record["rho_pbe_ref_grid"] = _RHO_PBE
        return record

    _run_dir, payload = _run(tmp_path, specs, loader)

    assert sorted(loaded) == ["loader_no_twin", "loader_twin"]
    assert payload["failures"] == {}
    assert payload["no_reference"] == []
    assert payload["no_pbe_twin"] == ["loader_no_twin"]
    assert sorted(payload["errors"]) == ["loader_twin"]
    _assert_closed_form(payload["errors"]["loader_twin"])
