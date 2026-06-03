"""Task 9 (density-fitting plan): pretrain-data generation must honor the
configured basis/grid_level, support density fitting on the per-atom SCF, and
carry a basis manifest so a basis change forces regeneration rather than silent
reuse of stale def2-svp data."""
import json
import os

import numpy as np
import pytest

from xcquinox.alec import pretrain_data_gen as pdg


# --- fast: manifest + is-current logic (no SCF) ----------------------------

def _touch_npz(path):
    np.savez(path, x=np.zeros(3))


def test_is_current_false_when_file_missing(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=1) is False


def test_is_current_false_when_no_manifest(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)                                   # legacy file, no manifest
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=1) is False


def test_is_current_true_on_matching_manifest(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)
    pdg._write_pretrain_manifest(p, basis="def2-svp", grid_level=1,
                                 density_fit=False)
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=1) is True


def test_is_current_false_on_basis_mismatch(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)
    pdg._write_pretrain_manifest(p, basis="def2-svp", grid_level=1,
                                 density_fit=False)
    # a def2-tzvp run must NOT reuse def2-svp data
    assert pdg.pretrain_data_is_current(p, basis="def2-tzvp", grid_level=1) is False


def test_is_current_false_on_grid_mismatch(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)
    pdg._write_pretrain_manifest(p, basis="def2-svp", grid_level=1,
                                 density_fit=False)
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=2) is False


def test_manifest_round_trips_density_fit_flag(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    pdg._write_pretrain_manifest(p, basis="def2-tzvp", grid_level=2,
                                 density_fit=True,
                                 auxbasis="def2-universal-jkfit")
    with open(pdg._pretrain_manifest_path(p)) as f:
        meta = json.load(f)
    assert meta == {"basis": "def2-tzvp", "grid_level": 2, "density_fit": True,
                    "auxbasis": "def2-universal-jkfit"}


def test_manifest_auxbasis_defaults_none(tmp_path):
    """Default-off (full-ERI) manifest records auxbasis=None."""
    p = os.path.join(tmp_path, "pretrain_data.npz")
    pdg._write_pretrain_manifest(p, basis="def2-svp", grid_level=1,
                                 density_fit=False)
    with open(pdg._pretrain_manifest_path(p)) as f:
        meta = json.load(f)
    assert meta["auxbasis"] is None


def test_legacy_manifest_without_auxbasis_stays_current(tmp_path):
    """A legacy manifest lacking the auxbasis key must NOT trigger a spurious
    regen on the full-ERI path (auxbasis reads as None -> matches None)."""
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)
    # Old-format manifest: basis + grid_level + density_fit, NO auxbasis key.
    with open(pdg._pretrain_manifest_path(p), "w") as f:
        json.dump({"basis": "def2-svp", "grid_level": 1, "density_fit": False}, f)
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=1) is True
    # A DF run (effective auxbasis set) correctly sees it as stale.
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1, auxbasis="def2-svp-jkfit") is False


# --- ensure-driver: skip-if-current, regen on basis change -----------------

def test_ensure_regenerates_only_when_stale(tmp_path, monkeypatch):
    calls = []

    def fake_generate(out_dir, *, atoms, basis, grid_level, polarized,
                      descriptors, density_fit, auxbasis=None,
                      cusp_log_transform=True):
        calls.append(basis)
        path = os.path.join(out_dir, "pretrain_data_polarized.npz"
                            if polarized else "pretrain_data.npz")
        _touch_npz(path)
        pdg._write_pretrain_manifest(
            path, basis=basis, grid_level=grid_level, density_fit=density_fit,
            auxbasis=pdg._effective_auxbasis(basis, density_fit, auxbasis))
        return path

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", fake_generate)

    # first call: file absent -> generates
    pdg.ensure_pretrain_data(str(tmp_path), basis="def2-svp", grid_level=1,
                             polarized=False)
    # second call, same basis -> current -> NO regen
    pdg.ensure_pretrain_data(str(tmp_path), basis="def2-svp", grid_level=1,
                             polarized=False)
    # third call, new basis -> stale -> regen
    pdg.ensure_pretrain_data(str(tmp_path), basis="def2-tzvp", grid_level=1,
                             polarized=False)
    assert calls == ["def2-svp", "def2-tzvp"]


# --- physics: DF on the per-atom SCF, DF-off byte-identical -----------------

def test_atom_columns_density_fit_runs():
    """A density-fitted per-atom SCF produces finite Fx/Fc columns."""
    cols = pdg._atom_columns("He", 0, "def2-svp", 1, polarized=False,
                             descriptors=False, density_fit=True)
    assert np.all(np.isfinite(cols["Fx"])) and np.all(np.isfinite(cols["Fc"]))
    assert cols["rho"].size > 0


def test_atom_columns_density_fit_off_matches_pre_df_path():
    """density_fit defaulting to off takes the same (non-DF) SCF code path and
    reproduces the pre-DF per-atom columns to SCF tolerance (the only residual
    is machine-epsilon BLAS nondeterminism, present between any two identical
    runs: not a code-path difference). Regression guard for the new kwarg."""
    base = pdg._atom_columns("He", 0, "def2-svp", 1, polarized=False,
                             descriptors=False)
    off = pdg._atom_columns("He", 0, "def2-svp", 1, polarized=False,
                            descriptors=False, density_fit=False)
    for k in base:
        np.testing.assert_allclose(off[k], base[k], rtol=0, atol=1e-10)


@pytest.mark.slow
def test_generate_writes_basis_tagged_manifest(tmp_path):
    """A full generation writes a manifest recording the basis it used."""
    path = pdg.generate_pretrain_data_npz(
        str(tmp_path), atoms=(("He", 0),), basis="def2-svp", grid_level=1,
        polarized=False, descriptors=False)
    assert pdg.pretrain_data_is_current(path, basis="def2-svp", grid_level=1)
    assert not pdg.pretrain_data_is_current(path, basis="def2-tzvp",
                                            grid_level=1)
