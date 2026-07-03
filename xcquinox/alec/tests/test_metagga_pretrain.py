"""M3: SCAN (meta-GGA) pretrain targets + iso-orbital alpha column.

DFS pretrains to SCAN (a meta-GGA); a GGA cannot fit SCAN's alpha-dependence, so
meta_gga archs need SCAN targets. These pin that pretrain_data_gen emits finite
SCAN Fx/Fc + the metagga alpha column (RKS + UKS), and that the staleness guard
forces a regen for pre-meta-GGA data files.
"""
import json
import numpy as np
import pytest

import xcquinox.alec.pretrain_data_gen as pdg


def test_atom_columns_emits_scan_and_metagga_rks():
    cols = pdg._atom_columns("He", 0, "def2-svp", 1, polarized=False,
                             descriptors=False, density_fit=False, auxbasis=None,
                             cusp_log_transform=True)
    for k in ("Fx_scan", "Fc_scan", "metagga"):
        assert k in cols, f"missing pretrain column {k!r}"
        assert np.all(np.isfinite(cols[k])), f"non-finite {k!r}"
    # metagga alpha is one per-grid column, aligned with rho
    assert cols["metagga"].shape == (cols["rho"].shape[0], 1)
    # SCAN targets are distinct from the PBE targets (the meta-GGA content matters)
    assert not np.allclose(cols["Fx_scan"], cols["Fx"])


def test_atom_columns_emits_scan_and_metagga_uks():
    # open-shell N (quartet) exercises the per-spin tau -> total alpha UKS path
    cols = pdg._atom_columns("N", 3, "def2-svp", 1, polarized=True,
                             descriptors=False, density_fit=False, auxbasis=None,
                             cusp_log_transform=True)
    for k in ("Fx_scan", "Fc_scan", "metagga"):
        assert np.all(np.isfinite(cols[k])), f"non-finite {k!r} (UKS)"
    assert cols["metagga"].shape == (cols["rho"].shape[0], 1)


def test_staleness_forces_regen_when_metagga_column_absent(tmp_path):
    # a pre-meta-GGA file: PBE targets but no metagga_all / Fx_scan_all
    p = tmp_path / "pretrain_data.npz"
    np.savez(p, rho_all=np.ones(3), sigma_all=np.ones(3), Fx_all=np.ones(3),
             Fc_all=np.ones(3), weights_all=np.ones(3))
    with open(str(p) + ".manifest.json", "w") as f:
        json.dump({"basis": "def2-svp", "grid_level": 1, "density_fit": False,
                   "auxbasis": None, "atoms": [["H", 1]]}, f)
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1, atoms=[("H", 1)]) is False
