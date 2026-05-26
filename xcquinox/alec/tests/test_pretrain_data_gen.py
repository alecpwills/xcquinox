"""Tests for the canonical pretrain-data generator (xcquinox.alec.pretrain_data_gen).

Uses a tiny 2-atom set (He closed-shell, H open-shell) on a coarse grid so the
PBE SCFs are fast, while still exercising both the spin=0 and spin=1 branches and
the zeta column.
"""
import numpy as np

from xcquinox.alec.pretrain_data_gen import generate_pretrain_data_npz

_TINY_ATOMS = (("He", 0), ("H", 1))


def test_polarized_npz_has_zeta_and_consistent_columns(tmp_path):
    path = generate_pretrain_data_npz(
        str(tmp_path), atoms=_TINY_ATOMS, polarized=True, descriptors=True)
    assert path.endswith("pretrain_data_polarized.npz")
    d = dict(np.load(path))
    # Required columns + zeta + descriptors all present.
    for key in ("rho_all", "sigma_all", "Fx_all", "Fc_all", "weights_all",
                "zeta_all", "cusp_all", "dm_all"):
        assert key in d, f"missing {key}"
    n = d["rho_all"].shape[0]
    assert n > 0
    # All per-point columns are aligned and finite.
    for key in ("sigma_all", "Fx_all", "Fc_all", "weights_all", "zeta_all"):
        assert d[key].shape[0] == n
        assert np.all(np.isfinite(d[key]))
    assert d["cusp_all"].shape[0] == n and d["dm_all"].shape[0] == n
    # zeta in [-1, 1]; ~0 on the closed-shell He points, ~+1 on the fully spin-
    # polarized H points -> the column must span from near 0 up toward 1.
    z = d["zeta_all"]
    assert np.all(z <= 1.0 + 1e-6) and np.all(z >= -1.0 - 1e-6)
    assert z.min() < 0.05      # He contributes ~zero polarization
    assert z.max() > 0.5       # H is strongly polarized


def test_unpolarized_npz_omits_zeta_and_default_name(tmp_path):
    path = generate_pretrain_data_npz(
        str(tmp_path), atoms=_TINY_ATOMS, polarized=False, descriptors=False)
    assert path.endswith("pretrain_data.npz")
    assert not path.endswith("pretrain_data_polarized.npz")
    d = dict(np.load(path))
    assert "zeta_all" not in d
    assert "cusp_all" not in d and "dm_all" not in d
    for key in ("rho_all", "sigma_all", "Fx_all", "Fc_all", "weights_all"):
        assert key in d
