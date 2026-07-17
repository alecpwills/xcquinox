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


# ---------------------------------------------------------------------------
# M3 wiring: the emitted SCAN/alpha columns must actually reach the pretrain
# network inputs at the layout the meta-GGA nets read, and run_pretrain must
# route the SCAN targets (not PBE) for a meta_gga arch.
# ---------------------------------------------------------------------------

def test_assemble_pretrain_input_includes_alpha_for_mgga():
    """The metagga alpha column reaches the pretrain net inputs at the right index.
    xnet input is [rho, sigma, alpha] (alpha at 2, xnet is zeta-blind); a polarized
    cnet input is [rho, sigma, zeta, alpha] (alpha AFTER the zeta column). A GGA arch
    carries no alpha column (input width 2). Indices are cross-checked against the
    materialized nets' ``metagga_alpha_index``."""
    from xcquinox.alec.pretrain import _assemble_pretrain_descriptors
    from xcquinox.alec.config import ARCHITECTURES, ArchitectureConfig
    from xcquinox.alec.networks import create_network_pair

    N = 6
    # metagga_all carries DISTINCT per-row values so the alpha column is
    # identifiable in the assembled input; zeta is distinct from it so column
    # ordering (zeta BEFORE alpha) is testable.
    alpha_col = np.arange(N, dtype=float).reshape(N, 1)
    zeta = np.linspace(-0.5, 0.5, N)
    pdata = {
        "rho_all": np.linspace(0.1, 2.0, N),
        "sigma_all": np.linspace(0.0, 1.0, N),
        "metagga_all": alpha_col,
        "zeta_all": zeta,
        "cusp_all": np.zeros(N),
        "rung35_all": np.zeros(N),
    }

    # (a) non-polarized meta-GGA: xnet and cnet inputs both [rho, sigma, alpha].
    mgga = ARCHITECTURES["deep_mgga_3x16"]
    assert mgga.meta_gga and not mgga.use_polarized_correlation
    xnet, cnet = create_network_pair(mgga)
    x_in = np.asarray(_assemble_pretrain_descriptors(mgga, pdata, for_cnet=False))
    c_in = np.asarray(_assemble_pretrain_descriptors(mgga, pdata, for_cnet=True))
    assert x_in.shape[1] == 3 and c_in.shape[1] == 3
    x_alpha_idx = 2 + xnet.metagga_alpha_index         # [rho, sigma] + descriptor offset
    c_alpha_idx = 2 + cnet.metagga_alpha_index         # no zeta column when unpolarized
    assert x_alpha_idx == 2 and c_alpha_idx == 2
    assert np.allclose(x_in[:, x_alpha_idx], alpha_col[:, 0])
    assert np.allclose(c_in[:, c_alpha_idx], alpha_col[:, 0])

    # (b) polarized meta-GGA: cnet inserts zeta at index 2, alpha AFTER it (index 3);
    #     xnet stays zeta-blind (exchange is spin-scaled, not zeta-fed).
    pol = ArchitectureConfig.from_spec(
        "mgga_pol", 3, 16, descriptors=["metagga"], meta_gga=True,
        use_polarized_correlation=True, descriptor_log_transform=True)
    pxnet, pcnet = create_network_pair(pol)
    px_in = np.asarray(_assemble_pretrain_descriptors(pol, pdata, for_cnet=False))
    pc_in = np.asarray(_assemble_pretrain_descriptors(pol, pdata, for_cnet=True))
    assert px_in.shape[1] == 3          # xnet unchanged by the polarized flag
    assert np.allclose(px_in[:, 2 + pxnet.metagga_alpha_index], alpha_col[:, 0])
    assert pc_in.shape[1] == 4          # [rho, sigma, zeta, alpha]
    pc_alpha_idx = 3 + pcnet.metagga_alpha_index
    assert pc_alpha_idx == 3
    assert np.allclose(pc_in[:, 2], zeta)                        # zeta column present
    assert np.allclose(pc_in[:, pc_alpha_idx], alpha_col[:, 0])  # alpha AFTER zeta

    # (c) GGA arch: no metagga column, input width 2.
    gga = ARCHITECTURES["deep_3x16"]
    assert not gga.meta_gga
    g_in = np.asarray(_assemble_pretrain_descriptors(gga, pdata, for_cnet=False))
    assert g_in.shape[1] == 2


def test_pretrain_target_routes_scan_for_mgga(tmp_path, monkeypatch):
    """run_pretrain feeds the SCAN targets (Fx_scan_all/Fc_scan_all) to a meta_gga
    arch and the PBE targets (Fx_all/Fc_all) to a GGA arch. The xcTrainer is stubbed
    to capture the target array each phase consumes, so the REAL routing branch in
    run_pretrain runs (a GGA cannot fit SCAN's alpha-dependence, so the meta-GGA nets
    must pretrain to SCAN). The targets are made numerically distinct so the consumed
    array identifies which branch was taken."""
    import xcquinox.train
    from xcquinox.alec.config import ARCHITECTURES, PretrainSpec
    from xcquinox.alec.pretrain import run_pretrain

    N = 5
    Fx_all = np.zeros(N)
    Fx_scan_all = np.full(N, 0.3)
    Fc_all = np.full(N, -0.1)
    Fc_scan_all = np.full(N, -0.4)
    np.savez(
        tmp_path / "pretrain_data.npz",
        rho_all=np.linspace(0.1, 2.0, N),
        sigma_all=np.linspace(0.0, 1.0, N),
        metagga_all=np.arange(N, dtype=float).reshape(N, 1),
        Fx_all=Fx_all, Fc_all=Fc_all,
        Fx_scan_all=Fx_scan_all, Fc_scan_all=Fc_scan_all,
        weights_all=np.ones(N),
    )

    def _run_capture(arch, ckpt_dir):
        """Drive run_pretrain with a stub trainer; return [Fx_target, Fc_target]."""
        captured = []

        class _StubTrainer:
            def __init__(self, *, model, **_kw):
                self.model = model

            def __call__(self, _epochs, _inputs, targets):
                captured.append(np.asarray(targets[0]))
                return self.model, [0.0]

        monkeypatch.setattr(xcquinox.train, "xcTrainer", _StubTrainer)
        run_pretrain(PretrainSpec(arch=arch, data_dir=str(tmp_path),
                                  checkpoint_dir=str(ckpt_dir), n_steps=1))
        return captured

    fx_m, fc_m = _run_capture(ARCHITECTURES["deep_mgga_3x16"], tmp_path / "ck_mgga")
    assert np.allclose(fx_m, Fx_scan_all) and not np.allclose(fx_m, Fx_all)
    assert np.allclose(fc_m, Fc_scan_all) and not np.allclose(fc_m, Fc_all)

    fx_g, fc_g = _run_capture(ARCHITECTURES["deep_3x16"], tmp_path / "ck_gga")
    assert np.allclose(fx_g, Fx_all) and not np.allclose(fx_g, Fx_scan_all)
    assert np.allclose(fc_g, Fc_all) and not np.allclose(fc_g, Fc_scan_all)
