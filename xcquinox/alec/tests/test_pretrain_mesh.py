"""Tests for the (s, alpha) parameter-space pretrain mesh.

The mesh exists because SCAN's enhancement factors are 3-D in (r_s, s, alpha)
and the atomic grids leave the alpha axis underdetermined (meta-GGA C-net
measured up to 0.457 from SCAN away from alpha=1, against <= 0.013 for the
GGA C-net on the same data). These tests pin the mesh's analytic construction,
its npz/staleness/manifest plumbing, the descriptor gate (mesh appended ONLY
for archs whose descriptor set is exactly ``(metagga,)``), the recorded
provenance, and the guarantee that non-mesh architectures are byte-identical
whether or not the mesh keys exist.
"""
import json

import numpy as np
import pytest

import xcquinox.alec.pretrain_data_gen as pdg
from xcquinox.alec.config import ArchitectureConfig, PretrainSpec
from xcquinox.alec.pretrain import run_pretrain


# ---------------------------------------------------------------------------
# The analytic mesh itself
# ---------------------------------------------------------------------------

def test_mesh_columns_realize_the_design_nodes():
    """Each (r_s, s, alpha) node is realized as a physical (rho, sigma, tau)
    triple, and compute_alpha applied to that triple must RECOVER the design
    alpha -- the self-consistency that guarantees the mesh's alpha column is
    the same quantity the SCF descriptor produces."""
    cols = pdg._mesh_columns()
    n = len(pdg.MESH_RS) * len(pdg.MESH_S) * len(pdg.MESH_ALPHA)
    assert cols["rho"].shape == (n,)
    assert cols["sigma"].shape == (n,)
    assert cols["metagga"].shape == (n, 1)
    for k in ("rho", "sigma", "Fx_scan", "Fc_scan", "weights", "zeta"):
        assert np.all(np.isfinite(cols[k])), k
    # The recovered alpha equals the design grid broadcast over (rs, s).
    design = np.broadcast_arrays(
        np.asarray(pdg.MESH_RS)[:, None, None],
        np.asarray(pdg.MESH_S)[None, :, None],
        np.asarray(pdg.MESH_ALPHA)[None, None, :])[2].ravel()
    np.testing.assert_allclose(cols["metagga"].reshape(-1), design,
                               rtol=1e-6, atol=1e-8)
    # Targets are enhancement-factor residuals in the same clipped convention
    # as the atomic path (ratio - 1, clipped to [-5, 5]).
    assert np.all(cols["Fx_scan"] >= -5.0) and np.all(cols["Fx_scan"] <= 5.0)
    assert np.all(cols["Fc_scan"] >= -5.0) and np.all(cols["Fc_scan"] <= 5.0)
    # The mesh spans genuinely non-UEG territory: alpha=1 rows alone cannot
    # produce this spread.
    assert cols["Fx_scan"].std() > 0.01


# ---------------------------------------------------------------------------
# npz plumbing: staleness + weight share
# ---------------------------------------------------------------------------

def _manifest(p, atoms=(("H", 1),)):
    with open(str(p) + ".manifest.json", "w") as f:
        json.dump({"basis": "def2-svp", "grid_level": 1, "density_fit": False,
                   "auxbasis": None,
                   "atoms": [[s, sp] for s, sp in atoms]}, f)


def test_staleness_forces_regen_when_mesh_absent(tmp_path):
    """A file with SCAN/metagga columns but no *_mesh keys predates the mesh;
    a meta-GGA arch pretrained on it has the underdetermined alpha axis, so
    the currency check must force a regen."""
    base = dict(rho_all=np.ones(3), sigma_all=np.ones(3), Fx_all=np.ones(3),
                Fc_all=np.ones(3), weights_all=np.ones(3),
                metagga_all=np.ones((3, 1)), Fx_scan_all=np.ones(3),
                Fc_scan_all=np.ones(3))
    p = tmp_path / "pretrain_data.npz"
    np.savez(p, **base)
    _manifest(p)
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1, atoms=[("H", 1)]) is False

    p2 = tmp_path / "sub" / "pretrain_data.npz"
    p2.parent.mkdir()
    np.savez(p2, **base, rho_mesh=np.ones(4), sigma_mesh=np.ones(4),
             Fx_scan_mesh=np.ones(4), Fc_scan_mesh=np.ones(4),
             metagga_mesh=np.ones((4, 1)), weights_mesh=np.ones(4))
    _manifest(p2)
    assert pdg.pretrain_data_is_current(
        p2, basis="def2-svp", grid_level=1, atoms=[("H", 1)]) is True


@pytest.mark.slow
def test_generator_writes_mesh_keys_and_weight_share(tmp_path):
    """The production generator emits the mesh columns alongside the atomic
    ones, with the mesh's quadrature-weight share equal to
    MESH_WEIGHT_FRACTION and zeta_mesh present on the polarized file."""
    path = pdg.generate_pretrain_data_npz(
        str(tmp_path), atoms=(("He", 0),), basis="sto-3g", grid_level=0,
        polarized=True, descriptors=False, density_fit=False)
    d = np.load(path)
    for k in ("rho_mesh", "sigma_mesh", "Fx_scan_mesh", "Fc_scan_mesh",
              "metagga_mesh", "weights_mesh", "zeta_mesh"):
        assert k in d.files, k
    w_atom = float(d["weights_all"].sum())
    w_mesh = float(d["weights_mesh"].sum())
    np.testing.assert_allclose(w_mesh / (w_atom + w_mesh),
                               pdg.MESH_WEIGHT_FRACTION, rtol=1e-10)
    meta = pdg.read_pretrain_manifest(path)
    assert meta["mesh"]["weight_fraction"] == pdg.MESH_WEIGHT_FRACTION
    assert meta["mesh"]["alpha"] == list(pdg.MESH_ALPHA)


# ---------------------------------------------------------------------------
# The descriptor gate + provenance + byte identity, on synthetic data
# ---------------------------------------------------------------------------

N_ATOMIC, N_MESH = 6, 4


def _write_synthetic(tmp_path, *, with_mesh=True):
    base = dict(
        rho_all=np.linspace(0.1, 2.0, N_ATOMIC),
        sigma_all=np.linspace(0.0, 1.0, N_ATOMIC),
        metagga_all=np.linspace(0.0, 2.0, N_ATOMIC).reshape(-1, 1),
        cusp_all=np.linspace(-0.5, 0.5, 2 * N_ATOMIC).reshape(N_ATOMIC, 2),
        Fx_all=np.zeros(N_ATOMIC), Fc_all=np.full(N_ATOMIC, -0.1),
        Fx_scan_all=np.full(N_ATOMIC, 0.3),
        Fc_scan_all=np.full(N_ATOMIC, -0.4),
        weights_all=np.ones(N_ATOMIC),
    )
    if with_mesh:
        base.update(
            rho_mesh=np.linspace(0.2, 1.0, N_MESH),
            sigma_mesh=np.linspace(0.1, 0.6, N_MESH),
            metagga_mesh=np.linspace(0.0, 3.0, N_MESH).reshape(-1, 1),
            Fx_scan_mesh=np.full(N_MESH, 0.7),
            Fc_scan_mesh=np.full(N_MESH, -0.7),
            weights_mesh=np.full(N_MESH, 0.25),
        )
    np.savez(tmp_path / "pretrain_data.npz", **base)


def _spec(tmp_path, arch, n_steps=3):
    ck = tmp_path / f"ck_{arch.name}"
    return PretrainSpec(arch=arch, data_dir=str(tmp_path),
                        checkpoint_dir=str(ck), n_steps=n_steps, seed=0)


def _mgga_arch(name="t_mgga", **kw):
    return ArchitectureConfig.from_spec(
        name, 2, 8, descriptors=["metagga"], meta_gga=True, **kw)


def test_mesh_rows_reach_only_the_pure_metagga_arch(tmp_path, monkeypatch):
    """Row counts and targets, captured from the real run_pretrain branch via
    a stub trainer: the pure-(metagga,) arch consumes atomic+mesh rows with
    the SCAN targets extended by the mesh targets; a GGA arch and a
    multi-descriptor meta-GGA arch consume the atomic rows alone."""
    import xcquinox.train

    _write_synthetic(tmp_path, with_mesh=True)
    captured = {}

    class _StubTrainer:
        def __init__(self, *, model, **_kw):
            self.model = model

        def __call__(self, _epochs, inputs, targets, **_kw):
            captured.setdefault("shapes", []).append(
                (np.asarray(inputs[0]).shape[0],
                 np.asarray(targets[0]).reshape(-1)))
            return self.model, [0.0]

    monkeypatch.setattr(xcquinox.train, "xcTrainer", _StubTrainer)

    mgga = _mgga_arch()
    run_pretrain(_spec(tmp_path, mgga))
    n_rows, fx = captured["shapes"][0]
    assert n_rows == N_ATOMIC + N_MESH
    np.testing.assert_allclose(fx[:N_ATOMIC], 0.3)
    np.testing.assert_allclose(fx[N_ATOMIC:], 0.7)
    meta = json.load(open(f"{tmp_path}/ck_{mgga.name}/pretrain_metadata.json"))
    assert meta["pretrain_mesh"] is True

    captured.clear()
    gga = ArchitectureConfig.from_spec("t_gga", 2, 8)
    run_pretrain(_spec(tmp_path, gga))
    n_rows, fx = captured["shapes"][0]
    assert n_rows == N_ATOMIC
    np.testing.assert_allclose(fx, 0.0)          # PBE targets, no mesh
    meta = json.load(open(f"{tmp_path}/ck_{gga.name}/pretrain_metadata.json"))
    assert meta["pretrain_mesh"] is False

    captured.clear()
    mixed = ArchitectureConfig.from_spec(
        "t_mgga_cusp", 2, 8, descriptors=["cusp", "metagga"], meta_gga=True)
    run_pretrain(_spec(tmp_path, mixed))
    n_rows, fx = captured["shapes"][0]
    assert n_rows == N_ATOMIC                    # gate holds: no mesh rows
    np.testing.assert_allclose(fx, 0.3)          # still SCAN targets
    meta = json.load(open(f"{tmp_path}/ck_{mixed.name}/pretrain_metadata.json"))
    assert meta["pretrain_mesh"] is False


def test_non_mesh_archs_byte_identical_with_and_without_mesh_keys(tmp_path):
    """The regression guard: for a GGA arch the trained checkpoint bytes must
    not depend on whether the npz carries mesh keys at all."""
    a = tmp_path / "with_mesh"
    b = tmp_path / "without_mesh"
    a.mkdir(); b.mkdir()
    _write_synthetic(a, with_mesh=True)
    _write_synthetic(b, with_mesh=False)
    arch = ArchitectureConfig.from_spec("t_gga_bytes", 2, 8)
    run_pretrain(_spec(a, arch, n_steps=5))
    run_pretrain(_spec(b, arch, n_steps=5))
    for f in ("xnet.eqx", "cnet.eqx"):
        ba = (a / "ck_t_gga_bytes" / f).read_bytes()
        bb = (b / "ck_t_gga_bytes" / f).read_bytes()
        assert ba == bb, f"{f} differs with mesh keys present"


def test_mesh_missing_for_mgga_arch_still_runs_with_warning(tmp_path, capsys):
    """A meta-GGA arch on mesh-less data must pretrain on the atomic rows
    (not crash) and say so -- the cluster path regenerates, but a local file
    may predate the mesh."""
    _write_synthetic(tmp_path, with_mesh=False)
    run_pretrain(_spec(tmp_path, _mgga_arch("t_mgga_nomesh")))
    out = capsys.readouterr().out
    assert "no (s, alpha) mesh" in out
    meta = json.load(
        open(f"{tmp_path}/ck_t_mgga_nomesh/pretrain_metadata.json"))
    assert meta["pretrain_mesh"] is False
