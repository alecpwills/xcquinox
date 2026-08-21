"""Per-system pretrain columns on the parent functional's own density.

Section 6 deviation 1 puts the pretraining set "at the production identity ...
on the parent functional's own self-consistent densities (PBE for GGA-rung,
SCAN for meta-GGA; DFS used PBE for both)". The density comes from
data.precompute_fixed_density_data(..., reference_xc=...), the one place this
library produces a frozen parent density, so the rows a network is fit on and
the rows the fidelity certificate measures it on are the same quadrature of the
same density by construction. These tests run real SCFs on tiny systems
(sto-3g, grid level 0 or 1, H / H2 / OH / H2O).

Tolerances are anchored to measured floors, quoted at each constant.
"""
import numpy as np
import pytest
from pyscf import dft, gto

import xcquinox.alec.pretrain_data_gen as pdg
from xcquinox.alec.pretrain_data_gen import (
    PretrainSystem, _atom_columns, _molecule_columns)


_H2 = PretrainSystem(name="h2", atom="H 0 0 0; H 0 0 0.74", charge=0, spin=0)
_OH = PretrainSystem(name="oh", atom="O 0 0 0; H 0 0 0.97", charge=0, spin=1)
_H2O = PretrainSystem(
    name="h2o", atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    charge=0, spin=0)

#: Integrated-energy agreement floor (Ha). The rebuilt quadrature omits the
#: points the rho floor drops; their exchange energy was measured at
#: <= 4.8e-12 Ha on OH/STO-3G level 0 and <= 3.3e-11 Ha on N and H2O at
#: def2-SVP level 1 (correlation <= 1e-22 Ha), and the point-wise sum agrees
#: with pyscf's own nr_rks / nr_uks to <= 5.3e-15 Ha.
_E_TOL = 1e-10

#: Density identification floors on H2O/STO-3G level 0 (closed shell, no
#: orbital degeneracy): the PBE and SCAN self-consistent densities differ by
#: 4.5e-2 at their maximum (2.3e-4 of the peak density), while a repeated SCF
#: of the same recipe reproduces a density to 2.6e-13 (PBE) / 8.5e-14 (SCAN).
#: The open-shell OH is NOT usable for this: its 2-Pi hole lands on a different
#: degenerate component from run to run (repeat spread 1.6e-3 for PBE, 0.45
#: for SCAN), the artifact the orientation lock exists for.
_RHO_SAME = 1e-8
_RHO_DIFFERENT = 1e-3


def _scf(system, basis, grid_level, reference_xc="pbe", max_cycle=None):
    """An independent pyscf SCF. Its post-kernel ``mf.grids`` is the grid pyscf
    itself integrates on -- the Becke-Lebedev grid pruned at the first
    ``get_veff`` call (``prune_small_rho_grids_``) -- reached through pyscf's
    own path rather than the builder's replay of it."""
    mol = gto.M(atom=system.atom, basis=basis, charge=system.charge,
                spin=system.spin, verbose=0)
    mf = dft.UKS(mol) if system.spin else dft.RKS(mol)
    mf.xc = reference_xc
    mf.grids.level = grid_level
    if max_cycle is not None:
        mf.max_cycle = max_cycle
    mf.kernel()
    return mol, mf


def _precompute(system, basis, grid_level, reference_xc="pbe"):
    from xcquinox.alec.data import precompute_fixed_density_data
    return precompute_fixed_density_data(
        pdg._mol_spec_for(system, basis, grid_level), required_keys=(),
        descriptors=(), reference_xc=reference_xc)


def _scf_density(system, basis, grid_level, reference_xc):
    """An independent SCF of ``reference_xc`` and its density on its grid."""
    mol, mf = _scf(system, basis, grid_level, reference_xc)
    dm = np.asarray(mf.make_rdm1())
    dm_tot = dm if dm.ndim == 2 else dm[0] + dm[1]
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=0)
    return np.einsum("pi,ij,pj->p", ao, dm_tot, ao)


def _record_from_scf(system, basis, grid_level, reference_xc, max_cycle=None):
    """A record in the conventions of ``precompute_fixed_density_data``
    (per-spin J for UKS, V_xc = V_eff - J_total), from an SCF that may be
    stopped before convergence."""
    mol, mf = _scf(system, basis, grid_level, reference_xc, max_cycle)
    dm = np.asarray(mf.make_rdm1())
    j = np.asarray(mf.get_j(mol, dm))
    veff = np.asarray(mf.get_veff(mol, dm))
    vxc = veff - (j.sum(axis=0)[None] if dm.ndim == 3 else j)
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=0)
    dm_tot = dm if dm.ndim == 2 else dm[0] + dm[1]
    record = {
        "dm_pbe": dm, "h_core": np.asarray(mf.get_hcore()),
        "s_matrix": np.asarray(mf.get_ovlp()), "j_matrix": j, "vxc_pbe": vxc,
        "rho_grid": np.einsum("pi,ij,pj->p", ao, dm_tot, ao),
        "grid_weights": np.asarray(mf.grids.weights),
    }
    return mf, record


def _rebuilt_energies(cols, x_key="Fx", c_key="Fc"):
    got_x = float(np.sum(cols["weights"] * cols["e_lda_x"] * (1.0 + cols[x_key])))
    got_c = float(np.sum(cols["weights"] * cols["e_lda_c"] * (1.0 + cols[c_key])))
    return got_x, got_c


# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

def test_molecule_columns_are_aligned_and_finite():
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=True,
                             descriptors=True)
    n = cols["rho"].shape[0]
    assert n > 0
    for key in ("sigma", "Fx", "Fc", "Fx_scan", "Fc_scan", "weights", "zeta",
                "e_lda_x", "e_lda_c"):
        assert np.asarray(cols[key]).shape == (n,), key
        assert np.all(np.isfinite(np.asarray(cols[key]))), key
    assert np.asarray(cols["metagga"]).shape == (n, 1)
    assert np.asarray(cols["cusp"]).shape == (n, 2)
    assert np.asarray(cols["dm"]).shape[0] == n
    assert np.asarray(cols["rung35"]).shape == (n, 2)
    assert np.asarray(cols["rung35ms"]).shape == (n, 6)
    # The JAX-computed blocks must come back in double precision too.
    for key in ("rho", "metagga", "cusp", "rung35", "rung35ms", "e_lda_x"):
        assert np.asarray(cols[key]).dtype == np.float64, key


def test_molecule_columns_reproduce_the_atom_path_for_a_free_atom():
    """A free atom is the single-nucleus case of the molecular builder. A
    divergence would mean the atomic rows and the molecular rows are not the
    same quantity, which is the failure the coverage change exists to remove."""
    a = _atom_columns("H", 1, "sto-3g", 0, polarized=True, descriptors=True)
    m = _molecule_columns(PretrainSystem("H", "H 0 0 0", 0, 1), "pbe",
                          "sto-3g", 0, polarized=True, descriptors=True)
    assert set(a) == set(m)
    for key in a:
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(m[key]),
                                      err_msg=key)


def test_columns_sit_on_the_precomputes_grid_and_density():
    """The rows are a quadrature of the SAME grid and the SAME density matrix
    the training features are built from; that identity is what makes the
    certificate's E_xc^NN - E_xc^parent a statement about the network rather
    than about two pipelines that were supposed to agree."""
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=False,
                             descriptors=False)
    md = _precompute(_H2, "sto-3g", 0)
    w = np.asarray(md["grid_weights"])
    rho = np.asarray(md["rho_grid"])
    keep = rho > pdg._RHO_FLOOR
    np.testing.assert_array_equal(cols["weights"], w[keep])
    np.testing.assert_allclose(cols["rho"], rho[keep], rtol=0, atol=1e-12)


def test_grid_guard_refuses_a_record_from_another_grid(monkeypatch):
    """The rebuilt grid must be the precompute's; a record whose weights are
    not the rebuilt ones is refused rather than integrated."""
    import xcquinox.alec.data as data_mod
    real = data_mod.precompute_fixed_density_data

    def _other_grid(*args, **kwargs):
        md = dict(real(*args, **kwargs))
        md["grid_weights"] = 1.01 * np.asarray(md["grid_weights"])
        return md

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _other_grid)
    with pytest.raises(RuntimeError, match="grid"):
        _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=False,
                          descriptors=False)


# ---------------------------------------------------------------------------
# Energy densities: the columns integrate to the parent's energies
# ---------------------------------------------------------------------------

def test_energy_density_columns_invert_the_stored_ratio():
    """w * e_lda * (1 + F) is the parent's energy quadrature. Summing it must
    reproduce pyscf's own integrated exchange and correlation on the same
    density and grid, up to the density floor and the +-5 clip on the ratio."""
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=False,
                             descriptors=False)
    md = _precompute(_H2, "sto-3g", 0)
    mol, mf = _scf(_H2, "sto-3g", 0)
    dm = np.asarray(md["dm_pbe"])
    assert mf.grids.weights.shape == np.asarray(md["grid_weights"]).shape
    ref_x = float(mf._numint.nr_rks(mol, mf.grids, "PBE,", dm)[1])
    ref_c = float(mf._numint.nr_rks(mol, mf.grids, ",PBE", dm)[1])
    got_x, got_c = _rebuilt_energies(cols)
    assert abs(got_x - ref_x) < _E_TOL, (got_x, ref_x)
    assert abs(got_c - ref_c) < _E_TOL, (got_c, ref_c)


def test_open_shell_energy_density_columns_use_the_spin_resolved_baseline():
    """The open-shell Fx / Fc are libxc spin=1 ratios, so their denominators are
    the SPIN-POLARIZED LDA and PW92 per-electron energies at the total density.
    e_lda_x / e_lda_c must be those same denominators times rho_tot, or the
    energy term would integrate a different functional than the fit."""
    cols = _molecule_columns(_OH, "pbe", "sto-3g", 0, polarized=True,
                             descriptors=False)
    md = _precompute(_OH, "sto-3g", 0)
    mol, mf = _scf(_OH, "sto-3g", 0)
    dm = np.asarray(md["dm_pbe"])
    assert mf.grids.weights.shape == np.asarray(md["grid_weights"]).shape
    ref_x = float(mf._numint.nr_uks(mol, mf.grids, "PBE,", dm)[1])
    ref_c = float(mf._numint.nr_uks(mol, mf.grids, ",PBE", dm)[1])
    got_x, got_c = _rebuilt_energies(cols)
    assert abs(got_x - ref_x) < _E_TOL, (got_x, ref_x)
    assert abs(got_c - ref_c) < _E_TOL, (got_c, ref_c)


def test_scan_columns_integrate_to_scans_energy_on_the_scan_density():
    """With a SCAN parent the meta-GGA targets, on the SCAN density and with
    the kinetic-energy density the columns carry, must integrate to pyscf's
    own SCAN exchange and correlation (spin=1 for the open shell; pyscf builds
    tau from the density matrix independently of the generator)."""
    cols = _molecule_columns(_OH, "scan", "sto-3g", 0, polarized=True,
                             descriptors=False)
    md = _precompute(_OH, "sto-3g", 0, reference_xc="scan")
    mol, mf = _scf(_OH, "sto-3g", 0, "scan")
    dm = np.asarray(md["dm_pbe"])
    assert mf.grids.weights.shape == np.asarray(md["grid_weights"]).shape
    ref_x = float(mf._numint.nr_uks(mol, mf.grids, "SCAN,", dm)[1])
    ref_c = float(mf._numint.nr_uks(mol, mf.grids, ",SCAN", dm)[1])
    got_x, got_c = _rebuilt_energies(cols, "Fx_scan", "Fc_scan")
    assert abs(got_x - ref_x) < _E_TOL, (got_x, ref_x)
    assert abs(got_c - ref_c) < _E_TOL, (got_c, ref_c)


# ---------------------------------------------------------------------------
# Parent selection
# ---------------------------------------------------------------------------

def test_scan_reference_uses_the_scan_density():
    """reference_xc='scan' must reach precompute_fixed_density_data and come
    back with SCAN's own self-consistent density: it matches an independent
    SCAN SCF and differs from the PBE one. H2 in a minimal basis cannot tell
    (one occupied orbital fixed by symmetry, the same density under every
    functional) and neither can OH (degenerate-component spread, see the
    floors above); the closed-shell H2O can."""
    cols = _molecule_columns(_H2O, "scan", "sto-3g", 0, polarized=False,
                             descriptors=False)
    rho_scan = _scf_density(_H2O, "sto-3g", 0, "scan")
    rho_pbe = _scf_density(_H2O, "sto-3g", 0, "pbe")
    keep = rho_scan > pdg._RHO_FLOOR
    assert cols["rho"].shape == rho_scan[keep].shape
    assert float(np.max(np.abs(cols["rho"] - rho_scan[keep]))) < _RHO_SAME
    assert float(np.max(np.abs(cols["rho"] - rho_pbe[keep]))) > _RHO_DIFFERENT


def test_system_columns_refuse_an_unknown_reference_xc():
    with pytest.raises(ValueError, match="reference_xc"):
        _molecule_columns(_H2, "blyp", "sto-3g", 0, polarized=False,
                          descriptors=False)


# ---------------------------------------------------------------------------
# Density sanity: converged, on its own grid, with its electrons
# ---------------------------------------------------------------------------

def test_scf_gradient_norm_reproduces_pyscfs_convergence_measure():
    """The stored (h, J, V_xc, P, S) record carries pyscf's own convergence
    measure: ||S^-1/2 (F P S - S P F) S^-1/2||_F / sqrt(2) is the norm of
    ``mf.get_grad`` for both the restricted (occupation 2, pyscf's factor of
    2) and the unrestricted (two channels, occupation 1) Fock matrices.
    Measured agreement: <= 6e-8 relative wherever the norm is above round-off."""
    for system, max_cycle in ((_H2O, None), (_H2O, 1), (_OH, None), (_OH, 1)):
        mf, record = _record_from_scf(system, "sto-3g", 0, "pbe", max_cycle)
        ref = float(np.linalg.norm(
            mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock())))
        got = pdg._scf_gradient_norm(record)
        assert abs(got - ref) <= 1e-12 + 1e-6 * ref, (system.name, max_cycle,
                                                     got, ref)


def test_require_sane_density_accepts_the_precomputes_record():
    """Converged records on level-0 grids pass: the quadrature error of the
    electron count is 2.5e-3 on H2 and 4.8e-3 on OH at this level (<= 5.3e-5
    at level 1 on H2O/def2-SVP), below the tolerance by construction."""
    for system, n_electrons in ((_H2, 2), (_OH, 9)):
        md = _precompute(system, "sto-3g", 0)
        pdg._require_sane_density(md, system, "pbe", "sto-3g", 0, n_electrons)


def test_require_sane_density_catches_a_density_that_lost_electrons():
    """The check that needs no cooperation from the precompute: the quadrature
    of the stored density against the electron count. It catches a grid too
    coarse to resolve a diffuse density and a density matrix that does not
    belong to the stored grid."""
    md = dict(_precompute(_H2, "sto-3g", 0))
    with pytest.raises(RuntimeError, match="h2"):
        pdg._require_sane_density(md, _H2, "pbe", "sto-3g", 0, 3)
    lost = dict(md)
    lost["rho_grid"] = 0.9 * np.asarray(md["rho_grid"])  # a tenth gone
    with pytest.raises(RuntimeError, match="electrons"):
        pdg._require_sane_density(lost, _H2, "pbe", "sto-3g", 0, 2)


def test_require_sane_density_reports_a_non_converged_scf_when_told():
    s = PretrainSystem("si2", "Si 0 0 0", 0, 0)
    md = {"rho_grid": np.full(4, 0.5), "grid_weights": np.full(4, 5.0),
          "scf_converged": False}
    with pytest.raises(RuntimeError, match="converge"):
        pdg._require_sane_density(md, s, "scan", "def2-svp", 3, 10)


def test_require_sane_density_detects_an_unconverged_density():
    """A stalled SCF still integrates to N electrons (any idempotent N-electron
    density matrix does), so only the Fock-density commutator can tell. An SCF
    stopped after one cycle (gradient norm 0.17 on OH, against the sqrt(1e-9)
    criterion) is refused from the record alone."""
    _, record = _record_from_scf(_OH, "sto-3g", 0, "pbe", max_cycle=1)
    with pytest.raises(RuntimeError, match="converge"):
        pdg._require_sane_density(record, _OH, "pbe", "sto-3g", 0, 9)


# ---------------------------------------------------------------------------
# Exchange footing and charge
# ---------------------------------------------------------------------------

def test_open_shell_molecule_carries_per_channel_exchange_rows():
    """Section 3.2: open-shell rows are posed per spin channel. The molecular
    path must reach the same row builder the atomic path does."""
    cols = _molecule_columns(_OH, "pbe", "sto-3g", 0, polarized=True,
                             descriptors=True,
                             exchange_footing="spin_channel")
    x = cols["x_rows"]
    assert x is not None
    assert x["rho"].ndim == 1
    assert x["rho"].shape[0] > cols["rho"].shape[0]
    np.testing.assert_allclose(x["rung35"][:, 0], x["rung35"][:, 1],
                               rtol=0, atol=1e-14)


def test_closed_shell_molecule_has_no_separate_exchange_rows():
    """rho_a = rho_b makes the doubled density the total one, so a closed
    shell's total-density rows ALREADY are the exact-spin-scaling rows."""
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=True,
                             descriptors=True,
                             exchange_footing="spin_channel")
    assert cols["x_rows"] is None


def test_charged_system_runs_at_its_charge():
    """F- is a BH76 species and a pretraining system; its SCF must carry the
    charge, or the row set is a different atom."""
    cols = _molecule_columns(PretrainSystem("H-", "H 0 0 0", -1, 0), "pbe",
                             "sto-3g", 0, polarized=False, descriptors=False)
    neutral = _atom_columns("H", 1, "sto-3g", 0, polarized=False,
                            descriptors=False)
    assert float(np.sum(cols["weights"] * cols["rho"])) > \
        float(np.sum(neutral["weights"] * neutral["rho"])) + 0.5


# ---------------------------------------------------------------------------
# One naming function for the data file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("polarized, reference_xc, expected", [
    (False, "pbe", "pretrain_data.npz"),
    (True, "pbe", "pretrain_data_polarized.npz"),
    (False, "scan", "pretrain_data_scan.npz"),
    (True, "scan", "pretrain_data_polarized_scan.npz"),
])
def test_pretrain_data_filename_encodes_polarization_and_parent(
        polarized, reference_xc, expected):
    assert pdg.pretrain_data_filename(polarized, reference_xc) == expected


def test_run_pretrain_and_the_writer_share_the_naming_function():
    """``run_pretrain`` looks the file up through ``pretrain._pretrain_data_filename``
    and the generator writes it through ``pretrain_data_filename``; a second
    spelling of the name in either place is a divergence waiting for the first
    non-PBE parent."""
    import types
    from xcquinox.alec.pretrain import _pretrain_data_filename
    for flag in (False, True):
        arch = types.SimpleNamespace(use_polarized_correlation=flag)
        assert _pretrain_data_filename(arch) == pdg.pretrain_data_filename(flag)


def test_generated_file_is_named_by_the_naming_function(tmp_path, monkeypatch):
    """The writer's filename and the skip-if-current probe's filename are the
    naming function's output, for both polarizations."""
    def _fake_cols(sym, spin, basis, grid_level, **kw):
        n = 3
        cols = {k: np.ones(n) for k in ("rho", "sigma", "Fx", "Fc", "Fx_scan",
                                         "Fc_scan", "weights", "zeta")}
        cols["metagga"] = np.ones((n, 1))
        return cols

    monkeypatch.setattr(pdg, "_atom_columns", _fake_cols)
    for flag in (False, True):
        path = pdg.generate_pretrain_data_npz(
            str(tmp_path), atoms=(("H", 1),), basis="sto-3g", grid_level=0,
            polarized=flag, descriptors=False)
        assert path.endswith(pdg.pretrain_data_filename(flag))
        assert pdg.ensure_pretrain_data(
            str(tmp_path), atoms=(("H", 1),), basis="sto-3g", grid_level=0,
            polarized=flag, descriptors=False) == path
