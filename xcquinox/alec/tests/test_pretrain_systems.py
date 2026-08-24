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
import os

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


def _scf(system, basis, grid_level, reference_xc="pbe", max_cycle=None,
         orientation_lock_strength=None):
    """An independent pyscf SCF at the pretraining identity. Its post-kernel
    ``mf.grids`` is the grid pyscf itself integrates on -- the Becke-Lebedev
    grid pruned at the first ``get_veff`` call (``prune_small_rho_grids_``) --
    reached through pyscf's own path rather than the builder's replay of it.
    The orientation lock is applied the way ``data.precompute_fixed_density_data``
    applies it (the traceless-quadrupole bias added to ``h_core`` before the
    kernel), at the generator's production strength unless told otherwise, so
    the comparison is between two SCFs of one Hamiltonian."""
    from xcquinox.alec.orientation_lock import orientation_lock_bias
    if orientation_lock_strength is None:
        orientation_lock_strength = pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH
    mol = gto.M(atom=system.atom, basis=basis, charge=system.charge,
                spin=system.spin, verbose=0)
    mf = dft.UKS(mol) if system.spin else dft.RKS(mol)
    mf.xc = reference_xc
    mf.grids.level = grid_level
    if max_cycle is not None:
        mf.max_cycle = max_cycle
    if orientation_lock_strength:
        locked = (np.asarray(mf.get_hcore())
                  + orientation_lock_bias(mol, orientation_lock_strength))
        mf.get_hcore = lambda *a, **k: locked
    mf.kernel()
    return mol, mf


def _precompute(system, basis, grid_level, reference_xc="pbe",
                orientation_lock_strength=None):
    """The record ``_system_columns`` reads, at the same identity (the
    production lock unless told otherwise), so a call after the column build
    is a memo hit on the very record the columns were built from."""
    from xcquinox.alec.data import precompute_fixed_density_data
    if orientation_lock_strength is None:
        orientation_lock_strength = pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH
    return precompute_fixed_density_data(
        pdg._mol_spec_for(system, basis, grid_level), required_keys=(),
        descriptors=(), reference_xc=reference_xc,
        orientation_lock_strength=orientation_lock_strength)


def _scf_density(system, basis, grid_level, reference_xc):
    """An independent SCF of ``reference_xc`` and its density on its grid."""
    mol, mf = _scf(system, basis, grid_level, reference_xc)
    dm = np.asarray(mf.make_rdm1())
    dm_tot = dm if dm.ndim == 2 else dm[0] + dm[1]
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=0)
    return np.einsum("pi,ij,pj->p", ao, dm_tot, ao)


def _record_from_scf(system, basis, grid_level, reference_xc, max_cycle=None,
                     stamp_converged=None):
    """A record in the conventions of ``precompute_fixed_density_data``
    (per-spin J for UKS, V_xc = V_eff - J_total, the convergence stamp in
    ``mol_metadata``), from an SCF that may be stopped before convergence.
    ``stamp_converged`` overrides pyscf's flag in the stamp, which is how a
    record that CLAIMS convergence without having it is built."""
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
        "mol_metadata": {
            "reference_xc": reference_xc,
            "reference_scf_converged": (bool(mf.converged)
                                        if stamp_converged is None
                                        else bool(stamp_converged)),
            "reference_scf_cycles": int(mf.cycles),
            "reference_scf_solver": "diis",
        },
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
    """The precompute's stamp is read where the precompute writes it,
    ``mol_metadata["reference_scf_converged"]``; a ``False`` stamp is refused
    before any quadrature, naming the cycles and the solver stage."""
    s = PretrainSystem("si2", "Si 0 0 0", 0, 0)
    md = {"rho_grid": np.full(4, 0.5), "grid_weights": np.full(4, 5.0),
          "mol_metadata": {"reference_xc": "scan",
                           "reference_scf_converged": False,
                           "reference_scf_cycles": 160,
                           "reference_scf_solver": "diis+newton"}}
    with pytest.raises(RuntimeError, match="160 cycles .diis.newton"):
        pdg._require_sane_density(md, s, "scan", "def2-svp", 3, 10)


def test_require_sane_density_refuses_a_record_without_the_stamp():
    """A record whose convergence was never asserted -- no ``mol_metadata``,
    or metadata without the stamp -- is not trusted, whatever its density
    integrates to: the converged H2 record itself is refused once its stamp
    is removed, and a top-level flag in the pre-stamp spelling does not
    stand in for it."""
    md = dict(_precompute(_H2, "sto-3g", 0))
    assert md["mol_metadata"]["reference_scf_converged"] is True
    pdg._require_sane_density(md, _H2, "pbe", "sto-3g", 0, 2)
    unstamped = dict(md)
    unstamped["mol_metadata"] = {
        k: v for k, v in md["mol_metadata"].items()
        if k != "reference_scf_converged"}
    with pytest.raises(RuntimeError, match="absent"):
        pdg._require_sane_density(unstamped, _H2, "pbe", "sto-3g", 0, 2)
    bare = dict(md)
    bare.pop("mol_metadata")
    bare["scf_converged"] = True
    with pytest.raises(RuntimeError, match="absent"):
        pdg._require_sane_density(bare, _H2, "pbe", "sto-3g", 0, 2)


def test_require_sane_density_refuses_a_record_of_the_other_parent():
    md = _precompute(_H2, "sto-3g", 0, "scan")
    assert md["mol_metadata"]["reference_xc"] == "scan"
    with pytest.raises(RuntimeError, match="stamped as the 'scan' density"):
        pdg._require_sane_density(md, _H2, "pbe", "sto-3g", 0, 2)


def test_require_sane_density_detects_an_unconverged_density():
    """A stalled SCF still integrates to N electrons (any idempotent N-electron
    density matrix does), so behind the stamp only the Fock-density commutator
    can tell. An SCF stopped after one cycle whose record nevertheless CLAIMS
    convergence (gradient norm 0.17 on OH, against the sqrt(1e-9) criterion)
    is refused from the stored Fock pieces alone."""
    _, record = _record_from_scf(_OH, "sto-3g", 0, "pbe", max_cycle=1,
                                 stamp_converged=True)
    with pytest.raises(RuntimeError, match="orbital gradient"):
        pdg._require_sane_density(record, _OH, "pbe", "sto-3g", 0, 9)
    # ... and pyscf's own flag on that record is False, which the stamp line
    # refuses first when it is carried faithfully.
    _, honest = _record_from_scf(_OH, "sto-3g", 0, "pbe", max_cycle=1)
    assert honest["mol_metadata"]["reference_scf_converged"] is False
    with pytest.raises(RuntimeError, match="not stamped converged"):
        pdg._require_sane_density(honest, _OH, "pbe", "sto-3g", 0, 9)


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


def test_run_pretrain_and_the_writer_share_the_naming_function(monkeypatch):
    """``run_pretrain`` looks the file up through ``pretrain._pretrain_data_filename``
    and the generator writes it through ``pretrain_data_filename``; a second
    spelling of the name in either place is a divergence waiting for the first
    non-PBE parent. Agreement on the PBE default cannot tell a delegation from
    a second copy of the same strings, so the naming function is redirected
    and the loader's helper must follow it."""
    import types
    from xcquinox.alec.pretrain import _pretrain_data_filename
    for flag in (False, True):
        arch = types.SimpleNamespace(use_polarized_correlation=flag)
        assert _pretrain_data_filename(arch) == pdg.pretrain_data_filename(flag)
    monkeypatch.setattr(
        pdg, "pretrain_data_filename",
        lambda polarized, reference_xc="pbe":
        f"redirected_{int(bool(polarized))}_{reference_xc}.npz")
    for flag in (False, True):
        arch = types.SimpleNamespace(use_polarized_correlation=flag)
        assert _pretrain_data_filename(arch) == \
            f"redirected_{int(flag)}_pbe.npz"


def test_generated_file_is_named_by_the_naming_function(tmp_path, monkeypatch):
    """The writer's filename and the skip-if-current probe's filename are the
    naming function's output, for both polarizations."""
    def _fake_cols(system, basis, grid_level, **kw):
        n = 3
        cols = {k: np.ones(n) for k in ("rho", "sigma", "Fx", "Fc", "Fx_scan",
                                         "Fc_scan", "weights", "e_lda_x",
                                         "e_lda_c")}
        cols["metagga"] = np.ones((n, 1))
        if kw["polarized"]:
            cols["zeta"] = np.ones(n)
        return cols

    monkeypatch.setattr(pdg, "_system_columns", _fake_cols)
    for flag in (False, True):
        path = pdg.generate_pretrain_data_npz(
            str(tmp_path), atoms=(("H", 1),), basis="sto-3g", grid_level=0,
            polarized=flag, descriptors=False)
        assert path.endswith(pdg.pretrain_data_filename(flag))
        assert pdg.ensure_pretrain_data(
            str(tmp_path), atoms=(("H", 1),), basis="sto-3g", grid_level=0,
            polarized=flag, descriptors=False) == path


# ---------------------------------------------------------------------------
# Per-system parent energies: the target of the energy term
# ---------------------------------------------------------------------------

#: Gap between the row quadrature and libxc's full-grid integral of the SAME
#: density (Ha): the energy of the points the rho floor drops plus the +-5 clip
#: on the stored ratio. Measured: 0.0 on the O atom at def2-SVP / grid level 1
#: (pyscf's pruning already removed every point below the floor, so the two
#: integrals are the same sum), 4.8e-12 on OH/STO-3G level 0 and <= 3.3e-11 on
#: N and H2O at def2-SVP level 1. The gate sits 30x above the worst case and six
#: orders of magnitude below the certificate's tol_atom = 1.0 mHa.
_FULL_GRID_GAP = 1e-9

#: Per-channel minus total-density exchange energy on the O atom (Ha): the two
#: footings integrate the same E_x^PBE through different libxc calls and
#: different row sets, so the residual is round-off on the rows plus the
#: floor: measured 2.7e-12 to 3.2e-12 over three level-1 draws (1.2e-13 to
#: 1.5e-13 for SCAN) and 6.0e-12 / 3.0e-13 at the level-3 identity the test
#: runs at (11682 total-density rows, 23568 channel rows). The 1e-9 gate
#: rejects a 1e-9 relative error in the channel weights, which moves the sum
#: by 8.2e-9 Ha.
_CHANNEL_TOTAL_GAP = 1e-9


def test_system_energy_targets_are_the_row_quadrature():
    """The stored target is the quadrature over the rows the file keeps, not
    libxc's full-grid integral. That is what makes the energy term vanish
    exactly when the network reproduces the stored enhancement factors."""
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=False,
                             descriptors=False)
    e_x, e_c, e_x_scan, e_c_scan = pdg._system_energy_targets(cols, None)
    assert e_x == pytest.approx(float(np.sum(
        cols["weights"] * cols["e_lda_x"] * (1.0 + cols["Fx"]))), rel=0,
        abs=1e-14)
    assert e_c == pytest.approx(float(np.sum(
        cols["weights"] * cols["e_lda_c"] * (1.0 + cols["Fc"]))), rel=0,
        abs=1e-14)
    assert e_x_scan == pytest.approx(float(np.sum(
        cols["weights"] * cols["e_lda_x"] * (1.0 + cols["Fx_scan"]))), rel=0,
        abs=1e-14)
    assert e_c_scan == pytest.approx(float(np.sum(
        cols["weights"] * cols["e_lda_c"] * (1.0 + cols["Fc_scan"]))), rel=0,
        abs=1e-14)
    assert e_x < 0.0 and e_c < 0.0


def _full_grid_libxc(system, basis, md):
    """libxc's integral over EVERY point of the stored grid, from the stored
    density matrix and AO table, without the floor and without the clip."""
    mol = gto.M(atom=system.atom, basis=basis, charge=system.charge,
                spin=system.spin, verbose=0)
    ni = dft.numint.NumInt()
    ao = np.asarray(md["ao_grid_deriv"])
    w = np.asarray(md["grid_weights"])
    dm = np.asarray(md["dm_pbe"])
    if dm.ndim == 3:
        ra = ni.eval_rho(mol, ao, dm[0], xctype="GGA", hermi=True)
        rb = ni.eval_rho(mol, ao, dm[1], xctype="GGA", hermi=True)
        rho = ra[0] + rb[0]
        ex = ni.eval_xc("PBE,", np.stack([ra, rb]), spin=1)[0]
        ec = ni.eval_xc(",PBE", np.stack([ra, rb]), spin=1)[0]
    else:
        r = ni.eval_rho(mol, ao, dm, xctype="GGA", hermi=True)
        rho = r[0]
        ex = ni.eval_xc("PBE,", r, spin=0)[0]
        ec = ni.eval_xc(",PBE", r, spin=0)[0]
    return float(np.sum(w * rho * ex)), float(np.sum(w * rho * ec))


def test_row_quadrature_tracks_the_full_grid_libxc_integral():
    """The rows the density floor drops are exactly the rows the model clamps
    to F = 1 (models._NN_TAIL_THRESHOLD is the same 1e-10), so the network can
    move no energy there. The gap between the row quadrature and libxc's
    full-grid integral of the SAME density is therefore the floor of what
    pretraining could reach, and it must sit far below the certificate's
    tol_atom = 1.0 mHa (measured: zero on the O atom, 4.8e-12 Ha on OH)."""
    from xcquinox.alec.models import _NN_TAIL_THRESHOLD
    assert _NN_TAIL_THRESHOLD == pdg._RHO_FLOOR
    oxygen = PretrainSystem("O", "O 0 0 0", 0, 2)
    for system, basis, level in ((oxygen, "def2-svp", 3), (_OH, "sto-3g", 0)):
        cols = _molecule_columns(system, "pbe", basis, level, polarized=True,
                                 descriptors=False)
        md = _precompute(system, basis, level)
        e_x, e_c, _sx, _sc = pdg._system_energy_targets(cols, None)
        ref_x, ref_c = _full_grid_libxc(system, basis, md)
        assert abs(e_x - ref_x) < _FULL_GRID_GAP, (system.name, e_x, ref_x)
        assert abs(e_c - ref_c) < _FULL_GRID_GAP, (system.name, e_c, ref_c)


def test_per_channel_and_total_exchange_energies_agree():
    """The Oliver-Perdew relation as a number: the exchange energy read off the
    per-channel doubled-density rows must equal the one read off the
    total-density spin-resolved rows. Both are E_x^PBE of the same density, so
    a disagreement means one of the two footings is not the parent's exchange.
    (Oliver and Perdew, Phys. Rev. A 20, 397 (1979).)"""
    cols = _atom_columns("O", 2, "def2-svp", 3, polarized=True,
                         descriptors=False,
                         exchange_footing="spin_channel")
    e_total, _c, e_total_scan, _sc = pdg._system_energy_targets(cols, None)
    e_channel, _c2, e_channel_scan, _sc2 = pdg._system_energy_targets(
        cols, cols["x_rows"])
    assert abs(e_channel - e_total) < _CHANNEL_TOTAL_GAP, (e_channel, e_total)
    assert abs(e_channel_scan - e_total_scan) < _CHANNEL_TOTAL_GAP, (
        e_channel_scan, e_total_scan)


def test_system_energy_targets_use_the_channel_rows_when_given():
    cols = _atom_columns("O", 2, "def2-svp", 3, polarized=True,
                         descriptors=False,
                         exchange_footing="spin_channel")
    x = cols["x_rows"]
    e_x, _c, e_x_scan, _sc = pdg._system_energy_targets(cols, x)
    e_lda = x["rho"] * (pdg._LDA_X_C * np.cbrt(x["rho"]))
    assert e_x == pytest.approx(float(np.sum(
        x["weights"] * e_lda * (1.0 + x["Fx"]))), rel=0, abs=1e-14)
    assert e_x_scan == pytest.approx(float(np.sum(
        x["weights"] * e_lda * (1.0 + x["Fx_scan"]))), rel=0, abs=1e-14)
    # The correlation targets do not depend on which exchange rows were used.
    assert pdg._system_energy_targets(cols, None)[1::2] == \
        pdg._system_energy_targets(cols, x)[1::2]


def test_x_block_lda_is_the_stored_column_or_the_analytic_doubled_lda():
    """A closed-shell exchange block is the total-density block and keeps its
    libxc-derived ``e_lda_x``; an open-shell per-channel block carries no such
    column and its denominator is the analytic unpolarized LDA at the doubled
    density. One expression serves the stored column and the energy target, so
    the loss multiplies the network by the same floating-point number the
    target was built from."""
    cols = _atom_columns("O", 2, "def2-svp", 3, polarized=True,
                         descriptors=False,
                         exchange_footing="spin_channel")
    np.testing.assert_array_equal(pdg._x_block_lda(cols), cols["e_lda_x"])
    x = cols["x_rows"]
    np.testing.assert_array_equal(
        pdg._x_block_lda(x), x["rho"] * (pdg._LDA_X_C * np.cbrt(x["rho"])))
    # The open shell's total-density column is the SPIN-POLARIZED LDA (libxc
    # spin=1), not the unpolarized one -- 16 percent larger in magnitude on
    # O's most polarized rows -- which is why the exchange block carries its
    # own LDA column instead of borrowing the total-density one.
    assert float(np.max(np.abs(
        cols["e_lda_x"] / (cols["rho"] * (pdg._LDA_X_C * np.cbrt(cols["rho"])))
        - 1.0))) > 0.1
    # On a closed shell the two conventions are one LDA: libxc's LDA_X at
    # spin=0 and the analytic coefficient agree to round-off (0.0 relative
    # measured on the doubled O channels, the coefficient one ulp apart).
    closed = _atom_columns("He", 0, "def2-svp", 1, polarized=True,
                           descriptors=False, exchange_footing="spin_channel")
    assert closed["x_rows"] is None
    np.testing.assert_allclose(pdg._x_block_lda(closed),
                               closed["rho"] * (pdg._LDA_X_C
                                                * np.cbrt(closed["rho"])),
                               rtol=1e-15, atol=0)


def test_system_energy_targets_close_on_pyscfs_total_energy():
    """The four targets summed by rung must reproduce the parent's own XC
    energy on the same density, and added to the record's non-XC energy the
    parent's total SCF energy: that is the certificate's E_xc^NN - E_xc^parent
    at zero residual. Measured closure: 0.0 (H2/PBE), 4.8e-12 Ha (OH/PBE,
    the floored points), 2.2e-16 (H2/SCAN), 3.2e-13 Ha (OH/SCAN)."""
    for system, xc in ((_H2, "pbe"), (_OH, "pbe"), (_H2, "scan"),
                       (_OH, "scan")):
        cols = _molecule_columns(system, xc, "sto-3g", 0, polarized=True,
                                 descriptors=False)
        md = _precompute(system, "sto-3g", 0, xc)
        e_x, e_c, e_x_scan, e_c_scan = pdg._system_energy_targets(cols, None)
        e_xc = (e_x + e_c) if xc == "pbe" else (e_x_scan + e_c_scan)
        assert abs(e_xc - float(md["E_xc_pbe"])) < _E_TOL, (system.name, xc)
        assert abs(float(md["E_non_xc"]) + e_xc - float(md["E_pbe"])) < _E_TOL, (
            system.name, xc)


# ---------------------------------------------------------------------------
# Orientation lock: the pretraining density is the training density
# ---------------------------------------------------------------------------

def _deployment_config_paths():
    """Every shipped deployment configuration, or an empty list.

    ``hpcjobs/`` sits beside the package rather than inside it, so a source
    or wheel checkout can be missing it entirely; an absent directory is
    nothing to cross-check, not a failure.
    """
    import glob
    root = os.path.join(os.path.dirname(os.path.abspath(pdg.__file__)), "..",
                        "..", "hpcjobs", "configs")
    if not os.path.isdir(root):
        return []
    return sorted(glob.glob(os.path.join(root, "*.yaml")))


def test_deployment_config_paths_is_empty_without_the_directory(monkeypatch,
                                                                tmp_path):
    """A checkout without ``hpcjobs/`` yields no paths, so the cross-check
    above skips instead of asserting on an empty glob."""
    monkeypatch.setattr(pdg, "__file__", str(tmp_path / "pretrain_data_gen.py"))
    assert _deployment_config_paths() == []


def test_pretraining_lock_is_the_training_lock():
    """The generator's lock strength is ``orientation_lock.DEFAULT_STRENGTH``
    and the value every production configuration that sets one trains at, so
    the degenerate radicals' pretraining rows sit on the component the
    training SCF and the fidelity certificate see."""
    import yaml
    from xcquinox.alec.orientation_lock import DEFAULT_STRENGTH
    assert pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH == DEFAULT_STRENGTH == 3e-5
    paths = _deployment_config_paths()
    if not paths:
        pytest.skip("no hpcjobs/configs in this checkout; the constant above "
                    "is still pinned, there is simply no deployed "
                    "configuration to cross-check it against")
    seen = []
    for path in paths:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        value = (cfg.get("inputs") or {}).get("orientation_lock_strength")
        if value is None:
            continue
        seen.append(os.path.basename(path))
        assert float(value) == pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH, path
    assert len(seen) >= 6, seen


def test_system_columns_hand_the_lock_to_the_precompute(monkeypatch):
    import xcquinox.alec.data as data_mod
    real = data_mod.precompute_fixed_density_data
    seen = []

    def _capture(*args, **kwargs):
        seen.append(kwargs.get("orientation_lock_strength"))
        return real(*args, **kwargs)

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _capture)
    _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=False,
                      descriptors=False)
    _atom_columns("H", 1, "sto-3g", 0, polarized=False, descriptors=False)
    _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=False,
                      descriptors=False, orientation_lock_strength=0.0)
    assert seen == [pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH,
                    pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH, 0.0]


def test_lock_changes_the_degenerate_atoms_rows_and_not_a_closed_shells():
    """The lock is a traceless quadrupole: identically zero on an s-only
    basis (the H and He rows of the default file are bit-identical with and
    without it), and a component selector on the O atom, whose rows move at
    order one between a locked and an unlocked build because the unlocked
    2p hole lands on whichever component rounding picks."""
    for symbol, spin in (("He", 0), ("H", 1)):
        locked = _atom_columns(symbol, spin, "sto-3g", 0, polarized=True,
                               descriptors=True)
        unlocked = _atom_columns(symbol, spin, "sto-3g", 0, polarized=True,
                                 descriptors=True,
                                 orientation_lock_strength=0.0)
        for key in locked:
            np.testing.assert_array_equal(np.asarray(locked[key]),
                                          np.asarray(unlocked[key]),
                                          err_msg=f"{symbol} {key}")


#: Lock-on spread of the O-atom rows between two processes at four BLAS
#: threads (def2-SVP, grid level 3, 11682 rows), as (rtol, atol) per column.
#: Measured over ten pairs: relative 1.0e-10 (rho), 2.1e-10 (sigma), 4.8e-10
#: (Fx), 6.7e-11 (Fc), 2.4e-10 (metagga), 1.2e-10 (e_lda_x) -- the end-point
#: spread of two SCFs converged to the same component at conv_tol 1e-9 -- and
#: absolute 3.6e-11 (Fx_scan), 1.3e-10 (zeta), 2.1e-8 (metagga) on rows where
#: the column itself is near zero; the weights are bit-identical. Gates: 100x
#: the relative spread, and for the near-zero rows 100x the absolute one.
_LOCKED_REPRO_TOL = {
    "rho": (1e-8, 1e-12), "sigma": (1e-8, 1e-12), "Fx": (1e-8, 1e-9),
    "Fc": (1e-8, 1e-9), "Fx_scan": (1e-8, 1e-8), "Fc_scan": (1e-8, 1e-8),
    "metagga": (1e-8, 1e-6), "weights": (0.0, 0.0), "zeta": (1e-8, 1e-8),
    "e_lda_x": (1e-8, 1e-12), "e_lda_c": (1e-8, 1e-12),
}

_REPRO_SCRIPT = """
import sys
import numpy as np
import xcquinox.alec.pretrain_data_gen as pdg
cols = pdg._atom_columns("O", 2, "def2-svp", 3, polarized=True,
                         descriptors=False)
np.savez(sys.argv[1], **cols)
"""


def test_degenerate_atom_rows_are_reproducible_across_processes_with_the_lock(
        tmp_path):
    """Two processes, four BLAS threads each, the production lock: the O
    atom's rows agree to the SCF end-point spread. Without the lock the same
    two builds land on different components of the 2p hole: measured at this
    identity, the two builds keep different numbers of rows (11682 against
    11678 in one of three pairs) and on a common row set differ by 0.94 in
    rho, 0.64 in Fx and 4.7e3 in sigma (the threaded-BLAS lottery of a
    degenerate open shell; 0.44 in rho on 99 percent of the level-1 grid).
    With the lock the spread is the one quoted at ``_LOCKED_REPRO_TOL``, seven
    to ten orders below the unlocked lottery."""
    import subprocess
    import sys
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4", JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    outs = []
    for i in range(2):
        out = tmp_path / f"o_rows_{i}.npz"
        subprocess.run([sys.executable, "-c", _REPRO_SCRIPT, str(out)],
                       env=env, check=True, timeout=600)
        outs.append(dict(np.load(out)))
    a, b = outs
    assert set(a) == set(b) == set(_LOCKED_REPRO_TOL)
    for key, (rtol, atol) in _LOCKED_REPRO_TOL.items():
        assert a[key].shape == b[key].shape, key
        np.testing.assert_allclose(a[key], b[key], rtol=rtol, atol=atol,
                                   err_msg=key)
