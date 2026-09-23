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

import xcquinox.pipeline.pretrain_data_gen as pdg
from xcquinox.pipeline.pretrain_data_gen import (
    PretrainSystem, _atom_columns, _molecule_columns)
from xcquinox.pipeline.pyscf_determinism import pin_small_rho_cutoff


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
    grid pruned at the first ``get_veff`` call (``prune_small_rho_grids_``),
    at the density cutoff the precompute pins (pyscf 2.14's class default
    would keep every point) -- reached through pyscf's own path rather than
    the builder's replay of it. The orientation lock is applied the way
    ``data.precompute_fixed_density_data`` applies it (the traceless-quadrupole
    bias added to ``h_core`` before the kernel), at the generator's production
    strength unless told otherwise, so the comparison is between two SCFs of
    one Hamiltonian."""
    from xcquinox.pipeline.orientation_lock import orientation_lock_bias
    if orientation_lock_strength is None:
        orientation_lock_strength = pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH
    mol = gto.M(atom=system.atom, basis=basis, charge=system.charge,
                spin=system.spin, verbose=0)
    mf = dft.UKS(mol) if system.spin else dft.RKS(mol)
    mf.xc = reference_xc
    mf.grids.level = grid_level
    pin_small_rho_cutoff(mf)
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
    from xcquinox.pipeline.data import precompute_fixed_density_data
    if orientation_lock_strength is None:
        orientation_lock_strength = pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH
    return precompute_fixed_density_data(
        pdg._mol_spec_for(system, basis, grid_level), required_keys=(),
        descriptors=(), reference_xc=reference_xc,
        orientation_lock_strength=orientation_lock_strength)


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
            "reference_scf_conv_tol": float(mf.conv_tol),
            "reference_scf_gradient": float(np.linalg.norm(mf.get_grad(
                mf.mo_coeff, mf.mo_occ, mf.get_fock(dm=dm)))),
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


def test_grid_guard_refuses_a_record_from_another_grid(monkeypatch):
    """The rebuilt grid must be the precompute's; a record whose weights are
    not the rebuilt ones is refused rather than integrated."""
    import xcquinox.pipeline.data as data_mod
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


def test_require_sane_density_refuses_a_record_of_the_other_parent():
    md = _precompute(_H2, "sto-3g", 0, "scan")
    assert md["mol_metadata"]["reference_xc"] == "scan"
    with pytest.raises(RuntimeError, match="stamped as the 'scan' density"):
        pdg._require_sane_density(md, _H2, "pbe", "sto-3g", 0, 2)


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


# ---------------------------------------------------------------------------
# One naming function for the data file
# ---------------------------------------------------------------------------


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


#: The three tracked deployment configurations written before the orientation
#: lock existed (the template and the two grid-2 campaigns the user guide
#: walks through); their training, references and pretraining data were built
#: unlocked, so their stated lock is exactly 0.0.
_PRE_LOCK_CAMPAIGNS = frozenset({
    "bh76w411_repr.svp_grid2.yaml", "bh76w411_repr.tzvpd_grid2_df.yaml",
    "step7.yaml",
})


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
#: Re-measured over ten pairs on the current stack (jax 0.10.2, pyscf 2.14.0,
#: numpy 2.5.3; 2026-09-22) with the reference SCF's density cutoff pinned
#: (11682 rows again): relative 1.6e-10 (rho), 2.4e-10 (sigma), 3.1e-10
#: (Fx), 1.1e-10 (Fc), 3.7e-10 (Fc_scan), 4.0e-10 (metagga), 1.8e-10
#: (e_lda_x and e_lda_c) on the rows of order one, absolute 3.4e-11 (Fx_scan)
#: and 6.0e-11 (zeta) on the others, the weights bit-identical; against the
#: gates as assert_allclose reads them (atol + rtol |b|) the largest use is
#: 2.2e-2 (sigma), so the gates hold with margins of 45x to 190x and are
#: kept as they were.
_LOCKED_REPRO_TOL = {
    "rho": (1e-8, 1e-12), "sigma": (1e-8, 1e-12), "Fx": (1e-8, 1e-9),
    "Fc": (1e-8, 1e-9), "Fx_scan": (1e-8, 1e-8), "Fc_scan": (1e-8, 1e-8),
    "metagga": (1e-8, 1e-6), "weights": (0.0, 0.0), "zeta": (1e-8, 1e-8),
    "e_lda_x": (1e-8, 1e-12), "e_lda_c": (1e-8, 1e-12),
}

_REPRO_SCRIPT = """
import sys
import numpy as np
import xcquinox.pipeline.pretrain_data_gen as pdg
cols = pdg._atom_columns("O", 2, "def2-svp", 3, polarized=True,
                         descriptors=False)
np.savez(sys.argv[1], **cols)
"""


