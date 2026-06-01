"""Generate a pretrain-data ``.npz`` for xcquinox.alec network pretraining.

This is the canonical, importable version of the recipe the step-4/5/6 notebooks
emit inline: for each pretraining atom, run a PBE SCF on a coarse grid and store
the per-grid-point exchange/correlation enhancement targets
``Fx = F_x^PBE/F_x^LDA - 1`` and ``Fc = F_c^PBE/F_c^LDA - 1`` (stored as ``F - 1``,
the network convention), with spin-RESOLVED libxc ``spin=1`` evaluation for
open-shell atoms (PBE 1996 §III spin-scaling — the ``spin=0`` total-density call
is wrong for open-shell).

The SPIN-POLARIZED variant additionally writes a ``zeta_all`` column
(ζ = (ρ_a − ρ_b)/ρ per grid point) so a spin-polarization-aware cnet
(``use_polarized_correlation``) is pretrained on the real ζ rather than a ζ=0
warm-start. ``run_pretrain`` auto-selects ``pretrain_data_polarized.npz`` for a
polarized architecture (see ``pretrain._pretrain_data_filename``).

Descriptor columns ``cusp_all`` / ``dm_all`` are included by default so the file
works for descriptor architectures (deep_cusp / deep_dm / deep_combined …); a
no-descriptor arch simply ignores them.
"""
from __future__ import annotations

import json
import os

import numpy as np
import jax.numpy as jnp
from pyscf import gto, dft

import xcquinox.features as _features
from xcquinox.alec.df_jk import default_auxbasis


# Same pretraining atoms / basis / grid as the step-6 notebook generator.
# (symbol, PySCF 2S spin): H, O, N are open-shell (UKS); He is closed-shell.
DEFAULT_PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("O", 2), ("N", 3))
DEFAULT_BASIS = "def2-svp"
DEFAULT_GRID_LEVEL = 1
_RHO_FLOOR = 1e-10  # strict > threshold for kept grid points


def _atom_columns(symbol, spin, basis, grid_level, *, polarized, descriptors,
                  density_fit=False, cusp_log_transform=True):
    """Per-atom pretrain columns. Returns a dict of equal-length 1-D arrays
    (rho, sigma, Fx, Fc, weights[, zeta][, cusp (N,2)][, dm (N,D)]).

    ``density_fit`` density-fits the Coulomb build of the per-atom PBE SCF
    (auxbasis from :func:`df_jk.default_auxbasis`). The Fx/Fc targets are a
    property of the converged density, so DF only changes them within DF error;
    it is wired here so the pretrain data can be regenerated at a large basis
    without the full ERI blowing up RAM (negligible cost for single atoms, but
    keeps the whole pipeline on one Coulomb backend)."""
    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, charge=0, spin=spin, verbose=0)
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=default_auxbasis(basis))
    mf.xc = "pbe"
    mf.grids.level = grid_level
    mf.kernel()

    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
    dm_ab = mf.make_rdm1()
    is_uks = (dm_ab.ndim == 3)

    if is_uks:
        # Spin-resolve and call libxc with spin=1 (UKS) for correct open-shell
        # Fx/Fc targets. The spin=0 total-density call is wrong for open shells.
        dm_total = dm_ab[0] + dm_ab[1]
        rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
        rho_b_gga = mf._numint.eval_rho(mol, ao, dm_ab[1], xctype="GGA", hermi=True)
        rho_gga_uks = np.stack([rho_a_gga, rho_b_gga], axis=0)
        rho_a, rho_b = rho_a_gga[0], rho_b_gga[0]
        rho = rho_a + rho_b
        nabla_total = rho_a_gga[1:4] + rho_b_gga[1:4]
        sigma = (nabla_total ** 2).sum(axis=0)
        zeta = (rho_a - rho_b) / np.maximum(rho, 1e-300)
        ex_pbe = mf._numint.eval_xc("PBE,", rho_gga_uks, spin=1)[0]
        ec_pbe = mf._numint.eval_xc(",PBE", rho_gga_uks, spin=1)[0]
        ex_lda = mf._numint.eval_xc("LDA_X,", (rho_a, rho_b), spin=1)[0]
        ec_lda = mf._numint.eval_xc(",LDA_C_PW", (rho_a, rho_b), spin=1)[0]
    else:
        dm_total = dm_ab
        rho_gga = mf._numint.eval_rho(mol, ao, dm_total, xctype="GGA", hermi=True)
        rho = rho_gga[0]
        sigma = rho_gga[1] ** 2 + rho_gga[2] ** 2 + rho_gga[3] ** 2
        zeta = np.zeros_like(rho)
        ex_pbe = mf._numint.eval_xc("PBE,", rho_gga, spin=0)[0]
        ec_pbe = mf._numint.eval_xc(",PBE", rho_gga, spin=0)[0]
        ex_lda = mf._numint.eval_xc("LDA_X,", rho, spin=0)[0]
        ec_lda = mf._numint.eval_xc(",LDA_C_PW", rho, spin=0)[0]

    ex_safe = np.where(np.abs(ex_lda) > 1e-12, ex_lda, 1e-12)
    ec_safe = np.where(np.abs(ec_lda) > 1e-12, ec_lda, 1e-12)
    fx = np.clip(ex_pbe / ex_safe - 1.0, -5.0, 5.0)
    fc = np.clip(ec_pbe / ec_safe - 1.0, -5.0, 5.0)

    valid = rho > _RHO_FLOOR
    cols = {
        "rho": rho[valid],
        "sigma": sigma[valid],
        "Fx": fx[valid],
        "Fc": fc[valid],
        "weights": np.asarray(mf.grids.weights)[valid],
    }
    if polarized:
        cols["zeta"] = zeta[valid]
    if descriptors:
        coords_v = mf.grids.coords[valid]
        # Match training: every cusp-using arch sets descriptor_log_transform=
        # True, and data.py computes the training cusp with that flag. The raw
        # default (False) saturates near nuclei, so a False pretrain cusp would
        # feed the network a different feature distribution than training does.
        cusp = _features.compute_cusp_descriptor(
            jnp.asarray(coords_v),
            jnp.asarray(mol.atom_coords()),
            jnp.asarray(mol.atom_charges()),
            log_transform=cusp_log_transform,
        )
        cols["cusp"] = np.asarray(cusp)
        # UKS: pass spin-resolved DM (3-D) so the UKS branch is used.
        dm_for_features = jnp.asarray(dm_ab) if is_uks else jnp.asarray(dm_total)
        dm_global = _features.compute_dm_features_array(
            dm_for_features, jnp.asarray(mol.intor("int1e_ovlp")))
        cols["dm"] = np.tile(np.asarray(dm_global), (len(cols["rho"]), 1))
    return cols


def _pretrain_manifest_path(npz_path):
    """Sidecar manifest path for a pretrain-data ``.npz`` (``<npz>.manifest.json``)."""
    return str(npz_path) + ".manifest.json"


def _write_pretrain_manifest(npz_path, *, basis, grid_level, density_fit):
    """Record the basis/grid_level/density_fit a pretrain ``.npz`` was built at.

    Written as a sidecar so the ``.npz`` array payload stays byte-identical to the
    pre-manifest format (legacy loaders that ignore the sidecar are unaffected)."""
    meta = {"basis": basis, "grid_level": int(grid_level),
            "density_fit": bool(density_fit)}
    with open(_pretrain_manifest_path(npz_path), "w") as f:
        json.dump(meta, f)


def read_pretrain_manifest(npz_path):
    """Return the pretrain-data manifest dict, or ``None`` if absent."""
    mpath = _pretrain_manifest_path(npz_path)
    if not os.path.isfile(mpath):
        return None
    with open(mpath) as f:
        return json.load(f)


def pretrain_data_is_current(npz_path, *, basis, grid_level):
    """True iff ``npz_path`` exists AND its manifest's basis+grid_level match.

    A missing file OR a missing/mismatched manifest returns ``False`` so the
    harness regenerates rather than silently reusing data built at a different
    basis (the stale-reuse bug Task 9 closes). Legacy manifest-less files
    therefore regenerate once, then carry a manifest thereafter."""
    if not os.path.isfile(npz_path):
        return False
    meta = read_pretrain_manifest(npz_path)
    if meta is None:
        return False
    return (meta.get("basis") == basis
            and int(meta.get("grid_level", -1)) == int(grid_level))


def ensure_pretrain_data(data_dir, *, atoms=DEFAULT_PRETRAIN_ATOMS,
                         basis=DEFAULT_BASIS, grid_level=DEFAULT_GRID_LEVEL,
                         polarized=True, descriptors=True, density_fit=False,
                         cusp_log_transform=True):
    """Skip-if-current driver for staged pretrain data.

    Returns the canonical ``.npz`` path, (re)generating it ONLY when the file is
    absent or its manifest's basis/grid_level differs from the requested pair.
    Idempotent — a second call at the same basis is a no-op. Used by the cluster
    harness so a basis change forces a regen instead of training on stale data."""
    fname = "pretrain_data_polarized.npz" if polarized else "pretrain_data.npz"
    out_path = os.path.join(data_dir, fname)
    if pretrain_data_is_current(out_path, basis=basis, grid_level=grid_level):
        return out_path
    return generate_pretrain_data_npz(
        data_dir, atoms=atoms, basis=basis, grid_level=grid_level,
        polarized=polarized, descriptors=descriptors, density_fit=density_fit,
        cusp_log_transform=cusp_log_transform)


def generate_pretrain_data_npz(out_dir, *, atoms=DEFAULT_PRETRAIN_ATOMS,
                               basis=DEFAULT_BASIS, grid_level=DEFAULT_GRID_LEVEL,
                               polarized=True, descriptors=True,
                               density_fit=False, cusp_log_transform=True):
    """Generate the pretrain-data ``.npz`` in ``out_dir`` and return its path.

    ``polarized=True`` writes ``pretrain_data_polarized.npz`` with a ``zeta_all``
    column (the spin-polarized run's data); ``polarized=False`` writes
    ``pretrain_data.npz`` (the unpolarized data). Both carry the same
    spin-resolved Fx/Fc targets and the same molecules — they differ only by the
    presence of ``zeta_all``.

    ``density_fit`` density-fits the per-atom SCF Coulomb build (so the data can
    be regenerated at a large basis without the full ERI exhausting RAM). A
    sidecar ``<npz>.manifest.json`` records the basis/grid_level/density_fit so
    :func:`pretrain_data_is_current` can detect a basis change and force a regen."""
    per_atom = [
        _atom_columns(sym, spin, basis, grid_level,
                      polarized=polarized, descriptors=descriptors,
                      density_fit=density_fit,
                      cusp_log_transform=cusp_log_transform)
        for sym, spin in atoms
    ]
    save_kwargs = {
        "rho_all": np.concatenate([c["rho"] for c in per_atom]),
        "sigma_all": np.concatenate([c["sigma"] for c in per_atom]),
        "Fx_all": np.concatenate([c["Fx"] for c in per_atom]),
        "Fc_all": np.concatenate([c["Fc"] for c in per_atom]),
        "weights_all": np.concatenate([c["weights"] for c in per_atom]),
    }
    if polarized:
        save_kwargs["zeta_all"] = np.concatenate([c["zeta"] for c in per_atom])
    if descriptors:
        save_kwargs["cusp_all"] = np.concatenate([c["cusp"] for c in per_atom])
        save_kwargs["dm_all"] = np.concatenate([c["dm"] for c in per_atom])

    os.makedirs(out_dir, exist_ok=True)
    fname = "pretrain_data_polarized.npz" if polarized else "pretrain_data.npz"
    out_path = os.path.join(out_dir, fname)
    np.savez(out_path, **save_kwargs)
    _write_pretrain_manifest(out_path, basis=basis, grid_level=grid_level,
                             density_fit=density_fit)
    return out_path
