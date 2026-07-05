"""Generate a pretrain-data ``.npz`` for xcquinox.alec network pretraining.

This is the canonical, importable version of the recipe the step-4/5/6 notebooks
emit inline: for each pretraining atom, run a PBE SCF on a coarse grid and store
the per-grid-point exchange/correlation enhancement targets
``Fx = F_x^PBE - 1`` and ``Fc = F_c^PBE - 1`` (stored as ``F - 1``,
the network convention), with spin-RESOLVED libxc ``spin=1`` evaluation for
open-shell atoms (PBE 1996 §III spin-scaling, the ``spin=0`` total-density call
is wrong for open-shell).

The SPIN-POLARIZED variant additionally writes a ``zeta_all`` column
(ζ = (ρ_a - ρ_b)/ρ per grid point) so a spin-polarization-aware cnet
(``use_polarized_correlation``) is pretrained on the real ζ rather than a ζ=0
warm-start. ``run_pretrain`` auto-selects ``pretrain_data_polarized.npz`` for a
polarized architecture (see ``pretrain._pretrain_data_filename``).

Descriptor columns ``cusp_all`` / ``dm_all`` are included by default so the file
works for descriptor architectures (deep_cusp / deep_dm / deep_combined ...); a
no-descriptor arch ignores them.
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
                  density_fit=False, auxbasis=None, cusp_log_transform=True):
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
        aux = auxbasis if auxbasis is not None else default_auxbasis(basis)
        mf = mf.density_fit(auxbasis=aux)
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

    # Meta-GGA (SCAN) pretrain targets + iso-orbital alpha column, computed
    # unconditionally so the shared pretrain data always supports meta_gga archs (a
    # GGA cannot be pretrained to SCAN -- SCAN is alpha-dependent). tau comes from
    # the deriv=1 AO gradients + DM (metagga.py); SCAN reads a [rho, grad, lapl, tau]
    # MGGA row with lapl=0 (SCAN ignores the laplacian).
    from xcquinox.alec.metagga import compute_tau_from_dm, compute_alpha
    _ag = jnp.asarray(ao[1:4])
    if is_uks:
        tau_a = np.asarray(compute_tau_from_dm(_ag, jnp.asarray(dm_ab[0])))
        tau_b = np.asarray(compute_tau_from_dm(_ag, jnp.asarray(dm_ab[1])))
        _lapl = np.zeros_like(rho_a)
        mgga_a = np.vstack([rho_a_gga, _lapl, tau_a])
        mgga_b = np.vstack([rho_b_gga, _lapl, tau_b])
        ex_scan = mf._numint.eval_xc("SCAN,", (mgga_a, mgga_b), spin=1)[0]
        ec_scan = mf._numint.eval_xc(",SCAN", (mgga_a, mgga_b), spin=1)[0]
        tau_tot = tau_a + tau_b
    else:
        tau_tot = np.asarray(compute_tau_from_dm(_ag, jnp.asarray(dm_total)))
        _lapl = np.zeros_like(rho)
        mgga = np.vstack([rho_gga, _lapl, tau_tot])
        ex_scan = mf._numint.eval_xc("SCAN,", mgga, spin=0)[0]
        ec_scan = mf._numint.eval_xc(",SCAN", mgga, spin=0)[0]
    fx_scan = np.clip(ex_scan / ex_safe - 1.0, -5.0, 5.0)
    fc_scan = np.clip(ec_scan / ec_safe - 1.0, -5.0, 5.0)
    alpha_col = np.asarray(compute_alpha(
        jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(tau_tot)))

    valid = rho > _RHO_FLOOR
    cols = {
        "rho": rho[valid],
        "sigma": sigma[valid],
        "Fx": fx[valid],
        "Fc": fc[valid],
        "Fx_scan": fx_scan[valid],
        "Fc_scan": fc_scan[valid],
        "metagga": alpha_col[valid].reshape(-1, 1),
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
        # Rung-3.5 per-spin local occupancy n_sigma = A^T P A on the valid grid,
        # mirroring the training-side computation in data.py so a rung35-descriptor
        # arch has its pretrain column (otherwise _assemble_pretrain_descriptors
        # KeyErrors). A is the density-independent projected-AO overlap; the
        # occupancy is linear in the PBE DM and bounded [0, 1]. Uses the default
        # alpha the rung35 archs are built with.
        from xcquinox.alec.rung35 import (
            compute_projected_ao, compute_rung35_occupancy, DEFAULT_RUNG35_ALPHA)
        proj_ao = compute_projected_ao(mol, coords_v, DEFAULT_RUNG35_ALPHA)
        rung35_feat = compute_rung35_occupancy(jnp.asarray(proj_ao), dm_for_features)
        cols["rung35"] = np.asarray(rung35_feat)
    return cols


def _pretrain_manifest_path(npz_path):
    """Sidecar manifest path for a pretrain-data ``.npz`` (``<npz>.manifest.json``)."""
    return str(npz_path) + ".manifest.json"


def _write_pretrain_manifest(npz_path, *, basis, grid_level, density_fit,
                             auxbasis=None, atoms=DEFAULT_PRETRAIN_ATOMS):
    """Record the basis/grid_level/density_fit/auxbasis/atoms a pretrain
    ``.npz`` was built at.

    Written as a sidecar so the ``.npz`` array payload stays byte-identical to the
    pre-manifest format (legacy loaders that ignore the sidecar are unaffected).
    ``auxbasis`` is the EFFECTIVE DF fitting basis (``None`` when density_fit is
    off) so a fitting-basis change forces a regen. ``atoms`` is recorded so an
    ATOM-SET change (e.g. extending pretraining coverage to every pool element)
    also forces a regen -- previously the manifest keyed only basis+grid and a
    species change silently reused stale data."""
    meta = {"basis": basis, "grid_level": int(grid_level),
            "density_fit": bool(density_fit), "auxbasis": auxbasis,
            "atoms": [[str(s), int(sp)] for s, sp in atoms]}
    with open(_pretrain_manifest_path(npz_path), "w") as f:
        json.dump(meta, f)


def read_pretrain_manifest(npz_path):
    """Return the pretrain-data manifest dict, or ``None`` if absent."""
    mpath = _pretrain_manifest_path(npz_path)
    if not os.path.isfile(mpath):
        return None
    with open(mpath) as f:
        return json.load(f)


def pretrain_data_is_current(npz_path, *, basis, grid_level, auxbasis=None,
                             atoms=DEFAULT_PRETRAIN_ATOMS):
    """True iff ``npz_path`` exists AND its manifest's
    basis+grid_level+auxbasis+atoms match.

    A missing file OR a missing/mismatched manifest returns ``False`` so the
    harness regenerates rather than silently reusing data built at a different
    basis (the stale-reuse bug Task 9 closes). Legacy manifest-less files
    therefore regenerate once, then carry a manifest thereafter. ``auxbasis`` is
    the EFFECTIVE DF fitting basis (``None`` when DF is off); a legacy manifest
    without an ``auxbasis`` key reads as ``None``, so the full-ERI path stays
    current without a spurious regen. A legacy manifest without an ``atoms``
    key reads as the historical DEFAULT_PRETRAIN_ATOMS, so existing default
    data stays current while any non-default atom set forces a regen."""
    if not os.path.isfile(npz_path):
        return False
    meta = read_pretrain_manifest(npz_path)
    if meta is None:
        return False
    want_atoms = [[str(s), int(sp)] for s, sp in atoms]
    have_atoms = meta.get(
        "atoms", [[str(s), int(sp)] for s, sp in DEFAULT_PRETRAIN_ATOMS])
    manifest_ok = (meta.get("basis") == basis
                   and int(meta.get("grid_level", -1)) == int(grid_level)
                   and meta.get("auxbasis") == auxbasis
                   and have_atoms == want_atoms)
    if not manifest_ok:
        return False
    # A descriptor-bearing file written before rung-3.5 support lacks the
    # ``rung35_all`` column; the manifest matches but ``run_pretrain`` would
    # KeyError on a rung35 arch. Treat such a file as stale so it regenerates.
    try:
        with np.load(npz_path) as _z:
            _keys = set(_z.files)
    except Exception:
        return False
    if "cusp_all" in _keys and "rung35_all" not in _keys:
        return False
    # A real pretrain-data file (has Fx_all) written before meta-GGA support lacks
    # the SCAN targets + metagga alpha column; a meta_gga arch would KeyError. Force
    # a regen so the columns appear. Gated on Fx_all so bare stub files (manifest-
    # only tests) are not spuriously flagged.
    if "Fx_all" in _keys and (
            "metagga_all" not in _keys or "Fx_scan_all" not in _keys):
        return False
    return True


def _effective_auxbasis(basis, density_fit, auxbasis):
    """Resolve the DF fitting basis actually used: explicit ``auxbasis`` if given,
    else :func:`df_jk.default_auxbasis(basis)`; ``None`` when DF is off."""
    if not density_fit:
        return None
    return auxbasis if auxbasis is not None else default_auxbasis(basis)


def ensure_pretrain_data(data_dir, *, atoms=DEFAULT_PRETRAIN_ATOMS,
                         basis=DEFAULT_BASIS, grid_level=DEFAULT_GRID_LEVEL,
                         polarized=True, descriptors=True, density_fit=False,
                         auxbasis=None, cusp_log_transform=True, progress=False):
    """Skip-if-current driver for staged pretrain data.

    Returns the canonical ``.npz`` path, (re)generating it ONLY when the file is
    absent or its manifest's basis/grid_level/auxbasis differs from the requested
    values. Idempotent, a second call at the same settings is a no-op. Used by
    the cluster harness so a basis OR fitting-basis change forces a regen instead
    of training on stale data."""
    eff_aux = _effective_auxbasis(basis, density_fit, auxbasis)
    fname = "pretrain_data_polarized.npz" if polarized else "pretrain_data.npz"
    out_path = os.path.join(data_dir, fname)
    if pretrain_data_is_current(out_path, basis=basis, grid_level=grid_level,
                                auxbasis=eff_aux, atoms=atoms):
        return out_path
    return generate_pretrain_data_npz(
        data_dir, atoms=atoms, basis=basis, grid_level=grid_level,
        polarized=polarized, descriptors=descriptors, density_fit=density_fit,
        auxbasis=auxbasis, cusp_log_transform=cusp_log_transform, progress=progress)


def generate_pretrain_data_npz(out_dir, *, atoms=DEFAULT_PRETRAIN_ATOMS,
                               basis=DEFAULT_BASIS, grid_level=DEFAULT_GRID_LEVEL,
                               polarized=True, descriptors=True,
                               density_fit=False, auxbasis=None,
                               cusp_log_transform=True, progress=False):
    """Generate the pretrain-data ``.npz`` in ``out_dir`` and return its path.

    ``polarized=True`` writes ``pretrain_data_polarized.npz`` with a ``zeta_all``
    column (the spin-polarized run's data); ``polarized=False`` writes
    ``pretrain_data.npz`` (the unpolarized data). Both carry the same
    spin-resolved Fx/Fc targets and the same molecules, they differ only by the
    presence of ``zeta_all``.

    ``density_fit`` density-fits the per-atom SCF Coulomb build (so the data can
    be regenerated at a large basis without the full ERI exhausting RAM). A
    sidecar ``<npz>.manifest.json`` records the basis/grid_level/density_fit so
    :func:`pretrain_data_is_current` can detect a basis change and force a regen."""
    per_atom = []
    for _i, (sym, spin) in enumerate(atoms, 1):
        if progress:
            print(f"  pretrain data: atom {_i}/{len(atoms)} {sym} (PBE SCF @ {basis}) ...",
                  flush=True)
        per_atom.append(_atom_columns(
            sym, spin, basis, grid_level,
            polarized=polarized, descriptors=descriptors,
            density_fit=density_fit, auxbasis=auxbasis,
            cusp_log_transform=cusp_log_transform))
    save_kwargs = {
        "rho_all": np.concatenate([c["rho"] for c in per_atom]),
        "sigma_all": np.concatenate([c["sigma"] for c in per_atom]),
        "Fx_all": np.concatenate([c["Fx"] for c in per_atom]),
        "Fc_all": np.concatenate([c["Fc"] for c in per_atom]),
        # SCAN (meta-GGA) targets + iso-orbital alpha column, always present so
        # meta_gga archs pretrain to SCAN (pretrain.py routes the target by the
        # arch's meta_gga flag); GGA archs ignore these keys.
        "Fx_scan_all": np.concatenate([c["Fx_scan"] for c in per_atom]),
        "Fc_scan_all": np.concatenate([c["Fc_scan"] for c in per_atom]),
        "metagga_all": np.concatenate([c["metagga"] for c in per_atom]),
        "weights_all": np.concatenate([c["weights"] for c in per_atom]),
    }
    if polarized:
        save_kwargs["zeta_all"] = np.concatenate([c["zeta"] for c in per_atom])
    if descriptors:
        save_kwargs["cusp_all"] = np.concatenate([c["cusp"] for c in per_atom])
        save_kwargs["dm_all"] = np.concatenate([c["dm"] for c in per_atom])
        save_kwargs["rung35_all"] = np.concatenate([c["rung35"] for c in per_atom])

    os.makedirs(out_dir, exist_ok=True)
    fname = "pretrain_data_polarized.npz" if polarized else "pretrain_data.npz"
    out_path = os.path.join(out_dir, fname)
    np.savez(out_path, **save_kwargs)
    _write_pretrain_manifest(
        out_path, basis=basis, grid_level=grid_level, density_fit=density_fit,
        auxbasis=_effective_auxbasis(basis, density_fit, auxbasis),
        atoms=atoms)
    return out_path
