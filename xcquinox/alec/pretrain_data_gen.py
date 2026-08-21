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

DFS parity deviation (SI Sec. III)
----------------------------------
DFS pretrains on its training molecules evaluated on their molecular grids,
augmented (SI Sec. III) for the EXCHANGE network by a regular 2-D
``(s, alpha)`` parameter grid at fixed density, ~10100 nodes, equal weight
per point. The recipe here samples the ``Fx`` / ``Fc`` targets on the atomic
grids of single atoms (``DEFAULT_PRETRAIN_ATOMS``; production configs override
with the pool's element set) from per-atom PBE SCFs -- the 21-molecule
molecular grids are not reproduced -- and, since 2026-08-10, augments
meta-GGA pretrains with the parameter-space mesh defined below. The mesh is a
deliberate EXTENSION of the SI's: 3-D ``(r_s, s, alpha)`` rather than 2-D at
fixed density, covering correlation as well as exchange, at a stated flat 30%
loss-weight share per channel rather than equal weight per node (see
LOSS_PRIMER Sec. 8 for the deviation row). The atomic warm-start plus mesh is
subsequently refined by self-consistent training on the molecular pool, so
the remaining deviation affects only the pretraining seed.
"""
from __future__ import annotations

import json
import os
from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from pyscf import gto, dft, scf

import xcquinox.features as _features
from xcquinox.alec.df_jk import default_auxbasis


# Same pretraining atoms / basis / grid as the step-6 notebook generator.
# (symbol, PySCF 2S spin): H, O, N are open-shell (UKS); He is closed-shell.
# NOTE: these four atoms are a DFS-parity deviation -- DFS SI Sec. III pretrains
# on the 21 training molecules' molecular grids plus a regular (s, alpha)
# parameter grid; see the module docstring for the full deviation note.
DEFAULT_PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("O", 2), ("N", 3))
DEFAULT_BASIS = "def2-svp"
DEFAULT_GRID_LEVEL = 1
_RHO_FLOOR = 1e-10  # strict > threshold for kept grid points

#: LDA exchange coefficient, ``eps_x^LDA(rho) = _LDA_X_C rho^(1/3)``. The same
#: constant libxc's ``LDA_X,`` returns at spin=0 and the same one
#: :func:`spin_channel_exchange_rows` divides by, kept here so the per-system
#: energy targets and the stored enhancement factors share one denominator.
_LDA_X_C = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)

#: One pretraining system: a geometry, a charge and a PySCF 2S spin. ``atom`` is
#: a PySCF geometry string in Angstrom. Free atoms are spelled
#: ``"<Sym> 0 0 0"`` so a pool atom and a ``pretrain.atoms`` entry for the same
#: element are one system rather than two.
PretrainSystem = namedtuple("PretrainSystem", ("name", "atom", "charge", "spin"))


def _system_name(symbol, charge):
    """Canonical system name for a free atom: the symbol, with the ion charge
    appended as a run of ``+`` or ``-`` (``F-``, ``Cl-``). Names are labels for
    provenance and for the validation split's record; the physics is carried by
    (geometry, charge, spin)."""
    if charge == 0:
        return str(symbol)
    sign = "+" if charge > 0 else "-"
    return f"{symbol}{sign * abs(int(charge))}"


def _geometry_key(atom_str):
    """Canonical hashable geometry: ``(symbol, x, y, z)`` per nucleus, rounded
    to 1e-8 angstrom and sorted.

    Two spellings of the same structure ("H 0 0 0" and "H 0.0 0.0 0.0", or two
    orderings of the same nuclei) collapse to one key, so the DFS inventory and
    the pool inventory deduplicate against each other and against an explicit
    ``pretrain.atoms`` entry without depending on how each source spells its
    geometry.
    """
    items = []
    for chunk in str(atom_str).replace("\n", ";").split(";"):
        parts = chunk.split()
        if not parts:
            continue
        if len(parts) != 4:
            raise ValueError(
                f"malformed PySCF geometry chunk {chunk!r} in {atom_str!r}: "
                "expected '<symbol> <x> <y> <z>' per nucleus."
            )
        items.append((parts[0].capitalize(), round(float(parts[1]), 8),
                      round(float(parts[2]), 8), round(float(parts[3]), 8)))
    if not items:
        raise ValueError(f"empty PySCF geometry {atom_str!r}.")
    return tuple(sorted(items))


def _n_atoms(atom_str):
    """Number of nuclei in a PySCF geometry string."""
    return len(_geometry_key(atom_str))


def _composition_from_atom(atom_str):
    """``((symbol, count), ...)``, sorted, for a PySCF geometry string.

    Derived from the geometry rather than trusted from a record, so a pool
    entry, a DFS entry and a ``(symbol, 2S)`` pair produce the same composition
    for the same molecule and therefore the same MoleculeSpec.
    """
    counts = {}
    for symbol, _x, _y, _z in _geometry_key(atom_str):
        counts[symbol] = counts.get(symbol, 0) + 1
    return tuple(sorted(counts.items()))


def normalize_system(obj):
    """Coerce a pretraining-system descriptor into a :class:`PretrainSystem`.

    Accepts a ``PretrainSystem``; a mapping carrying ``name``/``atom``/
    ``charge``/``spin`` (the schema of the committed pool JSON and of
    ``dfs_pretrain_set.dfs_pretrain_records``, whose extra ``kind``,
    ``atom_composition`` and ``g2_97_index`` entries are ignored); any object
    exposing those four attributes (``config.MoleculeSpec``); or a
    ``(symbol, 2S)`` pair, the historical ``pretrain.atoms`` form, which names a
    neutral free atom at the origin. Keeping the coercion in one place is what
    lets the set be assembled from three inventories written independently.
    """
    if isinstance(obj, PretrainSystem):
        return obj
    if isinstance(obj, dict):
        return PretrainSystem(name=str(obj["name"]), atom=str(obj["atom"]),
                              charge=int(obj.get("charge", 0)),
                              spin=int(obj.get("spin", 0)))
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        symbol, spin = obj
        return PretrainSystem(name=str(symbol), atom=f"{symbol} 0 0 0",
                              charge=0, spin=int(spin))
    if all(hasattr(obj, a) for a in ("name", "atom", "charge", "spin")):
        return PretrainSystem(name=str(obj.name), atom=str(obj.atom),
                              charge=int(obj.charge), spin=int(obj.spin))
    raise TypeError(
        f"cannot read {obj!r} as a pretraining system: expected a "
        "PretrainSystem, a mapping with name/atom/charge/spin, an object with "
        "those attributes, or a (symbol, 2S) pair."
    )


def _mol_spec_for(system, basis, grid_level):
    """The :class:`~xcquinox.alec.config.MoleculeSpec` for one pretraining
    system at the run's identity.

    This is the spec :func:`data.precompute_fixed_density_data` receives, so the
    pretraining rows and the training features of the same molecule are built
    from the same object.
    """
    from xcquinox.alec.config import MoleculeSpec
    system = normalize_system(system)
    return MoleculeSpec(
        name=system.name, atom=system.atom, basis=basis,
        charge=int(system.charge), spin=int(system.spin),
        atom_composition=_composition_from_atom(system.atom),
        grid_level=grid_level)


def pool_atom_systems():
    """Every single-atom species of the BH76 and W4-11 pools, de-duplicated.

    Fourteen distinct (symbol, charge, 2S) triples: the twelve neutral elements
    the two pools span -- Al, B, Be, C, Cl, F, H, N, O, P, S, Si, with their
    Hund's-rule ground-state spins -- plus the two closed-shell anions F- and
    Cl-, which are reactants of BH76 barrier heights and therefore systems the
    Section 3.3 certificate bounds at tol_atom. All of them are free atoms at
    the origin, so the geometry is the pool's own.

    Read from the committed pool JSON through ``full_benchmark_pools`` rather
    than transcribed, so a pool edit propagates here.
    """
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    mol_specs, _reactions = load_full_held_out_pools()
    seen = {}
    for ms in mol_specs.values():
        composition = dict(ms.atom_composition)
        if sum(composition.values()) != 1:
            continue
        seen.setdefault((str(next(iter(composition))), int(ms.charge),
                         int(ms.spin)), None)
    return tuple(
        PretrainSystem(name=_system_name(symbol, charge),
                       atom=f"{symbol} 0 0 0", charge=charge, spin=spin)
        for symbol, charge, spin in sorted(seen)
    )


def dfs_level_for_reference_xc(reference_xc):
    """Which DFS pretraining inventory a parent density's file uses.

    The DFS notebook ships two variants (spec Section 6): the GGA one with 22
    G2/97 molecules and the meta-GGA one with 20, the difference being H2 and N2
    (``dfs_pretrain_set.MGGA_EXCLUDED``). The parent functional and the
    inventory are the same rung choice, so one maps onto the other.
    """
    if reference_xc == "pbe":
        return "gga"
    if reference_xc == "scan":
        return "mgga"
    raise ValueError(
        f"reference_xc must be 'pbe' or 'scan'; got {reference_xc!r}.")


def resolve_parent_density(arch, parent_density):
    """The ``reference_xc`` whose self-consistent density ``arch`` pretrains on.

    ``pretrain.parent_density`` is the YAML knob; this is where its value becomes
    the ``reference_xc`` keyword of
    :func:`data.precompute_fixed_density_data`. ``"pbe"`` / ``"scan"`` pass
    through. ``"auto"`` is the rung baseline: SCAN for the meta-GGA rung, PBE
    otherwise (spec Section 1, "PBE for GGA-rung architectures, SCAN for
    meta-GGA architectures"). That is the map ``rungs.seed_xc_for_arch`` applies
    under its production ``"mgga_scan"`` policy, computed from the architecture
    OBJECT rather than a registry name so an architecture built ad hoc resolves
    too; the agreement with ``seed_xc_for_arch`` over the whole registry is
    pinned by test.

    The meta-GGA rung is read the way ``rungs.arch_ingredients`` reads it -- the
    ``meta_gga`` flag OR a ``"metagga"`` descriptor. The guard in
    ``ArchitectureConfig.from_spec`` rejects only the other direction (the flag
    without the descriptor); the dataclass constructor accepts either alone,
    so an architecture assembled outside ``from_spec`` can carry the descriptor
    alone,
    and resolving that one to PBE would pretrain a meta-GGA network on a density
    its own SCF never visits.
    """
    if parent_density in ("pbe", "scan"):
        return parent_density
    if parent_density != "auto":
        raise ValueError(
            "parent_density must be 'pbe', 'scan' or 'auto'; got "
            f"{parent_density!r}."
        )
    descriptor_names = {getattr(d, "name", None)
                        for d in getattr(arch, "descriptors", ())}
    has_meta_gga = (bool(getattr(arch, "meta_gga", False))
                    or "metagga" in descriptor_names)
    return "scan" if has_meta_gga else "pbe"


def _dfs_pretrain_records(level):
    """The DFS pretraining inventory for ``level`` ("gga" / "mgga").

    A named seam so the composition layer can be tested without the inventory
    and so an import failure names the module that supplies it. The RECORD form
    is read rather than ``dfs_pretrain_systems``: the composition layer is
    basis-free, and the basis and grid level are applied later by
    :func:`_mol_spec_for` at the run's own identity.
    """
    try:
        from xcquinox.alec.dfs_pretrain_set import dfs_pretrain_records
    except ImportError as exc:  # pragma: no cover - exercised when absent
        raise ImportError(
            "the DFS pretraining inventory lives in "
            "xcquinox.alec.dfs_pretrain_set (dfs_pretrain_records(level)); "
            "pretrain.dfs_set cannot be honored without it"
        ) from exc
    return dfs_pretrain_records(level)


def resolve_pretrain_systems(*, atoms=None, dfs_set=False, pool_atoms=False,
                             reference_xc="pbe"):
    """The ordered, de-duplicated pretraining set.

    Order is DFS inventory, then pool atoms, then the explicit ``atoms`` list,
    with the first occurrence of a (geometry, charge, spin) winning. ``atoms`` of
    ``None`` means the historical four-atom default when neither inventory is
    requested and NOTHING when one is: the set Section 7 binds is stated exactly
    ("the DFS pretraining set in its entirety, plus every atom of the BH76 /
    W4-11 pools"), and He belongs to neither.
    """
    if atoms is None:
        atoms = () if (dfs_set or pool_atoms) else DEFAULT_PRETRAIN_ATOMS
    ordered = []
    if dfs_set:
        ordered.extend(_dfs_pretrain_records(
            dfs_level_for_reference_xc(reference_xc)))
    if pool_atoms:
        ordered.extend(pool_atom_systems())
    ordered.extend(atoms)
    out = []
    seen = set()
    for entry in ordered:
        system = normalize_system(entry)
        key = (_geometry_key(system.atom), int(system.charge), int(system.spin))
        if key in seen:
            continue
        seen.add(key)
        out.append(system)
    return tuple(out)


def pretrain_data_filename(polarized, reference_xc="pbe"):
    """Canonical pretrain-data filename.

    ``reference_xc="pbe"`` reproduces the two historical names. The SCAN-density
    file carries its own suffix because it is built at a DIFFERENT
    self-consistent density (spec Section 6 deviation 1) and its rows are not
    interchangeable with the PBE file's.
    """
    base = "pretrain_data_polarized" if polarized else "pretrain_data"
    return (f"{base}.npz" if reference_xc == "pbe"
            else f"{base}_{reference_xc}.npz")


def spin_channel_exchange_rows(mol, mf, ao, dm_ab, *, descriptors=True,
                               cusp_log_transform=True, rho_floor=_RHO_FLOOR):
    """Open-shell exchange rows on the exact-spin-scaling footing.

    The production UKS exchange evaluates, per spin channel, the symmetric
    doubled density diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20,
    397 (1979)): density ``2 rho_sigma``, gradient invariant
    ``4 sigma_sigma_sigma``, kinetic-energy density ``2 tau_sigma``, and the
    descriptor features of that density matrix. Those are the inputs the network
    sees at SCF time on an open shell, so those are the inputs its exchange rows
    must be posed at, with the parent's SPIN-UNPOLARIZED enhancement factor at
    the same inputs as the target -- ``eval_xc(..., spin=0)`` on the doubled
    density, not the spin-polarized call on the physical one.

    Each row carries HALF the grid weight, because
    ``E_x = 1/2 (E_x[2 rho_a] + E_x[2 rho_b])``: summing
    ``w_row rho_row eps_x^LDA(rho_row) (1 + Fx_row)`` over both channels then
    reproduces the parent's open-shell exchange energy exactly.

    Returns 1-D columns (2-D for the descriptor blocks), alpha channel first
    then beta, with points at or below ``rho_floor`` in the DOUBLED density
    dropped. A
    channel with no electron (the beta channel of H) contributes no rows.
    Correlation is untouched: it is spin-interpolated rather than spin-scaled
    and keeps the total density with zeta.

    Parameters
    ----------
    mol, mf, ao : the converged parent calculation and its ``deriv=1`` AO
        values on ``mf.grids.coords``.
    dm_ab : array, shape (2, nao, nao). The parent's spin-resolved density
        matrix.
    """
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.alec.metagga import compute_alpha, compute_tau_from_dm
    from xcquinox.alec.rung35 import (
        DEFAULT_RUNG35_ALPHA, DEFAULT_RUNG35_MULTISHELL_ALPHAS,
        compute_projected_ao, compute_projected_ao_multishell,
        compute_rung35_multishell_occupancy, compute_rung35_occupancy)

    dm_j = jnp.asarray(dm_ab)
    ao_grad = jnp.asarray(ao[1:4])
    s_matrix = jnp.asarray(mol.intor("int1e_ovlp"))
    weights = np.asarray(mf.grids.weights)
    c_lda = _LDA_X_C

    names = ["rho", "sigma", "Fx", "Fx_scan", "metagga", "weights"]
    if descriptors:
        names += ["cusp", "dm", "rung35", "rung35ms"]
    parts = {k: [] for k in names}

    for s in (0, 1):
        dm_doubled = doubled_spin_dm(dm_j, s)
        rho_gga_s = mf._numint.eval_rho(mol, ao, np.asarray(dm_ab[s]),
                                        xctype="GGA", hermi=True)
        rho_d = 2.0 * rho_gga_s[0]
        grad_d = 2.0 * rho_gga_s[1:4]
        sigma_d = (grad_d ** 2).sum(axis=0)
        tau_d = np.asarray(compute_tau_from_dm(ao_grad, dm_doubled))
        keep = rho_d > rho_floor
        if not bool(keep.any()):
            continue
        gga_row = np.vstack([rho_d, grad_d])
        mgga_row = np.vstack([gga_row, np.zeros_like(rho_d), tau_d])
        # The parent's SPIN-UNPOLARIZED enhancement at the doubled inputs: this
        # is exactly what the spin-scaling relation asks the functional for.
        ex_pbe = mf._numint.eval_xc("PBE,", gga_row, spin=0)[0]
        ex_scan = mf._numint.eval_xc("SCAN,", mgga_row, spin=0)[0]
        ex_lda = c_lda * np.cbrt(np.clip(rho_d, 1e-300, None))
        ex_safe = np.where(np.abs(ex_lda) > 1e-12, ex_lda, 1e-12)
        parts["rho"].append(rho_d[keep])
        parts["sigma"].append(sigma_d[keep])
        parts["Fx"].append(np.clip(ex_pbe / ex_safe - 1.0, -5.0, 5.0)[keep])
        parts["Fx_scan"].append(
            np.clip(ex_scan / ex_safe - 1.0, -5.0, 5.0)[keep])
        parts["metagga"].append(np.asarray(compute_alpha(
            jnp.asarray(rho_d), jnp.asarray(sigma_d),
            jnp.asarray(tau_d)))[keep].reshape(-1, 1))
        # Half the grid weight per channel: E_x = 1/2 (E_x[2 rho_a] + E_x[2 rho_b]).
        parts["weights"].append(0.5 * weights[keep])
        if descriptors:
            coords_v = mf.grids.coords[keep]
            parts["cusp"].append(np.asarray(_features.compute_cusp_descriptor(
                jnp.asarray(coords_v),
                jnp.asarray(mol.atom_coords()),
                jnp.asarray(mol.atom_charges()),
                log_transform=cusp_log_transform,
            )))
            dm_global = _features.compute_dm_features_array(dm_doubled, s_matrix)
            parts["dm"].append(np.tile(np.asarray(dm_global),
                                       (int(keep.sum()), 1)))
            proj = compute_projected_ao(mol, coords_v, DEFAULT_RUNG35_ALPHA)
            parts["rung35"].append(np.asarray(compute_rung35_occupancy(
                jnp.asarray(proj), dm_doubled)))
            proj_ms = compute_projected_ao_multishell(
                mol, coords_v, DEFAULT_RUNG35_MULTISHELL_ALPHAS)
            parts["rung35ms"].append(np.asarray(
                compute_rung35_multishell_occupancy(jnp.asarray(proj_ms),
                                                    dm_doubled)))

    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def _scf_gradient_norm(mol_data):
    """pyscf's SCF orbital-gradient norm, rebuilt from a stored density record.

    ``precompute_fixed_density_data`` keeps the pieces of the Fock matrix at the
    density it returns -- ``h_core``, ``j_matrix`` (per spin for UKS),
    ``vxc_pbe`` -- together with ``dm_pbe`` and ``s_matrix``, and a
    self-consistent density is the one whose Fock matrix commutes with
    ``P S``. In the orthonormal MO basis ``C`` (``C^T S C = 1``) the commutator
    ``F P S - S P F`` has the occupied-virtual block ``n_occ F_ai`` and its
    negative transpose, so ``||S^-1/2 (F P S - S P F) S^-1/2||_F / sqrt(2)`` is
    exactly the norm pyscf's ``scf.hf.get_grad`` / ``scf.uhf.get_grad`` return
    (the factor of 2 in the restricted gradient is the occupation; the two
    unrestricted channels add in quadrature). ``S^-1/2`` stands in for the MO
    coefficients the record does not carry, the Frobenius norm being invariant
    under the orthogonal map between the two bases. Measured against
    ``np.linalg.norm(mf.get_grad(...))`` on H2O and OH, converged and stopped
    after one cycle: <= 6e-8 relative wherever the norm is above round-off.
    """
    s_matrix = np.asarray(mol_data["s_matrix"])
    dm = np.asarray(mol_data["dm_pbe"])
    h_core = np.asarray(mol_data["h_core"])
    j_matrix = np.asarray(mol_data["j_matrix"])
    vxc = np.asarray(mol_data["vxc_pbe"])
    evals, evecs = np.linalg.eigh(s_matrix)
    s_mhalf = (evecs / np.sqrt(evals)) @ evecs.T
    if dm.ndim == 3:
        j_total = j_matrix.sum(axis=0)
        pairs = ((h_core + j_total + vxc[0], dm[0]),
                 (h_core + j_total + vxc[1], dm[1]))
    else:
        pairs = ((h_core + j_matrix + vxc, dm),)
    total = 0.0
    for fock, dm_s in pairs:
        comm = fock @ dm_s @ s_matrix - s_matrix @ dm_s @ fock
        comm = s_mhalf @ comm @ s_mhalf
        total += float(np.sum(comm * comm))
    return float(np.sqrt(total / 2.0))


#: Electron-count tolerance, per electron, for :func:`_require_sane_density`.
#: The quadrature error of the integrated density was measured at <= 5.3e-6 per
#: electron on level-1 grids (H2O/def2-SVP, the worst of H, He, N, O, F-, H2O,
#: NH3) and at 1.3e-3 per electron on the level-0 grid of H2/STO-3G, the
#: coarsest grid the tests use; the defects the check exists for (a lost
#: electron, a density matrix on another molecule's grid) are O(1).
_N_ELECTRON_TOL = 1e-2


def _require_sane_density(mol_data, system, reference_xc, basis, grid_level,
                          n_electrons):
    """Raise unless the parent density is a converged density on this grid.

    The Fx / Fc targets and the per-system energies are properties of the
    CONVERGED parent density; an unconverged one is not a functional's density
    at all and enters the fit as noise no later stage can tell from a fit error.
    Three tests, none needing more than the record the precompute always
    returns:

    * an ``scf_converged`` flag of ``False`` is honored when a record carries
      one (the installed precompute records none);
    * the quadrature of the stored density against the electron count, which
      catches a grid too coarse to resolve a diffuse density and a density
      matrix that does not belong to the stored grid -- but NOT a stalled SCF,
      whose density matrix still integrates to N electrons;
    * the SCF orbital gradient rebuilt from the stored Fock pieces
      (:func:`_scf_gradient_norm`), held to pyscf's own convergence criterion
      ``conv_tol_grad = sqrt(conv_tol)`` (``pyscf/scf/hf.py``; 3.2e-5 at the
      default ``conv_tol`` of 1e-9). Measured on converged records: <= 4.2e-6
      (O/def2-SVP level 1); an SCF stopped after one cycle sits at 2e-3 (He)
      to 1 (F-), and an oxygen-atom SCAN run pyscf reported unconverged at
      6.7e-5. The energy-change half of pyscf's criterion needs the iteration
      history, which the record does not carry.
    """
    if mol_data.get("scf_converged") is False:
        raise RuntimeError(
            f"the {reference_xc} SCF for pretraining system {system.name!r} "
            f"(geometry {system.atom!r}, charge {system.charge}, 2S "
            f"{system.spin}, basis {basis}, grid level {grid_level}) did not "
            "converge"
        )
    rho = np.asarray(mol_data["rho_grid"])
    weights = np.asarray(mol_data["grid_weights"])
    n_grid = float(np.sum(weights * rho))
    tol = _N_ELECTRON_TOL * max(1.0, float(n_electrons))
    if not abs(n_grid - float(n_electrons)) < tol:
        raise RuntimeError(
            f"the {reference_xc} density of pretraining system "
            f"{system.name!r} integrates to {n_grid:.6f} electrons on its own "
            f"grid, against {n_electrons} expected (basis {basis}, grid level "
            f"{grid_level}); the grid does not resolve this density or the "
            "density matrix does not belong to it"
        )
    grad_norm = _scf_gradient_norm(mol_data)
    grad_tol = float(np.sqrt(scf.hf.SCF.conv_tol))
    if not grad_norm < grad_tol:
        raise RuntimeError(
            f"the {reference_xc} SCF for pretraining system {system.name!r} "
            f"(geometry {system.atom!r}, charge {system.charge}, 2S "
            f"{system.spin}, basis {basis}, grid level {grid_level}) did not "
            f"converge: the orbital gradient of its stored density is "
            f"{grad_norm:.3e}, against pyscf's criterion {grad_tol:.1e}"
        )


def _system_columns(system, basis, grid_level, *, reference_xc, polarized,
                    descriptors, density_fit=False, auxbasis=None,
                    cusp_log_transform=True, exchange_footing="total"):
    """Pretrain columns for ONE system on the parent functional's own density.

    The general case of :func:`_atom_columns`: an arbitrary geometry, charge and
    spin, and a parent functional that is PBE (the GGA rung's baseline) or SCAN
    (the meta-GGA rung's). The density is NOT computed here: it comes from
    ``data.precompute_fixed_density_data(mol_spec, reference_xc=...)``, the one
    place this library produces a frozen parent density. Training builds its
    features from that function's output and the Section 3.3 certificate
    measures ``E_xc^NN - E_xc^parent`` on it, so obtaining the pretraining rows
    the same way makes "the same density on the same grid" structural instead of
    a coincidence that has to be re-argued whenever the pipeline moves.

    Both the PBE and the SCAN enhancement targets are evaluated on whichever
    density the file was built at, exactly as the single-atom path has always
    done, so the column layout does not depend on the parent; the manifest
    records the ``reference_xc`` and ``run_pretrain`` refuses a file whose
    parent does not match the architecture's rung.

    Returns a dict of column arrays sharing one leading length (the descriptor
    blocks are 2-D; ``x_rows`` has its own row set): ``rho``, ``sigma``, ``Fx``,
    ``Fc``, ``Fx_scan``, ``Fc_scan``, ``metagga``, ``weights``, ``e_lda_x``,
    ``e_lda_c``, optionally ``zeta``, optionally ``cusp`` / ``dm`` / ``rung35``
    / ``rung35ms``, and under the ``spin_channel`` footing ``x_rows``. Points
    with a total density at or below ``_RHO_FLOOR`` are dropped.

    ``e_lda_x`` and ``e_lda_c`` are the LDA energy DENSITIES ``rho eps_x^LDA``
    and ``rho eps_c^PW92`` in the EXACT convention the ``Fx`` / ``Fc`` ratios
    were formed in (libxc ``spin=1`` for an open shell, ``spin=0`` for a closed
    one). Multiplying a stored enhancement factor by them returns Hartree per
    unit volume, which is what makes the per-system energy term integrate the
    same quantity the point-wise term fits: summed against ``weights`` they
    reproduce pyscf's own integrated exchange and correlation on the same
    density to <= 3.3e-11 Ha (the energy of the floored points; measured on N
    and H2O at def2-SVP level 1).

    ``exchange_footing`` selects how OPEN-SHELL exchange rows are posed.
    ``"total"`` is unchanged: one row per grid point at the total density with
    spin-resolved libxc targets. ``"spin_channel"`` additionally returns
    ``x_rows``, the per-channel rows of :func:`spin_channel_exchange_rows` --
    ``(2 rho_sigma, 4 sigma_sigma_sigma, features of diag(P_sigma, P_sigma))``
    with the parent's spin-unpolarized enhancement factor at those inputs as the
    target, which is what the exact spin scaling evaluates at SCF time (Oliver
    and Perdew, Phys. Rev. A 20, 397 (1979)). ``x_rows`` is ``None`` for a
    closed-shell system, whose total-density rows already are that footing.
    Correlation rows are untouched under either setting: correlation is
    spin-interpolated rather than spin-scaled and keeps the total density with
    zeta.

    ``density_fit`` is recorded in the manifest but no longer changes the parent
    SCF: the density is the precompute's, whose PBE / SCAN baseline is
    deliberately full-ERI so it is a fixed reference-quality anchor shared with
    training. ``auxbasis`` is forwarded for the same identity bookkeeping.
    """
    if reference_xc not in ("pbe", "scan"):
        raise ValueError(
            f"reference_xc must be 'pbe' or 'scan'; got {reference_xc!r}.")
    if exchange_footing not in ("total", "spin_channel"):
        raise ValueError(
            "exchange_footing must be 'total' or 'spin_channel'; got "
            f"{exchange_footing!r}."
        )
    system = normalize_system(system)
    from xcquinox.alec.data import precompute_fixed_density_data

    mol_spec = _mol_spec_for(system, basis, grid_level)
    # No descriptors and no reference keys are requested: the descriptor columns
    # below are built by the same calls the single-atom path has always used, so
    # an existing file's numbers do not move, and the precompute's own blocks
    # (which it would build at the same values) are not paid for twice.
    mol_data = precompute_fixed_density_data(
        mol_spec, required_keys=(), descriptors=(), auxbasis=auxbasis,
        reference_xc=reference_xc)

    mol = gto.M(atom=system.atom, basis=basis, charge=int(system.charge),
                spin=int(system.spin), verbose=0)
    # A mean field for its integration grid and its libxc handle ONLY: the
    # kernel is never run here. The record's grid is the one the SCF settled
    # on, and pyscf does not integrate on the bare Becke-Lebedev grid: at its
    # first ``get_veff`` call ``initialize_grids`` builds the grid and drops
    # the points where the INITIAL-GUESS density is negligible
    # (``prune_small_rho_grids_``: ``rho w <= small_rho_cutoff / n_points``,
    # 1e-7 by default; on H2/STO-3G level 0 this removes points, on a single
    # H it removes none). Replaying that method with the same initial guess
    # -- the same deterministic function of the geometry, the level and the
    # minao density -- reproduces the stored quadrature exactly, and the guard
    # refuses to continue if it does not, which turns "the same grid" from an
    # assumption into a check. The weights are compared exactly (one code path
    # builds both); the coordinates are pinned through the AO table the
    # precompute stored, which the same libcint kernels reproduce to round-off
    # against O(1) differences for a foreign grid.
    mf = dft.UKS(mol) if system.spin else dft.RKS(mol)
    if grid_level is not None:
        mf.grids.level = grid_level
    mf.initialize_grids(mol, mf.get_init_guess(mol, mf.init_guess,
                                               s1e=mf.get_ovlp(mol)))
    weights = np.asarray(mol_data["grid_weights"])
    coords = mf.grids.coords
    ao = np.asarray(mol_data["ao_grid_deriv"])
    same_grid = (
        np.asarray(mf.grids.weights).shape == weights.shape
        and np.array_equal(np.asarray(mf.grids.weights), weights)
        and np.allclose(mf._numint.eval_ao(mol, coords, deriv=0), ao[0],
                        rtol=0.0, atol=1e-10)
    )
    if not same_grid:
        raise RuntimeError(
            f"the rebuilt integration grid for pretraining system "
            f"{system.name!r} is not the one precompute_fixed_density_data "
            "used; the pretrain rows and the training features would be "
            "quadratures of different grids"
        )
    _require_sane_density(mol_data, system, reference_xc, basis, grid_level,
                          int(mol.nelectron))

    dm_ab = np.asarray(mol_data["dm_pbe"])
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
    # LDA energy densities in the SAME convention the ratios above were formed
    # in: ``ex_safe`` / ``ec_safe`` are the denominators the clips divided by,
    # so ``e_lda * (1 + F)`` returns the parent's energy density exactly
    # wherever the +-5 clip is inactive. These are the columns the per-system
    # energy term contracts with the quadrature weights.
    e_lda_x = rho * ex_safe
    e_lda_c = rho * ec_safe

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
        "weights": weights[valid],
        "e_lda_x": e_lda_x[valid],
        "e_lda_c": e_lda_c[valid],
    }
    if polarized:
        cols["zeta"] = zeta[valid]
    if descriptors:
        coords_v = coords[valid]
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
        # Multi-width twin at the descriptor's default widths, so a
        # rung35_multishell arch has its pretrain column. Column order matches
        # the descriptor exactly (alpha-major then spin), and
        # _assemble_pretrain_descriptors width-gates the result, so any
        # mismatch fails loudly rather than widening the network input.
        from xcquinox.alec.rung35 import (
            compute_projected_ao_multishell,
            compute_rung35_multishell_occupancy,
            DEFAULT_RUNG35_MULTISHELL_ALPHAS)
        proj_ao_ms = compute_projected_ao_multishell(
            mol, coords_v, DEFAULT_RUNG35_MULTISHELL_ALPHAS)
        cols["rung35ms"] = np.asarray(compute_rung35_multishell_occupancy(
            jnp.asarray(proj_ao_ms), dm_for_features))
    if exchange_footing == "spin_channel":
        cols["x_rows"] = (
            spin_channel_exchange_rows(
                mol, mf, ao, dm_ab, descriptors=bool(descriptors),
                cusp_log_transform=cusp_log_transform)
            if is_uks else None
        )
    return cols


def _atom_columns(symbol, spin, basis, grid_level, *, polarized, descriptors,
                  density_fit=False, auxbasis=None, cusp_log_transform=True,
                  exchange_footing="total"):
    """Per-atom pretrain columns: the single-nucleus case of
    :func:`_system_columns` on the PBE density.

    Kept as a named entry point because the historical pretraining set is a list
    of free atoms and because the atomic rows are the ones every pre-existing
    ``.npz`` was built from; the geometry spelling ``"<Sym> 0 0 0"`` is the one
    those files were generated with.
    """
    return _system_columns(
        PretrainSystem(name=str(symbol), atom=f"{symbol} 0 0 0", charge=0,
                       spin=int(spin)),
        basis, grid_level, reference_xc="pbe", polarized=polarized,
        descriptors=descriptors, density_fit=density_fit, auxbasis=auxbasis,
        cusp_log_transform=cusp_log_transform,
        exchange_footing=exchange_footing)


def _molecule_columns(mol_spec, reference_xc, basis, grid_level, *, polarized,
                      descriptors, density_fit=False, auxbasis=None,
                      cusp_log_transform=True, exchange_footing="total"):
    """Pretrain columns for one molecule of the set, on the parent's density.

    ``mol_spec`` is anything :func:`normalize_system` accepts: a
    ``PretrainSystem``, the mapping form the DFS inventory and the pool JSON
    use, or a ``config.MoleculeSpec``. The basis and grid level come from the
    run's production identity, not from the spec, so every system in a file
    shares one integration identity.
    """
    return _system_columns(
        mol_spec, basis, grid_level, reference_xc=reference_xc,
        polarized=polarized, descriptors=descriptors, density_fit=density_fit,
        auxbasis=auxbasis, cusp_log_transform=cusp_log_transform,
        exchange_footing=exchange_footing)


# --------------------------------------------------------------------------- #
# (s, alpha) parameter-space mesh -- the DFS-parity piece the atomic grids miss.
#
# WHY THIS EXISTS. PBE's F_c is a 2-D function of (r_s, s) and the atomic grids
# determine it: the GGA C-net reproduces PBE to <= 0.013 everywhere tested.
# SCAN's is 3-D in (r_s, s, alpha), and the SAME atomic data leaves the alpha
# axis underdetermined -- the meta-GGA C-net was measured at up to 0.457 from
# SCAN away from alpha=1 (it is exact AT alpha=1 only because the UEG gate pins
# it there by construction). The module docstring above predicted this: the
# deviation "is most consequential for the alpha-dependent meta-GGA / SCAN
# targets, whose alpha coordinate is undersampled". A regular mesh determines
# that axis directly instead of hoping the atomic grids happen to sample it.
#
# Analytic: (rho, sigma, tau) are synthesized to realize each (r_s, s, alpha)
# node and pushed through the SAME libxc calls the atomic path uses. No SCF.
# --------------------------------------------------------------------------- #

#: Mesh nodes. r_s spans core (0.1) to the diffuse tail (10); s spans the UEG
#: limit to the large-gradient tail; alpha spans single-orbital (0) through the
#: uniform gas (1) to the overlap region. Chosen to bracket the exchange-weighted
#: (r_s, s, alpha) region molecules actually occupy, measured on H2O/CH4.
MESH_RS = (0.1, 0.3, 0.7, 1.5, 3.0, 5.0, 10.0)
MESH_S = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
MESH_ALPHA = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
#: The mesh's share of the TOTAL integration weight. Stated and recorded in the
#: manifest rather than left emergent: the atomic rows carry physical quadrature
#: weights, the mesh rows carry none, so their relative influence on the
#: pretrain loss is a deliberate choice. 0.3 gives the alpha axis real pull
#: without letting a synthetic mesh outvote the physical densities.
MESH_WEIGHT_FRACTION = 0.3


def _mesh_columns(*, rs_grid=MESH_RS, s_grid=MESH_S, alpha_grid=MESH_ALPHA):
    """SCAN Fx/Fc on a regular ``(r_s, s, alpha)`` mesh.

    Returns the same column dict shape as :func:`_atom_columns` (minus the
    descriptor extras, which a mesh point has no geometry to define): ``rho``,
    ``sigma``, ``Fx_scan``, ``Fc_scan``, ``metagga``, ``weights`` (unnormalized;
    the caller rescales them to :data:`MESH_WEIGHT_FRACTION`).

    Each node is realized as a physical ``(rho, sigma, tau)`` triple --
    ``rho = 3/(4 pi r_s^3)``, ``sigma = (s * 2 k_F rho)^2``,
    ``tau = alpha * tau_unif + tau_W`` -- so the mesh's alpha column is produced
    by ``metagga.compute_alpha`` from the same triple the SCF would see. The
    targets come from the SAME ``eval_xc("SCAN,")`` / ``eval_xc(",SCAN")`` calls
    the atomic path uses, so mesh and atomic rows are the same quantity.
    """
    from pyscf import dft as _dft

    from xcquinox.alec.metagga import compute_alpha

    rs = np.asarray(rs_grid, dtype=float)[:, None, None]
    s = np.asarray(s_grid, dtype=float)[None, :, None]
    al = np.asarray(alpha_grid, dtype=float)[None, None, :]
    rs, s, al = np.broadcast_arrays(rs, s, al)
    rs, s, al = rs.ravel(), s.ravel(), al.ravel()

    rho = 3.0 / (4.0 * np.pi * rs ** 3)
    k_f = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    sigma = (s * 2.0 * k_f * rho) ** 2
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    tau = al * tau_unif + tau_w

    ni = _dft.numint.NumInt()
    grad = np.sqrt(sigma)
    zeros = np.zeros_like(rho)
    mgga = np.vstack([rho, grad, zeros, zeros, zeros, tau])
    gga = np.vstack([rho, grad, zeros, zeros])
    ex_scan = ni.eval_xc("SCAN,", mgga, spin=0)[0]
    ec_scan = ni.eval_xc(",SCAN", mgga, spin=0)[0]
    ex_lda = ni.eval_xc("LDA_X,", rho, spin=0)[0]
    ec_lda = ni.eval_xc(",LDA_C_PW", rho, spin=0)[0]
    ex_safe = np.where(np.abs(ex_lda) > 1e-12, ex_lda, 1e-12)
    ec_safe = np.where(np.abs(ec_lda) > 1e-12, ec_lda, 1e-12)
    del gga

    # alpha recomputed through the SHARED helper (not reused from `al`) so a
    # divergence between this mesh and the SCF descriptor would show up here.
    alpha_col = np.asarray(compute_alpha(
        jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(tau)))
    return {
        "rho": rho,
        "sigma": sigma,
        "Fx_scan": np.clip(ex_scan / ex_safe - 1.0, -5.0, 5.0),
        "Fc_scan": np.clip(ec_scan / ec_safe - 1.0, -5.0, 5.0),
        "metagga": alpha_col.reshape(-1, 1),
        "weights": np.ones_like(rho),
        "zeta": np.zeros_like(rho),
    }


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
            "atoms": [[str(s), int(sp)] for s, sp in atoms],
            # The (s, alpha) mesh the meta-GGA archs additionally pretrain on.
            # Recorded because its WEIGHT SHARE is a deliberate choice, not an
            # emergent property of a quadrature: mesh rows carry no physical
            # grid weight, so their pull on the pretrain loss is set here.
            "mesh": {"rs": list(MESH_RS), "s": list(MESH_S),
                     "alpha": list(MESH_ALPHA),
                     "weight_fraction": MESH_WEIGHT_FRACTION}}
    # Atomic for the same shared-dir reason as the npz write above.
    mpath = _pretrain_manifest_path(npz_path)
    tmp = f"{mpath}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, mpath)


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
    # Same argument one generation later: a file written before the multi-width
    # rung-3.5 support lacks ``rung35ms_all``.
    if "rung35_all" in _keys and "rung35ms_all" not in _keys:
        return False
    # A real pretrain-data file (has Fx_all) written before meta-GGA support lacks
    # the SCAN targets + metagga alpha column; a meta_gga arch would KeyError. Force
    # a regen so the columns appear. Gated on Fx_all so bare stub files (manifest-
    # only tests) are not spuriously flagged.
    if "Fx_all" in _keys and (
            "metagga_all" not in _keys or "Fx_scan_all" not in _keys):
        return False
    # A file written before the (s, alpha) parameter mesh lacks the *_mesh keys.
    # Without them a meta_gga arch pretrains on the atomic grids alone, which
    # leaves SCAN's alpha axis underdetermined -- measured at up to 0.457 error
    # in F_c away from alpha=1, against <= 0.013 for the GGA archs on the same
    # data. Force a regen so the mesh appears. Gated on Fx_all for the same
    # reason as above (bare manifest-only stubs are not flagged).
    if "Fx_all" in _keys and (
            "metagga_mesh" not in _keys or "Fx_scan_mesh" not in _keys):
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
    out_path = os.path.join(data_dir, pretrain_data_filename(polarized))
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
    # (s, alpha) parameter-space mesh, stored under SEPARATE *_mesh keys so the
    # atomic arrays every GGA arch reads stay byte-identical. pretrain.py
    # concatenates these ONLY for a meta_gga arch whose descriptor set the mesh
    # can actually define (see _mesh_columns).
    mesh = _mesh_columns()
    _w_atom = float(save_kwargs["weights_all"].sum())
    _n_mesh = mesh["rho"].shape[0]
    # Rescale the (weightless) mesh rows to a stated share of the total
    # integration weight: w_mesh_total / (w_atom + w_mesh_total) = FRACTION.
    _w_mesh_total = _w_atom * MESH_WEIGHT_FRACTION / (1.0 - MESH_WEIGHT_FRACTION)
    save_kwargs.update({
        "rho_mesh": mesh["rho"],
        "sigma_mesh": mesh["sigma"],
        "Fx_scan_mesh": mesh["Fx_scan"],
        "Fc_scan_mesh": mesh["Fc_scan"],
        "metagga_mesh": mesh["metagga"],
        "weights_mesh": np.full(_n_mesh, _w_mesh_total / _n_mesh),
    })
    if polarized:
        save_kwargs["zeta_mesh"] = mesh["zeta"]
    if polarized:
        save_kwargs["zeta_all"] = np.concatenate([c["zeta"] for c in per_atom])
    if descriptors:
        save_kwargs["cusp_all"] = np.concatenate([c["cusp"] for c in per_atom])
        save_kwargs["dm_all"] = np.concatenate([c["dm"] for c in per_atom])
        save_kwargs["rung35_all"] = np.concatenate([c["rung35"] for c in per_atom])
        save_kwargs["rung35ms_all"] = np.concatenate(
            [c["rung35ms"] for c in per_atom])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, pretrain_data_filename(polarized))
    # ATOMIC write (tmp + os.replace): the data dir is SHARED across sweep
    # runs, and two concurrently submitted runs whose datagen stages both see
    # a stale file would otherwise race a plain in-place np.savez -- a torn
    # zip that fails every reader. With the rename, concurrent regenerations
    # merely duplicate compute (last writer wins, both logically identical)
    # and a reader always sees a complete file. The tmp name is pid-tagged so
    # two writers do not collide on the tmp path either.
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    np.savez(tmp_path, **save_kwargs)
    # np.savez appends .npz to a name without it: normalize.
    if not tmp_path.endswith(".npz") and os.path.isfile(tmp_path + ".npz"):
        tmp_path = tmp_path + ".npz"
    os.replace(tmp_path, out_path)
    _write_pretrain_manifest(
        out_path, basis=basis, grid_level=grid_level, density_fit=density_fit,
        auxbasis=_effective_auxbasis(basis, density_fit, auxbasis),
        atoms=atoms)
    return out_path
