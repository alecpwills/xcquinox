"""Generate a pretrain-data ``.npz`` for xcquinox.alec network pretraining.

This is the canonical, importable version of the recipe the step-4/5/6 notebooks
emit inline: for each pretraining system -- a free atom or a molecule of the
set :func:`resolve_pretrain_systems` assembles -- take the parent functional's
self-consistent density from ``data.precompute_fixed_density_data`` (PBE for
the GGA rung, SCAN for the meta-GGA rung, at the training orientation lock) and
store the per-grid-point exchange/correlation enhancement targets
``Fx = F_x^PBE - 1`` and ``Fc = F_c^PBE - 1`` (stored as ``F - 1``,
the network convention) beside their SCAN counterparts, with spin-RESOLVED
libxc ``spin=1`` evaluation for open shells (PBE 1996 Sec. III spin-scaling,
the ``spin=0`` total-density call is wrong for an open shell). Under the
``spin_channel`` exchange footing the open-shell exchange rows are posed per
spin channel at the doubled density (the footing the production UKS exchange
evaluates) and written as a second row block; a per-row system index and a
per-system table of parent energies carry the energy term of the pretraining
objective. The schema is declared in :func:`pretrain_npz_keys` and closed in
both directions (:func:`_validate_pretrain_arrays`,
:func:`load_pretrain_data_npz`), and a sidecar manifest records the identity
the file was built at (:func:`pretrain_data_is_current`).

The SPIN-POLARIZED variant additionally writes a ``zeta_all`` column
(zeta = (rho_a - rho_b)/rho per grid point) so a spin-polarization-aware cnet
(``use_polarized_correlation``) is pretrained on the real zeta rather than a
zeta = 0 warm-start. ``run_pretrain`` auto-selects
``pretrain_data_polarized.npz`` for a polarized architecture (see
``pretrain._pretrain_data_filename``, which reads
:func:`pretrain_data_filename`).

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
import jax
import jax.numpy as jnp
from pyscf import gto, dft, scf

import xcquinox.features as _features
from xcquinox.alec.df_jk import default_auxbasis
from xcquinox.alec.orientation_lock import DEFAULT_STRENGTH as _LOCK_STRENGTH
from xcquinox.alec.metagga import ALPHA_DEFINITION as _ALPHA_DEFINITION

#: The indicator definition a manifest WITHOUT an ``alpha_definition`` key was
#: written under: every file predating the key carries the hard-clipped
#: indicator ``clip((tau - tau_W)/tau_unif, 0, 100)``.
_LEGACY_ALPHA_DEFINITION = "hard_clip"


# Same pretraining atoms / basis / grid as the step-6 notebook generator.
# (symbol, PySCF 2S spin): H, O, N are open-shell (UKS); He is closed-shell.
# NOTE: these four atoms are a DFS-parity deviation -- DFS SI Sec. III pretrains
# on the 21 training molecules' molecular grids plus a regular (s, alpha)
# parameter grid; see the module docstring for the full deviation note.
DEFAULT_PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("O", 2), ("N", 3))
DEFAULT_BASIS = "def2-svp"
DEFAULT_GRID_LEVEL = 1
_RHO_FLOOR = 1e-10  # strict > threshold for kept grid points

#: Orientation-lock strength the parent density is computed at: the coefficient
#: on the traceless-quadrupole ``h_core`` bias of ``orientation_lock.py`` that
#: makes a degenerate open shell's density reproducible (OH / NO / CH 2-Pi, the
#: 3P / 2P free atoms of the pools). It is ``orientation_lock.DEFAULT_STRENGTH``
#: and the ``inputs.orientation_lock_strength`` every production configuration
#: trains at (3e-5 in the six dfs6311 YAMLs that set one), so the rows a
#: network is fit on sit on the SAME component of the degenerate manifold the
#: training SCF and the fidelity certificate see. Without it the O-atom rows
#: of two four-thread builds differ by 0.44 in rho on 99 percent of the grid
#: (the threaded-BLAS lottery of the 2p hole); with it they agree to
#: round-off. Part of the data identity (``pretrain_data_is_current``).
PRETRAIN_ORIENTATION_LOCK_STRENGTH = _LOCK_STRENGTH

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
    Three tests, none needing more than the record the precompute returns:

    * the precompute's convergence stamp,
      ``mol_metadata["reference_scf_converged"]``, beside
      ``reference_scf_cycles`` and ``reference_scf_solver`` (``"diis"`` or
      ``"diis+newton"``). The precompute writes ``True`` in every record it
      returns and raises ``data.ReferenceSCFNotConverged`` otherwise, so a
      stamp of ``False`` or an ABSENT stamp (a record assembled elsewhere, or
      one predating the stamp) is refused: a density whose convergence was
      never asserted is not trusted. The stamp's ``reference_xc`` must also
      be the requested parent, so a record of the other functional cannot be
      integrated under this one's name;
    * the quadrature of the stored density against the electron count, which
      catches a grid too coarse to resolve a diffuse density and a density
      matrix that does not belong to the stored grid -- but NOT a stalled SCF,
      whose density matrix still integrates to N electrons;
    * the SCF orbital gradient rebuilt from the stored Fock pieces
      (:func:`_scf_gradient_norm`), the second line behind the stamp: held to
      pyscf's own convergence criterion ``conv_tol_grad = sqrt(conv_tol)``
      (``pyscf/scf/hf.py``; 3.2e-5 at the default ``conv_tol`` of 1e-9).
      Measured on converged records: <= 4.2e-6 (O/def2-SVP level 1); an SCF
      stopped after one cycle sits at 2e-3 (He) to 1 (F-), and an oxygen-atom
      SCAN run pyscf reported unconverged at 6.7e-5. The energy-change half of
      pyscf's criterion needs the iteration history, which the record does not
      carry.
    """
    where = (f"pretraining system {system.name!r} (geometry {system.atom!r}, "
             f"charge {system.charge}, 2S {system.spin}, basis {basis}, grid "
             f"level {grid_level})")
    stamp = mol_data.get("mol_metadata") or {}
    converged = stamp.get("reference_scf_converged")
    if converged is not True:
        detail = (f"{converged!r} after {stamp.get('reference_scf_cycles')} "
                  f"cycles ({stamp.get('reference_scf_solver')})"
                  if "reference_scf_converged" in stamp
                  else "absent: the record carries no convergence stamp")
        raise RuntimeError(
            f"the {reference_xc} SCF for {where} is not stamped converged: "
            f"mol_metadata['reference_scf_converged'] is {detail}; only a "
            "record the precompute returned as converged is integrated"
        )
    stamped_xc = stamp.get("reference_xc")
    if stamped_xc is not None and str(stamped_xc) != str(reference_xc):
        raise RuntimeError(
            f"the record for {where} is stamped as the {stamped_xc!r} "
            f"density, but the {reference_xc!r} parent was requested"
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
            f"the {reference_xc} SCF for {where} did not converge: the "
            f"orbital gradient of its stored density is {grad_norm:.3e}, "
            f"against pyscf's criterion {grad_tol:.1e}"
        )


def _system_columns(system, basis, grid_level, *, reference_xc, polarized,
                    descriptors, density_fit=False, auxbasis=None,
                    cusp_log_transform=True, exchange_footing="total",
                    orientation_lock_strength=PRETRAIN_ORIENTATION_LOCK_STRENGTH):
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

    ``orientation_lock_strength`` is handed to the precompute unchanged and
    defaults to the training lock (:data:`PRETRAIN_ORIENTATION_LOCK_STRENGTH`).
    A degenerate open shell (the O atom, OH) relaxes to whichever component of
    its degenerate manifold rounding picks, and two unlocked builds of the same
    atom differ by order one in rho at individual grid points while sharing one
    energy; locked, the rows are the ones the training SCF produces for the
    same species. On an s-only basis the bias vanishes identically.
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
        reference_xc=reference_xc,
        orientation_lock_strength=float(orientation_lock_strength))

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
                  exchange_footing="total",
                  orientation_lock_strength=PRETRAIN_ORIENTATION_LOCK_STRENGTH):
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
        exchange_footing=exchange_footing,
        orientation_lock_strength=orientation_lock_strength)


def _molecule_columns(mol_spec, reference_xc, basis, grid_level, *, polarized,
                      descriptors, density_fit=False, auxbasis=None,
                      cusp_log_transform=True, exchange_footing="total",
                      orientation_lock_strength=PRETRAIN_ORIENTATION_LOCK_STRENGTH):
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
        exchange_footing=exchange_footing,
        orientation_lock_strength=orientation_lock_strength)


def _x_block_lda(block):
    """``rho eps_x^LDA`` for one exchange block, in the block's own convention.

    A closed-shell system's exchange block IS its total-density block, which
    already carries the libxc-derived ``e_lda_x`` (a spin=0 call there, so the
    unpolarized LDA at the total density). An open shell's per-channel block
    carries no LDA column: its denominator is the analytic unpolarized LDA at
    the DOUBLED density ``2 rho_sigma``, a function of the stored ``rho``
    alone, and the expression here is the one :func:`spin_channel_exchange_rows`
    divided by (its floor and clip are inactive on every kept row, where
    ``rho > 1e-10`` puts ``|eps_x^LDA|`` above 3e-4). One expression serves the
    per-system target and the stored ``e_lda_x_x`` column, so the loss
    multiplies the network's enhancement factor by the same floating-point
    number the target was built from.
    """
    if "e_lda_x" in block:
        return np.asarray(block["e_lda_x"])
    rho = np.asarray(block["rho"])
    return rho * (_LDA_X_C * np.cbrt(rho))


def _system_energy_targets(cols, x_cols):
    """Per-system parent energies in Hartree: ``(e_x, e_c, e_x_scan, e_c_scan)``.

    Each is the ROW QUADRATURE over the rows this file stores,
    ``sum_i w_i e_LDA_i (1 + F_i)``, not libxc's full-grid integral. That
    choice is what makes the per-system energy term measure the fit and nothing
    else: the network's own energy on the same rows is
    ``sum_i w_i e_LDA_i F^NN_i``, so the residual vanishes exactly when the
    network reproduces the stored enhancement factors. The two integrals differ
    only by the rows the density floor drops and by the +-5 clip on the stored
    ratio, and the floor is the model's own tail threshold
    (``models._NN_TAIL_THRESHOLD`` = 1e-10), below which the model clamps F to 1
    and the network cannot move the energy at all -- the dropped rows are
    exactly the rows pretraining could not have fitted. Measured against libxc
    on the same density: zero on the O atom at def2-SVP / grid level 1 (pyscf's
    pruning leaves no point under the floor), 4.8e-12 Ha on OH/STO-3G level 0,
    <= 3.3e-11 Ha on N and H2O at def2-SVP level 1 -- six orders of magnitude
    under the certificate's tol_atom = 1.0 mHa. Summed by rung the targets
    reproduce the record's ``E_xc_pbe`` and, with ``E_non_xc``, its total SCF
    energy to the same floors.

    ``x_cols`` is the per-channel exchange block of
    :func:`spin_channel_exchange_rows`, or ``None`` when the exchange rows ARE
    the total-density rows (a closed-shell system, or the ``"total"`` footing).
    Its LDA denominator comes from :func:`_x_block_lda`. The correlation
    targets always come from the total-density rows: correlation is
    spin-interpolated rather than spin-scaled.
    """
    if x_cols is None:
        e_x = float(np.sum(cols["weights"] * cols["e_lda_x"]
                           * (1.0 + cols["Fx"])))
        e_x_scan = float(np.sum(cols["weights"] * cols["e_lda_x"]
                                * (1.0 + cols["Fx_scan"])))
    else:
        e_lda_x = _x_block_lda(x_cols)
        e_x = float(np.sum(x_cols["weights"] * e_lda_x
                           * (1.0 + x_cols["Fx"])))
        e_x_scan = float(np.sum(x_cols["weights"] * e_lda_x
                                * (1.0 + x_cols["Fx_scan"])))
    e_c = float(np.sum(cols["weights"] * cols["e_lda_c"] * (1.0 + cols["Fc"])))
    e_c_scan = float(np.sum(cols["weights"] * cols["e_lda_c"]
                            * (1.0 + cols["Fc_scan"])))
    return e_x, e_c, e_x_scan, e_c_scan


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
                             auxbasis=None, atoms=DEFAULT_PRETRAIN_ATOMS,
                             systems=None, reference_xc="pbe",
                             exchange_footing="total",
                             mesh_fraction=MESH_WEIGHT_FRACTION,
                             orientation_lock_strength=PRETRAIN_ORIENTATION_LOCK_STRENGTH,
                             allow_irreproducible_degenerate=False):
    """Record the identity a pretrain ``.npz`` was built at.

    Written as a sidecar so the ``.npz`` array payload stays byte-identical to
    the pre-manifest format (legacy loaders that ignore the sidecar are
    unaffected). Every key here is something a change of which changes the
    stored VALUES, so :func:`pretrain_data_is_current` treats all of them as
    the file's identity:

    - ``basis`` / ``grid_level`` / ``auxbasis``: the integration identity
      (``auxbasis`` is the EFFECTIVE DF fitting basis, ``None`` when DF is
      off, so a fitting-basis change forces a regeneration).
    - ``atoms``: the legacy projection ``[[name, 2S], ...]`` of the set, kept
      so a reader written before the set became a system list still resolves.
    - ``systems``: the set itself, ``[[name, geometry, charge, 2S], ...]``. A
      geometry change is a different physical system and must force a
      regeneration, which the atom-name projection cannot see.
    - ``reference_xc``: the functional whose SELF-CONSISTENT density the rows
      sit on (PBE for the GGA rung, SCAN for the meta-GGA rung).
    - ``exchange_footing``: ``"total"`` or ``"spin_channel"``. The open-shell
      exchange rows are a different row set under the two, so a footing change
      is a data change.
    - ``orientation_lock_strength``: the lock the parent density was computed
      at (:data:`PRETRAIN_ORIENTATION_LOCK_STRENGTH`). A degenerate open
      shell's rows are a different component of its manifold under a
      different lock, and an unlocked build is not reproducible at all.
    - ``allow_irreproducible_degenerate``: True iff the file carries a
      spatially degenerate free atom's rows built below
      :data:`COARSE_DEGENERATE_MIN_GRID_LEVEL` or with the orientation lock
      off, where they are one arbitrary member of the term's manifold rather
      than a reproducible quantity
      (:func:`_check_irreproducible_degenerate`). Recorded so a reader can
      see it; deliberately NOT part of the currency comparison, because the
      values it describes are the ones the other keys already identify, and
      :func:`ensure_pretrain_data` refuses that identity outright rather than
      serving such a file to a caller that did not ask for it.
    - ``x64``: whether JAX computed in double precision when the file was
      written (the generator refuses to write a single-precision column, so a
      file it wrote carries ``True``; recorded so a file from another writer
      declares its precision).
    - ``mesh.weight_fraction``: the share of the total integration weight the
      synthetic mesh carries. Recorded because it is a deliberate choice, not
      an emergent property of a quadrature: mesh rows carry no physical grid
      weight, so their pull on the pretrain loss is set here.
    - ``alpha_definition``: the definition of the iso-orbital indicator the
      ``metagga`` columns were computed with (``metagga.ALPHA_DEFINITION``,
      which names the smooth positive part and its width). The indicator is
      a stored column, so a change of its definition changes the file's
      values on every one-orbital row (the hard clip wrote 0.0 where the
      smooth positive part writes width / 2, on 1200 of 1200 rows of the
      default set's H atom and on the mesh's alpha = 0 nodes) without
      changing any other key; a file written under another definition is
      therefore stale, exactly as one built at another lock is.

    The writer's defaults are the PRODUCTION identity the generator and
    :func:`ensure_pretrain_data` use; a manifest key absent from a legacy file
    is read back as the HISTORICAL value (no lock, PBE, total footing, double
    precision, the hard-clipped indicator) by
    :func:`pretrain_data_is_current`.
    """
    meta = {"basis": basis, "grid_level": int(grid_level),
            "density_fit": bool(density_fit), "auxbasis": auxbasis,
            "atoms": [[str(s), int(sp)] for s, sp in atoms],
            "systems": (None if systems is None else
                        [[str(s.name), str(s.atom), int(s.charge),
                          int(s.spin)] for s in systems]),
            "reference_xc": str(reference_xc),
            "exchange_footing": str(exchange_footing),
            "orientation_lock_strength": float(orientation_lock_strength),
            "allow_irreproducible_degenerate":
                bool(allow_irreproducible_degenerate),
            "x64": bool(jax.config.jax_enable_x64),
            "alpha_definition": str(_ALPHA_DEFINITION),
            "mesh": {"rs": list(MESH_RS), "s": list(MESH_S),
                     "alpha": list(MESH_ALPHA),
                     "weight_fraction": float(mesh_fraction)}}
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


def _legacy_atom_rows(systems):
    """The ``[[symbol, 2S], ...]`` projection of a system list, or ``None``
    when some system is not a neutral free atom at the origin -- the only kind
    of system the legacy generator could write, so the only kind a manifest
    without a system list can be shown to hold."""
    rows = []
    for s in systems:
        key = _geometry_key(s.atom)
        if (int(s.charge) != 0 or len(key) != 1
                or key[0][1:] != (0.0, 0.0, 0.0) or key[0][0] != str(s.name)):
            return None
        rows.append([str(s.name), int(s.spin)])
    return rows


def _composition_matches(meta, systems):
    """Does the manifest's set equal ``systems``? A manifest with a system
    list is compared on every field; one without (written before the set
    became a system list) is compared through its atom projection, which
    identifies exactly the neutral free atoms at the origin, in order."""
    want = [[str(s.name), str(s.atom), int(s.charge), int(s.spin)]
            for s in systems]
    have = meta.get("systems")
    if have is not None:
        return [list(row) for row in have] == want
    legacy = _legacy_atom_rows(systems)
    if legacy is None:
        return False
    return meta.get(
        "atoms", [[str(s), int(sp)] for s, sp in DEFAULT_PRETRAIN_ATOMS]
    ) == legacy


def pretrain_data_is_current(npz_path, *, basis, grid_level, auxbasis=None,
                             atoms=DEFAULT_PRETRAIN_ATOMS, systems=None,
                             reference_xc="pbe", exchange_footing="total",
                             mesh_fraction=MESH_WEIGHT_FRACTION,
                             orientation_lock_strength=PRETRAIN_ORIENTATION_LOCK_STRENGTH,
                             x64=True):
    """True iff ``npz_path`` exists AND its manifest matches the requested
    identity.

    A missing file OR a missing/mismatched manifest returns ``False`` so the
    harness regenerates rather than silently reusing data built at a different
    identity. The identity is the full set :func:`_write_pretrain_manifest`
    records: basis, grid level, effective DF fitting basis, the system list
    (name, geometry, charge, spin of every system, in order), the parent
    functional, the exchange footing, the mesh share, the orientation-lock
    strength, the precision flag and the definition of the iso-orbital
    indicator (``metagga.ALPHA_DEFINITION``, the one constant the live
    generator writes its ``metagga`` columns with). A manifest key absent
    from a legacy file reads as the value the historical generator used --
    PBE, the ``total`` footing, the default mesh share, NO orientation lock,
    double precision (the production files were measured float64),
    ``auxbasis`` ``None``, the historical default atoms, the hard-clipped
    indicator -- so a legacy directory is current for a request at that
    identity and stale for the production one, whose lock its
    degenerate-atom rows were not computed at and whose indicator
    definition its alpha rows were not written under.

    ``systems`` is the resolved pretraining set. When given it replaces the
    ``atoms`` comparison (``atoms`` is otherwise resolved into systems the
    same way); a manifest without a system list is compared through its atom
    projection (:func:`_composition_matches`).

    THE WAIVER IS DELIBERATELY NOT COMPARED.
    ``allow_irreproducible_degenerate`` is recorded in the manifest and read
    by nothing here, and a waived file therefore cannot reach a caller that
    granted no waiver. The reason is that the waiver is a FUNCTION of the
    identity this check already compares: the generator's refusal
    (:func:`_check_irreproducible_degenerate`) fires on the systems, the basis,
    the grid level and the lock, all four of which must match for the file to
    be current at all, and :func:`ensure_pretrain_data` applies that refusal to
    the REQUESTED identity before ever asking whether the file is current. So
    at an identity that needs the waiver a non-waiving caller is refused
    before the comparison, and at an identity that does not need it the flag
    was never exercised -- the manifest records False there whatever the
    caller passed. Measured over the six combinations of the flag with (grid
    level 1, lock 3e-5), (grid level 3, lock 0) and (grid level 3, lock 3e-5),
    with the file present at exactly the requested identity: the only
    combination in which a non-waiving caller is served is the one at which
    the permission grants nothing.
    """
    if not os.path.isfile(npz_path):
        return False
    meta = read_pretrain_manifest(npz_path)
    if meta is None:
        return False
    if systems is None:
        systems = resolve_pretrain_systems(atoms=atoms)
    else:
        systems = tuple(normalize_system(s) for s in systems)
    manifest_ok = (meta.get("basis") == basis
                   and int(meta.get("grid_level", -1)) == int(grid_level)
                   and meta.get("auxbasis") == auxbasis
                   and _composition_matches(meta, systems)
                   and str(meta.get("reference_xc", "pbe"))
                   == str(reference_xc)
                   and str(meta.get("exchange_footing", "total"))
                   == str(exchange_footing)
                   and float(meta.get("mesh", {}).get(
                       "weight_fraction", MESH_WEIGHT_FRACTION))
                   == float(mesh_fraction)
                   and float(meta.get("orientation_lock_strength", 0.0))
                   == float(orientation_lock_strength)
                   and bool(meta.get("x64", True)) == bool(x64)
                   and str(meta.get("alpha_definition",
                                    _LEGACY_ALPHA_DEFINITION))
                   == str(_ALPHA_DEFINITION))
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
    # The manifest's exchange FOOTING against the file's own blocks. The
    # footing is not a property the manifest can assert on its own: the
    # ``*_x`` block IS the spin_channel footing, and its absence IS the total
    # one (pretrain_npz_layout reads the footing off ``rho_x`` for exactly
    # that reason). A manifest that names one while the file carries the other
    # matched every identity key above and would be served, and the exchange
    # rows the pretraining objective then reads are not the ones the run asked
    # for -- per channel at the doubled density, or on the total density, with
    # nothing to tell them apart downstream. Gated on ``Fx_all`` so a
    # manifest-only stub is not flagged.
    if "Fx_all" in _keys and ("rho_x" in _keys) != (
            str(exchange_footing) == "spin_channel"):
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


def ensure_pretrain_data(data_dir, *, atoms=None, basis=DEFAULT_BASIS,
                         grid_level=DEFAULT_GRID_LEVEL, polarized=True,
                         descriptors=True, density_fit=False, auxbasis=None,
                         cusp_log_transform=True, progress=False,
                         dfs_set=False, pool_atoms=False, reference_xc="pbe",
                         exchange_footing="total",
                         mesh_fraction=MESH_WEIGHT_FRACTION,
                         orientation_lock_strength=PRETRAIN_ORIENTATION_LOCK_STRENGTH,
                         allow_irreproducible_degenerate=False):
    """Skip-if-current driver for staged pretrain data.

    Returns the canonical ``.npz`` path, (re)generating it ONLY when the file
    is absent or its manifest's identity differs from the requested one.
    Idempotent: a second call at the same settings is a no-op. The set is
    resolved ONCE here and handed to both the currency check and the
    generator, so the file that is checked and the file that is written can
    never be built from different lists. ``atoms`` of ``None`` is the
    historical four-atom default unless ``dfs_set`` / ``pool_atoms`` supply
    the set (:func:`resolve_pretrain_systems`).

    The irreproducible-degenerate refusal is applied to the requested
    IDENTITY, before the currency check: a file already on disk at an identity
    the generator would refuse to produce must not be served either.
    """
    _check_generator_arguments(reference_xc, exchange_footing, mesh_fraction)
    eff_aux = _effective_auxbasis(basis, density_fit, auxbasis)
    systems = resolve_pretrain_systems(atoms=atoms, dfs_set=dfs_set,
                                       pool_atoms=pool_atoms,
                                       reference_xc=reference_xc)
    _check_irreproducible_degenerate(systems, basis, grid_level,
                                     orientation_lock_strength,
                                     allow_irreproducible_degenerate)
    out_path = os.path.join(data_dir,
                            pretrain_data_filename(polarized, reference_xc))
    if pretrain_data_is_current(out_path, basis=basis, grid_level=grid_level,
                                auxbasis=eff_aux, systems=systems,
                                reference_xc=reference_xc,
                                exchange_footing=exchange_footing,
                                mesh_fraction=mesh_fraction,
                                orientation_lock_strength=orientation_lock_strength):
        return out_path
    # ``systems`` alone: the generator takes the resolved tuple whenever it is
    # given and ignores ``atoms`` entirely, so passing both stated one input
    # twice and invited the two to disagree.
    return generate_pretrain_data_npz(
        data_dir, systems=systems, basis=basis,
        grid_level=grid_level, polarized=polarized, descriptors=descriptors,
        density_fit=density_fit, auxbasis=auxbasis,
        cusp_log_transform=cusp_log_transform, progress=progress,
        reference_xc=reference_xc, exchange_footing=exchange_footing,
        mesh_fraction=mesh_fraction,
        orientation_lock_strength=orientation_lock_strength,
        allow_irreproducible_degenerate=allow_irreproducible_degenerate)


# --------------------------------------------------------------------------- #
# The .npz schema: two row blocks, a system table, the mesh, one scalar.
#
# Every key the file can carry is declared here, so the writer can refuse a
# column it has no slot for instead of dropping it silently (the historical
# writer selected explicit keys and would have dropped a nested ``x_rows``) and
# the reader can refuse a file with a missing block or an unknown key.
# --------------------------------------------------------------------------- #

#: Column stems of the total-density block (``<stem>_all``): the historical
#: columns every file carries ...
_ALL_CORE = ("rho", "sigma", "Fx", "Fc", "Fx_scan", "Fc_scan", "metagga",
             "weights")
#: ... and the columns the pretraining protocol added beside them (the
#: per-row system index and the LDA energy densities of the energy term).
_ALL_PROTOCOL = ("system", "e_lda_x", "e_lda_c")
#: Descriptor columns, present iff the file was written with ``descriptors``.
_DESCRIPTOR_STEMS = ("cusp", "dm", "rung35", "rung35ms")
#: The exchange block (``<stem>_x``), present iff the file was written on the
#: ``spin_channel`` footing: the per-channel rows of every open shell and the
#: total-density rows of every closed shell, with their own system index and
#: their own LDA column (:func:`_x_block_lda`).
_X_CORE = ("rho", "sigma", "Fx", "Fx_scan", "metagga", "weights", "system",
           "e_lda_x")
#: The synthetic (r_s, s, alpha) mesh (``<stem>_mesh``).
_MESH_CORE = ("rho", "sigma", "Fx_scan", "Fc_scan", "metagga", "weights")
#: Per-system table: the parent energies in Hartree and the nucleus count.
_SYSTEM_TABLE = ("e_x_parent_sys", "e_c_parent_sys", "e_x_parent_scan_sys",
                 "e_c_parent_scan_sys", "system_natoms")
#: 0-d scalars.
_SCALARS = ("mesh_weight_fraction",)
#: Integer columns; everything else is float64.
_INT_KEYS = frozenset({"system_all", "system_x", "system_natoms"})
#: Trailing shape of the 2-D columns (the leading axis is the block's rows);
#: ``dm`` is 2-D with a width the descriptor defines.
_COLUMN_WIDTHS = {"metagga": (1,), "cusp": (2,), "rung35": (2,),
                  "rung35ms": (6,)}
_TWO_D_STEMS = ("metagga",) + _DESCRIPTOR_STEMS


def _stems_all(polarized, descriptors):
    stems = _ALL_CORE + _ALL_PROTOCOL
    if polarized:
        stems += ("zeta",)
    if descriptors:
        stems += _DESCRIPTOR_STEMS
    return stems


def _stems_x(descriptors):
    return _X_CORE + (_DESCRIPTOR_STEMS if descriptors else ())


def _stems_mesh(polarized):
    return _MESH_CORE + (("zeta",) if polarized else ())


def pretrain_npz_keys(*, polarized, descriptors, exchange_footing):
    """The exact key set a pretrain ``.npz`` written at this configuration
    carries."""
    keys = {f"{s}_all" for s in _stems_all(polarized, descriptors)}
    keys |= {f"{s}_mesh" for s in _stems_mesh(polarized)}
    keys |= set(_SYSTEM_TABLE) | set(_SCALARS)
    if exchange_footing == "spin_channel":
        keys |= {f"{s}_x" for s in _stems_x(descriptors)}
    return keys


#: Every key any configuration can carry: the reader's universe.
_KNOWN_KEYS = frozenset(
    pretrain_npz_keys(polarized=True, descriptors=True,
                      exchange_footing="spin_channel"))


#: The three keys whose PRESENCE declares a configuration, with what each
#: declares and the keys that accompany it. A file that lost one of them is
#: read as a file written without that configuration, and its companions then
#: have no slot -- so the refusal below would name the companions, which are
#: present and correct, and never the key that is actually missing. Each entry
#: is ``(sentinel, what it declares, companions)``.
_LAYOUT_SENTINELS = (
    ("zeta_all", "the spin-polarized (zeta-carrying) total-density block",
     ("zeta_mesh",)),
    ("cusp_all", "the descriptor columns",
     tuple(f"{s}_all" for s in _DESCRIPTOR_STEMS if s != "cusp")
     + tuple(f"{s}_x" for s in _DESCRIPTOR_STEMS)),
    ("rho_x", "the spin_channel exchange block",
     tuple(f"{s}_x" for s in _X_CORE if s != "rho")),
)


def _missing_layout_sentinel(keys):
    """The layout key a key set has LOST, or ``None``.

    A sentinel counts as lost only when its companions are present: a file
    that carries none of a block's columns was written without that block,
    which is a configuration and not a defect.
    """
    for sentinel, declares, companions in _LAYOUT_SENTINELS:
        if sentinel in keys:
            continue
        present = sorted(k for k in companions if k in keys)
        if present:
            return sentinel, declares, present
    return None


def pretrain_npz_layout(keys):
    """Which blocks a key set carries, and that it is a consistent schema.

    Returns ``{"polarized", "descriptors", "exchange_footing", "system_table",
    "mesh"}``. A key outside the schema is refused, and so is a file whose
    blocks are incomplete: a file carrying the protocol's system table must
    carry EXACTLY the key set of its configuration (:func:`pretrain_npz_keys`),
    an exchange block cannot appear without the system table, and a legacy
    file (no system table, written before the protocol) must carry the
    historical core of the total-density block plus whatever newer optional
    columns it has, nothing else. An existing production file is therefore
    still readable for the point-wise loss; a torn or foreign file is not.

    A file that has lost one of the three LAYOUT KEYS -- ``zeta_all``,
    ``cusp_all``, ``rho_x``, whose presence is what declares the polarization,
    the descriptors and the exchange footing -- is named for the key it lost
    (:func:`_missing_layout_sentinel`). Without that the layout reads as the
    other configuration and the refusal names the columns that accompany the
    missing key, every one of them present and correct, which points the
    reader at the wrong end of the file.
    """
    keys = set(keys)
    unknown = sorted(keys - _KNOWN_KEYS)
    if unknown:
        raise ValueError(
            f"pretrain data carries keys outside the schema: {unknown}")
    lost = _missing_layout_sentinel(keys)
    if lost is not None:
        sentinel, declares, present = lost
        raise ValueError(
            f"pretrain data is missing the layout key {sentinel!r} while "
            f"carrying {present}: {sentinel!r} is what declares "
            f"{declares}, so without it the file reads as one written without "
            "that configuration and the columns above have no slot in the "
            "layout. The missing key is the defect, not the columns")
    layout = {
        "polarized": "zeta_all" in keys,
        "descriptors": "cusp_all" in keys,
        "exchange_footing": ("spin_channel" if "rho_x" in keys else "total"),
        "system_table": "system_all" in keys,
        "mesh": "rho_mesh" in keys,
    }
    if layout["system_table"]:
        want = pretrain_npz_keys(polarized=layout["polarized"],
                                 descriptors=layout["descriptors"],
                                 exchange_footing=layout["exchange_footing"])
        missing = sorted(want - keys)
        if missing:
            raise ValueError(
                f"pretrain data is missing {missing} for its layout {layout}")
        extra = sorted(keys - want)
        if extra:
            raise ValueError(
                f"pretrain data carries {extra}, which its layout {layout} "
                "does not declare")
        return layout
    protocol = sorted(k for k in keys if k.endswith("_x")
                      or k in _SYSTEM_TABLE or k in _SCALARS
                      or k in ("e_lda_x_all", "e_lda_c_all"))
    if protocol:
        raise ValueError(
            f"pretrain data carries the protocol columns {protocol} without "
            "the per-row system index 'system_all': a torn file")
    missing = sorted(f"{s}_all" for s in ("rho", "sigma", "Fx", "Fc", "weights")
                     if f"{s}_all" not in keys)
    if missing:
        raise ValueError(
            f"pretrain data is missing the total-density columns {missing}")
    if layout["mesh"]:
        missing = sorted(f"{s}_mesh" for s in _MESH_CORE
                         if f"{s}_mesh" not in keys)
        if missing:
            raise ValueError(f"pretrain data is missing the mesh columns "
                             f"{missing}")
    return layout


def _validate_pretrain_arrays(arrays, *, expected=None):
    """Check a key -> array mapping against the schema and return its layout.

    ``expected`` (the writer's configuration, ``polarized`` / ``descriptors``
    / ``exchange_footing``) demands the exact key set of that configuration.
    Every array must be float64 -- a float32 column was computed in single
    precision and casting it up recovers nothing -- except the int32 system
    indices and nucleus counts; the rows of one block share one length, the
    2-D columns carry their declared widths, the scalars are 0-d, the system
    table is one row per system and every row index points into it.
    """
    keys = set(arrays)
    if expected is not None:
        want = pretrain_npz_keys(**expected)
        missing, extra = sorted(want - keys), sorted(keys - want)
        if missing or extra:
            raise ValueError(
                "the pretrain data columns do not match the declared schema "
                f"for {expected}: missing {missing}, unexpected {extra}")
    layout = pretrain_npz_layout(keys)
    for key, arr in arrays.items():
        want_dtype = np.int32 if key in _INT_KEYS else np.float64
        if arr.dtype != want_dtype:
            raise ValueError(
                f"pretrain data column {key!r} is {arr.dtype}, not "
                f"{np.dtype(want_dtype).name}")
    for key in _SCALARS:
        if key in arrays and arrays[key].shape != ():
            raise ValueError(f"{key!r} must be a 0-d scalar, got shape "
                             f"{arrays[key].shape}")
    n_sys = None
    if layout["system_table"]:
        n_sys = int(arrays["system_natoms"].shape[0])
        for key in _SYSTEM_TABLE:
            if arrays[key].shape != (n_sys,):
                raise ValueError(
                    f"system table column {key!r} has shape "
                    f"{arrays[key].shape}; expected ({n_sys},)")
    for suffix in ("_all", "_x", "_mesh"):
        block = {k: v for k, v in arrays.items() if k.endswith(suffix)}
        if not block:
            continue
        n_rows = int(block[f"rho{suffix}"].shape[0])
        for key, arr in block.items():
            stem = key[:-len(suffix)]
            if arr.ndim == 0 or arr.shape[0] != n_rows:
                raise ValueError(
                    f"pretrain data column {key!r} has {arr.shape} rows; the "
                    f"{suffix} block has {n_rows}")
            if stem in _TWO_D_STEMS:
                width = _COLUMN_WIDTHS.get(stem)
                if arr.ndim != 2 or (width is not None
                                     and arr.shape[1:] != width):
                    raise ValueError(
                        f"pretrain data column {key!r} has shape {arr.shape}; "
                        f"expected ({n_rows}, {width[0] if width else 'k'})")
            elif arr.ndim != 1:
                raise ValueError(
                    f"pretrain data column {key!r} has shape {arr.shape}; "
                    "expected a 1-D column")
        index_key = f"system{suffix}"
        if index_key in block and n_sys is not None and n_rows:
            idx = block[index_key]
            if int(idx.min()) < 0 or int(idx.max()) >= n_sys:
                raise ValueError(
                    f"pretrain data index {index_key!r} spans "
                    f"[{int(idx.min())}, {int(idx.max())}] but the system "
                    f"table has {n_sys} systems")
    return layout


def load_pretrain_data_npz(npz_path):
    """Read a pretrain ``.npz`` into ``{key: ndarray}`` after checking it
    against the schema (:func:`_validate_pretrain_arrays`): an unknown key, a
    missing block, a single-precision or misaligned column is refused rather
    than handed to the trainer. Every array is materialized, so the file
    handle is closed on return."""
    with np.load(npz_path) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    _validate_pretrain_arrays(arrays)
    return arrays


#: Grid level below which a spatially degenerate free atom's rows are not
#: reproducible between processes. Measured on the locked O atom: at the
#: generator's own DEFAULT_GRID_LEVEL of 1 independent processes differ at the
#: 1e-3..1e-1 level in rho, by of order unity in the iso-orbital indicator and
#: at the 1e-6 Ha level in the stored exchange energy, while at level 3 the
#: same comparison reproduces to 3e-11 relative. The spreads are stated as
#: orders of magnitude because they are samples of a process-to-process
#: scatter and not bounds: two independent sets of draws gave 3e-3 / 0.64 /
#: 3.7e-6 Ha and 5.7e-2 / 12.4 / 1.3e-6 Ha, and the indicator's spread over
#: six draw pairs ran 0.55 to 2.46, so unity is the middle of that scatter
#: rather than a floor under it. The lock fixes WHICH member of the
#: P-term manifold the SCF converges to; it cannot make a quadrature that
#: coarse resolve it.
COARSE_DEGENERATE_MIN_GRID_LEVEL = 3


def _degenerate_systems(systems, basis, grid_level):
    """The names of the spatially degenerate free atoms in ``systems``.

    A free atom with an open p shell of 1, 2, 4 or 5 electrons is a P term
    whose SCF can converge to any orientation of its hole
    (:func:`cluster.fidelity.is_degenerate_atom`, the one implementation of
    that rule; imported lazily because this module is imported by the harness
    parser paths that must not pull in the certificate). Half-filled, closed
    and s-shell atoms are spherical and excluded, as is every molecule.

    ``grid_level`` enters only through the mol spec the rule is asked about;
    the answer is a property of the shell, not of the quadrature.
    """
    from xcquinox.alec.cluster.fidelity import is_degenerate_atom
    flagged = []
    for system in systems:
        try:
            degenerate = is_degenerate_atom(
                _mol_spec_for(system, basis, grid_level))
        except ValueError:
            # Beyond argon: the degeneracy rule does not cover the shell, so
            # this guard makes no claim about it.
            continue
        if degenerate:
            flagged.append(system.name)
    return tuple(flagged)


def _check_irreproducible_degenerate(systems, basis, grid_level,
                                     orientation_lock_strength,
                                     allow_irreproducible_degenerate):
    """Refuse an unreproducible identity; return whether one was permitted.

    The manifest records an identity, and a file whose degenerate-atom rows
    are one arbitrary member of a manifold does not have the identity it
    claims. TWO conditions produce such a file, and one flag covers both
    because the defect is the same one either way:

    - **A coarse grid.** Below :data:`COARSE_DEGENERATE_MIN_GRID_LEVEL` the
      quadrature does not resolve the P term: locked draws of the O atom at
      level 1 differ at the 1e-3..1e-1 level in rho, by of order unity in the
      iso-orbital indicator and at the 1e-6 Ha level in the stored exchange
      energy, against 3e-11 relative at level 3. The figures are orders of
      magnitude spanning two independent sets of draws (3e-3 / 0.64 / 3.7e-6
      Ha and 5.7e-2 / 12.4 / 1.3e-6 Ha), not bounds; the indicator ran 0.55 to
      2.46 over six draw pairs, three of them below unity.
    - **No orientation lock.** With ``orientation_lock_strength`` at zero the
      SCF may land on any orientation of the hole however fine the grid is:
      unlocked draws of the O atom at level 3 keep different numbers of rows
      and disagree at the 3e-7 Ha level in the total energy (2.6e-7 Ha over
      one pair, 2.9e-7 Ha over a later triple), so the row set itself -- not
      merely its values -- depends on which process wrote it.

    The generation is refused under either condition unless the caller says
    explicitly that it wants the unreproducible build anyway -- the reference
    recorder and the unit tests that deliberately run coarse or unlocked,
    which is why the flag exists rather than a hard floor.

    Returns True only when the permission was actually EXERCISED (a flagged
    system under one of the two conditions), which is what the manifest
    records: a production file at grid level 3 with the lock on records False
    whether or not the caller passed the flag.
    """
    flagged = _degenerate_systems(systems, basis, grid_level)
    if not flagged:
        return False
    coarse = int(grid_level) < COARSE_DEGENERATE_MIN_GRID_LEVEL
    unlocked = float(orientation_lock_strength) == 0.0
    if not (coarse or unlocked):
        return False
    if not allow_irreproducible_degenerate:
        reasons = []
        if coarse:
            reasons.append(
                f"the grid is below level "
                f"{COARSE_DEGENERATE_MIN_GRID_LEVEL}, so the quadrature does "
                "not resolve the term (locked draws of O at level 1 differ at "
                "the 1e-3..1e-1 level in rho, by of order unity in the "
                "iso-orbital indicator and at the 1e-6 Ha level in the stored "
                "exchange energy between draws, against 3e-11 relative at "
                "level 3)")
        if unlocked:
            reasons.append(
                "the orientation lock is off, so the SCF may land on any "
                "orientation of the hole however fine the grid is (unlocked "
                "draws of O at level 3 keep different row counts and disagree "
                "at the 3e-7 Ha level in the total energy)")
        raise ValueError(
            f"pretraining system(s) {', '.join(flagged)} are spatially "
            f"degenerate free atoms, and their rows at grid level "
            f"{int(grid_level)} with "
            f"orientation_lock_strength={float(orientation_lock_strength):g} "
            "are not reproducible between processes, so the manifest would "
            "record an identity the file does not have: "
            + "; ".join(reasons) + ". Use grid level "
            f">= {COARSE_DEGENERATE_MIN_GRID_LEVEL} with "
            f"orientation_lock_strength={PRETRAIN_ORIENTATION_LOCK_STRENGTH:g}"
            ", or build the unreproducible file deliberately by passing "
            "allow_irreproducible_degenerate=True (from a harness "
            "configuration: inputs.allow_irreproducible_degenerate, which "
            "requires inputs.irreproducible_degenerate_reason)."
        )
    return True


def _check_generator_arguments(reference_xc, exchange_footing, mesh_fraction):
    """Refuse a bad parent, footing or mesh share before any SCF is paid for."""
    if reference_xc not in ("pbe", "scan"):
        raise ValueError(
            f"reference_xc must be 'pbe' or 'scan'; got {reference_xc!r}.")
    if exchange_footing not in ("total", "spin_channel"):
        raise ValueError(
            "exchange_footing must be 'total' or 'spin_channel'; got "
            f"{exchange_footing!r}."
        )
    mesh_fraction = float(mesh_fraction)
    if not (0.0 < mesh_fraction < 1.0):
        raise ValueError(
            "mesh_fraction is the mesh's share of the total integration "
            "weight, w_mesh / (w_atom + w_mesh), and must lie strictly "
            f"between 0 and 1; got {mesh_fraction!r}."
        )


def _exchange_block(cols):
    """The exchange rows of one system in the ``_x`` stems: the per-channel
    rows of an open shell (``x_rows``), the total-density rows of a closed
    shell, whose ``rho_a = rho_b`` makes the two the same rows (that block
    keeps its libxc ``e_lda_x``; the per-channel block carries none)."""
    x_rows = cols.get("x_rows")
    if x_rows is not None:
        return dict(x_rows)
    keep = set(_X_CORE + _DESCRIPTOR_STEMS) - {"system"}
    return {k: cols[k] for k in cols if k in keep}


def _check_block_columns(name, block, want_stems, suffix):
    """One system's column dict against the stems its block declares: a
    missing column, a column with no slot, and a column whose leading length
    is not the block's row count are refused by name."""
    have = frozenset(block)
    missing = sorted(f"{k}{suffix}" for k in want_stems - have)
    extra = sorted(f"{k}{suffix}" for k in have - want_stems)
    if missing or extra:
        raise ValueError(
            f"the columns of pretraining system {name!r} do not match the "
            f"schema: missing {missing}, without a slot {extra}")
    n_rows = int(np.asarray(block["rho"]).shape[0])
    for k in sorted(have):
        arr = np.asarray(block[k])
        if arr.ndim == 0 or arr.shape[0] != n_rows:
            raise ValueError(
                f"column {k + suffix!r} of pretraining system {name!r} has "
                f"shape {arr.shape} against {n_rows} rows")
    return n_rows


def _assemble_blocks(per_system, systems, *, polarized, descriptors,
                     exchange_footing, mesh_fraction):
    """Concatenate the per-system column dicts into the file's arrays.

    Every column a builder returned is mapped to a slot -- ``<stem>_all`` for
    the total-density rows, ``<stem>_x`` for the exchange rows -- and a column
    with no slot is refused rather than dropped, a missing or misaligned one
    by name, before any arithmetic is done on them. The per-row system
    indices, the exchange block's LDA column, the system table, the mesh and
    the mesh share are added here.
    """
    want_all = frozenset(_stems_all(polarized, descriptors)) - {"system"}
    for system, cols in zip(systems, per_system):
        _check_block_columns(
            system.name, {k: v for k, v in cols.items() if k != "x_rows"},
            want_all, "_all")
    arrays = {f"{k}_all": np.concatenate([np.asarray(c[k]) for c in per_system])
              for k in sorted(want_all)}
    arrays["system_all"] = np.concatenate(
        [np.full(np.asarray(c["rho"]).shape[0], i, dtype=np.int32)
         for i, c in enumerate(per_system)])
    if exchange_footing == "spin_channel":
        # One exchange block over EVERY system: the per-channel rows of an
        # open shell, and the total-density rows of a closed shell, which ARE
        # the per-channel rows there. The LDA column is computed per block
        # (:func:`_x_block_lda`) rather than carried, because only the
        # closed-shell blocks come with one.
        want_x = frozenset(_stems_x(descriptors)) - {"system", "e_lda_x"}
        x_blocks, x_lda = [], []
        for system, cols in zip(systems, per_system):
            block = _exchange_block(cols)
            lda = _x_block_lda(block)
            block = {k: v for k, v in block.items() if k != "e_lda_x"}
            _check_block_columns(system.name, block, want_x, "_x")
            x_blocks.append(block)
            x_lda.append(lda)
        for k in sorted(want_x):
            arrays[f"{k}_x"] = np.concatenate(
                [np.asarray(b[k]) for b in x_blocks])
        arrays["e_lda_x_x"] = np.concatenate(x_lda)
        arrays["system_x"] = np.concatenate(
            [np.full(np.asarray(b["rho"]).shape[0], i, dtype=np.int32)
             for i, b in enumerate(x_blocks)])
        targets = [_system_energy_targets(c, c.get("x_rows"))
                   for c in per_system]
    else:
        targets = [_system_energy_targets(c, None) for c in per_system]
    # Per-system parent energies, Hartree. Both the PBE and the SCAN targets,
    # for the same reason the Fx / Fx_scan columns are both present: the file's
    # density is the parent's, the target is the rung's.
    arrays.update({
        "e_x_parent_sys": np.array([t[0] for t in targets], dtype=np.float64),
        "e_c_parent_sys": np.array([t[1] for t in targets], dtype=np.float64),
        "e_x_parent_scan_sys": np.array([t[2] for t in targets],
                                        dtype=np.float64),
        "e_c_parent_scan_sys": np.array([t[3] for t in targets],
                                        dtype=np.float64),
        # Nuclei per system: the validation split holds out MOLECULES only.
        "system_natoms": np.array([_n_atoms(s.atom) for s in systems],
                                  dtype=np.int32),
    })
    # (s, alpha) parameter-space mesh, stored under SEPARATE *_mesh keys so the
    # atomic arrays every GGA arch reads stay byte-identical. pretrain.py
    # concatenates these ONLY for a meta_gga arch whose descriptor set the mesh
    # can actually define (see _mesh_columns).
    mesh = _mesh_columns()
    w_rows = float(arrays["weights_all"].sum())
    n_mesh = mesh["rho"].shape[0]
    # Rescale the (weightless) mesh rows to a stated share of the total
    # integration weight: w_mesh_total / (w_rows + w_mesh_total) = FRACTION.
    w_mesh_total = w_rows * mesh_fraction / (1.0 - mesh_fraction)
    arrays.update({
        "rho_mesh": mesh["rho"],
        "sigma_mesh": mesh["sigma"],
        "Fx_scan_mesh": mesh["Fx_scan"],
        "Fc_scan_mesh": mesh["Fc_scan"],
        "metagga_mesh": mesh["metagga"],
        "weights_mesh": np.full(n_mesh, w_mesh_total / n_mesh),
        # Stored beside the weights it produced so the loss reads the share the
        # DATA was built at rather than a constant that may have moved.
        "mesh_weight_fraction": np.asarray(float(mesh_fraction)),
    })
    if polarized:
        arrays["zeta_mesh"] = mesh["zeta"]
    return arrays


def generate_pretrain_data_npz(out_dir, *, atoms=None, basis=DEFAULT_BASIS,
                               grid_level=DEFAULT_GRID_LEVEL,
                               polarized=True, descriptors=True,
                               density_fit=False, auxbasis=None,
                               cusp_log_transform=True, progress=False,
                               dfs_set=False, pool_atoms=False,
                               reference_xc="pbe",
                               exchange_footing="total",
                               mesh_fraction=MESH_WEIGHT_FRACTION,
                               systems=None,
                               orientation_lock_strength=PRETRAIN_ORIENTATION_LOCK_STRENGTH,
                               allow_irreproducible_degenerate=False):
    """Generate the pretrain-data ``.npz`` in ``out_dir`` and return its path.

    ``polarized=True`` writes the zeta-carrying file; ``reference_xc="scan"``
    writes the SCAN-density file under its own name
    (:func:`pretrain_data_filename`). The set is
    ``resolve_pretrain_systems(atoms=..., dfs_set=..., pool_atoms=...,
    reference_xc=...)`` unless ``systems`` supplies an already-resolved tuple,
    which is how :func:`ensure_pretrain_data` guarantees the currency check
    and the generation see the same list. The parent density of every system
    is computed at ``orientation_lock_strength`` (the training lock by
    default, see :data:`PRETRAIN_ORIENTATION_LOCK_STRENGTH`).

    TWO ROW BLOCKS. The historical ``*_all`` block is the total-density block:
    it carries the correlation rows always, and the exchange rows too under the
    default ``"total"`` footing. Under ``exchange_footing="spin_channel"`` a
    second ``*_x`` block carries the exchange rows on the exact-spin-scaling
    footing -- per channel at ``(2 rho_sigma, 4 sigma_sigma_sigma, features of
    diag(P_sigma, P_sigma))`` for an open shell (Oliver and Perdew, Phys. Rev. A
    20, 397 (1979)), and the total-density rows for a closed shell, where
    rho_a = rho_b makes the two the same rows. The two blocks have different
    lengths on an open shell, which is why they cannot share one set of names,
    and the exchange block carries its own LDA column ``e_lda_x_x``, because
    an open shell's total-density ``e_lda_x_all`` is the SPIN-POLARIZED LDA
    while the per-channel ratio was formed against the unpolarized LDA at the
    doubled density.

    THE SYSTEM TABLE. ``system_all`` / ``system_x`` index each row into the
    system it came from, and ``e_{x,c}_parent[_scan]_sys`` hold that system's
    parent energy in Hartree as the row quadrature (see
    :func:`_system_energy_targets`). Together they are the per-system energy
    term of the pretraining objective: a network can no longer lower the
    point-wise residual while missing a system's energy.

    THE SCHEMA IS CLOSED. Every column a builder returns is written under its
    slot or refused (:func:`_assemble_blocks`), the written key set must be
    exactly :func:`pretrain_npz_keys` of the configuration, every column is
    float64 except the int32 indices, and the arrays are written only after
    they pass :func:`_validate_pretrain_arrays` -- the check
    :func:`load_pretrain_data_npz` repeats on the way back in.

    A sidecar ``<npz>.manifest.json`` records the identity the data was built
    at so :func:`pretrain_data_is_current` can force a regeneration.

    A spatially degenerate free atom is refused below
    :data:`COARSE_DEGENERATE_MIN_GRID_LEVEL`, and at any grid level with
    ``orientation_lock_strength`` at zero
    (:func:`_check_irreproducible_degenerate`): its rows are then one
    arbitrary member of the term's manifold rather than the reproducible
    quantity the manifest claims.
    ``allow_irreproducible_degenerate=True`` builds it anyway and the manifest
    records that it did.
    """
    from xcquinox.alec.data import clear_precompute_cache
    _check_generator_arguments(reference_xc, exchange_footing, mesh_fraction)
    systems = (tuple(normalize_system(s) for s in systems)
               if systems is not None
               else resolve_pretrain_systems(atoms=atoms, dfs_set=dfs_set,
                                             pool_atoms=pool_atoms,
                                             reference_xc=reference_xc))
    if not systems:
        raise ValueError(
            "the pretraining set is empty: pass atoms=..., or turn on "
            "dfs_set / pool_atoms."
        )
    # Before any SCF is paid for: a spatially degenerate free atom below
    # COARSE_DEGENERATE_MIN_GRID_LEVEL, or with the orientation lock off, is
    # refused unless the caller asked for the unreproducible build outright,
    # and the permission is recorded in the manifest when it was exercised.
    irreproducible_degenerate = _check_irreproducible_degenerate(
        systems, basis, grid_level, orientation_lock_strength,
        allow_irreproducible_degenerate)
    per_system = []
    for _i, system in enumerate(systems, 1):
        if progress:
            print(f"  pretrain data: system {_i}/{len(systems)} "
                  f"{system.name} ({reference_xc.upper()} density @ "
                  f"{basis}) ...",
                  flush=True)
        per_system.append(_system_columns(
            system, basis, grid_level, reference_xc=reference_xc,
            polarized=polarized,
            descriptors=descriptors, density_fit=density_fit,
            auxbasis=auxbasis, cusp_log_transform=cusp_log_transform,
            exchange_footing=exchange_footing,
            orientation_lock_strength=orientation_lock_strength))
        # precompute_fixed_density_data memoizes its MoleculeData in a
        # process-level dict, and each one holds the (4, n_grid, n_ao) AO
        # derivative tensor -- of order 0.8 GB for a ten-nucleus molecule at
        # 6-311++G(3df,2pd) and grid level 3. Retaining one per system would
        # exhaust the node long before the set is generated, and nothing here
        # revisits a system, so the cache is dropped as each system's columns
        # are extracted.
        clear_precompute_cache()
    save_kwargs = _assemble_blocks(
        per_system, systems, polarized=polarized, descriptors=descriptors,
        exchange_footing=exchange_footing, mesh_fraction=mesh_fraction)
    _validate_pretrain_arrays(
        save_kwargs, expected=dict(polarized=bool(polarized),
                                   descriptors=bool(descriptors),
                                   exchange_footing=exchange_footing))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            pretrain_data_filename(polarized, reference_xc))
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
        atoms=tuple((s.name, s.spin) for s in systems), systems=systems,
        reference_xc=reference_xc, exchange_footing=exchange_footing,
        mesh_fraction=mesh_fraction,
        orientation_lock_strength=orientation_lock_strength,
        allow_irreproducible_degenerate=irreproducible_degenerate)
    return out_path
