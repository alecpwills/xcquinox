"""xcquinox.alec.losses: AlecLoss ABC, registry, 6 concrete losses.

Implements THE SPEC §7.1 (base class) and §7.2 (A, B, C, D1, D2, D3).
"""
import abc
import math
import warnings
from typing import ClassVar

import jax.numpy as jnp
import equinox as eqx

from xcquinox.alec.oneshot import (
    fixed_density_total_energy,
    total_energy_for_solver,
    energy_trajectory_for_solver,
    scf_loss_tail_weights,
    dm_prediction_for_loss,
    grid_density_for_loss,
    compute_vxc_nn,
    _uks_spin_resolved_vxc,
)
from xcquinox.alec.descriptors import assemble_descriptor_features
from xcquinox.alec.energy_override import get_energy_override


# Scale-aware denominator floor for the D-family relative delta-AE loss.
# Set to (1 kcal/mol)^2 in Ha^2 so a compound PBE already describes to
# within ~1 kcal/mol cannot be over-weighted by a near-zero denominator. The
# 627.5094740631 kcal/mol-per-Ha factor is CODATA-2018 (matches
# domain.KCAL_PER_HA).
_DELTA_TGT_FLOOR_HA2 = (1.0 / 627.5094740631) ** 2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOSS_REGISTRY: dict[str, type["AlecLoss"]] = {}


def register_loss(name: str):
    """Decorator: register a loss class under `name`."""
    def wrapper(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return wrapper


def make_loss(name: str, **kwargs) -> "AlecLoss":
    """Construct a registered loss by name, forwarding all kwargs."""
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Unknown loss {name!r}; available: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)


def list_losses() -> list[str]:
    """Return sorted list of registered loss names."""
    return sorted(LOSS_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AlecLoss(eqx.Module, abc.ABC):
    """Stateless loss for AlecGGAModel. Returns (scalar_loss, aux_dict)."""
    registry_name: ClassVar[str] = ""
    required_mol_keys: ClassVar[tuple[str, ...]] = ()
    required_batch_keys: ClassVar[tuple[str, ...]] = ()

    atom_mol_idx: tuple[tuple[str, int], ...] = eqx.field(static=True)
    compound_idx: tuple[int, ...] = eqx.field(static=True)
    mol_names: tuple[str, ...] = eqx.field(static=True)
    compositions: tuple[tuple[tuple[str, int], ...], ...] = eqx.field(static=True)
    w_atomic: float = eqx.field(default=0.01, static=True)

    @abc.abstractmethod
    def __call__(self, model, batch) -> tuple[jnp.ndarray, dict]: ...

    @abc.abstractmethod
    def compute_components(self, model, batch, relative=False) -> dict[str, jnp.ndarray]:
        """Return individual loss components as a dict.

        Keys match the aux dict keys returned by __call__.
        Each term includes its own baseline multiplier (w_atomic, dm_weight,
        density_weight), but the balancing strategy controls how components
        are combined into the total loss.
        """
        ...

    @staticmethod
    def build_indices(molecules, *, require_compound: bool = True):
        """Build hashable (atom_mol_idx, compound_idx, mol_names, compositions).

        ``atom_mol_idx`` is keyed by element symbol and PREFERS neutral
        (charge=0) single-atom MoleculeSpecs over cations/anions. The
        downstream consumer (``_atomic_reg``) compares ``E_NN[idx]``
        against the neutral atom anchor in ``atom_energies[Z]``; pointing
        ``atom_map['Li']`` at a Li+ cation would train the network's
        cation energy toward the neutral Chakravorty value, biasing
        by the ionization-potential magnitude (~5 eV for Li). Mixed-pool
        specs that combine an IP13 pair with the species' neutral atom
        (e.g., HLi's Li anchor + Li_IP's neutral Li and Li+) hit this
        case.

        ``require_compound=False`` permits L5GradnormVxcStep7 to operate
        on specs containing no polyatomic species (BH76- or IP13-only
        subsets), where the AE channel sensibly contributes zero and the
        BH76/IP13 channels carry the loss.
        """
        atom_map: dict[str, int] = {}
        for i, m in enumerate(molecules):
            comp = dict(m.atom_composition)
            if sum(comp.values()) != 1:
                continue
            symbol = next(iter(comp))
            charge_i = int(m.charge)
            if symbol not in atom_map:
                atom_map[symbol] = i
            elif charge_i == 0 and int(molecules[atom_map[symbol]].charge) != 0:
                # Replace a non-neutral entry with the neutral one.
                atom_map[symbol] = i
            # Else: keep existing (already neutral, or both non-neutral,
            # in which case the first one observed wins, which is harmless
            # because dedup-by-(name, charge, spin) prevents duplicates).
        atom_mol_idx = tuple(sorted(atom_map.items()))
        compound_idx = tuple(
            i for i, m in enumerate(molecules)
            if sum(dict(m.atom_composition).values()) > 1
        )
        if not compound_idx and require_compound:
            raise ValueError(
                "Loss requires at least one compound molecule (atom_composition "
                f"sum > 1); got only atomic molecules: {[m.name for m in molecules]}"
            )
        mol_names = tuple(m.name for m in molecules)
        compositions = tuple(tuple(m.atom_composition) for m in molecules)
        return atom_mol_idx, compound_idx, mol_names, compositions

    @staticmethod
    def _validate_static_float(name: str, value) -> None:
        """Reject non-scalar types on static float fields."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"AlecLoss field {name!r} must be a plain Python int/float, "
                f"got {type(value).__name__} = {value!r}"
            )

    @staticmethod
    def _validate_static_bool(name: str, value) -> None:
        """Reject non-bool types on static bool fields."""
        if not isinstance(value, bool):
            raise TypeError(
                f"AlecLoss field {name!r} must be a plain Python bool, "
                f"got {type(value).__name__} = {value!r}"
            )


# ---------------------------------------------------------------------------
# Shared helpers (used inside loss __call__ bodies)
# ---------------------------------------------------------------------------

def _compute_energies(model, mol_data, N, solver_config=None):
    """Per-molecule NN total energies for the energy-loss term, dispatched on
    the solver MODE via :func:`xcquinox.alec.oneshot.total_energy_for_solver`:

    * ``FULL`` -> the self-consistent ``run_scf(...).total_energy`` (a coherent
      fixed point; backprop flows through the SCF cycles). This is the
      DFS/dpyscf self-consistent training target and is what evaluation
      (``TotalEnergyMetric`` / ``AtomizationEnergyMetric``) now measures under
      FULL, so training optimizes exactly what evaluation measures.
    * ``ONESHOT`` / ``FIXED_J`` / ``None`` -> the one-shot fixed-density
      functional ``E_total = E_non_xc[ρ_PBE] + E_xc^NN[ρ_PBE]``. FIXED_J stays
      one-shot on purpose: its run_scf energy is an incoherent J-pinned hybrid
      (see ``total_energy_for_solver``).
    """
    override = get_energy_override("scalar")
    if override is not None:
        # De-fused gradient pass: the per-molecule energies were computed
        # outside this graph and injected (see xcquinox.alec.defused_grad).
        return override
    return jnp.stack([
        total_energy_for_solver(model, mol_data[i], solver_config)
        for i in range(N)
    ])


def _compute_energy_trajectories(model, mol_data, N, solver_config=None):
    """Per-molecule SCF-energy TAIL trajectories, shape ``(N, T)``, for the DFS
    per-step tail-weighted energy loss. Each row is the convergence-tail
    energies from :func:`energy_trajectory_for_solver` (``T = min(max_cycles,
    scf_loss_tail)`` when the tail is enabled). All species share one
    ``solver_config`` so ``T`` is uniform and stackable. When the tail is
    disabled this is ``(N, 1)`` carrying the same scalar as
    :func:`_compute_energies` (the loss then reduces to the final-step form)."""
    override = get_energy_override("trajectory")
    if override is not None:
        # De-fused gradient pass: the per-molecule energy trajectories were
        # computed outside this graph and injected (see defused_grad).
        return override
    return jnp.stack([
        energy_trajectory_for_solver(model, mol_data[i], solver_config)
        for i in range(N)
    ])


def _ae_from_atoms(E_mol, comp_dict, atom_energies):
    """Positive-for-bound atomization energy from a fixed atomic anchor dict.

    AE = Σ n_Z · atom_energies[Z] - E_mol

    The atom anchor is a caller-supplied dict. Under the active
    ``dfs_step7`` domain profile this dict carries the Chakravorty 1993
    exact non-relativistic atomic totals (NOT PBE-consistent values).
    Using a fixed anchor rather than NN-predicted atomic totals is what
    lets the training loss and AtomizationEnergyMetric measure the same
    quantity.
    """
    return sum(n * atom_energies[Z] for Z, n in comp_dict.items()) - E_mol


def _atomic_reg(E_nn, atom_mol_idx_dict, atom_energies, step_w2=None):
    """Weak atomic regularization toward the caller-supplied atom anchor dict.

    Returns the MEAN squared relative error over the anchored atoms so the
    channel scale is independent of how many atoms are in the batch, matching
    the other mean-reduced channels (_ae_losses, _vxc_term, _grid_term).

    ``step_w2`` (the DFS per-SCF-step weights squared, shape ``(T,)``) enables
    the tail-weighted form: ``E_nn`` rows are then ``(T,)`` SCF-energy
    trajectories and each atom's squared error is reduced over the step axis as
    ``mean(step_w2 * diff^2)``. ``None`` -> the byte-identical scalar form.
    """
    if not atom_mol_idx_dict:
        return jnp.array(0.0)
    terms = []
    for Z in atom_mol_idx_dict:
        diff_sq = (E_nn[atom_mol_idx_dict[Z]] - atom_energies[Z]) ** 2
        if step_w2 is not None:
            diff_sq = jnp.mean(step_w2 * diff_sq)
        terms.append(diff_sq / (atom_energies[Z] ** 2 + 1e-8))
    return jnp.mean(jnp.stack(terms))


def _ae_losses(E_nn, compound_idx, comp_dicts, mol_names, targets, atom_energies,
               step_w2=None):
    """A-family: relative squared AE error per compound.

    AE is computed via `_ae_from_atoms` so the anchor dict is the same
    one that AtomizationEnergyMetric consumes at evaluation time.

    The relative normalization uses a scale-aware floor
    ``max(tgt**2, _DELTA_TGT_FLOOR_HA2)`` (mirroring `_delta_losses`)
    rather than the additive ``tgt**2 + 1e-8``: a near-zero AE target
    would otherwise inflate that compound's loss/gradient without bound
    (dfs_step7 forensics: Na2's 0.0273 Ha target inflated its channel
    ~1340x vs CO2-class targets, pegging the grad clip and dominating
    training). The floor guards the tgt->0 singularity; cross-compound
    gradient equalization proper comes from the reaction-form AE channel
    (absolute residuals), not from this floor.
    """
    terms = []
    for i in compound_idx:
        ae = _ae_from_atoms(E_nn[i], comp_dicts[i], atom_energies)
        tgt = targets[mol_names[i]]
        sq = (ae - tgt) ** 2
        if step_w2 is not None:
            # tail mode: ae is a (T,) SCF trajectory -> DFS weighted-MSE over
            # the step axis before the relative normalization.
            sq = jnp.mean(step_w2 * sq)
        terms.append(sq / jnp.maximum(tgt ** 2, _DELTA_TGT_FLOOR_HA2))
    return jnp.mean(jnp.stack(terms))


def _delta_losses(E_nn, mol_data, compound_idx, comp_dicts, mol_names, targets,
                  atom_energies, step_w2=None):
    """D-family: relative squared delta-AE error per compound.

    Both the NN and PBE atomization energies are computed from the same
    `atom_energies` anchor dict via `_ae_from_atoms`, so the atom-sum
    term cancels in `delta_nn = ae_nn - ae_pbe = E_pbe[mol] - E_nn[mol]`.
    The anchor dict still appears in `delta_tgt = target_AE - ae_pbe`,
    so the D-family target is a function of the anchor dict even though
    the NN/PBE delta itself is not.
    """
    # The relative normalization divides the squared delta error by a
    # SCALE-AWARE floor ``max(delta_tgt**2, _DELTA_TGT_FLOOR_HA2)`` rather
    # than an additive ``delta_tgt**2 + 1e-8``. ``delta_tgt`` is how much PBE
    # is off by; a compound PBE already nails (delta_tgt -> 0) would, under the
    # 1e-8 additive floor, have its error divided by ~1e-8 and be over-weighted
    # ~1e8x. The floor caps the denominator at (1 kcal/mol)^2, so near-exact-PBE
    # compounds cannot dominate the loss. No-op for pools whose PBE AE
    # errors ~1e-2..1e-1 Ha (delta_tgt**2 >> floor); matters only once the
    # pool adds compounds PBE describes near-exactly.
    terms = []
    for i in compound_idx:
        ae_nn = _ae_from_atoms(E_nn[i], comp_dicts[i], atom_energies)
        ae_pbe = _ae_from_atoms(mol_data[i]["E_pbe"], comp_dicts[i], atom_energies)
        delta_nn = ae_nn - ae_pbe
        delta_tgt = targets[mol_names[i]] - ae_pbe
        sq = (delta_nn - delta_tgt) ** 2
        if step_w2 is not None:
            # tail mode: delta_nn is a (T,) SCF trajectory -> DFS weighted-MSE.
            sq = jnp.mean(step_w2 * sq)
        terms.append(sq / jnp.maximum(delta_tgt ** 2, _DELTA_TGT_FLOOR_HA2))
    return jnp.mean(jnp.stack(terms))


def _dm_term(model, mol_data, iter_idx, solver_config=None, relative=False):
    """DM matching: Frobenius^2 mean-squared error per element.

    Absolute mode normalizes by the total number of DM elements
    (``n_ao^2`` for RKS; ``2 * n_ao^2`` for UKS, both spin channels).
    This yields a per-element scale invariant under RKS<->UKS, matching
    ``_vxc_term``'s convention; normalizing the UKS branch by ``n_ao^2``
    only would be off by a factor of 2 vs RKS and inconsistent with
    ``_vxc_term``.
    """
    terms = []
    n_skipped = 0
    n_total = 0
    for i in iter_idx:
        n_total += 1
        dm_ref = mol_data[i]["dm_target"]
        if dm_ref is None:
            n_skipped += 1
            continue
        dm_nn = dm_prediction_for_loss(model, mol_data[i], solver_config)
        dm_ref_arr = jnp.asarray(dm_ref)
        err = jnp.sum((dm_nn - dm_ref_arr) ** 2)
        if relative:
            err = err / (jnp.sum(dm_ref_arr ** 2) + 1e-8)
        else:
            # Per-element MSE: divide by total element count of dm_ref.
            # RKS: n_ao*n_ao; UKS (shape (2, n_ao, n_ao)): 2*n_ao*n_ao.
            #
            # Use ``math.prod`` over the static shape tuple instead of
            # ``int(jnp.prod(jnp.array(dm_ref_arr.shape)))``: under
            # ``eqx.filter_jit`` the latter returns a traced scalar and
            # ``int(...)`` raises ``ConcretizationTypeError``. ``shape``
            # is always a tuple of concrete Python ints (jit does not
            # trace shapes), so ``math.prod`` is both jit-safe and faster.
            n_elems = math.prod(dm_ref_arr.shape)
            err = err / float(n_elems)
        terms.append(err)
    if n_skipped:
        warnings.warn(
            f"_dm_term: {n_skipped} of {n_total} mol(s) had "
            f"dm_target=None and were skipped; dm channel may be zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    return jnp.mean(jnp.stack(terms)) if terms else jnp.array(0.0)


def _grid_term(model, mol_data, iter_idx, solver_config=None, relative=False,
               per_electron=False):
    """Grid density matching: weighted L2, normalized absolutely or relatively.

    ``per_electron=True`` (absolute mode only) divides the weighted-L2
    integral by N_e^2 with N_e = sum_g w_g rho_ref(g), making the channel
    INTENSIVE so large/many-electron species cannot dominate a multi-species
    loss by size alone. This is the dpyscf density-loss normalization
    (Dick & Fernandez-Serra 2021, og_dpyscf/ogdpyscf/losses.py:62:
    ``drho = sqrt(sum((rho-rho_ref)^2 w) / n_elec^2)``, then MSE vs 0 ==
    ``sum w (drho)^2 / N^2``); deviation: dpyscf normalizes per spin channel
    by N_sigma^2 (og_dpyscf/ogdpyscf/losses.py:70-73), we carry a spin-summed
    density so the total N_e^2 is used.
    """
    terms = []
    n_skipped = 0
    n_total = 0
    for i in iter_idx:
        n_total += 1
        rho_ref = mol_data[i]["rho_ref_grid"]
        if rho_ref is None:
            n_skipped += 1
            continue
        rho_nn = grid_density_for_loss(model, mol_data[i], solver_config)
        w = mol_data[i]["grid_weights"]
        err = jnp.sum(w * (rho_nn - rho_ref) ** 2)
        if relative:
            err = err / (jnp.sum(w * rho_ref ** 2) + 1e-8)
        elif per_electron:
            n_e = jnp.sum(w * rho_ref)
            err = err / (n_e ** 2 + 1e-8)
        terms.append(err)
    if n_skipped:
        warnings.warn(
            f"_grid_term: {n_skipped} of {n_total} mol(s) had "
            f"rho_ref_grid=None and were skipped; rho channel may be zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    return jnp.mean(jnp.stack(terms)) if terms else jnp.array(0.0)


def _vxc_term(model, mol_data, iter_idx, relative=False):
    """V_xc matching: Frobenius^2 of (V_xc^NN - V_xc^ref).

    Normalized by n_ao^2 (absolute) or ||V_xc^ref||_F^2 (relative).
    Skips molecules where vxc_ref is None.

    Supports both RKS references (shape ``(n_ao, n_ao)``) and UKS references
    (shape ``(2, n_ao, n_ao)``). For UKS, the NN's spin-resolved V_xc is
    constructed via :func:`_uks_spin_resolved_vxc` with the per-channel feature
    blocks of diag(P_sigma, P_sigma), and the squared error is summed across
    both spin channels.
    """
    terms = []
    n_skipped = 0
    n_total = 0
    for i in iter_idx:
        n_total += 1
        vxc_ref = mol_data[i]["vxc_ref"]
        if vxc_ref is None:
            n_skipped += 1
            continue
        vxc_ref_arr = jnp.asarray(vxc_ref)
        features = assemble_descriptor_features(model.descriptors, mol_data[i])

        if vxc_ref_arr.ndim == 3:  # UKS: (2, n_ao, n_ao)
            # Exchange channels take the block of their own doubled density
            # diag(P_sigma, P_sigma); correlation takes the total block.
            vxc_nn_a, vxc_nn_b = _uks_spin_resolved_vxc(
                model, mol_data[i],
                assemble_descriptor_features(model.descriptors, mol_data[i],
                                             spin_channel=0),
                assemble_descriptor_features(model.descriptors, mol_data[i],
                                             spin_channel=1),
                features,
            )
            err = jnp.sum((vxc_nn_a - vxc_ref_arr[0]) ** 2) \
                + jnp.sum((vxc_nn_b - vxc_ref_arr[1]) ** 2)
            if relative:
                err = err / (jnp.sum(vxc_ref_arr ** 2) + 1e-8)
            else:
                n_ao = vxc_ref_arr.shape[-1]
                # Two spin channels -> normalize by 2 * n_ao^2.
                err = err / (2 * n_ao * n_ao)
        else:  # RKS: (n_ao, n_ao)
            vxc_nn = compute_vxc_nn(
                model,
                mol_data[i]["rho_grid"],
                mol_data[i]["sigma_grid"],
                features,
                mol_data[i]["ao_grid"],
                mol_data[i]["grid_weights"],
                nabla_rho=mol_data[i].get("nabla_rho_grid"),
                ao_grad=mol_data[i].get("ao_grid_deriv"),
            )
            err = jnp.sum((vxc_nn - vxc_ref_arr) ** 2)
            if relative:
                err = err / (jnp.sum(vxc_ref_arr ** 2) + 1e-8)
            else:
                n_ao = vxc_ref_arr.shape[-1]
                err = err / (n_ao * n_ao)
        terms.append(err)
    if n_skipped:
        warnings.warn(
            f"_vxc_term: {n_skipped} of {n_total} mol(s) had "
            f"vxc_ref=None and were skipped; vxc channel may be zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    return jnp.mean(jnp.stack(terms)) if terms else jnp.array(0.0)


def _anchor_term(model, sample, weight: float) -> jnp.ndarray:
    """PBE-anchor loss: weight * mean((F_x_nn - F_x_PBE)^2) on a fixed sample."""
    if sample is None or weight == 0.0:
        return jnp.array(0.0)
    from xcquinox.alec.pbe_anchor import pbe_anchor_loss
    from xcquinox.alec.oneshot import _nn_fx_local_uks
    def _nn_fx(m, rho_alpha, rho_beta, s_vals):
        return _nn_fx_local_uks(m, rho_alpha, rho_beta, s_vals)
    return pbe_anchor_loss(model, sample, weight, _nn_fx)


def _rxn_residual_term(
    e_nn: jnp.ndarray,
    coeffs: jnp.ndarray,
    e_rxn_ref: jnp.ndarray,
    relative: bool = False,
    step_w2: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Squared residual of a generic reaction energy / barrier height.

    e_nn   : ``(n_species,)`` NN total energies for each species in the
             reaction, OR ``(n_species, T)`` per-SCF-step trajectories when the
             DFS tail loss is on (``step_w2`` provided).
    coeffs : (n_species,) signed stoichiometric coefficients (negative for
             reactants, positive for products)
    e_rxn_ref : scalar reference reaction-energy or barrier-height value
    step_w2 : ``(T,)`` per-step DFS weights squared, or ``None`` for the
             byte-identical final-step-only scalar form. When set, the per-step
             reaction-energy residual is reduced as DFS's ``mean(w^2 r^2)``
             over the tail, directly penalizing a non-converging SCF.

    Returns: ``(E_rxn_NN - E_rxn_ref)^2``; when ``relative`` is set, the
    relative form ``(.)^2 / (e_rxn_ref^2 + 1e-8)``: matching the AE/vxc/rho
    channels' normalization so that under ``loss_metric='relative'`` ALL five
    GradNorm channels measure the same dimensionless quantity (otherwise the
    BH76 channel would be absolute Ha^2 while the others are relative).

    Used by the BH76 task channel of L5_gradnorm_vxc_step7. In Dick &
    Fernandez-Serra PRB 104 L161109 (2021) the 0.01 factor is lambda_E,
    the weight on total energies L_E; BH76/atomization energies enter L_RE
    at weight 1 and density L_n at weight 20. Channel weighting depends on
    the update scheme: under ``update_scheme="batched"`` GradNorm (Chen et
    al. 2018, arXiv:1711.02257; alpha=1.5 default) discovers task weights
    adaptively, but the production per-molecule scheme (the cluster
    default) ignores the balancer and applies the fixed density-dominant
    weights {AE 1, BH76 1, IP13 1, vxc 1, rho 20}
    (train._DEFAULT_CHANNEL_WEIGHTS) -- the Letter's 1/20 structure. See
    notebooks/analysis/LOSS_PRIMER.md.
    """
    if step_w2 is None:
        e_rxn = jnp.sum(coeffs * e_nn)
        sq = (e_rxn - e_rxn_ref) ** 2
    else:
        # e_nn: (n_species, T) -> per-step reaction energy (T,), then DFS
        # weighted mean-squared residual over the SCF tail.
        e_rxn = jnp.sum(coeffs[:, None] * e_nn, axis=0)
        sq = jnp.mean(step_w2 * (e_rxn - e_rxn_ref) ** 2)
    if relative:
        return sq / (e_rxn_ref ** 2 + 1e-8)
    return sq


def _ip_residual_term(
    e_cation: jnp.ndarray,
    e_neutral: jnp.ndarray,
    ip_ref: jnp.ndarray,
    relative: bool = False,
    step_w2: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Squared residual of an ionization potential. IP = E_cation - E_neutral.

    When ``relative`` is set, the relative form ``(.)^2 / (ip_ref^2 + 1e-8)``,
    consistent with the other channels under ``loss_metric='relative'``.
    ``step_w2`` (``(T,)``) enables the DFS tail form: ``e_cation``/``e_neutral``
    are ``(T,)`` SCF trajectories and the residual is reduced as
    ``mean(w^2 r^2)`` over the tail; ``None`` -> byte-identical scalar form.

    Used by the IP13 task channel of L5_gradnorm_vxc_step7.
    """
    resid = e_cation - e_neutral - ip_ref
    if step_w2 is None:
        sq = resid ** 2
    else:
        sq = jnp.mean(step_w2 * resid ** 2)
    if relative:
        return sq / (ip_ref ** 2 + 1e-8)
    return sq


# ---------------------------------------------------------------------------
# Concrete losses
# ---------------------------------------------------------------------------

@register_loss("A_atomization")
class AtomizationLoss(AlecLoss):
    registry_name: ClassVar[str] = "A_atomization"
    required_mol_keys: ClassVar[tuple[str, ...]] = ()
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    molecules_only: bool = eqx.field(default=True, static=True)
    solver_config: object | None = eqx.field(default=None, static=True)
    vxc_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_sample: object | None = eqx.field(default=None, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 molecules_only: bool = True,
                 solver_config=None,
                 vxc_weight: float = 0.0,
                 pbe_anchor_weight: float = 0.0,
                 pbe_anchor_sample=None):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_bool("molecules_only", molecules_only)
        self._validate_static_float("vxc_weight", vxc_weight)
        self._validate_static_float("pbe_anchor_weight", pbe_anchor_weight)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.molecules_only = molecules_only
        self.solver_config = solver_config
        self.vxc_weight = vxc_weight
        self.pbe_anchor_weight = pbe_anchor_weight
        self.pbe_anchor_sample = pbe_anchor_sample

    def compute_components(self, model, batch, relative=False):
        atom_idx = dict(self.atom_mol_idx)
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N, solver_config=self.solver_config)
        loss_energy = _ae_losses(E_nn, self.compound_idx, comp_dicts,
                                 self.mol_names, targets, atom_energies)
        atomic_reg = self.w_atomic * _atomic_reg(E_nn, atom_idx, atom_energies)
        components = {"loss_energy": loss_energy, "atomic_reg": atomic_reg}
        if self.vxc_weight > 0:
            # Gate on molecules_only (default True) for consistency with
            # B/C/D2/D3. Atoms typically have vxc_ref=None and are skipped
            # inside _vxc_term anyway, but the explicit gate makes the API
            # uniform across all 6 losses.
            vxc_idx = self.compound_idx if self.molecules_only else tuple(range(N))
            components["loss_vxc"] = self.vxc_weight * _vxc_term(
                model, mol_data, vxc_idx, relative=relative,
            )
        if self.pbe_anchor_weight > 0.0 and self.pbe_anchor_sample is not None:
            components["loss_anchor"] = _anchor_term(
                model, self.pbe_anchor_sample, self.pbe_anchor_weight,
            )
        return components

    def __call__(self, model, batch):
        components = self.compute_components(model, batch)
        total = sum(components.values())
        return total, components


@register_loss("B_atomization_plus_dm")
class AtomizationPlusDMLoss(AlecLoss):
    registry_name: ClassVar[str] = "B_atomization_plus_dm"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("dm_target",)
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    dm_weight: float = eqx.field(default=0.1, static=True)
    molecules_only: bool = eqx.field(default=True, static=True)
    solver_config: object | None = eqx.field(default=None, static=True)
    vxc_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_sample: object | None = eqx.field(default=None, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 dm_weight: float = 0.1, molecules_only: bool = True,
                 solver_config=None, vxc_weight: float = 0.0,
                 pbe_anchor_weight: float = 0.0,
                 pbe_anchor_sample=None):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_float("dm_weight", dm_weight)
        self._validate_static_bool("molecules_only", molecules_only)
        self._validate_static_float("vxc_weight", vxc_weight)
        self._validate_static_float("pbe_anchor_weight", pbe_anchor_weight)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.dm_weight = dm_weight
        self.molecules_only = molecules_only
        self.solver_config = solver_config
        self.vxc_weight = vxc_weight
        self.pbe_anchor_weight = pbe_anchor_weight
        self.pbe_anchor_sample = pbe_anchor_sample

    def compute_components(self, model, batch, relative=False):
        atom_idx = dict(self.atom_mol_idx)
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N, solver_config=self.solver_config)
        loss_energy = _ae_losses(E_nn, self.compound_idx, comp_dicts,
                                 self.mol_names, targets, atom_energies)
        atomic_reg = self.w_atomic * _atomic_reg(E_nn, atom_idx, atom_energies)
        iter_idx = self.compound_idx if self.molecules_only else tuple(range(N))
        dm_loss = self.dm_weight * _dm_term(model, mol_data, iter_idx, solver_config=self.solver_config, relative=relative)
        components = {"loss_energy": loss_energy, "atomic_reg": atomic_reg, "loss_dm": dm_loss}
        if self.vxc_weight > 0:
            vxc_idx = self.compound_idx if self.molecules_only else tuple(range(N))
            components["loss_vxc"] = self.vxc_weight * _vxc_term(
                model, mol_data, vxc_idx, relative=relative,
            )
        if self.pbe_anchor_weight > 0.0 and self.pbe_anchor_sample is not None:
            components["loss_anchor"] = _anchor_term(
                model, self.pbe_anchor_sample, self.pbe_anchor_weight,
            )
        return components

    def __call__(self, model, batch):
        components = self.compute_components(model, batch)
        total = sum(components.values())
        return total, components


@register_loss("C_atomization_plus_grid")
class AtomizationPlusGridLoss(AlecLoss):
    registry_name: ClassVar[str] = "C_atomization_plus_grid"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("rho_ref_grid",)
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    density_weight: float = eqx.field(default=0.1, static=True)
    molecules_only: bool = eqx.field(default=True, static=True)
    solver_config: object | None = eqx.field(default=None, static=True)
    vxc_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_sample: object | None = eqx.field(default=None, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 density_weight: float = 0.1, molecules_only: bool = True,
                 solver_config=None, vxc_weight: float = 0.0,
                 pbe_anchor_weight: float = 0.0,
                 pbe_anchor_sample=None):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_float("density_weight", density_weight)
        self._validate_static_bool("molecules_only", molecules_only)
        self._validate_static_float("vxc_weight", vxc_weight)
        self._validate_static_float("pbe_anchor_weight", pbe_anchor_weight)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.density_weight = density_weight
        self.molecules_only = molecules_only
        self.solver_config = solver_config
        self.vxc_weight = vxc_weight
        self.pbe_anchor_weight = pbe_anchor_weight
        self.pbe_anchor_sample = pbe_anchor_sample

    def compute_components(self, model, batch, relative=False):
        atom_idx = dict(self.atom_mol_idx)
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N, solver_config=self.solver_config)
        loss_energy = _ae_losses(E_nn, self.compound_idx, comp_dicts,
                                 self.mol_names, targets, atom_energies)
        atomic_reg = self.w_atomic * _atomic_reg(E_nn, atom_idx, atom_energies)
        iter_idx = self.compound_idx if self.molecules_only else tuple(range(N))
        grid_loss = self.density_weight * _grid_term(model, mol_data, iter_idx, solver_config=self.solver_config, relative=relative)
        components = {"loss_energy": loss_energy, "atomic_reg": atomic_reg, "loss_grid": grid_loss}
        if self.vxc_weight > 0:
            vxc_idx = self.compound_idx if self.molecules_only else tuple(range(N))
            components["loss_vxc"] = self.vxc_weight * _vxc_term(
                model, mol_data, vxc_idx, relative=relative,
            )
        if self.pbe_anchor_weight > 0.0 and self.pbe_anchor_sample is not None:
            components["loss_anchor"] = _anchor_term(
                model, self.pbe_anchor_sample, self.pbe_anchor_weight,
            )
        return components

    def __call__(self, model, batch):
        components = self.compute_components(model, batch)
        total = sum(components.values())
        return total, components


@register_loss("D1_delta_ae")
class DeltaAELoss(AlecLoss):
    registry_name: ClassVar[str] = "D1_delta_ae"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("E_pbe",)
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    molecules_only: bool = eqx.field(default=True, static=True)
    solver_config: object | None = eqx.field(default=None, static=True)
    vxc_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_sample: object | None = eqx.field(default=None, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 molecules_only: bool = True,
                 solver_config=None,
                 vxc_weight: float = 0.0,
                 pbe_anchor_weight: float = 0.0,
                 pbe_anchor_sample=None):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_bool("molecules_only", molecules_only)
        self._validate_static_float("vxc_weight", vxc_weight)
        self._validate_static_float("pbe_anchor_weight", pbe_anchor_weight)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.molecules_only = molecules_only
        self.solver_config = solver_config
        self.vxc_weight = vxc_weight
        self.pbe_anchor_weight = pbe_anchor_weight
        self.pbe_anchor_sample = pbe_anchor_sample

    def compute_components(self, model, batch, relative=False):
        atom_idx = dict(self.atom_mol_idx)
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N, solver_config=self.solver_config)
        loss_delta = _delta_losses(E_nn, mol_data, self.compound_idx, comp_dicts,
                                   self.mol_names, targets, atom_energies)
        atomic_reg = self.w_atomic * _atomic_reg(E_nn, atom_idx, atom_energies)
        components = {"loss_delta": loss_delta, "atomic_reg": atomic_reg}
        if self.vxc_weight > 0:
            # Gate on molecules_only, matching B/C/D2/D3.
            vxc_idx = self.compound_idx if self.molecules_only else tuple(range(N))
            components["loss_vxc"] = self.vxc_weight * _vxc_term(
                model, mol_data, vxc_idx, relative=relative,
            )
        if self.pbe_anchor_weight > 0.0 and self.pbe_anchor_sample is not None:
            components["loss_anchor"] = _anchor_term(
                model, self.pbe_anchor_sample, self.pbe_anchor_weight,
            )
        return components

    def __call__(self, model, batch):
        components = self.compute_components(model, batch)
        total = sum(components.values())
        return total, components


@register_loss("D2_delta_ae_plus_dm")
class DeltaAEPlusDMLoss(AlecLoss):
    registry_name: ClassVar[str] = "D2_delta_ae_plus_dm"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("E_pbe", "dm_target")
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    dm_weight: float = eqx.field(default=0.1, static=True)
    molecules_only: bool = eqx.field(default=True, static=True)
    solver_config: object | None = eqx.field(default=None, static=True)
    vxc_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_sample: object | None = eqx.field(default=None, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 dm_weight: float = 0.1, molecules_only: bool = True,
                 solver_config=None, vxc_weight: float = 0.0,
                 pbe_anchor_weight: float = 0.0,
                 pbe_anchor_sample=None):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_float("dm_weight", dm_weight)
        self._validate_static_bool("molecules_only", molecules_only)
        self._validate_static_float("vxc_weight", vxc_weight)
        self._validate_static_float("pbe_anchor_weight", pbe_anchor_weight)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.dm_weight = dm_weight
        self.molecules_only = molecules_only
        self.solver_config = solver_config
        self.vxc_weight = vxc_weight
        self.pbe_anchor_weight = pbe_anchor_weight
        self.pbe_anchor_sample = pbe_anchor_sample

    def compute_components(self, model, batch, relative=False):
        atom_idx = dict(self.atom_mol_idx)
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N, solver_config=self.solver_config)
        loss_delta = _delta_losses(E_nn, mol_data, self.compound_idx, comp_dicts,
                                   self.mol_names, targets, atom_energies)
        atomic_reg = self.w_atomic * _atomic_reg(E_nn, atom_idx, atom_energies)
        iter_idx = self.compound_idx if self.molecules_only else tuple(range(N))
        dm_loss = self.dm_weight * _dm_term(model, mol_data, iter_idx, solver_config=self.solver_config, relative=relative)
        components = {"loss_delta": loss_delta, "atomic_reg": atomic_reg, "loss_dm": dm_loss}
        if self.vxc_weight > 0:
            vxc_idx = self.compound_idx if self.molecules_only else tuple(range(N))
            components["loss_vxc"] = self.vxc_weight * _vxc_term(
                model, mol_data, vxc_idx, relative=relative,
            )
        if self.pbe_anchor_weight > 0.0 and self.pbe_anchor_sample is not None:
            components["loss_anchor"] = _anchor_term(
                model, self.pbe_anchor_sample, self.pbe_anchor_weight,
            )
        return components

    def __call__(self, model, batch):
        components = self.compute_components(model, batch)
        total = sum(components.values())
        return total, components


@register_loss("D3_delta_ae_plus_grid")
class DeltaAEPlusGridLoss(AlecLoss):
    registry_name: ClassVar[str] = "D3_delta_ae_plus_grid"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("E_pbe", "rho_ref_grid")
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    density_weight: float = eqx.field(default=0.1, static=True)
    molecules_only: bool = eqx.field(default=True, static=True)
    solver_config: object | None = eqx.field(default=None, static=True)
    vxc_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_weight: float = eqx.field(default=0.0, static=True)
    pbe_anchor_sample: object | None = eqx.field(default=None, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 density_weight: float = 0.1, molecules_only: bool = True,
                 solver_config=None, vxc_weight: float = 0.0,
                 pbe_anchor_weight: float = 0.0,
                 pbe_anchor_sample=None):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_float("density_weight", density_weight)
        self._validate_static_bool("molecules_only", molecules_only)
        self._validate_static_float("vxc_weight", vxc_weight)
        self._validate_static_float("pbe_anchor_weight", pbe_anchor_weight)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.density_weight = density_weight
        self.molecules_only = molecules_only
        self.solver_config = solver_config
        self.vxc_weight = vxc_weight
        self.pbe_anchor_weight = pbe_anchor_weight
        self.pbe_anchor_sample = pbe_anchor_sample

    def compute_components(self, model, batch, relative=False):
        atom_idx = dict(self.atom_mol_idx)
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N, solver_config=self.solver_config)
        loss_delta = _delta_losses(E_nn, mol_data, self.compound_idx, comp_dicts,
                                   self.mol_names, targets, atom_energies)
        atomic_reg = self.w_atomic * _atomic_reg(E_nn, atom_idx, atom_energies)
        iter_idx = self.compound_idx if self.molecules_only else tuple(range(N))
        grid_loss = self.density_weight * _grid_term(model, mol_data, iter_idx, solver_config=self.solver_config, relative=relative)
        components = {"loss_delta": loss_delta, "atomic_reg": atomic_reg, "loss_grid": grid_loss}
        if self.vxc_weight > 0:
            vxc_idx = self.compound_idx if self.molecules_only else tuple(range(N))
            components["loss_vxc"] = self.vxc_weight * _vxc_term(
                model, mol_data, vxc_idx, relative=relative,
            )
        if self.pbe_anchor_weight > 0.0 and self.pbe_anchor_sample is not None:
            components["loss_anchor"] = _anchor_term(
                model, self.pbe_anchor_sample, self.pbe_anchor_weight,
            )
        return components

    def __call__(self, model, batch):
        components = self.compute_components(model, batch)
        total = sum(components.values())
        return total, components


# Sanity ceiling (Ha) for frozen reference reaction / atomization / IP energies.
# BH76 + W4-11 + IP13 references for first/second-row chemistry are all well
# under this in Hartree (the largest W4-11 total atomization energies are ~1-2
# Ha; IPs ~0.3-1.6 Ha). A reference above this almost certainly means a
# kcal/mol value was passed WITHOUT the kcal/mol->Ha conversion (a ~627x error;
# e.g. a forgotten /KCAL_PER_HA in the pool builder makes every value huge), so
# we fail loud at construction rather than silently train on it.
_HA_REF_SANITY_MAX = 10.0


def _guard_ref_is_hartree(value, label, kind):
    """Raise if a frozen reference energy looks like kcal/mol, not Hartree."""
    if value is not None and abs(value) > _HA_REF_SANITY_MAX:
        raise ValueError(
            f"{kind} {label!r}: reference energy {value} Ha exceeds the "
            f"{_HA_REF_SANITY_MAX} Ha sanity ceiling, this is almost certainly "
            f"a kcal/mol value passed without the kcal/mol->Ha conversion "
            f"(~627x too large). Convert references to Hartree before building "
            f"the loss (e.g. via KCAL_PER_HA)."
        )


def _freeze_rxn_specs(rxns) -> tuple:
    """Convert a list of BH76 reaction-spec dicts to a hashable tuple-of-tuples.

    Each reaction is normalized to:
      (name, reactants_tuple, products_tuple, coeffs_tuple, e_rxn_ref_or_None)

    Stored as eqx static field so jit cache keys remain stable.
    """
    out = []
    for r in rxns:
        name = r.get("name", "")
        reactants = tuple(r.get("reactants", ()))
        products = tuple(r.get("products", ()))
        coeffs = tuple(float(c) for c in r.get("coeffs", ()))
        e_ref = r.get("e_rxn_ref", None)
        if e_ref is not None:
            e_ref = float(e_ref)
            _guard_ref_is_hartree(e_ref, name, "BH76 reaction")
        out.append((name, reactants, products, coeffs, e_ref))
    return tuple(out)


def _freeze_ip_specs(pairs) -> tuple:
    """Convert a list of IP13 spec dicts to a hashable tuple-of-tuples.

    Each pair is normalized to:
      (name, neutral_label, cation_label, ip_ref_or_None)
    """
    out = []
    for p in pairs:
        name = p.get("name", "")
        neutral = p.get("neutral", "")
        cation = p.get("cation", "")
        ip_ref = p.get("ip_ref", None)
        if ip_ref is not None:
            ip_ref = float(ip_ref)
            _guard_ref_is_hartree(ip_ref, name, "IP13 pair")
        out.append((name, neutral, cation, ip_ref))
    return tuple(out)


@register_loss("L5_gradnorm_vxc_step7")
class L5GradnormVxcStep7(AlecLoss):
    """Step-7 extension of L5_gradnorm_vxc with BH76 + IP13 task channels.

    Five GradNorm task channels (per spec §5b):
      loss_AE   - atomization-energy residuals (existing alec mechanism)
      loss_BH76 - reaction-energy / barrier-height residuals via _rxn_residual_term
      loss_IP13 - ionization-potential residuals via _ip_residual_term
      loss_vxc  - V_xc residual (existing L3/L4/L5 mechanism)
      loss_rho  - grid-density residual (existing)

    These five keys correspond to the five GradNorm task channels. Each
    appears as a key in ``compute_components`` so the GradNorm balancer
    in ``train._run_gradnorm_loop`` (xcquinox/alec/train.py:408) treats
    them as five independent tasks.

    In Dick & Fernandez-Serra PRB 104 L161109 (2021) the 0.01 factor is
    lambda_E, the weight on total energies L_E; BH76/atomization energies
    enter L_RE at weight 1 and density L_n at weight 20. How the five
    channels are weighted depends on the training loop: the batched scheme
    balances them with GradNorm (Chen et al. 2018, arXiv:1711.02257;
    alpha=1.5 default at xcquinox/alec/balancing.py:55), while the
    per-molecule scheme -- the cluster default used by every dfs_step7
    production run -- ignores the balancer and applies the fixed
    density-dominant channel weights {AE 1, BH76 1, IP13 1, vxc 1, rho 20}
    (train._DEFAULT_CHANNEL_WEIGHTS), i.e. the Letter's lambda_RE=1 /
    lambda_n=20 structure; the vxc_weight/density_weight pre-scales below
    are forced to 1.0 there. See notebooks/analysis/LOSS_PRIMER.md.

    Constructor arguments
    ---------------------
    molecules : sequence of MoleculeSpec
        The full species set referenced by AE, BH76, and IP13 channels.
        Names must include all reactants/products for BH76 reactions and
        both neutral/cation labels for IP13 pairs.
    bh76_reactions : list[dict] | None
        Each dict has keys ``name``, ``reactants``, ``products``,
        ``coeffs`` (signed coefficients aligned with ``reactants +
        products``), and ``e_rxn_ref``. Empty list disables the channel
        (BH76 contribution becomes 0).
    ip13_pairs : list[dict] | None
        Each dict has keys ``name``, ``neutral`` (species name),
        ``cation`` (species name), and ``ip_ref``. Empty list disables
        the channel.
    dm_weight, vxc_weight, density_weight : float
        Per-channel scaling factors applied INSIDE the channel's residual
        before GradNorm reweighting. Channel weight tuning is GradNorm's
        job; these are pre-balancer scale factors only.
    """
    registry_name: ClassVar[str] = "L5_gradnorm_vxc_step7"
    required_mol_keys: ClassVar[tuple[str, ...]] = ()
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    target_kinds: ClassVar[tuple[str, ...]] = ("AE", "BH76", "IP13", "vxc", "rho")

    bh76_reactions: tuple = eqx.field(default=(), static=True)
    ip13_pairs: tuple = eqx.field(default=(), static=True)
    aux_only_names: tuple = eqx.field(default=(), static=True)
    molecules_only: bool = eqx.field(default=True, static=True)
    solver_config: object | None = eqx.field(default=None, static=True)
    vxc_weight: float = eqx.field(default=0.01, static=True)
    density_weight: float = eqx.field(default=0.1, static=True)
    # Per-electron^2 normalization of the density channel (dpyscf
    # losses.py:171 convention; see _grid_term). Default OFF so existing
    # domains are byte-identical; the dfs_step7 v2 sweep enables it.
    density_per_electron: bool = eqx.field(default=False, static=True)
    # Atom-symbol allowlist for `_atomic_reg`. None (default) regularizes
    # every single-atom MoleculeSpec in the spec, kept for back-compat.
    # Set to ("H", "Li") to mirror the
    # Dick & Fernandez-Serra 2021 SI §II atomic-density references; in
    # that case `_atomic_reg` ignores any other atomic species (C, N,
    # O, F, ...) that happen to appear in the spec via IP13 pairs or
    # mixed-pool TrainingPoint constructions.
    regularize_atom_syms: tuple | None = eqx.field(default=None, static=True)

    def __init__(
        self,
        *,
        molecules=None,
        bh76_reactions=None,
        ip13_pairs=None,
        aux_only_names=(),
        w_atomic: float = 0.01,
        molecules_only: bool = True,
        solver_config=None,
        vxc_weight: float = 0.01,
        density_weight: float = 0.1,
        density_per_electron: bool = False,
        regularize_atom_syms=None,
        _smoke_test: bool = False,
        **_unused_kwargs,
    ):
        # Fail-fast on the dead PBE-anchor knob: L5 has NO anchor channel
        # (target_kinds = AE/BH76/IP13/vxc/rho) and step-7 freezes pretraining
        # from step-6, so a positive pbe_anchor_weight would be silently
        # swallowed by **_unused_kwargs and do nothing.
        _anchor_w = _unused_kwargs.get("pbe_anchor_weight", 0.0) or 0.0
        if float(_anchor_w) > 0.0 or _unused_kwargs.get("pbe_anchor_sample") is not None:
            raise ValueError(
                "L5_gradnorm_vxc_step7 has no PBE-anchor channel; "
                "pbe_anchor_weight>0 / pbe_anchor_sample would be silently "
                "ignored. Step-7 freezes pretraining from step-6, so the PBE "
                "anchor is not used here, set pbe_anchor_weight=0."
            )
        # The smoke path is used by registry/contract tests where there is
        # no real training context (no molecules, no batch). It must still
        # initialize all required AlecLoss fields plus the new BH76/IP13
        # fields so eqx.Module field validation passes.
        bh76_frozen = _freeze_rxn_specs(bh76_reactions or ())
        ip13_frozen = _freeze_ip_specs(ip13_pairs or ())
        reg_syms_frozen = (
            tuple(regularize_atom_syms) if regularize_atom_syms is not None else None
        )

        if _smoke_test:
            self.atom_mol_idx = ()
            self.compound_idx = ()
            self.mol_names = ()
            self.compositions = ()
            self.w_atomic = w_atomic
            self.bh76_reactions = bh76_frozen
            self.ip13_pairs = ip13_frozen
            self.aux_only_names = tuple(aux_only_names)
            self.molecules_only = molecules_only
            self.solver_config = solver_config
            self.vxc_weight = vxc_weight
            self.density_weight = density_weight
            self.density_per_electron = density_per_electron
            self.regularize_atom_syms = reg_syms_frozen
            return

        if molecules is None:
            raise ValueError(
                "L5GradnormVxcStep7 requires `molecules` (or use _smoke_test=True "
                "for registry/contract tests)."
            )
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_bool("molecules_only", molecules_only)
        self._validate_static_float("vxc_weight", vxc_weight)
        self._validate_static_float("density_weight", density_weight)
        self._validate_static_bool("density_per_electron", density_per_electron)
        # require_compound=False: a BH76-only or IP13-only subset (no
        # polyatomic species) is a legitimate L5 configuration; the AE
        # channel returns 0 in compute_components and the BH76/IP13
        # channels carry the loss.  Aux filter applied below.
        ami, ci, mn, comp = self.build_indices(
            molecules, require_compound=False)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.aux_only_names = tuple(aux_only_names)
        aux_set = set(self.aux_only_names)
        self.compound_idx = tuple(
            i for i in self.compound_idx
            if self.mol_names[i] not in aux_set
        )
        # Empty compound_idx (after aux filter) is permitted: the AE
        # channel evaluates to 0 in compute_components and the loss
        # remains well-formed via BH76 + IP13 + atomic_reg + vxc + rho.
        # Validation that at least ONE channel has signal happens
        # implicitly, a spec with no compounds, no BH76 reactions, no
        # IP13 pairs, and no Dick atoms would yield a constant zero loss
        # which the GradNorm rebalance step would surface as NaN/Inf
        # gnorms, caller is expected to choose a non-degenerate subset.
        self.w_atomic = w_atomic
        self.bh76_reactions = bh76_frozen
        self.ip13_pairs = ip13_frozen
        self.molecules_only = molecules_only
        self.solver_config = solver_config
        self.vxc_weight = vxc_weight
        self.density_weight = density_weight
        self.density_per_electron = density_per_electron
        self.regularize_atom_syms = reg_syms_frozen

        # Validate that regularize_atom_syms is a subset of the
        # single-atom anchors actually present in atom_mol_idx.  An
        # unmatched symbol would be silently dropped in compute_components,
        # producing zero regularization for that species with no feedback.
        if reg_syms_frozen is not None:
            atom_map_keys = set(dict(ami).keys())
            missing = set(reg_syms_frozen) - atom_map_keys
            if missing:
                raise ValueError(
                    f"regularize_atom_syms contains symbols not present as "
                    f"single-atom MoleculeSpecs in `molecules`: "
                    f"{sorted(missing)}.  "
                    f"Available single-atom symbols: {sorted(atom_map_keys)}"
                )

        # Validate that every BH76 species and every IP13 species is
        # present in the `molecules` set. A missing species would cause
        # silent zeros at compute time.
        mol_name_set = set(self.mol_names)
        for (rname, reactants, products, coeffs, _eref) in self.bh76_reactions:
            for s in (*reactants, *products):
                if s not in mol_name_set:
                    raise ValueError(
                        f"BH76 reaction {rname!r}: species {s!r} not in "
                        f"`molecules` (have {sorted(mol_name_set)})"
                    )
            if len(coeffs) != len(reactants) + len(products):
                raise ValueError(
                    f"BH76 reaction {rname!r}: len(coeffs)={len(coeffs)} "
                    f"!= len(reactants)+len(products)="
                    f"{len(reactants) + len(products)}"
                )
        for (pname, neutral, cation, _ipref) in self.ip13_pairs:
            for s in (neutral, cation):
                if s not in mol_name_set:
                    raise ValueError(
                        f"IP13 pair {pname!r}: species {s!r} not in "
                        f"`molecules` (have {sorted(mol_name_set)})"
                    )

    def _bh76_channel(self, E_nn, relative=False, step_w2=None) -> jnp.ndarray:
        """Mean of squared reaction-energy residuals across BH76 reactions.

        E_NN_total values are looked up from the all-species `E_nn` vector
        by name via `mol_names`. A reaction with `e_rxn_ref=None` is
        skipped (treated as missing reference). If no usable reactions
        remain, returns 0.0 (so the channel contributes nothing under
        GradNorm without crashing). ``relative`` selects the dimensionless
        normalization so this channel is consistent with the others under
        ``loss_metric='relative'``.
        """
        if not self.bh76_reactions:
            return jnp.array(0.0)
        name_to_idx = {n: i for i, n in enumerate(self.mol_names)}
        terms = []
        for (_rname, reactants, products, coeffs, e_ref) in self.bh76_reactions:
            if e_ref is None:
                continue
            species = (*reactants, *products)
            idx = jnp.array([name_to_idx[s] for s in species])
            e_species = E_nn[idx]
            coeffs_arr = jnp.array(coeffs)
            terms.append(_rxn_residual_term(
                e_species, coeffs_arr, jnp.array(e_ref), relative=relative,
                step_w2=step_w2,
            ))
        if not terms:
            return jnp.array(0.0)
        return jnp.mean(jnp.stack(terms))

    def _ip13_channel(self, E_nn, relative=False, step_w2=None) -> jnp.ndarray:
        """Mean of squared IP residuals across IP13 pairs.

        Pairs with `ip_ref=None` are skipped. ``relative`` selects the
        dimensionless normalization (consistent with the other channels under
        ``loss_metric='relative'``).
        """
        if not self.ip13_pairs:
            return jnp.array(0.0)
        name_to_idx = {n: i for i, n in enumerate(self.mol_names)}
        terms = []
        for (_pname, neutral, cation, ip_ref) in self.ip13_pairs:
            if ip_ref is None:
                continue
            e_neutral = E_nn[name_to_idx[neutral]]
            e_cation = E_nn[name_to_idx[cation]]
            terms.append(_ip_residual_term(
                e_cation, e_neutral, jnp.array(ip_ref), relative=relative,
                step_w2=step_w2,
            ))
        if not terms:
            return jnp.array(0.0)
        return jnp.mean(jnp.stack(terms))

    def compute_components(self, model, batch, relative=False):
        """Return the 5 GradNorm task channels as a dict.

        Keys are the GradNorm task channels:
          loss_AE, loss_BH76, loss_IP13, loss_vxc, loss_rho
        """
        if not self.mol_names:
            raise RuntimeError(
                "L5GradnormVxcStep7 was constructed in smoke-test mode; "
                "compute_components requires a real `molecules` set."
            )
        atom_idx = dict(self.atom_mol_idx)
        # Restrict `_atomic_reg` to the user-specified atom set when
        # `regularize_atom_syms` is configured (e.g., the Dick 2021 SI §II
        # H/Li set). When None, regularize every single-atom MoleculeSpec
        # (kept for back-compat).
        if self.regularize_atom_syms is not None:
            allowed = set(self.regularize_atom_syms)
            atom_idx = {sym: i for sym, i in atom_idx.items() if sym in allowed}
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        # DFS tail loss: when enabled, score a quadratic-weighted window of the
        # SCF-energy TAIL (E_nn is then (N, T)) so the energy channels penalize
        # a non-converging SCF rather than one arbitrary final cycle. step_w2 is
        # None when disabled -> the scalar (N,) path, byte-identical to before.
        step_w = scf_loss_tail_weights(self.solver_config)
        if step_w is not None:
            E_nn = _compute_energy_trajectories(
                model, mol_data, N, solver_config=self.solver_config
            )
            step_w2 = step_w ** 2
        else:
            E_nn = _compute_energies(
                model, mol_data, N, solver_config=self.solver_config
            )
            step_w2 = None
        # AE channel: relative squared AE residual + atomic regularization,
        # mirroring AtomizationLoss but bundled into a single channel for
        # GradNorm. atomic_reg is folded into the AE channel because it is
        # a regularizer of the AE quantity, not an independent task.
        # Empty compound_idx -> AE-fitting term is 0 (BH76- / IP13-only
        # subsets); atomic_reg may still be nonzero if any Dick atom is
        # in atom_idx.
        if self.compound_idx:
            loss_ae = _ae_losses(
                E_nn, self.compound_idx, comp_dicts,
                self.mol_names, targets, atom_energies, step_w2=step_w2,
            )
        else:
            loss_ae = jnp.array(0.0)
        atomic_reg = self.w_atomic * _atomic_reg(
            E_nn, atom_idx, atom_energies, step_w2=step_w2,
        )
        loss_ae_total = loss_ae + atomic_reg

        # BH76 + IP13 channels: reaction / IP residuals (Dick 2021 SI II).
        # Pass `relative` so all 5 GradNorm channels share one metric under
        # loss_metric='relative' (else BH76/IP13 stay absolute Ha^2 while
        # AE/vxc/rho are relative, inconsistent quantities into GradNorm).
        loss_bh76 = self._bh76_channel(E_nn, relative=relative, step_w2=step_w2)
        loss_ip13 = self._ip13_channel(E_nn, relative=relative, step_w2=step_w2)

        # vxc + rho channels: existing alec mechanisms.
        iter_idx = self._iter_idx_for_aux_channels()
        loss_vxc = self.vxc_weight * _vxc_term(
            model, mol_data, iter_idx, relative=relative,
        )
        loss_rho = self.density_weight * _grid_term(
            model, mol_data, iter_idx, solver_config=self.solver_config,
            relative=relative, per_electron=self.density_per_electron,
        )

        return {
            "loss_AE": loss_ae_total,
            "loss_BH76": loss_bh76,
            "loss_IP13": loss_ip13,
            "loss_vxc": loss_vxc,
            "loss_rho": loss_rho,
        }

    def _iter_idx_for_aux_channels(self) -> tuple:
        """Indices used by V_xc / rho channels.  Includes ``aux_only_names``
        species (reaction-form AE compounds under ``ae_as_reactions``, or HBPT
        aux fixtures) which the AE channel excludes via ``compound_idx``.
        """
        N = len(self.mol_names)
        if not self.molecules_only:
            return tuple(range(N))
        aux_set = set(self.aux_only_names)
        aux_idx = tuple(
            i for i, n in enumerate(self.mol_names)
            if n in aux_set
        )
        return tuple(sorted(set(self.compound_idx) | set(aux_idx)))

    def __call__(self, model, batch):
        components = self.compute_components(model, batch)
        total = sum(components.values())
        return total, components
