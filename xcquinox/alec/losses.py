"""xcquinox.alec.losses — AlecLoss ABC, registry, 6 concrete losses.

Implements THE SPEC §7.1 (base class) and §7.2 (A, B, C, D1, D2, D3).
"""
import abc
import math
from typing import ClassVar

import jax.numpy as jnp
import equinox as eqx

from xcquinox.alec.oneshot import (
    fixed_density_total_energy,
    oneshot_dm_prediction_fast,
    oneshot_grid_density,
    compute_vxc_nn,
    _uks_spin_resolved_vxc,
)
from xcquinox.alec.descriptors import assemble_descriptor_features


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
    def build_indices(molecules):
        """Build hashable (atom_mol_idx, compound_idx, mol_names, compositions).

        Raises ValueError if no compound molecules are present.
        """
        atom_map = {}
        for i, m in enumerate(molecules):
            comp = dict(m.atom_composition)
            if sum(comp.values()) == 1:
                symbol = next(iter(comp))
                atom_map[symbol] = i
        atom_mol_idx = tuple(sorted(atom_map.items()))
        compound_idx = tuple(
            i for i, m in enumerate(molecules)
            if sum(dict(m.atom_composition).values()) > 1
        )
        if not compound_idx:
            raise ValueError(
                "Loss requires at least one compound molecule (atom_composition "
                f"sum > 1); got only atomic molecules: {[m.name for m in molecules]}"
            )
        mol_names = tuple(m.name for m in molecules)
        compositions = tuple(tuple(m.atom_composition) for m in molecules)
        return atom_mol_idx, compound_idx, mol_names, compositions

    @staticmethod
    def _validate_static_float(name: str, value) -> None:
        """D-H1: reject non-scalar types on static float fields."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"AlecLoss field {name!r} must be a plain Python int/float, "
                f"got {type(value).__name__} = {value!r}"
            )

    @staticmethod
    def _validate_static_bool(name: str, value) -> None:
        """L-B13-2: reject non-bool types on static bool fields."""
        if not isinstance(value, bool):
            raise TypeError(
                f"AlecLoss field {name!r} must be a plain Python bool, "
                f"got {type(value).__name__} = {value!r}"
            )


# ---------------------------------------------------------------------------
# Shared helpers (used inside loss __call__ bodies)
# ---------------------------------------------------------------------------

def _compute_energies(model, mol_data, N, solver_config=None):
    """Compute per-molecule NN total energies for the energy-loss term.

    The post-hoc fixed-density framework defines the total energy as a
    functional of the frozen reference density (``ρ_PBE``) with the NN's
    V_xc:

        E_total = e_nuc + Tr[h·ρ_PBE] + ½·Tr[J[ρ_PBE]·ρ_PBE] + E_xc^NN[ρ_PBE]

    This is what :class:`xcquinox.alec.evaluation.TotalEnergyMetric` measures
    at evaluation time. For training to *optimize* what evaluation
    *measures*, this function returns the same quantity regardless of
    ``solver_config`` — the solver governs DM / density / V_xc matching
    terms (``_dm_term``, ``_grid_term``, ``_vxc_term``), not the energy
    functional itself.

    (Prior to 2026-04-24 this function branched on ``solver_config`` and,
    for FIXED_J mode, returned ``run_scf(...).total_energy`` — a hybrid
    with ``J[ρ_PBE]`` acting on an SCF-evolved ``ρ_scf ≠ ρ_PBE`` that is
    not a valid energy functional of any single density. Training could
    drive that pseudo-energy to near-zero while producing a V_xc with a
    50+ kcal/mol atomization-energy error at evaluation. ``solver_config``
    is kept in the signature for callers; it is deliberately unused here.)
    """
    del solver_config  # energy functional is solver-invariant.
    return jnp.stack([
        fixed_density_total_energy(model, mol_data[i]) for i in range(N)
    ])


def _ae_from_atoms(E_mol, comp_dict, atom_energies):
    """Positive-for-bound atomization energy from a fixed atomic anchor dict.

    AE = Σ n_Z · atom_energies[Z] − E_mol

    The atom anchor is a caller-supplied dict (typically PBE-consistent
    atomic totals for a post-hoc NN XC on a frozen PBE density). Using a
    fixed anchor rather than NN-predicted atomic totals is what lets the
    training loss and AtomizationEnergyMetric measure the same quantity.
    """
    return sum(n * atom_energies[Z] for Z, n in comp_dict.items()) - E_mol


def _atomic_reg(E_nn, atom_mol_idx_dict, atom_energies):
    """Weak atomic regularization toward the caller-supplied atom anchor dict."""
    return sum(
        (E_nn[atom_mol_idx_dict[Z]] - atom_energies[Z]) ** 2
        / (atom_energies[Z] ** 2 + 1e-8)
        for Z in atom_mol_idx_dict
    )


def _ae_losses(E_nn, compound_idx, comp_dicts, mol_names, targets, atom_energies):
    """A-family: relative squared AE error per compound.

    AE is computed via `_ae_from_atoms` so the anchor dict is the same
    one that AtomizationEnergyMetric consumes at evaluation time.
    """
    terms = []
    for i in compound_idx:
        ae = _ae_from_atoms(E_nn[i], comp_dicts[i], atom_energies)
        tgt = targets[mol_names[i]]
        terms.append((ae - tgt) ** 2 / (tgt ** 2 + 1e-8))
    return jnp.mean(jnp.stack(terms))


def _delta_losses(E_nn, mol_data, compound_idx, comp_dicts, mol_names, targets, atom_energies):
    """D-family: relative squared delta-AE error per compound.

    Both the NN and PBE atomization energies are computed from the same
    `atom_energies` anchor dict via `_ae_from_atoms`, so the atom-sum
    term cancels in `delta_nn = ae_nn - ae_pbe = E_pbe[mol] - E_nn[mol]`.
    The anchor dict still appears in `delta_tgt = target_AE - ae_pbe`,
    so the D-family target is a function of the anchor dict even though
    the NN/PBE delta itself is not.
    """
    terms = []
    for i in compound_idx:
        ae_nn = _ae_from_atoms(E_nn[i], comp_dicts[i], atom_energies)
        ae_pbe = _ae_from_atoms(mol_data[i]["E_pbe"], comp_dicts[i], atom_energies)
        delta_nn = ae_nn - ae_pbe
        delta_tgt = targets[mol_names[i]] - ae_pbe
        terms.append((delta_nn - delta_tgt) ** 2 / (delta_tgt ** 2 + 1e-8))
    return jnp.mean(jnp.stack(terms))


def _dm_term(model, mol_data, iter_idx, solver_config=None, relative=False):
    """DM matching: Frobenius^2 mean-squared error per element.

    Absolute mode normalizes by the total number of DM elements
    (``n_ao^2`` for RKS; ``2 * n_ao^2`` for UKS — both spin channels).
    This yields a per-element scale invariant under RKS↔UKS, matching
    ``_vxc_term``'s convention (D5-loss audit fix: pre-fix UKS branch
    divided by n_ao^2 only, off by a factor of 2 vs RKS and inconsistent
    with _vxc_term).
    """
    terms = []
    for i in iter_idx:
        dm_ref = mol_data[i]["dm_target"]
        if dm_ref is None:
            continue
        dm_nn = oneshot_dm_prediction_fast(model, mol_data[i], solver_config=solver_config)
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
    return jnp.mean(jnp.stack(terms)) if terms else jnp.array(0.0)


def _grid_term(model, mol_data, iter_idx, solver_config=None, relative=False):
    """Grid density matching: weighted L2, normalized absolutely or relatively."""
    terms = []
    for i in iter_idx:
        rho_ref = mol_data[i]["rho_ref_grid"]
        if rho_ref is None:
            continue
        rho_nn = oneshot_grid_density(model, mol_data[i], solver_config=solver_config)
        w = mol_data[i]["grid_weights"]
        err = jnp.sum(w * (rho_nn - rho_ref) ** 2)
        if relative:
            err = err / (jnp.sum(w * rho_ref ** 2) + 1e-8)
        terms.append(err)
    return jnp.mean(jnp.stack(terms)) if terms else jnp.array(0.0)


def _vxc_term(model, mol_data, iter_idx, relative=False):
    """V_xc matching: Frobenius^2 of (V_xc^NN - V_xc^ref).

    Normalized by n_ao^2 (absolute) or ||V_xc^ref||_F^2 (relative).
    Skips molecules where vxc_ref is None.

    Supports both RKS references (shape ``(n_ao, n_ao)``) and UKS references
    (shape ``(2, n_ao, n_ao)``). For UKS, the NN's spin-resolved V_xc is
    constructed via :func:`_uks_spin_resolved_vxc` (spin-scaled approximation)
    and the squared error is summed across both spin channels.
    """
    terms = []
    for i in iter_idx:
        vxc_ref = mol_data[i]["vxc_ref"]
        if vxc_ref is None:
            continue
        vxc_ref_arr = jnp.asarray(vxc_ref)
        features = assemble_descriptor_features(model.descriptors, mol_data[i])

        if vxc_ref_arr.ndim == 3:  # UKS: (2, n_ao, n_ao)
            vxc_nn_a, vxc_nn_b = _uks_spin_resolved_vxc(
                model, mol_data[i], features
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
) -> jnp.ndarray:
    """Squared residual of a generic reaction energy / barrier height.

    e_nn   : (n_species,) NN total energies for each species in the reaction
    coeffs : (n_species,) signed stoichiometric coefficients (negative for
             reactants, positive for products)
    e_rxn_ref : scalar reference reaction-energy or barrier-height value

    Returns: scalar squared residual (E_rxn_NN - E_rxn_ref)^2.

    Used by the BH76 task channel of L5_gradnorm_vxc_step7. Per Dick 2021
    SI II, BH76 residuals were down-weighted by 0.01 in the original
    work; step-7 either reproduces that scaling explicitly or lets
    GradNorm (alpha=1.5; Chen et al. 2018, arXiv:1711.02257) discover the
    weight adaptively.
    """
    e_rxn = jnp.sum(coeffs * e_nn)
    return (e_rxn - e_rxn_ref) ** 2


def _ip_residual_term(
    e_cation: jnp.ndarray,
    e_neutral: jnp.ndarray,
    ip_ref: jnp.ndarray,
) -> jnp.ndarray:
    """Squared residual of an ionization potential. IP = E_cation - E_neutral.

    Used by the IP13 task channel of L5_gradnorm_vxc_step7.
    """
    return (e_cation - e_neutral - ip_ref) ** 2


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
            # D10-loss audit fix: gate on molecules_only (default True)
            # for consistency with B/C/D2/D3. Atoms typically have
            # vxc_ref=None and are skipped inside _vxc_term anyway, but
            # the explicit gate makes the API uniform across all 6 losses.
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
            # D10-loss audit fix: gate on molecules_only, matching B/C/D2/D3.
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

    Per Dick 2021 SI II, BH76 + IP13 residuals were down-weighted by
    0.01 in the original Dick training. Step-7 lets GradNorm (Chen et
    al. 2018, arXiv:1711.02257; alpha=1.5 default at
    xcquinox/alec/balancing.py:55) discover task weights adaptively
    rather than hard-coding the 0.01 factor.

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
    molecules_only: bool = eqx.field(default=True, static=True)
    solver_config: object | None = eqx.field(default=None, static=True)
    vxc_weight: float = eqx.field(default=0.01, static=True)
    density_weight: float = eqx.field(default=0.1, static=True)

    def __init__(
        self,
        *,
        molecules=None,
        bh76_reactions=None,
        ip13_pairs=None,
        w_atomic: float = 0.01,
        molecules_only: bool = True,
        solver_config=None,
        vxc_weight: float = 0.01,
        density_weight: float = 0.1,
        _smoke_test: bool = False,
        **_unused_kwargs,
    ):
        # The smoke path is used by registry/contract tests where there is
        # no real training context (no molecules, no batch). It must still
        # initialize all required AlecLoss fields plus the new BH76/IP13
        # fields so eqx.Module field validation passes.
        bh76_frozen = _freeze_rxn_specs(bh76_reactions or ())
        ip13_frozen = _freeze_ip_specs(ip13_pairs or ())

        if _smoke_test:
            self.atom_mol_idx = ()
            self.compound_idx = ()
            self.mol_names = ()
            self.compositions = ()
            self.w_atomic = w_atomic
            self.bh76_reactions = bh76_frozen
            self.ip13_pairs = ip13_frozen
            self.molecules_only = molecules_only
            self.solver_config = solver_config
            self.vxc_weight = vxc_weight
            self.density_weight = density_weight
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
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.bh76_reactions = bh76_frozen
        self.ip13_pairs = ip13_frozen
        self.molecules_only = molecules_only
        self.solver_config = solver_config
        self.vxc_weight = vxc_weight
        self.density_weight = density_weight

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

    def _bh76_channel(self, E_nn) -> jnp.ndarray:
        """Mean of squared reaction-energy residuals across BH76 reactions.

        E_NN_total values are looked up from the all-species `E_nn` vector
        by name via `mol_names`. A reaction with `e_rxn_ref=None` is
        skipped (treated as missing reference). If no usable reactions
        remain, returns 0.0 (so the channel contributes nothing under
        GradNorm without crashing).
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
                e_species, coeffs_arr, jnp.array(e_ref),
            ))
        if not terms:
            return jnp.array(0.0)
        return jnp.mean(jnp.stack(terms))

    def _ip13_channel(self, E_nn) -> jnp.ndarray:
        """Mean of squared IP residuals across IP13 pairs.

        Pairs with `ip_ref=None` are skipped.
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
                e_cation, e_neutral, jnp.array(ip_ref),
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
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(
            model, mol_data, N, solver_config=self.solver_config
        )
        # AE channel: relative squared AE residual + atomic regularization,
        # mirroring AtomizationLoss but bundled into a single channel for
        # GradNorm. atomic_reg is folded into the AE channel because it is
        # a regularizer of the AE quantity, not an independent task.
        loss_ae = _ae_losses(
            E_nn, self.compound_idx, comp_dicts,
            self.mol_names, targets, atom_energies,
        )
        atomic_reg = self.w_atomic * _atomic_reg(E_nn, atom_idx, atom_energies)
        loss_ae_total = loss_ae + atomic_reg

        # BH76 + IP13 channels: reaction / IP residuals (Dick 2021 SI II).
        loss_bh76 = self._bh76_channel(E_nn)
        loss_ip13 = self._ip13_channel(E_nn)

        # vxc + rho channels: existing alec mechanisms.
        iter_idx = self.compound_idx if self.molecules_only else tuple(range(N))
        loss_vxc = self.vxc_weight * _vxc_term(
            model, mol_data, iter_idx, relative=relative,
        )
        loss_rho = self.density_weight * _grid_term(
            model, mol_data, iter_idx, solver_config=self.solver_config,
            relative=relative,
        )

        return {
            "loss_AE": loss_ae_total,
            "loss_BH76": loss_bh76,
            "loss_IP13": loss_ip13,
            "loss_vxc": loss_vxc,
            "loss_rho": loss_rho,
        }

    def __call__(self, model, batch):
        components = self.compute_components(model, batch)
        total = sum(components.values())
        return total, components
