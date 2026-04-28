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
