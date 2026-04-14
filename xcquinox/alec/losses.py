"""xcquinox.alec.losses — AlecLoss ABC, registry, 6 concrete losses.

Implements THE SPEC §7.1 (base class) and §7.2 (A, B, C, D1, D2, D3).
"""
import abc
from typing import ClassVar

import jax.numpy as jnp
import equinox as eqx

from xcquinox.alec.oneshot import (
    fixed_density_total_energy,
    oneshot_dm_prediction_fast,
    oneshot_grid_density,
)


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

def _compute_energies(model, mol_data, N):
    """Compute per-molecule NN total energies."""
    return jnp.stack([fixed_density_total_energy(model, mol_data[i]) for i in range(N)])


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


def _dm_term(model, mol_data, iter_idx):
    """DM matching: Frobenius^2 / n_ao^2, averaged over molecules."""
    terms = []
    for i in iter_idx:
        dm_ref = mol_data[i]["dm_target"]
        if dm_ref is None:
            continue
        dm_nn = oneshot_dm_prediction_fast(model, mol_data[i])
        n_ao = dm_ref.shape[-1]
        terms.append(jnp.sum((dm_nn - dm_ref) ** 2) / (n_ao * n_ao))
    return jnp.mean(jnp.stack(terms)) if terms else jnp.array(0.0, dtype=jnp.float64)


def _grid_term(model, mol_data, iter_idx):
    """Grid density matching: weighted L2, averaged over molecules."""
    terms = []
    for i in iter_idx:
        rho_ref = mol_data[i]["rho_ccsd_grid"]
        if rho_ref is None:
            continue
        rho_nn = oneshot_grid_density(model, mol_data[i])
        w = mol_data[i]["grid_weights"]
        terms.append(jnp.sum(w * (rho_nn - rho_ref) ** 2))
    return jnp.mean(jnp.stack(terms)) if terms else jnp.array(0.0, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Concrete losses
# ---------------------------------------------------------------------------

@register_loss("A_atomization")
class AtomizationLoss(AlecLoss):
    registry_name: ClassVar[str] = "A_atomization"
    required_mol_keys: ClassVar[tuple[str, ...]] = ()
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")

    def __init__(self, *, molecules, w_atomic: float = 0.01):
        self._validate_static_float("w_atomic", w_atomic)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic

    def __call__(self, model, batch):
        atom_idx = dict(self.atom_mol_idx)
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N)
        loss_energy = _ae_losses(E_nn, self.compound_idx, comp_dicts,
                                 self.mol_names, targets, atom_energies)
        atomic_reg = _atomic_reg(E_nn, atom_idx, atom_energies)
        total = loss_energy + self.w_atomic * atomic_reg
        return total, {"loss_energy": loss_energy, "atomic_reg": atomic_reg}


@register_loss("B_atomization_plus_dm")
class AtomizationPlusDMLoss(AlecLoss):
    registry_name: ClassVar[str] = "B_atomization_plus_dm"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("dm_target",)
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    dm_weight: float = eqx.field(default=0.1, static=True)
    molecules_only: bool = eqx.field(default=True, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 dm_weight: float = 0.1, molecules_only: bool = True):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_float("dm_weight", dm_weight)
        self._validate_static_bool("molecules_only", molecules_only)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.dm_weight = dm_weight
        self.molecules_only = molecules_only

    def __call__(self, model, batch):
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N)
        loss_energy = _ae_losses(E_nn, self.compound_idx, comp_dicts,
                                 self.mol_names, targets, atom_energies)
        iter_idx = self.compound_idx if self.molecules_only else tuple(range(N))
        dm_loss = _dm_term(model, mol_data, iter_idx)
        total = loss_energy + self.dm_weight * dm_loss
        return total, {"loss_energy": loss_energy, "loss_dm": dm_loss}


@register_loss("C_atomization_plus_grid")
class AtomizationPlusGridLoss(AlecLoss):
    registry_name: ClassVar[str] = "C_atomization_plus_grid"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("rho_ccsd_grid",)
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    density_weight: float = eqx.field(default=0.1, static=True)
    molecules_only: bool = eqx.field(default=True, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 density_weight: float = 0.1, molecules_only: bool = True):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_float("density_weight", density_weight)
        self._validate_static_bool("molecules_only", molecules_only)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.density_weight = density_weight
        self.molecules_only = molecules_only

    def __call__(self, model, batch):
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N)
        loss_energy = _ae_losses(E_nn, self.compound_idx, comp_dicts,
                                 self.mol_names, targets, atom_energies)
        iter_idx = self.compound_idx if self.molecules_only else tuple(range(N))
        grid_loss = _grid_term(model, mol_data, iter_idx)
        total = loss_energy + self.density_weight * grid_loss
        return total, {"loss_energy": loss_energy, "loss_grid": grid_loss}


@register_loss("D1_delta_ae")
class DeltaAELoss(AlecLoss):
    registry_name: ClassVar[str] = "D1_delta_ae"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("E_pbe",)
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")

    def __init__(self, *, molecules, w_atomic: float = 0.01):
        self._validate_static_float("w_atomic", w_atomic)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic

    def __call__(self, model, batch):
        atom_idx = dict(self.atom_mol_idx)
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N)
        loss_delta = _delta_losses(E_nn, mol_data, self.compound_idx, comp_dicts,
                                   self.mol_names, targets, atom_energies)
        atomic_reg = _atomic_reg(E_nn, atom_idx, atom_energies)
        total = loss_delta + self.w_atomic * atomic_reg
        return total, {"loss_delta": loss_delta, "atomic_reg": atomic_reg}


@register_loss("D2_delta_ae_plus_dm")
class DeltaAEPlusDMLoss(AlecLoss):
    registry_name: ClassVar[str] = "D2_delta_ae_plus_dm"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("E_pbe", "dm_target")
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    dm_weight: float = eqx.field(default=0.1, static=True)
    molecules_only: bool = eqx.field(default=True, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 dm_weight: float = 0.1, molecules_only: bool = True):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_float("dm_weight", dm_weight)
        self._validate_static_bool("molecules_only", molecules_only)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.dm_weight = dm_weight
        self.molecules_only = molecules_only

    def __call__(self, model, batch):
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N)
        loss_delta = _delta_losses(E_nn, mol_data, self.compound_idx, comp_dicts,
                                   self.mol_names, targets, atom_energies)
        iter_idx = self.compound_idx if self.molecules_only else tuple(range(N))
        dm_loss = _dm_term(model, mol_data, iter_idx)
        total = loss_delta + self.dm_weight * dm_loss
        return total, {"loss_delta": loss_delta, "loss_dm": dm_loss}


@register_loss("D3_delta_ae_plus_grid")
class DeltaAEPlusGridLoss(AlecLoss):
    registry_name: ClassVar[str] = "D3_delta_ae_plus_grid"
    required_mol_keys: ClassVar[tuple[str, ...]] = ("E_pbe", "rho_ccsd_grid")
    required_batch_keys: ClassVar[tuple[str, ...]] = ("targets", "atom_energies")
    density_weight: float = eqx.field(default=0.1, static=True)
    molecules_only: bool = eqx.field(default=True, static=True)

    def __init__(self, *, molecules, w_atomic: float = 0.01,
                 density_weight: float = 0.1, molecules_only: bool = True):
        self._validate_static_float("w_atomic", w_atomic)
        self._validate_static_float("density_weight", density_weight)
        self._validate_static_bool("molecules_only", molecules_only)
        ami, ci, mn, comp = self.build_indices(molecules)
        self.atom_mol_idx = ami
        self.compound_idx = ci
        self.mol_names = mn
        self.compositions = comp
        self.w_atomic = w_atomic
        self.density_weight = density_weight
        self.molecules_only = molecules_only

    def __call__(self, model, batch):
        mol_data = batch["mol_data"]
        targets = batch["targets"]
        atom_energies = batch["atom_energies"]
        N = len(self.mol_names)
        comp_dicts = tuple(dict(c) for c in self.compositions)
        E_nn = _compute_energies(model, mol_data, N)
        loss_delta = _delta_losses(E_nn, mol_data, self.compound_idx, comp_dicts,
                                   self.mol_names, targets, atom_energies)
        iter_idx = self.compound_idx if self.molecules_only else tuple(range(N))
        grid_loss = _grid_term(model, mol_data, iter_idx)
        total = loss_delta + self.density_weight * grid_loss
        return total, {"loss_delta": loss_delta, "loss_grid": grid_loss}
