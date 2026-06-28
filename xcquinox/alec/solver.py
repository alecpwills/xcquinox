"""xcquinox.alec.solver: SolverConfig, ABCs, and SCF dispatcher.

Implements the SCF solver toggle design:
docs/superpowers/specs/2026-04-14-alec-scf-solver-and-ref-density-rename-design.md

This module intentionally contains no backend SCF logic. Backends are
lazily imported inside `run_scf` so that `import xcquinox.alec.solver`
never pulls `pyscfad` for users who only need the manual path.
"""
from enum import Enum


class SolverBackend(str, Enum):
    MANUAL = "manual"
    PYSCFAD = "pyscfad"


class SolverMode(str, Enum):
    ONESHOT = "oneshot"
    FIXED_J = "fixed_j"
    FULL = "full"


class FeaturePolicy(str, Enum):
    FROZEN = "frozen"
    REASSEMBLE = "reassemble"


from dataclasses import dataclass, field


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for SCF solver selection and behavior.

    Frozen + hashable so JAX jit caches work. mixer_kwargs is a
    tuple-of-tuples to stay hashable.
    """
    backend: SolverBackend = SolverBackend.MANUAL
    mode: SolverMode = SolverMode.ONESHOT
    max_cycles: int = 0
    conv_tol: float = 1e-6
    feature_policy: FeaturePolicy | None = None
    mixer_name: str = "linear"
    mixer_kwargs: tuple[tuple[str, float], ...] = (("alpha", 0.5),)
    convergence_name: str = "energy"
    # Density fitting: when True the Coulomb J is built from a precomputed
    # 3-index ``cderi`` (mol_data["cderi"]) instead of the full 4-index ERI,
    # making larger bases memory-feasible. ``auxbasis`` None -> auto-select
    # (df_jk.default_auxbasis from the orbital basis). Default off ->
    # byte-identical full-ERI path.
    density_fit: bool = False
    auxbasis: str | None = None
    # Gradient (activation) checkpointing of the unrolled SCF scan. When True,
    # each scan-cycle body is wrapped in ``jax.checkpoint`` (jax.remat) so the
    # reverse-mode tape stores ~O(sqrt(max_cycles)) cycle activations instead of
    # all of them (~2-3x lower backward-pass peak memory, ~1.5x recompute cost).
    # Default off -> the scan body is the original closure, giving a
    # byte-identical XLA graph and zero behavior change.
    scf_grad_checkpoint: bool = False

    def __post_init__(self):
        if self.max_cycles < 0:
            raise ValueError(f"max_cycles must be >= 0, got {self.max_cycles}")
        if self.conv_tol <= 0:
            raise ValueError(f"conv_tol must be > 0, got {self.conv_tol}")
        if self.mode == SolverMode.ONESHOT and self.max_cycles != 0:
            raise ValueError(
                f"oneshot mode requires max_cycles=0, got {self.max_cycles}"
            )
        if self.mode != SolverMode.ONESHOT and self.max_cycles == 0:
            raise ValueError(
                f"non-oneshot modes require max_cycles > 0, got 0 with mode={self.mode}"
            )
        # (FULL, FROZEN) is incoherent. FULL mode rebuilds
        # the Fock from D every cycle (J[D] explicit dependence on D), so
        # the descriptor features that condition the NN's V_xc must also
        # be reassembled with the current D, otherwise the NN sees stale
        # features that disagree with the J term, producing a Fock that
        # has no fixed point for the NN's actual functional. Allow only
        # ``feature_policy is None`` (auto-resolve to REASSEMBLE) or an
        # explicit REASSEMBLE.
        if (
            self.mode == SolverMode.FULL
            and self.feature_policy == FeaturePolicy.FROZEN
        ):
            raise ValueError(
                "feature_policy=FROZEN with mode=FULL is incoherent: FULL "
                "rebuilds the Fock per cycle so descriptor features must "
                "also be reassembled (FeaturePolicy.REASSEMBLE). Pass "
                "feature_policy=None for auto-resolution or REASSEMBLE "
                "explicitly."
            )

    @property
    def effective_feature_policy(self) -> FeaturePolicy:
        if self.feature_policy is not None:
            return self.feature_policy
        if self.mode == SolverMode.FIXED_J:
            return FeaturePolicy.FROZEN
        return FeaturePolicy.REASSEMBLE

    def describe(self) -> dict:
        """JSON-serializable form for test_metadata.json logging."""
        return {
            "backend": self.backend.value,
            "mode": self.mode.value,
            "max_cycles": self.max_cycles,
            "conv_tol": self.conv_tol,
            "feature_policy": self.effective_feature_policy.value,
            "mixer_name": self.mixer_name,
            "mixer_kwargs": dict(self.mixer_kwargs),
            "convergence_name": self.convergence_name,
            "density_fit": self.density_fit,
            "auxbasis": self.auxbasis,
            "scf_grad_checkpoint": self.scf_grad_checkpoint,
        }


import abc
from typing import NamedTuple
import jax.numpy as jnp


# Shared numerical-regularization constants used by every SCF backend.
# When duplicated as module-level constants in both ``oneshot.py`` and
# ``solver_manual.py``, a future adjustment in one site silently diverges
# the two paths. Defined once here so all backends import the same value.
#
# DEGENERACY_REG: uniform shift on the overlap matrix before Cholesky
# decomposition (S + ε·I). Conditions S so ``cholesky`` is stable for
# near-singular basis sets.
DEGENERACY_REG = 1e-10

# SYM_BREAK_SHIFT: magnitude of the NON-UNIFORM diagonal perturbation
# added to the transformed Fock matrix before ``jnp.linalg.eigh``.
# Required because eigh's reverse-mode JVP uses 1/(λ_i - λ_j) which
# returns NaN at exact degeneracies (linear-symmetry π MOs, atomic
# p_x/p_y/p_z). Size: 1e-8 is comfortably above float64 accumulation
# noise (~1e-13 relative) and orders of magnitude below any physical
# energy scale. See oneshot.py docstring for full discussion.
SYM_BREAK_SHIFT = 1e-8

# Golden-ratio constant used by the symmetry-breaking diagonal.
# Irrational so ``sin(idx · φ)`` produces a
# quasi-random spacing, no two indices give bit-equal values.
_GOLDEN_RATIO = 1.618033988749895


def _sym_break_diag(nao: int, dtype) -> jnp.ndarray:
    """Deterministic non-uniform diagonal that breaks eigh-degeneracy.

    Returns ``SYM_BREAK_SHIFT * sin(idx · φ)`` for ``idx ∈ [0, nao)``.
    Fully deterministic in ``nao`` alone (no PRNG), so forward results
    are reproducible across runs. Defined once here and re-exported by
    ``oneshot`` and ``solver_manual``.
    """
    idx = jnp.arange(nao, dtype=dtype)
    return SYM_BREAK_SHIFT * jnp.sin(idx * _GOLDEN_RATIO)


class MixerState(NamedTuple):
    """Base mixer state. Subclasses may extend via their own NamedTuple."""
    step_index: jnp.ndarray  # int32 scalar


# Mixer subclass registry, keyed by ``registry_name``. Populated via the
# ``register_mixer`` decorator below; ``_build_mixer`` in solver_manual.py
# consults this map instead of hard-coding the 'linear' branch. New mixers
# (e.g. DIIS, Pulay) only need to subclass ``Mixer`` with
# a unique ``registry_name`` and use ``@register_mixer``.
MIXER_REGISTRY: "dict[str, type[Mixer]]" = {}


def register_mixer(cls):
    """Decorator: register a Mixer subclass under its ``registry_name``."""
    name = getattr(cls, "registry_name", "") or ""
    if not name:
        raise ValueError(
            f"{cls.__name__} must set ``registry_name`` to a non-empty string "
            f"to use @register_mixer."
        )
    if name in MIXER_REGISTRY and MIXER_REGISTRY[name] is not cls:
        raise ValueError(
            f"mixer registry collision: {name!r} already maps to "
            f"{MIXER_REGISTRY[name].__name__}"
        )
    MIXER_REGISTRY[name] = cls
    return cls


class Mixer(abc.ABC):
    registry_name: str = ""

    @abc.abstractmethod
    def init_state(self, nao: int) -> MixerState:
        ...

    @abc.abstractmethod
    def step(self, state: MixerState, D_in: jnp.ndarray,
             D_out: jnp.ndarray) -> tuple[MixerState, jnp.ndarray]:
        """Returns (new_state, D_mixed)."""


@register_mixer
class LinearMixer(Mixer):
    registry_name = "linear"

    def __init__(self, alpha: float = 0.5):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def init_state(self, nao: int) -> MixerState:
        return MixerState(step_index=jnp.int32(0))

    def step(self, state, D_in, D_out):
        D_mixed = self.alpha * D_out + (1.0 - self.alpha) * D_in
        new_state = MixerState(step_index=state.step_index + jnp.int32(1))
        return new_state, D_mixed


# Convergence-criterion registry, keyed by ``registry_name``. Mirrors
# the ``MIXER_REGISTRY`` pattern: when ``_build_criterion`` in
# solver_manual.py is hard-coded to the 'energy' branch, any new criterion
# (e.g. density-RMS, Fock-error) would require editing the dispatch as well
# as defining the class.
CRITERION_REGISTRY: "dict[str, type['ConvergenceCriterion']]" = {}


def register_criterion(cls):
    """Decorator: register a ConvergenceCriterion subclass under its
    ``registry_name``."""
    name = getattr(cls, "registry_name", "") or ""
    if not name:
        raise ValueError(
            f"{cls.__name__} must set ``registry_name`` to a non-empty string "
            f"to use @register_criterion."
        )
    if name in CRITERION_REGISTRY and CRITERION_REGISTRY[name] is not cls:
        raise ValueError(
            f"criterion registry collision: {name!r} already maps to "
            f"{CRITERION_REGISTRY[name].__name__}"
        )
    CRITERION_REGISTRY[name] = cls
    return cls


class ConvergenceCriterion(abc.ABC):
    registry_name: str = ""

    @abc.abstractmethod
    def is_converged_from_energies(self, e_prev: jnp.ndarray,
                                   e_curr: jnp.ndarray) -> jnp.ndarray:
        """Return scalar JAX bool. Pure, safe inside jit'd scan body."""


@register_criterion
class EnergyConvergence(ConvergenceCriterion):
    registry_name = "energy"

    def __init__(self, tol: float):
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        self.tol = tol

    def is_converged_from_energies(self, e_prev, e_curr):
        return jnp.abs(e_curr - e_prev) < self.tol


@dataclass(frozen=True)
class SCFResult:
    """Result bundle returned by all SCF backend implementations."""
    density_matrix: jnp.ndarray     # (nao, nao)
    total_energy: jnp.ndarray       # scalar
    cycles_run: jnp.ndarray         # int32 scalar
    converged: jnp.ndarray          # bool scalar
    features_used: jnp.ndarray      # (n_grid, n_features) final cycle features
    energy_trace: jnp.ndarray | None = None  # (max_cycles,) per-cycle energies


def _contract_dm_to_grid(D: jnp.ndarray, ao_deriv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (rho, sigma) on-grid from a restricted-spin DM.

    ao_deriv matches the pyscf `eval_ao(..., deriv=1)` layout:
      shape (4, n_grid, nao): [value, d/dx, d/dy, d/dz]

    This is the layout stashed as mol_data['ao_grid_deriv'] by
    precompute_fixed_density_data.
    """
    rho, _, sigma = _contract_dm_to_grid_with_nabla(D, ao_deriv)
    return rho, sigma


def _contract_dm_to_grid_with_nabla(
    D: jnp.ndarray, ao_deriv: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (rho, nabla_rho, sigma) on-grid from a restricted-spin DM.

    Same layout convention as ``_contract_dm_to_grid``; the extra return
    is ``nabla_rho`` with shape ``(n_grid, 3)`` so callers can feed the
    GGA v_sigma term of ``compute_vxc_nn`` without recomputing it.
    """
    ao = ao_deriv[0]
    ao_grad = ao_deriv[1:4]
    rho = jnp.einsum("ij,gi,gj->g", D, ao, ao)
    nabla_rho = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_grad, ao)
    sigma = jnp.einsum("gd,gd->g", nabla_rho, nabla_rho)
    return rho, nabla_rho, sigma


def _reassemble_features(
    descriptors: tuple,
    dm: jnp.ndarray,
    s_matrix: jnp.ndarray,
    cusp_features: jnp.ndarray | None = None,
    n_grid: int | None = None,
    rung35_proj_ao: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Recompute descriptor features from the live (dm, S) + cached cusp.

    Used by REASSEMBLE policy. CuspDescriptor features are geometry-only
    (not DM-dependent) so they are passed in as the frozen precompute value.
    DMStatisticsDescriptor features use the live DM via compute_from_dm.
    DMRung35Descriptor features use the live DM contracted with the constant
    projected-AO matrix ``rung35_proj_ao`` (= mol_data['rung35_proj_ao'], which
    the manual backend computes on the precompute grid -- the same grid this
    backend integrates on).

    ``n_grid`` supplies the grid size for DMStatisticsDescriptor when no
    CuspDescriptor (and therefore no cusp_features) is present.
    """
    from xcquinox.alec.descriptors import (
        CuspDescriptor, DMStatisticsDescriptor, DMRung35Descriptor)
    if not descriptors:
        _ng = cusp_features.shape[0] if cusp_features is not None else (n_grid or 0)
        return jnp.zeros((_ng, 0))
    cols = []
    n_grid_hint = cusp_features.shape[0] if cusp_features is not None else n_grid
    for d in descriptors:
        if isinstance(d, CuspDescriptor):
            if cusp_features is None:
                raise ValueError(
                    "cusp_features must be provided when descriptors include CuspDescriptor"
                )
            cols.append(cusp_features)
            n_grid_hint = cusp_features.shape[0]
        elif isinstance(d, DMStatisticsDescriptor):
            if n_grid_hint is None:
                raise ValueError(
                    "_reassemble_features needs a grid-size hint; pass "
                    "cusp_features or n_grid"
                )
            cols.append(d.compute_from_dm(dm=dm, s_matrix=s_matrix, n_grid=n_grid_hint))
        elif isinstance(d, DMRung35Descriptor):
            if rung35_proj_ao is None:
                raise ValueError(
                    "rung35_proj_ao (mol_data['rung35_proj_ao']) must be provided "
                    "when descriptors include DMRung35Descriptor"
                )
            cols.append(d.compute_from_dm(proj_ao=rung35_proj_ao, dm=dm))
        else:
            raise NotImplementedError(
                f"_reassemble_features does not yet know how to recompute {type(d).__name__}"
            )
    return jnp.concatenate(cols, axis=1)


def _oneshot_result(model, mol_data: dict) -> "SCFResult":
    """Fast path for ONESHOT mode: calls the existing pure one-shot helpers
    and wraps their output in an SCFResult.

    Byte-identical to pre-SCF behavior, regression tests pin this.
    """
    from xcquinox.alec.oneshot import (
        oneshot_dm_prediction_fast, fixed_density_total_energy,
    )
    from xcquinox.alec.descriptors import assemble_descriptor_features
    D = oneshot_dm_prediction_fast(model, mol_data)
    E = fixed_density_total_energy(model, mol_data)
    features = assemble_descriptor_features(model.descriptors, mol_data)
    return SCFResult(
        density_matrix=D,
        total_energy=E,
        cycles_run=jnp.int32(0),
        converged=jnp.bool_(True),
        features_used=features,
    )


def run_scf(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    """Dispatch to the selected backend. Backends are imported lazily."""
    if config.backend == SolverBackend.MANUAL:
        from xcquinox.alec.solver_manual import run_manual_scf
        return run_manual_scf(config, model, mol_data)
    if config.backend == SolverBackend.PYSCFAD:
        from xcquinox.alec.solver_pyscfad import run_pyscfad_scf
        return run_pyscfad_scf(config, model, mol_data)
    raise ValueError(f"unknown solver backend: {config.backend}")
