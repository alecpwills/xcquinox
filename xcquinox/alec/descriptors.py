"""xcquinox.alec.descriptors: Descriptor ABC, registry, and concrete descriptors.

Implements THE SPEC §3: registry-driven descriptor composition for additional
network input features beyond (rho, sigma).
"""
import abc
import dataclasses

import equinox as eqx
import jax.numpy as jnp
from typing import ClassVar

from xcquinox.alec.rung35 import DEFAULT_RUNG35_ALPHA


class Descriptor(eqx.Module, abc.ABC):
    """Base class for all descriptors. Subclasses provide extra input features."""
    registry_name: ClassVar[str] = ""
    required_mol_keys: ClassVar[tuple[str, ...]] = ()
    n_features: int = eqx.field(static=True)

    @abc.abstractmethod
    def compute(self, mol_data: dict) -> jnp.ndarray:
        """Return descriptor features, shape (N, n_features) where N = grid size."""

    def describe(self) -> str:
        return f"{type(self).__name__}({self.registry_name}, n={self.n_features})"

    def __post_init__(self):
        _PRIMITIVE_TYPES = (int, float, bool, str)
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            ann = f.type
            if isinstance(ann, type) and issubclass(ann, _PRIMITIVE_TYPES):
                if isinstance(value, jnp.ndarray) or type(value).__module__.startswith("jax"):
                    raise TypeError(
                        f"{type(self).__name__}.{f.name} is declared as {ann.__name__} "
                        f"but received a jax.Array value; pass a plain Python {ann.__name__} instead."
                    )


DESCRIPTOR_REGISTRY: dict[str, type[Descriptor]] = {}


def register_descriptor(name: str):
    """Class decorator registering a Descriptor subclass under `name`."""
    def wrapper(cls):
        if name in DESCRIPTOR_REGISTRY:
            raise ValueError(f"Descriptor name {name!r} already registered")
        for f in dataclasses.fields(cls):
            if not f.metadata.get("static"):
                raise TypeError(
                    f"{cls.__name__}.{f.name} must be declared with "
                    f"eqx.field(..., static=True)"
                )
        cls.registry_name = name
        DESCRIPTOR_REGISTRY[name] = cls
        return cls
    return wrapper


def make_descriptor(name: str, **kwargs) -> Descriptor:
    """Look up DESCRIPTOR_REGISTRY[name] and instantiate with kwargs."""
    return DESCRIPTOR_REGISTRY[name](**kwargs)


def list_descriptors() -> list[str]:
    """Return sorted list of registered descriptor names."""
    return sorted(DESCRIPTOR_REGISTRY.keys())


@register_descriptor("cusp")
class CuspDescriptor(Descriptor):
    """Nuclear cusp proximity (2 features per grid point).

    Both features are bounded for network-friendly inputs:
      * Column 0 ``cusp_factor = exp(-2 Z_nearest r_min)`` ∈ [0, 1], where
        Z_nearest is the charge of the nearest nucleus and r_min is the
        distance to it. This is a heuristic proximity feature motivated by
        the electron-nucleus cusp condition. Kato (*Commun. Pure Appl.
        Math.* 10, 151 (1957)) fixes the wavefunction cusp
        ``(∂⟨ψ⟩/∂r)|_{r=0} = -Z·ψ(0)``; the corresponding spherically-averaged
        density relation ``(∂⟨ρ⟩/∂r)|_{r=0} = -2Z·ρ(0)`` is due to Steiner
        (*J. Chem. Phys.* 39, 2365 (1963)). The exponential decay here
        approximates the resulting density-form Slater envelope ``exp(-2 Z r)``
        (the density ρ=|ψ|² decays at twice the ``exp(-Z r)`` wavefunction
        rate) rather than enforcing the condition exactly.
      * Column 1 (``log_transform=True``):
        ``tanh(log(Σ_A Z_A / r_A) / 5)`` ∈ (-1, 1), log-compressed
        nuclear-attraction-like weight (Dick & Fernández-Serra XCDiff
        convention). With ``log_transform=False``, the same input is fed
        through ``tanh(Σ_A Z_A / r_A / 5)`` (no log).

    See ``xcquinox.features.compute_cusp_descriptor`` for the precise
    definitions and the dynamic-range argument for the /5 scaling.
    """
    n_features: int = eqx.field(default=2, static=True)
    # When True, apply the Dick XCDiff log compression to feature 1. When
    # False, feed the raw weighted-Z value through tanh only (old
    # checkpoints unpickle to this).
    log_transform: bool = eqx.field(default=False, static=True)
    required_mol_keys: ClassVar[tuple[str, ...]] = ("cusp_features",)

    def compute(self, mol_data):
        return mol_data["cusp_features"]


@register_descriptor("dm_statistics")
class DMStatisticsDescriptor(Descriptor):
    """Density-matrix correlation indicators. 3 features.

    The 3 features (see ``xcquinox.features.compute_dm_features_array``) are,
    in order:
      0. ``idempotency_error``: normalized Frobenius deviation from the
         single-determinant idempotency condition; ~0 for an HF/KS reference,
         growing with correlation.
      1. ``dm_entropy``: Shannon (von-Neumann-like) entropy of the natural-
         orbital occupations normalized to a probability distribution
         (``-sum_i p_i ln p_i`` with ``p_i = n_i / sum_j n_j``; the occupations
         ``n_i`` are the eigenvalues of ``D S``: Löwdin, Phys. Rev. 97, 1474
         (1955)). NOTE: this quantity is size-dependent (it scales
         roughly like ``ln N_occ``) and is nonzero even for a single
         determinant, so it is NOT a clean electron-correlation indicator,
         ``idempotency_error`` is the quantity that vanishes for a single
         determinant. See ``xcquinox.features.compute_dm_features`` for the
         full discussion.
      2. ``off_diag_norm``: Frobenius norm of the off-diagonal AO density-matrix
         block, normalized by the trace.

    ``idempotency_error`` and ``off_diag_norm`` grow with departure from a single
    Slater determinant; ``dm_entropy`` is a size-dependent natural-occupation
    entropy (see the caveat above), not a clean correlation indicator.

    SIZE-CONSISTENCY / LOCALITY CAVEAT: these are GLOBAL, molecule-level
    scalars that ``__call__`` ``jnp.tile``s identically to every grid point and
    feeds into the per-point (semilocal) enhancement factor. Consequences:
      * The per-point ε_xc then depends on whole-system quantities, so the
        functional is NOT size-consistent, the XC energy density at a point in
        fragment A shifts if a distant fragment B is added to the system.
      * ``dm_entropy`` is extensive (~ln N_occ), so as a constant local feature
        it largely encodes molecule identity/size (a label-leakage / overfitting
        handle on a small training pool) rather than local correlation physics.
    Making this defensible requires an ARCHITECTURE change (feed these as
    size-INTENSIVE molecule-level conditioning, or normalize per electron), which
    redefines the feature and invalidates checkpoints trained on the current
    values: a deferred design decision requiring sign-off, NOT changed here.
    """
    n_features: int = eqx.field(default=3, static=True)
    # When True, divides dm_entropy by ln(max(n_orb_eff, 2)), where
    # n_orb_eff = sum(occupations)/max_occ, so the feature is size-intensive
    # (range [0, 1]). When False, keeps the size-extensive ln(N_occ) form
    # (old checkpoints unpickle to this default).
    intensive: bool = eqx.field(default=False, static=True)
    required_mol_keys: ClassVar[tuple[str, ...]] = ("dm_features",)

    @staticmethod
    def compute_from_dm(dm: jnp.ndarray, s_matrix: jnp.ndarray,
                        n_grid: int, *, intensive: bool = False) -> jnp.ndarray:
        """Pure kernel: compute 3-feature vector from (dm, S) and tile to grid.

        Mirrors the precompute path in data.py:229-234 but accepts a live DM
        so the SCF REASSEMBLE policy can recompute features per cycle.
        """
        from xcquinox.features import compute_dm_features_array
        global_features = compute_dm_features_array(dm, s_matrix,
                                                     intensive=intensive)
        return jnp.tile(global_features, (n_grid, 1))

    def compute(self, mol_data):
        return mol_data["dm_features"]


@register_descriptor("rung35")
class DMRung35Descriptor(Descriptor):
    """Rung-3.5 localized density-matrix descriptor: per-spin bounded local
    occupancy ``n_sigma(r) = A(r)^T P^sigma A(r) in [0, 1]`` (Janesko,
    arXiv:2206.07118 Eq. 12-13; M11plus, Verma et al. *J. Chem. Theory Comput.*
    15, 4804 (2019)).

    Unlike :class:`DMStatisticsDescriptor` (GLOBAL per-molecule scalars tiled to
    every grid point -- a molecule-identity leak), this is a genuine PER-GRID-POINT
    contraction of the *non-local* Kohn-Sham 1-RDM against a localized Gaussian
    projector. It is therefore leak-free (size-intensive), self-consistent (a
    functional of the LIVE DM, recomputed each SCF cycle), NOT a static reference
    DM, and NOT a meta-GGA (no tau) -- its own rung on Jacob's ladder, between
    meta-GGA and hybrid. Bounded ``[0, 1]`` by Bessel's inequality (``P^sigma`` is
    PSD => ``>= 0``; orthonormal occupied orbitals + normalized projector =>
    ``<= 1``), so it is NaN-safe by construction.

    The two features are the alpha- and beta-spin occupancies. They feed BOTH the
    exchange and correlation networks: the M11plus rung-3.5 ingredient is a
    CORRELATION ingredient (so the C-net is a first-class consumer from the
    start), and the X-net receives it equally via the shared ``features`` extras
    mechanism -- complete X/C parity.

    See :mod:`xcquinox.alec.rung35` for the projected-AO overlap ``A_mu(r)`` (a
    fixed, density-independent precompute) and the occupancy contraction.
    """
    n_features: int = eqx.field(default=2, static=True)
    # Gaussian-projector width (a0^-2); a configurable hyperparameter. Default
    # grounded at the M11plus kernel scale (d^2 = 5 a0^2). A FIXED alpha makes
    # A_mu(r) a precompute-once constant (never differentiated); the occupancy is
    # then linear in the live DM.
    alpha: float = eqx.field(default=DEFAULT_RUNG35_ALPHA, static=True)
    required_mol_keys: ClassVar[tuple[str, ...]] = ("rung35_features",)

    @staticmethod
    def compute_from_dm(proj_ao: jnp.ndarray, dm: jnp.ndarray) -> jnp.ndarray:
        """Reassemble kernel: recompute the per-spin occupancy from a LIVE DM and
        the constant projected-AO matrix ``A`` (``proj_ao``), so the SCF REASSEMBLE
        policy keeps the descriptor self-consistent each cycle. Mirrors
        :meth:`DMStatisticsDescriptor.compute_from_dm` but per grid point (the
        occupancy is local, not a tiled global scalar)."""
        from xcquinox.alec.rung35 import compute_rung35_occupancy
        return compute_rung35_occupancy(proj_ao, dm)

    def compute(self, mol_data):
        return mol_data["rung35_features"]


def assemble_descriptor_features(descriptors: tuple[Descriptor, ...],
                                 mol_data: dict) -> jnp.ndarray:
    """Concatenate descriptor outputs left-to-right in declaration order."""
    if not descriptors:
        return jnp.zeros((mol_data["rho_grid"].shape[0], 0))
    return jnp.concatenate([d.compute(mol_data) for d in descriptors], axis=1)
