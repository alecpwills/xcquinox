"""xcquinox.alec.descriptors: Descriptor ABC, registry, and concrete descriptors.

Implements THE SPEC §3: registry-driven descriptor composition for additional
network input features beyond (rho, sigma).
"""
import abc
import dataclasses
import numbers

import equinox as eqx
import jax.numpy as jnp
from typing import ClassVar

from xcquinox.alec.rung35 import (DEFAULT_RUNG35_ALPHA,
                                  DEFAULT_RUNG35_MULTISHELL_ALPHAS)


def _validated_spin_channel(spin_channel) -> int:
    """Return the spin channel as the integer 0 (alpha) or 1 (beta).

    The channel selects an array index and a key slot, so only an integral
    scalar is admissible, and membership in ``(0, 1)`` is not by itself
    sufficient: ``True == 1`` and ``1.0 == 1`` both satisfy it. A boolean
    indexes a JAX array as a MASK rather than along the spin axis --
    ``jnp.zeros((2, 2, 2))[True]`` has shape ``(1, 2, 2, 2)`` and
    ``[False]`` shape ``(0, 2, 2, 2)`` -- so a Python bool would leave the
    ``(2, nao, nao)`` contract of :func:`doubled_spin_dm` without raising,
    while ``np.bool_(True)`` returned the beta block and ``np.bool_(False)``
    the alpha block. Floats passed the same membership test and were then
    either taken as a channel by accident or tripped the array indexer with an
    opaque TypeError. Each case is refused here instead.
    """
    is_boolean = (isinstance(spin_channel, bool)
                  or getattr(spin_channel, "dtype", None) == bool)
    if is_boolean or not isinstance(spin_channel, numbers.Integral):
        raise ValueError(
            "spin_channel must be an integral scalar, 0 (alpha) or 1 (beta); "
            f"got {spin_channel!r} of type {type(spin_channel).__name__}."
        )
    if spin_channel not in (0, 1):
        raise ValueError(
            f"spin_channel must be 0 (alpha) or 1 (beta); got {spin_channel!r}."
        )
    return int(spin_channel)


class Descriptor(eqx.Module, abc.ABC):
    """Base class for all descriptors. Subclasses provide extra input features."""
    registry_name: ClassVar[str] = ""
    required_mol_keys: ClassVar[tuple[str, ...]] = ()
    # Per-spin-channel precompute keys, (alpha, beta), holding this descriptor's
    # features for the symmetric doubled density diag(P_sigma, P_sigma). Empty
    # only for a geometry-only descriptor, whose per-channel block is the shared
    # one; __init_subclass__ enforces that pairing.
    spin_mol_keys: ClassVar[tuple[str, ...]] = ()
    # Whether the features depend on the density matrix, and therefore differ
    # between the physical density and the doubled density of a spin channel.
    # Declared rather than inferred from spin_mol_keys: an omitted key tuple on
    # a density-matrix descriptor would otherwise read as "geometry-only" and
    # reinstate the physical-density features in both exchange channels.
    density_matrix_dependent: ClassVar[bool] = False
    n_features: int = eqx.field(static=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.density_matrix_dependent and not cls.spin_mol_keys:
            raise TypeError(
                f"{cls.__name__} declares density_matrix_dependent = True but "
                "leaves spin_mol_keys empty, so its per-channel exchange block "
                "would fall back to the features of the physical density -- the "
                "defect that evaluating on diag(P_sigma, P_sigma) removes. "
                "Declare the (alpha, beta) precompute key names."
            )

    @abc.abstractmethod
    def compute(self, mol_data: dict) -> jnp.ndarray:
        """Return descriptor features, shape (N, n_features) where N = grid size."""

    def describe(self) -> str:
        return f"{type(self).__name__}({self.registry_name}, n={self.n_features})"

    def compute_for_spin_channel(self, mol_data: dict,
                                 spin_channel: int) -> jnp.ndarray:
        """Features of the symmetric doubled density ``diag(P_sigma, P_sigma)``.

        The exact exchange spin-scaling relation
        ``E_x[n_a, n_b] = (E_x[2 n_a] + E_x[2 n_b]) / 2`` (Oliver and Perdew,
        Phys. Rev. A 20, 397 (1979)) evaluates each channel on the fictitious
        spin-unpolarized system whose two spin blocks both hold ``P_sigma``.
        That system, not the physical one, is where a density-matrix descriptor
        must be evaluated for the relation to stay exact. A descriptor that
        declares ``density_matrix_dependent = False`` carries no such
        dependence, so its per-channel block is the shared block.
        """
        channel = _validated_spin_channel(spin_channel)
        if not self.spin_mol_keys:
            if self.density_matrix_dependent:
                raise TypeError(
                    f"{type(self).__name__} is density-matrix dependent but its "
                    "spin_mol_keys are empty, so no per-channel block exists; "
                    "returning the shared block here would evaluate exchange at "
                    "the physical density in both channels."
                )
            return self.compute(mol_data)
        key = self.spin_mol_keys[channel]
        value = mol_data.get(key)
        if value is None:
            raise KeyError(
                f"{type(self).__name__}.compute_for_spin_channel requires "
                f"mol_data[{key!r}], which is absent or None. Open-shell "
                "precompute populates the per-channel blocks; a closed-shell "
                "molecule has rho_a = rho_b and therefore one block, which is "
                "reached with spin_channel=None."
            )
        return value

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


def doubled_spin_dm(dm: jnp.ndarray, spin_channel: int) -> jnp.ndarray:
    """The symmetric doubled density matrix ``diag(P_sigma, P_sigma)``.

    The exact exchange spin-scaling relation (Oliver and Perdew, Phys. Rev. A
    20, 397 (1979)) refers each spin channel to the spin-unpolarized system
    built by placing ``P_sigma`` in BOTH spin slots. That system has total
    density ``2 rho_sigma``, gradient invariant ``4 sigma_sigma_sigma`` and
    kinetic-energy density ``2 tau_sigma``, and it is the system whose
    density-matrix descriptors define the channel's feature block: the
    iso-orbital indicator becomes
    ``alpha(2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma)``, the rung-3.5
    occupancy becomes ``[n_sigma, n_sigma]`` (still inside the Bessel bound
    ``[0, 1]``), and the density-matrix statistics become those of
    ``diag(P_sigma, P_sigma)``.

    Every descriptor kernel already produces the right quantity when handed
    this matrix: ``metagga.compute_tau_from_dm`` sums the two spin slots of a
    3-D density matrix, ``rung35.compute_rung35_occupancy`` contracts each slot
    separately, and ``features.compute_dm_features`` takes its per-spin
    idempotency branch on a 3-D argument. So this one transform carries the
    whole convention.
    """
    p = jnp.asarray(dm)
    if p.ndim != 3 or p.shape[0] != 2:
        raise ValueError(
            "doubled_spin_dm requires a spin-resolved (2, nao, nao) density "
            f"matrix; got shape {tuple(p.shape)}."
        )
    block = p[_validated_spin_channel(spin_channel)]
    return jnp.stack([block, block], axis=0)


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
    """Density-matrix correlation indicators. 2 features.

    The 2 features (see ``xcquinox.features.compute_dm_features_array``) are,
    in order:
      0. ``idempotency_error``: squared Frobenius deviation from the
         single-determinant idempotency condition, normalized by the electron
         count; ~0 for an HF/KS reference, growing with correlation.
      1. ``off_diag_norm``: Frobenius norm of the off-diagonal AO density-matrix
         block, normalized by the trace.

    Both grow with departure from a single Slater determinant.

    ``dm_entropy`` was REMOVED 2026-08-06 (width 3 -> 2). It had no usable
    gradient at any converged density -- the physical-bounds clip put every
    natural occupation on a boundary, and without the clip the ``eigh``
    eigenvector derivatives are ill-defined on the degenerate occupation
    spectrum of any idempotent density matrix. No spectral invariant can
    replace it: for a single determinant the eigenvalues of ``DS`` are exactly
    {2,...,2,0,...,0}, so any function of the spectrum alone is constant on the
    idempotent manifold. Removing it also took the dm_statistics architectures'
    energy/potential finite-difference residual from
    1.04e-02 to 2.1e-10 under the committed test's own parametrized ordering (residuals move up to ~5x with evaluation order; a fresh-process measurement gave 5.2e-03 to the same floor),
    since the dead gradient had been dominating it. See
    ``notebooks/analysis/DM_DESCRIPTOR_SPEC.md``.

    SIZE-CONSISTENCY / LOCALITY CAVEAT, unchanged and still open: these are
    GLOBAL, molecule-level scalars that ``__call__`` ``jnp.tile``s identically
    to every grid point and feeds into the per-point (semilocal) enhancement
    factor, so the XC energy density at a point in fragment A shifts if a
    distant fragment B is added. :class:`DMRung35Descriptor` and
    :class:`DMRung35MultishellDescriptor` are the leak-free members of this
    family -- genuine per-grid-point contractions of the same density matrix.
    Making the global form defensible requires an architecture change and is
    recorded in ``xcquinox/alec/DEFERRED_WORK.md``.
    """
    n_features: int = eqx.field(default=2, static=True)
    required_mol_keys: ClassVar[tuple[str, ...]] = ("dm_features",)
    spin_mol_keys: ClassVar[tuple[str, ...]] = ("dm_features_a", "dm_features_b")
    density_matrix_dependent: ClassVar[bool] = True

    @staticmethod
    def compute_from_dm(dm: jnp.ndarray, s_matrix: jnp.ndarray,
                        n_grid: int) -> jnp.ndarray:
        """Pure kernel: compute the 2-feature vector from (dm, S), tiled to grid.

        Mirrors the precompute path in data.py but accepts a live DM so the SCF
        REASSEMBLE policy can recompute features per cycle.
        """
        from xcquinox.features import compute_dm_features_array
        global_features = compute_dm_features_array(dm, s_matrix)
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
    spin_mol_keys: ClassVar[tuple[str, ...]] = ("rung35_features_a",
                                                "rung35_features_b")
    density_matrix_dependent: ClassVar[bool] = True

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


@register_descriptor("rung35_multishell")
class DMRung35MultishellDescriptor(Descriptor):
    """Multi-width rung-3.5 occupancy: per-spin, per-width local occupancies
    ``n_sigma(r; w) = A_w(r)^T P^sigma A_w(r) in [0, 1]``.

    The RADIAL generalization of the localized density-matrix projection used by
    NeuralXC (Dick and Fernandez-Serra, *Nat. Commun.* 11, 3509 (2020)) and
    carried in the DFS reference implementation, which projects the density
    matrix onto a localized basis and contracts the coefficients into
    rotationally invariant per-shell norms. Linear in the density matrix, so
    differentiable through the SCF with no eigendecomposition and no degeneracy
    hazard; bounded by the same Bessel argument as the single-width form.

    LIMITATION, stated because the name invites the stronger claim:
    ``fakemol_for_charges`` builds s-type projectors only, so this is the l = 0
    (radial) part of that construction. With one m per shell the invariant
    ``sqrt(sum_m c_{nlm}^2)`` collapses to the occupancy itself. Angular
    channels require solid-harmonic fakemols and are NOT implemented, so this
    should not be described as "the DFS descriptor" -- see
    ``xcquinox/alec/DEFERRED_WORK.md``.

    ``n_features`` is ``2 * len(alphas)`` (two spin channels per width) and the
    column order is ALPHA-MAJOR then spin. Setting ``alphas`` to a single width
    reproduces :class:`DMRung35Descriptor` bitwise.
    """
    n_features: int = eqx.field(default=6, static=True)
    alphas: tuple = eqx.field(default=DEFAULT_RUNG35_MULTISHELL_ALPHAS,
                              static=True)
    required_mol_keys: ClassVar[tuple[str, ...]] = ("rung35ms_features",)
    spin_mol_keys: ClassVar[tuple[str, ...]] = ("rung35ms_features_a",
                                                "rung35ms_features_b")
    density_matrix_dependent: ClassVar[bool] = True

    def __post_init__(self):
        # The base class rejects jax arrays in primitive-annotated fields; keep
        # that guard by delegating rather than shadowing it.
        super().__post_init__()
        # A YAML-sourced `alphas` arrives as a LIST (FeatureSpec._thaw maps an
        # untagged tuple back to a list by design), which makes the precompute
        # cache key unhashable. Coerce here so both the registry and the cache
        # see a tuple of plain floats.
        object.__setattr__(self, "alphas",
                           tuple(float(a) for a in self.alphas))
        if self.n_features != 2 * len(self.alphas):
            raise ValueError(
                f"DMRung35MultishellDescriptor.n_features ({self.n_features}) "
                f"must equal 2 * len(alphas) ({2 * len(self.alphas)}): two spin "
                f"channels per projector width."
            )

    @staticmethod
    def compute_from_dm(proj_ao_stack: jnp.ndarray,
                        dm: jnp.ndarray) -> jnp.ndarray:
        """Reassemble kernel: per-width occupancies from a LIVE DM and the
        constant projected-AO stack."""
        from xcquinox.alec.rung35 import compute_rung35_multishell_occupancy
        return compute_rung35_multishell_occupancy(proj_ao_stack, dm)

    def compute(self, mol_data):
        return mol_data["rung35ms_features"]


@register_descriptor("metagga")
class MetaGGAAlphaDescriptor(Descriptor):
    """Meta-GGA iso-orbital indicator ``alpha = (tau - tau_W)/tau_unif``: introduced by
    SCAN (Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015), Eq. 2), reused by DFS
    (Dick & Fernandez-Serra, PRB 104, L161109 (2021), Eq. 6). A genuine RUNG-3
    (meta-GGA) ingredient: the kinetic-energy density
    ``tau = 1/2 sum_{mu nu} P_{mu nu} grad chi_mu . grad chi_nu`` is a LINEAR
    contraction of the live one-particle DM against the AO gradients already on the
    grid (``eval_ao(deriv=1)``). So -- exactly like the rung-3.5 occupancy
    (:class:`DMRung35Descriptor`) -- it is self-consistent (a functional of the LIVE
    DM, recomputed each SCF cycle under REASSEMBLE), differentiable through the SCF,
    and needs NO new integrals, NO laplacian, NO ``deriv=2``.

    One feature: the total-density ``alpha`` (alpha=1 uniform gas, alpha=0 single
    orbital), clamped ``>= 0``. It feeds both the exchange and correlation networks
    RAW (the clamped alpha column); for the DFS-faithful meta-GGA net
    (``meta_gga=True``) alpha additionally drives the ``(x2 + tanh^2(x3))``
    UEG-recovery gate (x3 = ln((alpha+1)/2)) and the 1.174 Lieb-Oxford exchange
    ceiling. DEVIATION from DFS Eq. 10: DFS feeds the network the log-transformed
    x3 = ln((alpha+1)/2) as its alpha input, whereas here the raw alpha column is
    the MLP input and x3 enters only through the gate. Documented, not changed.
    See :mod:`xcquinox.alec.metagga`.
    """
    n_features: int = eqx.field(default=1, static=True)
    required_mol_keys: ClassVar[tuple[str, ...]] = ("metagga_features",)
    spin_mol_keys: ClassVar[tuple[str, ...]] = ("metagga_features_a",
                                                "metagga_features_b")
    density_matrix_dependent: ClassVar[bool] = True

    @staticmethod
    def compute_from_dm(ao_grad, rho, sigma, dm):
        """Reassemble kernel: total ``tau`` from the live DM contracted with the
        constant AO gradients, then the SCAN ``alpha`` from ``(rho, sigma, tau)`` --
        so the descriptor stays self-consistent each SCF cycle. ``ao_grad`` is the
        ``(3, N, nao)`` AO-gradient slice (``ao[1:4]`` of ``eval_ao(deriv=1)``)."""
        from xcquinox.alec.metagga import compute_tau_from_dm, compute_alpha
        tau = compute_tau_from_dm(ao_grad, dm)
        return compute_alpha(rho, sigma, tau).reshape(-1, 1)

    def compute(self, mol_data):
        return mol_data["metagga_features"]


def assemble_descriptor_features(descriptors: tuple[Descriptor, ...],
                                 mol_data: dict,
                                 spin_channel: int | None = None) -> jnp.ndarray:
    """Concatenate descriptor outputs left-to-right in declaration order.

    ``spin_channel=None`` returns the block of the physical (total) density,
    which the correlation term consumes. ``spin_channel=0`` / ``1`` returns the
    block of the symmetric doubled density ``diag(P_sigma, P_sigma)``, which is
    what the exact exchange spin scaling evaluates for the alpha / beta channel
    (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)). Column order and width are
    identical in all three blocks.
    """
    if not descriptors:
        return jnp.zeros((mol_data["rho_grid"].shape[0], 0))
    if spin_channel is None:
        return jnp.concatenate([d.compute(mol_data) for d in descriptors], axis=1)
    return jnp.concatenate(
        [d.compute_for_spin_channel(mol_data, spin_channel) for d in descriptors],
        axis=1,
    )
