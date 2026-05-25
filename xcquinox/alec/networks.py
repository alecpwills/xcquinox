"""xcquinox.alec.networks — Exchange and correlation enhancement networks.

Implements THE SPEC §5: AlecGGA_XNet (exchange), AlecGGA_CNet (correlation),
_AlecLOB (static-limit Lieb-Oxford squash), and create_network_pair factory.
"""
import math

import jax
import jax.numpy as jnp
import equinox as eqx
import xcquinox.net as _xnet

from xcquinox.alec.config import ArchitectureConfig
from xcquinox.alec.constraints import Constraint, _compose_constraints


class _AlecLOB(eqx.Module):
    """Output-range squash with limit marked static (D-H2 fix).

    Re-implements library xcquinox.net.LOB with limit as eqx.field(static=True)
    so gradient descent cannot modify the physical constant.

    When used for the exchange path (limit=1.804 = 1 + kappa_PBE) the
    functional form coincides with a local Lieb-Oxford-style ceiling on F_x
    (Lieb & Oxford 1981; PBE 1996 eq. 14).  Dick & Fernández-Serra (PRB 104
    L161109, 2021) use 1.174 for the exchange ceiling.

    When used for the correlation path (limit=2.0) the purpose is
    NON-NEGATIVITY of F_c, not a Lieb-Oxford bound.  The I₂(x) transform in
    Dick (2021) eq. (13) maps the network pre-activation through a symmetric
    sigmoid so that F_c ∈ [0, 2]; the upper value 2 is an incidental
    byproduct of the symmetric form, NOT a Lieb-Oxford constraint (the
    Lieb-Oxford theorem bounds the total E_xc, never F_c alone)."""
    limit: float = eqx.field(static=True)

    def __init__(self, limit: float):
        if not isinstance(limit, (int, float)) or isinstance(limit, bool):
            raise TypeError(
                f"_AlecLOB.limit must be a plain Python int/float "
                f"(got {type(limit).__name__})."
            )
        if not math.isfinite(limit):
            raise ValueError(
                f"_AlecLOB.limit must be finite (got {limit}); "
                f"float('inf')/float('nan') would produce NaN in the "
                f"forward pass via log(limit - 1)."
            )
        if limit <= 1.0:
            raise ValueError(
                f"_AlecLOB.limit must be > 1.0 (got {limit}); "
                f"the log(limit - 1) term requires a positive argument."
            )
        self.limit = limit

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Matches xcquinox/net.py:40 (LOB.__call__) exactly.
        return self.limit * jax.nn.sigmoid(x - jnp.log(self.limit - 1.0)) - 1.0


class AlecGGA_XNet(eqx.Module):
    """Exchange enhancement network matching GGA_FxNet_extended verbatim.

    Input: 1-D tensor [rho, sigma, *extras].
    Output: scalar 1 + F_x^enhancement.
    """
    n_extra_features: int = eqx.field(static=True)
    lob_lim: float | None = eqx.field(static=True)
    lower_rho_cutoff: float = eqx.field(static=True)
    use_self_attention: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    # Physical constraints enforced INTRINSICALLY by the network forward, so the
    # same constrained functional is used in pretraining, training, and eval.
    # Stored static (constraint params are all static -> no array leaves), so the
    # serialized leaf stream is unchanged and old checkpoints still deserialize.
    constraints: tuple = eqx.field(static=True)
    net: eqx.nn.MLP
    attention: _xnet.SelfAttentionBlock | None
    lobf: _AlecLOB | None

    def __init__(self, *, n_extra_features: int, depth: int, nodes: int,
                 use_self_attention: bool = False, seed: int = 42,
                 lob_lim: float | None = 1.804,
                 lower_rho_cutoff: float = 1e-12,
                 num_heads: int = 1,
                 constraints: tuple = ()):
        if use_self_attention and nodes % num_heads != 0:
            raise ValueError(
                f"AlecGGA_XNet: use_self_attention=True requires "
                f"nodes ({nodes}) divisible by num_heads ({num_heads})"
            )
        self.n_extra_features = n_extra_features
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        self.use_self_attention = use_self_attention
        self.num_heads = num_heads
        self.constraints = tuple(constraints)

        in_size = 1 + n_extra_features

        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 2)
        self.net = eqx.nn.MLP(
            in_size=in_size, out_size=1, depth=depth, width_size=nodes,
            activation=jax.nn.gelu, key=keys[0],
        )
        self.attention = (
            _xnet.SelfAttentionBlock(hidden_size=nodes, num_heads=num_heads, key=keys[1])
            if use_self_attention else None
        )
        self.lobf = _AlecLOB(limit=lob_lim) if lob_lim is not None else None

    def _core(self, rho, sigma, features):
        """Unconstrained exchange forward: (rho, sigma, features) -> 1 + F_x.

        This is the raw MLP path (reduced-gradient feature, tanh gate, optional
        built-in ``lobf`` wrap). The physical constraints in ``self.constraints``
        wrap THIS function — composed in ``__call__`` — so a constraint that
        re-invokes its inner_fn at a rescaled density (e.g. ScalingSymmetric)
        re-runs the full forward, exactly as the model-level composition did."""
        rho = jnp.maximum(rho, self.lower_rho_cutoff)
        sigma = jnp.maximum(sigma, 0.0)

        k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = jnp.atleast_1d(s).flatten()

        if self.n_extra_features > 0:
            extras = jnp.atleast_1d(features).flatten()
            netinp = jnp.concatenate([s, extras])
        else:
            netinp = s

        tanhterm = jnp.tanh(s) ** 2

        if self.attention is not None:
            x = netinp
            layers = self.net.layers
            for i, layer in enumerate(layers[:-1]):
                x = layer(x)
                x = jax.nn.gelu(x)
                if i == 0:
                    x = self.attention(x)
            netterm = layers[-1](x)
        else:
            netterm = self.net(netinp)

        gated = tanhterm * netterm
        if self.lobf is not None:
            lobterm = self.lobf(gated)
            return 1 + lobterm.squeeze()
        return 1 + gated.squeeze()

    def eval_core(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """UNCONSTRAINED 1 + F_x for a packed ``[rho, sigma, *extras]`` row.

        Exposed for introspection (constraint-violation reporting), which needs
        the raw value to compare against the constrained output."""
        return self._core(inputs[0], inputs[1], inputs[2:])

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        rho = inputs[0]
        sigma = inputs[1]
        features = inputs[2:]
        chain = _compose_constraints(self._core, self.constraints)
        return chain(rho, sigma, features)


class AlecGGA_CNet(eqx.Module):
    """Correlation enhancement network matching GGA_FcNet_extended verbatim.

    Differs from AlecGGA_XNet: feature base is [rs, s] (MLP in_size = 2 + extras),
    lob_lim defaults to 2.0.

    The default lob_lim=2.0 is NOT a Lieb-Oxford bound on F_c.  It implements
    the I₂(x) non-negativity squash from Dick & Fernández-Serra (PRB 104
    L161109, 2021) eq. (13): a symmetric sigmoid maps the pre-activation to
    F_c ∈ [0, 2], ensuring the correlation enhancement factor is non-negative.
    The value 2.0 is an artefact of the symmetric form; the Lieb-Oxford
    theorem constrains the total E_xc, not F_c individually.
    """
    n_extra_features: int = eqx.field(static=True)
    lob_lim: float | None = eqx.field(static=True)
    lower_rho_cutoff: float = eqx.field(static=True)
    use_self_attention: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    use_spin_polarization: bool = eqx.field(static=True)
    # Physical constraints enforced intrinsically by the forward (see XNet).
    constraints: tuple = eqx.field(static=True)
    net: eqx.nn.MLP
    attention: _xnet.SelfAttentionBlock | None
    lobf: _AlecLOB | None

    def __init__(self, *, n_extra_features: int, depth: int, nodes: int,
                 use_self_attention: bool = False, seed: int = 42,
                 lob_lim: float | None = 2.0,
                 lower_rho_cutoff: float = 1e-12,
                 num_heads: int = 1,
                 use_spin_polarization: bool = False,
                 constraints: tuple = ()):
        if use_self_attention and nodes % num_heads != 0:
            raise ValueError(
                f"AlecGGA_CNet: use_self_attention=True requires "
                f"nodes ({nodes}) divisible by num_heads ({num_heads})"
            )
        self.n_extra_features = n_extra_features
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        self.use_self_attention = use_self_attention
        self.num_heads = num_heads
        self.constraints = tuple(constraints)
        # P2-03: when True, the cnet takes a spin-polarization input feature
        # x1 = 1/2[(1+zeta)^{4/3}+(1-zeta)^{4/3}] (Dick & Fernández-Serra 2021,
        # input feature x1 / eq. (13)) inserted after [rs, s]. The model packs
        # zeta at inputs[2] and shifts descriptor extras to inputs[3:].
        self.use_spin_polarization = use_spin_polarization

        in_size = 2 + (1 if use_spin_polarization else 0) + n_extra_features

        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 2)
        self.net = eqx.nn.MLP(
            in_size=in_size, out_size=1, depth=depth, width_size=nodes,
            activation=jax.nn.gelu, key=keys[0],
        )
        self.attention = (
            _xnet.SelfAttentionBlock(hidden_size=nodes, num_heads=num_heads, key=keys[1])
            if use_self_attention else None
        )
        self.lobf = _AlecLOB(limit=lob_lim) if lob_lim is not None else None

    def _core(self, rho, sigma, features, zeta):
        """Unconstrained correlation forward: (rho, sigma, features) -> 1 + F_c.

        ``zeta`` (spin polarization) is threaded through as a closed-over scalar
        rather than via the constraint signature, because the c-constraints
        operate on (rho, sigma, F) only — matching the model-level
        ``_batched_network_apply_polarized`` base_fn that also captured zeta."""
        rho = jnp.maximum(rho, self.lower_rho_cutoff)
        sigma = jnp.maximum(sigma, 0.0)

        rs = (3 / (4 * jnp.pi * rho)) ** (1 / 3)
        k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)

        rs = jnp.atleast_1d(rs).flatten()
        s = jnp.atleast_1d(s).flatten()

        if self.use_spin_polarization:
            # P2-03: zeta = (rho_a - rho_b)/rho_tot feeds the bounded Dick
            # feature x1 = 1/2[(1+zeta)^{4/3}+(1-zeta)^{4/3}] (in [1, 2^{1/3}]
            # for zeta in [-1,1]; x1=1 at zeta=0, recovering the unpolarized
            # input so an RKS (zeta=0) call sees [rs, s, 1, extras]).
            zeta_c = jnp.clip(zeta, -1.0, 1.0)
            x1 = jnp.atleast_1d(
                0.5 * ((1.0 + zeta_c) ** (4 / 3) + (1.0 - zeta_c) ** (4 / 3))
            ).flatten()
            if self.n_extra_features > 0:
                extras = jnp.atleast_1d(features).flatten()
                netinp = jnp.concatenate([rs, s, x1, extras])
            else:
                netinp = jnp.concatenate([rs, s, x1])
        elif self.n_extra_features > 0:
            extras = jnp.atleast_1d(features).flatten()
            netinp = jnp.concatenate([rs, s, extras])
        else:
            netinp = jnp.concatenate([rs, s])

        tanhterm = jnp.tanh(s) ** 2

        if self.attention is not None:
            x = netinp
            layers = self.net.layers
            for i, layer in enumerate(layers[:-1]):
                x = layer(x)
                x = jax.nn.gelu(x)
                if i == 0:
                    x = self.attention(x)
            netterm = layers[-1](x)
        else:
            netterm = self.net(netinp)

        gated = tanhterm * netterm
        if self.lobf is not None:
            lobterm = self.lobf(gated)
            return 1 + lobterm.squeeze()
        return 1 + gated.squeeze()

    def eval_core(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """UNCONSTRAINED 1 + F_c for a packed input row. Exposed for
        constraint-violation introspection."""
        if self.use_spin_polarization:
            return self._core(inputs[0], inputs[1], inputs[3:], inputs[2])
        return self._core(inputs[0], inputs[1], inputs[2:], 0.0)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        rho = inputs[0]
        sigma = inputs[1]
        if self.use_spin_polarization:
            zeta = inputs[2]
            features = inputs[3:]
        else:
            zeta = 0.0
            features = inputs[2:]
        base = lambda r, s, f: self._core(r, s, f, zeta)  # noqa: E731
        chain = _compose_constraints(base, self.constraints)
        return chain(rho, sigma, features)


def create_network_pair(arch: ArchitectureConfig, seed: int = 42,
                        lower_rho_cutoff: float = 1e-12):
    """Build a fresh (xnet, cnet) pair for the architecture.

    C-H1: xnet lob_lim resolved via arch.resolved_xnet_lob_lim (None when
    LiebOxfordBound constraint is active).  Cnet lob_lim resolved via
    arch.resolved_cnet_lob_lim (default 2.0, a non-negativity squash on F_c
    per Dick & Fernández-Serra 2021 eq. (13) — not a Lieb-Oxford bound).

    Physical constraints are materialized from the arch and handed to the
    networks, which enforce them INTRINSICALLY in their forward pass.  The same
    constrained functional is therefore used everywhere the network is called —
    pretraining, training, and evaluation — rather than being applied only by
    the composed model at train/eval time.
    """
    n_extra_features = sum(d.n_features for d in arch.materialize_descriptors())
    xnet = AlecGGA_XNet(
        n_extra_features=n_extra_features, depth=arch.depth, nodes=arch.nodes,
        use_self_attention=arch.attention, seed=seed,
        lob_lim=arch.resolved_xnet_lob_lim,
        lower_rho_cutoff=lower_rho_cutoff,
        num_heads=arch.num_heads,
        constraints=arch.materialize_x_constraints(),
    )
    cnet = AlecGGA_CNet(
        n_extra_features=n_extra_features, depth=arch.depth, nodes=arch.nodes,
        use_self_attention=arch.attention, seed=seed + 1,
        lob_lim=arch.resolved_cnet_lob_lim,        # B-LOW audit fix
        lower_rho_cutoff=lower_rho_cutoff,
        num_heads=arch.num_heads,
        # P2-03: zeta-aware correlation network when the arch opts in.
        use_spin_polarization=arch.use_polarized_correlation,
        constraints=arch.materialize_c_constraints(),
    )
    return xnet, cnet
