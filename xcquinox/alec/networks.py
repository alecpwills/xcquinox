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


class _AlecLOB(eqx.Module):
    """Lieb-Oxford bound squash with limit marked static (D-H2 fix).

    Re-implements library xcquinox.net.LOB with limit as eqx.field(static=True)
    so gradient descent cannot modify the physical constant."""
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
    net: eqx.nn.MLP
    attention: _xnet.SelfAttentionBlock | None
    lobf: _AlecLOB | None

    def __init__(self, *, n_extra_features: int, depth: int, nodes: int,
                 use_self_attention: bool = False, seed: int = 42,
                 lob_lim: float | None = 1.804,
                 lower_rho_cutoff: float = 1e-12):
        self.n_extra_features = n_extra_features
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        self.use_self_attention = use_self_attention

        in_size = 1 + n_extra_features

        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 2)
        self.net = eqx.nn.MLP(
            in_size=in_size, out_size=1, depth=depth, width_size=nodes,
            activation=jax.nn.gelu, key=keys[0],
        )
        self.attention = (
            _xnet.SelfAttentionBlock(hidden_size=nodes, num_heads=1, key=keys[1])
            if use_self_attention else None
        )
        self.lobf = _AlecLOB(limit=lob_lim) if lob_lim is not None else None

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        rho = jnp.maximum(inputs[0], self.lower_rho_cutoff)
        sigma = jnp.maximum(inputs[1], 0.0)

        k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = jnp.atleast_1d(s).flatten()

        if self.n_extra_features > 0:
            extras = jnp.atleast_1d(inputs[2:2 + self.n_extra_features]).flatten()
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


class AlecGGA_CNet(eqx.Module):
    """Correlation enhancement network matching GGA_FcNet_extended verbatim.

    Differs from AlecGGA_XNet: feature base is [rs, s] (MLP in_size = 2 + extras),
    lob_lim defaults to 2.0.
    """
    n_extra_features: int = eqx.field(static=True)
    lob_lim: float | None = eqx.field(static=True)
    lower_rho_cutoff: float = eqx.field(static=True)
    use_self_attention: bool = eqx.field(static=True)
    net: eqx.nn.MLP
    attention: _xnet.SelfAttentionBlock | None
    lobf: _AlecLOB | None

    def __init__(self, *, n_extra_features: int, depth: int, nodes: int,
                 use_self_attention: bool = False, seed: int = 42,
                 lob_lim: float | None = 2.0,
                 lower_rho_cutoff: float = 1e-12):
        self.n_extra_features = n_extra_features
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        self.use_self_attention = use_self_attention

        in_size = 2 + n_extra_features

        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 2)
        self.net = eqx.nn.MLP(
            in_size=in_size, out_size=1, depth=depth, width_size=nodes,
            activation=jax.nn.gelu, key=keys[0],
        )
        self.attention = (
            _xnet.SelfAttentionBlock(hidden_size=nodes, num_heads=1, key=keys[1])
            if use_self_attention else None
        )
        self.lobf = _AlecLOB(limit=lob_lim) if lob_lim is not None else None

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        rho = jnp.maximum(inputs[0], self.lower_rho_cutoff)
        sigma = jnp.maximum(inputs[1], 0.0)

        rs = (3 / (4 * jnp.pi * rho)) ** (1 / 3)
        k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)

        rs = jnp.atleast_1d(rs).flatten()
        s = jnp.atleast_1d(s).flatten()

        if self.n_extra_features > 0:
            extras = jnp.atleast_1d(inputs[2:2 + self.n_extra_features]).flatten()
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


def create_network_pair(arch: ArchitectureConfig, seed: int = 42,
                        lower_rho_cutoff: float = 1e-12):
    """Build a fresh (xnet, cnet) pair for the architecture.

    C-H1: xnet lob_lim resolved via arch.resolved_xnet_lob_lim (None when
    LiebOxfordBound constraint is active). Cnet uses its default lob_lim=2.0.
    """
    n_extra_features = sum(d.n_features for d in arch.materialize_descriptors())
    xnet = AlecGGA_XNet(
        n_extra_features=n_extra_features, depth=arch.depth, nodes=arch.nodes,
        use_self_attention=arch.attention, seed=seed,
        lob_lim=arch.resolved_xnet_lob_lim,
        lower_rho_cutoff=lower_rho_cutoff,
    )
    cnet = AlecGGA_CNet(
        n_extra_features=n_extra_features, depth=arch.depth, nodes=arch.nodes,
        use_self_attention=arch.attention, seed=seed + 1,
        lower_rho_cutoff=lower_rho_cutoff,
    )
    return xnet, cnet
