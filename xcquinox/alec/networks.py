"""xcquinox.alec.networks: Exchange and correlation enhancement networks.

Implements THE SPEC §5: AlecGGA_XNet (exchange), AlecGGA_CNet (correlation),
_AlecLOB (static-limit Lieb-Oxford squash), and create_network_pair factory.
"""
import math

import jax
import jax.numpy as jnp
import equinox as eqx
import xcquinox.net as _xnet

from xcquinox.alec.config import DESCRIPTOR_COORDINATES, ArchitectureConfig
from xcquinox.alec.constraints import Constraint, _compose_constraints
from xcquinox.alec.metagga import (
    _ALPHA_MAX, _ALPHA_SMOOTHING_WIDTH, invert_smooth_positive_part)
from xcquinox.alec.parents import (
    PARENTS, lob_preimage, parent_fc, parent_for_arch, parent_fx)


# ---------------------------------------------------------------------------
# Row coordinates shared by the two networks
# ---------------------------------------------------------------------------

#: The offset inside the logarithm of the DFS density coordinate
#: (dpyscfl net.py line 39, ``self.loge = 1e-5``).
_DFS_LOG_EPS = 1e-5


def _dfs_log_transform(x):
    """``(1 - exp(-x^2)) ln(x + 1)``: the DFS reduced-gradient coordinate
    (dpyscfl net.py lines 198 and 204 with ``s_gam = 1``, line 40; Dick and
    Fernandez-Serra, PRB 104, L161109 (2021), eq. 9)."""
    return (1.0 - jnp.exp(-x * x)) * jnp.log(x + 1.0)


def _dfs_indicator_coordinate(alpha_raw):
    """``ln((alpha + 1)/2)`` of the RAW iso-orbital indicator (dpyscfl net.py
    line 220; PRB 104 L161109 eq. 10): the meta-GGA MLP coordinate under the
    DFS coordinates."""
    return jnp.log((alpha_raw + 1.0) / 2.0)


def _raw_indicator(alpha_column):
    """The raw iso-orbital indicator a stored ``metagga`` column encodes.

    The column is ``min(p(alpha_raw), _ALPHA_MAX)`` with ``p`` the smooth
    positive part of width ``_ALPHA_SMOOTHING_WIDTH``
    (``metagga.compute_alpha``, the manifest's ``ALPHA_DEFINITION``): below the
    ceiling ``invert_smooth_positive_part`` recovers ``alpha_raw`` exactly; at
    and above it the ceiling is returned, where SCAN's switching function has
    saturated (SPEC_parent_anchor.md Section 3.1). Differentiable in the
    column, so a potential taken through it inherits the column's
    regularization rather than the raw indicator's response. The inverse's
    division is guarded (the column is strictly positive by construction,
    floor ``width / 2``; the guard keeps the unselected branch finite under
    differentiation).
    """
    safe = jnp.where(alpha_column > 0.0, alpha_column, 1.0)
    return jnp.where(alpha_column < _ALPHA_MAX,
                     invert_smooth_positive_part(safe, _ALPHA_SMOOTHING_WIDTH),
                     alpha_column)


class _AlecLOB(eqx.Module):
    """Output-range squash with limit marked static.

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

    MLP coordinates, by ``descriptor_coordinates``:

    * ``"legacy"`` -- ``[s or its log transform, *extras]`` (today's layout,
      byte for byte; the meta-GGA extras carry the raw clamped indicator).
    * ``"dfs"`` -- ``[x_s, *extras]`` with ``x_s = (1 - exp(-s^2)) ln(s + 1)``
      of the doubled channel and, on the meta-GGA rung, ``x_alpha =
      ln((alpha + 1)/2)`` of the raw indicator in place of the indicator
      column: the inputs of the DFS exchange network, which reads the
      reduced gradient (and the indicator) and never the density (dpyscfl
      net.py ``get_scf`` lines 745 and 750, ``X_L(n_input=1, use=[1])`` and
      ``X_L(n_input=2, use=[1, 2])``). The width is the legacy width.

    With ``parent`` set the network is anchored to its parent functional
    (``parents.py``): see ``_core``.
    """
    n_extra_features: int = eqx.field(static=True)
    lob_lim: float | None = eqx.field(static=True)
    lower_rho_cutoff: float = eqx.field(static=True)
    use_self_attention: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    # When True, applies the Dick XCDiff log-transform
    # ``(1 - exp(-s²)) · log(s + 1)`` to the reduced-gradient feature before
    # the MLP. Prevents feature magnitude saturation for large ``s``. The
    # ``tanh(s)²`` UEG gate uses raw ``s`` (structural physics constraint, not
    # a feature transform).
    descriptor_log_transform: bool = eqx.field(default=False, static=True)
    # Meta-GGA (DFS-faithful, PRB 104 L161109 Eq. 12): when True the GGA UEG gate
    # ``tanh(s)^2`` becomes DFS's ``(x2 + tanh^2(x3))`` prefactor, where x2 is the
    # log-transformed s ``(1-e^{-s^2}) ln(s+1)`` and ``x3 = ln((alpha+1)/2)``, so the
    # gate -> 0 (UEG) at s=0 AND alpha=1. ``alpha`` is read from the descriptor
    # ``features`` at ``metagga_alpha_index`` (the MetaGGAAlphaDescriptor column).
    # Pair with ``lob_lim=1.174`` for the DFS exchange ceiling.
    meta_gga: bool = eqx.field(default=False, static=True)
    metagga_alpha_index: int = eqx.field(default=-1, static=True)
    # Physical constraints enforced INTRINSICALLY by the network forward, so the
    # same constrained functional is used in pretraining, training, and eval.
    # Stored static (constraint params are all static -> no array leaves) so the
    # serialized leaf stream is unchanged.
    constraints: tuple = eqx.field(static=True)
    # Parent anchor (SPEC_parent_anchor.md Section 3.2): "pbe" | "scan" | None.
    # The parent's F_x is evaluated on the row's physical inputs and the gated
    # MLP output is added in the PRE-IMAGE of the bounded map, so the network
    # returns the parent when its MLP is zero. Static: it is part of the
    # treedef, not of the leaf stream, so a checkpoint does not reveal it (the
    # record beside the checkpoint does). None = today's network, byte for byte.
    parent: str | None = eqx.field(default=None, static=True)
    # The coordinates the MLP reads the row in, "legacy" | "dfs" (class docstring).
    descriptor_coordinates: str = eqx.field(default="legacy", static=True)
    net: eqx.nn.MLP
    attention: _xnet.SelfAttentionBlock | None
    lobf: _AlecLOB | None

    def __init__(self, *, n_extra_features: int, depth: int, nodes: int,
                 use_self_attention: bool = False, seed: int = 42,
                 lob_lim: float | None = 1.804,
                 lower_rho_cutoff: float = 1e-12,
                 num_heads: int = 1,
                 constraints: tuple = (),
                 descriptor_log_transform: bool = False,
                 meta_gga: bool = False,
                 metagga_alpha_index: int = -1,
                 zero_init_final_layer: bool = False,
                 parent: str | None = None,
                 descriptor_coordinates: str = "legacy"):
        if use_self_attention and nodes % num_heads != 0:
            raise ValueError(
                f"AlecGGA_XNet: use_self_attention=True requires "
                f"nodes ({nodes}) divisible by num_heads ({num_heads})"
            )
        if parent is not None and parent not in PARENTS:
            raise ValueError(
                f"AlecGGA_XNet: parent must be one of {PARENTS} or None, "
                f"got {parent!r}"
            )
        if parent is not None and lob_lim is None:
            raise ValueError(
                "AlecGGA_XNet: a parent anchor adds the network's output in "
                "the pre-image of the built-in bounded map, so lob_lim=None "
                "(a lieb_oxford constraint without double_lob_clamp_allowed) "
                "cannot be anchored"
            )
        if descriptor_coordinates not in DESCRIPTOR_COORDINATES:
            raise ValueError(
                f"AlecGGA_XNet: descriptor_coordinates must be one of "
                f"{DESCRIPTOR_COORDINATES}, got {descriptor_coordinates!r}"
            )
        self.n_extra_features = n_extra_features
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        self.use_self_attention = use_self_attention
        self.num_heads = num_heads
        self.descriptor_log_transform = descriptor_log_transform
        self.meta_gga = meta_gga
        self.metagga_alpha_index = metagga_alpha_index
        self.constraints = tuple(constraints)
        self.parent = parent
        self.descriptor_coordinates = descriptor_coordinates

        # The exchange MLP's width is the same in both coordinate sets: the
        # DFS exchange network reads the reduced gradient (and the indicator,
        # which is already an extras column) and never the density.
        in_size = 1 + n_extra_features

        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 2)
        self.net = eqx.nn.MLP(
            in_size=in_size, out_size=1, depth=depth, width_size=nodes,
            activation=jax.nn.gelu, key=keys[0],
        )
        # Zero the final MLP layer so 1 + LOB(tanh(s)² · MLP) ≈ 1 at init,
        # ensuring the untrained network returns F_x=1 -- Slater/LDA exchange (the
        # uniform-gas limit, since F multiplies lda_x in models.py), NOT PBE.
        if zero_init_final_layer:
            self.net = eqx.tree_at(
                lambda m: (m.layers[-1].weight, m.layers[-1].bias),
                self.net,
                replace=(jnp.zeros_like(self.net.layers[-1].weight),
                         jnp.zeros_like(self.net.layers[-1].bias)))
        self.attention = (
            _xnet.SelfAttentionBlock(hidden_size=nodes, num_heads=num_heads, key=keys[1])
            if use_self_attention else None
        )
        self.lobf = _AlecLOB(limit=lob_lim) if lob_lim is not None else None

    def _core(self, rho, sigma, features, raw=False):
        """Unconstrained exchange forward: (rho, sigma, features) -> 1 + F_x.

        This is the raw MLP path (reduced-gradient feature, tanh gate, optional
        built-in ``lobf`` wrap). The physical constraints in ``self.constraints``
        wrap THIS function, composed in ``__call__``, so a constraint that
        re-invokes its inner_fn at a rescaled density (e.g. ScalingSymmetric)
        re-runs the full forward, exactly as the model-level composition did.

        With ``parent`` set the gated output enters in the pre-image of the
        bounded map at the parent's value, ``1 + L(z_parent + gated)`` with
        ``z_parent = lob_preimage(F_parent, limit)`` (SPEC_parent_anchor.md
        Section 3.2), so ``gated = 0`` returns ``F_parent`` to round-off and
        ``F`` stays in ``(0, limit)`` for every ``gated``; ``raw=True`` returns
        ``F_parent + gated`` instead, the unsquashed value ``eval_core``
        reports. ``raw`` has no effect on an unanchored network, whose path is
        today's byte for byte."""
        rho = jnp.maximum(rho, self.lower_rho_cutoff)
        sigma = jnp.maximum(sigma, 0.0)

        k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = jnp.atleast_1d(s).flatten()

        # The raw iso-orbital indicator the row's smoothed, capped column
        # encodes (``_raw_indicator``): read by the DFS coordinate x_alpha and
        # by the SCAN parent. None on the GGA rungs.
        alpha_raw = None
        if self.meta_gga:
            alpha_raw = _raw_indicator(
                jnp.atleast_1d(features).flatten()[self.metagga_alpha_index])

        # When descriptor_log_transform=True, feed the MLP a Dick XCDiff
        # log-compressed s; otherwise raw s. The tanh(s)² UEG gate below is
        # ALWAYS computed from raw s (it's a structural physics constraint,
        # not a feature transform).
        if self.descriptor_coordinates == "dfs":
            # DFS coordinates, dpyscfl net.py get_descriptors, spin-scaling
            # branch (the exchange network's): x_s = (1 - exp(-s^2)) ln(s + 1)
            # of the doubled channel (lines 195-198; PRB 104 L161109 eq. 9).
            # The exchange MLP reads x_s alone at the GGA level (get_scf line
            # 745, X_L(n_input=1, use=[1])) and x_s with x_alpha at the
            # meta-GGA level (line 750, use=[1, 2]; x_alpha below).
            s_mlp = _dfs_log_transform(s)
        elif self.descriptor_log_transform:
            s_mlp = (1.0 - jnp.exp(-s * s)) * jnp.log(s + 1.0)
        else:
            s_mlp = s

        if self.n_extra_features > 0:
            extras = jnp.atleast_1d(features).flatten()
            if self.descriptor_coordinates == "dfs" and self.meta_gga:
                # x_alpha = ln((alpha + 1)/2) of the RAW indicator (dpyscfl
                # net.py line 220; eq. 10) in place of the raw clamped column
                # the legacy layout feeds; the other extras are unchanged.
                extras = extras.at[self.metagga_alpha_index].set(
                    _dfs_indicator_coordinate(alpha_raw))
            netinp = jnp.concatenate([s_mlp, extras])
        else:
            netinp = s_mlp

        if self.meta_gga:
            # DFS Eq. 12 meta-GGA UEG-recovery prefactor (x2 + tanh^2(x3)): x2 =
            # log-transformed s, x3 = ln((alpha+1)/2). alpha is the descriptor
            # column at metagga_alpha_index (also an MLP input via `extras`), so it
            # is used in BOTH the gate and the MLP -- the DFS exchange (s, alpha).
            # DEVIATION from DFS Eq. 10/12: the MLP receives RAW clamped alpha (the
            # descriptor column), whereas DFS feeds the network the log-transformed
            # x3 = ln((alpha+1)/2); here x3 enters only through this gate, never as
            # an MLP input. Documented, not changed (feeding x3 to the MLP would
            # invalidate existing checkpoints).
            alpha = jnp.atleast_1d(features).flatten()[self.metagga_alpha_index]
            x2 = (1.0 - jnp.exp(-s * s)) * jnp.log(s + 1.0)
            x3 = jnp.log((alpha + 1.0) / 2.0)
            tanhterm = jnp.atleast_1d(x2).flatten() + jnp.tanh(x3) ** 2
        else:
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
        if self.parent is not None:
            f_parent = parent_fx(self.parent, rho, sigma, alpha_raw)
            if raw:
                return (f_parent + gated).squeeze()
            z_parent = lob_preimage(f_parent, self.lobf.limit)
            return 1 + self.lobf(z_parent + gated).squeeze()
        if self.lobf is not None:
            lobterm = self.lobf(gated)
            return 1 + lobterm.squeeze()
        return 1 + gated.squeeze()

    def eval_core(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """UNCONSTRAINED 1 + F_x for a packed ``[rho, sigma, *extras]`` row.

        Exposed for introspection (constraint-violation reporting), which needs
        the raw value to compare against the constrained output. An anchored
        network reports ``F_parent + gated``, the value before the bounded
        map; an unanchored one reports today's core value."""
        return self._core(inputs[0], inputs[1], inputs[2:], raw=True)

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

    MLP coordinates, by ``descriptor_coordinates``:

    * ``"legacy"`` -- ``[rs, s (each optionally log-transformed), x1 =
      spinscale (polarized only), *extras]``, today's layout byte for byte.
    * ``"dfs"`` -- ``[x0, x1, x_s, *extras]`` in the order the DFS correlation
      network consumes them (dpyscfl net.py ``get_scf`` lines 746 and 751,
      ``C_L(n_input=3)`` and ``C_L(n_input=4)``): ``x0 = ln(rho^(1/3) + 1e-5)``,
      ``x1 = ln(0.5 [(1 + zeta)^(4/3) + (1 - zeta)^(4/3)])``, ``x_s =
      (1 - exp(-s^2)) ln(s + 1)`` of the TOTAL density, and on the meta-GGA
      rung ``x_alpha = ln((alpha + 1)/2)`` of the raw indicator in place of
      the indicator column. Requires the polarized network (``x1`` is a DFS
      input); the width is ``3 + n_extra_features``, the legacy polarized
      width.

    With ``parent`` set the network is anchored to its parent functional
    relative to the model's polarized PW92 baseline (``parents.pbe_fc``):
    see ``_core``. An anchored correlation network must be
    polarization-aware; a zeta-blind one is refused at construction.
    """
    n_extra_features: int = eqx.field(static=True)
    lob_lim: float | None = eqx.field(static=True)
    lower_rho_cutoff: float = eqx.field(static=True)
    use_self_attention: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    use_spin_polarization: bool = eqx.field(static=True)
    # When True, applies the ``(1 - exp(-x²)) · log(x + 1)`` transform to both
    # MLP inputs ``rs`` and ``s`` before concatenation with extras (prevents
    # feature magnitude saturation). This is DFS's reduced-gradient form (Eq. 9);
    # applying it to the density feature ``rs`` DEVIATES from DFS Eq. 7, which
    # uses a plain log on the density variable ``x0 = n^{1/3}`` (see ``_core``).
    # The ``tanh(s)²`` UEG gate uses raw ``s`` (structural physics constraint).
    descriptor_log_transform: bool = eqx.field(default=False, static=True)
    # Meta-GGA (DFS Eq. 13): same (x2 + tanh^2(x3)) UEG-recovery prefactor as the
    # X-net; alpha read from the descriptor features at metagga_alpha_index. lob_lim
    # stays 2.0 (F_c non-negativity, not a Lieb-Oxford bound).
    meta_gga: bool = eqx.field(default=False, static=True)
    metagga_alpha_index: int = eqx.field(default=-1, static=True)
    # Physical constraints enforced intrinsically by the forward (see XNet).
    constraints: tuple = eqx.field(static=True)
    # Parent anchor, "pbe" | "scan" | None (see AlecGGA_XNet.parent).
    parent: str | None = eqx.field(default=None, static=True)
    # The coordinates the MLP reads the row in, "legacy" | "dfs" (class docstring).
    descriptor_coordinates: str = eqx.field(default="legacy", static=True)
    net: eqx.nn.MLP
    attention: _xnet.SelfAttentionBlock | None
    lobf: _AlecLOB | None

    def __init__(self, *, n_extra_features: int, depth: int, nodes: int,
                 use_self_attention: bool = False, seed: int = 42,
                 lob_lim: float | None = 2.0,
                 lower_rho_cutoff: float = 1e-12,
                 num_heads: int = 1,
                 use_spin_polarization: bool = False,
                 constraints: tuple = (),
                 descriptor_log_transform: bool = False,
                 meta_gga: bool = False,
                 metagga_alpha_index: int = -1,
                 zero_init_final_layer: bool = False,
                 parent: str | None = None,
                 descriptor_coordinates: str = "legacy"):
        if use_self_attention and nodes % num_heads != 0:
            raise ValueError(
                f"AlecGGA_CNet: use_self_attention=True requires "
                f"nodes ({nodes}) divisible by num_heads ({num_heads})"
            )
        if parent is not None and parent not in PARENTS:
            raise ValueError(
                f"AlecGGA_CNet: parent must be one of {PARENTS} or None, "
                f"got {parent!r}"
            )
        if parent is not None and not use_spin_polarization:
            raise ValueError(
                "AlecGGA_CNet: an anchored correlation network must be "
                "polarization-aware (use_spin_polarization=True). The parent's "
                "correlation is divided by the model's zeta-dependent PW92 "
                "baseline, and the pretraining data forms its open-shell Fc "
                "targets against that same baseline; a zeta-blind network "
                "multiplies the unpolarized baseline instead, and the two "
                "disagree on open shells (measured 14.9 mHa on the N atom's "
                "correlation term, SPEC_parent_anchor.md Section 3.1)"
            )
        if parent is not None and lob_lim is None:
            raise ValueError(
                "AlecGGA_CNet: a parent anchor adds the network's output in "
                "the pre-image of the built-in bounded map, so lob_lim=None "
                "cannot be anchored"
            )
        if descriptor_coordinates not in DESCRIPTOR_COORDINATES:
            raise ValueError(
                f"AlecGGA_CNet: descriptor_coordinates must be one of "
                f"{DESCRIPTOR_COORDINATES}, got {descriptor_coordinates!r}"
            )
        if descriptor_coordinates == "dfs" and not use_spin_polarization:
            raise ValueError(
                "AlecGGA_CNet: descriptor_coordinates='dfs' requires "
                "use_spin_polarization=True: x1 = ln(spinscale) is an input of "
                "the DFS correlation network (dpyscfl net.py line 191), which "
                "a zeta-blind row does not carry"
            )
        self.parent = parent
        self.descriptor_coordinates = descriptor_coordinates
        self.n_extra_features = n_extra_features
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        self.use_self_attention = use_self_attention
        self.num_heads = num_heads
        self.descriptor_log_transform = descriptor_log_transform
        self.meta_gga = meta_gga
        self.metagga_alpha_index = metagga_alpha_index
        self.constraints = tuple(constraints)
        # When True, the correlation network takes a spin-polarization input
        # feature x1 = 1/2[(1+zeta)^{4/3}+(1-zeta)^{4/3}] (Dick & Fernández-Serra
        # 2021 eq. 4) allowing the correlation functional to depend on relative
        # spin density. Zeta is packed at inputs[2], descriptor extras at
        # inputs[3:].
        self.use_spin_polarization = use_spin_polarization

        in_size = 2 + (1 if use_spin_polarization else 0) + n_extra_features

        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 2)
        self.net = eqx.nn.MLP(
            in_size=in_size, out_size=1, depth=depth, width_size=nodes,
            activation=jax.nn.gelu, key=keys[0],
        )
        # Zero the final MLP layer so 1 + LOB(tanh(s)² · MLP) ≈ 1 at init
        # (Fc -> 1, the PW92/LDA-correlation limit, NOT PBE).
        if zero_init_final_layer:
            self.net = eqx.tree_at(
                lambda m: (m.layers[-1].weight, m.layers[-1].bias),
                self.net,
                replace=(jnp.zeros_like(self.net.layers[-1].weight),
                         jnp.zeros_like(self.net.layers[-1].bias)))
        self.attention = (
            _xnet.SelfAttentionBlock(hidden_size=nodes, num_heads=num_heads, key=keys[1])
            if use_self_attention else None
        )
        self.lobf = _AlecLOB(limit=lob_lim) if lob_lim is not None else None

    def _core(self, rho, sigma, features, zeta, raw=False):
        """Unconstrained correlation forward: (rho, sigma, features) -> 1 + F_c.

        ``zeta`` (spin polarization) is threaded through as a closed-over scalar
        rather than via the constraint signature, because the c-constraints
        operate on (rho, sigma, F) only, matching the model-level
        ``_batched_network_apply_polarized`` base_fn that also captured zeta.

        With ``parent`` set the gated output enters in the pre-image of the
        bounded map at the parent's value (see ``AlecGGA_XNet._core``), the
        parent's ``F_c`` taken relative to the model's polarized PW92 baseline
        at the row's ``zeta``; ``raw=True`` returns ``F_parent + gated``.
        ``raw`` has no effect on an unanchored network."""
        rho = jnp.maximum(rho, self.lower_rho_cutoff)
        sigma = jnp.maximum(sigma, 0.0)

        rs = (3 / (4 * jnp.pi * rho)) ** (1 / 3)
        k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)

        rs = jnp.atleast_1d(rs).flatten()
        s = jnp.atleast_1d(s).flatten()

        # The raw iso-orbital indicator of the row's total density, from its
        # smoothed, capped column (``_raw_indicator``). None on the GGA rungs.
        alpha_raw = None
        if self.meta_gga:
            alpha_raw = _raw_indicator(
                jnp.atleast_1d(features).flatten()[self.metagga_alpha_index])

        # Log-transform BOTH rs and s for the MLP input when
        # descriptor_log_transform=True. DELIBERATE DEVIATION from DFS Eq. 7:
        # the C-net density feature is r_s (not DFS's x0 = n^{1/3}), and it is
        # passed through the s-style {1 - exp(-x²)}·log(x + 1) transform (the
        # reduced-gradient form, DFS Eq. 9) rather than the plain log that DFS
        # Eq. 7 applies to the density variable. Documented, not changed (a
        # plain-log density form would invalidate existing checkpoints). zeta
        # (x1) and extras are not transformed. The tanh(s)² UEG gate below is
        # ALWAYS computed from raw s. The "dfs" coordinates below are the DFS
        # form itself, a separate checkpoint family.
        if self.descriptor_log_transform:
            rs_mlp = (1.0 - jnp.exp(-rs * rs)) * jnp.log(rs + 1.0)
            s_mlp = (1.0 - jnp.exp(-s * s)) * jnp.log(s + 1.0)
        else:
            rs_mlp = rs
            s_mlp = s

        if self.descriptor_coordinates == "dfs":
            # DFS coordinates, dpyscfl net.py get_descriptors, the branch
            # without spin scaling (the correlation network's), in the order
            # C_L consumes them (get_scf lines 746 and 751):
            #   x0  = ln(rho^(1/3) + 1e-5)           line 190 (l_1, line 100;
            #                                        loge, line 39); eq. 7
            #   x1  = ln(0.5 [(1+zeta)^(4/3) + (1-zeta)^(4/3)])
            #                                        line 191; eq. 4
            #   x_s = (1 - exp(-s^2)) ln(s + 1)      lines 200, 204; eq. 9,
            #         s of the TOTAL density: the zeta rescaling of s on line
            #         202 is xcdiff's, not dpyscfl's, and is not applied
            #   x_alpha = ln((alpha + 1)/2)          lines 214, 220; eq. 10,
            #         on the meta-GGA rung, in place of the raw indicator
            #         column among the extras
            # (PRB 104 L161109 equation numbers.) The polarized row is
            # required (x1), enforced at construction.
            zeta_c = jnp.clip(zeta, -1.0, 1.0)
            spinscale = 0.5 * ((1.0 + zeta_c) ** (4 / 3) + (1.0 - zeta_c) ** (4 / 3))
            x0 = jnp.atleast_1d(jnp.log(rho ** (1 / 3) + _DFS_LOG_EPS)).flatten()
            x1 = jnp.atleast_1d(jnp.log(spinscale)).flatten()
            x_s = _dfs_log_transform(s)
            if self.n_extra_features > 0:
                extras = jnp.atleast_1d(features).flatten()
                if self.meta_gga:
                    extras = extras.at[self.metagga_alpha_index].set(
                        _dfs_indicator_coordinate(alpha_raw))
                netinp = jnp.concatenate([x0, x1, x_s, extras])
            else:
                netinp = jnp.concatenate([x0, x1, x_s])
        elif self.use_spin_polarization:
            # zeta = (rho_a - rho_b)/rho_tot feeds the bounded Dick feature
            # x1 = 1/2[(1+zeta)^{4/3}+(1-zeta)^{4/3}] (in [1, 2^{1/3}] for
            # zeta in [-1,1]; x1=1 at zeta=0, recovering the unpolarized input
            # so an RKS (zeta=0) call sees [rs, s, 1, extras]).
            zeta_c = jnp.clip(zeta, -1.0, 1.0)
            x1 = jnp.atleast_1d(
                0.5 * ((1.0 + zeta_c) ** (4 / 3) + (1.0 - zeta_c) ** (4 / 3))
            ).flatten()
            if self.n_extra_features > 0:
                extras = jnp.atleast_1d(features).flatten()
                netinp = jnp.concatenate([rs_mlp, s_mlp, x1, extras])
            else:
                netinp = jnp.concatenate([rs_mlp, s_mlp, x1])
        elif self.n_extra_features > 0:
            extras = jnp.atleast_1d(features).flatten()
            netinp = jnp.concatenate([rs_mlp, s_mlp, extras])
        else:
            netinp = jnp.concatenate([rs_mlp, s_mlp])

        if self.meta_gga:
            # meta-GGA UEG-recovery prefactor x2 + tanh^2(x3): this reuses the X-net
            # (DFS XC_L exchange) gate form (raw x2) -- a deliberate X/C unification.
            # NOTE it DEVIATES from DFS's correlation C_L, which applies tanh to x2
            # (tanh(x2) + tanh^2(x3); vendored dpyscf/net.py:820). Either form is
            # bounded by the correlation LOB below. alpha is the descriptor column
            # at metagga_alpha_index (same as the X-net). As in the X-net, the MLP
            # receives RAW clamped alpha, not DFS's log-transformed x3 =
            # ln((alpha+1)/2) network input (Eq. 10/12); x3 enters only through
            # this gate. Documented, not changed.
            alpha = jnp.atleast_1d(features).flatten()[self.metagga_alpha_index]
            x2 = (1.0 - jnp.exp(-s * s)) * jnp.log(s + 1.0)
            x3 = jnp.log((alpha + 1.0) / 2.0)
            tanhterm = jnp.atleast_1d(x2).flatten() + jnp.tanh(x3) ** 2
        else:
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
        if self.parent is not None:
            f_parent = parent_fc(self.parent, rho, sigma, zeta, alpha_raw)
            if raw:
                return (f_parent + gated).squeeze()
            z_parent = lob_preimage(f_parent, self.lobf.limit)
            return 1 + self.lobf(z_parent + gated).squeeze()
        if self.lobf is not None:
            lobterm = self.lobf(gated)
            return 1 + lobterm.squeeze()
        return 1 + gated.squeeze()

    def eval_core(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """UNCONSTRAINED 1 + F_c for a packed input row. Exposed for
        constraint-violation introspection. An anchored network reports
        ``F_parent + gated``, the value before the bounded map."""
        if self.use_spin_polarization:
            return self._core(inputs[0], inputs[1], inputs[3:], inputs[2],
                              raw=True)
        return self._core(inputs[0], inputs[1], inputs[2:], 0.0, raw=True)

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

    Xnet lob_lim resolved via arch.resolved_xnet_lob_lim (None when
    LiebOxfordBound constraint is active).  Cnet lob_lim resolved via
    arch.resolved_cnet_lob_lim (default 2.0, a non-negativity squash on F_c
    per Dick & Fernández-Serra 2021 eq. (13), not a Lieb-Oxford bound).

    Physical constraints are materialized from the arch and handed to the
    networks, which enforce them INTRINSICALLY in their forward pass.  The same
    constrained functional is therefore used everywhere the network is called,
    pretraining, training, and evaluation, rather than being applied only by
    the composed model at train/eval time.

    ``arch.parent_anchor`` anchors both networks to the architecture's parent
    (``parents.parent_for_arch``: PBE on the GGA rungs, SCAN on the meta-GGA
    rungs) and forces ``zero_init_final_layer`` so the pair returns the parent
    at initialization; it requires ``use_polarized_correlation`` (a zeta-blind
    correlation network is refused, ``ValueError`` naming the architecture),
    and a meta-GGA architecture is refused with ``NotImplementedError`` until
    the SCAN commit lands. ``arch.descriptor_coordinates`` selects the MLP
    coordinates ("dfs" requires the polarized correlation network). MLP input
    widths: exchange ``1 + n_extra`` in both coordinate sets; correlation
    ``2 + n_extra`` (legacy, zeta-blind), ``3 + n_extra`` (legacy polarized
    and dfs).
    """
    _descs = arch.materialize_descriptors()
    n_extra_features = sum(d.n_features for d in _descs)
    coordinates = getattr(arch, "descriptor_coordinates", "legacy")
    parent = None
    zero_init_final_layer = arch.zero_init_final_layer
    if getattr(arch, "parent_anchor", False):
        if not arch.use_polarized_correlation:
            raise ValueError(
                f"create_network_pair: architecture {arch.name!r} has "
                "parent_anchor=True with use_polarized_correlation=False. An "
                "anchored correlation network must be polarization-aware: the "
                "parent's correlation is divided by the model's zeta-dependent "
                "PW92 baseline, which the pretraining data's open-shell Fc "
                "targets are also formed against; a zeta-blind network would "
                "multiply the unpolarized baseline instead (measured 14.9 mHa "
                "on the N atom's correlation term). Build the architecture "
                "with use_polarized_correlation=True (every v6 configuration "
                "does)."
            )
        if ArchitectureConfig.is_meta_gga(arch):
            raise NotImplementedError(
                f"create_network_pair: architecture {arch.name!r} is on the "
                "meta-GGA rung and parent_anchor=True needs the SCAN parent, "
                "which lands in the SCAN commit that follows the PBE anchor "
                "(SPEC_parent_anchor.md Section 3.7); parents.scan_fx / "
                "scan_fc are not implemented yet."
            )
        parent = parent_for_arch(arch)
        # SPEC_parent_anchor.md Section 3.3: the final layer is zero whatever
        # the registry entry says, so gated = 0 and F = F_parent at
        # initialization on both the plain and the attention paths.
        zero_init_final_layer = True
    if coordinates == "dfs" and not arch.use_polarized_correlation:
        raise ValueError(
            f"create_network_pair: architecture {arch.name!r} has "
            "descriptor_coordinates='dfs' with use_polarized_correlation="
            "False; the DFS correlation network reads x1 = ln(spinscale), so "
            "the polarized correlation network is required."
        )
    # DFS-faithful meta-GGA: the arch flags meta_gga -> the nets use the
    # (x2 + tanh^2(x3)) UEG gate + (exchange) the 1.174 Lieb-Oxford ceiling, reading
    # alpha from the MetaGGAAlphaDescriptor's column in the concatenated features.
    meta_gga = bool(getattr(arch, "meta_gga", False))
    metagga_alpha_index = -1
    if meta_gga:
        _off = 0
        for d in _descs:
            if type(d).__name__ == "MetaGGAAlphaDescriptor":
                metagga_alpha_index = _off
                break
            _off += d.n_features
    # meta_gga hardcodes the DFS 1.174 exchange ceiling through the core _AlecLOB.
    # LATENT RISK: this bypasses the resolved_xnet_lob_lim -> None suppression
    # signal that disables the core _AlecLOB when a `lieb_oxford` x-constraint is
    # active. No meta_gga arch currently carries a lieb_oxford constraint, so the
    # two bounds never coexist today; if such an arch is ever added, that
    # constraint and this 1.174 core bound would both apply (double-bound) -- gate
    # on arch.resolved_xnet_lob_lim (or drop the constraint) before enabling it.
    xnet = AlecGGA_XNet(
        n_extra_features=n_extra_features, depth=arch.depth, nodes=arch.nodes,
        use_self_attention=arch.attention, seed=seed,
        lob_lim=(1.174 if meta_gga else arch.resolved_xnet_lob_lim),
        lower_rho_cutoff=lower_rho_cutoff,
        num_heads=arch.num_heads,
        constraints=arch.materialize_x_constraints(),
        descriptor_log_transform=arch.descriptor_log_transform,
        meta_gga=meta_gga, metagga_alpha_index=metagga_alpha_index,
        zero_init_final_layer=zero_init_final_layer,
        parent=parent, descriptor_coordinates=coordinates,
    )
    cnet = AlecGGA_CNet(
        n_extra_features=n_extra_features, depth=arch.depth, nodes=arch.nodes,
        use_self_attention=arch.attention, seed=seed + 1,
        lob_lim=arch.resolved_cnet_lob_lim,
        lower_rho_cutoff=lower_rho_cutoff,
        num_heads=arch.num_heads,
        # zeta-aware correlation network when the arch opts in.
        use_spin_polarization=arch.use_polarized_correlation,
        constraints=arch.materialize_c_constraints(),
        descriptor_log_transform=arch.descriptor_log_transform,
        meta_gga=meta_gga, metagga_alpha_index=metagga_alpha_index,
        zero_init_final_layer=zero_init_final_layer,
        parent=parent, descriptor_coordinates=coordinates,
    )
    return xnet, cnet
