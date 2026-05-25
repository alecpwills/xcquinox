"""xcquinox.alec.models — AlecGGAModel composite model.

Implements THE SPEC §5.3: composite model wrapping xnet + cnet with
constraint composition, descriptor materialization, and scalar eval path.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from xcquinox.utils import (
    lda_x, pw92c_unpolarized_scalar, pw92c_polarized_scalar,
)
from xcquinox.alec.config import ArchitectureConfig
from xcquinox.alec.networks import create_network_pair
from xcquinox.alec.descriptors import Descriptor


# Threshold below which NN inputs are sanitized to avoid sqrt(sigma)-derivative
# divergence in the networks' reduced-gradient transform. At these tail points
# Fx, Fc are masked to the LDA/PW92 limit (=1) — the physical limit as rho -> 0
# anyway — and the rho_safe * eps_xc prefactor vanishes, so energy contribution
# is negligible while gradients remain finite.
_NN_TAIL_THRESHOLD = 1e-10


def _pack_row(rho_scalar, sigma_scalar, features_row):
    """Assemble single-row feature vector [rho, sigma, *extras] for the network."""
    return jnp.concatenate([jnp.atleast_1d(rho_scalar),
                            jnp.atleast_1d(sigma_scalar),
                            features_row])


def _batched_network_apply(net, rho, sigma, features):
    """Apply single-point network to batched (N,) inputs via vmap.
    Shapes: rho, sigma (N,); features (N, F); returns (N,)."""
    def row_fn(r, s, f):
        return net(_pack_row(r, s, f)).squeeze()
    return jax.vmap(row_fn, in_axes=(0, 0, 0))(rho, sigma, features)


def _pack_row_polarized(rho_scalar, sigma_scalar, zeta_scalar, features_row):
    """Single-row vector [rho, sigma, zeta, *extras] for the spin-polarized
    correlation net (P2-03). zeta sits at index 2; descriptor extras follow at
    index 3 — matching ``AlecGGA_CNet.__call__`` when use_spin_polarization."""
    return jnp.concatenate([jnp.atleast_1d(rho_scalar),
                            jnp.atleast_1d(sigma_scalar),
                            jnp.atleast_1d(zeta_scalar),
                            features_row])


def _batched_network_apply_polarized(net, rho, sigma, zeta, features):
    """Batched apply with a per-point spin-polarization zeta (P2-03).
    Shapes: rho, sigma, zeta (N,); features (N, F); returns (N,)."""
    def row_fn(r, s, z, f):
        return net(_pack_row_polarized(r, s, z, f)).squeeze()
    return jax.vmap(row_fn, in_axes=(0, 0, 0, 0))(rho, sigma, zeta, features)


class AlecGGAModel(eqx.Module):
    """Composite model wrapping exchange + correlation networks with
    constraint composition and descriptor materialization."""
    xnet: eqx.Module
    cnet: eqx.Module
    descriptors: tuple[Descriptor, ...] = eqx.field(default=(), static=True)
    rho_cutoff: float = eqx.field(default=1e-18, static=True)

    # Physical constraints are now enforced INTRINSICALLY by the networks
    # (xnet/cnet carry and apply them in their forward), so the same constrained
    # functional is used in pretraining, training, and eval. These properties
    # expose the network-held constraints under the historical model-level API.
    @property
    def x_constraints(self) -> tuple:
        return getattr(self.xnet, "constraints", ())

    @property
    def c_constraints(self) -> tuple:
        return getattr(self.cnet, "constraints", ())

    @classmethod
    def from_arch(cls, arch: ArchitectureConfig, *, xnet=None, cnet=None,
                  seed: int = 42, rho_cutoff: float = 1e-18,
                  lower_rho_cutoff: float = 1e-12):
        """Materialize an AlecGGAModel from an ArchitectureConfig.

        If xnet/cnet are None, fresh networks are built via create_network_pair
        (which bakes the arch's constraints into the networks). If provided, they
        are used directly — they MUST already carry the arch's constraints (a
        skeleton built via create_network_pair does), and lower_rho_cutoff is
        ignored.
        """
        if xnet is None or cnet is None:
            xnet, cnet = create_network_pair(
                arch, seed=seed, lower_rho_cutoff=lower_rho_cutoff
            )
        return cls(
            xnet=xnet, cnet=cnet,
            descriptors=arch.materialize_descriptors(),
            rho_cutoff=rho_cutoff,
        )

    def eval_Fx(self, rho, sigma, features):
        """Constrained exchange enhancement. Shapes: (N,) -> (N,).

        The constraint chain is applied INSIDE the network forward, so this is
        simply the batched network apply (byte-identical to the former
        model-level ``_compose_constraints`` over the bare network)."""
        return _batched_network_apply(self.xnet, rho, sigma, features)

    def eval_Fc(self, rho, sigma, features, zeta=0.0):
        """Constrained correlation enhancement. Shapes: (N,) -> (N,).

        P2-03: when the cnet is spin-polarization-aware
        (``cnet.use_spin_polarization``), ``zeta`` (broadcast to rho's shape) is
        packed into the cnet row at index 2 and the descriptor extras shift to
        index 3 — so the cnet sees the bounded x1 feature. Otherwise ``zeta`` is
        ignored and the unpolarized packing [rho, sigma, *extras] is used.
        Constraints are enforced inside the network forward.
        """
        if getattr(self.cnet, "use_spin_polarization", False):
            zeta_arr = jnp.broadcast_to(jnp.asarray(zeta, dtype=rho.dtype), rho.shape)
            return _batched_network_apply_polarized(
                self.cnet, rho, sigma, zeta_arr, features)
        return _batched_network_apply(self.cnet, rho, sigma, features)

    def _ec_baseline(self, rho_safe, zeta):
        """Per-electron UEG correlation baseline eps_c. Spin-polarized PW92
        (zeta-dependent) when the cnet is polarization-aware, else unpolarized
        PW92 (P2-03). At zeta=0 the polarized baseline reduces EXACTLY to the
        unpolarized one (verified vs libxc)."""
        if getattr(self.cnet, "use_spin_polarization", False):
            half = 0.5 * (1.0 + jnp.asarray(zeta))
            return pw92c_polarized_scalar(rho_safe * half, rho_safe * (1.0 - half))
        return pw92c_unpolarized_scalar(rho_safe)

    def _exc_pieces(self, rho, sigma, features, zeta=0.0):
        """Shared core for eval_exc / eval_ex / eval_ec (batched).

        Returns ``(ex_density, ec_density)`` where
        ``ex_density = rho_safe * ex_lda * Fx`` (exchange-only energy density)
        and ``ec_density = rho_safe * ec_baseline * Fc`` (correlation-only). The
        tail masking and rho_safe prefactor are IDENTICAL to the original
        combined ``eval_exc``, so ``ex_density + ec_density == eval_exc``
        holds pointwise (see test_eval_exc_equals_eval_ex_plus_eval_ec_batched).

        Splitting exchange and correlation is required by SOLV-01: in UKS the
        exchange piece obeys the exact spin-scaling relation
        E_x[n_up,n_dn] = 1/2 (E_x[2 n_up] + E_x[2 n_dn]) (Oliver & Perdew,
        PRA 20, 397 (1979)), but correlation does NOT — correlation is
        spin-interpolated (von Barth & Hedin, J. Phys. C 5, 1629 (1972);
        PW92, PRB 45, 13244 (1992)).

        P2-03: ``zeta`` is the spin polarization (rho_a - rho_b)/rho_tot. When
        the cnet is spin-polarization-aware it is fed to BOTH the cnet (as the
        x1 input feature) and the correlation baseline (via the zeta-dependent
        ``_ec_baseline`` -> spin-polarized PW92). Exchange ignores zeta. For an
        unpolarized cnet (default) zeta is ignored and the unpolarized PW92
        baseline is used — byte-identical to the pre-P2-03 path; at zeta=0 the
        polarized baseline also reduces exactly to the unpolarized one.
        """
        tail_mask = rho > _NN_TAIL_THRESHOLD
        safe_rho = jnp.where(tail_mask, rho, jnp.ones_like(rho))
        safe_sigma = jnp.where(tail_mask, sigma, jnp.ones_like(sigma))
        Fx = self.eval_Fx(safe_rho, safe_sigma, features)
        Fc = self.eval_Fc(safe_rho, safe_sigma, features, zeta=zeta)
        # Mask Fx, Fc at tail to LDA/PW92 limit (F -> 1 as rho -> 0).
        Fx = jnp.where(tail_mask, Fx, jnp.ones_like(Fx))
        Fc = jnp.where(tail_mask, Fc, jnp.ones_like(Fc))
        rho_safe = jnp.maximum(rho, self.rho_cutoff)
        ex_lda = lda_x(rho_safe)
        ec_base = self._ec_baseline(rho_safe, zeta)
        ex_density = rho_safe * ex_lda * Fx
        ec_density = rho_safe * ec_base * Fc
        return ex_density, ec_density

    def eval_exc(self, rho, sigma, features, zeta=0.0):
        """Returns rho * epsilon_xc (energy density), shape (N,).
        Fx/Fc use raw rho; lda_x/pw92c and prefactor use rho_safe.

        At tail points (rho < ``_NN_TAIL_THRESHOLD``) the network inputs are
        sanitized to (rho=1, sigma=1) to avoid the sqrt(sigma)-derivative
        divergence in the networks' reduced-gradient transform; the network
        outputs at these points are then masked to the LDA/PW92 limit
        (Fx=Fc=1), and since ``rho_safe`` -> 0 at the tail, the energy
        contribution is negligible. This keeps both forward and backward
        (jax.grad) values finite for open-shell atoms where one spin channel
        has rho=sigma=0 identically (e.g., beta channel of H spin=1).

        P2-03: ``zeta`` (spin polarization) is forwarded to the correlation
        piece; ignored by a non-polarized cnet, defaults to 0 (RKS).
        """
        ex_density, ec_density = self._exc_pieces(rho, sigma, features, zeta=zeta)
        return ex_density + ec_density

    def eval_ex(self, rho, sigma, features):
        """Exchange-only energy density rho * eps_x (batched, shape (N,)).

        Exact split of ``eval_exc`` (eval_ex + eval_ec == eval_exc pointwise).
        Used by the UKS energy/potential to spin-scale the exchange piece
        (Oliver & Perdew, PRA 20, 397 (1979)). See ``_exc_pieces``.
        """
        ex_density, _ = self._exc_pieces(rho, sigma, features)
        return ex_density

    def eval_ec(self, rho, sigma, features, zeta=0.0):
        """Correlation-only energy density rho * eps_c (batched, shape (N,)).

        Exact split of ``eval_exc``. In UKS this is evaluated on the TOTAL
        density with the per-grid spin polarization ``zeta`` (P2-03): a
        polarization-aware cnet feeds zeta to both the x1 feature and the
        spin-polarized PW92 baseline (von Barth & Hedin 1972; PW92 1992). For a
        non-polarized cnet, zeta is ignored (zeta=0 unpolarized PW92). See
        ``_exc_pieces``.
        """
        _, ec_density = self._exc_pieces(rho, sigma, features, zeta=zeta)
        return ec_density

    def _exc_pieces_scalar(self, rho_scalar, sigma_scalar, features_scalar, zeta=0.0):
        """Shared scalar core for eval_exc_scalar / eval_ex_scalar /
        eval_ec_scalar. Returns ``(ex_density, ec_density)`` with IDENTICAL
        tail masking, rho_cutoff prefactor, and constraint composition as the
        original combined ``eval_exc_scalar`` — so the scalar split is exact
        (eval_ex_scalar + eval_ec_scalar == eval_exc_scalar pointwise).

        See ``_exc_pieces`` for the SOLV-01 physics rationale (Oliver &
        Perdew PRA 20, 397 (1979) for exchange spin-scaling; von Barth &
        Hedin 1972 / PW92 1992 for correlation spin interpolation).

        P2-03: ``zeta`` is the spin polarization at this point. A
        polarization-aware cnet receives it (packed at index 2 of the cnet row)
        and the correlation baseline becomes the spin-polarized PW92; exchange
        ignores zeta. Non-polarized cnet (default): zeta ignored, unpolarized
        PW92 — byte-identical to the pre-P2-03 scalar path.
        """
        rho_safe = jnp.maximum(rho_scalar, self.rho_cutoff)
        ex_lda = lda_x(rho_safe)
        ec_base = self._ec_baseline(rho_safe, zeta)

        tail_mask = rho_scalar > _NN_TAIL_THRESHOLD
        safe_rho_scalar = jnp.where(tail_mask, rho_scalar, jnp.ones_like(rho_scalar))
        safe_sigma_scalar = jnp.where(tail_mask, sigma_scalar, jnp.ones_like(sigma_scalar))

        polarized = getattr(self.cnet, "use_spin_polarization", False)

        # The networks enforce their constraints internally, so evaluating the
        # packed row directly yields the constrained Fx/Fc (no model-level
        # composition needed — byte-identical to the former chained path).
        x_row = jnp.concatenate(
            [jnp.atleast_1d(safe_rho_scalar), jnp.atleast_1d(safe_sigma_scalar),
             features_scalar])
        Fx_scalar = self.xnet(x_row).squeeze()
        if polarized:
            # P2-03: zeta at index 2, descriptor extras follow at index 3.
            c_row = jnp.concatenate(
                [jnp.atleast_1d(safe_rho_scalar), jnp.atleast_1d(safe_sigma_scalar),
                 jnp.atleast_1d(zeta), features_scalar])
        else:
            c_row = jnp.concatenate(
                [jnp.atleast_1d(safe_rho_scalar), jnp.atleast_1d(safe_sigma_scalar),
                 features_scalar])
        Fc_scalar = self.cnet(c_row).squeeze()
        Fx_scalar = jnp.where(tail_mask, Fx_scalar, jnp.ones_like(Fx_scalar))
        Fc_scalar = jnp.where(tail_mask, Fc_scalar, jnp.ones_like(Fc_scalar))
        ex_density = rho_safe * ex_lda * Fx_scalar
        ec_density = rho_safe * ec_base * Fc_scalar
        return ex_density, ec_density

    def eval_exc_scalar(self, rho_scalar, sigma_scalar, features_scalar, zeta=0.0):
        """Scalar energy-density at a single grid point. Used by compute_vxc_nn.

        E-H1: applies same constraint chain as eval_Fx/eval_Fc.
        M-C13-1: network inputs use raw rho (no pre-clip via rho_cutoff).

        Tail-point sanitization matches ``eval_exc``: at rho < threshold,
        network inputs are replaced with safe defaults and outputs masked
        to Fx=Fc=1 to keep gradients finite.

        P2-03: ``zeta`` forwards to the correlation piece (ignored by a
        non-polarized cnet; default 0).
        """
        ex_density, ec_density = self._exc_pieces_scalar(
            rho_scalar, sigma_scalar, features_scalar, zeta=zeta)
        return ex_density + ec_density

    def eval_ex_scalar(self, rho_scalar, sigma_scalar, features_scalar):
        """Exchange-only scalar energy density. JVP'd by ``compute_vxc_nn``
        with ``part="x"`` to build the spin-scaled exchange V_xc (Oliver &
        Perdew, PRA 20, 397 (1979)). Exact split of ``eval_exc_scalar``."""
        ex_density, _ = self._exc_pieces_scalar(
            rho_scalar, sigma_scalar, features_scalar)
        return ex_density

    def eval_ec_scalar(self, rho_scalar, sigma_scalar, features_scalar, zeta=0.0):
        """Correlation-only scalar energy density. JVP'd by ``compute_vxc_nn``
        with ``part="c"`` to build the correlation V_xc. P2-03: with a
        polarization-aware cnet, ``zeta`` feeds both the x1 feature and the
        spin-polarized PW92 baseline (von Barth & Hedin 1972, PW92 1992); the
        per-spin V_c in the UKS drivers comes from differentiating this w.r.t.
        the spin DM at the appropriate zeta. Non-polarized cnet: zeta ignored
        (zeta=0). Exact split of ``eval_exc_scalar``."""
        _, ec_density = self._exc_pieces_scalar(
            rho_scalar, sigma_scalar, features_scalar, zeta=zeta)
        return ec_density

    def constraint_report(self, rho, sigma, features) -> dict:
        """Returns {side: {name: {max, mean, l2}}} nested dict.

        Each constraint's violation is measured against the network's
        UNCONSTRAINED core (``_core``) — the networks now apply constraints in
        their forward, so the raw enhancement must come from the core, not the
        (already-constrained) ``__call__``."""
        def _x_raw(r, s, f):
            return jax.vmap(lambda rr, ss, ff: self.xnet._core(rr, ss, ff))(r, s, f)

        def _c_raw(r, s, f):
            # zeta=0: constraint_report has always used the unpolarized packing.
            return jax.vmap(
                lambda rr, ss, ff: self.cnet._core(rr, ss, ff, 0.0))(r, s, f)

        report = {"x": {}, "c": {}}
        for side, constraints, raw_fn in [
            ("x", self.x_constraints, _x_raw),
            ("c", self.c_constraints, _c_raw),
        ]:
            for c in constraints:
                v = c.violation(raw_fn, rho, sigma, features)
                report[side][c.registry_name] = {
                    "max": float(jnp.max(v)),
                    "mean": float(jnp.mean(v)),
                    "l2": float(jnp.sqrt(jnp.mean(v ** 2))),
                }
        return report
