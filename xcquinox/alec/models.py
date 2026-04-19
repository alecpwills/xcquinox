"""xcquinox.alec.models — AlecGGAModel composite model.

Implements THE SPEC §5.3: composite model wrapping xnet + cnet with
constraint composition, descriptor materialization, and scalar eval path.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from xcquinox.utils import lda_x, pw92c_unpolarized_scalar
from xcquinox.alec.config import ArchitectureConfig
from xcquinox.alec.networks import create_network_pair
from xcquinox.alec.constraints import Constraint, _compose_constraints
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


class AlecGGAModel(eqx.Module):
    """Composite model wrapping exchange + correlation networks with
    constraint composition and descriptor materialization."""
    xnet: eqx.Module
    cnet: eqx.Module
    descriptors: tuple[Descriptor, ...] = eqx.field(default=(), static=True)
    x_constraints: tuple[Constraint, ...] = eqx.field(default=(), static=True)
    c_constraints: tuple[Constraint, ...] = eqx.field(default=(), static=True)
    rho_cutoff: float = eqx.field(default=1e-18, static=True)

    @classmethod
    def from_arch(cls, arch: ArchitectureConfig, *, xnet=None, cnet=None,
                  seed: int = 42, rho_cutoff: float = 1e-18,
                  lower_rho_cutoff: float = 1e-12):
        """Materialize an AlecGGAModel from an ArchitectureConfig.

        If xnet/cnet are None, fresh networks are built via create_network_pair.
        If provided, they are used directly and lower_rho_cutoff is ignored.
        """
        if xnet is None or cnet is None:
            xnet, cnet = create_network_pair(
                arch, seed=seed, lower_rho_cutoff=lower_rho_cutoff
            )
        return cls(
            xnet=xnet, cnet=cnet,
            descriptors=arch.materialize_descriptors(),
            x_constraints=arch.materialize_x_constraints(),
            c_constraints=arch.materialize_c_constraints(),
            rho_cutoff=rho_cutoff,
        )

    def eval_Fx(self, rho, sigma, features):
        """Constrained exchange enhancement. Shapes: (N,) -> (N,)."""
        base_fn = lambda r, s, f: _batched_network_apply(self.xnet, r, s, f)
        constrained = _compose_constraints(base_fn, self.x_constraints)
        return constrained(rho, sigma, features)

    def eval_Fc(self, rho, sigma, features):
        """Constrained correlation enhancement. Shapes: (N,) -> (N,)."""
        base_fn = lambda r, s, f: _batched_network_apply(self.cnet, r, s, f)
        constrained = _compose_constraints(base_fn, self.c_constraints)
        return constrained(rho, sigma, features)

    def eval_exc(self, rho, sigma, features):
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
        """
        tail_mask = rho > _NN_TAIL_THRESHOLD
        safe_rho = jnp.where(tail_mask, rho, jnp.ones_like(rho))
        safe_sigma = jnp.where(tail_mask, sigma, jnp.ones_like(sigma))
        Fx = self.eval_Fx(safe_rho, safe_sigma, features)
        Fc = self.eval_Fc(safe_rho, safe_sigma, features)
        # Mask Fx, Fc at tail to LDA/PW92 limit (F -> 1 as rho -> 0).
        Fx = jnp.where(tail_mask, Fx, jnp.ones_like(Fx))
        Fc = jnp.where(tail_mask, Fc, jnp.ones_like(Fc))
        rho_safe = jnp.maximum(rho, self.rho_cutoff)
        ex_lda = lda_x(rho_safe)
        ec_pw92 = pw92c_unpolarized_scalar(rho_safe)
        return rho_safe * (ex_lda * Fx + ec_pw92 * Fc)

    def eval_exc_scalar(self, rho_scalar, sigma_scalar, features_scalar):
        """Scalar energy-density at a single grid point. Used by compute_vxc_nn.

        E-H1: applies same constraint chain as eval_Fx/eval_Fc.
        M-C13-1: network inputs use raw rho (no pre-clip via rho_cutoff).

        Tail-point sanitization matches ``eval_exc``: at rho < threshold,
        network inputs are replaced with safe defaults and outputs masked
        to Fx=Fc=1 to keep gradients finite.
        """
        rho_safe = jnp.maximum(rho_scalar, self.rho_cutoff)
        ex_lda = lda_x(rho_safe)
        ec_pw92 = pw92c_unpolarized_scalar(rho_safe)

        tail_mask = rho_scalar > _NN_TAIL_THRESHOLD
        safe_rho_scalar = jnp.where(tail_mask, rho_scalar, jnp.ones_like(rho_scalar))
        safe_sigma_scalar = jnp.where(tail_mask, sigma_scalar, jnp.ones_like(sigma_scalar))

        def x_base_scalar(r, s, f):
            row = jnp.concatenate([jnp.atleast_1d(r), jnp.atleast_1d(s), f])
            return self.xnet(row).squeeze()

        def c_base_scalar(r, s, f):
            row = jnp.concatenate([jnp.atleast_1d(r), jnp.atleast_1d(s), f])
            return self.cnet(row).squeeze()

        x_chain = _compose_constraints(x_base_scalar, self.x_constraints)
        c_chain = _compose_constraints(c_base_scalar, self.c_constraints)
        Fx_scalar = x_chain(safe_rho_scalar, safe_sigma_scalar, features_scalar)
        Fc_scalar = c_chain(safe_rho_scalar, safe_sigma_scalar, features_scalar)
        Fx_scalar = jnp.where(tail_mask, Fx_scalar, jnp.ones_like(Fx_scalar))
        Fc_scalar = jnp.where(tail_mask, Fc_scalar, jnp.ones_like(Fc_scalar))
        return rho_safe * (ex_lda * Fx_scalar + ec_pw92 * Fc_scalar)

    def constraint_report(self, rho, sigma, features) -> dict:
        """Returns {side: {name: {max, mean, l2}}} nested dict."""
        report = {"x": {}, "c": {}}
        for side, constraints, net in [
            ("x", self.x_constraints, self.xnet),
            ("c", self.c_constraints, self.cnet),
        ]:
            base_fn = lambda r, s, f, _net=net: _batched_network_apply(_net, r, s, f)
            for c in constraints:
                v = c.violation(base_fn, rho, sigma, features)
                report[side][c.registry_name] = {
                    "max": float(jnp.max(v)),
                    "mean": float(jnp.mean(v)),
                    "l2": float(jnp.sqrt(jnp.mean(v ** 2))),
                }
        return report
