import equinox as eqx
import jax
import jax.numpy as jnp

from xcquinox.alec.config import MoleculeSpec, get_architecture
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode, run_scf


def _spec():
    return MoleculeSpec.from_dict(
        name="H2O", atom="O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468",
        basis="def2-svp", charge=0, spin=0,
        atom_composition={"O": 1, "H": 2}, grid_level=1, external_data_path=None)


def _model():
    return AlecGGAModel.from_arch(get_architecture("deep"), seed=0)


def test_df_scf_energy_matches_full_eri_within_df_error():
    model = _model()
    md_full = precompute_fixed_density_data(_spec(), required_keys=("eri",))
    md_df = precompute_fixed_density_data(_spec(), required_keys=("cderi",))
    cfg_full = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                            max_cycles=3)
    cfg_df = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                          max_cycles=3, density_fit=True)
    e_full = float(run_scf(cfg_full, model, md_full).total_energy)
    e_df = float(run_scf(cfg_df, model, md_df).total_energy)
    assert abs(e_df - e_full) < 5e-3, (e_full, e_df)    # DF error ~mHa


def _o_triplet_spec():
    """Open-shell O atom (triplet) — exercises the UKS DF Coulomb path
    J = compute_j_df(Dα) + compute_j_df(Dβ) (solver_manual.py UKS branch)."""
    return MoleculeSpec.from_dict(
        name="O", atom="O 0 0 0", basis="def2-svp", charge=0, spin=2,
        atom_composition={"O": 1}, grid_level=1, external_data_path=None)


def test_df_scf_energy_matches_full_eri_uks():
    """UKS (open-shell) DF path: DF energy matches full-ERI within DF error.
    Regression for the audit finding that the open-shell DF Coulomb build —
    the exact path used by the radicals/atoms in the BH76+W4-11 pools — had
    zero test coverage."""
    model = _model()
    md_full = precompute_fixed_density_data(_o_triplet_spec(), required_keys=("eri",))
    md_df = precompute_fixed_density_data(_o_triplet_spec(), required_keys=("cderi",))
    cfg_full = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                            max_cycles=3, conv_tol=1e-5)
    cfg_df = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                          max_cycles=3, conv_tol=1e-5, density_fit=True)
    e_full = float(run_scf(cfg_full, model, md_full).total_energy)
    e_df = float(run_scf(cfg_df, model, md_df).total_energy)
    assert abs(e_df - e_full) < 5e-3, (e_full, e_df)


def test_df_scf_energy_is_differentiable_in_params_uks():
    """UKS DF SCF energy is differentiable w.r.t. model params (finite grads)."""
    md_df = precompute_fixed_density_data(_o_triplet_spec(), required_keys=("cderi",))
    cfg_df = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                          max_cycles=2, density_fit=True)

    def loss(m):
        return run_scf(cfg_df, m, md_df).total_energy

    grads = eqx.filter_grad(loss)(_model())
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


def test_df_scf_energy_is_differentiable_in_params():
    md_df = precompute_fixed_density_data(_spec(), required_keys=("cderi",))
    cfg_df = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                          max_cycles=2, density_fit=True)

    def loss(m):
        return run_scf(cfg_df, m, md_df).total_energy

    grads = eqx.filter_grad(loss)(_model())
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
