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


def test_df_scf_energy_is_differentiable_in_params():
    md_df = precompute_fixed_density_data(_spec(), required_keys=("cderi",))
    cfg_df = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                          max_cycles=2, density_fit=True)

    def loss(m):
        return run_scf(cfg_df, m, md_df).total_energy

    grads = eqx.filter_grad(loss)(_model())
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
