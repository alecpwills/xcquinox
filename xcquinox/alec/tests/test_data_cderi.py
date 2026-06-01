import jax.numpy as jnp

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec import df_jk


def _h2o_spec():
    return MoleculeSpec.from_dict(
        name="H2O",
        atom="O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468",
        basis="def2-svp", charge=0, spin=0,
        atom_composition={"O": 1, "H": 2}, grid_level=1,
        external_data_path=None,
    )


def test_cderi_requested_is_present_and_reconstructs_j():
    md = precompute_fixed_density_data(_h2o_spec(), required_keys=("cderi",))
    assert md.get("cderi") is not None
    assert md.get("eri") is None                       # not double-built
    cderi = jnp.asarray(md["cderi"])
    nao = cderi.shape[1]
    D = jnp.eye(nao)                                    # symmetric probe DM
    j_df = df_jk.compute_j_df(D, cderi)
    assert j_df.shape == (nao, nao)
    assert bool(jnp.all(jnp.isfinite(j_df)))


def test_cderi_not_built_when_not_requested():
    md = precompute_fixed_density_data(_h2o_spec(), required_keys=())
    assert md.get("cderi") is None
