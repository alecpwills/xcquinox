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


def test_precompute_forwards_auxbasis_to_build_cderi(monkeypatch):
    """GAP-1 regression: the configured auxbasis MUST reach build_cderi rather
    than being silently dropped (which made def2-tzvpd auto-select)."""
    from xcquinox.alec.data import clear_precompute_cache
    clear_precompute_cache()
    seen = {}
    real = df_jk.build_cderi

    def spy(mol, auxbasis=None):
        seen["auxbasis"] = auxbasis
        return real(mol, auxbasis=auxbasis)

    monkeypatch.setattr(df_jk, "build_cderi", spy)
    precompute_fixed_density_data(_h2o_spec(), required_keys=("cderi",),
                                  auxbasis="def2-svp-jkfit")
    assert seen["auxbasis"] == "def2-svp-jkfit"
    clear_precompute_cache()


def test_cderi_cache_distinguishes_auxbasis(monkeypatch):
    """Two fitting bases must NOT collide in the precompute cache: auxbasis is
    part of the cache key, so build_cderi is invoked for BOTH (a collision would
    serve the second call from the first's cached entry)."""
    from xcquinox.alec.data import clear_precompute_cache
    clear_precompute_cache()
    seen = []
    real = df_jk.build_cderi

    def spy(mol, auxbasis=None):
        seen.append(auxbasis)
        return real(mol, auxbasis=auxbasis)

    monkeypatch.setattr(df_jk, "build_cderi", spy)
    precompute_fixed_density_data(_h2o_spec(), required_keys=("cderi",),
                                  auxbasis="def2-svp-jkfit")
    precompute_fixed_density_data(_h2o_spec(), required_keys=("cderi",),
                                  auxbasis="def2-universal-jkfit")
    assert seen == ["def2-svp-jkfit", "def2-universal-jkfit"]
    clear_precompute_cache()


def test_auxbasis_does_not_perturb_non_df_precompute():
    """Default-off invariance: passing auxbasis with no cderi requested does not
    change any baseline (PBE) quantity. (auxbasis is never read off the DF path;
    the only residual difference between two independent SCF runs is BLAS
    last-bit noise, hence the tight tolerance rather than bit-equality.)"""
    from xcquinox.alec.data import clear_precompute_cache
    clear_precompute_cache()
    md0 = precompute_fixed_density_data(_h2o_spec(), required_keys=())
    clear_precompute_cache()
    md1 = precompute_fixed_density_data(_h2o_spec(), required_keys=(),
                                        auxbasis="def2-universal-jkfit")
    assert abs(float(md0["E_pbe"]) - float(md1["E_pbe"])) < 1e-9
    assert bool(jnp.allclose(jnp.asarray(md0["dm_pbe"]),
                             jnp.asarray(md1["dm_pbe"]), atol=1e-9, rtol=0))
    clear_precompute_cache()
