import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from xcquinox.alec import df_jk


def _h2o():
    return gto.M(atom="O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468",
                 basis="def2-svp", unit="angstrom", verbose=0)


def test_build_cderi_shape_and_dtype():
    mol = _h2o()
    cderi = df_jk.build_cderi(mol)                      # (naux, nao, nao)
    nao = mol.nao_nr()
    assert cderi.ndim == 3 and cderi.shape[1:] == (nao, nao)
    assert cderi.shape[0] > nao                          # naux > nao for jkfit


def test_df_j_matches_full_eri_within_df_error():
    mol = _h2o()
    nao = mol.nao_nr()
    rng = np.random.default_rng(0)
    Dh = rng.standard_normal((nao, nao))
    D = jnp.asarray(Dh + Dh.T)                            # symmetric DM-like
    eri = jnp.asarray(mol.intor("int2e", aosym="s1"))
    j_full = jnp.einsum("ijkl,kl->ij", eri, D)
    cderi = df_jk.build_cderi(mol)
    j_df = df_jk.compute_j_df(D, cderi)
    rel = float(jnp.linalg.norm(j_df - j_full) / jnp.linalg.norm(j_full))
    assert rel < 5e-3, rel                               # DF error on J is ~1e-3


def test_default_auxbasis_maps_known_bases():
    assert df_jk.default_auxbasis("def2-svp") == "def2-svp-jkfit"
    assert df_jk.default_auxbasis("def2-tzvp") == "def2-tzvp-jkfit"
    assert df_jk.default_auxbasis("DEF2-SVP") == "def2-svp-jkfit"  # case-insensitive
    assert df_jk.default_auxbasis("sto-3g") is None                # non-def2 -> auto
    assert df_jk.default_auxbasis(None) is None


def test_default_auxbasis_diffuse_def2_uses_universal_jkfit():
    """def2-tzvpd (and other def2 bases without a dedicated -jkfit) resolve to
    the cited Weigend universal Coulomb-fitting set, NOT pyscf auto-select.
    This is the GAP-1 fix: the configured fit is reproducible across stages."""
    assert df_jk.default_auxbasis("def2-tzvpd") == "def2-universal-jkfit"
    assert df_jk.default_auxbasis("def2-tzvppd") == "def2-universal-jkfit"
    assert df_jk.default_auxbasis("def2-qzvp") == "def2-universal-jkfit"
    # Any other def2-* orbital basis falls back to the universal set too.
    assert df_jk.default_auxbasis("def2-qzvpp") == "def2-universal-jkfit"
    # Non-def2 unknown bases still defer to pyscf auto-select.
    assert df_jk.default_auxbasis("cc-pvdz") is None


def test_build_cderi_outofcore_matches_incore():
    """build_cderi must assemble the same tensor whether pyscf keeps _cderi in
    RAM or spills it to a temp HDF5 file. Forcing max_memory tiny triggers the
    out-of-core path (which the old np.asarray(_cderi) approach crashed on)."""
    mol = _h2o()
    incore = np.asarray(df_jk.build_cderi(mol))
    outcore = np.asarray(df_jk.build_cderi(mol, max_memory=1e-3))  # force spill
    assert incore.shape == outcore.shape
    assert np.allclose(incore, outcore, atol=1e-10)


def test_compute_j_df_is_differentiable_in_D():
    mol = _h2o()
    cderi = df_jk.build_cderi(mol)
    nao = mol.nao_nr()
    D = jnp.eye(nao)

    def scalar(Dm):
        return jnp.sum(df_jk.compute_j_df(Dm, cderi) ** 2)

    g = jax.grad(scalar)(D)
    assert g.shape == (nao, nao)
    assert bool(jnp.all(jnp.isfinite(g)))
