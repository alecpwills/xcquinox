import pytest
import jax.numpy as jnp
from pyscf import gto

from xcquinox.alec import df_jk


@pytest.mark.slow
def test_def2tzvp_cderi_builds_for_large_molecule():
    # alcl3-sized system in def2-tzvp: the full s1 ERI is multi-GB; cderi is far
    # smaller. Confirms the memory property that motivates DF.
    mol = gto.M(atom="Al 0 0 0; Cl 2.06 0 0; Cl -1.03 1.78 0; Cl -1.03 -1.78 0",
                basis="def2-tzvp", unit="angstrom", verbose=0)
    cderi = df_jk.build_cderi(mol)
    nao = mol.nao_nr()
    assert cderi.shape[1:] == (nao, nao)
    assert cderi.size < (nao ** 4) // 4                 # DF << full ERI
    D = jnp.eye(nao)
    assert bool(jnp.all(jnp.isfinite(df_jk.compute_j_df(D, cderi))))


def test_inputs_density_fit_defaults_off():
    from xcquinox.alec.cluster.grid_config import _build_inputs
    inp = _build_inputs({
        "external_refs_dir": "x", "subset_ledger_path": "y",
        "basis": "def2-svp", "grid_level": 1, "output_root": "z"})
    assert inp.density_fit is False and inp.auxbasis is None


def test_inputs_density_fit_reads_config():
    from xcquinox.alec.cluster.grid_config import _build_inputs
    inp = _build_inputs({
        "external_refs_dir": "x", "subset_ledger_path": "y",
        "basis": "def2-tzvp", "grid_level": 2, "output_root": "z",
        "density_fit": True, "auxbasis": "def2-tzvp-jkfit"})
    assert inp.density_fit is True and inp.auxbasis == "def2-tzvp-jkfit"


def test_solver_config_from_named_threads_density_fit():
    from xcquinox.alec.cluster.spec_builder import _solver_config_from_named
    from xcquinox.alec.cluster.grid_config import SolverNamed

    # Build the REAL dataclass rather than a hand-written stand-in. A stub has
    # to be updated by hand every time SolverNamed gains a field, and when it is
    # not, the test fails on an AttributeError that says nothing about
    # density-fit threading -- which is what happened when the 2026-06-24 mixer
    # knobs (mixer_name / mixer_kwargs) landed. Constructing the dataclass keeps
    # this test pinned to the thing it claims to exercise.
    named = SolverNamed(mode="FULL", max_cycles=3)

    sc = _solver_config_from_named(named, density_fit=True,
                                   auxbasis="def2-svp-jkfit")
    assert sc.density_fit is True and sc.auxbasis == "def2-svp-jkfit"
