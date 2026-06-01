import pytest

import xcquinox.alec.external_refs as er


def test_cache_filename_includes_basis_and_df_tag():
    n1 = er._intermediate_cache_name("h2o", grid_level=1, basis="def2-svp",
                                     density_fit=False, kind="scf")
    n2 = er._intermediate_cache_name("h2o", grid_level=1, basis="def2-tzvp",
                                     density_fit=False, kind="scf")
    n3 = er._intermediate_cache_name("h2o", grid_level=1, basis="def2-svp",
                                     density_fit=True, kind="scf")
    assert n1 != n2 and n1 != n3          # basis + df distinguish cache files
    assert "def2-svp" in n1
    assert n1.endswith("_scf.npz")


def test_density_fit_off_cache_name_has_no_df_tag():
    n = er._intermediate_cache_name("h2o", grid_level=1, basis="def2-svp",
                                    density_fit=False, kind="ccsd")
    assert "_df_" not in n and n.endswith("_ccsd.npz")


def test_df_hf_matches_non_df_within_df_error():
    from pyscf import gto, scf
    mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468",
                basis="def2-svp", unit="angstrom", verbose=0)
    e_plain = scf.RHF(mol).kernel()
    mf_df = er._build_hf_meanfield(mol, False, density_fit=True,
                                   basis="def2-svp", auxbasis=None)
    e_df = mf_df.kernel()
    assert abs(e_df - e_plain) < 2e-3, (e_plain, e_df)


def test_build_hf_meanfield_default_is_plain():
    from pyscf import gto
    mol = gto.M(atom="He 0 0 0", basis="def2-svp", verbose=0)
    mf = er._build_hf_meanfield(mol, False)        # back-compat: no DF kwargs
    assert mf.__class__.__name__ in ("RHF", "SymAdaptedRHF")


@pytest.mark.slow
def test_df_ccsd_matches_non_df_within_df_error():
    """DF-HF→DF-CCSD total energy tracks the plain CCSD reference within DF
    error on a small closed-shell molecule."""
    from pyscf import gto, scf, cc
    mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468",
                basis="def2-svp", unit="angstrom", verbose=0)
    mf_plain = scf.RHF(mol).run()
    e_plain = mf_plain.e_tot + cc.CCSD(mf_plain).run().e_corr

    mf_df = er._build_hf_meanfield(mol, False, density_fit=True,
                                   basis="def2-svp", auxbasis=None).run()
    e_df = mf_df.e_tot + cc.CCSD(mf_df).run().e_corr
    assert abs(e_df - e_plain) < 5e-3, (e_plain, e_df)
