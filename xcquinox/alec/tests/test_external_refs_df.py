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


def test_precompute_all_forwards_density_fit(tmp_path, monkeypatch):
    """precompute_all must thread density_fit/auxbasis into the SCF and CCSD
    cache calls, else the harness's inputs.density_fit never reaches ref gen."""
    seen = {}

    spec = er.SpeciesEntry(name="H2", charge=0, spin=0, source="dfs_ae")

    def fake_scf(s, atoms, *, cache_dir, basis, grid_level,
                 density_fit=False, auxbasis=None):
        seen["scf"] = (density_fit, auxbasis)
        return {"dm": None}

    def fake_ccsd(s, atoms, *, scf_payload, cache_dir, basis, grid_level,
                  density_fit=False, auxbasis=None):
        seen["ccsd"] = (density_fit, auxbasis)
        return {}

    monkeypatch.setattr(er, "run_scf_with_cache", fake_scf)
    monkeypatch.setattr(er, "run_ccsd_with_cache", fake_ccsd)
    monkeypatch.setattr(er, "run_oep_cascade",
                        lambda *a, **k: None)
    monkeypatch.setattr(er, "resolve_geometry", lambda s: object())
    monkeypatch.setattr(er, "_npz_is_complete", lambda p: False)
    monkeypatch.setattr(er, "_validate_overrides", lambda species: None)

    er.precompute_all([spec], cache_dir=tmp_path, basis="def2-tzvp",
                      grid_level=1, run_preflight=False,
                      density_fit=True, auxbasis="def2-tzvp-jkfit")
    assert seen["scf"] == (True, "def2-tzvp-jkfit")
    assert seen["ccsd"] == (True, "def2-tzvp-jkfit")


@pytest.mark.slow
def test_df_ccsd_matches_non_df_within_df_error():
    """DF-HF -> DF-CCSD total energy tracks the plain CCSD reference within DF
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
