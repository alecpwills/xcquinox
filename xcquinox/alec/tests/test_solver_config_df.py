from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode


def test_solver_config_df_defaults_off_and_serialized():
    sc = SolverConfig()
    assert sc.density_fit is False
    assert sc.auxbasis is None
    d = sc.describe()                       # JSON-able dict
    assert d["density_fit"] is False
    assert d["auxbasis"] is None


def test_solver_config_df_on_is_hashable():
    sc = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                      max_cycles=3, density_fit=True, auxbasis="def2-tzvp-jkfit")
    assert sc.density_fit is True
    hash(sc)                                # frozen dataclass must stay hashable
    assert sc.describe()["auxbasis"] == "def2-tzvp-jkfit"
