from xcquinox.alec.eval_holdout import descriptors_and_required_keys
from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode


class _Arch:
    def materialize_descriptors(self):
        return ()


class _Spec:
    def __init__(self, sc):
        self.arch = _Arch()
        self.solver_config = sc


def test_full_solver_requests_eri_by_default():
    sc = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                      max_cycles=3)
    _, keys, _ = descriptors_and_required_keys(_Spec(sc))
    assert "eri" in keys and "cderi" not in keys


def test_full_solver_with_df_requests_cderi():
    sc = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                      max_cycles=3, density_fit=True)
    _, keys, _ = descriptors_and_required_keys(_Spec(sc))
    assert "cderi" in keys and "eri" not in keys


def test_oneshot_requests_neither():
    sc = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.ONESHOT)
    _, keys, _ = descriptors_and_required_keys(_Spec(sc))
    assert "eri" not in keys and "cderi" not in keys
