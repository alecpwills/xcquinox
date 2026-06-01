"""The held-out eval must build its pool in the run's configured basis,
not a hardcoded def2-svp (otherwise a basis bump silently evaluates in the
wrong basis -> invalid train/eval comparison)."""
import xcquinox.alec.cluster._eval_one_spec as eos


class _FakeInputs:
    def __init__(self, basis, grid_level):
        self.basis = basis
        self.grid_level = grid_level


class _FakeCfg:
    def __init__(self, basis, grid_level):
        self.inputs = _FakeInputs(basis, grid_level)


def test_held_out_basis_grid_reads_config():
    basis, grid = eos._held_out_basis_grid(_FakeCfg(basis="def2-tzvp", grid_level=2))
    assert basis == "def2-tzvp"
    assert grid == 2


def test_held_out_basis_grid_falls_back_when_missing():
    class _Empty:
        inputs = None
    basis, grid = eos._held_out_basis_grid(_Empty())
    assert basis == "def2-svp"
    assert grid == 1
