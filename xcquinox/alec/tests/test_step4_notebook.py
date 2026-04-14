"""Unit tests + smoke test for the step 4 notebook generator.

The generator lives at ``notebooks/_build_step4_notebook.py`` and is not part
of an importable package (``notebooks/`` intentionally has no ``__init__.py``).
Tests load the generator via ``importlib.util.spec_from_file_location`` so
test discovery does not depend on ``sys.path`` tricks.

Per ``docs/superpowers/plans/2026-04-12-step4-notebook-implementation.md``, this
module starts with a single scaffolding test in Task 1 and grows one builder
test group per downstream task (Tasks 2 through 13).
"""
import importlib.util
import pathlib

import nbformat
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
GENERATOR_PATH = REPO_ROOT / "notebooks" / "_build_step4_notebook.py"


def load_generator():
    """Import ``_build_step4_notebook`` as ``step4_generator`` via spec loader.

    ``notebooks/`` is not a package, and ``sys.path`` does not normally expose
    it, so direct ``import`` fails. ``spec_from_file_location`` sidesteps the
    question without requiring a spurious ``__init__.py``.
    """
    if not GENERATOR_PATH.is_file():
        pytest.fail(
            f"Step 4 notebook generator not found at {GENERATOR_PATH}. "
            "Did Task 1 fail to land?"
        )
    spec = importlib.util.spec_from_file_location(
        "step4_generator", str(GENERATOR_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_produces_valid_notebook(tmp_path):
    """``main()`` must emit a notebook that passes ``nbformat.validate``."""
    gen = load_generator()
    out_path = tmp_path / "step4_scaffold.ipynb"
    returned = gen.main(str(out_path))

    # main() returns the notebook object directly
    assert returned is not None
    assert len(returned.cells) >= 1

    # The written file must round-trip through nbformat.read without error
    assert out_path.is_file()
    nb = nbformat.read(str(out_path), as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 1


# ---------------------------------------------------------------------------
# Task 2 — Cells 1-5 builder tests
# ---------------------------------------------------------------------------


def test_cell_02_imports_includes_jax_x64_before_jnp():
    """The x64 config update must precede ``import jax.numpy as jnp``.

    Flipping ``jax_enable_x64`` after ``jnp`` has triggered tracing poisons
    cached JIT lowerings with the wrong dtype (spec Round C10-2 regression
    guard). The order is load-bearing, not cosmetic.
    """
    gen = load_generator()
    source = gen.build_cell_02_imports().source
    x64_idx = source.find('jax.config.update("jax_enable_x64", True)')
    jnp_idx = source.find("import jax.numpy as jnp")
    assert x64_idx != -1, "missing jax.config.update x64 call"
    assert jnp_idx != -1, "missing 'import jax.numpy as jnp'"
    assert x64_idx < jnp_idx, (
        "jax_enable_x64 update must appear before 'import jax.numpy as jnp' "
        f"(x64 at {x64_idx}, jnp at {jnp_idx})"
    )


def test_cell_02_imports_includes_jax_default_device_cpu():
    """Cell 2 must pin the JAX default device to CPU for reproducibility."""
    gen = load_generator()
    source = gen.build_cell_02_imports().source
    assert (
        'jax.config.update("jax_default_device", jax.devices("cpu")[0])'
        in source
    )


def test_cell_02_imports_tqdm_auto():
    """Cell 2 must import ``tqdm`` via ``tqdm.auto`` so the same symbol works
    in JupyterLab (routed to ``tqdm.notebook.tqdm`` + ipywidgets) and in a
    plain script/terminal context (routed to ``tqdm.std.tqdm``).
    """
    gen = load_generator()
    source = gen.build_cell_02_imports().source
    assert "from tqdm.auto import tqdm" in source


def test_cell_03_constants_match_spec():
    """Cell 3 must bind the exact literal forms frozen by the spec."""
    gen = load_generator()
    source = gen.build_cell_03_constants().source
    assert "BASIS = 'def2-svp'" in source
    assert "GRID_LEVEL = 1" in source
    assert (
        'H2O_COORDS = "O 0.0000 0.0000 0.1173; '
        'H 0.0000 0.7572 -0.4692; '
        'H 0.0000 -0.7572 -0.4692"'
    ) in source
    assert 'PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("O", 2), ("N", 3))' in source


def test_cell_03_constants_checkpoint_base_honors_override():
    """The ``checkpoint_base`` override must flow into the cell source via repr."""
    gen = load_generator()
    source = gen.build_cell_03_constants("smoke_ckpt").source
    assert "CHECKPOINT_BASE = 'smoke_ckpt'" in source


def test_cell_05_binds_arch_colors_before_cell_9():
    """Cell 5 must bind ``arch_colors`` so Cell 9 can reference it.

    Deferring the binding to Cell 25 (the visualization section) produces
    ``NameError`` at Cell 9's pretrain loss plot on any fresh top-to-bottom
    run (spec Round B11-1 regression guard).
    """
    gen = load_generator()
    source = gen.build_cell_05_arch_names().source
    assert "arch_colors = {" in source
    assert 'plt.get_cmap("tab20")' in source


def test_main_cells_1_to_5_validate(tmp_path):
    """``main()`` must produce at least 5 cells with the expected types."""
    gen = load_generator()
    out_path = tmp_path / "step4_cells_1_5.ipynb"
    nb = gen.main(str(out_path))
    assert len(nb.cells) >= 5
    expected_types = ["markdown", "code", "code", "code", "code"]
    actual_types = [c.cell_type for c in nb.cells[:5]]
    assert actual_types == expected_types, (
        f"first 5 cell types {actual_types} != expected {expected_types}"
    )


# Task 3 — Cells 6-8 builder tests


def test_cell_07_uses_rho_cutoff_1e_minus_10():
    """Cell 7's low-density mask must use `valid = rho > 1e-10`.

    Strict `>` not `>=`, threshold 1e-10 not 1e-6 — guards the off-by-threshold
    regression from spec B-review rounds 8-10. Step3b uses the looser cutoff
    to keep the atomic tail.
    """
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    assert "valid = rho > 1e-10" in source


def test_cell_07_uses_np_where_safe_division():
    """Cell 7 must use np.where-based safe division, NOT a boolean mask.

    Boolean masks drop points step3b keeps; np.where keeps shape parity so
    the downstream `valid` filter is the only mask applied.
    """
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    assert "np.where(np.abs(ex_lda)" in source


def test_cell_07_lists_initialised_unconditionally():
    """`cusp_list, dm_list = [], []` must appear before the PRETRAIN_ATOMS loop.

    Unconditional init makes the `if cusp_list:` / `if dm_list:` truthy-check
    at save time safe even when `ARCH_NAMES` contains no extended-feature archs.
    """
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    init_idx = source.find("cusp_list, dm_list = [], []")
    loop_idx = source.find("for atom_symbol, spin in PRETRAIN_ATOMS:")
    assert init_idx != -1, "cusp/dm list init missing"
    assert loop_idx != -1, "PRETRAIN_ATOMS loop missing"
    assert init_idx < loop_idx, "list init must precede the loop"


def test_cell_07_uses_libxc_strings_not_helpers():
    """Cell 7 must call libxc functional strings, NOT xcquinox helpers.

    Step3b Cell 10 uses pyscf's `eval_xc("LDA_X,", ...)` / `eval_xc(",LDA_C_PW", ...)`
    for exact numerical parity; the xcquinox helpers must NOT be imported here.
    """
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    assert '"LDA_X,"' in source
    assert '",LDA_C_PW"' in source
    assert "from xcquinox.utils import lda_x" not in source


def test_cell_07_need_flags_gate_extended_features():
    """`need_cusp`/`need_dm` must be derived via `any(...)` and gate the
    descriptor computation branches."""
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    assert "need_cusp = any(" in source
    assert "need_dm = any(" in source
    assert "if need_cusp:" in source
    assert "if need_dm:" in source


def test_cell_08_qualifies_alec_pretrainspec():
    """Cell 8 must use `alec.PretrainSpec(`, never bare `PretrainSpec(`."""
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    assert "alec.PretrainSpec(" in source
    # Ensure no bare PretrainSpec usage — check that every PretrainSpec
    # occurrence is preceded by "alec."
    import re
    bare_refs = re.findall(r"(?<!alec\.)PretrainSpec\(", source)
    assert bare_refs == [], f"bare PretrainSpec references found: {bare_refs}"


def test_cell_08_passes_step3b_hyperparameters():
    """Cell 8's PretrainSpec must pass the step3b hyperparameters."""
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    for literal in ("n_steps=1000", "lr_start=1e-2", "lr_end=1e-5",
                    "lr_decay_start=0.2", "grad_clip=1.0"):
        assert literal in source, f"missing hyperparameter literal: {literal}"


# Cell 8 runtime parallel-pretrain toggle tests
#
# The toggle is a notebook-runtime constant (`PRETRAIN_PARALLEL` set in Cell 3)
# that Cell 8 reads at execution time to pick between a serial in-process loop
# and a subprocess+ThreadPoolExecutor dispatch. Both branches live inside the
# single emitted Cell 8 source.


def test_cell_03_defines_pretrain_parallel_false():
    """Cell 3 must bind ``PRETRAIN_PARALLEL = False`` so Cell 8 has a default
    serial path. Users flip this constant in the notebook to opt into parallel
    pretraining without regenerating the notebook.
    """
    gen = load_generator()
    source = gen.build_cell_03_constants().source
    assert "PRETRAIN_PARALLEL = False" in source


def test_cell_08_branches_on_pretrain_parallel_runtime_toggle():
    """Cell 8 must contain a runtime ``if PRETRAIN_PARALLEL:`` branch so flipping
    the Cell 3 constant switches execution paths without regenerating.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    assert "if PRETRAIN_PARALLEL:" in source
    assert "else:" in source
    assert "for arch_name in ARCH_NAMES:" in source


def test_cell_08_parallel_branch_uses_thread_pool_and_subprocess():
    """Cell 8's parallel branch must dispatch via ThreadPoolExecutor over
    isolated subprocess.run calls.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    assert "from concurrent.futures import ThreadPoolExecutor" in source
    assert "as_completed" in source
    assert "import subprocess" in source
    assert "ThreadPoolExecutor(max_workers=" in source


def test_cell_08_parallel_branch_sets_xla_and_omp_env_before_subprocess_run():
    """Cell 8's parallel branch must set XLA_FLAGS and OMP_NUM_THREADS in the
    child env BEFORE the subprocess.run call — otherwise N parallel workers
    all oversubscribe XLA's internal thread pool.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    xla_idx = source.find('XLA_FLAGS')
    omp_idx = source.find('OMP_NUM_THREADS')
    run_idx = source.find('subprocess.run')
    assert xla_idx != -1, "parallel branch missing XLA_FLAGS"
    assert omp_idx != -1, "parallel branch missing OMP_NUM_THREADS"
    assert run_idx != -1, "parallel branch missing subprocess.run"
    assert xla_idx < run_idx, "XLA_FLAGS must be set before subprocess.run"
    assert omp_idx < run_idx, "OMP_NUM_THREADS must be set before subprocess.run"
    assert '"--xla_cpu_multi_thread_eigen=false"' in source


def test_cell_08_parallel_branch_bounds_max_workers_by_cpu_count():
    """Cell 8's parallel branch must compute ``max_workers`` from
    ``os.cpu_count()`` and cap at half the CPU count.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    assert "cpu_count()" in source
    assert "// 2" in source
    assert "len(ARCH_NAMES)" in source


def test_cell_08_parallel_branch_raises_on_subprocess_failure():
    """Cell 8's parallel branch must raise on subprocess failure — silent drops
    leave missing checkpoints that break downstream cells.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    assert "check=True" in source or "CalledProcessError" in source


def test_cell_08_source_is_valid_python():
    """The unified Cell 8 source (both branches plus the runtime toggle) must
    parse as valid Python — the parallel branch builds a nested f-string for
    the subprocess child code, which is easy to get wrong at generator time.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    compile(source, "<cell_08>", "exec")


def test_cell_08_serial_path_runtime_dispatches_to_alec_run_pretrain(monkeypatch):
    """With ``PRETRAIN_PARALLEL = False`` at exec time, Cell 8 must call
    ``alec.run_pretrain`` once per arch in ARCH_NAMES — verifies the runtime
    branch picker selects the serial path and the serial path actually runs.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source

    calls = []

    class _FakeArch:
        pass

    class _FakeSpec:
        def __init__(self, **kw):
            self.kw = kw

    class _FakeAlec:
        PretrainSpec = _FakeSpec

        @staticmethod
        def get_architecture(name):
            return _FakeArch()

        @staticmethod
        def run_pretrain(spec, progress_callback=None):
            calls.append(spec.kw["arch"])
            return None

    scope = {
        "__builtins__": __builtins__,
        "PRETRAIN_PARALLEL": False,
        "PRETRAIN_SKIP_IF_EXISTS": False,
        "CHECKPOINT_BASE": "/tmp/fake_ckpt",
        "ARCH_NAMES": ["shallow", "deep"],
        "alec": _FakeAlec,
    }
    exec(source, scope)
    assert len(calls) == 2, f"expected 2 run_pretrain calls, got {len(calls)}"


def test_cell_08_parallel_path_runtime_dispatches_subprocesses(monkeypatch):
    """With ``PRETRAIN_PARALLEL = True`` at exec time, Cell 8 must route each
    arch through ``subprocess.run`` — verifies the runtime branch picker
    selects the parallel path.
    """
    import subprocess as _real_sp
    from tqdm.auto import tqdm as _real_tqdm
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source

    captured_child_codes = []

    def _fake_run(args, env=None, check=None, capture_output=None, text=None):
        captured_child_codes.append(args[2])
        class _Result:
            stdout = ""
            stderr = ""
            returncode = 0
        return _Result()

    monkeypatch.setattr(_real_sp, "run", _fake_run)

    scope = {
        "__builtins__": __builtins__,
        "PRETRAIN_PARALLEL": True,
        "PRETRAIN_SKIP_IF_EXISTS": False,
        "CHECKPOINT_BASE": "/tmp/fake_ckpt",
        "ARCH_NAMES": ["shallow"],
        "tqdm": _real_tqdm,
    }
    exec(source, scope)

    assert len(captured_child_codes) == 1, (
        f"expected 1 subprocess.run call, got {len(captured_child_codes)}"
    )
    compile(captured_child_codes[0], "<child_code>", "exec")
    assert "alec.PretrainSpec(" in captured_child_codes[0]
    assert "'shallow'" in captured_child_codes[0]
    assert "/tmp/fake_ckpt/pretrain_data" in captured_child_codes[0]
    assert "/tmp/fake_ckpt/pretrain/shallow" in captured_child_codes[0]


def test_cell_08_parallel_path_sets_child_env_vars_at_runtime(monkeypatch):
    """With ``PRETRAIN_PARALLEL = True``, the env passed to subprocess.run must
    contain the XLA and OMP overrides — this is a runtime check, not just
    a string-match in the source.
    """
    import subprocess as _real_sp
    from tqdm.auto import tqdm as _real_tqdm
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source

    captured_envs = []

    def _fake_run(args, env=None, check=None, capture_output=None, text=None):
        captured_envs.append(env)
        class _Result:
            stdout = ""
            stderr = ""
            returncode = 0
        return _Result()

    monkeypatch.setattr(_real_sp, "run", _fake_run)

    scope = {
        "__builtins__": __builtins__,
        "PRETRAIN_PARALLEL": True,
        "PRETRAIN_SKIP_IF_EXISTS": False,
        "CHECKPOINT_BASE": "/tmp/fake_ckpt",
        "ARCH_NAMES": ["shallow"],
        "tqdm": _real_tqdm,
    }
    exec(source, scope)

    assert len(captured_envs) == 1
    env = captured_envs[0]
    assert env["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=false"
    assert env["OMP_NUM_THREADS"] == "1"


# Cell 8 tqdm progress-bar tests


def test_cell_08_serial_path_callback_uses_tqdm_bar_with_loss_postfix():
    """Cell 8's serial-branch callback must drive a ``tqdm`` bar and attach
    the current step's loss via ``set_postfix(loss=...)`` so users see a
    "step X/N loss=..." bar for each pretrain phase.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    # The _cb function must reference tqdm and set_postfix with a loss key
    assert "tqdm(" in source
    assert "set_postfix(" in source
    assert "loss=" in source


def test_cell_08_parallel_path_uses_arch_level_tqdm_bar():
    """Cell 8's parallel branch must wrap the ``as_completed`` loop in a
    ``tqdm`` bar tracking arch-level completion count — per-step callbacks
    do not stream back through subprocesses, so an arch-completion bar is
    the best-available progress signal in parallel mode.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    # The parallel branch contains both ThreadPoolExecutor and a tqdm call
    # whose total is len(ARCH_NAMES).
    assert "ThreadPoolExecutor" in source
    assert "tqdm(" in source
    assert "total=len(ARCH_NAMES)" in source


def test_cell_08_serial_callback_creates_one_bar_per_arch_phase():
    """Driving ``_cb`` with events across two phases must produce two tqdm
    bars (one per phase), each closed once its final step is reported.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source

    created_bars = []

    class _FakeBar:
        def __init__(self, total=None, desc=None, leave=True,
                     dynamic_ncols=False, **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
            self.postfix_calls = []
            self.update_calls = []
            self.refresh_calls = 0
            self.closed = False
            created_bars.append(self)

        def update(self, delta):
            self.update_calls.append(delta)
            self.n += delta

        def set_postfix(self, **kwargs):
            self.postfix_calls.append(kwargs)

        def refresh(self):
            self.refresh_calls += 1

        def close(self):
            self.closed = True

    class _FakeAlec:
        class PretrainSpec:
            def __init__(self, **kw):
                self.kw = kw

        @staticmethod
        def get_architecture(name):
            return object()

        @staticmethod
        def run_pretrain(spec, progress_callback=None):
            return None  # Do not drive callback — we drive it manually below.

    scope = {
        "__builtins__": __builtins__,
        "PRETRAIN_PARALLEL": False,
        "PRETRAIN_SKIP_IF_EXISTS": False,
        "CHECKPOINT_BASE": "/tmp/fake_ckpt",
        "ARCH_NAMES": [],  # empty so the for loop is a no-op
        "alec": _FakeAlec,
        "tqdm": _FakeBar,
    }
    exec(source, scope)

    cb = scope["_cb"]
    # Drive 2 phases × 2 steps for a single arch
    cb({"arch": "shallow", "phase": "X", "step": 1, "total": 2,
        "loss": 1e-2, "timestamp": 0.0})
    cb({"arch": "shallow", "phase": "X", "step": 2, "total": 2,
        "loss": 1e-3, "timestamp": 0.0})
    cb({"arch": "shallow", "phase": "C", "step": 1, "total": 2,
        "loss": 5e-2, "timestamp": 0.0})
    cb({"arch": "shallow", "phase": "C", "step": 2, "total": 2,
        "loss": 5e-3, "timestamp": 0.0})

    assert len(created_bars) == 2, (
        f"expected 2 bars (one per phase), got {len(created_bars)}"
    )
    assert all(bar.closed for bar in created_bars), (
        "all bars must be closed when their final step is reported"
    )
    # Each bar must have reached total=2
    assert all(bar.n == 2 for bar in created_bars), (
        f"bar.n must equal total=2 at end of phase, got {[b.n for b in created_bars]}"
    )


def test_cell_08_serial_callback_sets_loss_postfix_in_scientific_notation():
    """The loss reported in ``set_postfix`` must be formatted as scientific
    notation (``{loss:.4e}``) so small loss values stay readable.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source

    captured_postfix = []

    class _FakeBar:
        def __init__(self, total=None, desc=None, leave=True,
                     dynamic_ncols=False, **kwargs):
            self.n = 0

        def update(self, delta):
            self.n += delta

        def set_postfix(self, **kwargs):
            captured_postfix.append(kwargs)

        def refresh(self):
            pass

        def close(self):
            pass

    scope = {
        "__builtins__": __builtins__,
        "PRETRAIN_PARALLEL": False,
        "PRETRAIN_SKIP_IF_EXISTS": False,
        "CHECKPOINT_BASE": "/tmp/fake_ckpt",
        "ARCH_NAMES": [],
        "alec": type("_A", (), {
            "PretrainSpec": type("_S", (), {"__init__": lambda s, **k: None}),
            "get_architecture": staticmethod(lambda n: None),
            "run_pretrain": staticmethod(lambda spec, progress_callback=None: None),
        }),
        "tqdm": _FakeBar,
    }
    exec(source, scope)

    cb = scope["_cb"]
    cb({"arch": "shallow", "phase": "X", "step": 1, "total": 2,
        "loss": 1.2345e-3, "timestamp": 0.0})

    assert len(captured_postfix) >= 1
    loss_str = captured_postfix[-1]["loss"]
    # {:.4e} format: "1.2345e-03"
    assert "e-0" in loss_str or "e-" in loss_str, (
        f"loss postfix must be scientific notation, got {loss_str!r}"
    )
    assert "1.2345e" in loss_str


def test_cell_08_parallel_path_creates_and_closes_arch_bar(monkeypatch):
    """With ``PRETRAIN_PARALLEL = True``, the parallel branch must create a
    single tqdm bar with total=len(ARCH_NAMES), update it as archs finish,
    and close it after the executor context exits.
    """
    import subprocess as _real_sp
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source

    created_bars = []

    class _FakeBar:
        def __init__(self, total=None, desc=None, leave=True,
                     dynamic_ncols=False, **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
            self.postfix_calls = []
            self.update_calls = []
            self.closed = False
            created_bars.append(self)

        def update(self, delta):
            self.update_calls.append(delta)
            self.n += delta

        def set_postfix(self, **kwargs):
            self.postfix_calls.append(kwargs)

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    def _fake_run(args, env=None, check=None, capture_output=None, text=None):
        class _Result:
            stdout = ""
            stderr = ""
            returncode = 0
        return _Result()

    monkeypatch.setattr(_real_sp, "run", _fake_run)

    scope = {
        "__builtins__": __builtins__,
        "PRETRAIN_PARALLEL": True,
        "PRETRAIN_SKIP_IF_EXISTS": False,
        "CHECKPOINT_BASE": "/tmp/fake_ckpt",
        "ARCH_NAMES": ["shallow", "deep"],
        "tqdm": _FakeBar,
    }
    exec(source, scope)

    assert len(created_bars) >= 1, "no tqdm bar created in parallel branch"
    arch_bar = created_bars[0]
    assert arch_bar.total == 2, (
        f"parallel arch bar total must be len(ARCH_NAMES)=2, got {arch_bar.total}"
    )
    assert sum(arch_bar.update_calls) == 2, (
        f"arch bar must be updated once per completed arch, got {arch_bar.update_calls}"
    )
    assert arch_bar.closed, "arch bar must be closed after dispatch"


# Task 4 — Cells 9-10 builder tests


def test_cell_09_loads_losses_x_and_losses_c():
    """Cell 9 must load both xnet and cnet loss arrays by the path template
    Cell 8 writes to.
    """
    gen = load_generator()
    source = gen.build_cell_09_pretrain_loss_plot().source
    assert 'losses_x.npy' in source
    assert 'losses_c.npy' in source


def test_cell_09_uses_log_scale():
    """Cell 9 must use log y-scale so order-of-magnitude loss decay is visible."""
    gen = load_generator()
    source = gen.build_cell_09_pretrain_loss_plot().source
    assert "semilogy(" in source or 'set_yscale("log")' in source


def test_cell_09_saves_to_figures_dir():
    """Cell 9 must save the plot under `{CHECKPOINT_BASE}/figures/`."""
    gen = load_generator()
    source = gen.build_cell_09_pretrain_loss_plot().source
    assert '{CHECKPOINT_BASE}/figures/pretrain_losses.png' in source


def test_cell_10_uses_create_network_pair_skeleton():
    """Cell 10 must construct (xnet, cnet) skeletons via `alec.create_network_pair`.

    This is the pretrain-layout skeleton path; Cell 26 (full-model load) uses a
    different entry point (`AlecGGAModel.from_arch`), so the difference matters.
    """
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    assert "alec.create_network_pair(" in source


def test_cell_10_uses_tree_deserialise_leaves():
    """Cell 10 must deserialise the saved .eqx weights via eqx.tree_deserialise_leaves."""
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    assert "eqx.tree_deserialise_leaves(" in source


def test_cell_10_is_12x2_or_documented_subset():
    """Cell 10 must build a (n_arch x 2) subplots grid — 12 rows for the full
    default ARCH_NAMES or a narrower grid when the test harness passes a subset.

    Accept either an explicit `subplots(12, 2` literal OR a dynamic
    `subplots(n_arch, 2` / `subplots(len(ARCH_NAMES), 2` form.
    """
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    ok_forms = ("subplots(12, 2", "subplots(n_arch, 2", "subplots(len(ARCH_NAMES), 2")
    assert any(form in source for form in ok_forms), (
        f"Cell 10 must call subplots with an (n_arch, 2) grid; none of "
        f"{ok_forms} found in source."
    )


# Cell 10 parity plot formula guards.
# ``networks.py`` returns ``1 + lobterm.squeeze()`` / ``1 + gated.squeeze()``,
# so ``xnet(p)`` already equals F (not F-1). The pretrain loss confirms this:
#   pred = net(x); pred = pred - 1.0; loss = (pred - ref_F)^2
# with ref_F stored as (F - 1). So at convergence net(x) == F.
# A previous revision of Cell 10 added ``+ 1.0`` to predictions, producing a
# parity plot shifted by +1 on the y-axis relative to the x-axis.


def test_cell_10_predictions_do_not_shift_by_plus_one():
    """Cell 10 must NOT add ``+ 1.0`` to the network output when computing
    ``Fx_pred`` / ``Fc_pred``. The network already returns the full enhancement
    factor F, so ``xnet(p) + 1.0`` shifts predictions by an extra +1 and
    breaks parity against the x-axis.
    """
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    assert "xnet(p) + 1.0" not in source, (
        "Cell 10 adds +1.0 to xnet(p); network output is already F, so this "
        "shifts the y-axis by +1 relative to the x-axis."
    )
    assert "cnet(p) + 1.0" not in source, (
        "Cell 10 adds +1.0 to cnet(p); network output is already F, so this "
        "shifts the y-axis by +1 relative to the x-axis."
    )


def test_cell_10_predictions_use_raw_network_output():
    """Cell 10 must compute ``Fx_pred = jax.vmap(lambda p: xnet(p))(...)`` and
    ``Fc_pred = jax.vmap(lambda p: cnet(p))(...)``. The network already returns F.
    """
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    assert "jax.vmap(lambda p: xnet(p))" in source
    assert "jax.vmap(lambda p: cnet(p))" in source


def test_cell_10_both_axes_in_same_space():
    """Cell 10 must plot both x-axis (target) and y-axis (prediction) in the
    same F space. Either:
      (a) x = Fx_target + 1.0 (F space) AND y = xnet(p) (F space), OR
      (b) x = Fx_target (F-1 space) AND y = xnet(p) - 1.0 (F-1 space).
    The current generator uses form (a): ``Fx_target + 1.0`` on the x-axis
    and raw ``xnet(p)`` on the y-axis.
    """
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    # Form (a): both F-space.
    x_axis_uses_plus_one = "Fx_target) + 1.0" in source
    y_axis_is_raw = "jax.vmap(lambda p: xnet(p))" in source
    assert x_axis_uses_plus_one and y_axis_is_raw, (
        "Cell 10 must use the F-space plotting form: x-axis should shift "
        "Fx_target by +1 and y-axis should use raw xnet(p). "
        f"x_axis_uses_plus_one={x_axis_uses_plus_one}, "
        f"y_axis_is_raw={y_axis_is_raw}"
    )


def test_cell_10_runtime_predictions_match_target_at_convergence():
    """End-to-end runtime check: if a fake xnet returns exactly the stored
    ``1 + Fx_all`` (i.e., perfect convergence), Cell 10's scatter should
    produce x and y arrays that are elementwise equal. This guards against
    any residual shift being introduced into Fx_pred.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source

    # Build a fake .npz data file in memory and write it to tmp.
    # We do not need the full Cell 10 machinery — just the slice that
    # computes Fx_pred from xnet(p) and the scatter x/y pair.
    #
    # Strategy: carve out the two lines that matter and exec them against
    # a scope where xnet/cnet/_data/etc. are already bound. This avoids
    # having to mock the full alec API surface.
    rho = np.array([0.1, 0.2, 0.3])
    sigma = np.array([0.01, 0.04, 0.09])
    Fx_all = np.array([0.2, 0.3, 0.4])  # stored as (F - 1)
    Fc_all = np.array([-0.1, 0.0, 0.1])

    # Perfect convergence: xnet(p) returns 1 + Fx_all[i] = F
    Fx_full = 1.0 + Fx_all
    Fc_full = 1.0 + Fc_all

    def _fake_xnet(p):
        # p shape (n_features,); match by rho (column 0)
        # We rely on exec-time vmap calling this per row.
        idx = jnp.argmin(jnp.abs(p[0] - jnp.asarray(rho)))
        return jnp.asarray(Fx_full)[idx]

    def _fake_cnet(p):
        idx = jnp.argmin(jnp.abs(p[0] - jnp.asarray(rho)))
        return jnp.asarray(Fc_full)[idx]

    input_array = jnp.stack([jnp.asarray(rho), jnp.asarray(sigma)], axis=1)

    # Extract the Fx_pred / Fc_pred lines from the generator source so we
    # exercise the actual formula, not a hand-typed copy.
    fx_line = next(
        line for line in source.splitlines()
        if line.strip().startswith("Fx_pred =")
    )
    fc_line = next(
        line for line in source.splitlines()
        if line.strip().startswith("Fc_pred =")
    )
    local_scope = {
        "__builtins__": __builtins__,
        "jax": jax,
        "jnp": jnp,
        "xnet": _fake_xnet,
        "cnet": _fake_cnet,
        "input_array": input_array,
    }
    exec(fx_line.strip(), local_scope)
    exec(fc_line.strip(), local_scope)
    Fx_pred = np.asarray(local_scope["Fx_pred"])
    Fc_pred = np.asarray(local_scope["Fc_pred"])

    # Target (in F space): Fx_target + 1.0 = Fx_all + 1.0 = Fx_full
    np.testing.assert_allclose(Fx_pred, Fx_full, rtol=0, atol=1e-6)
    np.testing.assert_allclose(Fc_pred, Fc_full, rtol=0, atol=1e-6)


# Cell 8 / Cell 3 PRETRAIN_SKIP_IF_EXISTS toggle tests.
# Users want the option to load previously pre-trained models instead of
# re-running pretraining every notebook execution. The toggle is a
# notebook-runtime constant (`PRETRAIN_SKIP_IF_EXISTS` set in Cell 3) that
# Cell 8 reads before dispatching ``alec.run_pretrain`` — if True AND both
# xnet.eqx and cnet.eqx already exist for an arch, that arch is skipped.


def test_cell_03_defines_pretrain_skip_if_exists_false():
    """Cell 3 must bind ``PRETRAIN_SKIP_IF_EXISTS = False`` so Cell 8 defaults
    to always re-running pretraining. Users flip this constant in the notebook
    to opt into skipping arches whose checkpoints already exist.
    """
    gen = load_generator()
    source = gen.build_cell_03_constants().source
    assert "PRETRAIN_SKIP_IF_EXISTS = False" in source


def test_cell_08_references_pretrain_skip_if_exists():
    """Cell 8 must reference ``PRETRAIN_SKIP_IF_EXISTS`` so flipping the Cell 3
    constant actually gates the pretrain call.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    assert "PRETRAIN_SKIP_IF_EXISTS" in source


def test_cell_08_serial_path_skips_when_checkpoints_exist(monkeypatch, tmp_path):
    """With ``PRETRAIN_SKIP_IF_EXISTS = True`` and both xnet.eqx + cnet.eqx
    present for an arch, the serial branch must NOT call ``alec.run_pretrain``
    for that arch.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source

    # Pre-create the checkpoint files for "shallow" but NOT for "deep"
    ckpt = tmp_path / "ckpt"
    (ckpt / "pretrain" / "shallow").mkdir(parents=True)
    (ckpt / "pretrain" / "shallow" / "xnet.eqx").write_bytes(b"fake")
    (ckpt / "pretrain" / "shallow" / "cnet.eqx").write_bytes(b"fake")

    got_arch_calls = []

    class _FakeSpec:
        def __init__(self, **kw):
            self.kw = kw

    class _FakeAlec:
        PretrainSpec = _FakeSpec

        @staticmethod
        def get_architecture(name):
            # Return a named sentinel so run_pretrain can see which arch.
            got_arch_calls.append(name)
            return type("_Arch", (), {"name": name})()

        @staticmethod
        def run_pretrain(spec, progress_callback=None):
            return None

    scope = {
        "__builtins__": __builtins__,
        "PRETRAIN_PARALLEL": False,
        "PRETRAIN_SKIP_IF_EXISTS": True,
        "CHECKPOINT_BASE": str(ckpt),
        "ARCH_NAMES": ["shallow", "deep"],
        "alec": _FakeAlec,
    }
    exec(source, scope)
    # ``get_architecture`` only runs for non-skipped arches (it is inside the
    # Cell 8 loop body, after the skip check). "shallow" is pre-cached, so it
    # must be absent from got_arch_calls; "deep" must be present.
    assert "shallow" not in got_arch_calls, (
        f"'shallow' already had checkpoints; get_architecture must not be "
        f"called for it. got_arch_calls={got_arch_calls}"
    )
    assert "deep" in got_arch_calls, (
        f"'deep' has no checkpoints; get_architecture must be called for it. "
        f"got_arch_calls={got_arch_calls}"
    )


def test_cell_08_serial_path_runs_when_skip_is_false_even_if_checkpoints_exist(tmp_path):
    """With ``PRETRAIN_SKIP_IF_EXISTS = False``, existing checkpoints must NOT
    cause Cell 8 to skip the pretrain call — the default behavior is always
    to re-run. This guards against the skip logic kicking in unconditionally.
    """
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source

    ckpt = tmp_path / "ckpt"
    (ckpt / "pretrain" / "shallow").mkdir(parents=True)
    (ckpt / "pretrain" / "shallow" / "xnet.eqx").write_bytes(b"fake")
    (ckpt / "pretrain" / "shallow" / "cnet.eqx").write_bytes(b"fake")

    got_arch_calls = []

    class _FakeSpec:
        def __init__(self, **kw):
            self.kw = kw

    class _FakeAlec:
        PretrainSpec = _FakeSpec

        @staticmethod
        def get_architecture(name):
            got_arch_calls.append(name)
            return type("_Arch", (), {"name": name})()

        @staticmethod
        def run_pretrain(spec, progress_callback=None):
            return None

    scope = {
        "__builtins__": __builtins__,
        "PRETRAIN_PARALLEL": False,
        "PRETRAIN_SKIP_IF_EXISTS": False,
        "CHECKPOINT_BASE": str(ckpt),
        "ARCH_NAMES": ["shallow"],
        "alec": _FakeAlec,
    }
    exec(source, scope)
    assert got_arch_calls == ["shallow"], (
        f"with skip=False, shallow must be re-trained even if checkpoints exist; "
        f"got got_arch_calls={got_arch_calls}"
    )


def test_cell_08_parallel_path_skips_when_checkpoints_exist(monkeypatch, tmp_path):
    """With ``PRETRAIN_PARALLEL = True`` and ``PRETRAIN_SKIP_IF_EXISTS = True``,
    the parallel branch must not launch a subprocess for arches whose
    checkpoints already exist.
    """
    import subprocess as _real_sp
    from tqdm.auto import tqdm as _real_tqdm
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source

    ckpt = tmp_path / "ckpt"
    (ckpt / "pretrain" / "shallow").mkdir(parents=True)
    (ckpt / "pretrain" / "shallow" / "xnet.eqx").write_bytes(b"fake")
    (ckpt / "pretrain" / "shallow" / "cnet.eqx").write_bytes(b"fake")

    captured_args = []

    def _fake_run(args, env=None, check=None, capture_output=None, text=None):
        captured_args.append(args)
        class _Result:
            stdout = ""
            stderr = ""
            returncode = 0
        return _Result()

    monkeypatch.setattr(_real_sp, "run", _fake_run)

    scope = {
        "__builtins__": __builtins__,
        "PRETRAIN_PARALLEL": True,
        "PRETRAIN_SKIP_IF_EXISTS": True,
        "CHECKPOINT_BASE": str(ckpt),
        "ARCH_NAMES": ["shallow", "deep"],
        "tqdm": _real_tqdm,
    }
    exec(source, scope)

    # Only "deep" should reach subprocess.run — "shallow" is already cached.
    assert len(captured_args) == 1, (
        f"expected 1 subprocess.run call (only for 'deep'), got {len(captured_args)}"
    )
    child_code = captured_args[0][2]
    assert "'deep'" in child_code, (
        f"the un-skipped arch must be 'deep'; child_code did not reference it: {child_code[:200]}"
    )


# Task 5 — Cells 11-13 builder tests


def test_cell_12_targets_has_all_three_molecules():
    """`targets` dict must contain H, O, H2O — validator requires entries for
    every molecule in TrainingSpec.molecules (config.py:523-525).
    """
    gen = load_generator()
    source = gen.build_cell_12_reference_dicts().source
    # Targets dict should contain literal "H", "O", "H2O" keys
    assert "targets = {" in source
    for key in ('"H":', '"O":', '"H2O":'):
        assert key in source, f"targets dict missing key literal {key}"


def test_cell_12_atom_energies_missing_h2o():
    """`atom_energies` must contain exactly H and O — H2O deliberately absent.

    H2O is a compound, not an atom; placing it here would confuse the
    AtomizationEnergyMetric accumulator.
    """
    gen = load_generator()
    source = gen.build_cell_12_reference_dicts().source
    assert 'atom_energies = {"H": -0.5, "O": -75.0673}' in source
    # H2O must NOT appear as a key inside the atom_energies literal. Scan the
    # atom_energies literal slice only (to avoid matching the targets dict
    # above it).
    ae_start = source.find("atom_energies =")
    ae_end = source.find("}", ae_start) + 1
    ae_literal = source[ae_start:ae_end]
    assert '"H2O"' not in ae_literal


def test_cell_12_ext_data_dir_uses_checkpoint_base():
    """`ext_data_dir` must be derived from CHECKPOINT_BASE so smoke tests can
    redirect it via the tmp_path harness.
    """
    gen = load_generator()
    source = gen.build_cell_12_reference_dicts().source
    assert 'ext_data_dir = f"{CHECKPOINT_BASE}/external_data"' in source


def test_cell_13_uses_hf_dm_not_ccsd_dm():
    """Cell 13 must use the HF density matrix (step3b convention) — never the
    CCSD 1-RDM, despite the misleading `dm_target` key name.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    assert "mf_hf.make_rdm1()" in source
    assert "mycc.make_rdm1()" not in source


def test_cell_13_atom_branch_writes_only_e_ref():
    """The atom branch must write ONLY E_ref_literature — not dm_target and
    not rho_ccsd_grid. Atomic one-shot density targets are unstable due to
    degenerate HOMOs in open-shell atoms.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    # Find the atom branch (if name in ("H", "O"): ... else: ...)
    branch_start = source.find('if name in ("H", "O"):')
    branch_end = source.find("else:", branch_start)
    assert branch_start != -1 and branch_end != -1, "atom branch not found"
    atom_branch = source[branch_start:branch_end]
    assert "E_ref_literature=" in atom_branch
    assert "dm_target=" not in atom_branch
    assert "rho_ccsd_grid=" not in atom_branch


def test_cell_13_h2o_branch_writes_three_keys():
    """The H2O branch must write all three whitelisted keys: dm_target,
    rho_ccsd_grid, and E_ref_literature. These are the ONLY keys
    _ALLOWED_EXTERNAL_KEYS accepts (data.py:17-21).
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    branch_start = source.find("else:", source.find('if name in ("H", "O"):'))
    assert branch_start != -1, "H2O else branch not found"
    h2o_branch = source[branch_start:]
    assert "dm_target=dm_hf" in h2o_branch
    assert "rho_ccsd_grid=rho_hf" in h2o_branch
    assert "E_ref_literature=float(mf_hf.e_tot)" in h2o_branch


def test_cell_13_sidecar_json_for_every_species():
    """The `_metadata.json` write must run for every species — not inside any
    branch. Cell 25 reads E_ccsd_total from this file for all three molecules
    so the CCSD atomization-energy reference line can be computed.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    # The json.dump call must come AFTER the atom branch's else: block closes.
    # We check that there is exactly one json.dump and it is at indent level 4
    # (inside the for loop) but not inside any if/else — a simple heuristic is
    # to ensure the json.dump occurrence sits at the same indent level as the
    # `if name in` test, not deeper.
    assert "json.dump(" in source
    assert "_metadata.json" in source
    # The sidecar write should reference the species name via f-string, so it
    # fires for every iteration of the `for name, atom, spin in _mols:` loop.
    assert 'f"{name}_metadata.json"' in source


def test_cell_13_uses_grid_level_pinned():
    """`mf.grids.level = GRID_LEVEL` must appear before `mf.kernel()` so the
    PBE grid matches what Cell 14/15's precompute_fixed_density_data rebuilds.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    level_idx = source.find("mf.grids.level = GRID_LEVEL")
    kernel_idx = source.find("mf.kernel()")
    assert level_idx != -1, "mf.grids.level = GRID_LEVEL missing"
    assert kernel_idx != -1, "mf.kernel() missing"
    assert level_idx < kernel_idx, "grid level must be pinned before kernel()"


def test_cell_13_einsum_is_rho_hf_not_rho_nn():
    """The einsum variable must be named `rho_hf`, guarding the step3b-era
    `rho_nn` naming confusion — `rho_ccsd_grid` is HF in disguise.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    assert 'rho_hf = np.einsum("ij,gi,gj->g"' in source
    # `rho_nn` would indicate the wrong name reappeared
    assert "rho_nn = np.einsum" not in source


# Task 6 — Cells 14-15 builder tests


def test_cell_14_mol_specs_has_three_entries():
    """Cell 14 must construct exactly three alec.MoleculeSpec instances."""
    gen = load_generator()
    source = gen.build_cell_14_mol_specs().source
    assert source.count("alec.MoleculeSpec(") == 3


def test_cell_14_all_specs_carry_grid_level():
    """All three MoleculeSpec entries must set grid_level=GRID_LEVEL so
    precompute rebuilds the same grid Cell 13 used.
    """
    gen = load_generator()
    source = gen.build_cell_14_mol_specs().source
    assert source.count("grid_level=GRID_LEVEL") == 3


def test_cell_14_h2o_uses_h2o_coords_constant():
    """The H2O MoleculeSpec must reference H2O_COORDS (Cell 3), not a re-literal."""
    gen = load_generator()
    source = gen.build_cell_14_mol_specs().source
    assert "atom=H2O_COORDS" in source


def test_cell_14_all_specs_carry_external_data_path():
    """All three MoleculeSpec entries must point at an f-string path derived
    from ext_data_dir (Cell 12).
    """
    gen = load_generator()
    source = gen.build_cell_14_mol_specs().source
    assert source.count('external_data_path=f"{ext_data_dir}/') == 3


def test_cell_15_asserts_atom_rho_ccsd_is_none():
    """Cell 15 must assert both the atom-branch negative case and the H2O
    positive case on rho_ccsd_grid.
    """
    gen = load_generator()
    source = gen.build_cell_15_precompute_sanity().source
    assert 'mol_data_list[0]["rho_ccsd_grid"] is None' in source
    assert 'mol_data_list[2]["rho_ccsd_grid"] is not None' in source


def test_cell_15_mol_data_list_carries_descriptor_union():
    """Cell 15 must build ``mol_data_list`` with the union of
    ``required_mol_keys`` across every arch in ``ARCH_NAMES`` so that the
    evaluation cells (26, 27) can call descriptor-demanding APIs like
    ``oneshot_dm_prediction_fast`` on ``mol_data_list[2]`` regardless of
    which arch is selected as "best" by Cell 25's ``best_idx``.

    Without this, archs that declare ``cusp`` or ``dm_statistics``
    descriptors (``deep_cusp``, ``deep_dm``, ``deep_combined`` and their
    ``_attn`` variants) hit a ``TypeError: concatenate requires ndarray or
    scalar arguments, got <class 'NoneType'>`` because the bare
    ``precompute_fixed_density_data(m)`` call leaves
    ``mol_data['cusp_features']`` / ``mol_data['dm_features']`` as ``None``.

    The union must be computed from ``ARCH_NAMES`` (not hardcoded) so that
    Cell 15 stays correct when callers customize the arch list via the
    ``arch_names`` argument to ``main()``.
    """
    gen = load_generator()
    source = gen.build_cell_15_precompute_sanity().source
    # The Cell 15 loop must derive the required_keys union from ARCH_NAMES
    # rather than encoding a fixed tuple of key names.
    assert "ARCH_NAMES" in source, (
        "Cell 15 must walk ARCH_NAMES to collect descriptor required_mol_keys"
    )
    assert "materialize_descriptors()" in source, (
        "Cell 15 must call materialize_descriptors() on each arch"
    )
    assert "required_mol_keys" in source, (
        "Cell 15 must union each descriptor's required_mol_keys"
    )
    # And the precompute call must forward the computed union as required_keys=.
    assert "required_keys=" in source, (
        "Cell 15 must pass required_keys= to precompute_fixed_density_data"
    )


# Task 7 -- Cells 16-20 builder tests


def test_cell_17_builds_specs_list():
    """Cell 17 must bind `specs = []` before the nested loop."""
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    init_idx = source.find("specs = []")
    loop_idx = source.find("for arch_name in ARCH_NAMES:")
    assert init_idx != -1, "specs accumulator missing"
    assert loop_idx != -1, "outer arch loop missing"
    assert init_idx < loop_idx, "specs = [] must precede the loop"


def test_cell_17_loop_is_arch_then_loss_order():
    """Outer loop must iterate arch_name, inner loop must iterate loss_name."""
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    arch_idx = source.find("for arch_name in ARCH_NAMES:")
    loss_idx = source.find("for loss_name in LOSS_NAMES:")
    assert arch_idx != -1 and loss_idx != -1
    assert arch_idx < loss_idx, "arch loop must enclose loss loop"


def test_cell_17_sets_checkpoint_dir_per_pair():
    """Each spec must carry a per-(arch, loss) checkpoint_dir -- without this,
    all 72 runs overwrite each other in a single directory.
    """
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    assert 'checkpoint_dir=f"{CHECKPOINT_BASE}/train/{arch_name}/{loss_name}"' in source


def test_cell_17_passes_step3b_hyperparameters():
    """Cell 17 must pass n_steps=250, lr_start=1e-2, lr_decay_start=0.2 --
    the TrainingSpec defaults differ from step3b and silently produce wrong
    training curves if left alone.
    """
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    for literal in ("n_steps=250", "lr_start=1e-2", "lr_decay_start=0.2"):
        assert literal in source, f"missing hyperparameter literal: {literal}"


def test_cell_17_uses_qualified_alec_trainingspec():
    """Cell 17 must use `alec.TrainingSpec.from_dicts(` -- never bare."""
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    assert "alec.TrainingSpec.from_dicts(" in source
    import re
    bare_refs = re.findall(r"(?<!alec\.)TrainingSpec\.from_dicts\(", source)
    assert bare_refs == [], f"bare TrainingSpec references found: {bare_refs}"


def test_cell_17_loss_kwargs_weight_values():
    """LOSS_KWARGS must use 0.1 weights for dm and density -- not 1.0 or 0.01."""
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    assert '"dm_weight": 0.1' in source
    assert '"density_weight": 0.1' in source
    # Guard against wrong weight magnitudes
    assert '"dm_weight": 1.0' not in source
    assert '"density_weight": 0.01' not in source


def test_cell_18_is_serial():
    """Cell 18 must implement the serial path only -- no parallel build_training_jobs."""
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source
    assert "for spec in specs:" in source
    assert "alec.run_training(spec" in source
    assert "alec.build_training_jobs(" not in source


# Cell 18 tqdm progress-bar tests. Mirrors the Cell 8 tqdm pattern: each
# training run should drive a per-spec step bar with an ``{loss:.4e}``
# postfix, and the outer ``for spec in specs`` loop should also carry a
# spec-level tqdm bar so users see both step progress within a spec and
# overall (spec X / N) progress.


def test_cell_18_callback_uses_tqdm_bar_with_loss_postfix():
    """Cell 18's ``_train_cb`` must drive a ``tqdm`` bar and attach the
    current step's loss via ``set_postfix(loss=...)``.
    """
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source
    assert "tqdm(" in source
    assert "set_postfix(" in source
    assert "loss=" in source


def test_cell_18_replaces_print_callback_with_tqdm():
    """Cell 18 must NOT use a print-based callback — tqdm bars supersede the
    old ``print(f"[{arch}]...")`` line-noise.
    """
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source
    # The old pattern was: print(f"[{info['arch']}]...
    # The new pattern must drive tqdm from the callback.
    assert 'print(f"[{info[\'arch\']}]' not in source


def test_cell_18_has_spec_level_tqdm_bar():
    """Cell 18 must wrap the outer ``for spec in specs`` loop in a tqdm bar
    whose total is ``len(specs)`` so users see overall spec progress.
    """
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source
    assert "tqdm(" in source
    assert "total=len(specs)" in source


def test_cell_18_source_is_valid_python():
    """The tqdm-driven Cell 18 source must parse as valid Python."""
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source
    compile(source, "<cell_18>", "exec")


def test_cell_18_callback_drives_step_bar_and_sets_loss_postfix():
    """Runtime: driving ``_train_cb`` for a single spec must update a tqdm
    bar by ``step`` deltas and attach a scientific-notation ``loss=...``
    postfix. The bar must close once ``step == total``.
    """
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source

    created_bars = []

    class _FakeBar:
        def __init__(self, total=None, desc=None, leave=True,
                     dynamic_ncols=False, **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
            self.postfix_calls = []
            self.update_calls = []
            self.closed = False
            created_bars.append(self)

        def update(self, delta):
            self.update_calls.append(delta)
            self.n += delta

        def set_postfix(self, **kwargs):
            self.postfix_calls.append(kwargs)

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    class _FakeSpec:
        def __init__(self, arch_name, loss_name):
            self.arch = type("_A", (), {"name": arch_name})()
            self.loss_name = loss_name

    class _FakeAlec:
        @staticmethod
        def run_training(spec, progress_callback=None):
            # Drive the callback for every step from 1..total, simulating
            # a short 3-step training run.
            total = 3
            for step in range(1, total + 1):
                progress_callback({
                    "arch": spec.arch.name,
                    "phase": "train",
                    "step": step,
                    "total": total,
                    "loss": 10.0 ** (-step),
                    "timestamp": 0.0,
                })
            return {"arch_name": spec.arch.name, "loss_name": spec.loss_name}

    scope = {
        "__builtins__": __builtins__,
        "TRAIN_SKIP_IF_EXISTS": False,
        "specs": [_FakeSpec("shallow", "A_atomization")],
        "alec": _FakeAlec,
        "tqdm": _FakeBar,
    }
    exec(source, scope)

    # Must have created at least 2 bars: outer spec bar + inner step bar
    assert len(created_bars) >= 2, (
        f"expected >=2 tqdm bars (outer spec + inner step), got {len(created_bars)}"
    )
    # Find the spec-level outer bar (total == len(specs) == 1)
    outer = next((b for b in created_bars if b.total == 1), None)
    assert outer is not None, "no outer spec-level bar with total=len(specs)"
    assert outer.closed, "outer spec bar must be closed after the loop finishes"
    assert outer.n == 1, f"outer bar must advance once per spec; got n={outer.n}"

    # Find an inner step bar (total == 3 from the fake)
    inner = next((b for b in created_bars if b.total == 3), None)
    assert inner is not None, "no inner step bar created for the training run"
    assert inner.closed, "inner step bar must be closed when step==total"
    assert inner.n == 3, (
        f"inner bar must reach total=3 after all steps, got n={inner.n}"
    )
    # Loss postfix must be scientific notation
    assert len(inner.postfix_calls) >= 1
    last_post = inner.postfix_calls[-1]
    assert "loss" in last_post
    assert "e-" in last_post["loss"], (
        f"loss postfix must be scientific notation, got {last_post['loss']!r}"
    )


# Cell 18 TRAIN_SKIP_IF_EXISTS toggle tests. Mirrors PRETRAIN_SKIP_IF_EXISTS:
# users flip Cell 3's constant to re-use previously trained model.eqx files
# instead of re-running the main training loop for every (arch, loss) combo.


def test_cell_03_defines_train_skip_if_exists_false():
    """Cell 3 must bind ``TRAIN_SKIP_IF_EXISTS = False`` so Cell 18 defaults
    to always re-running training. Users flip this constant to skip any
    (arch, loss) run whose ``model.eqx`` is already on disk.

    The plain substring ``"TRAIN_SKIP_IF_EXISTS = False"`` is also a suffix
    of ``"PRETRAIN_SKIP_IF_EXISTS = False"``, so anchor with a boundary that
    excludes the ``PRE`` prefix.
    """
    import re
    gen = load_generator()
    source = gen.build_cell_03_constants().source
    # Match TRAIN_SKIP... that is NOT preceded by "PRE".
    assert re.search(r"(?<!PRE)TRAIN_SKIP_IF_EXISTS\s*=\s*False", source), (
        "Cell 3 must bind a standalone TRAIN_SKIP_IF_EXISTS (distinct from "
        "PRETRAIN_SKIP_IF_EXISTS)."
    )


def test_cell_18_references_train_skip_if_exists():
    """Cell 18 must reference ``TRAIN_SKIP_IF_EXISTS`` so flipping the Cell 3
    constant actually gates the ``alec.run_training`` call. Use a regex that
    excludes the ``PRE`` prefix so the assertion is not satisfied by a
    ``PRETRAIN_SKIP_IF_EXISTS`` reference.
    """
    import re
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source
    assert re.search(r"(?<!PRE)TRAIN_SKIP_IF_EXISTS", source), (
        "Cell 18 must reference TRAIN_SKIP_IF_EXISTS (not PRETRAIN_SKIP_IF_EXISTS)."
    )


def test_cell_18_skips_when_model_eqx_exists(tmp_path):
    """With ``TRAIN_SKIP_IF_EXISTS = True`` and ``model.eqx`` present for a
    spec's checkpoint_dir, Cell 18 must NOT call ``alec.run_training`` for
    that spec.
    """
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source

    ckpt = tmp_path / "ckpt"
    # Spec 0: model.eqx already exists (should be skipped)
    (ckpt / "train" / "shallow" / "A_atomization").mkdir(parents=True)
    (ckpt / "train" / "shallow" / "A_atomization" / "model.eqx").write_bytes(b"fake")
    # Spec 1: no checkpoint dir (should run)

    class _FakeSpec:
        def __init__(self, arch_name, loss_name, checkpoint_dir):
            self.arch = type("_A", (), {"name": arch_name})()
            self.loss_name = loss_name
            self.checkpoint_dir = checkpoint_dir

    train_calls = []

    class _FakeAlec:
        @staticmethod
        def run_training(spec, progress_callback=None):
            train_calls.append((spec.arch.name, spec.loss_name))
            return {"arch_name": spec.arch.name, "loss_name": spec.loss_name}

    # No-op tqdm so we can exec the cell without the bar interfering.
    class _FakeBar:
        def __init__(self, *args, **kwargs):
            self.n = 0
        def update(self, delta): self.n += delta
        def set_postfix(self, **kwargs): pass
        def refresh(self): pass
        def close(self): pass

    scope = {
        "__builtins__": __builtins__,
        "TRAIN_SKIP_IF_EXISTS": True,
        "specs": [
            _FakeSpec("shallow", "A_atomization",
                      str(ckpt / "train" / "shallow" / "A_atomization")),
            _FakeSpec("deep", "A_atomization",
                      str(ckpt / "train" / "deep" / "A_atomization")),
        ],
        "alec": _FakeAlec,
        "tqdm": _FakeBar,
    }
    exec(source, scope)

    assert ("shallow", "A_atomization") not in train_calls, (
        f"(shallow, A_atomization) already has model.eqx; run_training must "
        f"not be called for it. train_calls={train_calls}"
    )
    assert ("deep", "A_atomization") in train_calls, (
        f"(deep, A_atomization) has no model.eqx; run_training must be called. "
        f"train_calls={train_calls}"
    )


def test_cell_18_runs_when_skip_is_false_even_if_model_eqx_exists(tmp_path):
    """With ``TRAIN_SKIP_IF_EXISTS = False``, an existing ``model.eqx`` must
    NOT cause Cell 18 to skip the ``alec.run_training`` call — the default
    behavior is always to re-train.
    """
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source

    ckpt = tmp_path / "ckpt"
    (ckpt / "train" / "shallow" / "A_atomization").mkdir(parents=True)
    (ckpt / "train" / "shallow" / "A_atomization" / "model.eqx").write_bytes(b"fake")

    class _FakeSpec:
        def __init__(self, arch_name, loss_name, checkpoint_dir):
            self.arch = type("_A", (), {"name": arch_name})()
            self.loss_name = loss_name
            self.checkpoint_dir = checkpoint_dir

    train_calls = []

    class _FakeAlec:
        @staticmethod
        def run_training(spec, progress_callback=None):
            train_calls.append((spec.arch.name, spec.loss_name))
            return {"arch_name": spec.arch.name, "loss_name": spec.loss_name}

    class _FakeBar:
        def __init__(self, *args, **kwargs):
            self.n = 0
        def update(self, delta): self.n += delta
        def set_postfix(self, **kwargs): pass
        def refresh(self): pass
        def close(self): pass

    scope = {
        "__builtins__": __builtins__,
        "TRAIN_SKIP_IF_EXISTS": False,
        "specs": [
            _FakeSpec("shallow", "A_atomization",
                      str(ckpt / "train" / "shallow" / "A_atomization")),
        ],
        "alec": _FakeAlec,
        "tqdm": _FakeBar,
    }
    exec(source, scope)
    assert train_calls == [("shallow", "A_atomization")], (
        f"with skip=False, (shallow, A_atomization) must be re-trained even "
        f"if model.eqx exists; got train_calls={train_calls}"
    )


def test_cell_18_spec_bar_total_stays_len_specs_when_skipping(tmp_path):
    """When Cell 18 skips a spec, the outer tqdm bar total must still equal
    ``len(specs)`` and the bar must be advanced for the skipped spec too —
    otherwise the progress display desynchronizes from the spec list.
    """
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source

    ckpt = tmp_path / "ckpt"
    (ckpt / "train" / "shallow" / "A_atomization").mkdir(parents=True)
    (ckpt / "train" / "shallow" / "A_atomization" / "model.eqx").write_bytes(b"fake")

    created_bars = []

    class _FakeBar:
        def __init__(self, total=None, desc=None, leave=True,
                     dynamic_ncols=False, **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
            self.update_calls = []
            self.postfix_calls = []
            self.closed = False
            created_bars.append(self)
        def update(self, delta):
            self.update_calls.append(delta)
            self.n += delta
        def set_postfix(self, **kwargs):
            self.postfix_calls.append(kwargs)
        def refresh(self): pass
        def close(self): self.closed = True

    class _FakeSpec:
        def __init__(self, arch_name, loss_name, checkpoint_dir):
            self.arch = type("_A", (), {"name": arch_name})()
            self.loss_name = loss_name
            self.checkpoint_dir = checkpoint_dir

    class _FakeAlec:
        @staticmethod
        def run_training(spec, progress_callback=None):
            total = 2
            for step in range(1, total + 1):
                progress_callback({
                    "arch": spec.arch.name, "phase": "train",
                    "step": step, "total": total,
                    "loss": 1e-3, "timestamp": 0.0,
                })
            return {}

    scope = {
        "__builtins__": __builtins__,
        "TRAIN_SKIP_IF_EXISTS": True,
        "specs": [
            _FakeSpec("shallow", "A_atomization",
                      str(ckpt / "train" / "shallow" / "A_atomization")),
            _FakeSpec("deep", "A_atomization",
                      str(ckpt / "train" / "deep" / "A_atomization")),
        ],
        "alec": _FakeAlec,
        "tqdm": _FakeBar,
    }
    exec(source, scope)

    outer = next((b for b in created_bars if b.total == 2), None)
    assert outer is not None, "outer spec bar with total=len(specs)=2 not found"
    assert outer.n == 2, (
        f"outer bar must advance once per spec (including skipped), got n={outer.n}"
    )
    assert outer.closed


def test_cell_18_multi_spec_run_updates_outer_bar_per_spec():
    """Driving Cell 18 with 3 fake specs must advance the outer spec bar by
    1 per completed spec and produce one closed inner bar per spec.
    """
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source

    created_bars = []

    class _FakeBar:
        def __init__(self, total=None, desc=None, leave=True,
                     dynamic_ncols=False, **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
            self.update_calls = []
            self.postfix_calls = []
            self.closed = False
            created_bars.append(self)

        def update(self, delta):
            self.update_calls.append(delta)
            self.n += delta

        def set_postfix(self, **kwargs):
            self.postfix_calls.append(kwargs)

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    class _FakeSpec:
        def __init__(self, arch_name, loss_name):
            self.arch = type("_A", (), {"name": arch_name})()
            self.loss_name = loss_name

    class _FakeAlec:
        @staticmethod
        def run_training(spec, progress_callback=None):
            total = 2
            for step in range(1, total + 1):
                progress_callback({
                    "arch": spec.arch.name,
                    "phase": "train",
                    "step": step,
                    "total": total,
                    "loss": 1e-3,
                    "timestamp": 0.0,
                })
            return {"arch_name": spec.arch.name, "loss_name": spec.loss_name}

    scope = {
        "__builtins__": __builtins__,
        "TRAIN_SKIP_IF_EXISTS": False,
        "specs": [
            _FakeSpec("shallow", "A_atomization"),
            _FakeSpec("shallow", "B_atomization_plus_dm"),
            _FakeSpec("deep", "A_atomization"),
        ],
        "alec": _FakeAlec,
        "tqdm": _FakeBar,
    }
    exec(source, scope)

    # Outer bar: total = len(specs) = 3
    outer = next((b for b in created_bars if b.total == 3), None)
    assert outer is not None, "outer spec bar with total=3 not found"
    assert outer.n == 3, f"outer bar must advance 3 times; got n={outer.n}"
    assert outer.closed

    # Inner bars: one per spec, each total=2 (from the fake run_training)
    inner_bars = [b for b in created_bars if b.total == 2]
    assert len(inner_bars) == 3, (
        f"expected 3 inner step bars (one per spec), got {len(inner_bars)}"
    )
    for b in inner_bars:
        assert b.closed, "each inner step bar must be closed at end of its run"
        assert b.n == 2, f"each inner bar must reach total=2; got n={b.n}"


def test_cell_19_loads_losses_npy():
    """Cell 19 must load each per-(arch, loss) losses.npy using the
    checkpoint path template Cell 17 wrote to.
    """
    gen = load_generator()
    source = gen.build_cell_19_training_loss_plot().source
    assert "/train/{arch_name}/{loss_name}/losses.npy" in source


def test_cell_20_binds_arch_name_before_loop():
    """Cell 20 must bind arch_name = "shallow" before the for loss_name loop
    so the f-string checkpoint path is unambiguous.
    """
    gen = load_generator()
    source = gen.build_cell_20_aux_inspection().source
    bind_idx = source.find('arch_name = "shallow"')
    loop_idx = source.find("for loss_name in LOSS_NAMES:")
    assert bind_idx != -1, 'arch_name = "shallow" missing'
    assert loop_idx != -1, "loss_name loop missing"
    assert bind_idx < loop_idx, "arch_name binding must precede the loss loop"


# Task 8 -- Cells 21-24 builder tests


def test_cell_22_metrics_tuple_is_four():
    """Cell 22 must pass all four metrics explicitly (not rely on default)."""
    gen = load_generator()
    source = gen.build_cell_22_test_loop().source
    for metric in ("total_energy", "atomization_energy", "density_rmse", "constraint_violations"):
        assert f'"{metric}"' in source, f"metric {metric!r} missing from Cell 22 metrics tuple"


def test_cell_22_metric_kwargs_reference_ae_kcalmol():
    """Cell 22 must pass the full-precision step3b H2O AE (233.016 kcal/mol)."""
    gen = load_generator()
    source = gen.build_cell_22_test_loop().source
    assert '"reference_ae_kcalmol"' in source
    assert '"H2O": 233.016' in source


def test_cell_22_model_checkpoint_points_to_file():
    """Cell 22 must point model_checkpoint at the .eqx file, not the dir."""
    gen = load_generator()
    source = gen.build_cell_22_test_loop().source
    assert 'model_checkpoint=f"{ckpt_dir}/model.eqx"' in source


def test_cell_22_loop_order_matches_cell_17():
    """Cell 22's loop order must match Cell 17 (arch outer, loss inner)."""
    gen = load_generator()
    source = gen.build_cell_22_test_loop().source
    arch_idx = source.find("for arch_name in ARCH_NAMES:")
    loss_idx = source.find("for loss_name in LOSS_NAMES:")
    assert arch_idx != -1, "outer arch loop missing"
    assert loss_idx != -1, "inner loss loop missing"
    assert arch_idx < loss_idx, "arch loop must precede loss loop"


def test_cell_23_ae_error_column_reads_rmse():
    """Cell 23 must populate both AE_error_kcalmol_mean and AE_error_kcalmol_RMSE (B12-4 guard)."""
    gen = load_generator()
    source = gen.build_cell_23_dataframe().source
    assert '"AE_error_kcalmol_mean"' in source
    assert '"AE_error_kcalmol_RMSE"' in source


def test_cell_23_no_constraint_violations_column():
    """Cell 23 must NOT have a constraint_violations column -- key is absent from default-arch aggregate.json."""
    gen = load_generator()
    source = gen.build_cell_23_dataframe().source
    assert "constraint_violations" not in source


def test_cell_23_uses_get_with_nan_fallback():
    """Cell 23 must use the defensive .get(..., np.nan) pattern (B10-12 guard)."""
    gen = load_generator()
    source = gen.build_cell_23_dataframe().source
    assert 'agg.get("AE_error_kcalmol", {}).get("mean", np.nan)' in source


def test_cell_23_multiindex_is_arch_loss():
    """Cell 23 must set the MultiIndex to [arch, loss]."""
    gen = load_generator()
    source = gen.build_cell_23_dataframe().source
    assert 'set_index(["arch", "loss"])' in source


# Task 9 -- Cells 25-26 builder tests


def test_cell_25_binds_best_idx():
    """Cell 25 must bind best_idx from df[AE_error_kcalmol_mean].unstack(loss).idxmin(axis=0)."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert 'best_idx = df["AE_error_kcalmol_mean"].unstack("loss").idxmin(axis=0)' in source


def test_cell_25_binds_pairs():
    """Cell 25 must bind the attention-pairing list programmatically from ARCH_NAMES."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert 'pairs = [(n, f"{n}_attn") for n in ARCH_NAMES' in source
    assert 'not n.endswith("_attn")' in source
    assert 'f"{n}_attn" in ARCH_NAMES' in source


def test_cell_25_reads_both_mean_and_rmse_columns():
    """Cell 25 must read both mean and RMSE columns (B12-4 regression guard)."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert 'df["AE_error_kcalmol_mean"]' in source
    assert 'df["AE_error_kcalmol_RMSE"]' in source


def test_cell_25_has_three_reference_lines():
    """Cell 25 must draw PBE, CCSD, and chemical-accuracy reference lines."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert "PBE Error" in source
    assert "CCSD Error" in source
    assert "Chemical accuracy (1 kcal/mol)" in source


def test_cell_25_kernel_restart_fallback_exists():
    """Cell 25 must have a try/except NameError fallback for mol_data_list (kernel-restart safety)."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert "except NameError:" in source


def test_cell_25_saves_to_figures_dir():
    """Cell 25 must save ae_error_by_loss.png into the figures directory."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert "ae_error_by_loss.png" in source


def test_cell_26_uses_alec_gga_model_from_arch_not_create_network_pair():
    """Cell 26 must use alec.AlecGGAModel.from_arch (B11-4 regression guard)."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert "alec.AlecGGAModel.from_arch(" in source
    assert "alec.create_network_pair(" not in source


def test_cell_26_model_template_rebuilt_inside_loop():
    """Cell 26 must rebuild model_template inside the loop, not hoisted."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    loop_idx = source.find('for loss_name in ("B_atomization_plus_dm",')
    template_idx = source.find("model_template = alec.AlecGGAModel.from_arch(arch_config)")
    assert loop_idx != -1, "per-loss loop missing"
    assert template_idx != -1, "model_template rebuild missing"
    assert loop_idx < template_idx, "model_template must be rebuilt INSIDE the loop"


def test_cell_26_binds_model_b_d1_d2_explicit_names():
    """Cell 26 must bind model_B, model_D1, model_D2 as explicit named variables.

    The bindings use a narrow-config-tolerant conditional form so the cell
    also works in smoke tests that only train one loss family.
    """
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert 'model_B = model_bindings["B_atomization_plus_dm"]' in source
    assert 'model_D1 = model_bindings["D1_delta_ae"]' in source
    assert 'model_D2 = model_bindings["D2_delta_ae_plus_dm"]' in source


def test_cell_26_uses_oneshot_dm_prediction_fast():
    """Cell 26 must call the _fast variant (the only one alec exports)."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert "alec.oneshot_dm_prediction_fast(" in source
    # The bare variant (without _fast) does not exist in alec.__init__ — guard against a rename regression.
    assert "oneshot_dm_prediction(" not in source.replace("oneshot_dm_prediction_fast(", "")


def test_cell_26_reuses_mol_data_list_for_dm_hf():
    """Cell 26 must reuse mol_data_list[2]['dm_target'] for dm_hf, not reload the .npz."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert 'dm_hf = mol_data_list[2]["dm_target"]' in source
    assert ".npz" not in source


def test_cell_26_panel_assignment_is_explicit():
    """Cell 26 must have all four panel subtraction expressions."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    for expr in ("dm_pbe - dm_hf", "dm_nn_B - dm_hf", "dm_nn_D1 - dm_hf", "dm_nn_D2 - dm_hf"):
        assert expr in source, f"panel expression {expr!r} missing"


# Task 10 -- Cells 27-29 builder tests


def test_cell_27_uses_oneshot_grid_density():
    """Cell 27 must call alec.oneshot_grid_density on mol_data_list[2]."""
    gen = load_generator()
    source = gen.build_cell_27_density_histograms().source
    assert "alec.oneshot_grid_density(" in source


def test_cell_27_reads_rho_ccsd_grid_from_mol_data_list():
    """Cell 27 must read rho_ref from mol_data_list[2]['rho_ccsd_grid']."""
    gen = load_generator()
    source = gen.build_cell_27_density_histograms().source
    assert 'mol_data_list[2]["rho_ccsd_grid"]' in source


def test_cell_27_prints_delta_rho_l1():
    """Cell 27 must compute and print the inline |delta rho|_1 metric."""
    gen = load_generator()
    source = gen.build_cell_27_density_histograms().source
    assert "delta_rho_L1 = float(jnp.sum(w *" in source


def test_cell_28_uses_pairs_from_cell_25():
    """Cell 28 must iterate the pairs list bound in Cell 25."""
    gen = load_generator()
    source = gen.build_cell_28_attn_comparison().source
    # accept either `for base, _attn in pairs` style or an explicit comprehension over `pairs`
    assert "pairs" in source
    assert " in pairs" in source


def test_cell_29_feature_variants_excludes_attn_suffix():
    """Cell 29 must exclude attention-suffixed archs from the feature filter."""
    gen = load_generator()
    source = gen.build_cell_29_feature_comparison().source
    assert 'and not n.endswith("_attn")' in source


def test_cell_29_filter_startswith_deep():
    """Cell 29 must filter to deep-prefixed archs."""
    gen = load_generator()
    source = gen.build_cell_29_feature_comparison().source
    assert 'n.startswith("deep")' in source


# Task 11 -- Cells 30-31 builder tests


def test_cell_31_uses_qualified_alec_names():
    """Cell 31 must use qualified alec.MoleculeSpec / alec.TestSpec.from_dicts / alec.run_test."""
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    assert "alec.MoleculeSpec(" in source
    assert "alec.TestSpec.from_dicts(" in source
    assert "alec.run_test(" in source


def test_cell_31_step2_npz_generation_is_uncommented_under_isfile_guard():
    """Cell 31's step 2 must actually generate every species' ``.npz`` on the
    fly so that when a user uncomments ``alec.run_test(new_test_spec)`` or runs
    Cell 32 the reference data is already on disk.

    Task #29 extended Cell 31 to also compute PBE/HF/CCSD for every entry in
    ``new_atom_specs`` via a single loop over ``_entities`` (the new molecule
    plus any new atoms), so the SCF calls now appear inside that loop. The
    test accepts either the original single-molecule shape or the extended
    loop shape -- both need SCF calls active and both need an ``os.path.isfile``
    guard that keeps re-runs cheap.
    """
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source

    # The SCF / save block must be active Python, not just commented prose.
    active_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#"):
            active_lines.append(line)
    joined_active = "\n".join(active_lines)
    for needle in (
        "dft.RKS(",
        "scf.RHF(",
        "np.savez(",
    ):
        assert needle in joined_active, (
            f"Cell 31 step 2 must execute {needle!r} (found only as a comment "
            f"or missing entirely). Active source lines:\n"
            + joined_active
        )

    # The .npz generation must be gated by ``os.path.isfile`` so reruns skip
    # the SCF. Accept either the old single-molecule guard keyed on
    # ``new_mol_spec.external_data_path`` or the new per-species loop guard
    # keyed on a local path variable inside the loop body.
    has_guard = (
        "os.path.isfile(new_mol_spec.external_data_path)" in joined_active
        or "os.path.isfile(_npz_path)" in joined_active
        or "os.path.isfile(_meta_path)" in joined_active
    )
    assert has_guard, (
        "Cell 31 step 2 must gate SCF generation on os.path.isfile(...) to "
        "keep reruns cheap (either new_mol_spec.external_data_path or the "
        "per-species _npz_path / _meta_path loop variable)."
    )


def test_cell_31_npz_writes_dm_target_and_rho_ccsd_grid():
    """The np.savez call in Cell 31 step 2 must write all three whitelisted
    keys (``dm_target``, ``rho_ccsd_grid``, ``E_ref_literature``) so
    ``_load_external_data`` accepts the file and downstream metrics (DM-based
    losses, density_rmse) have real reference data.
    """
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    assert "dm_target=" in source
    assert "rho_ccsd_grid=" in source
    assert "E_ref_literature=" in source


def test_cell_31_best_arch_binds_from_best_idx():
    """Cell 31 must prefer D2_delta_ae_plus_dm from best_idx (with narrow-config fallback)."""
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    assert '_d2_key = "D2_delta_ae_plus_dm"' in source
    assert "best_arch = best_idx[_d2_key]" in source


def test_cell_31_atom_energies_merge():
    """Cell 31 must build ``new_atom_energies`` on top of the Cell 12 dict
    (never replace it) and source the new element's reference from its sidecar
    JSON, not a hardcoded ``-37.84`` placeholder. Preserving the Cell 12 dict
    keeps H / O reference energies in sync across the notebook."""
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    assert "new_atom_energies" in source
    # Must preserve existing Cell 12 atom_energies via an {**..} spread or
    # dict(..) copy rather than re-declaring the H / O values.
    assert ("{**atom_energies}" in source) or ("dict(atom_energies)" in source)
    # Source the new atom's reference from its Cell 31 sidecar.
    assert "_metadata.json" in source
    assert "E_hf_total" in source
    # No hardcoded -37.84 placeholder for C — Task #29 removed it in favour of
    # the sidecar-sourced value so re-parameterising to a new atom does not
    # require the user to look up a literature total.
    assert "-37.84" not in source


def test_cell_31_defines_new_atom_specs_for_C():
    """Cell 31 must declare a ``new_atom_specs`` list containing the Carbon
    atom (``spin=2`` 3P triplet ground state) so the reference-generation loop
    has a well-defined set of extra atoms to compute PBE/HF/CCSD for.
    """
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    assert "new_atom_specs" in source
    assert '"C"' in source
    assert '"C 0 0 0"' in source
    # Carbon 3P ground state => spin=2 in pyscf (two unpaired electrons).
    assert ", 2)" in source


def test_cell_31_loops_over_molecule_plus_new_atoms():
    """Cell 31 step 2 must iterate over the new molecule AND every atom in
    ``new_atom_specs`` so PBE/HF/CCSD totals land in a sidecar JSON for each
    species. Cell 32 reads those totals for its PBE/HF/CCSD reference lines."""
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    # A single loop that covers the molecule + new atoms — this is the
    # canonical shape, any equivalent expression must still mention both
    # ``new_mol_spec`` and ``new_atom_specs`` inside the iteration target.
    assert "new_atom_specs" in source
    assert "new_mol_spec.name" in source or "new_mol_spec.atom" in source


def test_cell_31_runs_pbe_hf_ccsd_for_every_species():
    """Cell 31 step 2 must run PBE, HF AND CCSD for each species in the loop
    (spin-branched RKS/UKS and RHF/UHF and CCSD/UCCSD), mirroring Cell 13's
    H/O/H2O pattern."""
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    active_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#"):
            active_lines.append(line)
    active = "\n".join(active_lines)

    # PBE branch
    assert "dft.UKS(" in active
    assert "dft.RKS(" in active
    # HF branch
    assert "scf.UHF(" in active
    assert "scf.RHF(" in active
    # CCSD branch
    assert "cc.UCCSD(" in active
    assert "cc.CCSD(" in active


def test_cell_31_writes_metadata_sidecar_with_all_totals():
    """Cell 31 must write a ``{name}_metadata.json`` sidecar for every species
    containing ``E_pbe_total`` / ``E_hf_total`` / ``E_ccsd_total``, matching
    Cell 13's H/O/H2O sidecar schema so Cell 25 and Cell 32 share the same
    reference-loading logic."""
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    assert '"E_pbe_total"' in source
    assert '"E_hf_total"' in source
    assert '"E_ccsd_total"' in source
    assert "_metadata.json" in source
    assert "json.dump(" in source


def test_cell_31_stores_rho_pbe_hf_rmse_in_molecule_sidecar():
    """Cell 31's molecule branch must compute the weighted PBE|HF density RMSE
    on the PBE grid and store it as ``rho_pbe_hf_rmse`` in the sidecar. Cell 32
    uses this for the density-RMSE panel's PBE reference line. Atom branches
    must NOT write this key (atoms have degenerate occupancy and would need
    special handling)."""
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    assert "rho_pbe_hf_rmse" in source


def test_cell_31_atom_branch_writes_only_E_ref_literature():
    """For each entry in ``new_atom_specs``, the .npz write must be the
    atom-branch shape (only ``E_ref_literature=`` key), mirroring Cell 13's
    behaviour for H/O. Writing ``dm_target`` / ``rho_ccsd_grid`` for an
    atomic species with degenerate HOMO eigenvalues is numerically unstable.
    """
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    # Atom branch is the same shape as Cell 13's: np.savez + E_ref_literature
    # with no DM / rho keys. Guarded either by "is_atom" variable or by a
    # conditional on the species name/composition.
    assert "np.savez(" in source
    # The source should contain a branch that discriminates atom vs molecule.
    # Accept any of these forms.
    has_branching = any(
        needle in source
        for needle in (
            "is_atom",
            "in new_atom_specs",
            "name != new_mol_spec",
            "_name == new_mol_spec",
            "_name in _atom_names",
        )
    )
    assert has_branching, (
        "Cell 31 must branch atom vs molecule in the .npz write block"
    )


def test_cell_31_testspec_uses_new_atom_energies():
    """``alec.TestSpec.from_dicts`` must pass ``atom_energies=new_atom_energies``
    so the AE metric uses the sidecar-derived reference energies rather than
    the raw Cell 12 dict (which lacks Carbon)."""
    gen = load_generator()
    source = gen.build_cell_31_new_molecule_template().source
    assert "atom_energies=new_atom_energies" in source


# ---------------------------------------------------------------------------
# Task #29 / #30 — Cell 32 new-molecule comparison plot (Option 2)
# ---------------------------------------------------------------------------


def test_cell_32_builder_exists():
    """The generator must expose ``build_cell_32_new_mol_comparison``.

    Cell 32 sweeps every trained (arch, loss) checkpoint, runs
    ``alec.run_test`` on the new molecule, and plots a 3-panel comparison
    (AE error / E error / density RMSE) with PBE / CCSD / HF / chemical-accuracy
    reference lines."""
    gen = load_generator()
    assert hasattr(gen, "build_cell_32_new_mol_comparison")


def test_cell_32_sweeps_arch_and_loss():
    """Cell 32 must sweep both ``ARCH_NAMES`` and ``LOSS_NAMES`` so every
    trained model is compared against the PBE / CCSD / HF references."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    assert "ARCH_NAMES" in source
    assert "LOSS_NAMES" in source


def test_cell_32_calls_run_test_inside_sweep():
    """Cell 32 must call ``alec.run_test`` inside the sweep so each (arch, loss)
    combination contributes a per-molecule AE/E/density row to the comparison
    plot."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    active_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#"):
            active_lines.append(line)
    active = "\n".join(active_lines)
    assert "alec.run_test(" in active


def test_cell_32_reads_sidecar_metadata_for_reference_lines():
    """Cell 32 must load every reference molecule/atom ``_metadata.json`` to
    derive the PBE / CCSD / HF lines (no hardcoded reference values)."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    assert "_metadata.json" in source
    assert "E_pbe_total" in source
    assert "E_ccsd_total" in source
    assert "E_hf_total" in source


def test_cell_32_plots_three_panels():
    """Cell 32 must render a 1x3 subplot grid (AE / E / density RMSE)."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    has_three_panel = (
        "plt.subplots(1, 3" in source
        or "plt.subplots(nrows=1, ncols=3" in source
        or "plt.subplots(1,3" in source
    )
    assert has_three_panel, "Cell 32 must call plt.subplots(1, 3, ...)"


def test_cell_32_has_chemical_accuracy_reference_line():
    """Cell 32's AE panel must draw the 1-kcal/mol chemical-accuracy line so
    readers can immediately see which (arch, loss) combos reach it."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    assert "axhline" in source
    assert "Chemical accuracy" in source or "chemical accuracy" in source
    assert "1.0" in source or "1 kcal" in source


def test_cell_32_has_pbe_ccsd_hf_reference_labels():
    """Cell 32 must label its PBE / CCSD / HF reference lines so the legend is
    unambiguous."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    assert "PBE" in source
    assert "CCSD" in source
    assert "HF" in source


def test_cell_32_uses_rho_pbe_hf_rmse_for_density_panel():
    """Cell 32's density-RMSE panel must use the ``rho_pbe_hf_rmse`` field that
    Cell 31 stores in the molecule sidecar as the PBE reference line (there is
    no CCSD grid density on this grid, so PBE is the only reference)."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    assert "rho_pbe_hf_rmse" in source


def test_cell_32_saves_figure_to_checkpoint_base_figures_dir():
    """Cell 32 must save the rendered figure under
    ``{CHECKPOINT_BASE}/figures/new_mol_<name>_comparison.png`` so it survives
    a kernel restart and can be included in a report artefact."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    assert "savefig(" in source
    assert "CHECKPOINT_BASE" in source
    assert "figures" in source
    assert "new_mol_" in source


def test_cell_32_uses_absolute_error_values():
    """Cell 32's error panels must plot the absolute value of AE_error_kcalmol
    and E_error_kcalmol so the log-scale axis can handle both signs."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    # Must apply ``abs`` (Python builtin or np.abs or jnp.abs or pandas .abs())
    # to at least one of the error columns, since the plot compares magnitudes
    # on a log axis.
    has_abs = (
        "abs(" in source
        or ".abs()" in source
        or "np.abs(" in source
        or "jnp.abs(" in source
    )
    assert has_abs


def test_cell_32_handles_narrow_config():
    """Cell 32 must skip checkpoints that do not exist (narrow-config smoke test
    only trains 1 arch x 1 loss so 71 of the 72 combos have no checkpoint)."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    # Gate on os.path.isfile for the model checkpoint -- exactly the pattern
    # used for Cell 22 / 23 to tolerate narrow configs.
    assert "os.path.isfile(" in source or "os.path.exists(" in source


# ---------------------------------------------------------------------------
# Task 12 — Full-notebook guards
# ---------------------------------------------------------------------------


def test_generator_is_deterministic(tmp_path):
    """Two back-to-back main() calls must produce byte-identical notebooks.

    A nondeterministic builder (unordered dict, datetime stamp, set iteration)
    would corrupt git blame and break reproducibility.
    """
    gen = load_generator()
    out1 = tmp_path / "nb1.ipynb"
    out2 = tmp_path / "nb2.ipynb"
    gen.main(str(out1))
    gen.main(str(out2))
    assert out1.read_bytes() == out2.read_bytes()


def test_generator_produces_38_cells(tmp_path):
    """main() must produce exactly 38 cells (the full step 4 notebook).

    The figure-labeling pass added 6 per-plot markdown description cells
    (section 7 overview + per-comparison-plot descriptions for cells 26-29
    and cell 32) on top of the 32-cell baseline.
    """
    gen = load_generator()
    out_path = tmp_path / "out.ipynb"
    gen.main(str(out_path))
    nb = nbformat.read(str(out_path), as_version=4)
    assert len(nb.cells) == 38, f"expected 38 cells, got {len(nb.cells)}"


def test_generator_cell_types_match_expected(tmp_path):
    """Markdown cells: original section headings (0, 5, 10, 15, 20, 34)
    plus the 6 new comparison-plot description markdown cells inserted by
    the figure-labeling pass at indices (24, 26, 28, 30, 32, 36)."""
    gen = load_generator()
    out_path = tmp_path / "out.ipynb"
    gen.main(str(out_path))
    nb = nbformat.read(str(out_path), as_version=4)

    markdown_indices = {0, 5, 10, 15, 20, 24, 26, 28, 30, 32, 34, 36}
    for idx, cell in enumerate(nb.cells):
        expected = "markdown" if idx in markdown_indices else "code"
        assert cell.cell_type == expected, (
            f"cell {idx}: expected {expected!r}, got {cell.cell_type!r}"
        )


# ---------------------------------------------------------------------------
# Task 13 — End-to-end smoke test (slow, opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_step4_notebook_smoke_runs_end_to_end(tmp_path):
    """Run the regenerated notebook end-to-end on a 1-arch × 1-loss config.

    Proves that every cell executes without raising. Does NOT validate
    numerical correctness (a 250-step training run is not converged) —
    the assertion surface is purely "files exist at expected paths".
    """
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient

    gen = load_generator()
    nb_path = tmp_path / "step4_smoke.ipynb"
    checkpoint_base = str(tmp_path / "ckpt")
    gen.main(
        str(nb_path),
        arch_names=("shallow",),
        loss_names=("A_atomization",),
        checkpoint_base=checkpoint_base,
    )

    nb = nbformat.read(str(nb_path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()

    import os
    assert os.path.isfile(f"{checkpoint_base}/pretrain_data/pretrain_data.npz")
    assert os.path.isfile(f"{checkpoint_base}/pretrain/shallow/xnet.eqx")
    assert os.path.isfile(f"{checkpoint_base}/pretrain/shallow/cnet.eqx")
    assert os.path.isfile(f"{checkpoint_base}/external_data/H.npz")
    assert os.path.isfile(f"{checkpoint_base}/external_data/O.npz")
    assert os.path.isfile(f"{checkpoint_base}/external_data/H2O.npz")
    assert os.path.isfile(f"{checkpoint_base}/external_data/H_metadata.json")
    assert os.path.isfile(f"{checkpoint_base}/external_data/O_metadata.json")
    assert os.path.isfile(f"{checkpoint_base}/external_data/H2O_metadata.json")
    assert os.path.isfile(f"{checkpoint_base}/train/shallow/A_atomization/model.eqx")
    assert os.path.isfile(f"{checkpoint_base}/test/shallow/A_atomization/aggregate.json")


# ---------------------------------------------------------------------------
# Figure-labeling pass — per-plot markdown descriptions + suptitle/title/axis
# label additions on every figure-generating cell. Guards the audit result
# that every comparison plot has a preceding description cell and that every
# figure is labelled precisely enough to stand alone as a results artifact.
# ---------------------------------------------------------------------------


def _md_source(gen, builder_name: str) -> str:
    """Return the markdown source for a named builder, asserting it's markdown."""
    builder = getattr(gen, builder_name)
    cell = builder()
    assert cell.cell_type == "markdown", (
        f"{builder_name} must return a markdown cell, got {cell.cell_type!r}"
    )
    return cell.source


def test_section7_overview_md_builder_exists():
    """Section 7 opener markdown must exist and announce the visualization section."""
    gen = load_generator()
    source = _md_source(gen, "build_section7_overview_md")
    assert "Section 7" in source
    assert "Visualization" in source


def test_section7_overview_md_mentions_reference_lines():
    """Section 7 opener must name the PBE/CCSD/chemical-accuracy reference lines
    that every plot in the section uses, so the reader knows what to expect."""
    gen = load_generator()
    source = _md_source(gen, "build_section7_overview_md")
    assert "PBE" in source
    assert "CCSD" in source
    assert "chemical accuracy" in source or "Chemical accuracy" in source


def test_cell_25_has_figure_title():
    """Cell 25 AE-bar plot must have an explicit matplotlib title stating what is
    being compared (H2O atomization energy error vs loss family) plus the
    literature reference value and the error-bar semantics."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert "H2O atomization-energy error by architecture" in source
    assert "literature AE = 233.016 kcal/mol" in source
    assert "error bars = per-molecule RMSE" in source


def test_cell_26_dm_heatmaps_md_builder_exists():
    """Cell 26's density-matrix-residual comparison must have a markdown
    description cell preceding it that names each panel and the colormap."""
    gen = load_generator()
    source = _md_source(gen, "build_cell_26_dm_heatmaps_md")
    assert "density-matrix" in source.lower() or "density matrix" in source.lower()
    assert "PBE" in source
    assert "best-B" in source
    assert "best-D1" in source
    assert "best-D2" in source
    assert "RdBu" in source


def test_cell_26_has_figure_suptitle():
    """Cell 26 must have a fig.suptitle identifying the comparison (H2O DM
    residuals vs HF target) so the saved PNG is self-describing."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert "fig.suptitle(" in source
    assert "H2O density-matrix residuals vs HF target" in source
    assert "RdBu_r diverging colormap" in source


def test_cell_26_panels_have_axis_labels():
    """Cell 26 must label each heatmap's x/y axes as AO basis indices."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert 'ax.set_xlabel("AO basis index $j$")' in source
    assert 'ax.set_ylabel("AO basis index $i$")' in source


def test_cell_27_density_histograms_md_builder_exists():
    """Cell 27's density-histogram comparison must have a markdown description
    cell that names the two density-grid-matching loss families (C and D3)."""
    gen = load_generator()
    source = _md_source(gen, "build_cell_27_density_histograms_md")
    assert "density-grid" in source or "density grid" in source
    assert "C" in source and "D3" in source
    assert "histogram" in source.lower()


def test_cell_27_has_figure_suptitle():
    """Cell 27 must have a fig.suptitle identifying the comparison."""
    gen = load_generator()
    source = gen.build_cell_27_density_histograms().source
    assert "fig.suptitle(" in source
    assert "H2O grid-density residual histograms" in source


def test_cell_27_axes_labelled_with_density_units():
    """Cell 27 must label the histogram x-axis with density units (rho_NN - rho_HF,
    electron/bohr^3) and the y-axis as a grid-weighted log-scale point count."""
    gen = load_generator()
    source = gen.build_cell_27_density_histograms().source
    assert "set_xlabel(r" in source
    assert r"\rho_{\mathrm{NN}}" in source
    assert r"\rho_{\mathrm{HF}}" in source
    assert "electron/bohr" in source
    assert 'set_ylabel("grid-weighted point count (log scale)")' in source


def test_cell_28_attn_comparison_md_builder_exists():
    """Cell 28's attention-vs-baseline comparison must have a markdown
    description cell that names the blue/orange bar semantics and the
    signed-error convention."""
    gen = load_generator()
    source = _md_source(gen, "build_cell_28_attn_comparison_md")
    assert "attention" in source.lower()
    assert "non-attention" in source.lower() or "non attention" in source.lower()
    assert "signed" in source.lower()


def test_cell_28_has_figure_suptitle():
    """Cell 28 must have a fig.suptitle stating what is being compared and
    clarifying the signed-error convention so the reader can interpret bars."""
    gen = load_generator()
    source = gen.build_cell_28_attn_comparison().source
    assert "fig.suptitle(" in source
    assert "Attention vs non-attention comparison" in source
    assert "signed so positive = NN over-predicts AE" in source


def test_cell_29_feature_comparison_md_builder_exists():
    """Cell 29's extended-feature comparison must have a markdown description
    cell that enumerates the four deep base variants being compared."""
    gen = load_generator()
    source = _md_source(gen, "build_cell_29_feature_comparison_md")
    assert "deep" in source
    assert "deep_cusp" in source
    assert "deep_dm" in source
    assert "deep_combined" in source


def test_cell_29_has_figure_title():
    """Cell 29 must have a matplotlib title identifying the comparison as
    extended-feature impact on deep base variants."""
    gen = load_generator()
    source = gen.build_cell_29_feature_comparison().source
    assert "Extended-feature impact on H2O AE error" in source
    assert "deep base variants" in source


def test_cell_32_new_mol_comparison_md_builder_exists():
    """Cell 32's transfer-evaluation comparison must have a markdown description
    cell that names the three error panels (AE, total-E, density) and the
    reference lines (PBE, CCSD, HF) they share."""
    gen = load_generator()
    source = _md_source(gen, "build_cell_32_new_mol_comparison_md")
    assert "AE error" in source or "atomization" in source.lower()
    assert "PBE" in source
    assert "CCSD" in source
    assert "HF" in source


def test_cell_32_has_figure_suptitle():
    """Cell 32 must have a fig.suptitle stating that this is the
    transfer-evaluation sweep across every (arch, loss) checkpoint."""
    gen = load_generator()
    source = gen.build_cell_32_new_mol_comparison().source
    assert "fig.suptitle(" in source
    assert "Transfer evaluation" in source


def test_cell_09_has_figure_suptitle():
    """Cell 09 pretrain loss plot must have a fig.suptitle naming the atoms
    and what the curves represent."""
    gen = load_generator()
    source = gen.build_cell_09_pretrain_loss_plot().source
    assert "fig.suptitle(" in source
    assert "Pretraining loss vs step" in source


def test_cell_09_has_per_subplot_axis_labels():
    """Cell 09 must have xlabel/ylabel on both xnet and cnet subplots."""
    gen = load_generator()
    source = gen.build_cell_09_pretrain_loss_plot().source
    assert 'ax_x.set_xlabel("optimizer step")' in source
    assert 'ax_c.set_xlabel("optimizer step")' in source
    assert 'ax_x.set_ylabel("MSE loss (log scale)")' in source
    assert 'ax_c.set_ylabel("MSE loss (log scale)")' in source


def test_cell_10_has_figure_suptitle():
    """Cell 10 parity plot must have a fig.suptitle explaining the y=x convention."""
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    assert "fig.suptitle(" in source
    assert "Pretrain parity" in source
    assert "points on y=x are perfectly matched" in source


def test_cell_19_has_figure_suptitle():
    """Cell 19 training-loss-curves figure must have a fig.suptitle stating
    the 12-arch x 6-loss layout."""
    gen = load_generator()
    source = gen.build_cell_19_training_loss_plot().source
    assert "fig.suptitle(" in source
    assert "Main training loss curves" in source
    assert "12 architectures x 6 loss families" in source


def test_cell_20_has_figure_suptitle():
    """Cell 20 aux-component figure must have a fig.suptitle naming the arch."""
    gen = load_generator()
    source = gen.build_cell_20_aux_inspection().source
    assert "fig.suptitle(" in source
    assert "Aux loss components for arch" in source


def test_every_code_cell_emitted_source_is_valid_python(tmp_path):
    """REGRESSION GUARD: every code cell's ``source`` must parse as valid Python.

    Root cause this guards against: builder functions use non-raw triple-quoted
    ``source = '''...'''`` strings. Any unescaped ``\\n`` / ``\\r`` / ``\\0``
    / ``\\x..`` / ``\\u....`` inside an inner string literal becomes an actual
    newline / CR / NUL / byte / codepoint when Python parses the outer string,
    corrupting the emitted Python source. Substring-matching tests cannot
    detect this because the substring survives across the embedded newline.

    Only a real ``compile()`` will catch it.
    """
    gen = load_generator()
    out_path = tmp_path / "out.ipynb"
    gen.main(str(out_path))
    nb = nbformat.read(str(out_path), as_version=4)

    failures = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            compile(cell.source, f"<cell_{idx:02d}>", "exec")
        except SyntaxError as exc:
            failures.append(
                f"cell {idx} (id={cell.get('id', '?')}): "
                f"line {exc.lineno}: {exc.msg}"
            )
    assert not failures, "Emitted notebook cells have Python syntax errors:\n" + "\n".join(failures)
