"""Structural tests for the step-7 notebook builder.

The builder at ``notebooks/_build_step7_notebook.py`` is loaded via
``importlib.util.spec_from_file_location`` (``notebooks/`` is intentionally not a
package, mirroring the step 4/5/6 notebook tests) and its ``build_cells()`` is
called directly. Every test therefore runs the real builder and inspects the
freshly emitted cells; a broken builder raises at load/build time and fails the
test rather than falling through to a committed, possibly-stale ``.ipynb`` on
disk.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO_ROOT / "notebooks" / "_build_step7_notebook.py"


def load_builder():
    """Load ``_build_step7_notebook`` via the spec loader (``notebooks/`` is not
    a package). Any import-time error raises here and fails the test."""
    if not BUILDER_PATH.is_file():
        pytest.fail(f"step-7 notebook builder not found at {BUILDER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "step7_builder", str(BUILDER_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_source(cell):
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _code_cells(cells):
    return [c for c in cells if c.get("cell_type") == "code"]


def _markdown_cells(cells):
    return [c for c in cells if c.get("cell_type") == "markdown"]


def _assert_all_code_cells_parse(cells):
    """Every emitted code cell must be valid Python. Guards against a builder
    change that emits a syntactically broken cell (the notebook would still open
    in Jupyter but every run would crash at that cell)."""
    for i, cell in enumerate(_code_cells(cells)):
        src = _cell_source(cell)
        if not src.strip():
            continue
        try:
            ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(f"code cell {i} fails to parse: {exc}")


def _find_code_cell_source(cells, needle):
    """Return the source of the first code cell containing ``needle``."""
    for cell in _code_cells(cells):
        src = _cell_source(cell)
        if needle in src:
            return src
    pytest.fail(f"no emitted code cell contains {needle!r}")


def _int_literals(node):
    """Collect every integer literal (including negatives written as a unary
    minus) reachable from ``node``; bool constants are excluded."""
    values = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub)
                and isinstance(n.operand, ast.Constant)
                and isinstance(n.operand.value, int)
                and not isinstance(n.operand.value, bool)):
            values.add(-n.operand.value)
        elif (isinstance(n, ast.Constant) and isinstance(n.value, int)
              and not isinstance(n.value, bool)):
            values.add(n.value)
    return values


def test_step7_notebook_builder_emits_override_markdown_cell():
    """The builder emits a markdown cell describing the per-species OEP override
    cascade (spec sec. 8). Built fresh from ``build_cells()`` -- never read from
    a committed notebook -- so a builder that drops the cell fails here."""
    cells = load_builder().build_cells()
    matches = [c for c in _markdown_cells(cells)
               if "Per-species OEP cascade overrides" in _cell_source(c)]
    assert len(matches) >= 1, (
        "builder no longer emits the per-species-OEP-overrides markdown cell"
    )


def test_step7_notebook_oom_detection_handles_sigkill_exit_code():
    """Regression pin (2026-05-07): the emitted ``_looks_like_gpu_oom`` helper
    takes an ``rc`` argument and treats the SIGKILL (-9 / 137) and SIGABRT
    (-6 / 134) exit codes as OOM evidence regardless of captured stderr. The OS
    OOM-killer dispatches SIGKILL with no time for the process to print
    JAX/CUDA OOM markers, so a marker-only check would leave
    ``_run_training_isolated`` raising 'CPU retry not attempted' on every
    kernel-OOM kill instead of falling back to CPU.

    Checked structurally (AST), not by an exact-literal substring, so a
    behavior-preserving refactor (reordering the exit codes, set-vs-tuple) does
    not spuriously break the pin."""
    cells = load_builder().build_cells()
    _assert_all_code_cells_parse(cells)

    src = _find_code_cell_source(cells, "def _looks_like_gpu_oom")
    tree = ast.parse(src)

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "_looks_like_gpu_oom"), None)
    assert fn is not None, "emitted cell must define _looks_like_gpu_oom"
    # New signature: a trailing rc parameter defaulting to None.
    assert fn.args.args and fn.args.args[-1].arg == "rc", (
        "_looks_like_gpu_oom must take a trailing rc parameter"
    )
    assert fn.args.defaults, "rc must have a default"
    last_default = fn.args.defaults[-1]
    assert isinstance(last_default, ast.Constant) and last_default.value is None, (
        "rc must default to None"
    )
    # SIGKILL (-9 / 137) and SIGABRT (-6 / 134, C++ std::bad_alloc -> abort())
    # all recognized by exit code alone.
    codes = _int_literals(fn)
    assert {-9, 137, -6, 134}.issubset(codes), (
        f"exit-code OOM branch must cover -9/137/-6/134; found {sorted(codes)}"
    )

    # Host (CPU) OOM markers assigned in the same cell -- a large-basis XLA/LLVM
    # compile OOM prints std::bad_alloc.
    marker_strs = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_CPU_OOM_MARKERS"
                        for t in node.targets)):
            marker_strs = {c.value for c in ast.walk(node.value)
                           if isinstance(c, ast.Constant)
                           and isinstance(c.value, str)}
    assert "std::bad_alloc" in marker_strs, (
        "_CPU_OOM_MARKERS must include the std::bad_alloc host-OOM marker"
    )

    # The isolated-training path threads the captured exit code through via an
    # ``rc=`` keyword so the exit-code branch can fire.
    passes_rc = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_looks_like_gpu_oom"
        and any(kw.arg == "rc" for kw in node.keywords)
        for node in ast.walk(tree)
    )
    assert passes_rc, (
        "_looks_like_gpu_oom must be called with rc= so the exit-code branch "
        "fires on a marker-less SIGKILL/SIGABRT"
    )


def test_step7_spec_builder_excludes_bh76_compounds_from_ae_channel():
    """Regression pin (2026-05-10): mixed-pool spec assembly must derive
    ``aux_only_names`` from the polyatomic species that did NOT come from an AE
    TrainingPoint (i.e. BH76 reactant/product compounds like HO, CH3, CH4, N2,
    N2O, F2). Without this, ``_ae_losses`` includes those species in
    ``compound_idx`` with target=0.0, the relative-error denominator collapses
    to (0**2 + 1e-8) = 1e-8, and a ~0.5 Ha NN-vs-anchor AE prediction blows up
    to ~2.5e+7, driving the trained NN to make ``compound energy =
    sum-of-atom-energies`` for those compounds -- an unphysical objective.

    bin01 (single AE point) trained correctly through this bug because no BH76
    species were chosen. bin02+ specs with BH76 reactions all learned the wrong
    objective.

    Checked structurally (AST) so a formatting-only refactor does not break the
    pin."""
    cells = load_builder().build_cells()
    _assert_all_code_cells_parse(cells)

    src = _find_code_cell_source(cells, "_aux_polyatomic_names")
    tree = ast.parse(src)

    # (1) The spec builder derives the _aux_polyatomic_names tuple.
    assigned = {t.id for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    assert "_aux_polyatomic_names" in assigned, (
        "spec-builder cell missing the _aux_polyatomic_names derivation "
        "(BH76 species would otherwise pollute the AE channel)"
    )

    # (2) The derivation excludes AE-reference compounds via a
    #     `ms.name not in _ae_ref_kcalmol` membership test.
    def _is_ms_name(node):
        return (isinstance(node, ast.Attribute) and node.attr == "name"
                and isinstance(node.value, ast.Name)
                and node.value.id == "ms")

    excludes_ae = any(
        isinstance(node, ast.Compare)
        and _is_ms_name(node.left)
        and any(isinstance(op, ast.NotIn) for op in node.ops)
        and any(isinstance(cmp, ast.Name) and cmp.id == "_ae_ref_kcalmol"
                for cmp in node.comparators)
        for node in ast.walk(tree)
    )
    assert excludes_ae, (
        "_aux_polyatomic_names derivation must exclude AE-reference compounds "
        "(ms.name not in _ae_ref_kcalmol); otherwise BH76 species (without AE "
        "targets) would be misclassified as AE compounds"
    )

    # (3) It is wired into loss_kwargs as aux_only_names.
    wired = any(
        isinstance(node, ast.Dict)
        and any(isinstance(k, ast.Constant) and k.value == "aux_only_names"
                and isinstance(v, ast.Name) and v.id == "_aux_polyatomic_names"
                for k, v in zip(node.keys, node.values) if k is not None)
        for node in ast.walk(tree)
    )
    assert wired, (
        "loss_kwargs missing aux_only_names -> _aux_polyatomic_names; BH76 "
        "compounds would enter the AE channel with target=0.0 placeholders"
    )
