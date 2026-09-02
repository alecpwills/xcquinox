"""Tests for xcquinox.alec.cluster.fidelity -- the per-architecture physics
certificate.

The cheap layer (the certificate predicate, the parent resolution, the oracle
set) is tested directly. The certificate itself is tested three ways: with the
per-system evaluation monkeypatched at the ``evaluate`` seam, so the verdict
arithmetic and the JSON schema are exercised with no SCF at all; for REAL on H
and H2 at sto-3g with networks built in the test, so the energy path, the three
parent routes and the atomization fold are pinned against physics; and with the
exact parent functional (PBE, SCAN through libxc) presented behind the model
interface on O, H and H2O, so the whole path is shown to be an identity when the
network is its parent and to report a known per-electron offset exactly.
"""
import itertools
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.alec.cluster import fidelity as fid


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _cfg(arch=("deep_3x16",), basis="sto-3g", grid_level=1,
         tol_AE=1.0, tol_atom=1.0, override_reason=None, enforce=True,
         polarized=False, pretrain_seed=42):
    """The attribute surface fidelity reads off a GridConfig."""
    return SimpleNamespace(
        sweep=SimpleNamespace(arch=tuple(arch)),
        inputs=SimpleNamespace(basis=basis, grid_level=grid_level,
                               density_fit=False, auxbasis=None,
                               orientation_lock_strength=0.0),
        pretrain=SimpleNamespace(seed=pretrain_seed),
        fidelity=SimpleNamespace(tol_AE=tol_AE, tol_atom=tol_atom,
                                 override_reason=override_reason,
                                 enforce=enforce),
        use_polarized_correlation=polarized,
    )


def _write_certificate(run_dir, arch, verdict="PASS", **extra):
    """Write a certificate; ``extra`` sets any schema key (``enforced``,
    ``summary``, ``tolerances``, ...)."""
    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    payload = {"verdict": verdict, "arch": arch}
    payload.update(extra)
    with open(os.path.join(d, fid.CERTIFICATE_FILENAME), "w") as f:
        json.dump(payload, f)
    return d


def _write_payload(run_dir, arch, payload):
    """Write an arbitrary certificate payload, including one that carries no
    ``verdict`` key at all."""
    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, fid.CERTIFICATE_FILENAME), "w") as f:
        json.dump(payload, f)
    return d


# ---------------------------------------------------------------------------
# Import weight: the module body stays a pure reader
# ---------------------------------------------------------------------------

# The stacks a certificate reader must never pull in, and the only xcquinox
# modules cheap enough for the module body: grid_config supplies the run
# layout, domain the kcal/mol conversion, materialize the atomic JSON writer
# the certificate writer uses. Each is stdlib-only in its own body.
_HEAVY_IMPORT_ROOTS = frozenset({
    "jax", "jaxlib", "equinox", "optax", "numpy", "scipy", "pyscf", "pyscfad",
    "ase", "pandas", "matplotlib", "torch", "h5py",
})
_CHEAP_XCQ_MODULES = frozenset({
    "xcquinox.alec.cluster.grid_config",
    "xcquinox.alec.cluster.domain",
    "xcquinox.alec.cluster.materialize",
    # The calibrated orientation-lock strength, and nothing else: the harness
    # parser's default reads it in its module body, and it must not drag
    # ``orientation_lock``'s numpy in behind it.
    "xcquinox.alec.orientation_lock_default",
    # The shared hard exit, reached only from this module's
    # ``if __name__ == "__main__"`` block, which an import never runs. It is
    # stdlib-only by construction, and whitelisting it rather than exempting
    # the block keeps the walk transitive: a heavy import added to the helper
    # would still be caught here.
    "xcquinox.alec.cluster._exit",
})

#: Whitelisted modules the SOURCE walk sees but an import never binds, because
#: the only statement importing them is the ``if __name__ == "__main__"``
#: block. The two tests below therefore treat them oppositely on purpose: the
#: walk accepts the name and recurses into it, and the closure test requires
#: the name to be ABSENT from ``sys.modules`` after the body has run -- which
#: is the measurement that the entry block really is off the import path.
_ENTRY_BLOCK_ONLY_MODULES = frozenset({"xcquinox.alec.cluster._exit"})

# Upper bound on the modules present in sys.modules after the file is executed
# with the package __init__ modules stubbed (the closure test below). The
# committed tree measures 123, of which 78 are the interpreter's own startup
# set. A module-body ``import pyscf`` measures 799 (numpy, scipy, pyscf) and an
# ``import jax`` in grid_config's body measures 612 (numpy, jax, jaxlib), so any
# bound between 123 and 612 discriminates; 300 leaves the readers room to grow a
# stdlib import without a test edit.
_CLOSURE_MODULE_BOUND = 300


def _module_body_imports(path, package):
    """Absolute module names imported by ONE file's module body, at any
    statement depth.

    The walk descends through every compound statement -- ``try``, ``if``,
    ``with``, ``for``, ``while`` -- because an import nested in one of those
    executes on import exactly as a top-level one does, while iterating
    ``tree.body`` alone cannot see it: ``if True:\\n    import pyscf`` in
    fidelity.py's body left the depth-1 form of this test passing with the
    module's measured closure at 799 modules.

    Function, async-function, class and lambda subtrees are PRUNED. A
    function-local import is what the contract asks for, so it must not
    register; a class body does execute on import, and is caught by the
    closure measurement below rather than by this walk.

    Returns ``(imports, parsed_to_nothing)``. The second value separates a
    module with no imports -- which a constants-only leaf legitimately is --
    from a file the walk failed to read.
    """
    import ast
    assert path is not None, (
        "no source file to read for a module body in package %r" % (package,))
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = []
    empty = not tree.body
    stack = list(reversed(tree.body))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                out.append(base + "." + node.module if node.module else base)
            else:
                out.append(node.module or "")
        else:
            stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return out, empty


def test_fidelity_module_body_carries_no_heavy_import():
    """`cluster status`, validate_run, merge and the train task's PARENT
    process all read certificates. fidelity.py's module BODY must therefore
    stay a pure reader -- path helpers, the certificate predicates, constants
    -- with every jax / pyscf / xcquinox.alec.data import inside a function,
    so a reader never triggers a model import or an SCF-capable stack it does
    not use. The PROHIBITION is what is pinned, not a frozen import list: the
    node entry point adds whatever stdlib modules it needs without touching
    this test, while no numeric or model stack may appear in the body and the
    only xcquinox modules allowed there are the cheap cluster readers.

    The walk is TRANSITIVE over those readers' own module bodies. A whitelist
    checked at depth 1 forbids nothing: `import domain` would satisfy it while
    domain's body pulled ASE, and this file would load the chain it promises
    not to. A name is therefore admitted only while its own body stays within
    the same prohibition.

    It is also transitive in STATEMENT depth: an import nested in a ``try``,
    an ``if`` or a ``with`` at module level runs on import, so it registers
    here as a top-level one does, while imports inside a function body do not.

    Checked on the source; the companion test below measures the same
    prohibition on the bindings, which is what catches an import this walk
    cannot see in the source (a class body) or reaches through a path it does
    not resolve."""
    import importlib.util
    import inspect
    queue = [("xcquinox.alec.cluster.fidelity", inspect.getsourcefile(fid))]
    seen = set()
    closure = []
    while queue:
        name, path = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        package = name.rsplit(".", 1)[0]
        imports, empty = _module_body_imports(path, package)
        # A module that parses to NOTHING is not the module the walk names --
        # a wrong path, or a file the loader never wrote. A module that parses
        # to statements but no imports is legitimate: the lock constant's leaf
        # home is exactly that, and is why the check is emptiness rather than
        # the absence of imports.
        assert not empty, (
            "%s (%s) parses to an empty module body; the file measured is "
            "not the module the walk names" % (name, path))
        for imported in imports:
            closure.append(imported)
            root = imported.split(".")[0]
            assert root not in _HEAVY_IMPORT_ROOTS, (name, imported)
            if root == "xcquinox":
                assert imported in _CHEAP_XCQ_MODULES, (name, imported)
                spec = importlib.util.find_spec(imported)
                assert spec is not None, (
                    "%s imports %s, which resolves to no module at all"
                    % (name, imported))
                assert spec.origin is not None, (
                    "%s imports %s, which resolves to a namespace package "
                    "with no source file, so the transitive walk cannot read "
                    "its body" % (name, imported))
                queue.append((imported, spec.origin))
    for deferred in ("xcquinox.alec.rungs", "xcquinox.alec.config",
                     "xcquinox.alec.full_benchmark_pools",
                     "xcquinox.alec.dfs_pretrain_set",
                     "xcquinox.alec.training_points"):
        assert deferred not in closure, deferred


def test_fidelity_module_body_loads_no_heavy_stack_when_executed():
    """The same prohibition measured on the BINDINGS rather than the source.

    The walk above reads statements; this one executes the file and counts
    what reaches ``sys.modules``, so a heavy import is caught whatever shape
    it was written in -- nested in a ``try`` or an ``if``, run from a class
    body, or pulled in by one of the cheap readers the walk whitelists.

    Measured with the three package ``__init__`` modules stubbed. Importing
    ``xcquinox.alec.cluster`` normally executes the package's own jax-carrying
    ``__init__``, which would load the whole stack before this module's body
    ran and mask its cost entirely; the stubs keep the real package
    ``__path__``, so ``domain``, ``grid_config`` and ``materialize`` still
    resolve to the real files and their bodies execute here too.

    The committed tree measures 123 modules loaded, of which 78 are the
    interpreter's own startup set, and pulls exactly the four cheap readers
    out of ``xcquinox`` (``domain``, ``grid_config``, ``materialize`` and
    ``orientation_lock_default``) -- so the deferred model, config and pool
    modules are pinned absent by binding and not only by name.
    """
    path = os.path.abspath(fid.__file__)
    cluster_dir = os.path.dirname(path)
    alec_dir = os.path.dirname(cluster_dir)
    xcq_dir = os.path.dirname(alec_dir)
    probe = """
import importlib.util, json, sys, types
paths = {"xcquinox": %r, "xcquinox.alec": %r, "xcquinox.alec.cluster": %r}
for name, path in paths.items():
    stub = types.ModuleType(name)
    stub.__path__ = [path]
    sys.modules[name] = stub
base = len(sys.modules)
spec = importlib.util.spec_from_file_location(
    "xcquinox.alec.cluster.fidelity", %r)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps({"base": base, "after": len(sys.modules),
                  "modules": sorted(sys.modules),
                  "filename": module.CERTIFICATE_FILENAME,
                  "ha_to_kcal": module.HA_TO_KCAL}))
""" % (xcq_dir, alec_dir, cluster_dir, path)
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    loaded = set(result["modules"])
    heavy = sorted(root for root in _HEAVY_IMPORT_ROOTS if root in loaded)
    assert not heavy, (heavy, result["after"])
    assert result["after"] < _CLOSURE_MODULE_BOUND, (
        result["after"], result["base"])
    # Exactly the four whitelisted readers, and the stubs the probe installed
    # itself: no deferred xcquinox module is reached through any route. The
    # entry-block-only names are subtracted rather than tolerated, so their
    # ABSENCE here is asserted: an import of this module must not reach them.
    assert {m for m in loaded if m.split(".")[0] == "xcquinox"} == (
        (set(_CHEAP_XCQ_MODULES) - _ENTRY_BLOCK_ONLY_MODULES)
        | {"xcquinox", "xcquinox.alec", "xcquinox.alec.cluster",
           "xcquinox.alec.cluster.fidelity"})
    # The module really executed under the stubs, so the counts above are the
    # cost of a complete import and not of a failed one.
    assert result["filename"] == fid.CERTIFICATE_FILENAME
    assert result["ha_to_kcal"] == fid.HA_TO_KCAL


def test_fidelity_imports_in_a_fresh_interpreter():
    """No import cycle: _pretrain imports fidelity, fidelity must not import
    _pretrain (it derives the distinct-arch list from _canon_axis instead)."""
    out = subprocess.run(
        [sys.executable, "-c", "import xcquinox.alec.cluster.fidelity"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------------------
# The one predicate every enforcement site calls
# ---------------------------------------------------------------------------

def test_certificate_status_missing(tmp_path):
    status, reason = fid.certificate_status_in(str(tmp_path))
    assert status == "MISSING"
    assert fid.CERTIFICATE_FILENAME in reason


def test_certificate_status_pass(tmp_path):
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="PASS")
    assert fid.certificate_status_in(d) == ("PASS", "fidelity certificate PASS")


def test_certificate_status_fail_carries_the_summary(tmp_path):
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                           summary={"max_atom_mHa": 13.7,
                                    "max_dAE_kcalmol": 25.7})
    status, reason = fid.certificate_status_in(d)
    assert status == "FAIL"
    assert "13.7" in reason and "25.7" in reason


def test_certificate_status_unreadable(tmp_path):
    d = tmp_path / "pretrain" / "deep_3x16"
    d.mkdir(parents=True)
    (d / fid.CERTIFICATE_FILENAME).write_text("{not json")
    status, reason = fid.certificate_status_in(str(d))
    assert status == "UNREADABLE"
    assert "JSON" in reason


_UNACTIONABLE_VERDICTS = [
    pytest.param({"arch": "deep_3x16"}, id="no-verdict-key"),
    pytest.param({"arch": "deep_3x16", "verdict": "pass"}, id="wrong-case"),
    pytest.param({"arch": "deep_3x16", "verdict": None}, id="null"),
    pytest.param({"arch": "deep_3x16", "verdict": 1}, id="integer"),
]


@pytest.mark.parametrize("payload", _UNACTIONABLE_VERDICTS)
def test_an_absent_or_unrecognised_verdict_is_unreadable(tmp_path, payload):
    """A certificate that records no verdict, or one the module does not
    recognise, states no outcome that can be acted on. Classifying it FAIL
    would make it waivable by ``enforced: false``, so a truncated or
    schema-less file would release an on-node gate; UNREADABLE never is."""
    d = _write_payload(str(tmp_path), "deep_3x16", payload)
    status, reason = fid.certificate_status_in(d)
    assert status == "UNREADABLE", payload
    assert status != fid.VERDICT_PASS
    assert "verdict" in reason
    assert fid.certificate_status(str(tmp_path), "deep_3x16")[0] != \
        fid.VERDICT_PASS


@pytest.mark.parametrize("payload", _UNACTIONABLE_VERDICTS)
def test_an_absent_or_unrecognised_verdict_never_releases_the_gate(tmp_path,
                                                                   payload):
    """The same certificates, carrying the fullest waiver a run can write:
    the gate still refuses, because only a recognised FAIL is waivable."""
    payload = dict(payload, enforced=False,
                   tolerances={"tol_AE": 1.0, "tol_atom": 1.0,
                               "override_reason": "workflow matrix"})
    _write_payload(str(tmp_path), "deep_3x16", payload)
    allowed, _message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False, payload


def test_certificate_status_by_run_dir_and_arch_uses_the_harness_layout(tmp_path):
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    _write_certificate(str(tmp_path), "deep_3x16")
    assert fid.certificate_path(str(tmp_path), "deep_3x16") == os.path.join(
        pretrain_checkpoint_dir(str(tmp_path), "deep_3x16"),
        fid.CERTIFICATE_FILENAME)
    assert fid.certificate_status(str(tmp_path), "deep_3x16")[0] == "PASS"


def test_read_certificate_returns_none_when_absent(tmp_path):
    assert fid.read_certificate(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# The ON-NODE gate: fidelity.enforce = False records the verdict and continues
# ---------------------------------------------------------------------------

def test_certificate_enforced_defaults_to_true_when_the_field_is_absent(
        tmp_path):
    """A certificate that does not say otherwise is enforcing; so is an
    absent one, which cannot say anything at all."""
    d = _write_certificate(str(tmp_path), "deep_3x16")
    assert fid.certificate_enforced_in(d) is True
    assert fid.certificate_enforced_in(str(tmp_path / "nowhere")) is True


def test_certificate_enforced_reads_the_recorded_flag(tmp_path):
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                           enforced=False)
    assert fid.certificate_enforced_in(d) is False


@pytest.mark.parametrize("extra, enforced", [
    pytest.param({}, True, id="absent"),
    pytest.param({"enforced": None}, True, id="null"),
    pytest.param({"enforced": True}, True, id="true"),
    pytest.param({"enforced": False}, False, id="false"),
    pytest.param({"enforced": "false"}, True, id="string"),
    pytest.param({"enforced": 0}, True, id="integer"),
])
def test_only_an_explicit_false_waives_enforcement(tmp_path, extra, enforced):
    """The waiver is the JSON literal ``false`` and nothing else. A truthiness
    test reads ``null`` -- the value a certificate written without the field
    populated carries -- as a waiver, and would release a FAIL on a node with
    no run ever having asked for it."""
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL", **extra)
    assert fid.certificate_enforced_in(d) is enforced


def test_the_gate_refuses_a_failure_whose_enforced_flag_is_null(tmp_path):
    """The on-node consequence of the rule above, with a complete waiver
    record otherwise: ``enforced: null`` is not a waiver."""
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=None,
                       tolerances={"override_reason": "workflow matrix"},
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7})
    allowed, _message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False


def test_gate_allows_a_passing_certificate(tmp_path):
    _write_certificate(str(tmp_path), "deep_3x16", verdict="PASS")
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is True
    assert "PASS" in message


def test_gate_refuses_an_enforced_failure(tmp_path):
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=True,
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7})
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False
    assert "13.7" in message


def test_gate_allows_a_recorded_failure_when_enforcement_is_off(tmp_path):
    """The Section 3.4 workflow matrix: a 50-step pretrain cannot meet the
    tolerance, but train and eval must still be exercised end to end with the
    real verdict on the record."""
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=False,
                       tolerances={"tol_AE": 1.0, "tol_atom": 1.0,
                                   "override_reason": "workflow matrix"},
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7})
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is True
    assert "enforcement is OFF" in message
    assert "workflow matrix" in message


@pytest.mark.parametrize("tolerances", [
    pytest.param(None, id="no-tolerances-block"),
    pytest.param({"tol_AE": 1.0, "tol_atom": 1.0}, id="no-reason-field"),
    pytest.param({"override_reason": None}, id="null"),
    pytest.param({"override_reason": ""}, id="empty"),
    pytest.param({"override_reason": "   "}, id="whitespace"),
])
def test_the_gate_refuses_a_waiver_that_records_no_reason(tmp_path,
                                                          tolerances):
    """``validate_grid_semantics`` refuses ``fidelity.enforce: false`` without
    a non-empty ``fidelity.override_reason``. The on-node gate imposes the
    same invariant on the certificate itself, so a hand-edited certificate or
    resolved_config.yaml on a node cannot release a FAIL with no reason on the
    record."""
    extra = {} if tolerances is None else {"tolerances": tolerances}
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=False,
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7},
                       **extra)
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False, tolerances
    assert "override_reason" in message


@pytest.mark.parametrize("recorded", [
    pytest.param(False, id="false"),
    pytest.param(True, id="true"),
    pytest.param(0, id="zero"),
    pytest.param(7, id="seven"),
    pytest.param([], id="empty-list"),
    pytest.param({}, id="empty-dict"),
])
def test_the_gate_refuses_a_waiver_whose_reason_is_not_prose(tmp_path,
                                                             recorded):
    """The reason is prose or it is nothing. ``grid_config._build_fidelity``
    refuses a non-string ``fidelity.override_reason`` rather than coercing it,
    because ``str(False)`` is the non-empty string 'False' and a boolean, a
    number or a container would then satisfy a strip-only test and authorise
    disabled gates that no author asked for. The certificate carries whatever
    was recorded, so the gate applies the same rule to the file it reads."""
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=False,
                       tolerances={"override_reason": recorded},
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7})
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False, recorded
    assert "override_reason" in message


def test_a_waiver_reason_is_accepted_with_surrounding_whitespace(tmp_path):
    """Emptiness is judged on the stripped string, so a reason a human typed
    with padding is a reason; the refusal above is for the empty ones only."""
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=False,
                       tolerances={"override_reason": "  workflow matrix  "},
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7})
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is True
    assert "workflow matrix" in message


def test_gate_never_allows_a_missing_certificate(tmp_path):
    """Enforcement can only be waived by a certificate that exists to record
    the waiver; an absent one waives nothing."""
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False
    assert "MISSING" in message or "was never checked" in message


def test_gate_never_allows_an_unreadable_certificate(tmp_path):
    d = tmp_path / "pretrain" / "deep_3x16"
    d.mkdir(parents=True)
    (d / fid.CERTIFICATE_FILENAME).write_text("{truncated")
    allowed, _message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False


# ---------------------------------------------------------------------------
# One document per decision
# ---------------------------------------------------------------------------

def _serve_documents(monkeypatch, path, documents):
    """Serve ``documents`` to successive READ opens of ``path``.

    The list returned collects one entry per read served, so a caller can
    state how many parses a decision rested on. Writes and every other path
    are passed through; once the list is exhausted its last entry repeats, so
    a caller that reads more often than the sequence is long is handed a
    complete document rather than an empty file.
    """
    import builtins
    import io
    real_open = builtins.open
    served: list = []

    def fake_open(file, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else "r")
        if str(file) == str(path) and "r" in mode:
            doc = documents[min(len(served), len(documents) - 1)]
            served.append(doc)
            return io.StringIO(doc if isinstance(doc, str)
                               else json.dumps(doc))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return served


# Three FAIL documents, each refused on its own: D1 records no waiver, D2
# records one that states no reason, D3 states a reason beside enforcement
# that is ON. A gate that classifies one document and reads the waiver off
# another releases on D1 -> D2 -> D3, which is a release granted to a
# certificate that never existed.
_D1 = {"verdict": "FAIL",
       "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7}}
_D2 = {"verdict": "FAIL", "enforced": False}
_D3 = {"verdict": "FAIL", "enforced": True,
       "tolerances": {"override_reason": "workflow matrix"}}


def test_the_gate_reads_the_certificate_once(tmp_path, monkeypatch):
    """One parse per gate decision, on the deepest path there is.

    The waiver path is the one that classifies, then asks whether enforcement
    is on, then reads the reason -- three chances for three documents to
    contribute to one release. The release and the message it states must
    come from a single document.
    """
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                           enforced=False,
                           tolerances={"override_reason": "workflow matrix"},
                           summary={"max_atom_mHa": 13.7,
                                    "max_dAE_kcalmol": 25.7})
    with open(os.path.join(d, fid.CERTIFICATE_FILENAME)) as f:
        document = f.read()
    served = _serve_documents(monkeypatch,
                              os.path.join(d, fid.CERTIFICATE_FILENAME),
                              [document])
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    monkeypatch.undo()
    assert allowed is True, message
    assert "workflow matrix" in message
    assert len(served) == 1, served


def test_no_document_sequence_releases_the_gate(tmp_path, monkeypatch):
    """A release corresponds to a document, not to a sequence of them.

    Each of the three documents below is refused on its own. Serving them to
    successive opens of the same path -- the state a certificate rewritten
    while the gate runs presents -- must not assemble a release out of the
    waiver of one, the reason of another and the verdict of a third.
    """
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL")
    path = os.path.join(d, fid.CERTIFICATE_FILENAME)
    for doc in (_D1, _D2, _D3):
        _serve_documents(monkeypatch, path, [doc])
        allowed, _message = fid.gate_certificate(str(tmp_path), "deep_3x16")
        monkeypatch.undo()
        assert allowed is False, doc
    for order in itertools.permutations((_D1, _D2, _D3)):
        served = _serve_documents(monkeypatch, path, list(order))
        allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
        monkeypatch.undo()
        assert allowed is False, (order, message)
        assert len(served) == 1, (order, served)


def test_the_gate_rule_applied_to_an_already_read_document(tmp_path):
    """The release rule, exposed for a caller that has already read the file.

    The worker reports the numbers off the certificate and gates on the same
    document; it takes the rule rather than the path so that its line and its
    exit code cannot describe two files.
    """
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                           enforced=False,
                           tolerances={"override_reason": "workflow matrix"},
                           summary={"max_atom_mHa": 13.7,
                                    "max_dAE_kcalmol": 25.7})
    read = fid.read_certificate_status_in(d)
    assert fid.gate_certificate_from_read(*read) == fid.gate_certificate(
        str(tmp_path), "deep_3x16")


# ---------------------------------------------------------------------------
# Parent resolution: the arch's RUNG picks the parent, not inputs.seed_xc
# ---------------------------------------------------------------------------

def test_parent_is_pbe_for_gga_rung_and_scan_for_meta_gga():
    assert fid.resolve_parent("deep_3x16") == "pbe"
    assert fid.resolve_parent("deep_cusp_3x16") == "pbe"
    assert fid.resolve_parent("deep_rung35_3x16") == "pbe"
    assert fid.resolve_parent("deep_mgga_3x16") == "scan"


def test_parent_agrees_with_the_rung_seed_policy():
    from xcquinox.alec.rungs import seed_xc_for_arch
    from xcquinox.alec.config import list_architectures
    for name in list_architectures():
        assert fid.resolve_parent(name) == seed_xc_for_arch(name)


def test_dfs_level_follows_the_parent():
    assert fid.dfs_level_for_parent("pbe") == "gga"
    assert fid.dfs_level_for_parent("scan") == "mgga"


def test_distinct_archs_matches_the_pretrain_workers_selector():
    from xcquinox.alec.cluster import _pretrain as pt
    cfg = _cfg(arch=("medium", "deep", "medium", "shallow"))
    assert fid._distinct_archs(cfg) == ["deep", "medium", "shallow"]
    assert fid._distinct_archs(cfg) == pt._distinct_archs(cfg)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_ha_to_kcal_is_the_harness_conversion_constant():
    """The certificate reports atomization offsets in kcal/mol and the harness
    reports its benchmark errors in kcal/mol; both must divide by one number.
    The constant is therefore taken from the domain table rather than restated
    here, where a truncated copy would put the certificate's tolerance and the
    campaign's error metric on slightly different scales."""
    from xcquinox.alec.cluster.domain import KCAL_PER_HA
    assert fid.HA_TO_KCAL == KCAL_PER_HA
    assert fid.HA_TO_KCAL is KCAL_PER_HA
    assert fid.HA_TO_MHA == 1000.0


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

def test_run_identity_carries_the_five_scf_identity_fields():
    cfg = _cfg(basis="6-311++G(3df,2pd)", grid_level=3)
    cfg.inputs.density_fit = True
    cfg.inputs.auxbasis = "def2-universal-jkfit"
    cfg.inputs.orientation_lock_strength = 0.02
    assert fid.run_identity(cfg) == {
        "basis": "6-311++G(3df,2pd)", "grid_level": 3, "density_fit": True,
        "auxbasis": "def2-universal-jkfit",
        "orientation_lock_strength": 0.02}


# ---------------------------------------------------------------------------
# Oracle set
# ---------------------------------------------------------------------------

def test_atom_system_names_are_canonical():
    assert fid.atom_system_name("O", 0) == "atom_O"
    assert fid.atom_system_name("F", -1) == "atom_F-"
    assert fid.atom_system_name("Na", 1) == "atom_Na+"
    assert fid.atom_system_name("O", -2) == "atom_O-2"


def test_build_oracle_set_returns_a_tuple():
    """The declared return type. Callers iterate the set more than once (the
    certificate's per-system loop, then the atomization fold), so a generator
    would silently be empty the second time."""
    systems = fid.build_oracle_set(_cfg(), "deep_3x16")
    assert isinstance(systems, tuple)
    assert list(systems) == list(systems)


def test_oracle_set_puts_atoms_first_then_molecules_each_sorted():
    systems = fid.build_oracle_set(_cfg(), "deep_3x16")
    names = [ms.name for ms in systems]
    atoms = [n for n in names if n.startswith("atom_")]
    mols = [n for n in names if not n.startswith("atom_")]
    assert names == atoms + mols
    assert atoms == sorted(atoms)
    assert mols == sorted(mols)


def test_oracle_set_carries_every_pool_free_atom_with_its_pool_spin():
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    pool, _ = load_full_held_out_pools(basis="sto-3g", grid_level=1)
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    seen = 0
    for ms in pool.values():
        comp = tuple(ms.atom_composition)
        if len(comp) != 1 or int(comp[0][1]) != 1:
            continue
        seen += 1
        name = fid.atom_system_name(comp[0][0], ms.charge)
        assert name in systems, name
        assert systems[name].spin == ms.spin
        assert systems[name].charge == ms.charge
    assert seen >= 14


def test_oracle_set_carries_the_dfs_molecules_and_the_unconditional_three():
    systems = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    for name in ("H2", "LiF", "AlCl3", "C4H6", "SiH4"):
        assert name in systems
    for name in ("H2O", "N2", "CH4"):
        assert name in systems


def test_meta_gga_oracle_set_drops_h2_but_keeps_n2_among_the_unconditional():
    """The meta-GGA DFS variant omits H2 and N2; the three unconditional
    molecules carry N2 for every rung, so every architecture is measured on a
    common N2 / H2O / CH4 core whatever its rung."""
    gga = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    mgga = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_mgga_3x16")}
    assert "H2" in gga and "H2" not in mgga
    assert "N2" in gga and "N2" in mgga


def test_oracle_set_supplies_a_free_atom_for_every_element_it_dissociates():
    systems = fid.build_oracle_set(_cfg(), "deep_3x16")
    names = {ms.name for ms in systems}
    for ms in systems:
        if fid.is_atom_system(ms):
            continue
        for sym, _n in ms.atom_composition:
            assert fid.atom_system_name(sym, 0) in names, (ms.name, sym)


def test_oracle_set_adds_lithium_and_sodium_which_no_pool_carries():
    names = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    assert "atom_Li" in names and "atom_Na" in names


def test_an_unconditional_molecule_composition_counts_every_nucleus():
    """The atomization fold weights each free atom's offset by its count, so
    an unconditional molecule's composition carries counts, not just the
    element set."""
    gga = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    mgga = {ms.name: ms
            for ms in fid.build_oracle_set(_cfg(), "deep_mgga_3x16")}
    assert gga["H2O"].atom_composition == (("H", 2), ("O", 1))
    assert mgga["N2"].atom_composition == (("N", 2),)


def test_an_element_with_no_ground_state_spin_is_refused(monkeypatch):
    """A molecule whose element has no recorded ground-state spin has no
    atomization reference; the oracle set refuses it instead of silently
    dropping the free atom the fold needs."""
    import xcquinox.alec.dfs_pretrain_set as dfs_set
    records = dfs_set.dfs_pretrain_records("gga")
    records.append({"name": "KrH", "kind": "molecule",
                    "atom": "Kr 0.0 0.0 0.0; H 0.0 0.0 1.4",
                    "charge": 0, "spin": 1,
                    "atom_composition": [["H", 1], ["Kr", 1]]})
    monkeypatch.setattr(dfs_set, "dfs_pretrain_records",
                        lambda level: records)
    with pytest.raises(ValueError, match="ground-state spin"):
        fid.build_oracle_set(_cfg(), "deep_3x16")


def test_a_dfs_records_composition_is_sorted_into_its_spec(monkeypatch):
    """``dfs_pretrain_systems`` sorts the composition it reads from the file
    rather than trusting the file's order, because MoleculeSpec is frozen and
    hashes every field: two orderings of one molecule are two different specs
    and two precompute-cache entries. The oracle set is built from the same
    records and sorts identically, so the certificate and the pretraining
    pipeline cannot disagree about a molecule's identity."""
    import xcquinox.alec.dfs_pretrain_set as dfs_set
    records = dfs_set.dfs_pretrain_records("gga")
    records.append({"name": "OH_probe", "kind": "molecule",
                    "atom": "O 0.0 0.0 0.0; H 0.0 0.0 0.97",
                    "charge": 0, "spin": 1,
                    "atom_composition": [["O", 1], ["H", 1]]})
    monkeypatch.setattr(dfs_set, "dfs_pretrain_records",
                        lambda level: records)
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    assert systems["OH_probe"].atom_composition == (("H", 1), ("O", 1))


def test_sources_that_disagree_on_a_free_atom_spin_are_refused(monkeypatch):
    """The pools and the DFS set must declare one spin per free atom: folding
    a molecule against an atom evaluated at a different multiplicity would
    change the atomization offset with nothing in the certificate to show it."""
    import xcquinox.alec.dfs_pretrain_set as dfs_set
    records = dfs_set.dfs_pretrain_records("gga")
    for record in records:
        if record["name"] == "H":
            record["spin"] = 3          # the pools carry the H atom at 2S = 1
    monkeypatch.setattr(dfs_set, "dfs_pretrain_records",
                        lambda level: records)
    with pytest.raises(ValueError, match="exactly one spin"):
        fid.build_oracle_set(_cfg(), "deep_3x16")


def test_ground_state_spin_table_agrees_with_the_pool_spins():
    """The certificate's Hund ground-state table is the atomization reference;
    it must agree species by species with the spins the BH76 / W4-11 pools
    carry, or a molecule would be folded against a different atom than the
    benchmark uses."""
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    pool, _ = load_full_held_out_pools(basis="sto-3g", grid_level=1)
    for ms in pool.values():
        comp = tuple(ms.atom_composition)
        if len(comp) != 1 or int(comp[0][1]) != 1 or ms.charge != 0:
            continue
        sym = comp[0][0]
        assert fid._ATOM_GROUND_SPIN[sym] == ms.spin, sym


def test_oracle_set_specs_carry_the_run_identity():
    cfg = _cfg(basis="def2-tzvpd", grid_level=2)
    for ms in fid.build_oracle_set(cfg, "deep_3x16"):
        assert ms.basis == "def2-tzvpd"
        assert ms.grid_level == 2
        assert ms.external_data_path is None


def test_is_atom_system():
    from xcquinox.alec.config import MoleculeSpec
    atom = MoleculeSpec(name="atom_O", atom="O 0 0 0", basis="sto-3g", spin=2,
                        atom_composition=(("O", 1),))
    mol = MoleculeSpec(name="OH", atom="O 0 0 0; H 0 0 1", basis="sto-3g",
                       spin=1, atom_composition=(("H", 1), ("O", 1)))
    assert fid.is_atom_system(atom)
    assert not fid.is_atom_system(mol)


# ---------------------------------------------------------------------------
# The three unconditional molecules are ONE molecule for every rung
# ---------------------------------------------------------------------------

def test_unconditional_molecules_are_identical_across_rungs():
    """dAE(H2O), dAE(N2) and dAE(CH4) are the certificate's headline numbers
    and the ones spec Section 2 tabulates. They are comparable between a
    GGA-rung and a meta-GGA architecture only if both are computed on the SAME
    molecule, so all three come from the pools whatever the rung."""
    gga = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    mgga = {ms.name: ms
            for ms in fid.build_oracle_set(_cfg(), "deep_mgga_3x16")}
    for name in ("H2O", "N2", "CH4"):
        assert name in gga and name in mgga, name
        assert gga[name].atom == mgga[name].atom, name
        assert gga[name].spin == mgga[name].spin, name
        assert gga[name].charge == mgga[name].charge, name
        assert gga[name].atom_composition == mgga[name].atom_composition, name


def test_unconditional_molecules_carry_the_pool_geometry():
    """The pool species are the ones the held-out atomization energies are
    scored on, so the certificate measures the functional on the geometry the
    campaign reports."""
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    pool, _rxns = load_full_held_out_pools(basis="sto-3g", grid_level=1)
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    assert fid._FIXED_MOLECULE_POOL_NAMES == (
        ("H2O", "H2O"), ("N2", "n2"), ("CH4", "CH4"))
    for name, pool_key in fid._FIXED_MOLECULE_POOL_NAMES:
        source = pool[pool_key]
        assert systems[name].atom == source.atom, name
        assert systems[name].spin == source.spin, name
        assert systems[name].charge == source.charge, name
        # The pool spec may carry a benchmark reference path; the certificate's
        # copy must not, or the precompute would try to load and shape-check it.
        assert systems[name].external_data_path is None, name


def test_a_dfs_record_never_overrides_an_unconditional_molecule():
    """The DFS pretraining set carries N2 (at its GGA level only) and CH4 at
    its own geometries. Neither may win, or the same architecture family is
    certified on two different N2 molecules depending on its rung."""
    from xcquinox.alec.dfs_pretrain_set import dfs_pretrain_records
    dfs = {r["name"]: r for r in dfs_pretrain_records("gga")}
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    for name in ("N2", "CH4"):
        assert name in dfs, f"the DFS set no longer carries {name}"
        assert systems[name].atom != dfs[name]["atom"], (
            f"{name} resolved to the DFS geometry; the pool geometry must win")


def test_the_dfs_molecules_keep_their_own_geometries():
    """Only the three unconditional names are overridden. Every other DFS
    molecule is still certified at the geometry it was pretrained on."""
    from xcquinox.alec.dfs_pretrain_set import dfs_pretrain_records
    dfs = {r["name"]: r for r in dfs_pretrain_records("gga")
           if r["kind"] == "molecule"}
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    overridden = {name for name, _key in fid._FIXED_MOLECULE_POOL_NAMES}
    checked = 0
    for name, record in dfs.items():
        if name in overridden:
            continue
        assert systems[name].atom == record["atom"], name
        checked += 1
    assert checked >= 18


# ---------------------------------------------------------------------------
# Anti-fork guard: no second construction of a precompute quantity
# ---------------------------------------------------------------------------

def test_fidelity_never_rebuilds_a_precompute_field():
    """Every grid quantity and every descriptor block reaches the certificate
    through data.precompute_fixed_density_data(..., reference_xc=...). A second
    construction here would have to be kept identical to data.py by hand
    forever, which is the failure class this certificate exists to remove.

    Assembling libxc's own input rows inside _parent_exc_on_stored_grid is not
    a construction of a mol_data field and is deliberately not listed."""
    import inspect
    src = inspect.getsource(fid)
    for forbidden in ("compute_rung35_occupancy",
                      "compute_rung35_multishell_occupancy",
                      "compute_dm_features_array",
                      "compute_alpha",
                      "doubled_spin_dm",
                      "nabla_rho_grid",
                      "rung35_proj_ao",
                      "rung35ms_proj_ao"):
        assert forbidden not in src, (
            f"fidelity.py references {forbidden!r}: the parent density's grid "
            "quantities and descriptor blocks must come from "
            "precompute_fixed_density_data(..., reference_xc=...), not from a "
            "second construction in this module")


# ---------------------------------------------------------------------------
# Model construction and the parent-density request
# ---------------------------------------------------------------------------

def test_build_certified_model_loads_the_checkpoint_not_the_skeleton(tmp_path):
    """The skeleton's seed fixes the tree SHAPE only; every array leaf comes
    from the checkpoint. A builder that returned the skeleton would certify a
    randomly initialised network -- exactly the state the gate exists to
    catch."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.networks import create_network_pair

    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_3x16", seed=7)
    arch = get_architecture("deep_3x16")
    _built, model = fid.build_certified_model(_cfg(pretrain_seed=99), run_dir,
                                              "deep_3x16")
    from_checkpoint, _ = create_network_pair(arch, seed=7)
    from_skeleton, _ = create_network_pair(arch, seed=99)
    got = jax.tree_util.tree_leaves(eqx.filter(model.xnet, eqx.is_array))
    want = jax.tree_util.tree_leaves(eqx.filter(from_checkpoint, eqx.is_array))
    other = jax.tree_util.tree_leaves(eqx.filter(from_skeleton, eqx.is_array))
    assert len(got) == len(want) == len(other)
    assert all(bool(jnp.allclose(a, b)) for a, b in zip(got, want))
    # The two seeds really do differ, so the assertion above has content.
    assert any(not bool(jnp.allclose(a, b)) for a, b in zip(want, other))


def test_evaluate_system_requests_the_parent_functionals_density(monkeypatch):
    """The certificate asks the library's one construction path for a record
    built on the PARENT functional's self-consistent density, and forwards the
    run's Coulomb backend and orientation lock unchanged."""
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.models import AlecGGAModel

    seen = {}
    original = data_mod.precompute_fixed_density_data

    def _spy(mol_spec, **kwargs):
        seen.update(kwargs)
        return original(mol_spec, **kwargs)

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _spy)
    arch = get_architecture("deep_3x16")
    model = AlecGGAModel.from_arch(arch, seed=0)
    rec = fid.evaluate_system(model, arch.materialize_descriptors(),
                              _tiny_oracle_set()[0], parent="pbe",
                              auxbasis=None, orientation_lock_strength=0.0)
    assert seen["reference_xc"] == "pbe"
    assert seen["orientation_lock_strength"] == 0.0
    assert rec["reference_xc"] == "pbe"
    assert rec["is_atom"] is True


def test_evaluate_system_refuses_a_record_built_on_another_functional(
        monkeypatch):
    """A record whose reference_xc is not the parent would measure the network
    against the wrong density; that raises rather than entering the table."""
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.models import AlecGGAModel

    original = data_mod.precompute_fixed_density_data

    def _mislabel(mol_spec, **kwargs):
        md = dict(original(mol_spec, **kwargs))
        md["reference_xc"] = "lda,vwn"
        return md

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _mislabel)
    arch = get_architecture("deep_3x16")
    model = AlecGGAModel.from_arch(arch, seed=0)
    with pytest.raises(ValueError, match="reference_xc"):
        fid.evaluate_system(model, arch.materialize_descriptors(),
                            _tiny_oracle_set()[0], parent="pbe")


def test_meta_gga_architecture_is_certified_against_scan(tmp_path, monkeypatch):
    """End to end for the rung that motivated reference_xc: a meta-GGA
    architecture's certificate must be computed against SCAN, on SCAN's own
    density."""
    import xcquinox.alec.data as data_mod
    seen = []
    original = data_mod.precompute_fixed_density_data

    def _spy(mol_spec, **kwargs):
        seen.append(kwargs.get("reference_xc"))
        return original(mol_spec, **kwargs)

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _spy)
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_mgga_3x16", seed=0)
    payload = fid.fidelity_certificate(
        _cfg(arch=("deep_mgga_3x16",), pretrain_seed=0), run_dir,
        "deep_mgga_3x16", oracle_set=_tiny_oracle_set())
    assert payload["parent"] == "scan"
    assert set(seen) == {"scan"}
    assert all(r["reference_xc"] == "scan" for r in payload["per_system"])


# ---------------------------------------------------------------------------
# The certificate, with the per-system evaluation mocked at the seam
# ---------------------------------------------------------------------------

def _fake_evaluate(table):
    """Build an ``evaluate`` seam returning canned dE_xc (mHa) per name."""
    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        d = table[mol_spec.name]
        return {"name": mol_spec.name, "spin": int(mol_spec.spin),
                "charge": int(mol_spec.charge),
                "is_atom": fid.is_atom_system(mol_spec),
                "n_grid": 10, "reference_xc": parent,
                "E_xc_nn": -1.0 + d / fid.HA_TO_MHA, "E_xc_parent": -1.0,
                "E_xc_parent_numint": -1.0, "E_xc_parent_record": -1.0,
                "parent_grid_diff_Ha": 0.0, "parent_record_diff_Ha": 0.0,
                "dE_xc_mHa": d, "duration_s": 0.0}
    return _evaluate


def _tiny_oracle_set(basis="sto-3g", grid_level=1):
    from xcquinox.alec.config import MoleculeSpec
    return (
        MoleculeSpec(name="atom_H", atom="H 0.0 0.0 0.0", basis=basis, spin=1,
                     atom_composition=(("H", 1),), grid_level=grid_level),
        MoleculeSpec(name="H2", atom="H 0 0 0.371395; H 0 0 -0.371395",
                     basis=basis, spin=0, atom_composition=(("H", 2),),
                     grid_level=grid_level),
    )


def _stub_checkpoint(run_dir, arch_name="deep_3x16", seed=42):
    """Write a real xnet.eqx + cnet.eqx pair for ``arch_name``."""
    import equinox as eqx
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    arch = get_architecture(arch_name)
    xnet, cnet = create_network_pair(arch, seed=seed)
    d = pretrain_checkpoint_dir(run_dir, arch_name)
    os.makedirs(d, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(d, "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(os.path.join(d, "cnet.eqx"), cnet)
    return d


def test_certificate_passes_within_tolerance_and_writes_the_schema(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg()
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16",
        oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 0.5, "H2": 1.0}))

    assert payload["verdict"] == "PASS"
    assert payload["arch"] == "deep_3x16"
    assert payload["parent"] == "pbe"
    assert payload["identity"] == fid.run_identity(cfg)
    assert payload["tolerances"] == {"tol_AE": 1.0, "tol_atom": 1.0,
                                     "override_reason": None}
    assert payload["enforced"] is True
    assert isinstance(payload["xcquinox_version"], str)
    assert payload["timestamp"].endswith("Z")
    assert payload["duration_s"] >= 0.0
    assert [r["name"] for r in payload["per_system"]] == ["atom_H", "H2"]
    assert [r["name"] for r in payload["per_atomization"]] == ["H2"]
    s = payload["summary"]
    assert s["n_systems"] == 2 and s["n_atoms"] == 1
    assert s["n_atomizations"] == 1 and s["n_failed_systems"] == 0
    assert s["max_parent_grid_diff_Ha"] == pytest.approx(0.0)
    assert s["max_parent_record_diff_Ha"] == pytest.approx(0.0)
    assert s["max_atom_mHa"] == pytest.approx(0.5)
    # dAE = dE_xc(H2) - 2 dE_xc(H) = 1.0 - 1.0 = 0 mHa.
    assert s["max_dAE_kcalmol"] == pytest.approx(0.0, abs=1e-12)
    assert s["failure_reasons"] == []

    on_disk = json.loads(
        open(fid.certificate_path(run_dir, "deep_3x16")).read())
    assert on_disk == payload
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "PASS"


def test_certificate_fails_on_the_atom_tolerance(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 13.7, "H2": 27.4}))
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] == pytest.approx(13.7)
    assert any("tol_atom" in r for r in payload["summary"]["failure_reasons"])
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"


def test_certificate_fails_on_the_atomization_tolerance(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    # dAE(H2) = (1.0 - 2 * 0.1) mHa = 0.8 mHa = 0.502 kcal/mol -> passes at
    # 1.0; scale it up until it does not.
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 0.1, "H2": 5.0}))
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] == pytest.approx(0.1)
    assert payload["summary"]["max_dAE_kcalmol"] == pytest.approx(
        (5.0 - 0.2) / fid.HA_TO_MHA * fid.HA_TO_KCAL)
    assert any("tol_AE" in r for r in payload["summary"]["failure_reasons"])


def test_certificate_honours_configured_tolerances(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg(tol_AE=2.0, tol_atom=2.0,
               override_reason=None)
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 1.5, "H2": 3.0}))
    assert payload["verdict"] == "PASS"
    assert payload["tolerances"]["tol_atom"] == 2.0


def test_certificate_records_the_override_reason(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg(tol_AE=5.0, tol_atom=5.0,
               override_reason="rung-3.5 control arm, documented in HISTORY")
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 4.0, "H2": 8.0}))
    assert payload["verdict"] == "PASS"
    assert payload["tolerances"]["override_reason"] == (
        "rung-3.5 control arm, documented in HISTORY")


def test_certificate_records_the_enforcement_flag(tmp_path):
    """A non-enforcing run still writes the TRUE verdict; only the gates
    change behaviour, and they read the flag out of the certificate."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg(enforce=False,
               override_reason="workflow matrix: 50-step pretrain")
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 13.7, "H2": 27.4}))
    assert payload["verdict"] == "FAIL"
    assert payload["enforced"] is False
    assert payload["tolerances"]["override_reason"] == (
        "workflow matrix: 50-step pretrain")
    # The record layers still see a FAIL ...
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"
    # ... while an on-node gate is allowed to continue.
    allowed, message = fid.gate_certificate(run_dir, "deep_3x16")
    assert allowed is True
    assert "enforcement is OFF" in message


def test_certificate_records_a_system_that_raised_and_fails(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)

    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        if mol_spec.name == "H2":
            raise RuntimeError("SCF blew up")
        return _fake_evaluate({"atom_H": 0.1})(
            model, descriptors, mol_spec, parent=parent)

    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_evaluate)
    assert payload["verdict"] == "FAIL"
    failed = [r for r in payload["per_system"] if "error" in r]
    assert [r["name"] for r in failed] == ["H2"]
    assert "SCF blew up" in failed[0]["error"]
    assert payload["summary"]["n_failed_systems"] == 1
    assert any("could not be evaluated" in r
               for r in payload["summary"]["failure_reasons"])


def test_certificate_fails_when_the_two_parent_grid_routes_disagree(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)

    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        rec = _fake_evaluate({"atom_H": 0.1, "H2": 0.2})(
            model, descriptors, mol_spec, parent=parent)
        rec["parent_grid_diff_Ha"] = 1e-3
        return rec

    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_evaluate)
    assert payload["verdict"] == "FAIL"
    assert any("grid" in r for r in payload["summary"]["failure_reasons"])


def test_certificate_fails_when_the_record_route_disagrees(tmp_path):
    """The third parent route -- the XC energy the reference SCF itself
    accumulated -- is bounded by the same PARENT_GRID_TOL_HA. A disagreement
    is a failure of the certificate's own consistency: it is reported as such
    and never averaged into the parent energy."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)

    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        rec = _fake_evaluate({"atom_H": 0.1, "H2": 0.2})(
            model, descriptors, mol_spec, parent=parent)
        rec["parent_record_diff_Ha"] = 1e-3
        return rec

    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_evaluate)
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_parent_record_diff_Ha"] == pytest.approx(
        1e-3)
    assert payload["summary"]["max_parent_grid_diff_Ha"] == pytest.approx(0.0)
    assert any("accumulated" in r for r in payload["summary"]["failure_reasons"])
    # The two tolerance checks themselves still pass; the consistency failure
    # is the only reason on record.
    assert len(payload["summary"]["failure_reasons"]) == 1


def test_certificate_records_the_checkpoint_digests(tmp_path):
    """The certificate names the exact networks it measured: the SHA-256 of
    xnet.eqx and cnet.eqx, so a checkpoint rewritten after certification can
    be told apart from the one the verdict refers to."""
    from xcquinox.alec.cluster.materialize import _sha256_file
    run_dir = str(tmp_path / "run")
    d = _stub_checkpoint(run_dir)
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 0.5, "H2": 1.0}))
    assert payload["checkpoint"] == {
        "dir": d,
        "xnet_sha256": _sha256_file(os.path.join(d, "xnet.eqx")),
        "cnet_sha256": _sha256_file(os.path.join(d, "cnet.eqx")),
    }
    assert len(payload["checkpoint"]["xnet_sha256"]) == 64


# ---------------------------------------------------------------------------
# The descriptor log transform: recorded, and compared where it is stated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch_name,on", [("deep_3x16", True),
                                          ("deep_notransform_3x16", False)])
def test_certificate_records_the_descriptor_log_transform(tmp_path, arch_name,
                                                          on):
    """The certificate states the descriptor log transform the certified
    networks read their inputs through, beside the anchor and the coordinates.

    Like those two it is a static field that changes no parameter shape, so
    the checkpoint digests the certificate carries say nothing about it: the
    same leaves are a different functional under the other value. Both values
    are exercised, on the two registry architectures that differ in this field
    and in nothing else that matters here.
    """
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, arch_name=arch_name)
    payload = fid.fidelity_certificate(
        _cfg(arch=(arch_name,)), run_dir, arch_name,
        oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 0.5, "H2": 1.0}))
    assert payload["descriptor_log_transform"] is on, payload
    on_disk = json.loads(open(fid.certificate_path(run_dir, arch_name)).read())
    assert on_disk["descriptor_log_transform"] is on


def test_a_certificate_stating_the_other_log_transform_is_a_mismatch():
    """A certificate recording a transform the run's architecture does not
    carry does not describe the networks this run trains, and is reported by
    the one comparison both the record layer and the pretrain keep check
    apply."""
    cert = {"arch": "deep_3x16", "parent_anchor": False,
            "descriptor_coordinates": "legacy",
            "descriptor_log_transform": False}
    assert fid.model_class_mismatches(_cfg(), cert) == [
        ("descriptor_log_transform", False, True)]


def test_a_certificate_stating_the_architectures_log_transform_agrees():
    """The control for the case above: the value the registry states is no
    mismatch, so the comparison is not one that refuses every certificate."""
    cert = {"arch": "deep_3x16", "parent_anchor": False,
            "descriptor_coordinates": "legacy",
            "descriptor_log_transform": True}
    assert fid.model_class_mismatches(_cfg(), cert) == []


def test_a_certificate_that_states_no_log_transform_still_validates():
    """Every certificate on the cluster was written before the key and carries
    the two class fields alone. Such a document states nothing about the
    transform, and is read exactly as it was."""
    cert = {"arch": "deep_3x16", "parent_anchor": False,
            "descriptor_coordinates": "legacy"}
    assert fid.model_class_mismatches(_cfg(), cert) == []


def test_the_expected_log_transform_follows_the_caller_s_architecture():
    """The expected value is the architecture the CALLER is asking about, as
    :func:`parent_mismatch`'s expected parent is.

    A certificate from another architecture's directory is refused on its
    ``arch`` line by both callers, but 23 of the 31 registered architectures
    set the transform, so such a document agrees on this field more often than
    not and the model-class report would go quiet about the field it exists to
    state. With the name passed, the comparison is against the architecture
    this run builds: ``medium`` carries the transform off and ``deep_3x16``
    carries it on.
    """
    cert = {"arch": "medium", "parent_anchor": False,
            "descriptor_coordinates": "legacy",
            "descriptor_log_transform": False}
    assert fid.model_class_mismatches(_cfg(), cert, "deep_3x16") == [
        ("descriptor_log_transform", False, True)]
    # Without a name the certificate's own is used, and it agrees with itself.
    assert fid.model_class_mismatches(_cfg(), cert) == []


def test_certificate_describes_run_states_the_log_transform_of_this_run(
        tmp_path):
    """The keep check hands its own architecture down, so the finding it
    reports is about the networks this run would train on."""
    cert = {"verdict": "PASS", "arch": "medium", "parent_anchor": False,
            "descriptor_coordinates": "legacy",
            "descriptor_log_transform": False}
    findings = fid.certificate_describes_run(
        _cfg(), str(tmp_path), "deep_3x16", cert)
    transform = [f for f in findings if "descriptor_log_transform" in f]
    assert len(transform) == 1, findings
    assert "this run builds descriptor_log_transform=True" in transform[0]


def test_a_log_transform_on_an_unregistered_arch_has_no_expected_value():
    """An architecture the registry does not carry has no value to compare
    with, and the comparison says nothing rather than guessing one; the
    unresolvable architecture is reported by the callers through
    ``parent_mismatch``."""
    cert = {"arch": "not_an_architecture", "parent_anchor": False,
            "descriptor_coordinates": "legacy",
            "descriptor_log_transform": True}
    assert fid.model_class_mismatches(_cfg(), cert) == []
    with pytest.raises(KeyError):
        fid.parent_mismatch("not_an_architecture", cert)


@pytest.mark.parametrize("enabled_before", [True, False])
def test_certificate_leaves_the_precompute_cache_as_it_found_it(
        tmp_path, enabled_before):
    """The certificate disables the precompute memo for its own loop (dozens of
    production-basis grids would exhaust a node) and restores the caller's
    setting afterwards, whichever it was."""
    import xcquinox.alec.data as data_mod
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    data_mod.set_precompute_cache_enabled(enabled_before)
    try:
        fid.fidelity_certificate(
            _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
            evaluate=_fake_evaluate({"atom_H": 0.5, "H2": 1.0}))
        assert data_mod._PRECOMPUTE_CACHE_ENABLED is enabled_before
    finally:
        data_mod.set_precompute_cache_enabled(True)


def test_certificate_applies_the_polarized_correlation_patch(tmp_path):
    """The pretrain worker builds a polarized cnet when the run is polarized;
    the certificate must load the checkpoint with the SAME architecture or the
    deserialise would fail on the cnet input width."""
    import dataclasses
    from xcquinox.alec.config import get_architecture
    run_dir = str(tmp_path / "run")
    arch = dataclasses.replace(get_architecture("deep_3x16"),
                               use_polarized_correlation=True)
    import equinox as eqx
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    xnet, cnet = create_network_pair(arch, seed=42)
    d = pretrain_checkpoint_dir(run_dir, "deep_3x16")
    os.makedirs(d, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(d, "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(os.path.join(d, "cnet.eqx"), cnet)

    built_arch, model = fid.build_certified_model(
        _cfg(polarized=True), run_dir, "deep_3x16")
    assert built_arch.use_polarized_correlation is True
    assert model is not None


# ---------------------------------------------------------------------------
# REAL physics: H and H2 at sto-3g, networks built in the test (seconds)
# ---------------------------------------------------------------------------

def test_certificate_real_physics_on_h_and_h2_at_sto3g(tmp_path):
    """The whole energy path, for real, on two tiny systems.

    ``deep_3x16`` is built with ``zero_init_final_layer=True``, so a freshly
    seeded network has Fx = Fc = 1 exactly and its E_xc is the LDA exchange
    plus PW92 correlation. Against PBE on the same frozen PBE density that is
    a large, definite offset, so this pins the sign, the magnitude, the
    atomization fold and the FAIL branch at once. Every number the certificate
    reports is re-derived in the test from an independent PySCF route.
    """
    import numpy as np
    from pyscf import dft, gto
    from pyscf.dft import numint

    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_3x16", seed=0)
    cfg = _cfg(pretrain_seed=0)
    systems = _tiny_oracle_set()

    payload = fid.fidelity_certificate(cfg, run_dir, "deep_3x16",
                                       oracle_set=systems)

    by_name = {r["name"]: r for r in payload["per_system"]}
    assert set(by_name) == {"atom_H", "H2"}
    assert by_name["atom_H"]["is_atom"] is True
    assert by_name["H2"]["is_atom"] is False
    assert by_name["atom_H"]["spin"] == 1 and by_name["H2"]["spin"] == 0

    # (1) Every record was built on the PARENT's own self-consistent density,
    #     and the parent energy is that functional on that density, on the
    #     SAME grid PySCF's own nr_rks / nr_uks uses.
    assert all(r["reference_xc"] == "pbe" for r in by_name.values())
    for ms in systems:
        rec = by_name[ms.name]
        mol = gto.M(atom=ms.atom, basis=ms.basis, charge=ms.charge,
                    spin=ms.spin, verbose=0)
        mf = dft.UKS(mol) if ms.spin else dft.RKS(mol)
        mf.xc = "pbe"
        mf.grids.level = ms.grid_level
        mf.kernel()
        grids = dft.Grids(mol)
        grids.level = ms.grid_level
        grids.build()
        ni = numint.NumInt()
        dm = mf.make_rdm1()
        if ms.spin:
            _v, exc, _ = ni.nr_uks(mol, grids, "PBE", dm)
        else:
            _v, exc, _ = ni.nr_rks(mol, grids, "PBE", dm)
        assert rec["E_xc_parent"] == pytest.approx(float(exc), abs=1e-8)
        assert rec["E_xc_parent_numint"] == pytest.approx(float(exc), abs=1e-8)
        assert abs(rec["parent_grid_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
        # Third independent route: the XC energy PySCF accumulated during the
        # reference SCF itself, carried on the record as E_xc_pbe.
        assert abs(rec["parent_record_diff_Ha"]) < fid.PARENT_GRID_TOL_HA

    # (2) dE_xc is exactly the difference the record carries, in mHa.
    for rec in by_name.values():
        assert rec["dE_xc_mHa"] == pytest.approx(
            (rec["E_xc_nn"] - rec["E_xc_parent"]) * fid.HA_TO_MHA, rel=1e-12)

    # (3) The atomization offset is the molecule minus its atoms, in kcal/mol.
    dae = {r["name"]: r["dAE_kcalmol"] for r in payload["per_atomization"]}
    expected = ((by_name["H2"]["dE_xc_mHa"] - 2 * by_name["atom_H"]["dE_xc_mHa"])
                / fid.HA_TO_MHA * fid.HA_TO_KCAL)
    assert dae["H2"] == pytest.approx(expected, rel=1e-12)

    # (4) An LDA-limit network is nowhere near PBE, so the certificate FAILS
    #     at the binding 1.0 mHa / 1.0 kcal/mol tolerances.
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] > 1.0
    assert abs(dae["H2"]) > 1.0
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"


def test_certificate_real_physics_passes_at_a_loosened_tolerance(tmp_path):
    """The PASS branch on real numbers: the same two systems under a
    deliberately loosened tolerance carrying its override reason."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_3x16", seed=0)
    cfg = _cfg(tol_AE=100.0, tol_atom=100.0, pretrain_seed=0,
               override_reason="unit test: pins the PASS branch on real "
                               "sto-3g numbers")
    payload = fid.fidelity_certificate(cfg, run_dir, "deep_3x16",
                                       oracle_set=_tiny_oracle_set())
    assert payload["verdict"] == "PASS"
    assert payload["summary"]["failure_reasons"] == []
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "PASS"


def test_scan_parent_routes_agree_on_h_and_h2_at_sto3g():
    """The meta-GGA parent through the three routes: the point-wise libxc
    evaluation on the stored grid (which must assemble tau for SCAN), PySCF's
    own nr_uks / nr_rks on a fresh grid, and the XC energy the SCAN reference
    SCF accumulated. All three agree within PARENT_GRID_TOL_HA on an open and
    a closed shell, and the point-wise number is re-derived here from an
    independent SCF."""
    from pyscf import dft, gto
    from pyscf.dft import numint
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.models import AlecGGAModel

    arch = get_architecture("deep_mgga_3x16")
    model = AlecGGAModel.from_arch(arch, seed=0)
    for ms in _tiny_oracle_set():
        rec = fid.evaluate_system(model, arch.materialize_descriptors(), ms,
                                  parent="scan")
        assert rec["reference_xc"] == "scan"
        assert abs(rec["parent_grid_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
        assert abs(rec["parent_record_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
        mol = gto.M(atom=ms.atom, basis=ms.basis, charge=ms.charge,
                    spin=ms.spin, verbose=0)
        mf = dft.UKS(mol) if ms.spin else dft.RKS(mol)
        mf.xc = "scan"
        mf.grids.level = ms.grid_level
        mf.kernel()
        grids = dft.Grids(mol)
        grids.level = ms.grid_level
        grids.build()
        ni = numint.NumInt()
        dm = mf.make_rdm1()
        if ms.spin:
            _nelec, exc, _vmat = ni.nr_uks(mol, grids, "SCAN", dm)
        else:
            _nelec, exc, _vmat = ni.nr_rks(mol, grids, "SCAN", dm)
        assert rec["E_xc_parent"] == pytest.approx(float(exc), abs=1e-8)


# ---------------------------------------------------------------------------
# The parent functional presented as the model: exactness of the energy path
# ---------------------------------------------------------------------------

class _ParentCNet(eqx.Module):
    """The one cnet attribute the UKS energy path reads."""
    use_spin_polarization: bool = eqx.field(static=True)


class _LibxcParentModel(eqx.Module):
    """The exact parent functional behind the model interface.

    ``fixed_density_total_energy`` reads ``descriptors``,
    ``cnet.use_spin_polarization``, ``eval_exc`` (closed shell) and
    ``eval_ex`` / ``eval_ec`` (open shell: exchange spin-scaled on the doubled
    channel densities, correlation on the total density with the production
    ``uks_zeta``). Each evaluation hands libxc exactly the rows that interface
    carries -- the density, the gradient invariant, the descriptor block and,
    through a polarization-aware cnet, zeta -- so the number that comes back
    is the parent's own energy density on the same points the network is
    integrated on. libxc runs inside ``jax.pure_callback`` because
    ``compute_exc_nn`` is jitted.

    GGA parent (``alpha_column < 0``): rows (rho, |grad rho|, 0, 0). PBE
    correlation depends on the spin densities and the TOTAL gradient only
    (Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996), Eq. 7-8), so the
    per-spin gradient split proportional to the spin densities is exact.

    Meta-GGA parent (``alpha_column >= 0``): the interface carries no tau,
    only the descriptor's clamped iso-orbital alpha, so tau is recovered as
    ``alpha tau_unif + sigma / (8 rho)`` (``metagga.compute_alpha`` inverted;
    exact wherever the [0, 100] clamp was inactive). The doubled exchange
    channels invert to ``2 tau_sigma`` directly. SCAN's correlation depends on
    the total density, total gradient, total tau and zeta only (Sun,
    Ruzsinszky, Perdew, PRL 115, 036402 (2015): alpha with d_s(zeta)), so the
    proportional per-spin split of tau is exact on physical data -- measured
    2.1e-18 Ha on the O-atom SCAN record against the true per-spin rows.

    ``offset_per_electron`` adds a constant ``c`` to the energy per electron,
    ``E_xc -> E_xc + c N``: the known offset the O-B oracle measures.
    """
    xc: str = eqx.field(static=True)
    descriptors: tuple = eqx.field(static=True)
    polarized: bool = eqx.field(static=True)
    offset_per_electron: float = eqx.field(static=True)
    alpha_column: int = eqx.field(static=True, default=-1)

    @property
    def cnet(self):
        return _ParentCNet(use_spin_polarization=self.polarized)

    def _tau(self, rho, sigma, alpha):
        rho_safe = np.maximum(rho, 1e-30)
        tau_unif = (0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
                    * rho_safe ** (5.0 / 3.0))
        return alpha * tau_unif + sigma / (8.0 * rho_safe)

    def _rows(self, rho, sigma, alpha):
        zero = np.zeros_like(rho)
        rows = [rho, np.sqrt(np.maximum(sigma, 0.0)), zero, zero]
        if self.alpha_column >= 0:
            rows.append(self._tau(rho, sigma, alpha))
        return np.vstack(rows)

    def _x(self, rho, sigma, alpha):
        from pyscf.dft import numint
        exc = numint.NumInt().eval_xc(f"{self.xc},",
                                      self._rows(rho, sigma, alpha), spin=0)[0]
        return rho * exc

    def _c(self, rho, sigma, alpha, zeta):
        from pyscf.dft import numint
        ni = numint.NumInt()
        if self.polarized:
            g = np.sqrt(np.maximum(sigma, 0.0))
            zero = np.zeros_like(rho)
            wa, wb = 0.5 * (1.0 + zeta), 0.5 * (1.0 - zeta)
            rows_a = [rho * wa, g * wa, zero, zero]
            rows_b = [rho * wb, g * wb, zero, zero]
            if self.alpha_column >= 0:
                tau = self._tau(rho, sigma, alpha)
                rows_a.append(tau * wa)
                rows_b.append(tau * wb)
            exc = ni.eval_xc(f",{self.xc}",
                             (np.vstack(rows_a), np.vstack(rows_b)), spin=1)[0]
        else:
            exc = ni.eval_xc(f",{self.xc}", self._rows(rho, sigma, alpha),
                             spin=0)[0]
        return rho * (exc + self.offset_per_electron)

    def _alpha(self, rho, features):
        if self.alpha_column >= 0:
            return features[:, self.alpha_column]
        return jnp.zeros_like(rho)

    def _callback(self, fn, rho, *args):
        out = jax.ShapeDtypeStruct(rho.shape, rho.dtype)
        return jax.pure_callback(fn, out, rho, *args)

    def eval_ex(self, rho, sigma, features):
        return self._callback(lambda r, s, a: np.asarray(self._x(r, s, a)),
                              rho, sigma, self._alpha(rho, features))

    def eval_ec(self, rho, sigma, features, zeta=0.0):
        zeta = jnp.broadcast_to(jnp.asarray(zeta, dtype=rho.dtype), rho.shape)
        return self._callback(
            lambda r, s, a, z: np.asarray(self._c(r, s, a, z)),
            rho, sigma, self._alpha(rho, features), zeta)

    def eval_exc(self, rho, sigma, features, zeta=0.0):
        return self.eval_ex(rho, sigma, features) + self.eval_ec(
            rho, sigma, features, zeta=zeta)


def _parent_oracle_set(basis="sto-3g", grid_level=1):
    """O, H and H2O: the two free atoms H2O dissociates into, so the
    atomization fold is exercised on a real molecule."""
    from xcquinox.alec.config import MoleculeSpec
    return (
        MoleculeSpec(name="atom_H", atom="H 0.0 0.0 0.0", basis=basis, spin=1,
                     atom_composition=(("H", 1),), grid_level=grid_level),
        MoleculeSpec(name="atom_O", atom="O 0.0 0.0 0.0", basis=basis, spin=2,
                     atom_composition=(("O", 1),), grid_level=grid_level),
        MoleculeSpec(name="H2O",
                     atom="O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; "
                          "H 0.0 -0.7572 -0.4692",
                     basis=basis, spin=0,
                     atom_composition=(("H", 2), ("O", 1)),
                     grid_level=grid_level),
    )


# (arch, libxc parent code, alpha column of the descriptor block)
_PARENT_DOUBLES = {
    "pbe": ("deep_3x16", "PBE", -1),
    "scan": ("deep_mgga_3x16", "SCAN", 0),
}


def _certify_with_parent_as_model(tmp_path, monkeypatch, parent, offset):
    from xcquinox.alec.config import get_architecture
    arch_name, xc, alpha_column = _PARENT_DOUBLES[parent]
    run_dir = str(tmp_path / f"run_{parent}_{offset:g}")
    _stub_checkpoint(run_dir, arch_name, seed=0)
    arch = get_architecture(arch_name)
    double = _LibxcParentModel(xc=xc, descriptors=arch.materialize_descriptors(),
                               polarized=True, offset_per_electron=offset,
                               alpha_column=alpha_column)
    monkeypatch.setattr(fid, "build_certified_model",
                        lambda cfg, rd, name: (arch, double))
    payload = fid.fidelity_certificate(
        _cfg(arch=(arch_name,), pretrain_seed=0), run_dir, arch_name,
        oracle_set=_parent_oracle_set())
    assert payload["parent"] == parent
    return payload, {r["name"]: r for r in payload["per_system"]}


# Bounds on |E_xc_nn - E_xc_parent| with the parent behind the model
# interface, anchored to the sto-3g / grid 1 measurements quoted in the test
# below: (O and H2O, H atom), in Ha.
_PARENT_AS_MODEL_BOUNDS = {
    "pbe": (1e-10, 5e-6),
    "scan": (1e-8, 5e-6),
}


@pytest.mark.parametrize("parent", ["pbe", "scan"])
def test_certificate_is_exact_when_the_model_is_the_parent_functional(
        tmp_path, monkeypatch, parent):
    """O-A. With the parent itself behind the model interface, the
    certificate's E_xc_nn - E_xc_parent on the O atom and on H2O is round-off
    and the verdict is PASS: the model path and the point-wise parent route
    reduce the same density, on the same points and weights, through the same
    libxc, the one in JAX and the other in numpy.

    Measured at sto-3g / grid 1. PBE: 3.6e-15 Ha (O) and 7.1e-15 Ha (H2O),
    a few ulps of an E_xc of order 8 Ha. SCAN: 2.4e-9 Ha (O), 3.4e-10 Ha
    (H2O) -- the meta-GGA interface carries the smoothed, clamped alpha, not
    tau, so the 572 (O) / 627 (H2O) tail points clamped at alpha = 100 carry
    1.5e-4 / 5.6e-4 electrons whose tau the double cannot recover, and the
    indicator's own smoothing floor (metagga._ALPHA_SMOOTHING_WIDTH / 2,
    which the certificate's inversion does not undo) enters its tau at the
    1e-9-Ha level. The SCAN figures replace a superseded record of 2.0e-10
    (O) and 1.6e-9 Ha (H2O) taken under the hard clip; the O atom moved 12x,
    so the 1e-8 bound now clears it by 4.2x rather than 50x -- the tightest
    margin in this test, and what a further move of the indicator's floor
    would trip. Both figures stay more than five orders inside tol_atom. The
    per-spin meta-GGA blocks are on trial here as much as the energy path: a
    wrong doubled-channel alpha would move the O atom by mHa.

    The H atom is pinned separately: the production path clips zeta to
    1 - 1e-6 (oneshot._ZETA_BOUNDARY_EPS) where the parent sees zeta = 1
    exactly, and the one-electron atom's correlation is not zero
    (self-correlation), so the double carries the zeta-derivative of
    rho eps_c across that clip -- 8.0e-7 Ha for PBE and 4.4e-8 Ha for SCAN,
    more than two orders inside tol_atom."""
    bound_heavy, bound_h = _PARENT_AS_MODEL_BOUNDS[parent]
    payload, by_name = _certify_with_parent_as_model(tmp_path, monkeypatch,
                                                     parent, 0.0)
    for name in ("atom_O", "H2O"):
        rec = by_name[name]
        assert abs(rec["E_xc_nn"] - rec["E_xc_parent"]) < bound_heavy, (
            name, rec)
        assert abs(rec["parent_grid_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
        assert abs(rec["parent_record_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
    assert abs(by_name["atom_H"]["E_xc_nn"]
               - by_name["atom_H"]["E_xc_parent"]) < bound_h
    assert payload["verdict"] == "PASS"
    assert payload["summary"]["failure_reasons"] == []
    assert payload["summary"]["max_atom_mHa"] < 1e-2
    assert payload["summary"]["max_dAE_kcalmol"] < 1e-2


def test_certificate_measures_a_known_per_electron_offset(tmp_path,
                                                          monkeypatch):
    """O-B. The parent plus a constant c = 0.5 mHa per electron must move
    every dE_xc by exactly c N_e_grid, N_e_grid the ELECTRON count the
    record's quadrature carries -- named to keep it apart from the payload's
    n_grid, the number of grid points -- (the shift against the c = 0
    certificate is compared, so the H atom's zeta-clip residual of the
    previous test drops out); the O atom (N_e = 8) then exceeds tol_atom by
    the predicted 4.0 mHa while the atomization fold cancels the offset to
    c (N_e_grid(H2O) - N_e_grid(O) - 2 N_e_grid(H)), the quadrature residual.
    That is why the certificate carries an atomic tolerance beside the
    atomization one: a per-electron offset is invisible to dAE. Measured
    shift minus prediction: 6.8e-12 mHa (O), 1.4e-12 mHa (H2O) at sto-3g /
    grid 1."""
    import numpy as np
    from xcquinox.alec.data import precompute_fixed_density_data
    c_mha = 0.5
    _payload0, base = _certify_with_parent_as_model(tmp_path, monkeypatch,
                                                    "pbe", 0.0)
    payload, by_name = _certify_with_parent_as_model(
        tmp_path, monkeypatch, "pbe", c_mha / fid.HA_TO_MHA)
    n_e_grid = {}
    for ms in _parent_oracle_set():
        # The record the certificate measured: the O atom is degenerate and
        # carries the certificate's orientation lock, the others none.
        md = precompute_fixed_density_data(
            ms, reference_xc="pbe",
            orientation_lock_strength=by_name[ms.name][
                "orientation_lock_strength"])
        n_e_grid[ms.name] = float(np.sum(np.asarray(md["grid_weights"])
                                         * np.asarray(md["rho_grid"])))
    assert by_name["atom_O"]["orientation_lock_strength"] == (
        fid.atom_orientation_lock_strength())
    assert by_name["atom_H"]["orientation_lock_strength"] == 0.0
    assert n_e_grid["atom_O"] == pytest.approx(8.0, abs=1e-3)
    for name, rec in by_name.items():
        shift = rec["dE_xc_mHa"] - base[name]["dE_xc_mHa"]
        assert shift == pytest.approx(c_mha * n_e_grid[name], abs=1e-7), name
    for name in ("atom_O", "H2O"):
        assert by_name[name]["dE_xc_mHa"] == pytest.approx(
            c_mha * n_e_grid[name], abs=1e-6), name
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] == pytest.approx(
        c_mha * n_e_grid["atom_O"], abs=1e-6)
    assert any("tol_atom" in r for r in payload["summary"]["failure_reasons"])
    predicted_dae = (c_mha * (n_e_grid["H2O"] - n_e_grid["atom_O"]
                              - 2 * n_e_grid["atom_H"])
                     / fid.HA_TO_MHA * fid.HA_TO_KCAL)
    dae = {r["name"]: r["dAE_kcalmol"] for r in payload["per_atomization"]}
    dae0 = {r["name"]: r["dAE_kcalmol"] for r in _payload0["per_atomization"]}
    assert dae["H2O"] - dae0["H2O"] == pytest.approx(predicted_dae, abs=1e-7)
    assert abs(dae["H2O"]) < 1.0
    assert not any("tol_AE" in r for r in payload["summary"]["failure_reasons"])


# ---------------------------------------------------------------------------
# The reference SCF must have converged
# ---------------------------------------------------------------------------

def _stamped(record, **stamp):
    """A copy of ``record`` whose metadata carries ``stamp``."""
    md = dict(record)
    md["mol_metadata"] = dict(md["mol_metadata"] or {}, **stamp)
    return md


def test_evaluate_system_refuses_an_unconverged_reference_record(monkeypatch):
    """A record whose metadata reports an unconverged reference SCF is not the
    parent's density (measured on H2O / SCAN at max_cycle=1: +7.2e-2 Ha in
    the total energy, 0.315 in the density matrix); the network is never
    measured on it."""
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.models import AlecGGAModel

    original = data_mod.precompute_fixed_density_data
    monkeypatch.setattr(
        data_mod, "precompute_fixed_density_data",
        lambda mol_spec, **kw: _stamped(original(mol_spec, **kw),
                                        reference_scf_converged=False,
                                        reference_scf_cycles=1))
    arch = get_architecture("deep_3x16")
    model = AlecGGAModel.from_arch(arch, seed=0)
    with pytest.raises(fid.ReferenceNotConverged, match="did not converge") \
            as info:
        fid.evaluate_system(model, arch.materialize_descriptors(),
                            _tiny_oracle_set()[0], parent="pbe")
    assert info.value.cycles == 1
    assert isinstance(info.value, ValueError)


def test_evaluate_system_copies_the_convergence_stamp_and_the_lock(
        monkeypatch):
    """The stamp the precompute wrote (converged flag and cycle count) and the
    orientation-lock strength the system was evaluated at are carried on the
    record; a record written without the stamp carries None and is not
    refused."""
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.models import AlecGGAModel

    arch = get_architecture("deep_3x16")
    model = AlecGGAModel.from_arch(arch, seed=0)
    descriptors = arch.materialize_descriptors()
    original = data_mod.precompute_fixed_density_data

    monkeypatch.setattr(
        data_mod, "precompute_fixed_density_data",
        lambda mol_spec, **kw: _stamped(original(mol_spec, **kw),
                                        reference_scf_converged=True,
                                        reference_scf_cycles=9))
    rec = fid.evaluate_system(model, descriptors, _tiny_oracle_set()[0],
                              parent="pbe", orientation_lock_strength=0.0)
    assert rec["reference_scf_converged"] is True
    assert rec["reference_scf_cycles"] == 9
    assert rec["orientation_lock_strength"] == 0.0

    def _unstamped(mol_spec, **kw):
        md = dict(original(mol_spec, **kw))
        meta = dict(md["mol_metadata"] or {})
        meta.pop("reference_scf_converged", None)
        meta.pop("reference_scf_cycles", None)
        md["mol_metadata"] = meta
        return md

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _unstamped)
    rec = fid.evaluate_system(model, descriptors, _tiny_oracle_set()[0],
                              parent="pbe", orientation_lock_strength=3e-5)
    assert rec["reference_scf_converged"] is None
    assert rec["reference_scf_cycles"] is None
    assert rec["orientation_lock_strength"] == 3e-5


def test_certificate_fails_by_name_when_a_reference_scf_did_not_converge(
        tmp_path):
    """An unconverged reference is a named consistency failure of its own:
    the per-system entry carries the stamp, the summary counts it, and the
    reason names the system -- it is never folded into the generic
    'could not be evaluated' bucket and can never PASS."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)

    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        if mol_spec.name == "H2":
            raise fid.ReferenceNotConverged(
                "the reference PBE SCF for 'H2' did not converge", cycles=50)
        return _fake_evaluate({"atom_H": 0.1})(
            model, descriptors, mol_spec, parent=parent)

    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_evaluate)
    assert payload["verdict"] == "FAIL"
    entry = [r for r in payload["per_system"] if r["name"] == "H2"][0]
    assert entry["reference_scf_converged"] is False
    assert entry["reference_scf_cycles"] == 50
    assert "did not converge" in entry["error"]
    s = payload["summary"]
    assert s["n_reference_unconverged"] == 1
    assert s["n_failed_systems"] == 1
    assert any("did not converge" in r and "H2" in r
               for r in s["failure_reasons"])
    assert not any("could not be evaluated" in r for r in s["failure_reasons"])
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"


# ---------------------------------------------------------------------------
# Degenerate free atoms are evaluated on an orientation-locked density
# ---------------------------------------------------------------------------

def _free_atom(symbol, charge=0, spin=0):
    from xcquinox.alec.config import MoleculeSpec
    return MoleculeSpec(name=fid.atom_system_name(symbol, charge),
                        atom=f"{symbol} 0 0 0", basis="sto-3g", charge=charge,
                        spin=spin, atom_composition=((symbol, 1),))


def test_degenerate_atoms_are_the_open_p_shell_ones():
    """A P term (1, 2, 4 or 5 p electrons) is spatially degenerate; s-shell,
    half-filled and closed-shell atoms are spherical."""
    for symbol in ("B", "C", "O", "F", "Al", "Si", "S", "Cl"):
        assert fid.is_degenerate_atom(_free_atom(symbol)) is True, symbol
    for symbol in ("H", "Li", "Be", "N", "Na", "Mg", "P"):
        assert fid.is_degenerate_atom(_free_atom(symbol)) is False, symbol
    # Ions follow their electron count, not their element.
    assert fid.is_degenerate_atom(_free_atom("F", charge=-1)) is False
    assert fid.is_degenerate_atom(_free_atom("Cl", charge=-1)) is False
    assert fid.is_degenerate_atom(_free_atom("O", charge=-1)) is True
    assert fid.is_degenerate_atom(_free_atom("Na", charge=1)) is False
    # Molecules never are, whatever their atoms.
    assert fid.is_degenerate_atom(_tiny_oracle_set()[1]) is False
    with pytest.raises(ValueError, match="beyond argon"):
        fid.is_degenerate_atom(_free_atom("Kr"))


def test_every_pool_free_atom_is_classified():
    """The predicate must answer for every free atom the oracle set can carry
    (the pools' twelve elements plus Li and Na), with no refusal."""
    systems = fid.build_oracle_set(_cfg(), "deep_3x16")
    degenerate = {ms.name for ms in systems if fid.is_degenerate_atom(ms)}
    assert degenerate == {"atom_B", "atom_C", "atom_O", "atom_F", "atom_Al",
                          "atom_Si", "atom_S", "atom_Cl"}


def test_atom_lock_strength_is_the_production_default():
    from xcquinox.alec.orientation_lock import DEFAULT_STRENGTH
    assert fid.atom_orientation_lock_strength() == DEFAULT_STRENGTH == 3e-5


def _lock_spy(seen):
    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        seen[mol_spec.name] = orientation_lock_strength
        return _fake_evaluate({"atom_H": 0.1, "atom_O": 0.2, "H2": 0.3})(
            model, descriptors, mol_spec, parent=parent)
    return _evaluate


def test_certificate_locks_degenerate_atoms_when_the_run_lock_is_off(tmp_path):
    """With inputs.orientation_lock_strength = 0 the degenerate O atom is
    evaluated at the default lock; H (s shell) and H2 (a molecule) are not,
    and the payload names what was locked."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    systems = _tiny_oracle_set() + (_free_atom("O", spin=2),)
    seen = {}
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=systems,
        evaluate=_lock_spy(seen))
    assert seen == {"atom_H": 0.0, "H2": 0.0,
                    "atom_O": fid.atom_orientation_lock_strength()}
    lock = payload["atom_orientation_lock"]
    assert lock["applied_to"] == ["atom_O"]
    assert lock["strength"] == fid.atom_orientation_lock_strength()
    assert lock["run_orientation_lock_strength"] == 0.0
    assert "orientation" in lock["note"]
    assert payload["identity"]["orientation_lock_strength"] == 0.0


def test_the_certificate_atom_lock_of_an_unlocked_run_is_the_calibrated_one(
        tmp_path):
    """A run that WAIVES the degenerate-atom refusal and stays unlocked has
    its pretraining rows built at 0.0, while the certificate still measures a
    degenerate free atom at ``orientation_lock.DEFAULT_STRENGTH``.

    The certificate's rule is deliberate -- an unlocked atomic E_xc depends on
    which orientation the SCF happened to reach, so the bound would not be a
    measurement -- and it is left alone. The consequence is pinned here rather
    than left to be rediscovered: for exactly the atoms whose degeneracy
    motivates the lock, such a run is certified against a density it was not
    pretrained on. No shipped configuration produces the combination (each one
    states the calibrated lock), and it is reachable only through the YAML
    waiver.

    The configuration is built through ``load_grid_config`` so the two halves
    of the statement -- what the harness accepts and what the certificate then
    does -- are joined on one object."""
    import yaml
    from xcquinox.alec.cluster.grid_config import load_grid_config
    from xcquinox.alec.orientation_lock import DEFAULT_STRENGTH
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    raw = _minimal_raw_config(archs=["deep_3x16"])
    raw["inputs"]["orientation_lock_strength"] = 0.0
    raw["inputs"]["allow_irreproducible_degenerate"] = True
    raw["inputs"]["irreproducible_degenerate_reason"] = (
        "an unlocked coarse identity, stated deliberately")
    cfg_path = tmp_path / "grid.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_grid_config(str(cfg_path))
    assert cfg.inputs.allow_irreproducible_degenerate is True
    assert cfg.inputs.orientation_lock_strength == 0.0

    systems = _tiny_oracle_set() + (_free_atom("O", spin=2),)
    seen = {}
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16", oracle_set=systems,
        evaluate=_lock_spy(seen))
    assert seen["atom_O"] == DEFAULT_STRENGTH
    assert seen["atom_O"] != cfg.inputs.orientation_lock_strength
    assert payload["atom_orientation_lock"]["strength"] == DEFAULT_STRENGTH
    assert payload["identity"]["orientation_lock_strength"] == 0.0


def test_certificate_keeps_the_runs_own_lock_when_it_is_on(tmp_path):
    """A run that locks orientations itself applies its strength to every
    system, degenerate atom or not; the certificate adds nothing."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg()
    cfg.inputs.orientation_lock_strength = 0.02
    systems = _tiny_oracle_set() + (_free_atom("O", spin=2),)
    seen = {}
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16", oracle_set=systems,
        evaluate=_lock_spy(seen))
    assert seen == {"atom_H": 0.02, "H2": 0.02, "atom_O": 0.02}
    lock = payload["atom_orientation_lock"]
    assert lock["applied_to"] == []
    # "strength" records what the degenerate atoms were actually evaluated
    # at: the run's own lock here (the pre-correction payloads recorded 0.0
    # for locked runs while every system was in fact evaluated at 0.02).
    assert lock["strength"] == 0.02
    assert lock["run_orientation_lock_strength"] == 0.02
    assert payload["identity"]["orientation_lock_strength"] == 0.02


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_selects_the_arch_by_index_and_returns_zero_on_pass(
        tmp_path, monkeypatch):
    import yaml
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw = _minimal_raw_config(archs=["deep", "medium", "shallow"])
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(raw, f)

    seen = {}

    def _fake(cfg, rd, arch_name, **kwargs):
        seen["arch"] = arch_name
        return {"verdict": "PASS", "enforced": True,
                "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                               "override_reason": None},
                "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2,
                            "n_systems": 2, "n_atoms": 1,
                            "n_atomizations": 1,
                            "failure_reasons": []}}

    monkeypatch.setattr(fid, "fidelity_certificate", _fake)
    assert fid.main([str(run_dir), "1"]) == 0
    assert seen["arch"] == "medium"


def test_main_returns_zero_on_a_failed_but_unenforced_certificate(
        tmp_path, monkeypatch):
    import yaml
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(_minimal_raw_config(archs=["deep"]), f)
    monkeypatch.setattr(fid, "fidelity_certificate", lambda *a, **k: {
        "verdict": "FAIL", "enforced": False,
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": "workflow matrix"},
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7,
                    "n_systems": 2, "n_atoms": 1, "n_atomizations": 1,
                    "failure_reasons": ["max_atom_mHa"]}})
    assert fid.main([str(run_dir), "0"]) == 0


def test_main_returns_one_on_a_failed_certificate(tmp_path, monkeypatch):
    import yaml
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(_minimal_raw_config(archs=["deep"]), f)
    monkeypatch.setattr(fid, "fidelity_certificate", lambda *a, **k: {
        "verdict": "FAIL", "enforced": True,
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": None},
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7,
                    "n_systems": 2, "n_atoms": 1, "n_atomizations": 1,
                    "failure_reasons": ["max_atom_mHa"]}})
    assert fid.main([str(run_dir), "0"]) == 1


def test_main_rejects_an_out_of_range_arch_index(tmp_path):
    import yaml
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(_minimal_raw_config(archs=["deep"]), f)
    assert fid.main([str(run_dir), "7"]) == 1


def test_main_reports_a_missing_config(tmp_path):
    assert fid.main([str(tmp_path), "0"]) == 1


def _minimal_raw_config(archs):
    """A complete-but-minimal raw config dict load_grid_config accepts."""
    return {
        "sweep": {"arch": list(archs), "loss": ["l2"], "metric": ["l2"],
                  "subset_size": [1], "solver": ["oneshot"]},
        "solvers": {"oneshot": {"mode": "oneshot", "max_cycles": 1}},
        "hyperparams": {"n_steps": 1, "lr_start": 1e-3, "lr_end": 1e-4,
                        "lr_decay_start": 0.5, "grad_clip": 1.0,
                        "gradnorm_alpha": 1.0, "vxc_weight": 1.0,
                        "density_weight": 1.0},
        "inputs": {"external_refs_dir": "/tmp/refs",
                   "subset_ledger_path": "/tmp/ledger.json",
                   "basis": "sto-3g", "grid_level": 1,
                   "output_root": "/tmp/out"},
        "pretrain": {"data_dir": "/tmp/pretrain_data"},
        "cluster": {"partition": "short", "time": "01:00:00", "mem": "8G",
                    "cpus_per_task": 1, "array_throttle": 1,
                    "eval_array_throttle": 1, "max_concurrent_tasks": 10},
        "domain_profile": "dfs_step7",
    }


# ---------------------------------------------------------------------------
# A non-finite measurement can satisfy no tolerance
# ---------------------------------------------------------------------------

def _strict_json(path):
    """Parse ``path`` under RFC 8259 rules: a NaN / Infinity token is refused
    (python's default json.load would silently accept the bare tokens its own
    default dump emits)."""
    def _refuse(token):
        raise AssertionError(f"non-RFC-8259 token in certificate: {token}")
    with open(path) as f:
        return json.load(f, parse_constant=_refuse)


def _two_atom_set():
    return (_free_atom("H", spin=1), _free_atom("O", spin=2))


@pytest.mark.parametrize("order", ["nan-first", "nan-last"])
def test_certificate_fails_on_a_nan_measurement_wherever_it_sits(tmp_path,
                                                                 order):
    """``nan > tol`` is False and ``max()`` returns NaN or swallows it
    depending on its slot, so an unguarded gate PASSes a NaN and its verdict
    depends on the oracle-set order. A non-finite measurement must be a named
    failure BEFORE any comparison, and the finite 500 mHa atom must still be
    caught whichever side of the NaN it sits on."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    systems = _two_atom_set()
    if order == "nan-last":
        systems = tuple(reversed(systems))
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=systems,
        evaluate=_fake_evaluate({"atom_H": float("nan"), "atom_O": 500.0}))
    assert payload["verdict"] == "FAIL"
    by_name = {r["name"]: r for r in payload["per_system"]}
    assert by_name["atom_H"]["non_finite"] == ["E_xc_nn", "dE_xc_mHa"]
    assert by_name["atom_H"]["E_xc_nn"] is None
    assert by_name["atom_H"]["dE_xc_mHa"] is None
    assert "non_finite" not in by_name["atom_O"]
    s = payload["summary"]
    # The NaN never entered max(): the finite atom's 500 mHa is the maximum.
    assert s["max_atom_mHa"] == pytest.approx(500.0)
    assert s["n_non_finite_systems"] == 1
    reasons = s["failure_reasons"]
    assert any("non-finite" in r and "atom_H" in r and "E_xc_nn" in r
               and "dE_xc_mHa" in r for r in reasons), reasons
    assert any("tol_atom" in r and "500.0" in r for r in reasons), reasons
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"
    _strict_json(fid.certificate_path(run_dir, "deep_3x16"))


def test_certificate_fails_on_a_nan_route_difference(tmp_path):
    """The route-consistency check is as blind to NaN as the tolerances:
    a non-finite route difference is the same named failure, and the finite
    routes still report their true maximum."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)

    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        rec = _fake_evaluate({"atom_H": 0.1, "H2": 0.2})(
            model, descriptors, mol_spec, parent=parent)
        if mol_spec.name == "H2":
            rec["parent_grid_diff_Ha"] = float("nan")
        return rec

    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_evaluate)
    assert payload["verdict"] == "FAIL"
    by_name = {r["name"]: r for r in payload["per_system"]}
    assert by_name["H2"]["non_finite"] == ["parent_grid_diff_Ha"]
    assert by_name["H2"]["parent_grid_diff_Ha"] is None
    s = payload["summary"]
    assert s["max_parent_grid_diff_Ha"] == pytest.approx(0.0)
    assert any("non-finite" in r and "H2" in r and "parent_grid_diff_Ha" in r
               for r in s["failure_reasons"])
    assert not any("fresh-grid" in r for r in s["failure_reasons"])
    _strict_json(fid.certificate_path(run_dir, "deep_3x16"))


def test_certificate_with_a_nan_weight_checkpoint_fails_and_writes_strict_json(
        tmp_path):
    """The production shape of the defect: a diverged pretraining leaves NaN
    weights in the checkpoint, every E_xc_nn is NaN, and an unguarded
    certificate would PASS it and emit a bare NaN token no RFC 8259 parser
    accepts. The real path must FAIL by name and the written file must parse
    strictly."""
    import equinox as eqx
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir

    run_dir = str(tmp_path / "run")
    arch = get_architecture("deep_3x16")
    xnet, cnet = create_network_pair(arch, seed=0)
    params, static = eqx.partition(xnet, eqx.is_inexact_array)
    params = jax.tree_util.tree_map(lambda a: jnp.full_like(a, jnp.nan),
                                    params)
    d = pretrain_checkpoint_dir(run_dir, "deep_3x16")
    os.makedirs(d, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(d, "xnet.eqx"),
                              eqx.combine(params, static))
    eqx.tree_serialise_leaves(os.path.join(d, "cnet.eqx"), cnet)

    payload = fid.fidelity_certificate(_cfg(pretrain_seed=0), run_dir,
                                       "deep_3x16",
                                       oracle_set=_tiny_oracle_set())
    assert payload["verdict"] == "FAIL"
    for rec in payload["per_system"]:
        assert "E_xc_nn" in rec["non_finite"], rec["name"]
        assert "dE_xc_mHa" in rec["non_finite"], rec["name"]
        assert rec["E_xc_nn"] is None
        # The parent side is untouched by the network's NaN.
        assert rec["E_xc_parent"] is not None
        assert abs(rec["parent_grid_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
    s = payload["summary"]
    assert s["n_non_finite_systems"] == 2
    assert s["max_atom_mHa"] is None
    assert any("non-finite" in r and "atom_H" in r and "H2" in r
               for r in s["failure_reasons"])
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"
    allowed, _message = fid.gate_certificate(run_dir, "deep_3x16")
    assert allowed is False
    _strict_json(fid.certificate_path(run_dir, "deep_3x16"))


def test_certificate_fails_on_a_non_finite_atomization_offset(tmp_path):
    """Every per-system measurement is finite and the FOLD still overflows.

    The atomization offset is a difference of per-system offsets, so a set
    whose members are each representable can still produce an infinite dAE:
    the molecule at +1e308 mHa against two free atoms at -1e308 mHa each
    overflows the binade in the doubled atom term, before the subtraction.
    The per-record nulling cannot see such a value -- it is formed afterwards
    -- so the fold carries its own finiteness check, and the offset is nulled
    and named before it can enter a maximum, a tolerance comparison, or the
    written file.
    """
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": -1e308, "H2": 1e308}))
    assert payload["verdict"] == "FAIL"
    # Both per-system records are finite: the defect is in the fold alone.
    for rec in payload["per_system"]:
        assert "non_finite" not in rec, rec["name"]
        assert math.isfinite(rec["dE_xc_mHa"]), rec["name"]
    fold = {r["name"]: r for r in payload["per_atomization"]}
    assert fold["H2"]["dAE_kcalmol"] is None
    assert "not finite" in fold["H2"]["error"]
    s = payload["summary"]
    assert s["max_dAE_kcalmol"] is None
    assert s["n_atomizations"] == 0
    assert s["n_non_finite_systems"] == 1
    reasons = s["failure_reasons"]
    assert any("non-finite" in r and "H2" in r and "dAE_kcalmol" in r
               for r in reasons), reasons
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"
    _strict_json(fid.certificate_path(run_dir, "deep_3x16"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_the_certificate_writer_refuses_a_non_finite_payload(tmp_path, value):
    """The writer is the last gate before the file, and it refuses.

    ``json.dump``'s default serializer emits the bare tokens ``NaN`` /
    ``Infinity``, which RFC 8259 does not define: a strict reader refuses the
    whole certificate, and a lenient one round-trips a number no tolerance can
    act on. Serializing with ``allow_nan=False`` first raises where a
    non-finite value escaped the per-record nulling -- a defect in the
    producer, not a verdict -- and leaves no file for a reader to act on.
    """
    path = str(tmp_path / fid.CERTIFICATE_FILENAME)
    with pytest.raises(ValueError):
        fid._write_certificate_payload(
            {"verdict": "PASS", "summary": {"max_atom_mHa": value}}, path)
    assert not os.path.exists(path)
    # No temporary file is left behind either.
    assert os.listdir(tmp_path) == []
    # The same writer still writes a finite payload, strictly parseable.
    fid._write_certificate_payload(
        {"verdict": "PASS", "summary": {"max_atom_mHa": 0.1}}, path)
    assert _strict_json(path)["summary"]["max_atom_mHa"] == 0.1


# ---------------------------------------------------------------------------
# The producer's own non-convergence refusal is counted by name
# ---------------------------------------------------------------------------

def test_certificate_counts_a_reference_the_producer_refused(tmp_path,
                                                             monkeypatch):
    """``precompute_fixed_density_data`` raises data.ReferenceSCFNotConverged
    (a RuntimeError) instead of returning an unconverged record. The
    certificate must land it on the SAME branch as its own stamped-record
    refusal -- converged flag, cycle count, n_reference_unconverged and the
    named reason -- not in the generic 'could not be evaluated' bucket.
    Driven through the REAL producer with the SCF cycle budgets forced to
    exhaustion."""
    import xcquinox.alec.data as data_mod
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_MAX_CYCLE", 1)
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_NEWTON_MAX_CYCLE", 0)
    h2 = _tiny_oracle_set()[1]
    with pytest.raises(data_mod.ReferenceSCFNotConverged) as info:
        data_mod.precompute_fixed_density_data(h2, reference_xc="pbe")
    expected_cycles = info.value.cycles
    assert expected_cycles is not None

    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_3x16", seed=0)
    payload = fid.fidelity_certificate(_cfg(pretrain_seed=0), run_dir,
                                       "deep_3x16", oracle_set=(h2,))
    assert payload["verdict"] == "FAIL"
    entry = payload["per_system"][0]
    assert entry["name"] == "H2"
    assert "did not converge" in entry["error"]
    assert entry["reference_scf_converged"] is False
    assert entry["reference_scf_cycles"] == expected_cycles
    s = payload["summary"]
    assert s["n_reference_unconverged"] == 1
    assert any("did not converge" in r and "H2" in r
               for r in s["failure_reasons"])
    assert not any("could not be evaluated" in r for r in s["failure_reasons"])
    _strict_json(fid.certificate_path(run_dir, "deep_3x16"))


# ---------------------------------------------------------------------------
# Per-system containment, diagnostics and config hygiene
# ---------------------------------------------------------------------------

def test_a_beyond_argon_atom_is_a_per_system_failure_not_an_abort(tmp_path):
    """The degeneracy rule refuses elements beyond argon; that refusal must
    be a per-system failure with the certificate still written, not an abort
    that leaves no file (an absent certificate says nothing about the systems
    that were evaluable)."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    systems = (_free_atom("Kr"),) + _tiny_oracle_set()
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=systems,
        evaluate=_fake_evaluate({"atom_H": 0.1, "H2": 0.2}))
    assert payload["verdict"] == "FAIL"
    by_name = {r["name"]: r for r in payload["per_system"]}
    assert "beyond argon" in by_name["atom_Kr"]["error"]
    assert by_name["atom_H"]["dE_xc_mHa"] == pytest.approx(0.1)
    assert by_name["H2"]["dE_xc_mHa"] == pytest.approx(0.2)
    assert any("could not be evaluated" in r and "atom_Kr" in r
               for r in payload["summary"]["failure_reasons"])
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"


def test_certificate_reports_the_boundary_value_at_full_precision(tmp_path):
    """1.0000000000000002 mHa exceeds tol_atom = 1.0 by one ulp; a reason
    printed at four decimals would read '1.0000 mHa exceeds tol_atom 1.0',
    an apparent contradiction. The raw value is printed."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 1.0000000000000002, "H2": 0.5}))
    assert payload["verdict"] == "FAIL"
    assert any("tol_atom" in r and "1.0000000000000002" in r
               for r in payload["summary"]["failure_reasons"]), \
        payload["summary"]["failure_reasons"]


@pytest.mark.parametrize("enforce", ["false", 0, 1, None])
def test_certificate_refuses_a_non_boolean_enforce(tmp_path, enforce):
    """``bool("false")`` is True and ``bool(None)`` is False: a coerced
    ``enforced`` field would record an enforcement state no configuration
    asked for, on the one flag that decides whether a FAIL releases a gate.
    Only a real boolean is recorded; anything else is refused before any
    system is evaluated and no certificate is written."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    seen = {}
    with pytest.raises(ValueError, match="enforce"):
        fid.fidelity_certificate(
            _cfg(enforce=enforce), run_dir, "deep_3x16",
            oracle_set=_tiny_oracle_set(), evaluate=_lock_spy(seen))
    assert seen == {}
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "MISSING"


def test_route_disagreement_reasons_name_the_system(tmp_path):
    """A route inconsistency is actionable only if the reason says WHERE:
    each offending system is named with its raw difference, per route."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)

    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        rec = _fake_evaluate({"atom_H": 0.1, "H2": 0.2})(
            model, descriptors, mol_spec, parent=parent)
        if mol_spec.name == "atom_H":
            rec["parent_grid_diff_Ha"] = 2e-6
        if mol_spec.name == "H2":
            rec["parent_record_diff_Ha"] = 3e-6
        return rec

    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_evaluate)
    assert payload["verdict"] == "FAIL"
    reasons = payload["summary"]["failure_reasons"]
    grid = [r for r in reasons if "fresh-grid" in r]
    record = [r for r in reasons if "accumulated" in r]
    assert len(grid) == 1 and len(record) == 1, reasons
    assert "atom_H" in grid[0] and "2e-06" in grid[0]
    assert "H2" not in grid[0]
    assert "H2" in record[0] and "3e-06" in record[0]
    assert "atom_H" not in record[0]


def test_version_cross_check_skips_the_unknown_fallback(tmp_path, monkeypatch,
                                                        capsys):
    """versioneer's '1+unknown' fallback carries no code identity: on the
    cluster every certificate AND manifest records it (21 of 21 production
    certificates), so equality there was vacuous, and a local recovery under
    a versioned tree FALSELY refused correct, digest-matching cluster
    artifacts. The comparison now fires only when BOTH sides carry real
    versions; an unknown side is reported as a warning, and the digest +
    identity + parent checks carry the provenance load."""
    cert = {"verdict": "PASS", "arch": "deep_3x16", "parent_anchor": True,
            "descriptor_coordinates": "dfs", "descriptor_log_transform": True,
            "xcquinox_version": "1+unknown"}
    monkeypatch.setattr(fid, "running_xcquinox_version",
                        lambda: "1.0.0+319.gdeadbee")
    findings = fid.certificate_describes_run(
        _cfg(), str(tmp_path), "deep_3x16", cert)
    assert not [f for f in findings if "xcquinox_version" in f], findings
    assert "no code identity" in capsys.readouterr().out
    # Both sides REAL and different: still a refusal finding.
    cert2 = dict(cert, xcquinox_version="0.9.0+100.gaaaaaaa")
    findings2 = fid.certificate_describes_run(
        _cfg(), str(tmp_path), "deep_3x16", cert2)
    assert [f for f in findings2 if "xcquinox_version" in f]
