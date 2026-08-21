"""Tests for xcquinox.alec.cluster.fidelity -- the per-architecture physics
certificate.

The cheap layer (the certificate predicate, the parent resolution, the oracle
set) is tested directly. The certificate itself is tested twice: once with the
per-system evaluation monkeypatched at the ``evaluate`` seam, so the verdict
arithmetic and the JSON schema are exercised with no SCF at all, and once for
REAL on H and H2 at sto-3g with networks built in the test, so the energy path,
the libxc parent route and the atomization fold are pinned against physics.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

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
})


def _module_body_imports(path, package):
    """Absolute module names imported by ONE file's module body."""
    import ast
    with open(path) as f:
        tree = ast.parse(f.read())
    out = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                out.append(base + "." + node.module if node.module else base)
            else:
                out.append(node.module or "")
    return out


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

    Checked on the source: importing any cluster module already executes the
    package's own jax-carrying __init__, so sys.modules cannot distinguish
    this file's cost from the package's."""
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
        imports = _module_body_imports(path, package)
        assert imports, "%s imports nothing at all" % name
        for imported in imports:
            closure.append(imported)
            root = imported.split(".")[0]
            assert root not in _HEAVY_IMPORT_ROOTS, (name, imported)
            if root == "xcquinox":
                assert imported in _CHEAP_XCQ_MODULES, (name, imported)
                origin = importlib.util.find_spec(imported).origin
                queue.append((imported, origin))
    for deferred in ("xcquinox.alec.rungs", "xcquinox.alec.config",
                     "xcquinox.alec.full_benchmark_pools",
                     "xcquinox.alec.dfs_pretrain_set",
                     "xcquinox.alec.training_points"):
        assert deferred not in closure, deferred


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
