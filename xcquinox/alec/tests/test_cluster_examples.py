"""Tests for the shipped example grid configs and the campaign-v6 submission.

The example YAML ``cluster/examples/grid_step7.yaml`` is a copy-me template
reproducing the step-7 40-spec sweep. These tests verify it (a) still parses
with ``load_grid_config``, (b) expands to exactly 40 cells, (c) covers every
config-dataclass field (so a future required field the example forgot is
caught), (d) carries no real email/account, and (e) fails
``validate_grid_semantics`` cleanly because its placeholder paths do not
exist. The workflow-matrix template is held to the same protocol
completeness.

The final sections pin the deployment configurations those templates are
copied into: ``hpcjobs/configs/dfs_step7.dfs6311_grid3_v6.yaml``, the
pretraining-fidelity program's campaign, and the five standalone group files
the campaign is submitted as. Their properties are pinned rather than reviewed
because each way they can go wrong -- an architecture added to the registry
after they were written, an architecture in two groups or in none, a waiver
carried over from a template, a loosened certificate tolerance, a pre-protocol
pretraining footing, a group that acquired a name from the other rung -- loads
without complaint and is invisible in a result.
"""
import dataclasses
import os

import pytest

from xcquinox.alec.cluster.grid_config import (
    GridConfig,
    SweepAxes,
    SolverNamed,
    HyperParams,
    InputPaths,
    PretrainConfig,
    ClusterResources,
    FidelityConfig,
    load_grid_config,
    expand_grid,
    validate_grid_semantics,
)
from xcquinox.alec.cluster.domain import get_domain_profile


# ---------------------------------------------------------------------------
# Locating the shipped example
# ---------------------------------------------------------------------------

def _example_path() -> str:
    """Absolute path to the shipped ``grid_step7.yaml`` example."""
    import xcquinox.alec.cluster as cluster_pkg
    pkg_dir = os.path.dirname(os.path.abspath(cluster_pkg.__file__))
    return os.path.join(pkg_dir, "examples", "grid_step7.yaml")


def test_example_yaml_exists():
    assert os.path.isfile(_example_path()), (
        f"shipped example grid_step7.yaml not found at {_example_path()}"
    )


def test_examples_dir_has_no_init():
    """examples/ ships as package DATA, not as a subpackage, no __init__.py."""
    examples_dir = os.path.dirname(_example_path())
    assert not os.path.exists(os.path.join(examples_dir, "__init__.py")), (
        "cluster/examples/ must NOT contain __init__.py: it is package data"
    )


# ---------------------------------------------------------------------------
# Load + expand (filesystem-free, no validate_grid_semantics here)
# ---------------------------------------------------------------------------

def test_example_loads_and_expands_to_40():
    """grid_step7.yaml loads via load_grid_config and expands to 40 specs.

    This is filesystem-free: it calls only load_grid_config + expand_grid.
    validate_grid_semantics is NOT called, the example's placeholder input
    paths intentionally do not exist (see the separate raises test).
    """
    pytest.importorskip("yaml")
    cfg = load_grid_config(_example_path())
    assert isinstance(cfg, GridConfig)
    cells = expand_grid(cfg)
    assert len(cells) == 40, (
        f"example grid expanded to {len(cells)} cells, expected 40 "
        "(10 subset sizes x 2 metrics x 2 solvers)"
    )


def test_example_reproduces_step7_grid_parameters():
    """The example matches the step-7 grid the notebook builds."""
    pytest.importorskip("yaml")
    cfg = load_grid_config(_example_path())
    assert cfg.sweep.arch == ("deep_combined_attn",)
    assert cfg.sweep.loss == ("L5_gradnorm_vxc_step7",)
    assert set(cfg.sweep.metric) == {"l2", "jsd"}
    assert set(cfg.sweep.subset_size) == {1, 2, 3, 4, 5, 6, 7, 12, 15, 18}
    assert set(cfg.sweep.solver) == {"oneshot", "full_3"}
    # named solvers
    assert cfg.solvers["oneshot"].mode == "ONESHOT"
    assert cfg.solvers["oneshot"].max_cycles == 0
    assert cfg.solvers["full_3"].mode == "FULL"
    assert cfg.solvers["full_3"].max_cycles == 3
    assert cfg.solvers["full_3"].feature_policy == "REASSEMBLE"
    # hyperparameters
    hp = cfg.hyperparams
    assert hp.n_steps == 100
    assert hp.lr_start == pytest.approx(1e-2)
    assert hp.lr_end == pytest.approx(1e-5)
    assert hp.lr_decay_start == pytest.approx(0.2)
    assert hp.grad_clip == pytest.approx(1.0)
    assert hp.gradnorm_alpha == pytest.approx(1.5)
    assert hp.vxc_weight == pytest.approx(0.01)
    assert hp.density_weight == pytest.approx(0.1)
    assert hp.pbe_anchor_weight == pytest.approx(0.0)
    # inputs
    assert cfg.inputs.basis == "def2-svp"
    assert cfg.inputs.grid_level == 1
    # domain profile is a real registered profile
    assert cfg.domain_profile == "dfs_step7"
    get_domain_profile(cfg.domain_profile)  # raises if unregistered
    assert cfg.bh76_mode == "reaction_energy"
    assert cfg.on_precompute_failure == "abort"


# ---------------------------------------------------------------------------
# Structural completeness, every dataclass field is covered
# ---------------------------------------------------------------------------

def _raw_yaml(path: str) -> dict:
    """Parse the example YAML to its raw dict (pre-dataclass)."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _assert_fields_covered(dc_type, raw_section, ctx):
    """Every dataclasses.field of ``dc_type`` is either present in
    ``raw_section`` (the parsed YAML mapping) or has a dataclass default.

    This catches a future required field added to a config dataclass that the
    shipped example forgot to fill in, without a default such a field would
    make the example unloadable, and this test pinpoints which field.
    """
    for f in dataclasses.fields(dc_type):
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        present = isinstance(raw_section, dict) and f.name in raw_section
        assert present or has_default, (
            f"{ctx}: config field {dc_type.__name__}.{f.name!r} is required "
            "(no dataclass default) but is absent from the example "
            "grid_step7.yaml: the example must be updated to set it"
        )


def test_example_structural_completeness():
    """Every field of GridConfig and every nested config dataclass is either
    set in the example YAML or has a dataclass default."""
    pytest.importorskip("yaml")
    raw = _raw_yaml(_example_path())

    # top-level GridConfig fields map to YAML sections / scalars
    _assert_fields_covered(GridConfig, raw, "GridConfig")
    # nested sections
    _assert_fields_covered(SweepAxes, raw.get("sweep"), "sweep")
    _assert_fields_covered(HyperParams, raw.get("hyperparams"), "hyperparams")
    _assert_fields_covered(InputPaths, raw.get("inputs"), "inputs")
    _assert_fields_covered(PretrainConfig, raw.get("pretrain"), "pretrain")
    _assert_fields_covered(ClusterResources, raw.get("cluster"), "cluster")
    _assert_fields_covered(FidelityConfig, raw.get("fidelity"), "fidelity")
    # every named solver covers SolverNamed's fields
    solvers = raw.get("solvers") or {}
    assert solvers, "example has no 'solvers' section"
    for name, sd in solvers.items():
        _assert_fields_covered(SolverNamed, sd, f"solvers.{name}")


# ---------------------------------------------------------------------------
# validate_grid_semantics raises on the placeholder paths
# ---------------------------------------------------------------------------

def test_example_validate_raises_on_placeholder_paths():
    """The example's unfilled placeholder input paths do not exist, so a
    semantic-validation pass surfaces them, either as a hard error or via a
    warning (the path checks in validate_grid_semantics are advisory).

    The example is a copy-me template; this asserts it is NOT silently usable
    as-is. We accept EITHER a raised error OR an emitted UserWarning naming a
    not-found path.
    """
    pytest.importorskip("yaml")
    import warnings

    cfg = load_grid_config(_example_path())
    domain = get_domain_profile(cfg.domain_profile)

    # The placeholder paths must genuinely not exist on this machine.
    for p in (
        cfg.inputs.external_refs_dir,
        cfg.inputs.subset_ledger_path,
        cfg.inputs.output_root,
        cfg.pretrain.data_dir,
    ):
        assert p is not None and not os.path.exists(p), (
            f"example placeholder path {p!r} unexpectedly exists, the "
            "example must use clearly-non-existent CHANGE_ME paths"
        )

    raised = None
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_grid_semantics(cfg, domain)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raised = exc

    if raised is not None:
        msg = str(raised).lower()
        assert ("not found" in msg or "exist" in msg or "no such" in msg), (
            f"validate raised, but the message does not mention a missing "
            f"path: {raised!r}"
        )
    else:
        # No hard error, then it MUST have warned about a not-found path.
        path_warnings = [
            w for w in caught
            if "not found" in str(w.message).lower()
            or "does not exist" in str(w.message).lower()
        ]
        assert path_warnings, (
            "validate_grid_semantics neither raised nor warned about the "
            "example's non-existent placeholder paths"
        )


# ---------------------------------------------------------------------------
# No real credentials committed
# ---------------------------------------------------------------------------

def test_example_has_no_real_credentials():
    """The example carries CHANGE_ME placeholders, not a real email/account."""
    with open(_example_path()) as f:
        text = f.read()
    assert "CHANGE_ME" in text, (
        "example must use CHANGE_ME placeholders for user-specific fields"
    )
    lowered = text.lower()
    for forbidden in ("@gmail.com", "@yahoo.com", "@stonybrook.edu",
                      "alec.p.wills"):
        assert forbidden not in lowered, (
            f"example grid_step7.yaml contains what looks like a real "
            f"credential ({forbidden!r}), it must ship only placeholders"
        )
    # mail_user is the example.com placeholder
    cfg = load_grid_config(_example_path()) if pytest.importorskip("yaml") \
        else None
    assert cfg.cluster.mail_user == "CHANGE_ME@example.com"
    assert cfg.cluster.account == "CHANGE_ME"


def test_example_ships_the_binding_certificate_tolerances():
    """The shipped template must carry the program's binding tolerances --
    tol_AE = 1.0 kcal/mol, tol_atom = 1.0 mHa -- explicitly, so a copy-me user
    sees them rather than inheriting an invisible default."""
    pytest.importorskip("yaml")
    raw = _raw_yaml(_example_path())
    assert raw.get("fidelity") is not None, (
        "grid_step7.yaml must ship a 'fidelity' block")
    assert raw["fidelity"]["tol_AE"] == 1.0
    assert raw["fidelity"]["tol_atom"] == 1.0
    assert raw["fidelity"]["override_reason"] is None
    assert raw["fidelity"]["enforce"] is True
    cfg = load_grid_config(_example_path())
    assert cfg.fidelity.tol_AE == 1.0
    assert cfg.fidelity.tol_atom == 1.0
    assert cfg.fidelity.enforce is True


def test_example_gives_datagen_a_wall_of_its_own():
    """Datagen builds the pretrain-data file(s) and every later stage waits on
    it, so it gets its own wall rather than inheriting the pretrain tier."""
    pytest.importorskip("yaml")
    cfg = load_grid_config(_example_path())
    assert cfg.cluster.datagen_time == "04:00:00"


def test_example_states_the_orientation_lock_explicitly():
    """The template names the run's lock rather than inheriting it.

    The value is authoritative for the whole run -- the training and eval SCF,
    the CCSD references and the pretraining data are all built at it -- and an
    unlocked degenerate open shell is not reproducible between processes, so a
    copy of the template is a complete statement of the Hamiltonian it solves
    rather than a set of invisible defaults."""
    pytest.importorskip("yaml")
    from xcquinox.alec.orientation_lock import DEFAULT_STRENGTH
    raw = _raw_yaml(_example_path())["inputs"]
    assert "orientation_lock_strength" in raw, (
        "grid_step7.yaml must state inputs.orientation_lock_strength")
    assert float(raw["orientation_lock_strength"]) == DEFAULT_STRENGTH
    cfg = load_grid_config(_example_path())
    assert cfg.inputs.orientation_lock_strength == DEFAULT_STRENGTH


# ---------------------------------------------------------------------------
# The shipped templates run at grid level 1, which no degenerate free atom is
# reproducible at
# ---------------------------------------------------------------------------

def _template_paths():
    """Both shipped templates: the copy-me sweep and the workflow matrix."""
    import xcquinox.alec.cluster as cluster_pkg
    pkg_dir = os.path.dirname(os.path.abspath(cluster_pkg.__file__))
    return [os.path.join(pkg_dir, "examples", name)
            for name in ("grid_step7.yaml", "workflow_matrix_template.yaml")]


@pytest.mark.parametrize("path", _template_paths())
def test_the_templates_waive_the_degenerate_refusal_with_a_reason(path):
    """Both templates are def2-svp at grid level 1 and their pretraining sets
    contain the O atom, whose 3P term that quadrature does not resolve, so the
    data generator refuses to build their rows unless the waiver is stated.
    Without it a copy of the canonical template fails its datagen stage and
    every later stage goes ``DependencyNeverSatisfied``.

    The waiver carries prose, and the templates' prose says what they are: a
    verification identity, never a campaign. A production configuration runs
    at grid level 3, where the same rows reproduce to 3e-11 relative, and
    needs no waiver at all."""
    pytest.importorskip("yaml")
    raw = _raw_yaml(path)["inputs"]
    assert raw.get("allow_irreproducible_degenerate") is True, path
    reason = raw.get("irreproducible_degenerate_reason")
    assert isinstance(reason, str) and reason.strip(), path
    assert "example" in reason and "never a campaign" in reason
    cfg = load_grid_config(path)
    assert cfg.inputs.allow_irreproducible_degenerate is True
    assert cfg.inputs.irreproducible_degenerate_reason == reason


@pytest.mark.parametrize("path", _template_paths())
def test_the_templates_state_the_certificate_consequence_of_an_unlocked_waiver(
        path):
    """A waived run may also be UNLOCKED, and the fidelity certificate still
    evaluates a degenerate free atom at the calibrated lock while such a run's
    pretraining rows sit at 0.0. The comment beside the flag says so, because
    the two are set in different files and the mismatch is invisible from
    either one."""
    with open(path) as f:
        lines = f.read().splitlines()
    keys = [i for i, line in enumerate(lines)
            if line.strip().startswith("allow_irreproducible_degenerate:")]
    assert len(keys) == 1, (path, keys)
    comment = []
    i = keys[0] - 1
    while i >= 0 and lines[i].strip().startswith("#"):
        comment.insert(0, lines[i].strip().lstrip("#").strip())
        i -= 1
    text = " ".join(comment)
    assert "certificate" in text, (path, text)
    assert "0.0" in text, (path, text)


#: The pretraining-protocol knobs, with the value both shipped templates run
#: at. They are the pre-protocol defaults: the canonical template reproduces
#: the step-7 pretraining, and the workflow matrix is a wiring check that
#: moves no knob away from the default it is verifying the wiring of. The v6
#: value sits in a comment beside each one in the files.
_PROTOCOL_KNOBS = (("dfs_set", False), ("pool_atoms", False),
                   ("parent_density", "pbe"),
                   ("exchange_footing", "total"),
                   ("mesh_fraction", 0.3), ("energy_term_weight", 0.0),
                   ("validation_fraction", 0.0), ("validation_seed", 0),
                   ("validate_every", 50), ("patience", 0))


@pytest.mark.parametrize("path", _template_paths())
def test_the_templates_set_every_pretraining_protocol_field(path):
    """BOTH shipped templates name each pretraining-protocol knob explicitly,
    so a copy of either is a complete statement of the protocol it runs rather
    than a set of invisible defaults.

    The matrix template is held to the same completeness as the canonical one:
    it is the file every architecture's workflow verification is rendered
    from, and a knob it leaves unstated is a knob whose default can change
    under a verification that reports nothing about it."""
    pytest.importorskip("yaml")
    raw = _raw_yaml(path)["pretrain"]
    name = os.path.basename(path)
    for key, value in _PROTOCOL_KNOBS:
        assert key in raw, f"{name} is missing pretrain.{key}"
        assert raw[key] == value, (name, key, raw[key], value)
    cfg = load_grid_config(path)
    for key, value in _PROTOCOL_KNOBS:
        assert getattr(cfg.pretrain, key) == value, (name, key)


@pytest.mark.parametrize("path", _template_paths())
def test_the_templates_state_the_v6_value_beside_each_protocol_knob(path):
    """Each knob carries the campaign-v6 value in a comment on its own line,
    which is what makes the stated default a decision rather than an
    inheritance. Read from the text, since a comment is invisible to the
    parser.

    Only the ``pretrain:`` block is scanned: ``validate_every`` names a knob
    in ``hyperparams`` too (the training loop's validation period), and the
    two are different periods of different loops."""
    with open(path) as f:
        lines = f.read().splitlines()
    start = lines.index("pretrain:")
    end = next((i for i in range(start + 1, len(lines))
                if lines[i][:1] not in ("", " ", "#")), len(lines))
    block = lines[start:end]
    for key, _value in _PROTOCOL_KNOBS:
        stated = [ln for ln in block if ln.strip().startswith(f"{key}:")]
        assert len(stated) == 1, (path, key, stated)
        assert "# v6:" in stated[0], (path, key, stated[0])


# ---------------------------------------------------------------------------
# The campaign-v6 deployment configuration
#
# `hpcjobs/configs/dfs_step7.dfs6311_grid3_v6.yaml` is the submission the
# pretraining-fidelity program exists to produce: every architecture in the
# registry, the certified pretraining protocol, and the certificate enforced
# at the binding tolerances. It is pinned here rather than left to review
# because three of its properties are silent failures -- an architecture added
# to the registry after the file was written, a waiver copied in from a
# template, and a certificate tolerance loosened -- each of which loads
# without complaint and none of which is visible in a result.
# ---------------------------------------------------------------------------

#: The reference configuration: the campaign's statement of the method. It is
#: not itself submitted -- the five group files below are -- but every property
#: the groups inherit is asserted on it too, so the six files are pinned as one
#: family.
_V6_REFERENCE = "dfs_step7.dfs6311_grid3_v6.yaml"


def _campaign_config_path(name):
    """Absolute path to ``hpcjobs/configs/<name>``, or None when absent.

    ``hpcjobs/`` sits beside the package rather than inside it, so a source or
    wheel checkout can be missing it entirely; an absent file is nothing to
    pin, not a failure.
    """
    import xcquinox.alec.cluster as cluster_pkg
    pkg_dir = os.path.dirname(os.path.abspath(cluster_pkg.__file__))
    path = os.path.normpath(os.path.join(
        pkg_dir, "..", "..", "..", "hpcjobs", "configs", name))
    return path if os.path.isfile(path) else None


def _campaign_configs_dir():
    """Absolute path to ``hpcjobs/configs/``, or None when it is absent.

    The directory's presence is what separates "this checkout has no
    deployment tree" (nothing to pin) from "a configuration file was
    deleted" (a campaign that no longer covers what it claims).
    """
    import xcquinox.alec.cluster as cluster_pkg
    pkg_dir = os.path.dirname(os.path.abspath(cluster_pkg.__file__))
    path = os.path.normpath(os.path.join(
        pkg_dir, "..", "..", "..", "hpcjobs", "configs"))
    return path if os.path.isdir(path) else None


def _missing_campaign_files(names, configs_dir):
    """Those of ``names`` with no file under ``configs_dir``."""
    return [n for n in names
            if not os.path.isfile(os.path.join(configs_dir, n))]


def _campaign_config(name):
    """The loaded configuration ``name``, skipping when the deployment tree is
    absent."""
    pytest.importorskip("yaml")
    path = _campaign_config_path(name)
    if path is None:
        pytest.skip(f"no hpcjobs/configs/{name} in this checkout")
    return path, load_grid_config(path)


def _v6_config_path():
    """Absolute path to the campaign-v6 reference configuration, or None."""
    return _campaign_config_path(_V6_REFERENCE)


def _v6_config():
    """The loaded v6 reference config, skipping when it is absent."""
    return _campaign_config(_V6_REFERENCE)


def _flat_config(obj, prefix=""):
    """Flatten a config tree to ``{dotted key: value}``, defaults included.

    Two configurations compared through this differ in a field the second
    leaves unstated only when the DEFAULT differs, which is the comparison an
    inheritance claim has to survive.
    """
    out = {}
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            out.update(_flat_config(getattr(obj, f.name), f"{prefix}{f.name}."))
    elif isinstance(obj, dict):
        for key in sorted(obj):
            out.update(_flat_config(obj[key], f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def test_v6_loads():
    """The deployment configuration parses through the harness loader.

    A campaign YAML that does not load is discovered on the login node at
    submission time, which is the wrong side of the change that broke it.
    """
    path, cfg = _v6_config()
    assert isinstance(cfg, GridConfig), path


def test_v6_runs_the_production_identity_and_needs_no_waiver():
    """The production identity -- basis, grid level, Coulomb backend and the
    calibrated orientation lock -- the SCF seed that goes with it, and NO
    irreproducible-degenerate waiver.

    ``fidelity.run_identity`` records five fields on every certificate the run
    writes and ``validate_run`` refuses a certificate whose identity differs
    from the configuration's, so all five are asserted here rather than the
    two that carry the waiver argument. Basis and density fitting are the
    silent half: v6 shares its reference caches, its subset ledger and its
    comparison lineage with v3-v5, and a changed basis or Coulomb backend
    re-identifies all of them while still loading, submitting and running.
    ``auxbasis`` is asserted absent, which is what makes the certificate's
    ``null`` the run's own resolved value rather than a transcription.

    The waiver and the grid level go together. Below grid level 3, or with the
    lock at zero, a spatially degenerate free atom's pretraining rows are one
    arbitrary member of its manifold and the data generator refuses to build
    them without a written waiver; at grid level 3 with the lock on it refuses
    nothing, so a waiver stated here would authorise a build this run never
    performs and ``validate_grid_semantics`` rejects it outright. The shipped
    templates carry the waiver because they run at grid level 1; a production
    copy must have dropped it, and this asserts it did.

    ``seed_xc: auto`` is the rung-seeding contract v5 introduced and v6
    inherits: the SCF seed is derived per architecture through
    ``rungs.seed_xc_for_arch`` -- SCAN for the five meta-GGA-rung
    architectures of this sweep, PBE for the other 26 -- which is the SAME
    predicate ``pretrain.parent_density: auto`` resolves the pretraining
    parent with. The literal ``pbe`` is not a harmless spelling of it: it
    seeds every architecture's SCF from PBE while the meta-GGA five are
    pretrained against, and certified against, SCAN, which is exactly the
    decoupling the sibling pin's docstring claims cannot happen.

    Every field is read from the YAML TEXT as well as from the parse. Three of
    them are values the loader also supplies as a default, and each default is
    the pre-protocol one -- ``density_fit`` False, ``seed_xc`` "pbe",
    ``auxbasis`` None -- so a value silently inherited from a dataclass would
    otherwise read here as a stated decision.
    """
    from xcquinox.alec.orientation_lock import DEFAULT_STRENGTH
    path, cfg = _v6_config()
    raw = _raw_yaml(path)["inputs"]
    # The five fields fidelity.run_identity records, in its own order.
    assert str(raw["basis"]) == "6-311++G(3df,2pd)", path
    assert cfg.inputs.basis == "6-311++G(3df,2pd)"
    assert int(raw["grid_level"]) == 3, path
    assert cfg.inputs.grid_level == 3
    assert raw["density_fit"] is True, path
    assert cfg.inputs.density_fit is True
    assert "auxbasis" not in raw, (
        f"{path}: auxbasis is deliberately unset -- auto-selected from the "
        "orbital basis, and recorded on the certificate as null; naming one "
        "here changes the Coulomb backend the shared caches were built with")
    assert cfg.inputs.auxbasis is None
    assert float(raw["orientation_lock_strength"]) == DEFAULT_STRENGTH == 3e-5
    assert cfg.inputs.orientation_lock_strength == DEFAULT_STRENGTH
    # The identity the certificate carries IS the one above, field for field.
    from xcquinox.alec.cluster.fidelity import run_identity
    assert run_identity(cfg) == {
        "basis": "6-311++G(3df,2pd)",
        "grid_level": 3,
        "density_fit": True,
        "auxbasis": None,
        "orientation_lock_strength": 3e-5,
    }, path
    # The SCF seed is the rung baseline, per architecture, not one functional
    # for the whole sweep.
    assert raw["seed_xc"] == "auto", (
        f"{path}: seed_xc must be 'auto' -- the per-architecture rung "
        "baseline rungs.seed_xc_for_arch resolves, which is the predicate "
        "pretrain.parent_density: auto uses for the pretraining parent. A "
        "literal seeds every architecture from one functional while the "
        "meta-GGA rung is certified against the other")
    assert cfg.inputs.seed_xc == "auto"
    # ... and "auto" is not a synonym for either literal over this sweep: it
    # resolves to BOTH, so neither spelling could stand in for it.
    from xcquinox.alec.cluster.spec_builder import resolve_seed_xc
    seeds = {resolve_seed_xc(cfg.inputs, name) for name in cfg.sweep.arch}
    assert seeds == {"pbe", "scan"}, (path, sorted(seeds))
    assert "allow_irreproducible_degenerate" not in raw, (
        "the v6 campaign runs at grid level 3 with the lock on, where the "
        "data generator refuses nothing; a stated waiver grants a permission "
        "the run never exercises and is refused at submit")
    assert "irreproducible_degenerate_reason" not in raw, path
    assert cfg.inputs.allow_irreproducible_degenerate is False


def test_v6_enforces_the_certificate_at_the_binding_tolerances():
    """tol_AE = 1.0 kcal/mol, tol_atom = 1.0 mHa, enforced, no override.

    These are the program's binding decision (SPEC_pretrain_fidelity_program
    Section 7). ``enforce: false`` or a tolerance above 2.0 would each require
    a written ``override_reason`` that is copied into every certificate the
    run writes; a run carrying one can never become a quantitative result,
    because validate_run, merge_v4_arms and the figure suite refuse it. The
    values are asserted from the text as well as the parse, so a tolerance
    silently inherited from a dataclass default is not mistaken for a stated
    decision.
    """
    path, cfg = _v6_config()
    raw = _raw_yaml(path)
    assert raw.get("fidelity") is not None, (
        f"{path} must state a 'fidelity' block")
    assert raw["fidelity"]["tol_AE"] == 1.0
    assert raw["fidelity"]["tol_atom"] == 1.0
    assert raw["fidelity"]["enforce"] is True
    assert raw["fidelity"]["override_reason"] is None
    assert cfg.fidelity.tol_AE == 1.0
    assert cfg.fidelity.tol_atom == 1.0
    assert cfg.fidelity.enforce is True
    assert cfg.fidelity.override_reason is None


def test_v6_sweeps_every_registry_architecture():
    """The arch axis IS ``sorted(ARCHITECTURES)``, name for name.

    v6 is defined as "every architecture resubmitted" (spec Section 3.5), so
    the axis is compared against the registry rather than against a
    transcribed list: an architecture added to ``xcquinox.alec.ARCHITECTURES``
    after this file was written turns this test red, which is the only signal
    that the campaign no longer covers what it claims to. Equality is asserted
    in both directions -- a name on the axis that the registry does not carry
    fails ``validate_grid_semantics`` on the login node, but only after the
    figure layer has been told to expect it.
    """
    from xcquinox.alec import ARCHITECTURES
    path, cfg = _v6_config()
    registry = sorted(ARCHITECTURES)
    axis = list(cfg.sweep.arch)
    assert sorted(set(axis)) == registry, (
        f"{os.path.basename(path)} sweeps {len(set(axis))} architectures; the "
        f"registry carries {len(registry)}. Missing: "
        f"{sorted(set(registry) - set(axis))}; unknown: "
        f"{sorted(set(axis) - set(registry))}")
    assert len(axis) == len(set(axis)), (
        f"{path}: the arch axis carries a duplicate name; expand_grid would "
        f"drop it with a warning: {axis}")
    # The expansion the SLURM array indexes is the product of the canonical
    # axes, so the cell count is the campaign's size on record.
    cells = expand_grid(cfg)
    assert len(cells) == len(registry) * len(set(cfg.sweep.subset_size))
    assert sorted({c.arch for c in cells}) == registry


def test_v6_pretrains_on_the_corrected_footing_against_the_rung_parent():
    """``exchange_footing: spin_channel`` and ``parent_density: auto``.

    The footing is the correction of the defect the program was opened on:
    the production UKS exchange evaluates each spin channel at the doubled
    density diag(P_sigma, P_sigma), and rows posed on the total density fit a
    network to inputs its deployment never sees. ``auto`` gives each
    architecture its rung's parent -- PBE for a GGA-rung one, SCAN for a
    meta-GGA one -- through the same predicate ``inputs.seed_xc: auto``
    resolves the SCF seed with, so a meta-GGA architecture cannot be fitted
    against one functional while its SCF is seeded from the other. Both are
    read from the text as well as the parse: each is a value the loader also
    supplies as a default, and the default is the pre-protocol one.
    """
    path, cfg = _v6_config()
    raw = _raw_yaml(path)["pretrain"]
    assert raw["exchange_footing"] == "spin_channel", path
    assert raw["parent_density"] == "auto", path
    assert cfg.pretrain.exchange_footing == "spin_channel"
    assert cfg.pretrain.parent_density == "auto"
    # The set the two switches select is the one Section 7 binds.
    assert raw["dfs_set"] is True and raw["pool_atoms"] is True, path
    assert cfg.pretrain.dfs_set is True
    assert cfg.pretrain.pool_atoms is True


def test_v6_mixed_rung_sweep_derives_both_parent_data_files():
    """``parent_density: auto`` over this sweep requires TWO data files.

    The sweep mixes GGA-rung and meta-GGA-rung architectures, and the two
    parents' self-consistent densities are different densities rather than two
    views of one, so the datagen stage must derive a file per parent. This
    pins the derivation the pretrain worker later opens its file through.
    """
    from xcquinox.alec.cluster._datagen import _required_data_specs
    path, cfg = _v6_config()
    specs = _required_data_specs(cfg)
    parents = sorted({ref for _pol, ref in specs})
    assert parents == ["pbe", "scan"], (path, specs)
    # Polarized correlation is a run-level flag, so one polarization.
    assert sorted({pol for pol, _ref in specs}) == [True], (path, specs)


def test_v6_carries_the_stony_brook_job_mail():
    """Every rendered script mails submission, completion and failure.

    The directives are rendered from these two fields, so pinning them here
    pins them on all five stage scripts.
    """
    path, cfg = _v6_config()
    assert cfg.cluster.mail_user == "alec.wills@stonybrook.edu", path
    assert cfg.cluster.mail_type == "BEGIN,END,FAIL", path


def test_v6_differs_from_v5_in_exactly_the_fields_it_claims():
    """The header states a MEASURED diff -- 15 of 137 resolved fields -- and
    names all fifteen. It is the file's own claim that everything except the
    method, the model class, the roots and the walls is v5's, which is what
    makes v5-vs-v6 on the five meta-GGA architectures a controlled comparison;
    a hyperparameter or a solver knob that drifted in would break that reading
    while loading, submitting and running exactly as before.

    The two fields the parent anchor adds are part of the method, not
    bookkeeping: ``model.parent_anchor`` and ``model.descriptor_coordinates``
    each define a different model class (the networks' forward is the parent
    plus a correction rather than a correction to F = 1, and the MLPs read the
    row in the DFS coordinates rather than the committed ones), so a v5 result
    and a v6 result are read against different starting functionals. The field
    count moves from 135 to 137 with them.

    Both trees are flattened, so DEFAULTS are compared too: a field v5 leaves
    unstated and v6 states at the same value is not a difference, and one v6
    states at a different value is.
    """
    path6, cfg6 = _v6_config()
    path5 = os.path.join(os.path.dirname(path6),
                         "dfs_step7.dfs6311_grid3_v5.yaml")
    if not os.path.isfile(path5):
        pytest.skip("no v5 configuration in this checkout")
    cfg5 = load_grid_config(path5)

    a, b = _flat_config(cfg5), _flat_config(cfg6)
    keys = sorted(set(a) | set(b))
    differing = sorted(k for k in keys
                       if a.get(k, "<absent>") != b.get(k, "<absent>"))
    assert len(keys) == 137, len(keys)
    assert differing == [
        "cluster.datagen_time",
        "cluster.pretrain_throttle",
        "cluster.time",
        "cluster.timeout_retry_time",
        "inputs.output_root",
        "model.descriptor_coordinates",
        "model.parent_anchor",
        "pretrain.data_dir",
        "pretrain.dfs_set",
        "pretrain.exchange_footing",
        "pretrain.parent_density",
        "pretrain.patience",
        "pretrain.pool_atoms",
        "pretrain.validation_fraction",
        "sweep.arch",
    ], differing
    # ... and the file says so, in the count it states.
    text = open(path6).read()
    assert "FIFTEEN differ" in text, path6
    assert "The remaining 122 fields are identical" in text, path6


def _v6_semantics(cfg):
    """Run the login-node semantic check on ``cfg``, swallowing the advisory
    path warnings the campaign's ``/gpfs`` roots raise off-cluster."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validate_grid_semantics(cfg, get_domain_profile(cfg.domain_profile))


def _axis_carries_a_meta_gga_architecture(cfg):
    """Whether any architecture on the sweep axis is on the meta-GGA rung."""
    from xcquinox.alec import get_architecture
    from xcquinox.alec.config import ArchitectureConfig
    return any(ArchitectureConfig.is_meta_gga(get_architecture(a))
               for a in cfg.sweep.arch)


def test_v6_submits_at_the_login_node_semantic_check():
    """AS COMMITTED the file is ACCEPTED by the semantic check, on both rungs.

    The file states ``model.parent_anchor: true`` and its axis carries the
    meta-GGA architectures, whose parent is SCAN. The PBE-anchor commit
    refused exactly this combination because ``parents.scan_fx`` / ``scan_fc``
    did not exist yet (SPEC_parent_anchor.md Section 3.8 sequences PBE first,
    SCAN second); both now do, so the rung is no ground for refusal and the
    whole 31-architecture axis submits.

    The weight refusal does not fire either: under the anchor every network
    EQUALS its parent at initialization, both terms of the objective are zero
    to round-off before the first step and the certificate passes there, so
    ``energy_term_weight: 0.0`` is the exact statement of this run's objective
    rather than a placeholder awaiting a sweep. The refusal of the weight-zero
    combination is kept for an UNANCHORED run, where it was measured unable to
    deliver the parent (2.3 to 56.1 kcal/mol of atomization offset,
    SPEC_pretrain_fidelity_program.md Section 2), and that is asserted below
    by turning the anchor off on the same configuration -- so what the file is
    exempt from is stated beside what it is accepted for.

    ``validate_grid_semantics`` is re-run by every submission surface --
    ``prepare``, ``submit``, ``resubmit``, ``resubmit-preflight`` and the
    deferred-eval path -- so the acceptance and the remaining refusal both
    reach the operator wherever the file is used.
    """
    path, cfg = _v6_config()
    assert cfg.pretrain.dfs_set is True
    assert cfg.fidelity.enforce is True
    assert cfg.model.parent_anchor is True, path
    assert cfg.model.descriptor_coordinates == "dfs", path
    assert cfg.pretrain.energy_term_weight == 0.0, (
        f"{path}: the weight has been changed; under the anchor 0.0 is the "
        "exact objective, and this pin describes the file as it ships")
    assert _axis_carries_a_meta_gga_architecture(cfg), path

    _v6_semantics(cfg)

    # ... and so is the same file restricted to either rung, so the acceptance
    # is not an artefact of the mixed axis.
    from xcquinox.alec import get_architecture
    from xcquinox.alec.config import ArchitectureConfig
    gga_only = tuple(a for a in cfg.sweep.arch
                     if not ArchitectureConfig.is_meta_gga(
                         get_architecture(a)))
    mgga_only = tuple(a for a in cfg.sweep.arch if a not in gga_only)
    assert gga_only and mgga_only, path
    for axis in (gga_only, mgga_only):
        _v6_semantics(dataclasses.replace(
            cfg, sweep=dataclasses.replace(cfg.sweep, arch=axis)))

    # ... while the same GGA-rung configuration UNANCHORED is refused for the
    # weight, which is the refusal the anchor exempts a run from.
    from xcquinox.alec.cluster.grid_config import ModelConfig
    unanchored = dataclasses.replace(
        cfg, sweep=dataclasses.replace(cfg.sweep, arch=gga_only),
        model=ModelConfig(parent_anchor=False, descriptor_coordinates="legacy"))
    with pytest.raises(ValueError) as excinfo:
        _v6_semantics(unanchored)
    message = str(excinfo.value)
    assert "pretrain.energy_term_weight" in message
    assert "hpcjobs/probe_pretrain_energy_weight.py" in message
    assert "56.1 kcal/mol" in message


def test_v6_states_the_objective_beside_the_weight():
    """The YAML says, at the key itself, why 0.0 is the objective and not a
    placeholder, and that the refusal it is exempt from still stands for an
    unanchored run. A banner at the top of a 500-line file is read once; the
    comment on the line being edited is read by whoever edits it."""
    path, _cfg = _v6_config()
    with open(path) as f:
        lines = f.read().splitlines()
    i = next(i for i, ln in enumerate(lines)
             if ln.strip().startswith("energy_term_weight:"))
    block = "\n".join(lines[max(0, i - 30):i])
    assert "validate_grid_semantics" in block, path
    assert "REFUSES" in block or "refuses" in block, path
    assert "parent_anchor" in block, path
    assert "exempt" in block, path


def test_v6_pins_the_pretraining_validation_block_and_the_mesh_share():
    """The stop criterion, the validation hold-out and the mesh share are
    protocol decisions of Section 7 / Section 6, each of which loads at a
    DIFFERENT dataclass default: ``validation_fraction`` 0.0 (no hold-out at
    all, i.e. no stop criterion), ``patience`` 0, ``mesh_fraction`` 0.3. A
    file that dropped them would still load and would run the pre-protocol
    schedule, so they are read from the text as well as the parse.

    The schedule also has to be non-degenerate: the training-side guard
    refuses a patience that cannot be reached in the number of checks the run
    performs, so the ratio is asserted rather than the values alone.
    """
    path, cfg = _v6_config()
    raw = _raw_yaml(path)["pretrain"]
    assert raw["validation_fraction"] == 0.2
    assert raw["validation_seed"] == 0
    assert raw["validate_every"] == 50
    assert raw["patience"] == 10
    assert cfg.pretrain.validation_fraction == 0.2
    assert cfg.pretrain.validation_seed == 0
    assert cfg.pretrain.validate_every == 50
    assert cfg.pretrain.patience == 10
    n_checks = cfg.pretrain.n_steps // cfg.pretrain.validate_every
    assert n_checks == 50
    assert cfg.pretrain.patience < n_checks - 1, (
        f"{path}: patience {cfg.pretrain.patience} against {n_checks} checks "
        "is the degenerate regime the training-side guard rejects")
    # The mesh is a regularizer at its anchored share, not a fitted knob.
    from xcquinox.alec.pretrain_data_gen import MESH_WEIGHT_FRACTION
    assert raw["mesh_fraction"] == 0.3
    assert cfg.pretrain.mesh_fraction == MESH_WEIGHT_FRACTION == 0.3


def test_v6_keeps_the_one_atom_neither_inventory_supplies():
    """``pretrain.atoms`` is all but redundant under ``dfs_set`` +
    ``pool_atoms`` and is kept for exactly one row: free Na, which neither
    inventory carries and on which Na2's atomization energy rests. Dropping
    the list loads, changes the set by one system, and is invisible in every
    other pin -- so the resolved set is asserted, not the list.

    He must stay OUT: PySCF has no He in 6-311++G(3df,2pd).
    """
    from xcquinox.alec.pretrain_data_gen import resolve_pretrain_systems
    path, cfg = _v6_config()
    raw = _raw_yaml(path)["pretrain"]
    assert raw["atoms"] == {"H": 1, "Li": 1, "C": 2, "N": 3, "O": 2, "F": 1,
                            "Na": 1}, path
    resolved = {
        ref: resolve_pretrain_systems(
            atoms=cfg.pretrain.atoms, dfs_set=cfg.pretrain.dfs_set,
            pool_atoms=cfg.pretrain.pool_atoms, reference_xc=ref)
        for ref in ("pbe", "scan")}
    assert len(resolved["pbe"]) == 38, path
    assert len(resolved["scan"]) == 36, path
    for ref, systems in resolved.items():
        names = {s.name for s in systems}
        assert "Na" in names, (path, ref)
        assert "He" not in names, (path, ref)
    # ... and Na is the only system the explicit list contributes.
    without = resolve_pretrain_systems(
        atoms=(), dfs_set=True, pool_atoms=True, reference_xc="pbe")
    assert {s.name for s in resolved["pbe"]} - {s.name for s in without} == \
        {"Na"}, path


def test_v6_train_wall_covers_the_four_channel_inline_eval():
    """The train wall has to hold the train AND the eval, because
    ``inline_eval`` runs the eval inside the train task.

    The anchor is v3's own measurement at this basis, grid and solver:
    ``deep_attn_3x16`` at subset 26 took 42.14 h under a THREE-channel inline
    eval (v3's config states ``inline_eval: true``, ``eval_time: 02:00:00``
    and no ``eval_coldstart``). v6 runs FOUR channels, one of them the
    25-cycle coldstart diagnostic, and budgets 08:00:00 for them. The same
    cell is then 42.14 - 2 + 8 = 48.1 h if the 42.14 h already contained a
    2 h eval, or 42.14 + 8 = 50.1 h if it was train time alone; both exceed a
    48 h wall, which is why the wall is no longer 48 h. v6 additionally puts
    five depth-4 / width-32 attention architectures on the axis, a size v3
    never ran (its three attention architectures were all 3x16), so the
    margin above 50.1 h is carrying real unmeasured load.

    ``timeout_retry_time`` is the wall a cell wall-killed BEFORE its first
    resume checkpoint is relaunched at: that one restarts from step zero, so
    a whole run has to fit inside one window. A cell that DID checkpoint is
    continued at the wall above instead, by the same
    ``cluster resubmit <run_dir> --submit`` -- the classifier gives the
    checkpoint precedence over the wall-kill record, and nothing requeues
    unattended. ``timeout_retry_partition`` stays unset
    for the reason v5 states for ``oom_retry_partition`` -- the submit
    partition is already the largest reachable node, so a re-route buys
    nothing and the wall is the only lever.
    """
    path, cfg = _v6_config()
    raw = _raw_yaml(path)["cluster"]

    def _hours(literal):
        d, _, rest = str(literal).rpartition("-")
        h, m, s = (int(x) for x in rest.split(":"))
        return int(d or 0) * 24 + h + m / 60.0 + s / 3600.0

    train_h = _hours(raw["time"])
    assert cfg.inline_eval is True and cfg.eval_coldstart is True, path
    # Both readings of the v3 anchor, from the file's own numbers.
    v3_measured, v3_eval_budget, v6_eval_budget = 42.14, 2.0, 8.0
    assert _hours(raw["eval_time"]) == v6_eval_budget
    inclusive = v3_measured - v3_eval_budget + v6_eval_budget
    exclusive = v3_measured + v6_eval_budget
    assert abs(inclusive - 48.1) < 0.05
    assert abs(exclusive - 50.1) < 0.05
    assert train_h > exclusive, (
        f"{path}: a {train_h:g} h train wall does not cover the worse "
        f"reading of the v3 anchor ({exclusive:.2f} h)")
    assert train_h == 72.0, path
    # The automatic recovery is longer than the wall it recovers from.
    assert cfg.cluster.timeout_retry_time is not None, (
        f"{path}: without timeout_retry_time a cell wall-killed before its "
        "first checkpoint restarts from zero on the wall it has already been "
        "shown to exceed")
    assert _hours(cfg.cluster.timeout_retry_time) > train_h, path
    assert cfg.cluster.timeout_retry_partition is None, path
    # The arithmetic is in the file, not only here.
    text = open(path).read()
    for quoted in ("42.14", "48.1 h", "50.1 h", "72:00:00", "sinfo"):
        assert quoted in text, (path, quoted)


# ---------------------------------------------------------------------------
# The five v6 GROUP submissions
#
# The campaign is submitted as five standalone group files rather than as the
# reference sweep: one question per file, one run directory per file, one
# certificate gate per file. A group's result is then readable on its own and a
# group's failure is local to it, which is the whole reason the architecture
# axis differs between the six files; everything else in them is the
# reference's, field for field.
#
# What such a split can lose silently is the PARTITION of the registry. An
# architecture that lands in no group, or in two, changes what the campaign
# measured while every file still loads, submits and runs; so does a group that
# acquired a name from the other rung, whose SCF seed and pretraining parent
# then differ from its neighbours' with nothing in the run reporting it. Both
# are asserted here from the parsed axes rather than from the file names.
# ---------------------------------------------------------------------------

#: The eleven architectures the campaign directive leaves OUT of v6: the
#: width/depth ablation group carries the baseline descriptors alone, plus or
#: minus attention, so the descriptor-carrying depth-4 forms, the whole
#: no-transform family and the rung-3.5-only form are not submitted. They are
#: NAMED rather than dropped, so the registry stays covered exactly: an
#: architecture may leave the campaign deliberately, on this list, but never by
#: falling through the gap between five files. Any one of them rejoins by a
#: line on a group's axis and the matching deletion here.
EXCLUDED_FROM_V6 = (
    "deep_combined",
    "deep_combined_attn",
    "deep_cusp",
    "deep_cusp_attn",
    "deep_dm",
    "deep_dm_attn",
    "deep_notransform",
    "deep_notransform_3x16",
    "deep_notransform_attn",
    "deep_notransform_attn_3x16",
    "deep_rung35only_3x16",
)

#: The ladder, in submission order: (file, architecture count, meta-GGA rung?,
#: the historical arm whose partition/retry scheme the file mirrors). The rung
#: is STATED here rather than derived from the axis, so a name that moved
#: between groups turns the rung pin red instead of redefining what the group
#: is.
_V6_GROUPS = (
    ("dfs_step7.dfs6311_grid3_v6g1_size.yaml", 4, False,
     "dfs_step7.dfs6311_grid3_v4gga.yaml"),
    ("dfs_step7.dfs6311_grid3_v6g2_families.yaml", 6, False,
     "dfs_step7.dfs6311_grid3_v4gga.yaml"),
    ("dfs_step7.dfs6311_grid3_v6g2_families_mgga.yaml", 5, True,
     "dfs_step7.dfs6311_grid3_v5.yaml"),
    ("dfs_step7.dfs6311_grid3_v6g3_dm.yaml", 3, False,
     "dfs_step7.dfs6311_grid3_v4gga.yaml"),
    ("dfs_step7.dfs6311_grid3_v6g4_ablations.yaml", 2, False,
     "dfs_step7.dfs6311_grid3_v4gga.yaml"),
)

#: The five group files in submission order.
_V6_GROUP_FILES = tuple(row[0] for row in _V6_GROUPS)

#: The reference and the five groups: the six files that must agree on the
#: identity, the protocol, the certificate, the placeholder and the walls.
_V6_FILES = (_V6_REFERENCE,) + _V6_GROUP_FILES

#: The subset-size axis every file shares, so a cell count is a claim about the
#: architecture axis alone.
_V6_SUBSET_SIZES = 11


def test_every_v6_campaign_file_is_present_in_the_deployment_tree():
    """All six campaign files exist wherever ``hpcjobs/configs/`` exists.

    Presence has to be its own assertion because every other pin in this
    section reaches its file through ``_campaign_config``, which SKIPS an
    absent one -- reasonable for a source or wheel checkout that carries no
    deployment tree at all, and exactly wrong for a group file that was
    deleted or never synced: nineteen pins describing that group would go
    quiet together and the run would still submit as a four-group campaign.
    The directory rather than any one file is the discriminator, so deleting
    the reference as well cannot make the check disappear.
    """
    configs_dir = _campaign_configs_dir()
    if configs_dir is None:
        pytest.skip("no hpcjobs/configs/ deployment tree in this checkout")
    missing = _missing_campaign_files(_V6_FILES, configs_dir)
    assert not missing, (
        f"{configs_dir} is missing {missing}: the campaign is submitted as "
        f"{len(_V6_GROUP_FILES)} group files against the reference, and every "
        "pin that names a missing one would otherwise skip rather than fail")


def test_a_deleted_group_file_fails_rather_than_skipping(tmp_path):
    """The predicate behind the pin above, driven on a directory a file was
    removed from -- so the pin is known to detect the deletion rather than
    merely to pass on a complete tree."""
    for name in _V6_FILES:
        (tmp_path / name).write_text("")
    assert _missing_campaign_files(_V6_FILES, str(tmp_path)) == []
    deleted = _V6_GROUP_FILES[3]
    os.unlink(str(tmp_path / deleted))
    assert _missing_campaign_files(_V6_FILES, str(tmp_path)) == [deleted]


def _v6_group_row(name):
    """The ``_V6_GROUPS`` row for ``name``."""
    return next(row for row in _V6_GROUPS if row[0] == name)


def _v6_expected_seeds(name):
    """The SCF seeds ``name``'s axis resolves to: one per rung, both for the
    mixed-rung reference."""
    if name == _V6_REFERENCE:
        return {"pbe", "scan"}
    return {"scan"} if _v6_group_row(name)[2] else {"pbe"}


# --- what every one of the six files states ---------------------------------

@pytest.mark.parametrize("name", _V6_FILES)
def test_every_v6_file_carries_the_production_identity(name):
    """The five identity fields, the rung-derived SCF seed and NO waiver, in
    every group file as in the reference.

    ``fidelity.run_identity`` records these five on every certificate a run
    writes, and ``validate_run`` refuses a certificate whose identity differs
    from its configuration's -- so a group whose identity drifted would fail
    late, on its own certificate. The five run directories are also one
    campaign only because their identities are the same object: a basis or a
    Coulomb backend that moved in one file re-identifies that group's shared
    reference caches while still loading, submitting and running.

    Read from the YAML text as well as the parse, because three of the fields
    have a loader default and each default is the pre-protocol one
    (``density_fit`` False, ``seed_xc`` "pbe", ``auxbasis`` None).
    """
    from xcquinox.alec.orientation_lock import DEFAULT_STRENGTH
    from xcquinox.alec.cluster.fidelity import run_identity
    from xcquinox.alec.cluster.spec_builder import resolve_seed_xc
    path, cfg = _campaign_config(name)
    raw = _raw_yaml(path)["inputs"]
    assert str(raw["basis"]) == "6-311++G(3df,2pd)", path
    assert int(raw["grid_level"]) == 3, path
    assert raw["density_fit"] is True, path
    assert "auxbasis" not in raw, (
        f"{path}: auxbasis is deliberately unset -- auto-selected from the "
        "orbital basis and recorded on the certificate as null")
    assert float(raw["orientation_lock_strength"]) == DEFAULT_STRENGTH == 3e-5
    assert run_identity(cfg) == {
        "basis": "6-311++G(3df,2pd)",
        "grid_level": 3,
        "density_fit": True,
        "auxbasis": None,
        "orientation_lock_strength": 3e-5,
    }, path
    # The seed is the rung baseline, per architecture, in every file.
    assert raw["seed_xc"] == "auto", (
        f"{path}: seed_xc must be 'auto' -- the per-architecture rung baseline "
        "rungs.seed_xc_for_arch resolves, which is the predicate "
        "pretrain.parent_density: auto uses for the pretraining parent")
    seeds = {resolve_seed_xc(cfg.inputs, arch) for arch in cfg.sweep.arch}
    assert seeds == _v6_expected_seeds(name), (path, sorted(seeds))
    assert "allow_irreproducible_degenerate" not in raw, (
        f"{path}: at grid level 3 with the lock on the data generator refuses "
        "nothing, so a stated waiver grants a permission the run never "
        "exercises and is refused at submit")
    assert "irreproducible_degenerate_reason" not in raw, path
    assert cfg.inputs.allow_irreproducible_degenerate is False


@pytest.mark.parametrize("name", _V6_FILES)
def test_every_v6_file_enforces_the_certificate_at_the_binding_tolerances(name):
    """tol_AE = 1.0 kcal/mol, tol_atom = 1.0 mHa, enforced, no override, in
    each of the six files.

    The gates are per group -- each group's pretrain task blocks its own train
    array and no other -- so a tolerance loosened in one file loosens that
    group alone, and nothing downstream reports the difference between two
    groups certified at different tolerances. Asserted from the text as well as
    the parse, so a value inherited from a dataclass default is not read as a
    stated decision.
    """
    path, cfg = _campaign_config(name)
    raw = _raw_yaml(path)
    assert raw.get("fidelity") is not None, f"{path} must state a 'fidelity' block"
    assert raw["fidelity"]["tol_AE"] == 1.0, path
    assert raw["fidelity"]["tol_atom"] == 1.0, path
    assert raw["fidelity"]["enforce"] is True, path
    assert raw["fidelity"]["override_reason"] is None, path
    assert cfg.fidelity.tol_AE == 1.0
    assert cfg.fidelity.tol_atom == 1.0
    assert cfg.fidelity.enforce is True
    assert cfg.fidelity.override_reason is None


@pytest.mark.parametrize("name", _V6_FILES)
def test_every_v6_file_runs_the_protocol_set_on_the_corrected_footing(name):
    """``dfs_set``, ``pool_atoms``, ``exchange_footing: spin_channel``,
    ``parent_density: auto``, the mesh share and the validation block, stated
    in every file.

    A certificate is comparable between two groups only if the networks were
    fitted to the same rows under the same objective, so the protocol block is
    not something a group file may inherit by omission: each knob loads at a
    DIFFERENT pre-protocol default (``dfs_set``/``pool_atoms`` False,
    ``exchange_footing`` "total", ``validation_fraction`` 0.0 -- no hold-out
    and so no stop criterion -- ``patience`` 0), and a file that dropped them
    would run the pre-protocol schedule and still load.
    """
    path, cfg = _campaign_config(name)
    raw = _raw_yaml(path)["pretrain"]
    assert raw["dfs_set"] is True and raw["pool_atoms"] is True, path
    assert raw["exchange_footing"] == "spin_channel", path
    assert raw["parent_density"] == "auto", path
    assert raw["mesh_fraction"] == 0.3, path
    assert raw["validation_fraction"] == 0.2, path
    assert raw["validation_seed"] == 0, path
    assert raw["validate_every"] == 50, path
    assert raw["patience"] == 10, path
    assert raw["atoms"] == {"H": 1, "Li": 1, "C": 2, "N": 3, "O": 2, "F": 1,
                            "Na": 1}, path
    assert cfg.pretrain.dfs_set is True
    assert cfg.pretrain.pool_atoms is True
    assert cfg.pretrain.exchange_footing == "spin_channel"
    assert cfg.pretrain.parent_density == "auto"
    from xcquinox.alec.pretrain_data_gen import MESH_WEIGHT_FRACTION
    assert cfg.pretrain.mesh_fraction == MESH_WEIGHT_FRACTION == 0.3
    n_checks = cfg.pretrain.n_steps // cfg.pretrain.validate_every
    assert cfg.pretrain.patience < n_checks - 1, (
        f"{path}: patience {cfg.pretrain.patience} against {n_checks} checks "
        "is the degenerate regime the training-side guard rejects")


@pytest.mark.parametrize("name", _V6_FILES)
def test_every_v6_file_submits_at_the_login_node_semantic_check(name):
    """Each of the six states ``model.parent_anchor: true``, ships
    ``energy_term_weight: 0.0``, and is ACCEPTED by the login-node check --
    the reference file and the meta-GGA group included.

    Under the anchor every network equals its parent at initialization, so the
    weight-zero objective is exact rather than a placeholder and the refusal
    that used to hold the campaign is lifted by construction; and both rungs
    now have their parent (``parents.pbe_*`` and ``parents.scan_*``,
    SPEC_parent_anchor.md Section 3.8), so the rung the file's axis sits on no
    longer decides whether it submits. The two files that carry the meta-GGA
    architectures were the ones the PBE-anchor commit refused, and they are
    the point of this pin.

    Each file's resolved parents are asserted as well as its acceptance: the
    parent is a property of the ARCHITECTURE's rung (``fidelity.resolve_parent``
    reads ``rungs.seed_xc_for_arch``), it is what the pretraining data, the
    certificate and the checkpoint identity are all posed against, and a file
    that submits while resolving the wrong one would buy the whole array
    before the mismatch showed.

    Executed per file, not asserted from the text.
    """
    from xcquinox.alec.cluster import fidelity as fid

    path, cfg = _campaign_config(name)
    assert cfg.model.parent_anchor is True, path
    assert cfg.model.descriptor_coordinates == "dfs", path
    assert cfg.pretrain.energy_term_weight == 0.0, (
        f"{path}: the weight has been changed; under the anchor 0.0 is the "
        "exact objective, and this pin describes the files as they ship")
    _v6_semantics(cfg)

    resolved = {a: fid.resolve_parent(a) for a in cfg.sweep.arch}
    assert set(resolved.values()) <= {"pbe", "scan"}, (path, resolved)
    from xcquinox.alec import get_architecture
    from xcquinox.alec.config import ArchitectureConfig
    for arch_name, parent in resolved.items():
        expected = "scan" if ArchitectureConfig.is_meta_gga(
            get_architecture(arch_name)) else "pbe"
        assert parent == expected, (path, arch_name, parent, expected)
    if _axis_carries_a_meta_gga_architecture(cfg):
        assert "scan" in set(resolved.values()), (path, resolved)
    else:
        assert set(resolved.values()) == {"pbe"}, (path, resolved)
    # ... and the file says so where the weight is edited, not only in a
    # banner: the anchored exemption and the refusal it is an exemption from.
    with open(path) as f:
        lines = f.read().splitlines()
    i = next(i for i, ln in enumerate(lines)
             if ln.strip().startswith("energy_term_weight:"))
    block = "\n".join(lines[max(0, i - 30):i])
    assert "validate_grid_semantics" in block, path
    assert "REFUSES" in block or "refuses" in block, path
    assert "parent_anchor" in block, path
    assert "exempt" in block, path


@pytest.mark.parametrize("name", _V6_FILES)
def test_every_v6_file_carries_the_v6_walls_and_the_queue_check(name):
    """The walls are the reference's in all six files, group by group.

    They are sized to the four-channel inline evaluation, not to the group:
    v3's 42.14 h measurement is of one 3x16 attention cell, and every group
    here carries at least one attention architecture at the same or a larger
    shape. A group file that quietly inherited v5's 48 h would wall-kill its
    large-subset attention cells (48.1 h on the smaller reading of the anchor,
    50.1 h on the larger) instead of finishing them.
    """
    path, cfg = _campaign_config(name)
    # The GGA groups ship the recorded long-40core figure (48 h, the value
    # v4gga ran at on that queue; an over-cap request is rejected at submit)
    # and rely on the 96 h retry for the ~50.1 h attention cells; the
    # meta-GGA group and the reference file carry 72 h on the 96-core class.
    expected_wall = "48:00:00" if "_v6g" in name and "mgga" not in name \
        else "72:00:00"
    assert cfg.cluster.time == expected_wall, path
    assert cfg.cluster.timeout_retry_time == "96:00:00", path
    assert cfg.cluster.eval_time == "08:00:00", path
    assert cfg.cluster.pretrain_time == "12:00:00", path
    assert cfg.cluster.preflight_time == "12:00:00", path
    assert cfg.cluster.datagen_time == "04:00:00", path
    assert cfg.cluster.benchmark_refs_time == "24:00:00", path
    assert cfg.inline_eval is True and cfg.eval_coldstart is True, path
    text = open(path).read()
    for quoted in ("42.14", "48.1 h", "50.1 h", expected_wall, "sinfo"):
        assert quoted in text, (path, quoted)


@pytest.mark.parametrize("name", _V6_FILES)
def test_every_v6_file_carries_the_stony_brook_job_mail(name):
    """Submission, completion and failure of every stage of every group reach
    the Stony Brook inbox. The directives are rendered from these two fields,
    so pinning them pins them on all five stage scripts of all six files."""
    path, cfg = _campaign_config(name)
    assert cfg.cluster.mail_user == "alec.wills@stonybrook.edu", path
    assert cfg.cluster.mail_type == "BEGIN,END,FAIL", path


# --- what makes the five groups a partition of the campaign -----------------

def test_v6_groups_and_the_exclusions_partition_the_registry():
    """The five group axes are pairwise disjoint, and their union plus
    ``EXCLUDED_FROM_V6`` is ``sorted(ARCHITECTURES)`` exactly.

    This is the pin the split exists to be safe under. An architecture in two
    groups is trained twice at different roots and read as two results; one in
    no group is missing from the campaign with nothing to say so, because each
    file is individually well formed. An architecture added to the registry
    after these files were written turns this red naming itself, which is the
    only signal that the campaign no longer covers what it claims to -- it then
    joins a group's axis or the exclusion list, deliberately either way.
    """
    from xcquinox.alec import ARCHITECTURES
    axes = {}
    for name in _V6_GROUP_FILES:
        path, cfg = _campaign_config(name)
        axis = list(cfg.sweep.arch)
        assert len(axis) == len(set(axis)), (
            f"{path}: the arch axis carries a duplicate name; expand_grid "
            f"would drop it with a warning: {axis}")
        axes[name] = set(axis)
    for i, first in enumerate(_V6_GROUP_FILES):
        for second in _V6_GROUP_FILES[i + 1:]:
            overlap = sorted(axes[first] & axes[second])
            assert not overlap, (
                f"{first} and {second} both sweep {overlap}: that "
                f"architecture would be trained twice, at two roots, and read "
                f"as two results")
    union = set().union(*axes.values())
    excluded = set(EXCLUDED_FROM_V6)
    assert len(EXCLUDED_FROM_V6) == len(excluded) == 11
    assert not (union & excluded), sorted(union & excluded)
    registry = sorted(ARCHITECTURES)
    assert sorted(union | excluded) == registry, (
        f"the five group axes and EXCLUDED_FROM_V6 cover "
        f"{len(union | excluded)} architectures; the registry carries "
        f"{len(registry)}. Missing from the campaign AND from the exclusion "
        f"list: {sorted(set(registry) - union - excluded)}; unknown to the "
        f"registry: {sorted((union | excluded) - set(registry))}")


@pytest.mark.parametrize("name", _V6_GROUP_FILES)
def test_v6_group_axis_is_one_rung(name):
    """Every architecture of a group is on the rung the group states.

    A group is one rung because the rung decides two things per architecture:
    the SCF seed ``inputs.seed_xc: auto`` resolves and the parent
    ``pretrain.parent_density: auto`` fits against. A stray name from the other
    rung would pull a second parent-density file into the group's datagen and
    be certified against a different functional from its neighbours, with the
    file loading and running exactly as before. The size group is GGA-only for
    a registry reason rather than a chosen one: no meta-GGA form of ``medium``
    or ``shallow`` exists.
    """
    from xcquinox.alec import ARCHITECTURES
    from xcquinox.alec.config import ArchitectureConfig
    path, cfg = _campaign_config(name)
    _f, n_arch, is_mgga, _arm = _v6_group_row(name)
    axis = list(cfg.sweep.arch)
    assert len(axis) == n_arch, (path, axis)
    for arch in axis:
        assert ArchitectureConfig.is_meta_gga(ARCHITECTURES[arch]) is is_mgga, (
            f"{path}: {arch} is on the "
            f"{'GGA' if is_mgga else 'meta-GGA'} rung, but this group is the "
            f"{'meta-GGA' if is_mgga else 'GGA'} one")


@pytest.mark.parametrize("name", _V6_GROUP_FILES)
def test_v6_group_cell_count(name):
    """The cell count each group's header states, derived from the parsed axes.

    The count is the group's size on record -- the SLURM array bound, the
    queue-time estimate and the figure layer's expectation all read it -- and
    it is a product, so a single name added to an axis moves it by eleven.
    """
    path, cfg = _campaign_config(name)
    _f, n_arch, _rung, _arm = _v6_group_row(name)
    cells = expand_grid(cfg)
    assert len(set(cfg.sweep.subset_size)) == _V6_SUBSET_SIZES, path
    assert len(cells) == n_arch * _V6_SUBSET_SIZES, path
    assert len(cells) == {4: 44, 6: 66, 5: 55, 3: 33, 2: 22}[n_arch], path
    assert f"{len(cells)} cells" in open(path).read(), (
        f"{path}: the header must state the cell count it expands to")


def test_v6_group_cells_sum_to_the_submitted_campaign():
    """44 + 66 + 55 + 33 + 22 = 220 cells submitted, out of the reference
    sweep's 341.

    The 121-cell gap is the eleven excluded architectures at eleven subset
    sizes, and it is arithmetic rather than a discrepancy: this asserts the
    three numbers close, so a group axis edited without the exclusion list (or
    the reverse) cannot leave the ladder quietly covering something else.
    """
    counts = []
    for name in _V6_GROUP_FILES:
        _path, cfg = _campaign_config(name)
        counts.append(len(expand_grid(cfg)))
    assert counts == [44, 66, 55, 33, 22], counts
    assert sum(counts) == 220
    _refpath, ref = _campaign_config(_V6_REFERENCE)
    assert len(expand_grid(ref)) == 341
    assert 341 - 220 == len(EXCLUDED_FROM_V6) * _V6_SUBSET_SIZES == 121


@pytest.mark.parametrize("name", _V6_GROUP_FILES)
def test_v6_group_partition_scheme_mirrors_its_historical_arm(name):
    """A GGA group carries v4gga's 40-core retry routing; the meta-GGA group
    carries v5's absence of it.

    The two schemes are opposites and the choice is per rung: a 40-core cell
    that OOMs or hits the wall is recoverable only by re-routing onto the
    larger Milan nodes, which is what v4gga's two retry partitions do, while a
    run already on those nodes has nowhere larger to go and v5 therefore leaves
    both unset (a same-partition retry cannot clear a deterministic compile
    OOM). The values are read from the historical files rather than transcribed,
    and the two arms are asserted to actually differ, so the mirror is a
    comparison and not a tautology.
    """
    path, cfg = _campaign_config(name)
    _f, _n, _rung, arm_name = _v6_group_row(name)
    arm_path, arm = _campaign_config(arm_name)
    gga_arm = load_grid_config(
        _campaign_config_path("dfs_step7.dfs6311_grid3_v4gga.yaml"))
    mgga_arm = load_grid_config(
        _campaign_config_path("dfs_step7.dfs6311_grid3_v5.yaml"))
    assert gga_arm.cluster.oom_retry_partition is not None
    assert gga_arm.cluster.timeout_retry_partition is not None
    assert mgga_arm.cluster.oom_retry_partition is None
    assert mgga_arm.cluster.timeout_retry_partition is None
    assert cfg.cluster.partition == arm.cluster.partition == "", (
        f"{path}: the partition is chosen at submit time (--partition is "
        f"required), so the config states none")
    assert cfg.cluster.oom_retry_partition == arm.cluster.oom_retry_partition, (
        path, arm_path)
    assert (cfg.cluster.timeout_retry_partition
            == arm.cluster.timeout_retry_partition), (path, arm_path)


@pytest.mark.parametrize("name", _V6_GROUP_FILES)
def test_v6_group_differs_from_the_reference_in_exactly_what_it_claims(name):
    """Flattened against the reference, defaults included, a group file differs
    in the architecture axis, the two roots and -- on the 40-core scheme -- the
    two retry partitions and the shipped 48 h wall. Nothing else.

    This is the inheritance claim every group header makes, and it is what
    makes five run directories one campaign: a hyperparameter, a solver knob, a
    reference root or a protocol field that drifted into one group would break
    the comparison between groups while loading, submitting and running exactly
    as before.
    """
    path, cfg = _campaign_config(name)
    ref_path, ref = _campaign_config(_V6_REFERENCE)
    _f, _n, is_mgga, _arm = _v6_group_row(name)
    a, b = _flat_config(ref), _flat_config(cfg)
    keys = sorted(set(a) | set(b))
    differing = sorted(k for k in keys
                       if a.get(k, "<absent>") != b.get(k, "<absent>"))
    expected = ["inputs.output_root", "pretrain.data_dir", "sweep.arch"]
    if not is_mgga:
        expected += ["cluster.oom_retry_partition",
                     "cluster.timeout_retry_partition",
                     # the 40-core class ships its recorded 48 h cap; the
                     # reference carries the 96-core class's 72 h.
                     "cluster.time"]
    # 137 since the parent anchor added ``model.parent_anchor`` and
    # ``model.descriptor_coordinates``; both are the campaign's model class
    # and are therefore IDENTICAL in every group and in the reference, so
    # neither joins the difference list here.
    assert len(keys) == 137, len(keys)
    assert differing == sorted(expected), (path, ref_path, differing)
    # The header makes the same claim in prose, and that is where an operator
    # reads it. What is asserted is the EXCEPT clause -- the list of fields
    # the file says it does not inherit -- because a clause naming fewer
    # differences than the file carries reads as an inheritance that can be
    # relied on. The 40-core groups ship 48 h where the reference carries
    # 72 h, so `cluster.time` belongs in that list.
    lines = open(path).read().splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("# Every field except"))
    end = next(i for i, ln in enumerate(lines[start:], start)
               if "is dfs_step7.dfs6311_grid3_v6.yaml's" in ln)
    except_clause = "\n".join(lines[start:end + 1])
    for field in ("architecture axis", "roots"):
        assert field in except_clause, (path, field)
    if not is_mgga:
        assert "cluster.time" in except_clause, (
            f"{path}: the header's inheritance paragraph must name the train "
            "wall among the fields NOT inherited -- this file ships 48 h "
            f"against the reference's 72 h. It reads:\n{except_clause}")


@pytest.mark.parametrize("name", _V6_GROUP_FILES)
def test_v6_group_writes_into_its_own_roots(name):
    """The run root and the pretraining-data root are the group's own, and
    carry its name.

    Two groups sharing an ``output_root`` interleave their run directories
    under one tree that ``status``, ``validate_run`` and the figure suite each
    read as one run. Two groups sharing ``pretrain.data_dir`` is worse: the
    parent-density file is named after the parent functional, so they collide
    on the name, and since the groups are independent submissions with no
    ordering between them the collision is a write race between two datagen
    jobs rather than a reuse. The per-group copies cost one datagen pass each
    and are value-identical by construction -- same set, same footing, same
    objective, same identity.
    """
    path, cfg = _campaign_config(name)
    stem = name[len("dfs_step7.dfs6311_grid3_"):-len(".yaml")]
    assert cfg.inputs.output_root.endswith(f"dfs6311_grid3_{stem}"), path
    assert cfg.pretrain.data_dir.endswith(f"_{stem}"), path
    roots, data = set(), set()
    for other in _V6_FILES:
        _p, other_cfg = _campaign_config(other)
        roots.add(other_cfg.inputs.output_root)
        data.add(other_cfg.pretrain.data_dir)
    assert len(roots) == len(_V6_FILES), sorted(roots)
    assert len(data) == len(_V6_FILES), sorted(data)


@pytest.mark.parametrize("name", _V6_GROUP_FILES)
def test_v6_group_datagen_builds_the_one_parent_its_rung_needs(name):
    """A single-rung group derives ONE pretraining-data file, on its own rung's
    parent; the reference's mixed axis is what needs two.

    This is the derivation the datagen stage builds from and the pretrain
    worker later opens its file through, so it is also the check that a group
    whose axis picked up a name from the other rung would fail here rather than
    at its certificate.
    """
    from xcquinox.alec.cluster._datagen import _required_data_specs
    path, cfg = _campaign_config(name)
    _f, _n, is_mgga, _arm = _v6_group_row(name)
    specs = _required_data_specs(cfg)
    parents = sorted({ref for _pol, ref in specs})
    assert parents == (["scan"] if is_mgga else ["pbe"]), (path, specs)
    assert sorted({pol for pol, _ref in specs}) == [True], (path, specs)


@pytest.mark.parametrize("name", _V6_GROUP_FILES)
def test_v6_group_states_its_place_in_the_ladder(name):
    """The header names the group, its question, its position and the whole
    ladder, and says the submissions are independent.

    An operator reads one file at a time; the ladder is a property of five, and
    a group file that did not carry it would be submitted without any way of
    knowing what else the campaign is or whether something has to run first.
    """
    path, _cfg = _campaign_config(name)
    text = open(path).read()
    order = _V6_GROUP_FILES.index(name) + 1
    assert "GROUP G" in text, path
    assert "THE QUESTION:" in text, path
    assert f"submission {order} of 5" in text, path
    assert "SUBMISSION IS PER GROUP AND INDEPENDENT" in text, path
    for other in _V6_GROUP_FILES:
        assert other in text, (path, other)


# QOS wall caps as measured (sacctmgr, 2026-08-27, both login instances):
# every long-* QOS holds MaxWall at 48 h; the extended-* partitions carry
# 7-day caps. Pinned here because SLURM rejects an over-cap wall only when
# the sbatch carrying it runs -- for a retry key that is days into a
# campaign, at recovery time -- and the first v6 groups shipped exactly that
# defect: timeout_retry_partition long-96core with timeout_retry_time 96 h,
# an escalation rejected whenever it fired, and an OOM re-route to the same
# 48 h QOS replaying scripts whose baked campaign wall is 72 h.
_QOS_MAX_WALL_HOURS = {
    "long-40core": 48.0,
    "long-96core": 48.0,
    "long-96core-shared": 48.0,
    "extended-40core": 168.0,
    "extended-96core": 168.0,
    "extended-96core-shared": 168.0,
}

#: The campaign submits the 40-core groups with ``--train-time "72:00:00"``
#: (recorded in the 2026-08-27 relaunch); an OOM re-route replays the captured
#: script at that baked wall, so the OOM target must admit it too.
_V6_CAMPAIGN_TRAIN_WALL_H = 72.0


def _wall_hours(literal) -> float:
    d, _, rest = str(literal).rpartition("-")
    h, m, sec = (int(x) for x in rest.split(":"))
    return int(d or 0) * 24 + h + m / 60.0 + sec / 3600.0


@pytest.mark.parametrize("name", _V6_GROUP_FILES)
def test_v6_group_retry_targets_admit_their_walls(name):
    """Every rendered recovery wall fits its target partition's QOS cap.

    The timeout escalation renders ``--partition=<timeout_retry_partition>
    --time=<timeout_retry_time>``; the OOM escalation renders
    ``--partition=<oom_retry_partition>`` and replays the captured script at
    its baked wall (the file's, or the campaign's 72 h ``--train-time``
    override, whichever was submitted). A target whose QOS cap is below the
    wall turns the recovery into a rejection at the moment it is needed --
    the shipped long-96core targets (48 h cap) against the 96 h retry and
    the 72 h replay failed this test before the 2026-08-30 retarget to
    extended-96core."""
    path = _campaign_config_path(name)
    if path is None:
        pytest.skip(f"no hpcjobs/configs/{name} in this checkout")
    raw = _raw_yaml(path)["cluster"]
    file_wall_h = _wall_hours(raw["time"])

    trp = raw.get("timeout_retry_partition")
    trt = raw.get("timeout_retry_time")
    assert trt is not None, name
    if trp is not None:
        cap = _QOS_MAX_WALL_HOURS[trp]
        assert _wall_hours(trt) <= cap, (
            f"{name}: timeout retry {trt} exceeds {trp}'s {cap:g} h cap")
    else:
        # Partition unset: the retry stays on the submit partition, whose
        # campaign home is extended-* (7-day) -- the wall must fit that.
        assert _wall_hours(trt) <= _QOS_MAX_WALL_HOURS["extended-96core"]

    orp = raw.get("oom_retry_partition")
    if orp is not None:
        cap = _QOS_MAX_WALL_HOURS[orp]
        for wall_h, label in ((file_wall_h, "file wall"),
                              (_V6_CAMPAIGN_TRAIN_WALL_H, "campaign wall")):
            assert wall_h <= cap, (
                f"{name}: the OOM re-route to {orp} replays the {label} "
                f"{wall_h:g} h above the {cap:g} h cap")


def test_v6_reference_names_the_five_group_files_in_order():
    """The reference file points at the ladder, in submission order, and says
    it is not itself the submission.

    It is the file the campaign's method is written in and the one an operator
    reaches for first; before the split it also carried the submission
    instruction, so it has to say where that moved or it reads as a 341-cell
    submission that is simply never made.
    """
    path, _cfg = _campaign_config(_V6_REFERENCE)
    text = open(path).read()
    positions = [text.index(name) for name in _V6_GROUP_FILES
                 if name in text]
    assert len(positions) == len(_V6_GROUP_FILES), (
        f"{path} must name all five group files; missing "
        f"{[n for n in _V6_GROUP_FILES if n not in text]}")
    assert positions == sorted(positions), (
        f"{path} names the group files out of submission order")
    assert "NOT THE SUBMISSION" in text, path
    assert "220 of the 341 cells" in text, path
