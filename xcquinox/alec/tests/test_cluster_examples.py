"""Tests for the shipped example grid configs and the campaign-v6 submission.

The example YAML ``cluster/examples/grid_step7.yaml`` is a copy-me template
reproducing the step-7 40-spec sweep. These tests verify it (a) still parses
with ``load_grid_config``, (b) expands to exactly 40 cells, (c) covers every
config-dataclass field (so a future required field the example forgot is
caught), (d) carries no real email/account, and (e) fails
``validate_grid_semantics`` cleanly because its placeholder paths do not
exist. The workflow-matrix template is held to the same protocol
completeness.

The final section pins the deployment configuration those templates are
copied into: ``hpcjobs/configs/dfs_step7.dfs6311_grid3_v6.yaml``, the
pretraining-fidelity program's campaign. Its properties are pinned rather
than reviewed because each way it can go wrong -- an architecture added to
the registry after it was written, a waiver carried over from a template, a
loosened certificate tolerance, a pre-protocol pretraining footing -- loads
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

def _v6_config_path():
    """Absolute path to the campaign-v6 configuration, or None.

    ``hpcjobs/`` sits beside the package rather than inside it, so a source or
    wheel checkout can be missing it entirely; an absent file is nothing to
    pin, not a failure.
    """
    import xcquinox.alec.cluster as cluster_pkg
    pkg_dir = os.path.dirname(os.path.abspath(cluster_pkg.__file__))
    path = os.path.normpath(os.path.join(
        pkg_dir, "..", "..", "..", "hpcjobs", "configs",
        "dfs_step7.dfs6311_grid3_v6.yaml"))
    return path if os.path.isfile(path) else None


def _v6_config():
    """The loaded v6 config, skipping when the deployment tree is absent."""
    pytest.importorskip("yaml")
    path = _v6_config_path()
    if path is None:
        pytest.skip("no hpcjobs/configs/dfs_step7.dfs6311_grid3_v6.yaml in "
                    "this checkout")
    return path, load_grid_config(path)


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
