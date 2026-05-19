"""Tests for the shipped example grid config ``cluster/examples/grid_step7.yaml``.

The example YAML is a copy-me template reproducing the step-7 40-spec sweep.
These tests verify it (a) still parses with ``load_grid_config``, (b) expands
to exactly 40 cells, (c) covers every config-dataclass field (so a future
required field the example forgot is caught), (d) carries no real
email/account, and (e) fails ``validate_grid_semantics`` cleanly because its
placeholder paths do not exist.
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
    """examples/ ships as package DATA, not as a subpackage — no __init__.py."""
    examples_dir = os.path.dirname(_example_path())
    assert not os.path.exists(os.path.join(examples_dir, "__init__.py")), (
        "cluster/examples/ must NOT contain __init__.py — it is package data"
    )


# ---------------------------------------------------------------------------
# Load + expand (filesystem-free — no validate_grid_semantics here)
# ---------------------------------------------------------------------------

def test_example_loads_and_expands_to_40():
    """grid_step7.yaml loads via load_grid_config and expands to 40 specs.

    This is filesystem-free: it calls only load_grid_config + expand_grid.
    validate_grid_semantics is NOT called — the example's placeholder input
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
# Structural completeness — every dataclass field is covered
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
    shipped example forgot to fill in — without a default such a field would
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
            "grid_step7.yaml — the example must be updated to set it"
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
    semantic-validation pass surfaces them — either as a hard error or via a
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
        cfg.pretrain.pretrain_root,
    ):
        assert p is not None and not os.path.exists(p), (
            f"example placeholder path {p!r} unexpectedly exists — the "
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
        # No hard error — then it MUST have warned about a not-found path.
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
            f"credential ({forbidden!r}) — it must ship only placeholders"
        )
    # mail_user is the example.com placeholder
    cfg = load_grid_config(_example_path()) if pytest.importorskip("yaml") \
        else None
    assert cfg.cluster.mail_user == "CHANGE_ME@example.com"
    assert cfg.cluster.account == "CHANGE_ME"
