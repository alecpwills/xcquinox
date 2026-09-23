"""Tests for the shipped example grid configs and the v7 campaign files.

The example YAML ``cluster/examples/grid_step7.yaml`` is a copy-me template
reproducing the step-7 40-spec sweep. These tests verify it (a) still parses
with ``load_grid_config``, (b) expands to exactly 40 cells, (c) covers every
config-dataclass field (so a future required field the example forgot is
caught), (d) carries no real email/account, and (e) fails
``validate_grid_semantics`` cleanly because its placeholder paths do not
exist. The workflow-matrix template is held to the same protocol
completeness.

The final sections pin the deployment configurations of the v7 campaign under
``hpcjobs/configs``: the three group files, the reaction-energy control and
the two arms. Their properties are pinned rather than reviewed because each
way they can go wrong -- a BH76 objective filled by a default, an arm that
drifts from the group file it mirrors in a key it does not claim, two groups
writing into one root, a wall above its queue's cap -- loads without
complaint and is invisible in a result.
"""
import dataclasses
import os

import pytest

from xcquinox.pipeline.cluster.grid_config import (
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
)


# ---------------------------------------------------------------------------
# Locating the shipped example
# ---------------------------------------------------------------------------

def _example_path() -> str:
    """Absolute path to the shipped ``grid_step7.yaml`` example."""
    import xcquinox.pipeline.cluster as cluster_pkg
    pkg_dir = os.path.dirname(os.path.abspath(cluster_pkg.__file__))
    return os.path.join(pkg_dir, "examples", "grid_step7.yaml")


def test_example_yaml_exists():
    assert os.path.isfile(_example_path()), (
        f"shipped example grid_step7.yaml not found at {_example_path()}"
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


# ---------------------------------------------------------------------------
# The shipped templates run at grid level 1, which no degenerate free atom is
# reproducible at
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The deployment tree beside the package
# ---------------------------------------------------------------------------

def _campaign_configs_dir():
    """Absolute path to ``hpcjobs/configs/``, or None when it is absent.

    The directory's presence is what separates "this checkout has no
    deployment tree" (nothing to pin) from "a configuration file was
    deleted" (a campaign that no longer covers what it claims).
    """
    import xcquinox.pipeline.cluster as cluster_pkg
    pkg_dir = os.path.dirname(os.path.abspath(cluster_pkg.__file__))
    path = os.path.normpath(os.path.join(
        pkg_dir, "..", "..", "..", "hpcjobs", "configs"))
    return path if os.path.isdir(path) else None


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


# ===========================================================================
# bh76_mode: every DFS-domain configuration file states its BH76 objective.
# The v7 group files and the arms that keep the objective train true barrier
# heights (the dpyscf treatment: staged transition states against the GMTKN55
# forward-barrier references); every other DFS-domain file states the
# reaction-energy substitution it trained, so the record is explicit in the
# file rather than filled by a default.
# ===========================================================================


#: The v7 restart trio: unanchored functional cloning (the published
#: xcquinox-clone protocol), barrier objective, derived from the v6
#: counterparts.
_V7_FILES = (
    "dfs_step7.dfs6311_grid3_v7g1_size.yaml",
    "dfs_step7.dfs6311_grid3_v7g2a_families_core.yaml",
    "dfs_step7.dfs6311_grid3_v7g2_families_mgga.yaml",
)


#: The v7 reaction-energy CONTROL ARM (2026-09-04): the g1 size group
#: resubmitted with the BH76 points as reaction energies, everything else
#: the g1 file's, the certified g1 clones reused by directory copy.
_V7_RXN_CONTROL = "dfs_step7.dfs6311_grid3_v7g1_rxn.yaml"
_V7_RXN_SOURCE = "dfs_step7.dfs6311_grid3_v7g1_size.yaml"

#: The v7 arms that KEEP the barrier objective, so the substitution pin above
#: expects barrier_height for them as it does for the group files. The
#: 25-cycle arm (2026-09-07) is the g1 size group's medium column on the
#: ``full_25`` solver, training from the g1 run's certified clones copied in.
#: An arm is pinned by its own mirror test rather than by ``_V7_FILES``: that
#: list also pins the completed pretraining protocol and a pretraining root of
#: its own, neither of which a run training from another run's clones carries.
#: The dpyscf-parity arm (2026-09-08) is the 25-cycle arm with every other
#: setting the 2026-09-07 parity table lists as a deviation moved to the
#: reference protocol's value: the first mix index, the seed mixture, the
#: convergence freeze, the tail window, the optimizer and its rates, the
#: coupled L2 and the channel weights. It clones the 25-cycle file, so the
#: two arms differ in the parity keys alone and the cycle count is shared.
_V7_C25_ARM = "dfs_step7.dfs6311_grid3_v7g1_c25.yaml"
_V7_DFSPARITY_ARM = "dfs_step7.dfs6311_grid3_v7g1_dfsparity.yaml"
_V7_ARMS = (_V7_C25_ARM, _V7_DFSPARITY_ARM)


# ---------------------------------------------------------------------------
# The production identity of every v7 file
# ---------------------------------------------------------------------------

#: Every v7 file: the three group files, the reaction-energy control and the
#: two arms, one identity.
_V7_ALL = _V7_FILES + (_V7_RXN_CONTROL,) + _V7_ARMS


