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

The last section pins the three arms of campaign 1 of the v8 program, which
differ from one another in the seed axis alone.
"""
import dataclasses
import os
import warnings

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


# ===========================================================================
# Campaign 1 of the v8 program: three arms differing in the seed axis alone
# ===========================================================================

#: The campaign-1 files. Arm S (``allsc``) is the base the other two clone: it
#: trains every molecule self-consistently from the mixed atomic start and
#: fits the four clones the other arms copy in. Arm P (``parity``) adds the
#: published self-consistency split and its weight on the non-self-consistent
#: points; arm A (``coldstart``) starts every SCF from the superposition of
#: atomic densities and drops the mixture. Everything else is held equal to
#: arm S by the diff pin below, so a drift in a key no arm claims -- a
#: pretraining knob, a wall, a ledger path -- is a failure rather than a
#: difference nothing records.
_V8_ARM_S = "dfs_step8.v8_dfs_allsc.yaml"
_V8_ARMS = (
    _V8_ARM_S,
    "dfs_step8.v8_dfs_parity.yaml",
    "dfs_step8.v8_dfs_coldstart.yaml",
)

#: The campaign's architecture axis, written in the order ``expand_grid``
#: canonicalizes to (``_canon_axis`` = ``sorted(set(...))``), so a cell's index
#: in the expansion is its SLURM array task id and the files read in task
#: order.
_V8_ARCHS = ("deep_3x16", "deep_attn_3x16", "deep_geom_3x16",
             "deep_geom_attn_3x16")

#: The keys each arm moves away from arm S, with the value it moves them to.
#: ``inputs.output_root`` moves in both arms and is added to the expected diff
#: separately: two arms writing into one root is the failure this set exists
#: to make impossible.
_V8_SEED_KEYS = {
    "dfs_step8.v8_dfs_parity.yaml": {
        "hyperparams.respect_sc_flag": True,
        "hyperparams.nonsc_weight": 0.5,
    },
    "dfs_step8.v8_dfs_coldstart.yaml": {
        "hyperparams.seed_mix_atomic": False,
        "inputs.seed_xc": "minao",
    },
}

#: Arm S's own value of every key an arm moves, so the diff is pinned on both
#: sides: a change made in arm S rather than in the arm claiming it would
#: otherwise leave the differing set intact.
_V8_ARM_S_SEED_VALUES = {
    "hyperparams.respect_sc_flag": False,
    "hyperparams.nonsc_weight": 1.0,
    "hyperparams.seed_mix_atomic": True,
    "inputs.seed_xc": "auto",
}

#: The published cloning protocol as the pretrain block of every arm carries
#: it: the sampled-row objective on 800 points per system, the 20000-step
#: schedule from 1e-3 to 1e-5 with the constant tail from 0.9, no clip, the
#: paper footing with no energy term, no validation split and no stop
#: criterion, and no Slim addition to the set. The v7 files run the other
#: values of the same keys (integration weighting, decay to the last step,
#: clip 1.0, spin-channel footing, energy weight 0.1, a 20 percent split,
#: patience 300), which is why each is stated here rather than inherited.
_V8_PRETRAIN_VALUES = (
    ("loss_weighting", "rho_w_sampled"),
    ("points_per_system", 800),
    ("sampling_seed", 42),
    ("n_steps", 20000),
    ("lr_start", 1.0e-3),
    ("lr_end", 1.0e-5),
    ("lr_decay_start", 0.5),
    ("lr_decay_end", 0.9),
    ("grad_clip", 0.0),
    ("exchange_footing", "paper"),
    ("energy_term_weight", 0.0),
    ("validation_fraction", 0.0),
    ("patience", 0),
    ("slim_set", ""),
)

#: The two set flags, held by identity rather than by equality: ``1 == True``,
#: so an integer left in the YAML would pass an equality pin.
_V8_PRETRAIN_FLAGS = (("dfs_set", True), ("pool_atoms", True))

#: The model class of the campaign: the published clone's coordinates and
#: uniform-gas gate, unanchored.
_V8_MODEL_VALUES = (("descriptor_coordinates", "paper"), ("ueg_gate", "x2"))


def _v8_paths():
    """``{arm file name: absolute path}``, or None when the tree is absent.

    The deployment directory's presence is what separates a checkout without a
    deployment tree (nothing to pin) from a deleted configuration file (a
    campaign that no longer states what it runs): an absent directory skips,
    an absent file fails.
    """
    cfg_dir = _campaign_configs_dir()
    if cfg_dir is None:
        return None
    return {name: os.path.join(cfg_dir, name) for name in _V8_ARMS}


def _require_v8_files():
    """The campaign-1 paths, skipping without the tree and failing without a
    file."""
    paths = _v8_paths()
    if paths is None:
        pytest.skip("cluster config tree not present in this checkout")
    for name, path in paths.items():
        assert os.path.isfile(path), (
            f"campaign-1 configuration missing: {path}; the three arms of "
            "campaign 1 are tracked files, not local copies"
        )
    return paths


def _flat_yaml(path: str) -> dict:
    """The raw YAML of ``path`` flattened to dotted keys.

    Comments carry no key and are absent from the result: ``yaml.safe_load``
    drops them, so a header naming the arm is not a difference between two
    files.
    """
    def _walk(mapping, prefix=""):
        flat = {}
        for key, value in (mapping or {}).items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict):
                flat.update(_walk(value, dotted + "."))
            else:
                flat[dotted] = value
        return flat
    return _walk(_raw_yaml(path))


def _wall_hours(literal: str) -> float:
    """Hours in an ``HH:MM:SS`` SLURM wall literal."""
    hh, mm, ss = str(literal).split(":")
    return int(hh) + int(mm) / 60.0 + int(ss) / 3600.0


def test_v8_arms_load_expand_to_four_cells_and_pass_the_semantics_guard():
    """Every campaign-1 arm loads, expands to its four cells and is accepted
    by the login-node guard.

    Oracle: ``load_grid_config`` + ``expand_grid`` +
    ``validate_grid_semantics`` executed on the tracked file, with the
    training-point pool the run draws from (``domain.DFS_POOL_SIZE``, the
    size the ``subset_size: 26`` axis is the whole of). The guard's advisory
    path warnings are recorded and not asserted on: the scratch roots it looks
    for resolve on the cluster alone. What the guard decides here is the seed
    axis -- the mixture requires per-molecule updates and refuses a ``minao``
    seed beside it, the self-consistency flag requires per-molecule updates --
    so an arm that states a seed protocol its training loop never applies
    fails at load rather than after a fortnight of wall time.
    """
    pytest.importorskip("yaml")
    from xcquinox.pipeline.cluster.domain import DFS_POOL_SIZE
    from xcquinox.pipeline.cluster.grid_config import validate_grid_semantics

    class _PoolStub:
        """The one attribute ``validate_grid_semantics`` reads off a domain."""
        pool_size = DFS_POOL_SIZE

    for name, path in _require_v8_files().items():
        cfg = load_grid_config(path)
        cells = expand_grid(cfg)
        assert [c.arch for c in cells] == list(_V8_ARCHS), (
            f"{name}: arch axis expanded to {[c.arch for c in cells]}, "
            f"expected the four architectures {list(_V8_ARCHS)} in that order"
        )
        assert list(_raw_yaml(path)["sweep"]["arch"]) == list(_V8_ARCHS), (
            f"{name}: the arch axis as written differs from the expansion "
            "order, so a cell's index is no longer its array task id"
        )
        assert len(cells) == 4, f"{name}: {len(cells)} cells, expected 4"
        for cell in cells:
            assert cell.solver == "full_25", (
                f"{name}: cell {cell.arch} runs solver {cell.solver!r}")
            assert cell.subset_size == 26, (
                f"{name}: cell {cell.arch} trains on subset size "
                f"{cell.subset_size}, expected the whole pool (26)")
            assert cell.metric == "jsd", (
                f"{name}: cell {cell.arch} draws its subset under metric "
                f"{cell.metric!r}")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            validate_grid_semantics(cfg, _PoolStub())


def test_v8_arms_differ_from_arm_s_in_the_seed_keys_alone():
    """Each arm departs from arm S in its own seed keys and the output root.

    Oracle: the flattened raw YAML of the tracked files, differenced key by
    key. The campaign's result is a comparison of three seed protocols, which
    only reads as one if nothing else moved: a pretraining knob, a solver
    field or a wall that drifted into one arm would be attributed to the seed
    start. Both sides of every moved key are pinned, so a change made in arm S
    instead of in the arm claiming it is caught as well.
    """
    pytest.importorskip("yaml")
    paths = _require_v8_files()
    base = _flat_yaml(paths[_V8_ARM_S])
    absent = object()
    for key, value in _V8_ARM_S_SEED_VALUES.items():
        if isinstance(value, bool):
            assert base.get(key, absent) is value, (
                f"{_V8_ARM_S}: {key} is {base.get(key, absent)!r}, "
                f"expected {value!r}")
        else:
            assert base.get(key, absent) == value, (
                f"{_V8_ARM_S}: {key} is {base.get(key, absent)!r}, "
                f"expected {value!r}")
    for name in _V8_ARMS[1:]:
        other = _flat_yaml(paths[name])
        differing = {
            key for key in set(base) | set(other)
            if base.get(key, absent) != other.get(key, absent)
        }
        expected = set(_V8_SEED_KEYS[name]) | {"inputs.output_root"}
        assert differing == expected, (
            f"{name} differs from {_V8_ARM_S} in {sorted(differing)}, "
            f"expected exactly {sorted(expected)}"
        )
        for key, value in _V8_SEED_KEYS[name].items():
            if isinstance(value, bool):
                assert other[key] is value, (
                    f"{name}: {key} is {other[key]!r}, expected {value!r}")
            else:
                assert other[key] == value, (
                    f"{name}: {key} is {other[key]!r}, expected {value!r}")
    roots = {name: _flat_yaml(path)["inputs.output_root"]
             for name, path in paths.items()}
    assert len(set(roots.values())) == len(_V8_ARMS), (
        f"the three arms share an output root: {roots}; two runs writing "
        "into one root interleave their run directories"
    )
    for name, root in roots.items():
        arm = name.split(".")[1]
        assert root.rstrip("/").endswith("/" + arm), (
            f"{name}: output_root {root!r} does not end in the arm's own "
            f"name {arm!r}, so a pull cannot tell the arms apart by path"
        )


def test_v8_arms_carry_the_published_pretraining_protocol_and_the_paper_model_block(
        tmp_path):
    """Every arm states the published cloning protocol, the published model
    class and the cold-start reporting path, and survives the resolved-config
    round trip.

    Oracle: the loaded ``GridConfig`` of each tracked file, and a second load
    of ``_config_to_raw_dict`` written back out as YAML. The round trip is the
    part that is not visible in the source file: every stage after submission
    rebuilds the run from ``resolved_config.yaml``, so a key that section
    drops resolves the run to a default while the source file states
    otherwise. The pretraining root is checked against every v7 file's: the
    campaign's data identity differs from the v7 rows' in the footing (the
    coordinates and the gate are network fields, not columns of the file),
    and a shared root would either read a file fitted under the other footing
    or rewrite one a completed campaign depends on.
    """
    pytest.importorskip("yaml")
    import yaml
    from xcquinox.pipeline.cluster.__main__ import _config_to_raw_dict

    cfg_dir = _campaign_configs_dir()
    v7_roots = {}
    for name in _V7_ALL:
        path = os.path.join(cfg_dir, name)
        assert os.path.isfile(path), f"tracked v7 config missing: {path}"
        v7_roots[name] = _raw_yaml(path)["pretrain"]["data_dir"]

    for name, path in _require_v8_files().items():
        cfg = load_grid_config(path)
        for key, value in _V8_PRETRAIN_VALUES:
            assert getattr(cfg.pretrain, key) == value, (
                f"{name}: pretrain.{key} is "
                f"{getattr(cfg.pretrain, key)!r}, the published protocol "
                f"states {value!r}")
        for key, value in _V8_PRETRAIN_FLAGS:
            assert getattr(cfg.pretrain, key) is value, (
                f"{name}: pretrain.{key} is "
                f"{getattr(cfg.pretrain, key)!r}, expected {value!r}")
        for v7_name, root in v7_roots.items():
            assert cfg.pretrain.data_dir != root, (
                f"{name}: pretrain.data_dir is {v7_name}'s root {root!r}; "
                "the campaign's rows are posed on a different footing and "
                "coordinate set and need a root of their own")
        assert cfg.model.parent_anchor is False, (
            f"{name}: model.parent_anchor is {cfg.model.parent_anchor!r}; "
            "the campaign clones the parent by fitting, not by construction")
        for key, value in _V8_MODEL_VALUES:
            assert getattr(cfg.model, key) == value, (
                f"{name}: model.{key} is {getattr(cfg.model, key)!r}, "
                f"expected {value!r}")
        assert cfg.eval_coldstart is True, (
            f"{name}: eval_coldstart is {cfg.eval_coldstart!r}; the "
            "campaign reports the cold-start channel pair")
        assert cfg.cluster.preflight_coldstart_census is True, (
            f"{name}: cluster.preflight_coldstart_census is "
            f"{cfg.cluster.preflight_coldstart_census!r}; the census over "
            "the training species is what the seed comparison is read with")
        assert cfg.hyperparams.update_scheme == "per_molecule", (
            f"{name}: hyperparams.update_scheme is "
            f"{cfg.hyperparams.update_scheme!r}; the seed mixture and the "
            "self-consistency flag are applied by the per-molecule loop "
            "alone")
        assert cfg.inputs.held_out_pools == ("bh76", "w411"), (
            f"{name}: inputs.held_out_pools is "
            f"{cfg.inputs.held_out_pools!r}; the in-loop validation slice is "
            "drawn from these pools, so a third pool would change the "
            "validation set and its cost in every arm. The diet set is "
            "re-evaluated from the pull instead")

        raw = _config_to_raw_dict(cfg)
        resolved = str(tmp_path / f"resolved_{name}")
        with open(resolved, "w") as handle:
            yaml.safe_dump(raw, handle, default_flow_style=False,
                           sort_keys=True)
        back = load_grid_config(resolved)
        assert dataclasses.asdict(back.pretrain) == \
            dataclasses.asdict(cfg.pretrain), (
                f"{name}: the pretrain block does not survive the "
                "resolved-config round trip")
        assert dataclasses.asdict(back.model) == \
            dataclasses.asdict(cfg.model), (
                f"{name}: the model block does not survive the "
                "resolved-config round trip")
        assert back.inputs.seed_xc == cfg.inputs.seed_xc
        assert back.inputs.held_out_pools == cfg.inputs.held_out_pools
        assert back.hyperparams.seed_mix_atomic is \
            cfg.hyperparams.seed_mix_atomic
        assert back.hyperparams.respect_sc_flag is \
            cfg.hyperparams.respect_sc_flag
        assert back.hyperparams.nonsc_weight == cfg.hyperparams.nonsc_weight
        assert back.eval_coldstart is cfg.eval_coldstart
        assert back.cluster.preflight_coldstart_census is \
            cfg.cluster.preflight_coldstart_census


def test_v8_arms_retry_walls_fit_the_queue_caps():
    """Every wall the campaign requests sits inside the QOS cap of the queue
    it is requested on, and the train wall forces the queue.

    Oracle: the walls as loaded, against the measured caps in
    ``_QOS_MAX_WALL_HOURS``. The train wall exceeds every long-* cap, so the
    campaign cannot be submitted on one of those queues and the submission
    sheet names ``extended-96core``; the timeout escalation stays on that
    queue at its own cap, which is the case a wall-killed cell with no
    checkpoint takes. SLURM rejects an over-cap wall only when the sbatch
    carrying it runs, i.e. for a retry key days into the campaign, at recovery
    time.
    """
    pytest.importorskip("yaml")
    extended_cap = _QOS_MAX_WALL_HOURS["extended-96core"]
    for name, path in _require_v8_files().items():
        cl = load_grid_config(path).cluster
        train = _wall_hours(cl.time)
        assert train == 96.0, (
            f"{name}: cluster.time is {cl.time!r} ({train} h), the campaign's "
            "train wall is 96 h")
        assert train <= extended_cap, (
            f"{name}: cluster.time {cl.time!r} exceeds extended-96core's "
            f"{extended_cap} h cap")
        assert train > _QOS_MAX_WALL_HOURS["long-96core-shared"], (
            f"{name}: cluster.time {cl.time!r} fits long-96core-shared's "
            f"{_QOS_MAX_WALL_HOURS['long-96core-shared']} h cap, so the "
            "submission sheet's choice of extended-96core is no longer "
            "forced by the wall")
        for qos, cap in _QOS_MAX_WALL_HOURS.items():
            if qos.startswith("long-"):
                assert train > cap, (
                    f"{name}: cluster.time {cl.time!r} fits {qos}'s {cap} h "
                    "cap")
        assert cl.timeout_retry_partition == "extended-96core", (
            f"{name}: cluster.timeout_retry_partition is "
            f"{cl.timeout_retry_partition!r}; the escalation stays on the "
            "queue whose cap admits it")
        retry = _wall_hours(cl.timeout_retry_time)
        assert retry == 168.0, (
            f"{name}: cluster.timeout_retry_time is "
            f"{cl.timeout_retry_time!r} ({retry} h), expected 168 h")
        assert retry <= extended_cap, (
            f"{name}: the timeout escalation asks {retry} h on "
            f"{cl.timeout_retry_partition}, whose cap is {extended_cap} h; "
            "it would be rejected exactly when it fired")
        pretrain = _wall_hours(cl.pretrain_time)
        assert pretrain == 48.0, (
            f"{name}: cluster.pretrain_time is {cl.pretrain_time!r} "
            f"({pretrain} h), expected 48 h")
        assert pretrain <= extended_cap, (
            f"{name}: cluster.pretrain_time {cl.pretrain_time!r} exceeds the "
            f"{extended_cap} h cap")
        # with inline_eval no evaluation job is rendered, so eval_time is
        # not a wall of this campaign; the reference job's is
        references = _wall_hours(cl.benchmark_refs_time)
        assert references == 24.0, (
            f"{name}: cluster.benchmark_refs_time is "
            f"{cl.benchmark_refs_time!r} ({references} h), expected 24 h")
        assert references <= extended_cap, (
            f"{name}: cluster.benchmark_refs_time {cl.benchmark_refs_time!r} "
            f"exceeds the {extended_cap} h cap")
        assert cl.mail_user == "alec.wills@stonybrook.edu", (
            f"{name}: cluster.mail_user is {cl.mail_user!r}; job mail goes "
            "to the institutional address")
        assert cl.mail_type == "BEGIN,END,FAIL", (
            f"{name}: cluster.mail_type is {cl.mail_type!r}; submission, "
            "completion and failure each have to reach the inbox")
