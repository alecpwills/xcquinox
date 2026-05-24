#!/usr/bin/env python
"""capture_notebook_spec_snapshot.py — capture the golden spec snapshot.

USER-RUN HELPER. This script is NOT executed by the test suite or by an agent.
It produces the golden-file fixture consumed by the slow faithfulness test
``xcquinox/alec/tests/test_cluster_spec_golden.py``.

What the snapshot is for
------------------------
The cluster harness (``xcquinox.alec.cluster``) is a de-notebooked extraction
of the step-7 spec-building logic in ``notebooks/_build_step7_notebook.py``.
The faithfulness test asserts the harness still reproduces the *physical
content* of a representative step-7 ``TrainingSpec`` — so a future refactor
that silently changes a target, an atom-energy anchor, a loss kwarg, or a
solver setting is caught.

To compare, the test needs a committed reference. This script builds one
representative step-7 ``TrainingSpec`` via the harness, serializes its
physically-meaningful content to JSON, stamps the current git SHA of
``notebooks/_build_step7_notebook.py`` (so a stale snapshot — captured before
the notebook builder changed — is detectable), and writes it to::

    xcquinox/alec/tests/data/notebook_spec_snapshot.json

How to run
----------
From the repo root, with the ``xcquinox`` conda env active::

    python scripts/capture_notebook_spec_snapshot.py

Then inspect the printed summary, ``git add`` the new
``xcquinox/alec/tests/data/notebook_spec_snapshot.json``, and commit it. The
slow test (``pytest -m slow xcquinox/alec/tests/test_cluster_spec_golden.py``)
will then run instead of skipping.

The representative spec
-----------------------
By default the snapshot captures the grid cell::

    arch=deep_combined_attn  loss=L5_gradnorm_vxc_step7
    metric=l2  subset_size=2  solver=oneshot

That is the cheapest cell (smallest subset, no-SCF solver) and matches the
``STEP7_SMOKE_ONLY`` cell the smoke harness already uses. Override with the
``--metric`` / ``--subset-size`` / ``--solver`` flags if a different cell is
wanted.

The consume-only harness no longer runs subset selection (that is a finished
offline pre-process whose result lives in ``subset_index_log.json``). For the
golden snapshot we only need a STABLE, valid subset to lock the spec's
structure, so a deterministic ledger entry is built from the first
``subset_size`` points of the (deterministically ordered) pool — no enumeration,
near-instant.

What the user must verify
-------------------------
After running, confirm the printed ``notebook_sha`` matches
``git log -1 --format=%H -- notebooks/_build_step7_notebook.py``. If you
captured against an uncommitted edit of the notebook builder, commit the
notebook builder first, then re-run this script so the SHA is meaningful.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys


# Repo root = parent of this script's directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOTEBOOK_BUILDER = os.path.join(
    REPO_ROOT, "notebooks", "_build_step7_notebook.py"
)
SNAPSHOT_PATH = os.path.join(
    REPO_ROOT, "xcquinox", "alec", "tests", "data",
    "notebook_spec_snapshot.json",
)

# Snapshot schema version — bump if the JSON layout below changes so a stale
# snapshot from an older script is rejected by the test.
SNAPSHOT_SCHEMA_VERSION = 1


def _git_sha_of(path: str) -> str:
    """Return the last-commit SHA touching ``path``, or a sentinel if it is
    not tracked / git is unavailable. A non-SHA sentinel makes a stale-vs-fresh
    determination impossible, which the test reports as a hard failure."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", path],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        sha = out.decode().strip()
        return sha or "UNTRACKED"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "GIT_UNAVAILABLE"


def _file_content_sha(path: str) -> str:
    """SHA-256 of the file's current bytes — detects uncommitted edits to the
    notebook builder that the commit-SHA alone would miss."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _solver_config_dict(sc) -> dict:
    """Serialize a ``SolverConfig`` to a plain dict of its physical settings."""
    def _name(v):
        return getattr(v, "name", None) or getattr(v, "value", None) or str(v)
    return {
        "mode": _name(sc.mode),
        "max_cycles": sc.max_cycles,
        "feature_policy": (
            _name(sc.feature_policy) if sc.feature_policy is not None else None
        ),
    }


def _molecule_dict(ms) -> dict:
    """Serialize a ``MoleculeSpec`` to its physically-meaningful content."""
    return {
        "name": ms.name,
        "atom": ms.atom,
        "basis": ms.basis,
        "charge": ms.charge,
        "spin": ms.spin,
        "atom_composition": dict(ms.atom_composition),
        "grid_level": ms.grid_level,
        "external_data_path_is_set": ms.external_data_path is not None,
    }


def _jsonable(obj):
    """Best-effort recursive coercion of harness objects to JSON-native types
    so ``loss_kwargs`` (which nests SolverConfig / tuples / dicts) round-trips
    deterministically."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(),
                                                        key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # SolverConfig (duck-typed) — it has mode + max_cycles.
    if hasattr(obj, "mode") and hasattr(obj, "max_cycles"):
        return _solver_config_dict(obj)
    # enum-like
    if hasattr(obj, "name") and hasattr(obj, "value"):
        return obj.name
    return repr(obj)


def _spec_snapshot(cell, spec) -> dict:
    """Serialize the physical content of one (GridCell, TrainingSpec) pair."""
    return {
        "grid_cell": {
            "arch": cell.arch,
            "loss": cell.loss,
            "metric": cell.metric,
            "subset_size": cell.subset_size,
            "solver": cell.solver,
        },
        "molecules": sorted(
            (_molecule_dict(ms) for ms in spec.molecules),
            key=lambda d: d["name"],
        ),
        "targets": _jsonable(spec.targets_dict),
        "atom_energies": _jsonable(spec.atom_energies_dict),
        "loss_name": spec.loss_name,
        "loss_kwargs": _jsonable(spec.loss_kwargs_dict),
        "solver_config": _solver_config_dict(spec.solver_config),
        "hyperparameters": {
            "n_steps": spec.n_steps,
            "lr_start": spec.lr_start,
            "lr_end": spec.lr_end,
            "lr_decay_start": spec.lr_decay_start,
            "grad_clip": spec.grad_clip,
            "seed": spec.seed,
            "pbe_anchor_weight": spec.pbe_anchor_weight,
        },
    }


def build_representative_spec(metric: str, subset_size: int, solver: str):
    """Build one representative step-7 ``TrainingSpec`` through the harness.

    Returns ``(GridCell, TrainingSpec)``. Raises on any harness error — this
    is a capture utility, so failing loudly is correct.
    """
    from xcquinox.alec.cluster.domain import get_domain_profile
    from xcquinox.alec.cluster.grid_config import (
        GridConfig, SweepAxes, SolverNamed, HyperParams, InputPaths,
        PretrainConfig, ClusterResources, expand_grid,
    )
    from xcquinox.alec.cluster.spec_builder import (
        build_training_specs, _ledger_key,
    )
    from xcquinox.alec.training_points import build_dfs_pool_points

    domain = get_domain_profile("dfs_step7")
    points = build_dfs_pool_points(bh76_mode="reaction_energy")

    # Single-cell sweep — only the requested representative cell.
    sweep = SweepAxes(
        arch=("deep_combined_attn",),
        loss=("L5_gradnorm_vxc_step7",),
        metric=(metric,),
        subset_size=(subset_size,),
        solver=(solver,),
    )
    solvers = {
        "oneshot": SolverNamed(mode="ONESHOT", max_cycles=0),
        "full_3": SolverNamed(
            mode="FULL", max_cycles=3, feature_policy="REASSEMBLE"
        ),
    }
    hyperparams = HyperParams(
        n_steps=100, lr_start=1e-2, lr_end=1e-5, lr_decay_start=0.2,
        grad_clip=1.0, gradnorm_alpha=1.5, vxc_weight=0.01,
        density_weight=0.1, pbe_anchor_weight=0.0, seed=42,
    )
    inputs = InputPaths(
        external_refs_dir="/nonexistent/external_refs",
        subset_ledger_path="/nonexistent/subset_index_log.json",
        basis="def2-svp", grid_level=1,
        output_root="/nonexistent/runs",
    )
    cluster = ClusterResources(
        partition="short-96core-shared", time="02:00:00", mem="",
        cpus_per_task=24, array_throttle=4, eval_array_throttle=4,
        max_concurrent_tasks=8,
    )
    pretrain = PretrainConfig(
        data_dir="/nonexistent/pretrain_data",
        pretrain_root="/nonexistent/pretrain",
    )
    cfg = GridConfig(
        sweep=sweep, solvers=solvers, hyperparams=hyperparams,
        inputs=inputs, pretrain=pretrain, cluster=cluster,
        domain_profile="dfs_step7",
    )

    cells = expand_grid(cfg)
    if len(cells) != 1:
        raise RuntimeError(
            f"expected a single-cell sweep, got {len(cells)} cells"
        )
    cell = cells[0]

    # The consume-only harness no longer runs subset selection (that is a
    # finished offline pre-process whose result lives in subset_index_log.json).
    # For the golden snapshot we only need a STABLE, valid subset to lock the
    # spec's structure (molecules/targets/atom_energies/loss_kwargs), so build a
    # deterministic ledger entry from the first `subset_size` pool points (the
    # pool order from build_dfs_pool_points is itself deterministic).
    if subset_size > len(points):
        raise RuntimeError(
            f"subset_size {subset_size} exceeds the pool size {len(points)}"
        )
    chosen_names = [tp.name for tp in points[:subset_size]]
    print(
        f"Representative cell: metric={metric} subset_size={subset_size} "
        f"solver={solver}; chosen points (first {subset_size} of "
        f"{len(points)}): {chosen_names}",
        flush=True,
    )

    # New consume-only ledger schema: {"<metric>/<r>": {"point_names": [...]}}.
    ledger = {
        _ledger_key(metric, subset_size): {
            "metric": metric,
            "subset_size": subset_size,
            "point_names": chosen_names,
        }
    }

    built = build_training_specs(
        points, ledger, cfg, domain,
        run_dir=os.path.join(REPO_ROOT, "_snapshot_run_dir"),
        cells=cells,
    )
    return built[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the golden step-7 TrainingSpec snapshot consumed by "
            "the slow faithfulness test test_cluster_spec_golden.py. "
            "USER-RUN: build the spec via the harness, then commit the "
            "generated JSON fixture."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--metric", default="l2", choices=["l2", "jsd"],
                        help="subset-selection metric for the captured cell")
    parser.add_argument("--subset-size", type=int, default=2,
                        help="subset size for the captured cell")
    parser.add_argument("--solver", default="oneshot",
                        choices=["oneshot", "full_3"],
                        help="named solver for the captured cell")
    parser.add_argument("--output", default=SNAPSHOT_PATH,
                        help="snapshot output path (default: the fixture path)")
    args = parser.parse_args(argv)

    if not os.path.isfile(NOTEBOOK_BUILDER):
        print(f"ERROR: notebook builder not found: {NOTEBOOK_BUILDER}",
              file=sys.stderr)
        return 1

    notebook_sha = _git_sha_of(NOTEBOOK_BUILDER)
    notebook_content_sha = _file_content_sha(NOTEBOOK_BUILDER)

    cell, spec = build_representative_spec(
        args.metric, args.subset_size, args.solver
    )

    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "notebook_builder_path": "notebooks/_build_step7_notebook.py",
        "notebook_sha": notebook_sha,
        "notebook_content_sha256": notebook_content_sha,
        "spec": _spec_snapshot(cell, spec),
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
        f.write("\n")

    print()
    print("=" * 70)
    print("Snapshot written:")
    print(f"  path           : {out_path}")
    print(f"  schema_version : {SNAPSHOT_SCHEMA_VERSION}")
    print(f"  notebook_sha   : {notebook_sha}")
    print(f"  cell           : metric={cell.metric} "
          f"subset_size={cell.subset_size} solver={cell.solver}")
    print(f"  n molecules    : {len(snapshot['spec']['molecules'])}")
    print(f"  n targets      : {len(snapshot['spec']['targets'])}")
    print("=" * 70)
    print()
    print("VERIFY: notebook_sha must match")
    print("  git log -1 --format=%H -- notebooks/_build_step7_notebook.py")
    print("If you captured against an uncommitted notebook-builder edit, "
          "commit it first and re-run this script.")
    print()
    print("Then: git add", os.path.relpath(out_path, REPO_ROOT),
          "&& git commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
