"""Per-architecture workflow matrix: the harness stage sequence at a tiny
identity, once per registered architecture.

SPEC_pretrain_fidelity_program.md 3.4 requires that, before any campaign YAML
is rendered, every architecture in the registry be driven through
``submit`` (dry-run) -> ``_datagen`` -> ``_pretrain`` -> the fidelity
certificate -> ``_preflight`` -> ``_train_task`` (two cells) ->
``_eval_one_spec`` (two cells) -> ``validate_run``, plus its spin-scaling
oracles, at def2-svp / grid level 1 against the repository's cached subset
ledger and CCSD references, with the held-out evaluation on a six-species
slice of the BH76 + W4-11 pool. The assertions are: every stage exits zero,
the expected artefacts exist, the certificate verdict is recorded, the
in-sample ``eval_df.csv`` and the sliced held-out channel are written, and the
architecture's oracles pass.

This module is a CALLER of the stage entry points, never a wrapper around
them: every stage runs as its own ``python -m`` subprocess with its own log,
exactly as SLURM would run it, so what the matrix verifies is the code the
cluster executes. ``runner`` is the single test seam (default
``subprocess.run``); with a fake runner the whole module is testable without
starting a process.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from xcquinox.alec.config import ARCHITECTURES

#: Cached inputs of the tiny identity, relative to the repository root.
CACHED_REFS_RELPATH = "notebooks/checkpoints_step7/external_refs"
CACHED_LEDGER_RELPATH = "notebooks/checkpoints_step7/alpha_on/subset_index_log.json"

#: Rendered grid config filename inside an architecture's work directory.
GRID_FILENAME = "grid.yaml"


def repo_root_path() -> Path:
    """The repository root, four parents up from this file.

    ``<root>/xcquinox/alec/cluster/workflow_matrix.py`` -> ``<root>``.
    """
    return Path(__file__).resolve().parents[3]


def template_path() -> Path:
    """The checked-in one-architecture template (package data)."""
    return Path(__file__).resolve().parent / "examples" / \
        "workflow_matrix_template.yaml"


def stage_cached_inputs(dest_root, *, repo_root) -> dict:
    """Copy the cached CCSD references into ``dest_root`` and locate the ledger.

    ``external_refs.precompute_all`` creates its cache directory, migrates
    legacy filenames inside it and writes a ``_run_log_<UTC>.json`` on EVERY
    call, and ``run_oep_cascade`` may rewrite a species npz; the repository
    copy of these references is tracked, so the matrix works on a copy (74 MB,
    one per work root, shared by every architecture) rather than a symlink
    farm, which would carry those writes back into the tree. Existing run logs
    are not copied.

    The subset ledger is read-only for the harness (only the JSON is read; no
    ``subset.traj`` is opened, see ``spec_builder``), so it is consumed in
    place.
    """
    dest_root = Path(dest_root)
    refs_src = Path(repo_root) / CACHED_REFS_RELPATH
    ledger = Path(repo_root) / CACHED_LEDGER_RELPATH
    if not refs_src.is_dir():
        raise FileNotFoundError(
            f"cached CCSD references not found at {refs_src}; the workflow "
            "matrix consumes the repository's step-7 cache."
        )
    if not ledger.is_file():
        raise FileNotFoundError(
            f"cached subset ledger not found at {ledger}; the workflow matrix "
            "consumes the repository's step-7 ledger."
        )
    refs_dst = dest_root / "_inputs" / "external_refs"
    if not refs_dst.exists():
        refs_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            refs_src, refs_dst,
            ignore=shutil.ignore_patterns("_run_log_*.json"))
    return {"external_refs_dir": str(refs_dst),
            "subset_ledger_path": str(ledger)}


def write_matrix_yaml(arch, out_dir, *, repo_root,
                      external_refs_dir=None, pretrain_data_dir=None) -> Path:
    """Render the one-architecture tiny grid config into ``<out_dir>/grid.yaml``.

    The template is parsed and its four CHANGE_ME values are replaced as data,
    not as text, so a malformed substitution cannot produce a syntactically
    valid but semantically wrong config. ``external_refs_dir`` and
    ``pretrain_data_dir`` default to per-architecture directories under
    ``out_dir``; the matrix passes shared ones so the 74 MB reference copy and
    the pretrain-data generation are paid once per shard instead of once per
    architecture.
    """
    import yaml

    if arch not in ARCHITECTURES:
        raise ValueError(
            f"{arch!r} is not a registered architecture; "
            f"valid names: {sorted(ARCHITECTURES)}"
        )
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with template_path().open() as f:
        raw = yaml.safe_load(f)

    if external_refs_dir is None:
        staged = stage_cached_inputs(out_dir, repo_root=repo_root)
        refs = staged["external_refs_dir"]
        ledger = staged["subset_ledger_path"]
    else:
        refs = str(Path(external_refs_dir).resolve())
        ledger = str((Path(repo_root) / CACHED_LEDGER_RELPATH).resolve())
    data_dir = Path(pretrain_data_dir).resolve() if pretrain_data_dir \
        else out_dir / "pretrain_data"
    # datagen writes into it and validate_grid_semantics warns when it is
    # absent on the submitting node.
    data_dir.mkdir(parents=True, exist_ok=True)

    raw["sweep"]["arch"] = [arch]
    raw["inputs"]["external_refs_dir"] = refs
    raw["inputs"]["subset_ledger_path"] = ledger
    raw["inputs"]["output_root"] = str(out_dir)
    raw["pretrain"]["data_dir"] = str(data_dir)

    path = out_dir / GRID_FILENAME
    with path.open("w") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=True)
    return path
