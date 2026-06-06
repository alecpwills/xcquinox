"""Orchestration tests for scripts/generate_step7_subsets.py.

The heavy library functions (PBE-SCF descriptor extraction, exhaustive
``select_subset``) are tested elsewhere; here we verify the runner's ledger /
subset.traj layout and that it covers BOTH metrics (l2, jsd) across the
requested alpha modes, with ``select_subset`` monkeypatched so no
combinatorics run.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "notebooks"))

g = importlib.import_module("generate_step7_subsets")

import numpy as np
_KEYS = ("rho_third", "s", "alpha")


def _dummy_href():
    return {k: np.zeros(g.ss.NBINS) for k in _KEYS}


def _dummy_edges():
    return {k: np.linspace(0.0, 1.0, g.ss.NBINS + 1) for k in _KEYS}


def test_mode_weights_mapping():
    """alpha_off drops alpha; alpha_on weights all descriptors equally."""
    assert g._MODE_WEIGHTS["alpha_off"] == {"alpha": 0.0}
    assert g._MODE_WEIGHTS["alpha_on"] is None
    assert "l2" in g.METRICS and "jsd" in g.METRICS


def test_resolve_sizes_include_full():
    """--include-full appends the full-pool size (complete training set) without
    touching the shared SUBSET_SIZES tuple; idempotent + order-preserving."""
    base = (1, 2, 3)
    assert g._resolve_sizes(base, 26, include_full=False) == [1, 2, 3]
    assert g._resolve_sizes(base, 26, include_full=True) == [1, 2, 3, 26]
    # already present -> no duplicate
    assert g._resolve_sizes((1, 2, 26), 26, include_full=True) == [1, 2, 26]
    # canonical step-7 ladder + full pool, for the 26-point DFS pool
    assert g._resolve_sizes(g.SUBSET_SIZES, 26, include_full=True)[-1] == 26
    assert 26 not in g.SUBSET_SIZES        # shared constant left untouched


def test_run_mode_writes_both_metrics_and_trajs(tmp_path, monkeypatch):
    """_run_mode writes a ledger with BOTH metrics for every subset size and a
    subset.traj per (metric, size, solver), under the mode's root."""
    from xcquinox.alec.training_points import build_dfs_pool_points

    points = build_dfs_pool_points()

    # Stub select_subset: return the first r pool indices + a dummy value, so
    # no C(n, r) enumeration runs. Signature mirrors the real one.
    def _fake_select_subset(point_descriptors, edges, h_ref, *, r, metric,
                            descriptor_weights=None, progress=True, **kw):
        return tuple(range(r)), 0.123 + (0.0 if metric == "l2" else 0.001)

    monkeypatch.setattr(g.ss, "select_subset", _fake_select_subset)
    monkeypatch.setattr(g, "CHECKPOINTS", tmp_path)

    ledger_path = g._run_mode(
        "alpha_off", points,
        point_descriptors=[], h_ref=_dummy_href(), edges=_dummy_edges(),
    )

    assert ledger_path == tmp_path / "alpha_off" / "subset_index_log.json"
    ledger = json.loads(ledger_path.read_text())
    # Both metrics present for every subset size (2-part keys "metric/r").
    for metric in ("l2", "jsd"):
        for r in g.SUBSET_SIZES:
            key = f"{metric}/{r}"
            assert key in ledger, f"missing ledger entry {key}"
            assert ledger[key]["chosen_indices"] == list(range(r))
            assert ledger[key]["tag"] == f"bin{r:02d}"
    # subset.traj written per (metric, size, solver).
    for metric in ("l2", "jsd"):
        for r in g.SUBSET_SIZES:
            for solver in g.SOLVERS:
                p = (tmp_path / "alpha_off" / metric / f"bin{r:02d}" /
                     g.ARCH_NAME / g.LOSS_NAME / solver / "subset.traj")
                assert p.is_file(), f"missing {p}"
    # Reference histogram cache written.
    assert (tmp_path / "alpha_off" / "dfs_pool_full_hist" / "reference.npz").is_file()


def test_run_mode_resumes_and_skips_cached(tmp_path, monkeypatch):
    """A second _run_mode call with a complete ledger + trajs re-selects
    nothing (skip-if-cached), proving idempotent resume."""
    from xcquinox.alec.training_points import build_dfs_pool_points

    points = build_dfs_pool_points()
    calls = {"n": 0}

    def _fake_select_subset(*a, r, metric, **kw):
        calls["n"] += 1
        return tuple(range(r)), 0.5

    monkeypatch.setattr(g.ss, "select_subset", _fake_select_subset)
    monkeypatch.setattr(g, "CHECKPOINTS", tmp_path)

    g._run_mode("alpha_on", points, point_descriptors=[],
                h_ref=_dummy_href(), edges=_dummy_edges())
    first = calls["n"]
    assert first == len(g.METRICS) * len(g.SUBSET_SIZES)
    # Second run: everything cached -> zero new selections.
    g._run_mode("alpha_on", points, point_descriptors=[],
                h_ref=_dummy_href(), edges=_dummy_edges())
    assert calls["n"] == first
