"""Slow faithfulness test, harness vs. the committed golden spec snapshot.

The cluster harness (``xcquinox.alec.cluster``) is a de-notebooked extraction
of the step-7 spec-building logic in ``notebooks/_build_step7_notebook.py``.
This test asserts the harness still reproduces the physical content of a
representative step-7 ``TrainingSpec``: molecules, targets, atom_energies,
loss_kwargs, solver_config, hyperparameters, so a silent regression is caught.

The golden reference lives at ``xcquinox/alec/tests/data/notebook_spec_snapshot.json``
and is produced by the USER-run helper ``scripts/capture_notebook_spec_snapshot.py``.
Until the user runs that script the fixture is absent, and this test SKIPS
with an actionable message. It FAILS only on a genuine content mismatch.
"""
import json
import os

import pytest


# ---------------------------------------------------------------------------
# Snapshot fixture location
# ---------------------------------------------------------------------------

_SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "notebook_spec_snapshot.json",
)

_CAPTURE_HINT = (
    "golden snapshot not found at "
    f"{_SNAPSHOT_PATH!r}. Generate it (USER-run) with:\n"
    "    python scripts/capture_notebook_spec_snapshot.py\n"
    "then `git add` and commit the JSON fixture. This test runs only once "
    "the fixture exists; it never fails on a missing fixture."
)


def _load_snapshot() -> dict:
    """Load the committed golden snapshot, or skip if it is absent."""
    if not os.path.isfile(_SNAPSHOT_PATH):
        pytest.skip(_CAPTURE_HINT)
    with open(_SNAPSHOT_PATH) as f:
        return json.load(f)


def test_golden_snapshot_atom_energies_match_domain():
    """Independent oracle (NOT the self-consistent rebuild): the committed golden
    snapshot's atom_energies must equal the live Chakravorty table (domain.py),
    their true source. Guards against a stale golden -- e.g. the prior S=-398.0
    placeholder vs the correct -398.1095, and a short 8-element table vs the
    current 14 -- which the rebuild-via-the-same-capture-path faithfulness test
    (which proves determinism, not correctness) cannot catch."""
    from xcquinox.alec.cluster.domain import ATOMIC_ENERGIES_CHAKRAVORTY
    ae = _load_snapshot()["spec"]["atom_energies"]
    assert set(ae) == set(ATOMIC_ENERGIES_CHAKRAVORTY), (
        "golden atom_energies element set drifted from domain.py")
    for el, e in ae.items():
        assert abs(e - ATOMIC_ENERGIES_CHAKRAVORTY[el]) < 1e-6, (el, e)


# ---------------------------------------------------------------------------
# Rebuild the spec the same way the capture script does
# ---------------------------------------------------------------------------

def _rebuild_spec_snapshot(grid_cell: dict) -> dict:
    """Rebuild the representative spec through the harness and serialize it the
    same way ``scripts/capture_notebook_spec_snapshot.py`` does.

    The capture script and this rebuilder MUST agree on the serialization
    schema; both go through ``_spec_snapshot`` in the capture module, which we
    import here so there is a single source of truth.
    """
    import importlib.util

    repo_root = os.path.dirname(  # .../xcquinox
        os.path.dirname(  # .../alec
            os.path.dirname(  # .../tests
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )
    script_path = os.path.join(
        repo_root, "scripts", "capture_notebook_spec_snapshot.py"
    )
    if not os.path.isfile(script_path):
        pytest.skip(
            f"capture script not found at {script_path!r}, cannot rebuild "
            "the spec for golden comparison"
        )
    spec_mod = importlib.util.spec_from_file_location(
        "_capture_notebook_spec_snapshot", script_path
    )
    cap = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(cap)

    cell, spec = cap.build_representative_spec(
        metric=grid_cell["metric"],
        subset_size=grid_cell["subset_size"],
        solver=grid_cell["solver"],
    )
    return cap._spec_snapshot(cell, spec)


# ---------------------------------------------------------------------------
# The faithfulness test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_harness_reproduces_golden_spec_snapshot():
    """The harness reproduces the committed golden step-7 spec snapshot.

    SKIPs (never fails) when the snapshot fixture is absent, it is absent
    until the user runs scripts/capture_notebook_spec_snapshot.py. It FAILS
    only on a genuine mismatch between the harness output and the committed
    reference.
    """
    snapshot = _load_snapshot()

    # Schema sanity, a snapshot from an incompatible capture-script version
    # is a setup error, not a faithfulness failure.
    assert "spec" in snapshot and "grid_cell" in snapshot["spec"], (
        "golden snapshot is missing the expected 'spec'/'grid_cell' keys, "
        "regenerate it with the current capture script"
    )

    golden_spec = snapshot["spec"]
    rebuilt = _rebuild_spec_snapshot(golden_spec["grid_cell"])

    # --- physical-content comparison, field by field for clear diffs -------
    assert rebuilt["grid_cell"] == golden_spec["grid_cell"], (
        "grid cell mismatch"
    )
    assert rebuilt["loss_name"] == golden_spec["loss_name"], (
        "loss_name mismatch"
    )
    assert rebuilt["solver_config"] == golden_spec["solver_config"], (
        "solver_config mismatch, the harness solver settings drifted from "
        "the notebook's SOLVER_CONFIGS"
    )
    assert rebuilt["hyperparameters"] == golden_spec["hyperparameters"], (
        "hyperparameters mismatch"
    )
    assert rebuilt["molecules"] == golden_spec["molecules"], (
        "molecule set mismatch, the chosen species union changed"
    )
    assert rebuilt["targets"] == golden_spec["targets"], (
        "targets mismatch, a target energy drifted from the notebook"
    )
    assert rebuilt["atom_energies"] == golden_spec["atom_energies"], (
        "atom_energies mismatch, an atomic-energy anchor drifted"
    )
    assert rebuilt["loss_kwargs"] == golden_spec["loss_kwargs"], (
        "loss_kwargs mismatch, a loss-channel input drifted from the "
        "notebook's _loss_kw"
    )

    # Whole-spec equality as a final catch-all.
    assert rebuilt == golden_spec, "spec snapshot mismatch"


@pytest.mark.slow
def test_golden_snapshot_records_notebook_sha():
    """If the snapshot exists, it must carry a usable notebook-builder SHA so
    a stale snapshot (captured before the notebook builder changed) is
    detectable. Skips when the fixture is absent."""
    snapshot = _load_snapshot()
    sha = snapshot.get("notebook_sha")
    assert sha, "golden snapshot is missing 'notebook_sha'"
    assert sha not in ("GIT_UNAVAILABLE", "UNTRACKED"), (
        f"golden snapshot notebook_sha is the sentinel {sha!r}, it was "
        "captured without a usable git SHA; regenerate it from a clean "
        "checkout so staleness can be detected"
    )
