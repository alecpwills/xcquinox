"""Tests for scripts/oep_per_species_tune.py harness."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "oep_per_species_tune.py"
DEFAULT_GRID = REPO_ROOT / "scripts" / "oep_tune_grids.yaml"


def test_script_dry_run_succeeds_with_default_grid(tmp_path):
    """`python oep_per_species_tune.py --dry-run` exits 0 sub-second."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--grid", str(DEFAULT_GRID),
         "--cache-dir", str(tmp_path),
         "--out-dir", str(tmp_path / "out"),
         "--dry-run"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    )
    assert "trial-enumeration plan" in proc.stdout.lower() or \
           "dry run" in proc.stdout.lower()


def test_script_dry_run_validates_yaml_path(tmp_path):
    """Malformed --grid path: non-zero exit + error references the file."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--grid", "/nonexistent/path.yaml",
         "--cache-dir", str(tmp_path),
         "--out-dir", str(tmp_path / "out"),
         "--dry-run"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode != 0
    assert ("nonexistent" in proc.stderr.lower()
            or "no such file" in proc.stderr.lower())


def test_yaml_grid_loader_parses_default_yaml():
    """The shipped scripts/oep_tune_grids.yaml parses to the schema."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_tune as harness
    grid = harness._load_yaml_grid(DEFAULT_GRID)
    assert "Be" in grid and "C+" in grid and "F2" in grid
    assert "F2O" in grid and "HF" in grid and "HS" in grid
    assert "N2O" in grid and "O3" in grid


def test_yaml_grid_loader_rejects_missing_target_floor():
    """A species block missing target_floor raises SystemExit."""
    import pytest
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_tune as harness
    bad_block = {
        "charge": 0, "spin": 0,
        "sweep": {"aux_basis": ["def2-svp-jkfit"]},
    }
    allowlist = frozenset({"aux_basis"})
    with pytest.raises(SystemExit, match="missing required keys"):
        harness._validate_yaml_species_block("BadSpecies", bad_block, allowlist)


def test_yaml_grid_loader_rejects_unknown_knob():
    """A typo'd knob in the sweep raises SystemExit (catches `aux_bais`)."""
    import pytest
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_tune as harness
    bad_block = {
        "charge": 0, "spin": 0, "target_floor": 1e-3,
        "sweep": {"aux_bais": ["def2-svp-jkfit"]},  # typo
    }
    allowlist = frozenset({"aux_basis"})
    with pytest.raises(SystemExit, match="unknown knobs"):
        harness._validate_yaml_species_block("BadSpecies", bad_block, allowlist)


def test_trial_enumeration_full_grid_no_coupling_violations():
    """Cartesian product on a coupling-clean grid: n_trials == prod(len(v))."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_tune as harness
    block = {
        "charge": 0, "spin": 0, "target_floor": 1e-3,
        "sweep": {
            "aux_basis": ["def2-svp-jkfit"],   # svp-jkfit doesn't trigger coupling
            "regularization": [1e-4, 1e-3],
            "grid_level": [1, 2],
        },
    }
    trials = harness._enumerate_trials("Test", block)
    assert len(trials) == 1 * 2 * 2  # Cartesian product, no filtering


def test_trial_enumeration_drops_aux_reg_coupling_violations():
    """tzvp-jkfit + reg=1e-4 combos are silently filtered."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_tune as harness
    block = {
        "charge": 0, "spin": 0, "target_floor": 1e-3,
        "sweep": {
            "aux_basis": ["def2-svp-jkfit", "def2-tzvp-jkfit"],
            "regularization": [1e-4, 1e-3],
        },
    }
    trials = harness._enumerate_trials("Test", block)
    # 4 combos total; tzvp-jkfit + 1e-4 is dropped → 3 trials
    assert len(trials) == 3
    # Verify no surviving trial has tzvp-jkfit with reg < 1e-3
    for t in trials:
        if "tzvp-jkfit" in t.get("aux_basis", "") or "qzvp-jkfit" in t.get("aux_basis", ""):
            assert t["regularization"] >= 1e-3


def test_parse_species_bare_name_no_overrides():
    """Bare species name returns (name, None, None)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_tune as harness
    out = harness._parse_species_arg(["Be", "C+"])
    assert out == [("Be", None, None), ("C+", None, None)]


def test_parse_species_triple_with_overrides():
    """Triple species,charge,spin parses to ints."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_tune as harness
    out = harness._parse_species_arg(["C+,1,1", "Be,0,0"])
    assert out == [("C+", 1, 1), ("Be", 0, 0)]


def test_jsonl_writer_appends_with_fsync(tmp_path, monkeypatch):
    """JSONL writer uses open(path, 'a') + os.fsync per spec sec. 6.1."""
    import os
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_tune as harness
    fsync_calls = []
    real_fsync = os.fsync
    def spy_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", spy_fsync)
    path = tmp_path / "Be.jsonl"
    record = {"trial_idx": 0, "species": {"name": "Be"}, "result": {}}
    harness._append_jsonl(path, record)
    harness._append_jsonl(path, record)
    assert len(fsync_calls) == 2
    # File contains 2 lines:
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    import json
    for ln in lines:
        json.loads(ln)   # parses cleanly
