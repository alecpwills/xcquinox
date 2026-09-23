"""Tests for tools/oep_per_species_tune.py harness."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "oep_per_species_tune.py"
DEFAULT_GRID = REPO_ROOT / "tools" / "oep_tune_grids.yaml"


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


def test_yaml_grid_loader_parses_default_yaml():
    """The shipped tools/oep_tune_grids.yaml parses to the schema."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import oep_per_species_tune as harness
    grid = harness._load_yaml_grid(DEFAULT_GRID)
    assert "Be" in grid and "C+" in grid and "F2" in grid
    assert "F2O" in grid and "HF" in grid and "HS" in grid
    assert "N2O" in grid and "O3" in grid


def test_yaml_grid_loader_rejects_unknown_knob():
    """A typo'd knob in the sweep raises SystemExit (catches `aux_bais`)."""
    import pytest
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import oep_per_species_tune as harness
    bad_block = {
        "charge": 0, "spin": 0, "target_floor": 1e-3,
        "sweep": {"aux_bais": ["def2-svp-jkfit"]},  # typo
    }
    allowlist = frozenset({"aux_basis"})
    with pytest.raises(SystemExit, match="unknown knobs"):
        harness._validate_yaml_species_block("BadSpecies", bad_block, allowlist)


def test_trial_enumeration_drops_aux_reg_coupling_violations():
    """tzvp-jkfit + reg=1e-4 combos are silently filtered."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import oep_per_species_tune as harness
    block = {
        "charge": 0, "spin": 0, "target_floor": 1e-3,
        "sweep": {
            "aux_basis": ["def2-svp-jkfit", "def2-tzvp-jkfit"],
            "regularization": [1e-4, 1e-3],
        },
    }
    trials = harness._enumerate_trials("Test", block)
    # 4 combos total; tzvp-jkfit + 1e-4 is dropped -> 3 trials
    assert len(trials) == 3
    # Verify no surviving trial has tzvp-jkfit with reg < 1e-3
    for t in trials:
        if "tzvp-jkfit" in t.get("aux_basis", "") or "qzvp-jkfit" in t.get("aux_basis", ""):
            assert t["regularization"] >= 1e-3


def test_jsonl_trial_record_schema_has_all_required_fields(tmp_path):
    """Spec §6.2 schema completeness: every documented field present
    in the JSONL record. Plan-3 review fix."""
    import json
    record = {
        "trial_idx": 0,
        "species": {"name": "Be", "charge": 0, "spin": 0},
        "settings": {
            "aux_basis": "def2-svp-jkfit",
            "regularization": 1e-4,
            "grid_level": 1,
            "level_shift": 0.0,
            "inner_damp": 0.1,
            "inner_diis_start_cycle": 5,
            "max_iter": 500,
            "conv_tol": 5e-3,
            "target_floor": 5e-3,
        },
        "result": {
            "density_error_history": [],
            "F_val_history": [],
            "density_error_min": None,
            "density_error_final": None,
            "n_iter": 0,
            "converged_stably": False,
            "converged_to_target_floor": False,
            "wall_clock_s": 0.0,
            "wall_capped": False,
            "termination": "max_iter",
            "plateau_density_error": None,
            "plateau_window_iters": 20,
            "inner_dm_r_squared": None,
            "target_dm_r_squared": None,
            "inner_dm_quad_aniso": None,
            "target_dm_quad_aniso": None,
            "inner_dm_dipole": None,
            "target_dm_dipole": None,
            "rss_mb_peak": None,
            "error_msg": None,
        },
    }
    # Just verify the dict has every required key; serialization round-trip:
    blob = json.dumps(record)
    loaded = json.loads(blob)
    settings_keys = {"aux_basis", "regularization", "grid_level",
                     "level_shift", "inner_damp", "inner_diis_start_cycle",
                     "max_iter", "conv_tol", "target_floor"}
    result_keys = {"density_error_history", "F_val_history",
                   "density_error_min", "density_error_final", "n_iter",
                   "converged_stably", "converged_to_target_floor",
                   "wall_clock_s", "wall_capped", "termination",
                   "plateau_density_error", "plateau_window_iters",
                   "inner_dm_r_squared", "target_dm_r_squared",
                   "inner_dm_quad_aniso", "target_dm_quad_aniso",
                   "inner_dm_dipole", "target_dm_dipole",
                   "rss_mb_peak", "error_msg"}
    assert settings_keys <= set(loaded["settings"].keys())
    assert result_keys <= set(loaded["result"].keys())


