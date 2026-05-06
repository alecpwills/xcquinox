"""Tests for scripts/oep_per_species_emit_overrides.py verifier."""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "oep_per_species_emit_overrides.py"


def _make_synthetic_summary(out_dir: Path) -> Path:
    """Write a minimal summary.json + Be.jsonl to out_dir."""
    summary = {
        "started_at_utc": "2026-05-04T00:00:00+00:00",
        "ended_at_utc":   "2026-05-04T01:00:00+00:00",
        "best_per_species": {
            "Be": {
                "trial_idx": 4,
                "settings": {
                    "aux_basis": "def2-tzvp-jkfit",
                    "regularization": 1e-3,
                    "grid_level": 1,
                    "conv_tol": 5e-3,
                    "target_floor": 5e-3,
                },
                "density_error_min": 1.2e-3,
                "wall_clock_s": 89.2,
            },
        },
        "n_trials_run": {"Be": 5},
        "short_circuited": ["Be"],
        "failed_target_floor": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary))
    # Minimal JSONL for Be (one converged trial)
    rec = {
        "trial_idx": 4,
        "species": {"name": "Be", "charge": 0, "spin": 0},
        "settings": summary["best_per_species"]["Be"]["settings"],
        "result": {
            "density_error_history": [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1.2e-3],
            "F_val_history": [],
            "density_error_min": 1.2e-3,
            "density_error_final": 1.2e-3,
            "n_iter": 6,
            "converged_stably": True,
            "converged_to_target_floor": True,
            "wall_clock_s": 89.2,
            "wall_capped": False,
            "termination": "early_stop_conv_tol",
            "plateau_density_error": None,
            "plateau_window_iters": 20,
            "inner_dm_r_squared": 4.13,
            "target_dm_r_squared": 4.10,
            "inner_dm_quad_aniso": 0.02,
            "target_dm_quad_aniso": 0.01,
            "inner_dm_dipole": None,
            "target_dm_dipole": None,
            "rss_mb_peak": None,
            "error_msg": None,
        },
    }
    (out_dir / "Be.jsonl").write_text(json.dumps(rec) + "\n")
    return out_dir / "summary.json"


def test_verifier_dry_run_succeeds_against_synthetic_summary(tmp_path):
    """--dry-run with a synthesized summary.json prints the snippet."""
    summary_path = _make_synthetic_summary(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--summary-path", str(summary_path),
         "--dry-run"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    # Snippet must include the species key and a citation tag
    assert '("Be", 0, 0)' in proc.stdout
    assert "[oep-tdl-" in proc.stdout
    assert "AUTHOR-RECALLED" in proc.stdout or "UNVERIFIED" in proc.stdout


def _stub_record(*, density_error_min, wall_clock_s,
                 termination="early_stop_conv_tol", level_shift=0.0,
                 history=None, target_r2=4.0, inner_r2=4.0,
                 target_q=0.0, inner_q=0.0,
                 target_dip=None, inner_dip=None,
                 converged_to_target_floor=True) -> dict:
    return {
        "species": {"name": "X", "charge": 0, "spin": 0},
        "settings": {"aux_basis": "def2-tzvp-jkfit", "level_shift": level_shift},
        "result": {
            "density_error_history": history or [3e-3] * 25,
            "density_error_min": density_error_min,
            "wall_clock_s": wall_clock_s,
            "n_iter": 25,
            "converged_to_target_floor": converged_to_target_floor,
            "termination": termination,
            "target_dm_r_squared": target_r2,
            "inner_dm_r_squared": inner_r2,
            "target_dm_quad_aniso": target_q,
            "inner_dm_quad_aniso": inner_q,
            "target_dm_dipole": target_dip,
            "inner_dm_dipole": inner_dip,
        },
    }


def test_select_winner_picks_lowest_density_error_min():
    """Among stably-converged candidates, lowest density_error_min wins."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_emit_overrides as ver
    records = [
        _stub_record(density_error_min=3e-3, wall_clock_s=100),
        _stub_record(density_error_min=1e-3, wall_clock_s=200),  # winner
        _stub_record(density_error_min=2e-3, wall_clock_s=50),
    ]
    winner = ver._select_winner(records, target_floor=5e-3)
    assert winner is not None
    assert winner["result"]["density_error_min"] == 1e-3


def test_select_winner_tie_break_by_wall_clock():
    """Equal density_error_min: cheaper wall wins."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_emit_overrides as ver
    records = [
        _stub_record(density_error_min=1e-3, wall_clock_s=200),
        _stub_record(density_error_min=1e-3, wall_clock_s=80),   # winner
    ]
    winner = ver._select_winner(records, target_floor=5e-3)
    assert winner["result"]["wall_clock_s"] == 80


def test_select_winner_returns_none_when_no_candidate_hits_floor():
    """All trials missed target_floor → no winner."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_emit_overrides as ver
    records = [
        _stub_record(density_error_min=2e-2, wall_clock_s=100,
                     converged_to_target_floor=False),
    ]
    winner = ver._select_winner(records, target_floor=5e-3)
    assert winner is None


def test_select_winner_short_history_with_conv_tol_marks_stable():
    """Carve-out: n_iter < plateau_window AND terminated_by==conv_tol
    → counted as stable (spec sec. 7.1 short-trial carve-out)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_emit_overrides as ver
    rec = _stub_record(density_error_min=1e-3, wall_clock_s=10,
                       termination="early_stop_conv_tol",
                       history=[1e-1, 5e-2, 1e-3])  # 3 iters < 20 window
    winner = ver._select_winner([rec], target_floor=5e-3)
    assert winner is not None


def test_dm_bias_check_skipped_at_level_shift_le_0_5():
    """At level_shift <= 0.5 the DM-bias check does not fire."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_emit_overrides as ver
    rec = _stub_record(density_error_min=1e-3, wall_clock_s=100,
                       level_shift=0.5,
                       target_q=0.0, inner_q=10.0)  # huge mismatch — ignored
    assert ver._passes_dm_bias_check(rec)


def test_dm_bias_check_excludes_quad_aniso_above_5pct():
    """At level_shift > 0.5: quad_aniso diff > 5%(target_r2) excludes."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_emit_overrides as ver
    rec = _stub_record(density_error_min=1e-3, wall_clock_s=100,
                       level_shift=1.0,
                       target_r2=4.0, target_q=0.0, inner_q=0.30)  # 7.5% of r^2
    assert not ver._passes_dm_bias_check(rec)


def test_round_2sigfig():
    """conv_tol = round_2sigfig(1.7 * density_error_min)."""
    import pytest
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import oep_per_species_emit_overrides as ver
    # Use approx for float-representation tolerance (e.g. 0.001*2.1 = 0.0021000000000000003)
    assert ver._round_2sigfig(1.7 * 1.234e-3) == pytest.approx(2.1e-3, rel=1e-9)
    assert ver._round_2sigfig(1.7 * 1.0e-3) == pytest.approx(1.7e-3, rel=1e-9)
