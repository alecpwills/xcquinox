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
