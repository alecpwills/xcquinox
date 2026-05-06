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
