"""Unit test for scripts/smoke_preflight_uks_oep.py.

Only exercises --dry-run, which imports the script's main() entry point
and the public preflight_uks_oep symbol but does NOT execute the slow
SCF/CCSD/OEP path. This keeps the unit test sub-second.

Full-execution verification is the user-driven `python scripts/smoke_preflight_uks_oep.py`
without --dry-run, NOT covered by pytest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "smoke_preflight_uks_oep.py"


def test_smoke_script_exists_and_is_executable_module():
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


def test_smoke_script_dry_run(tmp_path):
    """--dry-run imports preflight_uks_oep and exits 0 without running OEP."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--cache-dir", str(tmp_path), "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"--dry-run failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # The dry-run banner + resolved-symbol line must appear.
    assert "preflight_uks_oep on HO + HN" in proc.stdout
    assert "preflight_uks_oep" in proc.stdout
    assert "DRY-RUN" in proc.stdout
    # Cache directory must not have been populated.
    assert not (tmp_path / "HO.npz").exists()
    assert not (tmp_path / "HN.npz").exists()
