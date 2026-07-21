"""Tests for xcquinox.alec.procmem -- /proc-based process-memory readings.

The module must be importable without JAX (it is used by the CLI worker
before device routing), parse the kernel's kB fields into GiB exactly
(proc(5): VmRSS/VmHWM are reported in kB), and degrade to NaN rather than
raise where /proc is unavailable.
"""
import math
import sys

import pytest


def test_procmem_module_is_jax_free():
    """procmem's own imports must stay stdlib-only. Loaded by file path in a
    clean interpreter (importing via the package would pull the jax-heavy
    package __init__ and mask what THIS module imports)."""
    import os
    import subprocess

    mod_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "procmem.py"))
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('procmem', {mod_path!r}); "
        "m = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(m); "
        "sys.exit(1 if 'jax' in sys.modules else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()


def test_read_rss_gb_parses_kb_fields_to_gib(tmp_path):
    """proc(5) reports VmRSS/VmHWM in kB; 2097152 kB = 2 GiB exactly."""
    from xcquinox.alec.procmem import read_rss_gb

    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython\n"
        "VmPeak:\t 9999999 kB\n"
        "VmHWM:\t 3145728 kB\n"
        "VmRSS:\t 2097152 kB\n"
    )
    rss, hwm = read_rss_gb(str(status))
    assert rss == pytest.approx(2.0)
    assert hwm == pytest.approx(3.0)


def test_read_rss_gb_live_read_is_finite_and_ordered():
    """On Linux the default path yields finite positive values with
    high-water mark >= current RSS (single consistent snapshot)."""
    from xcquinox.alec.procmem import read_rss_gb

    if not sys.platform.startswith("linux"):
        pytest.skip("live /proc read is Linux-only")
    rss, hwm = read_rss_gb()
    assert math.isfinite(rss) and rss > 0.0
    assert math.isfinite(hwm) and hwm >= rss


def test_read_rss_gb_missing_file_returns_nan(tmp_path):
    from xcquinox.alec.procmem import read_rss_gb

    rss, hwm = read_rss_gb(str(tmp_path / "no_such_status"))
    assert math.isnan(rss)
    assert math.isnan(hwm)
