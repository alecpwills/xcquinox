"""Process-memory introspection from /proc (stdlib-only).

This module's own imports must stay stdlib-only so reading memory costs
nothing beyond the /proc read wherever it is used -- including callers that
run before JAX device routing has happened.

proc(5): ``VmRSS`` (current resident set) and ``VmHWM`` (peak resident set,
the high-water mark) are reported in kB; both are converted to GiB here.
"""
import math


def read_rss_gb(status_path="/proc/self/status"):
    """Return ``(vmrss_gib, vmhwm_gib)`` for the current process.

    Both values come from a single read of ``status_path``, so the pair is a
    consistent snapshot (``VmHWM >= VmRSS`` holds within one reading).
    Returns ``(nan, nan)`` where the file is unavailable (non-Linux) or a
    field is malformed -- callers embed the values in logs, so degradation
    must be silent rather than raising.
    """
    rss = hwm = math.nan
    try:
        with open(status_path, encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / (1024.0 * 1024.0)
                elif line.startswith("VmHWM:"):
                    hwm = float(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, IndexError, ValueError):
        return math.nan, math.nan
    return rss, hwm
