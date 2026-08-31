"""Worker CPU bind, importable WITHOUT the xcquinox package machinery.

The pin must run before ANY import that stands up the JAX CPU backend, and
``import xcquinox.alec.<anything>`` is such an import: the package __init__
chain reaches ``xcquinox/utils.py``, whose ``device=jax.devices('cpu')[0]``
DEFAULT ARGUMENT initializes the backend at import time -- measured on a real
worker, 41-43 pool threads already exist by then, ``sched_setaffinity`` from
that point binds the calling thread only, and a one-CPU worker still runs at
694-990 percent load with 52 OS threads. Applied before the first JAX import
the same pin gives 100 / 195 / 334 percent at budgets 1 / 2 / 4 with every
thread on the slice.

The worker scripts are launched BY PATH (``python <...>/train_worker.py``),
which puts this directory at ``sys.path[0]``, so ``import _cpu_bind`` loads
this file as a top-level module and executes nothing of the package. This
module must therefore stay stdlib-only. The parent-side API
(``parallel.apply_worker_cpu_bind``) delegates here by file path for the same
reason, and ``parallel`` pins its env-variable names equal to these by test.
"""
import os

WORKER_BIND_CPUS_ENV = "XCQUINOX_WORKER_BIND_CPUS"
WORKER_SLOT_ENV = "XCQUINOX_WORKER_SLOT"


def apply() -> int | None:
    """Pin this pool worker's CPU affinity; see parallel.apply_worker_cpu_bind.

    Reads the launcher's bind request (:data:`WORKER_BIND_CPUS_ENV`, the
    thread budget) and pool slot (:data:`WORKER_SLOT_ENV`) and pins the
    process to a slot-strided slice of the currently allowed CPUs (the
    current allowance, so a SLURM cgroup's restriction is respected). TSL
    sizes the XLA CPU intra-op pool from NumSchedulableCPUs, i.e.
    sched_getaffinity, so threads created after this point inherit the
    slice; nothing bounds threads created before it, which is why the call
    must precede the first JAX import. Slices of concurrently running pool
    members are mutually disjoint whenever slots x budget fits the
    allowance (the eval ladder guarantees workers x threads <= total CPUs);
    a slot pushed past the allowance wraps by modulo and may overlap -- a
    documented degradation, not an error, and unreachable from the shipped
    ladders.

    Returns the number of CPUs pinned, or None when unbound: no bind
    request, no slot, a budget at or above the allowance (nothing to
    bound), or a platform without sched_setaffinity.
    """
    budget = os.environ.get(WORKER_BIND_CPUS_ENV)
    slot = os.environ.get(WORKER_SLOT_ENV)
    if not budget or slot is None or not hasattr(os, "sched_setaffinity"):
        return None
    k = int(budget)
    s = int(slot)
    if k <= 0 or s < 0:
        return None
    allowed = sorted(os.sched_getaffinity(0))
    n = len(allowed)
    if k >= n:
        return None
    start = (s * k) % n
    cpus = {allowed[(start + j) % n] for j in range(k)}
    os.sched_setaffinity(0, cpus)
    return len(cpus)
