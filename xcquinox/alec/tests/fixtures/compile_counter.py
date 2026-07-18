"""Top-level XLA compilation counter.

A training loop over fixed molecules has fixed array shapes, so after warmup it
must reuse compiled kernels and compile nothing further. A nonzero steady-state
compile rate is a defect: every compilation retains its LLVM code mapping for the
life of the process (``jax.clear_caches()`` does not release backing stores), so a
per-step compile ratchets the process mapping count toward ``vm.max_map_count``
and terminates the job with SIGSEGV once the ceiling is crossed.

Compilations are counted from the ``jax._src.interpreters.pxla`` log under
``jax.log_compiles(True)``. Records beginning ``"Compiling module"`` come from the
separate FDO logger and are excluded. Absolute counts are not portable across JAX
versions -- they include warmup and per-constant helper kernels -- so tests assert
the *property* (steady state adds none) rather than an exact number.
"""
import logging

import jax


class CompileCounter:
    """Context manager counting top-level XLA compilations in ``self.count``."""

    LOGGER = "jax._src.interpreters.pxla"

    def __init__(self):
        self.count = 0
        self._handler = None
        self._logger = None
        self._prev_level = None
        self._prev_propagate = None
        self._dispatch = None
        self._dispatch_prev = None
        self._log_compiles = None

    def __enter__(self):
        counter = self

        class _Handler(logging.Handler):
            def emit(self, record):
                try:
                    msg = record.getMessage()
                except Exception:  # noqa: BLE001 - a broken record must not fail a test
                    return
                if msg.startswith("Compiling ") and not msg.startswith("Compiling module"):
                    counter.count += 1

        self._handler = _Handler()
        self._handler.setLevel(logging.DEBUG)
        self._logger = logging.getLogger(self.LOGGER)
        self._prev_level = self._logger.level
        self._prev_propagate = self._logger.propagate
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)
        # Suppress console output; the attached handler still receives records.
        self._logger.propagate = False
        self._dispatch = logging.getLogger("jax._src.dispatch")
        self._dispatch_prev = self._dispatch.level
        self._dispatch.setLevel(logging.ERROR)
        self._log_compiles = jax.log_compiles(True)
        self._log_compiles.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            if self._log_compiles is not None:
                self._log_compiles.__exit__(*exc)
        finally:
            if self._logger is not None:
                self._logger.removeHandler(self._handler)
                self._logger.setLevel(self._prev_level)
                self._logger.propagate = self._prev_propagate
            if self._dispatch is not None:
                self._dispatch.setLevel(self._dispatch_prev)
        return False
