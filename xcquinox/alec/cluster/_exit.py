"""xcquinox.alec.cluster._exit: the exit every harness entry point leaves through.

A harness stage is judged by the exit status its process hands SLURM, and that
status is NOT the stage's verdict when the interpreter dies during teardown.
JAX's atexit backend cleanup can corrupt the glibc heap after the work is
finished and its outputs are on disk -- ``corrupted size vs. prev_size while
consolidating``, SIGABRT, return code -6 (134 through a shell). The class was
first recorded on a pytest session (cluster job 2091615 batch 2: a green
summary followed by the abort, exit 134) and closed there by the same idiom in
``xcquinox/alec/tests/conftest.py``, which is the one remaining copy of it --
``hpcjobs/probe_pretrain_energy_weight.py`` carried a second one and now
leaves through this module like every other job stage.

The workflow-matrix smoke (cluster job 2134455, node dn024) measured it on a
harness WORKER: ``python -m xcquinox.alec.cluster._pretrain`` ran both
pretraining phases, computed and wrote ``fidelity_certificate.json``, wrote
``xnet.eqx`` and ``cnet.eqx``, logged ``pretrain SUCCEEDED``, and then aborted
at interpreter teardown. The stage was recorded as failed with rc -6 and every
stage after it was skipped. In a campaign that status marks the pretrain array
task FAILED, so the train array's ``afterok`` dependency never fires and the
whole run stalls behind a stage that had in fact completed.

Leaving through ``os._exit`` skips interpreter teardown entirely, so the status
the process hands SLURM is the value ``main`` returned. The streams are flushed
first because ``os._exit`` does not flush them.

Scope of the guarantee, and what it deliberately does NOT do:

* the return code is never swallowed -- a non-zero verdict stays non-zero, and
  an exception that escapes ``main`` keeps the interpreter's own status (1)
  with its traceback on stderr, so no consumer's exit-code contract moves;
* buffered output is never lost -- stdout, stderr and every file object opened
  from a path are flushed before the exit;
* a stage that aborts BEFORE finishing its work is unaffected: this changes
  only what happens after ``main`` has returned its verdict, so a genuine
  crash is still a crash.

``XCQ_NO_HARD_EXIT=1`` restores stock interpreter teardown (the same hatch name
the tests conftest uses) for the cases that need it -- subprocess coverage
measurement, or debugging a teardown fault rather than skipping it.
"""
import io
import os
import sys
import traceback
from typing import NoReturn

#: Environment variable that restores stock ``sys.exit`` teardown.
NO_HARD_EXIT_ENV = "XCQ_NO_HARD_EXIT"

#: Environment variable that skips the open-file sweep in :func:`flush_all`.
NO_FLUSH_SWEEP_ENV = "XCQ_NO_FLUSH_SWEEP"

#: Status for an exception that escaped ``main``. This is the interpreter's own
#: status for an uncaught exception, so wiring an entry point through here
#: leaves that contract exactly where it was; the traceback still reaches
#: stderr. A distinct code would silently redefine "the worker broke" for every
#: consumer that already classifies these return codes (``_train_task``'s
#: four-way outcome rule, ``workflow_matrix``'s stage records, the sbatch
#: epilogues).
EXIT_UNHANDLED = 1

#: Status for an interrupt, matching the shell's 128 + SIGINT.
EXIT_INTERRUPTED = 130


def _exit_status(value) -> int:
    """The process status for what ``main`` returned or ``SystemExit`` carried.

    Follows the interpreter's own rules: ``None`` is success, an integer is
    itself, and anything else is a message -- printed to stderr, status 1.
    """
    if value is None:
        return 0
    if isinstance(value, int):
        return int(value)
    print(value, file=sys.stderr)
    return 1


def _flush_open_files() -> None:
    """Flush every writable file object the process opened from a path.

    Worker logs are the reason: a stage that writes its log through a file
    object rather than through stdout would lose whatever is still in that
    object's buffer, since ``os._exit`` runs no finalizers. The sweep is over
    ``gc.get_objects`` (measured at 0.037 s for the 139k objects a loaded
    harness process carries, and linear in that count).

    Objects whose ``name`` is not a string are skipped: that is the pipe and
    socket case (``subprocess`` gives those file objects an integer fd as
    their name), where a flush can block on a reader that is already gone.
    Every step is guarded -- a file that cannot be flushed must not cost the
    process its exit status.
    """
    if os.environ.get(NO_FLUSH_SWEEP_ENV):
        return
    try:
        import gc
        objects = gc.get_objects()
    except Exception:  # noqa: BLE001 - nothing here may prevent the exit
        return
    for obj in objects:
        try:
            if not isinstance(obj, io.IOBase):
                continue
            if not isinstance(getattr(obj, "name", None), str):
                continue
            if obj.closed or not obj.writable():
                continue
            obj.flush()
        except Exception:  # noqa: BLE001 - per-object, never fatal
            continue


def flush_all() -> None:
    """Flush stdout, stderr, the logging handlers and every open log file.

    ``os._exit`` does not flush buffers, and a worker's stdout is a PIPE or a
    SLURM log file rather than a terminal, so it is block-buffered: without
    this the result line a parent parses (``parallel.run_workers`` reads the
    worker's stdout as JSON) and the tail of the SLURM log would be dropped by
    the very exit that preserves the status.
    """
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        try:
            if stream is not None:
                stream.flush()
        except Exception:  # noqa: BLE001 - nothing here may prevent the exit
            pass
    try:
        import logging
        logging.shutdown()
    except Exception:  # noqa: BLE001
        pass
    _flush_open_files()


def hard_exit(code) -> NoReturn:
    """Flush everything, then leave the process with ``code`` as its status.

    Never returns. ``os._exit`` is the LAST statement so that no interpreter
    teardown -- atexit handlers, C-extension finalizers, garbage collection of
    the JAX backend -- can run between the verdict and the status.
    """
    status = _exit_status(code)
    flush_all()
    if os.environ.get(NO_HARD_EXIT_ENV):
        sys.exit(status)
    os._exit(status)


def run_and_exit(main, argv=None) -> NoReturn:
    """Run ``main`` and leave through :func:`hard_exit` with its return code.

    EVERY path out of ``main`` leaves through that one exit, an unhandled
    exception included: the teardown is no safer because the stage broke, and a
    stage that dies with a traceback still owes whatever it wrote before the
    failure. The status classes are the interpreter's own, so nothing that
    reads these return codes has to change:

    * a returned value -- ``None`` is 0, an integer is itself;
    * ``SystemExit`` -- what it carries, which is how argparse's usage exit
      (2) and the harness's own refusals keep their codes;
    * ``KeyboardInterrupt`` -- 130, the shell's 128 + SIGINT;
    * anything else -- the traceback on stderr and status
      :data:`EXIT_UNHANDLED`.
    """
    try:
        code = main() if argv is None else main(argv)
    except SystemExit as exc:
        code = exc.code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        code = EXIT_INTERRUPTED
    except BaseException:  # noqa: BLE001 - nothing may skip the exit
        traceback.print_exc()
        code = EXIT_UNHANDLED
    hard_exit(code)
