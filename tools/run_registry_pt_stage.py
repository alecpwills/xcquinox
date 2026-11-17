"""Execute one stage of a registry proton-transfer prong notebook headless.

    python tools/run_registry_pt_stage.py --prong A --stage prep  --work-root <dir>
    python tools/run_registry_pt_stage.py --prong B --stage train --work-root <dir>
    python tools/run_registry_pt_stage.py --prong A --stage assemble --work-root <dir>
    python tools/run_registry_pt_stage.py --prong A --stage all --work-root <dir> --smoke

The notebook is regenerated from its builder into
``<work-root>/prong_<P>/executed/<stage>_<job>_<time>.ipynb`` and executed
there by nbclient in the running interpreter's kernel; the executed copy, with
every cell's output, is written back when the run ends for any reason, a
wall-clock SIGTERM included. jupyter_client starts the kernel in a session
of its own, so a SIGTERM to this process is forwarded to the kernel's
process group: the train stage drains (the children flush their resume
sets, the interrupted architectures return to pending) and the remaining
cells run. ``assemble`` builds the result tables from the readouts present
without executing the notebook.

Exit status: 0 when every cell ran (architectures that failed are recorded in
the prong's ledger, not raised); 1 when a cell raised (a stage-level failure);
143 on SIGTERM.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: whether the job's SIGTERM has reached this process
_TERM = {"seen": False}


def _flag_term(signum, frame):  # noqa: ARG001 -- signal-handler signature
    _TERM["seen"] = True


#: the exceptions a SIGTERM to the kernel produces in nbclient: the running
#: cell raising (KeyboardInterrupt, a dead worker) or the kernel dying
_TERM_SHAPED = ("CellExecutionError", "DeadKernelError")


def exit_code(exc, term_seen: bool) -> int:
    """The driver's exit status for an exception out of the notebook run:
    143 when a SIGTERM had arrived and the exception is one a SIGTERM
    produces; 1 otherwise, so a code error is never reported as an
    interruption."""
    if term_seen and type(exc).__name__ in _TERM_SHAPED:
        return 143
    return 1


def _kernel_pid(client):
    """The pid of the notebook's kernel process, or None before it starts."""
    provisioner = getattr(getattr(client, "km", None), "provisioner", None)
    for holder in (provisioner, getattr(provisioner, "process", None)):
        pid = getattr(holder, "pid", None) if holder is not None else None
        if isinstance(pid, int) and pid > 0:
            return pid
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prong", choices=("A", "B"), required=True)
    ap.add_argument("--stage", choices=("prep", "train", "all", "assemble"),
                    required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="the workstation identity (def2-svp, grid 1, 2 epochs, 4 archs)")
    args = ap.parse_args(argv)

    work_root = Path(os.path.expanduser(args.work_root)).resolve()
    prong_root = work_root / f"prong_{args.prong}"
    os.environ["REGISTRY_PT_WORK_ROOT"] = str(work_root)
    os.environ["REGISTRY_PT_STAGE"] = args.stage
    if args.smoke:
        os.environ["STEP_SMOKE"] = "1"
    else:
        os.environ.pop("STEP_SMOKE", None)

    if args.stage == "assemble":
        sys.path.insert(0, str(NOTEBOOKS))
        import registry_pt_prongs as rpp
        ledger_path = prong_root / "ledger.json"
        ledger = rpp.Ledger(ledger_path) if ledger_path.is_file() else None
        res = rpp.assemble(prong_root, clone_root=work_root / "shared" / "clone_readout",
                           ledger=ledger)
        print(f"assembled {res['n_readouts']} readouts, {res['n_clone_readouts']} "
              f"clone readouts into {prong_root / 'results'}", flush=True)
        return 0

    import nbclient
    import nbformat

    job = os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}"
    out_dir = prong_root / "executed"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / f"{args.stage}_{job}_{time.strftime('%Y%m%dT%H%M%S')}.ipynb"
    builder = _load(NOTEBOOKS / f"_build_registry_pt_prong_{args.prong}_notebook.py",
                    f"_build_registry_pt_prong_{args.prong}_notebook")
    builder.main(output_path=str(nb_path))
    nb = nbformat.read(str(nb_path), as_version=4)

    t0 = time.time()

    def _cell_done(cell_index=0, cell=None, **_kw):
        print(f"[{time.time() - t0:8.0f} s] cell {cell_index} done: "
              f"{(cell or {}).get('id', '?')}", flush=True)

    client = nbclient.NotebookClient(
        nb, timeout=None, kernel_name="python3",
        resources={"metadata": {"path": str(work_root)}},
        on_cell_executed=_cell_done)
    command = client.create_kernel_manager().format_kernel_cmd()
    if command[0] != sys.executable:
        print(f"FATAL: the kernel launches {command[0]}, not {sys.executable}",
              flush=True)
        return 1

    def _on_term(signum, frame):  # noqa: ARG001 -- signal-handler signature
        # jupyter_client starts the kernel in a session of its own, so the
        # job's TERM to this process group never reaches it: forward it to
        # the kernel's group (the kernel and the workers' children), and let
        # the notebook drain -- the train stage records every interrupted
        # architecture before its cell returns
        _TERM["seen"] = True
        pid = _kernel_pid(client)
        if pid is None:
            raise SystemExit(143)     # no kernel yet: nothing to drain
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            raise SystemExit(143) from None
        print(f"SIGTERM: forwarded to the kernel's process group ({pid}); "
              "draining", flush=True)

    signal.signal(signal.SIGTERM, _on_term)
    rc = 0
    try:
        client.execute()
    except (nbclient.exceptions.CellExecutionError,
            nbclient.exceptions.DeadKernelError) as exc:
        signal.signal(signal.SIGTERM, _flag_term)
        rc = exit_code(exc, _TERM["seen"])
        if rc == 143:
            print(f"interrupted (SIGTERM) after {time.time() - t0:.0f} s", flush=True)
        else:
            print(f"STAGE FAILED: a cell raised\n{exc}", flush=True)
    except SystemExit as exc:
        signal.signal(signal.SIGTERM, _flag_term)
        print(f"interrupted (exit {exc.code}) after {time.time() - t0:.0f} s", flush=True)
        rc = int(exc.code or 143)
    except Exception as exc:  # noqa: BLE001 -- the kernel's teardown can raise too
        signal.signal(signal.SIGTERM, _flag_term)
        traceback.print_exc()
        rc = exit_code(exc, _TERM["seen"])
        print(f"driver error after {time.time() - t0:.0f} s"
              f"{' (a SIGTERM had arrived)' if _TERM['seen'] else ''}; exit {rc}",
              flush=True)
    else:
        if _TERM["seen"]:
            rc = 143
            print(f"interrupted (SIGTERM) after {time.time() - t0:.0f} s; every "
                  "cell completed", flush=True)
    finally:
        signal.signal(signal.SIGTERM, _flag_term)
        nbformat.write(nb, str(nb_path))
        print(f"executed notebook: {nb_path}", flush=True)
    print(f"stage {args.stage} of prong {args.prong}: exit {rc} after "
          f"{time.time() - t0:.0f} s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
