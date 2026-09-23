"""The cold-start convergence census of a run's training species.

The census runs the training solver of a spec from the functional-free atomic
guess (the ``minao`` seed) on every training species of the spec, forward
only, with the model the spec starts training from (the certified pretrained
checkpoint through ``train._build_model``, or a model handed in), and records
per species whether the SCF converged, the cycles it ran, the energy it
reached, its last energy step and the whole energy trace. The solver is the
spec's own -- cycles, tolerance, mixer, tail and Coulomb footing -- with the
seed moved to the atomic guess (:func:`census_solver_config`), so the census
asks the question the cold-start arm poses: does the training SCF converge
from where that arm starts it. It is the pre-training pass of the published
dpyscf script (every training molecule from ``dm_realinit``, the atomic guess,
with the pretrained model) written as a record.

The census is a report. Nothing gates on it: the preflight runs it when
``cluster.preflight_coldstart_census`` is set, writes
``<run_dir>/coldstart_census.json`` and logs one summary line per architecture
(``_preflight._coldstart_census_impl``). A cell whose solver cannot start from
the atomic guess (a one-shot or fixed-J solver, the pyscfad backend) is
recorded with its refusal and no rows, and the other cells are still measured.
The document is rewritten after every cell, so a census cut short at a wall
leaves the cells completed.

Command line::

    python -m xcquinox.pipeline.cluster.coldstart_census <spec>... --out <json>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np

#: The keys every row carries.
ROW_KEYS = ("name", "converged", "cycles_run", "total_energy", "finite",
            "last_delta_e", "energy_trace", "wall_s")


def census_solver_config(sc):
    """The spec's solver started from the atomic guess: ``seed_source`` set to
    ``minao`` and the seed cache dropped, every other field kept. Raises
    ``ValueError`` for a solver that cannot take a non-pbe seed (the
    ``SolverConfig`` rule: FULL mode only)."""
    return dataclasses.replace(sc, seed_source="minao", seed_cache_dir=None)


def _solver_config_of(spec):
    """The solver config as the training loop reads it: ``loss_kwargs`` first,
    the field otherwise."""
    return spec.loss_kwargs_dict.get("solver_config") or spec.solver_config


def _spec_with_solver(spec, sc):
    """``spec`` carrying ``sc`` where the loop reads the solver config."""
    kwargs = spec.loss_kwargs_dict
    if "solver_config" in kwargs:
        kwargs["solver_config"] = sc
        return dataclasses.replace(
            spec, loss_kwargs=tuple(sorted(kwargs.items())))
    return dataclasses.replace(spec, solver_config=sc)


def _finite(x):
    """The float, or None where it is not finite: the document is strict
    JSON, which has no NaN or infinity."""
    x = float(x)
    return x if np.isfinite(x) else None


def _row(name, result, wall_s):
    """One species' record from the solver's result, as JSON scalars. A
    non-finite energy (a diverged SCF, the case the census exists to record)
    is written as null and the row's ``finite`` flag is false."""
    trace_raw = getattr(result, "energy_trace", None)
    trace = ([_finite(x) for x in np.asarray(trace_raw).ravel()]
             if trace_raw is not None else [])
    energy = _finite(result.total_energy)
    # The magnitude of the last energy step; None below two cycles, where the
    # trace holds no step, and None when either of the last two energies is
    # not finite. The trace is the solver's own (one entry per configured
    # cycle, the frozen tail included), which is what the tail loss reads;
    # cycles_run counts the cycles before the convergence freeze.
    last = (abs(trace[-1] - trace[-2])
            if len(trace) >= 2 and trace[-1] is not None
            and trace[-2] is not None else None)
    return {
        "name": str(name),
        "converged": bool(result.converged),
        "cycles_run": int(result.cycles_run),
        "total_energy": energy,
        "finite": energy is not None and all(x is not None for x in trace),
        "last_delta_e": last,
        "energy_trace": trace,
        "wall_s": float(wall_s),
    }


def census_rows(spec, model=None):
    """One record per training species of ``spec``, in the spec's order.

    The batch is the spec's training batch under the census solver (the
    supply layer then hands the solver the atomic guess as ``dm_seed``), the
    model is ``model`` or the one the spec starts training from, and every
    species runs the census solver forward only. Raises ``ValueError`` when
    the spec's solver cannot start from the atomic guess and
    ``NotImplementedError`` on a backend that refuses the seed.
    """
    from xcquinox.pipeline.losses import make_loss
    from xcquinox.pipeline.solver import run_scf
    from xcquinox.pipeline.train import _build_batch, _build_model

    sc = _solver_config_of(spec)
    if sc is None:
        raise ValueError(
            "the spec carries no solver config; the census needs the training "
            "solver to start it from the atomic guess")
    cold = census_solver_config(sc)
    cold_spec = _spec_with_solver(spec, cold)
    if model is None:
        model = _build_model(spec)
    loss = make_loss(
        cold_spec.loss_name,
        molecules=cold_spec.molecules,
        pbe_anchor_weight=cold_spec.pbe_anchor_weight,
        pbe_anchor_sample=cold_spec.pbe_anchor_sample,
        **cold_spec.loss_kwargs_dict,
    )
    batch = _build_batch(cold_spec, loss)
    rows = []
    for md in batch["mol_data"]:
        t0 = time.time()
        result = run_scf(cold, model, md, forward_only=True)
        rows.append(_row(md["name"], result, time.time() - t0))
    return rows


def _cell(spec_path, spec, rows=None, error=None):
    rows = list(rows or [])
    return {
        "arch": spec.arch.name,
        "spec_path": spec_path,
        "n_species": len(spec.molecules),
        "n_converged": sum(1 for r in rows if r["converged"]),
        "error": error,
        "rows": rows,
    }


def summary_line(cell) -> str:
    """One log line for a cell: the converged count and the names that did
    not converge, or the refusal that left the cell unmeasured."""
    if cell.get("error"):
        label = cell.get("arch") or os.path.basename(cell["spec_path"])
        return f"census {label}: not measured ({cell['error']})"
    names = [r["name"] for r in cell["rows"] if not r["converged"]]
    tail = f"; not converged: {', '.join(names)}" if names else ""
    return (f"census {cell['arch']}: {cell['n_converged']}/{cell['n_species']} "
            f"species converged from the atomic guess{tail}")


def _load_spec(path):
    from xcquinox.pipeline.checkpoint_class import load_pickle
    with open(path, "rb") as fh:
        return load_pickle(fh)


def _write_json_atomic(payload, path):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        # strict JSON: a non-finite value is refused here rather than written
        # as a token no other reader accepts (the rows write None for them)
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def _failed_cell(spec_path, spec, exc):
    """The cell of a spec that could not be measured (or read: ``spec`` is
    then None), carrying the error and no rows."""
    text = f"{type(exc).__name__}: {exc}"
    if spec is None:
        return {"arch": None, "spec_path": spec_path, "n_species": 0,
                "n_converged": 0, "error": text, "rows": []}
    return _cell(spec_path, spec, error=text)


def write_census(spec_paths, out_path, *, log=None) -> dict:
    """The census document over ``spec_paths`` (one cell each, in the order
    given), written to ``out_path`` after every cell; ``log`` receives the
    summary line of each cell. A cell that cannot be measured (a solver that
    cannot start from the atomic guess, a backend that refuses the seed, a
    spec file that cannot be read) is recorded with its error and the next
    cell is measured: the census is a report, and a report records its
    failures. Returns the document."""
    log = log or (lambda msg: None)
    cells = []
    for path in spec_paths:
        t0 = time.time()
        spec = None
        try:
            spec = _load_spec(path)
            cell = _cell(path, spec, census_rows(spec))
        except Exception as exc:  # noqa: BLE001 -- recorded, never dropped
            cell = _failed_cell(path, spec, exc)
        cell["wall_s"] = time.time() - t0
        cells.append(cell)
        # the document first, the line second: a line in the log names only
        # a cell the document holds
        _write_json_atomic({"cells": cells}, out_path)
        log(summary_line(cell))
    return {"cells": cells}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the training solver of each spec from the atomic "
                    "guess on every training species and write the record.")
    parser.add_argument("spec_paths", nargs="+", metavar="SPEC",
                        help="materialized TrainingSpec files, one cell each")
    parser.add_argument("--out", required=True, metavar="JSON",
                        help="the census document to write")
    args = parser.parse_args(argv)
    write_census(args.spec_paths, args.out,
                 log=lambda msg: print(msg, flush=True))
    return 0


if __name__ == "__main__":
    # float64 before any JAX import, as every worker entry point sets it
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    # The census is the status this process hands its caller, and JAX's
    # atexit teardown can abort the interpreter after main() has returned
    # it; run_and_exit flushes and leaves through os._exit, as every cluster
    # entry point does. See cluster/_exit.py.
    from xcquinox.pipeline.cluster._exit import run_and_exit
    run_and_exit(main)
