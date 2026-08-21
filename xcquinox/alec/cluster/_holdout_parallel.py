"""Adaptive-degradation orchestrator for the held-out eval.

Parallelizes a spec's held-out eval (full BH76 + W4-11, ~200+ molecules) across
molecule-shard worker subprocesses (``workers/eval_holdout_worker.py`` via
``parallel.run_workers``), descending a parallelism ladder and **retrying only
the molecules that did not finish** at lower parallelism (more memory per
worker), then finishing any stragglers in-process serial. Not finishing covers
both a shard that wrote no JSON and a species the shard wrote with a null/NaN
energy, i.e. one whose own evaluation raised. The per-molecule physics is the
SAME ``eval_holdout.compute_holdout_per_molecule`` the serial driver uses, so the
result is identical to serial; only the work distribution differs.

The serial final tier guarantees completion, so this never fails to produce the
artifacts. A catastrophic orchestrator error is the caller's cue to fall back to
the plain serial ``run_full_holdout_eval``.
"""
import json
import os
import sys
from pathlib import Path

from xcquinox.alec import parallel


#: Character cap on a forwarded worker line. A shard-level crash puts its whole
#: traceback on one JSON line, and 500 is the excerpt width this subsystem
#: already uses for worker output (parallel.run_workers' ``raw`` and
#: ``stderr_tail`` payload fields, backfill_holdout_nans' worker-failure text).
FORWARD_LINE_CHARS = 500


def _log(msg: str) -> None:
    print(f"[holdout-parallel] {msg}", flush=True)


def _forward_failed_lines(tier_no, si, res) -> None:
    """Echo a worker's per-species failure lines into the task log.

    The full captured streams are on disk (``<shard_dir>/worker_t*_s*.log``);
    only the ``FAILED:`` lines are repeated here so a 40-shard tier stays
    readable while a silently-failing species still shows up in the task log.
    Both streams are scanned -- the worker's own failure line has moved to
    stderr, but stdout is where older workers put it. Lines are truncated to
    ``FORWARD_LINE_CHARS``."""
    for stream in (getattr(res, "stdout", "") or "",
                   getattr(res, "stderr", "") or ""):
        for line in stream.splitlines():
            if "FAILED:" in line:
                print(f"[holdout-parallel] worker t{tier_no}/s{si}: "
                      f"{line.strip()[:FORWARD_LINE_CHARS]}",
                      file=sys.stderr, flush=True)


def _round_robin(names, k):
    """Partition ``names`` into ``k`` balanced shards (round-robin so a few
    heavy molecules don't all land in one shard)."""
    return [names[i::k] for i in range(k)]


def run_holdout_with_escalation(
    run_dir, spec_idx, training_spec, model, reactions, full_specs, out_dir, *,
    basis, grid_level, n_workers_top, total_cpus, strict=None,
    model_name="model.eqx", coldstart=False,
):
    """Run the held-out eval in parallel with adaptive degradation.

    Parameters mirror the serial ``run_full_holdout_eval`` plus the shard-launch
    context (``run_dir``/``spec_idx`` so workers reload the same spec+model) and
    the parallelism budget (``n_workers_top`` = ladder start, ``total_cpus`` =
    threads-per-worker base). ``training_spec``/``model`` are used only for the
    in-process serial leftover tier (workers reload their own). ``model_name``
    selects which checkpoint the shard workers reload (``model.eqx`` final /
    ``model_best.eqx`` best); the caller MUST pass the same checkpoint it loaded
    into ``model`` so the serial leftover tier and the workers agree. Returns the
    same summary dict as ``run_full_holdout_eval``.
    """
    from xcquinox.alec import eval_holdout

    if strict is None:
        strict = os.environ.get("XCQUINOX_HELDOUT_STRICT") == "1"

    out_dir = Path(out_dir)
    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    script = parallel.worker_script_path("eval_holdout_worker")

    remaining = list(full_specs.keys())
    shard_payloads = []

    ladder = parallel.eval_worker_ladder(total_cpus, top=n_workers_top)
    for tier_no, (n_workers, threads) in enumerate(ladder, start=1):
        if not remaining:
            break
        k = min(n_workers, len(remaining))
        shards = _round_robin(remaining, k)
        jobs = []
        out_shards = []
        for si, shard_names in enumerate(shards):
            names_file = shard_dir / f"names_t{tier_no}_s{si}.json"
            out_shard = shard_dir / f"shard_t{tier_no}_s{si}.json"
            names_file.write_text(json.dumps(shard_names))
            out_shards.append(out_shard)
            cmd = [
                sys.executable, script,
                "--run-dir", str(run_dir), "--spec-idx", str(spec_idx),
                "--names-file", str(names_file), "--out-shard", str(out_shard),
                "--basis", str(basis), "--grid-level", str(grid_level),
                "--threads", str(threads), "--model-name", str(model_name),
            ] + (["--coldstart"] if coldstart else [])
            jobs.append(parallel.WorkerJob(
                name=f"eval_t{tier_no}_s{si}", cmd=cmd,
                # The shard worker writes no progress file, so the watchdog
                # would report a "stall" every STALL_WARN_SEC for the whole run.
                progress_file=None,
                log_file=str(shard_dir / f"worker_t{tier_no}_s{si}.log"),
                thread_env=parallel._thread_env(threads)))

        _log(f"tier {tier_no}: {k} workers x {threads} thread(s) over "
             f"{len(remaining)} molecules")
        results = parallel.run_workers(jobs, max_parallel=n_workers)
        for si, res in enumerate(results):
            _forward_failed_lines(tier_no, si, res)

        done = set()
        for res, out_shard in zip(results, out_shards):
            if getattr(res, "status", "failed") != "success" or not out_shard.is_file():
                continue
            try:
                payload = json.loads(out_shard.read_text())
            except (OSError, ValueError):
                continue
            shard_payloads.append(payload)
            # A species counts as DONE only when its shard energy is finite.
            # A completed shard can still carry a null/NaN energy for a species
            # whose evaluation raised (transient host-allocation / compile
            # failures under tier-1 memory pressure), and treating the mere
            # PRESENCE of the name as completion retired those species without
            # ever retrying them at lower parallelism.
            for name, e in (payload.get("energies") or {}).items():
                if eval_holdout.is_finite_energy(e):
                    done.add(name)
        remaining = [n for n in remaining if n not in done]
        if remaining:
            _log(f"tier {tier_no} left {len(remaining)} molecules unfinished; "
                 f"degrading to lower parallelism")

    if remaining:
        _log(f"serial fallback for {len(remaining)} remaining molecule(s)")
        subset = {n: full_specs[n] for n in remaining}
        per = eval_holdout.compute_holdout_per_molecule(
            training_spec, model, subset)
        shard_payloads.append({
            "energies": per["energies"],
            "pbe_energies": per["pbe_energies"],
            "mol_records": per["mol_records"],
        })

    energies, pbe_energies, mol_records = eval_holdout.merge_holdout_shards(
        shard_payloads)
    # Every tier (workers + the serial sweep) has now had a go; a species still
    # non-finite is accepted with its last payload, but is NAMED so the operator
    # sees it instead of a silent NaN column in per_molecule.json.
    failed_everywhere = sorted(
        n for n in full_specs
        if not eval_holdout.is_finite_energy(energies.get(n)))
    if failed_everywhere:
        _log(f"{len(failed_everywhere)} species failed in every tier: "
             f"{', '.join(failed_everywhere)}")
    # MOLECULE-level names (single atoms excluded): held-out overlap is
    # molecule-level, else shared reference atoms (h, c, n, o, ...) drop nearly
    # the entire atomization held-out set. See eval_holdout.training_molecule_names.
    training_names = eval_holdout.training_molecule_names(training_spec)
    excl, key_map = eval_holdout.trained_reaction_exclusion(
        training_spec, full_specs)
    return eval_holdout._finalize_holdout_outputs(
        reactions, energies, pbe_energies, mol_records, training_names,
        n_species=len(full_specs), out_dir=out_dir, strict=strict,
        excluded_identities=excl, species_key_map=key_map)
