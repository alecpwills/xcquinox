"""Adaptive-degradation orchestrator for the held-out eval.

Parallelizes a spec's held-out eval (full BH76 + W4-11, ~200+ molecules) across
molecule-shard worker subprocesses (``workers/eval_holdout_worker.py`` via
``parallel.run_workers``), descending a parallelism ladder and **retrying only
the molecules whose shard failed** at lower parallelism (more memory per worker),
then finishing any stragglers in-process serial. The per-molecule physics is the
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


def _log(msg: str) -> None:
    print(f"[holdout-parallel] {msg}", flush=True)


def _round_robin(names, k):
    """Partition ``names`` into ``k`` balanced shards (round-robin so a few
    heavy molecules don't all land in one shard)."""
    return [names[i::k] for i in range(k)]


def run_holdout_with_escalation(
    run_dir, spec_idx, training_spec, model, reactions, full_specs, out_dir, *,
    basis, grid_level, n_workers_top, total_cpus, strict=None,
):
    """Run the held-out eval in parallel with adaptive degradation.

    Parameters mirror the serial ``run_full_holdout_eval`` plus the shard-launch
    context (``run_dir``/``spec_idx`` so workers reload the same spec+model) and
    the parallelism budget (``n_workers_top`` = ladder start, ``total_cpus`` =
    threads-per-worker base). ``training_spec``/``model`` are used only for the
    in-process serial leftover tier (workers reload their own). Returns the same
    summary dict as ``run_full_holdout_eval``.
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
                "--threads", str(threads),
            ]
            jobs.append(parallel.WorkerJob(
                name=f"eval_t{tier_no}_s{si}", cmd=cmd,
                progress_file=str(shard_dir / f"progress_t{tier_no}_s{si}.json"),
                thread_env=parallel._thread_env(threads)))

        _log(f"tier {tier_no}: {k} workers x {threads} thread(s) over "
             f"{len(remaining)} molecules")
        results = parallel.run_workers(jobs, max_parallel=n_workers)

        done = set()
        for res, out_shard in zip(results, out_shards):
            if getattr(res, "status", "failed") != "success" or not out_shard.is_file():
                continue
            try:
                payload = json.loads(out_shard.read_text())
            except (OSError, ValueError):
                continue
            shard_payloads.append(payload)
            done.update(payload.get("energies", {}).keys())
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
    # MOLECULE-level names (single atoms excluded): held-out overlap is
    # molecule-level, else shared reference atoms (h, c, n, o, ...) drop nearly
    # the entire atomization held-out set. See eval_holdout.training_molecule_names.
    training_names = eval_holdout.training_molecule_names(training_spec)
    return eval_holdout._finalize_holdout_outputs(
        reactions, energies, pbe_energies, mol_records, training_names,
        n_species=len(full_specs), out_dir=out_dir, strict=strict)
