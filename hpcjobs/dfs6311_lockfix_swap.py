"""Atomically swap the relocked CH/NO training references into the live cache.

Run on the login node after ``dfs6311_lockfix_chno_regen.sbatch`` has staged and
verified the regenerated references. Takes seconds; the production sweep keeps
running throughout.

WHY THIS IS SAFE MID-SWEEP. The training batch reads its references at TRAIN
time, not at datagen: ``train._build_batch`` calls
``data.precompute_fixed_density_data`` (train.py:358 via run_training), whose
in-memory cache key includes the reference file's ``(mtime_ns, size)``
(data.py:280-283, "so that re-running ... invalidates stale cache entries").
Every array task is a fresh process, so no stale entry can survive. The
replacement itself is a single ``os.replace``, which is atomic: a task loading
the file gets either the whole old inode or the whole new one, never a torn
read. Consequently every task the harness has NOT yet fired picks up the
relocked references automatically, and tasks already running are unaffected.

WHAT THIS CREATES. A deliberate provenance boundary inside one run: specs whose
training started before the swap trained against UNLOCKED CH/NO references,
specs started after train against locked ones. That is a real change in the
density channel mid-sweep, so the boundary MUST be recorded or later figures
cannot be interpreted. This script writes ``lockfix_swap_manifest.json`` into
the run directory capturing, at swap time: the swap timestamp, the per-species
before/after lock stamps and mtimes, and the exact spec partition (complete /
in flight / not yet started). The "not yet started" list is the set that will
train against the fixed references.

The displaced references are preserved as ``<name>.npz.prelock`` so the old
behaviour remains reproducible for the before/after comparison and for
re-running the affected cells later.

Usage:
    python dfs6311_lockfix_swap.py --stage <staging dir>            # dry run
    python dfs6311_lockfix_swap.py --stage <staging dir> --commit   # do it
"""

import argparse
import json
import os
import time

import numpy as np

LIVE_REFS = "/gpfs/scratch/awills/external_refs_dfs_6311ppg3df2pd_g3"
RUN_DIR = ("/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v3/runs"
           "/run_20260728T140018Z")
SPECIES = ("CH", "NO")
EXPECTED_LOCK = 3e-05
LOCK_KEY = "orientation_lock_strength"


def lock_stamp(path):
    """(lock_or_None, mtime_string) for a reference npz, or (None, None)."""
    if not os.path.isfile(path):
        return None, None
    # our own generated reference cache; allow_pickle is needed only because
    # the string identity stamps are stored as object arrays
    with np.load(path, allow_pickle=True) as z:
        lock = (float(np.asarray(z[LOCK_KEY]).item())
                if LOCK_KEY in z.files else None)
    return lock, time.ctime(os.stat(path).st_mtime)


def spec_partition(run_dir):
    """Partition the run's specs by training state at this instant.

    complete    -- completion.json present (trained on the OLD references)
    in_flight   -- resume state but no completion (started on the OLD ones)
    not_started -- neither; these will train on the NEW references
    """
    ck = os.path.join(run_dir, "checkpoints")
    out = {"complete": [], "in_flight": [], "not_started": []}
    if not os.path.isdir(ck):
        return out
    for name in sorted(os.listdir(ck)):
        d = os.path.join(ck, name)
        if not (name.startswith("spec_") and os.path.isdir(d)):
            continue
        if os.path.isfile(os.path.join(d, "completion.json")):
            out["complete"].append(name)
        elif any(f.startswith("resume_") for f in os.listdir(d)):
            out["in_flight"].append(name)
        else:
            out["not_started"].append(name)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, help="verified staging dir")
    ap.add_argument("--live", default=LIVE_REFS)
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--commit", action="store_true",
                    help="perform the swap (default: dry run)")
    args = ap.parse_args()

    print(f"swap time : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"staging   : {args.stage}")
    print(f"live refs : {args.live}")

    # --- gate: staged files must carry the production lock ------------------
    problems = []
    plan = {}
    for name in SPECIES:
        staged = os.path.join(args.stage, f"{name}.npz")
        live = os.path.join(args.live, f"{name}.npz")
        s_lock, s_mtime = lock_stamp(staged)
        l_lock, l_mtime = lock_stamp(live)
        plan[name] = {"staged_lock": s_lock, "staged_mtime": s_mtime,
                      "live_lock_before": l_lock, "live_mtime_before": l_mtime}
        if s_lock is None:
            problems.append(f"{name}: staged file missing or unstamped")
        elif f"{s_lock:g}" != f"{EXPECTED_LOCK:g}":
            problems.append(f"{name}: staged lock {s_lock:g} != "
                            f"{EXPECTED_LOCK:g}")
        print(f"  {name}: live lock "
              f"{'ABSENT' if l_lock is None else format(l_lock, 'g')} "
              f"({l_mtime})  ->  staged "
              f"{'ABSENT' if s_lock is None else format(s_lock, 'g')} "
              f"({s_mtime})")
    # The partition is informational and prints even when the gate fails, so a
    # dry run always shows where the boundary would fall.
    part = spec_partition(args.run_dir)
    print(f"\nspec partition at this instant ({args.run_dir}):")
    print(f"  complete    : {len(part['complete'])}  "
          f"{part['complete'][:3]}{' ...' if len(part['complete']) > 3 else ''}")
    print(f"  in flight   : {len(part['in_flight'])}  {part['in_flight']}")
    print(f"  not started : {len(part['not_started'])}  "
          f"(these will train against the RELOCKED references)")
    if part["not_started"]:
        print(f"      first: {part['not_started'][0]}   "
              f"last: {part['not_started'][-1]}")

    if problems:
        print("\nREFUSING TO SWAP -- staged references failed the gate:")
        for p in problems:
            print(f"  {p}")
        return 1

    if not args.commit:
        print("\nDRY RUN -- nothing changed. Re-run with --commit to swap.")
        return 0

    # --- atomic swap, preserving the displaced references -------------------
    swapped = []
    for name in SPECIES:
        staged = os.path.join(args.stage, f"{name}.npz")
        live = os.path.join(args.live, f"{name}.npz")
        if os.path.isfile(live):
            backup = live + ".prelock"
            os.replace(live, backup)
            print(f"  preserved {name}.npz -> {os.path.basename(backup)}")
        os.replace(staged, live)
        swapped.append(name)
        print(f"  swapped in {name}.npz")
    dfd = os.open(args.live, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)

    # --- provenance manifest -------------------------------------------------
    for name in SPECIES:
        lock, mtime = lock_stamp(os.path.join(args.live, f"{name}.npz"))
        plan[name]["live_lock_after"] = lock
        plan[name]["live_mtime_after"] = mtime
    manifest = {
        "what": "CH/NO training references relocked to "
                f"orientation_lock_strength={EXPECTED_LOCK:g} mid-sweep",
        "why": "references predated the lock stamp and the consumer guard is "
               "blind to a MISSING stamp; see notebooks/analysis/"
               "DENSITY_DIAGNOSIS.md",
        "swap_time_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "swap_time_epoch": int(time.time()),
        "live_refs_dir": args.live,
        "staging_dir": args.stage,
        "species": plan,
        "spec_partition_at_swap": part,
        "reading": ("specs in 'complete' and 'in_flight' trained against the "
                    "UNLOCKED references; specs in 'not_started' train against "
                    "the relocked ones. Any figure mixing them must disclose "
                    "the boundary."),
    }
    out = os.path.join(args.run_dir, "lockfix_swap_manifest.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(tmp, out)
    print(f"\nwrote provenance manifest: {out}")

    print("\nverification of the LIVE directory after the swap:")
    for name in SPECIES:
        lock, mtime = lock_stamp(os.path.join(args.live, f"{name}.npz"))
        state = "ABSENT" if lock is None else f"{lock:g}"
        print(f"  {name}: lock {state}  ({mtime})")
    print("\nSWAP COMPLETE. Every task the harness has not yet fired will read "
          "the relocked references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
