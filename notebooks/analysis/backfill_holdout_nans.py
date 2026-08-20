"""Local backfill of NN-NaN holdout species in pulled eval channels.

Transiently failed per-species evaluations leave ``E_total_nn`` null in a
channel's ``per_molecule.json`` while the model-free columns (E_pbe, the
density block) survive; the figure layer then stars the cell as an
incomplete hold-out eval. This driver recomputes EXACTLY those species
locally under the production eval identity and patches the energy/SCF
fields in place, leaving every other field verbatim.

No physics lives here. Each (spec, channel) group runs one
``xcquinox.alec.workers.eval_holdout_worker`` subprocess over the NaN
species plus a few finite CONTROL species; the worker path is the same
code the cluster shards run (spec pickle -> PBE seed + density fitting +
orientation lock -> truncated NN SCF -> record writer). A channel is
patched only when every control species reproduces its recorded
``E_total_nn`` and ``E_pbe`` within the gates, and each backfilled species
additionally requires its locally converged ``E_pbe`` to match the
recorded value -- a PBE SCF landing on a different solution (the
multireference-C2 class) disqualifies that species rather than seeding
the NN from the wrong density.

The density columns are NEVER touched: the benchmark CCSD reference
densities are not staged locally, so a recomputed row would null them.

Repair doctrine (mirrors ``cluster.coldstart_retro`` /
``refinalize_verbatim``): idempotent (done = no non-finite ``E_total_nn``
left), once-only ``per_molecule.pre_backfill.json`` backup, atomic
replace, a ``backfill_meta.json`` stamp per channel, and every computed
payload banked in ``backfill_ledger.json`` so a cluster pull that
overwrites the patched files can be re-applied instantly without
recomputation. Derived files (``per_reaction.json`` / ``test_set.csv``)
are regenerated afterwards by ``xcquinox.alec.refinalize_verbatim``
(``--refinalize``, default on when anything was patched).

Usage (from the repo root):

    python notebooks/analysis/backfill_holdout_nans.py <run_dir> \
        [--specs 42,44,45] [--channels eval_holdout ...] \
        [--controls 3] [--gate-nn ...] [--gate-pbe ...] \
        [--threads 6] [--chunk 4] [--measure-only | --dry-run]

``--measure-only`` evaluates the control species alone and reports the
reproduction deltas without writing anything -- the measurement the gate
defaults are anchored to.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CHANNEL_MODELS = {
    "eval_holdout": "model.eqx",
    "eval_holdout_best": "model_best.eqx",
    "eval_holdout_val_best": "model_val_best.eqx",
}

BACKUP_NAME = "per_molecule.pre_backfill.json"
LEDGER_NAME = "backfill_ledger.json"
STAMP_NAME = "backfill_meta.json"

# Gate defaults, anchored to the 2026-08-18 --measure-only run against
# the v4gga cluster evals (spec_0044/eval_holdout controls C2H5_2,
# ethanol, t-n2h2): max observed |dE_pbe| = 4.5e-13 Ha and |dE_nn| =
# 1.4e-12 Ha. The theoretical floor is the converged-PBE SCF tolerance
# (~1e-9 Ha cross-machine); the C2-class multi-solution signal is 4e-2
# Ha. 1e-7 / 1e-6 sit two decades above the conv-tol bound and five-plus
# below the signal.
DEFAULT_GATE_PBE = 1e-7
DEFAULT_GATE_NN = 1e-6

# Fields owned by the energy/SCF recomputation; everything else in a
# record is preserved verbatim (density block, quadrature bookkeeping,
# from_training_subset, and the recorded E_pbe). ``eval_error`` names the
# exception that produced the NaN row, so it belongs to the superseded
# evaluation and is dropped with the rest when a species is patched.
_SCF_PREFIXES = ("scf_energy_step_", "scf_energy_residual_")
_SCF_SCALARS = ("cycles_run", "scf_converged", "scf_total_energy",
                "eval_error")


def _is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def nonfinite_species(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Sorted species names whose ``E_total_nn`` is null/NaN/absent."""
    return sorted(str(r.get("molecule")) for r in records
                  if not _is_finite(r.get("E_total_nn")))


def pick_controls(records: Sequence[Dict[str, Any]], k: int) -> List[str]:
    """``k`` finite HELD-OUT species (training-subset rows excluded),
    evenly spaced over the sorted name list -- deterministic, so repeated
    runs measure the same reproduction deltas."""
    finite = sorted(str(r.get("molecule")) for r in records
                    if _is_finite(r.get("E_total_nn"))
                    and not r.get("from_training_subset"))
    if k >= len(finite):
        return finite
    if k <= 0:
        return []
    step = (len(finite) - 1) / max(1, k - 1) if k > 1 else 0
    idx = sorted({round(i * step) for i in range(k)})
    return [finite[i] for i in idx]


def read_basis_grid(run_dir: Path) -> Tuple[str, int]:
    """``(basis, grid_level)`` from ``resolved_config.yaml``'s inputs
    block -- the same hand parser as ``reeval_holdout_fixed``, so no yaml
    dependency."""
    basis: Optional[str] = None
    grid: Optional[int] = None
    for line in (Path(run_dir) / "resolved_config.yaml").read_text(
            ).splitlines():
        s = line.strip()
        if s.startswith("basis:") and basis is None:
            basis = s.split(":", 1)[1].strip().strip("'\"")
        elif s.startswith("grid_level:") and grid is None:
            grid = int(s.split(":", 1)[1].strip())
    if basis is None or grid is None:
        raise SystemExit(f"FATAL: basis/grid_level not found in "
                         f"{run_dir}/resolved_config.yaml")
    return basis, grid


def _payload_record(payload: Dict[str, Any], name: str
                    ) -> Optional[Dict[str, Any]]:
    for r in payload.get("mol_records") or []:
        if r.get("molecule") == name:
            return r
    return None


def patch_records(records: Sequence[Dict[str, Any]],
                  payload: Dict[str, Any], *,
                  controls: Sequence[str],
                  gate_nn: float, gate_pbe: float
                  ) -> Tuple[Optional[List[Dict[str, Any]]],
                             Dict[str, Any]]:
    """Pure patch step: ``(new_records, report)``.

    Controls gate the whole channel (any control whose recomputed
    ``E_total_nn``/``E_pbe`` misses its recorded value beyond the gates,
    or comes back non-finite, aborts with ``new_records=None``). Each NaN
    target is then patched only when its recomputed ``E_pbe`` matches the
    RECORDED one within ``gate_pbe`` -- the recorded ``E_pbe`` is kept and
    ``AE_nn`` recomputed against it. Density and every other field are
    preserved verbatim; stale ``scf_*`` keys are dropped before the new
    trace is appended."""
    by_old = {str(r.get("molecule")): r for r in records}
    energies = payload.get("energies") or {}
    pbe_energies = payload.get("pbe_energies") or {}
    report: Dict[str, Any] = {"patched": {}, "skipped": {}, "diverged": [],
                              "unresolved": [], "controls": {},
                              "aborted": False, "abort_reason": None}
    has_targets = any(not _is_finite(r.get("E_total_nn")) for r in records)
    if has_targets and not controls:
        # No finite held-out control species (or an explicit zero): the
        # replay identity was never confirmed against the recorded eval,
        # so nothing may be written to the canonical record. (Without
        # targets there is nothing to write and no identity to confirm.)
        report["aborted"] = True
        report["abort_reason"] = ("no control species available -- "
                                  "identity unconfirmed, refusing to patch")
        return None, report
    for c in controls:
        old = by_old.get(c) or {}
        e_nn, e_pbe = energies.get(c), pbe_energies.get(c)
        if not (_is_finite(e_nn) and _is_finite(e_pbe)
                and _is_finite(old.get("E_total_nn"))
                and _is_finite(old.get("E_pbe"))):
            report["controls"][c] = {"dE_nn": None, "dE_pbe": None}
            report["aborted"] = True
            report["abort_reason"] = (f"control {c} not finite on both "
                                      "sides -- identity unconfirmed")
            continue
        d_nn = abs(float(e_nn) - float(old["E_total_nn"]))
        d_pbe = abs(float(e_pbe) - float(old["E_pbe"]))
        report["controls"][c] = {"dE_nn": d_nn, "dE_pbe": d_pbe}
        if d_nn > gate_nn or d_pbe > gate_pbe:
            report["aborted"] = True
            report["abort_reason"] = (
                f"control {c} reproduction outside gates "
                f"(dE_nn={d_nn:.3e}, dE_pbe={d_pbe:.3e})")
    if report["aborted"]:
        return None, report

    out: List[Dict[str, Any]] = []
    for r in records:
        name = str(r.get("molecule"))
        if _is_finite(r.get("E_total_nn")):
            out.append(r)
            continue
        new_rec = _payload_record(payload, name)
        if new_rec is None:
            # No per-molecule record for this target -- either the worker
            # dropped the species entirely (a per-species precompute
            # failure) or the payload is inconsistent. Patching from the
            # energies map alone would write a row without its SCF
            # bookkeeping fields: refuse and name it.
            report["unresolved"].append(name)
            out.append(r)
            continue
        e_nn = energies.get(name)
        if e_nn is None:
            e_nn = new_rec.get("E_total_nn")
        if not _is_finite(e_nn):
            report["diverged"].append(name)
            out.append(r)
            continue
        e_pbe_local = pbe_energies.get(name)
        e_pbe_rec = r.get("E_pbe")
        if not (_is_finite(e_pbe_local) and _is_finite(e_pbe_rec)):
            report["skipped"][name] = {"reason": "pbe-not-finite",
                                       "dE_pbe": None}
            out.append(r)
            continue
        d_pbe = abs(float(e_pbe_local) - float(e_pbe_rec))
        if d_pbe > gate_pbe:
            report["skipped"][name] = {"reason": "pbe-mismatch",
                                       "dE_pbe": d_pbe}
            out.append(r)
            continue
        patched = {k: v for k, v in r.items()
                   if not (k.startswith(_SCF_PREFIXES) or k in _SCF_SCALARS)}
        patched["E_total_nn"] = float(e_nn)
        patched["AE_nn"] = float(e_nn) - float(e_pbe_rec)
        for k in _SCF_SCALARS:
            if new_rec is not None and k in new_rec:
                patched[k] = new_rec[k]
        if new_rec is not None:
            for k, v in new_rec.items():
                if k.startswith(_SCF_PREFIXES):
                    patched[k] = v
        report["patched"][name] = {"E_total_nn": float(e_nn),
                                   "dE_pbe": d_pbe}
        out.append(patched)
    return out, report


def write_patched(channel_dir: Path,
                  records: Sequence[Dict[str, Any]]) -> None:
    """Once-only backup of the pre-backfill file, then atomic replace,
    matching the cluster writer's format (indent=2)."""
    channel_dir = Path(channel_dir)
    target = channel_dir / "per_molecule.json"
    backup = channel_dir / BACKUP_NAME
    if target.is_file() and not backup.is_file():
        backup.write_bytes(target.read_bytes())
    tmp = channel_dir / f".tmp.{target.name}"
    with tmp.open("w") as f:
        json.dump(list(records), f, indent=2)
    os.replace(tmp, target)


def load_ledger(channel_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(channel_dir) / LEDGER_NAME
    if not p.is_file():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_ledger(channel_dir: Path, payload: Dict[str, Any]) -> None:
    """Merge ``payload`` into the channel's ledger (existing entries
    win: the first successful recomputation of a species is the one any
    re-apply reproduces)."""
    led = load_ledger(channel_dir) or {"energies": {}, "pbe_energies": {},
                                       "mol_records": []}
    have = {r.get("molecule") for r in led["mol_records"]}
    for k in ("energies", "pbe_energies"):
        for name, v in (payload.get(k) or {}).items():
            led[k].setdefault(name, v)
    for r in payload.get("mol_records") or []:
        if r.get("molecule") not in have:
            led["mol_records"].append(r)
    tmp = Path(channel_dir) / f".tmp.{LEDGER_NAME}"
    with tmp.open("w") as f:
        json.dump(led, f)
    os.replace(tmp, Path(channel_dir) / LEDGER_NAME)


def _merge_stamp(old: Optional[Dict[str, Any]],
                 new: Dict[str, Any]) -> Dict[str, Any]:
    """Union of two passes' per-species outcomes: a retry extends the
    stamp instead of erasing the earlier pass. Per species the newer
    classification wins, and a patched species leaves every other
    bucket."""
    if not old:
        return new
    merged = dict(new)
    merged["patched"] = {**(old.get("patched") or {}),
                         **(new.get("patched") or {})}
    merged["controls"] = {**(old.get("controls") or {}),
                          **(new.get("controls") or {})}
    newly_classified = (set(new.get("patched") or {})
                        | set(new.get("skipped") or {})
                        | set(new.get("diverged") or [])
                        | set(new.get("unresolved") or []))
    skipped = {k: v for k, v in (old.get("skipped") or {}).items()
               if k not in newly_classified}
    skipped.update(new.get("skipped") or {})
    merged["skipped"] = {k: v for k, v in skipped.items()
                         if k not in merged["patched"]}
    for key in ("diverged", "unresolved"):
        names = set(old.get(key) or []) - newly_classified
        names |= set(new.get(key) or [])
        merged[key] = sorted(n for n in names
                             if n not in merged["patched"]
                             and n not in merged["skipped"])
    return merged


def _write_stamp(channel_dir: Path, stamp: Dict[str, Any]) -> None:
    old = None
    p = Path(channel_dir) / STAMP_NAME
    if p.is_file():
        try:
            with p.open() as f:
                old = json.load(f)
        except (json.JSONDecodeError, OSError):
            old = None
    stamp = _merge_stamp(old, stamp)
    tmp = Path(channel_dir) / f".tmp.{STAMP_NAME}"
    with tmp.open("w") as f:
        json.dump(stamp, f, indent=2, sort_keys=True)
    os.replace(tmp, p)


def process_channel_records(channel_dir: Path, *,
                            controls: Sequence[str],
                            gate_nn: float, gate_pbe: float,
                            compute_fn: Callable[[List[str]],
                                                 Dict[str, Any]],
                            dry_run: bool = False) -> Dict[str, Any]:
    """One channel end to end: find targets, prefer the banked ledger,
    compute the remainder through ``compute_fn(names)`` (the worker seam),
    gate, patch, stamp. Returns the report dict (``status`` one of
    ``nothing-to-do`` / ``patched`` / ``would-patch`` / ``aborted``)."""
    channel_dir = Path(channel_dir)
    with (channel_dir / "per_molecule.json").open() as f:
        records = json.load(f)
    targets = nonfinite_species(records)
    if not targets:
        return {"status": "nothing-to-do", "targets": [], "patched": {}}

    # Ledger first (instant re-apply after a pull clobber) -- but the
    # ledger attempt only stands when it actually patches something; a
    # ledger whose coverage is stale, partial, or entirely gate-rejected
    # falls through to a fresh computation, so a channel can never
    # deadlock as nothing-to-do while non-finite entries remain.
    led = load_ledger(channel_dir)
    new_records = None
    report: Optional[Dict[str, Any]] = None
    source = None
    if led is not None:
        covered = {n for n in targets if _is_finite(
            (led.get("energies") or {}).get(n))}
        ctl_ok = all(_is_finite((led.get("energies") or {}).get(c))
                     for c in controls)
        if covered and ctl_ok:
            cand_records, cand = patch_records(
                records, led, controls=controls,
                gate_nn=gate_nn, gate_pbe=gate_pbe)
            if not cand["aborted"] and cand["patched"]:
                new_records, report, source = cand_records, cand, "ledger"
    if report is None:
        names = sorted(set(targets) | set(controls))
        payload = compute_fn(names)
        source = "compute"
        if not dry_run:
            save_ledger(channel_dir, payload)
        new_records, report = patch_records(records, payload,
                                            controls=controls,
                                            gate_nn=gate_nn,
                                            gate_pbe=gate_pbe)
    report["targets"] = targets
    report["source"] = source
    if report["aborted"]:
        report["status"] = "aborted"
        return report
    if dry_run:
        report["status"] = ("would-patch" if report["patched"]
                            else "no-species-patched")
        return report
    if report["patched"]:
        write_patched(channel_dir, new_records)
    _write_stamp(channel_dir, {
        "tool": "backfill_holdout_nans",
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "gates": {"gate_nn": gate_nn, "gate_pbe": gate_pbe},
        "controls": report["controls"],
        "patched": report["patched"],
        "skipped": report["skipped"],
        "diverged": report["diverged"],
        "unresolved": report["unresolved"],
    })
    report["status"] = ("patched" if report["patched"]
                        else "no-species-patched")
    return report


# ---------------------------------------------------------------------------
# Worker subprocess seam
# ---------------------------------------------------------------------------

def _worker_tag(spec_idx: int, model_name: str, names: Sequence[str],
                seq: int = 0) -> str:
    """Work-file tag unique per chunk: equal-length chunks of one
    (spec, model) must not share names/shard filenames -- ``seq`` is the
    caller's chunk ordinal."""
    return (f"s{spec_idx:04d}_{model_name.replace('.eqx', '')}"
            f"_{seq:02d}_{len(names)}")


def run_worker(run_dir: Path, spec_idx: int, names: Sequence[str],
               basis: str, grid_level: int, model_name: str, *,
               threads: int, workdir: Path,
               timeout_s: int = 3600, seq: int = 0) -> Dict[str, Any]:
    """One ``eval_holdout_worker`` subprocess over ``names``; returns the
    shard payload. The worker pins JAX_ENABLE_X64 and the BLAS caps before
    importing jax, exactly as the cluster shards do. Species the worker
    returns nothing for (a per-species precompute failure inside the
    shard) are named on stdout here; the patch layer classifies them as
    unresolved."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tag = _worker_tag(spec_idx, model_name, names, seq=seq)
    names_file = workdir / f"names_{tag}.json"
    out_shard = workdir / f"shard_{tag}.json"
    with names_file.open("w") as f:
        json.dump(list(names), f)
    cmd = [sys.executable, "-m",
           "xcquinox.alec.workers.eval_holdout_worker",
           "--run-dir", str(run_dir), "--spec-idx", str(spec_idx),
           "--names-file", str(names_file), "--out-shard", str(out_shard),
           "--basis", basis, "--grid-level", str(grid_level),
           "--threads", str(threads), "--model-name", model_name]
    env = dict(os.environ)
    env["MALLOC_ARENA_MAX"] = "2"
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_s, env=env)
    status = {}
    for line in reversed((proc.stdout or "").strip().splitlines()):
        try:
            status = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if proc.returncode != 0 or status.get("status") != "success":
        raise RuntimeError(
            f"worker failed for spec {spec_idx}/{model_name} "
            f"({len(names)} species, {time.time() - t0:.0f}s): "
            f"{status.get('error') or proc.stderr[-500:]}")
    with out_shard.open() as f:
        payload = json.load(f)
    returned = (set(payload.get("energies") or {})
                | {r.get("molecule") for r in
                   payload.get("mol_records") or []})
    dropped = sorted(set(names) - returned)
    if dropped:
        print(f"[backfill] WARNING: worker s{spec_idx:04d}/{model_name} "
              f"returned nothing for {dropped} (per-species failure "
              f"inside the shard; stderr tail: "
              f"{(proc.stderr or '')[-300:].strip()!r})", flush=True)
    return payload


def _chunked(seq: Sequence[str], n: int) -> List[List[str]]:
    seq = list(seq)
    return [seq[i:i + n] for i in range(0, len(seq), n)] if n > 0 else [seq]


def _merge_payloads(parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"energies": {}, "pbe_energies": {},
                           "mol_records": []}
    for p in parts:
        out["energies"].update(p.get("energies") or {})
        out["pbe_energies"].update(p.get("pbe_energies") or {})
        out["mol_records"].extend(p.get("mol_records") or [])
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Backfill NN-NaN holdout species locally under the "
                    "production eval identity.")
    p.add_argument("run_dir")
    p.add_argument("--specs", default=None,
                   help="comma-separated spec indices (default: every spec "
                        "with a NaN in a requested channel)")
    p.add_argument("--channels", nargs="+", default=list(CHANNEL_MODELS),
                   choices=list(CHANNEL_MODELS))

    def _controls_count(v: str) -> int:
        n = int(v)
        if n < 2:
            raise argparse.ArgumentTypeError(
                "at least 2 control species are required -- a patch "
                "without confirmed identity must not enter the record")
        return n

    p.add_argument("--controls", type=_controls_count, default=3)
    p.add_argument("--gate-nn", type=float, default=DEFAULT_GATE_NN)
    p.add_argument("--gate-pbe", type=float, default=DEFAULT_GATE_PBE)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--chunk", type=int, default=4,
                   help="species per worker call (progress granularity; "
                        "controls ride the first chunk's process)")
    p.add_argument("--timeout", type=int, default=3600,
                   help="per-worker-call timeout, seconds")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--measure-only", action="store_true",
                   help="evaluate the control species only and report the "
                        "reproduction deltas; writes nothing")
    p.add_argument("--no-refinalize", action="store_true",
                   help="skip the per_reaction/test_set regeneration pass")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not (run_dir / "checkpoints").is_dir():
        print(f"FATAL: {run_dir} has no checkpoints/ -- not a run dir")
        return 1
    basis, grid_level = read_basis_grid(run_dir)
    workdir = run_dir / "backfill_work"
    print(f"[backfill] {run_dir.name}: basis={basis} grid_level={grid_level}"
          f" gates: dE_pbe<={args.gate_pbe:g} dE_nn<={args.gate_nn:g}")

    if args.specs:
        indices = [int(s) for s in args.specs.split(",")]
    else:
        indices = sorted(
            int(d.name.split("_")[1])
            for d in (run_dir / "checkpoints").glob("spec_*")
            if any((d / ch / "per_molecule.json").is_file()
                   for ch in args.channels))

    any_patched = False
    rc = 0
    for idx in indices:
        spec_dir = run_dir / "checkpoints" / f"spec_{idx:04d}"
        for channel in args.channels:
            ch_dir = spec_dir / channel
            pm = ch_dir / "per_molecule.json"
            if not pm.is_file():
                continue
            with pm.open() as f:
                records = json.load(f)
            targets = nonfinite_species(records)
            controls = pick_controls(records, args.controls)
            model_name = CHANNEL_MODELS[channel]
            tag = f"spec_{idx:04d}/{channel}"

            if args.measure_only:
                if not controls:
                    print(f"[backfill] {tag}: no finite controls")
                    continue
                print(f"[backfill] {tag}: measuring {len(controls)} "
                      f"controls {controls} ...", flush=True)
                payload = run_worker(run_dir, idx, controls, basis,
                                     grid_level, model_name,
                                     threads=args.threads, workdir=workdir,
                                     timeout_s=args.timeout)
                by = {str(r.get("molecule")): r for r in records}
                for c in controls:
                    d_nn = abs(payload["energies"][c]
                               - by[c]["E_total_nn"])
                    d_pbe = abs(payload["pbe_energies"][c]
                                - by[c]["E_pbe"])
                    print(f"[backfill] {tag}: control {c}: "
                          f"dE_nn={d_nn:.3e} dE_pbe={d_pbe:.3e}")
                continue

            if not targets:
                continue
            print(f"[backfill] {tag}: {len(targets)} NaN species "
                  f"{targets}", flush=True)
            if args.dry_run:
                led = load_ledger(ch_dir) or {}
                covered = [n for n in targets if _is_finite(
                    (led.get("energies") or {}).get(n))]
                print(f"[backfill] {tag}: would evaluate "
                      f"{len(targets) - len(covered)} species + "
                      f"{len(controls)} controls {controls} "
                      f"({len(covered)} already banked in the ledger)")
                continue

            def _compute(names: List[str], _idx=idx, _mn=model_name,
                         _tag=tag) -> Dict[str, Any]:
                parts = []
                chunks = _chunked(names, args.chunk)
                for i, chunk in enumerate(chunks):
                    t0 = time.time()
                    print(f"[backfill] {_tag}: worker {i + 1}/"
                          f"{len(chunks)} over {chunk} ...", flush=True)
                    parts.append(run_worker(
                        run_dir, _idx, chunk, basis, grid_level, _mn,
                        threads=args.threads, workdir=workdir,
                        timeout_s=args.timeout, seq=i))
                    print(f"[backfill] {_tag}: worker {i + 1}/"
                          f"{len(chunks)} done in "
                          f"{time.time() - t0:.0f}s", flush=True)
                return _merge_payloads(parts)

            report = process_channel_records(
                ch_dir, controls=controls, gate_nn=args.gate_nn,
                gate_pbe=args.gate_pbe, compute_fn=_compute,
                dry_run=args.dry_run)
            ctl = report.get("controls") or {}
            if ctl:
                worst_nn = max((v["dE_nn"] for v in ctl.values()
                                if v["dE_nn"] is not None), default=None)
                worst_pbe = max((v["dE_pbe"] for v in ctl.values()
                                 if v["dE_pbe"] is not None), default=None)
                print(f"[backfill] {tag}: controls max dE_nn="
                      f"{worst_nn if worst_nn is None else f'{worst_nn:.3e}'}"
                      f" max dE_pbe="
                      f"{worst_pbe if worst_pbe is None else f'{worst_pbe:.3e}'}")
            unres = report.get("unresolved") or []
            print(f"[backfill] {tag}: {report['status']} -- "
                  f"{len(report.get('patched') or {})} patched, "
                  f"{len(report.get('skipped') or {})} skipped, "
                  f"{len(report.get('diverged') or [])} diverged locally, "
                  f"{len(unres)} unresolved"
                  + (f" {unres}" if unres else "")
                  + (f" ({report['abort_reason']})"
                     if report.get("abort_reason") else ""), flush=True)
            if report["status"] == "aborted" or unres:
                # An unresolved target is a tool failure (retryable),
                # unlike a gate skip or a measured local divergence,
                # which are documented terminal outcomes.
                rc = 1
            if report["status"] == "patched":
                any_patched = True

    if any_patched and not args.no_refinalize and not args.dry_run \
            and not args.measure_only:
        print("[backfill] regenerating per_reaction/test_set via "
              "refinalize_verbatim ...", flush=True)
        from xcquinox.alec.refinalize_verbatim import refinalize_run
        refinalize_run(run_dir, channels=tuple(args.channels))
    return rc


if __name__ == "__main__":
    sys.exit(main())
