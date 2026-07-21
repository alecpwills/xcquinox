#!/usr/bin/env python3
"""Report generator for the dfs6311 RSS-growth probe (hpcjobs/dfs6311_rss_probe.sbatch).

Prints, from the probe's .out log and its checkpoint dir:
  - compile-record totals and the post-epoch-1 count (the recompile-ratchet
    discriminator), a per-epoch compile histogram, and the names of functions
    that compile after epoch 1;
  - the per-update RSS/HWM curve with the mean growth rate, and the
    pre/post-validation RSS brackets (the boundary-burst discriminator).

stdlib-only for the normal path. The resume-state fallback (an OOM death
before aux_log.pkl is written at clean exit) unpickles model pytrees and
therefore needs the project conda env; the script says so if it hits that.

Deserialization safety: every pickle read here targets artifacts written by
this pipeline's own training code (aux_log.pkl / resume state) on this
filesystem, never data from an outside source.

Usage (defaults target the 2026-07-20 production run dir):
    python3 dfs6311_rss_probe_report.py [--run-dir DIR] [--log FILE] [--ckpt DIR]
"""
import argparse
import glob
import json
import os
import pickle  # noqa: S403 -- reading this pipeline's own artifacts
import re
import sys

DEFAULT_RUN = ("/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v3/"
               "runs/run_20260720T181059Z")
_COMPILE_RE = re.compile(r"Compiling ([^\s]+)")


def analyze_log(path):
    print(f"== log: {path}")
    n_compile_total = 0
    per_epoch = {}          # last-seen epoch -> compile count (0 = epoch-1 phase)
    late_names = {}
    cur_epoch = 0
    first_steps, last_steps = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            if '"kind": "step"' in line:
                try:
                    msg = json.loads(line[line.index("{"):])
                except (ValueError, json.JSONDecodeError):
                    continue
                cur_epoch = int(msg.get("step", cur_epoch))
                rec = (cur_epoch, msg.get("loss"), msg.get("rss_gb"),
                       msg.get("hwm_gb"))
                if len(first_steps) < 3:
                    first_steps.append(rec)
                last_steps.append(rec)
                if len(last_steps) > 5:
                    last_steps.pop(0)
                continue
            m = _COMPILE_RE.search(line)
            if m:
                n_compile_total += 1
                per_epoch[cur_epoch] = per_epoch.get(cur_epoch, 0) + 1
                if cur_epoch > 0:
                    late_names[m.group(1)] = late_names.get(m.group(1), 0) + 1
    n_late = sum(v for k, v in per_epoch.items() if k > 0)
    print(f"compile records total: {n_compile_total}")
    print(f"compiles AFTER epoch 1: {n_late}   <-- ratchet discriminator")
    print("per-epoch compile histogram (0 = the epoch-1 compile phase):")
    for k in sorted(per_epoch):
        print(f"  after epoch {k:3d}: {per_epoch[k]}")
    if late_names:
        print("late-compiling functions (post-epoch-1), by count:")
        for name, n in sorted(late_names.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {n:5d}  {name}")
    def _fmt(rec):
        ep, loss, rss, hwm = rec
        loss_s = f"{loss:.3e}" if isinstance(loss, float) else str(loss)
        rss_s = f"{rss:.2f}" if isinstance(rss, (int, float)) else str(rss)
        hwm_s = f"{hwm:.2f}" if isinstance(hwm, (int, float)) else str(hwm)
        return f"  epoch {ep:3d}  loss {loss_s}  rss {rss_s}G  hwm {hwm_s}G"
    print("first step lines:")
    for rec in first_steps:
        print(_fmt(rec))
    print("last step lines:")
    for rec in last_steps:
        print(_fmt(rec))


def _load_aux_log(ckpt_dir):
    aux_path = os.path.join(ckpt_dir, "aux_log.pkl")
    if os.path.isfile(aux_path):
        with open(aux_path, "rb") as f:
            return pickle.load(f), "aux_log.pkl"  # noqa: S301 -- own artifact
    print(f"no aux_log.pkl; dir has: "
          f"{sorted(os.path.basename(p) for p in glob.glob(ckpt_dir + '/*'))}")
    for p in sorted(glob.glob(os.path.join(ckpt_dir, "*.pkl"))):
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)  # noqa: S301 -- own artifact
        except Exception as exc:  # noqa: BLE001 -- report and try the next file
            print(f"  cannot unpickle {os.path.basename(p)}: {exc!r}")
            print("  (the resume state holds model pytrees -- activate the "
                  "xcquinox_j070 conda env and rerun this script)")
            continue
        if isinstance(obj, dict) and "aux_log" in obj:
            return obj["aux_log"], os.path.basename(p)
    return None, None


def analyze_aux(ckpt_dir):
    print(f"== checkpoint dir: {ckpt_dir}")
    log, source = _load_aux_log(ckpt_dir)
    if log is None:
        print("no aux_log recovered")
        return
    print(f"source: {source}")
    steps = [e for e in log
             if e.get("group") != "__validation__" and "rss_gb" in e]
    vals = [e for e in log if e.get("group") == "__validation__"]
    print(f"n updates: {len(steps)}   n validations: {len(vals)}")
    for i, e in enumerate(steps):
        if e["step"] < 6 or e["step"] % 20 == 0 or i == len(steps) - 1:
            row = (f"update {e['step']:4d} epoch {e['epoch']:3d}  "
                   f"rss {e['rss_gb']:7.2f}G  hwm {e['hwm_gb']:7.2f}G")
            if "live_gb" in e:
                row += f"  live {e['live_gb']:7.2f}G (n={e['live_n']})"
            print(row)
    if len(steps) > 1:
        d = steps[-1]["rss_gb"] - steps[0]["rss_gb"]
        rate = d / (len(steps) - 1) * 1024.0
        print(f"rss first->last: {steps[0]['rss_gb']:.2f} -> "
              f"{steps[-1]['rss_gb']:.2f} (delta {d:+.2f}G over "
              f"{len(steps)} updates = {rate:+.0f} MB/update)")
        if "live_gb" in steps[0] and "live_gb" in steps[-1]:
            dl = steps[-1]["live_gb"] - steps[0]["live_gb"]
            print(f"live first->last: {steps[0]['live_gb']:.2f} -> "
                  f"{steps[-1]['live_gb']:.2f} (delta {dl:+.2f}G; "
                  f"n {steps[0]['live_n']} -> {steps[-1]['live_n']})")
    for e in vals:
        pre = e.get("rss_gb_pre_val")
        post = e.get("rss_gb_post_val")
        hwm = e.get("hwm_gb_post_val")
        if pre is None or post is None:
            print(f"VAL after epoch {e.get('epoch')}: no rss brackets "
                  f"(entry keys: {sorted(e)})")
            continue
        hwm_s = f"{hwm:.2f}" if isinstance(hwm, (int, float)) else str(hwm)
        print(f"VAL after epoch {e['epoch']:3d}: pre {pre:7.2f}G  "
              f"post {post:7.2f}G  burst {post - pre:+.2f}G  hwm_post {hwm_s}G")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--log", default=None,
                    help="probe .out (default: newest rss_probe_*.out in "
                         "<run-dir>/logs)")
    ap.add_argument("--ckpt", default=None,
                    help="probe checkpoint dir (default: newest "
                         "<run-dir>/checkpoints/rss_probe*_spec_0003)")
    args = ap.parse_args(argv)
    log_path = args.log
    if log_path is None:
        cands = sorted(glob.glob(os.path.join(args.run_dir, "logs",
                                              "rss_probe_*.out")),
                       key=os.path.getmtime)
        if not cands:
            sys.exit(f"no rss_probe_*.out under {args.run_dir}/logs")
        log_path = cands[-1]
    analyze_log(log_path)
    print()
    ckpt = args.ckpt
    if ckpt is None:
        dirs = sorted(glob.glob(os.path.join(args.run_dir, "checkpoints",
                                             "rss_probe*_spec_0003")),
                      key=os.path.getmtime)
        ckpt = dirs[-1] if dirs else os.path.join(args.run_dir, "checkpoints",
                                                  "rss_probe_spec_0003")
    analyze_aux(ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
