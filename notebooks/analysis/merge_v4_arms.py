"""Build a merged VIEW of the v4 campaign arms for cross-arm figures.

The campaign runs as separate arms (the GGA/rung-3.5 v4 arm plus the two
SCAN-seeded v5 mgga arms; the PBE-seeded v4 mgga arms are retired and
excluded), each with its own run dir.
The figure collectors scan ``<run_dir>/checkpoints/spec_*``, so a merged
9-arch figure needs one directory whose spec dirs span every arm. This
script builds exactly that: a view directory of RENUMBERED SYMLINKS to the
arms' spec dirs -- no data is copied or modified, and every existing figure
function works on the view unchanged. Arms whose run dirs do not exist yet
are skipped, so the view grows as the campaign lands.

The view is rebuilt from scratch on every invocation (idempotent); its name
carries no ``run_YYYYMMDDT`` stamp, so the V_xc-provenance figure layer
conservatively draws no pre-correction marks on it -- correct, since every
arm postdates the correction.

Usage:
    python notebooks/analysis/merge_v4_arms.py [--results-root DIR]
                                               [--out DIR]

Default results root: ~/Documents/Research/xcquinox-results/runs/dfs_step7
(the pull target of pull_and_plot_v4.sh). The newest run under each arm's
``runs/`` directory is used.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# v5 era (2026-08-14): the retired v4 mgga arms (PBE-seeded, cancelled
# mid-array) are EXCLUDED; the roster is the still-valid GGA/rung-3.5 arm
# plus the two SCAN-seeded v5 arms. Per-arch seed provenance is VALIDATED
# against the rung-baseline policy before an arm enters the view.
ARM_BASES = ("dfs6311_grid3_v4gga", "dfs6311_grid3_v5",
             "dfs6311_grid3_v5mgga2")
DEFAULT_ROOT = Path.home() / "Documents/Research/xcquinox-results/runs/dfs_step7"


def newest_run(base_dir: Path) -> Path | None:
    """The lexically newest ``run_*`` under ``<base>/runs`` (timestamps sort)."""
    runs = base_dir / "runs"
    if not runs.is_dir():
        return None
    candidates = sorted(d for d in runs.iterdir()
                        if d.is_dir() and d.name.startswith("run_"))
    return candidates[-1] if candidates else None


_IDENTITY_KEYS = ("basis:", "density_fit:", "grid_level:")


def _config_identity(cfg: Path):
    """The production-identity lines of a resolved_config.yaml (basis /
    density_fit / grid_level), or None when the file is absent/unreadable."""
    try:
        lines = cfg.read_text().splitlines()
    except OSError:
        return None
    return tuple(next((ln.strip() for ln in lines
                       if ln.strip().startswith(k)), None)
                 for k in _IDENTITY_KEYS)


def _arm_manifest_entries(run: Path) -> dict:
    """{original_index: full manifest entry} from the arm's manifest.json
    (empty if absent/unreadable). Full entries so the merged manifest can
    carry the spec_file/sha256 provenance fields through."""
    mpath = run / "manifest.json"
    if not mpath.is_file():
        return {}
    try:
        manifest = json.loads(mpath.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {e["index"]: e
            for e in manifest.get("specs", []) if isinstance(e.get("index"), int)}


def _validate_arm_seed_policy(run: Path, arch_names) -> None:
    """Refuse an arm whose resolved seed diverges from the rung-baseline
    policy for any REGISTRY arch it carries.

    The grouped figures must only ever assemble correctly seeded data: a
    PBE-seeded mgga arm (the retired v4 protocol) resolves seed 'pbe'
    where the policy demands 'scan' and is rejected here, not silently
    merged. Archs not in the registry (test fixtures, legacy names) have
    no policy expectation and are skipped; a registry arch WITHOUT a
    loadable resolved_config.yaml is unverifiable and also refused.
    """
    from xcquinox.alec.rungs import seed_xc_for_arch
    registry_archs = {}
    for a in arch_names:
        if not a:
            continue
        try:
            registry_archs[a] = seed_xc_for_arch(a)
        except KeyError:
            continue
    if not registry_archs:
        return
    try:
        from xcquinox.alec.cluster.grid_config import load_grid_config
        from xcquinox.alec.cluster.spec_builder import resolve_seed_xc
        cfg = load_grid_config(str(run / "resolved_config.yaml"))
    except Exception as exc:  # noqa: BLE001 -- unverifiable = refused
        raise SystemExit(
            f"[merge] REFUSING {run}: cannot verify seed provenance for "
            f"registry archs {sorted(registry_archs)} "
            f"({type(exc).__name__}: {exc})")
    for arch, expected in sorted(registry_archs.items()):
        got = resolve_seed_xc(cfg.inputs, arch)
        if got != expected:
            raise SystemExit(
                f"[merge] REFUSING {run}: arch {arch} resolves seed "
                f"{got!r} but the rung-baseline policy demands "
                f"{expected!r} -- a mis-seeded arm cannot enter the "
                "grouped figures")


def build_view(results_root: Path, out_dir: Path) -> dict:
    """(Re)build the merged view; returns {arm_base: (run_name, n_specs)}.

    Alongside the renumbered spec symlinks a merged ``manifest.json`` is
    composed from the arms' own manifests -- the figure collectors join
    rows against it for the arch/subset labels, so without it every row
    would carry ``arch=None`` and the merged figures would be empty.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ck_out = out_dir / "checkpoints"
    ck_out.mkdir(parents=True)

    report: dict = {}
    merged_specs = []
    seen_cells: dict = {}
    idx = 0
    for base in ARM_BASES:
        run = newest_run(results_root / base)
        if run is None:
            report[base] = (None, 0)
            continue
        entries = _arm_manifest_entries(run)
        spec_dirs = sorted((run / "checkpoints").glob("spec_*")) \
            if (run / "checkpoints").is_dir() else []
        unlabeled = [int(sd.name.split("_", 1)[1]) for sd in spec_dirs
                     if int(sd.name.split("_", 1)[1]) not in entries]
        # A spec dir with no manifest entry merges with no arch/subset
        # labels, no duplicate-cell key, and no seed validation -- exactly
        # how a mis-seeded arm could slip in unlabeled. That state must be
        # loud whenever specs are actually present; an arm that has not
        # materialized anything yet only gets a low-key note, so the loud
        # message stays meaningful.
        if not entries:
            if spec_dirs:
                print(f"[merge] WARNING: {base} {run.name} has no usable "
                      "manifest entries (missing, unreadable, or empty "
                      "manifest.json) -- seed-provenance guard SKIPPED for "
                      "this arm; its specs merge without arch/subset labels "
                      "or duplicate-cell protection")
            else:
                print(f"[merge] note: {base} {run.name} has no manifest yet "
                      "-- seed-provenance validation deferred until the arm "
                      "materializes")
        elif unlabeled:
            shown = ", ".join(str(i) for i in unlabeled[:8])
            more = ("" if len(unlabeled) <= 8
                    else f", +{len(unlabeled) - 8} more")
            print(f"[merge] WARNING: {base} {run.name} manifest lacks "
                  f"entries for {len(unlabeled)} on-disk spec dir(s) "
                  f"(indices {shown}{more}) -- they merge without arch/"
                  "subset labels, duplicate-cell protection, or seed "
                  "validation")
        _validate_arm_seed_policy(
            run, {(e.get("cell") or {}).get("arch") for e in entries.values()})
        for sd in spec_dirs:
            orig_idx = int(sd.name.split("_", 1)[1])
            entry = entries.get(orig_idx, {})
            cell = entry.get("cell") or {}
            cell_key = (cell.get("arch"), cell.get("subset_size"))
            if all(v is not None for v in cell_key):
                owner = seen_cells.get(cell_key)
                if owner is not None and owner != base:
                    raise SystemExit(
                        f"[merge] REFUSING: cell {cell_key} arrives from "
                        f"both {owner} and {base} -- a duplicate cell is a "
                        "double-count, never a merge")
                seen_cells[cell_key] = base
            (ck_out / f"spec_{idx:04d}").symlink_to(sd.resolve())
            rec = {"index": idx, "cell": cell,
                   "arm": base, "arm_run": run.name, "arm_index": orig_idx}
            # Carry the integrity provenance through: spec_file/sha256 are
            # the expected hash record, and arm_run names the source run so
            # a verifier can resolve <arm>/runs/<arm_run>/specs/<spec_file>
            # without guessing the newest run (which moves as pulls land).
            for k in ("spec_file", "sha256"):
                if entry.get(k) is not None:
                    rec[k] = entry[k]
            merged_specs.append(rec)
            idx += 1
        report[base] = (run.name, len(spec_dirs))
        # Keep one provenance breadcrumb per arm; the eval count separates
        # finished cells from empty/mid-training spec dirs.
        n_eval = sum(
            1 for sd in spec_dirs
            if (sd / "eval_holdout" / "per_molecule.json").is_file()
            or (sd / "eval_holdout" / "per_reaction.json").is_file())
        with open(out_dir / "MERGED_ARMS.txt", "a") as f:
            f.write(f"{base}\t{run.name}\t{len(spec_dirs)} specs"
                    f"\t{n_eval} eval_holdout\n")
        # Propagate the run-identity + SCAN-cache files the figure loaders
        # resolve against the run-dir root (the arms share one production
        # identity, so the first copy wins): without resolved_config.yaml the
        # basis label degrades to "unknown" and the SCAN reference lines never
        # draw on the merged figures. The view is wiped on every rebuild, so
        # these must be copied here rather than dropped in by hand. Later arms
        # are checked against the view's identity -- a mismatched arm would
        # make the propagated caches/labels silently wrong for it.
        arm_id = _config_identity(run / "resolved_config.yaml")
        view_id = _config_identity(out_dir / "resolved_config.yaml")
        if view_id is not None and arm_id is not None and arm_id != view_id:
            print(f"[merge] WARNING: {base} {run.name} production identity "
                  f"{arm_id} differs from the view's {view_id} -- the "
                  "propagated SCAN caches/labels may not apply to this arm")
        for src in [run / "resolved_config.yaml",
                    *sorted(run.glob("scan_pool_*.json"))]:
            dst = out_dir / src.name
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
    (out_dir / "manifest.json").write_text(json.dumps(
        {"n_specs": idx, "specs": merged_specs,
         "merged_from": [b for b, (r, _n) in report.items() if r]}, indent=1))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-root", default=str(DEFAULT_ROOT))
    p.add_argument("--out", default=None,
                   help="view dir (default: <results-root>/merged_v4_arms)")
    args = p.parse_args(argv)
    root = Path(args.results_root)
    out = Path(args.out) if args.out else root / "merged_v4_arms"
    report = build_view(root, out)
    total = 0
    for base, (run, n) in report.items():
        print(f"[merge] {base:<28} {run or '(not pulled yet)':<28} {n} specs")
        total += n
    print(f"[merge] view: {out}  ({total} specs)")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
