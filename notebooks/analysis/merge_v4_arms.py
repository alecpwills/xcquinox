"""Build a merged VIEW of the v4 campaign arms for cross-arm figures.

The v4 re-sweep runs as separate arms (meta-GGA on 96-core, GGA-based on
40-core, the mgga stacking completions later), each with its own run dir.
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

ARM_BASES = ("dfs6311_grid3_v4", "dfs6311_grid3_v4gga", "dfs6311_grid3_v4mgga2")
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


def _arm_manifest_cells(run: Path) -> dict:
    """{original_index: cell} from the arm's manifest.json (empty if absent)."""
    mpath = run / "manifest.json"
    if not mpath.is_file():
        return {}
    try:
        manifest = json.loads(mpath.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {e["index"]: (e.get("cell") or {})
            for e in manifest.get("specs", []) if isinstance(e.get("index"), int)}


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
    idx = 0
    for base in ARM_BASES:
        run = newest_run(results_root / base)
        if run is None:
            report[base] = (None, 0)
            continue
        cells = _arm_manifest_cells(run)
        spec_dirs = sorted((run / "checkpoints").glob("spec_*")) \
            if (run / "checkpoints").is_dir() else []
        for sd in spec_dirs:
            orig_idx = int(sd.name.split("_", 1)[1])
            (ck_out / f"spec_{idx:04d}").symlink_to(sd.resolve())
            merged_specs.append({"index": idx,
                                 "cell": cells.get(orig_idx, {}),
                                 "arm": base, "arm_index": orig_idx})
            idx += 1
        report[base] = (run.name, len(spec_dirs))
        # Keep one provenance breadcrumb per arm.
        with open(out_dir / "MERGED_ARMS.txt", "a") as f:
            f.write(f"{base}\t{run.name}\t{len(spec_dirs)} specs\n")
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
