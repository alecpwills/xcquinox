#!/usr/bin/env python
"""Rebuild the pool JSON caches from the GMTKN55 source data.

Run this once after first adding the GMTKN55 source data to the repo, and
again any time the source `.res` files or per-species `coord` / `struc.xyz`
files change, or the composition files under data/slim/ or the diet lists
under data/dietgmtkn55-150/.

The output JSONs land under xcquinox/pipeline/data/:
  - bh76_full_pool.json   (76 reactions, ~50 species)
  - w411_full_pool.json   (140 atomizations, ~150 species)
  - slim05_pool.json, slim16_pool.json   (the Slim sets, with slim05's
    pretraining draw)
  - diet150_pool.json     (the 150-reaction diet set with its subset weights)
  - diet150_overlap.json, bh76_overlap.json, w411_overlap.json   (the
    per-reaction overlap reports against the Slim and DFS training sets)

Both are committed to the repo so the cluster eval task can load them
without re-parsing the source. The JSON structure is the same dict shape
that ``xcquinox.pipeline.full_benchmark_pools.load_full_{bh76,w411}`` returns
at runtime (species + reactions), so a diff after a re-run reveals exactly
which species or reactions changed.

Usage:
    python tools/rebuild_full_benchmark_pools.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the repo root is importable even when run as a script from outside.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from xcquinox.pipeline.full_benchmark_pools import (  # noqa: E402
    BH76_JSON_PATH,
    W411_JSON_PATH,
    build_bh76_pool_dict,
    build_w411_pool_dict,
)
from xcquinox.pipeline.gmtkn55_sets import (  # noqa: E402
    DIET150_JSON_PATH,
    OVERLAP_JSON_PATHS,
    POOL_SETS,
    SLIM_JSON_PATHS,
    build_diet150_pool_dict,
    build_overlap_reports,
    build_slim_pool_dict,
)


def _dump_atomic(path: Path, data: dict) -> None:
    """Write JSON to ``path`` atomically (write to .tmp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def main() -> int:
    print("Rebuilding GMTKN55-BH76 cache from .res + per-species files ...",
          flush=True)
    bh76 = build_bh76_pool_dict()
    print(f"  parsed {len(bh76['reactions'])} BH76 reactions over "
          f"{len(bh76['species'])} species", flush=True)
    _dump_atomic(BH76_JSON_PATH, bh76)
    print(f"  wrote {BH76_JSON_PATH}", flush=True)

    print("Rebuilding GMTKN55-W4-11 cache ...", flush=True)
    w411 = build_w411_pool_dict()
    print(f"  parsed {len(w411['reactions'])} W4-11 reactions over "
          f"{len(w411['species'])} species", flush=True)
    _dump_atomic(W411_JSON_PATH, w411)
    print(f"  wrote {W411_JSON_PATH}", flush=True)

    for name in POOL_SETS:
        print(f"Rebuilding the {name} cache from the composition file and the "
              "checkout ...", flush=True)
        pool = build_slim_pool_dict(name)
        print(f"  parsed {len(pool['reactions'])} reactions over "
              f"{len(pool['species'])} species; {len(pool['molecules'])} "
              "molecules under the published rule", flush=True)
        _dump_atomic(SLIM_JSON_PATHS[name], pool)
        print(f"  wrote {SLIM_JSON_PATHS[name]}", flush=True)

    print("Rebuilding the diet150 cache from its element list and its "
          "subset list ...", flush=True)
    diet = build_diet150_pool_dict()
    print(f"  parsed {len(diet['reactions'])} reactions over "
          f"{len(diet['species'])} species", flush=True)
    _dump_atomic(DIET150_JSON_PATH, diet)
    print(f"  wrote {DIET150_JSON_PATH}", flush=True)

    print("Rebuilding the overlap reports ...", flush=True)
    reports = build_overlap_reports()
    for key, path in OVERLAP_JSON_PATHS.items():
        _dump_atomic(path, reports[key])
        print(f"  wrote {path} ({len(reports[key])} reactions)", flush=True)

    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
