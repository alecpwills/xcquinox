#!/usr/bin/env python
"""Rebuild the BH76 + W4-11 JSON caches from the GMTKN55 source data.

Run this once after first adding the GMTKN55 source data to the repo, and
again any time the source `.res` files or per-species `coord` / `struc.xyz`
files change.

The output JSONs land at:
  - xcquinox/alec/data/bh76_full_pool.json   (76 reactions, ~50 species)
  - xcquinox/alec/data/w411_full_pool.json   (140 atomizations, ~150 species)

Both are committed to the repo so the cluster eval task can load them
without re-parsing the source. The JSON structure is the same dict shape
that ``xcquinox.alec.full_benchmark_pools.load_full_{bh76,w411}`` returns
at runtime (species + reactions), so a diff after a re-run reveals exactly
which species or reactions changed.

Usage:
    python scripts/rebuild_full_benchmark_pools.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the repo root is importable even when run as a script from outside.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from xcquinox.alec.full_benchmark_pools import (  # noqa: E402
    BH76_JSON_PATH,
    W411_JSON_PATH,
    build_bh76_pool_dict,
    build_w411_pool_dict,
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

    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
