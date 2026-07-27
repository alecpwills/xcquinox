"""Read-only provenance probe for the dfs6311 benchmark-reference cache.

Context: the held-out eval drops the species c2 with
    external rho_ref_grid shape (26840,) does not match rho_grid shape (26568,)
in every completed task of run_20260723T161502Z. The mismatch was traced to the
SCF-escalation path in external_refs.run_scf_with_cache: when the plain kernel
fails to converge (c2 is the only pool species where it does), the retry
mean-field's grid is density-pruned on the non-converged DIIS density instead
of the minao initial guess, and that drifted grid is what the reference npz
stores. 26840 is reproduced exactly by pruning on the 50-cycle non-converged
density with density_fit off and orientation lock 3e-5 (an early-era, pre-DF
invocation); the eval-side 26568 is the guess-pruned count in pyscf 2.9.0 and
2.11.0 alike.

This script confirms the file-side story before any cache deletion. It only
stats files and reads npz headers/arrays; nothing is written or modified.

Usage (login node is fine; runtime is seconds):
    python dfs6311_c2_ref_probe.py [refs_dir]
refs_dir defaults to the dfs6311 benchmark-refs cache on scratch.

Expected if the diagnosis holds:
  - c2.npz  : rho_ref_grid (26840,), orientation_lock_strength 3e-05 stamped,
              mtime older than (or from a different wave than) the controls
  - n2.npz  : rho_ref_grid (26616,)   [plain kernel converges -> guess-pruned]
  - f2.npz  : rho_ref_grid (26568,)   [plain kernel converges -> guess-pruned]
  - _intermediates/ holds c2 caches under the non-DF name
    c2_g3_b6-311++g(3df,2pd)_ol3e-05_{scf,ccsd}.npz (no _df_ infix)
"""

import glob
import os
import sys
import time

import numpy as np

DEFAULT_REFS = "/gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3"
SPECIES = ("c2", "n2", "f2", "co")
STAMP_KEYS = ("grid_level_used", "basis_used", "orientation_lock_strength",
              "ref_density_method")


def _mtime(path):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.stat(path).st_mtime))


def main(argv):
    refs = argv[1] if len(argv) > 1 else DEFAULT_REFS
    if not os.path.isdir(refs):
        print("refs dir not found: %s" % refs)
        return 1

    print("refs dir: %s" % refs)
    print()
    print("== final npz mtimes ==")
    for name in SPECIES:
        path = os.path.join(refs, "%s.npz" % name)
        if os.path.isfile(path):
            print("  %-8s %s  %10d bytes  %s" %
                  (name, _mtime(path), os.path.getsize(path), path))
        else:
            print("  %-8s MISSING  %s" % (name, path))

    print()
    print("== c2 intermediate caches (name encodes basis/DF/lock identity) ==")
    inter = sorted(glob.glob(os.path.join(refs, "_intermediates", "c2_*.npz")))
    if not inter:
        print("  (none found under %s)" % os.path.join(refs, "_intermediates"))
    for path in inter:
        print("  %s  %s" % (_mtime(path), os.path.basename(path)))

    print()
    print("== npz contents ==")
    for name in SPECIES[:3]:
        path = os.path.join(refs, "%s.npz" % name)
        if not os.path.isfile(path):
            print("  %-4s MISSING" % name)
            continue
        with np.load(path, allow_pickle=False) as z:
            keys = sorted(z.files)
            print("  %-4s keys: %s" % (name, keys))
            if "rho_ref_grid" in keys:
                print("       rho_ref_grid shape: %s" % (z["rho_ref_grid"].shape,))
            for k in STAMP_KEYS:
                if k in keys:
                    val = z[k]
                    val = val.item() if getattr(val, "ndim", 1) == 0 else val
                    print("       %-26s %s" % (k + ":", val))
                else:
                    print("       %-26s ABSENT" % (k + ":"))

    print()
    print("expected: c2 (26840,) lock 3e-05 non-DF intermediates; "
          "n2 (26616,); f2 (26568,)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
