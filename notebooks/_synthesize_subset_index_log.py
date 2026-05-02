"""Synthesize subset_index_log.json from the on-disk subset.traj files.

Usage:
    cd /home/awills/Documents/Research/xcquinox
    python notebooks/_synthesize_subset_index_log.py

When to use:
    The earlier subset-generation cell did not persist the ledger after each
    (metric, r) pair (only at the very end of all 20).  If a long run is
    killed mid-way, the subset.traj files for completed pairs are on disk
    but no ledger entry exists — so the new resume-aware cell would re-run
    those completed enumerations.

What it does:
    1. Walks notebooks/checkpoints_step7/{l2,jsd}/binNN[w]/.../subset.traj
    2. For each unique (metric, r, aug) triple, reads ONE subset.traj (both
       solvers under a given (metric, r, aug) carry identical content)
    3. Identifies the chosen AE-pool indices via at.info["dfs_hill"]
    4. Re-computes the L2/JSD metric value against the cached reference
       histogram (notebooks/checkpoints_step7/dfs_pool_full_hist/reference.npz)
       using the cached per-species descriptors
       (notebooks/checkpoints_step7/subset_descriptors/<idx>_<species>.npz).
       Re-computation is fast (no enumeration; ~1 binning + 1 metric eval
       per pair).  No SCF, no JAX, no GPU required.
    5. Writes notebooks/checkpoints_step7/subset_index_log.json with
       per-pair entries: chosen_indices, metric_value, atom_set, tag.

Idempotent: if a ledger already exists, entries from this synthesis OVERWRITE
matching keys (metric/r/aug) — the synthesized values are recomputed exactly
the way the cell would compute them, so this stays consistent.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import numpy as np
from ase.io import read as ase_read

REPO = Path(__file__).resolve().parents[1]
STEP7_ROOT = REPO / "notebooks" / "checkpoints_step7"
DESCRIPTOR_CACHE = STEP7_ROOT / "subset_descriptors"
REF_HIST_PATH = STEP7_ROOT / "dfs_pool_full_hist" / "reference.npz"
LEDGER_PATH = STEP7_ROOT / "subset_index_log.json"

ARCH_NAME = "deep_combined_attn"
LOSS_NAME = "L5_gradnorm_vxc_step7"


def _hill_to_pool_idx(pool_ae) -> dict[str, int]:
    """Build {dfs_hill -> pool index} so we can map subset.traj entries back."""
    out = {}
    for i, a in enumerate(pool_ae):
        hill = a.info.get("dfs_hill") or a.get_chemical_formula()
        out[hill] = i
    return out


def _load_descriptor(idx: int) -> dict:
    """Load cached PySCF descriptors for pool entry `idx`."""
    matches = sorted(DESCRIPTOR_CACHE.glob(f"{idx}_*.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No cached descriptor file matching {DESCRIPTOR_CACHE}/{idx}_*.npz"
        )
    z = np.load(matches[0])
    return {k: np.asarray(z[k])
            for k in ("rho_third", "s", "alpha", "weights")}


def _compute_metric_value(chosen_indices, ref_npz, metric: str) -> float:
    """Replay the cell's metric computation: bin the chosen subset's
    descriptors against the reference edges and call metric_l2/metric_jsd."""
    from xcquinox.alec.subset_selection import metric_l2, metric_jsd, LOG_REGULARIZER
    parts = {"rho_third": [], "s": [], "alpha": [], "weights": []}
    for idx in chosen_indices:
        d = _load_descriptor(idx)
        for k in parts:
            parts[k].append(d[k])
    cat = {k: np.concatenate(parts[k]) for k in parts}
    h_ref = {
        "rho_third": np.asarray(ref_npz["h_ref_rho"]),
        "s":         np.asarray(ref_npz["h_ref_s"]),
        "alpha":     np.asarray(ref_npz["h_ref_alpha"]),
    }
    edges = {
        "rho_third": np.asarray(ref_npz["e_rho"]),
        "s":         np.asarray(ref_npz["e_s"]),
        "alpha":     np.asarray(ref_npz["e_alpha"]),
    }
    h_cand = {}
    for k, ek in (("rho_third", "rho_third"),
                  ("s", "s"),
                  ("alpha", "alpha")):
        log_x = np.log10(cat[k] + LOG_REGULARIZER)
        h, _ = np.histogram(log_x, bins=edges[ek],
                            weights=cat["weights"], density=True)
        h_cand[k] = h
    if metric == "l2":
        return float(metric_l2(h_ref, h_cand))
    if metric == "jsd":
        return float(metric_jsd(h_ref, h_cand))
    raise ValueError(f"unknown metric {metric!r}")


def _is_hbpt_atoms(atoms) -> bool:
    """HBPT entries are 6-atom water dimers (H4O2)."""
    syms = sorted(atoms.get_chemical_symbols())
    return Counter(syms) == Counter(["H", "H", "H", "H", "O", "O"])


def _parse_subset_traj(traj_path: Path, hill_to_idx: dict[str, int]):
    """Return (chosen_indices_sorted, atom_set_sorted, has_hbpt: bool)."""
    atoms_list = ase_read(str(traj_path), ":")
    chosen: list[int] = []
    atom_syms_set: set[str] = set()
    has_hbpt = False
    for at in atoms_list:
        hill = at.info.get("dfs_hill")
        if hill is not None and hill in hill_to_idx:
            chosen.append(hill_to_idx[hill])
            continue
        if _is_hbpt_atoms(at):
            has_hbpt = True
            continue
        # Otherwise: single-atom reference (1 atom)
        if len(at) == 1:
            atom_syms_set.add(at.get_chemical_symbols()[0])
        else:
            # Unknown entry — skip but warn
            print(f"  [warn] {traj_path}: unrecognized entry "
                  f"(formula={at.get_chemical_formula()}, "
                  f"info_keys={list(at.info.keys())})")
    return sorted(set(chosen)), sorted(atom_syms_set), has_hbpt


def main() -> int:
    if not REF_HIST_PATH.exists():
        print(f"ERROR: reference histogram not found at {REF_HIST_PATH}")
        return 1
    if not DESCRIPTOR_CACHE.exists():
        print(f"ERROR: descriptor cache not found at {DESCRIPTOR_CACHE}")
        return 1

    from xcquinox.alec import dfs_pool
    print("Building DFS pool to recover Hill-formula -> pool-index mapping...")
    pool = dfs_pool.build_dfs_pool()
    hill_to_idx = _hill_to_pool_idx(pool["ae_molecules"])
    print(f"  pool size: {len(pool['ae_molecules'])} AE molecules")
    print(f"  hill -> idx: {hill_to_idx}")
    ref_npz = np.load(REF_HIST_PATH)

    # Load existing ledger (if any) to merge with synthesized entries.
    existing: dict = {}
    if LEDGER_PATH.exists():
        existing = json.loads(LEDGER_PATH.read_text())
        print(f"Found existing ledger with {len(existing)} entries; "
              f"will merge / overwrite.")

    out: dict = dict(existing)
    n_synth = 0
    n_skip = 0

    for traj_path in sorted(STEP7_ROOT.rglob("subset.traj")):
        # Path layout: STEP7_ROOT/<metric>/<tag>/<arch>/<loss>/<solver>/subset.traj
        rel = traj_path.relative_to(STEP7_ROOT)
        parts = rel.parts
        if len(parts) != 6 or parts[2] != ARCH_NAME or parts[3] != LOSS_NAME:
            continue
        metric, tag, _arch, _loss, _solver, _ = parts
        if metric not in ("l2", "jsd"):
            continue
        if not tag.startswith("bin"):
            continue
        try:
            r_str = tag[3:].rstrip("w")
            r = int(r_str)
        except ValueError:
            continue
        aug = tag.endswith("w")
        slashkey = f"{metric}/{r}/{aug}"

        # If we already have this triple synthesized in this run, skip.
        # (Both solvers under a (metric, tag) point at the same content.)
        if slashkey in out and out[slashkey].get("_synth_done"):
            continue

        chosen, atom_set, has_hbpt = _parse_subset_traj(traj_path, hill_to_idx)
        if len(chosen) != r:
            print(f"  [warn] {slashkey}: subset.traj has {len(chosen)} chosen "
                  f"AE entries but tag implies r={r} — skipping")
            n_skip += 1
            continue
        if has_hbpt != aug:
            print(f"  [warn] {slashkey}: subset.traj HBPT presence "
                  f"({has_hbpt}) does not match tag aug={aug}; trusting tag")

        try:
            mval = _compute_metric_value(chosen, ref_npz, metric)
        except FileNotFoundError as e:
            print(f"  [warn] {slashkey}: descriptor cache miss -> {e}; skip")
            n_skip += 1
            continue

        out[slashkey] = {
            "chosen_indices": chosen,
            "metric_value": mval,
            "atom_set": atom_set,
            "tag": tag,
            "_synth_done": True,  # internal marker for de-dup across solvers
        }
        n_synth += 1
        print(f"  [{metric}/r={r:>2d}/aug={aug}] chosen={chosen} "
              f"metric={mval:.6e}  atoms={atom_set}")

    # Strip the internal marker before writing.
    for v in out.values():
        v.pop("_synth_done", None)

    LEDGER_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(out)} ledger entries to {LEDGER_PATH}")
    print(f"  newly synthesized: {n_synth}  skipped: {n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
