"""Read-only probe: is the orientation lock recorded on the CH/NO references?

Context: the dfs6311 density channel never decreases during training, and in
every cell that contains them the open-shell radicals CH and NO own 68-98% of
it with an error the functional cannot close (model-free PBE reproduces CH's
error to within 0.4%). The mechanism is a degenerate-component mismatch: the
model-free PBE density error is bit-identical across specs for all 20
closed-shell species and varies up to 23x for exactly the three 2-Pi radicals
CH, HO and NO -- at constant total energy, which is the signature of the SCF
settling on a different member of the degenerate manifold than the stored
reference did. Full analysis: notebooks/analysis/DENSITY_DIAGNOSIS.md.

What this probe decides: the orientation lock is threaded correctly by
construction (run-level ``inputs.orientation_lock_strength`` is authoritative,
reaches the training SolverConfig, and both sides call the same operator), and
``data.py`` RAISES when a reference's stamped lock disagrees with the
consumer's. That guard did not fire on the production run, so a stamped
MISMATCH is already excluded. But the guard fires only when the reference
carries the key -- a reference written before the stamp existed has no key and
passes silently. Every reference in the local def2-svp cache is in exactly
that state, so the question is whether the 6311 CH/NO references are too.

The decisive field is therefore whether ``orientation_lock_strength`` is
PRESENT, not what its value is.

Usage (login node, read-only, seconds; activate the parity env first so numpy
is importable):

    source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
    conda activate /gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
    python /gpfs/projects/FernandezGroup/Alec/xcquinox/hpcjobs/dfs6311_lock_stamp_probe.py

Optional positional args override the reference directories.

Reading the result (CH and NO are TRAINING species, so the
``external_refs_dfs_*`` directory is the one that matters):

  * LOCK KEY ABSENT on CH/NO -- pre-stamp references; the consumer guard was
    structurally blind to them and the run trained a locked SCF against
    unlocked references. Remedy: regenerate those species' references, and
    make a MISSING key an error whenever a nonzero lock is configured.
  * PRESENT and equal on CH, NO and HO -- the lock was applied consistently on
    both sides, regeneration will not help, and the cause is either that this
    strength does not split CH's/NO's manifolds the way it splits OH's or that
    CH's CCSD reference is not a single-component density. Remedy then is
    per-species density down-weighting.
  * PRESENT but unequal -- should be impossible; the training-time guard would
    have aborted the run.
"""

import os
import sys
import time

import numpy as np

DEFAULT_REFS = (
    "/gpfs/scratch/awills/external_refs_dfs_6311ppg3df2pd_g3",
    "/gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3",
)
# CH and NO are the offenders; HO is the 2-Pi control that agrees with its
# reference; N2 is a closed-shell control that cannot be affected at all.
SPECIES = ("CH", "NO", "HO", "N2")
LOCK_KEY = "orientation_lock_strength"
STAMP_KEYS = ("basis_used", "grid_level_used", "ref_density_method",
              "oep_converged", "oep_n_electrons", "oep_density_error")


def _scalar(npz, key):
    """Scalar value of ``key``, or None when absent."""
    if key not in npz.files:
        return None
    arr = np.asarray(npz[key])
    return arr.item() if arr.shape == () else arr


def probe_dir(base):
    """Print one line per species; return {species: lock_value_or_None}."""
    print(f"\n=== {os.path.basename(base)}")
    if not os.path.isdir(base):
        print("    (directory absent)")
        return {}
    found = {}
    for name in SPECIES:
        path = os.path.join(base, f"{name}.npz")
        if not os.path.isfile(path):
            print(f"    {name:>3s}: file absent")
            continue
        # Our own generated reference cache; allow_pickle is needed only
        # because the string stamps are stored as object arrays.
        with np.load(path, allow_pickle=True) as npz:
            present = LOCK_KEY in npz.files
            lock = float(np.asarray(npz[LOCK_KEY]).item()) if present else None
            stamps = {k: _scalar(npz, k) for k in STAMP_KEYS}
        found[name] = lock
        verdict = (f"LOCK KEY PRESENT = {lock:g}" if present
                   else "LOCK KEY ABSENT  <-- pre-stamp reference")
        print(f"    {name:>3s}: {verdict}")
        print(f"         basis={stamps['basis_used']} "
              f"grid={stamps['grid_level_used']} "
              f"method={stamps['ref_density_method']}")
        print(f"         oep_converged={stamps['oep_converged']} "
              f"n_electrons={stamps['oep_n_electrons']} "
              f"density_error={stamps['oep_density_error']}")
        print(f"         mtime={time.ctime(os.stat(path).st_mtime)}")
    return found


def verdict(found):
    """Name the branch the training-reference directory lands in."""
    print("\n=== VERDICT (training references; CH and NO are training species)")
    offenders = {k: v for k, v in found.items() if k in ("CH", "NO")}
    control = found.get("HO")
    if not offenders:
        print("    CH and NO absent from this directory -- inconclusive.")
        return
    missing = [k for k, v in offenders.items() if v is None]
    if missing:
        print(f"    {', '.join(missing)} carry NO lock stamp: these are "
              "pre-stamp references, invisible to the training-time guard.")
        print("    => a locked SCF was trained against unlocked reference(s).")
        print("    => REMEDY: regenerate those species' references at the "
              "configured lock, and treat a missing key as an error whenever "
              "the consumer resolves a nonzero lock.")
        return
    values = set(offenders.values()) | ({control} if control is not None
                                        else set())
    if len(values) == 1:
        print(f"    CH, NO and HO all stamped {offenders['CH']:g} -- the lock "
              "was applied consistently on both sides.")
        print("    => regeneration will NOT help; the residual mismatch is "
              "either species-dependent lock efficacy or a CCSD reference "
              "that is not single-component.")
        print("    => REMEDY: per-species density down-weighting (lambda_n "
              "x 0.01) on CH and NO.")
    else:
        print(f"    Stamps disagree: {offenders} vs HO {control}. This should "
              "be impossible -- the training-time guard in data.py raises on a "
              "stamped mismatch, so the run could not have completed. "
              "Re-check which reference directory the run actually consumed.")


def main(argv):
    # The verdict interprets a LIVE reference cache (is the deployed state
    # defective?). Against a freshly regenerated staging directory it is a
    # false read by construction -- everything there is stamped because it was
    # just written -- so callers verifying staged output pass --no-verdict and
    # rely on their own gate instead.
    args = [a for a in argv[1:] if a != "--no-verdict"]
    show_verdict = "--no-verdict" not in argv[1:]
    bases = tuple(args) or DEFAULT_REFS
    training = {}
    for i, base in enumerate(bases):
        found = probe_dir(base)
        if i == 0:
            training = found
    if show_verdict:
        verdict(training)
    else:
        print("\n(verdict suppressed: --no-verdict; this listing is a staged "
              "verification, not a diagnosis of the deployed cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
