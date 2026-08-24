#!/usr/bin/env python
"""Measure the per-system energy-term weight of the pretraining objective.

The pretraining loss carries two terms: the point-wise enhancement-factor
residual and a per-system energy term
``mean_s (E_xc^NN_s - E_xc^parent_s)^2`` in Hartree^2, whose weight
``pretrain.energy_term_weight`` is dimensionful (inverse Hartree^2). The right
value of that weight is the one that buys the Section 3.3 certificate's
``tol_atom = 1.0 mHa`` without destroying the point-wise fit, and it is not
derivable: it is measured. This script performs that measurement -- one
pretraining run per (architecture, weight) cell -- and writes the table the
campaign-v6 configurations quote.

Two quantities are reported per cell, because they are not the same test:

* ``rms_dE_xc``  = ``sqrt(<dE_x^2>_s + <dE_c^2>_s)``, the RMS per-system XC
  energy error. This is what the pretraining metadata's
  ``energy_term_{x,c}_final`` pair records, and it is what the objective
  itself minimizes.
* ``max_dE_xc``  = ``max_s |dE_x,s + dE_c,s|``, the LARGEST per-system XC
  energy error. This is the certificate's own quantity: Section 3.3 gates on
  ``max |dE_xc| per atom <= tol_atom``, a maximum, not a mean. A set whose RMS
  clears the tolerance can still carry one system that does not.

Both are computed here from the per-system residuals rather than read off the
metadata, for two reasons. The metadata records only the MEAN of the squares,
from which no maximum can be recovered; and at ``energy_term_weight = 0`` the
loss short-circuits before the term is evaluated, so the recorded
``energy_term_{x,c}_final`` are identically zero and the weight-0 baseline row
-- the row every ratio in the table is taken against -- cannot be read from
them at all. The reconstruction is checked against the recorded mean on every
cell where the recorded mean is real (weight > 0); a disagreement above
``--recon-rtol`` is fatal.

Identity: the sweep runs at ``--basis``/``--grid-level`` with the pretraining
set of spec Section 7 (the DFS inventory in its entirety plus every atom of
the BH76 / W4-11 pools), the per-channel exchange footing, and the
orientation lock the data generator locks degenerate open shells with. The
GGA-rung architectures pretrain against the PBE parent and the meta-GGA rung
against SCAN, because the certificate is per rung and a meta-GGA network fit
on a PBE density is fit to a density its own SCF never visits.

The default identity is def2-svp / grid level 3. Level 3 is a floor, not a
preference: the generator refuses a degenerate open-shell atom below it (the
quadrature does not resolve the P term, so the file would not have the
identity its manifest claims), and the set contains O, C, N and every other
open p-shell pool atom.

Usage (cluster; see hpcjobs/probe_pretrain_energy_weight.sbatch):

    python hpcjobs/probe_pretrain_energy_weight.py \
        --data-dir /gpfs/scratch/awills/pretrain_energy_weight/data \
        --out      /gpfs/scratch/awills/pretrain_energy_weight/table.json

Usage (local identity check, seconds):

    python hpcjobs/probe_pretrain_energy_weight.py --smoke \
        --data-dir <tmp>/data --out <tmp>/table.json

Exit codes: 0 -- a weight cleared the gate on every architecture; 2 -- the
sweep completed and no weight did (a finding: the table is still written);
1 -- the sweep itself failed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# x64 is required before JAX is imported by anything below: the per-system
# energies are O(10 Ha) sums whose differences are read at the 1e-6 Ha level.
os.environ.setdefault("JAX_ENABLE_X64", "1")


# --------------------------------------------------------------------------- #
# Constants. Every one is a quoted requirement, not a tuned number.
# --------------------------------------------------------------------------- #

#: The certificate's per-atom tolerance on |E_xc^NN - E_xc^parent|, in mHa
#: (SPEC_pretrain_fidelity_program.md Sections 3.3 and 7: "tol_atom = 1.0 mHa").
TOL_ATOM_MHA = 1.0

#: Fraction of ``TOL_ATOM_MHA`` the sweep requires at its own reduced
#: identity. The sweep is not the certificate: it measures the pretraining
#: rows at the sweep's basis and grid, while the certificate measures the
#: production identity through the deployment energy path. Half the tolerance
#: leaves the difference between the two a factor of two of room.
MARGIN_FRACTION = 0.5

#: Largest factor by which either point-wise loss may rise from its weight-0
#: value. The energy term is a constraint added to the point-wise objective;
#: a weight that buys the energy by an order of magnitude of point-wise fit
#: has replaced the objective rather than constrained it.
POINTWISE_FACTOR = 3.0

DEFAULT_ARCHS = ("deep_3x16", "deep_cusp_3x16", "deep_rung35_3x16",
                 "deep_mgga_3x16")
DEFAULT_WEIGHTS = (0.0, 0.1, 1.0, 10.0, 100.0)
DEFAULT_BASIS = "def2-svp"
DEFAULT_GRID_LEVEL = 3
DEFAULT_N_STEPS = 1000
DEFAULT_SEED = 42

#: The smoke identity: two systems (one closed shell, one open) at STO-3G on
#: grid level 1, five optimizer steps. Coarse enough that the generator's
#: irreproducible-degenerate refusal has to be waived, which is why the flag
#: is passed only here. It measures the plumbing, never a weight.
SMOKE_BASIS = "sto-3g"
SMOKE_GRID_LEVEL = 1
SMOKE_ATOMS = (("He", 0), ("Li", 1))
SMOKE_ARCHS = ("deep_3x16", "deep_mgga_3x16")
SMOKE_WEIGHTS = (0.0, 1.0)
SMOKE_N_STEPS = 5

#: Relative agreement demanded between the per-system reconstruction here and
#: the mean-of-squares the pretraining metadata records. The two evaluate the
#: same expression on the same rows; they differ only in the reduction order
#: of a segmented sum and in whether the zero-weight mesh rows are carried,
#: so the gap is round-off.
DEFAULT_RECON_RTOL = 1.0e-6

_HARTREE_TO_MHA = 1000.0


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def _name_list(text):
    """Comma- and/or whitespace-separated names."""
    out = []
    for chunk in str(text).replace(",", " ").split():
        if chunk:
            out.append(chunk)
    if not out:
        raise argparse.ArgumentTypeError(f"empty list: {text!r}")
    return tuple(out)


def _weight_list(text):
    """Comma- and/or whitespace-separated non-negative finite weights."""
    out = []
    for chunk in _name_list(text):
        try:
            value = float(chunk)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"weight {chunk!r} is not a number")
        if not math.isfinite(value) or value < 0.0:
            raise argparse.ArgumentTypeError(
                f"weight {chunk!r} must be finite and >= 0")
        out.append(value)
    # Deduplicated, ascending: the recommendation rule takes the SMALLEST
    # clearing weight, so the order the table is swept in is the order it is
    # read in.
    return tuple(sorted(set(out)))


def build_parser():
    p = argparse.ArgumentParser(
        prog="probe_pretrain_energy_weight",
        description="Sweep pretrain.energy_term_weight and report the "
                    "per-system XC energy error it buys.")
    p.add_argument("--data-dir", required=True,
                   help="Root for the generated pretraining data. One "
                        "subdirectory per (polarization, parent density) "
                        "pair is created under it and reused across cells.")
    p.add_argument("--out", required=True,
                   help="Path of the JSON table written at the end.")
    p.add_argument("--archs", type=_name_list, default=None,
                   help=f"Architectures to sweep (default: "
                        f"{','.join(DEFAULT_ARCHS)}).")
    p.add_argument("--weights", type=_weight_list, default=None,
                   help=f"Energy-term weights (default: "
                        f"{','.join(repr(w) for w in DEFAULT_WEIGHTS)}).")
    p.add_argument("--n-steps", type=int, default=None,
                   help=f"Optimizer steps per network (default "
                        f"{DEFAULT_N_STEPS}).")
    p.add_argument("--basis", default=None,
                   help=f"Basis set (default {DEFAULT_BASIS!r}).")
    p.add_argument("--grid-level", type=int, default=None,
                   help=f"PySCF grid level (default {DEFAULT_GRID_LEVEL}). "
                        "Below 3 the generator refuses the degenerate "
                        "open-shell atoms of the set unless --smoke waives "
                        "it.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Network initialization seed (default {DEFAULT_SEED}"
                        "). One seed for every cell, so the comparison "
                        "between weights is not a comparison between draws.")
    p.add_argument("--loss-weighting", default="integration",
                   choices=("integration", "unweighted"),
                   help="Point-wise reduction (default: integration, which "
                        "is what the campaign trains with).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny identity used by the tests and for a local "
                        "plumbing check: two systems at STO-3G on grid level "
                        "1, five steps, the irreproducible-degenerate "
                        "refusal waived. Explicit flags still win.")
    p.add_argument("--recon-rtol", type=float, default=DEFAULT_RECON_RTOL,
                   help="Relative agreement demanded between the per-system "
                        "reconstruction and the recorded mean of squares "
                        f"(default {DEFAULT_RECON_RTOL:g}).")
    p.add_argument("--tol-atom-mha", type=float, default=TOL_ATOM_MHA,
                   help=f"Certificate tolerance on max |dE_xc| per system, "
                        f"in mHa (default {TOL_ATOM_MHA}).")
    p.add_argument("--margin-fraction", type=float, default=MARGIN_FRACTION,
                   help=f"Fraction of the tolerance the sweep demands "
                        f"(default {MARGIN_FRACTION}).")
    p.add_argument("--pointwise-factor", type=float,
                   default=POINTWISE_FACTOR,
                   help="Largest admissible rise of either point-wise loss "
                        f"from its weight-0 value (default "
                        f"{POINTWISE_FACTOR}).")
    return p


def parse_args(argv=None):
    """Parse the command line, applying the --smoke identity where the caller
    left a value unset. Returns the namespace with every field resolved, so
    nothing downstream has to know which defaults came from where."""
    args = build_parser().parse_args(argv)
    smoke = bool(args.smoke)
    if args.archs is None:
        args.archs = SMOKE_ARCHS if smoke else DEFAULT_ARCHS
    if args.weights is None:
        args.weights = SMOKE_WEIGHTS if smoke else DEFAULT_WEIGHTS
    if args.n_steps is None:
        args.n_steps = SMOKE_N_STEPS if smoke else DEFAULT_N_STEPS
    if args.basis is None:
        args.basis = SMOKE_BASIS if smoke else DEFAULT_BASIS
    if args.grid_level is None:
        args.grid_level = SMOKE_GRID_LEVEL if smoke else DEFAULT_GRID_LEVEL
    if args.n_steps <= 0:
        build_parser().error(f"--n-steps must be > 0, got {args.n_steps}")
    if args.grid_level < 0:
        build_parser().error(
            f"--grid-level must be >= 0, got {args.grid_level}")
    if not math.isfinite(args.recon_rtol) or args.recon_rtol <= 0:
        build_parser().error(
            f"--recon-rtol must be finite and > 0, got {args.recon_rtol}")
    for field in ("tol_atom_mha", "margin_fraction", "pointwise_factor"):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            build_parser().error(
                f"--{field.replace('_', '-')} must be finite and > 0, got "
                f"{value}")
    return args


# --------------------------------------------------------------------------- #
# The recommendation rule. Pure, so it is testable on a synthetic table.
# --------------------------------------------------------------------------- #

def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def recommend(rows, *, tol_atom_mha=TOL_ATOM_MHA,
              margin_fraction=MARGIN_FRACTION,
              pointwise_factor=POINTWISE_FACTOR):
    """Choose the production weight from a measured table.

    The rule, stated so the choice is reproducible rather than aesthetic:

      take the SMALLEST swept weight for which EVERY architecture's
      ``max_dE_xc_mHa`` is at or below ``margin_fraction * tol_atom_mha``
      AND neither of that architecture's point-wise losses has risen by more
      than ``pointwise_factor`` from its own weight-0 value.

    A weight measured on only part of the architecture set is not eligible:
    the gate is "on every architecture", and an absent cell is not a pass. A
    non-finite entry is a failure, not a missing value -- a diverged fit is
    exactly what the cap exists to reject.

    When nothing clears, the fallback is the weight that MINIMIZES the worst
    ``max_dE_xc_mHa`` among those still inside the point-wise cap (and among
    all of them if the cap is violated everywhere), with the smallest weight
    winning a tie. That is a finding rather than a choice: a pretraining that
    cannot reach the margin on its own rows will not pass the certificate at
    the production identity either.

    Returns a dict; ``cleared`` says whether the returned weight satisfies the
    rule or is the fallback.
    """
    margin_mha = float(margin_fraction) * float(tol_atom_mha)
    archs = sorted({str(r["arch"]) for r in rows})
    weights = sorted({float(r["weight"]) for r in rows})
    by_cell = {(str(r["arch"]), float(r["weight"])): r for r in rows}
    baseline = {a: by_cell.get((a, 0.0)) for a in archs}

    per_weight = []
    for w in weights:
        missing = [a for a in archs if (a, w) not in by_cell]
        worst_max, worst_arch = -1.0, None
        worst_ratio, worst_ratio_arch = -1.0, None
        gate_ok = not missing
        cap_ok = not missing
        no_baseline = []
        for a in archs:
            cell = by_cell.get((a, w))
            if cell is None:
                continue
            value = cell.get("max_dE_xc_mHa")
            if not _finite(value):
                gate_ok = False
                worst_max, worst_arch = float("inf"), a
            else:
                value = float(value)
                if value > worst_max:
                    worst_max, worst_arch = value, a
                if value > margin_mha:
                    gate_ok = False
            base = baseline.get(a)
            if base is None:
                # The cap is a ratio against the weight-0 cell; without it the
                # rise cannot be measured, so the weight cannot be certified.
                cap_ok = False
                no_baseline.append(a)
                continue
            for key in ("final_loss_x", "final_loss_c"):
                num, den = cell.get(key), base.get(key)
                if not _finite(num) or not _finite(den) or float(den) <= 0.0:
                    cap_ok = False
                    ratio = float("inf")
                else:
                    ratio = float(num) / float(den)
                    if ratio > float(pointwise_factor):
                        cap_ok = False
                if ratio > worst_ratio:
                    worst_ratio, worst_ratio_arch = ratio, a
        per_weight.append({
            "weight": w,
            "gate_ok": bool(gate_ok),
            "cap_ok": bool(cap_ok),
            "worst_max_dE_xc_mHa": (None if worst_max < 0 else worst_max),
            "worst_arch": worst_arch,
            "worst_pointwise_ratio": (None if worst_ratio < 0
                                      else worst_ratio),
            "worst_ratio_arch": worst_ratio_arch,
            "missing_archs": missing,
            "archs_without_baseline": no_baseline,
        })

    rule = (f"smallest weight with max |dE_xc| <= {margin_mha:g} mHa "
            f"({margin_fraction:g} x tol_atom = {tol_atom_mha:g} mHa) on "
            f"every architecture and neither point-wise loss above "
            f"{pointwise_factor:g}x its weight-0 value")

    eligible = [e for e in per_weight if e["gate_ok"] and e["cap_ok"]]
    if eligible:
        choice = min(eligible, key=lambda e: e["weight"])
        return {
            "weight": choice["weight"], "cleared": True, "rule": rule,
            "tol_atom_mHa": float(tol_atom_mha),
            "margin_mHa": margin_mha,
            "pointwise_factor": float(pointwise_factor),
            "per_weight": per_weight,
            "reason": (f"weight {choice['weight']:g} clears the gate on all "
                       f"{len(archs)} architectures (worst max |dE_xc| "
                       f"{choice['worst_max_dE_xc_mHa']:.4f} mHa on "
                       f"{choice['worst_arch']}) inside the point-wise cap "
                       f"(worst ratio "
                       f"{choice['worst_pointwise_ratio']:.3f})."),
        }

    pool = [e for e in per_weight
            if e["cap_ok"] and e["worst_max_dE_xc_mHa"] is not None]
    capped = bool(pool)
    if not pool:
        pool = [e for e in per_weight if e["worst_max_dE_xc_mHa"] is not None]
    if not pool:
        return {
            "weight": None, "cleared": False, "rule": rule,
            "tol_atom_mHa": float(tol_atom_mha),
            "margin_mHa": margin_mha,
            "pointwise_factor": float(pointwise_factor),
            "per_weight": per_weight,
            "reason": "no cell carries a finite max |dE_xc|; nothing to "
                      "choose between.",
        }
    choice = min(pool, key=lambda e: (e["worst_max_dE_xc_mHa"], e["weight"]))
    reason = (f"NO swept weight clears {margin_mha:g} mHa on every "
              f"architecture. Reported instead: weight {choice['weight']:g}, "
              f"which minimizes the worst max |dE_xc| "
              f"({choice['worst_max_dE_xc_mHa']:.4f} mHa on "
              f"{choice['worst_arch']})")
    reason += (" among the weights inside the point-wise cap."
               if capped else
               "; the point-wise cap is violated at every swept weight.")
    return {
        "weight": choice["weight"], "cleared": False, "rule": rule,
        "tol_atom_mHa": float(tol_atom_mha),
        "margin_mHa": margin_mha,
        "pointwise_factor": float(pointwise_factor),
        "per_weight": per_weight,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

_COLUMNS = (
    ("arch", "arch", 22, "s"),
    ("parent", "reference_xc", 6, "s"),
    ("w_E", "weight", 8, "g"),
    ("loss_x", "final_loss_x", 12, ".4e"),
    ("loss_c", "final_loss_c", 12, ".4e"),
    ("rms_dE/mHa", "rms_dE_xc_mHa", 12, ".4f"),
    ("max_dE/mHa", "max_dE_xc_mHa", 12, ".4f"),
    ("worst", "worst_system", 8, "s"),
    ("wall_s", "wall_seconds", 9, ".1f"),
)


def format_table(rows, recommendation=None):
    """The measured table as fixed-width text, newest column meanings first.

    Kept separate from the sweep so a table read back from JSON renders the
    same way it was printed."""
    head = "  ".join(f"{name:>{width}s}" for name, _key, width, _f in _COLUMNS)
    lines = [head, "-" * len(head)]
    for row in sorted(rows, key=lambda r: (str(r.get("arch")),
                                           float(r.get("weight", 0.0)))):
        cells = []
        for _name, key, width, fmt in _COLUMNS:
            value = row.get(key)
            if value is None:
                text = "-"
            elif fmt == "s":
                text = str(value)
            elif not _finite(value):
                text = str(value)
            else:
                text = format(float(value), fmt)
            cells.append(f"{text:>{width}s}")
        lines.append("  ".join(cells))
    if recommendation is not None:
        lines.append("")
        lines.append(f"rule: {recommendation['rule']}")
        weight = recommendation.get("weight")
        verdict = "CLEARS" if recommendation.get("cleared") else "DOES NOT CLEAR"
        lines.append(
            f"recommendation: energy_term_weight = "
            f"{'none' if weight is None else format(weight, 'g')}  [{verdict}]")
        lines.append(f"reason: {recommendation['reason']}")
    return "\n".join(lines)


def write_table(path, payload):
    """Write the JSON table, creating its directory. Returns the path."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #

def _energy_keys(arch):
    """The per-system parent-energy table keys this architecture's rung reads,
    the way ``run_pretrain`` selects them."""
    if bool(getattr(arch, "meta_gga", False)):
        return "e_x_parent_scan_sys", "e_c_parent_scan_sys"
    return "e_x_parent_sys", "e_c_parent_sys"


def per_system_energy_errors(arch, data_path, checkpoint_dir, *, seed):
    """Per-system ``E_xc^NN - E_xc^parent`` of a finished pretraining, in Ha.

    Returns ``(delta_x, delta_c, names)`` as numpy arrays and a list of system
    names. The expression is the one ``_PretrainLoss.parts`` evaluates --
    ``sum_{i in s} w_i e_LDA_i F^NN_i`` against the stored per-system target --
    evaluated on the physical rows alone. The synthetic mesh rows are omitted
    rather than padded because their energy row weight is identically zero;
    the agreement with the recorded mean of squares is what proves the two
    paths are the same quantity.
    """
    import numpy as np
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.pretrain import (_assemble_pretrain_descriptors,
                                        _energy_term_inputs)
    from xcquinox.alec.pretrain_data_gen import read_pretrain_manifest

    raw = np.load(data_path)
    data = {k: np.array(raw[k]) for k in raw.files}
    x_suffix = "_x" if "rho_x" in data else "_all"
    e_x_key, e_c_key = _energy_keys(arch)

    descriptors_x = _assemble_pretrain_descriptors(arch, data,
                                                   suffix=x_suffix)
    descriptors_c = _assemble_pretrain_descriptors(arch, data, for_cnet=True)

    xnet, cnet = create_network_pair(arch, seed=seed)
    xnet = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "xnet.eqx"), xnet)
    cnet = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "cnet.eqx"), cnet)

    def _delta(model, descriptors, weight_key, lda_key, segment_key,
               target_key):
        row_weight, segment, target, n_systems = _energy_term_inputs(
            data, weight_key=weight_key, lda_key=lda_key,
            segment_key=segment_key, target_key=target_key, n_mesh=0)
        pred = jax.vmap(model)(jnp.asarray(descriptors)).squeeze()
        energy = jax.ops.segment_sum(
            row_weight * pred, segment,
            num_segments=n_systems + 1)[:n_systems]
        return np.asarray(energy - target, dtype=np.float64)

    delta_x = _delta(xnet, descriptors_x, "weights" + x_suffix,
                     "e_lda_x" + x_suffix, "system" + x_suffix, e_x_key)
    delta_c = _delta(cnet, descriptors_c, "weights_all", "e_lda_c_all",
                     "system_all", e_c_key)

    manifest = read_pretrain_manifest(data_path) or {}
    systems = manifest.get("systems") or []
    names = [str(entry[0]) for entry in systems]
    if len(names) != delta_x.shape[0]:
        names = [f"sys{i:d}" for i in range(int(delta_x.shape[0]))]
    return delta_x, delta_c, names


def ensure_data(data_dir, *, polarized, reference_xc, basis, grid_level,
                lock_strength, allow_degenerate, smoke_atoms=None):
    """Generate (or reuse) one pretraining file and return the path
    ``run_pretrain`` will open for it.

    The file lives in its own ``parent_<reference_xc>`` subdirectory so the
    two parent densities never share a ``data_dir``. Inside that directory the
    generated name is also exposed under the name ``run_pretrain`` asks for:
    ``pretrain._pretrain_data_filename`` builds that name at the PBE default
    and never re-derives it once the run's parent is resolved, so a SCAN-parent
    architecture pointed at the SCAN file would otherwise look for a name the
    generator never wrote. The alias is a link, so both names resolve to one
    file and one manifest whatever the resolution rule becomes.
    """
    from xcquinox.alec.pretrain_data_gen import (ensure_pretrain_data,
                                                 pretrain_data_filename)

    target_dir = os.path.join(data_dir, f"parent_{reference_xc}")
    os.makedirs(target_dir, exist_ok=True)
    kwargs = dict(basis=basis, grid_level=grid_level, polarized=polarized,
                  descriptors=True, dfs_set=True, pool_atoms=True,
                  reference_xc=reference_xc,
                  exchange_footing="spin_channel",
                  orientation_lock_strength=lock_strength,
                  allow_irreproducible_degenerate=allow_degenerate,
                  progress=True)
    if smoke_atoms is not None:
        kwargs.update(atoms=tuple(smoke_atoms), dfs_set=False,
                      pool_atoms=False)
    path = ensure_pretrain_data(target_dir, **kwargs)

    wanted = os.path.join(target_dir, pretrain_data_filename(polarized))
    if os.path.abspath(wanted) != os.path.abspath(path):
        for suffix in ("", ".manifest.json"):
            link, real = wanted + suffix, path + suffix
            if not os.path.exists(real):
                continue
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.basename(real), link)
    return path


def run_cell(arch, arch_name, data_path, work_dir, *, weight, n_steps, seed,
             loss_weighting, recon_rtol, label):
    """One (architecture, weight) pretraining, measured. Returns the row."""
    import numpy as np

    from xcquinox.alec.config import PretrainSpec
    from xcquinox.alec.pretrain import run_pretrain

    checkpoint_dir = os.path.join(work_dir, f"{arch_name}_w{weight:g}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    spec = PretrainSpec(arch=arch, data_dir=os.path.dirname(data_path),
                        checkpoint_dir=checkpoint_dir, n_steps=n_steps,
                        seed=seed, loss_weighting=loss_weighting,
                        energy_term_weight=float(weight),
                        parent_density="auto", validation_fraction=0.0)

    started = time.time()
    state = {"last": started}

    def _progress(event):
        now = event.get("timestamp", time.time())
        step, total = int(event["step"]), int(event["total"])
        stride = max(1, total // 10)
        if step % stride and step != total:
            return
        if now - state["last"] < 1.0 and step != total:
            return
        state["last"] = now
        done = time.time() - started
        rate = done / max(1, step)
        log(f"[probe] {label} {event['phase']}-net step {step}/{total} "
            f"loss={event['loss']:.6e} elapsed={done:.1f}s "
            f"eta={rate * (total - step):.1f}s")

    md = run_pretrain(spec, progress_callback=_progress)
    wall = time.time() - started

    delta_x, delta_c, names = per_system_energy_errors(
        arch, data_path, checkpoint_dir, seed=seed)
    term_x = float(np.mean(delta_x ** 2))
    term_c = float(np.mean(delta_c ** 2))

    # Internal consistency: the reconstruction must reproduce the mean of
    # squares the run recorded. Only meaningful where the recorded value is
    # real -- at weight 0 the loss short-circuits and records 0.0.
    recon_dev = None
    if weight > 0.0:
        recon_dev = 0.0
        for got, want, which in ((term_x, md["energy_term_x_final"], "x"),
                                 (term_c, md["energy_term_c_final"], "c")):
            scale = max(abs(float(want)), abs(got), 1e-300)
            dev = abs(got - float(want)) / scale
            recon_dev = max(recon_dev, dev)
            if not (dev <= recon_rtol):
                raise RuntimeError(
                    f"{label}: per-system reconstruction of the {which} "
                    f"energy term disagrees with the recorded value by "
                    f"{dev:.3e} relative (reconstructed {got!r}, recorded "
                    f"{want!r}); the two are the same expression on the same "
                    f"rows, so a gap above --recon-rtol={recon_rtol:g} means "
                    f"they are no longer reading the same thing.")

    delta_xc = delta_x + delta_c
    worst = int(np.argmax(np.abs(delta_xc)))
    row = {
        "arch": arch_name,
        "reference_xc": str(md.get("reference_xc", "")),
        "weight": float(weight),
        "final_loss_x": float(md["final_loss_x"]),
        "final_loss_c": float(md["final_loss_c"]),
        "rms_dE_xc_mHa": _HARTREE_TO_MHA * math.sqrt(term_x + term_c),
        "max_dE_xc_mHa": _HARTREE_TO_MHA * float(np.max(np.abs(delta_xc))),
        "max_dE_x_mHa": _HARTREE_TO_MHA * float(np.max(np.abs(delta_x))),
        "max_dE_c_mHa": _HARTREE_TO_MHA * float(np.max(np.abs(delta_c))),
        "worst_system": names[worst] if worst < len(names) else str(worst),
        "energy_term_x_recon": term_x,
        "energy_term_c_recon": term_c,
        "energy_term_x_final": float(md["energy_term_x_final"]),
        "energy_term_c_final": float(md["energy_term_c_final"]),
        "recon_max_rel_dev": recon_dev,
        "n_systems": int(md.get("n_systems", len(names))),
        "n_rows_x": int(md.get("n_rows_x", 0)),
        "n_rows_c": int(md.get("n_rows_c", 0)),
        "exchange_footing": str(md.get("exchange_footing", "")),
        "pretrain_mesh": bool(md.get("pretrain_mesh", False)),
        "pretrain_steps": int(md.get("pretrain_steps", n_steps)),
        "wall_seconds": wall,
    }
    log(f"[probe] {label} DONE loss_x={row['final_loss_x']:.4e} "
        f"loss_c={row['final_loss_c']:.4e} "
        f"rms={row['rms_dE_xc_mHa']:.4f} mHa "
        f"max={row['max_dE_xc_mHa']:.4f} mHa on {row['worst_system']} "
        f"wall={wall:.1f}s")
    return row


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    from xcquinox.alec.config import get_architecture, list_architectures
    from xcquinox.alec.pretrain_data_gen import (
        PRETRAIN_ORIENTATION_LOCK_STRENGTH, resolve_parent_density)

    lock = PRETRAIN_ORIENTATION_LOCK_STRENGTH
    smoke_atoms = SMOKE_ATOMS if args.smoke else None
    # Refused by name rather than through a bare KeyError: a mistyped
    # architecture is a mistake worth an hour of queue time to make plainly.
    unknown = [n for n in args.archs if n not in list_architectures()]
    if unknown:
        raise SystemExit(
            f"probe_pretrain_energy_weight: unknown architecture(s) "
            f"{', '.join(repr(n) for n in unknown)}; the registry holds "
            f"{', '.join(list_architectures())}.")
    archs = [(name, get_architecture(name)) for name in args.archs]

    log(f"[probe] identity: basis={args.basis} grid_level={args.grid_level} "
        f"lock={lock:g} footing=spin_channel "
        f"loss_weighting={args.loss_weighting} n_steps={args.n_steps} "
        f"seed={args.seed} smoke={bool(args.smoke)}")
    log(f"[probe] archs: {', '.join(args.archs)}")
    log(f"[probe] weights: {', '.join(format(w, 'g') for w in args.weights)}")

    # One data file per distinct (polarization, parent) pair, generated once
    # and reused by every cell that reads it.
    paths = {}
    for name, arch in archs:
        polarized = bool(getattr(arch, "use_polarized_correlation", False))
        parent = resolve_parent_density(arch, "auto")
        key = (polarized, parent)
        if key in paths:
            continue
        started = time.time()
        log(f"[probe] data: polarized={polarized} parent={parent} ...")
        paths[key] = ensure_data(
            args.data_dir, polarized=polarized, reference_xc=parent,
            basis=args.basis, grid_level=args.grid_level, lock_strength=lock,
            allow_degenerate=bool(args.smoke), smoke_atoms=smoke_atoms)
        log(f"[probe] data: {paths[key]} "
            f"({time.time() - started:.1f}s, total {time.time() - t0:.1f}s)")

    work_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                            "cells")
    os.makedirs(work_dir, exist_ok=True)

    rows, failures = [], []
    total = len(archs) * len(args.weights)
    index = 0
    for name, arch in archs:
        polarized = bool(getattr(arch, "use_polarized_correlation", False))
        parent = resolve_parent_density(arch, "auto")
        data_path = paths[(polarized, parent)]
        for weight in args.weights:
            index += 1
            label = f"({index}/{total}) {name} w_E={weight:g}"
            log(f"[probe] {label} START (elapsed {time.time() - t0:.1f}s)")
            try:
                rows.append(run_cell(
                    arch, name, data_path, work_dir, weight=weight,
                    n_steps=args.n_steps, seed=args.seed,
                    loss_weighting=args.loss_weighting,
                    recon_rtol=args.recon_rtol, label=label))
            except Exception as exc:                     # noqa: BLE001
                failures.append({"arch": name, "weight": float(weight),
                                 "error": f"{type(exc).__name__}: {exc}"})
                log(f"[probe] {label} FAILED: {type(exc).__name__}: {exc}")

    verdict = recommend(rows, tol_atom_mha=args.tol_atom_mha,
                        margin_fraction=args.margin_fraction,
                        pointwise_factor=args.pointwise_factor)
    payload = {
        "identity": {
            "basis": args.basis, "grid_level": int(args.grid_level),
            "orientation_lock_strength": float(lock),
            "exchange_footing": "spin_channel",
            "dfs_set": smoke_atoms is None, "pool_atoms": smoke_atoms is None,
            "atoms": (None if smoke_atoms is None
                      else [list(a) for a in smoke_atoms]),
            "n_steps": int(args.n_steps), "seed": int(args.seed),
            "loss_weighting": args.loss_weighting,
            "validation_fraction": 0.0,
            "smoke": bool(args.smoke),
            "data_dir": os.path.abspath(args.data_dir),
        },
        "rows": rows,
        "failures": failures,
        "recommendation": verdict,
        "total_wall_seconds": time.time() - t0,
    }
    write_table(args.out, payload)

    log("")
    log(format_table(rows, verdict))
    log("")
    log(f"[probe] table: {os.path.abspath(args.out)}")
    log(f"[probe] TOTAL WALL = {time.time() - t0:.1f} s")
    if failures:
        log(f"[probe] {len(failures)} cell(s) FAILED: "
            + "; ".join(f"{f['arch']} w={f['weight']:g}" for f in failures))
        return 1
    return 0 if verdict["cleared"] else 2


if __name__ == "__main__":
    sys.exit(main())
