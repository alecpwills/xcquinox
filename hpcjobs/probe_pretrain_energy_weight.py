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

Four quantities are reported per cell, because they are not the same test:

* ``rms_dE_xc``  = ``sqrt(<dE_x^2>_s + <dE_c^2>_s)``, the RMS per-system XC
  energy error. This is what the pretraining metadata's
  ``energy_term_{x,c}_final`` pair records, and it is what the objective
  itself minimizes.
* ``max_atom_dE_xc`` = ``max_s |dE_x,s + dE_c,s|`` over the SINGLE-CENTER
  systems. This is the certificate's first gate: Section 3.3 bounds atoms at
  ``tol_atom = 1.0 mHa``, a maximum, not a mean. A set whose RMS clears the
  tolerance can still carry one atom that does not.
* ``max_dAE`` = ``max_m |dE_xc,m - sum_{a in m} dE_xc,a|`` in kcal/mol over the
  molecules whose constituent free atoms are all in the set. This is the
  certificate's second gate, ``tol_AE = 1.0 kcal/mol``: an error common to a
  molecule and its atoms cancels out of an atomization energy, and an error
  that does not cancel is what the deployment sees.
* ``max_dE_xc``  = the same maximum over EVERY system in the set, atoms and
  molecules together. Reported rather than gated: a molecular error is bounded
  by the certificate through the atomization energy, in kcal/mol, not by
  ``tol_atom``.

All are computed here from the per-system residuals rather than read off the
pretraining's own record of them: that record carries the two means of squares
and one maximum over EVERY system, and neither the per-atom maximum nor an
atomization energy can be recovered from those. Where the two DO describe the
same quantity -- the two means of squares, the all-system maximum and the RMS,
which ``run_pretrain`` measures on its saved network at any weight -- the
reconstruction here is required to reproduce it to ``--recon-rtol`` on every
cell, and a disagreement is fatal. That is also what pins the gate quantity:
the maximum is ``max_s |dE_x,s + dE_c,s|`` in two independent implementations,
one measuring the network in memory and one the checkpoint read back off disk,
so dropping either channel from the sum fails the cell rather than quietly
changing the number the campaign is handed.

Identity: the sweep runs at ``--basis``/``--grid-level`` with the pretraining
set of spec Section 7 (the DFS inventory in its entirety plus every atom of
the BH76 / W4-11 pools), the per-channel exchange footing, the polarized
correlation objective every production configuration under ``hpcjobs/configs/``
sets (``--no-polarized`` measures the unpolarized one instead), and the
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

The table is rewritten after every cell, so a job killed at the wall leaves the
cells it finished; ``--resume`` reads that table back, carries its rows over
and measures only the cells missing from it, refusing a table written at a
different identity. That is also how the sweep is batched over architectures
when one wall cannot hold it: several submissions, each with ``--archs`` and
``--resume`` on the same ``--out``, accumulate into one table.

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

#: The certificate's tolerance on the atomization-energy error, in kcal/mol
#: (SPEC_pretrain_fidelity_program.md Section 3.3 item 4: "tol_AE = 1.0
#: kcal/mol"). The two tolerances are not one gate in two units: an XC error
#: shared by a molecule and its atoms cancels out of an atomization energy, so
#: a set can hold every atom inside tol_atom and still miss tol_AE, and the
#: reverse.
TOL_AE_KCAL = 1.0

#: CODATA-2018, the constant the deployment paths convert with
#: (``xcquinox/alec/cluster/domain.py``: ``KCAL_PER_HA = 627.5094740631``).
KCAL_PER_HA = 627.5094740631

#: The production value of the run-level ``use_polarized_correlation`` knob.
#: It is a per-RUN flag, not an architecture property: no registry
#: architecture carries it, and the cluster patches it onto every swept
#: architecture before anything is read off it
#: (``cluster/_datagen.py:_swept_architectures``, ``spec_builder.py:597``).
#: Every configuration of the dfs6311 v3-v5 lineage under ``hpcjobs/configs/``
#: and the v6 template ``cluster/examples/workflow_matrix_template.yaml`` set
#: it true, so the weight is measured on the polarized objective it will be
#: applied to. ``--no-polarized`` measures the unpolarized one.
DEFAULT_POLARIZED = True

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
#: grid level 1, five optimizer steps. Neither is spatially degenerate, so the
#: generator accepts the pair at that grid level as it stands and the sweep
#: asks for no waiver at either identity. It measures the plumbing, never a
#: weight.
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
                   help="Path of the JSON table, rewritten after every cell "
                        "so a job killed at the wall keeps what it measured.")
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
                        "open-shell atoms of the set; the sweep never waives "
                        "that refusal.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Network initialization seed (default {DEFAULT_SEED}"
                        "). One seed for every cell, so the comparison "
                        "between weights is not a comparison between draws.")
    p.add_argument("--loss-weighting", default="integration",
                   choices=("integration", "unweighted"),
                   help="Point-wise reduction (default: integration, which "
                        "is what the campaign trains with).")
    polar = p.add_mutually_exclusive_group()
    polar.add_argument("--polarized", dest="polarized", action="store_true",
                       default=DEFAULT_POLARIZED,
                       help="Measure on the polarized correlation objective "
                            "(the zeta-aware cnet and the zeta-carrying data "
                            "file). Default, because every production "
                            "configuration sets use_polarized_correlation.")
    polar.add_argument("--no-polarized", dest="polarized",
                       action="store_false",
                       help="Measure on the unpolarized objective instead.")
    p.add_argument("--resume", action="store_true",
                   help="Read --out back and skip the cells already in it. "
                        "The stored identity must match the one requested; a "
                        "mismatch is refused rather than merged.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny identity used by the tests and for a local "
                        "plumbing check: two systems at STO-3G on grid level "
                        "1, five steps. Explicit flags still win.")
    p.add_argument("--recon-rtol", type=float, default=DEFAULT_RECON_RTOL,
                   help="Relative agreement demanded between the per-system "
                        "reconstruction and every quantity the pretraining "
                        "recorded for itself -- the two means of squares, the "
                        "all-system maximum and the RMS "
                        f"(default {DEFAULT_RECON_RTOL:g}).")
    p.add_argument("--tol-atom-mha", type=float, default=TOL_ATOM_MHA,
                   help=f"Certificate tolerance on max |dE_xc| per ATOM, "
                        f"in mHa (default {TOL_ATOM_MHA}).")
    p.add_argument("--tol-ae-kcal", type=float, default=TOL_AE_KCAL,
                   help=f"Certificate tolerance on max |dAE| per molecule, "
                        f"in kcal/mol (default {TOL_AE_KCAL}).")
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
    for field in ("tol_atom_mha", "tol_ae_kcal", "margin_fraction",
                  "pointwise_factor"):
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


def _gate_reading(cell, key, count_key):
    """One gate quantity of one cell, as ``(value, measured)``.

    ``None`` beside an explicit count of zero means the set carries no system
    of that kind -- the smoke's two free atoms define no atomization energy --
    so the gate is vacuous on that cell and nothing is claimed from it.
    ``None`` beside any other count, an absent count included, is NOT a pass: a
    table that does not carry the quantity cannot certify it. A non-finite
    entry is a failure, reported as an infinite value so it also loses the
    fallback comparison.
    """
    value = cell.get(key)
    if value is None:
        count = cell.get(count_key)
        if isinstance(count, (int, float)) and not isinstance(count, bool) \
                and int(count) == 0:
            return None, True
        return None, False
    if not _finite(value):
        return float("inf"), True
    return float(value), True


def recommend(rows, *, tol_atom_mha=TOL_ATOM_MHA, tol_ae_kcal=TOL_AE_KCAL,
              margin_fraction=MARGIN_FRACTION,
              pointwise_factor=POINTWISE_FACTOR):
    """Choose the production weight from a measured table.

    The rule, stated so the choice is reproducible rather than aesthetic:

      take the SMALLEST swept weight for which EVERY architecture clears BOTH
      halves of the Section 3.3 certificate with margin -- ``max |dE_xc|`` over
      the ATOMS at or below ``margin_fraction * tol_atom_mha``, and ``max
      |dAE|`` over the molecules at or below ``margin_fraction *
      tol_ae_kcal`` -- AND neither of that architecture's point-wise losses has
      risen by more than ``pointwise_factor`` from its own weight-0 value.

    Both halves are required because neither implies the other: an XC error
    common to a molecule and its atoms cancels out of the atomization energy,
    and an error that cancels there can still break the atom bound. The
    all-system maximum ``max_dE_xc_mHa`` is reported beside them and gates
    nothing -- bounding a molecule's absolute XC error at ``tol_atom`` is a
    tolerance the certificate does not impose, and reading it as the gate can
    refuse a weight the certificate would pass.

    A weight measured on only part of the architecture set is not eligible:
    the gate is "on every architecture", and an absent cell is not a pass. A
    non-finite entry is a failure, not a missing value -- a diverged fit is
    exactly what the cap exists to reject.

    When nothing clears, the fallback is the weight that MINIMIZES the worst
    gate quantity IN UNITS OF ITS OWN MARGIN (so the two tolerances compare)
    among those still inside the point-wise cap (and among all of them if the
    cap is violated everywhere), with the smallest weight winning a tie. That
    is a finding rather than a choice: a pretraining that cannot reach the
    margin on its own rows will not pass the certificate at the production
    identity either.

    Returns a dict; ``cleared`` says whether the returned weight satisfies the
    rule or is the fallback.
    """
    margin_mha = float(margin_fraction) * float(tol_atom_mha)
    margin_ae = float(margin_fraction) * float(tol_ae_kcal)
    archs = sorted({str(r["arch"]) for r in rows})
    weights = sorted({float(r["weight"]) for r in rows})
    by_cell = {(str(r["arch"]), float(r["weight"])): r for r in rows}
    baseline = {a: by_cell.get((a, 0.0)) for a in archs}

    gates = (("max_atom_dE_xc_mHa", "n_atom_systems", margin_mha),
             ("max_dAE_kcal", "n_ae_molecules", margin_ae))

    per_weight = []
    for w in weights:
        missing = [a for a in archs if (a, w) not in by_cell]
        worst_max, worst_arch = -1.0, None
        worst_atom, worst_atom_arch = -1.0, None
        worst_ae, worst_ae_arch = -1.0, None
        worst_gate, worst_gate_arch = None, None
        worst_ratio, worst_ratio_arch = -1.0, None
        gate_ok = not missing
        cap_ok = not missing
        no_baseline = []
        unmeasured = []
        for a in archs:
            cell = by_cell.get((a, w))
            if cell is None:
                continue
            # Reported, not gated: the all-system maximum.
            reported = cell.get("max_dE_xc_mHa")
            if not _finite(reported):
                worst_max, worst_arch = float("inf"), a
            elif float(reported) > worst_max:
                worst_max, worst_arch = float(reported), a
            for key, count_key, margin in gates:
                value, measured = _gate_reading(cell, key, count_key)
                if not measured:
                    gate_ok = False
                    unmeasured.append(f"{a}:{key}")
                    continue
                if value is None:            # vacuous on this cell
                    continue
                if value > margin:
                    gate_ok = False
                ratio = value / margin if margin > 0 else float("inf")
                if worst_gate is None or ratio > worst_gate:
                    worst_gate, worst_gate_arch = ratio, a
                if key == "max_atom_dE_xc_mHa" and value > worst_atom:
                    worst_atom, worst_atom_arch = value, a
                if key == "max_dAE_kcal" and value > worst_ae:
                    worst_ae, worst_ae_arch = value, a
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
            "worst_atom_dE_xc_mHa": (None if worst_atom < 0 else worst_atom),
            "worst_atom_arch": worst_atom_arch,
            "worst_dAE_kcal": (None if worst_ae < 0 else worst_ae),
            "worst_ae_arch": worst_ae_arch,
            "worst_gate_ratio": worst_gate,
            "worst_gate_arch": worst_gate_arch,
            "worst_pointwise_ratio": (None if worst_ratio < 0
                                      else worst_ratio),
            "worst_ratio_arch": worst_ratio_arch,
            "missing_archs": missing,
            "archs_without_baseline": no_baseline,
            "unmeasured_gates": unmeasured,
        })

    rule = (f"smallest weight with max |dE_xc| over atoms <= {margin_mha:g} "
            f"mHa ({margin_fraction:g} x tol_atom = {tol_atom_mha:g} mHa) AND "
            f"max |dAE| over molecules <= {margin_ae:g} kcal/mol "
            f"({margin_fraction:g} x tol_AE = {tol_ae_kcal:g} kcal/mol) on "
            f"every architecture, neither point-wise loss above "
            f"{pointwise_factor:g}x its weight-0 value")

    def _verdict(weight, cleared, reason):
        return {
            "weight": weight, "cleared": cleared, "rule": rule,
            "tol_atom_mHa": float(tol_atom_mha),
            "tol_AE_kcal": float(tol_ae_kcal),
            "margin_mHa": margin_mha,
            "margin_AE_kcal": margin_ae,
            "pointwise_factor": float(pointwise_factor),
            "per_weight": per_weight,
            "reason": reason,
        }

    def _gate_text(entry):
        atom = entry["worst_atom_dE_xc_mHa"]
        ae = entry["worst_dAE_kcal"]
        return (f"worst atom |dE_xc| "
                + ("none measured" if atom is None else f"{atom:.4f} mHa "
                   f"on {entry['worst_atom_arch']}")
                + ", worst |dAE| "
                + ("none measured" if ae is None else f"{ae:.4f} kcal/mol "
                   f"on {entry['worst_ae_arch']}"))

    eligible = [e for e in per_weight if e["gate_ok"] and e["cap_ok"]]
    if eligible:
        choice = min(eligible, key=lambda e: e["weight"])
        return _verdict(
            choice["weight"], True,
            f"weight {choice['weight']:g} clears both gates on all "
            f"{len(archs)} architectures ({_gate_text(choice)}) inside the "
            f"point-wise cap (worst ratio "
            f"{choice['worst_pointwise_ratio']:.3f}).")

    pool = [e for e in per_weight
            if e["cap_ok"] and e["worst_gate_ratio"] is not None]
    capped = bool(pool)
    if not pool:
        pool = [e for e in per_weight if e["worst_gate_ratio"] is not None]
    if not pool:
        return _verdict(
            None, False,
            "no cell carries a measured gate quantity; nothing to choose "
            "between.")
    choice = min(pool, key=lambda e: (e["worst_gate_ratio"], e["weight"]))
    reason = (f"NO swept weight clears {margin_mha:g} mHa on the atoms and "
              f"{margin_ae:g} kcal/mol on the atomization energies of every "
              f"architecture. Reported instead: weight "
              f"{choice['weight']:g}, which minimizes the worst gate quantity "
              f"in units of its own margin "
              f"({choice['worst_gate_ratio']:.3f} on "
              f"{choice['worst_gate_arch']}; {_gate_text(choice)})")
    reason += (" among the weights inside the point-wise cap."
               if capped else
               "; the point-wise cap is violated at every swept weight.")
    return _verdict(choice["weight"], False, reason)


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
    ("atom_dE/mHa", "max_atom_dE_xc_mHa", 12, ".4f"),
    ("worst_atom", "worst_atom", 11, "s"),
    ("dAE/kcal", "max_dAE_kcal", 10, ".4f"),
    ("worst_AE", "worst_ae_system", 9, "s"),
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


def _check_rung_consistency(name, arch):
    """Refuse an architecture whose two readings of "meta-GGA rung" disagree.

    The library reads that question in two places and not the same way:
    ``pretrain_data_gen.resolve_parent_density`` takes the ``meta_gga`` flag OR
    a ``"metagga"`` descriptor, while ``pretrain.run_pretrain`` selects the
    enhancement-factor targets and the per-system parent-energy keys from the
    FLAG alone. An architecture carrying the descriptor without the flag
    therefore pretrains on the SCAN self-consistent density against PBE
    targets, and the energy error this probe reports would be measured against
    the wrong parent. No architecture in the registry is in that state, so the
    refusal costs nothing today; it is here because an architecture assembled
    outside ``ArchitectureConfig.from_spec`` can be, and a sweep is not the
    place to discover it.
    """
    descriptor_names = {getattr(d, "name", None)
                        for d in getattr(arch, "descriptors", ())}
    flag = bool(getattr(arch, "meta_gga", False))
    descriptor = "metagga" in descriptor_names
    if flag != descriptor:
        raise SystemExit(
            f"probe_pretrain_energy_weight: architecture {name!r} carries "
            f"meta_gga={flag} with"
            + (" a" if descriptor else "out a")
            + " 'metagga' descriptor. The parent density is resolved from the "
              "flag OR the descriptor (pretrain_data_gen."
              "resolve_parent_density) while the pretraining targets and the "
              "per-system parent energies are selected from the flag alone "
              "(pretrain.run_pretrain), so this architecture would be fitted "
              "to one parent's targets on the other parent's density. Give it "
              "both or neither.")


def _geometry_elements(geometry):
    """The element symbols of a PySCF ``atom`` string, in order.

    The manifest stores each system's geometry exactly as the generator built
    the molecule, so the composition is read from the same string the SCF ran
    on rather than parsed out of a name -- ``CO`` is carbon and oxygen, and no
    name-based rule gets that right in general. A numbered label (``H1``) keeps
    its element.
    """
    import re

    out = []
    for chunk in str(geometry).replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.match(r"^([A-Za-z]+)", chunk.split()[0])
        if match:
            symbol = match.group(1)
            out.append(symbol[:1].upper() + symbol[1:].lower())
    return out


def classify_systems(systems):
    """Split a manifest system list into single-center systems and molecules.

    Returns ``(atom_indices, neutral_atoms, molecules)``:

    * ``atom_indices`` -- every SINGLE-CENTER system, neutral atoms and atomic
      ions alike. This is the set the certificate's ``tol_atom`` bounds; the
      pool's ``F-`` and ``Cl-`` are free species of the pretraining set and are
      held to the same bound as the neutral atoms.
    * ``neutral_atoms`` -- element symbol -> index of the neutral free atom of
      that element, the reference an atomization energy is taken against.
    * ``molecules`` -- ``(index, name, {symbol: count})`` per multi-center
      system.
    """
    atom_indices, neutral, molecules = [], {}, []
    for index, entry in enumerate(systems or ()):
        name, geometry, charge = str(entry[0]), str(entry[1]), int(entry[2])
        centers = _geometry_elements(geometry)
        if len(centers) == 1:
            atom_indices.append(index)
            if charge == 0:
                neutral.setdefault(centers[0], index)
        elif len(centers) > 1:
            counts = {}
            for symbol in centers:
                counts[symbol] = counts.get(symbol, 0) + 1
            molecules.append((index, name, counts))
    return atom_indices, neutral, molecules


def atomization_errors(delta_xc, systems):
    """Per-molecule atomization-energy error in kcal/mol, and what was skipped.

    ``dAE_m = dE_xc,m - sum_{a in m} n_a dE_xc,a`` on the per-system XC energy
    errors, converted with the deployment's own Hartree-to-kcal/mol constant.
    This is the quantity the Section 3.3 certificate bounds at ``tol_AE``: an
    error the network makes identically on a molecule and on its atoms cancels
    out of an atomization energy and is invisible to the deployment, while an
    error that does not cancel is exactly what the deployment sees.

    A molecule one of whose constituent NEUTRAL free atoms is absent from the
    set defines no atomization error here (``Na2`` in the Section 7 set: the
    pools contribute no sodium atom) and is returned as skipped rather than
    quietly dropped, so the count the gate certifies is stated.

    Returns ``([(name, dAE_kcal), ...], [skipped_name, ...])``.
    """
    _atoms, neutral, molecules = classify_systems(systems)
    errors, skipped = [], []
    for index, name, counts in molecules:
        if any(symbol not in neutral for symbol in counts):
            skipped.append(name)
            continue
        total = float(delta_xc[index])
        for symbol, count in counts.items():
            total -= count * float(delta_xc[neutral[symbol]])
        errors.append((name, KCAL_PER_HA * total))
    return errors, skipped


def summarize_energy_errors(delta_x, delta_c, systems, names):
    """The measured half of a table row, from the per-system errors alone.

    Pure and separate from the run so the gate quantities are testable without
    a pretraining. The gate quantity is the XC error, ``dE_x + dE_c`` per
    system: the exchange and correlation halves of one functional are not two
    measurements, and the certificate bounds their sum.
    """
    import numpy as np

    delta_x = np.asarray(delta_x, dtype=np.float64)
    delta_c = np.asarray(delta_c, dtype=np.float64)
    delta_xc = delta_x + delta_c
    magnitude = np.abs(delta_xc)
    worst = int(np.argmax(magnitude))

    # A file with no manifest describes no atoms and no molecules: the counts
    # are UNKNOWN, not zero, so the rule refuses to certify from the row
    # instead of reading an absent split as an empty one.
    atom_max, worst_atom, n_atoms = None, None, None
    ae_max, worst_ae, n_molecules, ae_skipped = None, None, None, None
    if systems:
        atom_indices, _neutral, _molecules = classify_systems(systems)
        n_atoms = len(atom_indices)
        if atom_indices:
            index = atom_indices[int(np.argmax(magnitude[atom_indices]))]
            atom_max = _HARTREE_TO_MHA * float(magnitude[index])
            worst_atom = names[index] if index < len(names) else str(index)
        ae_errors, ae_skipped = atomization_errors(delta_xc, systems)
        n_molecules = len(ae_errors)
        if ae_errors:
            worst_ae, value = max(ae_errors, key=lambda pair: abs(pair[1]))
            ae_max = abs(float(value))

    term_x = float(np.mean(delta_x ** 2))
    term_c = float(np.mean(delta_c ** 2))
    return {
        "energy_term_x_recon": term_x,
        "energy_term_c_recon": term_c,
        "rms_dE_xc_mHa": _HARTREE_TO_MHA * math.sqrt(term_x + term_c),
        "max_dE_xc_mHa": _HARTREE_TO_MHA * float(np.max(magnitude)),
        "max_dE_x_mHa": _HARTREE_TO_MHA * float(np.max(np.abs(delta_x))),
        "max_dE_c_mHa": _HARTREE_TO_MHA * float(np.max(np.abs(delta_c))),
        "worst_system": names[worst] if worst < len(names) else str(worst),
        "max_atom_dE_xc_mHa": atom_max,
        "worst_atom": worst_atom,
        "n_atom_systems": n_atoms,
        "max_dAE_kcal": ae_max,
        "worst_ae_system": worst_ae,
        "n_ae_molecules": n_molecules,
        "ae_skipped": ae_skipped,
        "per_system": {
            "names": list(names),
            "delta_x_mHa": [_HARTREE_TO_MHA * float(v) for v in delta_x],
            "delta_c_mHa": [_HARTREE_TO_MHA * float(v) for v in delta_c],
        },
    }


#: What the probe reconstructs, beside the pretraining's own record of it.
#: ``energy_term_max_abs_dE_mHa`` is the load-bearing one: it is
#: ``1000 max_s |dE_x,s + dE_c,s|`` computed by ``run_pretrain`` on the network
#: in memory (``pretrain.py``, ``_saved_network_energy_error`` +
#: ``system_energy_errors``), against the same quantity computed here from the
#: checkpoint deserialised off disk with the descriptors rebuilt. The gate
#: quantity is therefore pinned to an independent implementation on every cell
#: of every run: dropping either channel from the sum moves it by a factor of
#: about two and the cell fails immediately.
_RECON_KEYS = (
    ("energy_term_x_final", "energy_term_x_recon", "exchange energy term"),
    ("energy_term_c_final", "energy_term_c_recon", "correlation energy term"),
    ("energy_term_max_abs_dE_mHa", "max_dE_xc_mHa", "maximum |dE_xc| in mHa"),
    ("energy_term_rms_dE_mHa", "rms_dE_xc_mHa", "RMS dE_xc in mHa"),
)


def check_reconstruction(row, metadata, recon_rtol, label, *, weight):
    """Refuse a cell whose reconstruction has drifted from what was recorded.

    The reconstruction here and the pretraining's own record evaluate the same
    expressions on the same rows through different code -- the record measures
    the network in memory, this measures the checkpoint read back from disk
    with its descriptors reassembled -- so they differ only in the reduction
    order of a segmented sum and in whether the zero-weight mesh rows are
    carried. Anything above ``recon_rtol`` means the two are no longer reading
    the same quantity, and the table's maxima are then not the objective's.
    Returns the worst relative deviation, or ``None`` where there is nothing
    real to compare against.
    """
    if weight == 0.0 and metadata.get("energy_term_max_abs_dE_mHa") is None:
        # A pretraining written before the record measured its SAVED network
        # regardless of weight puts 0.0 here, where the loss short-circuits:
        # not the same quantity, and nothing to compare.
        return None
    worst = None
    for recorded_key, own_key, which in _RECON_KEYS:
        want = metadata.get(recorded_key)
        if want is None or not math.isfinite(float(want)):
            continue
        got, want = float(row[own_key]), float(want)
        scale = max(abs(want), abs(got), 1e-300)
        dev = abs(got - want) / scale
        worst = dev if worst is None else max(worst, dev)
        if not (dev <= recon_rtol):
            raise RuntimeError(
                f"{label}: the reconstructed {which} disagrees with the value "
                f"the run recorded by {dev:.3e} relative (reconstructed "
                f"{got!r}, recorded {want!r}); the two are the same expression "
                f"on the same rows, so a gap above --recon-rtol="
                f"{recon_rtol:g} means they are no longer reading the same "
                f"thing.")
    return worst


def per_system_energy_errors(arch, data_path, checkpoint_dir, *, seed):
    """Per-system ``E_xc^NN - E_xc^parent`` of a finished pretraining, in Ha.

    Returns ``(delta_x, delta_c, systems)`` as numpy arrays and the manifest's
    own ``[name, geometry, charge, spin]`` list in the order the segment index
    uses (``None`` when no manifest sits beside the file). The expression is
    the one ``_PretrainLoss.parts`` evaluates --
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
    systems = manifest.get("systems") or None
    if systems is not None and len(systems) != int(delta_x.shape[0]):
        # A manifest that does not describe these rows describes nothing: the
        # segment index and the system list must be the same order.
        systems = None
    return delta_x, delta_c, systems


def system_names(systems, count):
    """Display names for the measured systems, positional where none exist."""
    if not systems:
        return [f"sys{i:d}" for i in range(int(count))]
    return [str(entry[0]) for entry in systems]


def ensure_data(data_dir, *, polarized, reference_xc, basis, grid_level,
                lock_strength, smoke_atoms=None):
    """Generate (or reuse) one pretraining file and return the path
    ``run_pretrain`` will open for it.

    The file lives in its own ``parent_<reference_xc>`` subdirectory so the two
    parent densities never share a ``data_dir``, and it is written under the
    generator's own name: ``run_pretrain`` resolves the run's parent BEFORE it
    names the file, so the name the worker opens is the name the generator
    wrote, for both polarizations and both parents.

    The irreproducible-degenerate refusal is never waived. At the production
    identity grid level 3 is a floor precisely so no waiver is needed, and the
    smoke's two systems are not spatially degenerate at any level, so the
    generator accepts both identities as they stand. (The manifest's
    ``allow_irreproducible_degenerate`` records whether the permission was
    EXERCISED rather than whether it was offered, so it reads false here in
    either case; what is pinned by test is the argument passed.)
    """
    from xcquinox.alec.pretrain_data_gen import ensure_pretrain_data

    target_dir = os.path.join(data_dir, f"parent_{reference_xc}")
    os.makedirs(target_dir, exist_ok=True)
    kwargs = dict(basis=basis, grid_level=grid_level, polarized=polarized,
                  descriptors=True, dfs_set=True, pool_atoms=True,
                  reference_xc=reference_xc,
                  exchange_footing="spin_channel",
                  orientation_lock_strength=lock_strength,
                  allow_irreproducible_degenerate=False,
                  progress=True)
    if smoke_atoms is not None:
        kwargs.update(atoms=tuple(smoke_atoms), dfs_set=False,
                      pool_atoms=False)
    return ensure_pretrain_data(target_dir, **kwargs)


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

    delta_x, delta_c, systems = per_system_energy_errors(
        arch, data_path, checkpoint_dir, seed=seed)
    names = system_names(systems, delta_x.shape[0])

    row = {
        "arch": arch_name,
        "reference_xc": str(md.get("reference_xc", "")),
        "polarized": bool(getattr(arch, "use_polarized_correlation", False)),
        "weight": float(weight),
        "final_loss_x": float(md["final_loss_x"]),
        "final_loss_c": float(md["final_loss_c"]),
        "energy_term_x_final": md.get("energy_term_x_final"),
        "energy_term_c_final": md.get("energy_term_c_final"),
        "energy_term_max_abs_dE_mHa": md.get("energy_term_max_abs_dE_mHa"),
        "n_systems": int(md.get("n_systems", len(names))),
        "n_rows_x": int(md.get("n_rows_x", 0)),
        "n_rows_c": int(md.get("n_rows_c", 0)),
        "exchange_footing": str(md.get("exchange_footing", "")),
        "pretrain_mesh": bool(md.get("pretrain_mesh", False)),
        "pretrain_steps": int(md.get("pretrain_steps", n_steps)),
        "wall_seconds": wall,
    }
    row.update(summarize_energy_errors(delta_x, delta_c, systems, names))
    # Internal consistency, on every quantity the run recorded for itself:
    # the reconstruction here reads the checkpoint back off disk, the record
    # measured the network in memory, and the two must be the same number.
    row["recon_max_rel_dev"] = check_reconstruction(
        row, md, recon_rtol, label, weight=weight)
    atom_text = ("-" if row["max_atom_dE_xc_mHa"] is None
                 else f"{row['max_atom_dE_xc_mHa']:.4f} mHa on "
                      f"{row['worst_atom']}")
    ae_text = ("-" if row["max_dAE_kcal"] is None
               else f"{row['max_dAE_kcal']:.4f} kcal/mol on "
                    f"{row['worst_ae_system']}")
    log(f"[probe] {label} DONE loss_x={row['final_loss_x']:.4e} "
        f"loss_c={row['final_loss_c']:.4e} "
        f"rms={row['rms_dE_xc_mHa']:.4f} mHa "
        f"atom_max={atom_text} maxAE={ae_text} "
        f"max_all={row['max_dE_xc_mHa']:.4f} mHa on {row['worst_system']} "
        f"wall={wall:.1f}s")
    return row


#: The identity keys a resumed table has to agree with the requested run on.
#: The tolerances and the recommendation are re-derived from the rows on every
#: write, so they are not part of the identity; everything that decides what a
#: row MEASURES is.
_RESUME_IDENTITY_KEYS = ("basis", "grid_level", "orientation_lock_strength",
                         "exchange_footing", "dfs_set", "pool_atoms", "atoms",
                         "n_steps", "seed", "loss_weighting", "polarized",
                         "validation_fraction", "smoke")


def build_identity(args, lock, smoke_atoms):
    """The identity block: everything a measured row depends on."""
    return {
        "basis": args.basis, "grid_level": int(args.grid_level),
        "orientation_lock_strength": float(lock),
        "exchange_footing": "spin_channel",
        "dfs_set": smoke_atoms is None, "pool_atoms": smoke_atoms is None,
        "atoms": (None if smoke_atoms is None
                  else [list(a) for a in smoke_atoms]),
        "n_steps": int(args.n_steps), "seed": int(args.seed),
        "loss_weighting": args.loss_weighting,
        "polarized": bool(args.polarized),
        "validation_fraction": 0.0,
        "smoke": bool(args.smoke),
        "data_dir": os.path.abspath(args.data_dir),
    }


def load_resumable_rows(path, identity):
    """Rows of a previous run of THIS sweep, from the table it left behind.

    A twelve-hour reservation that is killed at the wall must not lose the
    cells it paid for, so the table is rewritten after every cell and read back
    here. EVERY row of the stored table is carried over, not only the cells
    this run would measure: that is how a sweep batched over architectures
    (one submission per batch, each with ``--resume`` on the same table)
    accumulates into a single table the rule can read across all of them.
    What makes that safe is the identity: rows measured at two identities are
    not one measurement, so a stored identity differing from the requested one
    is refused rather than merged. Duplicate cells keep their first occurrence.
    """
    if not os.path.isfile(path):
        return []
    with open(path) as handle:
        payload = json.load(handle)
    stored = payload.get("identity") or {}
    differing = [k for k in _RESUME_IDENTITY_KEYS
                 if stored.get(k) != identity.get(k)]
    if differing:
        raise SystemExit(
            f"probe_pretrain_energy_weight: --resume was given {path!r}, "
            f"whose identity differs from the requested one in "
            f"{', '.join(differing)} ("
            + "; ".join(f"{k}: stored {stored.get(k)!r} vs requested "
                        f"{identity.get(k)!r}" for k in differing)
            + "). Rows measured at two identities are not one table; point "
              "--out at a new file or drop --resume.")
    kept, seen = [], set()
    for row in payload.get("rows") or []:
        key = (str(row.get("arch")), float(row.get("weight", float("nan"))))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    import dataclasses

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
    # The run-level polarization flag is patched onto every swept architecture
    # before anything is read off it, exactly as cluster/_datagen.py's
    # _swept_architectures and spec_builder do it, so the objective measured
    # here is the objective the campaign trains.
    archs = [(name, dataclasses.replace(
        get_architecture(name),
        use_polarized_correlation=bool(args.polarized)))
        for name in args.archs]
    for name, arch in archs:
        _check_rung_consistency(name, arch)

    log(f"[probe] identity: basis={args.basis} grid_level={args.grid_level} "
        f"lock={lock:g} footing=spin_channel polarized={bool(args.polarized)} "
        f"loss_weighting={args.loss_weighting} n_steps={args.n_steps} "
        f"seed={args.seed} smoke={bool(args.smoke)}")
    log(f"[probe] archs: {', '.join(args.archs)}")
    log(f"[probe] weights: {', '.join(format(w, 'g') for w in args.weights)}")
    log(f"[probe] gates: max |dE_xc| over atoms <= "
        f"{args.margin_fraction * args.tol_atom_mha:g} mHa, max |dAE| over "
        f"molecules <= {args.margin_fraction * args.tol_ae_kcal:g} kcal/mol")

    identity = build_identity(args, lock, smoke_atoms)
    cells = [(name, float(w)) for name, _a in archs for w in args.weights]
    rows, failures = [], []
    if args.resume:
        rows = load_resumable_rows(args.out, identity)
        log(f"[probe] resume: {len(rows)} row(s) carried over from "
            f"{os.path.abspath(args.out)}"
            + (": " + ", ".join(f"{r['arch']} w={float(r['weight']):g}"
                                for r in rows) if rows else ""))
    done = {(str(r["arch"]), float(r["weight"])) for r in rows}
    log(f"[probe] cells: {len(cells)} requested, "
        f"{sum(1 for c in cells if c in done)} already measured")

    def _write(complete):
        verdict = recommend(rows, tol_atom_mha=args.tol_atom_mha,
                            tol_ae_kcal=args.tol_ae_kcal,
                            margin_fraction=args.margin_fraction,
                            pointwise_factor=args.pointwise_factor)
        write_table(args.out, {
            "identity": identity,
            "complete": bool(complete),
            "rows": rows,
            "failures": failures,
            "recommendation": verdict,
            "total_wall_seconds": time.time() - t0,
        })
        return verdict

    # One data file per distinct (polarization, parent) pair, generated once
    # and reused by every cell that reads it.
    paths = {}
    for name, arch in archs:
        if all((name, float(w)) in done for w in args.weights):
            continue                      # nothing left to measure for it
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
            smoke_atoms=smoke_atoms)
        log(f"[probe] data: {paths[key]} "
            f"({time.time() - started:.1f}s, total {time.time() - t0:.1f}s)")

    work_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                            "cells")
    os.makedirs(work_dir, exist_ok=True)

    total = len(cells)
    index = 0
    for name, arch in archs:
        polarized = bool(getattr(arch, "use_polarized_correlation", False))
        parent = resolve_parent_density(arch, "auto")
        for weight in args.weights:
            index += 1
            label = f"({index}/{total}) {name} w_E={weight:g}"
            if (name, float(weight)) in done:
                log(f"[probe] {label} SKIPPED (already in the table)")
                continue
            log(f"[probe] {label} START (elapsed {time.time() - t0:.1f}s)")
            try:
                rows.append(run_cell(
                    arch, name, paths[(polarized, parent)], work_dir,
                    weight=weight, n_steps=args.n_steps, seed=args.seed,
                    loss_weighting=args.loss_weighting,
                    recon_rtol=args.recon_rtol, label=label))
            except Exception as exc:                     # noqa: BLE001
                failures.append({"arch": name, "weight": float(weight),
                                 "error": f"{type(exc).__name__}: {exc}"})
                log(f"[probe] {label} FAILED: {type(exc).__name__}: {exc}")
            # Written after EVERY cell, finished or failed: a job killed at the
            # wall keeps the cells it paid for, and --resume reads them back.
            _write(complete=False)

    verdict = _write(complete=True)

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
    # The exit code IS the finding here (0 cleared / 2 did not / 1 broke), and
    # JAX's atexit backend cleanup can corrupt the heap during interpreter
    # shutdown ("corrupted size vs. prev_size", SIGABRT, code 134) AFTER the
    # table has been written -- the same teardown that once made a green
    # cluster regression batch read as FAILED (see
    # xcquinox/alec/tests/conftest.py). SLURM would then record a signal death
    # for a sweep that completed. Leaving through os._exit skips teardown
    # entirely; the streams are flushed explicitly because it does not.
    try:
        _code = main()
    except SystemExit as _exc:      # argparse, and the sweep's own refusals
        _code = _exc.code
        if _code is None:
            _code = 0
        elif not isinstance(_code, int):
            print(_code, file=sys.stderr)
            _code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(_code))
