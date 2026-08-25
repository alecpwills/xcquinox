#!/usr/bin/env python
"""Local pretraining board: every architecture, pretrained on this workstation.

The board is the statement that pretraining WORKS for each architecture before
any cluster time is spent on it. One small pretraining dataset is generated per
session (sto-3g, grid level 3, polarized, the production orientation lock) and
every registered architecture is pretrained on it twice for a short schedule:

* ANCHORED, in the DFS coordinates. At initialization the model IS its parent,
  so the per-system XC energy errors must sit at the oracle floor before a
  single optimizer step, and after the schedule they must still clear the
  fidelity certificate's per-atom (1.0 mHa) and per-atomization (1.0 kcal/mol)
  tolerances. That is the property the campaign depends on and the only one
  that gates.
* UNANCHORED, in the same DFS coordinates. Its numbers are RECORDED rather than
  gated: they are the measurement of what the coordinates alone deliver, which
  is what the anchor has to be read against. They are held only to a loose
  factor-of-two band around what was measured, so a regression shows without
  the board turning into a fit of the training noise.

The energy errors are the ones the fidelity certificate bounds -- per-system
``E_xc^NN - E_xc^parent``, summed over exchange and correlation, folded into
atomization energies against the free atoms of the same set -- computed from
the checkpoint on disk by the energy-weight probe's own machinery
(``hpcjobs/probe_pretrain_energy_weight.py``), not restated here.

Usage::

    python scripts/pretrain_board_local.py                 # the whole board
    python scripts/pretrain_board_local.py --arch deep_3x16
    python scripts/pretrain_board_local.py --steps 200 --threads 4

Exit status is 0 only when every gated row passes.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# The session's dataset identity
# ---------------------------------------------------------------------------

#: The smallest set that carries what the board has to measure: an open-shell
#: free atom (H, Li, O, N), a closed-shell molecule (H2O, LiH) and a molecule
#: whose atoms are all open shells (N2). Every element of every molecule has
#: its free atom in the set, so every atomization energy is defined.
BOARD_SYSTEMS = (
    ("H", 1),
    ("Li", 1),
    ("N", 3),
    ("O", 2),
    {"name": "H2O", "atom": "O 0.0 0.0 0.0; H 0.0 0.757 0.587; "
                            "H 0.0 -0.757 0.587", "charge": 0, "spin": 0},
    {"name": "LiH", "atom": "Li 0.0 0.0 0.0; H 0.0 0.0 1.5949",
     "charge": 0, "spin": 0},
    {"name": "N2", "atom": "N 0.0 0.0 0.0; N 0.0 0.0 1.0977",
     "charge": 0, "spin": 0},
)

#: sto-3g at grid level 3 is the cheapest identity that still poses every
#: quantity the board reads. The BASIS is what is cut to the bone; the GRID is
#: not, because the set carries the spatially degenerate free atoms O and N,
#: whose rows below level 3 are one arbitrary member of the P-term manifold --
#: locked draws at level 1 differ by of order unity in the iso-orbital
#: indicator, and the generator refuses to write a file whose manifest would
#: record an identity it does not have. It is NOT a production identity: the
#: numbers are a statement about the fit converging on this workstation, not
#: about the campaign's accuracy.
BOARD_BASIS = "sto-3g"
BOARD_GRID_LEVEL = 3

#: The certificate's binding tolerances (``SPEC_pretrain_fidelity_program.md``
#: Section 3.3), which the anchored rows are gated on after the schedule.
TOL_ATOM_MHA = 1.0
TOL_AE_KCAL = 1.0

#: The floor an anchored model has to sit at BEFORE any optimizer step: at
#: ``gated = 0`` the model is its parent, and the residual is the difference
#: between the JAX parent and the stored targets, which libxc produced.
#:
#: What sets it is the CORRELATION BASELINE, not the fit. The stored
#: ``e_lda_c`` column is libxc's ``LDA_C_PW`` at spin=1 while the anchored
#: parent divides by the repository's ``pw92c_polarized_scalar``; the two
#: parameter sets agree to 7.5e-9 relative (3.05e-7 at zeta = +-1), so on an
#: E_c of order 0.3 Ha the two conventions differ by of order 1e-6 mHa. The
#: floor is measured here at 6.32e-7 mHa (worst atom) and 7.94e-7 kcal/mol
#: (worst atomization) on the board's own identity, identical across
#: architectures because at ``gated = 0`` the model IS the parent whatever its
#: descriptors are. 1e-5 mHa clears it by 16x and still sits five orders under
#: the certificate.
TOL_INIT_MHA = 1e-5

#: What the UNANCHORED rows are held to: the measured value times this factor.
#: Recorded, not gated in the certificate's sense -- the band exists so a
#: regression in the coordinates shows.
UNANCHORED_MARGIN = 2.0

#: What the DFS coordinates alone deliver, measured on this workstation
#: (4 threads, 300 steps, the identity above, seed 0), in mHa on the worst free
#: atom and kcal/mol on the worst atomization energy. An architecture absent
#: from this table is REPORTED and not held, so adding one to the registry does
#: not turn the board red before it has been measured; the entries below were
#: recorded 2026-08-25 and are what a regression would show against.
#:
#: Read them beside the anchored rows, which sit at 3e-4 mHa and 2e-4 kcal/mol
#: after the same schedule: an unanchored fit in these coordinates is 1.4 to
#: 6.7 mHa per free atom and 1.2 to 2.5 kcal/mol per atomization energy, i.e.
#: OUTSIDE the certificate at 300 steps, which is the measurement the anchor
#: exists to remove.
UNANCHORED_REFERENCE: dict = {
    "deep": {"max_atom_mHa": 5.9677, "max_dAE_kcal": 2.0465},
    "deep_3x16": {"max_atom_mHa": 6.6742, "max_dAE_kcal": 2.4636},
    "deep_attn": {"max_atom_mHa": 1.4230, "max_dAE_kcal": 1.2328},
}


def _load_probe():
    """The energy-weight probe, loaded by path: ``hpcjobs`` is a deployment
    directory beside the package rather than an importable one."""
    path = os.path.join(REPO, "hpcjobs", "probe_pretrain_energy_weight.py")
    if not os.path.exists(path):
        raise RuntimeError(
            f"the per-system energy-error machinery is not in this checkout "
            f"({path}); the board reads it rather than restating it")
    spec = importlib.util.spec_from_file_location(
        "_pretrain_board_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def board_architectures(include_meta_gga=False):
    """The architectures the board covers, in registry order.

    The meta-GGA entries are excluded until the SCAN parent lands: an anchored
    meta-GGA architecture is refused at construction in the PBE commit, and an
    unanchored one has no parent to be measured against on this board.
    """
    import xcquinox.alec as alec
    from xcquinox.alec.config import ArchitectureConfig
    names = sorted(alec.ARCHITECTURES)
    if include_meta_gga:
        return tuple(names)
    return tuple(n for n in names
                 if not ArchitectureConfig.is_meta_gga(alec.ARCHITECTURES[n]))


def board_arch(name, *, anchor, coordinates="dfs"):
    """The registry entry as the board runs it: polarized correlation, the DFS
    coordinates, and the anchor on or off."""
    import dataclasses
    import xcquinox.alec as alec
    from xcquinox.alec.config import anchored
    arch = dataclasses.replace(alec.get_architecture(name),
                               use_polarized_correlation=True,
                               descriptor_coordinates=coordinates)
    return anchored(arch) if anchor else arch


def ensure_board_data(data_dir, *, systems=BOARD_SYSTEMS, basis=BOARD_BASIS,
                      grid_level=BOARD_GRID_LEVEL, reference_xc="pbe",
                      progress=False):
    """Generate (or reuse) the session's one pretraining file.

    Idempotent through the generator's own manifest identity check, so a second
    architecture in the same session pays nothing. The orientation lock is the
    production one: the O and N free atoms are spatially degenerate, and
    without it their rows are one arbitrary member of the P-term manifold.

    ``reference_xc`` is the parent whose SELF-CONSISTENT density the rows sit
    on -- "pbe" for the GGA rung, which is the whole board until the SCAN
    parent lands, and "scan" for the meta-GGA rows after it. The two are
    different densities and are written under different names, so a board that
    covers both rungs generates two files.
    """
    from xcquinox.alec.pretrain_data_gen import (
        PRETRAIN_ORIENTATION_LOCK_STRENGTH, ensure_pretrain_data)
    os.makedirs(data_dir, exist_ok=True)
    return ensure_pretrain_data(
        data_dir, atoms=tuple(systems), basis=basis, grid_level=grid_level,
        polarized=True, descriptors=True, dfs_set=False, pool_atoms=False,
        reference_xc=reference_xc, exchange_footing="spin_channel",
        orientation_lock_strength=PRETRAIN_ORIENTATION_LOCK_STRENGTH,
        allow_irreproducible_degenerate=False, progress=progress)


def measure(arch, data_path, checkpoint_dir, *, seed, probe=None):
    """The per-system energy errors of a checkpoint, summarized.

    Both the per-system table and the summary come from the probe, so the
    board and the energy-weight sweep read the same quantity through the same
    code.
    """
    probe = probe or _load_probe()
    delta_x, delta_c, systems = probe.per_system_energy_errors(
        arch, data_path, checkpoint_dir, seed=seed)
    names = probe.system_names(systems, delta_x.shape[0])
    return probe.summarize_energy_errors(delta_x, delta_c, systems, names)


def run_cell(name, *, anchor, data_path, work_dir, steps, seed,
             coordinates="dfs", probe=None):
    """One architecture, one anchor state: pretrain and measure.

    Returns a row carrying the measurement at INITIALIZATION (before any
    optimizer step) and after the schedule, with the wall clock of the fit.
    """
    import equinox as eqx

    from xcquinox.alec.config import PretrainSpec
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.pretrain import run_pretrain

    probe = probe or _load_probe()
    arch = board_arch(name, anchor=anchor, coordinates=coordinates)
    tag = f"{name}_{'anchored' if anchor else 'plain'}"
    checkpoint_dir = os.path.join(work_dir, tag)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # The untrained networks, written through the pretrain stage's own
    # serialization, so the initialization row is measured off disk exactly as
    # the trained one is.
    init_dir = os.path.join(work_dir, tag + "_init")
    os.makedirs(init_dir, exist_ok=True)
    xnet, cnet = create_network_pair(arch, seed=seed)
    eqx.tree_serialise_leaves(os.path.join(init_dir, "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(os.path.join(init_dir, "cnet.eqx"), cnet)
    init = measure(arch, data_path, init_dir, seed=seed, probe=probe)

    spec = PretrainSpec(arch=arch, data_dir=os.path.dirname(data_path),
                        checkpoint_dir=checkpoint_dir, n_steps=int(steps),
                        seed=seed, loss_weighting="integration",
                        energy_term_weight=0.0, parent_density="auto",
                        validation_fraction=0.0)
    started = time.time()
    metadata = run_pretrain(spec)
    wall = time.time() - started
    final = measure(arch, data_path, checkpoint_dir, seed=seed, probe=probe)

    return {
        "arch": name,
        "anchored": bool(anchor),
        "coordinates": coordinates,
        "steps": int(metadata.get("pretrain_steps_run", steps)),
        "loss_x": float(metadata["final_loss_x"]),
        "loss_c": float(metadata["final_loss_c"]),
        "init_max_atom_mHa": init["max_atom_dE_xc_mHa"],
        "init_max_dAE_kcal": init["max_dAE_kcal"],
        "init_max_mHa": init["max_dE_xc_mHa"],
        "max_atom_mHa": final["max_atom_dE_xc_mHa"],
        "max_dAE_kcal": final["max_dAE_kcal"],
        "max_mHa": final["max_dE_xc_mHa"],
        "rms_mHa": final["rms_dE_xc_mHa"],
        "worst_atom": final["worst_atom"],
        "worst_ae_system": final["worst_ae_system"],
        "wall_s": wall,
    }


def _finite(value):
    return value is not None and math.isfinite(float(value))


def verdict(row):
    """``(passed, [reason, ...])`` for one row.

    An ANCHORED row is gated three ways: at initialization it must sit at the
    oracle floor on both the atom and the atomization measures, and after the
    schedule it must clear the certificate's tolerances. An UNANCHORED row is
    gated only against its recorded value, and only where one exists.
    """
    reasons = []
    if row["anchored"]:
        for key, tol in (("init_max_atom_mHa", TOL_INIT_MHA),
                         ("init_max_dAE_kcal", TOL_INIT_MHA)):
            value = row[key]
            if not _finite(value):
                reasons.append(f"{key} is {value!r}")
            elif abs(float(value)) > tol:
                reasons.append(
                    f"{key} = {float(value):.3e} exceeds the initialization "
                    f"floor {tol:g} (an anchored model IS its parent at step 0)")
        for key, tol, unit in (("max_atom_mHa", TOL_ATOM_MHA, "mHa"),
                               ("max_dAE_kcal", TOL_AE_KCAL, "kcal/mol")):
            value = row[key]
            if not _finite(value):
                reasons.append(f"{key} is {value!r}")
            elif abs(float(value)) > tol:
                reasons.append(
                    f"{key} = {float(value):.4f} {unit} exceeds the "
                    f"certificate tolerance {tol:g} {unit}")
    else:
        reference = UNANCHORED_REFERENCE.get(row["arch"])
        if reference is None:
            reasons.append("no recorded value for this architecture "
                           "(reported, not held)")
            return True, reasons
        for key in ("max_atom_mHa", "max_dAE_kcal"):
            want = reference.get(key)
            value = row[key]
            if want is None or not _finite(value):
                continue
            if abs(float(value)) > UNANCHORED_MARGIN * abs(float(want)):
                reasons.append(
                    f"{key} = {float(value):.4f} is more than "
                    f"{UNANCHORED_MARGIN:g}x the recorded {float(want):.4f}")
    return (not any("exceeds" in r or "is more than" in r or "is None" in r
                    for r in reasons)), reasons


_COLUMNS = (
    ("arch", 26, "s"),
    ("anchored", 8, "s"),
    ("steps", 6, "d"),
    ("loss_x", 11, ".3e"),
    ("loss_c", 11, ".3e"),
    ("init_max_atom_mHa", 18, ".3e"),
    ("init_max_dAE_kcal", 18, ".3e"),
    ("max_atom_mHa", 13, ".4f"),
    ("max_dAE_kcal", 13, ".4f"),
    ("rms_mHa", 10, ".4f"),
    ("wall_s", 8, ".1f"),
    ("verdict", 8, "s"),
)


def format_table(rows):
    """The whole board as one table."""
    head = "  ".join(f"{name:>{width}}" for name, width, _fmt in _COLUMNS)
    lines = [head, "-" * len(head)]
    for row in rows:
        cells = []
        for name, width, fmt in _COLUMNS:
            value = row.get(name)
            if name == "anchored":
                text = "yes" if value else "no"
            elif value is None:
                text = "-"
            elif fmt == "s":
                text = str(value)
            else:
                text = format(float(value) if fmt != "d" else int(value), fmt)
            cells.append(f"{text:>{width}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def run_board(*, archs=None, steps=300, seed=0, work_dir, data_dir=None,
              coordinates="dfs", include_unanchored=True, progress=False,
              log=print):
    """Every architecture, both anchor states. Returns ``(rows, ok)``."""
    probe = _load_probe()
    data_dir = data_dir or os.path.join(work_dir, "data")
    data_path = ensure_board_data(data_dir, progress=progress)
    names = tuple(archs) if archs else board_architectures()
    rows = []
    ok = True
    for name in names:
        states = (True, False) if include_unanchored else (True,)
        for anchor in states:
            row = run_cell(name, anchor=anchor, data_path=data_path,
                           work_dir=work_dir, steps=steps, seed=seed,
                           coordinates=coordinates, probe=probe)
            passed, reasons = verdict(row)
            row["verdict"] = "PASS" if passed else "FAIL"
            row["reasons"] = reasons
            ok = ok and passed
            log(f"[board] {name} "
                f"{'anchored' if anchor else 'plain'}: {row['verdict']} "
                f"init_atom={row['init_max_atom_mHa']!r} "
                f"atom={row['max_atom_mHa']!r} mHa "
                f"AE={row['max_dAE_kcal']!r} kcal/mol "
                f"wall={row['wall_s']:.1f}s")
            for reason in reasons:
                log(f"[board]   {reason}")
            rows.append(row)
    return rows, ok


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", action="append", default=None,
                   help="architecture to run (repeatable); default every "
                        "registered GGA architecture")
    p.add_argument("--steps", type=int, default=300,
                   help="pretraining steps per architecture (default 300)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", default=None,
                   help="directory for the dataset and the checkpoints "
                        "(default: a temporary directory)")
    p.add_argument("--coordinates", default="dfs",
                   choices=("dfs", "legacy"))
    p.add_argument("--threads", type=int, default=4,
                   help="OpenMP/BLAS worker count (default 4)")
    p.add_argument("--anchored-only", action="store_true",
                   help="skip the unanchored control rows")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(int(args.threads)))
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    work_dir = args.work_dir
    tmp = None
    if work_dir is None:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="pretrain_board_")
        work_dir = tmp
    os.makedirs(work_dir, exist_ok=True)

    def _log(message):
        if not args.quiet:
            print(message, flush=True)

    started = time.time()
    rows, ok = run_board(archs=args.arch, steps=args.steps, seed=args.seed,
                         work_dir=work_dir, coordinates=args.coordinates,
                         include_unanchored=not args.anchored_only,
                         progress=not args.quiet, log=_log)
    print()
    print(f"Local pretraining board -- {BOARD_BASIS}, grid level "
          f"{BOARD_GRID_LEVEL}, polarized, {args.steps} steps, "
          f"{len(BOARD_SYSTEMS)} systems, {args.threads} threads")
    print(format_table(rows))
    failed = [r for r in rows if r["verdict"] == "FAIL"]
    print()
    print(f"{len(rows) - len(failed)}/{len(rows)} rows pass; "
          f"total wall {time.time() - started:.1f} s"
          + (f"; work dir {work_dir}" if tmp is None else ""))
    for row in failed:
        for reason in row["reasons"]:
            print(f"FAIL {row['arch']} "
                  f"{'anchored' if row['anchored'] else 'plain'}: {reason}")
    return 0 if ok else 1


if __name__ == "__main__":
    # The shared hard exit: JAX's atexit backend cleanup corrupts the glibc
    # heap during interpreter teardown ("double free or corruption", SIGABRT,
    # 134 through a shell) AFTER the board has printed its table, so a run
    # whose every row passed still hands 134 to whatever launched it. Measured
    # on this very script before the change.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
