#!/usr/bin/env python
"""Multi-mode constraint/pretraining evaluation across the self-consistency ladder.

Extends ``constraint_pretrain_gmtkn55_demo.py`` from a single fixed-rho evaluation to
THREE evaluation modes on the frozen-PBE starting density, with increasing
self-consistency:

  - **fixed_rho** : NN XC energy on the frozen PBE density (0 Roothaan steps) --
    ``oneshot.fixed_density_total_energy``.
  - **one_shot**  : ONE Roothaan step -- ``run_scf(FIXED_J, max_cycles=1, FROZEN)``
    (build the NN Fock from the PBE density, J pinned to J[rho_PBE], diagonalize once).
  - **3step**     : 3-cycle SCF -- ``run_scf(FULL, max_cycles=3, REASSEMBLE)``
    (J + descriptor features rebuilt each cycle).

For each mode x constraint level it computes, for the 3 metrics (BH76 reaction
energy, per-species |E-E_PBE|, W4-11 atomization):
  - **random**: an independent 16-seed sweep -> mean / worst / std (the same 16
    random initializations are run through every mode -- a paired comparison; each
    mode gets its own full 16-seed evaluation), plus the SCF **divergence rate**
    (fraction of seeds whose energies go non-finite -- a result in itself: the LO
    bound is expected to stabilize otherwise-divergent random functionals).
  - **pretrained**: the single-seed cloned net, plus a per-species |E-E_PBE|
    breakdown grouped into {pretrain atoms, other atoms, molecules} and a
    per-W4-11-reaction AE decomposition (molecule term vs sum-of-atoms term).

Scope: the spin-polarized 1000-step config (production-relevant; reuses the demo's
polarized builders). Unpolarized is a documented follow-up (needs its own,
non-zeta pretrain-data file). Results are written incrementally to JSON so a crash
keeps partial data; it is a long job, so it prints progress.

Run:
    python notebooks/analysis/multimode_constraint_eval.py            # full (16 seeds)
    python notebooks/analysis/multimode_constraint_eval.py --seeds 2 --species-limit 4  # smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# JAX fp64 + CPU BEFORE any jax-backed import (mirrors the harness workers).
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from xcquinox.alec.solver import (  # noqa: E402
    run_scf, SolverConfig, SolverMode, FeaturePolicy, SolverBackend,
)

# Mirror the demo's constants here (hardcoded) so importing this module for the
# pure helpers / unit tests does NOT require importing the demo module or pyscf.
# (verified against constraint_pretrain_gmtkn55_demo: KCAL_PER_HA and
# PRETRAIN_ATOMS = (("H",1),("He",0),("N",3),("O",2)).)
KCAL_PER_HA = 627.5094740631
PRETRAIN_ATOM_SYMS = frozenset({"H", "He", "N", "O"})
MODES = ("fixed_rho", "one_shot", "3step")
_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def solver_config_for_mode(mode: str):
    """Map a mode name to its SolverConfig, or ``None`` for the fixed-rho fast
    path (which uses ``fixed_density_total_energy`` directly, no SCF)."""
    if mode == "fixed_rho":
        return None
    if mode == "one_shot":
        return SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
                            max_cycles=1, feature_policy=FeaturePolicy.FROZEN)
    if mode == "3step":
        return SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                            max_cycles=3, feature_policy=FeaturePolicy.REASSEMBLE)
    raise ValueError(f"unknown mode {mode!r} (expected one of {MODES})")


def group_species(deviation_by_name: dict, atom_element_by_name: dict) -> dict:
    """Group per-species deviations into pretrain-atoms / other-atoms / molecules.

    ``atom_element_by_name`` maps a *single-atom* species name -> its element
    symbol; names absent from it are molecules. Pure."""
    groups = {"pretrain_atoms": {}, "other_atoms": {}, "molecules": {}}
    for name, dev in deviation_by_name.items():
        elem = atom_element_by_name.get(name)
        if elem is None:
            groups["molecules"][name] = dev
        elif elem in PRETRAIN_ATOM_SYMS:
            groups["pretrain_atoms"][name] = dev
        else:
            groups["other_atoms"][name] = dev
    return groups


def reaction_mae_robust(energies: dict, reactions: list) -> float:
    """MAE (kcal/mol) over reactions whose every species energy is finite.

    Skips reactions touching a diverged (non-finite) species so one bad species
    does not poison the whole metric. NaN if no reaction is fully finite."""
    errs = []
    for rxn in reactions:
        names = list(rxn["reactants"]) + list(rxn["products"])
        es = [energies[n] for n in names]
        if not all(math.isfinite(e) for e in es):
            continue
        de = sum(c * e for c, e in zip(rxn["coeffs"], es))
        errs.append(abs(de * KCAL_PER_HA - rxn["reaction_energy_ref"]))
    return float(np.mean(errs)) if errs else float("nan")


def pbe_dev_mae_robust(energies: dict, e_pbe_by_name: dict) -> float:
    """Mean |E_nn - E_PBE| (kcal/mol) over species with finite energy. NaN if none."""
    errs = [abs(energies[n] - e_pbe_by_name[n]) * KCAL_PER_HA
            for n in energies if math.isfinite(energies[n])]
    return float(np.mean(errs)) if errs else float("nan")


def aggregate_seed_metrics(per_seed: list, metric_keys) -> dict:
    """Aggregate a list of per-seed metric dicts into mean/worst/std over the
    seeds with a FINITE value, plus n_used / n_total / divergence_rate per metric.

    ``per_seed`` is a list of ``{metric_key: value}`` (value may be NaN when that
    seed diverged for that metric). Pure."""
    out = {}
    n_total = len(per_seed)
    for key in metric_keys:
        vals = [d[key] for d in per_seed if key in d and math.isfinite(d[key])]
        n_used = len(vals)
        out[key] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "worst": float(np.max(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
            "n_used": n_used,
            "n_total": n_total,
            "divergence_rate": (n_total - n_used) / n_total if n_total else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# evaluation (compute) helpers
# ---------------------------------------------------------------------------

def species_energies_mode(model, mol_data_by_name: dict, mode: str) -> dict:
    """Per-species total energy (Ha) for ``model`` through ``mode``. A species
    that raises or yields a non-finite energy is recorded as NaN (diverged)."""
    import xcquinox.alec as alec  # lazy: pulls pyscf, only needed at compute time
    cfg = solver_config_for_mode(mode)
    out = {}
    for name, md in mol_data_by_name.items():
        try:
            if cfg is None:
                e = float(alec.fixed_density_total_energy(model, md))
            else:
                e = float(run_scf(cfg, model, md).total_energy)
        except Exception:
            e = float("nan")
        out[name] = e if math.isfinite(e) else float("nan")
    return out


def metrics_from_energies(energies, bh76_rxns, w411_rxns, e_pbe_by_name) -> dict:
    return {
        "bh76": reaction_mae_robust(energies, bh76_rxns),
        "pbe_dev": pbe_dev_mae_robust(energies, e_pbe_by_name),
        "w411_ae": reaction_mae_robust(energies, w411_rxns),
    }


def atom_element_map(mol_specs: dict) -> dict:
    """name -> element symbol for SINGLE-atom species (molecules omitted), using
    ``MoleculeSpec.atom_composition`` ({element: count})."""
    out = {}
    for name, spec in mol_specs.items():
        comp = dict(spec.atom_composition)
        if sum(comp.values()) == 1:
            out[name] = next(iter(comp))
    return out


def per_species_deviation(energies, e_pbe_by_name) -> dict:
    return {n: (abs(energies[n] - e_pbe_by_name[n]) * KCAL_PER_HA
                if math.isfinite(energies[n]) else float("nan"))
            for n in energies}


_METRIC_KEYS = ("bh76", "pbe_dev", "w411_ae")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    # Lazy imports (pull pyscf / the demo module) — kept out of module scope so the
    # pure helpers + unit tests import cheaply.
    sys.path.insert(0, _HERE)
    import constraint_pretrain_gmtkn55_demo as demo  # noqa: E402
    import xcquinox.alec as alec  # noqa: E402

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=["polarized", "unpolarized"],
                   default="polarized", help="network config (default: polarized)")
    p.add_argument("--seeds", type=int, default=demo.N_SEEDS,
                   help="random-init seeds per mode (default: %(default)s)")
    p.add_argument("--pretrain-steps", type=int, default=demo.PRETRAIN_N_STEPS,
                   help="pretraining steps (default: %(default)s)")
    p.add_argument("--modes", default=",".join(MODES),
                   help="comma list of modes (default: all three)")
    p.add_argument("--species-limit", type=int, default=None,
                   help="(smoke) cap the number of species precomputed")
    p.add_argument("--out", default=os.path.join(_HERE, "demo_logs",
                                                  "multimode_polarized.json"))
    args = p.parse_args(argv)

    if args.config == "unpolarized":
        raise NotImplementedError(
            "unpolarized config needs its own non-zeta pretrain_data.npz; this "
            "driver currently scopes to the polarized config (the demo's builders "
            "are polarized). Unpolarized is a documented follow-up.")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        solver_config_for_mode(m)  # validate early

    t0 = time.time()
    workdir = os.path.join(demo.OUTDIR, "multimode")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("[1/4] Building eval pools (BH76-RC + W4-11) ...", flush=True)
    bh76_specs, bh76_rxns = demo.build_bh76_pool()
    w411_specs, w411_rxns = demo.build_w411_ae_pool()
    mol_specs = {**bh76_specs, **w411_specs}
    if args.species_limit is not None:
        keep = set(list(mol_specs)[:args.species_limit])
        mol_specs = {k: v for k, v in mol_specs.items() if k in keep}
        bh76_rxns = [r for r in bh76_rxns
                     if set(r["reactants"]) | set(r["products"]) <= keep]
        w411_rxns = [r for r in w411_rxns
                     if set(r["reactants"]) | set(r["products"]) <= keep]
    atom_elem = atom_element_map(mol_specs)
    print(f"      {len(mol_specs)} species "
          f"({sum(1 for n in mol_specs if n in atom_elem)} atoms); "
          f"{len(bh76_rxns)} BH76, {len(w411_rxns)} W4-11", flush=True)

    print("[2/4] Precomputing PBE SCF + grids + ERI (once per species) ...", flush=True)
    mol_data, e_pbe = {}, {}
    for i, (name, spec) in enumerate(mol_specs.items(), 1):
        md = alec.precompute_fixed_density_data(spec, descriptors=(),
                                                required_keys=("eri",))
        mol_data[name] = md
        e_pbe[name] = float(md["E_pbe"])
        print(f"      [precompute {i}/{len(mol_specs)}] {name}", flush=True)

    print("[3/4] Generating polarized pretrain data ...", flush=True)
    data_dir = os.path.join(demo.OUTDIR, "pretrain_data")
    demo.generate_pretrain_data(data_dir)

    levels = demo.make_constraint_levels()
    n_cells = len(modes) * len(levels)
    results = {
        "config": args.config, "seeds": args.seeds,
        "pretrain_steps": args.pretrain_steps, "modes": modes,
        "n_species": len(mol_specs), "metrics": list(_METRIC_KEYS),
        "pbe_baseline": metrics_from_energies(
            {n: e_pbe[n] for n in mol_specs}, bh76_rxns, w411_rxns, e_pbe),
        "cells": {},
    }

    # Pretrain each constrained level ONCE (mode-independent); evaluate per mode.
    print("[4/4] Pretraining constrained levels (once) ...", flush=True)
    pretrained = {}
    for label, spec in levels:
        if spec is None:
            continue
        ckpt = os.path.join(workdir, "pretrain",
                            label.replace("+", "").replace("(", "").replace(")", "").strip())
        if args.pretrain_steps != demo.PRETRAIN_N_STEPS:
            demo.PRETRAIN_N_STEPS = args.pretrain_steps  # honor CLI override
        pretrained[label] = demo.pretrain_and_load(spec, data_dir, ckpt)
        print(f"      pretrained {label}", flush=True)

    # Build the 16 random models once per level, evaluate each through every mode.
    print(f"Evaluating {n_cells} cells "
          f"({args.seeds} random seeds x {len(levels)} levels x {len(modes)} modes) ...",
          flush=True)
    cell = 0
    for label, spec in levels:
        rand_models = [demo.build_random_model(spec, s) for s in range(args.seeds)]
        for mode in modes:
            cell += 1
            tcell = time.time()
            per_seed = []
            n_div = 0
            for s, model in enumerate(rand_models):
                en = species_energies_mode(model, mol_data, mode)
                if any(not math.isfinite(v) for v in en.values()):
                    n_div += 1
                per_seed.append(metrics_from_energies(en, bh76_rxns, w411_rxns, e_pbe))
            agg = aggregate_seed_metrics(per_seed, _METRIC_KEYS)
            entry = {"random": agg,
                     "random_any_species_divergence_rate": n_div / max(args.seeds, 1)}
            if spec is not None:
                pen = species_energies_mode(pretrained[label], mol_data, mode)
                entry["pretrained"] = metrics_from_energies(
                    pen, bh76_rxns, w411_rxns, e_pbe)
                dev = per_species_deviation(pen, e_pbe)
                grouped = group_species(dev, atom_elem)
                entry["pretrained_per_species"] = {
                    grp: {n: round(v, 4) for n, v in d.items()}
                    for grp, d in grouped.items()}
            results["cells"].setdefault(mode, {})[label] = entry
            # incremental write so a crash keeps partial data
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{cell}/{n_cells}] mode={mode} {label}: "
                  f"random pbe_dev mean={agg['pbe_dev']['mean']:.2f} "
                  f"div={entry['random_any_species_divergence_rate']:.0%} "
                  f"({time.time() - tcell:.0f}s)", flush=True)

    results["elapsed_s"] = time.time() - t0
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {results['elapsed_s']:.0f}s. Results -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
