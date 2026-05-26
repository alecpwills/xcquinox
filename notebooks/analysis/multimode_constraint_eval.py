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

def _one_energy(alec, cfg, model, md) -> float:
    """Total energy (Ha) of one model on one species' precomputed ``md`` through
    the given solver ``cfg`` (None ⇒ fixed-ρ). Non-finite / exception ⇒ NaN
    (recorded as diverged)."""
    try:
        if cfg is None:
            e = float(alec.fixed_density_total_energy(model, md))
        else:
            e = float(run_scf(cfg, model, md).total_energy)
    except Exception:
        e = float("nan")
    return e if math.isfinite(e) else float("nan")


def species_energies_mode(model, mol_data_by_name: dict, mode: str) -> dict:
    """Per-species total energy (Ha) for ``model`` through ``mode``."""
    import xcquinox.alec as alec  # lazy: pulls pyscf, only needed at compute time
    cfg = solver_config_for_mode(mode)
    return {name: _one_energy(alec, cfg, model, md)
            for name, md in mol_data_by_name.items()}


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
WEIGHTINGS = ("unweighted", "integration")


def steps_to_converge(trajectory, frac: float = 1.05):
    """First step (1-based) whose loss <= ``frac`` * min(loss) over the trajectory
    — a simple "how quickly it converged" measure. NaN for an empty/all-NaN
    trajectory. Pure."""
    t = np.asarray(trajectory, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return float("nan")
    thresh = frac * float(np.min(t))
    return int(np.argmax(t <= thresh)) + 1  # argmax = first True; min always <= thresh


def _convergence_from(md: dict, ckpt_dir: str) -> dict:
    """Assemble a convergence record from run_pretrain's return dict + the saved
    losses_{x,c}.npy trajectories."""
    out = {k: md.get(k) for k in ("final_loss_x", "final_loss_c",
                                  "min_loss_x", "min_loss_c", "duration_seconds")}
    for net in ("x", "c"):
        path = os.path.join(ckpt_dir, f"losses_{net}.npy")
        if os.path.isfile(path):
            traj = np.load(path)
            out[f"steps_to_converge_{net}"] = steps_to_converge(traj)
            out[f"n_steps_{net}"] = int(np.asarray(traj).size)
    return out


def should_reuse_checkpoint(meta: dict, n_steps: int, weighting: str) -> bool:
    """True iff a saved pretrain_metadata dict matches the requested ``n_steps``
    AND ``loss_weighting`` — so a shorter (e.g. smoke) or other-weighting checkpoint
    is never silently reused. Pure."""
    return (meta.get("pretrain_steps") == n_steps
            and meta.get("loss_weighting") == weighting)


def _build_arch(demo, label, x_constraints, c_constraints, polarized):
    """A no-descriptor GGA arch with the given constraints, polarized or not."""
    safe = (label.replace("+", "").replace("(", "").replace(")", "").strip()
            or "base")
    return demo.ArchitectureConfig.from_spec(
        safe, demo.DEPTH, demo.NODES,
        x_constraints=list(x_constraints), c_constraints=list(c_constraints),
        use_polarized_correlation=polarized)


def build_random_model(demo, spec, seed, polarized):
    """Random-init model for a level (config-aware). ``spec is None`` ⇒ truly
    unconstrained (lob_lim=None, no constraints); the cnet is spin-polarization-
    aware iff ``polarized``."""
    if spec is None:
        xnet = demo.AlecGGA_XNet(n_extra_features=0, depth=demo.DEPTH,
                                 nodes=demo.NODES, seed=seed, lob_lim=None)
        cnet = demo.AlecGGA_CNet(n_extra_features=0, depth=demo.DEPTH,
                                 nodes=demo.NODES, seed=seed + 1, lob_lim=None,
                                 use_spin_polarization=polarized)
        base = demo.ArchitectureConfig.from_spec(
            "base", demo.DEPTH, demo.NODES, use_polarized_correlation=polarized)
        return demo.AlecGGAModel.from_arch(base, xnet=xnet, cnet=cnet)
    x_constraints, c_constraints = spec
    return demo.AlecGGAModel.from_arch(
        _build_arch(demo, "lvl", x_constraints, c_constraints, polarized), seed=seed)


def _pretrain_one(demo, alec, level_spec, weighting, data_dir, ckpt_dir, n_steps,
                  seed, polarized, reuse=True):
    """Pretrain (or reuse) ONE (level, weighting); return (AlecGGAModel, conv dict).

    ``level_spec`` is None for the truly-unconstrained level (built lob_lim=None and
    pretrained via the run_pretrain ``networks=`` override, since
    create_network_pair cannot express lob_lim=None), else ``(x_constraints,
    c_constraints)``. ``polarized`` selects the spin-polarized vs unpolarized arch
    (which also picks the pretrain-data file inside run_pretrain). ``reuse=True``
    skips pretraining when ``ckpt_dir`` already has xnet.eqx/cnet.eqx +
    pretrain_metadata.json (reads convergence from disk)."""
    eqx = demo.eqx
    if level_spec is None:
        arch = demo.ArchitectureConfig.from_spec(
            "base", demo.DEPTH, demo.NODES, use_polarized_correlation=polarized)
        mk_x = lambda: demo.AlecGGA_XNet(n_extra_features=0, depth=demo.DEPTH,
                                         nodes=demo.NODES, seed=seed, lob_lim=None)
        mk_c = lambda: demo.AlecGGA_CNet(n_extra_features=0, depth=demo.DEPTH,
                                         nodes=demo.NODES, seed=seed + 1,
                                         lob_lim=None, use_spin_polarization=polarized)
        networks = (mk_x(), mk_c())          # lob_lim=None override
        skel = (mk_x, mk_c)                  # matching reload skeletons
    else:
        x_constraints, c_constraints = level_spec
        arch = _build_arch(demo, "lvl", x_constraints, c_constraints, polarized)
        networks = None                      # build from arch via create_network_pair
        pair = demo.create_network_pair(arch, seed=seed)
        skel = (lambda p=pair[0]: p, lambda p=pair[1]: p)

    # Reuse ONLY when an existing checkpoint was trained for the SAME n_steps AND
    # weighting (else a shorter smoke/other-weighting run would be silently reused).
    meta_path = os.path.join(ckpt_dir, "pretrain_metadata.json")
    have = (os.path.isfile(os.path.join(ckpt_dir, "xnet.eqx"))
            and os.path.isfile(os.path.join(ckpt_dir, "cnet.eqx"))
            and os.path.isfile(meta_path))
    md = None
    if reuse and have:
        with open(meta_path) as f:
            cand = json.load(f)
        if should_reuse_checkpoint(cand, n_steps, weighting):
            md = cand
    if md is None:
        md = alec.run_pretrain(
            alec.PretrainSpec(arch=arch, data_dir=data_dir, checkpoint_dir=ckpt_dir,
                              n_steps=n_steps, loss_weighting=weighting, seed=seed),
            networks=networks)

    xnet = eqx.tree_deserialise_leaves(os.path.join(ckpt_dir, "xnet.eqx"), skel[0]())
    cnet = eqx.tree_deserialise_leaves(os.path.join(ckpt_dir, "cnet.eqx"), skel[1]())
    model = demo.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    return model, _convergence_from(md, ckpt_dir)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    # Lazy imports (pull pyscf / the demo module) — kept out of module scope so the
    # pure helpers + unit tests import cheaply.
    sys.path.insert(0, _HERE)
    import constraint_pretrain_gmtkn55_demo as demo  # noqa: E402
    import xcquinox.alec as alec  # noqa: E402
    import jax  # noqa: E402 — for clear_caches() between species (bound JIT memory)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=["polarized", "unpolarized"],
                   default="polarized", help="network config (default: polarized)")
    p.add_argument("--seeds", type=int, default=demo.N_SEEDS,
                   help="random-init seeds per mode (default: %(default)s)")
    p.add_argument("--pretrain-steps", type=int, default=demo.PRETRAIN_N_STEPS,
                   help="pretraining steps (default: %(default)s)")
    p.add_argument("--modes", default=",".join(MODES),
                   help="comma list of modes (default: all three)")
    p.add_argument("--weightings", default=",".join(WEIGHTINGS),
                   help="comma list of pretraining loss weightings "
                        "(default: unweighted,integration)")
    p.add_argument("--species-limit", type=int, default=None,
                   help="(smoke) cap the number of species precomputed")
    p.add_argument("--fresh-pretrain", action="store_true",
                   help="re-pretrain even if checkpoints exist (default: reuse them)")
    p.add_argument("--out", default=None,
                   help="results JSON (default: demo_logs/multimode_<config>.json)")
    args = p.parse_args(argv)

    polarized = (args.config == "polarized")
    if args.out is None:
        args.out = os.path.join(_HERE, "demo_logs", f"multimode_{args.config}.json")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        solver_config_for_mode(m)  # validate early
    weightings = [w.strip() for w in args.weightings.split(",") if w.strip()]

    t0 = time.time()
    workdir = os.path.join(demo.OUTDIR, "multimode", args.config)
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

    print(f"[3/4] Generating {args.config} pretrain data ...", flush=True)
    data_dir = os.path.join(demo.OUTDIR, "pretrain_data")
    demo.generate_pretrain_data(data_dir, polarized=polarized)

    levels = demo.make_constraint_levels()
    n_cells = len(modes) * len(levels)
    results = {
        "config": args.config, "seeds": args.seeds,
        "pretrain_steps": args.pretrain_steps, "modes": modes,
        "weightings": weightings,
        "n_species": len(mol_specs), "metrics": list(_METRIC_KEYS),
        "pbe_baseline": metrics_from_energies(
            {n: e_pbe[n] for n in mol_specs}, bh76_rxns, w411_rxns, e_pbe),
        "cells": {}, "convergence": {},
    }

    # Pretrain every (weighting x level) ONCE — incl. the truly-unconstrained level
    # (via the run_pretrain networks= override). Mode-independent; evaluated per
    # mode below. Capture the convergence record for each.
    print(f"[4/4] Pretraining {len(weightings)}x{len(levels)} (weighting x level) ...",
          flush=True)
    pretrained = {}
    for w in weightings:
        for label, spec in levels:
            safe = label.replace("+", "").replace("(", "").replace(")", "").strip()
            ckpt = os.path.join(workdir, "pretrain", w, safe)
            model, conv = _pretrain_one(demo, alec, spec, w, data_dir, ckpt,
                                        args.pretrain_steps, demo.SEED, polarized,
                                        reuse=not args.fresh_pretrain)
            pretrained[(w, label)] = model
            results["convergence"].setdefault(w, {})[label] = conv
            print(f"      pretrained [{w}] {label}: "
                  f"final_loss_x={conv.get('final_loss_x'):.4g} "
                  f"steps_x={conv.get('steps_to_converge_x')}", flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    # Build the 16 random models once per level, evaluate each through every mode.
    print(f"Evaluating {n_cells} cells "
          f"({args.seeds} random seeds x {len(levels)} levels x {len(modes)} modes) ...",
          flush=True)
    cell = 0
    for label, spec in levels:
        rand_models = [build_random_model(demo, spec, s, polarized)
                       for s in range(args.seeds)]
        for mode in modes:
            cell += 1
            tcell = time.time()
            cfg = solver_config_for_mode(mode)
            # SPECIES-OUTER: compile the SCF for a species ONCE (reused across all
            # random seeds + pretrained models), then jax.clear_caches() to free
            # that compiled executable before the next species. Bounds the resident
            # XLA executables to ~1 (the full set's 29 distinct shapes otherwise
            # accumulate and segfault the CPU compiler).
            rand_en = [dict() for _ in rand_models]   # per-seed {species: E}
            pre_en = {w: dict() for w in weightings}  # per-weighting {species: E}
            for name, md in mol_data.items():
                for s, model in enumerate(rand_models):
                    rand_en[s][name] = _one_energy(alec, cfg, model, md)
                for w in weightings:
                    pre_en[w][name] = _one_energy(alec, cfg, pretrained[(w, label)], md)
                if mode != "fixed_rho":
                    jax.clear_caches()
            per_seed = [metrics_from_energies(rand_en[s], bh76_rxns, w411_rxns, e_pbe)
                        for s in range(args.seeds)]
            n_div = sum(1 for s in range(args.seeds)
                        if any(not math.isfinite(v) for v in rand_en[s].values()))
            agg = aggregate_seed_metrics(per_seed, _METRIC_KEYS)
            entry = {"random": agg,
                     "random_any_species_divergence_rate": n_div / max(args.seeds, 1)}
            entry["pretrained"] = {}
            entry["pretrained_per_species"] = {}
            for w in weightings:
                entry["pretrained"][w] = metrics_from_energies(
                    pre_en[w], bh76_rxns, w411_rxns, e_pbe)
                grouped = group_species(per_species_deviation(pre_en[w], e_pbe), atom_elem)
                entry["pretrained_per_species"][w] = {
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
