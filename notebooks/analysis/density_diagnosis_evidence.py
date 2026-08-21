#!/usr/bin/env python3
"""Regenerate every number in DENSITY_DIAGNOSIS.md from the pulled run.

Reads only pulled artifacts (aux_log.pkl, eval/per_molecule.json,
eval_holdout*/per_molecule.json, manifest.json) -- no SCF, no model load, no
cluster access. The ``aux_log.pkl`` inputs are training logs this repo's own
harness wrote (``train._run_per_molecule_loop``) and pulled from our cluster
scratch, so the pickle load is inside the trust boundary; nothing here reads
third-party or network-sourced pickles. Run from the repo root:

    python notebooks/analysis/density_diagnosis_evidence.py [--run <run_dir>]

The default run dir is the dfs6311_grid3_v3 pull the diagnosis was written
against; pass --run to re-point it at a later pull.
"""
from __future__ import annotations

import argparse
import json
import pickle  # noqa: S403 -- trusted local training logs
import statistics
from pathlib import Path

import numpy as np

from xcquinox.alec.eval_holdout import assert_channel_not_sliced

# The effective per-channel weights of the per-molecule loop
# (train._DEFAULT_CHANNEL_WEIGHTS); see notebooks/analysis/LOSS_PRIMER.md.
CHANNEL_WEIGHTS = {"loss_AE": 1.0, "loss_BH76": 1.0, "loss_IP13": 1.0,
                   "loss_vxc": 1.0, "loss_rho": 20.0}
CHANNELS = list(CHANNEL_WEIGHTS)
DEFAULT_RUN = (Path.home() / "Documents/Research/xcquinox-results/runs/dfs_step7"
               / "dfs6311_grid3_v3/runs/run_20260728T140018Z")
EDGE_EPOCHS = 5          # window at each end of training
ARTIFACT_EPS = 0.05      # per-electron L1 an order above the population


def _steps(aux):
    """Optimizer-update entries (validation records excluded)."""
    return [e for e in aux
            if "aux" in e and e.get("group") != "__validation__"]


def _edges(steps):
    ep = sorted({e["epoch"] for e in steps})
    first = [e for e in steps if e["epoch"] <= ep[0] + EDGE_EPOCHS - 1]
    last = [e for e in steps if e["epoch"] >= ep[-1] - EDGE_EPOCHS + 1]
    return first, last


def _mean(entries, channel):
    vals = [e["aux"][channel] for e in entries if channel in e["aux"]]
    return sum(vals) / len(vals) if vals else float("nan")


def weight_identity(run: Path) -> None:
    """The loop's effective weights, verified against the recorded totals."""
    print("\n[1] Effective channel weights (loss == sum_k w_k * component_k)")
    worst = 0.0
    n_updates = n_specs = 0
    schemes = set()
    for d in sorted((run / "checkpoints").glob("spec_*")):
        p = d / "aux_log.pkl"
        if not p.is_file():
            continue
        aux = pickle.load(open(p, "rb"))  # noqa: S301
        steps = _steps(aux)
        if not steps:
            continue
        n_specs += 1
        for e in steps:
            n_updates += 1
            total = sum(CHANNEL_WEIGHTS[k] * v for k, v in e["aux"].items())
            worst = max(worst, abs(total - e["loss"])
                        / max(abs(e["loss"]), 1e-300))
        schemes |= {e.get("update_scheme") for e in aux}
    print(f"    {n_updates} updates over {n_specs} specs; "
          f"max relative deviation {worst:.3e}")
    print(f"    update_scheme values: {sorted(s for s in schemes if s)}")


def channel_budget(run: Path) -> None:
    """Per-channel share of the loss, and whether each channel decreases."""
    print("\n[2] Channel budget and trajectory (first vs last "
          f"{EDGE_EPOCHS} epochs)")
    print(f"    {'spec':>10s} {'rho first':>11s} {'rho last':>11s} "
          f"{'ratio':>6s} {'rho share end':>13s} {'AE ratio':>9s}")
    ratios = []
    for d in sorted((run / "checkpoints").glob("spec_*")):
        p = d / "aux_log.pkl"
        if not p.is_file():
            continue
        steps = _steps(pickle.load(open(p, "rb")))  # noqa: S301
        if not steps:
            continue
        first, last = _edges(steps)
        rho_a, rho_b = _mean(first, "loss_rho"), _mean(last, "loss_rho")
        ae_a, ae_b = _mean(first, "loss_AE"), _mean(last, "loss_AE")
        total = sum(e["loss"] for e in last) / len(last)
        ratios.append(rho_b / rho_a if rho_a else float("nan"))
        print(f"    {d.name:>10s} {rho_a:11.4e} {rho_b:11.4e} "
              f"{ratios[-1]:6.3f} {100 * 20 * rho_b / total:12.1f}% "
              f"{ae_b / ae_a if ae_a else float('nan'):9.3f}")
    fell = sum(1 for r in ratios if r < 1.0)
    print(f"    density channel decreased in {fell}/{len(ratios)} specs; "
          f"median ratio {statistics.median(ratios):.3f}")
    # the conclusion must not depend on the window: widen it, and regress
    # over the whole trajectory
    print("    window robustness (the flatness is not an edge-window effect):")
    for width in (5, 10, 20, 50):
        rs = []
        for d in sorted((run / "checkpoints").glob("spec_*")):
            p = d / "aux_log.pkl"
            if not p.is_file():
                continue
            steps = _steps(pickle.load(open(p, "rb")))  # noqa: S301
            if not steps:
                continue
            ep = sorted({e["epoch"] for e in steps})
            f = [e for e in steps if e["epoch"] <= ep[0] + width - 1]
            l = [e for e in steps if e["epoch"] >= ep[-1] - width + 1]  # noqa: E741
            a, b = _mean(f, "loss_rho"), _mean(l, "loss_rho")
            if a:
                rs.append(b / a)
        print(f"      window {width:>2d} epochs: median ratio "
              f"{statistics.median(rs):.4f}   rose in "
              f"{sum(1 for r in rs if r > 1.0)}/{len(rs)}")
    slopes = []
    for d in sorted((run / "checkpoints").glob("spec_*")):
        p = d / "aux_log.pkl"
        if not p.is_file():
            continue
        steps = _steps(pickle.load(open(p, "rb")))  # noqa: S301
        by_ep: dict[int, list] = {}
        for e in steps:
            by_ep.setdefault(e["epoch"], []).append(e["aux"]["loss_rho"])
        xs = sorted(by_ep)
        ys = [sum(by_ep[k]) / len(by_ep[k]) for k in xs]
        if len(xs) < 10:
            continue
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        if den and my:
            slopes.append((sum((x - mx) * (y - my)
                               for x, y in zip(xs, ys)) / den) / my)
    if slopes:
        med = statistics.median(slopes)
        print(f"      OLS over all epochs: median relative slope "
              f"{med:.3e}/epoch => {100 * med * 200:+.2f}% over 200 epochs")


def per_group_attribution(run: Path,
                          specs=("spec_0006", "spec_0008", "spec_0016")) -> None:
    """Which training group carries the density channel."""
    print("\n[3] Per-group density attribution (last "
          f"{EDGE_EPOCHS} epochs)")
    for name in specs:
        p = run / "checkpoints" / name / "aux_log.pkl"
        if not p.is_file():
            continue
        steps = _steps(pickle.load(open(p, "rb")))  # noqa: S301
        _, last = _edges(steps)
        by_group: dict[str, list] = {}
        for e in last:
            by_group.setdefault(e["group"], []).append(e)
        total_rho = sum(e["aux"]["loss_rho"] for e in last)
        print(f"    {name}:")
        ranked = sorted(by_group.items(),
                        key=lambda kv: -sum(x["aux"]["loss_rho"]
                                            for x in kv[1]) / len(kv[1]))
        for group, entries in ranked[:4]:
            mean_rho = sum(x["aux"]["loss_rho"] for x in entries) / len(entries)
            share = 100 * sum(x["aux"]["loss_rho"] for x in entries) / total_rho
            # same estimator (mean) for the value and for its share of the
            # step, so the two numbers on this line describe one quantity
            mean_step = sum(x["loss"] for x in entries) / len(entries)
            of_step = 100 * 20 * mean_rho / mean_step if mean_step else 0.0
            print(f"      {group:<28s} mean rho {mean_rho:.4e}  "
                  f"{share:5.1f}% of the channel  "
                  f"({of_step:.1f}% of its own step's loss)")
        rest = [e for e in last if not e["group"].endswith(":CH")]
        if len(rest) != len(last):
            print(f"      channel WITHOUT the CH group: "
                  f"{sum(e['aux']['loss_rho'] for e in rest) / len(rest):.4e}"
                  f"  (vs {total_rho / len(last):.4e} with)")


def insample_density(run: Path, specs=("spec_0006",)) -> None:
    """NN vs model-free PBE on the TRAINING molecules."""
    print("\n[4] In-sample per-molecule density, NN vs PBE (both vs CCSD)")
    for name in specs:
        p = run / "checkpoints" / name / "eval" / "per_molecule.json"
        if not p.is_file():
            continue
        print(f"    {name}:  {'molecule':>10s} {'eps_NN':>11s} "
              f"{'eps_PBE':>11s} {'NN/PBE':>7s}")
        for r in json.load(open(p)):
            nn, pbe = r.get("density_eps_l1"), r.get("density_eps_l1_pbe")
            if not (isinstance(nn, (int, float))
                    and isinstance(pbe, (int, float)) and pbe > 0):
                continue
            flag = "   <-- artifact scale" if pbe > ARTIFACT_EPS else ""
            print(f"              {r['molecule']:>10s} {nn:11.4e} "
                  f"{pbe:11.4e} {nn / pbe:7.3f}{flag}")


HELDOUT_CHANNELS = ("eval_holdout", "eval_holdout_best",
                    "eval_holdout_val_best")


def heldout_summary(run: Path) -> None:
    """Held-out density: contamination check and checkpoint comparison."""
    print("\n[5] Held-out density")
    # Every channel this function pools is checked before the first read: a
    # species slice covers a handful of species for a workflow test, so its
    # ratios are not the pool's and must not enter the medians below.
    for d in sorted((run / "checkpoints").glob("spec_*")):
        for sub in HELDOUT_CHANNELS:
            assert_channel_not_sliced(d, sub)
    pbe_all: dict[str, list] = {}
    for d in sorted((run / "checkpoints").glob("spec_*")):
        p = d / "eval_holdout_val_best" / "per_molecule.json"
        if not p.is_file():
            continue
        for r in json.load(open(p)):
            pbe = r.get("density_eps_l1_pbe")
            if isinstance(pbe, (int, float)) and pbe > 0:
                pbe_all.setdefault(r["molecule"], []).append(pbe)
    if pbe_all:
        # the model-free PBE error must be spec-invariant; if it is not, a
        # single draw per species would silently hide a degenerate manifold
        # (see [7]), so report the worst spread before summarizing
        spread = max(max(v) / min(v) for v in pbe_all.values())
        by_species = {m: statistics.median(v) for m, v in pbe_all.items()}
        vals = sorted(by_species.values())
        contaminated = [m for m, v in by_species.items() if v > ARTIFACT_EPS]
        print(f"    worst within-species spread across specs: {spread:.4f}x "
              + ("(spec-invariant, so a per-species summary is safe)"
                 if spread < 1.001 else
                 "WARNING: not spec-invariant -- summarizing medians"))
        print(f"    PBE eps over {len(vals)} held-out species: median "
              f"{statistics.median(vals):.4e}, max {vals[-1]:.4e}")
        print(f"    species at artifact scale (> {ARTIFACT_EPS}): "
              f"{len(contaminated)} {sorted(contaminated)}")

    def medians(subdir):
        out = {}
        for d in sorted((run / "checkpoints").glob("spec_*")):
            p = d / subdir / "per_molecule.json"
            if not p.is_file():
                continue
            r = [x["density_eps_l1"] / x["density_eps_l1_pbe"]
                 for x in json.load(open(p))
                 if isinstance(x.get("density_eps_l1"), (int, float))
                 and isinstance(x.get("density_eps_l1_pbe"), (int, float))
                 and x["density_eps_l1_pbe"] > 0]
            if r:
                out[d.name] = statistics.median(r)
        return out

    fin, best, vb = (medians("eval_holdout"), medians("eval_holdout_best"),
                     medians("eval_holdout_val_best"))
    common = sorted(set(fin) & set(vb) & set(best))
    if not common:
        return
    mm = lambda d: statistics.median([d[s] for s in common])  # noqa: E731
    print(f"    median-of-medians NN/PBE eps ratio over {len(common)} specs:")
    print(f"      final {mm(fin):.4f} | train-best {mm(best):.4f} | "
          f"val-best {mm(vb):.4f}  "
          f"(difference {100 * (mm(vb) - mm(fin)) / mm(fin):+.2f}%)")
    paired = [vb[s] - fin[s] for s in common]
    print(f"      paired per-spec (val-best - final): median "
          f"{statistics.median(paired):+.5f}, val-best better in "
          f"{sum(1 for x in paired if x < 0)}/{len(paired)} specs")
    print(f"      specs beating PBE at the final checkpoint: "
          f"{sum(1 for s in common if fin[s] < 1)}/{len(common)}")
    # pooled over species rows, per checkpoint -- do not mix the two
    for label, sub in (("final", "eval_holdout"),
                       ("val-best", "eval_holdout_val_best")):
        rows = []
        for d in sorted((run / "checkpoints").glob("spec_*")):
            p = d / sub / "per_molecule.json"
            if not p.is_file():
                continue
            rows += [x["density_eps_l1"] / x["density_eps_l1_pbe"]
                     for x in json.load(open(p))
                     if isinstance(x.get("density_eps_l1"), (int, float))
                     and isinstance(x.get("density_eps_l1_pbe"), (int, float))
                     and x["density_eps_l1_pbe"] > 0]
        if rows:
            print(f"      pooled species rows ({label}): n={len(rows)} "
                  f"median {statistics.median(rows):.4f}, "
                  f"{100 * sum(1 for x in rows if x < 1) / len(rows):.2f}% "
                  "better than PBE")


def outcome_correlates(run: Path) -> None:
    """What the held-out density outcome tracks: CH presence, architecture,
    and the energy ratio (the trade-off test)."""
    print("\n[6] Held-out density vs CH presence, architecture, and energy")
    manifest = json.load(open(run / "manifest.json"))
    cells = {f"spec_{s['index']:04d}": s["cell"] for s in manifest["specs"]}
    rows = []
    for d in sorted((run / "checkpoints").glob("spec_*")):
        # Both legs of the correlate below are held-out quantities; on a
        # sliced channel both describe a handful of species.
        assert_channel_not_sliced(d, "eval_holdout")
        pm = d / "eval_holdout" / "per_molecule.json"
        pr = d / "eval_holdout" / "per_reaction.json"
        tm = d / "train_metadata.json"
        if not (pm.is_file() and pr.is_file() and tm.is_file()):
            continue
        dens = [x["density_eps_l1"] / x["density_eps_l1_pbe"]
                for x in json.load(open(pm))
                if isinstance(x.get("density_eps_l1"), (int, float))
                and isinstance(x.get("density_eps_l1_pbe"), (int, float))
                and x["density_eps_l1_pbe"] > 0]
        rxn = json.load(open(pr))
        nn = [r["abs_error_nn_kcalmol"] for r in rxn
              if isinstance(r.get("abs_error_nn_kcalmol"), (int, float))]
        pbe = [r["abs_error_pbe_kcalmol"] for r in rxn
               if isinstance(r.get("abs_error_pbe_kcalmol"), (int, float))]
        if not (dens and nn and pbe):
            continue
        c = cells[d.name]
        rows.append((d.name, c["arch"], c["subset_size"],
                     "CH" in json.load(open(tm)).get("molecules", []),
                     statistics.median(dens),
                     (sum(nn) / len(nn)) / (sum(pbe) / len(pbe))))
    if not rows:
        return
    print(f"    {'spec':>10s} {'arch':>16s} {'ss':>3s} {'CH':>3s} "
          f"{'density':>8s} {'energy':>7s}")
    for name, arch, ss, has_ch, dr, er in rows:
        print(f"    {name:>10s} {arch:>16s} {ss:>3} "
              f"{'yes' if has_ch else ' no':>3s} {dr:8.4f} {er:7.4f}")
    for arch in sorted({r[1] for r in rows}):
        for label, keep in (("clean", False), ("CH", True)):
            vals = [r[4] for r in rows if r[1] == arch and r[3] is keep]
            if vals:
                print(f"      {arch:>16s} {label:>5s}: n={len(vals)} "
                      f"median density {statistics.median(vals):.4f}")
    xs = [r[5] for r in rows]
    ys = [r[4] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    # r = cov / (sx * sy) is undefined when either sample has zero variance
    # (one evaluated spec, or several with identical ratios); the table above
    # is still worth having, so the coefficient is reported as undefined.
    if sx == 0.0 or sy == 0.0:
        print(f"    Pearson r(energy ratio, density ratio) over {n} specs = "
              "undefined (zero spread on "
              + ("both axes" if sx == 0.0 and sy == 0.0
                 else "the energy axis" if sx == 0.0 else "the density axis")
              + ")")
        return
    print(f"    Pearson r(energy ratio, density ratio) over {n} specs = "
          f"{cov / (sx * sy):.3f}   "
          "(negative would mean better energies cost densities)")


def degenerate_component_test(results_root: Path | None = None) -> None:
    """The discriminator for WHY the dominant species' error is irreducible.

    ``density_rmse_pbe`` is model-free: a fixed functional against a fixed
    reference file. It cannot vary between specs unless the SCF settles
    somewhere different. Pooling EVERY pulled def2-svp evaluation (not one
    draw per run -- a single draw hides the spread entirely) separates
    species whose SCF solution is unique from species with a degenerate
    manifold, and the PBE total-energy spread says whether the variation
    costs any energy.
    """
    print("\n[7] Degenerate-component test over all pulled def2-svp evals")
    root = results_root or (Path.home() / "Documents/Research/xcquinox-results")
    base = root / "runs/dfs_step7"
    acc: dict[str, dict[str, list]] = {}
    for basis in ("svp_grid2_v3", "svp_grid2", "svp_grid2_v3_full25"):
        run_root = base / basis / "runs"
        if not run_root.is_dir():
            continue
        for run_dir in sorted(run_root.glob("run_*")):
            for d in sorted((run_dir / "checkpoints").glob("spec_*")):
                p = d / "eval" / "per_molecule.json"
                if not p.is_file():
                    continue
                for x in json.load(open(p)):
                    rho, energy = x.get("density_rmse_pbe"), x.get("E_pbe")
                    if not isinstance(rho, (int, float)):
                        continue
                    slot = acc.setdefault(x["molecule"], {"rho": [], "E": []})
                    slot["rho"].append(rho)
                    if isinstance(energy, (int, float)):
                        slot["E"].append(energy)
    if not acc:
        print("    (no pulled def2-svp evaluations found)")
        return
    print(f"    {'species':>8s} {'n':>4s} {'rho min':>11s} {'rho max':>11s} "
          f"{'max/min':>8s} {'E_pbe spread (Ha)':>18s}")
    ranked = sorted(acc.items(),
                    key=lambda kv: -(max(kv[1]["rho"]) / min(kv[1]["rho"])
                                     if min(kv[1]["rho"]) > 0 else 0))
    scattering = []
    for name, slot in ranked:
        rho, energy = slot["rho"], slot["E"]
        if len(rho) < 5:
            continue
        spread = max(rho) / min(rho) if min(rho) > 0 else float("nan")
        if spread > 1.001:
            scattering.append(name)
        if spread > 1.001 or len(scattering) == 0 or name in ("N2", "CH4"):
            print(f"    {name:>8s} {len(rho):>4d} {min(rho):11.4e} "
                  f"{max(rho):11.4e} {spread:8.3f} "
                  f"{(max(energy) - min(energy) if energy else float('nan')):18.3e}")
    stable = [n for n, s in ranked if len(s["rho"]) >= 5
              and n not in scattering]
    print(f"    species whose model-free PBE density VARIES across specs: "
          f"{scattering}")
    print(f"    species bit-identical across every spec and run: "
          f"{len(stable)} (all closed-shell)")
    # a genuine functional-vs-reference gap would be ONE number per species;
    # draws below the closed-shell population median cannot happen that way
    closed = [v for n in stable for v in acc[n]["rho"]]
    for name in ("CH", "NO", "HO"):
        if name not in acc or not closed:
            continue
        rho = acc[name]["rho"]
        pop = statistics.median(closed)
        below = sum(1 for v in rho if v <= pop)
        print(f"    {name}: n={len(rho)} spread "
              f"{max(rho) / min(rho):.1f}x; draws at or below the "
              f"closed-shell median ({pop:.4e}): {below}/{len(rho)}"
              + (f"  (smallest = {min(rho) / pop:.2f}x that median)"
                 if below else ""))
    refs = root / "external_refs/external_refs_dfs_svp_g2"
    if not refs.is_dir():
        return
    print("    stored SVP reference diagnostics (offenders vs controls):")
    for name in ("CH", "NO", "HO", "CN", "C"):
        p = refs / f"{name}.npz"
        if not p.is_file():
            continue
        # our own generated reference cache; allow_pickle is needed only for
        # the string stamps stored as object arrays
        z = np.load(p, allow_pickle=True)

        def stamp(key):
            if key not in z.files:
                return None
            v = z[key]
            return v.item() if v.shape == () else v

        print(f"      {name:>3s}: oep_converged={stamp('oep_converged')} "
              f"oep_density_error={float(stamp('oep_density_error')):.4e} "
              f"oep_n_electrons={float(stamp('oep_n_electrons')):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    if not (args.run / "checkpoints").is_dir():
        raise SystemExit(f"no checkpoints/ under {args.run}")
    print(f"run: {args.run}")
    weight_identity(args.run)
    channel_budget(args.run)
    per_group_attribution(args.run)
    insample_density(args.run)
    heldout_summary(args.run)
    outcome_correlates(args.run)
    degenerate_component_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
