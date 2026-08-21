#!/usr/bin/env python3
"""Regenerate every number in MGGA_DIAGNOSIS.md from the pulled run.

The meta-GGA architecture ``deep_mgga_3x16`` entered the dfs6311 sweep at
subset_size 1-2 and does not follow the GGA archs: where every GGA arch drops to
or below the PBE line by subset_size 2, the meta-GGA stays ~2.3x worse than PBE.
This script reproduces the measurements that localize why.

Sections, in the order the argument runs:

  1. HELD-OUT MAE per (arch, subset_size) against the PBE line -- the observation.
  2. OVERFIT ASYMMETRY -- in-sample atomization MAE vs held-out reaction MAE.
     The meta-GGA fits its training molecules BETTER than the GGA archs and
     generalizes worse, which is what makes this overfitting rather than a
     broken functional.
  3. PRETRAIN CONVERGENCE -- final X/C losses per arch, from the pretrain
     metadata; rules out a failed clone.
  4. F_x AGAINST ITS OWN TARGET -- the meta-GGA archs pretrain to SCAN
     (pretrain.py selects Fx_scan_all for meta_gga archs), NOT to PBE, so the
     oracle here is libxc MGGA_X_SCAN. Comparing against PBE instead is the
     mistake this diagnosis had to correct: SCAN's F_x really is nearly flat in
     s, so "flat in s" is correct meta-GGA behaviour and not a defect.
  5. SCF CONVERGENCE -- the 3-cycle solver's converged fraction and last-cycle
     energy drift per arch, which bounds how much of the gap the protocol owns.

Sections 1-3 and 5 read only pulled artifacts (manifest.json, per_molecule.json,
pretrain_metadata.json). Section 4 additionally deserializes pulled checkpoints
(model.eqx / pretrain xnet.eqx) and calls libxc; it is skipped with a message
when the weights were not pulled. The ``.spec`` pickles are this repo's own
harness output pulled from our cluster scratch, so the load is inside the trust
boundary; nothing here reads third-party or network-sourced pickles.

Run from the repo root:

    python notebooks/analysis/mgga_diagnosis_evidence.py [--run <run_dir>]
"""
from __future__ import annotations

import argparse
import json
import pickle  # noqa: S403 -- trusted local harness output
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_RUN = (Path.home() / "Documents/Research/xcquinox-results/runs/dfs_step7"
               / "dfs6311_grid3_v3/runs/run_20260728T140018Z")
HA_TO_KCAL = 627.509474

# --------------------------------------------------------------------------- #
# Pre-committed decision rule for "does the meta-GGA overfit resolve with more
# training data?", fixed on 2026-08-06 while only specs 0033-0034 (ss=1,2)
# existed -- so the verdict is a prediction, not a reading taken after the fact.
#
# WHY ss=5 DECIDES IT. The GGA archs realize 42-87% of their entire
# ss=1 -> ss=26 improvement by ss=2 and 60-93% by ss=3, then plateau:
#
#     arch              ss1   ss2   ss3   ss5   ss26
#     deep_3x16        13.3  11.1  10.1  10.7   8.0
#     deep_attn_3x16   14.4   7.9  11.3   7.9   6.4
#     deep_cusp_3x16   18.0  10.0   9.5   9.8   8.8
#
# So the meta-GGA's ss=3/4/5 cells (specs 0035-0037) settle this; its ss=26 cell
# (spec 0043) adds little. If the 42x train->held-out blow-up at ss=2 is a
# small-data artifact it must be largely gone by ss=5; if it is structural the
# curve plateaus at its own high level exactly as the GGA curves plateau at
# theirs.
# --------------------------------------------------------------------------- #
MGGA_RECOVER_BELOW = 20.0   # held-out MAE at ss=5 -> tracking a GGA-like plateau
MGGA_PERSIST_ABOVE = 24.0   # held-out MAE at ss=5 -> plateauing high; structural
MGGA_VERDICT_SS = 5         # the subset size the rule is evaluated at
#: The alpha and s values the diagnosis tabulates F_x on.
ALPHA_GRID: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0)
S_GRID: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)


# --------------------------------------------------------------------------- #
# pulled-artifact readers
# --------------------------------------------------------------------------- #

def _cells(run: Path) -> Dict[int, Dict[str, Any]]:
    man = json.loads((run / "manifest.json").read_text())
    return {s["index"]: s["cell"] for s in man["specs"]}


def _mae(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [abs(r[key]) for r in records
            if isinstance(r.get(key), (int, float))]
    return float(np.mean(vals)) if vals else None


def _read_json(path: Path) -> Optional[list]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def holdout_table(run: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """``{(arch, ss): {nn, pbe, n_rxn, idx}}`` over every completed cell."""
    # Imported at the guard, not at module scope: this script has no
    # other use for the training package and importing it here would
    # pull jax / pyscf / equinox into every invocation.
    from xcquinox.alec.eval_holdout import assert_channel_not_sliced
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for idx, cell in sorted(_cells(run).items()):
        # The learning-curve verdict is read off this table: a six-species
        # workflow slice averaged into one cell would move the curve.
        assert_channel_not_sliced(run / "checkpoints" / f"spec_{idx:04d}",
                                  "eval_holdout")
        rec = _read_json(run / "checkpoints" / f"spec_{idx:04d}"
                         / "eval_holdout" / "per_reaction.json")
        if not rec:
            continue
        out[(cell["arch"], cell["subset_size"])] = {
            "nn": _mae(rec, "error_nn_kcalmol"),
            "pbe": _mae(rec, "error_pbe_kcalmol"),
            "n_rxn": len(rec), "idx": idx}
    return out


def insample_ae(run: Path, idx: int) -> Tuple[Optional[float], int]:
    """``(in-sample atomization MAE, n_species)`` for one spec."""
    rec = _read_json(run / "checkpoints" / f"spec_{idx:04d}" / "eval"
                     / "per_molecule.json")
    if not rec:
        return None, 0
    vals = [abs(r["AE_error_kcalmol"]) for r in rec
            if isinstance(r.get("AE_error_kcalmol"), (int, float))]
    return (float(np.mean(vals)) if vals else None), len(vals)


def scf_stats(run: Path, idx: int) -> Optional[Dict[str, Any]]:
    """Converged fraction + last-cycle energy drift (kcal/mol) for one spec.

    The drift ``|E(step2) - E(step1)|`` is the honest measure of distance from
    the fixed point here: ``scf_energy_residual_2`` is written as 0.0 on the
    final recorded cycle, so reading it would report perfect convergence for
    every spec.
    """
    # Imported at the guard, not at module scope: this script has no
    # other use for the training package and importing it here would
    # pull jax / pyscf / equinox into every invocation.
    from xcquinox.alec.eval_holdout import assert_channel_not_sliced
    # These statistics describe the pool's molecules, not a workflow slice's.
    assert_channel_not_sliced(run / "checkpoints" / f"spec_{idx:04d}",
                              "eval_holdout")
    rec = _read_json(run / "checkpoints" / f"spec_{idx:04d}" / "eval_holdout"
                     / "per_molecule.json")
    if not rec:
        return None
    drift = sorted(
        abs(r["scf_energy_step_2"] - r["scf_energy_step_1"]) * HA_TO_KCAL
        for r in rec
        if isinstance(r.get("scf_energy_step_2"), (int, float))
        and isinstance(r.get("scf_energy_step_1"), (int, float)))
    n_conv = sum(1 for r in rec if r.get("scf_converged"))
    if not drift:
        return {"n": len(rec), "n_conv": n_conv, "median": None,
                "p90": None, "max": None}
    return {"n": len(rec), "n_conv": n_conv,
            "median": statistics.median(drift),
            "p90": drift[int(0.9 * len(drift))], "max": drift[-1]}


def pretrain_losses(run: Path) -> Dict[str, Dict[str, float]]:
    """``{arch: {final_loss_x, final_loss_c}}`` from each pretrain metadata."""
    out: Dict[str, Dict[str, float]] = {}
    pdir = run / "pretrain"
    if not pdir.is_dir():
        return out
    for sub in sorted(pdir.iterdir()):
        meta = sub / "pretrain_metadata.json"
        if not meta.is_file():
            continue
        try:
            m = json.loads(meta.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out[sub.name] = {"x": m.get("final_loss_x"), "c": m.get("final_loss_c")}
    return out


# --------------------------------------------------------------------------- #
# section 4 -- F_x against its own target
# --------------------------------------------------------------------------- #

def scan_fx(s: float, alpha: float, rho: float = 1.0) -> float:
    """SCAN exchange enhancement ``F_x(s, alpha)`` from libxc ``MGGA_X_SCAN``.

    The INDEPENDENT oracle: libxc's own SCAN, not this repo's SCAN path, so a
    bug shared by the repo's pretrain-target generation and its evaluation
    cannot hide here. ``tau`` is solved from the requested alpha at fixed
    ``(rho, s)`` by inverting ``alpha = (tau - tau_W)/tau_unif``.
    """
    from pyscf import dft

    k_f = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    sigma = (s * 2.0 * k_f * rho) ** 2
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    tau = alpha * tau_unif + tau_w
    # unpolarized meta-GGA layout: (rho, dx, dy, dz, laplacian, tau)
    arr = np.array([[rho], [np.sqrt(sigma)], [0.0], [0.0], [0.0], [tau]])
    e_xc = dft.libxc.eval_xc("MGGA_X_SCAN", arr, spin=0, deriv=0)[0][0]
    e_x_lda = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (1.0 / 3.0)
    return float(e_xc / e_x_lda)


def model_fx(model, s: float, alpha: float, rho: float = 1.0) -> float:
    """``F_x(s, alpha)`` from a loaded model, alpha placed in the descriptor
    column the meta-GGA reads."""
    import jax.numpy as jnp

    k_f = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    sigma = (s * 2.0 * k_f * rho) ** 2
    n_extra = int(getattr(model.xnet, "n_extra_features", 0)) or 1
    feats = np.zeros((1, n_extra), dtype=float)
    feats[0, int(getattr(model.xnet, "metagga_alpha_index", 0))] = alpha
    fx = model.eval_Fx(jnp.asarray([rho]), jnp.asarray([sigma]),
                       jnp.asarray(feats))
    return float(np.asarray(fx)[0])


def load_models(run: Path, idx: int, arch: str):
    """``(trained, pretrained)`` models for a spec, or ``(None, None)`` when the
    checkpoints were not pulled. ``pretrained`` is the same model with the
    pre-task-training xnet swapped in -- the control that separates what
    pretraining produced from what task training did to it."""
    import equinox as eqx

    from xcquinox.alec import eval_holdout

    spec_path = run / "specs" / f"spec_{idx:04d}.spec"
    model_path = run / "checkpoints" / f"spec_{idx:04d}" / "model.eqx"
    pre_path = run / "pretrain" / arch / "xnet.eqx"
    if not (spec_path.is_file() and model_path.is_file()):
        return None, None
    with spec_path.open("rb") as fh:
        spec = pickle.load(fh)
    trained = eval_holdout.load_trained_model(spec, model_path)
    pre = None
    if pre_path.is_file():
        pre = eqx.tree_at(
            lambda m: m.xnet, trained,
            eqx.tree_deserialise_leaves(pre_path, trained.xnet))
    return trained, pre


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def saturation(nn_by_ss: Dict[int, float]) -> Dict[int, float]:
    """``{ss: fraction of this arch's total gain realized by ss}``.

    Measured against the arch's OWN first and best values, so a family whose
    absolute errors sit higher is still comparable on shape -- which is the
    question here (does the meta-GGA follow the GGA saturation curve, offset,
    or does it plateau early at its own level?).
    """
    sizes = sorted(nn_by_ss)
    # Two points carry no shape: the later one is trivially 100% of the span
    # between them. Refuse rather than print a meaningless 100%.
    if len(sizes) < 3:
        return {}
    first, best = nn_by_ss[sizes[0]], min(nn_by_ss.values())
    span = first - best
    if span <= 0:
        return {ss: 0.0 for ss in sizes}
    out, running = {}, first
    for ss in sizes:
        running = min(running, nn_by_ss[ss])
        out[ss] = (first - running) / span
    return out


def mgga_verdict(nn_by_ss: Dict[int, float]) -> str:
    """Apply the pre-committed rule to the meta-GGA learning curve."""
    val = nn_by_ss.get(MGGA_VERDICT_SS)
    if val is None:
        have = sorted(nn_by_ss)
        return (f"UNDECIDED -- rule evaluates at ss={MGGA_VERDICT_SS}; "
                f"completed meta-GGA cells so far: {have or 'none'}. "
                f"Waiting on spec_0037 (ss=5).")
    if val < MGGA_RECOVER_BELOW:
        return (f"RECOVERS -- ss={MGGA_VERDICT_SS} held-out MAE {val:.2f} < "
                f"{MGGA_RECOVER_BELOW:.0f}: the ss<=2 blow-up was a small-data "
                "artifact, and the meta-GGA is tracking a GGA-like plateau.")
    if val > MGGA_PERSIST_ABOVE:
        return (f"PERSISTS -- ss={MGGA_VERDICT_SS} held-out MAE {val:.2f} > "
                f"{MGGA_PERSIST_ABOVE:.0f}: the curve is plateauing at its own "
                "high level, so the gap is structural, not small-data.")
    return (f"AMBIGUOUS -- ss={MGGA_VERDICT_SS} held-out MAE {val:.2f} falls "
            f"between {MGGA_RECOVER_BELOW:.0f} and {MGGA_PERSIST_ABOVE:.0f}; "
            "the rule deliberately refuses a verdict in this band. Read "
            "ss=6,7,12 before concluding.")


def _fx_table(title: str, fn) -> None:
    print(f"\n  {title}")
    print("     s\\a " + "".join(f"{a:>9.2f}" for a in ALPHA_GRID))
    for s in S_GRID:
        print(f"    {s:>4.1f} " + "".join(f"{fn(s, a):>9.3f}" for a in ALPHA_GRID))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(DEFAULT_RUN), type=Path,
                   help=f"pulled run dir (default: {DEFAULT_RUN})")
    p.add_argument("--skip-checkpoints", action="store_true",
                   help="skip section 4 (needs pulled model.eqx + pyscf/jax)")
    args = p.parse_args(argv)
    run = Path(args.run).expanduser()
    if not (run / "manifest.json").is_file():
        print(f"no manifest.json under {run}")
        return 1
    print(f"run: {run.name}")

    ht = holdout_table(run)
    archs = [a for a in ("deep_3x16", "deep_attn_3x16", "deep_cusp_3x16",
                         "deep_mgga_3x16") if any(k[0] == a for k in ht)]
    sizes = sorted({ss for _a, ss in ht})

    print("\n[1] HELD-OUT reaction MAE (kcal/mol) per (arch, subset_size)")
    print("     ss  " + "".join(f"{a.replace('_3x16',''):>14}" for a in archs)
          + f"{'PBE':>10}")
    for ss in sizes:
        row = f"    {ss:>3}  "
        pbe = None
        for a in archs:
            cell = ht.get((a, ss))
            if cell is None:
                row += f"{'-':>14}"
            else:
                row += f"{cell['nn']:>14.2f}"
                pbe = cell["pbe"]
        print(row + (f"{pbe:>10.2f}" if pbe is not None else ""))

    print("\n[2] OVERFIT ASYMMETRY -- in-sample AE MAE vs held-out reaction MAE")
    print(f"    {'arch':<18}{'ss':>4}{'in-sample':>12}{'held-out':>11}"
          f"{'blow-up':>10}{'n_train':>9}")
    for ss in sizes:
        for a in archs:
            cell = ht.get((a, ss))
            if cell is None:
                continue
            ins, n_ins = insample_ae(run, cell["idx"])
            if ins is None or not ins:
                continue
            print(f"    {a:<18}{ss:>4}{ins:>12.3f}{cell['nn']:>11.2f}"
                  f"{cell['nn']/ins:>9.1f}x{n_ins:>9}")

    print("\n[3] PRETRAIN convergence (final loss; meta-GGA archs target SCAN, "
          "GGA archs PBE)")
    for arch, m in sorted(pretrain_losses(run).items()):
        tag = "SCAN" if "mgga" in arch else "PBE "
        print(f"    {arch:<26} target={tag}  X={m['x']:.3e}  C={m['c']:.3e}")

    print("\n[5] SCF convergence at the 3-cycle solver "
          "(drift = |E(step2)-E(step1)|, kcal/mol)")
    print(f"    {'arch':<18}{'ss':>4}{'converged':>12}{'median':>10}"
          f"{'p90':>9}{'max':>9}")
    for ss in sizes:
        for a in archs:
            cell = ht.get((a, ss))
            if cell is None:
                continue
            st = scf_stats(run, cell["idx"])
            if not st or st["median"] is None:
                continue
            print(f"    {a:<18}{ss:>4}{st['n_conv']:>7}/{st['n']:<4}"
                  f"{st['median']:>10.3f}{st['p90']:>9.2f}{st['max']:>9.1f}")

    print("\n[6] LEARNING CURVE -- fraction of each arch's total gain realized "
          "by subset_size")
    curves = {a: {ss: ht[(a, ss)]["nn"] for ss in sizes if (a, ss) in ht}
              for a in archs}
    shown = [ss for ss in sizes if any(ss in c for c in curves.values())]
    print(f"    {'arch':<18}" + "".join(f"{ss:>7}" for ss in shown))
    for a in archs:
        sat = saturation(curves[a])
        if not sat:
            print(f"    {a:<18}  ({len(curves[a])} cell(s) -- a shape needs "
                  ">= 3; two points are trivially 0%/100%)")
            continue
        # The span is printed beside the percentages because 100% means "this
        # arch has stopped improving", NOT "this arch did well": an arch that
        # plateaus early at a bad value reaches 100% of its own small span.
        # Without the span, a plateaued meta-GGA reads like a converged GGA.
        first = curves[a][min(curves[a])]
        best = min(curves[a].values())
        print(f"    {a:<18}"
              + "".join(f"{sat[ss]:>6.0%}" + " " if ss in sat else f"{'-':>7}"
                        for ss in shown)
              + f"   [{first:.1f} -> {best:.1f}]")
    print("\n    VERDICT (rule fixed 2026-08-06, before specs 0035-0037 ran; "
          f"recover < {MGGA_RECOVER_BELOW:.0f} < ambiguous < "
          f"{MGGA_PERSIST_ABOVE:.0f} < persists, at ss={MGGA_VERDICT_SS}):")
    for a in archs:
        if "mgga" in a:
            print(f"      {a}: {mgga_verdict(curves[a])}")

    if args.skip_checkpoints:
        print("\n[4] skipped (--skip-checkpoints)")
        return 0
    print("\n[4] F_x AGAINST ITS OWN TARGET (rho=1)")
    print("    The meta-GGA pretrains to SCAN, not PBE. Comparing it against PBE")
    print("    reads SCAN's genuine flatness in s as a defect -- SCAN's own F_x")
    print("    runs 1.174 -> 1.075 over s=0..4 at alpha=0 (ceiling 1.174).")
    try:
        _fx_table("SCAN (libxc MGGA_X_SCAN) -- the target", scan_fx)
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"    (SCAN oracle unavailable: {type(exc).__name__}: {exc})")
    mgga_cells = sorted((ss, c["idx"]) for (a, ss), c in ht.items()
                        if a == "deep_mgga_3x16")
    for ss, idx in mgga_cells:
        try:
            trained, pre = load_models(run, idx, "deep_mgga_3x16")
        except Exception as exc:  # noqa: BLE001
            print(f"    (spec_{idx:04d} not loadable: "
                  f"{type(exc).__name__}: {exc})")
            continue
        if trained is None:
            print(f"    (spec_{idx:04d}: model.eqx not pulled)")
            continue
        if pre is not None and ss == mgga_cells[0][0]:
            _fx_table("deep_mgga_3x16 PRETRAINED (before task training)",
                      lambda s, a: model_fx(pre, s, a))
        _fx_table(f"deep_mgga_3x16 TRAINED spec_{idx:04d} (ss={ss})",
                  lambda s, a, m=trained: model_fx(m, s, a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
