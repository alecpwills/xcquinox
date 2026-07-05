#!/usr/bin/env python
"""Plot per-architecture pretraining loss curves for a cluster training run.

Reads ``<run_dir>/pretrain/<arch>/{losses_x.npy, losses_c.npy,
pretrain_metadata.json}`` and writes a two-panel figure -- X-net and C-net
enhancement-factor pretraining loss on a log-y axis, one line per architecture.

This is a standalone quick-look, NOT part of ``make_ablation_arch_figure.py
--suite``: the suite's figures read a run's held-out eval (``eval_holdout/``),
which only exists once training completes, whereas pretraining is an earlier,
independent stage. When a run's training stalls (e.g. the 6-311++G(3df,2pd)+grid3
XLA compile OOM), the pretraining curves are often the only clean signal, and
their smooth convergence confirms the nets pretrained fine (so the failure is in
the SCF-training stage, not the pretrain).

Semantics (see xcquinox/alec/pretrain.py): X-net and C-net are pretrained
separately; each array holds one MSE-loss value per pretraining step, so the
x-axis is the step index ``0 .. pretrain_steps-1`` and
``len(losses_x) == pretrain_metadata["pretrain_steps"]``.

Usage:
    python notebooks/analysis/plot_pretraining_curves.py <run_dir> [-o out.png]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

# Shared cross-figure styling (rung-keyed palette + rung grouping). ``arch_style``
# is light (no heavy analysis deps, unlike make_ablation_arch_figure), loaded by
# path since this directory is not an importable package. Falls back to a colormap
# if it can't be imported -- the curves stay correct, only the styling differs.
import importlib.util as _ilu  # noqa: E402

_AS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch_style.py")
try:
    _as_spec = _ilu.spec_from_file_location("arch_style", _AS_PATH)
    arch_style = _ilu.module_from_spec(_as_spec)
    sys.modules.setdefault("arch_style", arch_style)
    _as_spec.loader.exec_module(arch_style)
except Exception:  # pragma: no cover - only if arch_style is missing/broken
    arch_style = None

_FALLBACK_CMAP = plt.get_cmap("tab10")
# Linestyle by rung so GGA / meta-GGA / rung-3.5 / combined read apart even in
# grayscale (pretraining has no solver axis to map to linestyle).
_RUNG_LS = {"GGA": "-", "meta-GGA": "--", "rung-3.5": ":", "rung-3.5+meta-GGA": "-."}


def _arch_color(arch: str, idx: int):
    """Rung-keyed shared color if available, else a stable tab10 fallback."""
    if arch_style is not None:
        return arch_style.arch_color(arch)
    return _FALLBACK_CMAP(idx % 10)


def _arch_linestyle(arch: str):
    """Rung-keyed linestyle (solid GGA .. dashdot combined); solid if unavailable."""
    if arch_style is None:
        return "-"
    return _RUNG_LS.get(arch_style.rung_of(arch), "-")


def _order_archs(archs):
    """Rung-grouped order (GGA -> meta-GGA -> rung-3.5 -> combined) if available."""
    return arch_style.sort_by_rung(archs) if arch_style is not None else sorted(archs)


def load_pretrain_curves(run_dir):
    """Load every arch's pretraining loss arrays + metadata from a run dir.

    Returns ``{arch_name: {"x": ndarray, "c": ndarray, "meta": dict}}`` for each
    ``<run_dir>/pretrain/<arch>/`` that has both ``losses_x.npy`` and
    ``losses_c.npy``. Raises ``FileNotFoundError`` if there is no ``pretrain/``
    dir or no arch with both loss arrays.
    """
    pdir = Path(run_dir) / "pretrain"
    if not pdir.is_dir():
        raise FileNotFoundError(f"no pretrain/ dir under {run_dir}")
    out: dict[str, dict] = {}
    for arch_dir in sorted(p for p in pdir.iterdir() if p.is_dir()):
        lx, lc = arch_dir / "losses_x.npy", arch_dir / "losses_c.npy"
        if not (lx.is_file() and lc.is_file()):
            continue
        meta: dict = {}
        mp = arch_dir / "pretrain_metadata.json"
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text())
            except (ValueError, OSError):
                meta = {}
        out[arch_dir.name] = {
            "x": np.asarray(np.load(lx), dtype=float).ravel(),
            "c": np.asarray(np.load(lc), dtype=float).ravel(),
            "meta": meta,
        }
    if not out:
        raise FileNotFoundError(
            f"no arch subdir with both losses_x.npy and losses_c.npy under {pdir}")
    return out


def plot_pretraining_curves(curves, out_path, run_label=""):
    """Render the two-panel (X-net | C-net) log-y pretraining-loss figure.

    ``curves`` is the mapping returned by :func:`load_pretrain_curves`. Writes a
    PNG to ``out_path`` (creating parent dirs) and returns that path.
    """
    archs = _order_archs(list(curves))  # rung-grouped: GGA -> mGGA -> 3.5 -> both
    fig, (axx, axc) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for i, arch in enumerate(archs):
        d = curves[arch]
        color = _arch_color(arch, i)
        ls = _arch_linestyle(arch)
        meta = d.get("meta") or {}
        fx, fc = meta.get("final_loss_x"), meta.get("final_loss_c")
        lbl_x = arch if fx is None else f"{arch}  (final {fx:.2e})"
        lbl_c = arch if fc is None else f"{arch}  (final {fc:.2e})"
        axx.plot(np.arange(d["x"].size), d["x"], color=color, lw=1.3, ls=ls, label=lbl_x)
        axc.plot(np.arange(d["c"].size), d["c"], color=color, lw=1.3, ls=ls, label=lbl_c)
    for ax, title in ((axx, "X-net  (exchange $F_x$)"),
                      (axc, "C-net  (correlation $F_c$)")):
        ax.set_yscale("log")
        ax.set_xlabel("pretraining step")
        ax.set_ylabel("MSE loss vs target enhancement factor")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, framealpha=0.9)
    any_meta = next((c["meta"] for c in curves.values() if c.get("meta")), {})
    nsteps = any_meta.get("pretrain_steps")
    if nsteps is None:
        nsteps = max(c["x"].size for c in curves.values())
    sup = f"Pretraining loss curves -- {len(archs)} archs, {nsteps} steps"
    if run_label:
        sup += f"\n{run_label}"
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir",
                    help="run dir containing pretrain/<arch>/losses_*.npy")
    ap.add_argument("-o", "--out", default=None,
                    help="output PNG (default: <run_dir>/pretraining_curves.png)")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.run_dir, "pretraining_curves.png")
    curves = load_pretrain_curves(args.run_dir)
    written = plot_pretraining_curves(
        curves, out, run_label=os.path.basename(os.path.normpath(args.run_dir)))
    print(f"wrote {written}  ({len(curves)} archs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
