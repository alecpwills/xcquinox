#!/usr/bin/env python
"""Pretrained-network enhancement factors against the parent baseline.

For every architecture with a pretrained ``xnet.eqx``/``cnet.eqx`` pair under
``<run_dir>/pretrain/<arch>/``, the exchange and correlation enhancement
factors of the PRETRAINED network are drawn over the parent functional's
curves, together with difference panels -- under the v6 parent anchor the two
sit within the fidelity certificate's tolerance of each other, so the
difference panel is where the learned correction is visible at all.

Slices (the conventions of ``enhancement_factors.py``, whose curve helpers
are reused verbatim):
  * ``F_x(s)`` at rho = 1, zero extra descriptors.
  * ``F_c(s; r_s)`` at zeta = 0 and r_s in {0.5, 2, 5}.

The baselines are the anchor's OWN parent functions (``parents.pbe_fx`` /
``parents.pbe_fc``, libxc constants, pinned against libxc at 6.7e-16 /
1.5e-14): with the pre-image anchor the model IS that parent plus the
learned correction, so any other PBE implementation reads as a spurious
correction (the rounded-constant analytic helper differs by 4.553e-6 in
F_x on this grid).

Models are loaded through the production builder
(``cluster.fidelity.build_certified_model``): the run's own resolved
configuration supplies the model class (parent anchor, descriptor
coordinates, polarized correlation), so the plotted network is the class the
checkpoint was pretrained as. Meta-GGA architectures are refused by name (the
PBE curves here are the wrong parent for them).

Outputs, into ``--outdir``:
  * ``pretrain_fx_fc_<arch>.png``      per-arch 2x2 (overlay + difference)
  * ``pretrain_fx_fc_delta_all.png``   cross-arch difference panels
  * ``pretrain_fx_fc_curves.csv``      long-form curves (arch, channel, rs, s,
                                       f_model, f_parent)

Usage:
  python notebooks/analysis/pretrain_fx_fc.py \\
      --run-dir <pulled run dir> --outdir <figure dir> [--archs a,b,...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_style import ARCH_COLOR, arch_color  # noqa: E402
from enhancement_factors import (  # noqa: E402
    model_fc_curve,
    model_fx_curve,
    rs_to_rho,
    s_to_sigma,
)


def parent_fx_curve(s_grid: np.ndarray, rho: float = 1.0) -> np.ndarray:
    """PBE F_x from the anchor's own parent (``parents.pbe_fx``, libxc
    constants). ``enhancement_factors.pbe_fx_curve`` (the rounded-constant
    analytic helper) differs from it by up to 4.553e-6 on this grid, which
    would read as a spurious learned correction under the anchor."""
    import jax.numpy as jnp

    from xcquinox.alec import parents
    rho_arr = np.full(s_grid.shape[0], rho, dtype=float)
    sigma = s_to_sigma(rho_arr, s_grid)
    return np.asarray(parents.pbe_fx(jnp.asarray(rho_arr),
                                     jnp.asarray(sigma)), dtype=float)


def parent_fc_curve(s_grid: np.ndarray, rs: float,
                    zeta: float = 0.0) -> np.ndarray:
    """PBE F_c from the anchor's own parent (``parents.pbe_fc``): the PBE
    correlation over the model's PW92 baseline, on the model's conventions."""
    import jax.numpy as jnp

    from xcquinox.alec import parents
    rho = rs_to_rho(rs)
    rho_arr = np.full(s_grid.shape[0], rho, dtype=float)
    sigma = s_to_sigma(rho_arr, s_grid)
    zeta_arr = np.full(s_grid.shape[0], float(zeta))
    return np.asarray(parents.pbe_fc(jnp.asarray(rho_arr),
                                     jnp.asarray(sigma),
                                     jnp.asarray(zeta_arr)), dtype=float)

S_GRID = np.linspace(0.0, 6.0, 241)
RS_VALUES = (0.5, 2.0, 5.0)
#: One lightness step per r_s (sequential: r_s is a magnitude), anchored to
#: the model's arch colour via alpha; the parent counterparts are grey steps.
RS_ALPHA = {0.5: 1.0, 2.0: 0.72, 5.0: 0.45}
RS_GREY = {0.5: "0.15", 2.0: "0.45", 5.0: "0.65"}
_PARENT_STYLE = dict(color="0.25", linestyle="--", linewidth=2.0, zorder=1)


def discover_archs(run_dir: Path) -> list[str]:
    """Architectures under ``<run_dir>/pretrain/`` holding both networks."""
    root = run_dir / "pretrain"
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if (p / "xnet.eqx").is_file() and (p / "cnet.eqx").is_file()
    )


def load_pretrained_model(run_dir: Path, arch_name: str):
    """(arch, model) through the production builder, from the run's own
    resolved configuration."""
    from xcquinox.alec.cluster.fidelity import build_certified_model
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
    arch, model = build_certified_model(cfg, str(run_dir), arch_name)
    if getattr(arch, "meta_gga", False):
        raise ValueError(
            f"{arch_name} is a meta-GGA architecture; its parent is SCAN and "
            "the PBE curves drawn here are the wrong baseline for it.")
    return arch, model


def _certificate_line(run_dir: Path, arch_name: str) -> str:
    path = run_dir / "pretrain" / arch_name / "fidelity_certificate.json"
    try:
        with open(path) as fh:
            cert = json.load(fh)
    except (OSError, ValueError):
        return "certificate: unavailable"
    s = cert.get("summary", {})
    return (f"certificate {cert.get('verdict', '?')}: "
            f"max atom {s.get('max_atom_mHa', float('nan')):.2e} mHa, "
            f"max dAE {s.get('max_dAE_kcalmol', float('nan')):.2e} kcal/mol "
            f"(parent {cert.get('parent', '?')})")


def compute_curves(model) -> dict:
    """All plotted curves for one model: F_x and F_c(s; r_s) with parents."""
    curves = {
        "fx_model": model_fx_curve(model, S_GRID),
        "fx_parent": parent_fx_curve(S_GRID),
        "fc": {},
    }
    for rs in RS_VALUES:
        parent = parent_fc_curve(S_GRID, rs)
        curves["fc"][rs] = {
            "model": model_fc_curve(model, S_GRID, rs),
            "parent": parent,
        }
    return curves


def render_arch_figure(arch_name: str, curves: dict, outdir: Path,
                       footer: str) -> Path:
    """Per-arch 2x2: F_x overlay | delta F_x / F_c overlays | delta F_c."""
    color = ARCH_COLOR.get(arch_name, arch_color(arch_name))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    (ax_fx, ax_dfx), (ax_fc, ax_dfc) = axes

    ax_fx.plot(S_GRID, curves["fx_parent"], label="PBE parent",
               **_PARENT_STYLE)
    ax_fx.plot(S_GRID, curves["fx_model"], color=color, linewidth=2.0,
               label=f"{arch_name} (pretrained)", zorder=2)
    ax_fx.set_ylabel(r"$F_x(s)$")
    ax_fx.legend(frameon=False, fontsize=8)

    ax_dfx.axhline(0.0, color="0.7", linewidth=1.0)
    ax_dfx.plot(S_GRID, curves["fx_model"] - curves["fx_parent"],
                color=color, linewidth=2.0)
    ax_dfx.set_ylabel(r"$F_x^{\mathrm{NN}} - F_x^{\mathrm{PBE}}$")

    for rs in RS_VALUES:
        pair = curves["fc"][rs]
        if pair["parent"] is not None:
            ax_fc.plot(S_GRID, pair["parent"], color=RS_GREY[rs],
                       linestyle="--", linewidth=1.6, zorder=1,
                       label=rf"PBE, $r_s={rs:g}$")
        ax_fc.plot(S_GRID, pair["model"], color=color, linewidth=2.0,
                   alpha=RS_ALPHA[rs], zorder=2,
                   label=rf"NN, $r_s={rs:g}$")
        if pair["parent"] is not None:
            ax_dfc.plot(S_GRID, pair["model"] - pair["parent"], color=color,
                        alpha=RS_ALPHA[rs], linewidth=2.0,
                        label=rf"$r_s={rs:g}$")
    ax_dfc.axhline(0.0, color="0.7", linewidth=1.0)
    ax_fc.set_ylabel(r"$F_c(s;\,r_s)$  ($\zeta=0$)")
    ax_fc.legend(frameon=False, fontsize=7, ncol=2)
    ax_dfc.set_ylabel(r"$F_c^{\mathrm{NN}} - F_c^{\mathrm{PBE}}$")
    ax_dfc.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel(r"reduced gradient $s$")
        ax.grid(True, color="0.92", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(f"{arch_name}: pretrained network against the PBE parent",
                 fontsize=12)
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.97))
    out = outdir / f"pretrain_fx_fc_{arch_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_delta_figure(all_curves: dict[str, dict], outdir: Path,
                        footer: str, rs_for_fc: float = 2.0) -> Path:
    """Cross-arch difference panels: every arch's learned correction."""
    fig, (ax_dfx, ax_dfc) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for arch_name, curves in all_curves.items():
        color = ARCH_COLOR.get(arch_name, arch_color(arch_name))
        ax_dfx.plot(S_GRID, curves["fx_model"] - curves["fx_parent"],
                    color=color, linewidth=2.0, label=arch_name)
        pair = curves["fc"][rs_for_fc]
        if pair["parent"] is not None:
            ax_dfc.plot(S_GRID, pair["model"] - pair["parent"], color=color,
                        linewidth=2.0, label=arch_name)
    for ax, label in ((ax_dfx, r"$F_x^{\mathrm{NN}} - F_x^{\mathrm{PBE}}$"),
                      (ax_dfc,
                       rf"$F_c^{{\mathrm{{NN}}}} - F_c^{{\mathrm{{PBE}}}}$"
                       rf"  ($r_s={rs_for_fc:g}$)")):
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.set_xlabel(r"reduced gradient $s$")
        ax.set_ylabel(label)
        ax.grid(True, color="0.92", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Pretrained corrections to the PBE parent, all architectures",
                 fontsize=12)
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    out = outdir / "pretrain_fx_fc_delta_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_curves_csv(all_curves: dict[str, dict], outdir: Path) -> Path:
    out = outdir / "pretrain_fx_fc_curves.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "channel", "rs", "s", "f_model", "f_parent"])
        for arch_name, curves in all_curves.items():
            for s, fm, fp in zip(S_GRID, curves["fx_model"],
                                 curves["fx_parent"]):
                w.writerow([arch_name, "fx", "", f"{s:.6f}", repr(float(fm)),
                            repr(float(fp))])
            for rs in RS_VALUES:
                pair = curves["fc"][rs]
                parent = (pair["parent"] if pair["parent"] is not None
                          else [float("nan")] * len(S_GRID))
                for s, fm, fp in zip(S_GRID, pair["model"], parent):
                    w.writerow([arch_name, "fc", f"{rs:g}", f"{s:.6f}",
                                repr(float(fm)), repr(float(fp))])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--archs", default=None,
                    help="comma-separated subset (default: every arch with "
                         "both networks on disk)")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    archs = (tuple(a.strip() for a in args.archs.split(","))
             if args.archs else tuple(discover_archs(run_dir)))
    if not archs:
        print(f"no pretrained xnet/cnet pairs under {run_dir}/pretrain "
              "(pull them with --profile full or an explicit rsync)")
        return 1

    all_curves: dict[str, dict] = {}
    for arch_name in archs:
        arch, model = load_pretrained_model(run_dir, arch_name)
        curves = compute_curves(model)
        dfx = float(np.max(np.abs(curves["fx_model"] - curves["fx_parent"])))
        footer = (f"run {run_dir.name}; slices: F_x at rho=1, zero extra "
                  f"descriptors; F_c at zeta=0; max|dF_x| {dfx:.2e}. "
                  + _certificate_line(run_dir, arch_name))
        out = render_arch_figure(arch_name, curves, outdir, footer)
        print(f"wrote {out} (max|dF_x| {dfx:.3e})")
        all_curves[arch_name] = curves

    shared_footer = (f"run {run_dir.name}; pretrained networks loaded through "
                     "the run's resolved configuration (parent anchor class); "
                     "parent curves: parents.pbe_fx / parents.pbe_fc (libxc constants)")
    out = render_delta_figure(all_curves, outdir, shared_footer)
    print(f"wrote {out}")
    out = write_curves_csv(all_curves, outdir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
