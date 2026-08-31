#!/usr/bin/env python
"""Pretrained-network enhancement factors against each arch's parent baseline.

For every architecture with a pretrained ``xnet.eqx``/``cnet.eqx`` pair under
``<run_dir>/pretrain/<arch>/``, the exchange and correlation enhancement
factors of the PRETRAINED network are drawn over the parent functional's
curves, together with difference panels -- under the v6 parent anchor the two
sit within the fidelity certificate's tolerance of each other, so the
difference panel is where the learned correction is visible at all.

The parent is resolved PER ARCHITECTURE from its own rung
(``parents.parent_for_arch``): PBE for the GGA rungs, SCAN for the meta-GGA
ones. A run mixing rungs draws each arch against ITS parent, and every
figure states which parent each panel uses.

Slices, by parent (the s and r_s conventions of ``enhancement_factors.py``,
whose curve helpers are reused verbatim):
  * PBE archs:  ``F_x(s)`` at rho = 1, zero extra descriptors;
    ``F_c(s; r_s)`` at zeta = 0 and r_s in {0.5, 2, 5}.
  * SCAN archs: the same curves at the fixed iso-orbital slices alpha = 0
    and alpha = 1 (the F_x(s)-at-fixed-alpha convention of Sun, Ruzsinszky
    and Perdew, PRL 115, 036402 (2015), Fig. 1; DFS, PRB 104, L161109
    (2021), uses the same (r_s, s, alpha) coordinates). alpha is the EXACT
    raw indicator of the slice: the value placed in the network's alpha
    column is the stored-column encoding of that indicator
    (:func:`alpha_column_value`), which ``networks._raw_indicator`` inverts
    exactly, so the anchored parent inside the network is evaluated at
    exactly the slice's alpha.

The baselines are the anchor's OWN parent functions (``parents.pbe_fx`` /
``parents.pbe_fc``, pinned against libxc at 6.7e-16 / 1.5e-14, and
``parents.scan_fx`` / ``parents.scan_fc``, pinned at 2.6e-15 / 1.5e-13,
all at libxc constants): with the pre-image anchor the model IS that parent
plus the learned correction, so any other implementation of the parent reads
as a spurious correction (the rounded-constant analytic PBE helper differs
by 4.553e-6 in F_x on this grid).

Models are loaded through the production builder
(``cluster.fidelity.build_certified_model``): the run's own resolved
configuration supplies the model class (parent anchor, descriptor
coordinates, polarized correlation), so the plotted network is the class the
checkpoint was pretrained as.

Outputs, into ``--outdir``:
  * ``pretrain_fx_fc_<arch>.png``      per-arch panels (overlay + difference;
                                       2x2 for a PBE arch, 2x3 with the two
                                       alpha slices for a SCAN arch)
  * ``pretrain_fx_fc_delta_all.png``   cross-arch difference panels
  * ``pretrain_fx_fc_curves.csv``      long-form curves (arch, channel, rs, s,
                                       f_model, f_parent; an alpha column is
                                       added when a SCAN arch is drawn)

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
    alpha_column_value,
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


def parent_fx_curve_scan(s_grid: np.ndarray, alpha: float,
                         rho: float = 1.0) -> np.ndarray:
    """SCAN F_x from the anchor's own parent (``parents.parent_fx('scan')``,
    libxc constants) at the EXACT iso-orbital indicator ``alpha`` of the
    slice."""
    import jax.numpy as jnp

    from xcquinox.alec import parents
    rho_arr = np.full(s_grid.shape[0], rho, dtype=float)
    sigma = s_to_sigma(rho_arr, s_grid)
    alpha_arr = np.full(s_grid.shape[0], float(alpha))
    return np.asarray(parents.parent_fx("scan", jnp.asarray(rho_arr),
                                        jnp.asarray(sigma),
                                        jnp.asarray(alpha_arr)), dtype=float)


def parent_fc_curve_scan(s_grid: np.ndarray, rs: float, alpha: float,
                         zeta: float = 0.0) -> np.ndarray:
    """SCAN F_c from the anchor's own parent (``parents.parent_fc('scan')``):
    the SCAN correlation over the model's polarized PW92 baseline, at the
    EXACT indicator ``alpha`` of the slice."""
    import jax.numpy as jnp

    from xcquinox.alec import parents
    rho = rs_to_rho(rs)
    rho_arr = np.full(s_grid.shape[0], rho, dtype=float)
    sigma = s_to_sigma(rho_arr, s_grid)
    zeta_arr = np.full(s_grid.shape[0], float(zeta))
    alpha_arr = np.full(s_grid.shape[0], float(alpha))
    return np.asarray(parents.parent_fc("scan", jnp.asarray(rho_arr),
                                        jnp.asarray(sigma),
                                        jnp.asarray(zeta_arr),
                                        jnp.asarray(alpha_arr)), dtype=float)


def model_fx_curve_mgga(model, s_grid: np.ndarray, alpha: float,
                        rho: float = 1.0) -> np.ndarray:
    """Model F_x(s) at the EXACT iso-orbital slice ``alpha``, zero other
    descriptors. The shared helper encodes the alpha column itself
    (``enhancement_factors.model_fx_curve`` places
    :func:`~enhancement_factors.alpha_column_value` in the column), so the
    raw slice value is passed through."""
    return model_fx_curve(model, s_grid, rho=rho, alpha=float(alpha))


def model_fc_curve_mgga(model, s_grid: np.ndarray, rs: float, alpha: float,
                        zeta: float = 0.0) -> np.ndarray:
    """Model F_c(s; r_s) at the EXACT iso-orbital slice ``alpha``, zero other
    descriptors.

    Mirrors ``enhancement_factors.model_fc_curve`` (which has no alpha
    channel) with the alpha column placed at the C-net's
    ``metagga_alpha_index``, carrying the stored-column encoding
    (``enhancement_factors.alpha_column_value``). The packing matches
    ``pretrain._append_pretrain_mesh``: the C-net row is
    ``[rho, sigma, (zeta,) *extras]`` with zeta supplied through
    ``eval_Fc``'s keyword on a polarization-aware C-net.
    """
    import jax.numpy as jnp
    n = s_grid.shape[0]
    rho = rs_to_rho(rs)
    rho_arr = np.full(n, rho, dtype=float)
    sigma = s_to_sigma(rho_arr, s_grid)
    n_extra = int(getattr(model.cnet, "n_extra_features", 0))
    feats = np.zeros((n, n_extra), dtype=float)
    idx = int(model.cnet.metagga_alpha_index)
    feats[:, idx] = alpha_column_value(alpha)
    fc = model.eval_Fc(jnp.asarray(rho_arr), jnp.asarray(sigma),
                       jnp.asarray(feats), zeta=zeta)
    return np.asarray(fc, dtype=float)


S_GRID = np.linspace(0.0, 6.0, 241)
RS_VALUES = (0.5, 2.0, 5.0)
#: Iso-orbital slices the SCAN-parent curves are drawn at: the single-orbital
#: corner (alpha = 0) and the uniform-gas point (alpha = 1), the two slices
#: SCAN's F_x(s) is conventionally plotted on (Sun, Ruzsinszky and Perdew,
#: PRL 115, 036402 (2015), Fig. 1).
ALPHA_VALUES = (0.0, 1.0)
#: One lightness step per r_s (sequential: r_s is a magnitude), anchored to
#: the model's arch colour via alpha; the parent counterparts are grey steps.
RS_ALPHA = {0.5: 1.0, 2.0: 0.72, 5.0: 0.45}
RS_GREY = {0.5: "0.15", 2.0: "0.45", 5.0: "0.65"}
#: Linestyle per alpha slice where both share an axis (model curves; the
#: F_c panels split by PANEL instead, one per alpha, and keep solid lines).
ALPHA_LINESTYLE = {0.0: "-", 1.0: "-."}
#: Parent counterparts of the two alpha slices (both grey, so the linestyle
#: is what separates them).
_PARENT_ALPHA_LINESTYLE = {0.0: "--", 1.0: ":"}
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
    resolved configuration. Every rung loads here; the parent the curves are
    drawn against is resolved per arch downstream
    (:func:`compute_curves_for_arch`)."""
    from xcquinox.alec.cluster.fidelity import build_certified_model
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
    return build_certified_model(cfg, str(run_dir), arch_name)


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
    """All plotted curves for one PBE-parent model: F_x and F_c(s; r_s)."""
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


def compute_curves_scan(model) -> dict:
    """All plotted curves for one SCAN-parent (meta-GGA) model: F_x and
    F_c(s; r_s) at each iso-orbital slice of :data:`ALPHA_VALUES`, each with
    its SCAN parent at the same exact alpha."""
    curves = {"fx_alpha": {}, "fc_alpha": {}}
    for alpha in ALPHA_VALUES:
        curves["fx_alpha"][alpha] = {
            "model": model_fx_curve_mgga(model, S_GRID, alpha),
            "parent": parent_fx_curve_scan(S_GRID, alpha),
        }
        curves["fc_alpha"][alpha] = {}
        for rs in RS_VALUES:
            curves["fc_alpha"][alpha][rs] = {
                "model": model_fc_curve_mgga(model, S_GRID, rs, alpha),
                "parent": parent_fc_curve_scan(S_GRID, rs, alpha),
            }
    return curves


def compute_curves_for_arch(arch, model) -> tuple:
    """``(parent, curves)`` with the parent resolved from the arch's OWN rung
    (``parents.parent_for_arch``: SCAN on the meta-GGA rung, PBE otherwise),
    so one invocation path serves a run mixing rungs with no cross-parent
    draw."""
    from xcquinox.alec import parents
    parent = parents.parent_for_arch(arch)
    if parent == "scan":
        return parent, compute_curves_scan(model)
    return parent, compute_curves(model)


def _max_abs_fx_delta(curves: dict) -> float:
    """max|F_x^NN - F_x^parent| over every drawn F_x slice."""
    if "fx_alpha" in curves:
        return max(
            float(np.max(np.abs(pair["model"] - pair["parent"])))
            for pair in curves["fx_alpha"].values())
    return float(np.max(np.abs(curves["fx_model"] - curves["fx_parent"])))


def render_arch_figure(arch_name: str, curves: dict, outdir: Path,
                       footer: str) -> Path:
    """Per-arch panels against the arch's parent: the 2x2 PBE layout, or the
    2x3 SCAN layout with the two alpha slices for a meta-GGA arch."""
    if "fx_alpha" in curves:
        return _render_arch_figure_scan(arch_name, curves, outdir, footer)
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


def _render_arch_figure_scan(arch_name: str, curves: dict, outdir: Path,
                             footer: str) -> Path:
    """Per-arch 2x3 against the SCAN parent. Columns: F_x (both alpha slices,
    separated by linestyle) | F_c at alpha=0 | F_c at alpha=1; overlays on the
    top row, differences below each overlay."""
    color = ARCH_COLOR.get(arch_name, arch_color(arch_name))
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6))
    (ax_fx, ax_fc0, ax_fc1), (ax_dfx, ax_dfc0, ax_dfc1) = axes

    for alpha in ALPHA_VALUES:
        pair = curves["fx_alpha"][alpha]
        ax_fx.plot(S_GRID, pair["parent"], color="0.25",
                   linestyle=_PARENT_ALPHA_LINESTYLE[alpha], linewidth=2.0,
                   zorder=1, label=rf"SCAN, $\alpha={alpha:g}$")
        ax_fx.plot(S_GRID, pair["model"], color=color,
                   linestyle=ALPHA_LINESTYLE[alpha], linewidth=2.0, zorder=2,
                   label=rf"NN, $\alpha={alpha:g}$")
        ax_dfx.plot(S_GRID, pair["model"] - pair["parent"], color=color,
                    linestyle=ALPHA_LINESTYLE[alpha], linewidth=2.0,
                    label=rf"$\alpha={alpha:g}$")
    ax_fx.set_ylabel(r"$F_x(s;\,\alpha)$")
    ax_fx.legend(frameon=False, fontsize=8)
    ax_dfx.axhline(0.0, color="0.7", linewidth=1.0)
    ax_dfx.set_ylabel(r"$F_x^{\mathrm{NN}} - F_x^{\mathrm{SCAN}}$")
    ax_dfx.legend(frameon=False, fontsize=8)

    # One F_c overlay/difference column per alpha slice; strict so a changed
    # ALPHA_VALUES fails loudly instead of silently dropping a slice.
    for alpha, ax_fc, ax_dfc in zip(ALPHA_VALUES, (ax_fc0, ax_fc1),
                                    (ax_dfc0, ax_dfc1), strict=True):
        for rs in RS_VALUES:
            pair = curves["fc_alpha"][alpha][rs]
            ax_fc.plot(S_GRID, pair["parent"], color=RS_GREY[rs],
                       linestyle="--", linewidth=1.6, zorder=1,
                       label=rf"SCAN, $r_s={rs:g}$")
            ax_fc.plot(S_GRID, pair["model"], color=color, linewidth=2.0,
                       alpha=RS_ALPHA[rs], zorder=2,
                       label=rf"NN, $r_s={rs:g}$")
            ax_dfc.plot(S_GRID, pair["model"] - pair["parent"], color=color,
                        alpha=RS_ALPHA[rs], linewidth=2.0,
                        label=rf"$r_s={rs:g}$")
        ax_dfc.axhline(0.0, color="0.7", linewidth=1.0)
        ax_fc.set_ylabel(
            rf"$F_c(s;\,r_s)$  ($\zeta=0,\ \alpha={alpha:g}$)")
        ax_fc.legend(frameon=False, fontsize=7, ncol=2)
        ax_dfc.set_ylabel(
            rf"$F_c^{{\mathrm{{NN}}}} - F_c^{{\mathrm{{SCAN}}}}$"
            rf"  ($\alpha={alpha:g}$)")
        ax_dfc.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel(r"reduced gradient $s$")
        ax.grid(True, color="0.92", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(f"{arch_name}: pretrained network against the SCAN parent",
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
    """Cross-arch difference panels: every arch's learned correction against
    its OWN parent (a SCAN arch contributes one curve per alpha slice,
    separated by linestyle; the legend names each curve's parent when the
    figure holds more than one)."""
    if any("fx_alpha" in c for c in all_curves.values()):
        return _render_delta_figure_with_scan(all_curves, outdir, footer,
                                              rs_for_fc)
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


def _render_delta_figure_with_scan(all_curves: dict[str, dict], outdir: Path,
                                   footer: str, rs_for_fc: float) -> Path:
    """The cross-arch difference panels when at least one SCAN arch is drawn:
    each SCAN arch appears once per alpha slice (linestyle-separated) against
    parents.scan_*, each PBE arch once against parents.pbe_*, with the parent
    named per curve in the legend."""
    has_pbe = any("fx_model" in c for c in all_curves.values())
    parent_tag = r"\mathrm{parent}" if has_pbe else r"\mathrm{SCAN}"
    fig, (ax_dfx, ax_dfc) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for arch_name, curves in all_curves.items():
        color = ARCH_COLOR.get(arch_name, arch_color(arch_name))
        if "fx_alpha" in curves:
            for alpha in ALPHA_VALUES:
                pair = curves["fx_alpha"][alpha]
                ax_dfx.plot(S_GRID, pair["model"] - pair["parent"],
                            color=color, linestyle=ALPHA_LINESTYLE[alpha],
                            linewidth=2.0,
                            label=rf"{arch_name} (vs SCAN, $\alpha={alpha:g}$)")
                pair = curves["fc_alpha"][alpha][rs_for_fc]
                ax_dfc.plot(S_GRID, pair["model"] - pair["parent"],
                            color=color, linestyle=ALPHA_LINESTYLE[alpha],
                            linewidth=2.0,
                            label=rf"{arch_name} (vs SCAN, $\alpha={alpha:g}$)")
        else:
            ax_dfx.plot(S_GRID, curves["fx_model"] - curves["fx_parent"],
                        color=color, linewidth=2.0,
                        label=f"{arch_name} (vs PBE)")
            pair = curves["fc"][rs_for_fc]
            if pair["parent"] is not None:
                ax_dfc.plot(S_GRID, pair["model"] - pair["parent"],
                            color=color, linewidth=2.0,
                            label=f"{arch_name} (vs PBE)")
    for ax, label in ((ax_dfx,
                       rf"$F_x^{{\mathrm{{NN}}}} - F_x^{{{parent_tag}}}$"),
                      (ax_dfc,
                       rf"$F_c^{{\mathrm{{NN}}}} - F_c^{{{parent_tag}}}$"
                       rf"  ($r_s={rs_for_fc:g}$)")):
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.set_xlabel(r"reduced gradient $s$")
        ax.set_ylabel(label)
        ax.grid(True, color="0.92", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(
        "Pretrained corrections to each architecture's own parent "
        "(PBE for GGA, SCAN for meta-GGA)" if has_pbe else
        "Pretrained corrections to the SCAN parent, all architectures",
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
    if any("fx_alpha" in c for c in all_curves.values()):
        return _write_curves_csv_with_alpha(all_curves, out)
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


def _write_curves_csv_with_alpha(all_curves: dict[str, dict],
                                 out: Path) -> Path:
    """The long-form CSV when at least one SCAN arch is drawn: the schema
    gains an ``alpha`` column, which PBE-arch rows leave empty exactly as fx
    rows leave ``rs`` empty."""
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "channel", "rs", "alpha", "s", "f_model",
                    "f_parent"])
        for arch_name, curves in all_curves.items():
            if "fx_alpha" in curves:
                for alpha in ALPHA_VALUES:
                    pair = curves["fx_alpha"][alpha]
                    for s, fm, fp in zip(S_GRID, pair["model"],
                                         pair["parent"]):
                        w.writerow([arch_name, "fx", "", f"{alpha:g}",
                                    f"{s:.6f}", repr(float(fm)),
                                    repr(float(fp))])
                for alpha in ALPHA_VALUES:
                    for rs in RS_VALUES:
                        pair = curves["fc_alpha"][alpha][rs]
                        for s, fm, fp in zip(S_GRID, pair["model"],
                                             pair["parent"]):
                            w.writerow([arch_name, "fc", f"{rs:g}",
                                        f"{alpha:g}", f"{s:.6f}",
                                        repr(float(fm)), repr(float(fp))])
            else:
                for s, fm, fp in zip(S_GRID, curves["fx_model"],
                                     curves["fx_parent"]):
                    w.writerow([arch_name, "fx", "", "", f"{s:.6f}",
                                repr(float(fm)), repr(float(fp))])
                for rs in RS_VALUES:
                    pair = curves["fc"][rs]
                    parent = (pair["parent"] if pair["parent"] is not None
                              else [float("nan")] * len(S_GRID))
                    for s, fm, fp in zip(S_GRID, pair["model"], parent):
                        w.writerow([arch_name, "fc", f"{rs:g}", "",
                                    f"{s:.6f}", repr(float(fm)),
                                    repr(float(fp))])
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
    parent_by_arch: dict[str, str] = {}
    for arch_name in archs:
        arch, model = load_pretrained_model(run_dir, arch_name)
        parent, curves = compute_curves_for_arch(arch, model)
        parent_by_arch[arch_name] = parent
        dfx = _max_abs_fx_delta(curves)
        if parent == "scan":
            footer = (f"run {run_dir.name}; parent SCAN; slices: F_x at "
                      "rho=1 at the exact iso-orbital indicator alpha in "
                      "{0, 1}, zero other descriptors; F_c at zeta=0, same "
                      f"alpha slices; max|dF_x| {dfx:.2e}. "
                      + _certificate_line(run_dir, arch_name))
        else:
            footer = (f"run {run_dir.name}; slices: F_x at rho=1, zero extra "
                      f"descriptors; F_c at zeta=0; max|dF_x| {dfx:.2e}. "
                      + _certificate_line(run_dir, arch_name))
        out = render_arch_figure(arch_name, curves, outdir, footer)
        if parent == "scan":
            print(f"wrote {out} (parent scan, max|dF_x| {dfx:.3e})")
        else:
            print(f"wrote {out} (max|dF_x| {dfx:.3e})")
        all_curves[arch_name] = curves

    if any(p == "scan" for p in parent_by_arch.values()):
        shared_footer = (
            f"run {run_dir.name}; pretrained networks loaded through the "
            "run's resolved configuration (parent anchor class); each arch "
            "drawn against its own parent: parents.pbe_fx / parents.pbe_fc "
            "(GGA archs), parents.scan_fx / parents.scan_fc at alpha in "
            "{0, 1} (meta-GGA archs); libxc constants")
    else:
        shared_footer = (
            f"run {run_dir.name}; pretrained networks loaded through "
            "the run's resolved configuration (parent anchor class); "
            "parent curves: parents.pbe_fx / parents.pbe_fc (libxc constants)")
    out = render_delta_figure(all_curves, outdir, shared_footer)
    print(f"wrote {out}")
    out = write_curves_csv(all_curves, outdir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
