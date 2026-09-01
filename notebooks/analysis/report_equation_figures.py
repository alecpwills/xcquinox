#!/usr/bin/env python
"""Graphs of the governing equations of REPORT_pretraining_evolution.md and
REPORT_problem_species.md.

Every curve here is produced by CALLING the repository's own implementation --
``xcquinox.alec.parents`` (parent enhancement factors and the pre-image of the
bounded map), ``xcquinox.alec.networks._AlecLOB`` (the bounded map itself),
``xcquinox.alec.metagga`` (the smooth positive part, its inverse and the
iso-orbital indicator), ``xcquinox.alec.pretrain_data_gen`` (the synthetic
mesh) and ``xcquinox.alec.oneshot`` (the spin-polarization clip) -- so a figure
cannot drift from the code it documents. Two deliberate exceptions exist, and
both are contrasts rather than substitutes:

* the bind thresholds of the pre-image clamp are evaluated from the closed form
  in the ``parents.lob_preimage`` docstring and then CHECKED against
  ``lob_preimage`` itself (a parent that far from a bound must map to exactly
  ``+-z_max``);
* the PW92 spin interpolation ``f(zeta)`` and its analytic second derivative are
  written out, because the repository carries ``f`` only inside
  ``parents._pw92_mod_eps``; the written form is checked by reconstructing that
  function's return value from it through the repository's own ``G(r_s)``
  parametrizations and ``_PW_MOD_FZ20``.

The physical coordinates of a synthetic row -- ``rho = 3 / (4 pi r_s^3)``,
``sigma = (2 s k_F rho)^2``, ``tau = alpha tau_unif + tau_W`` -- are formed as
``pretrain_data_gen._mesh_columns`` forms them (lines 1104-1109), so the curves
sit on the same (r_s, s, alpha) coordinates the pretraining mesh does.

Each figure writes a PNG and a same-stem CSV of the plotted series in tidy long
form (``panel, series, x_name, x, y_name, y`` and, where a third quantity is
drawn, one further named column), so every plotted number is recoverable
without re-running the script.

Usage::

    JAX_PLATFORMS=cpu python notebooks/analysis/report_equation_figures.py \
        --outdir notebooks/analysis/figures_report_pretraining
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                       # noqa: E402
import matplotlib                                             # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt                               # noqa: E402
import numpy as np                                            # noqa: E402
from matplotlib.colors import LogNorm                          # noqa: E402

from xcquinox.alec import metagga, parents                     # noqa: E402
from xcquinox.alec.networks import _AlecLOB                    # noqa: E402
from xcquinox.alec.oneshot import _ZETA_BOUNDARY_EPS           # noqa: E402
from xcquinox.alec.pretrain_data_gen import (                  # noqa: E402
    MESH_ALPHA, MESH_RS, MESH_S, MESH_WEIGHT_FRACTION)

# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #

#: Okabe-Ito qualitative palette (Okabe and Ito, "Color Universal Design",
#: 2008): eight hues separated for the three common dichromacies.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

#: Default output directory, relative to this file.
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "figures_report_pretraining"

#: The DIIS trajectory the C2 bistability figure is drawn from.
DEFAULT_C2_LOG = (Path(__file__).resolve().parents[2]
                  / "scratch" / "v6_diag" / "repro_c2_pbe_branch.log")

#: Bounded-map limits in use: the DFS meta-GGA exchange ceiling, the GGA
#: exchange ceiling ``1 + kappa_PBE``, and the correlation squash
#: (``networks._AlecLOB``; ``config`` resolves them per architecture).
LIMIT_X_MGGA = 1.174
LIMIT_X_GGA = 1.804
LIMIT_C = 2.0

#: The pre-image clamp of ``parents.lob_preimage`` (its ``z_max`` default).
Z_MAX = 40.0

HARTREE_PER_KCAL = 627.5094740631


def _style() -> None:
    """Plain scientific defaults: serif-free labels, no gridline clutter."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.7,
        "font.size": 10,
    })


_BASE_FIELDS = ("panel", "series", "x_name", "x", "y_name", "y")


def _write_csv(path, rows, extra_fields=()):
    """Tidy long-form CSV of the plotted series."""
    fields = list(_BASE_FIELDS) + list(extra_fields)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return path


def _series_rows(panel, series, x_name, x, y_name, y, **extra):
    """Rows for one (x, y) series; ``extra`` maps a field name to an array."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"{panel}/{series}: x {x.shape} vs y {y.shape}")
    out = []
    for i in range(x.size):
        row = {"panel": panel, "series": series, "x_name": x_name,
               "x": repr(float(x[i])), "y_name": y_name,
               "y": repr(float(y[i]))}
        for key, arr in extra.items():
            row[key] = repr(float(np.asarray(arr, dtype=float).ravel()[i]))
        out.append(row)
    return out


def _save(fig, outdir, stem, dpi):
    path = Path(outdir) / f"{stem}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Physical coordinates of a synthetic row
# --------------------------------------------------------------------------- #

_K_F_COEF = (3.0 * math.pi ** 2) ** (1.0 / 3.0)
#: ``tau_unif = _TAU_UNIF_COEF rho^(5/3)``.
_TAU_UNIF_COEF = 0.3 * _K_F_COEF ** 2


def rho_from_rs(rs):
    """``rho = 3 / (4 pi r_s^3)`` (``pretrain_data_gen._mesh_columns`` l. 1104)."""
    return 3.0 / (4.0 * math.pi * np.asarray(rs, dtype=float) ** 3)


def sigma_from_s(rho, s):
    """``sigma = (s 2 k_F rho)^2`` (``_mesh_columns`` l. 1105-1106): the gradient
    invariant realizing a reduced gradient ``s`` at density ``rho``."""
    rho = np.asarray(rho, dtype=float)
    k_f = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)
    return (np.asarray(s, dtype=float) * 2.0 * k_f * rho) ** 2


def tau_unif(rho):
    """``tau_unif = (3/10)(3 pi^2)^(2/3) rho^(5/3)`` (``metagga`` module head)."""
    return _TAU_UNIF_COEF * np.asarray(rho, dtype=float) ** (5.0 / 3.0)


def lob_slope(z, limit):
    """``L'(z)`` of ``networks._AlecLOB`` by ``jax.grad`` through the class."""
    lob = _AlecLOB(limit=limit)
    grad = jax.grad(lambda zz: 1.0 + lob(zz))
    return np.asarray(jax.vmap(grad)(jnp.asarray(z, dtype=jnp.float64)))


def lob_map(z, limit):
    """``F = 1 + L(z)`` of ``networks._AlecLOB``."""
    lob = _AlecLOB(limit=limit)
    return np.asarray(1.0 + lob(jnp.asarray(z, dtype=jnp.float64)))


def parent_slope(f_parent, limit):
    """``L'`` at the parent's pre-image: the factor every anchored parameter
    gradient carries (``REPORT_pretraining_evolution.md`` Section 6.2)."""
    z = np.asarray(parents.lob_preimage(jnp.asarray(f_parent), limit))
    return lob_slope(z, limit), z


# --------------------------------------------------------------------------- #
# Figure 1 -- the bounded map and its pre-image
# --------------------------------------------------------------------------- #

def make_bounded_map(outdir, dpi):
    stem = "bounded_map"
    limits = (LIMIT_X_MGGA, LIMIT_X_GGA, LIMIT_C)
    colors = (OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"])
    labels = (r"$\Lambda = 1.174$ (meta-GGA $F_x$)",
              r"$\Lambda = 1.804$ (GGA $F_x$)",
              r"$\Lambda = 2.0$ ($F_c$)")

    z = np.linspace(-45.0, 45.0, 1801)
    rows = []
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    for lim, col, lab in zip(limits, colors, labels):
        f = lob_map(z, lim)
        ax0.plot(z, f, color=col, label=lab)
        ax0.axhline(lim, color=col, lw=0.7, ls=":", alpha=0.8)
        rows += _series_rows("a_map", f"F_limit_{lim}", "z", z, "F", f)

    for sign in (-1.0, 1.0):
        ax0.axvline(sign * Z_MAX, color=OKABE_ITO["black"], lw=0.8, ls="--",
                    alpha=0.6)
    ax0.axhline(0.0, color=OKABE_ITO["black"], lw=0.6, alpha=0.4)
    ax0.text(Z_MAX, 0.12, r" $z_{\max}=+40$", fontsize=8, rotation=90,
             va="bottom", ha="left")
    ax0.text(-Z_MAX, 0.12, r"$z_{\max}=-40$ ", fontsize=8, rotation=90,
             va="bottom", ha="right")
    ax0.plot([0.0], [1.0], marker="o", ms=5, mfc="white",
             color=OKABE_ITO["black"], zorder=5, lw=0)
    ax0.annotate(r"$F(0) = 1$ for every $\Lambda$", xy=(0.0, 1.0),
                 xytext=(9.0, 0.50), fontsize=8.5,
                 arrowprops=dict(arrowstyle="-", lw=0.7,
                                 color=OKABE_ITO["black"]))
    ax0.set_xlabel(r"pre-activation $z$ (dimensionless)")
    ax0.set_ylabel(r"$F = 1 + L(z)$ (dimensionless)")
    ax0.set_title(r"(a) $L(z) = \Lambda\,\sigma(z - \ln(\Lambda-1)) - 1$",
                  loc="left")
    ax0.set_xlim(-46, 46)
    ax0.set_ylim(-0.08, 2.30)
    ax0.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    # Panel (b): the pre-image, sampled uniformly in the map's own coordinate so
    # both clamped tails are resolved.
    t = np.linspace(-46.0, 46.0, 1601)
    u = 1.0 / (1.0 + np.exp(-t))
    thresh_lines = []
    for lim, col, lab in zip(limits, colors, labels):
        f = lim * u
        zz = np.asarray(parents.lob_preimage(jnp.asarray(f), lim))
        ax1.plot(f, zz, color=col, label=lab)
        rows += _series_rows("b_preimage", f"z_limit_{lim}", "F_parent", f,
                             "z_parent", zz)
        upper = lim * (lim - 1.0) * math.exp(-Z_MAX)
        lower = lim * math.exp(-Z_MAX) / (lim - 1.0)
        # Oracle: a parent exactly that far from a bound maps to +-z_max.
        z_hi = float(parents.lob_preimage(lim - upper, lim))
        z_lo = float(parents.lob_preimage(lower, lim))
        if not (math.isclose(z_hi, Z_MAX, rel_tol=1e-9)
                and math.isclose(z_lo, -Z_MAX, rel_tol=1e-9)):
            raise AssertionError(
                f"bind thresholds do not reproduce the clamp at limit={lim}: "
                f"z(upper)={z_hi}, z(lower)={z_lo}")
        thresh_lines.append(
            f"$\\Lambda$={lim}:  {upper:.2e} / {lower:.2e}")
        rows += _series_rows("c_bind", f"bind_limit_{lim}", "bound",
                             [lim, 0.0], "distance_in_F", [upper, lower])

    for sign in (-1.0, 1.0):
        ax1.axhline(sign * Z_MAX, color=OKABE_ITO["black"], lw=0.8, ls="--",
                    alpha=0.6)
    ax1.text(1.28, Z_MAX, r"clamp $z = +40$", fontsize=8, va="bottom")
    ax1.text(0.02, -Z_MAX, r"clamp $z = -40$", fontsize=8, va="top")
    ax1.set_xlabel(r"parent value $F_{\mathrm{parent}}$ (dimensionless)")
    ax1.set_ylabel(r"$z_{\mathrm{parent}}$ (dimensionless)")
    ax1.set_title(r"(b) pre-image $z = \ln[(\Lambda-1)F / (\Lambda-F)]$",
                  loc="left")
    ax1.set_xlim(-0.03, 2.05)
    ax1.set_ylim(-52, 52)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.02, 0.82))
    ax1.text(0.62, 0.30,
             "clamp binds within (upper / lower)\n" + "\n".join(thresh_lines),
             transform=ax1.transAxes, fontsize=7.8, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7",
                       lw=0.6))

    fig.suptitle("Bounded map and its pre-image", fontsize=12.5, y=1.01)
    path = _save(fig, outdir, stem, dpi)
    _write_csv(Path(outdir) / f"{stem}.csv", rows)
    return path


# --------------------------------------------------------------------------- #
# Figure 2 -- pre-image sensitivity
# --------------------------------------------------------------------------- #

def make_preimage_sensitivity(outdir, dpi):
    stem = "preimage_sensitivity"
    rows = []
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    # --- panel (a): exchange -------------------------------------------------
    s = np.linspace(0.0, 20.0, 801)
    rho = 1.0                       # F_x is scale-free; s is the only argument
    sigma = sigma_from_s(rho, s)

    fx_pbe = np.asarray(parents.pbe_fx(rho, jnp.asarray(sigma)))
    lp_pbe, z_pbe = parent_slope(fx_pbe, LIMIT_X_GGA)
    ax0.plot(s, lp_pbe, color=OKABE_ITO["vermillion"],
             label=r"PBE $F_x$, $\Lambda = 1.804$")
    rows += _series_rows("a_exchange", "pbe_fx", "s", s, "L_prime", lp_pbe,
                         F_parent=fx_pbe, z_parent=z_pbe)

    scan_style = {0.0: ("-", OKABE_ITO["blue"]), 1.0: ("--", OKABE_ITO["sky"])}
    for alpha, (ls, col) in scan_style.items():
        fx = np.asarray(parents.scan_fx(rho, jnp.asarray(sigma), alpha))
        lp, zz = parent_slope(fx, LIMIT_X_MGGA)
        ax0.plot(s, lp, color=col, ls=ls,
                 label=rf"SCAN $F_x$, $\alpha = {alpha:.0f}$, $\Lambda = 1.174$")
        rows += _series_rows("a_exchange", f"scan_fx_alpha{alpha:.0f}", "s", s,
                             "L_prime", lp, F_parent=fx, z_parent=zz)
        if alpha == 0.0:
            lp_scan0 = lp

    ax0.plot([0.0], [lp_pbe[0]], marker="o", ms=5, color=OKABE_ITO["vermillion"],
             lw=0)
    ax0.annotate(rf"$L' = {lp_pbe[0]:.4f}$ at $s = 0$", xy=(0.0, lp_pbe[0]),
                 xytext=(1.6, 0.505), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=0.7,
                                 color=OKABE_ITO["vermillion"]))
    ax0.annotate(rf"$L' = {lp_pbe[-1]:.4f}$ at $s = 20$",
                 xy=(20.0, lp_pbe[-1]), xytext=(11.2, 0.090), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=0.7,
                                 color=OKABE_ITO["vermillion"]))
    ax0.plot([0.0], [lp_scan0[0]], marker="s", ms=5, mfc="white",
             color=OKABE_ITO["blue"], lw=0)
    ax0.annotate(r"$L' = 0$", xy=(0.0, 0.0), xytext=(0.55, 0.048),
                 fontsize=8.5, color=OKABE_ITO["blue"],
                 arrowprops=dict(arrowstyle="->", lw=0.7,
                                 color=OKABE_ITO["blue"]))
    ax0.set_xlabel(r"reduced gradient $s$ (dimensionless)")
    ax0.set_ylabel(r"$L'(z_{\mathrm{parent}}) = F(1 - F/\Lambda)$"
                   "\n(dimensionless)")
    ax0.set_title("(a) exchange sensitivity", loc="left")
    ax0.set_xlim(-0.5, 20.5)
    ax0.set_ylim(-0.02, 0.85)
    ax0.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99))

    inset = ax0.inset_axes([0.60, 0.40, 0.33, 0.24])
    m = s <= 3.0
    inset.semilogy(s[m], lp_scan0[m], color=OKABE_ITO["blue"], lw=1.2)
    inset.set_xlim(0, 3)
    inset.set_ylim(1e-11, 1e0)
    inset.set_xlabel(r"$s$", fontsize=7.5, labelpad=1)
    inset.set_ylabel(r"$L'$", fontsize=7.5, labelpad=1)
    inset.tick_params(labelsize=6.5)
    inset.set_title(r"SCAN $\alpha=0$, log scale", fontsize=7.4, pad=3)
    ax0.text(1.0, 0.635,
             r"SCAN $\alpha = 0$ at $s = 0$: $F_x = \Lambda$, so"
             "\n"
             r"$z_{\mathrm{parent}}$ clamps at $+40$ and $L'$ vanishes",
             fontsize=8.2, va="top")

    # --- panel (b): the correlation mirror -----------------------------------
    sc = np.linspace(0.0, 6.0, 601)
    rs_c = 2.0
    rho_c = float(rho_from_rs(rs_c))
    sigma_c = sigma_from_s(rho_c, sc)
    fc = np.asarray(parents.pbe_fc(rho_c, jnp.asarray(sigma_c), 0.0))
    lp_c, z_c = parent_slope(fc, LIMIT_C)

    ax1.semilogy(sc, fc, color=OKABE_ITO["black"], ls=":",
                 label=r"$F_c^{\mathrm{PBE}}$")
    ax1.semilogy(sc, lp_c, color=OKABE_ITO["green"],
                 label=r"$L'(z_{\mathrm{parent}})$, $\Lambda = 2.0$")
    rows += _series_rows("b_correlation", "pbe_fc", "s", sc, "F_parent", fc)
    rows += _series_rows("b_correlation", "pbe_fc_slope", "s", sc, "L_prime",
                         lp_c, F_parent=fc, z_parent=z_c)

    label_offsets = {0.0: (8, -13), 2.0: (7, 9), 6.0: (-4, 11)}
    for s_mark in (0.0, 2.0, 6.0):
        i = int(np.argmin(np.abs(sc - s_mark)))
        ax1.plot([sc[i]], [lp_c[i]], marker="o", ms=4.5,
                 color=OKABE_ITO["green"], lw=0)
        ax1.annotate(f"{lp_c[i]:.4f}", xy=(sc[i], lp_c[i]),
                     xytext=label_offsets[s_mark], textcoords="offset points",
                     fontsize=8, ha="right" if s_mark == 6.0 else "left")
    ax1.set_xlabel(r"reduced gradient $s$ (dimensionless)")
    ax1.set_ylabel(r"value (dimensionless)")
    ax1.set_title(r"(b) correlation mirror, $r_s = 2$ bohr, $\zeta = 0$",
                  loc="left")
    ax1.set_xlim(-0.15, 6.4)
    ax1.set_ylim(5e-4, 3.0)
    ax1.legend(loc="lower left")
    ax1.text(0.40, 0.90,
             r"$L' \to F_c$ as $F_c \to 0$:" "\n"
             "trainability vanishes where\nthe parent does",
             transform=ax1.transAxes, fontsize=8, va="top")

    fig.suptitle("Pre-image sensitivity of the anchored correction",
                 fontsize=12.5, y=1.01)
    path = _save(fig, outdir, stem, dpi)
    _write_csv(Path(outdir) / f"{stem}.csv", rows,
               extra_fields=("F_parent", "z_parent"))
    return path


# --------------------------------------------------------------------------- #
# Figure 3 -- the smooth positive part
# --------------------------------------------------------------------------- #

def make_smooth_positive_part(outdir, dpi):
    stem = "smooth_positive_part"
    w = metagga._ALPHA_SMOOTHING_WIDTH
    rows = []
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    x = np.linspace(-5e-5, 5e-5, 1001)
    p = np.asarray(metagga.smooth_positive_part(jnp.asarray(x), w))
    hard = np.maximum(x, 0.0)

    ax0.plot(x, hard, color=OKABE_ITO["black"], ls="--", lw=1.3,
             label=r"$\max(x, 0)$")
    ax0.plot(x, p, color=OKABE_ITO["vermillion"],
             label=rf"$p(x)$, $w = {w:.0e}$")
    rows += _series_rows("a_value", "hard_positive_part", "x", x,
                         "value", hard)
    rows += _series_rows("a_value", "smooth_positive_part", "x", x,
                         "value", p)

    p0 = float(metagga.smooth_positive_part(jnp.array(0.0), w))
    ax0.plot([0.0], [p0], marker="o", ms=5, color=OKABE_ITO["vermillion"], lw=0)
    ax0.annotate(rf"$p(0) = w/2 = {p0:.0e}$" "\n" r"slope $1/2$",
                 xy=(0.0, p0), xytext=(-4.7e-5, 1.45e-5), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=0.7,
                                 color=OKABE_ITO["vermillion"]))
    ax0.set_xlabel(r"raw indicator $x$ (dimensionless)"
                   "\n" r"$x = (\tau - \tau_W)/\tau_{\mathrm{unif}}$")
    ax0.set_ylabel("positive part (dimensionless)")
    ax0.set_title(r"(a) $p(x) = \frac{1}{2}(x + \sqrt{x^2 + w^2})$", loc="left")
    ax0.legend(loc="lower right")
    ax0.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # inset: the excess over the hard clip, log-log, against its own asymptote
    ax_abs = np.logspace(-9.0, np.log10(5e-5), 300)
    p_abs = np.asarray(metagga.smooth_positive_part(jnp.asarray(ax_abs), w))
    excess = p_abs - ax_abs
    asym = w * w / (4.0 * ax_abs)
    inset = ax0.inset_axes([0.16, 0.55, 0.34, 0.35])
    inset.loglog(ax_abs, excess, color=OKABE_ITO["blue"], lw=1.2)
    inset.loglog(ax_abs, asym, color=OKABE_ITO["black"], ls=":", lw=1.0)
    inset.axhline(w / 2.0, color=OKABE_ITO["orange"], ls="--", lw=0.9)
    inset.set_xlabel(r"$|x|$", fontsize=7.5, labelpad=1)
    inset.set_ylabel(r"$p - \max(x,0)$", fontsize=7.5, labelpad=1)
    inset.tick_params(labelsize=6.5)
    inset.set_ylim(1e-7, 2e-5)
    inset.text(1.3e-9, 6.5e-6, r"$w/2$", fontsize=7,
               color=OKABE_ITO["orange"])
    inset.text(4e-6, 1.5e-7, r"$w^2/4|x|$", fontsize=7)
    rows += _series_rows("a_inset", "excess_over_hard_clip", "abs_x", ax_abs,
                         "excess", excess)
    rows += _series_rows("a_inset", "asymptote_w2_over_4x", "abs_x", ax_abs,
                         "excess", asym)

    # --- panel (b): the inversion round trip ---------------------------------
    back = np.asarray(metagga.invert_smooth_positive_part(jnp.asarray(p), w))
    err = np.abs(back - x)
    # Conditioning of the inversion: dx/dp = 1 + w^2/(4 p^2), so one unit of
    # round-off in p returns eps |x| (1 + w^2 / 4 p^2) in x. This is a
    # first-order SCALE, not a rigorous bound -- several roundings enter the
    # forward and inverse evaluations -- and the measured excursion above it is
    # reported rather than absorbed into a fitted constant.
    scale = np.finfo(float).eps * np.maximum(np.abs(x), w) * (
        1.0 + w * w / (4.0 * p * p))
    ratio_max = float(np.max(err / scale))
    ax1.plot(x, err, color=OKABE_ITO["purple"], lw=1.3,
             label=r"$|p^{-1}(p(x)) - x|$")
    ax1.plot(x, scale, color=OKABE_ITO["black"], ls=":", lw=1.0,
             label=r"conditioning scale $\varepsilon\,|x|\,(1 + w^2/4p^2)$")
    rows += _series_rows("b_roundtrip", "inversion_error", "x", x,
                         "abs_error", err)
    rows += _series_rows("b_roundtrip", "conditioning_scale", "x", x,
                         "abs_error", scale)
    ax1.set_xlabel(r"raw indicator $x$ (dimensionless)")
    ax1.set_ylabel(r"absolute error (dimensionless)")
    ax1.set_title("(b) exactness of the stored-column inversion", loc="left")
    ax1.legend(loc="upper center")
    ax1.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax1.text(0.03, 0.62,
             f"max error over the grid: {err.max():.2e}\n"
             f"max relative: "
             f"{np.max(err[x != 0] / np.abs(x[x != 0])):.2e}\n"
             f"largest excursion above the conditioning\n"
             f"scale: {ratio_max:.2f}x",
             transform=ax1.transAxes, fontsize=8, va="top")

    fig.suptitle(rf"Smooth positive part at width $w = {w:.0e}$",
                 fontsize=12.5, y=1.06)
    path = _save(fig, outdir, stem, dpi)
    _write_csv(Path(outdir) / f"{stem}.csv", rows)
    return path


# --------------------------------------------------------------------------- #
# Figure 4 -- the indicator ceiling
# --------------------------------------------------------------------------- #

def make_alpha_ceiling(outdir, dpi):
    stem = "alpha_ceiling"
    amax = metagga._ALPHA_MAX
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    alpha = np.logspace(math.log10(amax), 7.0, 401)
    rho = 1.0
    colors = {0.0: OKABE_ITO["vermillion"], 1.0: OKABE_ITO["blue"],
              4.0: OKABE_ITO["green"]}
    saturation = {}
    for s, col in colors.items():
        sigma = float(sigma_from_s(rho, s))
        f_cap = float(parents.scan_fx(rho, sigma, amax))
        f = np.asarray(parents.scan_fx(rho, jnp.asarray(sigma),
                                       jnp.asarray(alpha)))
        d = np.abs(f - f_cap)
        ax.loglog(alpha, d, color=col, label=rf"$s = {s:.0f}$")
        rows += _series_rows("a_residual", f"s{s:.0f}", "alpha", alpha,
                             "abs_delta_Fx", d, Fx=f)
        saturation[s] = float(d[-1])

    ax.axvline(amax, color=OKABE_ITO["black"], lw=0.9, ls="--", alpha=0.7)
    ax.text(amax * 1.2, 1.35e-5,
            rf"ceiling $\alpha = {amax:.0f}$ (`metagga._ALPHA_MAX`)",
            fontsize=8.5, va="bottom")
    ax.axhline(saturation[0.0], color=OKABE_ITO["vermillion"], lw=0.8, ls=":",
               alpha=0.8)
    ax.annotate(rf"saturation ${saturation[0.0]:.3g}$ at $s = 0$",
                xy=(1e6, saturation[0.0]), xytext=(1.5e3, 3.4e-3),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.7,
                                color=OKABE_ITO["vermillion"]))
    ax.set_xlabel(r"exact indicator $\alpha$ (dimensionless)")
    ax.set_ylabel(r"$|F_x^{\mathrm{SCAN}}(s,\alpha)"
                  r" - F_x^{\mathrm{SCAN}}(s,100)|$ (dimensionless)")
    ax.set_title("Indicator ceiling: SCAN exchange residual on a capped row")
    ax.set_ylim(1e-5, 6e-3)
    ax.legend(loc="lower right", title="reduced gradient",
              title_fontsize=8.5)
    ax.text(0.055, 0.30,
            "the residual is zero by construction at the cap;\n"
            "a row whose exact indicator exceeds it is\n"
            "evaluated at 100 and prices this difference",
            transform=ax.transAxes, fontsize=8, va="top")

    path = _save(fig, outdir, stem, dpi)
    _write_csv(Path(outdir) / f"{stem}.csv", rows, extra_fields=("Fx",))
    return path


# --------------------------------------------------------------------------- #
# Figure 5 -- parent enhancement factors
# --------------------------------------------------------------------------- #

def make_parent_enhancement(outdir, dpi):
    stem = "parent_enhancement"
    rows = []
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    s = np.linspace(0.0, 6.0, 601)
    rho = 1.0
    sigma = sigma_from_s(rho, s)

    fx_pbe = np.asarray(parents.pbe_fx(rho, jnp.asarray(sigma)))
    ax0.plot(s, fx_pbe, color=OKABE_ITO["vermillion"], label="PBE")
    rows += _series_rows("a_exchange", "pbe_fx", "s", s, "F_x", fx_pbe)
    for alpha, ls, col in ((0.0, "-", OKABE_ITO["blue"]),
                           (1.0, "--", OKABE_ITO["sky"])):
        fx = np.asarray(parents.scan_fx(rho, jnp.asarray(sigma), alpha))
        ax0.plot(s, fx, color=col, ls=ls,
                 label=rf"SCAN, $\alpha = {alpha:.0f}$")
        rows += _series_rows("a_exchange", f"scan_fx_alpha{alpha:.0f}", "s", s,
                             "F_x", fx)

    ax0.axhline(LIMIT_X_GGA, color=OKABE_ITO["vermillion"], lw=0.7, ls=":")
    ax0.axhline(LIMIT_X_MGGA, color=OKABE_ITO["blue"], lw=0.7, ls=":")
    ax0.text(5.95, LIMIT_X_GGA + 0.012, r"$1 + \kappa = 1.804$", fontsize=7.5,
             ha="right", color=OKABE_ITO["vermillion"])
    ax0.text(5.95, LIMIT_X_MGGA + 0.012, r"$h_x^0 = 1.174$", fontsize=7.5,
             ha="right", color=OKABE_ITO["blue"])
    ax0.plot([0.0], [fx_pbe[0]], marker="o", ms=4.5, lw=0,
             color=OKABE_ITO["vermillion"])
    ax0.plot([0.0], [LIMIT_X_MGGA], marker="s", ms=4.5, lw=0,
             color=OKABE_ITO["blue"])
    ax0.annotate(r"$F_x(0) = 1$ (PBE and SCAN $\alpha=1$);"
                 "\n" r"SCAN $\alpha=0$ starts at its ceiling",
                 xy=(0.0, 1.0), xytext=(2.15, 1.28), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=0.7,
                                 color=OKABE_ITO["black"]))
    ax0.set_xlabel(r"reduced gradient $s$ (dimensionless)")
    ax0.set_ylabel(r"$F_x$ (dimensionless)")
    ax0.set_title("(a) exchange", loc="left")
    ax0.set_xlim(-0.1, 6.1)
    ax0.set_ylim(0.55, 1.95)
    ax0.legend(loc="lower left")

    rs_colors = {0.5: OKABE_ITO["orange"], 2.0: OKABE_ITO["green"],
                 5.0: OKABE_ITO["purple"]}
    for rs, col in rs_colors.items():
        rho_c = float(rho_from_rs(rs))
        sigma_c = sigma_from_s(rho_c, s)
        fc_pbe = np.asarray(parents.pbe_fc(rho_c, jnp.asarray(sigma_c), 0.0))
        fc_scan = np.asarray(parents.scan_fc(rho_c, jnp.asarray(sigma_c),
                                             0.0, 0.0))
        ax1.plot(s, fc_pbe, color=col, ls="-",
                 label=rf"PBE, $r_s = {rs}$")
        ax1.plot(s, fc_scan, color=col, ls="--",
                 label=rf"SCAN $\alpha=0$, $r_s = {rs}$")
        rows += _series_rows("b_correlation", f"pbe_fc_rs{rs}", "s", s,
                             "F_c", fc_pbe)
        rows += _series_rows("b_correlation", f"scan_fc_alpha0_rs{rs}", "s", s,
                             "F_c", fc_scan)

    ax1.set_xlabel(r"reduced gradient $s$ (dimensionless)")
    ax1.set_ylabel(r"$F_c$ relative to the PW92 baseline (dimensionless)")
    ax1.set_title(r"(b) correlation, $\zeta = 0$", loc="left")
    ax1.set_xlim(-0.1, 6.1)
    ax1.set_ylim(-0.03, 1.10)
    ax1.legend(loc="upper right", ncol=1)
    ax1.text(2.45, 0.60,
             "PBE correlation is switched off by the\n"
             "gradient; the SCAN single-orbital branch\n"
             "retains a finite floor",
             fontsize=8, va="top")

    fig.suptitle("Parent enhancement factors at libxc constants",
                 fontsize=12.5, y=1.01)
    path = _save(fig, outdir, stem, dpi)
    _write_csv(Path(outdir) / f"{stem}.csv", rows)
    return path


# --------------------------------------------------------------------------- #
# Figure 6 -- the PW92 spin interpolation and its curvature
# --------------------------------------------------------------------------- #

_FZ_NORM = 2.0 ** (4.0 / 3.0) - 2.0


def f_spin(zeta):
    """PW92 spin interpolation ``f(zeta)`` (Perdew and Wang, PRB 45, 13244
    (1992), eq. 9), written out because the repository carries it only inside
    ``parents._pw92_mod_eps``; the written form is verified against that
    function by :func:`_verify_f_spin`."""
    z = np.asarray(zeta, dtype=float)
    return ((1.0 + z) ** (4.0 / 3.0) + (1.0 - z) ** (4.0 / 3.0) - 2.0) / _FZ_NORM


def f_spin_second_analytic(zeta):
    """``f''(zeta) = (4/9)[(1+z)^(-2/3) + (1-z)^(-2/3)] / (2^(4/3) - 2)``."""
    z = np.asarray(zeta, dtype=float)
    return (4.0 / 9.0) * ((1.0 + z) ** (-2.0 / 3.0)
                          + (1.0 - z) ** (-2.0 / 3.0)) / _FZ_NORM


def _verify_f_spin(rs_values=(0.5, 2.0), n=81):
    """Reconstruct ``parents._pw92_mod_eps`` from :func:`f_spin` through the
    repository's own ``G(r_s)`` sets; returns the worst absolute deviation."""
    worst = 0.0
    zs = np.linspace(-0.999, 0.999, n)
    fz = f_spin(zs)
    z4 = zs ** 4
    for rs in rs_values:
        eps_repo = np.asarray(parents._pw92_mod_eps(
            rs, jnp.asarray(zs), jnp.asarray(1.0 + zs), jnp.asarray(1.0 - zs)))
        g0 = float(parents._pw92_mod_g(0, rs))
        g1 = float(parents._pw92_mod_g(1, rs))
        g2 = float(parents._pw92_mod_g(2, rs))
        eps_built = (g0 - g2 * fz / parents._PW_MOD_FZ20 * (1.0 - z4)
                     + (g1 - g0) * fz * z4)
        worst = max(worst, float(np.max(np.abs(eps_repo - eps_built))))
    return worst


def make_zeta_pole(outdir, dpi):
    stem = "zeta_pole"
    z_bound = 1.0 - _ZETA_BOUNDARY_EPS
    rows = []

    worst_eps = _verify_f_spin()
    if worst_eps > 1e-15:
        raise AssertionError(
            "the written f(zeta) does not reproduce parents._pw92_mod_eps "
            f"(worst absolute deviation {worst_eps:.3e})")
    fz20_dev = abs(f_spin_second_analytic(0.0) - parents._PW_MOD_FZ20)
    if fz20_dev > 1e-15:
        raise AssertionError(
            "the analytic f''(0) does not reproduce parents._PW_MOD_FZ20 "
            f"(deviation {fz20_dev:.3e})")

    core = np.linspace(0.0, 0.9, 181)
    tail = 1.0 - np.logspace(-1.0, math.log10(_ZETA_BOUNDARY_EPS), 141)
    half = np.unique(np.concatenate([core, tail, [0.5, 0.9, z_bound]]))
    half = half[half <= z_bound]
    zeta = np.unique(np.concatenate([-half, half]))

    fz = f_spin(zeta)
    f2_an = f_spin_second_analytic(zeta)
    # Central second difference at a step scaled by the distance to the pole so
    # the stencil never leaves [-1, 1].
    h = np.minimum(1e-3, 0.05 * (1.0 - np.abs(zeta)))
    f2_fd = (f_spin(zeta + h) - 2.0 * f_spin(zeta) + f_spin(zeta - h)) / h ** 2

    clean = np.abs(zeta) <= 0.99
    worst_rel = float(np.max(np.abs(f2_fd[clean] - f2_an[clean])
                             / f2_an[clean]))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    ax0.plot(zeta, fz, color=OKABE_ITO["blue"], label=r"$f(\zeta)$")
    rows += _series_rows("a_interpolation", "f_zeta", "zeta", zeta, "f", fz)
    for sign in (-1.0, 1.0):
        ax0.axvline(sign * z_bound, color=OKABE_ITO["black"], lw=0.9, ls="--",
                    alpha=0.7)
    ax0.set_xlabel(r"spin polarization $\zeta$ (dimensionless)")
    ax0.set_ylabel(r"$f(\zeta)$ (dimensionless)")
    ax0.set_title(r"(a) $f = [(1+\zeta)^{4/3} + (1-\zeta)^{4/3} - 2]"
                  r"/(2^{4/3} - 2)$", loc="left", fontsize=10)
    ax0.set_xlim(-1.06, 1.06)
    ax0.set_ylim(-0.05, 1.12)
    ax0.text(0.0, 0.30,
             rf"$f(\pm 1) = 1$, $f(0) = 0$" "\n"
             rf"$f''(0) = {parents._PW_MOD_FZ20:.10f}$" "\n"
             r"(`parents._PW_MOD_FZ20`, reproduced exactly)",
             ha="center", fontsize=8)
    ax0.text(z_bound, 0.62, r" clip $|\zeta| = 1 - 10^{-6}$", fontsize=8,
             rotation=90, va="center", ha="right")
    ax0.legend(loc="upper center")

    ax1.semilogy(zeta, f2_an, color=OKABE_ITO["vermillion"],
                 label="analytic")
    step = max(1, zeta.size // 90)
    ax1.semilogy(zeta[::step], f2_fd[::step], lw=0, marker="o", ms=3.2,
                 mfc="none", color=OKABE_ITO["black"],
                 label="central difference")
    rows += _series_rows("b_curvature", "f_second_analytic", "zeta", zeta,
                         "f_second", f2_an)
    rows += _series_rows("b_curvature", "f_second_finite_difference", "zeta",
                         zeta, "f_second", f2_fd, fd_step=h)
    for sign in (-1.0, 1.0):
        ax1.axvline(sign * z_bound, color=OKABE_ITO["black"], lw=0.9, ls="--",
                    alpha=0.7)
    ax1.set_xlabel(r"spin polarization $\zeta$ (dimensionless)")
    ax1.set_ylabel(r"$f''(\zeta)$ (dimensionless)")
    ax1.set_title(r"(b) curvature $f''(\zeta) = (4/9)[(1+\zeta)^{-2/3} + "
                  r"(1-\zeta)^{-2/3}]/(2^{4/3}-2)$", loc="left", fontsize=9.5)
    ax1.set_xlim(-1.06, 1.06)
    ax1.set_ylim(1.0, 3e4)
    ax1.legend(loc="upper center")
    ax1.text(0.12, 0.72,
             "analytic vs central difference\n"
             rf"agree to {worst_rel:.1e} relative for $|\zeta| \leq 0.99$",
             transform=ax1.transAxes, fontsize=8, va="top")

    inset = ax1.inset_axes([0.62, 0.11, 0.35, 0.33])
    m = zeta > 0
    inset.loglog(1.0 - zeta[m], f2_an[m], color=OKABE_ITO["vermillion"], lw=1.2)
    inset.axvline(_ZETA_BOUNDARY_EPS, color=OKABE_ITO["black"], lw=0.9,
                  ls="--")
    inset.set_xlim(2e-7, 2.0)
    inset.set_xlabel(r"$1 - \zeta$", fontsize=7.5, labelpad=1)
    inset.set_ylabel(r"$f''$", fontsize=7.5, labelpad=1)
    inset.tick_params(labelsize=6.5)
    inset.set_title(r"pole, clipped at $10^{-6}$", fontsize=7.5, pad=2)

    fig.suptitle("PW92 spin interpolation and the clipped curvature pole",
                 fontsize=12.5, y=1.01)
    path = _save(fig, outdir, stem, dpi)
    _write_csv(Path(outdir) / f"{stem}.csv", rows, extra_fields=("fd_step",))
    return path


# --------------------------------------------------------------------------- #
# Figure 7 -- the synthetic pretraining mesh
# --------------------------------------------------------------------------- #

def make_dfs_mesh(outdir, dpi):
    stem = "dfs_mesh"
    amax = metagga._ALPHA_MAX
    rs = np.asarray(MESH_RS, dtype=float)[:, None, None]
    s = np.asarray(MESH_S, dtype=float)[None, :, None]
    al = np.asarray(MESH_ALPHA, dtype=float)[None, None, :]
    rs, s, al = np.broadcast_arrays(rs, s, al)
    rs, s, al = rs.ravel(), s.ravel(), al.ravel()
    n_nodes = rs.size

    rows = _series_rows("a_projection", "mesh_node", "s", s, "alpha", al,
                        r_s=rs)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11.0, 4.5), gridspec_kw={"width_ratios": [1.75, 1.0]})

    # Deterministic dodge in s by r_s index so all nodes of the projection are
    # visible; the s coordinate of each node is the undodged value in the CSV.
    rs_index = np.array([MESH_RS.index(v) for v in rs], dtype=float)
    dodge = (rs_index - (len(MESH_RS) - 1) / 2.0) * 0.052
    sc = ax0.scatter(s + dodge, al, c=rs, cmap="viridis",
                     norm=LogNorm(vmin=min(MESH_RS), vmax=max(MESH_RS)),
                     s=22, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax0, pad=0.02)
    cb.set_label(r"$r_s$ (bohr)", fontsize=9.5)
    ax0.set_xlabel(r"reduced gradient $s$ (dimensionless)"
                   "\n" r"(nodes dodged in $s$ by $r_s$ for visibility)")
    ax0.set_ylabel(r"iso-orbital indicator $\alpha$ (dimensionless)")
    ax0.set_title(r"(a) $(s, \alpha)$ projection", loc="left")
    ax0.set_xlim(-0.4, 5.4)
    ax0.set_ylim(-0.4, 6.9)
    ax0.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax0.set_xticks(list(MESH_S), minor=True)
    ax0.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax0.set_yticks(list(MESH_ALPHA), minor=True)
    ax0.tick_params(labelsize=8)
    ax0.text(0.03, 0.97,
             f"{n_nodes} nodes = {len(MESH_RS)} "
             r"$r_s$ $\times$ " f"{len(MESH_S)} " r"$s$ $\times$ "
             f"{len(MESH_ALPHA)} " r"$\alpha$" "\n"
             f"mesh share of the total integration weight: "
             f"{MESH_WEIGHT_FRACTION}",
             transform=ax0.transAxes, fontsize=8.2, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7",
                       lw=0.6))

    # Panel (b): the alpha axis against the stored column's ceiling.
    nodes = np.asarray(MESH_ALPHA, dtype=float)
    idx = np.arange(1, nodes.size + 1, dtype=float)
    ax1.axhspan(max(MESH_ALPHA), amax, color=OKABE_ITO["orange"], alpha=0.25,
                lw=0)
    ax1.plot(idx, nodes, marker="o", ms=6.5, lw=0.9, ls=":",
             color=OKABE_ITO["blue"], zorder=4, label=r"mesh $\alpha$ nodes")
    ax1.axhline(amax, color=OKABE_ITO["vermillion"], lw=1.5, ls="--",
                label=rf"`metagga._ALPHA_MAX` $= {amax:.0f}$")
    ax1.set_yscale("symlog", linthresh=0.05, linscale=0.5)
    ax1.set_xlim(0.3, 10.7)
    ax1.set_ylim(-0.015, 400.0)
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"{v:g}" for v in MESH_ALPHA], fontsize=7.5,
                        rotation=45)
    ax1.set_xlabel(r"the ten $\alpha$ nodes, in order")
    ax1.set_ylabel(r"iso-orbital indicator $\alpha$ (dimensionless)")
    ax1.set_title(r"(b) $\alpha$ coverage against the stored ceiling",
                  loc="left")
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    ax1.text(5.5, math.sqrt(max(MESH_ALPHA) * amax),
             "unsampled: "
             rf"{max(MESH_ALPHA):.0f} $< \alpha \leq$ {amax:.0f}"
             rf" ({amax / max(MESH_ALPHA):.0f}x)",
             ha="center", va="center", fontsize=8.5)
    rows += _series_rows("b_alpha_axis", "mesh_alpha_node", "alpha", nodes,
                         "marker", np.zeros_like(nodes))
    rows += _series_rows("b_alpha_axis", "alpha_max", "alpha", [amax],
                         "marker", [0.0])

    fig.suptitle("SCAN pretraining mesh", fontsize=12.5, y=1.02)
    path = _save(fig, outdir, stem, dpi)
    _write_csv(Path(outdir) / f"{stem}.csv", rows, extra_fields=("r_s",))
    return path


# --------------------------------------------------------------------------- #
# Figure 8 -- the C2 DIIS trajectory
# --------------------------------------------------------------------------- #

_TRAJ_RE = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+e[+-]\d+)")
_SOSCF_RE = re.compile(r"SOSCF conv=(\w+) macros=(\d+) E=(-?\d+\.\d+)")
_STAB_RE = re.compile(r"\[E=(-?\d+\.\d+)[^]]*\] stability: internal=(\w+)")
_BELOW_RE = re.compile(r"cycles below the basin midpoint\s+(-?\d+\.\d+):\s+"
                       r"(\d+) of (\d+)")


def parse_c2_log(path):
    """Per-cycle trajectory, converged solutions and their internal stability."""
    text = Path(path).read_text()
    cycles, energies, grads = [], [], []
    in_traj = False
    for line in text.splitlines():
        if line.startswith("per-cycle DIIS trajectory"):
            in_traj = True
            continue
        if not in_traj:
            continue
        if not line.strip():
            if cycles:
                break
            continue
        m = _TRAJ_RE.match(line)
        if m is None:
            if cycles:
                break
            continue
        cycles.append(int(m.group(1)))
        energies.append(float(m.group(2)))
        grads.append(float(m.group(3)))
    if not cycles:
        raise ValueError(f"no per-cycle DIIS trajectory found in {path}")
    if cycles != list(range(len(cycles))):
        raise ValueError(f"trajectory cycles are not consecutive in {path}")

    solutions = {}
    for conv, macros, energy in _SOSCF_RE.findall(text):
        if conv != "True":
            continue
        # Keyed on the printed decimal string so no precision is lost; two
        # solutions printed identically are one solution.
        solutions[float(energy)] = int(macros)
    if len(solutions) != 2:
        raise ValueError(
            f"expected two converged SOSCF solutions in {path}, "
            f"found {sorted(solutions)}")
    stability = {}
    for energy, internal in _STAB_RE.findall(text):
        stability[round(float(energy), 6)] = internal
    labelled = {}
    for energy in solutions:
        key = round(energy, 6)
        if key not in stability:
            raise ValueError(f"no stability record for E={energy} in {path}")
        labelled[energy] = stability[key]

    below = _BELOW_RE.search(text)
    if below is None:
        raise ValueError(f"no basin-midpoint summary in {path}")
    stated_below = int(below.group(2))

    return {
        "cycle": np.asarray(cycles, dtype=int),
        "energy": np.asarray(energies, dtype=float),
        "grad": np.asarray(grads, dtype=float),
        "solutions": labelled,
        "stated_cycles_below_midpoint": stated_below,
    }


def make_c2_diis_trajectory(outdir, dpi, log_path=DEFAULT_C2_LOG):
    stem = "c2_diis_trajectory"
    data = parse_c2_log(log_path)
    cyc, e, g = data["cycle"], data["energy"], data["grad"]
    solutions = data["solutions"]
    e_low = min(solutions)
    e_high = max(solutions)
    midpoint = 0.5 * (e_low + e_high)
    n_below = int((e < midpoint).sum())
    if n_below != data["stated_cycles_below_midpoint"]:
        raise AssertionError(
            f"basin occupancy disagrees with the log: computed {n_below}, "
            f"log states {data['stated_cycles_below_midpoint']}")
    gap_kcal = (e_high - e_low) * HARTREE_PER_KCAL
    i_e = int(np.argmin(e))
    i_g = int(np.argmin(g))

    rows = _series_rows("a_trajectory", "energy", "cycle", cyc, "E_hartree", e,
                        grad_norm=g)
    rows += _series_rows("b_solutions", "converged_soscf",
                         "E_hartree", [e_low, e_high], "index", [0, 1])
    rows += _series_rows("c_markers", "lowest_energy_cycle", "cycle",
                         [cyc[i_e]], "E_hartree", [e[i_e]],
                         grad_norm=[g[i_e]])
    rows += _series_rows("c_markers", "lowest_gradient_cycle", "cycle",
                         [cyc[i_g]], "E_hartree", [e[i_g]],
                         grad_norm=[g[i_g]])
    rows += _series_rows("c_markers", "basin_midpoint", "cycle", [-1],
                         "E_hartree", [midpoint])

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax2 = ax.twinx()
    h_grad, = ax2.semilogy(cyc, g, color=OKABE_ITO["orange"], lw=0.9,
                           alpha=0.75, zorder=1,
                           label=r"$|g|$ (right axis)")
    ax2.set_ylabel(r"orbital gradient norm $|g|$ (a.u.)")
    ax2.set_ylim(1e-3, 3.0)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    h_energy, = ax.plot(cyc, e, color=OKABE_ITO["blue"], lw=1.3, marker="o",
                        ms=2.6, zorder=3, label=r"$E$ (left axis)")
    h_low = ax.axhline(e_low, color=OKABE_ITO["green"], lw=1.2, ls="--",
                       zorder=2,
                       label=f"SOSCF {e_low:.10f} Ha (internally "
                             f"{solutions[e_low].lower()})")
    h_high = ax.axhline(e_high, color=OKABE_ITO["vermillion"], lw=1.2,
                        ls="--", zorder=2,
                        label=f"SOSCF {e_high:.10f} Ha (internally "
                              f"{solutions[e_high].lower()})")
    h_mid = ax.axhline(midpoint, color=OKABE_ITO["black"], lw=0.9, ls=":",
                       zorder=2,
                       label=f"basin midpoint {midpoint:.6f} Ha")
    h_mine, = ax.plot([cyc[i_e]], [e[i_e]], marker="v", ms=9, lw=0,
                      color=OKABE_ITO["black"], zorder=6,
                      label=f"lowest $E$: cycle {cyc[i_e]}, "
                            f"{e[i_e]:.10f} Ha")
    h_ming, = ax.plot([cyc[i_g]], [e[i_g]], marker="^", ms=9, lw=0,
                      mfc="none", color=OKABE_ITO["purple"], zorder=6,
                      label=f"lowest $|g|$: cycle {cyc[i_g]}, "
                            f"{g[i_g]:.3e} a.u.")

    ax.set_xlabel("DIIS cycle")
    ax.set_ylabel(r"total energy $E$ (Ha)")
    ax.set_xlim(-2, 101)
    ax.set_ylim(-75.828, -75.640)
    ax.set_title("C2 (RKS/PBE, 6-311++G(3df,2pd), grid 3): "
                 "an oscillating DIIS trajectory between two solutions",
                 pad=24)
    ax.text(0.5, 1.015,
            f"trajectory spread {e.max() - e.min():.4e} Ha; "
            f"{n_below} of {cyc.size} cycles in the lower basin; "
            f"inter-branch gap {e_high - e_low:.4e} Ha = "
            f"{gap_kcal:.2f} kcal/mol",
            transform=ax.transAxes, fontsize=8.5, ha="center")

    ax.legend(handles=[h_energy, h_grad, h_low, h_high, h_mid, h_mine,
                       h_ming],
              loc="upper left", ncol=2, fontsize=7.6,
              handlelength=1.8, columnspacing=1.2, borderaxespad=0.2)

    path = _save(fig, outdir, stem, dpi)
    _write_csv(Path(outdir) / f"{stem}.csv", rows,
               extra_fields=("grad_norm",))
    return path


# --------------------------------------------------------------------------- #
# Figure 9 -- the iso-orbital indicator
# --------------------------------------------------------------------------- #

def make_alpha_indicator(outdir, dpi):
    stem = "alpha_indicator"
    w = metagga._ALPHA_SMOOTHING_WIDTH
    amax = metagga._ALPHA_MAX
    rows = []

    rho = 1.0
    s_fixed = 1.0
    sigma = float(sigma_from_s(rho, s_fixed))
    t_w = sigma / (8.0 * rho)
    t_unif = float(tau_unif(rho))
    ratio_w = t_w / t_unif

    def alpha_of(raw):
        tau = t_w + np.asarray(raw, dtype=float) * t_unif
        return np.asarray(metagga.compute_alpha(
            jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(tau)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    raw = np.linspace(0.0, 20.0, 801)
    a = alpha_of(raw)
    ax0.plot(raw + ratio_w, a, color=OKABE_ITO["blue"],
             label=r"`metagga.compute_alpha`")
    rows += _series_rows("a_linear", "compute_alpha", "tau_over_tau_unif",
                         raw + ratio_w, "alpha", a, raw_indicator=raw)

    ax0.axvline(ratio_w, color=OKABE_ITO["vermillion"], lw=1.0, ls="--")
    ax0.axvline(ratio_w + 1.0, color=OKABE_ITO["green"], lw=1.0, ls="--")
    ax0.annotate(r"$\tau = \tau_W$: $\alpha \to 0$"
                 "\n" rf"(floor $w/2 = {w / 2:.0e}$)",
                 xy=(ratio_w, 0.0), xytext=(3.2, 2.2), fontsize=8,
                 color=OKABE_ITO["vermillion"],
                 arrowprops=dict(arrowstyle="->", lw=0.8,
                                 color=OKABE_ITO["vermillion"]))
    ax0.annotate(r"$\tau = \tau_W + \tau_{\mathrm{unif}}$: $\alpha = 1$"
                 "\n" "(uniform electron gas)",
                 xy=(ratio_w + 1.0, 1.0), xytext=(2.6, 11.6), fontsize=8,
                 color=OKABE_ITO["green"],
                 arrowprops=dict(arrowstyle="->", lw=0.8,
                                 color=OKABE_ITO["green"]))
    ax0.set_xlabel(r"$\tau / \tau_{\mathrm{unif}}$ (dimensionless)")
    ax0.set_ylabel(r"stored indicator $\alpha$ (dimensionless)")
    ax0.set_title(rf"(a) $\rho = 1$ bohr$^{{-3}}$, $s = {s_fixed:.0f}$"
                  rf" ($\tau_W/\tau_{{\mathrm{{unif}}}} = {ratio_w:.4f}$)",
                  loc="left")
    ax0.set_xlim(ratio_w - 1.0, ratio_w + 21.0)
    ax0.set_ylim(-1.0, 21.0)
    ax0.legend(loc="upper left")

    raw_in = np.linspace(-5e-5, 5e-5, 601)
    a_in = alpha_of(raw_in)
    inset = ax0.inset_axes([0.55, 0.11, 0.40, 0.31])
    inset.semilogy(raw_in, a_in, color=OKABE_ITO["blue"], lw=1.2)
    inset.axhline(w / 2.0, color=OKABE_ITO["orange"], ls="--", lw=0.9)
    inset.set_xlabel(r"$(\tau - \tau_W)/\tau_{\mathrm{unif}}$", fontsize=7,
                     labelpad=1)
    inset.set_ylabel(r"$\alpha$", fontsize=7.5, labelpad=1)
    inset.tick_params(labelsize=6.5)
    inset.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    inset.xaxis.get_offset_text().set_fontsize(6)
    inset.set_title("smooth floor", fontsize=7.5, pad=2)
    rows += _series_rows("a_inset", "compute_alpha_near_zero",
                         "raw_indicator", raw_in, "alpha", a_in)

    raw_hi = np.logspace(-1.0, 3.0, 601)
    a_hi = alpha_of(raw_hi)
    ax1.loglog(raw_hi + ratio_w, a_hi, color=OKABE_ITO["blue"],
               label=r"`metagga.compute_alpha`")
    ax1.loglog(raw_hi + ratio_w, raw_hi, color=OKABE_ITO["black"], ls=":",
               lw=1.0, label=r"raw $(\tau - \tau_W)/\tau_{\mathrm{unif}}$")
    rows += _series_rows("b_ceiling", "compute_alpha", "tau_over_tau_unif",
                         raw_hi + ratio_w, "alpha", a_hi, raw_indicator=raw_hi)
    rows += _series_rows("b_ceiling", "raw_indicator", "tau_over_tau_unif",
                         raw_hi + ratio_w, "alpha", raw_hi,
                         raw_indicator=raw_hi)
    ax1.axhline(amax, color=OKABE_ITO["vermillion"], lw=1.1, ls="--")
    ax1.text(1.4e2, 22.0, rf"ceiling $\alpha = {amax:.0f}$" "\n"
             "(`metagga._ALPHA_MAX`)", fontsize=8.5, va="center",
             color=OKABE_ITO["vermillion"])
    ax1.set_xlabel(r"$\tau / \tau_{\mathrm{unif}}$ (dimensionless)")
    ax1.set_ylabel(r"stored indicator $\alpha$ (dimensionless)")
    ax1.set_title("(b) the ceiling on the low-density tail", loc="left")
    ax1.set_ylim(5e-2, 2e3)
    ax1.legend(loc="lower right")

    fig.suptitle("Iso-orbital indicator: smooth floor and hard ceiling",
                 fontsize=12.5, y=1.01)
    path = _save(fig, outdir, stem, dpi)
    _write_csv(Path(outdir) / f"{stem}.csv", rows,
               extra_fields=("raw_indicator",))
    return path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

FIGURES = (
    ("bounded_map", make_bounded_map),
    ("preimage_sensitivity", make_preimage_sensitivity),
    ("smooth_positive_part", make_smooth_positive_part),
    ("alpha_ceiling", make_alpha_ceiling),
    ("parent_enhancement", make_parent_enhancement),
    ("zeta_pole", make_zeta_pole),
    ("dfs_mesh", make_dfs_mesh),
    ("c2_diis_trajectory", make_c2_diis_trajectory),
    ("alpha_indicator", make_alpha_indicator),
)


def main(outdir=DEFAULT_OUTDIR, dpi=200, c2_log=DEFAULT_C2_LOG):
    """Write every figure and its CSV into ``outdir``; returns the PNG paths."""
    _style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, fn in FIGURES:
        if stem == "c2_diis_trajectory":
            path = fn(outdir, dpi, log_path=c2_log)
        else:
            path = fn(outdir, dpi)
        print(f"wrote {path}")
        written.append(path)
    return written


def _cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help="directory for the PNGs and CSVs")
    ap.add_argument("--dpi", type=int, default=200,
                    help="raster resolution (>= 150)")
    ap.add_argument("--c2-log", default=str(DEFAULT_C2_LOG),
                    help="DIIS trajectory log for the C2 bistability figure")
    args = ap.parse_args()
    if args.dpi < 150:
        ap.error("--dpi must be at least 150")
    main(outdir=args.outdir, dpi=args.dpi, c2_log=args.c2_log)


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _cli()
