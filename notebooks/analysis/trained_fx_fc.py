#!/usr/bin/env python
"""Trained-network enhancement factors against the parent baseline.

The final-result counterpart of ``pretrain_fx_fc.py``: where that module draws
the network as it leaves the pretraining stage (the parent plus whatever the
fit added), this one draws the network the TRAINING stage produced, cell by
cell, so the learned correction can be read against the same parent curves and
against the training-set size that produced it.

The parent is resolved PER ARCHITECTURE from its own rung
(``parents.parent_for_arch``): PBE for the GGA rungs, SCAN for the meta-GGA
ones. A run mixing rungs draws each arch against ITS parent, and every figure
states which parent each panel uses.

Slices, by parent (the conventions of ``enhancement_factors.py`` and
``pretrain_fx_fc.py``, whose curve helpers are reused verbatim):
  * PBE archs:  ``F_x(s)`` at rho = 1, zero extra descriptors;
    ``F_c(s; r_s)`` at zeta = 0; the figures draw r_s = 2, the CSV carries
    every r_s of ``pretrain_fx_fc.RS_VALUES``.
  * SCAN archs: the same curves at the fixed iso-orbital slices
    ``pretrain_fx_fc.ALPHA_VALUES`` (alpha = 0 and alpha = 1, the SCAN
    Fig. 1 convention), one panel column per slice; alpha is the EXACT raw
    indicator, carried into the network as the stored-column encoding the
    shared curve helpers apply.

The baselines are the anchor's OWN parent functions (``parents.pbe_fx`` /
``parents.pbe_fc`` and ``parents.scan_fx`` / ``parents.scan_fc``, libxc
constants), imported from ``pretrain_fx_fc`` so the pretrained and trained
figures are drawn against one baseline: with the pre-image anchor the model
IS that parent plus the learned correction, and any other implementation of
the parent reads as a spurious correction (the rounded-constant analytic PBE
helper differs by 4.553e-6 in F_x on this grid).

Loading. The run's own ``resolved_config.yaml`` supplies the model class (the
parent anchor, the descriptor coordinates, the polarized correlation network),
the skeleton is built from it as the training stage built it, and the leaves
are read through ``checkpoint_class.load_trained_checkpoint`` -- the canonical
loader, which holds the class record beside the checkpoint to the ``.eqx`` on
disk by its SHA-256 and then to the class of the skeleton. Neither the anchor
nor the coordinates changes a parameter shape, so a checkpoint of another
class would otherwise deserialize here in silence and plot as a plausible
curve -- at zero-initialized final layers one that agrees with the parent to
round-off (2.2e-16), and with perturbed legacy leaves one that sits 8.3e-3
off, an O(1 percent) silently wrong functional. A refusal is raised, not swallowed: a figure drawn from a checkpoint
nothing on disk describes would be worse than no figure.

``enhancement_factors.load_trained_model`` is the sibling reader of the same
checkpoints; it is not reused here because it reads the pickled
``specs/spec_NNNN.spec`` (which the default pull profile does not carry) and
names ``model.eqx`` literally, while these figures select a channel.

Outputs, into ``--outdir``:
  * ``trained_fx_fc_<arch>.png``     per-arch panels, one curve per completed
                                     subset-size cell (light -> dark) with
                                     difference panels (2x2 for a PBE arch,
                                     2x4 with one column per alpha slice for
                                     a SCAN arch)
  * ``trained_fx_fc_delta_best.png`` cross-arch differences, each arch at its
                                     best held-out cell against its own parent
  * ``trained_fx_fc_curves.csv``     long-form curves (arch, subset_size,
                                     channel, rs, s, f_model, f_parent,
                                     eval_channel; an alpha column is added
                                     when a SCAN arch is drawn)

The ``eval_channel`` column states the channel each row's weights actually came
from, so a row reading ``final`` under ``--eval-channel val_best`` is a cell
that had no ``model_val_best.eqx``.

Usage:
  python notebooks/analysis/trained_fx_fc.py \\
      --run-dir <pulled run dir> --outdir <figure dir> \\
      [--archs a,b,...] [--eval-channel val_best|final]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_style import ARCH_COLOR, arch_color  # noqa: E402
from enhancement_factors import model_fc_curve, model_fx_curve  # noqa: E402
from pretrain_fx_fc import (  # noqa: E402
    ALPHA_LINESTYLE,
    ALPHA_VALUES,
    RS_VALUES,
    S_GRID,
    model_fc_curve_mgga,
    model_fx_curve_mgga,
    parent_fc_curve,
    parent_fc_curve_scan,
    parent_fx_curve,
    parent_fx_curve_scan,
)

#: Checkpoint filename per evaluation channel. ``val_best`` is the held-out
#: VALIDATION-best snapshot (the checkpoint the figure suite plots); ``final``
#: is the last training step. ``model_best.eqx`` (minimum TRAINING loss) is
#: deliberately absent: it selects the most-overfit step on the overfit-prone
#: architectures and no figure reads it.
CHANNEL_FILENAMES = {"val_best": "model_val_best.eqx", "final": "model.eqx"}
#: The channel a missing checkpoint falls back to. Only the val-best channel
#: has one: ``model.eqx`` is written by every completed train task, while
#: ``model_val_best.eqx`` exists only where the run validated.
CHANNEL_FALLBACK = {"val_best": "final"}
#: The held-out evaluation directory written from each channel's weights, read
#: for the best-cell selection so the ranking is scored on the SAME weights
#: the curves are drawn from.
CHANNEL_EVAL_DIR = {"val_best": "eval_holdout_val_best", "final": "eval_holdout"}
#: The held-out row and column the best cell is selected on.
BEST_CELL_SET = "test_set_held_out_combined"
BEST_CELL_COLUMN = "mae_nn_kcalmol"
#: The r_s the correlation panels are drawn at (r_s = 2 is the valence-density
#: scale the reaction sets sample); the CSV carries every RS_VALUES entry.
RS_FIGURE = 2.0
#: Lightest tint of an arch's hue, as the fraction of the hue kept when the
#: colour is blended toward white. The lightest subset-size curve must stay
#: legible on white, which is what sets the floor.
SHADE_LIGHTEST = 0.32
_PARENT_STYLE = dict(color="0.25", linestyle="--", linewidth=2.0, zorder=1)
#: The local root run dirs are pulled under, used only to reconstruct the
#: ``--category`` of the pull command named in the missing-checkpoint report.
_LOCAL_ROOT = Path("~/Documents/Research/xcquinox-results/runs").expanduser()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One completed sweep cell and the checkpoint the curves are drawn from."""
    index: int
    arch: str
    subset_size: int
    path: Path
    channel: str
    fallback: bool


def read_manifest(run_dir: Path) -> Tuple[int, Dict[int, dict]]:
    """``(width, {spec_index: cell})`` from ``manifest.json``.

    Raises ``FileNotFoundError`` when the manifest is absent: without it the
    spec index carries no architecture and no subset size, and a figure keyed
    on a directory name alone would be unlabelled.
    """
    path = run_dir / "manifest.json"
    with open(path) as fh:
        manifest = json.load(fh)
    width = int(manifest.get("width", 4))
    cells: Dict[int, dict] = {}
    for entry in manifest.get("specs", []):
        idx = entry.get("index")
        if isinstance(idx, int):
            cells[idx] = dict(entry.get("cell") or {})
    return width, cells


def spec_dir(run_dir: Path, index: int, width: int) -> Path:
    return run_dir / "checkpoints" / f"spec_{index:0{width}d}"


def discover_cells(run_dir: Path, eval_channel: str,
                   archs: Optional[Sequence[str]] = None
                   ) -> Tuple[List[Cell], List[Tuple[int, str, int]]]:
    """``(cells, missing)`` for ``run_dir`` on ``eval_channel``.

    ``cells`` are the manifest entries whose checkpoint is on disk (the
    requested channel, else its fallback); ``missing`` are the
    ``(index, arch, subset_size)`` of the entries with neither, which is what
    a run pulled without the weights looks like.
    """
    width, manifest_cells = read_manifest(run_dir)
    wanted = set(archs) if archs else None
    found: List[Cell] = []
    missing: List[Tuple[int, str, int]] = []
    for index in sorted(manifest_cells):
        cell = manifest_cells[index]
        arch, subset_size = cell.get("arch"), cell.get("subset_size")
        if arch is None or subset_size is None:
            continue
        if wanted is not None and arch not in wanted:
            continue
        directory = spec_dir(run_dir, index, width)
        chain = [eval_channel]
        fallback = CHANNEL_FALLBACK.get(eval_channel)
        if fallback is not None:
            chain.append(fallback)
        for channel in chain:
            path = directory / CHANNEL_FILENAMES[channel]
            if path.is_file():
                found.append(Cell(index=index, arch=arch,
                                  subset_size=int(subset_size), path=path,
                                  channel=channel,
                                  fallback=channel != eval_channel))
                break
        else:
            missing.append((index, arch, int(subset_size)))
    return found, missing


def held_out_mae(run_dir: Path, index: int, width: int,
                 eval_channel: str) -> Optional[float]:
    """The combined held-out MAE (kcal/mol) of one cell on ``eval_channel``,
    or ``None`` when that evaluation is not on disk."""
    path = (spec_dir(run_dir, index, width)
            / CHANNEL_EVAL_DIR[eval_channel] / "test_set.csv")
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    for row in rows:
        if row.get("set") == BEST_CELL_SET:
            try:
                value = float(row[BEST_CELL_COLUMN])
            except (KeyError, TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None
    return None


def missing_checkpoints_message(run_dir: Path, eval_channel: str,
                                missing: Sequence[Tuple[int, str, int]],
                                width: int) -> str:
    """What to say when no cell of a run has its trained weights on disk.

    The run has its evaluation tables (the pull carried them) and no ``.eqx``,
    which is what a run pulled before the default profile carried the weights
    looks like. The report names the files that are absent and the pull that
    fetches them rather than reporting an empty arch set.
    """
    names = [CHANNEL_FILENAMES[eval_channel]]
    fallback = CHANNEL_FALLBACK.get(eval_channel)
    if fallback is not None:
        names.append(CHANNEL_FILENAMES[fallback])
    examples = [
        str(spec_dir(run_dir, index, width).relative_to(run_dir) / names[0])
        for index, _arch, _ss in missing[:3]
    ]
    try:
        category = run_dir.expanduser().resolve().parent.relative_to(_LOCAL_ROOT)
        category_arg = f" --category {category}"
    except ValueError:
        category_arg = " --category <category under the local root>"
    return (
        f"no trained checkpoint is on disk for any of the {len(missing)} "
        f"cells in the manifest of {run_dir.name}. The {eval_channel} channel "
        "reads "
        + names[0]
        + (f" (falling back to {names[1]})" if len(names) > 1 else "")
        + f", and none of {', '.join(examples)}"
        + (", ..." if len(missing) > 3 else "")
        + " exists: the evaluation tables of this run were pulled without the "
          "weights themselves. Fetch them with\n"
          f"  python -m xcquinox.alec.cluster pull {run_dir.name}"
          f"{category_arg} --profile summaries\n"
          "(the summaries profile carries model.eqx / model_val_best.eqx and "
          "their .class.json records), then re-run this script.")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def arch_from_config(cfg, arch_name: str):
    """The architecture ``arch_name`` resolves to under a run's config.

    The registry entry patched with the run-level polarized-correlation
    override and the run's ``model:`` block (the parent anchor, the descriptor
    coordinates), exactly as ``cluster.fidelity.build_certified_model`` and the
    training stage resolve it -- so the skeleton built here is the class the
    checkpoint was trained as.
    """
    import dataclasses

    from xcquinox.alec.config import apply_model_block, get_architecture
    arch = get_architecture(arch_name)
    if getattr(cfg, "use_polarized_correlation", False):
        arch = dataclasses.replace(arch, use_polarized_correlation=True)
    model_block = getattr(cfg, "model", None)
    if model_block is not None:
        arch = apply_model_block(arch, model_block)
    return arch


def load_run_config(run_dir: Path):
    """The run's resolved configuration."""
    from xcquinox.alec.cluster.grid_config import load_grid_config
    return load_grid_config(str(run_dir / "resolved_config.yaml"))


def load_trained_model(cfg, arch_name: str, checkpoint_path: Path):
    """``(arch, model)`` for one trained checkpoint, class record honoured.

    The skeleton is built from the run's own configuration and filled by
    ``checkpoint_class.load_trained_checkpoint``, which refuses a checkpoint
    the record beside it does not describe (``ClassRecordStale``) and one
    written as another model class (``ModelClassMismatch``). Both refusals
    propagate. Every rung loads here; the parent the curves are drawn
    against is resolved per arch downstream (``parents.parent_for_arch``).
    """
    from xcquinox.alec.checkpoint_class import load_trained_checkpoint
    from xcquinox.alec.models import AlecGGAModel
    arch = arch_from_config(cfg, arch_name)
    skeleton = AlecGGAModel.from_arch(arch, seed=0)
    model = load_trained_checkpoint(checkpoint_path, skeleton,
                                    what="trained checkpoint")
    return arch, model


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def parent_curves() -> dict:
    """The PBE baselines on the plotted grid, computed once per run."""
    return {
        "fx": parent_fx_curve(S_GRID),
        "fc": {rs: parent_fc_curve(S_GRID, rs) for rs in RS_VALUES},
    }


def parent_curves_scan() -> dict:
    """The SCAN baselines on the plotted grid at each alpha slice, computed
    once per run (the meta-GGA counterpart of :func:`parent_curves`)."""
    return {
        "fx_alpha": {alpha: parent_fx_curve_scan(S_GRID, alpha)
                     for alpha in ALPHA_VALUES},
        "fc_alpha": {alpha: {rs: parent_fc_curve_scan(S_GRID, rs, alpha)
                             for rs in RS_VALUES}
                     for alpha in ALPHA_VALUES},
    }


def compute_curves(model, parents: dict) -> dict:
    """All plotted curves for one trained PBE-parent model, against cached
    parents.

    Same shape as ``pretrain_fx_fc.compute_curves``; the parent curves are
    passed in rather than recomputed because a size sweep evaluates dozens of
    cells against one baseline.
    """
    curves = {
        "fx_model": model_fx_curve(model, S_GRID),
        "fx_parent": parents["fx"],
        "fc": {},
    }
    for rs in RS_VALUES:
        curves["fc"][rs] = {
            "model": model_fc_curve(model, S_GRID, rs),
            "parent": parents["fc"][rs],
        }
    return curves


def compute_curves_scan(model, parents_scan: dict) -> dict:
    """All plotted curves for one trained SCAN-parent (meta-GGA) model,
    against cached parents; same shape as
    ``pretrain_fx_fc.compute_curves_scan``. The model curves carry the
    stored-column alpha encoding through the shared helpers."""
    curves = {"fx_alpha": {}, "fc_alpha": {}}
    for alpha in ALPHA_VALUES:
        curves["fx_alpha"][alpha] = {
            "model": model_fx_curve_mgga(model, S_GRID, alpha),
            "parent": parents_scan["fx_alpha"][alpha],
        }
        curves["fc_alpha"][alpha] = {}
        for rs in RS_VALUES:
            curves["fc_alpha"][alpha][rs] = {
                "model": model_fc_curve_mgga(model, S_GRID, rs, alpha),
                "parent": parents_scan["fc_alpha"][alpha][rs],
            }
    return curves


def max_abs_dfx(curves: dict) -> float:
    """max|F_x^NN - F_x^parent| over every drawn F_x slice."""
    if "fx_alpha" in curves:
        return max(
            float(np.max(np.abs(pair["model"] - pair["parent"])))
            for pair in curves["fx_alpha"].values())
    return float(np.max(np.abs(curves["fx_model"] - curves["fx_parent"])))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def subset_shades(color: str, n: int) -> List[str]:
    """``n`` shades of ``color``, light -> dark, for ``n`` subset sizes.

    A sequential ramp of ONE hue: subset size is a magnitude, so the family
    must read as an ordered sweep rather than as n unrelated architectures.
    The hue is blended toward white, which keeps the arch's palette identity
    at every step.
    """
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    if n <= 1:
        return [mcolors.to_hex(rgb)]
    return [mcolors.to_hex(tuple(1.0 - t * (1.0 - rgb)))
            for t in np.linspace(SHADE_LIGHTEST, 1.0, n)]


def _cell_label(cell: Cell) -> str:
    return (f"{cell.subset_size} mol"
            + ("" if cell.subset_size == 1 else "s")
            + (" [final]" if cell.fallback else ""))


def render_arch_figure(arch_name: str, cells: Sequence[Cell],
                       curves_by_index: Dict[int, dict], outdir: Path,
                       footer: str) -> Path:
    """Per-arch panels against the arch's parent: the 2x2 PBE layout, or the
    2x4 SCAN layout (one column per alpha slice) for a meta-GGA arch.

    One curve per completed subset-size cell, ascending, light -> dark.
    """
    ordered = sorted(cells, key=lambda c: (c.subset_size, c.index))
    if "fx_alpha" in curves_by_index[ordered[0].index]:
        return _render_arch_figure_scan(arch_name, ordered, curves_by_index,
                                        outdir, footer)
    shades = subset_shades(ARCH_COLOR.get(arch_name, arch_color(arch_name)),
                           len(ordered))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    (ax_fx, ax_dfx), (ax_fc, ax_dfc) = axes

    first = curves_by_index[ordered[0].index]
    ax_fx.plot(S_GRID, first["fx_parent"], label="PBE parent", **_PARENT_STYLE)
    ax_fc.plot(S_GRID, first["fc"][RS_FIGURE]["parent"],
               label=rf"PBE parent, $r_s={RS_FIGURE:g}$", **_PARENT_STYLE)
    for shade, cell in zip(shades, ordered):
        curves = curves_by_index[cell.index]
        label = (f"{cell.subset_size} mol"
                 + ("" if cell.subset_size == 1 else "s")
                 + (" [final]" if cell.fallback else ""))
        ax_fx.plot(S_GRID, curves["fx_model"], color=shade, linewidth=2.0,
                   label=label, zorder=2)
        ax_dfx.plot(S_GRID, curves["fx_model"] - curves["fx_parent"],
                    color=shade, linewidth=2.0, label=label)
        pair = curves["fc"][RS_FIGURE]
        ax_fc.plot(S_GRID, pair["model"], color=shade, linewidth=2.0, zorder=2,
                   label=label)
        ax_dfc.plot(S_GRID, pair["model"] - pair["parent"], color=shade,
                    linewidth=2.0, label=label)

    ax_fx.set_ylabel(r"$F_x(s)$")
    ax_dfx.set_ylabel(r"$F_x^{\mathrm{NN}} - F_x^{\mathrm{PBE}}$")
    ax_fc.set_ylabel(rf"$F_c(s;\,r_s={RS_FIGURE:g})$  ($\zeta=0$)")
    ax_dfc.set_ylabel(r"$F_c^{\mathrm{NN}} - F_c^{\mathrm{PBE}}$")
    for ax in (ax_dfx, ax_dfc):
        ax.axhline(0.0, color="0.7", linewidth=1.0)
    for ax in axes.ravel():
        ax.set_xlabel(r"reduced gradient $s$")
        ax.grid(True, color="0.92", linewidth=0.8)
        ax.set_axisbelow(True)
        # A dozen curves per panel leave the legend sitting over some of them,
        # so it gets a soft opaque frame rather than the sibling module's bare
        # text (which is legible there because each panel holds few curves).
        ax.legend(fontsize=7, ncol=2, title="training subset", frameon=True,
                  framealpha=0.85, edgecolor="0.85").get_title().set_fontsize(7)
    fig.suptitle(f"{arch_name}: trained networks against the PBE parent",
                 fontsize=12)
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.97))
    out = outdir / f"trained_fx_fc_{arch_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _render_arch_figure_scan(arch_name: str, ordered: Sequence[Cell],
                             curves_by_index: Dict[int, dict], outdir: Path,
                             footer: str) -> Path:
    """Per-arch 2x4 against the SCAN parent: one column per (channel, alpha)
    -- [F_x a=0 | F_x a=1 | F_c a=0 | F_c a=1], overlays on the top row and
    differences below, so every panel is a single-alpha family of cell curves
    exactly like the PBE panels. ``ordered`` is the cell list already sorted
    by (subset_size, index)."""
    shades = subset_shades(ARCH_COLOR.get(arch_name, arch_color(arch_name)),
                           len(ordered))
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 7.6))
    top, bottom = axes
    first = curves_by_index[ordered[0].index]

    for j, alpha in enumerate(ALPHA_VALUES):
        ax, axd = top[j], bottom[j]
        ax.plot(S_GRID, first["fx_alpha"][alpha]["parent"],
                label=rf"SCAN parent, $\alpha={alpha:g}$", **_PARENT_STYLE)
        for shade, cell in zip(shades, ordered):
            pair = curves_by_index[cell.index]["fx_alpha"][alpha]
            ax.plot(S_GRID, pair["model"], color=shade, linewidth=2.0,
                    label=_cell_label(cell), zorder=2)
            axd.plot(S_GRID, pair["model"] - pair["parent"], color=shade,
                     linewidth=2.0, label=_cell_label(cell))
        ax.set_ylabel(rf"$F_x(s)$  ($\alpha={alpha:g}$)")
        axd.set_ylabel(rf"$F_x^{{\mathrm{{NN}}}} - F_x^{{\mathrm{{SCAN}}}}$"
                       rf"  ($\alpha={alpha:g}$)")

    for j, alpha in enumerate(ALPHA_VALUES):
        ax, axd = top[2 + j], bottom[2 + j]
        ax.plot(S_GRID, first["fc_alpha"][alpha][RS_FIGURE]["parent"],
                label=rf"SCAN parent, $r_s={RS_FIGURE:g}$, "
                      rf"$\alpha={alpha:g}$",
                **_PARENT_STYLE)
        for shade, cell in zip(shades, ordered):
            pair = curves_by_index[cell.index]["fc_alpha"][alpha][RS_FIGURE]
            ax.plot(S_GRID, pair["model"], color=shade, linewidth=2.0,
                    label=_cell_label(cell), zorder=2)
            axd.plot(S_GRID, pair["model"] - pair["parent"], color=shade,
                     linewidth=2.0, label=_cell_label(cell))
        ax.set_ylabel(rf"$F_c(s;\,r_s={RS_FIGURE:g})$"
                      rf"  ($\zeta=0,\ \alpha={alpha:g}$)")
        axd.set_ylabel(rf"$F_c^{{\mathrm{{NN}}}} - F_c^{{\mathrm{{SCAN}}}}$"
                       rf"  ($\alpha={alpha:g}$)")

    for ax in bottom:
        ax.axhline(0.0, color="0.7", linewidth=1.0)
    for ax in axes.ravel():
        ax.set_xlabel(r"reduced gradient $s$")
        ax.grid(True, color="0.92", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7, ncol=2, title="training subset", frameon=True,
                  framealpha=0.85, edgecolor="0.85").get_title().set_fontsize(7)
    fig.suptitle(f"{arch_name}: trained networks against the SCAN parent",
                 fontsize=12)
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.97))
    out = outdir / f"trained_fx_fc_{arch_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_best_figure(best: Sequence[Tuple[Cell, dict, Optional[float]]],
                       outdir: Path, footer: str) -> Path:
    """Cross-arch differences, each arch at its best held-out cell, drawn
    against its OWN parent (a SCAN arch contributes one curve per alpha
    slice, separated by linestyle, with the parent named per curve)."""
    if any("fx_alpha" in curves for _cell, curves, _mae in best):
        return _render_best_figure_with_scan(best, outdir, footer)
    fig, (ax_dfx, ax_dfc) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for cell, curves, mae in best:
        color = ARCH_COLOR.get(cell.arch, arch_color(cell.arch))
        label = (f"{cell.arch} ({cell.subset_size} mol"
                 + ("" if cell.subset_size == 1 else "s")
                 + (f", {mae:.2f} kcal/mol" if mae is not None else "")
                 + (", final" if cell.fallback else "") + ")")
        ax_dfx.plot(S_GRID, curves["fx_model"] - curves["fx_parent"],
                    color=color, linewidth=2.0, label=label)
        pair = curves["fc"][RS_FIGURE]
        ax_dfc.plot(S_GRID, pair["model"] - pair["parent"], color=color,
                    linewidth=2.0, label=label)
    for ax, ylabel in (
            (ax_dfx, r"$F_x^{\mathrm{NN}} - F_x^{\mathrm{PBE}}$"),
            (ax_dfc, rf"$F_c^{{\mathrm{{NN}}}} - F_c^{{\mathrm{{PBE}}}}$"
                     rf"  ($r_s={RS_FIGURE:g}$)")):
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.set_xlabel(r"reduced gradient $s$")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.92", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Trained corrections to the PBE parent, each architecture at "
                 "its best held-out cell", fontsize=12)
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    out = outdir / "trained_fx_fc_delta_best.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _render_best_figure_with_scan(
        best: Sequence[Tuple[Cell, dict, Optional[float]]],
        outdir: Path, footer: str) -> Path:
    """The best-cell difference panels when at least one SCAN arch is drawn:
    each SCAN arch appears once per alpha slice (linestyle-separated) against
    parents.scan_*, each PBE arch once against parents.pbe_*, with the parent
    named per curve in the legend."""
    has_pbe = any("fx_model" in curves for _cell, curves, _mae in best)
    parent_tag = r"\mathrm{parent}" if has_pbe else r"\mathrm{SCAN}"
    fig, (ax_dfx, ax_dfc) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for cell, curves, mae in best:
        color = ARCH_COLOR.get(cell.arch, arch_color(cell.arch))
        stem = (f"{cell.arch} ({cell.subset_size} mol"
                + ("" if cell.subset_size == 1 else "s")
                + (f", {mae:.2f} kcal/mol" if mae is not None else "")
                + (", final" if cell.fallback else ""))
        if "fx_alpha" in curves:
            for alpha in ALPHA_VALUES:
                label = stem + rf", vs SCAN, $\alpha={alpha:g}$)"
                pair = curves["fx_alpha"][alpha]
                ax_dfx.plot(S_GRID, pair["model"] - pair["parent"],
                            color=color, linestyle=ALPHA_LINESTYLE[alpha],
                            linewidth=2.0, label=label)
                pair = curves["fc_alpha"][alpha][RS_FIGURE]
                ax_dfc.plot(S_GRID, pair["model"] - pair["parent"],
                            color=color, linestyle=ALPHA_LINESTYLE[alpha],
                            linewidth=2.0, label=label)
        else:
            label = stem + ", vs PBE)"
            ax_dfx.plot(S_GRID, curves["fx_model"] - curves["fx_parent"],
                        color=color, linewidth=2.0, label=label)
            pair = curves["fc"][RS_FIGURE]
            ax_dfc.plot(S_GRID, pair["model"] - pair["parent"], color=color,
                        linewidth=2.0, label=label)
    for ax, ylabel in (
            (ax_dfx, rf"$F_x^{{\mathrm{{NN}}}} - F_x^{{{parent_tag}}}$"),
            (ax_dfc, rf"$F_c^{{\mathrm{{NN}}}} - F_c^{{{parent_tag}}}$"
                     rf"  ($r_s={RS_FIGURE:g}$)")):
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.set_xlabel(r"reduced gradient $s$")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.92", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Trained corrections to each architecture's own parent at its best "
        "held-out cell (PBE for GGA, SCAN for meta-GGA)" if has_pbe else
        "Trained corrections to the SCAN parent, each architecture at its "
        "best held-out cell", fontsize=12)
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    out = outdir / "trained_fx_fc_delta_best.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_curves_csv(cells: Sequence[Cell], curves_by_index: Dict[int, dict],
                     outdir: Path) -> Path:
    """Long-form curves; ``eval_channel`` is the channel each row was read
    from, so a ``final`` row under ``--eval-channel val_best`` is a fallback
    cell. When at least one SCAN arch is drawn the schema gains an ``alpha``
    column, which PBE-arch rows leave empty exactly as fx rows leave ``rs``
    empty."""
    out = outdir / "trained_fx_fc_curves.csv"
    ordered = sorted(cells, key=lambda c: (c.arch, c.subset_size, c.index))
    if any("fx_alpha" in curves_by_index[c.index] for c in ordered):
        return _write_curves_csv_with_alpha(ordered, curves_by_index, out)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "subset_size", "channel", "rs", "s", "f_model",
                    "f_parent", "eval_channel"])
        for cell in ordered:
            curves = curves_by_index[cell.index]
            for s, fm, fp in zip(S_GRID, curves["fx_model"],
                                 curves["fx_parent"]):
                w.writerow([cell.arch, cell.subset_size, "fx", "",
                            f"{s:.6f}", repr(float(fm)), repr(float(fp)),
                            cell.channel])
            for rs in RS_VALUES:
                pair = curves["fc"][rs]
                for s, fm, fp in zip(S_GRID, pair["model"], pair["parent"]):
                    w.writerow([cell.arch, cell.subset_size, "fc", f"{rs:g}",
                                f"{s:.6f}", repr(float(fm)), repr(float(fp)),
                                cell.channel])
    return out


def _write_curves_csv_with_alpha(ordered: Sequence[Cell],
                                 curves_by_index: Dict[int, dict],
                                 out: Path) -> Path:
    """The long-form CSV when at least one SCAN arch is drawn (``ordered`` is
    the cell list already sorted by (arch, subset_size, index))."""
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "subset_size", "channel", "rs", "alpha", "s",
                    "f_model", "f_parent", "eval_channel"])
        for cell in ordered:
            curves = curves_by_index[cell.index]
            if "fx_alpha" in curves:
                for alpha in ALPHA_VALUES:
                    pair = curves["fx_alpha"][alpha]
                    for s, fm, fp in zip(S_GRID, pair["model"],
                                         pair["parent"]):
                        w.writerow([cell.arch, cell.subset_size, "fx", "",
                                    f"{alpha:g}", f"{s:.6f}",
                                    repr(float(fm)), repr(float(fp)),
                                    cell.channel])
                for alpha in ALPHA_VALUES:
                    for rs in RS_VALUES:
                        pair = curves["fc_alpha"][alpha][rs]
                        for s, fm, fp in zip(S_GRID, pair["model"],
                                             pair["parent"]):
                            w.writerow([cell.arch, cell.subset_size, "fc",
                                        f"{rs:g}", f"{alpha:g}", f"{s:.6f}",
                                        repr(float(fm)), repr(float(fp)),
                                        cell.channel])
            else:
                for s, fm, fp in zip(S_GRID, curves["fx_model"],
                                     curves["fx_parent"]):
                    w.writerow([cell.arch, cell.subset_size, "fx", "", "",
                                f"{s:.6f}", repr(float(fm)), repr(float(fp)),
                                cell.channel])
                for rs in RS_VALUES:
                    pair = curves["fc"][rs]
                    for s, fm, fp in zip(S_GRID, pair["model"],
                                         pair["parent"]):
                        w.writerow([cell.arch, cell.subset_size, "fc",
                                    f"{rs:g}", "", f"{s:.6f}",
                                    repr(float(fm)), repr(float(fp)),
                                    cell.channel])
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _fallback_note(cells: Sequence[Cell], eval_channel: str) -> str:
    fell_back = sorted({f"{c.arch}/{c.subset_size}" for c in cells if c.fallback})
    if not fell_back:
        return ""
    return (f"  {len(fell_back)} cell(s) had no "
            f"{CHANNEL_FILENAMES[eval_channel]} and are drawn from "
            f"{CHANNEL_FILENAMES[CHANNEL_FALLBACK[eval_channel]]} "
            f"(labelled 'final'): {', '.join(fell_back)}.")


def best_cells(run_dir: Path, cells: Sequence[Cell], width: int
               ) -> Tuple[List[Tuple[Cell, Optional[float]]], List[str]]:
    """``([(cell, mae)], unranked_archs)``: one cell per architecture.

    The cell kept is the one with the smallest combined held-out MAE on the
    channel its own weights came from -- the ranking is scored on the weights
    the curve is drawn from, not on a sibling checkpoint's evaluation. An
    architecture with no held-out evaluation on disk is drawn at its largest
    completed subset size and named in the returned list, so the figure can
    say the cell was not selected on a measurement.
    """
    by_arch: Dict[str, List[Cell]] = {}
    for cell in cells:
        by_arch.setdefault(cell.arch, []).append(cell)
    out: List[Tuple[Cell, Optional[float]]] = []
    unranked: List[str] = []
    for arch_name in sorted(by_arch):
        scored = [(held_out_mae(run_dir, c.index, width, c.channel), c)
                  for c in by_arch[arch_name]]
        ranked = [(m, c) for m, c in scored if m is not None]
        if ranked:
            mae, cell = min(ranked, key=lambda t: (t[0], t[1].index))
            out.append((cell, mae))
        else:
            unranked.append(arch_name)
            out.append((max(by_arch[arch_name],
                            key=lambda c: (c.subset_size, c.index)), None))
    return out, unranked


def build_all(run_dir: Path, outdir: Path, *, eval_channel: str = "val_best",
              archs: Optional[Sequence[str]] = None) -> int:
    """Render every figure and the CSV for ``run_dir``. Returns an exit code."""
    try:
        width, manifest_cells = read_manifest(run_dir)
    except (OSError, ValueError) as exc:
        print(f"no readable manifest.json under {run_dir} ({exc}): the spec "
              "index alone carries neither the architecture nor the subset "
              "size, so there is nothing to label a curve with.")
        return 2
    cells, missing = discover_cells(run_dir, eval_channel, archs)
    if not cells and not missing:
        present = sorted({c.get("arch") for c in manifest_cells.values()
                          if c.get("arch")})
        print(f"--archs {','.join(archs or ())} matches no cell of "
              f"{run_dir.name}; its manifest holds {', '.join(present)}.")
        return 2
    if not cells:
        print(missing_checkpoints_message(run_dir, eval_channel, missing,
                                          width))
        return 2

    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(run_dir)
    from xcquinox.alec.parents import parent_for_arch
    parents = parent_curves()
    parents_scan: Optional[dict] = None  # built on the first SCAN arch
    parent_by_arch: Dict[str, str] = {}
    curves_by_index: Dict[int, dict] = {}
    for cell in sorted(cells, key=lambda c: (c.arch, c.subset_size)):
        arch, model = load_trained_model(cfg, cell.arch, cell.path)
        parent = parent_for_arch(arch)
        parent_by_arch[cell.arch] = parent
        if parent == "scan":
            if parents_scan is None:
                parents_scan = parent_curves_scan()
            curves_by_index[cell.index] = compute_curves_scan(model,
                                                              parents_scan)
        else:
            curves_by_index[cell.index] = compute_curves(model, parents)
        print(f"loaded spec {cell.index} {cell.arch} ss={cell.subset_size} "
              f"[{cell.channel}] max|dF_x| "
              f"{max_abs_dfx(curves_by_index[cell.index]):.3e}", flush=True)

    by_arch: Dict[str, List[Cell]] = {}
    for cell in cells:
        by_arch.setdefault(cell.arch, []).append(cell)

    channel_note = (f"checkpoint channel {eval_channel} "
                    f"({CHANNEL_FILENAMES[eval_channel]})")
    for arch_name in sorted(by_arch):
        arch_cells = by_arch[arch_name]
        worst = max(max_abs_dfx(curves_by_index[c.index]) for c in arch_cells)
        if parent_by_arch[arch_name] == "scan":
            footer = (f"run {run_dir.name}; {channel_note}; parent SCAN; "
                      "slices: F_x at rho=1 at the exact iso-orbital "
                      "indicator alpha in {0, 1}, zero other descriptors; "
                      f"F_c at zeta=0, r_s={RS_FIGURE:g}, same alpha slices; "
                      f"{len(arch_cells)} completed cell(s); max|dF_x| "
                      f"{worst:.2e}; parent curves parents.scan_fx / "
                      "parents.scan_fc (libxc constants)."
                      + _fallback_note(arch_cells, eval_channel))
        else:
            footer = (f"run {run_dir.name}; {channel_note}; slices: F_x at "
                      "rho=1, "
                      f"zero extra descriptors; F_c at zeta=0, r_s={RS_FIGURE:g}; "
                      f"{len(arch_cells)} completed cell(s); max|dF_x| "
                      f"{worst:.2e}; parent curves parents.pbe_fx / parents.pbe_fc "
                      "(libxc constants)." + _fallback_note(arch_cells,
                                                            eval_channel))
        out = render_arch_figure(arch_name, arch_cells, curves_by_index,
                                 outdir, footer)
        print(f"wrote {out} ({len(arch_cells)} cells, max|dF_x| {worst:.3e})")

    selected, unranked = best_cells(run_dir, cells, width)
    best = [(cell, curves_by_index[cell.index], mae)
            for cell, mae in selected]

    used_dirs = sorted({CHANNEL_EVAL_DIR[cell.channel]
                        for cell, mae in selected if mae is not None})
    score_src = "/".join(used_dirs) if used_dirs else \
        CHANNEL_EVAL_DIR[eval_channel]
    if any(p == "scan" for p in parent_by_arch.values()):
        parent_note = ("each arch drawn against its own parent: "
                       "parents.pbe_fx / parents.pbe_fc (GGA archs), "
                       "parents.scan_fx / parents.scan_fc at alpha in {0, 1} "
                       "(meta-GGA archs); libxc constants.")
    else:
        parent_note = ("parent curves parents.pbe_fx / parents.pbe_fc "
                       "(libxc constants).")
    footer = (f"run {run_dir.name}; {channel_note}; one cell per architecture, "
              f"selected by the smallest {BEST_CELL_COLUMN} of the "
              f"{BEST_CELL_SET} row of "
              f"checkpoints/spec_*/{score_src}/test_set.csv -- each cell "
              "scored on the channel its drawn weights came from; "
              + parent_note)
    if unranked:
        footer += (f"  No held-out evaluation on disk for {', '.join(unranked)}"
                   " -- drawn at the largest completed subset size instead.")
    if missing:
        footer += (f"  {len(missing)} manifest cell(s) have no checkpoint on "
                   "disk and are not drawn.")
    out = render_best_figure(best, outdir, footer)
    print(f"wrote {out}")
    out = write_curves_csv(cells, curves_by_index, outdir)
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--archs", default=None,
                    help="comma-separated subset (default: every arch in the "
                         "run's manifest with a checkpoint on disk)")
    ap.add_argument("--eval-channel", default="val_best",
                    choices=sorted(CHANNEL_FILENAMES),
                    help="which trained checkpoint to draw: 'val_best' (the "
                         "held-out-validation-best weights, the default) or "
                         "'final' (the last training step). A cell with no "
                         "model_val_best.eqx falls back to model.eqx and is "
                         "labelled as such.")
    args = ap.parse_args(argv)
    archs = (tuple(a.strip() for a in args.archs.split(","))
             if args.archs else None)
    return build_all(Path(args.run_dir).expanduser(),
                     Path(args.outdir).expanduser(),
                     eval_channel=args.eval_channel, archs=archs)


if __name__ == "__main__":
    sys.exit(main())
