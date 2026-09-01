#!/usr/bin/env python
"""Anchored against unanchored enhancement-factor corrections, pretrain to end.

One 2x2 figure holding the two campaign families side by side on the same
axes. Columns are the two channels: the exchange correction ``F_x^NN - F_x^PBE``
at rho = 1 (left) and the correlation correction ``F_c^NN - F_c^PBE`` at
zeta = 0, r_s = 2 (right). Rows are the two stages of a cell's life: the
PRETRAINED network as it leaves the pretraining stage (top) and the OPTIMIZED
network the training stage produced, read from the validation-best checkpoint
(bottom). The zero line of every panel is the parent itself
(``parents.pbe_fx`` / ``parents.pbe_fc`` at libxc constants), so a curve's
height IS the learned correction.

The comparison the figure exists to make: the UNANCHORED generations (v3,
v4gga; dashed) pretrain to a network that already sits 0.01--0.04 off PBE and
then train from there, while the ANCHORED v6 G1 generation (medium,
medium_attn; solid) pretrains to the parent itself and builds every
correction during training. Reading the two rows together separates what the
optimizer discovered from what the pretraining stage happened to leave behind.

Data. Nothing is loaded from a checkpoint here: the curves are read from the
long-form CSVs the two sibling modules already committed for each generation
(``pretrain_fx_fc.py`` -> ``pretrain_fx_fc_curves.csv``, ``trained_fx_fc.py``
-> ``trained_fx_fc_curves.csv``), and the plotted quantity is the
``f_model - f_parent`` of those rows. Both writers evaluate the model and its
parent on one grid inside a single call, so the difference is exact rather
than an interpolation of two separately sampled curves.

Representative of each generation, as read off the committed figure this
module reproduces (``anchored_vs_unanchored_fx_fc.png`` of
``figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best``, rendered 2026-08-30):
its bottom-row panel titles read "OPTIMIZED (val-best, ss=18)", so the trained
stage is a FIXED subset-size cell -- ss = 18 -- for every generation, not each
generation's best held-out cell. Its legend names three curves, "v3 deep_3x16
(unanchored)", "v4gga deep_3x16 (unanchored)" and "v6 medium (anchored)"; the
anchored attention twin is absent there because the v6 G1 run had reached only
8 medium_attn cells at that time, ss = 18 not among them. The refreshed
``trained_fx_fc_curves.csv`` carries the medium_attn ss = 18 cell, so it is
drawn here as the second anchored curve.

GGA rows only. Both v6 G1 architectures drawn here sit on the GGA rung and
their curves carry no ``alpha`` column; a selected row bearing a non-empty
alpha is refused rather than plotted, because a meta-GGA row is a slice
through a SCAN parent and does not belong on a PBE-parent difference axis.

Outputs, into ``--outdir``:
  * ``anchored_vs_unanchored_fx_fc.png``  the 2x2 figure
  * ``anchored_vs_unanchored_fx_fc.csv``  the plotted series, long form
    (series, generation, arch, anchoring, stage, channel, rs, subset_size,
    eval_channel, s, f_model, f_parent, delta)

Usage:
  python notebooks/analysis/anchored_vs_unanchored_fx_fc.py [--outdir DIR]
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: Directory the per-generation figure sets live in; also the default root the
#: source CSVs are resolved against.
ANALYSIS_DIR = Path(__file__).resolve().parent

#: Long-form curve files written by the two sibling modules.
PRETRAIN_CSV = "pretrain_fx_fc_curves.csv"
TRAINED_CSV = "trained_fx_fc_curves.csv"

#: Columns both writers emit. ``subset_size`` and ``eval_channel`` are the
#: trained writer's additions and are required only of that file; ``alpha`` is
#: optional and appears only once a SCAN-parent arch shares the file.
REQUIRED_COLUMNS = ("arch", "channel", "rs", "s", "f_model", "f_parent")
TRAINED_ONLY_COLUMNS = ("subset_size", "eval_channel")

#: The correlation slice drawn in the right column (r_s = 2 is the
#: valence-density scale the reaction sets sample; the convention of
#: ``trained_fx_fc.RS_FIGURE``).
RS_FIGURE = 2.0
#: The trained-stage cell drawn for every generation, read off the committed
#: figure's bottom-row panel titles ("OPTIMIZED (val-best, ss=18)").
TRAINED_SUBSET_SIZE = 18

STAGES = ("pretrained", "optimized")
CHANNELS = ("fx", "fc")

#: Qualitative colours from the Okabe-Ito colour-blind-safe set (Okabe and Ito,
#: "Color Universal Design", 2008), plus a neutral grey for the oldest
#: generation. Series here are GENERATIONS rather than architectures, so the
#: per-arch palette of ``arch_style`` is deliberately not used: two of the four
#: curves share an architecture family and would collide there.
_GREY = "#7f7f7f"
_BLUE = "#0072b2"
_GREEN = "#009e73"
_PURPLE = "#cc79a7"
#: Anchoring is carried by linestyle (solid = anchored, dashed = unanchored) so
#: the split survives a grey-scale print. The two unanchored generations get
#: DIFFERENT dash periods: their pretrained curves coincide to about 1e-5
#: (same pretraining protocol and seed), and one dash pattern over another
#: reads as a single curve where two identical ones would read as one series.
_SOLID = "-"
_DASH_LONG = (0, (7, 3))
_DASH_SHORT = (0, (3, 2))


@dataclass(frozen=True)
class SeriesSpec:
    """One drawn generation/architecture pair and how it is styled."""
    key: str
    generation: str
    figure_dir: str
    arch: str
    anchored: bool
    color: str
    linestyle: object

    @property
    def label(self) -> str:
        state = "anchored" if self.anchored else "unanchored"
        return f"{self.generation} {state} ({self.arch})"


#: The drawn series, in legend order: the two unanchored generations first
#: (oldest first), then the anchored G1 pair.
SERIES: Tuple[SeriesSpec, ...] = (
    SeriesSpec(key="v3", generation="v3",
               figure_dir="figures_dfs_step7_dfs6311_grid3_v3_val_best",
               arch="deep_3x16", anchored=False, color=_GREY,
               linestyle=_DASH_LONG),
    SeriesSpec(key="v4gga", generation="v4gga",
               figure_dir="figures_dfs_step7_dfs6311_grid3_v4gga_val_best",
               arch="deep_3x16", anchored=False, color=_BLUE,
               linestyle=_DASH_SHORT),
    SeriesSpec(key="v6_medium", generation="v6",
               figure_dir="figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best",
               arch="medium", anchored=True, color=_GREEN,
               linestyle=_SOLID),
    SeriesSpec(key="v6_medium_attn", generation="v6",
               figure_dir="figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best",
               arch="medium_attn", anchored=True, color=_PURPLE,
               linestyle=_SOLID),
)

#: The generation whose trained coverage the footer reports (the figure is
#: published into that generation's own directory).
COVERAGE_GENERATION = "v6"

DEFAULT_OUTDIR = (ANALYSIS_DIR
                  / "figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best")

FIGURE_STEM = "anchored_vs_unanchored_fx_fc"


class CurveSourceError(RuntimeError):
    """A source CSV cannot supply a requested curve.

    Raised rather than skipped: a panel silently short of one generation
    would still look like a complete comparison.
    """


@dataclass(frozen=True)
class Curve:
    """One drawn curve: the model, its parent, and the grid they share."""
    s: np.ndarray
    f_model: np.ndarray
    f_parent: np.ndarray
    eval_channel: str

    @property
    def delta(self) -> np.ndarray:
        """The plotted quantity, ``f_model - f_parent``."""
        return self.f_model - self.f_parent


def _rs_matches(cell: str, rs: Optional[float]) -> bool:
    """Whether a row's ``rs`` cell selects the requested slice.

    ``rs is None`` selects the exchange rows, which the writers leave empty;
    a float selects the correlation rows by VALUE, so the ``%g`` formatting of
    the writer ("2" for 2.0) does not have to be reproduced here.
    """
    cell = cell.strip()
    if rs is None:
        return cell == ""
    if cell == "":
        return False
    return float(cell) == float(rs)


def _available(path: Path) -> List[str]:
    """``arch`` (and subset sizes, where the file carries them) on disk, for
    the message of a request that matched nothing."""
    seen: Dict[str, set] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            sizes = seen.setdefault(row["arch"], set())
            if row.get("subset_size"):
                sizes.add(int(row["subset_size"]))
    out = []
    for arch in sorted(seen):
        sizes = sorted(seen[arch])
        out.append(f"{arch} (ss {', '.join(str(v) for v in sizes)})"
                   if sizes else arch)
    return out


def read_curve(path: Path, arch: str, channel: str, *,
               rs: Optional[float] = None,
               subset_size: Optional[int] = None) -> Curve:
    """One curve from a long-form CSV, as ``(s, f_model, f_parent)``.

    ``rs`` selects the correlation slice (``None`` selects the exchange rows,
    whose ``rs`` cell is empty); ``subset_size`` selects a trained cell and
    also switches on the trained-file column requirements. A selected row
    bearing a non-empty ``alpha`` is refused: that is a SCAN-parent slice, and
    this figure's zero line is the PBE parent.
    """
    path = Path(path)
    if not path.is_file():
        raise CurveSourceError(
            f"{path} is not on disk; this figure is built from the committed "
            "long-form curves of each generation's figure set, so a source "
            "root without that file has nothing to draw")
    required = list(REQUIRED_COLUMNS)
    if subset_size is not None:
        required += list(TRAINED_ONLY_COLUMNS)
    rows: List[dict] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or ())
        absent = [c for c in required if c not in fields]
        if absent:
            raise CurveSourceError(
                f"{path} is missing the column(s) {', '.join(absent)}; it "
                f"carries {', '.join(fields) if fields else 'no header'}")
        for row in reader:
            if row["arch"] != arch or row["channel"] != channel:
                continue
            if subset_size is not None and \
                    int(row["subset_size"]) != subset_size:
                continue
            if not _rs_matches(row["rs"], rs):
                continue
            alpha = (row.get("alpha") or "").strip()
            if alpha:
                raise CurveSourceError(
                    f"{path} carries a meta-GGA row for {arch} "
                    f"(channel {channel}, alpha={alpha}); this figure draws "
                    "GGA architectures against the PBE parent, and an "
                    "alpha slice belongs on a SCAN-parent axis")
            rows.append(row)
    if not rows:
        cell = "" if subset_size is None else f", ss={subset_size}"
        raise CurveSourceError(
            f"{path} holds no {channel} row for arch {arch}{cell} at "
            f"rs={'(exchange)' if rs is None else f'{rs:g}'}; it holds "
            f"{'; '.join(_available(path))}")
    s = np.array([float(r["s"]) for r in rows], dtype=float)
    order = np.argsort(s, kind="stable")
    s = s[order]
    if s.size > 1 and not np.all(np.diff(s) > 0.0):
        raise CurveSourceError(
            f"{path} repeats an s value for {arch} {channel}: the selection "
            "matched more than one curve, so the request is ambiguous")
    f_model = np.array([float(r["f_model"]) for r in rows], dtype=float)[order]
    f_parent = np.array([float(r["f_parent"]) for r in rows],
                        dtype=float)[order]
    eval_channel = ""
    if subset_size is not None:
        channels = sorted({r["eval_channel"] for r in rows})
        if len(channels) != 1:
            raise CurveSourceError(
                f"{path} mixes evaluation channels ({', '.join(channels)}) "
                f"within {arch} ss={subset_size} {channel}")
        eval_channel = channels[0]
    return Curve(s=s, f_model=f_model, f_parent=f_parent,
                 eval_channel=eval_channel)


def read_series(spec: SeriesSpec, root: Path) -> Dict[str, Dict[str, Curve]]:
    """Both stages of one series, keyed ``[stage][channel]``."""
    source = Path(root) / spec.figure_dir
    out: Dict[str, Dict[str, Curve]] = {}
    for stage in STAGES:
        filename = PRETRAIN_CSV if stage == "pretrained" else TRAINED_CSV
        subset = None if stage == "pretrained" else TRAINED_SUBSET_SIZE
        path = source / filename
        out[stage] = {
            "fx": read_curve(path, spec.arch, "fx", subset_size=subset),
            "fc": read_curve(path, spec.arch, "fc", rs=RS_FIGURE,
                             subset_size=subset),
        }
    return out


def read_all(root: Path, series: Optional[Sequence[SeriesSpec]] = None
             ) -> List[Tuple[SeriesSpec, Dict[str, Dict[str, Curve]]]]:
    """Every configured series, with the shared s grid enforced.

    The four series are drawn on one axis, so a generation written on a
    different grid would be silently stretched across it. The grids come from
    one module-level constant upstream and are compared exactly. ``series``
    defaults to :data:`SERIES`, resolved at CALL time so the module-level
    tuple is the single definition of what is drawn.
    """
    series = SERIES if series is None else series
    drawn: List[Tuple[SeriesSpec, Dict[str, Dict[str, Curve]]]] = []
    grid: Optional[np.ndarray] = None
    reference = ""
    for spec in series:
        curves = read_series(spec, root)
        for stage in STAGES:
            for channel in CHANNELS:
                s = curves[stage][channel].s
                if grid is None:
                    grid, reference = s, f"{spec.key}/{stage}/{channel}"
                elif not np.array_equal(s, grid):
                    raise CurveSourceError(
                        f"{spec.key}/{stage}/{channel} is written on a "
                        f"different s grid than {reference} "
                        f"({s.size} points spanning {s.min():g}-{s.max():g} "
                        f"against {grid.size} spanning "
                        f"{grid.min():g}-{grid.max():g})")
        drawn.append((spec, curves))
    return drawn


def trained_coverage(path: Path) -> Tuple[Dict[str, List[int]], List[str]]:
    """``({arch: [subset sizes]}, [evaluation channels])`` of a trained file.

    The cell count is what the footer reports as the coverage the bottom row
    was drawn at: a partially drained sweep puts a different number of cells
    behind the same figure.
    """
    sizes: Dict[str, set] = {}
    channels: set = set()
    with open(Path(path), newline="") as fh:
        reader = csv.DictReader(fh)
        absent = [c for c in list(REQUIRED_COLUMNS) + list(TRAINED_ONLY_COLUMNS)
                  if c not in (reader.fieldnames or ())]
        if absent:
            raise CurveSourceError(
                f"{path} is missing the column(s) {', '.join(absent)}")
        for row in reader:
            sizes.setdefault(row["arch"], set()).add(int(row["subset_size"]))
            channels.add(row["eval_channel"])
    return ({arch: sorted(sizes[arch]) for arch in sorted(sizes)},
            sorted(channels))


def _worst_delta(drawn, anchored: bool) -> Tuple[float, float]:
    """``(smallest, largest)`` of ``max|delta|`` over the PRETRAINED curves of
    the anchored or unanchored series, one entry per drawn channel."""
    worst = [float(np.max(np.abs(curves["pretrained"][channel].delta)))
             for spec, curves in drawn if spec.anchored == anchored
             for channel in CHANNELS]
    return (min(worst), max(worst)) if worst else (float("nan"), float("nan"))


def _unanchored_spread(drawn) -> Optional[float]:
    """The largest gap between the PRETRAINED curves of the unanchored series,
    over both channels, or ``None`` when fewer than two are drawn. The two
    legacy generations pretrain under one protocol, so this is the number that
    says whether their top-row curves overlap."""
    unanchored = [curves for spec, curves in drawn if not spec.anchored]
    if len(unanchored) < 2:
        return None
    gaps = [float(np.max(np.abs(a["pretrained"][channel].delta
                                - b["pretrained"][channel].delta)))
            for channel in CHANNELS
            for i, a in enumerate(unanchored) for b in unanchored[i + 1:]]
    return max(gaps)


def footer_text(drawn, root: Path) -> str:
    """The provenance and coverage line under the figure.

    Every number in it is computed from the rows this run read.
    """
    dirs = []
    for spec, _curves in drawn:
        if spec.figure_dir not in dirs:
            dirs.append(spec.figure_dir)
    eval_channels = sorted({curves["optimized"][channel].eval_channel
                            for _spec, curves in drawn
                            for channel in CHANNELS})
    coverage_dirs = sorted({spec.figure_dir for spec, _curves in drawn
                            if spec.generation == COVERAGE_GENERATION})
    parts = [
        "Sources (committed long-form curves, one row per grid point): "
        + f"{PRETRAIN_CSV} and {TRAINED_CSV} of " + ", ".join(dirs) + ".",
        f"Bottom row: the ss={TRAINED_SUBSET_SIZE} cell of each series on the "
        + "/".join(eval_channels) + " channel.",
    ]
    for figure_dir in coverage_dirs:
        sizes, channels = trained_coverage(Path(root) / figure_dir
                                           / TRAINED_CSV)
        cells = sum(len(v) for v in sizes.values())
        detail = ", ".join(f"{arch} {len(sizes[arch])}" for arch in sizes)
        parts.append(
            f"{COVERAGE_GENERATION} trained coverage in {figure_dir}: "
            f"{cells} cells ({detail}), every row on the "
            + "/".join(channels) + " channel.")
    parts.append(
        "Zero line = parents.pbe_fx / parents.pbe_fc (libxc constants), the "
        "anchor's own parent.")
    # Only classes actually drawn are quantified; a figure stating a range
    # over an empty set would print a placeholder as a measurement.
    scales = []
    lo_a, hi_a = _worst_delta(drawn, anchored=True)
    if np.isfinite(hi_a):
        scales.append(f"anchored {lo_a:.1e} to {hi_a:.1e} (below the line "
                      "width at this scale)")
    lo_u, hi_u = _worst_delta(drawn, anchored=False)
    if np.isfinite(hi_u):
        scales.append(f"unanchored {lo_u:.1e} to {hi_u:.1e}")
    if scales:
        parts.append("Pretrained max|delta| over the drawn channels: "
                     + ", ".join(scales) + ".")
    spread = _unanchored_spread(drawn)
    if spread is not None:
        parts.append(
            f"The unanchored pretrains agree to {spread:.1e}, so their "
            "top-row curves lie on top of one another (separate dash "
            "periods).")
    return " ".join(parts)


def render(drawn, outdir: Path, footer: str) -> Path:
    """The 2x2 figure: stages down the rows, channels across the columns."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    titles = {
        ("pretrained", "fx"):
            r"PRETRAINED: $F_x^{\mathrm{NN}} - F_x^{\mathrm{PBE}}$",
        ("pretrained", "fc"):
            r"PRETRAINED: $F_c^{\mathrm{NN}} - F_c^{\mathrm{PBE}}$"
            rf"  ($r_s = {RS_FIGURE:g}$)",
        ("optimized", "fx"):
            rf"OPTIMIZED (val-best, ss={TRAINED_SUBSET_SIZE}): "
            r"$F_x^{\mathrm{NN}} - F_x^{\mathrm{PBE}}$",
        ("optimized", "fc"):
            rf"OPTIMIZED (val-best, ss={TRAINED_SUBSET_SIZE}): "
            r"$F_c^{\mathrm{NN}} - F_c^{\mathrm{PBE}}$"
            rf"  ($r_s = {RS_FIGURE:g}$)",
    }
    for i, stage in enumerate(STAGES):
        for j, channel in enumerate(CHANNELS):
            ax = axes[i][j]
            ax.axhline(0.0, color="0.7", linewidth=1.0, zorder=1)
            for spec, curves in drawn:
                curve = curves[stage][channel]
                ax.plot(curve.s, curve.delta, color=spec.color,
                        linestyle=spec.linestyle, linewidth=2.0, zorder=2,
                        label=spec.label)
            ax.set_title(titles[(stage, channel)], fontsize=10)
            ax.set_xlabel(r"reduced gradient $s$")
            ax.grid(True, color="0.92", linewidth=0.8)
            ax.set_axisbelow(True)
    # ONE legend for the figure: all four panels carry the same four series,
    # and an in-axes legend covers the anchored exchange peak of the optimized
    # panel -- the feature the figure exists to show.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncol=len(labels), frameon=False,
               fontsize=9)
    fig.suptitle("Anchored vs unanchored corrections: pretrained start "
                 "and optimized end", fontsize=13, y=0.99)
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.925))
    out = outdir / f"{FIGURE_STEM}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


CSV_COLUMNS = ("series", "generation", "arch", "anchoring", "stage",
               "channel", "rs", "subset_size", "eval_channel", "s",
               "f_model", "f_parent", "delta")


def write_csv(drawn, outdir: Path) -> Path:
    """The plotted series in long form, one row per drawn grid point."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{FIGURE_STEM}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for spec, curves in drawn:
            anchoring = "anchored" if spec.anchored else "unanchored"
            for stage in STAGES:
                subset = ("" if stage == "pretrained"
                          else str(TRAINED_SUBSET_SIZE))
                for channel in CHANNELS:
                    curve = curves[stage][channel]
                    rs = "" if channel == "fx" else f"{RS_FIGURE:g}"
                    for s, fm, fp in zip(curve.s, curve.f_model,
                                         curve.f_parent):
                        w.writerow([spec.key, spec.generation, spec.arch,
                                    anchoring, stage, channel, rs, subset,
                                    curve.eval_channel, f"{s:.6f}",
                                    repr(float(fm)), repr(float(fp)),
                                    repr(float(fm) - float(fp))])
    return out


def build(root: Path, outdir: Path) -> Tuple[Path, Path]:
    """Read, draw and tabulate; returns ``(png, csv)``."""
    drawn = read_all(Path(root))
    footer = footer_text(drawn, Path(root))
    png = render(drawn, outdir, footer)
    table = write_csv(drawn, outdir)
    return png, table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help="where the figure and its table are written "
                         "(default: the v6 G1 validation-best figure set)")
    ap.add_argument("--source-root", default=str(ANALYSIS_DIR),
                    help="directory holding the per-generation figure sets "
                         "the curves are read from (default: the analysis "
                         "directory this script lives in)")
    args = ap.parse_args(argv)
    try:
        png, table = build(Path(args.source_root).expanduser(),
                           Path(args.outdir).expanduser())
    except CurveSourceError as exc:
        print(f"cannot draw the comparison: {exc}")
        return 2
    print(f"wrote {png}")
    print(f"wrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
