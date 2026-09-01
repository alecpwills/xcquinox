#!/usr/bin/env python
"""Tests for anchored_vs_unanchored_fx_fc.py -- the two-generation comparison.

The module draws nothing itself: every curve is a ``f_model - f_parent``
difference of rows already committed in the per-generation long-form CSVs, so
the load-bearing pins are (a) that the committed sources carry the schema the
reader assumes, (b) that a plotted value equals the difference of the two
columns of the SAME row -- checked against an independent parse of the file,
one pinned value per stage -- and (c) that every refusal the reader can raise
has been seen to fire on an input that triggers it.

The content pin is the figure's own claim: the anchored pretrained curves sit
at the parent (max|delta| below 1e-4) while the unanchored ones start 1e-2
away, so a module that read the wrong generation into the wrong style could
not pass it.
"""
import csv
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchored_vs_unanchored_fx_fc as A  # noqa: E402

MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "anchored_vs_unanchored_fx_fc.py")

#: A float placeholder that reached the caption. Anchored on word boundaries:
#: the bare substring occurs inside "unanchored", which is legitimate text.
_NOT_A_NUMBER = re.compile(r"\bnan\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# An independent parse of a committed CSV, deliberately not reusing the
# module's reader: a shared parsing bug would otherwise cancel out.
# ---------------------------------------------------------------------------

def _raw_rows(path):
    with open(path) as fh:
        lines = [line.rstrip("\n") for line in fh if line.strip()]
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def _raw_delta_at(path, *, arch, channel, s, rs="", subset_size=None):
    """``f_model - f_parent`` of the single matching row, parsed by hand."""
    hits = []
    for row in _raw_rows(path):
        if row["arch"] != arch or row["channel"] != channel:
            continue
        if row["rs"] != rs:
            continue
        if subset_size is not None and row["subset_size"] != str(subset_size):
            continue
        if float(row["s"]) != s:
            continue
        hits.append(float(row["f_model"]) - float(row["f_parent"]))
    assert len(hits) == 1, (path, arch, channel, s, rs, subset_size, len(hits))
    return hits[0]


def _source(spec, filename):
    return os.path.join(A.ANALYSIS_DIR, spec.figure_dir, filename)


# ---------------------------------------------------------------------------
# Synthetic generations, for the refusals and for the CLI seam
# ---------------------------------------------------------------------------

def _write_generation(root, figure_dir, arch, s_values, *, alpha=None,
                      drop_column=None, duplicate=False,
                      subset_size=A.TRAINED_SUBSET_SIZE):
    """A minimal pair of long-form CSVs under ``root/figure_dir``."""
    directory = root / figure_dir
    directory.mkdir(parents=True, exist_ok=True)
    for filename in (A.PRETRAIN_CSV, A.TRAINED_CSV):
        trained = filename == A.TRAINED_CSV
        columns = ["arch", "channel", "rs", "s", "f_model", "f_parent"]
        if trained:
            columns = (["arch", "subset_size"] + columns[1:]
                       + ["eval_channel"])
        if alpha is not None:
            columns.append("alpha")
        if drop_column is not None and drop_column in columns:
            columns.remove(drop_column)
        rows = []
        for channel, rs in (("fx", ""), ("fc", f"{A.RS_FIGURE:g}")):
            for i, s in enumerate(s_values):
                values = {"arch": arch, "channel": channel, "rs": rs,
                          "s": f"{s:.6f}", "f_model": repr(1.0 + 0.01 * i),
                          "f_parent": "1.0", "subset_size": str(subset_size),
                          "eval_channel": "val_best", "alpha": alpha or ""}
                rows.append([values[c] for c in columns])
                if duplicate:
                    rows.append([values[c] for c in columns])
        with open(directory / filename, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(columns)
            w.writerows(rows)
    return directory


def _synthetic_spec(figure_dir, arch, **kwargs):
    defaults = dict(key=figure_dir, generation="test", figure_dir=figure_dir,
                    arch=arch, anchored=True, color="#000000", linestyle="-")
    defaults.update(kwargs)
    return A.SeriesSpec(**defaults)


# ---------------------------------------------------------------------------
# The committed sources
# ---------------------------------------------------------------------------

def test_source_csvs_exist_with_the_expected_columns():
    """Every source the module names is on disk with the schema it reads."""
    seen = set()
    for spec in A.SERIES:
        for filename in (A.PRETRAIN_CSV, A.TRAINED_CSV):
            path = _source(spec, filename)
            if path in seen:
                continue
            seen.add(path)
            assert os.path.isfile(path), path
            with open(path) as fh:
                header = fh.readline().rstrip("\n").split(",")
            expected = list(A.REQUIRED_COLUMNS)
            if filename == A.TRAINED_CSV:
                expected += list(A.TRAINED_ONLY_COLUMNS)
            assert not [c for c in expected if c not in header], (path, header)
    assert len(seen) == 6, sorted(seen)


def test_no_source_carries_a_meta_gga_row_for_a_drawn_arch():
    """The drawn architectures are GGA: no ``alpha`` column, and so no SCAN
    slice smuggled onto a PBE-parent difference axis."""
    for spec in A.SERIES:
        for filename in (A.PRETRAIN_CSV, A.TRAINED_CSV):
            path = _source(spec, filename)
            with open(path) as fh:
                header = fh.readline().rstrip("\n").split(",")
            if "alpha" not in header:
                continue
            index = header.index("alpha")
            with open(path) as fh:
                next(fh)
                offenders = [line for line in fh
                             if line.split(",")[0] == spec.arch
                             and line.split(",")[index].strip()]
            assert not offenders, (path, spec.arch, offenders[:2])


def test_trained_subset_size_is_the_largest_cell_common_to_every_series():
    """The drawn cell is not an arbitrary pick.

    ss=18, the cell the committed figure's bottom-row titles name, is the
    LARGEST subset size every drawn generation has reached, which is what
    makes the bottom row a like-for-like comparison rather than four cells of
    different training-set sizes. Should a later pull give every generation a
    larger shared cell, this fires: redraw at that cell, or record why the
    figure stays where it is.
    """
    shared = None
    for spec in A.SERIES:
        sizes, _channels = A.trained_coverage(
            os.path.join(A.ANALYSIS_DIR, spec.figure_dir, A.TRAINED_CSV))
        available = set(sizes[spec.arch])
        shared = available if shared is None else shared & available
    assert shared, [spec.key for spec in A.SERIES]
    assert A.TRAINED_SUBSET_SIZE in shared, sorted(shared)
    assert A.TRAINED_SUBSET_SIZE == max(shared), sorted(shared)


def test_every_series_resolves_on_one_grid():
    """All four series read, both stages, on the one shared s grid."""
    drawn = A.read_all(A.ANALYSIS_DIR)
    assert [spec.key for spec, _ in drawn] == [s.key for s in A.SERIES]
    grid = drawn[0][1]["pretrained"]["fx"].s
    assert grid[0] == 0.0 and grid[-1] == 6.0 and grid.size > 100
    for spec, curves in drawn:
        for stage in A.STAGES:
            for channel in A.CHANNELS:
                curve = curves[stage][channel]
                assert np.array_equal(curve.s, grid), (spec.key, stage,
                                                       channel)
                assert np.all(np.isfinite(curve.delta))
                if stage == "optimized":
                    assert curve.eval_channel, (spec.key, channel)
                else:
                    assert curve.eval_channel == ""


def test_pretrained_value_matches_an_independent_parse_of_the_csv():
    """One pinned PRETRAINED value, computed from the file inside the test."""
    spec = next(s for s in A.SERIES if s.key == "v4gga")
    path = _source(spec, A.PRETRAIN_CSV)
    curve = A.read_curve(path, spec.arch, "fx")
    s_pin = 3.0
    expected = _raw_delta_at(path, arch=spec.arch, channel="fx", s=s_pin)
    got = float(curve.delta[int(np.argmin(np.abs(curve.s - s_pin)))])
    assert got == pytest.approx(expected, rel=0.0, abs=0.0)
    # The pin is on the DIFFERENCE, not on either column alone.
    assert abs(expected) > 1e-2, expected


def test_optimized_value_matches_an_independent_parse_of_the_csv():
    """One pinned OPTIMIZED value: the anchored correlation correction at the
    top of the grid, where the pre-image suppression is largest."""
    spec = next(s for s in A.SERIES if s.key == "v6_medium")
    path = _source(spec, A.TRAINED_CSV)
    curve = A.read_curve(path, spec.arch, "fc", rs=A.RS_FIGURE,
                         subset_size=A.TRAINED_SUBSET_SIZE)
    s_pin = 6.0
    expected = _raw_delta_at(path, arch=spec.arch, channel="fc", s=s_pin,
                             rs=f"{A.RS_FIGURE:g}",
                             subset_size=A.TRAINED_SUBSET_SIZE)
    got = float(curve.delta[int(np.argmin(np.abs(curve.s - s_pin)))])
    assert got == pytest.approx(expected, rel=0.0, abs=0.0)
    assert curve.eval_channel == "val_best"


def test_anchored_pretrains_sit_at_the_parent_and_unanchored_do_not():
    """The figure's own claim, as a discriminating pin: swapping which
    generation is drawn as anchored breaks both halves at once."""
    drawn = A.read_all(A.ANALYSIS_DIR)
    for spec, curves in drawn:
        worst = max(float(np.max(np.abs(curves["pretrained"][c].delta)))
                    for c in A.CHANNELS)
        if spec.anchored:
            assert worst < 1e-4, (spec.key, worst)
        else:
            assert worst > 1e-2, (spec.key, worst)


def test_optimized_corrections_are_orders_above_the_anchored_start():
    """The bottom row is not the top row: every drawn cell moves off its
    pretrained state by at least an order of magnitude more than the anchored
    pretrain sat from the parent."""
    drawn = A.read_all(A.ANALYSIS_DIR)
    for spec, curves in drawn:
        worst = max(float(np.max(np.abs(curves["optimized"][c].delta)))
                    for c in A.CHANNELS)
        assert worst > 1e-2, (spec.key, worst)


# ---------------------------------------------------------------------------
# Refusals -- each seen to fire on an input that triggers it
# ---------------------------------------------------------------------------

def test_alpha_bearing_row_is_refused_and_a_blank_alpha_is_not(tmp_path):
    """The guard fires on the alpha VALUE, not on the column's presence: the
    same fixture with an empty alpha cell reads normally."""
    bearing = _write_generation(tmp_path, "gen_alpha", "medium",
                                [0.0, 3.0, 6.0], alpha="1")
    with pytest.raises(A.CurveSourceError) as exc:
        A.read_curve(bearing / A.PRETRAIN_CSV, "medium", "fx")
    assert "alpha" in str(exc.value)
    blank = _write_generation(tmp_path, "gen_blank", "medium",
                              [0.0, 3.0, 6.0], alpha="")
    curve = A.read_curve(blank / A.PRETRAIN_CSV, "medium", "fx")
    assert curve.s.size == 3


def test_absent_cell_is_refused_and_names_what_is_on_disk():
    """A subset size the sweep has not reached is a refusal, not an empty
    panel; the message carries the cells that do exist."""
    spec = next(s for s in A.SERIES if s.key == "v6_medium")
    path = _source(spec, A.TRAINED_CSV)
    absent = max(A.trained_coverage(path)[0][spec.arch]) + 1
    with pytest.raises(A.CurveSourceError) as exc:
        A.read_curve(path, spec.arch, "fx", subset_size=absent)
    assert spec.arch in str(exc.value) and str(absent) in str(exc.value)


def test_absent_arch_is_refused():
    spec = next(s for s in A.SERIES if s.key == "v3")
    with pytest.raises(A.CurveSourceError):
        A.read_curve(_source(spec, A.PRETRAIN_CSV), "no_such_arch", "fx")


def test_missing_column_is_refused(tmp_path):
    directory = _write_generation(tmp_path, "gen_thin", "medium",
                                  [0.0, 3.0, 6.0], drop_column="f_parent")
    with pytest.raises(A.CurveSourceError) as exc:
        A.read_curve(directory / A.PRETRAIN_CSV, "medium", "fx")
    assert "f_parent" in str(exc.value)


def test_ambiguous_selection_is_refused(tmp_path):
    """Two rows at one s means the selection matched more than one curve."""
    directory = _write_generation(tmp_path, "gen_dup", "medium",
                                  [0.0, 3.0, 6.0], duplicate=True)
    with pytest.raises(A.CurveSourceError) as exc:
        A.read_curve(directory / A.PRETRAIN_CSV, "medium", "fx")
    assert "repeats an s value" in str(exc.value)


def test_mismatched_grid_is_refused(tmp_path):
    """Two generations written on different grids cannot share an axis."""
    _write_generation(tmp_path, "gen_a", "medium", np.linspace(0.0, 6.0, 5))
    _write_generation(tmp_path, "gen_b", "medium", np.linspace(0.0, 6.0, 7))
    series = (_synthetic_spec("gen_a", "medium"),
              _synthetic_spec("gen_b", "medium"))
    with pytest.raises(A.CurveSourceError) as exc:
        A.read_all(tmp_path, series)
    assert "s grid" in str(exc.value)
    # Same fixture on ONE grid resolves, so the refusal is about the grid.
    _write_generation(tmp_path, "gen_c", "medium", np.linspace(0.0, 6.0, 5))
    ok = A.read_all(tmp_path, (series[0], _synthetic_spec("gen_c", "medium")))
    assert len(ok) == 2


# ---------------------------------------------------------------------------
# Footer, table and the command-line seam
# ---------------------------------------------------------------------------

def test_footer_states_the_v6_trained_coverage():
    """The coverage sentence counts the cells actually in the source file."""
    drawn = A.read_all(A.ANALYSIS_DIR)
    spec = next(s for s in A.SERIES
                if s.generation == A.COVERAGE_GENERATION)
    path = os.path.join(A.ANALYSIS_DIR, spec.figure_dir, A.TRAINED_CSV)
    pairs = {(row["arch"], row["subset_size"]) for row in _raw_rows(path)}
    channels = {row["eval_channel"] for row in _raw_rows(path)}
    footer = A.footer_text(drawn, A.ANALYSIS_DIR)
    assert f"{len(pairs)} cells" in footer, footer
    for channel in channels:
        assert channel in footer
    assert f"ss={A.TRAINED_SUBSET_SIZE}" in footer
    assert not _NOT_A_NUMBER.search(footer), footer


def test_footer_quantifies_only_the_classes_actually_drawn(tmp_path):
    """A series set holding one anchoring class states a range for that class
    alone: an empty min/max would otherwise be printed as a measurement."""
    for name in ("gen_a", "gen_b"):
        _write_generation(tmp_path, name, "medium", np.linspace(0.0, 6.0, 5))
    series = (_synthetic_spec("gen_a", "medium"),
              _synthetic_spec("gen_b", "medium"))
    drawn = A.read_all(tmp_path, series)
    footer = A.footer_text(drawn, tmp_path)
    assert not _NOT_A_NUMBER.search(footer), footer
    assert "anchored" in footer and "unanchored" not in footer, footer


def test_cli_writes_the_figure_and_its_table(tmp_path):
    """End to end on the committed sources, into a scratch directory."""
    assert A.main(["--outdir", str(tmp_path)]) == 0
    png = tmp_path / f"{A.FIGURE_STEM}.png"
    table = tmp_path / f"{A.FIGURE_STEM}.csv"
    assert png.is_file() and png.stat().st_size > 0
    assert table.is_file() and table.stat().st_size > 0
    with open(table) as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0].keys()) == A.CSV_COLUMNS
    grid = A.read_all(A.ANALYSIS_DIR)[0][1]["pretrained"]["fx"].s
    assert len(rows) == (len(A.SERIES) * len(A.STAGES) * len(A.CHANNELS)
                         * grid.size)
    for row in rows[:: max(1, len(rows) // 40)]:
        assert (float(row["delta"])
                == pytest.approx(float(row["f_model"])
                                 - float(row["f_parent"]), rel=0.0, abs=0.0))
    optimized = [r for r in rows if r["stage"] == "optimized"]
    assert optimized and all(
        r["subset_size"] == str(A.TRAINED_SUBSET_SIZE) and r["eval_channel"]
        for r in optimized)
    assert all(r["subset_size"] == "" and r["eval_channel"] == ""
               for r in rows if r["stage"] == "pretrained")


def test_cli_reports_a_source_refusal_instead_of_a_traceback(tmp_path):
    """A source root without the generations exits 2 with a message."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert A.main(["--outdir", str(tmp_path / "out"),
                   "--source-root", str(empty)]) == 2


def test_cli_honours_an_alternate_source_root(tmp_path, monkeypatch):
    """``--source-root`` is what the reader resolves against."""
    for name in ("gen_a", "gen_b"):
        _write_generation(tmp_path / "src", name, "medium",
                          np.linspace(0.0, 6.0, 5))
    monkeypatch.setattr(A, "SERIES", (_synthetic_spec("gen_a", "medium"),
                                      _synthetic_spec("gen_b", "medium")))
    monkeypatch.setattr(A, "COVERAGE_GENERATION", "test")
    out = tmp_path / "out"
    assert A.main(["--outdir", str(out),
                   "--source-root", str(tmp_path / "src")]) == 0
    assert (out / f"{A.FIGURE_STEM}.png").is_file()
    assert (out / f"{A.FIGURE_STEM}.csv").is_file()


# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------

#: Tokens that must not appear in a committed module: process meta-commentary,
#: attribution, and the puffery register.
_BANNED = ("subagent", "adversarial", "auditor", "co-authored",
           "generated with", "multi-agent", "refute", "delve", "leverage",
           "seamless", "rigorous", "compelling", "as an ai", "claude",
           "anthropic")


def _style_offences(text):
    lowered = text.lower()
    found = [token for token in _BANNED if token in lowered]
    found += [f"non-ascii {ord(ch):#x}" for ch in sorted(set(text))
              if ord(ch) > 127]
    return found


def test_module_text_is_ascii_and_free_of_process_meta_commentary():
    with open(MODULE_PATH, encoding="utf-8") as fh:
        text = fh.read()
    assert _style_offences(text) == []
    # The sweep is not vacuous: it fires on each class of offence it covers.
    # The dash is written as an escape so this file stays ASCII on disk.
    assert _style_offences("a rigorous check") == ["rigorous"]
    assert _style_offences("an em dash \u2014 here") == ["non-ascii 0x2014"]
