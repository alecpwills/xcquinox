"""Tests for ``notebooks/analysis/plot_subset_jsd.py`` (subset-JSD figure + table).

The script draws the training-subset selection record: the Jensen-Shannon
divergence of each chosen subset against the full 26-point pool as a function
of the subset size r, the three descriptor marginals of the pool with the
chosen subsets overlaid, and a CSV carrying each subset's members and its
divergence.

Expected values here are computed with the package's own selection primitives
(``xcquinox.alec.subset_selection``), the functions that produced the shipped
ledger; the tests therefore pin the script to the same definition of the
metric rather than to a re-derivation of it. Everything is synthetic: no SCF,
no cluster data, matplotlib on Agg.

The script under test is loaded by path INSIDE each test (the sibling script
tests load at module import). The lazy load is deliberate: while the script is
absent every test reports its own FileNotFoundError instead of the module
collapsing into a single collection error, so the RED state names each
requirement separately.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from xcquinox.alec import subset_selection as ss  # noqa: E402

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "plot_subset_jsd.py"
_MODNAME = "plot_subset_jsd"
_KEYS = ("rho_third", "s", "alpha")


def _load_script():
    """Load the script under test from its path, fresh on every call.

    A fresh module object per test keeps one test's monkeypatched module
    attributes out of the next one's globals.
    """
    spec = importlib.util.spec_from_file_location(_MODNAME, _SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build a module spec for {_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODNAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(_MODNAME, None)
        raise
    return mod


# ---------------------------------------------------------------------------
# synthetic pool: per-point descriptor blocks, stub training points
# ---------------------------------------------------------------------------

def _synthetic_blocks(sizes=(320, 260, 400, 180, 300),
                      means=(0.60, 1.00, 1.50, 2.10, 0.85)):
    """Five per-point descriptor blocks with unit quadrature weights.

    One seeded generator drives every block, so the pool is reproducible; the
    per-block means differ so that a strict subset is genuinely divergent from
    the pool (a pool of identical blocks would make every JSD zero and every
    comparison below vacuous).
    """
    rng = np.random.default_rng(0)
    blocks = []
    for n, mu in zip(sizes, means):
        blocks.append({
            "rho_third": np.abs(rng.normal(mu, 0.30, n)),
            "s": np.abs(rng.normal(1.5 * mu, 0.70, n)),
            "alpha": np.abs(rng.normal(0.4 + mu, 0.35, n)),
            "weights": np.ones(n),
        })
    return blocks


def _concat(blocks, indices):
    sel = [blocks[i] for i in indices]
    out = {k: np.concatenate([b[k] for b in sel]) for k in _KEYS}
    out["weights"] = np.concatenate([b["weights"] for b in sel])
    return out


def _expected_jsd(h_ref, edges, blocks, indices):
    """(total, per-marginal parts) for a subset, from the package primitives."""
    h_cand = ss._bin_with_edges(_concat(blocks, indices), edges)
    total = ss.metric_jsd(h_ref, h_cand)
    parts = {}
    for k in _KEYS:
        p = ss._to_pmf(h_ref[k])
        q = ss._to_pmf(h_cand[k])
        m = 0.5 * (p + q)
        parts[k] = 0.5 * (ss._kl(p, m) + ss._kl(q, m))
    return total, parts


class _StubPoint:
    """Stand-in for ``xcquinox.alec.training_points.TrainingPoint``.

    Carries the three attributes the figure code reads: ``name``, ``kind`` and
    the tuple of ASE ``Atoms`` in ``species``.
    """

    def __init__(self, name, kind, species):
        self.name = name
        self.kind = kind
        self.species = tuple(species)
        self.metadata = {}

    def __repr__(self):  # pragma: no cover - diagnostics only
        return f"_StubPoint({self.name!r}, {self.kind!r})"


def _atoms(symbols, name, charge, spin):
    at = Atoms(symbols)
    at.info.update({"name": name, "charge": int(charge), "spin": int(spin)})
    return at


def _pool_points():
    """Five stub points over the five synthetic blocks, one shared H anchor.

    The shared anchor exercises the species union: H must appear once, in the
    order it was first met.
    """
    h = _atoms("H", "H", 0, 1)
    return [
        _StubPoint("H2O", "ae",
                   [_atoms("H2O", "H2O", 0, 0), h, _atoms("O", "O", 0, 2)]),
        _StubPoint("CH4", "ae",
                   [_atoms("CH4", "CH4", 0, 0), _atoms("C", "C", 0, 2), h]),
        _StubPoint("OH+N2_to_H+N2O", "bh76",
                   [_atoms("OH", "HO", 0, 1), _atoms("N2", "N2", 0, 0), h,
                    _atoms("N2O", "N2O", 0, 0)]),
        _StubPoint("Li_IP", "ip13",
                   [_atoms("Li", "Li", 0, 1), _atoms("Li", "Li+", 1, 0)]),
        _StubPoint("CO", "ae",
                   [_atoms("CO", "CO", 0, 0), _atoms("C", "C", 0, 2),
                    _atoms("O", "O", 0, 2)]),
    ]


def _species_union(points, indices):
    out = []
    for i in indices:
        for sp in points[i].species:
            nm = sp.info["name"]
            if nm not in out:
                out.append(nm)
    return out


def _expected_row(r, indices, points, blocks, h_ref, edges):
    total, parts = _expected_jsd(h_ref, edges, blocks, indices)
    kinds = [points[i].kind for i in indices]
    return {
        "r": r,
        "jsd_ledger": total,
        "jsd": total,
        "jsd_rho": parts["rho_third"],
        "jsd_s": parts["s"],
        "jsd_alpha": parts["alpha"],
        "point_names": [points[i].name for i in indices],
        "point_kinds": kinds,
        "species": _species_union(points, indices),
        "n_ae": kinds.count("ae"),
        "n_bh76": kinds.count("bh76"),
        "n_ip13": kinds.count("ip13"),
        # the members as pool indices: the marginal panels bin the subset's
        # blocks from these, and the CSV carries them beside the names
        "chosen_indices": list(indices),
    }


_ROW_KEYS = ("r", "jsd_ledger", "jsd", "jsd_rho", "jsd_s", "jsd_alpha",
             "point_names", "point_kinds", "species", "n_ae", "n_bh76",
             "n_ip13", "chosen_indices")

# r = 1, 2 and 4 over the 5-point pool: every entry a STRICT subset, so every
# ledger value is positive and no comparison below is satisfied by zero.
_PLAN = {1: (0,), 2: (0, 3), 4: (0, 1, 2, 3)}


def _plan_rows(points, blocks, h_ref, edges):
    return [_expected_row(r, idx, points, blocks, h_ref, edges)
            for r, idx in sorted(_PLAN.items())]


def _ledger_from_rows(rows):
    return {
        row["r"]: {
            "chosen_indices": list(_PLAN[row["r"]]),
            "metric_value": row["jsd"],
            "point_names": list(row["point_names"]),
            "point_kinds": list(row["point_kinds"]),
            "tag": f"bin{row['r']:02d}",
        }
        for row in rows
    }


def _names_number(message, n):
    """True when ``n`` stands alone in ``message`` (not inside a float).

    A bare ``str(n) in message`` would be satisfied by the digit appearing
    inside a printed divergence such as 0.0474, which is not the ledger key
    the error is required to name.
    """
    return re.search(rf"(?<![\w.]){n}(?![\w.])", message) is not None


def _write_species_cache(path, n, seed, mean):
    """Write one per-species descriptor cache in the shipped npz layout."""
    rng = np.random.default_rng(seed)
    arrs = {
        "rho_third": np.abs(rng.normal(mean, 0.30, n)),
        "s": np.abs(rng.normal(1.5 * mean, 0.70, n)),
        "alpha": np.abs(rng.normal(0.4 + mean, 0.35, n)),
        "weights": np.ones(n),
    }
    np.savez(path, **arrs)
    return arrs


def _concat_arrays(arr_list):
    out = {k: np.concatenate([a[k] for a in arr_list]) for k in _KEYS}
    out["weights"] = np.concatenate([a["weights"] for a in arr_list])
    return out


# ---------------------------------------------------------------------------
# figure introspection helpers
# ---------------------------------------------------------------------------

def _secondary_axes(ax):
    """The secondary axes of a panel (``ax.secondary_xaxis``).

    A secondary axis is a CHILD axis of its panel (``ax.child_axes``), not an
    entry of ``fig.axes``, so the panel count is ``len(fig.axes)`` and the
    secondary labels are only reachable through the children.
    """
    return [c for c in ax.child_axes
            if any(k.__name__ == "SecondaryAxis" for k in type(c).__mro__)]


def _figure_texts(fig):
    """Every string the figure carries: titles, labels, legends, free text,
    including the labels of the secondary (child) axes."""
    out = [t.get_text() for t in fig.texts]
    for lg in fig.legends:
        out += [t.get_text() for t in lg.get_texts()]
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.texts]
        lg = ax.get_legend()
        if lg is not None:
            out += [t.get_text() for t in lg.get_texts()]
        for sec in _secondary_axes(ax):
            out += [sec.get_xlabel(), sec.get_ylabel()]
    return [t for t in out if t]


# ===========================================================================
# T1: subset_jsd
# ===========================================================================

def test_subset_jsd_is_the_package_metric_and_its_marginal_parts():
    mod = _load_script()
    blocks = _synthetic_blocks()
    h_ref, edges = ss.build_reference_histograms(blocks)

    indices = (0, 3)
    expect_total, expect_parts = _expected_jsd(h_ref, edges, blocks, indices)
    # a strict subset of a pool whose blocks differ really is divergent; without
    # this the equalities below would hold trivially at zero.
    assert expect_total > 1e-6

    total, parts = mod.subset_jsd(h_ref, edges, blocks, indices)

    # the total is metric_jsd of the subset binned ON THE REFERENCE EDGES
    assert total == pytest.approx(expect_total, rel=1e-12, abs=1e-15)
    assert set(parts) == set(_KEYS)
    # equal marginal weights: the three parts are the total
    assert sum(parts.values()) == pytest.approx(total, rel=1e-12, abs=1e-15)
    for k in _KEYS:
        assert parts[k] == pytest.approx(expect_parts[k], rel=1e-12, abs=1e-15)
        # Lin 1991: a PMF-normalized marginal JSD lies in [0, ln 2]
        assert 0.0 <= parts[k] <= np.log(2.0) + 1e-12

    # binning the subset on its OWN percentile edges gives a different number;
    # the reference-edge value is the one the ledger holds.
    own_edges = ss.metric_jsd(h_ref, ss.bin_descriptors(_concat(blocks, indices)))
    assert abs(own_edges - expect_total) > 1e-6

    # the whole pool IS the reference: zero divergence, zero in every marginal
    full = tuple(range(len(blocks)))
    total_full, parts_full = mod.subset_jsd(h_ref, edges, blocks, full)
    assert total_full == pytest.approx(0.0, abs=1e-12)
    for k in _KEYS:
        assert parts_full[k] == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# T2: subset_rows
# ===========================================================================

def test_subset_rows_reproduces_the_ledger_and_names_the_members():
    mod = _load_script()
    blocks = _synthetic_blocks()
    points = _pool_points()
    h_ref, edges = ss.build_reference_histograms(blocks)
    expected = _plan_rows(points, blocks, h_ref, edges)
    ledger = _ledger_from_rows(expected)

    rows = mod.subset_rows(ledger, points, blocks, h_ref, edges)

    assert [row["r"] for row in rows] == [1, 2, 4]
    for row, exp in zip(rows, expected):
        for key in _ROW_KEYS:
            assert key in row, f"row r={exp['r']} is missing {key!r}"
        assert row["jsd_ledger"] == pytest.approx(exp["jsd_ledger"],
                                                  rel=1e-12, abs=1e-15)
        assert row["jsd"] == pytest.approx(exp["jsd"], rel=1e-12, abs=1e-15)
        assert row["jsd"] > 0.0
        for key, ref in (("jsd_rho", exp["jsd_rho"]), ("jsd_s", exp["jsd_s"]),
                         ("jsd_alpha", exp["jsd_alpha"])):
            assert row[key] == pytest.approx(ref, rel=1e-12, abs=1e-15)
        assert (row["jsd_rho"] + row["jsd_s"] + row["jsd_alpha"]
                == pytest.approx(row["jsd"], rel=1e-12, abs=1e-15))
        assert list(row["point_names"]) == exp["point_names"]
        assert list(row["point_kinds"]) == exp["point_kinds"]
        # the species union is deduplicated (H is an anchor of three points)
        # and keeps first-appearance order
        assert list(row["species"]) == exp["species"]
        assert len(set(row["species"])) == len(row["species"])
        assert (row["n_ae"], row["n_bh76"], row["n_ip13"]) == (
            exp["n_ae"], exp["n_bh76"], exp["n_ip13"])

    # the r = 4 subset draws one point of each kind plus a second AE point
    r4 = rows[-1]
    assert (r4["n_ae"], r4["n_bh76"], r4["n_ip13"]) == (2, 1, 1)
    assert r4["n_ae"] + r4["n_bh76"] + r4["n_ip13"] == 4

    # a ledger value that does not reproduce is refused, and the entry named.
    # 1e-3 is six orders above the 1e-9 agreement the recomputation holds to.
    bad = {r: dict(v) for r, v in ledger.items()}
    bad[4]["metric_value"] = float(bad[4]["metric_value"]) + 1e-3
    with pytest.raises(ValueError) as exc:
        mod.subset_rows(bad, points, blocks, h_ref, edges)
    assert _names_number(str(exc.value), 4), str(exc.value)

    # the ledger's own member names and kinds are cross-checked against the pool:
    # a renamed point changes the published table only through an error.
    for field in ("point_names", "point_kinds"):
        renamed = {r: dict(v) for r, v in ledger.items()}
        renamed[2][field] = list(renamed[2][field])
        renamed[2][field][0] = "xx"
        with pytest.raises(ValueError) as exc:
            mod.subset_rows(renamed, points, blocks, h_ref, edges)
        assert field in str(exc.value) and _names_number(str(exc.value), 2), str(exc.value)


def test_subset_rows_refuses_a_non_finite_recomputation():
    """The guard must fail CLOSED: a subset with no in-range mass in one marginal
    makes the package metric non-finite (NaN through density=True), and
    ``abs(nan - x) > tol`` is False, so a guard written as ``>`` accepts it."""
    mod = _load_script()
    blocks = _synthetic_blocks()
    points = _pool_points()
    h_ref, edges = ss.build_reference_histograms(blocks)
    # a sixth block entirely above the s range of the reference edges
    far = {k: np.full(50, 1.0) for k in _KEYS}
    far["s"] = np.full(50, float(edges["s"][-1]) + 100.0)
    far["weights"] = np.ones(50)
    blocks_plus = list(blocks) + [far]
    points_plus = list(points) + [_StubPoint("FAR", "ae", [_atoms("H", "H", 0, 1)])]

    # (numpy's density=True divides by the zero in-range weight: the NaN itself
    # is the premise of the test, its warning is not the subject)
    with np.errstate(invalid="ignore"):
        total, parts = mod.subset_jsd(h_ref, edges, blocks_plus, (5,))
    assert not np.isfinite(total)   # the premise of the test

    ledger = {1: {"chosen_indices": [5], "metric_value": 0.5,
                  "point_names": ["FAR"], "point_kinds": ["ae"], "tag": "bin01"}}
    with pytest.raises(ValueError) as exc, np.errstate(invalid="ignore"):
        mod.subset_rows(ledger, points_plus, blocks_plus, h_ref, edges)
    assert _names_number(str(exc.value), 1), str(exc.value)


# ===========================================================================
# T3: pool_blocks
# ===========================================================================

def test_pool_blocks_reads_the_caches_by_the_stable_prefix(tmp_path, monkeypatch):
    mod = _load_script()
    cache = tmp_path / "subset_descriptors"
    cache.mkdir()
    n_li, n_h = 41, 27
    li = _write_species_cache(cache / "Liplus_c1_s0_Li.npz", n_li, seed=11,
                              mean=1.7)
    h = _write_species_cache(cache / "H_c0_s1_H.npz", n_h, seed=12, mean=0.9)

    # The cation's cache is keyed ('Li+', 1, 0) but named with '+' spelled
    # 'plus', and its formula suffix ('Li') is not its name: only the
    # <name>_c<charge>_s<spin>_ prefix locates the file.
    points = [
        _StubPoint("Li_IP", "ip13",
                   [_atoms("Li", "Li+", 1, 0), _atoms("H", "H", 0, 1)]),
        _StubPoint("H2", "ae", [_atoms("H", "H", 0, 1)]),
    ]
    monkeypatch.setattr(mod, "build_dfs_pool_points", lambda *a, **k: points)

    got_points, blocks = mod.pool_blocks(cache)

    assert len(got_points) == 2
    assert [p.name for p in got_points] == ["Li_IP", "H2"]
    assert len(blocks) == 2
    # block 0 concatenates both of its species, block 1 carries H alone
    for key in ("rho_third", "s", "alpha", "weights"):
        assert blocks[0][key].size == n_li + n_h
        assert blocks[1][key].size == n_h
    assert np.allclose(np.sort(blocks[1]["rho_third"]),
                       np.sort(h["rho_third"]))
    assert np.allclose(np.sort(blocks[0]["alpha"]),
                       np.sort(np.concatenate([li["alpha"], h["alpha"]])))

    # two caches under one prefix are refused rather than the first taken: a
    # silent first-match would read another species' grid.
    _write_species_cache(cache / "H_c0_s1_H2.npz", 5, seed=13, mean=0.9)
    with pytest.raises(ValueError) as exc:
        mod.pool_blocks(cache)
    assert "('H', 0, 1)" in str(exc.value), str(exc.value)
    (cache / "H_c0_s1_H2.npz").unlink()

    # a cache that is not on disk is refused, with the key it was looked up by
    (cache / "H_c0_s1_H.npz").unlink()
    with pytest.raises((FileNotFoundError, KeyError)) as exc:
        mod.pool_blocks(cache)
    msg = str(exc.value)
    assert "('H', 0, 1)" in msg or "H_c0_s1" in msg, msg


# ===========================================================================
# T4: plot_subset_jsd and write_subset_table
# ===========================================================================

def test_plot_subset_jsd_draws_four_panels_with_the_bound_and_the_mass_note(
        tmp_path, monkeypatch):
    mod = _load_script()
    blocks = _synthetic_blocks()
    points = _pool_points()
    h_ref, edges = ss.build_reference_histograms(blocks)
    rows = _plan_rows(points, blocks, h_ref, edges)

    # one populated reference bin far below any plotting "floor": every positive
    # mass must be drawn on the log axis, only exact zeros may be left out.
    tiny_bin = 7
    assert h_ref["rho_third"][tiny_bin] > 0.0
    h_ref["rho_third"][tiny_bin] = 1e-15 * float(h_ref["rho_third"].max())

    import matplotlib.figure as mfig
    cap = {}
    real_savefig = mfig.Figure.savefig

    def _capture(self, *args, **kwargs):
        cap["texts"] = _figure_texts(self)
        cap["panels"] = len(self.axes)
        cap["lines"] = [len(ax.lines) for ax in self.axes]
        cap["ydata"] = [{ln.get_label(): np.asarray(ln.get_ydata(), dtype=float)
                         for ln in ax.lines} for ax in self.axes]
        cap["secondary"] = [_secondary_axes(ax) for ax in self.axes]
        return real_savefig(self, *args, **kwargs)

    monkeypatch.setattr(mfig.Figure, "savefig", _capture)

    show_r = (1, 2)
    out_png = tmp_path / "figs" / "subset_jsd_vs_full.png"
    mod.plot_subset_jsd(rows, h_ref, edges, blocks, out_png, show_r=show_r)

    assert out_png.is_file()
    assert out_png.stat().st_size > 2000
    assert "texts" in cap, "the figure must be written through Figure.savefig"
    assert cap["panels"] == 4

    joined = " ".join(cap["texts"])
    assert "ln 2" in joined          # the per-marginal bound (Lin 1991)
    assert "first bin" in joined     # where the reference mass sits
    # the first-bin statement carries the PMF's own first-bin mass, per marginal
    for k in _KEYS:
        share = 100.0 * float(ss._to_pmf(h_ref[k])[0])
        assert f"first bin: {share:.1f}% of the reference mass" in joined, (k, share, joined)

    # the marginal panels carry the pool AND each subset in show_r; a panel
    # drawn from the reference alone would hold one line.
    drawn = sum(1 for n in cap["lines"] if n >= 1 + len(show_r))
    assert drawn >= 3, cap["lines"]

    # panel (b): the reference line draws every populated bin, the tiny one
    # included (finite count == positive-bin count), and nothing else
    p_ref = ss._to_pmf(h_ref["rho_third"])
    ref_line = [y for label, y in cap["ydata"][1].items() if "reference" in label]
    assert len(ref_line) == 1
    finite = np.isfinite(ref_line[0])
    assert finite.sum() == np.count_nonzero(p_ref > 0.0)
    assert finite[tiny_bin]

    # panels (b)-(d) carry a secondary axis mapping the bin index to the bin's
    # LEFT edge (bin 1 -> edges[1]), labelled with the marginal's symbol
    assert [len(s) for s in cap["secondary"]] == [0, 1, 1, 1]
    for ax_i, k in zip((1, 2, 3), _KEYS):
        sec = cap["secondary"][ax_i][0]
        fwd, inv = sec._functions
        assert float(fwd(1)) == pytest.approx(float(edges[k][1]), rel=1e-12)
        assert float(fwd(0)) == pytest.approx(float(edges[k][0]), rel=1e-12)
        assert float(inv(float(edges[k][3]))) == pytest.approx(3.0, abs=1e-9)
        label = sec.get_xlabel()
        assert "left edge" in label
        assert {"rho_third": "rho", "s": "$s$", "alpha": "alpha"}[k] in label, label
        assert "rho_third" not in label


def test_bin_edge_helpers_and_out_of_range_share():
    mod = _load_script()
    edges = np.linspace(2.0, 12.0, 6)          # bins of width 2 from 2 to 12
    assert mod.bin_left_edge(edges, 0) == pytest.approx(2.0)
    assert mod.bin_left_edge(edges, 3) == pytest.approx(8.0)
    assert np.allclose(mod.bin_left_edge(edges, np.array([1, 4])), [4.0, 10.0])
    assert mod.bin_index(edges, 8.0) == pytest.approx(3.0)
    assert mod.bin_index(edges, mod.bin_left_edge(edges, 2.5)) == pytest.approx(2.5)
    # 3 of 10 units of weight sit outside [2, 12]: 1 below, 2 above
    block = {"s": np.array([0.0, 2.0, 5.0, 12.0, 13.0]),
             "weights": np.array([1.0, 3.0, 3.0, 1.0, 2.0])}
    assert mod.out_of_range_share(block, edges, "s") == pytest.approx(30.0)
    # unit weights when the block carries none
    block2 = {"s": np.array([1.0, 5.0, 6.0, 7.0])}
    assert mod.out_of_range_share(block2, edges, "s") == pytest.approx(25.0)


def test_write_subset_table_carries_the_members_and_the_divergence(tmp_path):
    mod = _load_script()
    blocks = _synthetic_blocks()
    points = _pool_points()
    h_ref, edges = ss.build_reference_histograms(blocks)
    rows = _plan_rows(points, blocks, h_ref, edges)

    path = tmp_path / "subset_table.csv"
    mod.write_subset_table(rows, path)

    text = path.read_text()
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert len(parsed) == len(rows)
    header = list(parsed[0].keys())
    for key in _ROW_KEYS:
        assert key in header, f"CSV header is missing {key!r}: {header}"
    assert [int(rec["r"]) for rec in parsed] == [row["r"] for row in rows]

    for rec, row in zip(parsed, rows):
        assert float(rec["jsd"]) == pytest.approx(row["jsd"], rel=1e-6)
        assert float(rec["jsd_ledger"]) == pytest.approx(row["jsd_ledger"],
                                                         rel=1e-6)
        for col, key in (("jsd_rho", "jsd_rho"), ("jsd_s", "jsd_s"),
                         ("jsd_alpha", "jsd_alpha")):
            assert float(rec[col]) == pytest.approx(row[key], rel=1e-6)
        assert int(rec["n_ae"]) == row["n_ae"]
        assert int(rec["n_bh76"]) == row["n_bh76"]
        assert int(rec["n_ip13"]) == row["n_ip13"]
        # list-valued columns are joined by ';' so the report can split them
        for col in ("point_names", "point_kinds", "species"):
            assert [c.strip() for c in rec[col].split(";")] == list(row[col])
        assert [int(c) for c in rec["chosen_indices"].split(";")] == list(row["chosen_indices"])


# ===========================================================================
# T5: main
# ===========================================================================

def test_main_writes_the_figure_and_the_table(tmp_path, monkeypatch):
    mod = _load_script()

    cache = tmp_path / "subset_descriptors"
    cache.mkdir()
    sp = {
        ("H2O", 0, 0): _write_species_cache(cache / "H2O_c0_s0_H2O.npz", 140,
                                            seed=21, mean=0.70),
        ("H", 0, 1): _write_species_cache(cache / "H_c0_s1_H.npz", 90,
                                          seed=22, mean=1.20),
        ("H2", 0, 0): _write_species_cache(cache / "H2_c0_s0_H2.npz", 120,
                                           seed=23, mean=0.95),
        ("Li", 0, 1): _write_species_cache(cache / "Li_c0_s1_Li.npz", 110,
                                           seed=24, mean=1.80),
        ("Li+", 1, 0): _write_species_cache(cache / "Liplus_c1_s0_Li.npz", 80,
                                            seed=25, mean=2.30),
    }
    points = [
        _StubPoint("H2O", "ae",
                   [_atoms("H2O", "H2O", 0, 0), _atoms("H", "H", 0, 1)]),
        _StubPoint("H2", "ae",
                   [_atoms("H2", "H2", 0, 0), _atoms("H", "H", 0, 1)]),
        _StubPoint("Li_IP", "ip13",
                   [_atoms("Li", "Li", 0, 1), _atoms("Li", "Li+", 1, 0)]),
    ]
    monkeypatch.setattr(mod, "build_dfs_pool_points", lambda *a, **k: points)

    blocks = [
        _concat_arrays([sp[("H2O", 0, 0)], sp[("H", 0, 1)]]),
        _concat_arrays([sp[("H2", 0, 0)], sp[("H", 0, 1)]]),
        _concat_arrays([sp[("Li", 0, 1)], sp[("Li+", 1, 0)]]),
    ]
    h_ref, edges = ss.build_reference_histograms(blocks)

    ref_path = tmp_path / "reference.npz"
    np.savez(ref_path,
             h_ref_rho=h_ref["rho_third"], e_rho=edges["rho_third"],
             h_ref_s=h_ref["s"], e_s=edges["s"],
             h_ref_alpha=h_ref["alpha"], e_alpha=edges["alpha"])

    plan = {1: (0,), 2: (0, 2)}
    jsd = {r: _expected_jsd(h_ref, edges, blocks, idx)[0]
           for r, idx in plan.items()}
    assert all(v > 1e-6 for v in jsd.values())
    ledger = {
        f"jsd/{r}": {
            "chosen_indices": list(idx),
            "metric_value": jsd[r],
            "point_names": [points[i].name for i in idx],
            "point_kinds": [points[i].kind for i in idx],
            "tag": f"bin{r:02d}",
        }
        for r, idx in plan.items()
    }
    # The shipped ledger holds both metrics; the l2 entries must not be read as
    # jsd values (their metric_value would fail the recomputation guard).
    ledger["l2/1"] = {"chosen_indices": [1], "metric_value": 9.9,
                      "point_names": ["H2"], "point_kinds": ["ae"],
                      "tag": "bin01"}
    ledger_path = tmp_path / "subset_index_log.json"
    ledger_path.write_text(json.dumps(ledger, indent=2))

    # the npz key mapping the reference file uses
    got_h, got_edges = mod.load_reference(ref_path)
    for k in _KEYS:
        assert np.allclose(got_h[k], h_ref[k])
        assert np.allclose(got_edges[k], edges[k])
    entries = mod.load_ledger(ledger_path, metric="jsd")
    assert sorted(entries) == [1, 2]

    outdir = tmp_path / "out"
    rc = mod.main(["--ledger", str(ledger_path),
                   "--reference", str(ref_path),
                   "--descriptors", str(cache),
                   "--outdir", str(outdir)])
    assert rc in (0, None)

    png = outdir / "subset_jsd_vs_full.png"
    csv_path = outdir / "subset_table.csv"
    assert png.is_file() and png.stat().st_size > 2000
    assert csv_path.is_file()

    parsed = list(csv.DictReader(io.StringIO(csv_path.read_text())))
    assert [int(rec["r"]) for rec in parsed] == [1, 2]
    for rec in parsed:
        assert float(rec["jsd"]) == pytest.approx(jsd[int(rec["r"])], rel=1e-6)
    assert "H2O" in parsed[0]["point_names"]
    assert "Li+" in parsed[1]["species"]
