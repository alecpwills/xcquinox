"""Failing tests for ``arm_vs_size_density.py`` (the 25-cycle comparison).

The script reads a merged family view, pairs every protocol arm with the size cell of the
same stored architecture and subset, and reports the per-species held-out density-RMSE
ratio arm/size, a caveat naming each cell's SCF cycle budget and converged count, a
figure, a CSV and the LaTeX rows of the deck table.

The module does not exist yet, so every test here fails on its own ``ImportError`` from
:func:`_load_script` rather than the file collapsing into one collection error: the RED
state names each requirement separately, as ``test_report_tables.py`` does.

The synthetic view built by :func:`_build_view` is read through the figure suite's own
held-out density reader, so the fixture has to satisfy that reader:

* no ``train_metadata.json`` is written in any spec directory, so the alias repair
  (``_spec_alias_names``) finds nothing to drop and every row reaches the caller;
* no ``sliced_eval.json`` marker exists in any channel directory and every
  ``eval_metadata.json`` carries ``species_slice: null``, the two marks
  ``assert_channel_not_sliced`` refuses a channel on;
* ``density_rmse_pbe`` is identical for a species in every cell (and identical across the
  ``h2``/``H2`` twins), because a model-free reference that disagrees across specs by more
  than five percent is dropped from every cell by the reader's consistency repair;
* every species carries a positive ``density_rmse_pbe``, because the twin-collapse the
  script mirrors (``cell_species_density_ratios``) drops a species without one;
* the rows of each ``per_molecule.json`` are written in an order that is NOT the ratio
  order, so a sort that omits the ratio key is visible.

Mutation coverage (the design page's list) is named above each test.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pytest  # noqa: E402

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "arm_vs_size_density.py"
_MODNAME = "arm_vs_size_density"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load_script():
    """Load the module under test from its path, fresh on every call."""
    if not _SCRIPT.exists():
        raise ImportError(f"{_SCRIPT} does not exist: the comparison script has not been "
                          "written")
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


def _suite():
    """The figure suite, loaded from its path once (the loader ``report_tables._suite``
    uses, so the script and the test share one module object)."""
    cached = sys.modules.get("make_ablation_arch_figure")
    if isinstance(cached, types.ModuleType) and \
            callable(getattr(cached, "collect_holdout_density_rows", None)):
        return cached
    path = _HERE / "make_ablation_arch_figure.py"
    spec = importlib.util.spec_from_file_location("make_ablation_arch_figure", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["make_ablation_arch_figure"] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except BaseException:
        sys.modules.pop("make_ablation_arch_figure", None)
        raise
    return mod


# ---------------------------------------------------------------------------
# the synthetic view: three size cells, four arms, one of them unpaired
# ---------------------------------------------------------------------------

CHANNEL = "val_best"
EVAL_SUBDIR = "eval_holdout_val_best"

#: (index, stored arch, subset_size, protocol, solver) -- the manifest cells. Indices 0
#: (size at 1) and 5 (the arm at 15) have no partner and get no spec directory.
MANIFEST_SPECS = [
    (0, "medium", 1, None, "full_3"),
    (1, "medium", 7, None, "full_3"),
    (2, "medium", 12, None, "full_3"),
    (3, "medium", 7, "25 cycles", "full_25"),
    (4, "medium", 12, "25 cycles", "full_25"),
    (5, "medium", 15, "25 cycles", "full_25"),
    (6, "medium", 7, "dpyscf parity", "full_25"),
]

#: model-free PBE density RMSE per casefolded species, one value for the whole view
PBE_RMSE = {
    "h2": 5.0e-4,
    "ch4": 9.0e-4,
    "c2": 7.0e-4,
    "nh3": 3.0e-4,      # NN 6e-4 in every cell: ratio 2 everywhere, the fixture's recurring tail
    "co": 2.0e-4,
    "lonely": 3.0e-4,
    "arm_only": 4.0e-4,
    "hcn": 8.0e-4,
    "zero": 1.0e-4,
    "h2o": 5.0e-4,      # the supervised row of cell 1, dropped by the reader
}

#: per spec index: (molecule, density_rmse, density_eps_l1, cycles_run, scf_converged),
#: in file order. "h2"/"H2" are the case twins; their NN errors differ within every cell, and
#: between the size cell and the 25-cycle arm at r = 7 (size (2, 4) against arm (0.5, 2.5)), so
#: a first-wins collapse and a mean collapse give different ratios. "lonely" sits in the
#: size cell of r = 7 only, "arm_only" and "hcn" in the arm cell of r = 12 only. No two
#: cells carry the same row count, so each converged count has its own denominator.
CELL_ROWS = {
    1: [("co", 1.0e-4, 0.050, 3, False),
        ("nh3", 6.0e-4, 0.040, 3, False),
        ("h2", 2.0e-4, 0.010, 3, True),
        ("ch4", 8.0e-4, 0.020, 3, False),
        ("H2", 4.0e-4, 0.014, 2, False),
        ("c2", 4.0e-4, 0.030, 3, False),
        ("lonely", 7.0e-4, None, 3, False),
        ("h2o", 5.0e-4, 0.030, 3, True, True)],
    2: [("nh3", 6.0e-4, 0.040, 3, False),
        ("ch4", 8.0e-4, 0.020, 2, False),
        ("c2", 4.0e-4, 0.030, 3, False),
        ("zero", 0.0, 0.020, 3, False)],
    3: [("nh3", 6.0e-4, 0.038, 23, True),
        ("co", 8.0e-4, 0.048, 25, False),
        ("c2", 3.0e-4, 0.028, 22, True),
        ("h2", 0.5e-4, 0.004, 25, True),
        ("H2", 2.5e-4, 0.012, 19, True),
        ("ch4", 5.0e-4, 0.018, 21, True)],
    4: [("nh3", 6.0e-4, 0.038, 24, True),
        ("ch4", 16.0e-4, 0.044, 25, False),
        ("c2", 2.0e-4, 0.026, 25, True),
        ("arm_only", 5.0e-4, 0.050, 25, True),
        ("hcn", 9.0e-4, 0.055, 25, True),
        ("zero", 1.0e-4, 0.020, 25, True)],
    6: [("ch4", 6.0e-4, 0.016, 25, True),
        ("h2", 2.0e-4, 0.009, 24, True),
        ("H2", 4.0e-4, 0.011, 20, None),
        ("c2", 1.0e-4, 0.026, 25, True)],
}

#: solver_config.max_cycles per spec index: the size cells ran the training budget, the
#: arms the 25-cycle one
MAX_CYCLES = {1: 3, 2: 3, 3: 25, 4: 25, 6: 25}

#: the caveat phrase per spec index, by hand from CELL_ROWS over the twin-collapsed species:
#: a species converged when every twin did (h2 of cell 1: True and False), unknown when a
#: twin lacks the flag (H2 of cell 6), the supervised h2o of cell 1 dropped by the reader
CAVEAT_PHRASES = {
    1: "size cell at r = 7: max_cycles 3, 0 of 6 converged",
    2: "size cell at r = 12: max_cycles 3, 0 of 4 converged",
    3: "25 cycles at r = 7: max_cycles 25, 4 of 5 converged",
    4: "25 cycles at r = 12: max_cycles 25, 5 of 6 converged",
    6: "dpyscf parity at r = 7: max_cycles 25, 2 of 2 converged, 1 unknown",
}

# The ratios the collapse produces, computed by hand from CELL_ROWS.
#   r = 7, "25 cycles" (size 1 against arm 3):
#     h2  (2.0 + 4.0) / 2 = 3.0 against (0.5 + 2.5) / 2 = 1.5 -> 0.5
#     ch4 8.0 against 5.0 -> 0.625
#     c2  4.0 against 3.0 -> 0.75
#     nh3 6.0 against 6.0 -> 1.0
#     co  1.0 against 8.0 -> 8.0
#     "lonely" is dropped (size cell only)
#   r = 7, "dpyscf parity" (size 1 against arm 6):
#     c2 4.0 against 1.0 -> 0.25;  ch4 8.0 against 6.0 -> 0.75
#     h2  (2.0 + 4.0) / 2 = 3.0 against (2.0 + 4.0) / 2 = 3.0 -> exactly 1.0: the same twin
#         values on both sides, so the two means are the same floating-point number
#     nh3, co and "lonely" are dropped (size cell only)
#   r = 12, "25 cycles" (size 2 against arm 4):
#     c2 4.0 against 2.0 -> 0.5;  nh3 6.0 against 6.0 -> 1.0;  ch4 8.0 against 16.0 -> 2.0
#     "arm_only" and "hcn" are dropped (arm cell only)
RATIOS = {
    (7, "25 cycles"): {"h2": 0.5, "ch4": 0.625, "c2": 0.75, "nh3": 1.0, "co": 8.0},
    (7, "dpyscf parity"): {"c2": 0.25, "ch4": 0.75, "h2": 1.0},
    (12, "25 cycles"): {"c2": 0.5, "nh3": 1.0, "ch4": 2.0},
}

# n, n_below_1 (strict) and the median of each group, by hand from RATIOS. The means are
# 2.175, 0.6667 and 1.1667, none of them the median; counting with <= would give 4, 3 and 2.
SUMMARY = {
    (7, "25 cycles"): {"n": 5, "n_below_1": 3, "median_ratio": 0.75},
    (7, "dpyscf parity"): {"n": 3, "n_below_1": 2, "median_ratio": 0.75},
    (12, "25 cycles"): {"n": 3, "n_below_1": 1, "median_ratio": 1.0},
}

CSV_HEADER = ("subset_size,arm,species,rmse_size,rmse_arm,ratio,eps_size,eps_arm,"
              "cycles_size,cycles_arm,converged_size,converged_arm,tail")

#: (subset_size, arm, species) in the order write_csv must emit them
CSV_ORDER = [
    (7, "25 cycles", "h2"), (7, "25 cycles", "ch4"), (7, "25 cycles", "c2"),
    (7, "25 cycles", "nh3"), (7, "25 cycles", "co"),
    (7, "dpyscf parity", "c2"), (7, "dpyscf parity", "ch4"), (7, "dpyscf parity", "h2"),
    (12, "25 cycles", "c2"), (12, "25 cycles", "nh3"), (12, "25 cycles", "ch4"),
]


def _pm_row(name, rmse, eps, cycles, converged, supervised=False):
    """One ``per_molecule.json`` record, the held-out per-species schema; ``supervised``
    is the eval's own from_training_subset flag, which the reader drops."""
    return {
        "molecule": name,
        "E_total_nn": -76.0 - 0.25 * len(name),
        "E_pbe": -76.0 - 0.25 * len(name) + 0.05,
        "density_rmse": rmse,
        "density_rmse_pbe": PBE_RMSE[name.casefold()],
        "density_eps_l1": eps,
        "cycles_run": cycles,
        "scf_converged": converged,
        "from_training_subset": supervised,
    }


def _build_view(tmp_path):
    """A merged family view: ``manifest.json`` plus one channel directory per paired
    spec. No ``train_metadata.json`` and no ``sliced_eval.json`` are written (see the
    module docstring); ``resolved_config.yaml`` is absent, so no run-level protocol tag
    is appended to the shown architecture names."""
    run = Path(tmp_path) / "run_20260908T153908Z"
    (run / "checkpoints").mkdir(parents=True)
    specs = []
    for idx, arch, ss, protocol, solver in MANIFEST_SPECS:
        cell = {"arch": arch, "loss": "L5_gradnorm_vxc_step7", "metric": "jsd",
                "solver": solver, "subset_size": ss}
        if protocol is not None:
            cell["protocol"] = protocol
        specs.append({"index": idx, "cell": cell,
                      "category": "dfs6311_grid3_v7g1_size" if protocol is None
                      else "dfs6311_grid3_v7g1_arm",
                      "run": "run_20260902T145245Z",
                      "spec_file": f"spec_{idx:04d}.spec"})
    (run / "manifest.json").write_text(
        json.dumps({"n_specs": len(specs), "specs": specs}, indent=1))
    for idx, rows in CELL_ROWS.items():
        ch = run / "checkpoints" / f"spec_{idx:04d}" / EVAL_SUBDIR
        ch.mkdir(parents=True)
        (ch / "per_molecule.json").write_text(
            json.dumps([_pm_row(*r) for r in rows], indent=1))
        (ch / "eval_metadata.json").write_text(json.dumps({
            "channel": EVAL_SUBDIR,
            "channel_override": None,
            "model": "model_val_best.eqx",
            "n_species": len(rows),
            "solver_config": {"backend": "manual", "mode": "full", "conv_tol": 1e-6,
                              "density_fit": True, "max_cycles": MAX_CYCLES[idx],
                              "mixer_name": "decaying_linear", "seed_source": "pbe"},
            "species_slice": None,
        }, indent=1))
    return run


def _hd_rows(run):
    """The suite's held-out density rows for the synthetic view."""
    return _suite().collect_holdout_density_rows(Path(run), EVAL_SUBDIR)


# ---------------------------------------------------------------------------
# the two pool CSVs of a figure set
# ---------------------------------------------------------------------------

# the real header of both files (verified against
# figures_dfs_step7_v7_family_val_best/holdout_by_pool_3x3_eps.csv and
# .../holdout_by_pool_3x3.csv, which share it)
POOL_HEADER = ("leg,arch,arch_stored,subset_size,n_reactions,n_density_species,E_kcalmol,"
               "D_rmse,gamma,gammaD_kcalmol,ED_kcalmol,E_pbe_kcalmol,D_pbe_rmse,"
               "ED_pbe_kcalmol,beats_pbe,E_scan_kcalmol,D_scan_rmse,ED_scan_kcalmol,"
               "beats_scan,ED_pbe_cell_kcalmol,ED_scan_cell_kcalmol,n_reactions_slice,"
               "D_insample_rmse,n_insample_species")
POOL_FIELDS = POOL_HEADER.split(",")

# the eps file names its legs with the DFS-units suffix, the grid file does not
COMBINED_EPS = "combined_wtmad2_eps_gamma_dfs"
BH76_EPS = "bh76_wtmad2_eps_gamma_dfs"
COMBINED_GRID = "combined_wtmad2"
BH76_GRID = "bh76_wtmad2"

E_PBE = 8.915174138633205          # the eps leg's model-free anchor, one value per leg
D_PBE_EPS = 0.009229535775259713
D_PBE_GRID = 0.00020200017710636945


def _pool_row(**kw):
    """One CSV data line: the named fields, the untouched ones blank."""
    unknown = set(kw) - set(POOL_FIELDS)
    assert not unknown, f"the fixture names fields the real header does not carry: {unknown}"
    values = {name: "" for name in POOL_FIELDS}
    values.update({k: str(v) for k, v in kw.items()})
    return ",".join(f'"{values[n]}"' if "," in values[n] else values[n] for n in POOL_FIELDS)


#: (arch, arch_stored, subset, E_kcalmol, eps, ED_kcalmol, ED_pbe_cell, grid D_rmse).
#: The first three rows are the paired cell of the deck table; the tagged rows come
#: first in the file, so a reader that keeps the file order gets the size row last.
_TABLE_CELLS = [
    ("deep_3x16 [dpyscf parity]", "medium", 7,
     7.444123456, 0.0090294567, 8.459123456, 9.4123456, 0.000162123),
    ("deep_3x16 [25 cycles]", "medium", 7,
     6.917123456, 0.0097234567, 8.355123456, 9.3552345, 0.00017512345),
    ("deep_3x16", "medium", 7,
     6.443359742702014, 0.010165365077236168, 8.134181768597308, 9.48819003489076,
     0.0001934022414146754),
    # an untagged medium cell at a subset with no arm: not a paired cell
    ("deep_3x16", "medium", 12,
     7.412669319292577, 0.010473536695909851, 8.972088232246204, 9.560469003646006,
     0.00021234567),
    # another stored architecture at a paired subset
    ("deep_2x8", "shallow", 7,
     12.512345678, 0.0155123456, 15.012345678, 9.481234567, 0.00031234567),
    # the stored key "deep_3x16" is the shown name of the stored key "medium": a reader
    # that pairs on the shown name instead of the stored key reaches for this row
    ("deep0_3x16", "deep_3x16", 7,
     9.912345678, 0.0121234567, 11.012345678, 9.491234567, 0.00025234567),
]

#: the rows latex_rows must return, hand-written from _TABLE_CELLS with the page's
#: formats: the size row first, then the arms by tag, then the PBE row; underscores
#: escaped, the protocol tag kept, two decimals for WTMAD-2 and ED, five for eps and the
#: eps PBE reference, ``%.2e`` for the grid RMSE, "--" for the columns PBE has none of.
LATEX_ROWS = [
    "deep\\_3x16 & 7 & 6.44 & 0.01017 & 1.93e-04 & 8.13 & 9.49",
    "deep\\_3x16 [25 cycles] & 7 & 6.92 & 0.00972 & 1.75e-04 & 8.36 & 9.36",
    "deep\\_3x16 [dpyscf parity] & 7 & 7.44 & 0.00903 & 1.62e-04 & 8.46 & 9.41",
    "PBE & -- & 8.92 & 0.00923 & 2.02e-04 & -- & --",
]


def _write_family_dir(tmp_path, name="figures_dfs_step7_v7_family_val_best", cells=None):
    """The two pool CSVs of a figure set. Every cell appears in both files, on the
    combined leg each file names in its own way, and again on a BH76 leg whose numbers
    differ, so a reader that takes the wrong leg is visible."""
    cells = _TABLE_CELLS if cells is None else cells
    fam = Path(tmp_path) / name
    fam.mkdir(parents=True, exist_ok=True)
    eps_lines, grid_lines = [POOL_HEADER], [POOL_HEADER]
    for leg, e_pbe, d_pbe in ((COMBINED_EPS, E_PBE, D_PBE_EPS),
                              (BH76_EPS, 22.114068697861534, 0.007771234567)):
        for arch, stored, ss, e, eps, ed, cap, _grid in cells:
            bump = 0.0 if leg == COMBINED_EPS else 13.5
            eps_lines.append(_pool_row(
                leg=leg, arch=arch, arch_stored=stored, subset_size=ss,
                n_reactions=175, n_density_species=181, E_kcalmol=e + bump,
                D_rmse=eps if leg == COMBINED_EPS else eps + 0.004,
                gamma=1084.87, ED_kcalmol=ed + bump, E_pbe_kcalmol=e_pbe,
                D_pbe_rmse=d_pbe, ED_pbe_kcalmol=9.432182242030425, beats_pbe=True,
                ED_pbe_cell_kcalmol=cap + bump, n_reactions_slice=175))
    for leg, d_pbe in ((COMBINED_GRID, D_PBE_GRID), (BH76_GRID, 0.00031200017710636945)):
        for arch, stored, ss, e, _eps, ed, cap, grid in cells:
            bump = 100.0 if leg == COMBINED_GRID else 113.5
            grid_lines.append(_pool_row(
                leg=leg, arch=arch, arch_stored=stored, subset_size=ss,
                n_reactions=175, n_density_species=181, E_kcalmol=e + bump,
                D_rmse=grid if leg == COMBINED_GRID else grid + 5e-5,
                gamma=109475.4916289835, ED_kcalmol=ed + bump,
                E_pbe_kcalmol=22.114068697861534, D_pbe_rmse=d_pbe,
                ED_pbe_kcalmol=22.114068697861534, beats_pbe=True,
                ED_pbe_cell_kcalmol=cap + bump, n_reactions_slice=175))
    (fam / "holdout_by_pool_3x3_eps.csv").write_text("\n".join(eps_lines) + "\n")
    (fam / "holdout_by_pool_3x3.csv").write_text("\n".join(grid_lines) + "\n")
    return fam


def _png_ok(path):
    """True when the file exists and carries the PNG signature."""
    p = Path(path)
    return p.is_file() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _by_key(rows):
    """``{(subset_size, arm, species): row}`` over the script's species rows."""
    return {(int(r["subset_size"]), r["arm"], r["species"]): r for r in rows}


# ---------------------------------------------------------------------------
# 1. the pairing
# ---------------------------------------------------------------------------

def test_pairs_from_manifest_are_ordered_and_the_unpaired_arm_is_skipped(tmp_path, capsys):
    """Every arm is paired with the size cell of the same stored architecture and subset,
    the pairs come out in (subset_size, protocol) order, and the arm at r = 15, which has
    no size partner, is skipped and named on stdout."""
    mod = _load_script()
    run = _build_view(tmp_path)
    capsys.readouterr()
    pairs = mod.pairs_from_manifest(run)
    out = capsys.readouterr().out
    assert [(p.arm_idx, p.size_idx, p.subset_size, p.arch_stored, p.protocol)
            for p in pairs] == [
        (3, 1, 7, "medium", "25 cycles"),
        (6, 1, 7, "medium", "dpyscf parity"),
        (4, 2, 12, "medium", "25 cycles"),
    ]
    skipped = [ln for ln in out.splitlines() if "15" in ln and "25 cycles" in ln]
    assert skipped, "the arm at r = 15 has no size partner and must be reported by name"


# ---------------------------------------------------------------------------
# 2. the twin collapse, the ratio direction and the unmatched species
# kills: twins not collapsed; ratio inverted (size/arm); the unmatched species kept
# ---------------------------------------------------------------------------

def test_species_rows_collapse_case_twins_and_drop_unmatched_species(tmp_path, capsys):
    """The case twins ``h2`` and ``H2`` reduce to ONE species row whose errors are the
    twin means, ``cycles_*`` the twin maximum and ``converged_*`` the twin conjunction;
    the ratio is arm over size; a species that only one cell of a pair carries is dropped
    and named on stdout, never kept with a non-finite ratio."""
    mod = _load_script()
    run = _build_view(tmp_path)
    pairs = mod.pairs_from_manifest(run)
    hd = _hd_rows(run)
    capsys.readouterr()
    rows = mod.species_rows(hd, pairs, {"co"})
    out = capsys.readouterr().out

    assert len(rows) == 11, "five, three and three scored species over the three pairs"
    assert [r for r in rows if r["species"] == "H2"] == [], "the twin key is casefolded"
    twins = [r for r in rows
             if r["species"] == "h2" and int(r["subset_size"]) == 7
             and r["arm"] == "25 cycles"]
    assert len(twins) == 1, "h2 and H2 are one species, not two rows"
    t = twins[0]
    # size (2.0 + 4.0) / 2 = 3.0e-4 against arm (0.5 + 2.5) / 2 = 1.5e-4
    assert t["rmse_size"] == pytest.approx(3.0e-4)
    assert t["rmse_arm"] == pytest.approx(1.5e-4)
    assert t["ratio"] == pytest.approx(0.5)     # arm / size, not 2.0
    # eps (0.010 + 0.014) / 2 = 0.012 against (0.004 + 0.012) / 2 = 0.008
    assert t["eps_size"] == pytest.approx(0.012)
    assert t["eps_arm"] == pytest.approx(0.008)
    assert int(t["cycles_size"]) == 3 and int(t["cycles_arm"]) == 25   # twin maxima
    assert bool(t["converged_size"]) is False    # True and False
    assert bool(t["converged_arm"]) is True      # True and True

    # a second, twin-free pair pins the ratio direction again
    co = _by_key(rows)[(7, "25 cycles", "co")]
    assert co["ratio"] == pytest.approx(8.0)     # 8.0e-4 arm over 1.0e-4 size

    assert all(r["species"] != "lonely" for r in rows), "size-only species are dropped"
    assert all(r["species"] != "arm_only" for r in rows), "arm-only species are dropped"
    assert all(math.isfinite(float(r["ratio"])) for r in rows)
    assert "lonely" in out and "arm_only" in out
    # the species with a zero size-cell RMSE has no ratio: dropped and named, never divided
    assert all(r["species"] != "zero" for r in rows)
    assert "zero" in out
    # the supervised row never reaches the rows (the reader drops it)
    assert all(r["species"] != "h2o" for r in rows)
    # a twin without the convergence flag leaves the species unknown, not False
    assert _by_key(rows)[(7, "dpyscf parity", "h2")]["converged_arm"] is None


# ---------------------------------------------------------------------------
# 3. the summaries
# kills: median replaced by mean; n_below_1 counted with <=
# ---------------------------------------------------------------------------

def test_summaries_count_below_one_strictly_and_report_the_median(tmp_path):
    """Per (subset_size, arm): the species count, the count of ratios strictly below one
    (a ratio of exactly 1.0 sits in every group, so ``<=`` overcounts each of them) and
    the median (the three group means are 2.175, 0.667 and 1.167, none of them a median)."""
    mod = _load_script()
    run = _build_view(tmp_path)
    pairs = mod.pairs_from_manifest(run)
    rows = mod.species_rows(_hd_rows(run), pairs, set())
    got = mod.summarize(rows)
    assert set(got) == set(SUMMARY)
    for key, want in SUMMARY.items():
        assert int(got[key]["n"]) == want["n"], key
        assert int(got[key]["n_below_1"]) == want["n_below_1"], key
        assert float(got[key]["median_ratio"]) == pytest.approx(want["median_ratio"]), key


# ---------------------------------------------------------------------------
# 4. the tail flag
# ---------------------------------------------------------------------------

def test_tail_flag_marks_only_the_species_in_the_tail_set(tmp_path):
    """``tail`` is 1 exactly for the casefolded species handed in, 0 for every other,
    in every pair the species appears in."""
    mod = _load_script()
    run = _build_view(tmp_path)
    pairs = mod.pairs_from_manifest(run)
    rows = mod.species_rows(_hd_rows(run), pairs, {"co", "ch4"})
    flagged = {(int(r["subset_size"]), r["arm"], r["species"]) for r in rows
               if int(r["tail"]) == 1}
    assert flagged == {(7, "25 cycles", "co"), (7, "25 cycles", "ch4"),
                       (7, "dpyscf parity", "ch4"), (12, "25 cycles", "ch4")}
    assert all(int(r["tail"]) == 0 for r in rows
               if r["species"] not in ("co", "ch4"))
    # the same rows with an empty tail set carry no flag at all
    plain = mod.species_rows(_hd_rows(run), pairs, set())
    assert all(int(r["tail"]) == 0 for r in plain)


# ---------------------------------------------------------------------------
# 5. the CSV
# kills: a sort key without the ratio
# ---------------------------------------------------------------------------

def test_csv_header_is_exact_and_rows_sort_by_subset_arm_ratio(tmp_path):
    """The CSV carries the thirteen columns of the page in order, and its rows are
    ordered by (subset_size, arm, ratio). Neither the file order of the fixture's
    ``per_molecule.json`` records nor the order the pairs are built in is the ratio
    order, so a sort that omits the ratio is visible here."""
    mod = _load_script()
    run = _build_view(tmp_path)
    pairs = mod.pairs_from_manifest(run)
    rows = mod.species_rows(_hd_rows(run), pairs, {"co"})
    path = Path(tmp_path) / "arm_vs_size_density.csv"
    mod.write_csv(rows, path)
    lines = path.read_text().splitlines()
    assert lines[0] == CSV_HEADER
    with path.open() as f:
        got = list(csv.DictReader(f))
    assert [(int(r["subset_size"]), r["arm"], r["species"]) for r in got] == CSV_ORDER
    assert [float(r["ratio"]) for r in got] == pytest.approx(
        [RATIOS[(s, a)][sp] for s, a, sp in CSV_ORDER])


# ---------------------------------------------------------------------------
# 6. the caveat and the figure
# kills: a caveat without the converged counts
# ---------------------------------------------------------------------------

def test_cycle_caveat_carries_the_counts_and_the_figure_shows_them(tmp_path):
    """The caveat names the channel, each cell's ``solver_config.max_cycles`` and each
    cell's converged count of its row count, so a reader cannot mistake the arms'
    advantage for a training-protocol effect alone. The figure carries one panel per
    subset, a legend line per arm with that arm's counts and median, and the caveat as a
    footer text; it returns its Figure so those can be read back."""
    mod = _load_script()
    run = _build_view(tmp_path)
    pairs = mod.pairs_from_manifest(run)
    rows = mod.species_rows(_hd_rows(run), pairs, {"co"})
    summaries = mod.summarize(rows)

    caveat = mod.cycle_caveat(run, pairs, EVAL_SUBDIR)
    assert CHANNEL in caveat
    assert "max_cycles 3" in caveat
    assert "25" in caveat
    for idx, phrase in CAVEAT_PHRASES.items():
        assert phrase in caveat, f"spec_{idx:04d}: {phrase!r} missing from {caveat!r}"
    assert "of 7" not in caveat and "of 8" not in caveat, "rows counted instead of species"

    png = Path(tmp_path) / "arm_vs_size_density.png"
    fig = mod.make_figure(rows, summaries, caveat, png)
    assert _png_ok(png)
    assert fig is not None, "make_figure must return its Figure for the caller to inspect"

    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert {"r = 7", "r = 12"} <= titles

    texts = [t.get_text() for ax in fig.axes if ax.get_legend() is not None
             for t in ax.get_legend().get_texts()]
    texts += [t.get_text() for leg in getattr(fig, "legends", []) for t in leg.get_texts()]
    for want in ("25 cycles: 3 of 5 below 1, median 0.75",
                 "dpyscf parity: 2 of 3 below 1, median 0.75",
                 "25 cycles: 1 of 3 below 1, median 1.00"):
        assert want in texts, f"legend line missing: {want}"
    assert len([t for t in texts if "below 1" in t]) == 3

    assert any(caveat in t.get_text() for t in fig.texts), "the caveat is the figure footer"


# ---------------------------------------------------------------------------
# 7. the LaTeX rows
# kills: pairing on the shown name instead of the stored key
# ---------------------------------------------------------------------------

def test_latex_rows_pair_on_the_stored_key_and_end_with_the_pbe_row(tmp_path):
    """The deck table: the size row of a paired cell first, then its arms by tag, then
    one PBE row. The energy, the eps column and the two ED columns come from the
    combined DFS-units leg of the eps file, the grid RMSE from the combined leg of the
    grid file (whose legs carry no suffix), and the PBE row from the two files' own PBE
    columns. A cell of another stored architecture, an untagged cell at a subset with no
    arm, and every row of the BH76 leg stay out."""
    mod = _load_script()
    fam = _write_family_dir(tmp_path)
    assert mod.latex_rows(fam) == LATEX_ROWS


# ---------------------------------------------------------------------------
# 8. the command line
# ---------------------------------------------------------------------------

def test_cli_writes_the_csv_and_png_into_the_out_dir(tmp_path, capsys):
    """``main`` reads the view's ``eval_holdout_<channel>`` rows, writes
    ``arm_vs_size_density.csv`` and ``arm_vs_size_density.png`` into the output directory
    (not into the family directory it read the pool CSVs from) and prints the summaries,
    the caveat and the LaTeX rows."""
    mod = _load_script()
    run = _build_view(tmp_path)
    fam = _write_family_dir(tmp_path, cells=_TABLE_CELLS + [
        ("deep_3x16 [25 cycles]", "medium", 12,
         7.166123456, 0.0098284567, 8.571123456, 9.5601234, 0.00019912345)])
    out_dir = Path(tmp_path) / "out"
    rc = mod.main(["--run-dir", str(run), "--eval-channel", CHANNEL,
                   "--family-dir", str(fam), "--out-dir", str(out_dir)])
    assert rc in (0, None)
    csv_path = out_dir / "arm_vs_size_density.csv"
    assert csv_path.is_file()
    assert _png_ok(out_dir / "arm_vs_size_density.png")
    assert not (fam / "arm_vs_size_density.csv").exists()
    assert csv_path.read_text().splitlines()[0] == CSV_HEADER

    printed = capsys.readouterr().out
    assert "3 of 5" in printed and "2 of 3" in printed and "1 of 3" in printed
    assert "max_cycles 3" in printed
    assert LATEX_ROWS[-1] in printed
    # the recurring tail the command line computes over the view reaches the CSV: nh3 sits
    # at twice its PBE reference in every cell, no other species is above 1.5 in two cells
    tail_line = [ln for ln in printed.splitlines() if "recurring" in ln]
    assert tail_line and "nh3" in tail_line[0], printed
    with csv_path.open() as f:
        got = list(csv.DictReader(f))
    assert {r["species"] for r in got if int(r["tail"]) == 1} == {"nh3"}
    assert sum(int(r["tail"]) for r in got) == 2, "nh3 is shared by the two 25-cycle pairs"


# ---------------------------------------------------------------------------
# 9. arms the manifest lists but the channel has not evaluated
# kills: the pair filter removed (every species reported as unmatched, the caveat naming an
#        unknown cell); the filter without the finite-NN test (a PBE-only cell kept)
# ---------------------------------------------------------------------------

def test_unevaluated_arms_are_skipped_as_one_line_each_and_left_out_of_the_caveat(
        tmp_path, capsys):
    """A merged view lists every arm the run will evaluate, so an arm without a held-out NN
    evaluation is paired by the manifest and must be dropped before the species join: the
    arm at r = 12 has a channel directory holding PBE-only rows (no NN leg), the arm at
    r = 1 has no channel directory and neither has its size cell. One printed line names
    each skipped pair by spec, no species is reported as unmatched for them, the rows and
    summaries are those of the evaluated pairs alone, and the caveat does not name them."""
    mod = _load_script()
    run = _build_view(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    for idx, ss, protocol in ((7, 12, "dpyscf parity"), (8, 1, "25 cycles")):
        manifest["specs"].append({
            "index": idx,
            "cell": {"arch": "medium", "loss": "L5_gradnorm_vxc_step7", "metric": "jsd",
                     "solver": "full_25", "subset_size": ss, "protocol": protocol},
            "category": "dfs6311_grid3_v7g1_arm", "run": "run_20260902T145245Z",
            "spec_file": f"spec_{idx:04d}.spec"})
    manifest["n_specs"] = len(manifest["specs"])
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # spec 7: a PBE-only re-evaluation, rows with the model-free leg and no NN leg
    ch = run / "checkpoints" / "spec_0007" / EVAL_SUBDIR
    ch.mkdir(parents=True)
    pbe_only = []
    for name in ("nh3", "ch4", "c2"):
        row = _pm_row(name, None, None, None, None)
        pbe_only.append(row)
    (ch / "per_molecule.json").write_text(json.dumps(pbe_only, indent=1))
    (ch / "eval_metadata.json").write_text(json.dumps({
        "channel": EVAL_SUBDIR, "channel_override": None, "model": "model_val_best.eqx",
        "n_species": 3, "solver_config": {"max_cycles": 25}, "species_slice": None},
        indent=1))

    pairs = mod.pairs_from_manifest(run)
    assert (7, 2, 12, "medium", "dpyscf parity") in [tuple(p) for p in pairs]
    assert (8, 0, 1, "medium", "25 cycles") in [tuple(p) for p in pairs]
    hd = _hd_rows(run)
    assert any(r["idx"] == 7 for r in hd), "the PBE-only rows reach the reader's output"
    capsys.readouterr()
    kept = mod.evaluated_pairs(hd, pairs)
    out = capsys.readouterr().out
    assert [tuple(p) for p in kept] == [
        (3, 1, 7, "medium", "25 cycles"),
        (6, 1, 7, "medium", "dpyscf parity"),
        (4, 2, 12, "medium", "25 cycles"),
    ]
    assert len([ln for ln in out.splitlines() if "spec_0007" in ln]) == 1, out
    eight = [ln for ln in out.splitlines() if "spec_0008" in ln]
    assert len(eight) == 1 and "spec_0000" in eight[0], out
    assert "nh3" not in out and "ch4" not in out, "no species list for a skipped pair"

    rows = mod.species_rows(hd, kept, set())
    assert len(rows) == 11
    assert set(mod.summarize(rows)) == set(SUMMARY)
    caveat = mod.cycle_caveat(run, kept, EVAL_SUBDIR, hd_rows=hd)
    assert "dpyscf parity at r = 12" not in caveat
    assert "at r = 1:" not in caveat
    assert "max_cycles unknown" not in caveat and "convergence unknown" not in caveat

    # species_rows on its own also refuses to list species for a pair without NN rows
    capsys.readouterr()
    rows_all = mod.species_rows(hd, pairs, set())
    out = capsys.readouterr().out
    assert len(rows_all) == 11
    for ln in out.splitlines():
        if "spec_0007" in ln or "spec_0008" in ln:
            assert "nh3" not in ln and "ch4" not in ln, ln

    # the command line applies the filter: the summaries are unchanged and no species
    # list is printed for the unevaluated arms
    fam = _write_family_dir(tmp_path)
    capsys.readouterr()
    rc = mod.main(["--run-dir", str(run), "--eval-channel", CHANNEL,
                   "--family-dir", str(fam), "--out-dir", str(Path(tmp_path) / "out2")])
    printed = capsys.readouterr().out
    assert rc in (0, None)
    assert "3 of 5" in printed and "2 of 3" in printed and "1 of 3" in printed
    assert "dpyscf parity at r = 12" not in printed
    assert printed.count("spec_0007") == 1 and printed.count("spec_0008") == 1
