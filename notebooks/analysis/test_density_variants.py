"""Tests for the outlier-free held-out density variants, the tail table, the
twin-consistent published counts and the in-sample parity work (commits A, B
and C of ``spec_part2_minimal.md``).

The module under test is loaded by ``test_make_ablation_arch_figure`` (bound
there as ``fig``); its fixtures are imported rather than copied so the two
files describe one synthetic run layout. Every test here is RED on
f0c7b7865: the names the design defines (``exclude_cf``, ``variant_note``,
``load_t1_table``, ``write_holdout_density_tail_csv``,
``insample_density_by_arch_subset``, ``D_insample_rmse``,
``n_insample_species``) do not exist yet, and the twin-collapse assertions
contradict the current raw-name keying.

The rich fixture (:func:`_rich_run_dir`) is the one the design calls for:
four cells, five held-out density species each, an ``H2``/``h2`` case-twin
pair carrying DIFFERENT NN values (so a row mean and a twin-then-species mean
differ), one species (``CH2O``) present only in the run-level PBE table, and
``pbe_density_errors.json`` whose Eq. 20 eps column is backfilled for only
some species. ``NO`` is the constructed outlier: its NN/PBE ratio is 7 in
every cell, so it is the species the T1/tail machinery must surface and the
one the variant excludes.

The closing section covers the recurring-tail variant. Neither
``cell_species_density_ratios`` (the per-cell NN/PBE reduction the tail table
and the variant share) nor ``recurring_tail_species`` (the majority rule over
the cells that carry a species) exists yet, and no ``<fdir>_excl_tail``
directory is written, so those tests are RED as the rest of the file was
before its commits landed.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
from pathlib import Path

import pytest

from test_make_ablation_arch_figure import (
    fig,
    _make_dfs_results,
    _make_run_dir,
    _png_ok,
    _write_pm,
)

_HERE = Path(__file__).resolve().parent
_GOLDEN = _HERE / "golden" / "density_variants"
_GOLDEN_CSVS = ("holdout_ed_combined.csv",
                "holdout_by_pool_3x3.csv",
                "holdout_by_pool_3x3_eps.csv")
# the tail table joins the byte-identity set (2026-09-08: its per-cell rule is shared
# with the recurring-tail variant, so the refactor is pinned here); the commit-B column
# test keeps the three ED tables it was written for
_GOLDEN_BYTE_CSVS = _GOLDEN_CSVS + ("holdout_density_tail.csv",
                                    "insample_by_pool_3x3_eps.csv")
# The in-sample table is rendered from the SIBLING fixture (:func:`_insample_run_dir`):
# the Eq. 20 eps columns its in-sample rows need would fill D_insample_rmse on
# the eps legs of holdout_ed_combined.csv, a column the goldens above pin blank.
_INSAMPLE_GOLDENS = frozenset({"insample_by_pool_3x3_eps.csv"})
_README = _HERE / "README_density_figures.md"

# ---------------------------------------------------------------------------
# Rich fixture: two archs x two subset sizes, five held-out density species
# ---------------------------------------------------------------------------

# spec index -> cell, for the four evaluated specs of _make_run_dir
_RICH_CELLS = {("deep", 1): 0, ("deep", 3): 1,
               ("deep_notransform", 1): 2, ("deep_notransform", 3): 3}

# NN channels scale by (1 + 0.1*spec_index) so the four cells differ.
_NN_RMSE = {"H2": 1.0e-3, "h2": 3.0e-3, "CO": 5.0e-3, "NO": 7.0e-3,
            "N2": 2.0e-3}
_NN_EPS = {"H2": 1.5e-3, "h2": 2.5e-3, "CO": 4.0e-3, "NO": 6.0e-3,
           "N2": 1.0e-3}
# Model-free PBE columns: identical across specs (the cross-spec consistency
# guard drops a species whose reference drifts) and identical within the
# H2/h2 twin group (else the guard drops the pair once it keys on casefold).
_PBE_RMSE = {"H2": 4.0e-3, "h2": 4.0e-3, "CO": 8.0e-3, "NO": 1.0e-3,
             "N2": 2.0e-3}
_PBE_EPS = {"H2": 5.0e-3, "h2": 5.0e-3, "CO": 9.0e-3, "NO": 1.0e-3,
            "N2": 3.0e-3}

# run-level table: CH2O has no held-out rows (anchor-only species), and the
# eps column is backfilled for four of the six keys only.
_PBE_TABLE = {
    "H2": {"density_rmse_pbe": 4.0e-3, "density_l1_pbe": 1.0e-4,
           "density_eps_l1_pbe": 5.0e-3},
    "h2": {"density_rmse_pbe": 4.0e-3, "density_l1_pbe": 1.0e-4},
    "CO": {"density_rmse_pbe": 8.0e-3, "density_l1_pbe": 2.0e-4,
           "density_eps_l1_pbe": 9.0e-3},
    "NO": {"density_rmse_pbe": 1.0e-3, "density_l1_pbe": 5.0e-5,
           "density_eps_l1_pbe": 1.0e-3},
    "N2": {"density_rmse_pbe": 2.0e-3, "density_l1_pbe": 8.0e-5,
           "density_eps_l1_pbe": 3.0e-3},
    "CH2O": {"density_rmse_pbe": 1.0e-2, "density_l1_pbe": 4.0e-4},
}

_EXCLUDED_CF = frozenset({"no"})
_VARIANT_NOTE = ("1 species excluded in every cell (T1 > 0.02, Lee-Taylor "
                 "1989): ['NO']")


def _cell_factor(idx: int) -> float:
    return 1.0 + 0.1 * idx


def _rich_density_rows(idx: int):
    """One spec's held-out ``per_molecule.json`` rows."""
    f = _cell_factor(idx)
    rows = []
    for m in ("H2", "h2", "CO", "NO", "N2"):
        r = {
            "molecule": m,
            "density_rmse": _NN_RMSE[m] * f,
            "density_l1": _NN_RMSE[m] * f * 0.1,
            "density_rmse_pbe": _PBE_RMSE[m],
            "density_l1_pbe": _PBE_RMSE[m] * 0.1,
            "density_eps_l1": _NN_EPS[m] * f,
            "density_eps_l1_pbe": _PBE_EPS[m],
            "ref_density_method": "ccsd",
            "from_training_subset": False,
            "n_electrons": {"H2": 2.0, "h2": 2.0, "CO": 14.0, "NO": 15.0,
                            "N2": 14.0}[m],
            "scf_energy_residual_0": 0.22 if m == "NO" else 1.0e-6,
            "scf_converged": m != "NO",
        }
        if m != "N2":                 # N2 exercises the .get(None) default
            r["cycles_run"] = 25 if m == "NO" else 12
        rows.append(r)
    # an atom row with no density channels at all (dropped on read)
    rows.append({"molecule": "H", "density_rmse": None, "density_l1": None,
                 "density_rmse_pbe": None, "density_l1_pbe": None,
                 "density_eps_l1": None, "density_eps_l1_pbe": None,
                 "ref_density_method": None, "from_training_subset": False})
    return rows


def _rich_insample_rows(idx: int):
    """One spec's in-sample ``eval/per_molecule.json`` rows: the two AE
    species of the shared fixture plus a constructed ``H2``/``h2`` twin pair
    with different values (the in-sample sets are twin-free in production;
    the pair is what makes a row mean and a species mean differ)."""
    return [
        {"molecule": "HO", "AE_error_kcalmol": 6.0 + idx, "density_rmse": 3e-3,
         "density_l1": 3e-4, "skipped": False, "scf_converged": False,
         "scf_energy_residual_0": 2.0e-5, "cycles_run": 18,
         "n_electrons": 9.0},
        {"molecule": "CH4", "AE_error_kcalmol": -2.0 - idx,
         "density_rmse": 1e-3, "density_l1": 1e-4, "skipped": False,
         "scf_converged": True, "scf_energy_residual_0": 1.0e-7,
         "cycles_run": 9, "n_electrons": 10.0},
        {"molecule": "H2", "AE_error_kcalmol": None, "density_rmse": 1.0e-3,
         "skipped": False, "scf_converged": True},
        {"molecule": "h2", "AE_error_kcalmol": None, "density_rmse": 5.0e-3,
         "skipped": False, "scf_converged": True},
        {"molecule": "H", "skipped": True, "skip_reason": "atomic_system",
         "AE_error_kcalmol": None, "density_rmse": None},
    ]


def _rich_reactions(idx: int):
    """The shared fixture's two reactions plus two whose species lists carry
    the density species, so the per-channel (3x3) views have density content
    in both channels."""
    return [
        {"name": "bh76_a", "pool": "bh76",
         "reactants": ["HO", "h"], "products": ["HOh_ts"],
         "reaction_energy_ref_kcalmol": 17.7,
         "de_nn_kcalmol": -91.0 + idx, "de_pbe_kcalmol": -91.2 + idx,
         "abs_error_nn_kcalmol": 108.7 - idx,
         "abs_error_pbe_kcalmol": 108.9 - idx},
        {"name": "w411_b", "pool": "w411",
         "reactants": ["HO"], "products": ["h", "o"],
         "reaction_energy_ref_kcalmol": 120.0,
         "de_nn_kcalmol": 118.0 + idx, "de_pbe_kcalmol": 119.0 + idx,
         "abs_error_nn_kcalmol": 2.0 + idx, "abs_error_pbe_kcalmol": 1.0 + idx},
        {"name": "w411_co", "pool": "w411",
         "reactants": ["CO"], "products": ["c", "o"],
         "reaction_energy_ref_kcalmol": 257.0,
         "de_nn_kcalmol": 250.0 + idx, "de_pbe_kcalmol": 248.0 + idx,
         "abs_error_nn_kcalmol": 7.0 - idx, "abs_error_pbe_kcalmol": 9.0 - idx},
        {"name": "bh76_h2no", "pool": "bh76",
         "reactants": ["H2", "NO"], "products": ["N2", "h"],
         "reaction_energy_ref_kcalmol": 30.0,
         "de_nn_kcalmol": 26.0 + idx, "de_pbe_kcalmol": 24.0 + idx,
         "abs_error_nn_kcalmol": 4.0 - idx, "abs_error_pbe_kcalmol": 6.0 - idx},
    ]


def _rich_run_dir(root: Path) -> Path:
    """The shared run fixture with the rich density layer written over it."""
    run = _make_run_dir(root)
    for sd in sorted((run / "checkpoints").glob("spec_*")):
        idx = int(sd.name.split("_")[1])
        eh = sd / "eval_holdout"
        if not (eh / "per_reaction.json").is_file():
            continue
        (eh / "per_reaction.json").write_text(json.dumps(_rich_reactions(idx)))
        (eh / "per_molecule.json").write_text(
            json.dumps(_rich_density_rows(idx)))
        (sd / "eval" / "per_molecule.json").write_text(
            json.dumps(_rich_insample_rows(idx)))
    (run / "pbe_density_errors.json").write_text(json.dumps(
        {"basis": "def2-svp", "grid_level": 2, "refs_dir": "/refs",
         "errors": _PBE_TABLE, "failures": {}}))
    return run


# -- independent oracles (plain python; no reduction from the module) -------

def _species_mean(values_by_raw_name, keep=None):
    """Twin-then-species mean: average the raw names sharing a casefold key,
    then average those group values. ``keep`` filters on the casefold key."""
    groups = {}
    for name, val in values_by_raw_name.items():
        if val is None:
            continue
        key = name.casefold()
        if keep is not None and key not in keep:
            continue
        groups.setdefault(key, []).append(float(val))
    if not groups:
        return float("nan")
    per_species = [sum(v) / len(v) for v in groups.values()]
    return sum(per_species) / len(per_species)


def _expected_d_rmse(idx, keep=None):
    f = _cell_factor(idx)
    return _species_mean({m: v * f for m, v in _NN_RMSE.items()}, keep)


def _expected_d_eps(idx, keep=None):
    f = _cell_factor(idx)
    return _species_mean({m: v * f for m, v in _NN_EPS.items()}, keep)


def _expected_d_pbe(key, keep=None):
    return _species_mean({m: d.get(key) for m, d in _PBE_TABLE.items()}, keep)


_ALL_CF = frozenset({"h2", "co", "no", "n2", "ch2o"})
_KEPT_CF = _ALL_CF - _EXCLUDED_CF


# ---------------------------------------------------------------------------
# One shared pair of builder runs (standard + variant) for the rich fixture
# ---------------------------------------------------------------------------

class _Builds:
    def __init__(self):
        self.run = None
        self.std = None
        self.var = None
        self.notes = {"std": [], "var": []}
        # the builder's own stdout per run, so a skip line printed at build
        # time is assertable without a second render
        self.out = {"std": "", "var": ""}
        self.error = None

    def require(self):
        """Re-raise the builder's own failure inside the calling test, so a
        missing keyword surfaces as a TypeError there rather than as a
        fixture error."""
        if self.error is not None:
            raise self.error


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    """Standard and variant builder runs over the rich fixture, rendered once
    for the module. The standard run passes the new keywords at their
    documented defaults -- that call IS the no-op claim commit A makes."""
    b = _Builds()
    root = tmp_path_factory.mktemp("rich")
    b.run = _rich_run_dir(root)
    b.std = root / "std"
    b.var = root / "var"
    real_stamp = fig._stamp_parity_footer
    bucket = {"which": "std"}

    def _spy(mpl_fig, **kw):
        b.notes[bucket["which"]].append(kw)
        return real_stamp(mpl_fig, **kw)

    fig._stamp_parity_footer = _spy
    try:
        for which, outdir, excl, note in (
                ("std", b.std, frozenset(), ""),
                ("var", b.var, _EXCLUDED_CF, _VARIANT_NOTE)):
            bucket["which"] = which
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    fig.build_density_energy_figures(b.run, outdir,
                                                     exclude_cf=excl,
                                                     variant_note=note)
            finally:
                b.out[which] = buf.getvalue()
    except Exception as exc:                      # RED until commit A lands
        b.error = exc
    finally:
        fig._stamp_parity_footer = real_stamp
    return b


def _read_csv(path: Path):
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def _by_leg_cell(rows):
    return {(r["leg"], r["arch"], int(r["subset_size"])): r for r in rows}


# ---------------------------------------------------------------------------
# Commit A -- exclusion by casefold, the T1 loader, the tail table
# ---------------------------------------------------------------------------

def test_collect_holdout_density_rows_exclude_cf_drops_both_case_twins(
        tmp_path, capsys):
    """Kills the mutation `raw-name exclusion (both twins dropped)`: a filter
    on the raw name leaves the other case variant in every cell mean."""
    run = _rich_run_dir(tmp_path)
    rows = fig.collect_holdout_density_rows(run, exclude_cf=frozenset({"h2"}))
    assert rows, "the remaining species must survive"
    assert not [r for r in rows if str(r["molecule"]).casefold() == "h2"]
    assert {str(r["molecule"]) for r in rows} == {"CO", "NO", "N2"}
    out = capsys.readouterr().out
    assert "H2" in out and "h2" in out, out
    # the count: 8 dropped rows (2 spellings x 4 evaluated specs), or the
    # 2 species behind them -- the message must carry one of them
    assert re.search(r"\b(8|2)\b", out), out


def test_collect_holdout_density_rows_empty_exclusion_is_identical(tmp_path):
    """The empty set must be byte-identical behaviour -- the variant hook
    cannot perturb the standard figure set."""
    run = _rich_run_dir(tmp_path)
    base = fig.collect_holdout_density_rows(run)
    same = fig.collect_holdout_density_rows(run, exclude_cf=frozenset())
    assert same == base


def test_load_pbe_density_table_exclude_cf_is_case_insensitive(tmp_path):
    """Kills the mutation `table filter on raw keys`: the table lists the
    twins under both spellings, so a raw-key filter leaves one anchor in."""
    run = _rich_run_dir(tmp_path)
    full = fig.load_pbe_density_table(run)
    assert set(full) == set(_PBE_TABLE)
    cut = fig.load_pbe_density_table(run, exclude_cf=frozenset({"h2", "no"}))
    assert set(cut) == {"CO", "N2", "CH2O"}
    assert fig.load_pbe_density_table(run, exclude_cf=frozenset()) == full


def test_load_t1_table_casefolds_keys_and_returns_empty_when_absent(tmp_path):
    """Kills the mutation `T1 keys not casefolded`: the diagnostics file
    carries the pools' raw names, while the species present in the held-out
    rows are compared as casefolded keys, so an un-casefolded table
    intersects to nothing and the variant is silently skipped."""
    run = _rich_run_dir(tmp_path)
    assert fig.load_t1_table(run) == {}
    (run / "t1_diagnostics.json").write_text(json.dumps(
        {"t1": {"NO": 0.05, "N2": 0.008}, "threshold": 0.02,
         "source": "CCSD T1 diagnostic (Lee-Taylor 1989)"}))
    t1 = fig.load_t1_table(run)
    assert set(t1["t1"]) == {"no", "n2"}
    assert t1["t1"]["no"] == pytest.approx(0.05)
    assert t1["threshold"] == pytest.approx(0.02)
    assert "Lee-Taylor" in t1["source"]


def test_density_collectors_carry_scf_diagnostic_columns(tmp_path):
    """The tail table's diagnostic columns must reach it from BOTH density
    collectors, absent values as None rather than a KeyError."""
    run = _rich_run_dir(tmp_path)
    hd = {str(r["molecule"]): r for r in fig.collect_holdout_density_rows(run)
          if r["idx"] == 0}
    for col in ("scf_energy_residual_0", "scf_converged", "cycles_run",
                "n_electrons"):
        assert all(col in r for r in hd.values()), col
    assert hd["NO"]["scf_energy_residual_0"] == pytest.approx(0.22)
    assert hd["NO"]["scf_converged"] is False
    assert hd["NO"]["cycles_run"] == 25
    assert hd["N2"]["cycles_run"] is None          # absent -> None, not KeyError
    assert hd["CO"]["n_electrons"] == pytest.approx(14.0)
    ins = {str(r["molecule"]): r for r in fig.collect_insample_density_rows(run)
           if r["idx"] == 0}
    for col in ("scf_energy_residual_0", "scf_converged", "cycles_run",
                "n_electrons"):
        assert all(col in r for r in ins.values()), col
    assert ins["HO"]["cycles_run"] == 18
    assert ins["H2"]["cycles_run"] is None


def test_holdout_density_tail_csv_collapses_twins_and_orders_by_ratio(
        tmp_path):
    """One row per (arch, subset_size, casefolded species) above the ratio
    floor: the twin pair collapses to a single row whose residual is the MAX
    and whose converged flag is the AND of the pair, ordered ratio-descending
    within the cell."""
    rows = [
        # cell (deep, 1): the twin pair (ratio 5.0 once averaged) written
        # first, so a tail table that preserved input order would fail
        {"arch": "deep", "subset_size": 1, "molecule": "H2",
         "density_rmse": 1.0e-3, "density_rmse_pbe": 4.0e-4,
         "scf_energy_residual_0": 0.05, "scf_converged": True,
         "cycles_run": 10},
        {"arch": "deep", "subset_size": 1, "molecule": "h2",
         "density_rmse": 3.0e-3, "density_rmse_pbe": 4.0e-4,
         "scf_energy_residual_0": 0.20, "scf_converged": False,
         "cycles_run": 25},
        {"arch": "deep", "subset_size": 1, "molecule": "CO",
         "density_rmse": 1.0e-3, "density_rmse_pbe": 8.0e-4,
         "scf_energy_residual_0": 1e-7, "scf_converged": True,
         "cycles_run": 9},                       # ratio 1.25 -> below floor
        {"arch": "deep", "subset_size": 1, "molecule": "NO",
         "density_rmse": 9.0e-3, "density_rmse_pbe": 1.0e-3,
         "scf_energy_residual_0": 0.22, "scf_converged": False,
         "cycles_run": 25},                      # ratio 9.0
        {"arch": "deep", "subset_size": 1, "molecule": "N2",
         "density_rmse": None, "density_rmse_pbe": 2.0e-3,
         "scf_energy_residual_0": None, "scf_converged": None,
         "cycles_run": None},                    # no NN leg -> no row
        {"arch": "deep", "subset_size": 3, "molecule": "NO",
         "density_rmse": 4.0e-3, "density_rmse_pbe": 1.0e-3,
         "scf_energy_residual_0": 0.1, "scf_converged": True,
         "cycles_run": 20},
    ]
    out = tmp_path / "tail.csv"
    fig.write_holdout_density_tail_csv(rows, out, t1={"h2": 0.031})
    got = _read_csv(out)
    assert list(got[0]) == ["arch", "subset_size", "molecule", "ratio",
                            "density_rmse", "density_rmse_pbe",
                            "scf_energy_residual_0", "scf_converged",
                            "cycles_run", "t1_diagnostic"]
    cell = [r for r in got if r["arch"] == "deep" and r["subset_size"] == "1"]
    assert [r["molecule"].casefold() for r in cell] == ["no", "h2"]
    twin = cell[1]
    assert float(twin["ratio"]) == pytest.approx(5.0)
    assert float(twin["density_rmse"]) == pytest.approx(2.0e-3)
    assert float(twin["density_rmse_pbe"]) == pytest.approx(4.0e-4)
    assert float(twin["scf_energy_residual_0"]) == pytest.approx(0.20)
    assert twin["scf_converged"] == "False"
    assert twin["cycles_run"] == "25"
    assert float(twin["t1_diagnostic"]) == pytest.approx(0.031)
    assert cell[0]["t1_diagnostic"] == ""          # no T1 entry -> blank
    assert {r["subset_size"] for r in got} == {"1", "3"}
    # the floor is a parameter, not a constant baked into the reduction
    out2 = out.with_name("tail6.csv")
    fig.write_holdout_density_tail_csv(rows, out2, ratio_threshold=6.0)
    assert [r["molecule"].casefold() for r in _read_csv(out2)] == ["no"]


def test_builder_writes_holdout_density_tail_csv(builds):
    """The tail table is written beside the builder's other CSVs and stays
    out of the PNG-only return contract."""
    builds.require()
    tail = builds.std / "holdout_density_tail.csv"
    assert tail.is_file()
    got = _read_csv(tail)
    # NO is the only species above the 1.5 ratio floor in the fixture
    assert {r["molecule"].casefold() for r in got} == {"no"}
    assert len(got) == len(_RICH_CELLS)
    ratios = {(r["arch"], int(r["subset_size"])): float(r["ratio"])
              for r in got}
    for cell, idx in _RICH_CELLS.items():
        assert ratios[cell] == pytest.approx(
            _NN_RMSE["NO"] * _cell_factor(idx) / _PBE_RMSE["NO"], rel=1e-9)


def test_variant_csv_d_rmse_is_species_mean_without_excluded(builds):
    """Kills the mutation `twin-deduplicating exclusion`: the cell density
    mean is the twin-then-species mean over the SURVIVING species. The H2/h2
    pair carries different NN values, so dropping (or double-counting) a twin
    row moves D_rmse away from the oracle."""
    builds.require()
    std = _by_leg_cell(_read_csv(builds.std
                                 / "holdout_ed_combined.csv"))
    var = _by_leg_cell(_read_csv(builds.var
                                 / "holdout_ed_combined.csv"))
    for (arch, ss), idx in _RICH_CELLS.items():
        s = std[("wtmad2", arch, ss)]
        v = var[("wtmad2", arch, ss)]
        assert float(s["D_rmse"]) == pytest.approx(_expected_d_rmse(idx),
                                                   rel=1e-9)
        assert float(v["D_rmse"]) == pytest.approx(
            _expected_d_rmse(idx, _KEPT_CF), rel=1e-9)
        # the DFS-units leg carries the same rule on the eps channel
        se = std[("wtmad2_eps_gamma_dfs", arch, ss)]
        ve = var[("wtmad2_eps_gamma_dfs", arch, ss)]
        assert float(se["D_rmse"]) == pytest.approx(_expected_d_eps(idx),
                                                    rel=1e-9)
        assert float(ve["D_rmse"]) == pytest.approx(
            _expected_d_eps(idx, _KEPT_CF), rel=1e-9)


def test_variant_csv_d_pbe_rmse_filtered_on_both_legs(builds):
    """Kills the mutation `eps leg left unfiltered`: the run-level table is
    the anchor source for BOTH channels, and the excluded species carries an
    eps entry, so an exclusion applied only to the RMSE map leaves the
    DFS-units anchor at its unfiltered value."""
    builds.require()
    std = _by_leg_cell(_read_csv(builds.std
                                 / "holdout_ed_combined.csv"))
    var = _by_leg_cell(_read_csv(builds.var
                                 / "holdout_ed_combined.csv"))
    exp = {
        ("wtmad2", "density_rmse_pbe"): (_expected_d_pbe("density_rmse_pbe"),
                                         _expected_d_pbe("density_rmse_pbe",
                                                          _KEPT_CF)),
        ("wtmad2_eps_gamma_dfs", "density_eps_l1_pbe"):
            (_expected_d_pbe("density_eps_l1_pbe"),
             _expected_d_pbe("density_eps_l1_pbe", _KEPT_CF)),
    }
    for (leg, _key), (want_std, want_var) in exp.items():
        assert want_std != pytest.approx(want_var), "oracle must discriminate"
        for (arch, ss) in _RICH_CELLS:
            assert float(std[(leg, arch, ss)]["D_pbe_rmse"]) == pytest.approx(
                want_std, rel=1e-9), leg
            assert float(var[(leg, arch, ss)]["D_pbe_rmse"]) == pytest.approx(
                want_var, rel=1e-9), leg


def test_variant_note_and_dataset_reach_the_footer(builds):
    """Kills the mutation `variant_note dropped`: the recorded footer note of
    the variant run must carry the exclusion rule and the excluded name, and
    its dataset band must lose exactly the excluded species."""
    builds.require()
    var_notes = [kw.get("note") or "" for kw in builds.notes["var"]]
    std_notes = [kw.get("note") or "" for kw in builds.notes["std"]]
    assert var_notes, "no figure stamped a footer"
    carrying = [n for n in var_notes if _VARIANT_NOTE in n]
    assert len(carrying) >= 3, var_notes[:3]
    assert not [n for n in std_notes if "excluded in every cell" in n]

    def _species(kws):
        counts = set()
        for kw in kws:
            m = re.search(r"density: (\d+) species", kw.get("dataset") or "")
            if m:
                counts.add(int(m.group(1)))
        return counts

    std_n, var_n = _species(builds.notes["std"]), _species(builds.notes["var"])
    assert len(std_n) == 1 and len(var_n) == 1, (std_n, var_n)
    assert std_n.pop() - var_n.pop() == 1


def test_standard_dir_csvs_byte_identical_to_golden(builds, ins_builds):
    """The variant hook is a no-op at its defaults: the CSVs of the standard
    directory match, byte for byte, goldens produced by the unmodified
    builder on this same fixture.

    Commit B moves `n_density_species` by design (case twins collapse), so
    the goldens are re-baselined in that commit; regenerate with
    DENSITY_VARIANTS_REGOLD=1 pytest -k byte_identical and re-read the diff.
    ``insample_by_pool_3x3_eps.csv`` is pinned from the in-sample fixture's
    standard directory (see :data:`_INSAMPLE_GOLDENS`).
    """
    builds.require()
    ins_builds.require()

    def _src(name: str) -> Path:
        base = ins_builds.std if name in _INSAMPLE_GOLDENS else builds.std
        return base / name

    if os.environ.get("DENSITY_VARIANTS_REGOLD"):
        _GOLDEN.mkdir(parents=True, exist_ok=True)
        for name in _GOLDEN_BYTE_CSVS:
            (_GOLDEN / name).write_bytes(_src(name).read_bytes())
        pytest.fail("goldens rewritten; re-run without DENSITY_VARIANTS_REGOLD")
    for name in _GOLDEN_BYTE_CSVS:
        got = _src(name).read_bytes()
        want = (_GOLDEN / name).read_bytes()
        assert got == want, name


def test_suite_loop_renders_the_t1_variant_directory(tmp_path):
    """The suite renders `<fdir>_excl_t1` beside the standard directory when
    the run carries `t1_diagnostics.json`, and nothing when it does not."""
    root, run = _make_dfs_results(tmp_path)
    for sd in sorted((run / "checkpoints").glob("spec_*")):
        idx = int(sd.name.split("_")[1])
        eh = sd / "eval_holdout"
        if not (eh / "per_reaction.json").is_file():
            continue
        (eh / "per_molecule.json").write_text(
            json.dumps(_rich_density_rows(idx)))
    (run / "pbe_density_errors.json").write_text(json.dumps(
        {"basis": "def2-svp", "grid_level": 2, "refs_dir": "/refs",
         "errors": _PBE_TABLE, "failures": {}}))

    plain = tmp_path / "figs_plain"
    fig.build_bh76w411_suite(results_root=root, outroot=plain,
                             bases=("svp_grid2",), domain="dfs_step7")
    assert (plain / "figures_dfs_step7_svp").is_dir()
    assert not list(plain.glob("*_excl_t1"))

    # raw pool names in the file; the species set it is intersected with is
    # casefolded, so the loader must casefold before the intersection
    (run / "t1_diagnostics.json").write_text(json.dumps(
        {"t1": {"NO": 0.05, "N2": 0.008}, "threshold": 0.02,
         "source": "CCSD T1 diagnostic (Lee-Taylor 1989)"}))
    outroot = tmp_path / "figs_t1"
    # the architecture restriction must reach the variant call too: the two
    # directories describe the same run and would otherwise disagree on which
    # cells they draw
    fig.build_bh76w411_suite(results_root=root, outroot=outroot,
                             bases=("svp_grid2",), domain="dfs_step7",
                             archs=("deep",))
    variant = outroot / "figures_dfs_step7_svp_excl_t1"
    assert variant.is_dir(), sorted(p.name for p in outroot.iterdir())
    assert (variant / "holdout_density_vs_ccsd.png").is_file()
    csv_rows = [r for r in _read_csv(
        variant / "holdout_ed_combined.csv")
        if r["leg"] == "wtmad2"]
    assert csv_rows
    assert {r["arch"] for r in csv_rows} == {"deep"}
    for r in csv_rows:
        assert float(r["D_pbe_rmse"]) == pytest.approx(
            _expected_d_pbe("density_rmse_pbe", _KEPT_CF), rel=1e-9)


def test_readme_documents_every_written_png_and_csv(builds, ins_builds):
    """Kills the mutation `README walk over PNGs only`: the walk covers the
    CSVs too, so a tail table with no README row is caught. Both standard
    directories are walked -- the in-sample fixture is the only one whose run
    carries training reactions, so the in-sample 3x3 family is written there
    and nowhere else."""
    builds.require()
    ins_builds.require()
    text = _README.read_text()
    written = sorted({p.name for d in (builds.std, ins_builds.std)
                      for p in d.iterdir() if p.suffix in (".png", ".csv")})

    def documented(name):
        if name in text:
            return True
        # a per-arch family (a NEW_OUTPUTS stem followed by ``_<arch>``) is
        # documented under its pattern name
        for stem in getattr(fig, "NEW_OUTPUTS", ()):
            if name.startswith(stem + "_") and name.endswith(".png"):
                return f"{stem}_<arch>.png" in text
        return False

    missing = [n for n in written if not documented(n)]
    assert not missing, missing
    # the family rule is live: the channel family's pattern name is what
    # the README carries, and a name outside any family is still literal
    assert documented("training_loss_channels_medium.png")
    assert not documented("a_file_nothing_writes.png")


# ---------------------------------------------------------------------------
# Commit B -- twin-consistent published counts
# ---------------------------------------------------------------------------

def _twin_density_rows(order=("H2", "h2", "CO")):
    return [{"arch": "deep", "subset_size": 1, "molecule": m,
             "density_rmse": {"H2": 1.0e-3, "h2": 3.0e-3, "CO": 5.0e-3}[m],
             "density_rmse_pbe": {"H2": 4.0e-3, "h2": 4.0e-3,
                                  "CO": 8.0e-3}[m]}
            for m in order]


def test_cell_counts_collapses_twins_and_counts_nameless_rows():
    """Kills the mutation `_cell_counts branching on name is None`: the
    density rows carry no reaction name, and two distinct nameless rows must
    still count two while the case twins count one."""
    rows = _twin_density_rows()
    assert fig._cell_counts(rows, "density_rmse") == {("deep", 1): 2}
    nameless = [{"arch": "deep", "subset_size": 1, "x": 1.0},
                {"arch": "deep", "subset_size": 1, "x": 2.0}]
    assert fig._cell_counts(nameless, "x") == {("deep", 1): 2}
    # named reaction rows keep their name-dedup semantics unchanged
    rxn = [{"arch": "deep", "subset_size": 1, "name": "r1",
            "abs_error_nn_kcalmol": 1.0},
           {"arch": "deep", "subset_size": 1, "name": "r1",
            "abs_error_nn_kcalmol": 2.0},
           {"arch": "deep", "subset_size": 1, "name": "r2",
            "abs_error_nn_kcalmol": 3.0}]
    assert fig._cell_counts(rxn, "abs_error_nn_kcalmol") == {("deep", 1): 2}


def test_holdout_eval_note_counts_species_not_case_twins():
    """The dataset band publishes a species count; the twins are one
    species."""
    note = fig._holdout_eval_note([], _twin_density_rows())
    assert "density: 2 species" in note, note


def test_density_parity_panel_title_counts_species_and_averages_twins():
    """One point per (arch, subset_size, casefolded species): the panel title
    publishes that count, not the row count."""
    plt = pytest.importorskip("matplotlib.pyplot")
    pbe_mol = {"h2": 4.0e-3, "co": 8.0e-3}
    mfig, ax = plt.subplots()
    try:
        fig._density_parity_panel(ax, _twin_density_rows(), pbe_mol)
        assert "2 points" in ax.get_title(), ax.get_title()
        drawn = sorted((float(x), float(y)) for c in ax.collections
                       for x, y in c.get_offsets())
        # the twin point sits at the MEAN of the pair, not at either row
        assert drawn == [pytest.approx((4.0e-3, 2.0e-3)),
                         pytest.approx((8.0e-3, 5.0e-3))], drawn
    finally:
        plt.close(mfig)


def test_density_cell_coverage_warning_ignores_case_twins():
    """Cells whose species sets differ only by the spelling of a twin cover
    the same species, so the divergence warning must stay empty."""
    rows = [{"arch": "deep", "subset_size": 1, "molecule": "H2",
             "density_rmse": 1e-3},
            {"arch": "deep", "subset_size": 1, "molecule": "CO",
             "density_rmse": 5e-3},
            {"arch": "deep", "subset_size": 3, "molecule": "h2",
             "density_rmse": 2e-3},
            {"arch": "deep", "subset_size": 3, "molecule": "CO",
             "density_rmse": 6e-3}]
    assert fig._density_cell_coverage_warning(rows) == ""


def test_insample_density_strip_ticks_collapse_twins_and_keep_display_order():
    """Kills the mutation `casefold-key ordering`: groups are ordered by the
    DISPLAY name, so `NO` stays before `Na2` (as today's raw-name sort has
    it) while the H2/h2 pair collapses to one tick."""
    plt = pytest.importorskip("matplotlib.pyplot")
    rows = [{"arch": "deep", "subset_size": 1, "molecule": m,
             "density_rmse": v, "density_rmse_pbe": 2 * v}
            for m, v in (("NO", 1e-3), ("NO2", 2e-3), ("Na2", 3e-3),
                         ("H2", 4e-3), ("h2", 5e-3))]
    mfig, ax = plt.subplots()
    try:
        fig._insample_density_strip_panel(ax, rows)
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        assert ticks == ["H2", "NO", "NO2", "Na2"], ticks
        assert "4 trained species" in ax.get_title(), ax.get_title()
    finally:
        plt.close(mfig)


def test_insample_ae_strip_ticks_collapse_twins():
    """The AE strip carries the same grouping rule as the density strip."""
    plt = pytest.importorskip("matplotlib.pyplot")
    rows = [{"arch": "deep", "subset_size": 1, "molecule": m,
             "AE_error_kcalmol": v}
            for m, v in (("NO", 1.0), ("NO2", 2.0), ("Na2", 3.0),
                         ("H2", 4.0), ("h2", 5.0))]
    mfig, ax = plt.subplots()
    try:
        fig._insample_ae_strip_panel(ax, rows)
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        assert ticks == ["H2", "NO", "NO2", "Na2"], ticks
        assert "4 trained" in ax.get_title(), ax.get_title()
    finally:
        plt.close(mfig)


def test_insample_density_strip_displays_min_of_twin_group_in_both_row_orders():
    """Kills the mutation `first-seen display name`: the group's display name
    is min(group), so the tick reads `H2` whichever spelling the rows list
    first."""
    plt = pytest.importorskip("matplotlib.pyplot")
    for order in (("H2", "h2", "CO"), ("h2", "H2", "CO")):
        mfig, ax = plt.subplots()
        try:
            fig._insample_density_strip_panel(ax, _twin_density_rows(order))
            ticks = [t.get_text() for t in ax.get_xticklabels()]
            assert ticks == ["CO", "H2"], (order, ticks)
        finally:
            plt.close(mfig)


def test_inconsistent_pbe_density_reference_drops_both_case_twins(tmp_path,
                                                                  capsys):
    """Kills the mutation `the cross-spec-inconsistency drop on raw names`:
    the reference drift is a property of the physical species, so a pair
    whose two spellings disagree across specs must leave no row behind."""
    run = tmp_path / "run_drift"
    for spec, val in (("spec_0000", 8.0e-4), ("spec_0001", 2.0e-4)):
        _write_pm(run, spec,
                  [{"molecule": "H2", "density_rmse": 1e-4,
                    "density_rmse_pbe": val},
                   {"molecule": "h2", "density_rmse": 1e-4,
                    "density_rmse_pbe": 8.0e-4},
                   {"molecule": "CO", "density_rmse": 1e-4,
                    "density_rmse_pbe": 5.0e-4}])
    rows = fig.collect_holdout_density_rows(run)
    assert {str(r["molecule"]) for r in rows} == {"CO"}
    assert "H2" in capsys.readouterr().out


def test_commit_b_changes_only_the_density_count_column(builds):
    """The published counts move in one column only: every other field of the
    three CSVs still matches the golden."""
    builds.require()
    for name in _GOLDEN_CSVS:
        got = _read_csv(builds.std / name)
        want = _read_csv(_GOLDEN / name)
        assert len(got) == len(want), name
        for a, b in zip(got, want):
            a = dict(a)
            b = dict(b)
            a.pop("n_density_species", None)
            b.pop("n_density_species", None)
            assert a == b, name


# ---------------------------------------------------------------------------
# Commit C -- in-sample parity
# ---------------------------------------------------------------------------

def test_insample_density_by_arch_subset_collapses_twins():
    """Kills the mutation `D_insample_rmse over rows`: the in-sample cell
    value is the twin-then-species mean, as the held-out leg already is."""
    rows = [{"arch": "deep", "subset_size": 1, "molecule": m,
             "density_rmse": v, "density_eps_l1": 2 * v}
            for m, v in (("H2", 1.0e-3), ("h2", 5.0e-3), ("CO", 9.0e-3))]
    got = fig.insample_density_by_arch_subset(rows, "density_rmse")
    # twin-then-species: mean(mean(1, 5), 9) = 6; the row mean is 5
    assert got == {("deep", 1): pytest.approx(6.0e-3)}
    row_mean = sum(r["density_rmse"] for r in rows) / len(rows)
    assert got[("deep", 1)] != pytest.approx(row_mean), "row mean must not pass"
    eps = fig.insample_density_by_arch_subset(rows, "density_eps_l1")
    assert eps == {("deep", 1): pytest.approx(12.0e-3)}


def test_ed_csv_fields_carry_the_insample_columns():
    """The two new columns are appended to the field list (order matters --
    the header assertion in the module's own suite pins the set)."""
    assert fig._ED_CSV_FIELDS[-2:] == ["D_insample_rmse", "n_insample_species"]


def test_write_combined_ed_csv_writes_insample_columns(tmp_path):
    """Filled from the maps when they are supplied, blank when they are not
    (the per-channel 3x3 calls, the held-out pair and the in-sample one, pass nothing)."""
    energy = {("deep", 1): 8.0, ("deep_attn", 1): 20.0}
    density = {("deep", 1): 0.004, ("deep_attn", 1): 0.02}
    wt = fig.combined_ed_by_cell(energy, 10.0, density, 0.005)
    out = tmp_path / "ed.csv"
    fig.write_combined_ed_csv(
        {"wtmad2": wt}, out, n_reactions={}, n_density={},
        d_insample={("deep", 1): 0.0021}, n_insample={("deep", 1): 3})
    rows = {r["arch"]: r for r in _read_csv(out)}
    assert float(rows["deep"]["D_insample_rmse"]) == pytest.approx(0.0021)
    assert rows["deep"]["n_insample_species"] == "3"
    assert rows["deep_attn"]["D_insample_rmse"] == ""
    assert rows["deep_attn"]["n_insample_species"] == ""
    out2 = tmp_path / "ed2.csv"
    fig.write_combined_ed_csv({"wtmad2": wt}, out2, n_reactions={},
                              n_density={})
    blank = _read_csv(out2)
    assert all(r["D_insample_rmse"] == "" and r["n_insample_species"] == ""
               for r in blank)


def test_builder_fills_insample_columns_in_the_combined_csv_only(builds):
    """The combined CSV carries the in-sample leg; the per-channel 3x3 CSVs
    leave both columns blank."""
    builds.require()
    combined = _read_csv(builds.std / "holdout_ed_combined.csv")
    # in-sample species: HO, CH4 and the H2/h2 pair -> (3e-3 + 1e-3 + 3e-3)/3
    want = (3e-3 + 1e-3 + (1.0e-3 + 5.0e-3) / 2) / 3
    assert {r["leg"] for r in combined} >= {"wtmad2", "wtmad2_eps_gamma_dfs"}
    for r in combined:
        if "eps" in r["leg"]:
            # the eps legs carry the in-sample eps mean in their own units;
            # this fixture's in-sample rows have no eps columns -> blank
            assert r["D_insample_rmse"] == "", r["leg"]
        else:
            assert float(r["D_insample_rmse"]) == pytest.approx(want, rel=1e-9)
        assert r["n_insample_species"] == "3"
    for name in ("holdout_by_pool_3x3.csv",
                 "holdout_by_pool_3x3_eps.csv"):
        for r in _read_csv(builds.std / name):
            assert r["D_insample_rmse"] == "", name
            assert r["n_insample_species"] == "", name
    # the in-sample 3x3's own table (written by the in-sample fixture's run)
    # is a per-channel table too: its density leg IS the in-sample value, so
    # the two trailing columns stay blank there as well
    for r in _read_csv(_GOLDEN / "insample_by_pool_3x3_eps.csv"):
        assert r["D_insample_rmse"] == "", "insample_by_pool_3x3_eps.csv"
        assert r["n_insample_species"] == "", "insample_by_pool_3x3_eps.csv"
    # the variant directory carries the SAME in-sample columns: the exclusion
    # filters the held-out rows only, never the trained species
    var = _read_csv(builds.var / "holdout_ed_combined.csv")
    key = lambda r: (r["leg"], r["arch"], r["subset_size"])
    std_ins = {key(r): (r["D_insample_rmse"], r["n_insample_species"]) for r in combined}
    var_ins = {key(r): (r["D_insample_rmse"], r["n_insample_species"]) for r in var}
    assert var_ins == std_ins


def test_insample_density_ccsd_gains_a_ratio_panel(tmp_path, monkeypatch):
    """The in-sample figure gains a per-cell NN/PBE ratio panel (the parity
    of the held-out family). Pinned on the axes the figure carries, which
    the plotter builds through ``plt.subplots``."""
    plt = pytest.importorskip("matplotlib.pyplot")
    made = []
    real = plt.subplots

    def _spy(*a, **kw):
        out = real(*a, **kw)
        made.append(out[0])
        return out

    monkeypatch.setattr(fig.plt, "subplots", _spy)
    rows = [{"arch": "deep", "subset_size": s, "molecule": m,
             "density_rmse": v, "density_rmse_pbe": 2.0 * v}
            for s in (1, 3)
            for m, v in (("HO", 3e-3), ("CH4", 1e-3))]
    fig.plot_insample_density_ccsd(
        rows, tmp_path / "insample_density_vs_ccsd.png", "run_x")
    assert made, "the plotter no longer builds its figure through plt.subplots"
    axes = made[0].axes
    assert len(axes) >= 3, len(axes)
    labels = " | ".join((ax.get_title() or "") + " " + (ax.get_ylabel() or "")
                        for ax in axes).lower()
    assert "ratio" in labels or "nn/pbe" in labels, labels
    # the ratio is NN over PBE: every cell of the fixture has density_rmse_pbe
    # = 2 x density_rmse, so the arch line sits at 0.5 and the unit line at 1
    ratio_ax = [ax for ax in axes
                if "ratio" in ((ax.get_title() or "") + (ax.get_ylabel() or "")).lower()][0]
    ys = sorted({round(float(y), 9) for line in ratio_ax.get_lines()
                 for y in line.get_ydata()})
    assert ys == [0.5, 1.0], ys


def test_insample_density_ccsd_eps_twin_iff_eps_columns(tmp_path):
    """The DFS-units twin is written when (and only when) the rows carry both
    Eq. 20 eps columns -- the production in-sample rows all do, the module's
    own fixture rows carry none, so its exact-set assertions survive."""
    rows = [{"arch": "deep", "subset_size": s, "molecule": m,
             "density_rmse": v, "density_rmse_pbe": 2.0 * v}
            for s in (1, 3)
            for m, v in (("HO", 3e-3), ("CH4", 1e-3))]
    plain = tmp_path / "plain"
    plain.mkdir()
    fig.plot_insample_density_ccsd(
        rows, plain / "insample_density_vs_ccsd.png", "run_x")
    assert not (plain
                / "insample_density_vs_ccsd_eps.png").exists()
    eps_rows = [dict(r, density_eps_l1=r["density_rmse"] * 1.5,
                     density_eps_l1_pbe=r["density_rmse_pbe"] * 1.5)
                for r in rows]
    withe = tmp_path / "with_eps"
    withe.mkdir()
    fig.plot_insample_density_ccsd(
        eps_rows, withe / "insample_density_vs_ccsd.png", "run_x")
    twin = withe / "insample_density_vs_ccsd_eps.png"
    assert _png_ok(twin)


def test_eps_twin_uses_eps_suffix(tmp_path):
    """The per-electron-units twin is suffixed ``_eps``, not the holdover
    ``_dfs_units``: the suffix is appended to whatever stem the caller gives,
    and nothing else is written beside the base figure."""
    rows = [{"arch": "deep", "subset_size": s, "molecule": m,
             "density_rmse": v, "density_rmse_pbe": 2.0 * v,
             "density_eps_l1": v * 1.5, "density_eps_l1_pbe": 3.0 * v}
            for s in (1, 3)
            for m, v in (("HO", 3e-3), ("CH4", 1e-3))]
    out = tmp_path / "eps_suffix"
    out.mkdir()
    base = out / "insample_density_vs_ccsd.png"
    fig.plot_insample_density_ccsd(rows, base, "run_x")
    assert _png_ok(out / "insample_density_vs_ccsd_eps.png")
    assert not (out / "insample_density_vs_ccsd_dfs_units.png").exists()
    assert {p.name for p in out.iterdir()} == {
        "insample_density_vs_ccsd.png", "insample_density_vs_ccsd_eps.png"}

    # the suffix is generic: a different stem takes the same twin name
    other = tmp_path / "other_stem"
    other.mkdir()
    fig.plot_insample_density_ccsd(rows, other / "stem_x.png", "run_x")
    assert _png_ok(other / "stem_x_eps.png")
    assert not (other / "stem_x_dfs_units.png").exists()


# ---------------------------------------------------------------------------
# Review findings on commits A-C (2026-09-07): the variant note's scope, the
# loaders' robustness, the tail table's provenance, the units of the in-sample
# leg, and the converged-channel variant
# ---------------------------------------------------------------------------

_UNFILTERED_TITLE_STARTS = ("In-sample", "Held-out energy")


def test_rf_variant_note_reaches_only_the_filtered_figures(builds):
    """The exclusion note belongs to the held-out density family: the
    energy-only and in-sample figures of a variant directory hold unfiltered
    data and must not assert an exclusion."""
    builds.require()
    var = builds.notes["var"]
    unfiltered = [kw for kw in var
                  if str(kw.get("title", "")).startswith(_UNFILTERED_TITLE_STARTS)
                  or "rung" in str(kw.get("title", "")).lower()]
    filtered = [kw for kw in var if not (
        str(kw.get("title", "")).startswith(_UNFILTERED_TITLE_STARTS)
        or "rung" in str(kw.get("title", "")).lower())]
    assert len(unfiltered) >= 5 and len(filtered) >= 3, (len(unfiltered), len(filtered))
    leaked = [kw["title"] for kw in unfiltered
              if "excluded in every cell" in (kw.get("note") or "")]
    assert not leaked, leaked
    missing = [kw["title"] for kw in filtered
               if "excluded in every cell" not in (kw.get("note") or "")]
    assert not missing, missing


def test_rf_load_t1_table_tolerates_malformed_files(tmp_path):
    """A malformed or empty diagnostics file is "no table", never an abort of
    the whole figure build."""
    run = _rich_run_dir(tmp_path)
    for text in ("[1, 2, 3]", '{"t1": {"NO": 0.05}, "threshold": null}',
                 '{"t1": {"NO": 0.05}, "threshold": "hi"}', '{"t1": "x"}', "{}",
                 '{"t1": {"NO": "bad"}, "threshold": 0.02}'):
        (run / "t1_diagnostics.json").write_text(text)
        assert fig.load_t1_table(run) == {}, text


def test_rf_exclude_cf_is_normalized(tmp_path):
    """Any spelling and any iterable: the set is casefolded on entry, so a
    raw-name set is not a silent no-op."""
    run = _rich_run_dir(tmp_path)
    rows = fig.collect_holdout_density_rows(run, exclude_cf={"NO"})
    assert {str(r["molecule"]) for r in rows} == {"H2", "h2", "CO", "N2"}
    tab = fig.load_pbe_density_table(run, exclude_cf=["H2"])
    assert set(tab) == {"CO", "NO", "N2", "CH2O"}


def test_rf_tail_table_skips_rows_without_a_molecule(tmp_path):
    rows = [{"arch": "deep", "subset_size": 1, "density_rmse": 9e-3,
             "density_rmse_pbe": 1e-3},
            {"arch": "deep", "subset_size": 1, "molecule": None,
             "density_rmse": 9e-3, "density_rmse_pbe": 1e-3},
            {"arch": "deep", "subset_size": 1, "molecule": "NO",
             "density_rmse": 9e-3, "density_rmse_pbe": 1e-3}]
    out = tmp_path / "tail.csv"
    fig.write_holdout_density_tail_csv(rows, out)
    assert [r["molecule"] for r in _read_csv(out)] == ["NO"]


def test_rf_variant_tail_table_holds_the_filtered_rows(builds):
    """The tail table describes its own directory's cell means: the variant's
    table lacks the excluded species."""
    builds.require()
    std = {r["molecule"].casefold()
           for r in _read_csv(builds.std / "holdout_density_tail.csv")}
    var = {r["molecule"].casefold()
           for r in _read_csv(builds.var / "holdout_density_tail.csv")}
    assert std == {"no"} and var == set(), (std, var)


def test_rf_insample_leg_units_follow_the_leg(tmp_path):
    """On the eps legs the in-sample column holds the in-sample eps mean (the
    leg's own units), blank when no in-sample eps map is given; the species
    count is filled on every leg."""
    energy = {("deep", 1): 8.0}
    density = {("deep", 1): 0.004}
    wt = fig.combined_ed_by_cell(energy, 10.0, density, 0.005)
    out = tmp_path / "ed.csv"
    fig.write_combined_ed_csv(
        {"wtmad2": wt, "wtmad2_eps_gamma_dfs": wt}, out, n_reactions={},
        n_density={}, d_insample={("deep", 1): 0.0021},
        n_insample={("deep", 1): 3}, d_insample_eps={("deep", 1): 0.007})
    rows = {r["leg"]: r for r in _read_csv(out)}
    assert float(rows["wtmad2"]["D_insample_rmse"]) == pytest.approx(0.0021)
    assert float(rows["wtmad2_eps_gamma_dfs"]["D_insample_rmse"]) == pytest.approx(0.007)
    assert rows["wtmad2_eps_gamma_dfs"]["n_insample_species"] == "3"
    out2 = tmp_path / "ed2.csv"
    fig.write_combined_ed_csv(
        {"wtmad2_eps_gamma_dfs": wt}, out2, n_reactions={}, n_density={},
        d_insample={("deep", 1): 0.0021}, n_insample={("deep", 1): 3})
    assert _read_csv(out2)[0]["D_insample_rmse"] == ""


def test_rf_unconverged_variant_directory(tmp_path):
    """The converged-channel variant drops the species whose NN SCF did not
    converge in any cell, and renders nothing when every species converged."""
    run = _rich_run_dir(tmp_path)
    for sd in sorted((run / "checkpoints").glob("spec_*")):
        eh = sd / "eval_holdout"
        if not (eh / "per_reaction.json").is_file():
            continue
        conv = sd / "eval_holdout_converged"
        conv.mkdir()
        (conv / "per_reaction.json").write_text((eh / "per_reaction.json").read_text())
        (conv / "per_molecule.json").write_text((eh / "per_molecule.json").read_text())
    fdir = tmp_path / "figs_conv"
    written = fig._build_outlier_free_variants(run, fdir,
                                               eval_subdir="eval_holdout_converged")
    vdir = tmp_path / "figs_conv_excl_unconverged"
    assert vdir.is_dir() and written
    rows = [r for r in _read_csv(vdir / "holdout_ed_combined.csv")
            if r["leg"] == "wtmad2"]
    assert rows
    for r in rows:
        assert float(r["D_pbe_rmse"]) == pytest.approx(
            _expected_d_pbe("density_rmse_pbe", _KEPT_CF), rel=1e-9)
    for sd in sorted((run / "checkpoints").glob("spec_*")):
        pm = sd / "eval_holdout_converged" / "per_molecule.json"
        if pm.is_file():
            payload = json.loads(pm.read_text())
            for r in payload:
                r["scf_converged"] = True
            pm.write_text(json.dumps(payload))
    fdir2 = tmp_path / "figs_conv2"
    written2 = fig._build_outlier_free_variants(
        run, fdir2, eval_subdir="eval_holdout_converged")
    # the tail variant of this fixture (NO above in every cell) still renders;
    # the unconverged one must not
    assert not [p for p in written2 if "figs_conv2_excl_unconverged" in str(p)]
    assert not (tmp_path / "figs_conv2_excl_unconverged").exists()


def test_insample_panel_helpers_take_channel_keywords():
    """The panel bodies are reusable on the eps channel; their defaults stay
    the strings the shipped figures carry today."""
    plt = pytest.importorskip("matplotlib.pyplot")
    rows = [{"arch": "deep", "subset_size": s, "molecule": m,
             "density_rmse": v, "density_rmse_pbe": 2.0 * v,
             "density_eps_l1": 10.0 * v, "density_eps_l1_pbe": 20.0 * v}
            for s in (1, 3)
            for m, v in (("HO", 3e-3), ("CH4", 1e-3))]
    mfig, ax = plt.subplots()
    try:
        fig._insample_density_lines_panel(ax, rows)
        assert ax.get_xlabel() == "training subset_size"
        assert ax.get_ylabel() == "density RMSE vs CCSD (grid, weighted-mean)"
        assert ax.get_title() == "In-sample density fit vs CCSD (per arch)"
        assert "PBE vs CCSD" in ax.get_legend_handles_labels()[1]
    finally:
        plt.close(mfig)
    mfig, ax = plt.subplots()
    try:
        fig._insample_density_lines_panel(
            ax, rows, key="density_eps_l1", pbe_key="density_eps_l1_pbe",
            xlabel="ss", ylabel="eps", title="eps lines", pbe_label="PBE eps")
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
            "ss", "eps", "eps lines")
        assert "PBE eps" in ax.get_legend_handles_labels()[1]
        ys = [float(y) for line in ax.get_lines() for y in line.get_ydata()]
        # per-subset means on the eps channel: NN (0.03+0.01)/2, PBE
        # (0.06+0.02)/2 -- the default channel would top out at 0.004
        assert max(ys) == pytest.approx(0.04)
        assert min(ys) == pytest.approx(0.02)
    finally:
        plt.close(mfig)
    mfig, ax = plt.subplots()
    try:
        fig._insample_density_strip_panel(ax, rows)
        assert ax.get_ylabel() == "density RMSE vs CCSD"
        assert ax.get_title() == ("Per-molecule (every point; 2 trained "
                                  "species)")
    finally:
        plt.close(mfig)
    mfig, ax = plt.subplots()
    try:
        fig._insample_density_strip_panel(
            ax, rows, key="density_eps_l1", pbe_key="density_eps_l1_pbe",
            ylabel="eps strip", title_fmt="eps over {n} species")
        assert ax.get_ylabel() == "eps strip"
        assert ax.get_title() == "eps over 2 species"
    finally:
        plt.close(mfig)


# ---------------------------------------------------------------------------
# The recurring held-out tail variant (2026-09-08): the per-cell ratio
# reduction the tail table and the variant share, the majority rule over the
# cells that carry a species, and the `<fdir>_excl_tail` directory
# ---------------------------------------------------------------------------

def _ratio_rows():
    """Held-out rows for the per-cell reduction: a case-twin pair whose NN
    legs differ (so a sum and a mean are distinguishable), a species below
    the tail floor, and five rows the reduction must drop."""
    return [
        {"arch": "deep", "subset_size": 1, "molecule": "H2",
         "density_rmse": 1.0e-3, "density_rmse_pbe": 4.0e-4},
        {"arch": "deep", "subset_size": 1, "molecule": "h2",
         "density_rmse": 3.0e-3, "density_rmse_pbe": 4.0e-4},
        {"arch": "deep", "subset_size": 1, "molecule": "CO",
         "density_rmse": 1.0e-3, "density_rmse_pbe": 8.0e-4},   # ratio 1.25
        {"arch": "deep", "subset_size": 1, "molecule": "N2",
         "density_rmse": None, "density_rmse_pbe": 2.0e-3},     # no NN leg
        {"arch": "deep", "subset_size": 1, "molecule": "O2",
         "density_rmse": float("nan"), "density_rmse_pbe": 2.0e-3},
        {"arch": "deep", "subset_size": 1, "molecule": "F2",
         "density_rmse": 5.0e-3, "density_rmse_pbe": 0.0},      # PBE leg 0
        {"arch": "deep", "subset_size": 1, "molecule": "Cl2",
         "density_rmse": 5.0e-3, "density_rmse_pbe": None},     # no PBE leg
        {"arch": "deep", "subset_size": 1, "molecule": None,
         "density_rmse": 9.0e-3, "density_rmse_pbe": 1.0e-3},   # nameless
        {"arch": None, "subset_size": 1, "molecule": "NO",
         "density_rmse": 9.0e-3, "density_rmse_pbe": 1.0e-3},   # no cell
        {"arch": "deep", "subset_size": 3, "molecule": "NO",
         "density_rmse": 4.0e-3, "density_rmse_pbe": 1.0e-3},   # ratio 4.0
    ]


def test_cell_species_density_ratios_collapses_twins_and_drops_broken_legs(
        tmp_path):
    """The per-cell NN/PBE density-RMSE reduction, keyed on (arch,
    subset_size, casefolded species): the case twins collapse to ONE key
    whose NN and PBE legs are the twin MEANS -- a sum on both legs leaves the
    ratio right and both legs wrong -- and a row with no finite NN leg, no
    PBE leg, a non-positive PBE leg, no name or no cell contributes nothing.
    The reduction carries no ratio floor of its own (the below-floor species
    is present); the floor belongs to its callers. The tail table is the same
    reduction read through that floor: the rows it writes are exactly the
    keys above 1.5, each carrying this helper's ratio."""
    got = fig.cell_species_density_ratios(_ratio_rows())
    assert set(got) == {("deep", 1, "h2"), ("deep", 1, "co"),
                        ("deep", 3, "no")}
    twin = got[("deep", 1, "h2")]
    assert twin["nn"] == pytest.approx(2.0e-3)      # the mean, not the 4e-3 sum
    assert twin["pbe"] == pytest.approx(4.0e-4)     # the mean, not the 8e-4 sum
    assert twin["ratio"] == pytest.approx(5.0)
    assert got[("deep", 1, "co")]["ratio"] == pytest.approx(1.25)
    assert got[("deep", 3, "no")]["ratio"] == pytest.approx(4.0)
    out = tmp_path / "tail.csv"
    fig.write_holdout_density_tail_csv(_ratio_rows(), out)
    written = {(r["arch"], int(r["subset_size"]), r["molecule"].casefold()):
               float(r["ratio"]) for r in _read_csv(out)}
    above = {k: v["ratio"] for k, v in got.items() if v["ratio"] > 1.5}
    assert set(written) == set(above), (sorted(written), sorted(above))
    for key, ratio in written.items():
        assert ratio == pytest.approx(above[key], rel=1e-12), key


def _tail_rule_rows(cells, above, present=None):
    """One row per (cell, species): the NN leg sits at 5x the PBE leg in the
    cell indices listed in ``above[species]`` and at 1x elsewhere.
    ``present`` restricts a species to the cell indices it names."""
    rows = []
    for ci, (arch, ss) in enumerate(cells):
        for sp, hot in above.items():
            if present is not None and ci not in present.get(
                    sp, range(len(cells))):
                continue
            rows.append({"arch": arch, "subset_size": ss, "molecule": sp,
                         "density_rmse": 5.0e-3 if ci in hot else 1.0e-3,
                         "density_rmse_pbe": 1.0e-3})
    return rows


_RULE_CELLS_3 = (("deep", 1), ("deep", 3), ("deep_notransform", 1))


def test_recurring_tail_species_needs_a_majority_of_the_cells_carrying_it():
    """The rule the variant is selected by: a species joins the recurring
    tail when its per-cell ratio exceeds the floor in at least
    ``cell_fraction`` of the cells that CARRY it (ceil(fraction x n_cells)),
    and the second return value is the number of cells the rows describe.
    NO is above in two of three cells and N2 in three of three (both
    recurring); CO is above in one of three (a one-off, kept); BN is carried
    by one cell and above there, a single observation, kept, and CO2 is carried
    by two cells and above in one (half of them, one observation), kept too: a
    recurring species is above in at least two cells. The H2/h2 pair is above the floor on one
    spelling in every cell and below it once the twins are collapsed, so a
    rule keyed on raw names reports the pair as recurring."""
    rows = _tail_rule_rows(_RULE_CELLS_3,
                           {"NO": (0, 1), "CO": (0,), "N2": (0, 1, 2),
                            "BN": (0,), "CO2": (0,)},
                           present={"BN": (0,), "CO2": (0, 1)})
    for ci, (arch, ss) in enumerate(_RULE_CELLS_3):
        rows += [
            {"arch": arch, "subset_size": ss, "molecule": "H2",
             "density_rmse": 2.0e-4, "density_rmse_pbe": 4.0e-4},
            {"arch": arch, "subset_size": ss, "molecule": "h2",
             "density_rmse": 8.0e-4, "density_rmse_pbe": 4.0e-4},
        ]
    assert fig.recurring_tail_species(rows) == ({"no", "n2"}, 3)
    # both keywords are parameters of the rule, not constants inside it
    assert fig.recurring_tail_species(rows, cell_fraction=1.0) == (
        {"n2"}, 3)
    assert fig.recurring_tail_species(rows, ratio_threshold=6.0) == (set(), 3)


def test_recurring_tail_species_rounds_the_fraction_up_not_to_even():
    """ceil, not round: at five carrying cells half is 2.5, which ceil takes to 3
    and half-to-even rounding to 2, so a species above in two of five is kept
    by the rule and would be excluded by a rounded one."""
    cells = tuple(("deep", ss) for ss in (1, 2, 3, 4, 5))
    rows = _tail_rule_rows(cells, {"NO": (0, 1)}, present={"NO": (0, 1, 2, 3, 4)})
    assert fig.recurring_tail_species(rows) == (set(), 5)
    rows = _tail_rule_rows(cells, {"NO": (0, 1, 2)}, present={"NO": (0, 1, 2, 3, 4)})
    assert fig.recurring_tail_species(rows) == ({"no"}, 5)


def test_recurring_tail_species_counts_cells_by_the_finite_nn_leg():
    """The cell count is the builder's: a cell whose only species carries a
    finite NN leg and no PBE leg draws a cell in the figures and counts here,
    while it produces no ratio."""
    rows = _tail_rule_rows(_RULE_CELLS_3, {"NO": (0, 1)})
    rows.append({"arch": "deep", "subset_size": 9, "molecule": "CO",
                 "density_rmse": 1.0e-3, "density_rmse_pbe": None})
    assert fig.recurring_tail_species(rows) == ({"no"}, 4)


def test_recurring_tail_species_includes_a_species_above_in_exactly_half():
    """Kills the mutation `cell_fraction compared with > instead of >=`: a
    species above the floor in two of the four cells that carry it sits at
    exactly the default fraction, and the rule reads `at least half`. Raising
    the fraction to three quarters drops it again."""
    cells = (("deep", 1), ("deep", 3), ("deep_notransform", 1),
             ("deep_notransform", 3))
    rows = _tail_rule_rows(cells, {"NO": (0, 1)})
    assert fig.recurring_tail_species(rows) == ({"no"}, 4)
    assert fig.recurring_tail_species(rows, cell_fraction=0.75) == (set(), 4)


def _tail_variant_run_dir(root: Path, above=(0, 1, 2, 3)) -> Path:
    """The rich run with NO -- the constructed outlier, 7x the PBE leg -- held
    above the tail floor in the cell indices named by ``above`` and pushed to
    half the PBE leg in the others. The PBE columns are untouched, so the
    cross-spec consistency guard keeps every species."""
    run = _rich_run_dir(root)
    for sd in sorted((run / "checkpoints").glob("spec_*")):
        idx = int(sd.name.split("_")[1])
        pm = sd / "eval_holdout" / "per_molecule.json"
        if not pm.is_file() or idx in above:
            continue
        payload = json.loads(pm.read_text())
        for r in payload:
            if str(r.get("molecule")).casefold() == "no":
                r["density_rmse"] = 5.0e-4 * _cell_factor(idx)
        pm.write_text(json.dumps(payload))
    return run


def test_tail_variant_directory_drops_the_recurring_species(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """The third variant: NO sits above 1.5x PBE in all four cells of the
    rich fixture, so `<fdir>_excl_tail` renders with NO dropped from every
    cell. Kills the mutation `variant suffix misspelled` (the directory is
    named, and it is the only variant directory written) and the mutation
    `exclusion applied to the PBE table only` (the NN cell means move too,
    each to the twin-then-species mean over the surviving species). The
    footer note names the rule, the count and the species, and the variant's
    own tail table is empty once its cause is gone."""
    run = _tail_variant_run_dir(tmp_path)
    stamped = []
    real_stamp = fig._stamp_parity_footer

    def _spy(mpl_fig, **kw):
        stamped.append(kw)
        return real_stamp(mpl_fig, **kw)

    monkeypatch.setattr(fig, "_stamp_parity_footer", _spy)
    fdir = tmp_path / "figs"
    written = fig._build_outlier_free_variants(run, fdir)
    vdir = tmp_path / "figs_excl_tail"
    assert vdir.is_dir(), sorted(p.name for p in tmp_path.iterdir())
    assert [p.name for p in tmp_path.iterdir() if "_excl_" in p.name] == [
        "figs_excl_tail"]
    assert written and all(vdir in Path(p).parents for p in written)
    assert (vdir / "holdout_density_vs_ccsd.png").is_file()

    var = _by_leg_cell(_read_csv(vdir / "holdout_ed_combined.csv"))
    for (arch, ss), idx in _RICH_CELLS.items():
        row = var[("wtmad2", arch, ss)]
        want_nn = _expected_d_rmse(idx, _KEPT_CF)
        assert want_nn != pytest.approx(_expected_d_rmse(idx)), \
            "oracle must discriminate"
        assert float(row["D_rmse"]) == pytest.approx(want_nn, rel=1e-9)
        want_pbe = _expected_d_pbe("density_rmse_pbe", _KEPT_CF)
        assert want_pbe != pytest.approx(_expected_d_pbe("density_rmse_pbe")), \
            "oracle must discriminate"
        assert float(row["D_pbe_rmse"]) == pytest.approx(want_pbe, rel=1e-9)
    assert _read_csv(vdir / "holdout_density_tail.csv") == []

    notes = [kw.get("note") or "" for kw in stamped]
    carrying = [n for n in notes if "excluded in every cell" in n]
    assert carrying, notes[:3]
    note = carrying[0]
    assert "recurring held-out tail" in note, note
    assert "1.5" in note and "at least half" in note, note
    assert re.search(r"at least half [(]and at least two[)] of the cells carrying the species; "
                     r"the run renders 4 cells", note), note
    assert "diagnostic view" in note, note
    assert "1 of 4 held-out species" in note and "NO" in note, note
    out = capsys.readouterr().out
    assert "excl_tail" in out, out


def test_tail_variant_skipped_when_no_species_recurs(tmp_path, capsys):
    """A one-off outlier is not a tail: NO is above 1.5x PBE in one of the
    four cells, below it in the other three, so the rule selects nothing, no
    directory is written and the skip line names the variant. Without the
    printed line this test would pass on a build that never considered the
    variant at all."""
    run = _tail_variant_run_dir(tmp_path, above=(0,))
    pre = tmp_path / "precheck.csv"
    fig.write_holdout_density_tail_csv(
        fig.collect_holdout_density_rows(run), pre)
    assert [r["molecule"].casefold() for r in _read_csv(pre)] == ["no"], \
        "the fixture must carry a one-off outlier, above the floor once"
    capsys.readouterr()                       # drop the precheck's own output
    written = fig._build_outlier_free_variants(run, tmp_path / "figs")
    out = capsys.readouterr().out
    assert written == []
    assert not [p.name for p in tmp_path.iterdir() if "_excl_" in p.name]
    assert "excl_tail" in out, out
    assert "nothing to exclude" in out, out


# ---------------------------------------------------------------------------
# The in-sample 3x3 in DFS units (2026-09-08): WTMAD-2 / eps / ED per pool on
# the cells' OWN training reactions
# ---------------------------------------------------------------------------

# Element anchors of the training record; a reaction whose PRODUCTS are all
# anchors is an atomization (w411), otherwise a barrier (bh76).
_INS_ATOM_ENERGIES = {"H": -0.5, "O": -75.0, "C": -37.8}

# Model-free PBE species energies (Ha), spec-invariant as the real ones are.
# 'h' (the BH76 spelling of the hydrogen atom) differs from the W4-11 'H', so
# a casefolding species lookup cannot reproduce the barrier's energy.
_INS_E_PBE = {"HO": -75.62, "H": -0.505, "O": -75.01, "h": -0.49,
              "HOh_ts": -76.06, "CH4": -40.30, "C": -37.79}

# Eq. 20 eps channels of the in-sample rows. The H2/h2 twins carry DIFFERENT
# NN values, so a row mean and a twin-then-species mean differ.
_INS_NN_EPS = {"HO": 2.0e-3, "CH4": 4.0e-3, "H2": 1.0e-3, "h2": 3.0e-3}
_INS_PBE_EPS = {"HO": 5.0e-3, "CH4": 6.0e-3, "H2": 4.0e-3, "h2": 4.0e-3}
_INS_ALL_CF = frozenset(m.casefold() for m in _INS_NN_EPS)

# The variant excludes an in-sample species as well as the held-out outlier:
# without CH4 in the set, "the in-sample rows are unfiltered" would be
# untestable on this fixture (nothing excluded reaches them).
_INS_EXCLUDED_CF = frozenset({"no", "ch4"})
_INS_KEPT_CF = _INS_ALL_CF - _INS_EXCLUDED_CF
_INS_VARIANT_NOTE = ("2 species excluded in every cell (T1 > 0.02, Lee-Taylor "
                     "1989): ['CH4', 'NO']")

_INS_KCAL = 627.5094740631        # CODATA-2018 Hartree -> kcal/mol
_INS_WTMAD2_SCALE = 56.84         # GMTKN55 global mean |dE| (Goerigk 2017)

_SS_BY_IDX = {idx: ss for (_arch, ss), idx in _RICH_CELLS.items()}

_INS_RXN_HO = {"name": "HO", "reactants": ["HO"], "products": ["H", "O"],
               "coeffs": [-1.0, 1.0, 1.0], "e_rxn_ref": 0.20}
_INS_RXN_CH4 = {"name": "CH4", "reactants": ["CH4"], "products": ["C", "H"],
                "coeffs": [-1.0, 1.0, 4.0], "e_rxn_ref": 0.65}
_INS_RXN_TS = {"name": "OH+h_to_HOh_ts", "reactants": ["HO", "h"],
               "products": ["HOh_ts"], "coeffs": [-1.0, -1.0, 1.0],
               "e_rxn_ref": 0.03}


def _ins_e_nn(idx):
    """One cell's self-consistent species energies (Ha). The molecules drift
    with the spec index (the atoms do not), so the four cells carry different
    reaction errors and a per-cell reduction is distinguishable from a pooled
    one."""
    return {"HO": -75.65 - 0.002 * idx, "H": -0.50, "O": -75.00, "h": -0.48,
            "HOh_ts": -76.09 - 0.001 * idx, "CH4": -40.33 - 0.003 * idx,
            "C": -37.80}


def _ins_reactions(idx):
    """The cell's OWN training reactions: the ss=1 cells train on HO alone
    (one atomization + one barrier), the ss=3 cells add the CH4 atomization.
    The cells therefore do NOT share a reaction set, as in a real subset-size
    sweep -- a leg reduced over the run's union instead of the cell's own
    reactions lands on different numbers."""
    return ([_INS_RXN_HO, _INS_RXN_TS] if _SS_BY_IDX[idx] == 1
            else [_INS_RXN_HO, _INS_RXN_CH4, _INS_RXN_TS])


def _ins_pool(rxn):
    return ("w411" if all(p in _INS_ATOM_ENERGIES for p in rxn["products"])
            else "bh76")


def _insample_rows_with_energies(idx):
    """``_rich_insample_rows`` plus the two columns the in-sample 3x3 needs:
    the Eq. 20 eps density channels and the per-species self-consistent
    energies the energy leg is formed from."""
    e_nn, e_pbe = _ins_e_nn(idx), _INS_E_PBE
    f = _cell_factor(idx)
    rows = [dict(r) for r in _rich_insample_rows(idx)]
    for r in rows:
        m = r.get("molecule")
        if m in _INS_NN_EPS:
            r["density_eps_l1"] = _INS_NN_EPS[m] * f
            r["density_eps_l1_pbe"] = _INS_PBE_EPS[m]
        if m in e_nn:
            r["E_total_nn"] = e_nn[m]
        if m in e_pbe:
            r["E_pbe"] = e_pbe[m]
    have = {r.get("molecule") for r in rows}
    for m in sorted(set(e_nn) | set(e_pbe)):
        if m in have:
            continue
        # atoms and the transition state carry no density channels
        rows.append({"molecule": m, "skipped": m in _INS_ATOM_ENERGIES,
                     "AE_error_kcalmol": None, "density_rmse": None,
                     "E_total_nn": e_nn.get(m), "E_pbe": e_pbe.get(m)})
    return rows


def _insample_run_dir(root: Path) -> Path:
    """The rich fixture with the in-sample energy layer written over it: each
    spec's training record carries the cell's own reactions, the element
    anchors and an IP13 pair (a reaction of neither pool), and its
    ``eval/per_molecule.json`` the species energies beside the eps columns.

    A SIBLING of :func:`_rich_run_dir` on purpose: those eps columns fill
    ``D_insample_rmse`` on the eps legs of ``holdout_ed_combined.csv``, which
    the standard fixture's byte goldens pin blank."""
    run = _rich_run_dir(root)
    for sd in sorted((run / "checkpoints").glob("spec_*")):
        idx = int(sd.name.split("_")[1])
        if not (sd / "eval_holdout" / "per_reaction.json").is_file():
            continue
        tm = sd / "train_metadata.json"
        meta = json.loads(tm.read_text())
        meta["atom_energies"] = dict(_INS_ATOM_ENERGIES)
        meta["loss_kwargs"] = {
            "bh76_reactions": _ins_reactions(idx),
            "ip13_pairs": [{"name": "Li_IP", "neutral": "Li",
                            "cation": "Li+", "ip_ref": 0.198}],
        }
        tm.write_text(json.dumps(meta))
        (sd / "eval" / "per_molecule.json").write_text(
            json.dumps(_insample_rows_with_energies(idx)))
    return run


# -- independent oracles (plain python; no reducer from the module) ---------

def _expected_ins_e(idx):
    """2-subset WTMAD-2 over the cell's own training reactions:
    (scale/N) * sum_pool N_pool * MAD_pool / mean|ref|_pool, with the reaction
    energies formed from the signed coefficients over reactants-then-products
    and both legs converted to kcal/mol."""
    e = _ins_e_nn(idx)
    pools = {}
    for rxn in _ins_reactions(idx):
        species = list(rxn["reactants"]) + list(rxn["products"])
        de = sum(c * e[s]
                 for c, s in zip(rxn["coeffs"], species)) * _INS_KCAL
        ref = rxn["e_rxn_ref"] * _INS_KCAL
        pools.setdefault(_ins_pool(rxn), []).append((abs(de - ref), abs(ref)))
    n_total = sum(len(v) for v in pools.values())
    acc = 0.0
    for vals in pools.values():
        mad = sum(a for a, _ in vals) / len(vals)
        mean_ref = sum(r for _, r in vals) / len(vals)
        acc += len(vals) * mad / mean_ref
    return _INS_WTMAD2_SCALE / n_total * acc


def _expected_ins_d_eps(idx, keep=None):
    """Twin-then-species mean of the in-sample Eq. 20 eps column."""
    f = _cell_factor(idx)
    return _species_mean({m: v * f for m, v in _INS_NN_EPS.items()}, keep)


class _InsBuilds:
    def __init__(self):
        self.run = None
        self.std = None
        self.var = None
        self.deep = None
        self.out = {}
        # footer kwargs per run (the _stamp_parity_footer spy of `builds`)
        # and the (file name, yscale) of every plot_density_energy_3x3 call,
        # so the in-sample 3x3's note and its _logy twin's scale are
        # assertable without a second render
        self.notes = {}
        self.calls3x3 = {}
        self.error = None

    def require(self):
        if self.error is not None:
            raise self.error


@pytest.fixture(scope="module")
def ins_builds(tmp_path_factory):
    """Three builder runs over the in-sample fixture: the standard directory,
    an outlier-free variant whose exclusion set names an in-sample species,
    and an architecture-restricted directory."""
    b = _InsBuilds()
    root = tmp_path_factory.mktemp("insample")
    b.run = _insample_run_dir(root)
    b.std, b.var, b.deep = root / "std", root / "var", root / "deep"
    real_stamp = fig._stamp_parity_footer
    real_3x3 = fig.plot_density_energy_3x3
    bucket = {"which": "std"}

    def _spy_stamp(mpl_fig, **kw):
        b.notes.setdefault(bucket["which"], []).append(kw)
        return real_stamp(mpl_fig, **kw)

    def _spy_3x3(rows, hd_rows, out_path, run_id, **kw):
        b.calls3x3.setdefault(bucket["which"], []).append(
            (Path(out_path).name, kw.get("yscale", "linear"),
             kw.get("note") or "", kw.get("title") or ""))
        return real_3x3(rows, hd_rows, out_path, run_id, **kw)

    fig._stamp_parity_footer = _spy_stamp
    fig.plot_density_energy_3x3 = _spy_3x3
    try:
        for which, outdir, kw in (
                ("std", b.std, {}),
                ("var", b.var, {"exclude_cf": _INS_EXCLUDED_CF,
                                "variant_note": _INS_VARIANT_NOTE}),
                ("deep", b.deep, {"archs": ("deep",)})):
            bucket["which"] = which
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    fig.build_density_energy_figures(b.run, outdir, **kw)
            finally:
                b.out[which] = buf.getvalue()
    except Exception as exc:            # RED until the in-sample 3x3 lands
        b.error = exc
    finally:
        fig._stamp_parity_footer = real_stamp
        fig.plot_density_energy_3x3 = real_3x3
    return b


_INS_LEG = "combined_wtmad2_eps_gamma_dfs"


def test_insample_by_pool_3x3_eps_energy_leg_is_the_training_reactions(
        ins_builds):
    """Kills the mutations `the 3x3 written from the held-out rows`,
    `coefficients applied to the reactants only`, `e_rxn_ref left in Hartree`
    and `the pool rule inverted`: the combined leg's E per cell is the
    plain-python 2-subset WTMAD-2 over THAT cell's training reactions, and its
    D the twin-collapsed mean of the in-sample eps column -- neither is the
    held-out value for the same cell."""
    ins_builds.require()
    png = ins_builds.std / "insample_by_pool_3x3_eps.png"
    logy = ins_builds.std / "insample_by_pool_3x3_eps_logy.png"
    table = ins_builds.std / "insample_by_pool_3x3_eps.csv"
    assert _png_ok(png), sorted(p.name for p in ins_builds.std.iterdir())
    assert _png_ok(logy)
    assert table.is_file()

    got = _by_leg_cell(_read_csv(table))
    holdout = _by_leg_cell(_read_csv(ins_builds.std
                                     / "holdout_by_pool_3x3_eps.csv"))
    # the oracle discriminates the cells: four different energies, four
    # different densities (a single pooled number would satisfy neither)
    assert len({round(_expected_ins_e(i), 9)
                for i in _RICH_CELLS.values()}) == 4
    assert len({round(_expected_ins_d_eps(i), 12)
                for i in _RICH_CELLS.values()}) == 4
    for (arch, ss), idx in _RICH_CELLS.items():
        row = got[(_INS_LEG, arch, ss)]
        assert float(row["E_kcalmol"]) == pytest.approx(
            _expected_ins_e(idx), rel=1e-9), (arch, ss)
        assert float(row["D_rmse"]) == pytest.approx(
            _expected_ins_d_eps(idx), rel=1e-9), (arch, ss)
        # the held-out leg of the same cell is a different number, so the
        # in-sample table cannot have been rendered from the held-out rows
        hd = holdout[(_INS_LEG, arch, ss)]
        assert float(row["E_kcalmol"]) != pytest.approx(
            float(hd["E_kcalmol"]), rel=1e-6)
        assert float(row["D_rmse"]) != pytest.approx(
            float(hd["D_rmse"]), rel=1e-6)


def test_insample_by_pool_3x3_eps_skipped_without_training_reactions(builds):
    """A run whose training record carries no reactions has no in-sample
    energy leg: nothing is written and the builder says so. Without the
    printed line this would pass on a build that never considered the
    figure."""
    builds.require()
    for name in ("insample_by_pool_3x3_eps.png",
                 "insample_by_pool_3x3_eps_logy.png",
                 "insample_by_pool_3x3_eps.csv"):
        assert not (builds.std / name).exists(), name
        assert not (builds.var / name).exists(), name
    said = [ln for ln in builds.out["std"].splitlines()
            if re.search(r"(?i)in-?sample", ln)
            and re.search(r"(?i)skip", ln)]
    assert said, builds.out["std"]


def test_insample_by_pool_3x3_eps_is_unfiltered_by_the_variant(ins_builds):
    """Kills the mutation `the in-sample rows filtered by exclude_cf`: every
    in-sample figure reads the cells' full training set, so the variant
    directory's in-sample table is byte-identical to the standard one -- while
    its held-out twin, on the same run, is not."""
    ins_builds.require()
    name = "insample_by_pool_3x3_eps.csv"
    assert "ch4" in _INS_EXCLUDED_CF, \
        "the exclusion must reach an in-sample species"
    assert (ins_builds.std / name).read_bytes() == \
        (ins_builds.var / name).read_bytes()
    # the variant run really did exclude something: its held-out twin moved
    assert (ins_builds.std / "holdout_by_pool_3x3_eps.csv").read_bytes() != \
        (ins_builds.var / "holdout_by_pool_3x3_eps.csv").read_bytes()
    # ... and the identity is the UNFILTERED value on both sides, not two
    # equally-filtered tables
    for outdir in (ins_builds.std, ins_builds.var):
        rows = _by_leg_cell(_read_csv(outdir / name))
        for (arch, ss), idx in _RICH_CELLS.items():
            full = _expected_ins_d_eps(idx)
            cut = _expected_ins_d_eps(idx, _INS_KEPT_CF)
            assert full != pytest.approx(cut), "oracle must discriminate"
            assert float(rows[(_INS_LEG, arch, ss)]["D_rmse"]) == \
                pytest.approx(full, rel=1e-9), (outdir.name, arch, ss)


def test_insample_by_pool_3x3_eps_carries_the_plain_note_and_its_logy_scale(
        ins_builds):
    """Two review findings on the first implementation: the variant run
    stamped its exclusion note on the in-sample 3x3 (whose data the
    variant never filters), and the ``_logy`` twin sat outside the module's
    log-scale guard. The in-sample calls of the variant run carry the plain
    note and an ``In-sample`` title (the unfiltered-title rule of
    ``test_rf_variant_note_reaches_only_the_filtered_figures``), and the
    ``_logy`` call carries ``yscale="log"``."""
    ins_builds.require()
    calls = {name: (scale, note, title)
             for name, scale, note, title in ins_builds.calls3x3["var"]}
    assert "insample_by_pool_3x3_eps.png" in calls, sorted(calls)
    assert "insample_by_pool_3x3_eps_logy.png" in calls, sorted(calls)
    for name in ("insample_by_pool_3x3_eps.png",
                 "insample_by_pool_3x3_eps_logy.png"):
        scale, note, title = calls[name]
        assert "excluded in every cell" not in note, (name, note)
        assert title.startswith("In-sample"), (name, title)
    assert calls["insample_by_pool_3x3_eps.png"][0] == "linear"
    assert calls["insample_by_pool_3x3_eps_logy.png"][0] == "log"
    # the held-out twin of the same run DOES carry the exclusion note
    hd = calls["holdout_by_pool_3x3_eps.png"]
    assert "excluded in every cell" in hd[1], hd[1]
    # and the footer the figure stamped says what its comparator lines are
    stamped = [kw for kw in ins_builds.notes["var"]
               if str(kw.get("title", "")).startswith("In-sample per-channel")]
    assert stamped, [kw.get("title") for kw in ins_builds.notes["var"]]
    for kw in stamped:
        assert "excluded in every cell" not in (kw.get("note") or "")
        assert "UNION" in (kw.get("caveat") or ""), kw.get("caveat")
        assert "Capped horizontal spans" in (kw.get("provenance") or "")


def test_insample_eval_note_counts_deduplicated_reactions_and_species():
    """The in-sample dataset line: the run's union of training reactions per
    pool counted once per name (a reaction trained by several cells is one
    reaction), the trained species with an eps value counted once per
    casefolded name, and the checkpoint stated."""
    rows = [{"name": "HO", "pool": "w411", "abs_error_nn_kcalmol": 1.0,
             "ref_kcalmol": 100.0},
            {"name": "HO", "pool": "w411", "abs_error_nn_kcalmol": 2.0,
             "ref_kcalmol": 100.0},
            {"name": "bh76_a", "pool": "bh76", "abs_error_nn_kcalmol": 1.0,
             "ref_kcalmol": 10.0}]
    drows = [{"molecule": "HO", "density_eps_l1": 1e-3},
             {"molecule": "H2", "density_eps_l1": 1e-3},
             {"molecule": "h2", "density_eps_l1": 2e-3},
             {"molecule": "CO", "density_eps_l1": None}]
    note = fig._insample_eval_note(rows, drows)
    assert note.startswith("In-sample (final checkpoint): ")
    assert "BH76 1 + W4-11 1" in note, note
    assert "2 trained species" in note, note
    assert fig._insample_eval_note([], []) == ""


def test_insample_by_pool_3x3_eps_follows_the_arch_restriction(ins_builds):
    """The architecture restriction reaches the in-sample rows as it reaches
    every other row set of this builder: a directory drawn for one arch must
    not carry another arch's cells in its in-sample table (the two directories
    describe the same run and would otherwise disagree on which cells they
    draw)."""
    ins_builds.require()
    rows = _read_csv(ins_builds.deep / "insample_by_pool_3x3_eps.csv")
    assert rows
    assert {r["arch"] for r in rows} == {"deep"}
    assert {int(r["subset_size"]) for r in rows} == {1, 3}
