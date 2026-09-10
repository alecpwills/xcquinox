"""Tests for ``notebooks/analysis/report_tables.py`` (the per-cell tables of the v7 documents).

The module generates the report's and the summary's per-cell tables from the family CSVs and
splices them into the documents between named marker lines, so that the numbers in the prose
tables are the numbers in the files rather than a transcription. The tests below are T1-T3 of
the design page ``scratch/density_audit_2026-09-07/page_documents_merged_set.md``:

* T1 ``read_leg_rows`` / ``holdout_table`` / ``insample_table``: one leg of the CSV, in the
  file's order, rendered with the report's column formats (E two decimals, eps five decimals,
  ED two decimals, dED signed and bold when negative, ``yes``/``no``).
* T2 ``splice``: the block between ``<!-- table:NAME -->`` and ``<!-- /table:NAME -->`` is
  replaced, the markers and everything around them kept, a missing marker refused.
* T3 ``main``: the seven tables over a fixture family directory, written into the marker blocks
  that exist and into no others.

Everything is synthetic: the fixture CSVs are built in ``tmp_path`` from the real header of
``figures_dfs_step7_v7_family_val_best/holdout_by_pool_3x3_eps.csv`` (copied verbatim into
``HEADER`` below) with two architectures at two subset sizes on the three legs. No repository
CSV, no repository document, no network, no jax, no pyscf.

Two properties of the fixture are deliberate and are not properties of the real files:

* the rows are interleaved leg by leg, so that "the file's order" is pinned as the order of the
  leg's own rows and not as a contiguous slice of the file (the real file groups by leg);
* ``n_reactions_slice`` carries a value different from ``n_reactions`` in every row, so that the
  ``n rxn`` column is pinned to a named field. In both real CSVs the two agree in every row
  (0 mismatches over 99 and 92 rows), so real data cannot distinguish them.

The module under test is loaded from its path inside each test, as the sibling
``test_md_to_tex`` does: while ``report_tables.py`` is absent every test reports its own
ImportError instead of the file collapsing into one collection error, so the RED state names
each requirement separately.

Mutation coverage (the page's list): m1 (the dED sign flipped) ->
test_holdout_table_rows_are_the_report_format_with_the_signed_delta and
test_holdout_table_pins_the_second_architecture_rows; m2 (bold on a positive dED) -> the same
two tests; m3 (splice drops the markers) ->
test_splice_keeps_the_markers_and_the_surrounding_text and
test_splice_is_idempotent_on_its_own_output; m5 (main writes every table into every document)
-> test_main_fills_only_the_marker_blocks_that_exist and
test_main_leaves_a_document_without_markers.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "report_tables.py"
_MODNAME = "report_tables"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load_script():
    """Load the module under test from its path, fresh on every call."""
    if not _SCRIPT.exists():
        raise ImportError(f"{_SCRIPT} does not exist: the table generator has not been written")
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
# the fixture CSVs: the real header, two architectures, two subsets, three legs
# ---------------------------------------------------------------------------

# copied verbatim from figures_dfs_step7_v7_family_val_best/holdout_by_pool_3x3_eps.csv
HEADER = ("leg,arch,arch_stored,subset_size,n_reactions,n_density_species,E_kcalmol,D_rmse,"
          "gamma,gammaD_kcalmol,ED_kcalmol,E_pbe_kcalmol,D_pbe_rmse,ED_pbe_kcalmol,beats_pbe,"
          "E_scan_kcalmol,D_scan_rmse,ED_scan_kcalmol,beats_scan,ED_pbe_cell_kcalmol,"
          "ED_scan_cell_kcalmol,n_reactions_slice,D_insample_rmse,n_insample_species")
FIELDS = HEADER.split(",")

# the three legs of both real files (`cut -d, -f1 ... | sort | uniq -c`)
BH76 = "bh76_wtmad2_eps_gamma_dfs"
W411 = "w411_wtmad2_eps_gamma_dfs"
COMBINED = "combined_wtmad2_eps_gamma_dfs"

GAMMA = "1084.87"

# the expected header of the generated held-out table (interface 2 of the page)
TABLE_HEADER = ("| Architecture | Subset | n rxn | n species | E NN | E PBE (pool) | eps NN | "
                "eps PBE (pool) | ED NN | ED PBE (cell) | dED | beats |")
# the part of the header that does not depend on the PBE columns' parenthetical, used where a
# table is only being counted
HEADER_STEM = "| Architecture | Subset | n rxn | n species | E NN |"


def _row(**kw) -> str:
    """One CSV line: the named fields, the untouched ones blank, gamma the family value."""
    values = {name: "" for name in FIELDS}
    values["gamma"] = GAMMA
    unknown = set(kw) - set(FIELDS)
    assert not unknown, f"the fixture names fields the real header does not carry: {unknown}"
    values.update({k: str(v) for k, v in kw.items()})
    return ",".join(values[name] for name in FIELDS)


# the four held-out cells of every leg: deep_3x16 is the shown name whose stored key is
# ``medium``, deep0_3x16 the shown name whose stored key is ``deep_3x16``, so a table printing
# the stored key would print ``medium`` for the first pair.
# (arch, arch_stored, subset_size, n_reactions, n_density_species, n_reactions_slice)
_HOLDOUT_CELLS = [
    ("deep_3x16", "medium", 1, 58, 66, 41),
    ("deep_3x16", "medium", 7, 57, 61, 39),
    ("deep0_3x16", "deep_3x16", 1, 58, 66, 41),
    ("deep0_3x16", "deep_3x16", 7, 57, 61, 39),
]

# per (leg, cell): E_kcalmol, D_rmse, ED_kcalmol, E_pbe_kcalmol, D_pbe_rmse, ED_pbe_cell, beats.
# The BH76 cells 1 and 2 are the real medium rows at subsets 1 and 7 of the family CSV; cell 1
# has a negative dED with beats true and eps above PBE, cell 2 a positive dED with beats false
# and eps below PBE, cell 3 a positive dED, cell 4 a negative one.
_HOLDOUT_NUMBERS = {
    BH76: [
        (21.26724287879999, 0.009345380408212495, 13.731136483086104,
         22.114068697861534, 0.009277848775182807, 13.847707802039247, True),
        (23.60991859962919, 0.008131234567890123, 16.662123579562966,
         22.114068697861534, 0.009277848775182807, 13.939161436181097, False),
        (21.523456789012345, 0.009561234567890123, 14.001234567,
         22.114068697861534, 0.009277848775182807, 13.847707802039247, False),
        (14.641234567890123, 0.010881234567890123, 13.071234567,
         22.114068697861534, 0.009277848775182807, 14.031234567, True),
    ],
    W411: [
        (91.11123456789012, 0.021111234567890123, 61.111234567,
         92.22223456789012, 0.022222234567890123, 62.222234567, True),
        (93.33123456789012, 0.023331234567890123, 63.331234567,
         92.22223456789012, 0.022222234567890123, 62.222234567, False),
        (94.44123456789012, 0.024441234567890123, 64.441234567,
         92.22223456789012, 0.022222234567890123, 62.222234567, False),
        (95.55123456789012, 0.025551234567890123, 61.551234567,
         92.22223456789012, 0.022222234567890123, 62.222234567, True),
    ],
    COMBINED: [
        (71.11123456789012, 0.031111234567890123, 41.111234567,
         72.22223456789012, 0.032222234567890123, 42.222234567, True),
        (73.33123456789012, 0.033331234567890123, 43.331234567,
         72.22223456789012, 0.032222234567890123, 42.222234567, False),
        (74.44123456789012, 0.034441234567890123, 44.441234567,
         72.22223456789012, 0.032222234567890123, 42.222234567, False),
        (75.55123456789012, 0.035551234567890123, 41.551234567,
         72.22223456789012, 0.032222234567890123, 42.222234567, True),
    ],
}

# the four held-out BH76 rows, hand-written from the numbers above with the report's formats:
# E and ED two decimals, eps five decimals, dED = ED NN - ED PBE (cell) signed to two decimals
# and bold when negative, beats yes/no from beats_pbe, the architecture the shown name.
BH76_ROWS = [
    "| deep_3x16 | 1 | 58 | 66 | 21.27 | 22.11 | 0.00935 | 0.00928 | 13.73 | 13.85 | "
    "**-0.12** | yes |",
    "| deep_3x16 | 7 | 57 | 61 | 23.61 | 22.11 | 0.00813 | 0.00928 | 16.66 | 13.94 | "
    "+2.72 | no |",
    "| deep0_3x16 | 1 | 58 | 66 | 21.52 | 22.11 | 0.00956 | 0.00928 | 14.00 | 13.85 | "
    "+0.15 | no |",
    "| deep0_3x16 | 7 | 57 | 61 | 14.64 | 22.11 | 0.01088 | 0.00928 | 13.07 | 14.03 | "
    "**-0.96** | yes |",
]

# the in-sample twin: the same header and the same three leg names (verified against
# figures_dfs_step7_v7_family_val_best/insample_by_pool_3x3_eps.csv), the cells' own training
# reactions and species, so n rxn and n species are small.
_INSAMPLE_CELLS = [
    ("deep_3x16", "medium", 2, 1, 2, 7),
    ("deep_3x16", "medium", 7, 5, 6, 11),
    ("deep0_3x16", "deep_3x16", 2, 1, 2, 7),
    ("deep0_3x16", "deep_3x16", 7, 5, 6, 11),
]

_INSAMPLE_NUMBERS = {
    BH76: [
        (2.6570331685381, 0.008053522653281797, 4.0748546217426025,
         18.603755219635225, 0.007876211567269732, 10.970490150373726, True),
        (3.331234567890123, 0.009331234567890123, 5.331234567,
         18.603755219635225, 0.007876211567269732, 7.891234567, True),
        (7.512345678901234, 0.009912345678901234, 12.221234567,
         18.603755219635225, 0.007876211567269732, 10.970490150373726, False),
        (4.441234567890123, 0.008441234567890123, 6.441234567,
         18.603755219635225, 0.007876211567269732, 7.891234567, True),
    ],
    W411: [
        (51.11123456789012, 0.041111234567890123, 31.111234567,
         52.22223456789012, 0.042222234567890123, 32.222234567, True),
        (53.33123456789012, 0.043331234567890123, 33.331234567,
         52.22223456789012, 0.042222234567890123, 32.222234567, False),
        (54.44123456789012, 0.044441234567890123, 34.441234567,
         52.22223456789012, 0.042222234567890123, 32.222234567, False),
        (55.55123456789012, 0.045551234567890123, 31.551234567,
         52.22223456789012, 0.042222234567890123, 32.222234567, True),
    ],
    COMBINED: [
        (61.11123456789012, 0.051111234567890123, 21.111234567,
         62.22223456789012, 0.052222234567890123, 22.222234567, True),
        (63.33123456789012, 0.053331234567890123, 23.331234567,
         62.22223456789012, 0.052222234567890123, 22.222234567, False),
        (64.44123456789012, 0.054441234567890123, 24.441234567,
         62.22223456789012, 0.052222234567890123, 22.222234567, False),
        (65.55123456789012, 0.055551234567890123, 21.551234567,
         62.22223456789012, 0.052222234567890123, 22.222234567, True),
    ],
}

# the four in-sample BH76 rows, hand-written the same way
INSAMPLE_BH76_ROWS = [
    "| deep_3x16 | 2 | 1 | 2 | 2.66 | 18.60 | 0.00805 | 0.00788 | 4.07 | 10.97 | "
    "**-6.90** | yes |",
    "| deep_3x16 | 7 | 5 | 6 | 3.33 | 18.60 | 0.00933 | 0.00788 | 5.33 | 7.89 | "
    "**-2.56** | yes |",
    "| deep0_3x16 | 2 | 1 | 2 | 7.51 | 18.60 | 0.00991 | 0.00788 | 12.22 | 10.97 | "
    "+1.25 | no |",
    "| deep0_3x16 | 7 | 5 | 6 | 4.44 | 18.60 | 0.00844 | 0.00788 | 6.44 | 7.89 | "
    "**-1.45** | yes |",
]

# eps values unique to one leg of one file, used where a table's presence or absence in a
# document is asserted (a five-decimal eps cannot be a substring of another column)
EPS_HOLDOUT_BH76 = "0.00935"
EPS_HOLDOUT_W411 = "0.02111"
EPS_HOLDOUT_COMBINED = "0.03111"
EPS_INSAMPLE_BH76 = "0.00805"
EPS_INSAMPLE_W411 = "0.04111"
EPS_INSAMPLE_COMBINED = "0.05111"


def _csv_text(cells, numbers, shift: float = 0.0) -> str:
    """A fixture CSV: the real header, the cells' rows with the three legs interleaved.

    ``shift`` moves every ED of the network, so that a second directory (the tail-excluded
    sibling) carries visibly different numbers for the same cells.
    """
    lines = [HEADER]
    for k, (arch, stored, subset, n_rxn, n_species, n_slice) in enumerate(cells):
        for leg in (BH76, W411, COMBINED):
            e, d, ed, e_pbe, d_pbe, ed_pbe_cell, beats = numbers[leg][k]
            lines.append(_row(
                leg=leg, arch=arch, arch_stored=stored, subset_size=subset,
                n_reactions=n_rxn, n_density_species=n_species, n_reactions_slice=n_slice,
                E_kcalmol=repr(e), D_rmse=repr(d), ED_kcalmol=repr(ed + shift),
                E_pbe_kcalmol=repr(e_pbe), D_pbe_rmse=repr(d_pbe),
                ED_pbe_kcalmol=repr(ed_pbe_cell), ED_pbe_cell_kcalmol=repr(ed_pbe_cell),
                # the suite's flag is the cell comparison; a shifted ED keeps it consistent
                beats_pbe=str((ed + shift) < ed_pbe_cell if shift else beats)))
    return "\n".join(lines) + "\n"


def _family_dir(tmp_path, name="figures_family_val_best", shift=0.0):
    """A fixture family directory carrying the two CSVs under their real names."""
    d = Path(tmp_path) / name
    d.mkdir(exist_ok=True)
    (d / "holdout_by_pool_3x3_eps.csv").write_text(_csv_text(_HOLDOUT_CELLS, _HOLDOUT_NUMBERS,
                                                             shift))
    (d / "insample_by_pool_3x3_eps.csv").write_text(_csv_text(_INSAMPLE_CELLS, _INSAMPLE_NUMBERS,
                                                              shift))
    return d


def _holdout_csv(tmp_path) -> Path:
    return _family_dir(tmp_path) / "holdout_by_pool_3x3_eps.csv"


def _insample_csv(tmp_path) -> Path:
    return _family_dir(tmp_path) / "insample_by_pool_3x3_eps.csv"


def _table_lines(table: str):
    """The table's markdown rows, in order."""
    return [ln for ln in table.splitlines() if ln.startswith("|")]


def _cells(row: str):
    return [c.strip() for c in row.strip().strip("|").split("|")]


# ---------------------------------------------------------------------------
# T1  read_leg_rows: one leg, the file's order
# ---------------------------------------------------------------------------

def test_read_leg_rows_returns_one_leg_in_the_files_order(tmp_path):
    """Only the leg's rows, in the order the file carries them, with the file's own fields."""
    mod = _load_script()
    rows = mod.read_leg_rows(_holdout_csv(tmp_path), BH76)

    assert len(rows) == 4, "the four cells of the leg and nothing from the other two legs"
    assert {r["leg"] for r in rows} == {BH76}
    assert [r["arch"] for r in rows] == ["deep_3x16", "deep_3x16", "deep0_3x16", "deep0_3x16"], \
        "the file's order, not the alphabetical one (deep0_3x16 sorts before deep_3x16)"
    assert [str(r["subset_size"]) for r in rows] == ["1", "7", "1", "7"]
    assert set(FIELDS) <= set(rows[0]), "the rows keep the CSV's own field names"
    assert str(rows[0]["arch_stored"]) == "medium", "the stored key travels with the row"


def test_read_leg_rows_of_the_other_legs_are_their_own(tmp_path):
    mod = _load_script()
    csv_path = _holdout_csv(tmp_path)
    w411 = mod.read_leg_rows(csv_path, W411)
    combined = mod.read_leg_rows(csv_path, COMBINED)
    assert len(w411) == 4 and len(combined) == 4
    assert {r["leg"] for r in w411} == {W411}
    assert {r["leg"] for r in combined} == {COMBINED}
    assert float(w411[0]["E_kcalmol"]) != float(combined[0]["E_kcalmol"])


def test_read_leg_rows_of_an_absent_leg_yields_nothing(tmp_path):
    """A leg name the file does not carry must not return another leg's rows."""
    mod = _load_script()
    try:
        rows = mod.read_leg_rows(_holdout_csv(tmp_path), "no_such_leg")
    except (KeyError, ValueError):
        return
    assert rows == [], "an absent leg yields no rows (or is refused)"


# ---------------------------------------------------------------------------
# T1  holdout_table: the columns, the formats, the signed dED, the bold, yes/no
# ---------------------------------------------------------------------------

def test_holdout_table_header_and_row_order(tmp_path):
    mod = _load_script()
    table = mod.holdout_table(_holdout_csv(tmp_path), BH76, title="Held-out per cell, BH76 leg")
    lines = _table_lines(table)

    assert lines[0] == TABLE_HEADER, "the twelve columns of the page, in its order"
    assert _cells(lines[1]) == ["---"] * 12, "the markdown separator row, one cell per column"
    assert len(lines) == 2 + 4, "every cell of the leg is a row (header, separator, four cells)"
    assert lines[2:] == BH76_ROWS, "the rows are the file's order, deep_3x16 before deep0_3x16"


def test_holdout_table_rows_are_the_report_format_with_the_signed_delta(tmp_path):
    """The first two cells, hand-written: E two decimals, eps five, ED two, dED signed.

    Kills m1 (the dED sign flipped: +0.12 instead of -0.12) and m2 (bold on a positive dED:
    the second row would carry ``**+2.72**``).
    """
    mod = _load_script()
    table = mod.holdout_table(_holdout_csv(tmp_path), BH76, title="Held-out per cell, BH76 leg")
    lines = _table_lines(table)

    assert lines[2] == BH76_ROWS[0]
    cells = _cells(lines[2])
    assert cells[0] == "deep_3x16", "the Architecture column is the shown name"
    assert cells[1] == "1"
    assert cells[2] == "58" and cells[3] == "66", \
        "n rxn and n species are n_reactions and n_density_species, not n_reactions_slice (41)"
    assert cells[4] == "21.27" and cells[5] == "22.11", "E and E PBE at two decimals"
    assert cells[6] == "0.00935" and cells[7] == "0.00928", "eps at five decimals, NN above PBE"
    assert cells[8] == "13.73" and cells[9] == "13.85", "ED and ED PBE (cell) at two decimals"
    assert cells[10] == "**-0.12**", "a negative dED is signed and bold"
    assert cells[11] == "yes", "beats_pbe True prints yes"

    assert lines[3] == BH76_ROWS[1]
    second = _cells(lines[3])
    assert second[10] == "+2.72", "a positive dED carries its sign and is not bold"
    assert "**" not in second[10]
    assert float(second[6]) < float(second[7]), "this cell's eps is below PBE's"
    assert second[11] == "no", "beats_pbe False prints no"


def test_holdout_table_pins_the_second_architecture_rows(tmp_path):
    """The deep0_3x16 pair, hand-written; the stored key ``deep_3x16`` is never printed."""
    mod = _load_script()
    table = mod.holdout_table(_holdout_csv(tmp_path), BH76, title="Held-out per cell, BH76 leg")
    lines = _table_lines(table)
    assert lines[4] == BH76_ROWS[2]
    assert lines[5] == BH76_ROWS[3]
    assert _cells(lines[4])[10] == "+0.15"
    assert _cells(lines[5])[10] == "**-0.96**"
    assert "medium" not in table, \
        "the stored keys (medium, deep_3x16) are not the shown names of these rows"


def test_holdout_table_carries_the_title_before_the_table(tmp_path):
    mod = _load_script()
    title = "Held-out per cell, BH76 leg"
    table = mod.holdout_table(_holdout_csv(tmp_path), BH76, title=title)
    assert title in table, "the caption text is the given title"
    assert table.index(title) < table.index(TABLE_HEADER), "the caption precedes the table"


def test_holdout_table_of_another_leg_carries_only_that_legs_numbers(tmp_path):
    mod = _load_script()
    csv_path = _holdout_csv(tmp_path)
    bh76 = mod.holdout_table(csv_path, BH76, title="BH76")
    w411 = mod.holdout_table(csv_path, W411, title="W4-11")
    assert bh76 != w411
    assert EPS_HOLDOUT_W411 in w411 and EPS_HOLDOUT_W411 not in bh76, \
        "the leg filter selects the rows"
    assert EPS_HOLDOUT_BH76 in bh76 and EPS_HOLDOUT_BH76 not in w411


# ---------------------------------------------------------------------------
# T1  insample_table: the twin, its own numbers
# ---------------------------------------------------------------------------

def test_insample_table_has_the_same_columns_and_the_files_counts(tmp_path):
    mod = _load_script()
    table = mod.insample_table(_insample_csv(tmp_path), BH76,
                               title="In-sample per cell, BH76 leg")
    lines = _table_lines(table)
    header = _cells(lines[0])

    # the page says "the same columns"; the parenthetical of the two PBE columns is left open
    # here because the document the table replaces labels them (union), not (pool)
    assert len(header) == 12, "the in-sample twin carries the held-out table's twelve columns"
    assert header[:5] == ["Architecture", "Subset", "n rxn", "n species", "E NN"]
    assert header[5].startswith("E PBE")
    assert header[6] == "eps NN" and header[7].startswith("eps PBE")
    assert header[8] == "ED NN" and header[9] == "ED PBE (cell)"
    assert header[10] == "dED" and header[11] == "beats"
    assert len(lines) == 2 + 4, "every in-sample cell of the leg is a row"
    assert lines[2:] == INSAMPLE_BH76_ROWS, "the cells' own training reactions and species"


def test_insample_table_first_row_is_the_hand_written_one(tmp_path):
    mod = _load_script()
    table = mod.insample_table(_insample_csv(tmp_path), BH76, title="In-sample per cell")
    cells = _cells(_table_lines(table)[2])
    assert cells[:4] == ["deep_3x16", "2", "1", "2"], \
        "n rxn and n species are the cell's training counts, not the slice field (7)"
    assert cells[4] == "2.66" and cells[6] == EPS_INSAMPLE_BH76
    assert cells[10] == "**-6.90**" and cells[11] == "yes"


def test_insample_table_is_not_the_holdout_table(tmp_path):
    mod = _load_script()
    fam = _family_dir(tmp_path)
    held = mod.holdout_table(fam / "holdout_by_pool_3x3_eps.csv", COMBINED, title="held")
    ins = mod.insample_table(fam / "insample_by_pool_3x3_eps.csv", COMBINED, title="in")
    assert held != ins
    assert EPS_HOLDOUT_COMBINED in held and EPS_HOLDOUT_COMBINED not in ins
    assert EPS_INSAMPLE_COMBINED in ins and EPS_INSAMPLE_COMBINED not in held


# ---------------------------------------------------------------------------
# T2  splice: the block, the markers, the surrounding text
# ---------------------------------------------------------------------------

_OPEN = "<!-- table:holdout_bh76 -->"
_CLOSE = "<!-- /table:holdout_bh76 -->"

_DOC = (
    "## 5.7 Results per cell\n"
    "\n"
    "The tables below are read from `holdout_by_pool_3x3_eps.csv`.\n"
    "\n"
    + _OPEN + "\n"
    "| stale | table |\n"
    "|---|---|\n"
    "| 1 | 2 |\n"
    + _CLOSE + "\n"
    "\n"
    "### 5.8 Reading\n"
    "\n"
    "The text after the block.\n"
)

_A_TABLE = "| Architecture | Subset |\n|---|---|\n| deep_3x16 | 1 |\n"


def test_splice_keeps_the_markers_and_the_surrounding_text():
    """Kills m3: a splice that dropped the markers could not be run a second time."""
    mod = _load_script()
    out = mod.splice(_DOC, "holdout_bh76", _A_TABLE)

    assert out.count(_OPEN) == 1 and out.count(_CLOSE) == 1, "the markers are kept, once each"
    before = _DOC[:_DOC.index(_OPEN) + len(_OPEN)]
    after = _DOC[_DOC.index(_CLOSE):]
    assert out.startswith(before), "everything up to and including the opening marker is kept"
    assert out.endswith(after), "everything from the closing marker on is kept"

    inner = out[out.index(_OPEN) + len(_OPEN):out.index(_CLOSE)]
    assert inner.strip() == _A_TABLE.strip(), "the block between the markers is the table"
    assert "| stale | table |" not in out, "the old block is gone"


def test_splice_is_idempotent_on_its_own_output():
    mod = _load_script()
    once = mod.splice(_DOC, "holdout_bh76", _A_TABLE)
    twice = mod.splice(once, "holdout_bh76", _A_TABLE)
    assert twice == once, "the markers survive, so a second splice of the same table is a no-op"


def test_splice_refuses_a_missing_marker():
    mod = _load_script()
    with pytest.raises(ValueError, match="holdout_bh76"):
        mod.splice("no markers here at all\n", "holdout_bh76", _A_TABLE)


def test_splice_refuses_a_block_whose_closing_marker_is_absent():
    mod = _load_script()
    half = _DOC.replace(_CLOSE + "\n", "")
    with pytest.raises(ValueError, match="holdout_bh76"):
        mod.splice(half, "holdout_bh76", _A_TABLE)


def test_splice_touches_only_the_named_block():
    mod = _load_script()
    doc = (_DOC
           + "<!-- table:insample_combined -->\n| other | block |\n"
             "<!-- /table:insample_combined -->\n")
    out = mod.splice(doc, "holdout_bh76", _A_TABLE)
    assert "| other | block |" in out, "another table's block is untouched"
    assert out.count("<!-- table:insample_combined -->") == 1


# ---------------------------------------------------------------------------
# T3  main: the seven tables, only into the marker blocks that exist
# ---------------------------------------------------------------------------

_TABLE_NAMES = ("holdout_bh76", "holdout_w411", "holdout_combined", "holdout_combined_excl_tail",
                "insample_bh76", "insample_w411", "insample_combined")

# the ED of every network row of the --excl-dir fixture, moved so that the tail-excluded table
# is distinguishable from the family one cell by cell
_EXCL_SHIFT = -3.0


def _marked(name: str, filler: str = "PLACEHOLDER\n") -> str:
    return f"<!-- table:{name} -->\n{filler}<!-- /table:{name} -->\n"


def _block_of(text: str, name: str) -> str:
    o, c = f"<!-- table:{name} -->", f"<!-- /table:{name} -->"
    assert o in text and c in text, f"the {name} markers must survive the run"
    return text[text.index(o) + len(o):text.index(c)]


def _dirs(tmp_path):
    fam = _family_dir(tmp_path, "figures_family_val_best")
    excl = _family_dir(tmp_path, "figures_family_val_best_excl_tail", shift=_EXCL_SHIFT)
    return fam, excl


def _documents(tmp_path):
    """One document carrying three of the seven marker blocks, and one carrying none."""
    doc_a = Path(tmp_path) / "REPORT_fixture.md"
    doc_a.write_text(
        "# fixture report\n"
        "\n"
        "## 5.7 Results per cell\n"
        "\n"
        "Prose that must survive verbatim.\n"
        "\n"
        + _marked("holdout_bh76")
        + "\n"
        "Prose between the blocks.\n"
        "\n"
        + _marked("holdout_combined_excl_tail")
        + "\n"
        + _marked("insample_combined")
        + "\n"
        "The closing paragraph.\n"
    )
    doc_b = Path(tmp_path) / "SUMMARY_fixture.md"
    doc_b.write_text("# fixture summary\n\nNo marker anywhere in this document.\n")
    return doc_a, doc_b


def test_main_fills_only_the_marker_blocks_that_exist(tmp_path, capsys):
    """Kills m5: a main writing all seven tables everywhere would leak the other legs."""
    mod = _load_script()
    fam, excl = _dirs(tmp_path)
    doc_a, doc_b = _documents(tmp_path)

    rc = mod.main(["--family-dir", str(fam), "--excl-dir", str(excl),
                   "--splice", str(doc_a), str(doc_b)])
    assert not rc, "a successful run returns 0 (or None)"

    out = doc_a.read_text()
    assert "PLACEHOLDER" not in out, "every marker block present is filled"

    bh76 = _block_of(out, "holdout_bh76")
    assert TABLE_HEADER in bh76
    for row in BH76_ROWS:
        assert row in bh76, "the held-out BH76 table is the generated one"

    ins = _block_of(out, "insample_combined")
    assert EPS_INSAMPLE_COMBINED in ins, "the in-sample table comes from the in-sample CSV"
    assert EPS_HOLDOUT_COMBINED not in ins, "and not from the held-out one"

    excl_block = _block_of(out, "holdout_combined_excl_tail")
    assert "38.11" in excl_block, \
        "the tail-excluded table comes from --excl-dir (ED 41.11 moved to 38.11)"
    assert "41.11" not in excl_block

    # the four tables without a marker are written nowhere in the document
    assert EPS_HOLDOUT_W411 not in out, "holdout_w411 has no marker here"
    assert EPS_INSAMPLE_W411 not in out, "insample_w411 has no marker here"
    assert EPS_INSAMPLE_BH76 not in out, "insample_bh76 has no marker here"
    assert out.count(HEADER_STEM) == 3, "three marker blocks, three tables"

    printed = capsys.readouterr().out
    assert doc_a.name in printed, "one line per table and document is printed"


def test_main_keeps_the_prose_around_the_blocks(tmp_path):
    mod = _load_script()
    fam, excl = _dirs(tmp_path)
    doc_a, doc_b = _documents(tmp_path)
    before = doc_a.read_text()

    mod.main(["--family-dir", str(fam), "--excl-dir", str(excl),
              "--splice", str(doc_a), str(doc_b)])
    after = doc_a.read_text()

    head = before[:before.index("<!-- table:holdout_bh76 -->")]
    tail = before[before.index("The closing paragraph."):]
    assert after.startswith(head), "the text before the first block is byte-identical"
    assert after.endswith(tail), "the text after the last block is byte-identical"
    assert "Prose between the blocks." in after
    for name in ("holdout_bh76", "holdout_combined_excl_tail", "insample_combined"):
        assert after.count(f"<!-- table:{name} -->") == 1
        assert after.count(f"<!-- /table:{name} -->") == 1


def test_main_leaves_a_document_without_markers(tmp_path):
    """Kills m5 from the other side: a document with no marker is not appended to."""
    mod = _load_script()
    fam, excl = _dirs(tmp_path)
    doc_a, doc_b = _documents(tmp_path)
    before = doc_b.read_bytes()

    rc = mod.main(["--family-dir", str(fam), "--excl-dir", str(excl),
                   "--splice", str(doc_a), str(doc_b)])
    assert not rc
    assert doc_b.read_bytes() == before, "a document without a marker is left byte-identical"


def test_main_writes_every_table_when_every_marker_is_present(tmp_path):
    mod = _load_script()
    fam, excl = _dirs(tmp_path)
    doc = Path(tmp_path) / "REPORT_all.md"
    doc.write_text("# all seven\n\n" + "\n".join(_marked(n) for n in _TABLE_NAMES))

    rc = mod.main(["--family-dir", str(fam), "--excl-dir", str(excl), "--splice", str(doc)])
    assert not rc
    out = doc.read_text()
    assert "PLACEHOLDER" not in out, "all seven blocks are filled"
    assert out.count(HEADER_STEM) == 7
    assert EPS_HOLDOUT_W411 in _block_of(out, "holdout_w411")
    assert EPS_INSAMPLE_W411 in _block_of(out, "insample_w411")
    assert EPS_INSAMPLE_BH76 in _block_of(out, "insample_bh76")
    assert EPS_HOLDOUT_COMBINED in _block_of(out, "holdout_combined")
    assert "38.11" in _block_of(out, "holdout_combined_excl_tail")


# --------------------------------------------------------------------------- #
# The decks' rows and the CSV's own consistency (2026-09-09, the decisions on
# the objections)
# --------------------------------------------------------------------------- #

def test_latex_rows_escape_names_and_bold_the_negative_delta(tmp_path):
    """One tabular row per cell, underscores escaped, the negative dED bold."""
    mf = _load_script()
    csv_path = _holdout_csv(tmp_path)
    rows = mf.latex_rows(csv_path, "bh76")
    assert len(rows) == 4
    assert rows[0].startswith("deep\\_3x16 & 1 & ")
    assert rows[0].endswith("& \\textbf{-0.12}")
    assert "\\textbf" not in rows[1] and rows[1].endswith("& +2.72")
    assert all("_" not in r.split(" & ")[0].replace("\\_", "") for r in rows)


def test_a_verdict_that_disagrees_with_the_cell_values_is_refused(tmp_path):
    """``beats_pbe`` is the suite's cell comparison; a flag that contradicts the
    ED difference is a defect of the CSV and is never printed."""
    mf = _load_script()
    csv_path = _holdout_csv(tmp_path)
    text = csv_path.read_text()
    lines = text.splitlines()
    header = lines[0].split(",")
    i = header.index("beats_pbe")
    cells = lines[1].split(",")
    cells[i] = "False" if cells[i] == "True" else "True"
    lines[1] = ",".join(cells)
    csv_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="beats_pbe"):
        mf.holdout_table(csv_path, "bh76")
