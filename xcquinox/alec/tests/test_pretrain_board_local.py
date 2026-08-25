"""The local pretraining board, as a test: pretraining SHOWN to work for every
architecture on this workstation before any cluster time is spent on it.

Every case here runs a real pretraining and is therefore marked ``slow``
(deselected by the repository's default ``-m "not slow"``). The whole board,
printed as one table, is ``scripts/pretrain_board_local.py``; this module is
the same measurement per architecture, so a single architecture can be run in
isolation::

    pytest xcquinox/alec/tests/test_pretrain_board_local.py -m slow \\
        -k deep_3x16

What each case asserts, on the session's shared dataset (sto-3g, grid level 3,
polarized, the production orientation lock, seven systems: the free atoms H,
Li, N, O and the molecules H2O, LiH, N2, so every atomization energy in the set
is defined) -- built on PBE's self-consistent density for the GGA rung and on
SCAN's for the meta-GGA rung, whose parent is SCAN:

* At INITIALIZATION an anchored model IS its parent, so the per-system XC
  energy errors sit at the oracle floor: 1e-6 mHa per free atom and per
  atomization energy, five orders under the certificate's binding tolerances.
  The design measures -1.3e-8 mHa on the N atom's correlation term
  (``SPEC_parent_anchor.md`` Section 3.5).
* After a short schedule (300 steps) the same errors still clear the
  certificate: 1.0 mHa per free atom, 1.0 kcal/mol per atomization energy
  (``SPEC_pretrain_fidelity_program.md`` Section 3.3). This is the property
  the campaign depends on, and it is what the anchor was built to make
  unconditional -- the energy-weight sweep measured 1.68 to 8.00 kcal/mol for
  an unanchored ``deep_3x16`` at every weight it tried (job 2134963).
* The UNANCHORED nets in the same DFS coordinates are RECORDED beside them, as
  the measurement of what the coordinates alone deliver. They are held only
  loosely, to a factor of two around the value the board measured, so a
  regression shows without the case becoming a fit of the training noise.

The energy errors are the certificate's own quantity -- per-system
``E_xc^NN - E_xc^parent``, exchange plus correlation, folded into atomization
energies against the free atoms of the same set -- computed from the
checkpoint on disk by the energy-weight probe's machinery rather than restated
here.

Wall clock measured on this workstation at 4 threads: the dataset is generated
once per session, and each architecture's 300-step fit is quoted in the board's
own table.
"""
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

board = importlib.import_module("pretrain_board_local")

import xcquinox.alec as alec  # noqa: E402
from xcquinox.alec.config import ArchitectureConfig  # noqa: E402


#: Steps every case runs. Short enough to keep each architecture inside the
#: board's own wall budget on this workstation and long enough for the
#: schedule's decay to bite (``lr_decay_start`` is 0.2 of the run).
BOARD_STEPS = 300
BOARD_SEED = 0

_GGA_ARCHS = tuple(name for name in sorted(alec.ARCHITECTURES)
                   if not ArchitectureConfig.is_meta_gga(alec.ARCHITECTURES[name]))
_MGGA_ARCHS = tuple(name for name in sorted(alec.ARCHITECTURES)
                    if ArchitectureConfig.is_meta_gga(alec.ARCHITECTURES[name]))


@pytest.fixture(scope="module")
def board_data(tmp_path_factory):
    """The session's PBE-density pretraining dataset, shared by every GGA-rung
    case.

    Module-scoped because the generator's cost is a reference SCF per system
    and the file's identity is fixed; the generator's own manifest check makes
    a second call a no-op, and the shared path makes the checkpoints of the
    different architectures comparable by construction.
    """
    work = tmp_path_factory.mktemp("pretrain_board")
    return str(work), board.ensure_board_data(str(work / "data"))


@pytest.fixture(scope="module")
def board_data_scan(tmp_path_factory):
    """The same dataset on SCAN's self-consistent density, which the meta-GGA
    rung's rows sit on.

    Its own file rather than the PBE one: the parent density is a different
    SCF solution, the rows are not interchangeable, and ``run_pretrain``
    resolves the parent from the architecture before it names the file (and
    refuses the PBE name for a SCAN-parent run). Generation measured at 12.2 s
    for the board's seven systems at sto-3g / grid level 3.
    """
    work = tmp_path_factory.mktemp("pretrain_board_scan")
    return str(work), board.ensure_board_data(str(work / "data"),
                                              reference_xc="scan")


def _run(name, anchor, board_data):
    work_dir, data_path = board_data
    return board.run_cell(name, anchor=anchor, data_path=data_path,
                          work_dir=work_dir, steps=BOARD_STEPS,
                          seed=BOARD_SEED, coordinates="dfs")


@pytest.mark.slow
@pytest.mark.parametrize("arch_name", _GGA_ARCHS)
def test_the_anchored_architecture_pretrains_within_the_certificate(
        arch_name, board_data, capsys):
    """One anchored architecture, pretrained locally in the DFS coordinates.

    Three statements, in the order a failure should be read: the model is its
    parent at step 0 (1e-6 mHa on both measures); the fit stays inside the
    certificate after 300 steps (1.0 mHa per atom, 1.0 kcal/mol per
    atomization); and the pointwise losses and the wall clock are printed, so
    the board's row for this architecture is in the test output whether it
    passes or fails.
    """
    row = _run(arch_name, True, board_data)
    passed, reasons = board.verdict(row)
    row["verdict"] = "PASS" if passed else "FAIL"
    with capsys.disabled():
        print()
        print(board.format_table([row]))
    assert passed, (arch_name, reasons)
    assert row["init_max_atom_mHa"] is not None
    assert abs(row["init_max_atom_mHa"]) <= board.TOL_INIT_MHA
    assert abs(row["init_max_dAE_kcal"]) <= board.TOL_INIT_MHA
    assert abs(row["max_atom_mHa"]) <= board.TOL_ATOM_MHA
    assert abs(row["max_dAE_kcal"]) <= board.TOL_AE_KCAL


@pytest.mark.slow
@pytest.mark.parametrize("arch_name", _GGA_ARCHS)
def test_the_unanchored_architecture_in_the_same_coordinates(
        arch_name, board_data, capsys):
    """The control row: the same architecture and coordinates WITHOUT the
    anchor, recorded rather than gated.

    The numbers are the measurement of what the DFS coordinates alone deliver
    on this dataset, which is what the anchored row has to be read against.
    They are held to a factor of two around the recorded value where the board
    has one (``pretrain_board_local.UNANCHORED_REFERENCE``); an architecture
    with no recorded value is reported and not held, so adding one to the
    registry does not turn the board red before it has been measured.

    The case also states the thing the anchor exists for: an unanchored model
    is NOT its parent at step 0, so its initialization errors are far above
    the anchored floor.
    """
    row = _run(arch_name, False, board_data)
    passed, reasons = board.verdict(row)
    row["verdict"] = "PASS" if passed else "FAIL"
    with capsys.disabled():
        print()
        print(board.format_table([row]))
    assert passed, (arch_name, reasons)
    assert row["init_max_atom_mHa"] is not None
    assert abs(row["init_max_atom_mHa"]) > board.TOL_INIT_MHA, (
        "an unanchored network starts at F = 1, the LDA/PW92 limit, not at "
        "its parent; if this holds the control is not a control")


@pytest.mark.slow
@pytest.mark.parametrize("arch_name", _MGGA_ARCHS)
def test_the_anchored_meta_gga_architecture_pretrains_within_the_certificate(
        arch_name, board_data_scan, capsys):
    """The meta-GGA rows of the board: the same three statements as the GGA
    rung, against SCAN.

    The rung's parent is SCAN, so its rows sit on SCAN's SELF-CONSISTENT
    density and not on the shared PBE file -- ``run_pretrain`` resolves the
    parent from the architecture before it names the file and refuses the PBE
    one by name -- which is why this case takes its own dataset fixture.

    Measured for ``deep_mgga_3x16`` on this workstation (sto-3g, grid level 3,
    polarized, 300 steps, seed 0, 4 threads): at initialization 5.726e-7 mHa
    on the worst free atom and 2.919e-6 kcal/mol on the worst atomization,
    both under the board's 1e-5 initialization floor; after the schedule
    0.0049 mHa and 0.0011 kcal/mol, two to three orders inside the
    certificate's 1.0 mHa and 1.0 kcal/mol. Pointwise losses 9.11e-12 (x) and
    2.54e-12 (c); wall 14.8 s.
    """
    work_dir, data_path = board_data_scan
    row = board.run_cell(arch_name, anchor=True, data_path=data_path,
                         work_dir=work_dir, steps=BOARD_STEPS,
                         seed=BOARD_SEED, coordinates="dfs")
    passed, reasons = board.verdict(row)
    row["verdict"] = "PASS" if passed else "FAIL"
    with capsys.disabled():
        print()
        print(board.format_table([row]))
    assert passed, (arch_name, reasons)
    assert row["init_max_atom_mHa"] is not None
    assert abs(row["init_max_atom_mHa"]) <= board.TOL_INIT_MHA
    assert abs(row["init_max_dAE_kcal"]) <= board.TOL_INIT_MHA
    assert abs(row["max_atom_mHa"]) <= board.TOL_ATOM_MHA
    assert abs(row["max_dAE_kcal"]) <= board.TOL_AE_KCAL


@pytest.mark.slow
def test_the_unanchored_meta_gga_architecture_in_the_same_coordinates(
        board_data_scan, capsys):
    """The meta-GGA control row: ``deep_mgga_3x16`` in the DFS coordinates,
    fitted to SCAN WITHOUT the anchor, recorded rather than gated.

    It is the measurement the anchored row has to be read against, and it is
    the worst of the recorded architectures: measured 577.6 mHa on the worst
    free atom at initialization (an unanchored network starts at ``F = 1``,
    the LDA/PW92 limit, which is further from SCAN than from PBE) and, after
    300 steps, 19.8748 mHa and 8.6720 kcal/mol -- an order outside the
    certificate, against the anchored row's 0.0049 mHa and 0.0011 kcal/mol on
    the same data and schedule.
    """
    work_dir, data_path = board_data_scan
    row = board.run_cell("deep_mgga_3x16", anchor=False, data_path=data_path,
                         work_dir=work_dir, steps=BOARD_STEPS,
                         seed=BOARD_SEED, coordinates="dfs")
    passed, reasons = board.verdict(row)
    row["verdict"] = "PASS" if passed else "FAIL"
    with capsys.disabled():
        print()
        print(board.format_table([row]))
    assert passed, reasons
    assert abs(row["init_max_atom_mHa"]) > board.TOL_INIT_MHA, (
        "an unanchored network starts at F = 1, the LDA/PW92 limit, not at "
        "its parent; if this holds the control is not a control")
    assert abs(row["max_dAE_kcal"]) > board.TOL_AE_KCAL, (
        "the unanchored fit is outside the certificate at 300 steps, which is "
        "the measurement the anchor exists to remove")


def test_the_board_generates_one_dataset_per_parent_density(monkeypatch):
    """``run_board`` asks for the file each architecture's rung resolves, once
    per parent rather than once per architecture.

    The parent is read from the ARCHITECTURE (``resolve_parent_density`` under
    the rung baseline), not from the anchor state, so an unanchored control
    row reads the same file as its anchored twin and the two stay comparable;
    a board spanning both rungs therefore generates exactly two files however
    many architectures it holds. Driven with the generator and the cell runner
    stubbed, so the wiring is stated without a pretraining.
    """
    requested = []
    cells = []

    monkeypatch.setattr(board, "_load_probe", lambda: None)
    monkeypatch.setattr(
        board, "ensure_board_data",
        lambda data_dir, **kw: (requested.append(kw.get("reference_xc", "pbe"))
                                or f"{data_dir}/{kw.get('reference_xc')}.npz"))

    def _cell(name, *, anchor, data_path, **kw):
        cells.append((name, anchor, data_path))
        return {"arch": name, "anchored": anchor, "init_max_atom_mHa": 0.0,
                "init_max_dAE_kcal": 0.0, "max_atom_mHa": 0.0,
                "max_dAE_kcal": 0.0, "wall_s": 0.0}

    monkeypatch.setattr(board, "run_cell", _cell)
    rows, ok = board.run_board(
        archs=("deep_3x16", "deep_mgga_3x16", "deep_rung35_mgga_3x16"),
        work_dir="/tmp/board_wiring", log=lambda message: None)
    assert ok and len(rows) == 6
    assert sorted(requested) == ["pbe", "scan"], requested
    by_arch = {(name, anchor): path for name, anchor, path in cells}
    assert by_arch[("deep_3x16", True)].endswith("pbe.npz")
    assert by_arch[("deep_3x16", False)] == by_arch[("deep_3x16", True)]
    for name in ("deep_mgga_3x16", "deep_rung35_mgga_3x16"):
        assert by_arch[(name, True)].endswith("scan.npz"), name
        assert by_arch[(name, False)] == by_arch[(name, True)], name
    assert all(row["parent"] == ("scan" if row["arch"].endswith("mgga_3x16")
                                 else "pbe") for row in rows)


# ---------------------------------------------------------------------------
# The board's own contract -- cheap, so not marked slow
# ---------------------------------------------------------------------------

def test_the_board_covers_every_registered_gga_architecture():
    """The board's architecture list IS the registry's GGA rung: an
    architecture added to the registry joins the board without a second edit,
    which is what makes "every architecture was shown to pretrain" a statement
    about the registry rather than about a hand-kept list."""
    assert board.board_architectures() == _GGA_ARCHS
    assert set(board.board_architectures(include_meta_gga=True)) == \
        set(alec.ARCHITECTURES)


def test_the_board_runs_anchored_polarized_dfs_architectures():
    """What the board pretrains: the registry entry with polarized
    correlation, the DFS coordinates and the anchor, which is the campaign's
    model class. The unanchored control differs in the anchor alone."""
    anchored_arch = board.board_arch("deep_3x16", anchor=True)
    plain = board.board_arch("deep_3x16", anchor=False)
    assert anchored_arch.parent_anchor is True
    assert anchored_arch.zero_init_final_layer is True
    assert anchored_arch.use_polarized_correlation is True
    assert anchored_arch.descriptor_coordinates == "dfs"
    assert plain.parent_anchor is False
    assert plain.descriptor_coordinates == "dfs"
    assert plain.use_polarized_correlation is True


def test_the_board_set_defines_every_atomization_energy():
    """Every element of every molecule in the board's set has its own free
    atom in the set, so no atomization energy is skipped for want of a
    reference (the probe returns such molecules as skipped rather than
    dropping them silently)."""
    from xcquinox.alec.pretrain_data_gen import resolve_pretrain_systems

    systems = resolve_pretrain_systems(atoms=board.BOARD_SYSTEMS)
    single, multi = set(), []
    for system in systems:
        symbols = [chunk.split()[0]
                   for chunk in system.atom.split(";") if chunk.strip()]
        if len(symbols) == 1 and system.charge == 0:
            single.add(symbols[0])
        elif len(symbols) > 1:
            multi.append((system.name, symbols))
    assert multi, "the board needs at least one molecule"
    assert any(s.spin == 0 for s in systems if ";" in s.atom), \
        "the board needs a closed-shell molecule"
    assert any(s.spin > 0 for s in systems if ";" not in s.atom), \
        "the board needs an open-shell free atom"
    for name, symbols in multi:
        missing = sorted(set(symbols) - single)
        assert not missing, (name, missing)


def test_the_board_gates_the_anchored_rows_and_reports_the_others():
    """The verdict rule, exercised on synthetic rows so the gate is testable
    without a pretraining.

    An anchored row fails when it starts off its parent or ends outside the
    certificate; an unanchored row with no recorded value passes with its
    reason stated, and one that has drifted past twice its recorded value
    fails.
    """
    good = dict(arch="deep_3x16", anchored=True, init_max_atom_mHa=1e-9,
                init_max_dAE_kcal=1e-9, max_atom_mHa=0.2, max_dAE_kcal=0.3)
    assert board.verdict(good)[0] is True

    off_parent = dict(good, init_max_atom_mHa=1e-3)
    passed, reasons = board.verdict(off_parent)
    assert passed is False and any("initialization floor" in r for r in reasons)

    off_certificate = dict(good, max_dAE_kcal=2.5)
    passed, reasons = board.verdict(off_certificate)
    assert passed is False and any("certificate tolerance" in r for r in reasons)

    unheld = dict(arch="not_a_registered_arch", anchored=False,
                  max_atom_mHa=99.0, max_dAE_kcal=99.0)
    passed, reasons = board.verdict(unheld)
    assert passed is True and any("no recorded value" in r for r in reasons)


def test_the_board_table_carries_the_gated_columns():
    """The table states, per row, what the gate read: both initialization
    measures, both certificate measures, the two pointwise losses and the wall
    clock. A column dropped from the table is a number the board stopped
    reporting."""
    row = dict(arch="deep_3x16", anchored=True, steps=300, loss_x=5e-7,
               loss_c=3.4e-4, init_max_atom_mHa=1.3e-8,
               init_max_dAE_kcal=2.0e-8, max_atom_mHa=0.21,
               max_dAE_kcal=0.34, rms_mHa=0.15, wall_s=42.0, verdict="PASS")
    text = board.format_table([row])
    for column in ("init_max_atom_mHa", "init_max_dAE_kcal", "max_atom_mHa",
                   "max_dAE_kcal", "loss_x", "loss_c", "wall_s", "verdict"):
        assert column in text, column
    assert "deep_3x16" in text and "PASS" in text
