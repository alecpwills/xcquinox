"""The donor-path resolution and the free-atom gate in
``plot_certificate_summary.py``.

The certificate bars plot, per (label, arch), the numbers each architecture
WARM-STARTED under. A run whose resolved config warm-starts an arch from a
donor has no run-local certificate for it, so the collector must resolve
through the config -- in both directions: the donor's certificate is the
arch's row, and a file left in the run-local slot of a donated arch is not
plotted beside the donor's as a second word on the same arch. The
run-local-slot direction is the discriminating one: a plain run-local glob
plots it, so each such case below fails against a run-local
implementation.

Oracle: ``collect_certificates`` on synthetic runs whose one architecture is
donor-backed, with the certificate placed in the donor directory, the
run-local slot, or neither.

A certificate has a second gate beside the atomization energies: the largest
|dE_xc| over its free atoms (``summary.max_atom_mHa``) against its recorded
``tol_atom``, and a certificate can fail on the atoms alone with its
atomization statistics inside both atomization gates. The records, the CSV
and a second panel carry that value and each certificate's own tolerance.
Oracle: synthetic certificates in the schema ``fidelity.fidelity_certificate``
writes, read through ``collect_certificates``; the CSV that ``write_csv``
writes; and the figure ``plot_certificate_summary`` draws, read back from its
two axes (bars, horizontal lines, annotations, ranges, tick labels) beside
the manifest the function returns.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


plot = _load("plot_certificate_summary")

_ARCH = "deep_3x16"


def _write_certificate(directory: Path, *, dAE: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": _ARCH,
                   "tolerances": {"tol_AE": 0.5},
                   "per_atomization": [{"name": "H2O", "dAE_kcalmol": dAE},
                                       {"name": "NH3", "dAE_kcalmol": 0.1}]},
                  f)


def _run_config(run: Path, donor: Path) -> None:
    import yaml
    cfg = {
        "sweep": {"arch": [_ARCH], "loss": ["delta_ae"], "metric": ["l2"],
                  "subset_size": [4], "solver": ["fast"]},
        "solvers": {"fast": {"mode": "fixed_density", "max_cycles": 1}},
        "hyperparams": {"n_steps": 200, "lr_start": 1e-3, "lr_end": 1e-5,
                        "lr_decay_start": 0.2, "grad_clip": 1.0,
                        "gradnorm_alpha": 1.5, "vxc_weight": 1.0,
                        "density_weight": 0.5},
        "inputs": {"external_refs_dir": "/refs", "subset_ledger_path":
                   "/ledger.json", "basis": "def2-tzvp", "grid_level": 3,
                   "output_root": "/out"},
        "pretrain": {"data_dir": "/pretrain_data",
                     "donor_checkpoints": {_ARCH: str(donor)}},
        "cluster": {"partition": "short", "time": "01:00:00", "mem": "8G",
                    "cpus_per_task": 1, "array_throttle": 1,
                    "eval_array_throttle": 1, "max_concurrent_tasks": 4},
        "domain_profile": "gmtkn55_subset",
    }
    with open(run / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)


def _donor_backed_run(tmp_path: Path, *, donor_certificate: bool,
                      run_local_certificate: bool = False) -> Path:
    donor = tmp_path / "donor_run" / "pretrain" / _ARCH
    donor.mkdir(parents=True)
    if donor_certificate:
        _write_certificate(donor, dAE=0.2)
    run = tmp_path / "run"
    run.mkdir()
    if run_local_certificate:
        _write_certificate(run / "pretrain" / _ARCH, dAE=9.9)
    _run_config(run, donor)
    return run


def test_collect_plots_the_donor_certificate_of_a_donor_backed_arch(tmp_path):
    run = _donor_backed_run(tmp_path, donor_certificate=True)
    records = plot.collect_certificates([("v7", str(run))])
    assert [(label, arch) for label, arch, _ in records] == [("v7", _ARCH)]
    assert records[0][2]["max"] == 0.2


def test_collect_ignores_a_run_local_certificate_of_a_donor_backed_arch(
        tmp_path):
    """A leftover PASS in the run-local slot is not the donor's word on the
    arch: the donor carries nothing here, so the run is refused as holding
    no certificate -- a run-local glob would plot the leftover at 9.9."""
    run = _donor_backed_run(tmp_path, donor_certificate=False,
                            run_local_certificate=True)
    try:
        plot.collect_certificates([("v7", str(run))])
    except ValueError as exc:
        assert "no fidelity_certificate.json" in str(exc)
    else:
        raise AssertionError("a donor without a certificate was plotted")


def test_collect_keeps_the_run_local_glob_without_a_config(tmp_path):
    """A run directory with no loadable resolved config plots exactly as it
    did before the donor rule existed."""
    run = tmp_path / "run"
    run.mkdir()
    _write_certificate(run / "pretrain" / _ARCH, dAE=0.3)
    records = plot.collect_certificates([("v7", str(run))])
    assert [(label, arch) for label, arch, _ in records] == [("v7", _ARCH)]
    assert records[0][2]["max"] == 0.3


# ---------------------------------------------------------------------------
# The free-atom gate
# ---------------------------------------------------------------------------

#: the tolerances block a certificate records under the two-tier
#: atomization gate
_TOLERANCES = {"tol_AE": 1.0, "tol_atom": 1.0, "tol_AE_aggregate": "mae",
               "tol_AE_max_backstop": 2.0, "override_reason": None}

#: the CSV columns that precede the free-atom columns, in their order
_CSV_HEAD = ["label", "arch", "arch_stored", "verdict", "n_atomizations",
             "mean_abs_dAE_kcalmol", "rmse_dAE_kcalmol",
             "max_abs_dAE_kcalmol", "species_over_1_kcalmol",
             "tol_AE", "tol_AE_aggregate", "tol_AE_max_backstop"]

_FAILED_ATOM_MHA = 1.6850708101152634
_FAILED_MAX_DAE = 1.5413649959273232


def _gate_run(tmp_path: Path, label: str, *, verdict: str, dAE: dict,
              summary: dict, per_system: list | None = None,
              tol_atom: float | None = 1.0) -> tuple:
    """``(label, run_dir)`` for a run holding one ``_ARCH`` certificate in
    its run-local slot. The run has no resolved config, so the run-local
    glob resolves the file. ``per_system`` is left out of the file when
    ``None``, and ``tol_atom`` out of its tolerances when ``None``."""
    tolerances = dict(_TOLERANCES)
    if tol_atom is None:
        del tolerances["tol_atom"]
    else:
        tolerances["tol_atom"] = tol_atom
    cert = {"verdict": verdict, "arch": _ARCH, "tolerances": tolerances,
            "per_atomization": [{"name": name, "dAE_kcalmol": value}
                                for name, value in dAE.items()],
            "summary": summary}
    if per_system is not None:
        cert["per_system"] = per_system
    run = tmp_path / label
    directory = run / "pretrain" / _ARCH
    directory.mkdir(parents=True)
    with open(directory / "fidelity_certificate.json", "w") as f:
        json.dump(cert, f)
    return label, str(run)


def _csv_cells(path: Path) -> tuple:
    """The CSV header, and its rows as ``{column: cell}`` keyed by
    ``(label, stored arch)``."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    return header, {(row[0], row[2]): dict(zip(header, row))
                    for row in rows[1:]}


def _number(cell: str):
    """A CSV cell as a float, or ``None`` for an empty cell: a value the
    certificate does not carry is written as nothing."""
    return float(cell) if cell else None


def _drawn(axes) -> dict:
    """What ``axes`` shows, read from its own artists: the bars as
    ``{x centre: (height, face colour, hatched)}``, the heights of the
    horizontal lines spanning it, the annotation texts, the two ranges, the
    x tick labels with their rotations, and the y label."""
    return {
        "bars": {round(p.get_x() + p.get_width() / 2, 9):
                 (p.get_height(), p.get_facecolor(), bool(p.get_hatch()))
                 for p in axes.patches},
        "gates": sorted(float(line.get_ydata()[0]) for line in axes.lines
                        if list(line.get_xdata()) == [0, 1]),
        "texts": sorted(t.get_text() for t in axes.texts),
        "xlim": tuple(axes.get_xlim()),
        "ylim": tuple(axes.get_ylim()),
        "ticks": [(t.get_text(), t.get_rotation())
                  for t in axes.get_xticklabels()],
        "ylabel": axes.get_ylabel(),
    }


def _bar_x(panel: dict, height: float) -> float:
    """The x centre of the one bar of ``panel`` that has this height."""
    (x,) = [x for x, bar in panel["bars"].items() if bar[0] == height]
    return x


@pytest.fixture
def render(monkeypatch):
    """``render(records, out_path)``: the manifest that
    ``plot_certificate_summary`` returns and the two panels it drew, the
    atomization panel then the free-atom panel, each as :func:`_drawn` reads
    it. The manifest is written beside each draw call; the axes hold what a
    viewer sees, so the figure is kept past the tool's own ``close`` and
    read."""
    close = plot.plt.close
    figures = []
    monkeypatch.setattr(plot.plt, "close", figures.append)

    def _render(records, out_path):
        manifest = plot.plot_certificate_summary(records, str(out_path))
        upper, lower = figures[-1].axes
        return manifest, _drawn(upper), _drawn(lower)

    yield _render
    for figure in figures:
        close(figure)


def test_a_failure_on_the_free_atoms_is_drawn_against_its_gate(tmp_path,
                                                               render):
    """A certificate failing on its free atoms alone shows the failing value
    against the atom gate, in the records, the figure and the CSV.

    Two labels over one architecture, both recording ``tol_atom`` 1.0 mHa: a
    PASS at 0.2 mHa and a FAIL at 1.685 mHa whose atomization offsets sit
    inside both atomization gates (mean 0.61 against ``tol_AE`` 1.0, max
    1.54 against the 2.0 backstop), the form of the failed
    ``deep_geom_3x16`` certificate. The records carry each largest free-atom
    deviation and the recorded tolerance. The lower axes hold one bar per
    certificate, at the x position and in the colour of that certificate's
    bar in the upper panel and as tall as its atom value, the FAIL alone
    hatched; one horizontal line at 1.0 (two certificates recording one
    value share one line) with its caption and no other text; and a y range
    that is the stated cap and reaches the 1.685 bar. The two panels share
    the architecture axis, whose name stands on the lower axes, slanted so
    that neighbouring names stay apart, and the lower axis is labelled in
    mHa. The upper panel keeps its own two gate lines and hatches the FAIL
    there too. The CSV keeps its twelve columns in place and appends the atom
    value and the tolerance. The summary is the only source of the atom
    value here (no per-system rows are written), and its
    ``max_dAE_kcalmol`` differs from it, so the bars show the recorded
    ``max_atom_mHa`` and nothing else.
    """
    runs = [
        _gate_run(tmp_path, "v7", verdict="PASS",
                  dAE={"H2O": 0.1, "NH3": -0.35},
                  summary={"max_atom_mHa": 0.2, "max_dAE_kcalmol": 0.35}),
        _gate_run(tmp_path, "S", verdict="FAIL",
                  dAE={"AlCl3": _FAILED_MAX_DAE, "CH4": 0.1, "C2H2": -0.2},
                  summary={"max_atom_mHa": _FAILED_ATOM_MHA,
                           "max_dAE_kcalmol": _FAILED_MAX_DAE,
                           "failure_reasons": [
                               "max |dE_xc| over free atoms "
                               f"{_FAILED_ATOM_MHA!r} mHa exceeds tol_atom "
                               "1.0 mHa"]}),
    ]
    records = plot.collect_certificates(runs)
    by_key = {(label, arch): r for label, arch, r in records}
    assert {key: r["max_atom"] for key, r in by_key.items()} == {
        ("v7", _ARCH): 0.2, ("S", _ARCH): _FAILED_ATOM_MHA}
    assert {key: r["tol_atom"] for key, r in by_key.items()} == {
        ("v7", _ARCH): 1.0, ("S", _ARCH): 1.0}
    passed, failed = by_key[("v7", _ARCH)], by_key[("S", _ARCH)]
    assert failed["mean"] < 1.0 and failed["max"] < 2.0

    manifest, upper, lower = render(records, tmp_path / "summary.png")
    assert manifest["atom_bars"] == {("v7", _ARCH): 0.2,
                                     ("S", _ARCH): _FAILED_ATOM_MHA}
    assert [tuple(k) for k in manifest["atom_hatched"]] == [("S", _ARCH)]
    assert [value for value, _text in manifest["atom_gate_lines"]] == [1.0]
    assert manifest["atom_y_cap"] >= _FAILED_ATOM_MHA
    assert manifest["atom_notes"] == {}
    assert {"out_path", "y_cap", "gate_lines", "hatched", "clipped",
            "colors", "notes"} <= set(manifest)
    assert sorted(value for value, _text in manifest["gate_lines"]) == [
        1.0, 2.0]
    assert [tuple(k) for k in manifest["hatched"]] == [("S", _ARCH)]

    # the figure itself: each certificate's bar in the upper panel is found
    # by its mean, and the lower panel is read against it
    x_pass, x_fail = _bar_x(upper, passed["mean"]), _bar_x(upper, failed["mean"])
    assert (upper["bars"][x_pass][2], upper["bars"][x_fail][2]) == (False, True)
    assert lower["bars"] == {
        x_pass: (0.2, upper["bars"][x_pass][1], False),
        x_fail: (_FAILED_ATOM_MHA, upper["bars"][x_fail][1], True)}
    assert (upper["gates"], lower["gates"]) == ([1.0, 2.0], [1.0])
    assert lower["texts"] == [text for _value, text
                              in manifest["atom_gate_lines"]]
    assert lower["ylim"] == (0.0, manifest["atom_y_cap"])
    assert lower["xlim"] == upper["xlim"]
    assert [name for name, _rotation in lower["ticks"]] == [
        plot.display_name(_ARCH)]
    assert all(rotation for _name, rotation in lower["ticks"])
    assert "mHa" in lower["ylabel"]

    csv_path = tmp_path / "summary.csv"
    plot.write_csv(records, str(csv_path))
    header, cells = _csv_cells(csv_path)
    assert header == _CSV_HEAD + ["max_atom_mHa", "tol_atom"]
    for key, verdict, atom, max_dae in (
            (("v7", _ARCH), "PASS", 0.2, 0.35),
            (("S", _ARCH), "FAIL", _FAILED_ATOM_MHA, _FAILED_MAX_DAE)):
        assert cells[key]["verdict"] == verdict
        assert _number(cells[key]["max_abs_dAE_kcalmol"]) == max_dae
        assert _number(cells[key]["max_atom_mHa"]) == atom
        assert _number(cells[key]["tol_atom"]) == 1.0


def test_the_atom_value_falls_back_to_the_free_atom_rows(tmp_path, render):
    """Without a value in ``summary.max_atom_mHa`` the atom value is the
    largest magnitude of ``dE_xc_mHa`` over the per-system rows marked
    ``is_atom``.

    The rows hold a negative atom deviation (-0.9) larger in magnitude than
    any other atom's, so a signed maximum (0.4) shows; two molecules larger
    than every atom (2.5 and -3.0), so a maximum over all rows shows; a
    nulled atom row, which is skipped; and an evaluation-error row, which
    carries no ``is_atom`` and no value, as the certificate writer records
    one. A second certificate records 0.6 in its summary beside rows that
    would give 0.9: the summary holds the value the verdict was formed from,
    so where it is recorded it is read in preference to a recomputation. A
    third holds a null there beside the same rows: a null is no recorded
    value, and the rows are read. Each value is the height of a bar on the
    lower axes.
    """
    rows = [
        {"name": "atom_H", "is_atom": True, "dE_xc_mHa": 0.4},
        {"name": "atom_C", "is_atom": True, "dE_xc_mHa": -0.9},
        {"name": "atom_O", "is_atom": True, "dE_xc_mHa": None},
        {"name": "atom_N", "error": "RuntimeError: evaluation failed"},
        {"name": "H2O", "is_atom": False, "dE_xc_mHa": 2.5},
        {"name": "CH4", "is_atom": False, "dE_xc_mHa": -3.0},
    ]
    runs = [
        _gate_run(tmp_path, "rows", verdict="PASS",
                  dAE={"H2O": 0.2, "CH4": 0.1},
                  summary={"max_dAE_kcalmol": 0.2}, per_system=rows),
        _gate_run(tmp_path, "summary", verdict="PASS",
                  dAE={"H2O": 0.2, "CH4": 0.1},
                  summary={"max_atom_mHa": 0.6, "max_dAE_kcalmol": 0.2},
                  per_system=rows),
        _gate_run(tmp_path, "null_summary", verdict="PASS",
                  dAE={"H2O": 0.2, "CH4": 0.1},
                  summary={"max_atom_mHa": None, "max_dAE_kcalmol": 0.2},
                  per_system=rows),
    ]
    records = plot.collect_certificates(runs)
    assert {(label, arch): r["max_atom"] for label, arch, r in records} == {
        ("rows", _ARCH): 0.9, ("summary", _ARCH): 0.6,
        ("null_summary", _ARCH): 0.9}

    _manifest, _upper, lower = render(records, tmp_path / "summary.png")
    assert sorted(height for height, _colour, _hatched
                  in lower["bars"].values()) == [0.6, 0.9, 0.9]


def test_a_certificate_without_the_atom_value_is_noted_not_drawn(tmp_path,
                                                                 render):
    """A certificate with no free-atom value draws no bar, is noted, and
    states no number in the CSV; the other panel's gap removes no atom bar.

    Four certificates carry no free-atom value: one in the writer's form
    when no free atom was evaluated (``summary.max_atom_mHa`` null, the atom
    rows errored or nulled, a FAIL); one in the form written before the
    free-atom gate (no atom field in its summary, no per-system rows, no
    recorded ``tol_atom``); one whose summary holds a NaN; and one with no
    summary value whose atom rows hold a NaN after a finite 0.4. A value
    that is not finite is a failed measurement: it is no height to draw and
    no axis limit to set, and the largest magnitude over rows that include
    one is not the 0.4 that ``max`` returns for that row order. None of the
    four draws a bar; all four are among the lower panel's notes, which
    stand on the lower axes beside the gate caption and nothing else; and
    none states a number in the CSV, where a zero would read as a perfect
    clone and a ``nan`` cell as a value. The second records no tolerance, so
    its record carries none and it adds no gate line: the others record
    0.5, so a line at 1.0 could come only from a default standing in for a
    tolerance the certificate never recorded. Each note names its
    certificate's verdict, since a record without a bar has nothing to
    hatch. A fifth certificate has an atom value of 0.3 and no atomization
    offset (every molecule failed, a FAIL): its atomization bar is replaced
    by a note, and its atom bar is the one bar of the lower panel, hatched
    although 0.3 is inside its own 0.5 gate, because the hatch is the
    verdict.
    """
    legacy = tmp_path / "legacy"
    _write_certificate(legacy / "pretrain" / _ARCH, dAE=0.3)
    nan = float("nan")
    runs = [
        _gate_run(tmp_path, "null", verdict="FAIL", dAE={"H2O": None},
                  summary={"max_atom_mHa": None},
                  per_system=[
                      {"name": "atom_H",
                       "error": "RuntimeError: evaluation failed"},
                      {"name": "atom_O", "is_atom": True, "dE_xc_mHa": None},
                      {"name": "H2O", "is_atom": False, "dE_xc_mHa": 0.4}],
                  tol_atom=0.5),
        ("legacy", str(legacy)),
        _gate_run(tmp_path, "nan_summary", verdict="FAIL",
                  dAE={"H2O": 0.1, "NH3": 0.2},
                  summary={"max_atom_mHa": nan}, tol_atom=0.5),
        _gate_run(tmp_path, "nan_row", verdict="FAIL",
                  dAE={"H2O": 0.1, "NH3": 0.2}, summary={},
                  per_system=[
                      {"name": "atom_H", "is_atom": True, "dE_xc_mHa": 0.4},
                      {"name": "atom_O", "is_atom": True, "dE_xc_mHa": nan}],
                  tol_atom=0.5),
        _gate_run(tmp_path, "atoms_only", verdict="FAIL", dAE={},
                  summary={"max_atom_mHa": 0.3}, tol_atom=0.5),
    ]
    without = {("null", _ARCH), ("legacy", _ARCH), ("nan_summary", _ARCH),
               ("nan_row", _ARCH)}
    records = plot.collect_certificates(runs)
    assert {(label, arch): r["max_atom"] for label, arch, r in records} == {
        ("null", _ARCH): None, ("legacy", _ARCH): None,
        ("nan_summary", _ARCH): None, ("nan_row", _ARCH): None,
        ("atoms_only", _ARCH): 0.3}
    assert {(label, arch): r["tol_atom"] for label, arch, r in records} == {
        ("null", _ARCH): 0.5, ("legacy", _ARCH): None,
        ("nan_summary", _ARCH): 0.5, ("nan_row", _ARCH): 0.5,
        ("atoms_only", _ARCH): 0.5}

    manifest, _upper, lower = render(records, tmp_path / "summary.png")
    assert manifest["atom_bars"] == {("atoms_only", _ARCH): 0.3}
    assert [tuple(k) for k in manifest["atom_hatched"]] == [
        ("atoms_only", _ARCH)]
    assert set(manifest["atom_notes"]) == without
    assert all(r["verdict"] in manifest["atom_notes"][(label, arch)]
               for label, arch, r in records if (label, arch) in without)
    assert [value for value, _text in manifest["atom_gate_lines"]] == [0.5]
    assert [(height, hatched) for height, _colour, hatched
            in lower["bars"].values()] == [(0.3, True)]
    assert lower["gates"] == [0.5]
    assert lower["texts"] == sorted(
        list(manifest["atom_notes"].values())
        + [text for _value, text in manifest["atom_gate_lines"]])

    csv_path = tmp_path / "summary.csv"
    plot.write_csv(records, str(csv_path))
    _header, cells = _csv_cells(csv_path)
    for key in without:
        assert _number(cells[key]["max_atom_mHa"]) is None
    assert _number(cells[("legacy", _ARCH)]["tol_atom"]) is None
    assert _number(cells[("atoms_only", _ARCH)]["max_atom_mHa"]) == 0.3


def test_each_recorded_atom_tolerance_draws_its_own_gate_line(tmp_path,
                                                              render):
    """The free-atom gate lines are the certificates' own tolerances.

    Two certificates record ``tol_atom`` 0.5 and 2.0 mHa: one horizontal
    line stands on the lower axes at each value, so a line fixed at 1.0
    shows. The 1.2 mHa bar is a PASS under its own 2.0 gate and is not
    hatched, so hatching formed from a comparison against any other gate
    shows. The y range is the stated cap, which reaches the 2.0 line above
    the tallest bar, and each CSV row carries its own certificate's
    tolerance.
    """
    runs = [
        _gate_run(tmp_path, "tight", verdict="PASS",
                  dAE={"H2O": 0.1, "NH3": 0.2},
                  summary={"max_atom_mHa": 0.3}, tol_atom=0.5),
        _gate_run(tmp_path, "loose", verdict="PASS",
                  dAE={"H2O": 0.1, "NH3": 0.2},
                  summary={"max_atom_mHa": 1.2}, tol_atom=2.0),
    ]
    records = plot.collect_certificates(runs)
    assert {(label, arch): (r["max_atom"], r["tol_atom"])
            for label, arch, r in records} == {
        ("tight", _ARCH): (0.3, 0.5), ("loose", _ARCH): (1.2, 2.0)}

    manifest, _upper, lower = render(records, tmp_path / "summary.png")
    assert sorted(value for value, _text in manifest["atom_gate_lines"]) == [
        0.5, 2.0]
    assert list(manifest["atom_hatched"]) == []
    assert manifest["atom_y_cap"] >= 2.0
    assert lower["gates"] == [0.5, 2.0]
    assert [hatched for _height, _colour, hatched
            in lower["bars"].values()] == [False, False]
    assert lower["ylim"] == (0.0, manifest["atom_y_cap"])

    csv_path = tmp_path / "summary.csv"
    plot.write_csv(records, str(csv_path))
    _header, cells = _csv_cells(csv_path)
    assert _number(cells[("tight", _ARCH)]["tol_atom"]) == 0.5
    assert _number(cells[("loose", _ARCH)]["tol_atom"]) == 2.0
