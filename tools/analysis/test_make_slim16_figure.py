"""The Slim16 figure: each held-out subset's error per network beside PBE-DF.

The figure puts the subsets on the x axis and draws, per subset, one bar per
evaluated network and one PBE-DF bar, each the subset's mean absolute error of
the reaction energies against the benchmark references over every reaction
the network reports, converged or not. Each network's WTMAD-2 (the paper's
form) is computed with the FULL set's weights twice, over all reactions and
over the converged ones only, and the legend carries both; a subset with no
converged reaction is dropped from the converged total, its weight is not
redistributed, and the unconverged reactions are named. Where a subset holds
converged and unconverged reactions both, the converged-only error is marked
on the bar. Bars are coloured by architecture and hatched by network group.
The collect layer reuses the table's row assembly (``slim16_table``), so the
figure and the metrics CSV state one set of numbers.

Oracle: synthetic run dirs whose PBE-DF reaction energies are exact in
kcal/mol (41.0, 121.8, 80.6 and 65.0 for the four reactions) and whose
benchmark references are 40, 120, 80 and 60 (mean 75, four reactions, so the
scale is 18.75), so every per-subset error and every WTMAD-2 is a hand value:
18.75 times the sum over the subsets of N_i MAD_i / m_i.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import pytest

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


figure = _load("make_slim16_figure")
table = figure.slim16_table
arch_style = _load("arch_style")

_KCAL = 627.5094740631

# Reaction A, subset ALKBDE10 (name and stoichiometry from load_full_slim):
# the PBE-DF reaction energy is 41.0 kcal/mol, the reference 40.0.
_A = {
    "name": "alkbde10_006",
    "reactants": ["slim16@alkbde10_lif"],
    "products": ["slim16@alkbde10_li", "slim16@alkbde10_f"],
    "coeffs": [-1.0, 1.0, 1.0],
}
_A_SPECIES = {"slim16@alkbde10_lif": -1.0,
              "slim16@alkbde10_li": -0.5,
              "slim16@alkbde10_f": -0.5 + 41.0 / _KCAL}
_A_REF = 40.0
# Reaction B, subset RC21: PBE-DF 121.8 kcal/mol, reference 120.0.
_B = {
    "name": "rc21_010",
    "reactants": ["slim16@rc21_4e"],
    "products": ["slim16@rc21_4p", "slim16@rc21_me"],
    "coeffs": [-1.0, 1.0, 1.0],
}
_B_SPECIES = {"slim16@rc21_4e": -2.0,
              "slim16@rc21_4p": -1.0,
              "slim16@rc21_me": -1.0 + 121.8 / _KCAL}
_B_REF = 120.0
# Reactions E and H, the two RSE43 reactions of the pool: PBE-DF 80.6 and
# 65.0 kcal/mol, references 80.0 and 60.0. They share two species.
_E = {
    "name": "rse43_033",
    "reactants": ["slim16@rse43_E35", "slim16@rse43_P1"],
    "products": ["slim16@rse43_P35", "slim16@rse43_E1"],
    "coeffs": [-1.0, -1.0, 1.0, 1.0],
}
_H = {
    "name": "rse43_019",
    "reactants": ["slim16@rse43_E21", "slim16@rse43_P1"],
    "products": ["slim16@rse43_P21", "slim16@rse43_E1"],
    "coeffs": [-1.0, -1.0, 1.0, 1.0],
}
_RSE43_SPECIES = {"slim16@rse43_E35": -1.0,
                  "slim16@rse43_P1": -0.5,
                  "slim16@rse43_P35": -1.2,
                  "slim16@rse43_E1": -0.3 + 80.6 / _KCAL,
                  "slim16@rse43_E21": -1.0,
                  "slim16@rse43_P21": -1.2 - 15.6 / _KCAL}
_E_REF = 80.0
_H_REF = 60.0
# Reaction D: its species are evaluated and converged but NOT in pbe_df.json
# and the record carries no benchmark reference, so the network that reports
# it alone has nothing that can be scored against the references.
_D = {
    "name": "isrc26_001",
    "reactants": ["slim16@isrc26_ch3"],
    "products": ["slim16@isrc26_h", "slim16@isrc26_ch4"],
    "coeffs": [-1.0, 1.0, 1.0],
}
_D_SPECIES = {"slim16@isrc26_ch3": -0.6,
              "slim16@isrc26_h": -0.3,
              "slim16@isrc26_ch4": -0.55}
# Reaction F, subset ADIM6, converges under every network but carries no
# benchmark reference: it is outside every reference-basis number (its
# PBE-DF reaction energy, 10.0 kcal/mol, keeps it inside the PBE-DF basis).
_F = {
    "name": "adim6_001",
    "reactants": ["slim16@adim6_AD2"],
    "products": ["slim16@adim6_AM2"],
    "coeffs": [-1.0, 2.0],
}
_F_SPECIES = {"slim16@adim6_AD2": -2.0,
              "slim16@adim6_AM2": -1.0 + 5.0 / _KCAL}

# The hand values against the references, with the full set's weights:
# N 4, mean |ref| 75, scale 75 / 4 = 18.75; N_i 1 for ALKBDE10 (m 40) and
# RC21 (m 120), 2 for RSE43 (m 70). total = 18.75 * sum N_i MAD_i / m_i.
# S_a: A 0.4, B 1.2, H 0.7 converged, E 3.5 unconverged ->
#   all 18.75 * (0.01 + 0.01 + 2 * 2.1 / 70) = 1.5,
#   converged 18.75 * (0.01 + 0.01 + 2 * 0.7 / 70) = 0.75.
# slim05_b: A 0.2, B 0.72 converged, E 1.4 unconverged, H with no reported
#   energy (unscored: out of every mean, inside the weights) ->
#   all 18.75 * (0.005 + 0.006 + 2 * 1.4 / 70) = 0.95625,
#   converged 18.75 * (0.005 + 0.006) = 0.20625, RSE43 dropped. (No hand
#   total sits on a two-decimal rounding tie: the legend text is read back.)
# S_e: everything converged, A 0.4, B 1.2, E 0.7, H without an energy
#   (RSE43 keeps its weight of two, its mean is E's) -> 0.75 both.
# S_g: the same errors as S_e with nothing converged -> all 0.75, converged
#   none (every subset dropped).
# PBE-DF: 1.0, 1.8, 0.6, 5.0 -> 18.75 * (0.025 + 0.015 + 2 * 2.8 / 70) = 2.25;
# its RSE43 bar, 2.8, is the tallest bar of the figure.
_MAE_S_A = {"ALKBDE10": 0.4, "RC21": 1.2, "RSE43": 2.1}
_MAE_S_A_CONV = {"ALKBDE10": 0.4, "RC21": 1.2, "RSE43": 0.7}
_WT_S_A = (1.5, 0.75)
_MAE_SLIM05_B = {"ALKBDE10": 0.2, "RC21": 0.72, "RSE43": 1.4}
_MAE_SLIM05_B_CONV = {"ALKBDE10": 0.2, "RC21": 0.72, "RSE43": None}
_WT_SLIM05_B = (0.95625, 0.20625)
_MAE_S_E = {"ALKBDE10": 0.4, "RC21": 1.2, "RSE43": 0.7}
_WT_S_E = (0.75, 0.75)
_MAE_S_G_CONV = {"ALKBDE10": None, "RC21": None, "RSE43": None}
_WT_S_G_ALL = 0.75
_MAE_PBE = {"ALKBDE10": 1.0, "RC21": 1.8, "RSE43": 2.8}
_WT_PBE = 2.25

_ARCH_SHARED = "deep_3x16"
_ARCH_OTHER = "deep_geom_3x16"


def _pbe_df_json() -> dict:
    species = {name: {"E_pbe_df": value}
               for name, value in {**_A_SPECIES, **_B_SPECIES,
                                   **_RSE43_SPECIES, **_F_SPECIES}.items()}
    return {"identity": {"pool": "slim16"}, "n_species_not_df": 0,
            "species": species}


def _reaction_record(record: dict, de_nn: float, ref) -> dict:
    out = dict(record)
    out["de_nn_kcalmol"] = de_nn
    out["reaction_energy_ref_kcalmol"] = ref
    return out


def _molecules(species: dict, *, converged, unconverged=()) -> list:
    """One per-molecule record per species, ``scf_converged`` as given,
    the names in ``unconverged`` overriding it to False."""
    return [{"molecule": name, "E_total_nn": value - 0.001,
             "E_pbe": value + 1e-5,
             "scf_converged": bool(converged) and name not in unconverged}
            for name, value in species.items()]


def _channel(run: Path, idx: int, molecules: list, reactions: list) -> None:
    d = run / "checkpoints" / f"spec_{idx:04d}" / "eval_holdout_converged"
    d.mkdir(parents=True)
    (d / "per_molecule.json").write_text(json.dumps(molecules))
    (d / "per_reaction.json").write_text(json.dumps(reactions))


def _run(tmp_path: Path) -> Path:
    """Six networks: S_a (deep_3x16; E unconverged through its species
    P35), slim05_b (deep_3x16; E unconverged, H reported without an energy,
    so RSE43 has no converged reaction), S_e (deep_geom_3x16; everything
    converged, H reported without an energy), S_g (deep_geom_3x16; nothing converged), slim05_c with no
    channel at all, and v7_d whose one reaction has species absent from the
    PBE-DF table and no reference. Every evaluated network also reports F,
    which carries no reference."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "kind": "slim16_eval", "width": 4,
        "identity": {"pool": "slim16"},
        "networks": [
            {"index": 0, "label": "S_a", "kind": "pretrain",
             "arch_name": _ARCH_SHARED, "certificate_verdict": "PASS"},
            {"index": 1, "label": "slim05_b", "kind": "pretrain",
             "arch_name": _ARCH_SHARED, "certificate_verdict": "PASS"},
            {"index": 2, "label": "S_e", "kind": "pretrain",
             "arch_name": _ARCH_OTHER, "certificate_verdict": "PASS"},
            {"index": 5, "label": "S_g", "kind": "pretrain",
             "arch_name": _ARCH_OTHER, "certificate_verdict": "PASS"},
            {"index": 3, "label": "slim05_c", "kind": "pretrain",
             "arch_name": "deep_attn_3x16", "certificate_verdict": "PASS"},
            {"index": 4, "label": "v7_d", "kind": "trained",
             "arch_name": "medium", "certificate_verdict": None},
        ]}))
    (run / "pbe_df.json").write_text(json.dumps(_pbe_df_json()))
    always = _molecules({**_A_SPECIES, **_B_SPECIES, **_F_SPECIES},
                        converged=True)
    without_reference = [_reaction_record(_F, 7.0, None)]
    _channel(run, 0,
             always + _molecules(_RSE43_SPECIES, converged=True,
                                 unconverged=("slim16@rse43_P35",)),
             [_reaction_record(_A, 40.4, _A_REF),
              _reaction_record(_B, 121.2, _B_REF),
              _reaction_record(_E, 83.5, _E_REF),
              _reaction_record(_H, 60.7, _H_REF)] + without_reference)
    _channel(run, 1,
             always + _molecules(_RSE43_SPECIES, converged=True,
                                 unconverged=("slim16@rse43_P35",)),
             [_reaction_record(_A, 40.2, _A_REF),
              _reaction_record(_B, 120.72, _B_REF),
              _reaction_record(_E, 81.4, _E_REF),
              _reaction_record(_H, float("nan"), _H_REF)] + without_reference)
    _channel(run, 2,
             always + _molecules(_RSE43_SPECIES, converged=True),
             [_reaction_record(_A, 40.4, _A_REF),
              _reaction_record(_B, 121.2, _B_REF),
              _reaction_record(_E, 80.7, _E_REF),
              _reaction_record(_H, float("nan"), _H_REF)] + without_reference)
    _channel(run, 5,
             _molecules({**_A_SPECIES, **_B_SPECIES, **_F_SPECIES,
                         **_RSE43_SPECIES}, converged=False),
             [_reaction_record(_A, 40.4, _A_REF),
              _reaction_record(_B, 121.2, _B_REF),
              _reaction_record(_E, 80.7, _E_REF),
              _reaction_record(_H, 60.7, _H_REF)] + without_reference)
    _channel(run, 4,
             _molecules(_D_SPECIES, converged=True),
             [_reaction_record(_D, -0.25 * _KCAL, None)])
    return run


def _approx_map(values: dict):
    return {key: (None if value is None else pytest.approx(value))
            for key, value in values.items()}


def test_collect_set_errors_against_the_references(tmp_path):
    """Each evaluated network is scored on every reaction it reports, with
    the weights of the full set of referenced reactions: the per-subset
    errors over all reactions, and the WTMAD-2 over all reactions and over
    the converged ones alone, the converged total keeping the full weights
    and dropping, by name, a subset with no converged reaction; a network
    with nothing converged has no converged total. A reaction reported
    without an energy keeps its weight, enters no mean and is named. The
    unconverged reactions are named per network. A network without a
    channel and one with no reaction that carries a reference are absent,
    each with its own reason. PBE-DF is scored over every reaction its table
    covers. A subset whose references average to zero carries no weight and
    is named. The table's rows give the same numbers."""
    run = _run(tmp_path)
    order, records, pbe, comparators = figure.collect_set_errors(run)
    assert order == ["S_a", "slim05_b", "S_e", "S_g", "slim05_c", "v7_d"]
    # the run carries no comparator table
    assert comparators == []

    expected = {
        "S_a": (_MAE_S_A, _MAE_S_A_CONV, _WT_S_A, ["rse43_033"], [], [],
                _ARCH_SHARED, "S"),
        "slim05_b": (_MAE_SLIM05_B, _MAE_SLIM05_B_CONV, _WT_SLIM05_B,
                     ["rse43_033"], ["rse43_019"], ["RSE43"], _ARCH_SHARED,
                     "slim05"),
        "S_e": (_MAE_S_E, _MAE_S_E, _WT_S_E, [], ["rse43_019"], [], _ARCH_OTHER, "S"),
        "S_g": (_MAE_S_E, _MAE_S_G_CONV, (_WT_S_G_ALL, None),
                ["alkbde10_006", "rc21_010", "rse43_019", "rse43_033"], [],
                ["ALKBDE10", "RC21", "RSE43"], _ARCH_OTHER, "S"),
    }
    for label, (mae, conv, wt, unconverged, unscored, dropped, arch,
                group) in expected.items():
        record = records[label]
        assert record["mae"] == _approx_map(mae), label
        assert record["mae_converged"] == _approx_map(conv), label
        assert record["wtmad2_all"] == pytest.approx(wt[0]), label
        if wt[1] is None:
            assert record["wtmad2_converged"] is None, label
        else:
            assert record["wtmad2_converged"] == pytest.approx(wt[1]), label
        assert record["unconverged"] == unconverged, label
        assert record["unscored"] == unscored, label
        assert record["dropped"] == dropped, label
        assert record["arch_name"] == arch
        assert record["group"] == group
        assert record["reason"] is None
    assert "ADIM6" not in records["S_a"]["mae"]

    absent_c = records["slim05_c"]
    assert absent_c["reason"] == "no evaluated channel"
    assert absent_c["mae"] == {} and absent_c["wtmad2_all"] is None
    absent_d = records["v7_d"]
    assert absent_d["reason"] and absent_d["reason"] != "no evaluated channel"
    assert absent_d["mae"] == {} and absent_d["wtmad2_all"] is None
    assert absent_d["group"] == "v7"

    assert pbe["mae"] == _approx_map(_MAE_PBE)
    assert pbe["wtmad2"] == pytest.approx(_WT_PBE)
    assert pbe["n_reactions"] == 4 and pbe["unscored"] == []

    # the table's own rows and totals state the same numbers
    _manifest, width, pbe_df, subsets, _n = table.run_context(run)
    molecules, reactions = table.channel_records(run, width, 0)
    rows_nn, rows_pbe = table.reference_rows(molecules, reactions, pbe_df, subsets)
    scored = table.wtmad2_full_weights(rows_nn)
    assert scored["all"] == pytest.approx(_WT_S_A[0])
    assert scored["converged"] == pytest.approx(_WT_S_A[1])
    assert {name: entry["MAD_all"] for name, entry in scored["per_subset"].items()} \
        == _approx_map(_MAE_S_A)
    assert scored["per_subset"]["RSE43"]["MAD_converged"] == pytest.approx(0.7)
    assert scored["per_subset"]["RSE43"]["N_i"] == 2
    assert scored["unconverged"] == ["rse43_033"] and scored["dropped"] == []
    assert scored["n_reactions"] == 4
    assert table.wtmad2_full_weights(rows_pbe)["all"] == pytest.approx(_WT_PBE)
    # slim05_b's unscored reaction keeps RSE43's weight of two
    molecules, reactions = table.channel_records(run, width, 1)
    rows_nn, _rows_pbe = table.reference_rows(molecules, reactions, pbe_df, subsets)
    scored = table.wtmad2_full_weights(rows_nn)
    assert scored["per_subset"]["RSE43"]["N_i"] == 2
    assert scored["n_reactions"] == 4 and scored["unscored"] == ["rse43_019"]
    # a subset whose references average to zero carries no weight: named,
    # its term out of both totals, its reaction still counted in N
    scored = table.wtmad2_full_weights([
        {"name": "z1", "subset": "Z", "de_ref": 0.0, "de_nn": 0.3, "converged": True},
        {"name": "y1", "subset": "Y", "de_ref": 10.0, "de_nn": 11.0, "converged": True}])
    assert scored["unweighted"] == ["Z"]
    assert scored["all"] == pytest.approx(0.25) and scored["converged"] == pytest.approx(0.25)

    # a network scored on another list of referenced reactions is refused by
    # name: the weights in the legend are one set
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["networks"].append({"index": 6, "label": "S_h", "kind": "pretrain",
                                 "arch_name": _ARCH_OTHER, "certificate_verdict": "PASS"})
    manifest_path.write_text(json.dumps(manifest))
    _channel(run, 6, _molecules({**_A_SPECIES, **_B_SPECIES}, converged=True),
             [_reaction_record(_A, 40.4, _A_REF), _reaction_record(_B, 121.2, _B_REF)])
    with pytest.raises(ValueError, match="S_h"):
        figure.collect_set_errors(run)


def test_the_drawn_figure_reads_back_the_set_errors(tmp_path, monkeypatch):
    """Read from the matplotlib axes: the subsets on the x axis in name
    order, per subset one bar per evaluated network (in manifest order) and
    then one PBE-DF bar, each at its own x and as tall as the subset's error
    over all reactions; a marker on the bar at the converged-only error
    where the subset holds converged and unconverged reactions both, and no
    marker where every reaction converged or none did. Bars of one
    architecture share a colour and differ from another architecture's, bars
    of one network group share a hatch and differ from another group's, the
    PBE-DF bars share one colour of their own, every bar lies inside a y
    range common to the axes, the legend reads each network's label with its
    two WTMAD-2 values and PBE-DF with its one, and the title names the
    absent networks, the unconverged reactions and the dropped subset."""
    closed = []
    monkeypatch.setattr(figure.plt, "close",
                        lambda fig=None, *a, **k: closed.append(fig))
    order, records, pbe, _comparators = figure.collect_set_errors(_run(tmp_path))
    manifest = figure.plot_slim16_set_errors(order, records, pbe,
                                             tmp_path / "fig.png")
    fig = closed[-1] if closed and closed[-1] is not None else figure.plt.gcf()
    try:
        subsets = ["ALKBDE10", "RC21", "RSE43"]
        assert manifest["subsets"] == subsets
        ticks = [t.get_text() for ax in fig.axes for t in ax.get_xticklabels()]
        assert [t for t in ticks if t] == subsets

        bars = sorted((p for ax in fig.axes for p in ax.patches
                       if p.get_width() > 0),
                      key=lambda p: p.get_x() + p.get_width() / 2)
        centres = [round(p.get_x() + p.get_width() / 2, 9) for p in bars]
        assert len(set(centres)) == len(bars)
        expected = []
        for name in subsets:
            for label, mae in (("S_a", _MAE_S_A), ("slim05_b", _MAE_SLIM05_B),
                               ("S_e", _MAE_S_E), ("S_g", _MAE_S_E)):
                expected.append((label, mae[name]))
            expected.append(("PBE-DF", _MAE_PBE[name]))
        assert len(bars) == len(expected)
        assert [p.get_height() for p in bars] == pytest.approx(
            [value for _label, value in expected])

        # the one marker: S_a's RSE43 bar, at the converged-only error,
        # black; slim05_b and S_g have nothing converged there, S_e nothing
        # unconverged
        lines = [line for ax in fig.axes for line in ax.lines]
        markers = [(round(float(line.get_xdata()[0]), 9), float(line.get_ydata()[0]))
                   for line in lines]
        rse43_s_a = [centres[i] for i, (label, _v) in enumerate(expected)
                     if label == "S_a"][2]
        assert markers == [(rse43_s_a, pytest.approx(0.7))]
        assert mcolors.to_hex(lines[0].get_color()) == "#000000"
        assert manifest["markers"] == [{"label": "S_a", "subset": "RSE43",
                                        "value": pytest.approx(0.7)}]

        style = {}
        for (label, _value), patch in zip(expected, bars):
            style.setdefault(label, set()).add(
                (mcolors.to_hex(patch.get_facecolor(), keep_alpha=False),
                 patch.get_hatch() or ""))
        assert all(len(found) == 1 for found in style.values()), style
        color = {label: next(iter(found))[0] for label, found in style.items()}
        hatch = {label: next(iter(found))[1] for label, found in style.items()}
        assert color["S_a"] == mcolors.to_hex(arch_style.arch_color(_ARCH_SHARED))
        assert color["S_e"] == mcolors.to_hex(arch_style.arch_color(_ARCH_OTHER))
        assert color["slim05_b"] == color["S_a"]
        assert color["S_e"] != color["S_a"]
        assert color["PBE-DF"] not in {color["S_a"], color["S_e"]}
        assert hatch["S_e"] == hatch["S_a"]
        assert hatch["slim05_b"] != hatch["S_a"]

        limits = {ax.get_ylim() for ax in fig.axes if ax.patches}
        assert len(limits) == 1
        low, high = limits.pop()
        assert all(low <= 0.0 and p.get_height() <= high for p in bars)

        legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes
                                       if ax.get_legend() is not None]
        texts = [t.get_text() for legend in legends for t in legend.get_texts()]
        assert texts == [f"S_a ({_WT_S_A[0]:.2f} / {_WT_S_A[1]:.2f})",
                         f"slim05_b ({_WT_SLIM05_B[0]:.2f} / {_WT_SLIM05_B[1]:.2f})",
                         f"S_e ({_WT_S_E[0]:.2f} / {_WT_S_E[1]:.2f})",
                         f"S_g ({_WT_S_G_ALL:.2f} / n/a)",
                         f"PBE-DF ({_WT_PBE:.2f})"]
        assert len(legends) == 1 and "4 reactions" in legends[0].get_title().get_text()
        assert manifest["n_reactions"] == 4

        titles = " ".join(
            [fig._suptitle.get_text() if fig._suptitle is not None else ""]
            + [ax.get_title() for ax in fig.axes])
        for text in ("slim05_c", "v7_d", "rse43_033", "rse43_019", "RSE43",
                     records["slim05_c"]["reason"], records["v7_d"]["reason"]):
            assert text in titles
        assert set(manifest["absent"]) == {"slim05_c", "v7_d"}
        assert manifest["unconverged"] == {
            "S_a": ["rse43_033"], "slim05_b": ["rse43_033"],
            "S_g": ["alkbde10_006", "rc21_010", "rse43_019", "rse43_033"]}
        assert manifest["unscored"] == {"slim05_b": ["rse43_019"], "S_e": ["rse43_019"]}
        assert "reported without an energy" in titles and "slim05_b: rse43_019" in titles
        assert manifest["dropped"] == {"slim05_b": ["RSE43"],
                                       "S_g": ["ALKBDE10", "RC21", "RSE43"]}
    finally:
        monkeypatch.undo()
        figure.plt.close(fig)

    # the wrap: fourteen subsets of one network go into rows of thirteen
    # and one, each row an axes with its own tick labels
    closed.clear()
    monkeypatch.setattr(figure.plt, "close",
                        lambda fig=None, *a, **k: closed.append(fig))
    names = [f"SET{i:02d}" for i in range(14)]
    # SET01 and SET02 hold converged and unconverged reactions both; the
    # converged-only error of SET01 stands above every bar
    record = {"mae": {n: 1.0 for n in names},
              "mae_converged": {**{n: 1.0 for n in names}, "SET01": 5.0},
              "mixed": ["SET01", "SET02"],
              "wtmad2_all": 1.0, "wtmad2_converged": 1.0, "unconverged": [],
              "unscored": [], "dropped": [], "unweighted": [],
              "n_reactions": 14,
              "arch_name": _ARCH_SHARED, "group": "S", "reason": None}
    wide = figure.plot_slim16_set_errors(
        ["only"], {"only": record},
        {"mae": {}, "wtmad2": None, "n_reactions": 0, "unscored": []},
        tmp_path / "wide.png")
    fig = closed[-1]
    try:
        per_axes = [[t.get_text() for t in ax.get_xticklabels() if t.get_text()]
                    for ax in fig.axes]
        assert per_axes == [names[:13], names[13:]]
        assert wide["subsets"] == names and wide["legend"] == ["only (1.00 / 1.00)"]
        assert wide["yscale"] == "linear" and wide["y_floor"] == 0.0
        marks = [(line.get_ydata()[0], line.axes.get_ylim()[1])
                 for ax in fig.axes for line in ax.lines]
        assert [y for y, _high in marks] == pytest.approx([5.0, 1.0])
        assert all(y <= high for y, high in marks)
        assert wide["y_cap"] == pytest.approx(5.5)
        assert "14 reactions" in fig.legends[0].get_title().get_text()
    finally:
        monkeypatch.undo()
        figure.plt.close(fig)

    # the log copy: every axis logarithmic, the floor half the smallest
    # drawn value so every bar stays inside the limits
    closed.clear()
    monkeypatch.setattr(figure.plt, "close",
                        lambda fig=None, *a, **k: closed.append(fig))
    record["mae"]["SET00"] = record["mae_converged"]["SET00"] = 0.01
    logged = figure.plot_slim16_set_errors(
        ["only"], {"only": record},
        {"mae": {}, "wtmad2": None, "n_reactions": 0, "unscored": []},
        tmp_path / "wide_log.png", log=True)
    fig = closed[-1]
    try:
        assert logged["yscale"] == "log" and logged["y_floor"] == pytest.approx(0.005)
        assert all(ax.get_yscale() == "log" for ax in fig.axes)
        for ax in fig.axes:
            low, high = ax.get_ylim()
            assert low == pytest.approx(0.005)
            assert all(low < p.get_height() <= high for p in ax.patches)
    finally:
        monkeypatch.undo()
        figure.plt.close(fig)


def test_build_table_matches_the_hand_values_after_the_rewiring(tmp_path):
    """The metrics table keeps its WTMAD-2 against PBE-DF (the paper's
    converged filter, and over all reactions) and ends with the three
    reference-basis totals: the network over all its reactions and over the
    converged ones with the full weights, and PBE-DF; the CSV the tool
    writes carries them in those columns, a missing converged total as
    ``nan``."""
    run = _run(tmp_path)
    rows = table.build_table(run)
    by_label = {r["label"]: r for r in rows}
    a, b, e = by_label["S_a"], by_label["slim05_b"], by_label["S_e"]
    assert a["n_converged"] == 13
    assert a["n_reactions"] == 5
    # against PBE-DF (41.0, 121.8, 65.0, 10.0 for A, B, H, F; E out of the
    # converged rows): S_a |d| 0.6, 0.6, 4.3, 3.0
    assert a["wtmad2_paper_converged"] == pytest.approx(
        ((41.0 + 121.8 + 65.0 + 10.0) / 4) / 4
        * (0.6 / 41.0 + 0.6 / 121.8 + 4.3 / 65.0 + 3.0 / 10.0))
    # over all rows RSE43 holds E (|83.5 - 80.6| = 2.9) and H together
    assert a["wtmad2_paper_all"] == pytest.approx(
        ((41.0 + 121.8 + 2 * (80.6 + 65.0) / 2 + 10.0) / 5) / 5
        * (0.6 / 41.0 + 0.6 / 121.8 + 2 * (2.9 + 4.3) / 2 / ((80.6 + 65.0) / 2)
           + 3.0 / 10.0))
    # against the benchmark references, with the full set's weights
    assert a["wtmad2_ref_nn_all"] == pytest.approx(_WT_S_A[0])
    assert a["wtmad2_ref_nn_converged"] == pytest.approx(_WT_S_A[1])
    assert b["wtmad2_ref_nn_all"] == pytest.approx(_WT_SLIM05_B[0])
    assert b["wtmad2_ref_nn_converged"] == pytest.approx(_WT_SLIM05_B[1])
    assert e["wtmad2_ref_nn_all"] == pytest.approx(_WT_S_E[0])
    assert e["wtmad2_ref_nn_converged"] == pytest.approx(_WT_S_E[1])
    for row in (a, b, e):
        assert row["wtmad2_ref_pbe_df"] == pytest.approx(_WT_PBE)
    assert by_label["S_g"]["wtmad2_ref_nn_all"] == pytest.approx(_WT_S_G_ALL)
    assert math.isnan(by_label["S_g"]["wtmad2_ref_nn_converged"])
    # the three WTMAD-2 columns, then the comparators' six (empty here)
    assert table.COLUMNS[-9:] == ("wtmad2_ref_nn_all", "wtmad2_ref_nn_converged",
                                  "wtmad2_ref_pbe_df") + table.COMPARATOR_COLUMNS
    assert all(a.get(c) is None for c in table.COMPARATOR_COLUMNS)
    # slim05_c has no channel: its row carries the identity fields only
    assert "n_converged" not in by_label["slim05_c"]
    assert by_label["slim05_c"]["kind"] == "pretrain"

    assert table.main([str(run)]) == 0
    with open(run / "slim16_metrics.csv", newline="", encoding="utf-8") as handle:
        cells = {row["label"]: row for row in csv.DictReader(handle)}
    assert float(cells["S_a"]["wtmad2_ref_nn_all"]) == pytest.approx(_WT_S_A[0], abs=5e-4)
    assert float(cells["S_a"]["wtmad2_ref_nn_converged"]) == pytest.approx(
        _WT_S_A[1], abs=5e-4)
    assert float(cells["S_a"]["wtmad2_ref_pbe_df"]) == pytest.approx(_WT_PBE, abs=5e-4)
    assert cells["S_g"]["wtmad2_ref_nn_converged"] == "nan"
    assert cells["slim05_c"]["wtmad2_ref_nn_all"] == ""


def test_main_writes_the_png_beside_the_run(tmp_path):
    run = _run(tmp_path)
    assert figure.main([str(run)]) == 0
    assert figure.main([str(run), "--log"]) == 0
    assert (run / "slim16_wtmad2_log.png").is_file()
    out = run / "slim16_wtmad2.png"
    assert out.is_file()
    assert out.stat().st_size > 0

# The comparator tables beside PBE-DF: hand reaction energies (kcal/mol) of
# the run's four referenced reactions and of F; the per-species energies
# realize them on the fixture's stoichiometry. References: A 40, B 120, E 80,
# H 60; the scale of the full-weights WTMAD-2 is (40 + 120 + 2 * 70) / 4 / 4
# = 18.75 and a subset's term is N_i MAD_i / m_i (RSE43: N_i 2, m_i 70).
# r2SCAN: errors 0.6, 3.6, 1.4, 0.7 -> 18.75 (0.6/40 + 3.6/120 + 2 (1.05)/70)
#   = 1.40625; reaction MAE 1.575; its RC21 bar (3.6) is the figure's tallest.
# B3LYP: errors 1.2, -, 2.6, 0.4 (rc21_me carries no energy: rc21_010 is
#   unscored, its weight kept) -> 18.75 (1.2/40 + 2 (1.5)/70) = 1.3661; MAE 1.4.
# wB97M-V: errors 0.2, 0.9, 0.5, 1.1 -> 18.75 (0.2/40 + 0.9/120 + 2 (0.8)/70)
#   = 0.6629; MAE 0.675.
_CMP_RXN = {"r2scan": {"A": 40.6, "B": 116.4, "E": 81.4, "H": 59.3, "F": 6.0},
            "b3lyp": {"A": 38.8, "B": None, "E": 82.6, "H": 60.4, "F": 9.0},
            "wb97m-v": {"A": 40.2, "B": 120.9, "E": 79.5, "H": 61.1, "F": 4.0}}
_CMP_LABEL = {"r2scan": "r2SCAN", "b3lyp": "B3LYP", "wb97m-v": "wB97M-V"}
_CMP_WT = {"r2scan": 18.75 * (0.6 / 40 + 3.6 / 120 + 2 * 1.05 / 70),
           "b3lyp": 18.75 * (1.2 / 40 + 2 * 1.5 / 70),
           "wb97m-v": 18.75 * (0.2 / 40 + 0.9 / 120 + 2 * 0.8 / 70)}
_CMP_MAE = {"r2scan": 1.575, "b3lyp": 1.4, "wb97m-v": 0.675}
_CMP_SET_MAE = {"r2scan": {"ALKBDE10": 0.6, "RC21": 3.6, "RSE43": 1.05},
                "b3lyp": {"ALKBDE10": 1.2, "RSE43": 1.5},
                "wb97m-v": {"ALKBDE10": 0.2, "RC21": 0.9, "RSE43": 0.8}}


def _comparator_species(rxn: dict) -> dict:
    """Per-species energies (Hartree) whose reaction energies are ``rxn``
    on the stoichiometry of the fixture's reactions; None for a species
    that failed."""
    a, b, e, h, f = (rxn[k] for k in "ABEHF")
    return {"slim16@alkbde10_lif": -1.0, "slim16@alkbde10_li": -0.5,
            "slim16@alkbde10_f": -0.5 + a / _KCAL,
            "slim16@rc21_4e": -2.0, "slim16@rc21_4p": -1.0,
            "slim16@rc21_me": None if b is None else -1.0 + b / _KCAL,
            "slim16@rse43_E35": -1.0, "slim16@rse43_P1": -0.5,
            "slim16@rse43_P35": -1.2, "slim16@rse43_E1": -0.3 + e / _KCAL,
            "slim16@rse43_E21": -1.0, "slim16@rse43_P21": -1.2 + (h - e) / _KCAL,
            "slim16@adim6_AD2": -2.0, "slim16@adim6_AM2": -1.0 + f / (2.0 * _KCAL)}


def _functionals_df_json() -> dict:
    """The comparator file of the job (``comparators-df``), its functionals
    in the reverse of the display order and a ``pbe`` entry, the job's
    sanity pin, which is never a comparator."""
    functionals = {}
    for key in ("pbe", "wb97m-v", "b3lyp", "r2scan"):
        rxn = _CMP_RXN.get(key, _CMP_RXN["r2scan"])
        species = {name: ({"error": "RuntimeError: the SCF did not converge"}
                          if energy is None else
                          {"E_df": energy, "n_ao": 1, "reference_eri_path": "df-aux240"})
                   for name, energy in _comparator_species(rxn).items()}
        functionals[key] = {"label": _CMP_LABEL.get(key, "PBE"), "xc": key,
                            "vv10": key == "wb97m-v", "hybrid": key != "r2scan",
                            "meta_gga": key != "b3lyp", "n_species": len(species),
                            "n_converged": sum(1 for v in species.values() if "E_df" in v),
                            "n_species_not_df": 0, "species": species}
    return {"identity": {"pool": "slim16", "nlc_grid_level": 3},
            "species_slice": None, "functionals": functionals}


def test_the_comparator_tables_draw_grey_bars_after_pbe_df(tmp_path, monkeypatch):
    """A run carrying the comparator file: the reader returns the tables in
    the display order (r2SCAN, B3LYP, wB97M-V) without the pbe entry, the
    failed species without an energy; the table's six comparator columns
    equal the hand WTMAD-2 and reaction MAE on every evaluated network's
    row; and the figure, read off the axes, draws per subset the networks,
    PBE-DF and then one bar per comparator as tall as the subset's hand
    error (B3LYP none in RC21), no two bars overlapping, every bar inside
    the y range (the tallest is r2SCAN's), the legend reading the tracked
    five lines and then ``label (WTMAD-2)`` per comparator in its grey, the
    three greys distinct from each other, from PBE-DF's and from the
    architecture colours of the drawn networks."""
    run = _run(tmp_path)
    (run / "functionals_df.json").write_text(json.dumps(_functionals_df_json()))

    found = table.comparator_tables(run)
    assert [(k, label) for k, label, _t in found] == [
        ("r2scan", "r2SCAN"), ("b3lyp", "B3LYP"), ("wb97m-v", "wB97M-V")]
    assert dict(found[1][2])["slim16@rc21_me"] is None
    rows = {row["label"]: row for row in table.build_table(run)}
    for label in ("S_a", "slim05_b", "S_e", "S_g"):
        for key, column in (("r2scan", "r2scan"), ("b3lyp", "b3lyp"), ("wb97m-v", "wb97mv")):
            assert rows[label][f"wtmad2_ref_{column}"] == pytest.approx(_CMP_WT[key])
            assert rows[label][f"mae_rxn_ref_{column}"] == pytest.approx(_CMP_MAE[key])
    assert rows["S_a"]["wtmad2_ref_pbe_df"] == pytest.approx(_WT_PBE)

    closed = []
    monkeypatch.setattr(figure.plt, "close",
                        lambda fig=None, *a, **k: closed.append(fig))
    order, records, pbe, comparators = figure.collect_set_errors(run)
    assert [c["key"] for c in comparators] == ["r2scan", "b3lyp", "wb97m-v"]
    for c in comparators:
        assert c["mae"] == _approx_map(_CMP_SET_MAE[c["key"]])
        assert c["wtmad2"] == pytest.approx(_CMP_WT[c["key"]]) and c["n_reactions"] == 4
    manifest = figure.plot_slim16_set_errors(order, records, pbe, tmp_path / "fig.png",
                                             comparators=comparators)
    fig = closed[-1]
    try:
        subsets = ["ALKBDE10", "RC21", "RSE43"]
        bars = sorted((p for ax in fig.axes for p in ax.patches if p.get_width() > 0),
                      key=lambda p: p.get_x() + p.get_width() / 2)
        expected = []
        for i, name in enumerate(subsets):
            for label, mae in (("S_a", _MAE_S_A), ("slim05_b", _MAE_SLIM05_B),
                               ("S_e", _MAE_S_E), ("S_g", _MAE_S_E)):
                expected.append((label, i, mae[name]))
            expected.append(("PBE-DF", i, _MAE_PBE[name]))
            for key in ("r2scan", "b3lyp", "wb97m-v"):
                if name in _CMP_SET_MAE[key]:
                    expected.append((_CMP_LABEL[key], i, _CMP_SET_MAE[key][name]))
        assert len(bars) == len(expected)
        assert [p.get_height() for p in bars] == pytest.approx([v for _l, _i, v in expected])
        centres = [p.get_x() + p.get_width() / 2 for p in bars]
        assert all(abs(c - i) < 0.5 for c, (_l, i, _v) in zip(centres, expected))
        edges = sorted((p.get_x(), p.get_x() + p.get_width()) for p in bars)
        assert all(right <= left + 1e-12 for (_a, right), (left, _b) in zip(edges, edges[1:]))
        low, high = next(ax.get_ylim() for ax in fig.axes if ax.patches)
        tallest = max(p.get_height() for p in bars)
        assert low <= 0.0 and tallest == pytest.approx(3.6) and tallest <= high

        colour = {}
        for (label, _i, _v), patch in zip(expected, bars):
            colour.setdefault(label, set()).add(
                mcolors.to_hex(patch.get_facecolor(), keep_alpha=False))
        assert all(len(found_) == 1 for found_ in colour.values()), colour
        greys = [next(iter(colour[_CMP_LABEL[k]])) for k in ("r2scan", "b3lyp", "wb97m-v")]
        assert len(set(greys)) == 3
        assert all(mcolors.to_rgb(g)[0] == mcolors.to_rgb(g)[1] == mcolors.to_rgb(g)[2]
                   for g in greys)
        others = {next(iter(colour[label])) for label in ("S_a", "S_e", "PBE-DF")}
        assert not set(greys) & others

        legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes
                                       if ax.get_legend() is not None]
        assert len(legends) == 1
        texts = [t.get_text() for t in legends[0].get_texts()]
        assert texts == [f"S_a ({_WT_S_A[0]:.2f} / {_WT_S_A[1]:.2f})",
                         f"slim05_b ({_WT_SLIM05_B[0]:.2f} / {_WT_SLIM05_B[1]:.2f})",
                         f"S_e ({_WT_S_E[0]:.2f} / {_WT_S_E[1]:.2f})",
                         f"S_g ({_WT_S_G_ALL:.2f} / n/a)",
                         f"PBE-DF ({_WT_PBE:.2f})"] + [
                             f"{_CMP_LABEL[k]} ({_CMP_WT[k]:.2f})"
                             for k in ("r2scan", "b3lyp", "wb97m-v")]
        handles = list(legends[0].legend_handles)
        assert [mcolors.to_hex(h.get_facecolor(), keep_alpha=False)
                for h in handles[-3:]] == greys
        assert [c["key"] for c in manifest["comparators"]] == ["r2scan", "b3lyp", "wb97m-v"]
        assert sum(1 for b in manifest["bars"] if b["label"] in _CMP_LABEL.values()) == 8
    finally:
        monkeypatch.undo()
        figure.plt.close(fig)
