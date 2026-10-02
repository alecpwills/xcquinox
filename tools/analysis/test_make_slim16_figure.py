"""The Slim16 figure: the paper's WTMAD-2 per network per subset.

The figure stacks each network's subsets into one horizontal bar whose width
is the network's total WTMAD-2, so a reader sees which subsets dominate each
network on one axis. The collect layer reuses the table's own row assembly
(``slim16_table``), so the figure and the metrics CSV state one set of
numbers.

Oracle: synthetic run dirs with hand-computed WTMAD-2 (two subsets chosen so
the shares are exact: both contributions equal, so each share is 0.5 and the
total is exactly 0.15 kcal/mol), the converged-only filter exercised by a
network whose second reaction never converged, and ``build_table`` checked
against the same hand values after its rewiring onto the shared functions.
"""
from __future__ import annotations

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


figure = _load("make_slim16_figure")
table = figure.slim16_table

_KCAL = 627.5094740631

# Reaction A, subset ALKBDE10 (name and stoichiometry from load_full_slim):
# de_ref = (+1.0 - 0.5 - 0.4) Ha = 0.1 Ha; with de_nn offset +0.05 kcal the
# subset's MAD is 0.05 and its contribution 0.05/(0.1*KCAL) = 0.5/KCAL.
_A = {
    "name": "alkbde10_006",
    "reactants": ["slim16@alkbde10_lif"],
    "products": ["slim16@alkbde10_li", "slim16@alkbde10_f"],
    "coeffs": [-1.0, 1.0, 1.0],
}
_A_SPECIES = {"slim16@alkbde10_lif": -1.0,
              "slim16@alkbde10_li": -0.5,
              "slim16@alkbde10_f": -0.4}
# Reaction B, subset RC21: de_ref = (2.0 - 1.0 - 0.5) Ha = 0.5 Ha; the +0.25
# kcal offset makes its contribution 0.25/(0.5*KCAL) = 0.5/KCAL too, so the
# total is (0.3*KCAL/2)*(1/KCAL) = 0.15 kcal/mol EXACTLY and each share 0.5.
_B = {
    "name": "rc21_010",
    "reactants": ["slim16@rc21_4e"],
    "products": ["slim16@rc21_4p", "slim16@rc21_me"],
    "coeffs": [-1.0, 1.0, 1.0],
}
_B_SPECIES = {"slim16@rc21_4e": -2.0,
              "slim16@rc21_4p": -1.0,
              "slim16@rc21_me": -0.5}
# Reaction D, the smoke-run shape: its species are evaluated and converged,
# but they are NOT in pbe_df.json (the slice runs' footing covers fewer
# species than the reactions name), so de_ref is nan and the network's
# WTMAD-2 is not finite.
_D = {
    "name": "isrc26_001",
    "reactants": ["slim16@isrc26_ch3"],
    "products": ["slim16@isrc26_h", "slim16@isrc26_ch4"],
    "coeffs": [-1.0, 1.0, 1.0],
}
_D_SPECIES = {"slim16@isrc26_ch3": -0.6,
              "slim16@isrc26_h": -0.3,
              "slim16@isrc26_ch4": -0.55}


def _pbe_df_json() -> dict:
    species = {name: {"E_pbe_df": value}
               for name, value in {**_A_SPECIES, **_B_SPECIES}.items()}
    return {"identity": {"pool": "slim16"}, "n_species_not_df": 0,
            "species": species}


def _reaction_record(record: dict, de_nn: float) -> dict:
    out = dict(record)
    out["de_nn_kcalmol"] = de_nn
    return out


def _molecules(species: dict, *, converged: bool) -> list:
    return [{"molecule": name, "E_total_nn": value - 0.001,
             "E_pbe": value + 1e-5, "scf_converged": converged}
            for name, value in species.items()]


def _channel(run: Path, idx: int, molecules: list, reactions: list) -> None:
    d = run / "checkpoints" / f"spec_{idx:04d}" / "eval_holdout_converged"
    d.mkdir(parents=True)
    (d / "per_molecule.json").write_text(json.dumps(molecules))
    (d / "per_reaction.json").write_text(json.dumps(reactions))


def _run(tmp_path: Path) -> Path:
    """Four networks: netA both reactions converged (total 0.15, shares
    0.5/0.5); netB carrying both reactions but B's species unconverged (the
    converged filter keeps A alone: total 0.10, share 1.0); netC with no
    channel at all (recorded absent); netD the smoke-run shape (species
    evaluated and converged, absent from the PBE-DF table, WTMAD-2 not
    finite)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "kind": "slim16_eval", "width": 4,
        "identity": {"pool": "slim16"},
        "networks": [
            {"index": 0, "label": "netA", "kind": "pretrain",
             "arch_name": "deep_3x16", "certificate_verdict": "PASS"},
            {"index": 1, "label": "netB", "kind": "pretrain",
             "arch_name": "deep_3x16", "certificate_verdict": "PASS"},
            {"index": 2, "label": "netC", "kind": "trained",
             "arch_name": "medium", "certificate_verdict": None},
            {"index": 3, "label": "netD", "kind": "trained",
             "arch_name": "medium", "certificate_verdict": None},
        ]}))
    (run / "pbe_df.json").write_text(json.dumps(_pbe_df_json()))
    _channel(run, 0,
             _molecules(_A_SPECIES, converged=True)
             + _molecules(_B_SPECIES, converged=True),
             [_reaction_record(_A, 0.1 * _KCAL + 0.05),
              _reaction_record(_B, 0.5 * _KCAL + 0.25)])
    # netB: A converged with offset +0.10 (MAD 0.10, the one-subset identity:
    # total = MAD); B present but its species never converged, with an offset
    # (+0.30) that would move the total to 0.24 if the filter were dropped.
    _channel(run, 1,
             _molecules(_A_SPECIES, converged=True)
             + _molecules(_B_SPECIES, converged=False),
             [_reaction_record(_A, 0.1 * _KCAL + 0.10),
              _reaction_record(_B, 0.5 * _KCAL + 0.30)])
    # netD: everything about the channel says evaluated and converged; only
    # the PBE-DF table is short a species, so the total is not finite.
    _channel(run, 3,
             _molecules(_D_SPECIES, converged=True),
             [_reaction_record(_D, -0.25 * _KCAL)])
    return run


def test_collect_matches_the_hand_values(tmp_path):
    order, records = figure.collect_subset_wtmad2(_run(tmp_path))
    assert order == ["netA", "netB", "netC", "netD"]
    a = records["netA"]
    assert a["total"] == pytest.approx(0.15)
    assert set(a["shares"]) == {"ALKBDE10", "RC21"}
    assert a["shares"]["ALKBDE10"] == pytest.approx(0.5)
    assert a["shares"]["RC21"] == pytest.approx(0.5)
    # the shares are shares OF THE TOTAL: they sum to the printed number
    assert sum(a["shares"].values()) * a["total"] == pytest.approx(a["total"])
    b = records["netB"]
    assert b["total"] == pytest.approx(0.10)
    assert b["shares"] == {"ALKBDE10": pytest.approx(1.0)}


def test_an_unevaluated_network_is_recorded_absent(tmp_path):
    order, records = figure.collect_subset_wtmad2(_run(tmp_path))
    assert records["netC"]["total"] is None
    assert records["netC"]["reason"] == "no evaluated channel"
    manifest = figure.plot_slim16_wtmad2(
        order, records, tmp_path / "fig.png")
    assert "netC" in manifest["absent"]
    assert {b["label"] for b in manifest["bars"]} == {"netA", "netB"}


def test_a_not_finite_wtmad2_is_not_labeled_no_evaluated_channel(tmp_path):
    """The smoke-run shape, found on real data: the channel exists and every
    species converged, but the run's PBE-DF table does not cover the
    reaction's species, so the total is nan. Lumping that network under the
    no-channel label would state a falsehood about the run."""
    order, records = figure.collect_subset_wtmad2(_run(tmp_path))
    d = records["netD"]
    assert d["total"] is None
    assert d["shares"] == {}
    assert d["reason"] != "no evaluated channel"
    assert "not finite" in d["reason"]
    manifest = figure.plot_slim16_wtmad2(order, records, tmp_path / "fig.png")
    assert "netD" in manifest["absent"]
    assert "netD" not in manifest["totals"]
    assert not any(b["label"] == "netD" for b in manifest["bars"])


def test_the_converged_filter_holds(tmp_path):
    """netB carries both reactions; the unconverged one must not enter, or
    its 0.30-offset reaction would move the total off 0.10."""
    _order, records = figure.collect_subset_wtmad2(_run(tmp_path))
    assert records["netB"]["total"] == pytest.approx(0.10)
    assert "RC21" not in records["netB"]["shares"]


def test_the_bars_and_totals_reach_the_manifest(tmp_path):
    run = _run(tmp_path)
    order, records = figure.collect_subset_wtmad2(run)
    manifest = figure.plot_slim16_wtmad2(order, records, tmp_path / "fig.png")
    assert manifest["order"] == ["netA", "netB", "netC", "netD"]
    assert manifest["subsets"] == ["ALKBDE10", "RC21"]
    assert manifest["totals"]["netA"] == pytest.approx(0.15)
    assert manifest["totals"]["netB"] == pytest.approx(0.10)
    net_a = [b for b in manifest["bars"] if b["label"] == "netA"]
    assert len(net_a) == 2
    # the tab20 cycle: the two subsets take the first two colors, and a
    # subset appears under both networks with the SAME color
    colors = {b["subset"]: b["color"] for b in manifest["bars"]}
    assert len(set(colors.values())) == len(colors)
    for bar in manifest["bars"]:
        assert bar["share"] == pytest.approx(
            records[bar["label"]]["shares"][bar["subset"]])


def test_build_table_matches_the_hand_values_after_the_rewiring(tmp_path):
    rows = table.build_table(_run(tmp_path))
    by_label = {r["label"]: r for r in rows}
    assert by_label["netA"]["n_converged"] == 6
    assert by_label["netA"]["n_reactions"] == 2
    assert by_label["netA"]["wtmad2_paper_converged"] == pytest.approx(0.15)
    assert by_label["netA"]["wtmad2_paper_all"] == pytest.approx(0.15)
    assert by_label["netB"]["n_converged"] == 3
    assert by_label["netB"]["wtmad2_paper_converged"] == pytest.approx(0.10)
    # rows_all still sees both reactions: the 0.30 offset moves it to 0.24
    assert by_label["netB"]["wtmad2_paper_all"] == pytest.approx(0.24)
    # netC has no channel: its row carries the identity fields only, exactly
    # as before the rewiring (the CSV prints the empty string via .get)
    assert "n_converged" not in by_label["netC"]
    assert by_label["netC"]["kind"] == "trained"


def test_main_writes_the_png_beside_the_run(tmp_path):
    run = _run(tmp_path)
    assert figure.main([str(run)]) == 0
    out = run / "slim16_wtmad2.png"
    assert out.is_file()
    assert out.stat().st_size > 0
