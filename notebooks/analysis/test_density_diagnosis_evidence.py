"""Held-out readers of ``density_diagnosis_evidence.py``.

The bulk of the script is reporting over pulled artifacts. What is pinned
here is the one property a reader of a held-out channel must have: a channel
evaluated on a workflow-verification species slice is refused rather than
summarized as though it covered the pool. The script is loaded from-file (no
package layout), mirroring test_make_ablation_arch_figure.py.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from xcquinox.alec.eval_holdout import SlicedChannelError

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


dd = _load("density_diagnosis_evidence")

_SLICE = ["h", "h2", "o", "oh", "n2o", "n2ohts"]

def _per_molecule(scale: float):
    return [
        {"molecule": "h2o", "density_eps_l1": 2.0e-3 * scale,
         "density_eps_l1_pbe": 2.5e-3},
        {"molecule": "nh3", "density_eps_l1": 1.5e-3 * scale,
         "density_eps_l1_pbe": 2.0e-3},
    ]


def _per_reaction(scale: float):
    return [
        {"name": "w411_a", "abs_error_nn_kcalmol": 3.0 * scale,
         "abs_error_pbe_kcalmol": 4.0},
    ]


def _make_run(tmp_path: Path, n_specs: int = 2,
              channels=("eval_holdout", "eval_holdout_best",
                        "eval_holdout_val_best")) -> Path:
    """Specs carrying every held-out channel plus their training records.

    Two specs by default: the correlate table's Pearson coefficient needs a
    non-degenerate spread on both axes, so the ratios differ per spec.
    """
    run = tmp_path / "run_20260821T000000Z"
    (run / "checkpoints").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(
        {"specs": [{"index": i,
                    "cell": {"arch": "deep_3x16", "subset_size": i + 1}}
                   for i in range(n_specs)]}))
    for i in range(n_specs):
        sd = run / "checkpoints" / f"spec_{i:04d}"
        sd.mkdir()
        (sd / "train_metadata.json").write_text(json.dumps(
            {"molecules": ["CH", "h", "c"] if i == 0 else ["HO", "h", "o"]}))
        scale = 1.0 + 0.25 * i
        for ch in channels:
            (sd / ch).mkdir()
            (sd / ch / "per_molecule.json").write_text(
                json.dumps(_per_molecule(scale)))
            (sd / ch / "per_reaction.json").write_text(
                json.dumps(_per_reaction(scale)))
    return run


def _mark_sliced(run: Path, channel: str, spec: str = "spec_0000") -> None:
    chan = run / "checkpoints" / spec / channel
    chan.mkdir(parents=True, exist_ok=True)
    (chan / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": list(_SLICE), "n_species": len(_SLICE),
         "n_reactions": 1, "env_var": "XCQUINOX_HELDOUT_SPECIES_SLICE"}))


def test_unsliced_run_is_summarized(tmp_path: Path, capsys) -> None:
    """The guards are a no-op on an unmarked run: both readers still report."""
    run = _make_run(tmp_path)
    dd.heldout_summary(run)
    dd.outcome_correlates(run)
    out = capsys.readouterr().out
    assert "Held-out density" in out
    assert "spec_0000" in out
    assert "Pearson r" in out


def test_outcome_correlates_reports_a_degenerate_correlation(tmp_path: Path,
                                                             capsys) -> None:
    """Pearson's r is undefined when either sample has zero variance -- one
    evaluated spec, or several with identical ratios. The table is still
    worth printing, so the coefficient is reported as undefined rather than
    dividing by zero."""
    run = _make_run(tmp_path, n_specs=1)
    dd.outcome_correlates(run)
    out = capsys.readouterr().out
    assert "spec_0000" in out
    assert "undefined" in out


@pytest.mark.parametrize("channel", ["eval_holdout", "eval_holdout_best",
                                     "eval_holdout_val_best"])
def test_heldout_summary_refuses_a_sliced_channel(tmp_path: Path,
                                                  channel: str) -> None:
    """The summary pools density ratios across all three checkpoints'
    channels, so a slice in ANY of them would enter the medians."""
    run = _make_run(tmp_path)
    _mark_sliced(run, channel)
    with pytest.raises(SlicedChannelError) as exc:
        dd.heldout_summary(run)
    msg = str(exc.value)
    assert "run_20260821T000000Z" in msg
    assert "spec_0000" in msg
    assert channel in msg
    assert "'n2ohts'" in msg


def test_outcome_correlates_refuses_a_sliced_channel(tmp_path: Path) -> None:
    """The correlate table pairs a spec's held-out density ratio with its
    held-out energy ratio; on a sliced channel both are slice quantities."""
    run = _make_run(tmp_path)
    _mark_sliced(run, "eval_holdout")
    with pytest.raises(SlicedChannelError) as exc:
        dd.outcome_correlates(run)
    msg = str(exc.value)
    assert "run_20260821T000000Z" in msg
    assert "spec_0000" in msg
    assert "eval_holdout" in msg
    assert "'n2ohts'" in msg
