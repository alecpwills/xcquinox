"""Parity-probe verdicts (verify_holdout_parity.compare_spec)."""
import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


vp = _load("verify_holdout_parity")

_R = {"name": "r1", "de_nn_kcalmol": 1.0, "de_pbe_kcalmol": 2.0,
      "abs_error_nn_kcalmol": 0.5, "abs_error_pbe_kcalmol": 0.7}


def test_parity_verdict():
    rep = vp.compare_spec([dict(_R)], [dict(_R)])
    assert rep["verdict"] == "parity"
    assert rep["max_abs_delta"] == 0.0


def test_stale_rule_verdict_on_set_difference():
    rep = vp.compare_spec([dict(_R)],
                          [dict(_R), dict(_R, name="r2")])
    assert rep["verdict"] == "stale-rule"
    assert rep["only_recon"] == ["r2"]


def test_value_mismatch_verdict():
    rep = vp.compare_spec([dict(_R)],
                          [dict(_R, de_nn_kcalmol=1.5)], tol=1e-9)
    assert rep["verdict"] == "value-mismatch"
    assert rep["max_abs_delta"] > 0.4


def test_no_cluster_file_verdict():
    rep = vp.compare_spec(None, [dict(_R)])
    assert rep["verdict"] == "no-cluster-file"


def test_duplicate_name_rows_compare_by_best_match():
    # The pool's duplicate-name entries produce two identical rows; each
    # reconstruction row matches its best cluster twin, not the worst.
    a = [dict(_R), dict(_R)]
    rep = vp.compare_spec(a, a)
    assert rep["verdict"] == "parity"


def test_one_side_finite_is_a_mismatch_not_parity():
    a = [dict(_R)]
    b = [dict(_R, de_nn_kcalmol=None)]
    rep = vp.compare_spec(a, b)
    assert rep["verdict"] == "value-mismatch"


class _StubFig:
    """Stands in for the figure module's reconstruction loader, so the parity
    probe's OWN read of the cluster ``per_reaction.json`` is what is tested."""

    def __init__(self, rows):
        self._rows = rows

    def collect_holdout_reaction_rows(self, run_dir, eval_subdir="eval_holdout"):
        return list(self._rows)


def test_verify_run_refuses_a_sliced_channel(tmp_path):
    """The probe compares a channel's cluster file against a reconstruction;
    on a sliced channel both legs describe a handful of species, and a
    "parity" verdict would read as evidence about the pool."""
    import json

    from xcquinox.alec.eval_holdout import SlicedChannelError

    run = tmp_path / "run_20260821T000000Z"
    chan = run / "checkpoints" / "spec_0000" / "eval_holdout"
    chan.mkdir(parents=True)
    (chan / "per_reaction.json").write_text(json.dumps([dict(_R)]))
    (chan / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": ["h", "h2", "o", "oh", "n2o", "n2ohts"],
         "n_species": 6, "n_reactions": 1,
         "env_var": "XCQUINOX_HELDOUT_SPECIES_SLICE"}))
    stub = _StubFig([dict(_R, idx=0)])
    with pytest.raises(SlicedChannelError) as exc:
        vp.verify_run(run, channels=["eval_holdout"], _fig=stub)
    msg = str(exc.value)
    assert "run_20260821T000000Z" in msg
    assert "spec_0000" in msg
    assert "eval_holdout" in msg
    assert "'n2ohts'" in msg
