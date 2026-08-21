"""Tests for the guarded dfs_step7 basis-comparison launcher.

Covers the presence guard (the only new logic); figure rendering itself stays covered by
test_make_ablation_arch_figure.py. The launcher is loaded from-file (no package layout),
mirroring test_make_ablation_arch_figure.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


regen = _load("regen_dfs_step7_basis_comparison")


def _add_basis(root: Path, basis: str, *, with_eval: bool = True,
               stamp: str = "run_20260611T000000Z") -> None:
    """Create <root>/dfs_step7/<basis>/runs/<stamp>/checkpoints/spec_0010[/eval_holdout]."""
    spec_dir = root / "dfs_step7" / basis / "runs" / stamp / "checkpoints" / "spec_0010"
    (spec_dir / "eval_holdout").mkdir(parents=True, exist_ok=True)
    if with_eval:
        (spec_dir / "eval_holdout" / "per_reaction.json").write_text("[]")


def test_missing_when_tzvpd_absent(tmp_path: Path) -> None:
    _add_basis(tmp_path, "svp_grid2")
    assert regen.missing_bases(tmp_path) == ["tzvpd_grid2_df"]


def test_none_missing_when_both_present(tmp_path: Path) -> None:
    _add_basis(tmp_path, "svp_grid2")
    _add_basis(tmp_path, "tzvpd_grid2_df")
    assert regen.missing_bases(tmp_path) == []


def test_both_missing_when_empty(tmp_path: Path) -> None:
    assert regen.missing_bases(tmp_path) == list(regen.BASES)


def test_basis_present_dir_but_no_eval_counts_as_missing(tmp_path: Path) -> None:
    # run dir exists but has no per_reaction.json -> not real coverage
    _add_basis(tmp_path, "svp_grid2")
    _add_basis(tmp_path, "tzvpd_grid2_df", with_eval=False)
    assert regen.missing_bases(tmp_path) == ["tzvpd_grid2_df"]


def test_newest_run_with_eval_is_detected(tmp_path: Path) -> None:
    # an older empty run plus a newer run WITH eval -> present
    _add_basis(tmp_path, "svp_grid2")
    _add_basis(tmp_path, "tzvpd_grid2_df", with_eval=False, stamp="run_20260601T000000Z")
    _add_basis(tmp_path, "tzvpd_grid2_df", with_eval=True, stamp="run_20260615T000000Z")
    assert regen.missing_bases(tmp_path) == []


def test_main_refuses_and_reports_when_missing(tmp_path: Path, capsys) -> None:
    _add_basis(tmp_path, "svp_grid2")  # only svp -> must refuse, must NOT shell out
    rc = regen.main(["--results-root", str(tmp_path), "--outroot", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "NOT generated" in err and "tzvpd_grid2_df" in err


def _mark_sliced(root: Path, basis: str,
                 stamp: str = "run_20260611T000000Z") -> None:
    """Write the pre-eval slice marker into that basis run's only channel."""
    import json
    chan = (root / "dfs_step7" / basis / "runs" / stamp / "checkpoints"
            / "spec_0010" / "eval_holdout")
    chan.mkdir(parents=True, exist_ok=True)
    (chan / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": ["h", "h2", "o", "oh", "n2o", "n2ohts"],
         "n_species": 6, "n_reactions": 1,
         "env_var": "XCQUINOX_HELDOUT_SPECIES_SLICE"}))


def test_basis_has_eval_refuses_a_sliced_channel(tmp_path: Path) -> None:
    """The presence guard decides whether the whole comparison suite is
    rendered; a six-species workflow slice is not held-out coverage of the
    pool, and reading it as such would put slice MAEs on the basis figures."""
    import pytest

    from xcquinox.alec.eval_holdout import SlicedChannelError

    _add_basis(tmp_path, "svp_grid2")
    _mark_sliced(tmp_path, "svp_grid2")
    with pytest.raises(SlicedChannelError) as exc:
        regen.basis_has_eval(tmp_path, "svp_grid2")
    msg = str(exc.value)
    assert "run_20260611T000000Z" in msg
    assert "spec_0010" in msg
    assert "eval_holdout" in msg
    assert "'n2ohts'" in msg
    with pytest.raises(SlicedChannelError):
        regen.missing_bases(tmp_path)
