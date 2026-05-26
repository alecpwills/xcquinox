"""Unit test for the pure accessors in notebooks/analysis/make_multimode_figure.py
(the multimode figure builder). Loaded by file path; touches no matplotlib render."""
import importlib.util
import os

import pytest

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "notebooks", "analysis",
    "make_multimode_figure.py",
)


@pytest.fixture(scope="module")
def mod():
    p = os.path.abspath(_PATH)
    if not os.path.isfile(p):
        pytest.skip(f"figure builder not found at {p}")
    spec = importlib.util.spec_from_file_location("make_multimode_figure", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_short_level(mod):
    assert mod.short_level("+LO(x)") == "+LO"
    assert mod.short_level("+LO+UEG+NNc(c)") == "+LO+UEG+NNc"
    assert mod.short_level("unconstrained") == "unconstrained"


def test_cell_random_accessor(mod):
    results = {"cells": {"3step": {"+LO(x)": {"random": {
        "pbe_dev": {"mean": 12.3, "worst": 45.6, "std": 7.8}}}}}}
    r = mod.cell_random(results, "3step", "+LO(x)", "pbe_dev")
    assert r["mean"] == 12.3 and r["worst"] == 45.6 and r["std"] == 7.8
