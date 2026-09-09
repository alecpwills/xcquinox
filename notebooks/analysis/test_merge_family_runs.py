"""Tests for the merged FAMILY view builder (``merge_family_runs``).

One listed set of runs composes one view directory that the existing figure
suite renders once. The list (``family_runs.yaml``) is the only place a new
run enters, so the tests pin the list's grammar, the composition (renumbering,
the protocol tag, the duplicate and identity refusals, the arch restriction,
the pretrain carry), the coverage guard and the command line.

The module under test does not exist yet: it is imported lazily inside
:func:`_mf` so a missing module FAILS each test rather than erroring
collection. The run fixtures reuse the sibling arm-merge test's loadable grid
configuration (``test_merge_v4_arms._full_config_yaml``), whose ``inputs``
block already carries the three identity keys the merge compares, and the
held-out reaction payload of the figure suite's display fixture
(``test_make_ablation_arch_figure._make_display_run_dir``).
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from test_merge_v4_arms import _full_config_yaml            # noqa: E402

# The two REGISTRY architectures the fixtures use. Registry membership is what
# arms the seed-provenance and fidelity-certificate guards -- a fixture name
# outside the registry carries no policy expectation and would leave both
# guards unexercised. Their shown names are "deep_3x16" (medium) and
# "deep0_3x16" (deep_3x16); the merge and its manifest hold STORED keys.
ARCH_A = "medium"
ARCH_B = "deep_3x16"
ARCH_C = "medium_attn"

VIEW = "v7_family"
DOMAIN = "dfs_step7"


def _mf():
    """The module under test, imported at call time."""
    import merge_family_runs as mf
    return mf


# --------------------------------------------------------------------------- #
# Fixtures: a pulled run, and a family list
# --------------------------------------------------------------------------- #
def _reactions(i):
    """Two held-out reaction rows, as the display fixture writes them."""
    return [
        {"name": "bh76_a", "pool": "bh76",
         "reactants": ["HO", "h"], "products": ["HOh_ts"],
         "reaction_energy_ref_kcalmol": 17.7,
         "de_nn_kcalmol": -91.0 + i, "de_pbe_kcalmol": -91.2 + i,
         "abs_error_nn_kcalmol": 108.7 - i,
         "abs_error_pbe_kcalmol": 108.9 - i},
        {"name": "w411_b", "pool": "w411",
         "reactants": ["HO"], "products": ["h", "o"],
         "reaction_energy_ref_kcalmol": 120.0,
         "de_nn_kcalmol": 118.0 + i, "de_pbe_kcalmol": 119.0 + i,
         "abs_error_nn_kcalmol": 2.0 + i,
         "abs_error_pbe_kcalmol": 1.0 + i},
    ]


def _mk_run(root, category, run_name, cells, *, evaluated=False,
            config=None, certify=True, verdicts=None, domain=DOMAIN):
    """A pulled run at ``<root>/<domain>/<category>/runs/<run_name>``.

    ``cells`` is ``[(arch, subset_size), ...]`` in spec order. Each spec dir
    holds a completion marker and, with ``evaluated``, a held-out reaction
    channel. A PASS fidelity certificate is written under
    ``pretrain/<arch>`` for every architecture the manifest names (the
    verdict is overridable per arch through ``verdicts``), and the resolved
    configuration is the loadable one with ``seed_xc: auto``, so the seed
    policy resolves each arch's own rung baseline and the certificate is what
    decides the run's fate.
    """
    run = root / domain / category / "runs" / run_name
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    specs = []
    for i, (arch, ss) in enumerate(cells):
        d = ck / f"spec_{i:04d}"
        d.mkdir()
        (d / "completion.json").write_text(
            json.dumps({"category": category, "spec": i}))
        if evaluated:
            eh = d / "eval_holdout"
            eh.mkdir()
            (eh / "per_reaction.json").write_text(json.dumps(_reactions(i)))
        specs.append({"index": i, "spec_file": f"spec_{i:04d}.spec",
                      "sha256": f"{i:064d}",
                      "cell": {"arch": arch, "subset_size": ss}})
    (run / "manifest.json").write_text(json.dumps(
        {"model": {"descriptor_coordinates": "dfs", "parent_anchor": False},
         "n_specs": len(specs), "specs": specs}))
    (run / "resolved_config.yaml").write_text(
        config if config is not None else _full_config_yaml("auto"))
    if certify:
        for arch in sorted({a for a, _ss in cells}):
            pd = run / "pretrain" / arch
            pd.mkdir(parents=True, exist_ok=True)
            (pd / "fidelity_certificate.json").write_text(json.dumps(
                {"verdict": (verdicts or {}).get(arch, "PASS"), "arch": arch,
                 "summary": {"max_atom_mHa": 0.31, "max_dAE_kcalmol": 0.62,
                             "n_systems": 40, "failure_reasons": []}}))
    return run


def _family(tmp_path, text, name="family_runs.yaml"):
    """Write a family list and return its path."""
    p = tmp_path / name
    p.write_text(text)
    return p


def _view_dir(root, stamp, view=VIEW, domain=DOMAIN):
    """The view directory the merge composes: named by the newest source
    run's stamp, under the view's own ``runs/``."""
    return root / domain / view / "runs" / f"run_{stamp}"


def _targets(view):
    """``{link name: resolved target}`` over the view's spec links."""
    return {p.name: p.resolve() for p in (view / "checkpoints").iterdir()}


# --------------------------------------------------------------------------- #
# T1 -- the list's grammar
# --------------------------------------------------------------------------- #
def test_t1_family_list_parses_to_dataclasses_with_defaults(tmp_path):
    """Two entries parse to ``FamilySpec``/``RunEntry`` with the stated
    defaults: every architecture the manifest names, no protocol tag, and the
    run's certified pretrain directories carried."""
    mf = _mf()
    path = _family(tmp_path, f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
  - category: dfs6311_grid3_v7g1_c25
    run: run_20260908T153856Z
    archs: [medium, medium_attn]
    protocol: 25 cycles
    pretrain: false
""")
    spec = mf.load_family_runs(path)
    assert isinstance(spec, mf.FamilySpec)
    assert spec.view == VIEW
    assert spec.domain == DOMAIN
    assert len(spec.entries) == 2
    first, second = spec.entries
    assert isinstance(first, mf.RunEntry)
    assert first.category == "dfs6311_grid3_v7g1_size"
    assert first.run == "latest"
    assert first.archs is None
    assert first.protocol is None
    assert first.pretrain is True
    assert second.category == "dfs6311_grid3_v7g1_c25"
    assert second.run == "run_20260908T153856Z"
    assert list(second.archs) == ["medium", "medium_attn"]
    assert second.protocol == "25 cycles"
    assert second.pretrain is False


@pytest.mark.parametrize("bad,needle", [
    # a missing category: nothing to resolve, so the entry has no meaning
    ("  - run: latest\n", "category"),
    # a run that is neither `latest` nor a run_* name
    ("  - category: cat_a\n    run: newest\n", "newest"),
    # a non-string protocol: the tag is appended to a shown name
    ("  - category: cat_a\n    run: latest\n    protocol: 25\n", "protocol"),
    # an unknown key: a typo must not be read as a default
    ("  - category: cat_a\n    run: latest\n    pretrian: false\n",
     "pretrian"),
])
def test_t1_malformed_entries_are_refused_naming_the_entry(tmp_path, bad,
                                                           needle):
    """Each malformed entry raises ValueError, and the message names both the
    defect and the entry it is in."""
    mf = _mf()
    path = _family(tmp_path,
                   f"view: {VIEW}\ndomain: {DOMAIN}\nruns:\n{bad}")
    with pytest.raises(ValueError) as exc:
        mf.load_family_runs(path)
    msg = str(exc.value)
    assert needle in msg
    # the entry itself is named: by its category where it has one, and by
    # the missing key where it has not
    assert ("cat_a" in msg) or ("category" in msg)


# --------------------------------------------------------------------------- #
# T2 -- composition
# --------------------------------------------------------------------------- #
_T2_YAML = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g2a_families_core
    run: latest
  - category: dfs6311_grid3_v7g1_size
    run: latest
"""


def test_t2_two_runs_compose_one_renumbered_view(tmp_path):
    """Five spec links in LIST order (not sorted-category order), each
    resolving to its source spec dir; the merged manifest carries the source
    cells with their category, run and source index; the view directory is
    named by the NEWEST source run's stamp."""
    mf = _mf()
    root = tmp_path / "runs"
    core = _mk_run(root, "dfs6311_grid3_v7g2a_families_core",
                   "run_20260902T145247Z",
                   [(ARCH_B, 1), (ARCH_B, 2), (ARCH_B, 3)])
    size = _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260908T153856Z",
                   [(ARCH_A, 1), (ARCH_A, 2)])
    spec = mf.load_family_runs(_family(tmp_path, _T2_YAML))
    report = mf.build_view(spec, root)

    assert report["dfs6311_grid3_v7g2a_families_core"] == (
        "run_20260902T145247Z", 3, 0)
    assert report["dfs6311_grid3_v7g1_size"] == ("run_20260908T153856Z", 2, 0)

    view = _view_dir(root, "20260908T153856Z")
    assert view.is_dir(), sorted(
        p.name for p in (root / DOMAIN / VIEW / "runs").iterdir())
    links = _targets(view)
    assert sorted(links) == [f"spec_{i:04d}" for i in range(5)]
    # list order: the core entry's three specs first, then the size entry's
    assert links["spec_0000"] == (core / "checkpoints/spec_0000").resolve()
    assert links["spec_0002"] == (core / "checkpoints/spec_0002").resolve()
    assert links["spec_0003"] == (size / "checkpoints/spec_0000").resolve()
    assert links["spec_0004"] == (size / "checkpoints/spec_0001").resolve()
    assert (view / "checkpoints/spec_0003/completion.json").read_text() == \
        json.dumps({"category": "dfs6311_grid3_v7g1_size", "spec": 0})

    m = json.loads((view / "manifest.json").read_text())
    assert m["n_specs"] == 5
    assert m["merged_from"] == ["dfs6311_grid3_v7g2a_families_core",
                                "dfs6311_grid3_v7g1_size"]
    assert [s["index"] for s in m["specs"]] == list(range(5))
    assert [s["cell"]["arch"] for s in m["specs"]] == \
        [ARCH_B, ARCH_B, ARCH_B, ARCH_A, ARCH_A]
    assert [s["cell"]["subset_size"] for s in m["specs"]] == [1, 2, 3, 1, 2]
    assert m["specs"][3]["category"] == "dfs6311_grid3_v7g1_size"
    assert m["specs"][3]["run"] == "run_20260908T153856Z"
    assert m["specs"][3]["source_index"] == 0
    assert m["specs"][2]["source_index"] == 2
    assert m["specs"][0]["fidelity_status"] == "PASS"
    assert m["specs"][0]["spec_file"] == "spec_0000.spec"
    assert m["specs"][0]["sha256"] == f"{0:064d}"


# --------------------------------------------------------------------------- #
# T3 -- stable renumbering
# --------------------------------------------------------------------------- #
_T3_TWO = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
  - category: dfs6311_grid3_v7g2a_families_core
    run: latest
"""

_T3_THREE = _T3_TWO + """  - category: dfs6311_grid3_v7g1_c25
    run: latest
"""


def test_t3_renumbering_is_stable_and_appends(tmp_path):
    """Rebuilding yields byte-identical manifest and identical link targets;
    an entry added at the END of the list appends indices and moves none of
    the earlier ones -- so the indices follow the LIST, not the sorted
    category names (the added category sorts first of the three)."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1), (ARCH_A, 2)])
    _mk_run(root, "dfs6311_grid3_v7g2a_families_core",
            "run_20260901T000000Z", [(ARCH_B, 1)])
    assert sorted(["dfs6311_grid3_v7g1_size",
                   "dfs6311_grid3_v7g2a_families_core",
                   "dfs6311_grid3_v7g1_c25"])[0] == "dfs6311_grid3_v7g1_c25"

    two = mf.load_family_runs(_family(tmp_path, _T3_TWO, "two.yaml"))
    mf.build_view(two, root)
    view = _view_dir(root, "20260902T000000Z")
    first_bytes = (view / "manifest.json").read_bytes()
    first_links = _targets(view)
    assert sorted(first_links) == ["spec_0000", "spec_0001", "spec_0002"]

    mf.build_view(two, root)
    assert (view / "manifest.json").read_bytes() == first_bytes
    assert _targets(view) == first_links

    # a third run lands at the end of the list, with an older stamp so the
    # view keeps its name
    _mk_run(root, "dfs6311_grid3_v7g1_c25", "run_20260831T000000Z",
            [(ARCH_C, 1), (ARCH_C, 2)])
    three = mf.load_family_runs(_family(tmp_path, _T3_THREE, "three.yaml"))
    mf.build_view(three, root)
    grown = _targets(view)
    assert sorted(grown) == [f"spec_{i:04d}" for i in range(5)]
    for name, target in first_links.items():
        assert grown[name] == target, name
    m = json.loads((view / "manifest.json").read_text())
    assert [s["category"] for s in m["specs"]] == [
        "dfs6311_grid3_v7g1_size", "dfs6311_grid3_v7g1_size",
        "dfs6311_grid3_v7g2a_families_core",
        "dfs6311_grid3_v7g1_c25", "dfs6311_grid3_v7g1_c25"]


# --------------------------------------------------------------------------- #
# T4 -- the protocol tag, through the display boundary onto a figure
# --------------------------------------------------------------------------- #
_T4_YAML = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
  - category: dfs6311_grid3_v7g1_c25
    run: latest
    protocol: 25 cycles
    pretrain: false
"""


def test_t4_protocol_tag_reaches_the_figure_axis(tmp_path, monkeypatch):
    """A tagged entry repeating the first run's (arch, subset_size) cells is
    ADMITTED -- the tag is part of the cell key -- its cells carry
    ``protocol`` in the merged manifest, the display boundary appends the tag
    to the shown name, and the per-arch figure draws the tagged architecture
    beside its untagged one."""
    mf = _mf()
    import make_ablation_arch_figure as fig
    import matplotlib.figure as mfig

    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1), (ARCH_A, 3)], evaluated=True)
    _mk_run(root, "dfs6311_grid3_v7g1_c25", "run_20260908T000000Z",
            [(ARCH_A, 1), (ARCH_A, 3)], evaluated=True)
    spec = mf.load_family_runs(_family(tmp_path, _T4_YAML))
    report = mf.build_view(spec, root)
    assert report["dfs6311_grid3_v7g1_c25"] == ("run_20260908T000000Z", 2, 2)
    view = _view_dir(root, "20260908T000000Z")

    m = json.loads((view / "manifest.json").read_text())
    assert [s["cell"].get("protocol") for s in m["specs"]] == \
        [None, None, "25 cycles", "25 cycles"]
    assert "protocol" not in m["specs"][0]["cell"]

    cells = fig.ccp._read_manifest_cells(view)
    assert {c["arch"] for c in cells.values()} == \
        {"deep_3x16", "deep_3x16 [25 cycles]"}
    assert {c["arch_stored"] for c in cells.values()} == {ARCH_A}

    rows = fig.collect_holdout_reaction_rows(view)
    assert len(rows) == 8
    assert {r["arch"] for r in rows} == \
        {"deep_3x16", "deep_3x16 [25 cycles]"}

    seen = []
    real = mfig.Figure.savefig

    def _cap(self, *a, **k):
        for ax in self.axes:
            seen.extend(t.get_text() for t in ax.get_xticklabels())
        return real(self, *a, **k)

    monkeypatch.setattr(mfig.Figure, "savefig", _cap)
    fig.plot_mae_by_arch(rows, [], tmp_path / "mae_by_arch.png", "run_x")
    heads = {t.split("\n")[0] for t in seen}
    assert "deep_3x16" in heads, sorted(heads)
    assert "deep_3x16 [25 cycles]" in heads, sorted(heads)
    assert "25 cycles" in fig.key_line(fig._archs_present(rows))


# --------------------------------------------------------------------------- #
# T5 -- the duplicate-cell refusal
# --------------------------------------------------------------------------- #
_T5_YAML = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
  - category: dfs6311_grid3_v7g1_resubmit
    run: latest
"""


def test_t5_duplicate_untagged_cell_is_refused(tmp_path):
    """The same (arch, subset_size) from two UNTAGGED entries is a
    double-count, not a merge: the build is refused and both categories are
    named."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1), (ARCH_A, 2)])
    _mk_run(root, "dfs6311_grid3_v7g1_resubmit", "run_20260903T000000Z",
            [(ARCH_A, 2)])
    spec = mf.load_family_runs(_family(tmp_path, _T5_YAML))
    with pytest.raises(SystemExit) as exc:
        mf.build_view(spec, root)
    msg = str(exc.value)
    assert "dfs6311_grid3_v7g1_size" in msg
    assert "dfs6311_grid3_v7g1_resubmit" in msg
    assert ARCH_A in msg


# --------------------------------------------------------------------------- #
# T6 -- the production identity
# --------------------------------------------------------------------------- #
_T6_UNTAGGED = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
  - category: dfs6311_grid3_v7g1_other
    run: latest
"""

_T6_TAGGED = _T6_UNTAGGED + "    protocol: dpyscf parity\n"


def _other_basis_config():
    """The loadable configuration with ONE identity key changed."""
    text = _full_config_yaml("auto")
    assert "basis: 6-311++G(3df,2pd)" in text
    return text.replace("basis: 6-311++G(3df,2pd)", "basis: other")


def test_t6_differing_identity_is_refused_without_a_tag(tmp_path):
    """A later entry whose production identity differs from the view's is
    refused when it carries no protocol tag: its cells would be drawn under
    the first run's basis label and SCAN caches."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1)])
    _mk_run(root, "dfs6311_grid3_v7g1_other", "run_20260903T000000Z",
            [(ARCH_A, 2)], config=_other_basis_config())
    spec = mf.load_family_runs(_family(tmp_path, _T6_UNTAGGED))
    with pytest.raises(SystemExit) as exc:
        mf.build_view(spec, root)
    msg = str(exc.value)
    assert "basis" in msg
    assert "other" in msg
    assert "dfs6311_grid3_v7g1_other" in msg


def test_t6_differing_identity_is_refused_with_a_tag_too(tmp_path):
    """A protocol tag names a training protocol, not a basis or a grid: the
    view keeps the first entry's configuration (its basis label, its SCAN
    references), so cells measured at another identity cannot be drawn under
    it even when tagged. Refused with the same message as the untagged case,
    and no view is left on disk."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1)])
    _mk_run(root, "dfs6311_grid3_v7g1_other", "run_20260903T000000Z",
            [(ARCH_A, 2)], config=_other_basis_config())
    spec = mf.load_family_runs(_family(tmp_path, _T6_TAGGED))
    with pytest.raises(SystemExit) as exc:
        mf.build_view(spec, root)
    msg = str(exc.value)
    assert "basis" in msg and "other" in msg
    assert "dfs6311_grid3_v7g1_other" in msg
    assert not (root / DOMAIN / VIEW).exists() or \
        not list((root / DOMAIN / VIEW / "runs").glob("run_*"))


# --------------------------------------------------------------------------- #
# T7 -- an absent run
# --------------------------------------------------------------------------- #
_T7_YAML = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
  - category: dfs6311_grid3_v7g2_families_mgga
    run: latest
"""


def test_t7_absent_listed_run_refuses_the_build(tmp_path):
    """A listed category with no local run is REFUSED, not skipped: a view
    built without it would be read as the whole family, and the message names
    the missing category."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1)])
    spec = mf.load_family_runs(_family(tmp_path, _T7_YAML))
    with pytest.raises((SystemExit, FileNotFoundError, RuntimeError,
                        ValueError)) as exc:
        mf.build_view(spec, root)
    assert "dfs6311_grid3_v7g2_families_mgga" in str(exc.value)


# --------------------------------------------------------------------------- #
# T8 -- the architecture restriction
# --------------------------------------------------------------------------- #
_T8_YAML = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
    archs: [{ARCH_A}]
"""


def test_t8_arch_restriction_limits_specs_and_certificates(tmp_path):
    """``archs`` restricts the manifest entries AND the spec directories: the
    other architecture's specs are not linked and its certificate is not
    carried into the view."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1), (ARCH_B, 1), (ARCH_A, 2), (ARCH_B, 2)])
    spec = mf.load_family_runs(_family(tmp_path, _T8_YAML))
    report = mf.build_view(spec, root)
    assert report["dfs6311_grid3_v7g1_size"][1] == 2
    view = _view_dir(root, "20260902T000000Z")
    assert sorted(_targets(view)) == ["spec_0000", "spec_0001"]
    m = json.loads((view / "manifest.json").read_text())
    assert {s["cell"]["arch"] for s in m["specs"]} == {ARCH_A}
    assert [s["source_index"] for s in m["specs"]] == [0, 2]
    assert sorted(p.name for p in (view / "pretrain").iterdir()) == [ARCH_A]


# --------------------------------------------------------------------------- #
# T9 -- pretrain: false
# --------------------------------------------------------------------------- #
_T9_YAML = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
  - category: dfs6311_grid3_v7g1_c25
    run: latest
    protocol: 25 cycles
    pretrain: false
"""


def test_t9_pretrain_false_carries_no_certificate_but_still_gates(tmp_path):
    """``pretrain: false`` keeps the entry's certificate OUT of the view: the
    view's pretrain slot stays the first run's, and the substitution is
    recorded (the figures read the certificate by the slot). The certificate
    guard still runs on the entry's OWN architectures -- a failed one refuses
    the build -- and an architecture with no carried slot is refused, since
    the figures would read it as uncertified."""
    mf = _mf()
    root = tmp_path / "runs"
    size = _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
                   [(ARCH_A, 1)])
    _mk_run(root, "dfs6311_grid3_v7g1_c25", "run_20260908T000000Z",
            [(ARCH_A, 1)])
    spec = mf.load_family_runs(_family(tmp_path, _T9_YAML))
    mf.build_view(spec, root)
    view = _view_dir(root, "20260908T000000Z")
    assert sorted(p.name for p in (view / "pretrain").iterdir()) == [ARCH_A]
    assert (view / "pretrain" / ARCH_A).resolve() == \
        (size / "pretrain" / ARCH_A).resolve()
    txt = (view / "MERGED_RUNS.txt").read_text()
    assert "slot carried from dfs6311_grid3_v7g1_size" in txt
    m = json.loads((view / "manifest.json").read_text())
    # the status map holds statuses only; the slot substitution is its own record
    assert m["fidelity"]["by_run"]["dfs6311_grid3_v7g1_c25"][ARCH_A] == "PASS"
    assert m["fidelity"]["by_run"]["dfs6311_grid3_v7g1_size"][ARCH_A] == "PASS"
    assert m["fidelity"]["slot_from"] == {
        "dfs6311_grid3_v7g1_c25": {ARCH_A: "dfs6311_grid3_v7g1_size"}}
    assert m["fidelity"]["n_waived"] == 0

    # the same list over a run whose own architecture FAILED its certificate
    root2 = tmp_path / "runs2"
    _mk_run(root2, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1)])
    _mk_run(root2, "dfs6311_grid3_v7g1_c25", "run_20260908T000000Z",
            [(ARCH_A, 1)], verdicts={ARCH_A: "FAIL"})
    with pytest.raises(SystemExit) as exc:
        mf.build_view(spec, root2)
    msg = str(exc.value)
    assert ARCH_A in msg
    assert "dfs6311_grid3_v7g1_c25" in msg

    # an architecture with no carried slot cannot ride on pretrain: false
    root3 = tmp_path / "runs3"
    _mk_run(root3, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1)])
    _mk_run(root3, "dfs6311_grid3_v7g1_c25", "run_20260908T000000Z",
            [(ARCH_B, 1)])
    with pytest.raises(SystemExit) as exc:
        mf.build_view(spec, root3)
    assert ARCH_B in str(exc.value) and "slot" in str(exc.value)


# --------------------------------------------------------------------------- #
# T10 -- the coverage guard
# --------------------------------------------------------------------------- #
_T10_YAML = _T4_YAML


def test_t10_coverage_guard_names_a_missing_cell(tmp_path):
    """``evaluated_cells`` is the union of the listed runs' evaluated cells,
    keyed per (category, arch, protocol, subset_size); a spec link removed
    from the view makes ``assert_view_complete`` raise NAMING the cell -- a
    guard that merely counted links could not name it."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1), (ARCH_A, 3)], evaluated=True)
    _mk_run(root, "dfs6311_grid3_v7g1_c25", "run_20260908T000000Z",
            [(ARCH_A, 1), (ARCH_A, 3)], evaluated=True)
    spec = mf.load_family_runs(_family(tmp_path, _T10_YAML))
    mf.build_view(spec, root)
    view = _view_dir(root, "20260908T000000Z")

    assert set(mf.evaluated_cells(view)) == {
        ("dfs6311_grid3_v7g1_size", ARCH_A, None, 1),
        ("dfs6311_grid3_v7g1_size", ARCH_A, None, 3),
        ("dfs6311_grid3_v7g1_c25", ARCH_A, "25 cycles", 1),
        ("dfs6311_grid3_v7g1_c25", ARCH_A, "25 cycles", 3),
    }
    mf.assert_view_complete(spec, root, view)

    (view / "checkpoints" / "spec_0003").unlink()
    with pytest.raises((SystemExit, AssertionError, RuntimeError,
                        ValueError)) as exc:
        mf.assert_view_complete(spec, root, view)
    msg = str(exc.value)
    assert "dfs6311_grid3_v7g1_c25" in msg
    assert ARCH_A in msg
    assert "25 cycles" in msg
    assert "3" in msg


# --------------------------------------------------------------------------- #
# T11 -- the command line
# --------------------------------------------------------------------------- #
_T11_YAML = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
  - category: dfs6311_grid3_v7g1_c25
    run: latest
    protocol: 25 cycles
    pretrain: false
"""


def test_t11_cli_prints_an_inventory_line_per_entry(tmp_path, capsys):
    """``main`` prints one inventory line per listed entry (category, run,
    spec count, evaluated count, protocol) plus the view path, and returns 0
    when the view carries an evaluated cell."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1), (ARCH_A, 3)], evaluated=True)
    _mk_run(root, "dfs6311_grid3_v7g1_c25", "run_20260908T000000Z",
            [(ARCH_A, 1)], evaluated=True)
    path = _family(tmp_path, _T11_YAML)
    rc = mf.main(["--family", str(path), "--results-root", str(root)])
    assert rc == 0
    out = capsys.readouterr().out
    size_lines = [ln for ln in out.splitlines()
                  if "dfs6311_grid3_v7g1_size" in ln]
    c25_lines = [ln for ln in out.splitlines()
                 if "dfs6311_grid3_v7g1_c25" in ln]
    assert len(size_lines) == 1, out
    assert len(c25_lines) == 1, out
    assert "run_20260902T000000Z" in size_lines[0]
    # the two counts, as standalone tokens: the run names carry no bare
    # "2"/"1" token, so a line without the inventory numbers fails here
    assert [t for t in size_lines[0].split() if t == "2"], size_lines[0]
    assert [t for t in c25_lines[0].split() if t == "1"], c25_lines[0]
    assert "25 cycles" in c25_lines[0]
    assert str(_view_dir(root, "20260908T000000Z")) in out


def test_t11_cli_exits_1_without_an_evaluated_cell(tmp_path):
    """A view with no evaluated cell renders nothing: the command reports it
    by exiting non-zero rather than leaving an empty figure set."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1)])
    _mk_run(root, "dfs6311_grid3_v7g1_c25", "run_20260908T000000Z",
            [(ARCH_A, 1)])
    path = _family(tmp_path, _T11_YAML)
    assert mf.main(["--family", str(path), "--results-root", str(root)]) == 1


# --------------------------------------------------------------------------- #
# T12 -- the tracked list
# --------------------------------------------------------------------------- #
def test_t12_the_real_family_list_names_the_five_v7_categories():
    """The tracked list is the v7 family: G1, G2a, the meta-GGA run, and the
    two arms with their tags. The arms set ``pretrain: false`` because they
    reuse G1's certificates, and every entry takes the newest local run."""
    mf = _mf()
    spec = mf.load_family_runs(os.path.join(_HERE, "family_runs.yaml"))
    assert spec.view == "v7_family"
    assert spec.domain == "dfs_step7"
    assert [e.category for e in spec.entries] == [
        "dfs6311_grid3_v7g1_size",
        "dfs6311_grid3_v7g2a_families_core",
        "dfs6311_grid3_v7g2_families_mgga",
        "dfs6311_grid3_v7g1_c25",
        "dfs6311_grid3_v7g1_dfsparity",
    ]
    assert [e.protocol for e in spec.entries] == [
        None, None, None, "25 cycles", "dpyscf parity"]
    assert [e.pretrain for e in spec.entries] == [
        True, True, True, False, False]
    assert [e.run for e in spec.entries] == ["latest"] * 5



# --------------------------------------------------------------------------- #
# Review findings (2026-09-09): stray checkpoint entries, in-run duplicates,
# the ledger in the identity, the evaluated channels
# --------------------------------------------------------------------------- #
_T13_YAML = f"""
view: {VIEW}
domain: {DOMAIN}
runs:
  - category: dfs6311_grid3_v7g1_size
    run: latest
"""


def test_a_stray_entry_under_checkpoints_is_refused_by_name(tmp_path):
    """A file or a partial directory named like a spec under ``checkpoints/``
    is neither labelled nor linkable: refused with the path named, never a
    bare ValueError from the index parse."""
    mf = _mf()
    root = tmp_path / "runs"
    run = _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
                  [(ARCH_A, 1)])
    (run / "checkpoints" / "spec_0001.partial").mkdir()
    spec = mf.load_family_runs(_family(tmp_path, _T13_YAML))
    with pytest.raises(SystemExit) as exc:
        mf.build_view(spec, root)
    assert "spec_0001.partial" in str(exc.value)
    assert "REFUSING" in str(exc.value)


def test_a_duplicate_cell_within_one_run_is_refused(tmp_path):
    """Two spec directories of ONE run with the same (arch, subset_size)
    would both be reduced into one cell: refused, the run named twice."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1), (ARCH_A, 1)])
    spec = mf.load_family_runs(_family(tmp_path, _T13_YAML))
    with pytest.raises(SystemExit) as exc:
        mf.build_view(spec, root)
    assert "twice from dfs6311_grid3_v7g1_size" in str(exc.value)


def test_the_subset_ledger_is_part_of_the_identity(tmp_path):
    """The view keeps the first run's configuration, whose ledger captions
    every subset's training content; a run on another ledger is refused."""
    mf = _mf()
    root = tmp_path / "runs"
    _mk_run(root, "dfs6311_grid3_v7g1_size", "run_20260902T000000Z",
            [(ARCH_A, 1)])
    other = _full_config_yaml("auto").replace(
        "subset_ledger_path:", "subset_ledger_path_was:") + \
        "\nsubset_ledger_path: /elsewhere/other_ledger.json\n"
    assert "subset_ledger_path:" in other
    _mk_run(root, "dfs6311_grid3_v7g2a_families_core", "run_20260903T000000Z",
            [(ARCH_B, 1)], config=other)
    spec = mf.load_family_runs(_family(tmp_path, _T2_YAML))
    with pytest.raises(SystemExit) as exc:
        mf.build_view(spec, root)
    assert "subset_ledger_path" in str(exc.value)


def test_only_the_rendered_channels_count_as_evaluated(tmp_path):
    """A spec holding the cold-start channel alone is not an evaluated cell
    of the suite's sets; the inventory and the exit code do not count it."""
    mf = _mf()
    d = tmp_path / "spec_0000"
    (d / "eval_holdout_coldstart").mkdir(parents=True)
    (d / "eval_holdout_coldstart" / "per_reaction.json").write_text("[]")
    assert mf._evaluated(d) is False
    (d / "eval_holdout_val_best").mkdir()
    (d / "eval_holdout_val_best" / "per_reaction.json").write_text("[]")
    assert mf._evaluated(d) is True
