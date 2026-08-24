"""Task 9 (density-fitting plan): pretrain-data generation must honor the
configured basis/grid_level, support density fitting on the per-atom SCF, and
carry a basis manifest so a basis change forces regeneration rather than silent
reuse of stale def2-svp data."""
import json
import os

import numpy as np
import pytest

from xcquinox.alec import pretrain_data_gen as pdg


# --- fast: manifest + is-current logic (no SCF) ----------------------------

def _touch_npz(path):
    np.savez(path, x=np.zeros(3))


def test_is_current_false_when_file_missing(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=1) is False


def test_is_current_false_when_no_manifest(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)                                   # legacy file, no manifest
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=1) is False


def test_is_current_true_on_matching_manifest(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)
    pdg._write_pretrain_manifest(p, basis="def2-svp", grid_level=1,
                                 density_fit=False)
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=1) is True


def test_is_current_false_on_basis_mismatch(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)
    pdg._write_pretrain_manifest(p, basis="def2-svp", grid_level=1,
                                 density_fit=False)
    # a def2-tzvp run must NOT reuse def2-svp data
    assert pdg.pretrain_data_is_current(p, basis="def2-tzvp", grid_level=1) is False


def test_is_current_false_on_grid_mismatch(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)
    pdg._write_pretrain_manifest(p, basis="def2-svp", grid_level=1,
                                 density_fit=False)
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=2) is False


def test_manifest_round_trips_density_fit_flag(tmp_path):
    p = os.path.join(tmp_path, "pretrain_data.npz")
    pdg._write_pretrain_manifest(p, basis="def2-tzvp", grid_level=2,
                                 density_fit=True,
                                 auxbasis="def2-universal-jkfit")
    with open(pdg._pretrain_manifest_path(p)) as f:
        meta = json.load(f)
    assert meta == {"basis": "def2-tzvp", "grid_level": 2, "density_fit": True,
                    "alpha_definition": str(pdg._ALPHA_DEFINITION),
                    "auxbasis": "def2-universal-jkfit",
                    "atoms": [[s, sp] for s, sp in pdg.DEFAULT_PRETRAIN_ATOMS],
                    # The pretraining-protocol identity: the system list (None
                    # when the writer was handed atoms only), the parent
                    # functional, the exchange footing, the orientation lock
                    # the parent density was computed at, and the precision.
                    # A degenerate atom's rows below grid level 3, or with
                    # the lock off, are one arbitrary member of its manifold;
                    # the writer records whether that was permitted.
                    "allow_irreproducible_degenerate": False,
                    "systems": None, "reference_xc": "pbe",
                    "exchange_footing": "total",
                    "orientation_lock_strength":
                        pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH,
                    "x64": True,
                    # The (s, alpha) parameter mesh record (2026-08-10): its
                    # weight share is a deliberate choice the manifest states.
                    "mesh": {"rs": list(pdg.MESH_RS), "s": list(pdg.MESH_S),
                             "alpha": list(pdg.MESH_ALPHA),
                             "weight_fraction": pdg.MESH_WEIGHT_FRACTION}}


def test_manifest_auxbasis_defaults_none(tmp_path):
    """Default-off (full-ERI) manifest records auxbasis=None."""
    p = os.path.join(tmp_path, "pretrain_data.npz")
    pdg._write_pretrain_manifest(p, basis="def2-svp", grid_level=1,
                                 density_fit=False)
    with open(pdg._pretrain_manifest_path(p)) as f:
        meta = json.load(f)
    assert meta["auxbasis"] is None


def test_legacy_manifest_without_auxbasis_stays_current(tmp_path):
    """A legacy manifest lacking the auxbasis key must NOT trigger a spurious
    regen on the full-ERI path (auxbasis reads as None -> matches None)."""
    p = os.path.join(tmp_path, "pretrain_data.npz")
    _touch_npz(p)
    # Old-format manifest: basis + grid_level + density_fit, NO auxbasis key.
    # Such a file was also built without the orientation lock, so its identity
    # is asked for at lock 0.0 (the production default would regenerate it).
    with open(pdg._pretrain_manifest_path(p), "w") as f:
        json.dump({"basis": "def2-svp", "grid_level": 1, "density_fit": False,
                   "alpha_definition": str(pdg._ALPHA_DEFINITION)}, f)
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1,
        orientation_lock_strength=0.0) is True
    # A DF run (effective auxbasis set) correctly sees it as stale.
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1, auxbasis="def2-svp-jkfit",
        orientation_lock_strength=0.0) is False


# --- ensure-driver: skip-if-current, regen on basis change -----------------

def test_ensure_regenerates_only_when_stale(tmp_path, monkeypatch):
    calls = []

    # ``atoms`` has a default because ``ensure_pretrain_data`` hands the
    # generator the RESOLVED system list and nothing else; a fake that still
    # required the atom list would pin a duplicate argument rather than the
    # basis behaviour this test is about.
    def fake_generate(out_dir, *, basis, grid_level, polarized,
                      descriptors, density_fit, atoms=None, auxbasis=None,
                      cusp_log_transform=True, **kwargs):
        calls.append(basis)
        path = os.path.join(out_dir, "pretrain_data_polarized.npz"
                            if polarized else "pretrain_data.npz")
        _touch_npz(path)
        pdg._write_pretrain_manifest(
            path, basis=basis, grid_level=grid_level, density_fit=density_fit,
            auxbasis=pdg._effective_auxbasis(basis, density_fit, auxbasis))
        return path

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", fake_generate)

    # The default set carries O, and this test runs at grid level 1
    # deliberately (it is about the currency check, and the generator is
    # faked), so the irreproducible-degenerate refusal is waived throughout.
    coarse = dict(allow_irreproducible_degenerate=True)
    # first call: file absent -> generates
    pdg.ensure_pretrain_data(str(tmp_path), basis="def2-svp", grid_level=1,
                             polarized=False, **coarse)
    # second call, same basis -> current -> NO regen
    pdg.ensure_pretrain_data(str(tmp_path), basis="def2-svp", grid_level=1,
                             polarized=False, **coarse)
    # third call, new basis -> stale -> regen
    pdg.ensure_pretrain_data(str(tmp_path), basis="def2-tzvp", grid_level=1,
                             polarized=False, **coarse)
    assert calls == ["def2-svp", "def2-tzvp"]


# --- physics: DF on the per-atom SCF, DF-off byte-identical -----------------

def test_atom_columns_density_fit_runs():
    """A density-fitted per-atom SCF produces finite Fx/Fc columns."""
    cols = pdg._atom_columns("He", 0, "def2-svp", 1, polarized=False,
                             descriptors=False, density_fit=True)
    assert np.all(np.isfinite(cols["Fx"])) and np.all(np.isfinite(cols["Fc"]))
    assert cols["rho"].size > 0


def test_atom_columns_density_fit_off_matches_pre_df_path():
    """density_fit defaulting to off takes the same (non-DF) SCF code path and
    reproduces the pre-DF per-atom columns to SCF tolerance (the only residual
    is machine-epsilon BLAS nondeterminism, present between any two identical
    runs: not a code-path difference). Regression guard for the new kwarg."""
    base = pdg._atom_columns("He", 0, "def2-svp", 1, polarized=False,
                             descriptors=False)
    off = pdg._atom_columns("He", 0, "def2-svp", 1, polarized=False,
                            descriptors=False, density_fit=False)
    for k in base:
        if k in ("Fx_scan", "Fc_scan", "metagga"):
            # The meta-GGA columns divide by tau_unif ~ rho^{5/3}, so the SCAN
            # targets + iso-orbital alpha are ill-conditioned in the low-density
            # tail and NOT reproducible to machine-epsilon run-to-run (unlike the
            # GGA columns, which DO guard the density_fit code path at 1e-10 above:
            # a real DF-vs-non-DF difference is ~1e-4 and would trip them). Just
            # sanity-check shape + finiteness for the new columns.
            assert off[k].shape == base[k].shape
            assert np.all(np.isfinite(off[k])) and np.all(np.isfinite(base[k]))
            continue
        np.testing.assert_allclose(off[k], base[k], rtol=0, atol=1e-10)


@pytest.mark.slow
def test_generate_writes_basis_tagged_manifest(tmp_path):
    """A full generation writes a manifest recording the basis it used."""
    path = pdg.generate_pretrain_data_npz(
        str(tmp_path), atoms=(("He", 0),), basis="def2-svp", grid_level=1,
        polarized=False, descriptors=False)
    assert pdg.pretrain_data_is_current(path, basis="def2-svp", grid_level=1)
    assert not pdg.pretrain_data_is_current(path, basis="def2-tzvp",
                                            grid_level=1)


def test_manifest_records_and_compares_atoms(tmp_path, monkeypatch):
    """The pretrain-data manifest must key the ATOM SET too: extending
    pretraining coverage (e.g. +Li/C/F/Na for dfs_step7 v2) must force a
    regen instead of silently reusing 4-atom data; legacy manifests without
    the key read as the historical default (no spurious regen)."""
    import json
    import numpy as np
    import xcquinox.alec.pretrain_data_gen as pdg

    calls = []

    def fake_cols(system, basis, grid_level, **kw):
        calls.append(system.name)
        return {k: np.ones(2) for k in ("rho", "sigma", "Fx", "Fc", "weights",
                                        "zeta", "Fx_scan", "Fc_scan",
                                        "e_lda_x", "e_lda_c")} | {
            "cusp": np.ones((2, 2)), "dm": np.ones((2, 2)),
            "rung35": np.ones((2, 2)), "rung35ms": np.ones((2, 6)),
            "metagga": np.ones((2, 1))}

    # The generator builds every system through _system_columns (the atom
    # wrapper is a named entry point for callers, not the generator's seam).
    monkeypatch.setattr(pdg, "_system_columns", fake_cols)
    # The default set carries O and the default grid level is 1, so every
    # ensure below waives the irreproducible-degenerate refusal deliberately:
    # this test is about the ATOM SET in the manifest and the column builder
    # is faked.
    coarse = dict(allow_irreproducible_degenerate=True)
    default_path = pdg.ensure_pretrain_data(str(tmp_path), **coarse)
    assert calls == [s for s, _ in pdg.DEFAULT_PRETRAIN_ATOMS]
    meta = json.loads(open(default_path + ".manifest.json").read())
    assert meta["atoms"] == [[s, sp] for s, sp in pdg.DEFAULT_PRETRAIN_ATOMS]

    # same atoms -> current, no regen
    calls.clear()
    pdg.ensure_pretrain_data(str(tmp_path), **coarse)
    assert calls == []

    # extended atom set -> stale -> regen with the new set
    full = pdg.DEFAULT_PRETRAIN_ATOMS + (("Na", 1), ("Li", 1))
    pdg.ensure_pretrain_data(str(tmp_path), atoms=full, **coarse)
    assert calls == [s for s, _ in full]

    # legacy manifest without 'atoms' key (and without the system list that
    # now accompanies it) == DEFAULT (no spurious regen) but stale for a
    # non-default set
    meta2 = json.loads(open(default_path + ".manifest.json").read())
    meta2.pop("atoms")
    meta2.pop("systems")
    open(default_path + ".manifest.json", "w").write(json.dumps(meta2))
    assert pdg.pretrain_data_is_current(
        default_path, basis=pdg.DEFAULT_BASIS,
        grid_level=pdg.DEFAULT_GRID_LEVEL) is True
    assert pdg.pretrain_data_is_current(
        default_path, basis=pdg.DEFAULT_BASIS,
        grid_level=pdg.DEFAULT_GRID_LEVEL, atoms=full) is False


def test_parse_pretrain_atoms_forms():
    from xcquinox.alec.cluster.grid_config import _parse_pretrain_atoms
    assert _parse_pretrain_atoms(None) == ()
    assert _parse_pretrain_atoms({}) == ()
    assert _parse_pretrain_atoms({"H": 1, "Na": 1}) == (("H", 1), ("Na", 1))
    # round-tripped resolved_config (lists of pairs)
    assert _parse_pretrain_atoms([["H", 1], ["O", 2]]) == (("H", 1), ("O", 2))


def test_generator_writes_are_atomic(tmp_path, monkeypatch):
    """The pretrain-data dir is SHARED across sweep runs; two concurrently
    submitted runs can both reach a stale file and regenerate. Both the npz
    and its manifest must land via tmp + os.replace so a reader never sees a
    torn file -- concurrent regeneration then only duplicates compute."""
    import numpy as np
    import xcquinox.alec.pretrain_data_gen as pdg

    def fake_cols(system, basis, grid_level, **kw):
        return {k: np.ones(2) for k in ("rho", "sigma", "Fx", "Fc", "weights",
                                        "zeta", "Fx_scan", "Fc_scan",
                                        "e_lda_x", "e_lda_c")} | {
            "cusp": np.ones((2, 2)), "dm": np.ones((2, 2)),
            "rung35": np.ones((2, 2)), "rung35ms": np.ones((2, 6)),
            "metagga": np.ones((2, 1))}

    monkeypatch.setattr(pdg, "_system_columns", fake_cols)
    replaces = []
    real_replace = os.replace
    monkeypatch.setattr(
        pdg.os, "replace",
        lambda src, dst: (replaces.append((src, dst)), real_replace(src, dst)))

    out = pdg.generate_pretrain_data_npz(str(tmp_path), atoms=(("H", 1),),
                                         basis="def2-svp", grid_level=1)
    # Final artifacts valid and loadable.
    d = np.load(out)
    assert "rho_all" in d.files and "rho_mesh" in d.files
    assert pdg.read_pretrain_manifest(out)["basis"] == "def2-svp"
    # Both writes went through an atomic rename onto their final paths.
    dsts = [dst for _s, dst in replaces]
    assert out in dsts, "npz was not written via os.replace"
    assert pdg._pretrain_manifest_path(out) in dsts, \
        "manifest was not written via os.replace"
    # No tmp remnants survive.
    leftovers = [f for f in os.listdir(tmp_path) if ".tmp." in f]
    assert leftovers == [], leftovers
