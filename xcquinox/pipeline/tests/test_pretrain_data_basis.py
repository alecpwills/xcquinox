"""Task 9 (density-fitting plan): pretrain-data generation must honor the
configured basis/grid_level, support density fitting on the per-atom SCF, and
carry a basis manifest so a basis change forces regeneration rather than silent
reuse of stale def2-svp data."""
import json
import os

import numpy as np

from xcquinox.pipeline import pretrain_data_gen as pdg


# --- fast: manifest + is-current logic (no SCF) ----------------------------

def _touch_npz(path):
    np.savez(path, x=np.zeros(3))


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


def test_generator_writes_are_atomic(tmp_path, monkeypatch):
    """The pretrain-data dir is SHARED across sweep runs; two concurrently
    submitted runs can both reach a stale file and regenerate. Both the npz
    and its manifest must land via tmp + os.replace so a reader never sees a
    torn file -- concurrent regeneration then only duplicates compute."""
    import numpy as np
    import xcquinox.pipeline.pretrain_data_gen as pdg

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
