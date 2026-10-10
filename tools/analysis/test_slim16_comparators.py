"""The comparator tables of the Slim16 job (hpcjobs/slim16_eval.py comparators-df).

The job's generic density-fitted table is held to an independent pyscf mean
field built here at the run's identity (def2-TZVP, the def2 core potential
through config.mole_ecp, the default auxiliary basis, grid level 4, the
pinned density cutoff) on the H atom (spin 1, an empty beta channel kept
density fitted) and the Be atom: PBE, r2SCAN, B3LYP and wB97M-V to 1e-9 Ha
(the cached helper and such a build agree to 6e-13 Ha), the PBE entries to
the PBE-DF table the job writes through its own mode (1e-9 Ha), the VV10
term of wB97M-V resolved against the build without it (11 kcal/mol on Be,
2.8 on H), libxc's classification of each functional, the identity block,
and the refusal of a shard whose mean field would not apply the term. The
shard workers are subprocesses, so the species slice reaches them through
the harness's environment variable.
"""
import contextlib
import functools
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from pyscf import dft, gto

from xcquinox.pipeline import parallel
from xcquinox.pipeline.config import mole_ecp
from xcquinox.pipeline.df_jk import default_auxbasis
from xcquinox.pipeline.eval_holdout import KCAL_PER_HA
from xcquinox.pipeline.full_benchmark_pools import (
    HELDOUT_SPECIES_SLICE_ENV, load_held_out_pools,
)
from xcquinox.pipeline.pyscf_determinism import pin_small_rho_cutoff

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hpcjobs"))
import slim16_eval as job  # noqa: E402

pytestmark = pytest.mark.slow

_H, _BE = "slim16@bh76_h", "slim16@dc13_be"
_FOUR = ("pbe", "r2scan", "b3lyp", "wb97m-v")
#: (vv10, hybrid, meta_gga) of each functional in its defining paper
_CLASS = {"pbe": (False, False, False), "r2scan": (False, False, True),
          "b3lyp": (False, True, False), "wb97m-v": (True, True, True)}


@functools.lru_cache(maxsize=None)
def _pool():
    specs, _reactions = load_held_out_pools(("slim16",), basis=job.BASIS,
                                            grid_level=job.GRID_LEVEL)
    return specs


@functools.lru_cache(maxsize=None)
def _oracle(name, xc, nlc=True):
    """The total energy of ``name`` under ``xc`` from a mean field built
    here at the run's identity; ``nlc=False`` switches the VV10 term off."""
    ms = _pool()[name]
    lines = [(tok.split()[0], tuple(float(v) for v in tok.split()[1:4]))
             for tok in ms.atom.split(";") if tok.split()]
    mol = gto.M(atom=lines, basis=job.BASIS, charge=int(ms.charge), spin=int(ms.spin),
                unit="angstrom", verbose=0, ecp=mole_ecp(job.BASIS, lines))
    mf = (dft.UKS(mol) if mol.spin else dft.RKS(mol)).density_fit(
        auxbasis=default_auxbasis(job.BASIS))
    mf.xc = xc
    mf.grids.level = job.GRID_LEVEL
    pin_small_rho_cutoff(mf)
    if not nlc:
        mf.nlc = False
    energy = float(mf.kernel())
    assert mf.converged, (name, xc, nlc)
    return {"E": energy, "nao": int(mol.nao), "do_nlc": bool(mf.do_nlc()),
            "nlc_level": int(mf.nlcgrids.level)}


def _manifest(run):
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "kind": "slim16_eval", "width": 4, "n_specs": 0,
        "identity": {"pool": "slim16"}, "networks": []}))
    return run


def _invoke(argv):
    """``(status, stderr)`` of one in-process call of the job's main."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return job.main(argv), err.getvalue()
    except SystemExit as exc:
        return exc.code, err.getvalue()


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    """pbe-df and comparators-df (the four functionals) on H and Be, one
    shard worker at one thread, in run directories of their own."""
    root = tmp_path_factory.mktemp("slim16_comparators")
    runs = {label: _manifest(root / f"{label}_run") for label in ("pbe", "four")}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JAX_ENABLE_X64", "1")
        mp.setenv(HELDOUT_SPECIES_SLICE_ENV, ",".join((_H, _BE)))
        mp.setattr(parallel, "detect_available_cpus", lambda: 1)
        status, text = _invoke(["pbe-df", str(runs["pbe"]), "--workers", "1"])
        assert status == 0, text
        status, text = _invoke(["comparators-df", str(runs["four"]), "--workers", "1",
                                "--functionals", ",".join(_FOUR)])
        assert status == 0, text
    pbe = json.loads((runs["pbe"] / job.PBE_DF_FILE).read_text(encoding="utf-8"))
    four = json.loads((runs["four"] / job.FUNCTIONALS_DF_FILE).read_text(encoding="utf-8"))
    return pbe, four


def test_the_comparator_energies_are_the_independent_build(tables):
    pbe, four = tables
    assert set(four) == {"identity", "species_slice", "functionals"}
    assert four["identity"] == {
        **job.identity(), "nlc_grid_level": _oracle(_BE, "wb97m-v")["nlc_level"],
        "n_pool_species": len(_pool()),
        "pool_species_sha256": hashlib.sha256(
            "\n".join(sorted(_pool())).encode("utf-8")).hexdigest()}
    assert pbe["identity"] == job.identity()
    assert set(four["functionals"]) == set(_FOUR)
    for key in _FOUR:
        entry = four["functionals"][key]
        libxc = (bool(dft.libxc.is_nlc(key)), bool(dft.libxc.is_hybrid_xc(key)),
                 bool(dft.libxc.is_meta_gga(key)))
        assert libxc == _CLASS[key] == (entry["vv10"], entry["hybrid"], entry["meta_gga"])
        assert entry["n_species"] == entry["n_converged"] == 2
        assert entry["n_species_not_df"] == 0
        for name in (_H, _BE):
            value = entry["species"][name]
            reference = _oracle(name, key)
            assert value["reference_eri_path"] == "df-aux240"
            assert value["n_ao"] == reference["nao"]
            assert abs(value["E_df"] - reference["E"]) <= 1e-9, (key, name)
            if key == "pbe":
                assert abs(value["E_df"] - pbe["species"][name]["E_pbe_df"]) <= 1e-9
    assert four["functionals"]["b3lyp"]["libxc"] == [{"id": 402, "name": "HYB_GGA_XC_B3LYP"}]
    assert four["functionals"]["wb97m-v"]["libxc"] == [
        {"id": 531, "name": "HYB_MGGA_XC_WB97M_V"}]


def test_the_vv10_term_is_applied_and_its_absence_refused(tables, tmp_path, monkeypatch):
    _pbe, four = tables
    for name in (_BE, _H):
        with_term, without = _oracle(name, "wb97m-v"), _oracle(name, "wb97m-v", nlc=False)
        assert with_term["do_nlc"] and not without["do_nlc"]
        energy = four["functionals"]["wb97m-v"]["species"][name]["E_df"]
        assert abs(energy - with_term["E"]) <= 1e-9
        assert abs(energy - without["E"]) * KCAL_PER_HA > 1.0
    run = _manifest(tmp_path / "refusal_run")
    names = run / "names.json"
    names.write_text(json.dumps([_BE]))
    out = run / "out.json"
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    monkeypatch.setattr(dft.rks.KohnShamDFT, "do_nlc", lambda self: False)
    cache = tmp_path / "refusal_cache"
    status, _text = _invoke(["df-shard", str(run), str(names), str(out), "wb97m-v",
                             str(cache), "E_df"])
    assert status == 0
    entry = json.loads(out.read_text(encoding="utf-8"))[_BE]
    assert "E_df" not in entry and "VV10" in entry["error"]
    # the stamp is cached with the record: a later call reads the cache and
    # refuses again, the mean field's patch undone or not
    monkeypatch.undo()
    assert list((cache / "_intermediates").glob("*.npz"))
    status, _text = _invoke(["df-shard", str(run), str(names), str(out), "wb97m-v",
                             str(cache), "E_df"])
    assert status == 0
    entry = json.loads(out.read_text(encoding="utf-8"))[_BE]
    assert "E_df" not in entry and "VV10" in entry["error"]
