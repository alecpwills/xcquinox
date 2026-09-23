"""Tests for xcquinox.pipeline.pyscf_determinism: fixed quadrature blocking.

The defect these pin: pyscf sizes the block loop of the XC quadrature from
``max_memory - lib.current_memory()``, so the summation order of the
reference SCF -- and its converged energy and density at the 1e-13 level --
follows the memory history of the process. The block-count tests exercise
the seam directly on ``NumInt.block_loop``; the subprocess test runs the
library's own reference path in two processes with different memory
histories and requires bitwise agreement with the pin, and disagreement
without it.
"""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from pyscf import dft, gto, lib, scf
from pyscf.dft.gen_grid import BLKSIZE

from xcquinox.pipeline.config import MoleculeSpec
from xcquinox.pipeline import pyscf_determinism as pd
from xcquinox.pipeline.pyscf_determinism import (
    REFERENCE_XC_BLKSIZE, pin_reference_scf, pin_xc_block_size,
    pinned_xc_block_size)

_H2O_ATOM = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"
_O_ATOM = "O 0 0 0"
_PRODUCTION_BASIS = "6-311++G(3df,2pd)"


def _h2o_rks(level=3, basis="def2-svp", build_grid=True):
    mol = gto.M(atom=_H2O_ATOM, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = level
    # The pruned point counts pinned below (30632 at level 3, 9304 at level 1)
    # are the grids of the density cutoff this module fixes, on any release.
    mf.small_rho_cutoff = pd.REFERENCE_SMALL_RHO_CUTOFF
    if build_grid:
        mf.grids.build()
    return mol, mf


def _block_sizes(mf, **kwargs):
    mol = mf.mol
    return [int(weight.shape[0]) for _, _, weight, _
            in mf._numint.block_loop(mol, mf.grids, mol.nao, 1, **kwargs)]


def test_pinned_block_count_is_independent_of_max_memory():
    _, mf = _h2o_rks()
    ngrids = mf.grids.weights.size
    assert pin_xc_block_size(mf) == REFERENCE_XC_BLKSIZE
    expected = [REFERENCE_XC_BLKSIZE] * (ngrids // REFERENCE_XC_BLKSIZE)
    if ngrids % REFERENCE_XC_BLKSIZE:
        expected.append(ngrids % REFERENCE_XC_BLKSIZE)
    assert expected == [12544, 12544, 8616]
    for budget in (4000, 1, -100, 2000000):
        assert _block_sizes(mf, max_memory=budget) == expected
    # The pin holds for the value-only loop of the grid pruning as well.
    lda_blocks = [int(w.shape[0]) for _, _, w, _
                  in mf._numint.block_loop(mf.mol, mf.grids, mf.mol.nao, 0,
                                           max_memory=-100)]
    assert lda_blocks == expected


def test_pin_reaches_the_second_order_and_density_fitting_wrappers():
    """newton() and density_fit() copy the mean-field's __dict__, so they
    share the pinned integrator; the reference paths rely on that."""
    _, mf = _h2o_rks()
    pin_xc_block_size(mf)
    so = mf.newton()
    assert so._numint is mf._numint
    assert pinned_xc_block_size(so) == REFERENCE_XC_BLKSIZE
    df = mf.density_fit()
    assert df._numint is mf._numint
    assert pinned_xc_block_size(df) == REFERENCE_XC_BLKSIZE
    # And the other order: pinning the wrapper pins the object it wraps.
    _, mf2 = _h2o_rks()
    df2 = mf2.density_fit()
    pin_xc_block_size(df2)
    assert pinned_xc_block_size(mf2) == REFERENCE_XC_BLKSIZE


def test_pin_is_idempotent_and_refuses_a_second_value():
    _, mf = _h2o_rks()
    assert pin_xc_block_size(mf, 4 * BLKSIZE) == 4 * BLKSIZE
    assert pin_xc_block_size(mf, 4 * BLKSIZE) == 4 * BLKSIZE
    assert _block_sizes(mf, max_memory=4000)[0] == 4 * BLKSIZE
    with pytest.raises(ValueError, match="already pinned at 224"):
        pin_xc_block_size(mf, 8 * BLKSIZE)
    # The refusal left the first pin in place.
    assert pinned_xc_block_size(mf) == 4 * BLKSIZE


def test_pin_small_rho_cutoff_reports_pyscfs_previous_default():
    """The density cutoff of the reference grid is pinned, not inherited.

    PySCF 2.14.0 moved the Kohn-Sham class default ``small_rho_cutoff`` from
    1e-7 to 0, which turns off the density-based pruning of the integration
    grid at the first SCF cycle (``prune_small_rho_grids_``, ``rho w >
    threshold / n_points``) and leaves the small-density tail rows in: 9 to 15
    percent more points on the reference probes of this suite. Pinned at 1e-7
    the grids of the two releases are identical bit for bit, coordinates and
    weights, which is what makes the recorded numbers release-independent.
    """
    assert pd.REFERENCE_SMALL_RHO_CUTOFF == 1e-7
    mol, mf = _h2o_rks(level=1, build_grid=False)
    mf.small_rho_cutoff = 0.0          # the 2.14 class default, set explicitly
    assert pd.pin_small_rho_cutoff(mf) == 1e-7
    assert mf.small_rho_cutoff == 1e-7
    # A Hartree-Fock object has no density cutoff to pin: nothing is set and
    # the pin reports that it did not apply.
    hf = scf.RHF(mol)
    assert pd.pin_small_rho_cutoff(hf) is None
    assert not hasattr(hf, "small_rho_cutoff")


def test_pinned_scf_changes_only_the_summation_order():
    """The pin changes the order the quadrature is summed in, nothing else:
    the converged energy agrees to the round-off of the quadrature and the
    pruned grid is the same set of points with the same weights (the
    pruning decides per point, so the block size cannot move it)."""
    # The grid is left for the SCF to build, so that it prunes it on the
    # initial guess exactly as the reference paths do.
    _, plain = _h2o_rks(build_grid=False)
    _, pinned = _h2o_rks(build_grid=False)
    pin_xc_block_size(pinned)
    plain.kernel()
    pinned.kernel()
    assert plain.converged and pinned.converged
    # Measured 1.4e-14 Ha on this grid (3 blocks against 1); the bound is
    # 1e4 times the measured order difference and 1e5 times below the SCF
    # convergence criterion, so a pin that changed the physics fails it.
    assert abs(float(plain.e_tot) - float(pinned.e_tot)) < 1e-10
    assert plain.grids.weights.shape == pinned.grids.weights.shape
    assert np.array_equal(plain.grids.weights, pinned.grids.weights)
    assert np.array_equal(plain.grids.coords, pinned.grids.coords)
    assert plain.grids.weights.size == 30632


def test_precompute_records_the_pins_in_the_metadata():
    from xcquinox.pipeline.data import precompute_fixed_density_data
    spec = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                        charge=0, spin=0, atom_composition=(("H", 2),),
                        grid_level=1)
    previous = lib.num_threads()
    try:
        lib.num_threads(1)
        record = precompute_fixed_density_data(spec)
    finally:
        lib.num_threads(previous)
    meta = record["mol_metadata"]
    assert meta["reference_xc_blksize"] == REFERENCE_XC_BLKSIZE
    assert meta["reference_blas_threads"] == 1
    assert meta["reference_small_rho_cutoff"] == 1e-7
    assert isinstance(meta["reference_xc_blksize"], int)
    assert isinstance(meta["reference_blas_threads"], int)
    assert isinstance(meta["reference_small_rho_cutoff"], float)


# ---------------------------------------------------------------------------
# End to end: two processes with different memory histories.
# ---------------------------------------------------------------------------

_CHILD = textwrap.dedent("""
    import hashlib, json, os, sys
    import numpy as np
    hold = np.ones(int(float(sys.argv[1]) * 2 ** 30 / 8)) if float(sys.argv[1]) else None
    from pyscf import lib
    from pyscf.dft import numint
    counts = []
    _unpinned = numint.NumInt.block_loop
    def counting(self, *args, **kwargs):
        n = 0
        for item in _unpinned(self, *args, **kwargs):
            n += 1
            yield item
        counts.append(n)
    numint.NumInt.block_loop = counting
    import xcquinox.pipeline.data as data_mod
    from xcquinox.pipeline.config import MoleculeSpec
    if sys.argv[2] == "off":
        # Only the memory-dependent pins are switched off. The density cutoff
        # stays pinned, so the two children prune the same grid and differ in
        # the summation order alone.
        from xcquinox.pipeline.pyscf_determinism import ReferencePins, pin_small_rho_cutoff
        data_mod.pin_reference_scf = lambda mf: ReferencePins(None, lib.num_threads(), "unpinned",
                                                              pin_small_rho_cutoff(mf))
    def digest(x):
        return hashlib.sha1(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()).hexdigest()
    out = {"rss_mb": lib.current_memory()[0], "threads": lib.num_threads()}
    specs = {
        "O": dict(atom="O 0 0 0", charge=0, spin=2, atom_composition=(("O", 1),), lock=3e-5),
        "H2O": dict(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", charge=0, spin=0,
                    atom_composition=(("O", 1), ("H", 2)), lock=0.0),
    }
    for name, s in specs.items():
        counts.clear()
        spec = MoleculeSpec(name=name, atom=s["atom"], basis="def2-svp", charge=s["charge"],
                            spin=s["spin"], atom_composition=s["atom_composition"], grid_level=3)
        md = data_mod.precompute_fixed_density_data(spec, orientation_lock_strength=s["lock"])
        out[name] = {k: digest(md[k]) for k in ("dm_pbe", "rho_grid", "sigma_grid",
                     "nabla_rho_grid", "grid_weights", "ao_grid", "ao_grid_deriv", "vxc_pbe", "j_matrix")}
        out[name]["E_pbe"] = float(md["E_pbe"]).hex()
        out[name]["E_non_xc"] = float(md["E_non_xc"]).hex()
        out[name]["max_blocks"] = max(counts)
        out[name]["blksize"] = md["mol_metadata"]["reference_xc_blksize"]
        out[name]["threads"] = md["mol_metadata"]["reference_blas_threads"]
    print("RESULT " + json.dumps(out))
""")


def _run_child(hold_gib, pin):
    env = dict(os.environ)
    env.update({
        # One OpenMP thread: pyscf's threaded reductions are not associative,
        # so bitwise agreement is only defined at one thread (the module
        # docstring records the measurement); the block-size pin is what is
        # under test here.
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
        # pyscf's ceiling, so that a 2 GiB hold puts the process above it
        # (the regime a production process reaches at its default 4000 MB
        # once jax and a few precomputes are resident).
        "PYSCF_MAX_MEMORY": "2000",
    })
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(hold_gib), pin],
                          env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(line[-1][len("RESULT "):])


_COMPARED = ("dm_pbe", "rho_grid", "sigma_grid", "nabla_rho_grid",
             "grid_weights", "ao_grid", "ao_grid_deriv", "vxc_pbe",
             "j_matrix", "E_pbe", "E_non_xc")


def test_reference_records_are_bitwise_identical_across_memory_histories():
    """Two processes, one clean and one holding 2 GiB above pyscf's ceiling,
    produce the same O (locked) and H2O records bit for bit with the pin --
    and different ones without it, on the same code, in the same session:
    the pin is what makes the difference."""
    clean = _run_child(0.0, "on")
    heavy = _run_child(2.0, "on")
    assert heavy["rss_mb"] > 2000 > clean["rss_mb"]
    for name in ("O", "H2O"):
        assert clean[name]["blksize"] == heavy[name]["blksize"] == \
            REFERENCE_XC_BLKSIZE
        assert clean[name]["threads"] == heavy[name]["threads"] == 1
        assert clean[name]["max_blocks"] == heavy[name]["max_blocks"]
        for key in _COMPARED:
            assert clean[name][key] == heavy[name][key], (name, key)
    clean_off = _run_child(0.0, "off")
    heavy_off = _run_child(2.0, "off")
    for name in ("O", "H2O"):
        assert clean_off[name]["blksize"] is None
        # The defect: the block count follows the memory history ...
        assert clean_off[name]["max_blocks"] == 1
        assert heavy_off[name]["max_blocks"] > 1
        # ... and with it the density matrix and the grid columns built
        # from it (the grid itself, the AO tables and the core Hamiltonian
        # do not depend on the summation order).
        assert clean_off[name]["dm_pbe"] != heavy_off[name]["dm_pbe"], name
        assert clean_off[name]["rho_grid"] != heavy_off[name]["rho_grid"]
        assert clean_off[name]["grid_weights"] == heavy_off[name]["grid_weights"]
        assert clean_off[name]["ao_grid_deriv"] == heavy_off[name]["ao_grid_deriv"]
        # The pinned record of a clean process is the clean unpinned one
        # whenever the grid fits one block (O: 11904 points), and a
        # different summation order otherwise (H2O: 30632 points, 3 blocks).
    assert clean["O"]["dm_pbe"] == clean_off["O"]["dm_pbe"]
    assert clean["O"]["E_pbe"] == clean_off["O"]["E_pbe"]
    assert clean["H2O"]["max_blocks"] == 3


# ---------------------------------------------------------------------------
# The two-electron integral path: pyscf's incore/direct choice follows
# process memory too (SCF._is_mem_enough), and the two paths differ at the
# 1e-13 level. It is pinned to the system size.
# ---------------------------------------------------------------------------

from xcquinox.pipeline.pyscf_determinism import (  # noqa: E402
    REFERENCE_ERI_INCORE_MB, eri_path_for_nao, pin_eri_path)


def test_eri_path_rule_is_the_packed_tensor_estimate_against_the_budget():
    # nao**4 / 1e6 MB, the estimate SCF._is_mem_enough uses, without its
    # lib.current_memory() term: 211**4 / 1e6 = 1982 MB fits 2000 MB,
    # 212**4 / 1e6 = 2020 MB does not.
    assert eri_path_for_nao(211) == "incore"
    assert eri_path_for_nao(212) == "direct"
    assert eri_path_for_nao(24) == "incore"       # H2O / def2-svp
    assert eri_path_for_nao(99) == "incore"       # CH4 at the production basis
    assert eri_path_for_nao(315) == "direct"      # C5H8 at the production basis
    assert eri_path_for_nao(315, incore_budget_mb=1e5) == "incore"
    assert eri_path_for_nao(2, incore_budget_mb=0.0) == "direct"


def test_pinned_eri_path_ignores_max_memory():
    _, mf = _h2o_rks(build_grid=False)
    mf.max_memory = 0                      # pyscf alone would go direct
    assert pin_eri_path(mf) == "incore"
    assert mf._is_mem_enough() is True
    assert mf._eri is None
    mf.kernel()
    assert mf.converged
    assert mf._eri is not None             # the tensor was held in memory
    _, direct = _h2o_rks(build_grid=False)
    direct.max_memory = 1e6                # pyscf alone would go incore
    assert pin_eri_path(direct, incore_budget_mb=0.0) == "direct"
    assert direct._is_mem_enough() is False
    direct.kernel()
    assert direct.converged
    assert direct._eri is None             # never materialised
    # Both paths are the same physics: agreement to the round-off of the
    # screened, incremental build (measured 1e-14..1e-13 Ha).
    assert abs(float(mf.e_tot) - float(direct.e_tot)) < 1e-10


def test_pinned_mean_fields_are_freed_by_refcount_alone():
    """The pins must not create reference cycles: a pinned mean-field (and
    its integrator) dies at del with the cyclic collector disabled, exactly
    as an unpinned one does. The OEP inner loop builds and drops one pinned
    object per objective evaluation, so a cycle turns the loop into an
    accumulator that only gc.collect() drains (measured before the fix:
    the object survives del, and an 80-iteration build-and-discard loop
    peaks 6x higher than unpinned)."""
    import gc
    import weakref
    gc.disable()
    try:
        refs = []
        for _ in range(3):
            mol = gto.M(atom=_H2O_ATOM, basis="def2-svp", verbose=0)
            mf = dft.RKS(mol)
            pin_reference_scf(mf)
            refs.append((weakref.ref(mf), weakref.ref(mf._numint)))
            del mf
        assert all(r() is None for r, _ in refs), \
            "pinned mean-fields are kept alive by a reference cycle"
        assert all(n() is None for _, n in refs), \
            "pinned integrators are kept alive by a reference cycle"
        mol = gto.M(atom=_H2O_ATOM, basis="def2-svp", verbose=0)
        hf = scf.RHF(mol)
        pin_reference_scf(hf)
        wr = weakref.ref(hf)
        del hf
        assert wr() is None
    finally:
        gc.enable()


# ---------------------------------------------------------------------------
# The OEP path builds its own mean-field objects: the baseline KS SCF and a
# fresh Hartree-Fock object per inner SCF (J plus a fixed potential matrix).
# ---------------------------------------------------------------------------


_DF_CHILD = textwrap.dedent("""
    import hashlib, json, sys, tempfile
    import numpy as np
    hold = np.ones(int(float(sys.argv[1]) * 2 ** 30 / 8)) if float(sys.argv[1]) else None
    from ase import Atoms
    from pyscf import df as pyscf_df
    from pyscf import gto, lib
    aux_calls = []
    real_loop = pyscf_df.df.DF.loop
    def counting(self, blksize=None):
        aux_calls.append(blksize)
        yield from real_loop(self, blksize)
    pyscf_df.df.DF.loop = counting
    from xcquinox.pipeline import external_refs as ext
    def digest(x):
        return hashlib.sha1(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()).hexdigest()
    atom = ("C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; "
            "H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276")
    basis = "6-311++G(3df,2pd)"
    entry = ext.SpeciesEntry(name="CH4", charge=0, spin=0, source="test")
    atoms = Atoms("CH4", positions=[[0, 0, 0], [0.6276, 0.6276, 0.6276],
                                    [-0.6276, -0.6276, 0.6276],
                                    [-0.6276, 0.6276, -0.6276],
                                    [0.6276, -0.6276, -0.6276]])
    tmp = tempfile.mkdtemp(prefix="d29_df_")
    rec = ext.run_scf_with_cache(entry, atoms, cache_dir=tmp, basis=basis,
                                 grid_level=3, density_fit=True)
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    aux_calls.clear()
    mf_hf = ext._prepare_converged_hf(mol, dm0=np.asarray(rec["dm"]),
                                      is_uks=False, density_fit=True,
                                      basis=basis)
    out = {"rss_mb": lib.current_memory()[0],
           "dm": digest(rec["dm"]), "e_tot": float(rec["e_tot"]).hex(),
           "eri_path": rec["reference_eri_path"],
           "hf_dm": digest(mf_hf.make_rdm1()),
           "hf_e_tot": float(mf_hf.e_tot).hex(),
           "aux_blksizes": sorted(set(a for a in aux_calls if a is not None)),
           "naux": int(mf_hf.with_df.get_naoaux())}
    print("RESULT " + json.dumps(out))
""")


