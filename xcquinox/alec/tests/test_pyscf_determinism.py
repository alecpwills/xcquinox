"""Tests for xcquinox.alec.pyscf_determinism: fixed quadrature blocking.

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

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.pyscf_determinism import (
    REFERENCE_XC_BLKSIZE, ReferencePins, pin_reference_scf,
    pin_xc_block_size, pinned_xc_block_size, reference_thread_count)

_H2O_ATOM = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"
_O_ATOM = "O 0 0 0"
_PRODUCTION_BASIS = "6-311++G(3df,2pd)"


def _h2o_rks(level=3, basis="def2-svp", build_grid=True):
    mol = gto.M(atom=_H2O_ATOM, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = level
    if build_grid:
        mf.grids.build()
    return mol, mf


def _block_sizes(mf, **kwargs):
    mol = mf.mol
    return [int(weight.shape[0]) for _, _, weight, _
            in mf._numint.block_loop(mol, mf.grids, mol.nao, 1, **kwargs)]


def test_reference_block_size_is_a_positive_multiple_of_pyscf_blksize():
    # block_loop asserts blksize % BLKSIZE == 0 (the AO screening table is
    # indexed per BLKSIZE-aligned sub-block); the value itself is the one the
    # recorded metadata and the memory bound below are stated for.
    assert REFERENCE_XC_BLKSIZE > 0
    assert REFERENCE_XC_BLKSIZE % BLKSIZE == 0
    assert REFERENCE_XC_BLKSIZE == 224 * BLKSIZE == 12544


def test_unpinned_block_count_follows_max_memory():
    """The seam of the defect: pyscf's own block loop takes a different
    number of blocks for the same grid at two memory budgets."""
    _, mf = _h2o_rks()
    ngrids = mf.grids.weights.size
    assert ngrids == 33704
    generous = _block_sizes(mf, max_memory=4000)
    starved = _block_sizes(mf, max_memory=1)
    assert generous == [ngrids]
    assert len(starved) > 1
    assert sum(starved) == ngrids
    # Above the memory ceiling (a negative budget) pyscf falls to its floor
    # of 4 * BLKSIZE points, the regime a process past 4000 MB runs in.
    above_ceiling = _block_sizes(mf, max_memory=-100)
    assert above_ceiling[0] == 4 * BLKSIZE
    assert len(above_ceiling) == -(-ngrids // (4 * BLKSIZE))


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


def test_pin_reports_its_value_and_reads_back():
    _, mf = _h2o_rks()
    assert pinned_xc_block_size(mf) is None
    assert pin_xc_block_size(mf) == REFERENCE_XC_BLKSIZE
    assert pinned_xc_block_size(mf) == REFERENCE_XC_BLKSIZE
    assert mf._numint.block_loop.__doc__.startswith(
        f"NumInt.block_loop pinned at {REFERENCE_XC_BLKSIZE}")


def test_explicit_block_size_is_honoured_over_the_pin():
    # A caller that names a block size has sized its own buffer for it.
    _, mf = _h2o_rks()
    pin_xc_block_size(mf)
    sizes = _block_sizes(mf, blksize=10 * BLKSIZE)
    assert sizes[0] == 10 * BLKSIZE
    assert len(sizes) == -(-mf.grids.weights.size // (10 * BLKSIZE))


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


def test_pin_routes_through_an_instrumented_class_block_loop(monkeypatch):
    """A class-level wrapper of the form (self, *args, **kwargs) -- what a
    counting or logging instrument installs -- is forwarded to, not
    refused, and still sees the pinned block size."""
    from pyscf.dft import numint
    seen = []
    unpatched = numint.NumInt.block_loop

    def counting(self, *args, **kwargs):
        seen.append(kwargs.get("blksize"))
        return unpatched(self, *args, **kwargs)

    monkeypatch.setattr(numint.NumInt, "block_loop", counting)
    _, mf = _h2o_rks()
    assert pin_xc_block_size(mf) == REFERENCE_XC_BLKSIZE
    assert _block_sizes(mf, max_memory=1) == [12544, 12544, 8616]
    assert seen == [REFERENCE_XC_BLKSIZE]


def test_pin_refuses_a_block_loop_it_cannot_forward_to(monkeypatch):
    from pyscf.dft import numint

    def foreign(self, mol, grids, chunk=None):
        return iter(())

    monkeypatch.setattr(numint.NumInt, "block_loop", foreign)
    _, mf = _h2o_rks()
    with pytest.raises(RuntimeError, match="does not take the parameters"):
        pin_xc_block_size(mf)
    assert pinned_xc_block_size(mf) is None


def test_pin_is_idempotent_and_refuses_a_second_value():
    _, mf = _h2o_rks()
    assert pin_xc_block_size(mf, 4 * BLKSIZE) == 4 * BLKSIZE
    assert pin_xc_block_size(mf, 4 * BLKSIZE) == 4 * BLKSIZE
    assert _block_sizes(mf, max_memory=4000)[0] == 4 * BLKSIZE
    with pytest.raises(ValueError, match="already pinned at 224"):
        pin_xc_block_size(mf, 8 * BLKSIZE)
    # The refusal left the first pin in place.
    assert pinned_xc_block_size(mf) == 4 * BLKSIZE


@pytest.mark.parametrize("bad", [0, -BLKSIZE, 100, BLKSIZE + 1, 2.5 * BLKSIZE])
def test_pin_refuses_a_block_size_that_is_not_a_positive_multiple(bad):
    _, mf = _h2o_rks()
    with pytest.raises(ValueError, match="positive multiple"):
        pin_xc_block_size(mf, bad)
    assert pinned_xc_block_size(mf) is None


def test_hartree_fock_has_no_quadrature_to_pin():
    mol = gto.M(atom=_H2O_ATOM, basis="def2-svp", verbose=0)
    for mf in (scf.RHF(mol), scf.UHF(mol), scf.RHF(mol).density_fit()):
        assert pin_xc_block_size(mf) is None
        assert pinned_xc_block_size(mf) is None
    pins = pin_reference_scf(scf.RHF(mol))
    assert pins == ReferencePins(xc_blksize=None,
                                 threads=reference_thread_count(),
                                 eri_path="incore")


def test_reference_thread_count_is_pyscfs_openmp_count():
    assert reference_thread_count() == int(lib.num_threads())
    previous = lib.num_threads()
    try:
        lib.num_threads(1)
        assert reference_thread_count() == 1
    finally:
        lib.num_threads(previous)


def test_pin_reference_scf_reports_both_stamps():
    _, mf = _h2o_rks()
    pins = pin_reference_scf(mf)
    assert pins.xc_blksize == REFERENCE_XC_BLKSIZE
    assert pins.threads == int(lib.num_threads())
    assert pinned_xc_block_size(mf) == REFERENCE_XC_BLKSIZE


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


def test_block_memory_bound_holds_for_the_largest_pool_species():
    """One GGA / meta-GGA block costs 5 * blksize * nao * 8 bytes (four AO
    components plus one scratch table); the block size was chosen so that
    this stays under 200 MB for every species of the benchmark pools at the
    production basis. Anchored to the pools rather than to a number: a
    larger species added to a pool that breaks the bound fails here."""
    from xcquinox.alec import full_benchmark_pools
    largest = 0
    for loader in (full_benchmark_pools.load_full_bh76,
                   full_benchmark_pools.load_full_w411):
        specs, _ = loader(basis=_PRODUCTION_BASIS, grid_level=3)
        for spec in specs.values():
            mol = gto.M(atom=spec.atom, basis=_PRODUCTION_BASIS,
                        charge=spec.charge, spin=spec.spin, verbose=0)
            largest = max(largest, int(mol.nao))
    assert largest == 315  # bh76 C5H8 / RKT22 (13 atoms)
    block_bytes = 5 * REFERENCE_XC_BLKSIZE * largest * 8
    assert block_bytes < 200e6
    assert block_bytes == 158_054_400
    # ... and the small probes of the test suite and the closed-shell
    # fixture (H2O, def2-svp, grid level 1: 9304 pruned points) stay one
    # block, i.e. the order a clean process already summed them in.
    mol = gto.M(atom=_H2O_ATOM, basis="def2-svp", verbose=0)
    mf = dft.RKS(mol)
    mf.grids.level = 1
    mf.initialize_grids(mol, mf.get_init_guess(mol, mf.init_guess))
    assert mf.grids.weights.size == 9304 <= REFERENCE_XC_BLKSIZE


def test_precompute_records_the_pins_in_the_metadata():
    from xcquinox.alec.data import precompute_fixed_density_data
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
    assert isinstance(meta["reference_xc_blksize"], int)
    assert isinstance(meta["reference_blas_threads"], int)


def test_run_scf_with_cache_records_the_pins_and_keeps_an_older_cache(
        tmp_path):
    """The stamps are payload, not identity: a cache written before they
    existed is still a hit and reports them as None."""
    from ase import Atoms
    from xcquinox.alec.external_refs import (
        SpeciesEntry, _intermediate_cache_name, run_scf_with_cache)
    entry = SpeciesEntry(name="H2", charge=0, spin=0, source="test")
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    rec = run_scf_with_cache(entry, atoms, cache_dir=tmp_path, basis="sto-3g",
                             grid_level=1)
    assert rec["reference_xc_blksize"] == REFERENCE_XC_BLKSIZE
    assert rec["reference_blas_threads"] == int(lib.num_threads())
    path = tmp_path / "_intermediates" / _intermediate_cache_name(
        "H2", grid_level=1, basis="sto-3g", density_fit=False, kind="scf")
    with np.load(path, allow_pickle=False) as z:
        assert int(z["reference_xc_blksize"]) == REFERENCE_XC_BLKSIZE
        older = {k: z[k] for k in z.files
                 if k not in ("reference_xc_blksize",
                              "reference_blas_threads")}
    np.savez_compressed(path, **older)
    hit = run_scf_with_cache(entry, atoms, cache_dir=tmp_path, basis="sto-3g",
                             grid_level=1)
    assert hit["reference_xc_blksize"] is None
    assert hit["reference_blas_threads"] is None
    assert np.array_equal(hit["dm"], rec["dm"])
    assert hit["e_tot"] == rec["e_tot"]


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
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.config import MoleculeSpec
    if sys.argv[2] == "off":
        from xcquinox.alec.pyscf_determinism import ReferencePins
        data_mod.pin_reference_scf = lambda mf: ReferencePins(None, lib.num_threads(), "unpinned")
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

from xcquinox.alec.pyscf_determinism import (  # noqa: E402
    REFERENCE_ERI_INCORE_MB, eri_path_for_nao, pin_eri_path)


def test_eri_budget_is_pyscfs_own_integral_tensor_budget():
    from pyscf.df import incore
    assert REFERENCE_ERI_INCORE_MB == 2000.0 == incore.MAX_MEMORY


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


def test_unpinned_eri_choice_follows_max_memory():
    """The seam: pyscf's predicate reads max_memory against current memory."""
    _, mf = _h2o_rks(build_grid=False)
    mf.max_memory = 1e6
    assert mf._is_mem_enough() is True
    mf.max_memory = 0
    assert mf._is_mem_enough() is False


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


def test_eri_pin_is_idempotent_and_refuses_a_second_path():
    _, mf = _h2o_rks(build_grid=False)
    assert pin_eri_path(mf) == "incore"
    assert pin_eri_path(mf) == "incore"
    with pytest.raises(ValueError, match="already pinned to 'incore'"):
        pin_eri_path(mf, incore_budget_mb=0.0)
    assert mf._is_mem_enough() is True


def test_eri_pin_leaves_a_density_fitted_object_alone():
    _, mf = _h2o_rks(build_grid=False)
    df = mf.density_fit()
    assert pin_eri_path(df) == "df"
    assert not hasattr(df, "_xcquinox_eri_path")
    mol = gto.M(atom=_H2O_ATOM, basis="def2-svp", verbose=0)
    assert pin_eri_path(scf.RHF(mol).density_fit()) == "df"
    assert pin_eri_path(scf.RHF(mol)) == "incore"


def test_eri_pin_reaches_the_second_order_wrapper():
    _, mf = _h2o_rks(build_grid=False)
    mf.max_memory = 0
    pin_eri_path(mf)
    so = mf.newton()
    assert so._is_mem_enough() is True
    assert so._xcquinox_eri_path == "incore"


def test_pin_reference_scf_reports_the_eri_path():
    _, mf = _h2o_rks(build_grid=False)
    assert pin_reference_scf(mf).eri_path == "incore"
    mol = gto.M(atom=_H2O_ATOM, basis="def2-svp", verbose=0)
    assert pin_reference_scf(scf.UHF(mol)).eri_path == "incore"
    assert pin_reference_scf(scf.UHF(mol).density_fit()).eri_path == "df"


def test_precompute_records_the_eri_path():
    from xcquinox.alec.data import precompute_fixed_density_data
    spec = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                        charge=0, spin=0, atom_composition=(("H", 2),),
                        grid_level=1)
    record = precompute_fixed_density_data(spec)
    assert record["mol_metadata"]["reference_eri_path"] == "incore"


# ---------------------------------------------------------------------------
# The OEP path builds its own mean-field objects: the baseline KS SCF and a
# fresh Hartree-Fock object per inner SCF (J plus a fixed potential matrix).
# ---------------------------------------------------------------------------

def test_oep_baseline_mean_field_carries_both_pins(monkeypatch):
    from pyscf.dft import numint
    from xcquinox.alec import oep
    seen = []
    unpatched = numint.NumInt.block_loop

    def counting(self, *args, **kwargs):
        seen.append(kwargs.get("blksize"))
        return unpatched(self, *args, **kwargs)

    monkeypatch.setattr(numint.NumInt, "block_loop", counting)
    spec = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                        charge=0, spin=0, atom_composition=(("H", 2),),
                        grid_level=1)
    mol, mf = oep._build_mol_and_mf(spec, baseline_xc="pbe")
    assert mf.converged
    assert pinned_xc_block_size(mf) == REFERENCE_XC_BLKSIZE
    assert mf._xcquinox_eri_path == "incore"
    # Every quadrature loop of the baseline SCF ran at the pinned size.
    assert seen and set(seen) == {REFERENCE_XC_BLKSIZE}
    # The inner objects: one fresh RHF (closed shell) or UHF (a three-index
    # potential) per call, each pinned before its kernel.
    import pyscf.scf
    made = []
    real_rhf, real_uhf = pyscf.scf.RHF, pyscf.scf.UHF

    def recording_rhf(mol_):
        made.append(real_rhf(mol_))
        return made[-1]

    def recording_uhf(mol_):
        made.append(real_uhf(mol_))
        return made[-1]

    monkeypatch.setattr(pyscf.scf, "RHF", recording_rhf)
    monkeypatch.setattr(pyscf.scf, "UHF", recording_uhf)
    dm = mf.make_rdm1()
    vxc = oep._baseline_vxc_matrix(mol, mf, dm)
    dm_r, _, _, ok_r = oep._ks_from_vxc_matrix(mol, mf, vxc, dm0=dm)
    vxc_ab = np.stack([vxc, vxc])
    dm_u, _, _, ok_u = oep._ks_from_vxc_matrix(mol, mf, vxc_ab, dm0=dm)
    assert ok_r and ok_u
    assert len(made) == 2
    assert made[0].__class__.__name__.endswith("RHF")
    assert made[1].__class__.__name__.endswith("UHF")
    for inner in made:
        assert pinned_xc_block_size(inner) is None      # no quadrature on HF
        assert inner._xcquinox_eri_path == "incore"
        inner.max_memory = 0                             # pyscf alone: direct
        assert inner._is_mem_enough() is True
        assert inner._eri is not None                    # held in memory
    # The pinned inner SCF reproduces the baseline density it was seeded
    # with, as the Wu-Yang identity requires at the baseline potential.
    assert np.allclose(dm_r, dm, atol=1e-6)
    assert np.allclose(dm_u[0] + dm_u[1], dm, atol=1e-6)
