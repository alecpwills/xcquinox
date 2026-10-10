"""The Laplacian-level descriptor and its architectures: the density
Laplacian from the density matrix against pyscf, the compressed column, its
scaling, the precompute record, the live solver's potential, size
consistency, the pretraining file and its identity, the rung, the registry
entries, the certificate path and the pretraining step.

Oracles: pyscf's eval_ao(deriv=2) and eval_rho(..., xctype="MGGA",
with_lapl=True) on the records' own grids, central differences, the analytic
Gaussian density, jax.grad of the energy the solver minimizes, and the file
the generator writes (jax 0.10.2, pyscf 2.14.0, numpy 2.5.3, x64)."""
import dataclasses
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pytest  # noqa: E402
from pyscf import dft, gto  # noqa: E402

jax.config.update("jax_enable_x64", True)

import xcquinox.pipeline as pipeline  # noqa: E402
from xcquinox.features import compute_reduced_laplacian  # noqa: E402
from xcquinox.pipeline import config as C  # noqa: E402
from xcquinox.pipeline import networks as N  # noqa: E402
from xcquinox.pipeline import pretrain_data_gen as pdg  # noqa: E402
from xcquinox.pipeline.config import MoleculeSpec  # noqa: E402
from xcquinox.pipeline.data import precompute_fixed_density_data  # noqa: E402
from xcquinox.pipeline.pyscf_determinism import pin_small_rho_cutoff  # noqa: E402

from xcquinox.pipeline.tests import test_parent_anchor as _pa  # noqa: E402
from xcquinox.pipeline.tests import test_solv01_split_xc as _solv  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]


def _repo_module(relative):
    """A module outside the package (the figure tools, the pretraining board,
    the energy-weight probe), loaded by path."""
    path = _REPO / relative
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    sys.modules[name] = module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _tool(filename):
    """A figure tool under ``tools/analysis`` (not a package)."""
    return _repo_module(f"tools/analysis/{filename}")


AS = _tool("arch_style.py")

#: The two registry entries and the entry each is, with its descriptor list.
_NEW = ("deep_lap_3x16", "deep_lap_geom_3x16")
_BASE = {"deep_lap_3x16": ("deep_3x16", ("lap",)),
         "deep_lap_geom_3x16": ("deep_geom_3x16", ("cusp", "lap"))}
#: The records of test_parent_anchor's fixtures: OH (open shell) and H2O.
_OH = ("OH", "O 0.0 0.0 0.0; H 0.0 0.0 0.97", 1, (("H", 1), ("O", 1)))
_H2O = ("H2O", "O 0.0 0.0 0.0; H 0.0 0.757 0.587; H 0.0 -0.757 0.587", 0,
        (("H", 2), ("O", 1)))
#: The pretraining file of the datagen, certificate and pretraining tests.
_ATOMS = (("He", 0), ("Li", 1))
#: The points the column is compared on: rho above the networks' tail
#: threshold and the generator's floor (pdg._RHO_FLOOR, 1e-10).
_VALID = 1e-10
#: 4 (3 pi^2)^(2/3): q = lap rho / (_CF rho^(5/3)).
_CF = 4.0 * (3.0 * math.pi ** 2) ** (2.0 / 3.0)
#: The column against pyscf's Laplacian: 1.9e-15 (OH), 2.8e-15 (H2O) and
#: 2.9e-15 (OH's doubled channels) between pyscf's eval_rho and the numpy
#: contraction on every grid point, the largest at the nuclei; held to 1e-12.
_COL_TOL = 1e-12
#: The historical descriptor-name -> column-stem map of pretrain.py
#: (_key_map before the item), which DESCRIPTOR_STEM_OF must keep.
_HISTORICAL_STEMS = {"dm_statistics": "dm", "cusp": "cusp", "rung35": "rung35",
                     "rung35_multishell": "rung35ms", "metagga": "metagga"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec(system, basis, grid_level):
    name, atom, spin, comp = system
    return MoleculeSpec(name=name, atom=atom, basis=basis, charge=0, spin=spin,
                        atom_composition=comp, grid_level=grid_level)


def _record(system, basis, grid_level, descriptors=(), extra_keys=()):
    keys = tuple(sorted({k for d in descriptors for k in d.required_mol_keys}
                        | set(extra_keys)))
    return precompute_fixed_density_data(_spec(system, basis, grid_level),
                                         required_keys=keys,
                                         descriptors=tuple(descriptors))


def _grid(system, basis, grid_level, md):
    """The record's grid points, rebuilt as pretrain_data_gen._system_columns
    rebuilds them (the pinned density cutoff, the grid pruned on the minao
    guess); the weights must be the record's and the AO values its stored
    table, to 1e-10."""
    _name, atom, spin, _comp = system
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    pin_small_rho_cutoff(mf)
    mf.grids.level = grid_level
    mf.initialize_grids(mol, mf.get_init_guess(mol, mf.init_guess,
                                               s1e=mf.get_ovlp(mol)))
    coords = np.asarray(mf.grids.coords)
    assert np.array_equal(np.asarray(mf.grids.weights),
                          np.asarray(md["grid_weights"]))
    ao0 = mf._numint.eval_ao(mol, coords, deriv=0)
    assert float(np.abs(ao0 - np.asarray(md["ao_grid"])).max()) < 1e-10
    return mol, coords


def _pyscf(mol, coords, p):
    """(rho, lap rho, ao2) of the 2-D matrix ``p`` from pyscf's MGGA rows."""
    ao2 = dft.numint.eval_ao(mol, coords, deriv=2)
    r = dft.numint.NumInt().eval_rho(mol, ao2, p, xctype="MGGA", with_lapl=True)
    return np.asarray(r[0]), np.asarray(r[4]), ao2


def _q(rho, lap):
    return np.asarray(compute_reduced_laplacian(jnp.asarray(rho),
                                                jnp.asarray(lap)))


def _total(dm):
    dm = np.asarray(dm)
    return dm if dm.ndim == 2 else dm[0] + dm[1]


def _gaussian(points, lam=1.0, occupation=1.0):
    """One s function chi = lam^(3/2) exp(-lam^2 r^2 / 2) on ``points``: its
    value (n, 1), gradient (3, n, 1) and Laplacian (n, 1) tables, written out
    (grad chi = -lam^2 r chi, lap chi = (lam^4 r^2 - 3 lam^2) chi), the
    density matrix
    [[occupation]] and the density, occupation lam^3 exp(-lam^2 r^2): the
    density lam^3 rho_1(lam r) of rho_1 = exp(-r^2) scaled uniformly."""
    pts = np.asarray(points, dtype=float)
    r2 = np.sum(pts * pts, axis=1)
    chi = lam ** 1.5 * np.exp(-0.5 * lam ** 2 * r2)
    grad = -(lam ** 2) * pts.T[:, :, None] * chi[None, :, None]
    lapl = (lam ** 4 * r2 - 3.0 * lam ** 2) * chi
    p = np.array([[occupation]])
    return (jnp.asarray(chi[:, None]), jnp.asarray(grad),
            jnp.asarray(lapl[:, None]), jnp.asarray(p),
            jnp.asarray(occupation * chi ** 2), r2)


def _q_closed(r2):
    """q of rho = exp(-r^2): lap rho = (4 r^2 - 6) exp(-r^2), so
    q = (4 r^2 - 6) exp(2 r^2 / 3) / (4 (3 pi^2)^(2/3))."""
    return (4.0 * r2 - 6.0) * np.exp(2.0 * r2 / 3.0) / _CF


def _points(n, r_max, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u * rng.uniform(0.0, r_max, n)[:, None]


def _lap():
    from xcquinox.pipeline.descriptors import make_descriptor
    return make_descriptor("lap")


def _model(name, polarized=True, seed=0):
    arch = dataclasses.replace(C.get_architecture(name),
                               use_polarized_correlation=polarized)
    xnet, cnet = N.create_network_pair(arch, seed=seed)
    return pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _model_record(model, system, basis="sto-3g", grid_level=1):
    return _record(system, basis, grid_level, model.descriptors, ("eri",))


def _rks_features_of(model, md):
    """The RKS P -> features map, as the manual solver builds it."""
    from xcquinox.pipeline.solver import (_contract_dm_to_grid_with_nabla,
                                          _reassemble_features)
    ao = jnp.asarray(md["ao_grid"])
    deriv = jnp.asarray(md["ao_grid_deriv"])
    lapl = jnp.asarray(md["ao_grid_lapl"])
    n = int(np.asarray(md["grid_weights"]).shape[0])

    def features_of(D):
        rho, _nab, sigma = _contract_dm_to_grid_with_nabla(D, deriv)
        return _reassemble_features(
            descriptors=model.descriptors, dm=D,
            s_matrix=jnp.asarray(md["s_matrix"]),
            cusp_features=md.get("cusp_features"), n_grid=n,
            rung35_proj_ao=md.get("rung35_proj_ao"),
            rung35ms_proj_ao=md.get("rung35ms_proj_ao"),
            ao_grad=deriv[1:4], rho=rho, sigma=sigma, ao=ao, ao_lapl=lapl)
    return features_of


def _uks_features_of(model, md):
    """The three UKS P -> block maps, as the manual solver builds them."""
    from xcquinox.pipeline.solver import make_uks_feature_fns
    return make_uks_feature_fns(
        descriptors=model.descriptors,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"),
        ao_lapl=jnp.asarray(md["ao_grid_lapl"]))


def _dash(style):
    """A line style as the dash pattern matplotlib draws."""
    from matplotlib import lines
    off, dashes = lines._get_dash_pattern(style)
    return (float(off), None if dashes is None else tuple(float(x) for x in dashes))


def _legacy(path, tmp_path):
    """A file of the generation before the column, built from the file at
    ``path``: the same arrays without ``lap_all`` and ``lap_x`` (the key set
    the generator wrote before the column), and its manifest without the
    stems and the column's definition (the keys the manifest gained)."""
    dst = tmp_path / "legacy"
    dst.mkdir()
    out = str(dst / os.path.basename(path))
    arrays = {k: v for k, v in _arrays(path).items() if k not in ("lap_all", "lap_x")}
    np.savez(out, **arrays)
    meta = {k: v for k, v in pdg.read_pretrain_manifest(path).items()
            if k not in ("descriptor_stems", "lap_definition")}
    Path(pdg._pretrain_manifest_path(out)).write_text(json.dumps(meta))
    return out


def _arrays(path):
    with np.load(path) as z:
        return {k: np.array(z[k]) for k in z.files}


def _datagen_oracle(scale, basis="sto-3g", grid_level=0):
    """The column of the He/Li file computed here: per system, the parent
    (PBE) density matrix of the generator's own precompute call, the grid
    rebuilt as the generator rebuilds it, pyscf's MGGA rows, tanh(q / scale)
    on the points above the generator's floor. The total block for every
    system; the exchange block per channel at diag(P_s, P_s) (alpha, then
    beta) for an open shell and the total rows for a closed one. Returns
    {system index: (rho, column)} for each block; its rows are the file's
    own to 3.0e-15 relative, row for row."""
    systems = pdg.resolve_pretrain_systems(atoms=_ATOMS)
    out_all, out_x = {}, {}
    for i, system in enumerate(systems):
        md = precompute_fixed_density_data(
            pdg._mol_spec_for(system, basis, grid_level), required_keys=(),
            descriptors=(), reference_xc="pbe",
            orientation_lock_strength=float(pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH))
        mol = gto.M(atom=system.atom, basis=basis, charge=int(system.charge),
                    spin=int(system.spin), ecp=C.mole_ecp(basis, system.atom),
                    verbose=0)
        mf = dft.UKS(mol) if system.spin else dft.RKS(mol)
        pin_small_rho_cutoff(mf)
        mf.grids.level = grid_level
        mf.initialize_grids(mol, mf.get_init_guess(mol, mf.init_guess,
                                                   s1e=mf.get_ovlp(mol)))
        coords = np.asarray(mf.grids.coords)
        assert np.array_equal(np.asarray(mf.grids.weights),
                              np.asarray(md["grid_weights"]))
        dm = np.asarray(md["dm_pbe"])
        rho, lap, ao2 = _pyscf(mol, coords, _total(dm))
        assert float(np.abs(ao2[0] - np.asarray(md["ao_grid"])).max()) < 1e-10
        keep = rho > pdg._RHO_FLOOR
        out_all[i] = (rho[keep], np.tanh(_q(rho, lap) / scale)[keep])
        if dm.ndim == 3:
            rows, cols = [], []
            for s in (0, 1):
                r = dft.numint.NumInt().eval_rho(mol, ao2, 2.0 * dm[s],
                                                 xctype="MGGA", with_lapl=True)
                k = r[0] > pdg._RHO_FLOOR
                rows.append(r[0][k])
                cols.append(np.tanh(_q(r[0], r[4]) / scale)[k])
            out_x[i] = (np.concatenate(rows), np.concatenate(cols))
        else:
            out_x[i] = out_all[i]
    return out_all, out_x


@pytest.fixture(scope="module")
def lap_file(tmp_path_factory):
    """The He/Li file (sto-3g, grid level 0, polarized, descriptors, the
    spin_channel footing) written by the generator as it stands, and its
    arrays."""
    d = tmp_path_factory.mktemp("a5_lap_file")
    path = pdg.generate_pretrain_data_npz(
        str(d), atoms=_ATOMS, basis="sto-3g", grid_level=0, polarized=True,
        descriptors=True, exchange_footing="spin_channel")
    return path, _arrays(path)


# ---------------------------------------------------------------------------
# 1. The kernel
# ---------------------------------------------------------------------------

def test_the_kernel_is_the_density_laplacian():
    """The AO Laplacian table (data.ao_laplacian_on_grid) and the density
    Laplacian from the density matrix (descriptors.lap_rho_from_dm) on OH
    (UKS) and H2O (RKS), def2-svp, grid level 1, the records' own PBE
    densities and grids.

    The table: components 4 + 7 + 9 of pyscf's eval_ao(deriv=2) to 1e-12 of
    its largest element, at block sizes 997, the default and the whole grid
    alike (eval_ao on 997-point chunks differs from one call on the whole
    grid by 2.6e-14, not bitwise); against a central difference of the AO
    values (h = 1e-4, 300 points) to 1e-4 (|lap chi| + 1), measured 8.5e-6
    against 1.6e4 for the components 4 + 5 + 6.

    The density Laplacian: pyscf's eval_rho(xctype="MGGA", with_lapl=True)
    row 4 on the points rho > 1e-10, to 1e-10 relative plus 1e-13 of
    max|lap rho|; measured 2.3e-9 (OH) and 3.0e-9 (H2O) absolute against
    max|lap rho| of 1.17e6 and 1.16e6 (2.0e-15 and 2.6e-15 of it) and
    7.9e-13 relative per point at worst. A spin-resolved
    (2, n, n) matrix is summed: the result of its total, to 1e-12 of
    max|lap rho|. Dropping the 4 tau term moves the Laplacian by 2.8e-2 of
    max|lap rho|, the factor 2 by 0.50, the components 4 + 5 + 6 by 0.67."""
    from xcquinox.pipeline.data import ao_laplacian_on_grid
    from xcquinox.pipeline.descriptors import lap_rho_from_dm
    for system in (_OH, _H2O):
        md = _record(system, "def2-svp", 1)
        mol, coords = _grid(system, "def2-svp", 1, md)
        n = coords.shape[0]
        ao2 = dft.numint.eval_ao(mol, coords, deriv=2)
        ref = ao2[4] + ao2[7] + ao2[9]
        top_t = float(np.abs(ref).max())
        table = np.asarray(ao_laplacian_on_grid(mol, coords, block=997))
        assert table.shape == ref.shape == (n, mol.nao), (table.shape, ref.shape)
        assert float(np.abs(table - ref).max()) <= 1e-12 * top_t
        for other in (ao_laplacian_on_grid(mol, coords),
                      ao_laplacian_on_grid(mol, coords, block=n)):
            assert float(np.abs(np.asarray(other) - ref).max()) <= 1e-12 * top_t
        pick = np.random.default_rng(20261007).choice(n, 300, replace=False)
        h = 1e-4
        fd = np.zeros((300, mol.nao))
        for e in np.eye(3) * h:
            fd += (dft.numint.eval_ao(mol, coords[pick] + e, deriv=0)
                   - 2.0 * ao2[0][pick]
                   + dft.numint.eval_ao(mol, coords[pick] - e, deriv=0)) / h ** 2
        assert np.all(np.abs(table[pick] - fd) <= 1e-4 * (np.abs(table[pick]) + 1.0))

        dm = np.asarray(md["dm_pbe"])
        rho, lap_ref, _ao2 = _pyscf(mol, coords, _total(dm))
        valid = rho > _VALID
        top = float(np.abs(lap_ref).max())
        args = (jnp.asarray(ao2[0]), jnp.asarray(ao2[1:4]), jnp.asarray(table))
        lap = np.asarray(lap_rho_from_dm(*args, jnp.asarray(dm))).reshape(-1)
        assert lap.size == n
        dev = np.abs(lap - lap_ref)[valid]
        assert np.all(dev <= 1e-10 * np.abs(lap_ref[valid]) + 1e-13 * top), (
            system[0], float(dev.max()))
        if dm.ndim == 3:
            summed = np.asarray(lap_rho_from_dm(*args, jnp.asarray(_total(dm))))
            assert float(np.abs(summed.reshape(-1) - lap).max()) <= 1e-12 * top


# ---------------------------------------------------------------------------
# 2. The column
# ---------------------------------------------------------------------------

def test_the_column_is_tanh_of_the_reduced_laplacian():
    """The registered descriptor "lap": one feature, its precompute key and
    its two per-channel keys, density-matrix dependent with a
    compute_from_dm (which switches the feature-response term on), bounds
    (-1, 1) (range 2), and a positive finite scale read from the instance.

    The analytic Gaussian (one s function, P = [[1]], rho = exp(-r^2) on 400
    points with r <= 2.2): lap rho = (4 r^2 - 6) exp(-r^2) to 1e-14 relative
    plus 1e-15, and the column tanh(q / scale) with q = (4 r^2 - 6)
    exp(2 r^2 / 3) / (4 (3 pi^2)^(2/3)) to 1e-13 (closed forms); strictly
    inside (-1, 1) there (|q| <= 8.8); 0 to 1e-14 at r^2 = 3/2, where q = 0;
    within [-1, 1] at r up to 4, where tanh saturates (q = 6.5e4 at r = 4).

    The records (OH and H2O, def2-svp, grid level 1, pyscf's AO tables): the
    column is tanh(compute_reduced_laplacian(rho, lap_rho_from_dm) / scale)
    to 1e-14 everywhere, finite and within [-1, 1], and tanh(q / scale) of
    pyscf's Laplacian to 1e-12 on rho > 1e-10."""
    from xcquinox.pipeline.descriptors import (DESCRIPTOR_REGISTRY,
                                               LaplacianDescriptor,
                                               lap_rho_from_dm)
    d = _lap()
    assert type(d) is LaplacianDescriptor
    assert DESCRIPTOR_REGISTRY["lap"] is LaplacianDescriptor
    assert d.n_features == 1
    assert LaplacianDescriptor.required_mol_keys == ("lap_features",)
    assert LaplacianDescriptor.spin_mol_keys == ("lap_features_a",
                                                 "lap_features_b")
    assert LaplacianDescriptor.density_matrix_dependent is True
    assert hasattr(LaplacianDescriptor, "compute_from_dm")
    assert tuple(tuple(float(v) for v in b) for b in d.column_bounds) == ((-1.0, 1.0),)
    assert tuple(d.column_ranges) == (2.0,)
    scale = float(d.scale)
    assert math.isfinite(scale) and scale > 0.0

    pts = np.concatenate([_points(400, 2.2, 7), [[math.sqrt(1.5), 0.0, 0.0]]])
    ao, grad, lapl, p, rho, r2 = _gaussian(pts)
    lap = np.asarray(lap_rho_from_dm(ao, grad, lapl, p)).reshape(-1)
    want = (4.0 * r2 - 6.0) * np.exp(-r2)
    assert np.all(np.abs(lap - want) <= 1e-14 * np.abs(want) + 1e-15)
    col = np.asarray(d.compute_from_dm(ao, grad, lapl, rho, p))
    assert col.shape == (pts.shape[0], 1)
    assert float(np.abs(col[:, 0] - np.tanh(_q_closed(r2) / scale)).max()) <= 1e-13
    assert float(np.abs(col[:, 0]).max()) < 1.0
    assert abs(float(col[-1, 0])) <= 1e-14
    far = _points(200, 4.0, 8)
    ao, grad, lapl, p, rho, r2 = _gaussian(far)
    col = np.asarray(d.compute_from_dm(ao, grad, lapl, rho, p))[:, 0]
    assert np.all(np.isfinite(col)) and np.all(np.abs(col) <= 1.0)
    assert float(np.abs(col - np.tanh(_q_closed(r2) / scale)).max()) <= 1e-13

    for system in (_OH, _H2O):
        md = _record(system, "def2-svp", 1)
        mol, coords = _grid(system, "def2-svp", 1, md)
        dm = np.asarray(md["dm_pbe"])
        rho_ref, lap_ref, ao2 = _pyscf(mol, coords, _total(dm))
        tables = (jnp.asarray(ao2[0]), jnp.asarray(ao2[1:4]),
                  jnp.asarray(ao2[4] + ao2[7] + ao2[9]))
        rho = jnp.asarray(md["rho_grid"])
        col = np.asarray(d.compute_from_dm(*tables, rho, jnp.asarray(dm)))
        assert col.shape == (coords.shape[0], 1)
        lap = lap_rho_from_dm(*tables, jnp.asarray(dm))
        composed = np.tanh(np.asarray(compute_reduced_laplacian(
            rho, jnp.asarray(lap).reshape(-1))) / scale)
        assert float(np.abs(col[:, 0] - composed).max()) <= 1e-14
        assert np.all(np.isfinite(col)) and np.all(np.abs(col) <= 1.0)
        valid = rho_ref > _VALID
        oracle = np.tanh(_q(rho_ref, lap_ref) / scale)
        assert float(np.abs(col[valid, 0] - oracle[valid]).max()) <= _COL_TOL


# ---------------------------------------------------------------------------
# 3. Scaling: coordinates and spin
# ---------------------------------------------------------------------------

def test_the_column_under_coordinate_scaling_and_spin_doubling():
    """q is invariant under uniform coordinate scaling, lambda^3 rho(lambda r)
    at r against rho at lambda r: the analytic Gaussian at lambda = 0.5, 2,
    3.7 (the kernel on the scaled tables, 300 points with lambda r <= 2.2)
    against the unscaled column and the closed form, to 1e-13.

    The doubled channel: q[2 rho] = 2^(-2/3) q[rho]. On the Gaussian,
    P = [[2]] gives tanh(2^(-2/3) q / scale) to 1e-13. On OH (def2-svp,
    grid level 1) the record's lap_features_a / _b are tanh(q / scale) of
    pyscf's density and Laplacian of diag(P_s, P_s) to 1e-12 on every point
    (measured 2.4e-15 and 2.9e-15), the same as tanh(2^(-2/3) q_s / scale)
    of the channel's own density wherever rho_s > 1e-12 (4.9e-16 relative
    in q); each differs from the total block by more than 0.1 (0.25 and 0.55),
    which a channel block built from the total density would not.

    A closed shell: H2O's record stores no per-channel blocks (None, as every
    density-matrix descriptor's), and the live UKS maps at P = [D/2, D/2]
    return the total block for both channels to 1e-14."""
    d = _lap()
    scale = float(d.scale)
    base = _points(300, 2.2, 9)
    ao, grad, lapl, p, rho, r2 = _gaussian(base)
    ref = np.asarray(d.compute_from_dm(ao, grad, lapl, rho, p))[:, 0]
    assert float(np.abs(ref - np.tanh(_q_closed(r2) / scale)).max()) <= 1e-13
    for lam in (0.5, 2.0, 3.7):
        ao, grad, lapl, p, rho, _r2 = _gaussian(base / lam, lam=lam)
        got = np.asarray(d.compute_from_dm(ao, grad, lapl, rho, p))[:, 0]
        assert float(np.abs(got - ref).max()) <= 1e-13, lam
    ao, grad, lapl, p, rho, r2 = _gaussian(base, occupation=2.0)
    got = np.asarray(d.compute_from_dm(ao, grad, lapl, rho, p))[:, 0]
    want = np.tanh(2.0 ** (-2.0 / 3.0) * _q_closed(r2) / scale)
    assert float(np.abs(got - want).max()) <= 1e-13

    md = _record(_OH, "def2-svp", 1, (d,))
    mol, coords = _grid(_OH, "def2-svp", 1, md)
    dm = np.asarray(md["dm_pbe"])
    total = np.asarray(md["lap_features"])[:, 0]
    for s, key in ((0, "lap_features_a"), (1, "lap_features_b")):
        block = np.asarray(md[key])
        assert block.shape == (coords.shape[0], 1), key
        rho_d, lap_d, _ao2 = _pyscf(mol, coords, 2.0 * dm[s])
        assert float(np.abs(block[:, 0] - np.tanh(_q(rho_d, lap_d) / scale)).max()) \
            <= _COL_TOL, key
        rho_s, lap_s, _ao2 = _pyscf(mol, coords, dm[s])
        above = rho_s > 1e-12
        scaled = np.tanh(2.0 ** (-2.0 / 3.0) * _q(rho_s, lap_s) / scale)
        assert float(np.abs(block[above, 0] - scaled[above]).max()) <= _COL_TOL, key
        assert float(np.abs(block[:, 0] - total).max()) > 0.1, key

    md = _record(_H2O, "def2-svp", 1, (d,))
    assert md["lap_features_a"] is None and md["lap_features_b"] is None
    model = SimpleNamespace(descriptors=(d,))
    fa, fb, ft = _uks_features_of(model, md)
    half = 0.5 * jnp.asarray(md["dm_pbe"])
    P = jnp.stack([half, half])
    tot = np.asarray(md["lap_features"])
    for block in (fa(P), fb(P), ft(P)):
        assert float(np.abs(np.asarray(block) - tot).max()) <= 1e-14


# ---------------------------------------------------------------------------
# 4. The precompute record
# ---------------------------------------------------------------------------

def test_the_precompute_record_carries_the_column():
    """The record of OH (def2-svp, grid level 1) with the descriptor: the AO
    Laplacian table ao_grid_lapl (n_grid, n_ao), pyscf's components
    4 + 7 + 9 on the rebuilt grid to 1e-12 of its largest element; the
    column lap_features (n_grid, 1) the kernel on the record's own tables,
    density and matrix to 1e-13 and pyscf's column to 1e-12 on rho > 1e-10;
    lap_features_a / _b the kernel on diag(P_s, P_s) with 2 rho_s to 1e-12.
    Without the descriptor the record carries neither the table nor the
    blocks (None). The record type declares the four keys.

    The padded record (padding.canonicalize_mol_data to n_ao + 3 and
    n_grid + 17): the table is (n_grid + 17, n_ao + 3) with the real block
    unchanged, the padded AO columns zero and the padded rows the last real
    row; the three blocks carry n_grid + 17 rows, the real rows unchanged;
    and the live per-channel and total maps on the padded record reproduce
    the unpadded blocks on the real rows to 1e-12.

    The instance's scale is the column's: a record built with half the
    default scale, after one at the default in the same process (the
    precompute memo enabled), carries tanh(2 q / scale) of pyscf's
    Laplacian to 1e-12, and so does the live map of that instance; the two
    records' columns differ by more than 0.1 (0.30)."""
    from xcquinox.pipeline import data as D
    from xcquinox.pipeline.descriptors import (LaplacianDescriptor,
                                               doubled_spin_dm, make_descriptor)
    from xcquinox.pipeline.padding import PadTarget, canonicalize_mol_data
    from xcquinox.pipeline.solver import make_uks_feature_fns
    d = _lap()
    assert type(d) is LaplacianDescriptor
    scale = float(d.scale)
    for key in ("ao_grid_lapl", "lap_features", "lap_features_a",
                "lap_features_b"):
        assert key in D.MoleculeData.__annotations__, key
    md = _record(_OH, "def2-svp", 1, (d,))
    mol, coords = _grid(_OH, "def2-svp", 1, md)
    n, nao = coords.shape[0], mol.nao
    ao2 = dft.numint.eval_ao(mol, coords, deriv=2)
    ref = ao2[4] + ao2[7] + ao2[9]
    table = np.asarray(md["ao_grid_lapl"])
    assert table.shape == (n, nao)
    assert float(np.abs(table - ref).max()) <= 1e-12 * float(np.abs(ref).max())
    ao = jnp.asarray(md["ao_grid"])
    grad = jnp.asarray(md["ao_grid_deriv"])[1:4]
    dm = jnp.asarray(md["dm_pbe"])
    col = np.asarray(md["lap_features"])
    assert col.shape == (n, 1)
    kernel = np.asarray(d.compute_from_dm(
        ao, grad, jnp.asarray(table), jnp.asarray(md["rho_grid"]), dm))
    assert float(np.abs(col - kernel).max()) <= 1e-13
    rho_ref, lap_ref, _ao2 = _pyscf(mol, coords, _total(dm))
    valid = rho_ref > _VALID
    assert float(np.abs(col[valid, 0] - np.tanh(_q(rho_ref, lap_ref) / scale)[valid]).max()) \
        <= _COL_TOL
    for s, key in ((0, "lap_features_a"), (1, "lap_features_b")):
        rho_s = jnp.einsum("gi,ij,gj->g", ao, dm[s], ao)
        want = np.asarray(d.compute_from_dm(
            ao, grad, jnp.asarray(table), 2.0 * rho_s, doubled_spin_dm(dm, s)))
        assert np.asarray(md[key]).shape == (n, 1), key
        assert float(np.abs(np.asarray(md[key]) - want).max()) <= 1e-12, key

    bare = _record(_OH, "def2-svp", 1)
    for key in ("ao_grid_lapl", "lap_features", "lap_features_a",
                "lap_features_b"):
        assert bare.get(key) is None, key
    cusp_only = _record(_OH, "def2-svp", 1, (make_descriptor("cusp"),))
    assert cusp_only.get("ao_grid_lapl") is None

    target = PadTarget(n_ao=nao + 3, n_grid=n + 17, naux=None)
    padded = canonicalize_mol_data(md, target)
    pt = np.asarray(padded["ao_grid_lapl"])
    assert pt.shape == (n + 17, nao + 3)
    np.testing.assert_array_equal(pt[:n, :nao], table)
    assert not np.any(pt[:, nao:])
    np.testing.assert_array_equal(pt[n:, :nao], np.repeat(table[-1:], 17, axis=0))
    for key in ("lap_features", "lap_features_a", "lap_features_b"):
        block = np.asarray(padded[key])
        assert block.shape == (n + 17, 1), key
        np.testing.assert_array_equal(block[:n], np.asarray(md[key]))
        assert np.all(np.isfinite(block)), key
    fa, fb, ft = make_uks_feature_fns(
        descriptors=(d,), ao_deriv=jnp.asarray(padded["ao_grid_deriv"]),
        s_matrix=jnp.asarray(padded["s_matrix"]), n_grid=n + 17,
        ao_lapl=jnp.asarray(padded["ao_grid_lapl"]))
    P = jnp.asarray(padded["dm_pbe"])
    for fn, key in ((fa, "lap_features_a"), (fb, "lap_features_b"),
                    (ft, "lap_features")):
        live = np.asarray(fn(P))
        assert live.shape == (n + 17, 1), key
        assert float(np.abs(live[:n] - np.asarray(md[key])).max()) <= 1e-12, key

    D.set_precompute_cache_enabled(True)
    D.clear_precompute_cache()
    _record(_OH, "def2-svp", 1, (d,))
    half = make_descriptor("lap", scale=0.5 * scale)
    md_half = _record(_OH, "def2-svp", 1, (half,))
    # the scale enters the precompute cache key, so this call re-runs the PBE
    # SCF, and a second convergence under threaded BLAS need not land on the
    # first record's iterate (rho differing at the 1e-2 level at four
    # threads, run to run); the oracle is this record's own reference, which
    # its column matches at the 1e-15 level
    mol_half, coords_half = _grid(_OH, "def2-svp", 1, md_half)
    rho_half, lap_half, _ao2 = _pyscf(mol_half, coords_half,
                                      _total(jnp.asarray(md_half["dm_pbe"])))
    valid_half = rho_half > _VALID
    oracle = np.tanh(2.0 * _q(rho_half, lap_half) / scale)
    col_half = np.asarray(md_half["lap_features"])[:, 0]
    assert float(np.abs(col_half[valid_half] - oracle[valid_half]).max()) \
        <= _COL_TOL
    assert float(np.abs(col_half - col[:, 0]).max()) > 0.1
    model = SimpleNamespace(descriptors=(half,))
    _fa, _fb, ft_half = _uks_features_of(model, md_half)
    live = np.asarray(ft_half(jnp.asarray(md_half["dm_pbe"])))[:, 0]
    assert float(np.abs(live[valid_half] - oracle[valid_half]).max()) \
        <= _COL_TOL


# ---------------------------------------------------------------------------
# 5. The live solver
# ---------------------------------------------------------------------------

def test_the_live_solver_fock_is_the_derivative_of_its_energy(monkeypatch):
    """deep_lap_3x16 (polarized correlation, seed 0) in the manual solver on
    H2O (RKS) and OH (UKS), sto-3g, grid level 1.

    Three cycles of FULL / REASSEMBLE: finite energies, and the features the
    solver returns are the column of its final density matrix (1e-12).

    One cycle at mixing 1 returns the aufbau matrix of the first Fock; that
    step rebuilt from the public pieces (the solver's own feature maps,
    compute_vxc_nn and, on the open shell, the polarized correlation
    potential, the feature-response term, solver_manual's Coulomb matrix and
    diagonalization) reproduces it to 1e-12, and the same rebuild WITHOUT the
    feature-response term does not (more than 1e3 times further): 0.0
    against 1.4e-4 for deep_lap_3x16, 0.0 against 6.7e-4 to 2.2e-3 for the
    density-matrix columns of deep_rung35only_3x16 and deep_mgga_3x16.

    RKS: V_xc (analytic plus feature response) equals sym(jax.grad E_xc(P))
    elementwise to 1e-8 of max|grad| (test_training_gradient_consistency's
    bound; measured 1.5e-12, 3.0e-5 without the term), the analytic part
    alone more than 100 times further; and a central difference along a
    random symmetric direction (eps 1e-5) to 1e-6 relative (measured
    1.9e-11). UKS: test_solv01_split_xc's O2 harness (rotation path,
    straddle mask, its _TOL_UKS = 5e-7) with maps that carry the Laplacian
    table (measured 2.8e-9, no point masked).

    FROZEN against REASSEMBLE (FIXED_J, three cycles; FULL refuses FROZEN):
    on the open shell, where FROZEN freezes the descriptor blocks only, the
    control deep_3x16 gives the same energy trace bit for bit and
    deep_lap_3x16 a different one (by more than 1e-8 Ha; measured 2.2e-5 Ha,
    against 7.9e-4 and 1.6e-3 Ha for the density-matrix columns of the
    tree). On a closed shell FROZEN freezes rho and sigma as well (deep_3x16
    moves by 1.9e-2 Ha there), so the two policies differ by more than the
    column and the comparison is made on the open shell."""
    from xcquinox.pipeline import solver_manual as SM
    from xcquinox.pipeline.oneshot import (
        compute_vc_polarized_per_spin, compute_vxc_nn, feature_energy_derivative,
        feature_response_vxc, has_dm_dependent_descriptor, uks_zeta)
    from xcquinox.pipeline.solver import (FeaturePolicy, SolverBackend,
                                          SolverConfig, SolverMode,
                                          _contract_dm_to_grid_with_nabla,
                                          run_scf)
    model = _model("deep_lap_3x16")
    assert has_dm_dependent_descriptor(model)
    cfg3 = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=3, conv_tol=1e-12,
                        feature_policy=FeaturePolicy.REASSEMBLE)
    cfg1 = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=1, conv_tol=1e-12,
                        feature_policy=FeaturePolicy.REASSEMBLE,
                        mixer_kwargs=(("alpha", 1.0),))

    # --- RKS: H2O
    md = _model_record(model, _H2O)
    features_of = _rks_features_of(model, md)
    ao = jnp.asarray(md["ao_grid"])
    deriv = jnp.asarray(md["ao_grid_deriv"])
    w = jnp.asarray(md["grid_weights"])
    res = run_scf(cfg3, model, md, forward_only=True)
    assert np.all(np.isfinite(np.asarray(res.energy_trace)))
    assert np.isfinite(float(res.total_energy))
    assert float(np.abs(np.asarray(res.features_used)
                        - np.asarray(features_of(res.density_matrix))).max()) <= 1e-12

    def energy(D):
        rho, _nab, sigma = _contract_dm_to_grid_with_nabla(D, deriv)
        return jnp.sum(w * model.eval_exc(rho, sigma, features_of(D)))

    def potential(D, response=True):
        rho, nab, sigma = _contract_dm_to_grid_with_nabla(D, deriv)
        f = features_of(D)
        v = compute_vxc_nn(model, rho, sigma, f, ao, w, nabla_rho=nab,
                           ao_grad=deriv)
        if response:
            v = v + feature_response_vxc(
                feature_energy_derivative(model, rho, sigma, f), w,
                features_of, D)
        return v

    D0 = jnp.asarray(md["dm_seed"])
    D1 = np.asarray(run_scf(cfg1, model, md, forward_only=True).density_matrix)
    J = SM._resolve_coulomb(cfg1, md)(D0)
    h, S = jnp.asarray(md["h_core"]), jnp.asarray(md["s_matrix"])
    with_t = np.asarray(SM._diagonalize_roothaan(h + J + potential(D0), S, md["nocc"]))
    without = np.asarray(SM._diagonalize_roothaan(h + J + potential(D0, False), S,
                                                   md["nocc"]))
    e_with, e_without = np.abs(D1 - with_t).max(), np.abs(D1 - without).max()
    assert e_with <= 1e-12, e_with
    assert e_without > 1e3 * max(e_with, 1e-15), (e_with, e_without)

    G = jax.grad(energy)(D0)
    G = 0.5 * (G + G.T)
    top = float(jnp.max(jnp.abs(G)))
    gap_with = float(jnp.max(jnp.abs(potential(D0) - G))) / top
    gap_without = float(jnp.max(jnp.abs(potential(D0, False) - G))) / top
    assert gap_with < 1e-8, gap_with
    assert gap_without > 100.0 * max(gap_with, 1e-12), (gap_with, gap_without)
    W = np.random.default_rng(20261007).standard_normal(D0.shape)
    W = jnp.asarray(0.5 * (W + W.T))
    eps = 1e-5
    fd = float((energy(D0 + eps * W) - energy(D0 - eps * W)) / (2.0 * eps))
    an = float(jnp.sum(potential(D0) * W))
    assert abs(fd - an) / max(abs(fd), abs(an)) < 1e-6, (fd, an)

    # --- UKS: OH
    md = _model_record(model, _OH)
    fa, fb, ft = _uks_features_of(model, md)
    res = run_scf(cfg3, model, md, forward_only=True)
    assert np.all(np.isfinite(np.asarray(res.energy_trace)))
    assert float(np.abs(np.asarray(res.features_used)
                        - np.asarray(ft(res.density_matrix))).max()) <= 1e-12
    ao = jnp.asarray(md["ao_grid"])
    deriv = jnp.asarray(md["ao_grid_deriv"])
    xyz = deriv[1:4]
    w = jnp.asarray(md["grid_weights"])

    def spin(D):
        rho = jnp.einsum("ij,gi,gj->g", D, ao, ao)
        nab = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, xyz, ao)
        return rho, nab, jnp.einsum("gd,gd->g", nab, nab)

    def potentials(P, response=True):
        ra, na, saa = spin(P[0])
        rb, nb, sbb = spin(P[1])
        nt = na + nb
        st = jnp.einsum("gd,gd->g", nt, nt)
        f_a, f_b, f_t = fa(P), fb(P), ft(P)
        va = compute_vxc_nn(model, 2 * ra, 4 * saa, f_a, ao, w,
                            nabla_rho=2 * na, ao_grad=deriv, part="x")
        vb = compute_vxc_nn(model, 2 * rb, 4 * sbb, f_b, ao, w,
                            nabla_rho=2 * nb, ao_grad=deriv, part="x")
        ca, cb = compute_vc_polarized_per_spin(model, ra, rb, st, f_t, ao, w,
                                               nt, deriv)
        va, vb = va + ca, vb + cb
        if response:
            v = feature_response_vxc(0.5 * feature_energy_derivative(
                model, 2 * ra, 4 * saa, f_a, part="x"), w, fa, P)
            v = v + feature_response_vxc(0.5 * feature_energy_derivative(
                model, 2 * rb, 4 * sbb, f_b, part="x"), w, fb, P)
            v = v + feature_response_vxc(feature_energy_derivative(
                model, ra + rb, st, f_t, part="c", zeta=uks_zeta(ra, rb)),
                w, ft, P)
            va, vb = va + v[0], vb + v[1]
        return va, vb

    D0 = jnp.asarray(md["dm_seed"])
    D1 = np.asarray(run_scf(cfg1, model, md, forward_only=True).density_matrix)
    coulomb = SM._resolve_coulomb(cfg1, md)
    jt = coulomb(D0[0]) + coulomb(D0[1])
    h, S = jnp.asarray(md["h_core"]), jnp.asarray(md["s_matrix"])
    rebuilt = []
    for response in (True, False):
        va, vb = potentials(D0, response)
        rebuilt.append(np.stack([
            np.asarray(SM._diagonalize_roothaan_unrestricted(h + jt + va, S, md["nocc_a"])),
            np.asarray(SM._diagonalize_roothaan_unrestricted(h + jt + vb, S, md["nocc_b"]))]))
    e_with, e_without = np.abs(D1 - rebuilt[0]).max(), np.abs(D1 - rebuilt[1]).max()
    assert e_with <= 1e-12, e_with
    assert e_without > 1e3 * max(e_with, 1e-15), (e_with, e_without)
    monkeypatch.setattr(_solv, "_live_uks_features_fns", _uks_features_of)
    _solv._assert_uks_fd_consistency(model, md, "deep_lap_3x16", "OH")

    traces = {}
    for name in ("deep_3x16", "deep_lap_3x16"):
        m = _model(name)
        rec = _model_record(m, _OH)
        for policy in (FeaturePolicy.FROZEN, FeaturePolicy.REASSEMBLE):
            cfg = SolverConfig(backend=SolverBackend.MANUAL,
                               mode=SolverMode.FIXED_J, max_cycles=3,
                               conv_tol=1e-12, feature_policy=policy)
            traces[name, policy] = np.asarray(
                run_scf(cfg, m, rec, forward_only=True).energy_trace)
    np.testing.assert_array_equal(traces["deep_3x16", FeaturePolicy.FROZEN],
                                  traces["deep_3x16", FeaturePolicy.REASSEMBLE])
    moved = np.abs(traces["deep_lap_3x16", FeaturePolicy.FROZEN]
                   - traces["deep_lap_3x16", FeaturePolicy.REASSEMBLE]).max()
    assert moved > 1e-8, moved


# ---------------------------------------------------------------------------
# 6. Size consistency
# ---------------------------------------------------------------------------

def test_the_column_is_size_consistent():
    """The one-shot energy (oneshot.fixed_density_total_energy) of
    deep_lap_3x16 (polarized correlation, seed 0), sto-3g, grid level 1, of
    two fragments 80 bohr apart against the fragments alone, for the
    identical pair Ne...Ne and the non-identical pair Ne...He, total and XC
    part, to 1e-8 Ha.

    The floor of the comparison is 6.1e-12 Ha (Ne...Ne) and 4.5e-12 Ha
    (Ne...He) for deep_3x16 and a local density-matrix column
    (deep_rung35only_3x16); the Laplacian column reaches 6.1e-12 / 4.4e-12,
    while the same column replaced by its system-wide rho*w mean (an
    intensive, non-local column) passes the identical pair at 6.3e-12 and
    fails the non-identical pair at 3.1e-5 Ha. The identical pair alone
    cannot tell the two apart, which is why both pairs are asserted."""
    from xcquinox.pipeline.oneshot import fixed_density_total_energy
    model = _model("deep_lap_3x16")
    r = 80.0 * 0.52917721092
    systems = {
        "Ne": ("Ne", "Ne 0 0 0", 0, (("Ne", 1),)),
        "He": ("He", "He 0 0 0", 0, (("He", 1),)),
        "NeNe": ("NeNe", f"Ne 0 0 0; Ne 0 0 {r:.6f}", 0, (("Ne", 2),)),
        "NeHe": ("NeHe", f"Ne 0 0 0; He 0 0 {r:.6f}", 0, (("He", 1), ("Ne", 1))),
    }
    energy, xc = {}, {}
    for key, system in systems.items():
        md = _record(system, "sto-3g", 1, model.descriptors)
        energy[key] = float(fixed_density_total_energy(model, md))
        xc[key] = energy[key] - float(md["E_non_xc"])
    for a, b, ab in (("Ne", "Ne", "NeNe"), ("Ne", "He", "NeHe")):
        assert abs(energy[ab] - energy[a] - energy[b]) <= 1e-8, (ab, energy)
        assert abs(xc[ab] - xc[a] - xc[b]) <= 1e-8, (ab, xc)


# ---------------------------------------------------------------------------
# 7. The datagen
# ---------------------------------------------------------------------------

def test_the_datagen_writes_the_column_and_the_manifest_its_stems(lap_file, tmp_path):
    """The He/Li file (sto-3g, grid level 0, polarized, descriptors, the
    spin_channel footing).

    Columns: lap_all and lap_x of width 1 beside the columns of the
    generation before (a file of that generation is the same file without
    the two, with a manifest without the stems); the schema reader accepts
    the file. lap_all is tanh(q / scale) of the parent density on the
    generator's rows, lap_x the same of diag(P_s, P_s) per channel (alpha
    then beta) for Li and of the total density for He, each from pyscf's
    MGGA rows on the rebuilt grid to 1e-12; the rows are the file's own (rho
    to 1e-12 relative; 3.0e-15 measured).

    The stem map: DESCRIPTOR_STEM_OF names a stem for every registered
    descriptor, "lap" -> "lap" and the historical five as the loader had
    them; "lap" is a descriptor stem of width 1.

    The manifest states the stems the file carries (descriptor_stems: the
    descriptor stems; none for a file written without descriptors). A
    manifest without the key reads through the file's keys as the four
    historical stems: the earlier file is current for a request of any of
    them or of none, stale for one naming "lap", and the schema reader still
    accepts it (a reader demanding every stem would refuse every such file).
    The new file is current for every subset of its stems, and a file whose
    manifest states another definition of the column is stale for a request
    naming the stem. ensure_pretrain_data takes the requested stems as
    descriptor_stems and passes them to that check (on_stale="refuse" raises
    for "lap" on the earlier file and returns it untouched for "cusp"); the
    datagen stage's call (datagen_call, which the preflight and the
    energy-weight probe also make) carries descriptor_stems with "lap" for a
    sweep with a Laplacian architecture and no "lap" for one without.

    The loader: _assemble_pretrain_descriptors gives deep_lap_3x16 the rows
    [rho, sigma, lap] (exchange block) and [rho, sigma, zeta, lap]
    (correlation), deep_lap_geom_3x16 the cusp pair before the column, and
    refuses the legacy file naming the stem."""
    path, got = lap_file
    assert "lap_all" in got and "lap_x" in got, sorted(got)
    assert "lap" in pdg._DESCRIPTOR_STEMS
    assert tuple(pdg._COLUMN_WIDTHS["lap"]) == (1,)
    from xcquinox.pipeline.descriptors import DESCRIPTOR_REGISTRY
    stem_of = pdg.DESCRIPTOR_STEM_OF
    assert stem_of["lap"] == "lap"
    assert set(DESCRIPTOR_REGISTRY) <= set(stem_of), sorted(set(DESCRIPTOR_REGISTRY) - set(stem_of))
    assert {k: stem_of[k] for k in _HISTORICAL_STEMS} == _HISTORICAL_STEMS

    legacy_path = _legacy(path, tmp_path)
    legacy = _arrays(legacy_path)
    assert set(got) == set(legacy) | {"lap_all", "lap_x"}
    loaded = pdg.load_pretrain_data_npz(path)
    assert set(loaded) == set(got)
    assert got["lap_all"].shape == (got["rho_all"].shape[0], 1)
    assert got["lap_x"].shape == (got["rho_x"].shape[0], 1)

    scale = float(_lap().scale)
    oracle_all, oracle_x = _datagen_oracle(scale)
    for block, oracle in (("all", oracle_all), ("x", oracle_x)):
        index = got[f"system_{block}"]
        for i, (rho, col) in oracle.items():
            rows = index == i
            np.testing.assert_allclose(got[f"rho_{block}"][rows], rho,
                                       rtol=1e-12, atol=0.0)
            dev = np.abs(got[f"lap_{block}"][rows, 0] - col)
            assert float(dev.max()) <= _COL_TOL, (block, i, float(dev.max()))

    meta = pdg.read_pretrain_manifest(path)
    assert set(meta["descriptor_stems"]) == set(pdg._DESCRIPTOR_STEMS)
    assert "descriptor_stems" not in pdg.read_pretrain_manifest(legacy_path)
    # the column's definition is part of the file's identity: a lap file
    # written under another definition is stale for a request naming the stem
    # and current for one that does not name it
    from xcquinox.pipeline.descriptors import LAP_DEFINITION
    assert meta["lap_definition"] == LAP_DEFINITION
    other_dir = tmp_path / "other"
    shutil.copytree(os.path.dirname(path), other_dir)
    other_path = str(other_dir / os.path.basename(path))
    other_manifest = Path(pdg._pretrain_manifest_path(other_path))
    other_manifest.write_text(json.dumps(dict(meta, lap_definition="other")))
    systems = pdg.resolve_pretrain_systems(atoms=_ATOMS)
    ident = dict(basis="sto-3g", grid_level=0, systems=systems,
                 exchange_footing="spin_channel")
    historical = ("cusp", "dm", "rung35", "rung35ms")
    for stems, want in (((), True), (("cusp",), True), (historical, True),
                        (("lap",), False), (("cusp", "lap"), False)):
        assert pdg.pretrain_data_is_current(legacy_path, descriptor_stems=stems,
                                            **ident) is want, stems
    for stems in ((), ("lap",), ("cusp", "lap"), tuple(pdg._DESCRIPTOR_STEMS)):
        assert pdg.pretrain_data_is_current(path, descriptor_stems=stems,
                                            **ident) is True, stems
    assert pdg.pretrain_data_is_current(other_path, descriptor_stems=("lap",),
                                        **ident) is False
    assert pdg.pretrain_data_is_current(other_path, descriptor_stems=("cusp",),
                                        **ident) is True
    assert set(pdg.load_pretrain_data_npz(legacy_path)) == set(legacy)

    legacy_dir = os.path.dirname(legacy_path)
    mtime = os.path.getmtime(legacy_path)
    kw = dict(atoms=_ATOMS, basis="sto-3g", grid_level=0, polarized=True,
              descriptors=True, exchange_footing="spin_channel",
              on_stale="refuse")
    with pytest.raises(pdg.PretrainDataStale):
        pdg.ensure_pretrain_data(legacy_dir, descriptor_stems=("lap",), **kw)
    assert pdg.ensure_pretrain_data(legacy_dir, descriptor_stems=("cusp",),
                                    **kw) == legacy_path
    assert os.path.getmtime(legacy_path) == mtime

    bare_dir = tmp_path / "bare"
    bare = pdg.generate_pretrain_data_npz(
        str(bare_dir), atoms=(("He", 0),), basis="sto-3g", grid_level=0,
        polarized=True, descriptors=False)
    assert list(pdg.read_pretrain_manifest(bare)["descriptor_stems"]) == []
    bare_ident = dict(basis="sto-3g", grid_level=0,
                      systems=pdg.resolve_pretrain_systems(atoms=(("He", 0),)))
    assert pdg.pretrain_data_is_current(bare, descriptor_stems=(), **bare_ident)
    assert not pdg.pretrain_data_is_current(bare, descriptor_stems=("cusp",),
                                            **bare_ident)

    from xcquinox.pipeline.cluster import _datagen

    def cfg(archs):
        pt = SimpleNamespace(data_dir="/d/pt", atoms=(), dfs_set=False,
                             pool_atoms=False, parent_density="auto",
                             exchange_footing="spin_channel", mesh_fraction=0.3)
        return SimpleNamespace(
            sweep=SimpleNamespace(arch=list(archs)),
            use_polarized_correlation=True, pretrain=pt,
            inputs=SimpleNamespace(basis="def2-svp", grid_level=3,
                                   density_fit=False, auxbasis=None,
                                   orientation_lock_strength=3e-5))

    with_lap = cfg(["deep_3x16", "deep_lap_3x16"])
    assert _datagen._required_data_specs(with_lap) == [(True, "pbe")]
    _dir, keywords = _datagen.datagen_call(with_lap, True, "pbe")
    assert "lap" in set(keywords["descriptor_stems"]), keywords
    _dir, keywords = _datagen.datagen_call(cfg(["deep_3x16", "deep_geom_3x16"]),
                                           True, "pbe")
    assert "lap" not in set(keywords.get("descriptor_stems", ())), keywords
    # the iso-orbital indicator is a core column every file carries, not a
    # requested stem: a meta-GGA sweep's request names only descriptor
    # stems, the file just written is current for it, and a request naming
    # the core column is refused as a caller's error rather than read as a
    # stale file
    _dir, keywords = _datagen.datagen_call(
        cfg(["deep_3x16", "deep_mgga_3x16", "deep_rung35_mgga_3x16"]), True, "pbe")
    assert set(keywords["descriptor_stems"]) == {"cusp", "rung35"}, keywords
    assert set(keywords["descriptor_stems"]) <= set(pdg._DESCRIPTOR_STEMS)
    assert pdg.pretrain_data_is_current(
        path, descriptor_stems=keywords["descriptor_stems"], **ident) is True
    with pytest.raises(ValueError, match="metagga"):
        pdg.pretrain_data_is_current(path, descriptor_stems=("metagga",), **ident)
    assert pdg.descriptor_stems_for([C.get_architecture("deep_mgga_3x16")]) == ()
    assert pdg.descriptor_stems_for(
        [C.get_architecture("deep_lap_geom_3x16")]) == ("cusp", "lap")
    # the definition carries the scale, so another scale is another identity
    assert f"{pdg._LAP_DEFINITION}".startswith("tanh(q / ") and \
        f"{float(_lap().scale):g}" in pdg._LAP_DEFINITION
    # a manifest without the stems key of a file without descriptors reads
    # through the file's keys as carrying none
    bare_meta = Path(pdg._pretrain_manifest_path(bare))
    kept = json.loads(bare_meta.read_text())
    kept.pop("descriptor_stems")
    bare_meta.write_text(json.dumps(kept))
    assert pdg.pretrain_data_is_current(bare, descriptor_stems=(), **bare_ident)
    assert not pdg.pretrain_data_is_current(bare, descriptor_stems=("cusp",),
                                            **bare_ident)
    # the board and the energy-weight probe state their architectures' stems
    # on their own data calls
    captured = {}

    def _capture(data_dir, **kw):
        captured.update(kw)
        return os.path.join(data_dir, "x.npz")

    board = _repo_module("tools/pretrain_board_local.py")
    real = pdg.ensure_pretrain_data
    pdg.ensure_pretrain_data = _capture
    try:
        board.ensure_board_data(str(tmp_path / "board"), archs=[
            C.get_architecture("deep_lap_3x16"), C.get_architecture("deep_3x16")])
        assert captured["descriptor_stems"] == ("lap",), captured
        captured.clear()
        probe = _repo_module("hpcjobs/probe_pretrain_energy_weight.py")
        probe.ensure_data(str(tmp_path / "probe"), polarized=True,
                          reference_xc="pbe", basis="sto-3g", grid_level=0,
                          lock_strength=0.0, smoke_atoms=_ATOMS,
                          archs=[C.get_architecture("deep_lap_geom_3x16")])
        assert captured["descriptor_stems"] == ("cusp", "lap"), captured
    finally:
        pdg.ensure_pretrain_data = real

    from xcquinox.pipeline.pretrain import _assemble_pretrain_descriptors
    data = {k: jnp.asarray(v) for k, v in got.items()}
    old = {k: jnp.asarray(v) for k, v in legacy.items()}
    for name, n_extra in (("deep_lap_3x16", 1), ("deep_lap_geom_3x16", 3)):
        arch = dataclasses.replace(C.get_architecture(name),
                                   use_polarized_correlation=True)
        x = np.asarray(_assemble_pretrain_descriptors(arch, data, suffix="_x"))
        c = np.asarray(_assemble_pretrain_descriptors(arch, data, for_cnet=True))
        assert x.shape == (got["rho_x"].shape[0], 2 + n_extra), name
        assert c.shape == (got["rho_all"].shape[0], 3 + n_extra), name
        np.testing.assert_array_equal(x[:, -1], got["lap_x"][:, 0])
        np.testing.assert_array_equal(c[:, -1], got["lap_all"][:, 0])
        np.testing.assert_array_equal(c[:, 2], got["zeta_all"])
        if n_extra == 3:
            np.testing.assert_array_equal(x[:, 2:4], got["cusp_x"])
        with pytest.raises(KeyError, match="lap"):
            _assemble_pretrain_descriptors(arch, old, suffix="_x")
    # a file carries the column at the module's scale only, so a descriptor at
    # another scale is refused by the loader naming the scale
    from xcquinox.pipeline.config import FeatureSpec
    for scale in (1.0, 4.0):
        other_scale = dataclasses.replace(
            C.get_architecture("deep_lap_3x16"), use_polarized_correlation=True,
            descriptors=(FeatureSpec.of(("lap", {"scale": scale})),))
        with pytest.raises(ValueError, match="scale"):
            _assemble_pretrain_descriptors(other_scale, data, suffix="_x")
    # run_pretrain holds a lap architecture to the file's statement of the
    # column's definition: the copy whose manifest states another
    # definition is refused naming it
    from xcquinox.pipeline.pretrain import run_pretrain
    arch = dataclasses.replace(C.get_architecture("deep_lap_3x16"),
                               use_polarized_correlation=True)
    with pytest.raises(ValueError, match="lap_definition|definition"):
        run_pretrain(C.PretrainSpec(arch=arch, data_dir=str(other_dir),
                                    checkpoint_dir=str(tmp_path / "other_pre"),
                                    n_steps=1, seed=0))
    assert not (tmp_path / "other_pre").exists() or not any(
        (tmp_path / "other_pre").glob("*.eqx"))


def test_the_datagen_column_is_aligned_where_rows_are_dropped(tmp_path):
    """The He/Li file at grid level 1 (sto-3g, polarized, descriptors, the
    spin_channel footing), whose total block DROPS rows below the generator's
    floor (the level-0 file drops none on the total block, so a column built
    on the valid rows' tables with the full grid's density, or on the first
    rows instead of the valid ones, passes there): at least one system drops
    rows, and lap_all and lap_x equal pyscf's column on the kept rows to
    1e-12, row for row."""
    path = pdg.generate_pretrain_data_npz(
        str(tmp_path / "pruned"), atoms=_ATOMS, basis="sto-3g", grid_level=1,
        polarized=True, descriptors=True, exchange_footing="spin_channel")
    got = _arrays(path)
    scale = float(_lap().scale)
    oracle_all, oracle_x = _datagen_oracle(scale, grid_level=1)
    systems = pdg.resolve_pretrain_systems(atoms=_ATOMS)
    dropped = 0
    for i, system in enumerate(systems):
        md = precompute_fixed_density_data(
            pdg._mol_spec_for(system, "sto-3g", 1), required_keys=(),
            descriptors=(), reference_xc="pbe",
            orientation_lock_strength=float(pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH))
        dropped += int(np.asarray(md["grid_weights"]).shape[0]) - int(
            (got["system_all"] == i).sum())
    assert dropped > 0, "the pruned file drops no row; choose another grid"
    for block, oracle in (("all", oracle_all), ("x", oracle_x)):
        index = got[f"system_{block}"]
        for i, (rho, col) in oracle.items():
            rows = index == i
            np.testing.assert_allclose(got[f"rho_{block}"][rows], rho,
                                       rtol=1e-12, atol=0.0)
            dev = np.abs(got[f"lap_{block}"][rows, 0] - col)
            assert float(dev.max()) <= _COL_TOL, (block, i, float(dev.max()))


# ---------------------------------------------------------------------------
# 8. The rung
# ---------------------------------------------------------------------------

def test_the_laplacian_rung(monkeypatch):
    """The ladder (the design fixes it): GGA < Laplacian < meta-GGA <
    rung-3.5 < rung-3.5+meta-GGA, ranks 0..4. The ingredients are three
    flags (meta-GGA indicator, rung-3.5 occupancy, the "lap" descriptor);
    the Laplacian ranks below both: a flag table of all eight cases, the
    two-argument call keeping its meaning. The two entries are on the
    Laplacian rung, deep_3x16 on GGA, a (registered here, hypothetical)
    metagga + lap architecture on meta-GGA, a rung35 + lap one on rung-3.5,
    a cusp + rung35 + metagga + lap one on rung-3.5+meta-GGA. The seed is
    PBE for the Laplacian rung under both policies and unchanged for every
    other entry; the pretraining parent is PBE.

    The figures: arch_style's rung of both names and of the token fallback
    ("deep_lap", "deep_lap_geom"), the legacy tokens unchanged, sort_by_rung
    placing the rung between GGA and meta-GGA; make_ablation_arch_figure's
    short tag "Lap" and a line style per rung, five distinct dash patterns;
    and plot_pretraining_curves' line style per rung, five distinct (its
    table is keyed by the rung names and falls back to the GGA rung's solid
    line for a rung it does not hold)."""
    from xcquinox.pipeline import rungs
    from xcquinox.pipeline.cluster import fidelity as fid
    from xcquinox.pipeline.parents import parent_for_arch
    LAP = rungs.RUNG_LAP
    assert LAP == "Laplacian"
    assert rungs.RUNG_ORDER == (rungs.RUNG_GGA, LAP, rungs.RUNG_MGGA,
                                rungs.RUNG_R35, rungs.RUNG_R35_MGGA)
    assert rungs.RUNG_RANK == {r: i for i, r in enumerate(rungs.RUNG_ORDER)}
    table = {(False, False, False): rungs.RUNG_GGA,
             (False, False, True): LAP,
             (True, False, False): rungs.RUNG_MGGA,
             (True, False, True): rungs.RUNG_MGGA,
             (False, True, False): rungs.RUNG_R35,
             (False, True, True): rungs.RUNG_R35,
             (True, True, False): rungs.RUNG_R35_MGGA,
             (True, True, True): rungs.RUNG_R35_MGGA}
    for (meta, r35, lap), want in table.items():
        assert rungs.rung_from_ingredients(meta, r35, lap) == want
        assert rungs.rung_from_ingredients(meta, r35, has_lap=lap) == want
        if not lap:
            assert rungs.rung_from_ingredients(meta, r35) == want
    for name in C.ARCHITECTURES:
        arch = C.get_architecture(name)
        names = {s.name for s in arch.descriptors}
        flags = (C.ArchitectureConfig.is_meta_gga(arch),
                 any(n.startswith("rung35") for n in names), "lap" in names)
        assert tuple(rungs.arch_ingredients(name)) == flags, name
        assert rungs.rung_of(name) == table[flags], name
        if name not in _NEW:
            assert rungs.seed_xc_for_arch(name) == ("scan" if flags[0] else "pbe")
            assert rungs.seed_xc_for_arch(name, "beyond_gga_scan") == (
                "scan" if flags[0] or flags[1] else "pbe"), name
    for name in _NEW:
        arch = C.get_architecture(name)
        assert rungs.rung_of(name) == LAP
        assert rungs.seed_xc_for_arch(name) == "pbe"
        assert rungs.seed_xc_for_arch(name, "beyond_gga_scan") == "pbe"
        assert pdg.resolve_parent_density(arch, "auto") == "pbe"
        assert parent_for_arch(arch) == "pbe" == fid.resolve_parent(name)
    assert rungs.rung_of("deep_3x16") == rungs.RUNG_GGA
    for key, descriptors, meta, want in (
            ("t_mgga_lap_3x16", ["metagga", "lap"], True, rungs.RUNG_MGGA),
            ("t_rung35_lap_3x16", ["rung35", "lap"], False, rungs.RUNG_R35),
            ("t_all_lap_3x16", ["cusp", "rung35", "metagga", "lap"], True,
             rungs.RUNG_R35_MGGA)):
        monkeypatch.setitem(C.ARCHITECTURES, key, C.ArchitectureConfig.from_spec(
            key, 3, 16, descriptors=descriptors, meta_gga=meta,
            dm_entropy_intensive=True, descriptor_log_transform=True))
        assert rungs.rung_of(key) == want, key
        assert AS.rung_of(key) == want, key

    for name in _NEW:
        assert AS.rung_of(name) == LAP
        assert AS.rung_of(AS.display_name(name)) == LAP
    assert AS.RUNG_ORDER == rungs.RUNG_ORDER
    for token, want in (("deep_lap", LAP), ("deep_lap_geom", LAP),
                        ("deep_mgga", rungs.RUNG_MGGA),
                        ("deep_rung35", rungs.RUNG_R35),
                        ("deep_rung35_mgga", rungs.RUNG_R35_MGGA),
                        ("deep", rungs.RUNG_GGA)):
        assert AS.rung_of(token) == want, token
    assert AS.sort_by_rung(["deep_mgga_3x16", "deep_lap_3x16", "deep_3x16"]) == [
        "deep_3x16", "deep_lap_3x16", "deep_mgga_3x16"]

    fig = _tool("make_ablation_arch_figure.py")
    assert fig._RUNG_SHORT[LAP] == "Lap"
    assert set(fig._RUNG_SHORT) == set(rungs.RUNG_ORDER)
    # the own-rung nonempirical reference: PBE for the Laplacian rung (its
    # parent), SCAN only for the orbital-dependent rungs
    assert fig.arch_reference_kinds(
        ["deep_lap_3x16", "deep_lap_geom_3x16", "deep_3x16", "deep_mgga_3x16",
         "deep_rung35_3x16"]) == {
        "deep_lap_3x16": "pbe", "deep_lap_geom_3x16": "pbe", "deep_3x16": "pbe",
        "deep_mgga_3x16": "scan", "deep_rung35_3x16": "scan"}
    for styles in (fig._RUNG_LS, _tool("plot_pretraining_curves.py")._RUNG_LS):
        assert set(rungs.RUNG_ORDER) <= set(styles), sorted(styles)
        patterns = [_dash(styles[r]) for r in rungs.RUNG_ORDER]
        assert len(set(patterns)) == len(rungs.RUNG_ORDER), patterns


# ---------------------------------------------------------------------------
# 9. The registry
# ---------------------------------------------------------------------------

def test_the_registry_entries_are_the_stated_ones():
    """The two entries (the design fixes their definitions):
    deep_lap_3x16 is deep_3x16 with descriptors ["lap"], deep_lap_geom_3x16
    is deep_geom_3x16 with ["cusp", "lap"] (dataclass equality, the cusp pair
    first); 43 entries; each its own shown name (tokens "lap", "lap_geom");
    the expanded key naming arch_names._DESCRIPTOR_TEXT["lap"] (and the cusp
    text for the geometric entry) and no meta-GGA; the materialized
    descriptors and their bounds ((-1, 1), after the cusp pair's (0, 1) and
    (-1, 1)); the networks build and return a finite F on rows across the
    column's range, in the registry's configuration and the campaign's
    (polarized, paper coordinates, x2 gate); the bounded column admits the
    Fourier map (m = 16) and the spline network (grid 5, order 3), both of
    which refuse the unbounded density-matrix statistics (the check is
    live).

    The figure layer: both orders and ARCH_ORDER carry the pair directly
    after deep_notransform_attn_3x16 and before deep_rung35_3x16; each name
    has its own colour, not the unknown-architecture grey and carried by no
    other architecture, rung accent or band; the Laplacian rung has an
    accent and a band of its own. The matrix reads the registry's 43
    names."""
    from xcquinox.pipeline import arch_names as AN
    from xcquinox.pipeline import rungs
    from xcquinox.pipeline.cluster import workflow_matrix as wm
    from xcquinox.pipeline.config import FeatureSpec
    from xcquinox.pipeline.descriptors import CuspDescriptor, LaplacianDescriptor
    for name, (base, descriptors) in _BASE.items():
        want = dataclasses.replace(C.ARCHITECTURES[base], name=name,
                                   descriptors=tuple(FeatureSpec.of(x) for x in descriptors))
        assert C.ARCHITECTURES[name] == want, name
    assert len(C.ARCHITECTURES) == 43
    text = AN._DESCRIPTOR_TEXT["lap"]
    assert isinstance(text, str) and text
    for name, tokens in (("deep_lap_3x16", "lap"), ("deep_lap_geom_3x16", "lap_geom")):
        assert AN.DISPLAY_NAME[name] == name == AN.display_name(name)
        assert AN._tokens(name) == tokens
        assert AN.stored_key(name) == name
        expanded = AN.expanded_key(name)
        assert text in expanded and "meta-GGA" not in expanded, expanded
        if "geom" in name:
            assert AN._DESCRIPTOR_TEXT["cusp"] in expanded
        assert C.ARCHITECTURES[name].describe()["display_name"] == name
    plain, geom = (C.ARCHITECTURES[n] for n in _NEW)
    (only,) = plain.materialize_descriptors()
    assert type(only) is LaplacianDescriptor
    cusp, lap = geom.materialize_descriptors()
    assert type(cusp) is CuspDescriptor and cusp.log_transform is True
    assert type(lap) is LaplacianDescriptor
    assert plain.extra_feature_bounds == ((-1.0, 1.0),)
    assert geom.extra_feature_bounds == ((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    assert (plain.n_extra_features, geom.n_extra_features) == (1, 3)

    rows = [(0.3, 0.05), (2.0, 1.5), (1e-3, 1e-7)]
    cols = (-0.47, 0.0, 0.99)
    cusp_cols = (0.2, -0.3)
    for arch in (plain, geom):
        for variant in (arch, dataclasses.replace(
                arch, use_polarized_correlation=True,
                descriptor_coordinates="paper", ueg_gate="x2")):
            xnet, cnet = N.create_network_pair(variant, seed=0)
            for (rho, sigma), c in zip(rows, cols):
                extras = ([*cusp_cols, c] if arch is geom else [c])
                fx = float(xnet(jnp.asarray([rho, sigma, *extras])))
                zeta = [0.1] if variant.use_polarized_correlation else []
                fc = float(cnet(jnp.asarray([rho, sigma, *zeta, *extras])))
                assert math.isfinite(fx) and math.isfinite(fc), (arch.name, rho)
        N.create_network_pair(dataclasses.replace(arch, fourier_features=16),
                              seed=0)
    N.create_network_pair(C.ArchitectureConfig.from_spec(
        "t_lap_kan_2x6", 2, 6, descriptors=["lap"], dm_entropy_intensive=True,
        descriptor_log_transform=True, network="kan", kan_grid=5, kan_order=3),
        seed=0)
    with pytest.raises(ValueError, match="not bounded"):
        dataclasses.replace(C.ARCHITECTURES["deep_dm_3x16"], fourier_features=16)
    with pytest.raises(ValueError, match="not bounded"):
        C.ArchitectureConfig.from_spec(
            "t_dm_kan_2x6", 2, 6, descriptors=["dm_statistics"],
            dm_entropy_intensive=True, descriptor_log_transform=True,
            network="kan", kan_grid=5, kan_order=3)

    lap_bases = {"deep_lap", "deep_lap_geom"}

    def own(key):
        return AS._SIZE_SUFFIX.sub("", AS.stored_key(key)) in lap_bases

    colours = {name: AS.arch_color(name).lower() for name in _NEW}
    accents = {v.lower() for v in AS.RUNG_ACCENT.values()}
    bands = {v.lower() for v in AS.RUNG_BAND.values()}
    others = {v.lower() for k, v in AS.ARCH_COLOR.items() if not own(k)}
    assert len(set(colours.values())) == 2, colours
    for name, colour in colours.items():
        assert colour != "#333333", name
        assert colour not in others | accents | bands, (name, colour)
    every = {v.lower() for v in AS.ARCH_COLOR.values()}
    accent = AS.RUNG_ACCENT[rungs.RUNG_LAP].lower()
    band = AS.RUNG_BAND[rungs.RUNG_LAP].lower()
    assert AS.rung_color(rungs.RUNG_LAP).lower() == accent
    other_accents = {v.lower() for k, v in AS.RUNG_ACCENT.items() if k != rungs.RUNG_LAP}
    other_bands = {v.lower() for k, v in AS.RUNG_BAND.items() if k != rungs.RUNG_LAP}
    assert accent not in every | other_accents | bands
    assert band not in every | accents | other_bands | {"#ffffff"}
    want = ("deep_notransform_attn_3x16", "deep_lap_3x16", "deep_lap_geom_3x16",
            "deep_rung35_3x16")
    for order in (AS._STORED_ORDER, AS._DISPLAY_ORDER, AS.ARCH_ORDER):
        i = order.index("deep_notransform_attn_3x16")
        assert tuple(order[i:i + 4]) == want, order[i:i + 4]

    archs = sorted(C.ARCHITECTURES)
    assert sorted(wm.ARCHITECTURES) == archs and len(archs) == 43


# ---------------------------------------------------------------------------
# 10. The certificate path
# ---------------------------------------------------------------------------

def test_the_certificate_path_runs_with_the_column(lap_file, tmp_path):
    """deep_lap_3x16 through the production writers and readers: a two-step
    pretraining on the He/Li file and the fidelity certificate on the tiny
    oracle set (the H atom and H2, def2-svp, grid level 1;
    test_parent_anchor's), against deep_3x16 through the same calls.

    The payload: the PBE parent, every system evaluated (no error, finite
    dE_xc), no model-class mismatch; the potential record beside it
    (fidelity_vxc.json) measured on every system with the feature-response
    term included, as it is for every density-matrix column. No new
    field: the pretraining metadata, the payload, mlp_class_of,
    model_class_of_arch and the class record a checkpoint of the
    architecture carries (checkpoint_class.class_record) have deep_3x16's
    key sets (the descriptor is part of the architecture identity
    already)."""
    from xcquinox.pipeline import checkpoint_class as K
    from xcquinox.pipeline.cluster import fidelity as fid
    from xcquinox.pipeline.cluster._pretrain import resolve_run_architecture
    from xcquinox.pipeline.cluster.grid_config import pretrain_checkpoint_dir
    from xcquinox.pipeline.pretrain import run_pretrain
    path, _got = lap_file
    data_dir = os.path.dirname(path)
    out = {}
    for name in ("deep_lap_3x16", "deep_3x16"):
        cfg = _pa._anchored_cfg(arch=(name,), parent_anchor=False)
        arch = resolve_run_architecture(cfg, C.get_architecture(name))
        run_dir = str(tmp_path / name)
        pre = pretrain_checkpoint_dir(run_dir, name)
        md = run_pretrain(C.PretrainSpec(arch=arch, data_dir=data_dir,
                                         checkpoint_dir=pre, n_steps=2, seed=0))
        payload = fid.fidelity_certificate(cfg, run_dir, name,
                                           oracle_set=_pa._tiny_oracle_set())
        out[name] = (arch, md, payload, fid.read_potential(pre), cfg)
    arch, md, payload, potential, cfg = out["deep_lap_3x16"]
    assert payload["parent"] == "pbe"
    for rec in payload["per_system"]:
        assert not rec.get("error"), rec
        assert math.isfinite(float(rec["dE_xc_mHa"])), rec
    assert fid.model_class_mismatches(cfg, payload, "deep_lap_3x16") == []
    assert potential is not None
    assert potential["summary"]["n_failed"] == 0, potential["per_system"]
    assert all(e.get("feature_response_included") is True
               for e in potential["per_system"]), potential["per_system"]
    arch0, md0, payload0, _potential0, _cfg0 = out["deep_3x16"]
    assert set(md) == set(md0), set(md) ^ set(md0)
    assert set(payload) == set(payload0), set(payload) ^ set(payload0)
    assert set(K.mlp_class_of(arch)) == set(K.mlp_class_of(arch0))
    assert set(K.model_class_of_arch(arch)) == set(K.model_class_of_arch(arch0))
    digest = "0" * 64
    assert (set(K.class_record(arch, sha256=digest, size=0))
            == set(K.class_record(arch0, sha256=digest, size=0)))


# ---------------------------------------------------------------------------
# 11. The pretraining step
# ---------------------------------------------------------------------------

def test_one_pretraining_step_moves_the_networks(lap_file):
    """One Adam step (1e-3) of _PretrainLoss on the He/Li file's rows as the
    pretraining assembles them (exchange block [rho, sigma, lap], the
    correlation block [rho, sigma, zeta, lap]; the geometric entry with the
    cusp pair before the column), per network of both entries (polarized):
    finite loss and gradients; the loss depends on the column (its gradient
    with respect to the column's input is nonzero) and so does the first
    layer (the gradient of the weights reading the column, the last input,
    is nonzero); the step moves those weights and leaves a finite loss. The
    same calls on deep_geom_3x16, whose last input is a cusp column, give
    gradients of 2e-6 to 6e-4 and a step of 1e-3."""
    from xcquinox.pipeline.pretrain import _PretrainLoss, _assemble_pretrain_descriptors
    _path, got = lap_file
    data = {k: jnp.asarray(v) for k, v in got.items()}
    loss_fn, optimizer = _PretrainLoss(), optax.adam(1e-3)
    for name in _NEW:
        arch = dataclasses.replace(C.get_architecture(name),
                                   use_polarized_correlation=True)
        xnet, cnet = N.create_network_pair(arch, seed=0)
        blocks = ((xnet, _assemble_pretrain_descriptors(arch, data, suffix="_x"),
                   data["Fx_x"]),
                  (cnet, _assemble_pretrain_descriptors(arch, data, for_cnet=True),
                   data["Fc_all"]))
        for net, rows, ref in blocks:
            loss, grads = eqx.filter_value_and_grad(loss_fn)(net, rows, ref)
            leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
            assert np.isfinite(float(loss))
            assert all(np.all(np.isfinite(np.asarray(g))) for g in leaves)
            g_rows = jax.grad(lambda r: loss_fn(net, r, ref))(rows)
            assert float(jnp.max(jnp.abs(g_rows[:, -1]))) > 0.0, name
            g_col = np.asarray(grads.net.layers[0].weight)[:, -1]
            assert float(np.abs(g_col).max()) > 0.0, name
            params = eqx.filter(net, eqx.is_array)
            updates, _ = optimizer.update(eqx.filter(grads, eqx.is_array),
                                          optimizer.init(params), params)
            stepped = eqx.apply_updates(net, updates)
            moved = np.abs(np.asarray(stepped.net.layers[0].weight)[:, -1]
                           - np.asarray(net.net.layers[0].weight)[:, -1])
            assert float(moved.max()) > 0.0, name
            assert np.isfinite(float(loss_fn(stepped, rows, ref))), name
