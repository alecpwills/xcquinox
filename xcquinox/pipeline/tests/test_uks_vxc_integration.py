"""End-to-end integration test: UKS OEP -> save -> load -> _vxc_term."""
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import gto, scf

import xcquinox.pipeline as pipeline
from xcquinox.pipeline.config import MoleculeSpec
from xcquinox.pipeline.data import precompute_fixed_density_data


def test_uks_oep_to_vxc_term_end_to_end():
    """Pipeline: Li-atom UHF DM -> UKS OEP -> save_vxc_ref -> external_data_path
    loaded into mol_data -> _vxc_term returns finite loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Build target UHF DM for Li atom (spin=1, avoids p-degeneracy)
        mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
        mf_hf = scf.UHF(mol); mf_hf.kernel()
        dm_target = mf_hf.make_rdm1()
        assert dm_target.ndim == 3 and dm_target.shape[0] == 2

        # 2. UKS OEP inversion
        spec = MoleculeSpec(
            name="Li", atom="Li 0 0 0", basis="sto-3g",
            charge=0, spin=1, atom_composition=(("Li", 1),),
            grid_level=1,
        )
        oep_result = pipeline.run_oep_inversion(
            spec, dm_target, max_iter=50, conv_tol=1e-4,
        )
        assert oep_result.vxc_matrix.shape == (2, 5, 5), (
            f"expected (2,5,5), got {oep_result.vxc_matrix.shape}"
        )

        # 3. Save to .npz via save_vxc_ref
        npz_path = os.path.join(tmpdir, "Li.npz")
        pipeline.save_vxc_ref(
            oep_result, npz_path,
            dm_target=dm_target,
            method="uhf",
        )
        assert os.path.isfile(npz_path)

        # 4. Build a new MoleculeSpec pointing at the saved .npz,
        # re-precompute with external_data_path, verify vxc_ref is loaded
        spec2 = MoleculeSpec(
            name="Li", atom="Li 0 0 0", basis="sto-3g",
            charge=0, spin=1, atom_composition=(("Li", 1),),
            grid_level=1, external_data_path=npz_path,
        )
        md = precompute_fixed_density_data(spec2, required_keys=("vxc_ref",))
        assert md["vxc_ref"] is not None
        vxc_loaded = np.asarray(md["vxc_ref"])
        assert vxc_loaded.shape == (2, 5, 5)
        # Loaded matrix should match what OEP produced (within 1e-10)
        assert np.max(np.abs(vxc_loaded - oep_result.vxc_matrix)) < 1e-10

        # 5. Compute _vxc_term using this mol_data and a random NN
        from xcquinox.pipeline.losses import _vxc_term
        arch = pipeline.get_architecture("deep")
        xnet, cnet = pipeline.create_network_pair(arch, seed=0)
        model = pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
        val = _vxc_term(model, [md], [0])
        assert jnp.isfinite(val), f"vxc term not finite: {val}"
        # Random NN won't match OEP-derived V_xc, so loss > 0
        assert float(val) > 0, f"expected positive loss for random NN, got {val}"
