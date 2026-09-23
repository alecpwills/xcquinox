"""Step-7 loss-extension unit tests."""
from __future__ import annotations


import jax.numpy as jnp
import pytest

from xcquinox.pipeline import losses


def test_rxn_residual_basic_zero_residual():
    """E_NN_products − E_NN_reactants = E_ref -> residual = 0."""
    e_nn = jnp.array([2.0, 1.0])  # [reactant, product]
    coeffs = jnp.array([-1.0, +1.0])  # reactant subtracted, product added
    e_rxn_ref = jnp.array(-1.0)  # 1.0 - 2.0 = -1.0
    res = losses._rxn_residual_term(e_nn, coeffs, e_rxn_ref)
    assert float(res) == pytest.approx(0.0, abs=1e-12)


def test_rxn_residual_off_by_one():
    e_nn = jnp.array([2.0, 1.0])
    coeffs = jnp.array([-1.0, +1.0])
    e_rxn_ref = jnp.array(0.0)  # but actual rxn energy = -1.0 -> residual = 1.0
    res = losses._rxn_residual_term(e_nn, coeffs, e_rxn_ref)
    assert float(res) == pytest.approx(1.0, abs=1e-12)


def test_ip_residual_squared_displacement():
    e_cation = jnp.array(5.0)
    e_neutral = jnp.array(2.0)
    ip_ref = jnp.array(2.0)  # actual = 3.0; residual^2 = 1.0
    res = losses._ip_residual_term(e_cation, e_neutral, ip_ref)
    assert float(res) == pytest.approx(1.0, abs=1e-12)


# --- relative metric (L5 all-5-channels consistency) -----------------------

def test_rxn_residual_relative_normalizes():
    """relative=True normalizes the BH76 residual by ref^2+1e-8, matching the
    AE/vxc/rho channels so all 5 GradNorm channels are dimensionless."""
    e_nn = jnp.array([2.0, 1.0])
    coeffs = jnp.array([-1.0, +1.0])     # e_rxn = -1.0
    e_ref = jnp.array(-2.0)              # residual^2 = 1.0
    absolute = losses._rxn_residual_term(e_nn, coeffs, e_ref)
    relative = losses._rxn_residual_term(e_nn, coeffs, e_ref, relative=True)
    assert float(absolute) == pytest.approx(1.0, abs=1e-12)
    assert float(relative) == pytest.approx(1.0 / (4.0 + 1e-8), rel=1e-9)


# --- Hartree-units guard on frozen reference energies ----------------------

def test_freeze_rxn_specs_rejects_kcalmol_magnitude():
    """A reaction reference > 10 Ha is almost certainly a kcal/mol value passed
    without conversion, must raise rather than silently train (~627x error)."""
    with pytest.raises(ValueError, match="kcal/mol"):
        losses._freeze_rxn_specs([{
            "name": "R1", "reactants": ("A",), "products": ("B",),
            "coeffs": (-1.0, 1.0), "e_rxn_ref": 50.0,  # 50 "Ha" => implausible
        }])


# --- C1-03 scale-aware floor on the D-family relative delta-AE loss ----------

def test_delta_losses_c1_03_floor_caps_near_zero_target():
    """When PBE already nails the AE (delta_tgt -> 0), the relative delta loss
    denominator is floored at (1 kcal/mol)^2 instead of the old 1e-8 additive
    floor, so a near-exact-PBE compound cannot be over-weighted ~1e8x."""
    from xcquinox.pipeline.losses import _delta_losses, _DELTA_TGT_FLOOR_HA2
    # atom_energies=0 and E_pbe=0 => ae_pbe=0; targets=0 => delta_tgt=0;
    # E_nn=0.1 => delta_nn=-0.1 => residual^2 = 0.01.
    E_nn = jnp.array([0.1])
    mol_data = ({"E_pbe": 0.0},)
    loss = float(_delta_losses(
        E_nn, mol_data, [0], [{"H": 2}], ["X"], {"X": 0.0}, {"H": 0.0},
    ))
    assert loss == pytest.approx(0.01 / _DELTA_TGT_FLOOR_HA2, rel=1e-6)
    # Far below the old additive-1e-8 blowup (0.01/1e-8 = 1e6).
    assert loss < 0.01 / 1e-8


def test_step7_loss_class_registered():
    """The step-7 loss family registers under the pipeline loss registry."""
    from xcquinox.pipeline import losses as pipeline_losses
    assert "L5_gradnorm_vxc_step7" in pipeline_losses.list_losses()


def test_build_indices_prefers_neutral_atom_over_cation():
    """When a spec contains both neutral Li (charge=0) AND Li+ (charge=1)
    as single-atom MoleculeSpecs, atom_mol_idx['Li'] must point at the
    NEUTRAL entry, _atomic_reg compares E_NN[atom_mol_idx[Z]] against
    atom_energies[Z] (neutral Chakravorty value), so pointing at the
    cation would train the cation energy toward the neutral anchor,
    biasing the loss by the IP magnitude (~5 eV for Li).  Mixed-pool
    specs combining HLi (Li anchor) + Li_IP (neutral Li and Li+) hit
    this exact case (jsd/r=5 onward, l2/r=7 onward)."""
    from xcquinox.pipeline.losses import AlecLoss
    from xcquinox.pipeline.config import MoleculeSpec
    li = MoleculeSpec.from_dict(
        name="Li", atom="Li 0 0 0", atom_composition={"Li": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    li_plus = MoleculeSpec.from_dict(
        name="Li+", atom="Li 0 0 0", atom_composition={"Li": 1},
        basis="sto-3g", charge=1, spin=0,
    )
    h_atom = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    h_li = MoleculeSpec.from_dict(
        name="HLi", atom="H 0 0 0; Li 0 0 1",
        atom_composition={"H": 1, "Li": 1},
        basis="sto-3g", charge=0, spin=0,
    )
    # Order with cation FIRST (would be the failing case under the old
    # last-wins logic).  build_indices must still pick neutral Li.
    molecules = (li_plus, h_atom, h_li, li)
    ami, ci, mn, _ = AlecLoss.build_indices(molecules)
    ami_dict = dict(ami)
    assert "Li" in ami_dict and "H" in ami_dict
    li_idx = ami_dict["Li"]
    assert int(molecules[li_idx].charge) == 0, (
        f"atom_mol_idx['Li'] = idx {li_idx} -> "
        f"{mn[li_idx]} (charge={molecules[li_idx].charge}); "
        f"expected neutral Li (charge=0)"
    )
    # And the reverse order (neutral first, cation later), neutral
    # should still win.
    molecules2 = (li, h_atom, h_li, li_plus)
    ami2, _, _, _ = AlecLoss.build_indices(molecules2)
    li_idx2 = dict(ami2)["Li"]
    assert int(molecules2[li_idx2].charge) == 0


# ---------------------------------------------------------------------------
# regularize_atom_syms subset validation
# ---------------------------------------------------------------------------

def test_regularize_atom_syms_typo_raises_value_error():
    """Constructing L5GradnormVxcStep7 with a regularize_atom_syms element
    not present among the single-atom molecules in the spec must raise
    ValueError naming the missing symbols.  A typo'd symbol (e.g. 'Xx')
    should not be silently dropped."""
    from xcquinox.pipeline.losses import L5GradnormVxcStep7
    from xcquinox.pipeline.config import MoleculeSpec

    h_atom = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    li = MoleculeSpec.from_dict(
        name="Li", atom="Li 0 0 0", atom_composition={"Li": 1},
        basis="sto-3g", charge=0, spin=1,
    )

    with pytest.raises(ValueError, match="regularize_atom_syms"):
        L5GradnormVxcStep7(
            molecules=(h_atom, li),
            bh76_reactions=(),
            ip13_pairs=(),
            regularize_atom_syms=("H", "Xx"),  # "Xx" not in atom_mol_idx
        )


# ---------------------------------------------------------------------------
# LOSS-04: RuntimeWarning when any channel has None references skipped
# ---------------------------------------------------------------------------


def test_grid_term_per_electron_normalization(monkeypatch):
    """per_electron=True divides the weighted-L2 integral by N_e^2
    (N_e = sum w*rho_ref; dpyscf losses.py:171 convention), making the
    channel intensive: same per-electron error => same loss regardless of
    electron count."""
    import jax.numpy as jnp
    import xcquinox.pipeline.losses as L

    # 2-electron and 8-electron fake systems with the SAME relative error
    md_small = {"rho_ref_grid": jnp.array([2.0, 2.0]),
                "grid_weights": jnp.array([0.5, 0.5])}        # N_e = 2
    md_big = {"rho_ref_grid": jnp.array([8.0, 8.0]),
              "grid_weights": jnp.array([0.5, 0.5])}          # N_e = 8
    monkeypatch.setattr(L, "grid_density_for_loss",
                        lambda model, md, solver_config=None:
                        md["rho_ref_grid"] * 1.01)            # +1% density
    out_small = L._grid_term(object(), [md_small], (0,), per_electron=True)
    out_big = L._grid_term(object(), [md_big], (0,), per_electron=True)
    assert float(out_small) == pytest.approx(float(out_big), rel=1e-9)
    # absolute (default) mode keeps the size-extensive behavior: 16x ratio
    abs_small = L._grid_term(object(), [md_small], (0,))
    abs_big = L._grid_term(object(), [md_big], (0,))
    assert float(abs_big) / float(abs_small) == pytest.approx(16.0, rel=1e-9)


