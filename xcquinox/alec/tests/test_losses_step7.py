"""Step-7 loss-extension unit tests."""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from xcquinox.alec import losses


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


def test_ip_residual_basic_zero_residual():
    """IP = E_cation - E_neutral; residual = (IP_NN - IP_ref)^2 -> 0 when match."""
    e_cation = jnp.array(5.0)
    e_neutral = jnp.array(2.0)
    ip_ref = jnp.array(3.0)
    res = losses._ip_residual_term(e_cation, e_neutral, ip_ref)
    assert float(res) == pytest.approx(0.0, abs=1e-12)


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


def test_ip_residual_relative_normalizes():
    e_cation = jnp.array(5.0)
    e_neutral = jnp.array(3.0)          # IP = 2.0
    ip_ref = jnp.array(1.0)            # residual^2 = 1.0
    relative = losses._ip_residual_term(e_cation, e_neutral, ip_ref, relative=True)
    assert float(relative) == pytest.approx(1.0 / (1.0 + 1e-8), rel=1e-9)


# --- Hartree-units guard on frozen reference energies ----------------------

def test_freeze_rxn_specs_rejects_kcalmol_magnitude():
    """A reaction reference > 10 Ha is almost certainly a kcal/mol value passed
    without conversion, must raise rather than silently train (~627x error)."""
    with pytest.raises(ValueError, match="kcal/mol"):
        losses._freeze_rxn_specs([{
            "name": "R1", "reactants": ("A",), "products": ("B",),
            "coeffs": (-1.0, 1.0), "e_rxn_ref": 50.0,  # 50 "Ha" ⇒ implausible
        }])


def test_freeze_rxn_specs_accepts_hartree():
    out = losses._freeze_rxn_specs([{
        "name": "R1", "reactants": ("A",), "products": ("B",),
        "coeffs": (-1.0, 1.0), "e_rxn_ref": 0.08,  # ~50 kcal/mol in Ha
    }])
    assert out[0][-1] == pytest.approx(0.08)


def test_freeze_ip_specs_rejects_kcalmol_magnitude():
    with pytest.raises(ValueError, match="kcal/mol"):
        losses._freeze_ip_specs([{
            "name": "Li", "neutral": "Li", "cation": "Li+", "ip_ref": 124.0,
        }])


# --- C1-03 scale-aware floor on the D-family relative delta-AE loss ----------

def test_delta_losses_c1_03_floor_caps_near_zero_target():
    """When PBE already nails the AE (delta_tgt -> 0), the relative delta loss
    denominator is floored at (1 kcal/mol)^2 instead of the old 1e-8 additive
    floor, so a near-exact-PBE compound cannot be over-weighted ~1e8x."""
    from xcquinox.alec.losses import _delta_losses, _DELTA_TGT_FLOOR_HA2
    # atom_energies=0 and E_pbe=0 ⇒ ae_pbe=0; targets=0 ⇒ delta_tgt=0;
    # E_nn=0.1 ⇒ delta_nn=-0.1 ⇒ residual^2 = 0.01.
    E_nn = jnp.array([0.1])
    mol_data = ({"E_pbe": 0.0},)
    loss = float(_delta_losses(
        E_nn, mol_data, [0], [{"H": 2}], ["X"], {"X": 0.0}, {"H": 0.0},
    ))
    assert loss == pytest.approx(0.01 / _DELTA_TGT_FLOOR_HA2, rel=1e-6)
    # Far below the old additive-1e-8 blowup (0.01/1e-8 = 1e6).
    assert loss < 0.01 / 1e-8


def test_delta_losses_c1_03_floor_noop_for_normal_targets():
    """For a normal PBE error (delta_tgt ~ 0.05 Ha >> floor) the loss is
    unchanged: max(delta_tgt^2, floor) == delta_tgt^2."""
    from xcquinox.alec.losses import _delta_losses
    # ae_pbe=0; delta_tgt = targets = 0.05 (>> sqrt(floor) ~1.6e-3);
    # delta_nn=-0.1 ⇒ residual = (-0.1-0.05)^2 = 0.0225; denom = 0.05^2 = 0.0025.
    E_nn = jnp.array([0.1])
    mol_data = ({"E_pbe": 0.0},)
    loss = float(_delta_losses(
        E_nn, mol_data, [0], [{"H": 2}], ["X"], {"X": 0.05}, {"H": 0.0},
    ))
    assert loss == pytest.approx((0.15 ** 2) / (0.05 ** 2), rel=1e-5)


def test_step7_loss_class_registered():
    """The step-7 loss family registers under the alec loss registry."""
    from xcquinox.alec import losses as alec_losses
    assert "L5_gradnorm_vxc_step7" in alec_losses.list_losses()


def test_step7_loss_target_kinds():
    """Per spec §5b: 5 GradNorm channels (AE, BH76, IP13, vxc, rho)."""
    from xcquinox.alec import losses as alec_losses
    cls = alec_losses.make_loss("L5_gradnorm_vxc_step7", _smoke_test=True)
    assert sorted(cls.target_kinds) == sorted(["AE", "BH76", "IP13", "vxc", "rho"])


def test_step7_loss_rejects_dead_pbe_anchor_knob():
    """CW2 fix: L5 has no PBE-anchor channel, so a positive pbe_anchor_weight
    (or a non-None pbe_anchor_sample) must fail loudly instead of being
    silently swallowed by **_unused_kwargs."""
    import pytest
    from xcquinox.alec import losses as alec_losses
    with pytest.raises(ValueError, match="no PBE-anchor channel"):
        alec_losses.make_loss(
            "L5_gradnorm_vxc_step7", _smoke_test=True, pbe_anchor_weight=0.1)
    with pytest.raises(ValueError, match="no PBE-anchor channel"):
        alec_losses.make_loss(
            "L5_gradnorm_vxc_step7", _smoke_test=True,
            pbe_anchor_sample=object())
    # weight=0 (the default) must NOT raise.
    alec_losses.make_loss(
        "L5_gradnorm_vxc_step7", _smoke_test=True, pbe_anchor_weight=0.0)


def test_step7_loss_smoke_constructor_with_dict_inputs():
    """Construct via dict inputs (no training context); verify field accessors."""
    from xcquinox.alec import losses as alec_losses
    from xcquinox.alec import dfs_pool

    pool = dfs_pool.build_dfs_pool()
    # dfs_pool stores references in kcal/mol; the L5 loss contract is Hartree
    # (production converts via domain.{bh76,ip13}_meta_to_loss_dict before
    # constructing the loss). Mirror that here so the Hartree-units guard passes.
    from xcquinox.alec.cluster.domain import KCAL_PER_HA

    def _to_ha(d, key):
        v = d.get(key)
        return {**d, key: (None if v is None else v / KCAL_PER_HA)}

    bh76_ha = [_to_ha(r, "e_rxn_ref") for r in pool["bh76_reactions"]]
    ip13_ha = [_to_ha(p, "ip_ref") for p in pool["ip13_pairs"]]
    inst = alec_losses.make_loss(
        "L5_gradnorm_vxc_step7",
        bh76_reactions=bh76_ha,
        ip13_pairs=ip13_ha,
        _smoke_test=True,
    )
    assert len(inst.bh76_reactions) == 3
    assert len(inst.ip13_pairs) == 2
    assert sorted(inst.target_kinds) == sorted(["AE", "BH76", "IP13", "vxc", "rho"])


def test_aux_only_names_excludes_from_compound_idx():
    """Species in aux_only_names are filtered out of compound_idx."""
    from xcquinox.alec.losses import L5GradnormVxcStep7
    from xcquinox.alec.config import MoleculeSpec
    h2o = MoleculeSpec.from_dict(
        name="H2O", atom="O 0 0 0; H 0 0 1; H 0 1 0",
        atom_composition={"O": 1, "H": 2},
        basis="sto-3g", charge=0, spin=0,
    )
    h_atom = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    o_atom = MoleculeSpec.from_dict(
        name="O", atom="O 0 0 0", atom_composition={"O": 1},
        basis="sto-3g", charge=0, spin=2,
    )
    hbwd = MoleculeSpec.from_dict(
        name="HBWD", atom="O 0 0 0; H 0 0 1; H 0 1 0; "
                          "O 3 0 0; H 3 0 1; H 3 1 0",
        atom_composition={"O": 2, "H": 4},
        basis="sto-3g", charge=1, spin=1,
    )
    loss = L5GradnormVxcStep7(
        molecules=(h2o, h_atom, o_atom, hbwd),
        bh76_reactions=(),
        ip13_pairs=(),
        aux_only_names=("HBWD",),
    )
    name_idx = {n: i for i, n in enumerate(loss.mol_names)}
    assert name_idx["H2O"] in loss.compound_idx
    assert name_idx["HBWD"] not in loss.compound_idx, (
        "HBWD should be excluded from compound_idx via aux_only_names")
    iter_idx = loss._iter_idx_for_aux_channels()
    assert name_idx["HBWD"] in iter_idx


def test_aux_only_names_default_empty_tuple():
    """Default aux_only_names=() doesn't change behavior."""
    from xcquinox.alec.losses import L5GradnormVxcStep7
    from xcquinox.alec.config import MoleculeSpec
    h2o = MoleculeSpec.from_dict(
        name="H2O", atom="O 0 0 0; H 0 0 1; H 0 1 0",
        atom_composition={"O": 1, "H": 2},
        basis="sto-3g", charge=0, spin=0,
    )
    h_atom = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    o_atom = MoleculeSpec.from_dict(
        name="O", atom="O 0 0 0", atom_composition={"O": 1},
        basis="sto-3g", charge=0, spin=2,
    )
    loss = L5GradnormVxcStep7(
        molecules=(h2o, h_atom, o_atom),
        bh76_reactions=(),
        ip13_pairs=(),
    )
    name_idx = {n: i for i, n in enumerate(loss.mol_names)}
    assert name_idx["H2O"] in loss.compound_idx


def test_iter_idx_default_equals_compound_idx():
    """With aux_only_names=(), iter_idx must equal compound_idx exactly."""
    from xcquinox.alec.losses import L5GradnormVxcStep7
    from xcquinox.alec.config import MoleculeSpec
    h2o = MoleculeSpec.from_dict(
        name="H2O", atom="O 0 0 0; H 0 0 1; H 0 1 0",
        atom_composition={"O": 1, "H": 2},
        basis="sto-3g", charge=0, spin=0,
    )
    h_atom = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    o_atom = MoleculeSpec.from_dict(
        name="O", atom="O 0 0 0", atom_composition={"O": 1},
        basis="sto-3g", charge=0, spin=2,
    )
    loss = L5GradnormVxcStep7(
        molecules=(h2o, h_atom, o_atom),
        bh76_reactions=(),
        ip13_pairs=(),
    )
    assert loss._iter_idx_for_aux_channels() == loss.compound_idx


def test_aux_only_names_filtering_all_compounds_is_permitted():
    """When aux_only_names removes every compound, __init__ no longer
    raises (2026-05-10 mixed-pool relaxation): a BH76- or IP13-only
    subset is a legitimate L5 configuration where the AE channel
    contributes 0 and BH76 / IP13 carry the loss.  Verify compound_idx
    is empty after the filter and loss object still builds."""
    from xcquinox.alec.losses import L5GradnormVxcStep7
    from xcquinox.alec.config import MoleculeSpec
    h2o = MoleculeSpec.from_dict(
        name="H2O", atom="O 0 0 0; H 0 0 1; H 0 1 0",
        atom_composition={"O": 1, "H": 2},
        basis="sto-3g", charge=0, spin=0,
    )
    h_atom = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    loss = L5GradnormVxcStep7(
        molecules=(h2o, h_atom),
        bh76_reactions=(),
        ip13_pairs=(),
        aux_only_names=("H2O",),
    )
    assert loss.compound_idx == ()
    # H atom remains in atom_mol_idx; H2O is filtered out of compound_idx.
    assert dict(loss.atom_mol_idx) == {"H": 1}


def test_build_indices_prefers_neutral_atom_over_cation():
    """When a spec contains both neutral Li (charge=0) AND Li+ (charge=1)
    as single-atom MoleculeSpecs, atom_mol_idx['Li'] must point at the
    NEUTRAL entry, _atomic_reg compares E_NN[atom_mol_idx[Z]] against
    atom_energies[Z] (neutral Chakravorty value), so pointing at the
    cation would train the cation energy toward the neutral anchor,
    biasing the loss by the IP magnitude (~5 eV for Li).  Mixed-pool
    specs combining HLi (Li anchor) + Li_IP (neutral Li and Li+) hit
    this exact case (jsd/r=5 onward, l2/r=7 onward)."""
    from xcquinox.alec.losses import AlecLoss
    from xcquinox.alec.config import MoleculeSpec
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


def test_build_indices_require_compound_false_permits_atomic_only():
    """build_indices(require_compound=False) does NOT raise when every
    molecule is single-atom, supports L5GradnormVxcStep7 specs that
    contain only IP13 species (Li, Li+) with no compound molecules."""
    from xcquinox.alec.losses import AlecLoss
    from xcquinox.alec.config import MoleculeSpec
    li = MoleculeSpec.from_dict(
        name="Li", atom="Li 0 0 0", atom_composition={"Li": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    li_plus = MoleculeSpec.from_dict(
        name="Li+", atom="Li 0 0 0", atom_composition={"Li": 1},
        basis="sto-3g", charge=1, spin=0,
    )
    # Default require_compound=True still raises.
    with pytest.raises(ValueError, match="at least one compound molecule"):
        AlecLoss.build_indices((li, li_plus))
    # require_compound=False permits empty compound_idx.
    ami, ci, mn, _ = AlecLoss.build_indices(
        (li, li_plus), require_compound=False)
    assert ci == ()
    assert dict(ami) == {"Li": 0}  # neutral Li


def test_l5_handles_ip13_only_spec_with_no_compound():
    """L5GradnormVxcStep7 must accept a pure-IP13 spec (Li + Li+ only)
    without any polyatomic compounds.  The AE-fitting term is 0; the
    IP13 channel and atomic_reg carry the loss."""
    from xcquinox.alec.losses import L5GradnormVxcStep7
    from xcquinox.alec.config import MoleculeSpec
    li = MoleculeSpec.from_dict(
        name="Li", atom="Li 0 0 0", atom_composition={"Li": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    li_plus = MoleculeSpec.from_dict(
        name="Li+", atom="Li 0 0 0", atom_composition={"Li": 1},
        basis="sto-3g", charge=1, spin=0,
    )
    loss = L5GradnormVxcStep7(
        molecules=(li, li_plus),
        bh76_reactions=(),
        ip13_pairs=({"name": "Li_IP", "neutral": "Li", "cation": "Li+",
                     "ip_ref": 0.198},),
        regularize_atom_syms=("Li",),
    )
    assert loss.compound_idx == ()
    assert dict(loss.atom_mol_idx) == {"Li": 0}


# ---------------------------------------------------------------------------
# regularize_atom_syms subset validation
# ---------------------------------------------------------------------------

def test_regularize_atom_syms_typo_raises_value_error():
    """Constructing L5GradnormVxcStep7 with a regularize_atom_syms element
    not present among the single-atom molecules in the spec must raise
    ValueError naming the missing symbols.  A typo'd symbol (e.g. 'Xx')
    should not be silently dropped."""
    from xcquinox.alec.losses import L5GradnormVxcStep7
    from xcquinox.alec.config import MoleculeSpec

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


def test_regularize_atom_syms_absent_anchor_raises_value_error():
    """A symbol that is a valid element but not present as a single-atom
    MoleculeSpec in the spec must also raise ValueError."""
    from xcquinox.alec.losses import L5GradnormVxcStep7
    from xcquinox.alec.config import MoleculeSpec

    h_atom = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", charge=0, spin=1,
    )

    with pytest.raises(ValueError, match="regularize_atom_syms"):
        L5GradnormVxcStep7(
            molecules=(h_atom,),
            bh76_reactions=(),
            ip13_pairs=(),
            regularize_atom_syms=("H", "Li"),  # "Li" not in molecules
        )


def test_regularize_atom_syms_valid_subset_does_not_raise():
    """A valid subset of present single-atom species must not raise."""
    from xcquinox.alec.losses import L5GradnormVxcStep7
    from xcquinox.alec.config import MoleculeSpec

    h_atom = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", charge=0, spin=1,
    )
    li = MoleculeSpec.from_dict(
        name="Li", atom="Li 0 0 0", atom_composition={"Li": 1},
        basis="sto-3g", charge=0, spin=1,
    )

    loss = L5GradnormVxcStep7(
        molecules=(h_atom, li),
        bh76_reactions=(),
        ip13_pairs=(),
        regularize_atom_syms=("H",),  # valid strict subset
    )
    assert loss.regularize_atom_syms == ("H",)


# ---------------------------------------------------------------------------
# LOSS-04: RuntimeWarning when any channel has None references skipped
# ---------------------------------------------------------------------------

def _make_mock_mol_data_with_none(n: int, *, vxc_none_idx=(0,), rho_none_idx=(),
                                  dm_none_idx=()):
    """Build a minimal list of mol_data dicts where some references are None."""
    mols = []
    for i in range(n):
        mols.append({
            "vxc_ref": None if i in vxc_none_idx else object(),
            "rho_ref_grid": None if i in rho_none_idx else object(),
            "dm_target": None if i in dm_none_idx else object(),
        })
    return mols


def test_vxc_term_warns_on_none_references():
    """_vxc_term must emit a RuntimeWarning when any mol has vxc_ref=None."""
    from xcquinox.alec.losses import _vxc_term

    class _FakeModel:
        pass

    # Two mols, one with vxc_ref=None, should warn about 1 skipped entry.
    mol_data = [
        {"vxc_ref": None},
        {"vxc_ref": None},
    ]
    with pytest.warns(RuntimeWarning, match="vxc"):
        _vxc_term(_FakeModel(), mol_data, iter_idx=(0, 1))


def test_grid_term_warns_on_none_references():
    """_grid_term must emit a RuntimeWarning when any mol has rho_ref_grid=None."""
    from xcquinox.alec.losses import _grid_term

    class _FakeModel:
        pass

    mol_data = [
        {"rho_ref_grid": None},
        {"rho_ref_grid": None},
    ]
    with pytest.warns(RuntimeWarning, match="rho"):
        _grid_term(_FakeModel(), mol_data, iter_idx=(0, 1))


def test_dm_term_warns_on_none_references():
    """_dm_term must emit a RuntimeWarning when any mol has dm_target=None."""
    from xcquinox.alec.losses import _dm_term

    class _FakeModel:
        pass

    mol_data = [
        {"dm_target": None},
        {"dm_target": None},
    ]
    with pytest.warns(RuntimeWarning, match="dm"):
        _dm_term(_FakeModel(), mol_data, iter_idx=(0, 1))
