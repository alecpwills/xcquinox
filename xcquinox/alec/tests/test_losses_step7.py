"""Step-7 loss-extension unit tests."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from xcquinox.alec import losses


def test_rxn_residual_basic_zero_residual():
    """E_NN_products − E_NN_reactants = E_ref → residual = 0."""
    e_nn = jnp.array([2.0, 1.0])  # [reactant, product]
    coeffs = jnp.array([-1.0, +1.0])  # reactant subtracted, product added
    e_rxn_ref = jnp.array(-1.0)  # 1.0 - 2.0 = -1.0
    res = losses._rxn_residual_term(e_nn, coeffs, e_rxn_ref)
    assert float(res) == pytest.approx(0.0, abs=1e-12)


def test_rxn_residual_off_by_one():
    e_nn = jnp.array([2.0, 1.0])
    coeffs = jnp.array([-1.0, +1.0])
    e_rxn_ref = jnp.array(0.0)  # but actual rxn energy = -1.0 → residual = 1.0
    res = losses._rxn_residual_term(e_nn, coeffs, e_rxn_ref)
    assert float(res) == pytest.approx(1.0, abs=1e-12)


def test_ip_residual_basic_zero_residual():
    """IP = E_cation - E_neutral; residual = (IP_NN - IP_ref)^2 → 0 when match."""
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


def test_step7_loss_class_registered():
    """The step-7 loss family registers under the alec loss registry."""
    from xcquinox.alec import losses as alec_losses
    assert "L5_gradnorm_vxc_step7" in alec_losses.list_losses()


def test_step7_loss_target_kinds():
    """Per spec §5b: 5 GradNorm channels (AE, BH76, IP13, vxc, rho)."""
    from xcquinox.alec import losses as alec_losses
    cls = alec_losses.make_loss("L5_gradnorm_vxc_step7", _smoke_test=True)
    assert sorted(cls.target_kinds) == sorted(["AE", "BH76", "IP13", "vxc", "rho"])


def test_step7_loss_smoke_constructor_with_dict_inputs():
    """Construct via dict inputs (no training context); verify field accessors."""
    from xcquinox.alec import losses as alec_losses
    from xcquinox.alec import dfs_pool

    pool = dfs_pool.build_dfs_pool()
    inst = alec_losses.make_loss(
        "L5_gradnorm_vxc_step7",
        bh76_reactions=pool["bh76_reactions"],
        ip13_pairs=pool["ip13_pairs"],
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


def test_aux_only_names_filtering_all_compounds_raises():
    """If aux_only_names removes every compound, __init__ raises ValueError."""
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
    with pytest.raises(ValueError, match="aux_only_names filtered out all"):
        L5GradnormVxcStep7(
            molecules=(h2o, h_atom),
            bh76_reactions=(),
            ip13_pairs=(),
            aux_only_names=("H2O",),
        )
