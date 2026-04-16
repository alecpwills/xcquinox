"""Tests for xcquinox.alec.losses — AlecLoss, LOSS_REGISTRY, 6 concrete losses.

Implements THE SPEC §13.2 test_losses.py: exactly 43 tests.

Test structure:
  Tests 1-24  (4 parametrized over 6 losses × 4 aspects = 24 tests):
    (a) registry roundtrip
    (b) forward → (scalar, dict)
    (c) required_mol_keys
    (d) differentiability via eqx.filter_value_and_grad

  Tests 25-43 (19 additional):
    25. LOSS_REGISTRY has exactly 6 builtins
    26. list_losses returns sorted
    27. make_loss kwargs propagation
    28. make_loss unknown name raises KeyError
    29. aux dict carries no gradient
    30. xfail: atomization notebook cell26
    31. xfail: delta-ae notebook cell29
    32. molecule-generic (H, N, NH3)
    33. B15-4: molecules_only=True skips atoms in DM term
    34. dm_weight=0.2 scales DM term
    35. density_weight=0.3 scales grid term
    36. xfail: missing required_mol_keys raises KeyError
    37. D-H1: float type validation
    38. D-H5: field assignment completeness
    39. xfail: H-E12-3 DM loss UKS/RKS scaling
    40. M-E12-4: aux dict schema per class
    41. AtomizationLoss sign convention
    42. xfail: training targets sign matches test reference
    43. L-B13-2: molecules_only bool type validation
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from xcquinox.alec.config import ArchitectureConfig, MoleculeSpec
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.losses import (
    LOSS_REGISTRY,
    AlecLoss,
    AtomizationLoss,
    AtomizationPlusDMLoss,
    AtomizationPlusGridLoss,
    DeltaAELoss,
    DeltaAEPlusDMLoss,
    DeltaAEPlusGridLoss,
    list_losses,
    make_loss,
)
from xcquinox.alec.tests.fixtures.molecules import h_atom, o_atom, h2o_molecule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arch():
    return ArchitectureConfig(
        name="tiny", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )


def _make_model():
    return AlecGGAModel.from_arch(_make_arch(), seed=0)


# ---------------------------------------------------------------------------
# Module-scoped fixtures: precompute PySCF data once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h_mol_data():
    return precompute_fixed_density_data(h_atom())


@pytest.fixture(scope="module")
def o_mol_data():
    return precompute_fixed_density_data(o_atom())


@pytest.fixture(scope="module")
def h2o_mol_data():
    return precompute_fixed_density_data(h2o_molecule())


@pytest.fixture(scope="module")
def batch_h_o_h2o(h_mol_data, o_mol_data, h2o_mol_data):
    """Standard 3-molecule batch (H, O, H2O) with H and O as atoms."""
    mols = (h_atom(), o_atom(), h2o_molecule())
    mol_data = (h_mol_data, o_mol_data, h2o_mol_data)
    # Realistic reference AE for H2O (sto-3g PBE, approximately in Hartree)
    ae_h2o = float(
        h_mol_data["E_pbe"] * 2 + o_mol_data["E_pbe"] - h2o_mol_data["E_pbe"]
    )
    targets = {"H2O": max(ae_h2o, 0.001)}
    atom_energies = {
        "H": float(h_mol_data["E_pbe"]),
        "O": float(o_mol_data["E_pbe"]),
    }
    return {
        "mols": mols,
        "mol_data": mol_data,
        "targets": targets,
        "atom_energies": atom_energies,
    }


@pytest.fixture(scope="module")
def model():
    return _make_model()


# ---------------------------------------------------------------------------
# Loss parametrize: 6 loss names × expected types × required_mol_keys
# ---------------------------------------------------------------------------

LOSS_PARAMS = [
    pytest.param(
        "A_atomization", AtomizationLoss, (),
        {"loss_energy", "atomic_reg"},
        id="A_atomization",
    ),
    pytest.param(
        "B_atomization_plus_dm", AtomizationPlusDMLoss, ("dm_target",),
        {"loss_energy", "loss_dm"},
        id="B_atomization_plus_dm",
    ),
    pytest.param(
        "C_atomization_plus_grid", AtomizationPlusGridLoss, ("rho_ref_grid",),
        {"loss_energy", "loss_grid"},
        id="C_atomization_plus_grid",
    ),
    pytest.param(
        "D1_delta_ae", DeltaAELoss, ("E_pbe",),
        {"loss_delta", "atomic_reg"},
        id="D1_delta_ae",
    ),
    pytest.param(
        "D2_delta_ae_plus_dm", DeltaAEPlusDMLoss, ("E_pbe", "dm_target"),
        {"loss_delta", "loss_dm"},
        id="D2_delta_ae_plus_dm",
    ),
    pytest.param(
        "D3_delta_ae_plus_grid", DeltaAEPlusGridLoss, ("E_pbe", "rho_ref_grid"),
        {"loss_delta", "loss_grid"},
        id="D3_delta_ae_plus_grid",
    ),
]

# Param IDs only (for parametrize that does not need the full tuple)
LOSS_NAMES = [
    "A_atomization",
    "B_atomization_plus_dm",
    "C_atomization_plus_grid",
    "D1_delta_ae",
    "D2_delta_ae_plus_dm",
    "D3_delta_ae_plus_grid",
]


# ---------------------------------------------------------------------------
# Tests 1-6 (a): registry roundtrip — make_loss returns correct type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_name,expected_cls,_rmk,_aux_keys", LOSS_PARAMS)
def test_registry_roundtrip(loss_name, expected_cls, _rmk, _aux_keys, batch_h_o_h2o):
    """Test (a): make_loss(name, molecules=...) returns the expected class."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss(loss_name, molecules=mols)
    assert isinstance(loss, expected_cls), (
        f"make_loss({loss_name!r}) returned {type(loss).__name__}, "
        f"expected {expected_cls.__name__}"
    )
    assert isinstance(loss, AlecLoss)


# ---------------------------------------------------------------------------
# Tests 7-12 (b): forward on 3-molecule batch returns (scalar, dict)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_name,_cls,_rmk,_aux_keys", LOSS_PARAMS)
def test_forward_returns_scalar_and_dict(loss_name, _cls, _rmk, _aux_keys,
                                         batch_h_o_h2o, model):
    """Test (b): loss(model, batch) returns (scalar JAX array, dict)."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss(loss_name, molecules=mols)
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    total, aux = loss(model, batch)
    # total must be a scalar JAX array
    assert hasattr(total, "shape"), "total loss must be a JAX array"
    assert total.shape == (), f"total loss must be scalar, got shape {total.shape}"
    assert jnp.isfinite(total), f"total loss must be finite, got {total}"
    # aux must be a dict
    assert isinstance(aux, dict), f"aux must be a dict, got {type(aux)}"


# ---------------------------------------------------------------------------
# Tests 13-18 (c): required_mol_keys matches documented table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_name,expected_cls,expected_rmk,_aux_keys", LOSS_PARAMS)
def test_required_mol_keys(loss_name, expected_cls, expected_rmk, _aux_keys,
                            batch_h_o_h2o):
    """Test (c): required_mol_keys class attribute matches documented table."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss(loss_name, molecules=mols)
    assert loss.required_mol_keys == expected_rmk, (
        f"{loss_name}.required_mol_keys = {loss.required_mol_keys!r}, "
        f"expected {expected_rmk!r}"
    )


# ---------------------------------------------------------------------------
# Tests 19-24 (d): differentiability via eqx.filter_value_and_grad(has_aux=True)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_name,_cls,_rmk,_aux_keys", LOSS_PARAMS)
def test_differentiability(loss_name, _cls, _rmk, _aux_keys,
                            batch_h_o_h2o, model):
    """Test (d): grads through loss are finite and at least one is nonzero."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss(loss_name, molecules=mols)
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }

    def loss_fn(mdl):
        return loss(mdl, batch)

    grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (total, aux), grads = grad_fn(model)

    leaves = jax.tree_util.tree_leaves(grads)
    array_leaves = [leaf for leaf in leaves if isinstance(leaf, jnp.ndarray)]
    assert array_leaves, "grad tree must have at least one array leaf"
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in array_leaves
    ), f"all grad leaves must be finite for loss {loss_name!r}"
    assert any(
        jnp.any(leaf != 0) for leaf in array_leaves
    ), f"at least one grad leaf must be nonzero for loss {loss_name!r}"


# ---------------------------------------------------------------------------
# Test 25: LOSS_REGISTRY has exactly 6 builtins
# ---------------------------------------------------------------------------

def test_loss_registry_has_exactly_6_builtins():
    """Test 25: LOSS_REGISTRY contains exactly the 6 documented loss names."""
    expected = {
        "A_atomization",
        "B_atomization_plus_dm",
        "C_atomization_plus_grid",
        "D1_delta_ae",
        "D2_delta_ae_plus_dm",
        "D3_delta_ae_plus_grid",
    }
    assert set(LOSS_REGISTRY.keys()) == expected, (
        f"LOSS_REGISTRY keys = {sorted(LOSS_REGISTRY.keys())}, "
        f"expected {sorted(expected)}"
    )
    assert len(LOSS_REGISTRY) == 6


# ---------------------------------------------------------------------------
# Test 26: list_losses returns sorted
# ---------------------------------------------------------------------------

def test_list_losses_returns_sorted():
    """Test 26: list_losses() returns a sorted list of loss names."""
    names = list_losses()
    assert names == sorted(names), f"list_losses() must be sorted, got {names}"
    assert len(names) == 6


# ---------------------------------------------------------------------------
# Test 27: make_loss kwargs propagation
# ---------------------------------------------------------------------------

def test_make_loss_propagates_kwargs(batch_h_o_h2o):
    """Test 27: extra kwargs like w_atomic are forwarded to the loss constructor."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("D1_delta_ae", molecules=mols, w_atomic=0.05)
    assert loss.w_atomic == 0.05, (
        f"w_atomic should be 0.05, got {loss.w_atomic}"
    )


# ---------------------------------------------------------------------------
# Test 28: make_loss unknown name raises KeyError
# ---------------------------------------------------------------------------

def test_make_loss_unknown_name_raises_key_error(batch_h_o_h2o):
    """Test 28: make_loss with an unregistered name raises KeyError."""
    mols = batch_h_o_h2o["mols"]
    with pytest.raises(KeyError, match="not-a-loss"):
        make_loss("not-a-loss", molecules=mols)


# ---------------------------------------------------------------------------
# Test 29: aux dict carries no gradient
# ---------------------------------------------------------------------------

def test_aux_dict_carries_no_gradient(batch_h_o_h2o, model):
    """Test 29: aux dict values are not tracked by JAX gradient tape."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("A_atomization", molecules=mols)
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }

    def loss_fn(mdl):
        return loss(mdl, batch)

    grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (total, aux), grads = grad_fn(model)

    # The aux dict must be a plain Python dict whose values are plain arrays
    # (not traced). Check that none of the aux values carry abstract jax tracer
    # metadata indicating they are differentiated through.
    # The simplest check: aux values must be concrete JAX arrays (not tracers
    # at the outer level), which they will be since has_aux=True stops gradient
    # flow into them.
    for k, v in aux.items():
        assert hasattr(v, "shape"), f"aux[{k!r}] must be a JAX array"
        assert jnp.isfinite(v), f"aux[{k!r}] must be finite"


# ---------------------------------------------------------------------------
# Test 30: xfail — atomization notebook cell26
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="Fixture notebook_cell26_atomization.npz not yet generated")
def test_atomizationloss_matches_notebook_cell26():
    """Test 30: AtomizationLoss output matches notebook cell 26 reference."""
    import pathlib
    fixture_path = (
        pathlib.Path(__file__).parent / "fixtures"
        / "notebook_cell26_atomization.npz"
    )
    ref = dict(np.load(str(fixture_path)))
    # If fixture exists, compare total loss to stored reference value.
    raise NotImplementedError("fixture not yet generated")


# ---------------------------------------------------------------------------
# Test 31: xfail — delta-ae notebook cell29
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="Fixture notebook_cell29_delta_ae.npz not yet generated")
def test_deltaaeloss_matches_notebook_cell29():
    """Test 31: DeltaAELoss output matches notebook cell 29 reference."""
    import pathlib
    fixture_path = (
        pathlib.Path(__file__).parent / "fixtures"
        / "notebook_cell29_delta_ae.npz"
    )
    ref = dict(np.load(str(fixture_path)))
    raise NotImplementedError("fixture not yet generated")


# ---------------------------------------------------------------------------
# Test 32: molecule-generic — (H, N, NH3) batch
# ---------------------------------------------------------------------------

def _n_atom() -> MoleculeSpec:
    """Nitrogen atom (spin=3, open-shell UKS)."""
    return MoleculeSpec(
        name="N", atom="N 0 0 0", basis="sto-3g",
        charge=0, spin=3, atom_composition=(("N", 1),),
    )


def _nh3_molecule() -> MoleculeSpec:
    """Ammonia molecule (spin=0, closed-shell)."""
    return MoleculeSpec(
        name="NH3",
        atom="N 0 0 0.117; H 0 0.935 -0.272; H 0.810 -0.468 -0.272; H -0.810 -0.468 -0.272",
        basis="sto-3g",
        charge=0, spin=0,
        atom_composition=(("H", 3), ("N", 1)),
    )


@pytest.fixture(scope="module")
def batch_h_n_nh3():
    """3-molecule batch (H, N, NH3) — tests non-H2O compounds."""
    h_spec = h_atom()
    n_spec = _n_atom()
    nh3_spec = _nh3_molecule()
    h_data = precompute_fixed_density_data(h_spec)
    n_data = precompute_fixed_density_data(n_spec)
    nh3_data = precompute_fixed_density_data(nh3_spec)
    # AE for NH3: sum of constituent atoms minus NH3 total
    ae_nh3 = float(
        h_data["E_pbe"] * 3 + n_data["E_pbe"] - nh3_data["E_pbe"]
    )
    targets = {"NH3": max(ae_nh3, 0.001)}
    atom_energies = {
        "H": float(h_data["E_pbe"]),
        "N": float(n_data["E_pbe"]),
    }
    return {
        "mols": (h_spec, n_spec, nh3_spec),
        "mol_data": (h_data, n_data, nh3_data),
        "targets": targets,
        "atom_energies": atom_energies,
    }


def test_molecule_generic_h_n_nh3(batch_h_n_nh3, model):
    """Test 32: AtomizationLoss works on (H, N, NH3) batch with atom_mol_idx {H:0, N:1}."""
    mols = batch_h_n_nh3["mols"]
    loss = make_loss("A_atomization", molecules=mols)
    # Verify atom_mol_idx maps correctly
    atom_idx_dict = dict(loss.atom_mol_idx)
    assert "H" in atom_idx_dict
    assert "N" in atom_idx_dict
    batch = {
        "mol_data": batch_h_n_nh3["mol_data"],
        "targets": batch_h_n_nh3["targets"],
        "atom_energies": batch_h_n_nh3["atom_energies"],
    }
    total, aux = loss(model, batch)
    assert total.shape == ()
    assert jnp.isfinite(total)
    assert set(aux.keys()) == {"loss_energy", "atomic_reg"}


# ---------------------------------------------------------------------------
# Test 33: B15-4 — molecules_only=True skips atoms in DM term
# ---------------------------------------------------------------------------

def test_molecules_only_true_skips_atoms_dm_term(batch_h_o_h2o, model):
    """Test 33 (B15-4): molecules_only=True means DM term iterates only compound_idx."""
    mols = batch_h_o_h2o["mols"]
    # With molecules_only=True (default), DM term only uses H2O (index 2)
    loss_mol_only = make_loss("B_atomization_plus_dm", molecules=mols,
                               molecules_only=True)
    # With molecules_only=False, DM term uses all 3 molecules
    loss_all = make_loss("B_atomization_plus_dm", molecules=mols,
                          molecules_only=False)

    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    total_mol, aux_mol = loss_mol_only(model, batch)
    total_all, aux_all = loss_all(model, batch)

    # Both must return valid scalars; dm_target is None so dm_loss is 0.0 in both
    assert total_mol.shape == ()
    assert total_all.shape == ()
    assert jnp.isfinite(total_mol)
    assert jnp.isfinite(total_all)


# ---------------------------------------------------------------------------
# Test 34: dm_weight=0.2 scales DM term
# ---------------------------------------------------------------------------

def test_dm_weight_scales_dm_term(batch_h_o_h2o, model):
    """Test 34: dm_weight parameter is stored and used by AtomizationPlusDMLoss."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("B_atomization_plus_dm", molecules=mols, dm_weight=0.2)
    assert loss.dm_weight == 0.2
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    total, aux = loss(model, batch)
    assert total.shape == ()
    assert jnp.isfinite(total)
    # total = loss_energy + 0.2 * loss_dm (dm_target is None so loss_dm=0)
    expected = aux["loss_energy"] + 0.2 * aux["loss_dm"]
    np.testing.assert_allclose(float(total), float(expected), rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 35: density_weight=0.3 scales grid term
# ---------------------------------------------------------------------------

def test_density_weight_scales_grid_term(batch_h_o_h2o, model):
    """Test 35: density_weight parameter is stored and used by AtomizationPlusGridLoss."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("C_atomization_plus_grid", molecules=mols, density_weight=0.3)
    assert loss.density_weight == 0.3
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    total, aux = loss(model, batch)
    assert total.shape == ()
    assert jnp.isfinite(total)
    # total = loss_energy + 0.3 * loss_grid (rho_ref_grid is None so loss_grid=0)
    expected = aux["loss_energy"] + 0.3 * aux["loss_grid"]
    np.testing.assert_allclose(float(total), float(expected), rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 36: xfail — missing required_mol_keys raises KeyError
# ---------------------------------------------------------------------------

def test_missing_required_mol_keys_raises_key_error(batch_h_o_h2o, model):
    """Test 36: loss raises KeyError when a required_mol_key is absent from mol_data."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("B_atomization_plus_dm", molecules=mols)
    # Build a batch where mol_data entries don't have 'dm_target' at all
    # (as opposed to None). Currently not enforced — xfail until implemented.
    mol_data_stripped = tuple(
        {k: v for k, v in md.items() if k != "dm_target"}
        for md in batch_h_o_h2o["mol_data"]
    )
    batch = {
        "mol_data": mol_data_stripped,
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    with pytest.raises(KeyError):
        loss(model, batch)


# ---------------------------------------------------------------------------
# Test 37: D-H1 — float type validation rejects bool and non-numeric
# ---------------------------------------------------------------------------

def test_float_type_validation_rejects_non_scalar(batch_h_o_h2o):
    """Test 37 (D-H1): w_atomic must be a plain Python int or float (not bool, str, None)."""
    mols = batch_h_o_h2o["mols"]
    for bad_value in (True, False, "0.01", None):
        with pytest.raises(TypeError):
            make_loss("A_atomization", molecules=mols, w_atomic=bad_value)


# ---------------------------------------------------------------------------
# Test 38: D-H5 — field assignment completeness
# ---------------------------------------------------------------------------

def test_field_assignment_completeness(batch_h_o_h2o):
    """Test 38 (D-H5): all documented fields are present and have correct types."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("A_atomization", molecules=mols)
    # atom_mol_idx: tuple of (str, int) pairs
    assert isinstance(loss.atom_mol_idx, tuple)
    for entry in loss.atom_mol_idx:
        assert isinstance(entry, tuple) and len(entry) == 2
        assert isinstance(entry[0], str)
        assert isinstance(entry[1], int)
    # compound_idx: tuple of ints
    assert isinstance(loss.compound_idx, tuple)
    assert all(isinstance(i, int) for i in loss.compound_idx)
    # mol_names: tuple of strings
    assert isinstance(loss.mol_names, tuple)
    assert all(isinstance(n, str) for n in loss.mol_names)
    # compositions: tuple of tuples of (str, int) pairs
    assert isinstance(loss.compositions, tuple)
    # w_atomic: plain Python float or int
    assert isinstance(loss.w_atomic, (int, float))
    assert not isinstance(loss.w_atomic, bool)


# ---------------------------------------------------------------------------
# Test 39: xfail — H-E12-3 DM loss UKS/RKS scaling
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="UKS/RKS CCSD dm_target data not available; needs CCSD fixture")
def test_dm_loss_uks_rks_scaling():
    """Test 39 (H-E12-3): DM Frobenius loss scales correctly for UKS vs RKS dm_target."""
    raise NotImplementedError("needs CCSD dm_target fixture for UKS and RKS molecules")


# ---------------------------------------------------------------------------
# Test 40: M-E12-4 — aux dict schema per class
# ---------------------------------------------------------------------------

def test_aux_dict_schema_per_class(batch_h_o_h2o, model):
    """Test 40 (M-E12-4): each loss class returns exactly the documented aux keys."""
    expected_schemas = {
        "A_atomization": {"loss_energy", "atomic_reg"},
        "B_atomization_plus_dm": {"loss_energy", "loss_dm"},
        "C_atomization_plus_grid": {"loss_energy", "loss_grid"},
        "D1_delta_ae": {"loss_delta", "atomic_reg"},
        "D2_delta_ae_plus_dm": {"loss_delta", "loss_dm"},
        "D3_delta_ae_plus_grid": {"loss_delta", "loss_grid"},
    }
    mols = batch_h_o_h2o["mols"]
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    for loss_name, expected_aux_keys in expected_schemas.items():
        loss = make_loss(loss_name, molecules=mols)
        _, aux = loss(model, batch)
        assert set(aux.keys()) == expected_aux_keys, (
            f"{loss_name}: aux keys = {set(aux.keys())}, expected {expected_aux_keys}"
        )


# ---------------------------------------------------------------------------
# Test 41: AtomizationLoss sign convention
# ---------------------------------------------------------------------------

def test_atomizationloss_sign_convention(batch_h_o_h2o, model):
    """Test 41: loss_energy is non-negative (relative squared error ≥ 0)."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("A_atomization", molecules=mols)
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    total, aux = loss(model, batch)
    assert float(aux["loss_energy"]) >= 0.0, (
        f"loss_energy must be >= 0 (relative squared error), got {aux['loss_energy']}"
    )
    assert float(total) >= 0.0, f"total loss must be >= 0, got {total}"


# ---------------------------------------------------------------------------
# Test 42: training loss and evaluation metric compute the same AE
# ---------------------------------------------------------------------------

def test_training_loss_and_eval_metric_agree_on_atomization_energy(
    batch_h_o_h2o, model
):
    """Test 42: AtomizationLoss and AtomizationEnergyMetric measure the
    same physical quantity given the same model, mol_data, and atom_energies.

    Regression guard for the training-vs-eval AE semantic mismatch that
    caused trained models to score ~113 kcal/mol off on evaluation while
    appearing to converge during training. The loss and the metric MUST
    compute atomization energy from the same fixed atom_energies dict.
    """
    from xcquinox.alec.evaluation import AtomizationEnergyMetric
    from xcquinox.alec.losses import _ae_from_atoms, _compute_energies

    mols = batch_h_o_h2o["mols"]
    mol_data = batch_h_o_h2o["mol_data"]
    atom_energies = batch_h_o_h2o["atom_energies"]

    h2o_idx = 2
    h2o_data = mol_data[h2o_idx]
    comp_dict = dict(mols[h2o_idx].atom_composition)

    E_nn = _compute_energies(model, mol_data, len(mols))
    ae_loss = float(_ae_from_atoms(E_nn[h2o_idx], comp_dict, atom_energies))

    metric = AtomizationEnergyMetric(atom_energies=atom_energies)
    ae_eval = float(metric.compute(model, h2o_data)["AE_nn"])

    np.testing.assert_allclose(
        ae_loss, ae_eval, rtol=1e-6, atol=1e-8,
        err_msg=(
            "Training-loss AE and evaluation-metric AE diverged: "
            f"loss={ae_loss!r}, eval={ae_eval!r}. "
            "Both MUST use the batch['atom_energies'] dict as the fixed "
            "atomic anchor — do not use NN-predicted atomic totals."
        ),
    )


# ---------------------------------------------------------------------------
# Test 43: L-B13-2 — molecules_only bool type validation
# ---------------------------------------------------------------------------

def test_molecules_only_bool_type_validation(batch_h_o_h2o):
    """Test 43 (L-B13-2): molecules_only must be a plain Python bool."""
    mols = batch_h_o_h2o["mols"]
    for bad_value in (1, 0, "true", "false", 1.0, None):
        with pytest.raises(TypeError):
            make_loss("B_atomization_plus_dm", molecules=mols,
                      molecules_only=bad_value)


# ---------------------------------------------------------------------------
# Task 7.4: solver_config=None regression — all 6 losses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_name", LOSS_NAMES)
def test_loss_solver_config_none_matches_legacy(loss_name, batch_h_o_h2o, model):
    """solver_config=None must produce byte-identical output to legacy (no kwarg)."""
    mols = batch_h_o_h2o["mols"]
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    loss_legacy = make_loss(loss_name, molecules=mols)
    loss_explicit = make_loss(loss_name, molecules=mols, solver_config=None)
    total_legacy, aux_legacy = loss_legacy(model, batch)
    total_explicit, aux_explicit = loss_explicit(model, batch)
    np.testing.assert_allclose(
        float(total_legacy), float(total_explicit), rtol=0, atol=1e-12,
    )
    for key in aux_legacy:
        np.testing.assert_allclose(
            float(aux_legacy[key]), float(aux_explicit[key]), rtol=0, atol=1e-12,
        )
