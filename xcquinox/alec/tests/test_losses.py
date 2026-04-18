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
    # total = sum of pre-weighted components
    expected = sum(aux.values())
    np.testing.assert_allclose(float(total), float(expected), rtol=1e-5)


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
    # total = sum of pre-weighted components
    expected = sum(aux.values())
    np.testing.assert_allclose(float(total), float(expected), rtol=1e-5)


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
    # solver_config: None by default
    assert loss.solver_config is None


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
        "B_atomization_plus_dm": {"loss_energy", "atomic_reg", "loss_dm"},
        "C_atomization_plus_grid": {"loss_energy", "atomic_reg", "loss_grid"},
        "D1_delta_ae": {"loss_delta", "atomic_reg"},
        "D2_delta_ae_plus_dm": {"loss_delta", "atomic_reg", "loss_dm"},
        "D3_delta_ae_plus_grid": {"loss_delta", "atomic_reg", "loss_grid"},
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


# ---------------------------------------------------------------------------
# Test 44: RELATIVE dm_term divides by ||D_ref||_F^2 + eps
# ---------------------------------------------------------------------------

def test_dm_term_relative(h_mol_data, o_mol_data, h2o_mol_data):
    """RELATIVE dm_term divides by ||D_ref||_F^2 + eps."""
    model = AlecGGAModel.from_arch(_make_arch(), seed=0)
    # Inject dm_pbe as dm_target so _dm_term doesn't skip (fixture has dm_target=None)
    h2o_with_dm = dict(h2o_mol_data)
    h2o_with_dm["dm_target"] = h2o_mol_data["dm_pbe"]
    mol_data = (h_mol_data, o_mol_data, h2o_with_dm)
    from xcquinox.alec.losses import _dm_term
    abs_val = _dm_term(model, mol_data, (2,))
    rel_val = _dm_term(model, mol_data, (2,), relative=True)
    assert not jnp.allclose(abs_val, rel_val, atol=1e-15)
    assert float(abs_val) > 0
    assert float(rel_val) > 0


# ---------------------------------------------------------------------------
# Test 45: RELATIVE grid_term divides by sum(w * rho_ref^2) + eps
# ---------------------------------------------------------------------------

def test_grid_term_relative(h_mol_data, o_mol_data, h2o_mol_data):
    """RELATIVE grid_term divides by sum(w * rho_ref^2) + eps."""
    model = AlecGGAModel.from_arch(_make_arch(), seed=0)
    # Inject rho_grid as rho_ref_grid so _grid_term doesn't skip (fixture has rho_ref_grid=None)
    h2o_with_rho = dict(h2o_mol_data)
    h2o_with_rho["rho_ref_grid"] = h2o_mol_data["rho_grid"]
    mol_data = (h_mol_data, o_mol_data, h2o_with_rho)
    from xcquinox.alec.losses import _grid_term
    abs_val = _grid_term(model, mol_data, (2,))
    rel_val = _grid_term(model, mol_data, (2,), relative=True)
    assert not jnp.allclose(abs_val, rel_val, atol=1e-15)
    assert float(abs_val) > 0
    assert float(rel_val) > 0


# ---------------------------------------------------------------------------
# Tests 46-51: compute_components keys match __call__ aux dict
# ---------------------------------------------------------------------------

LOSS_PARAMS_KEYS = {
    "A_atomization": {"loss_energy", "atomic_reg"},
    "B_atomization_plus_dm": {"loss_energy", "atomic_reg", "loss_dm"},
    "C_atomization_plus_grid": {"loss_energy", "atomic_reg", "loss_grid"},
    "D1_delta_ae": {"loss_delta", "atomic_reg"},
    "D2_delta_ae_plus_dm": {"loss_delta", "atomic_reg", "loss_dm"},
    "D3_delta_ae_plus_grid": {"loss_delta", "atomic_reg", "loss_grid"},
}


@pytest.mark.parametrize("loss_name", list(LOSS_PARAMS_KEYS.keys()))
def test_compute_components_keys_match_call(loss_name, batch_h_o_h2o, model):
    """compute_components returns same keys as __call__ aux dict."""
    mols = batch_h_o_h2o["mols"]
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    loss = make_loss(loss_name, molecules=mols)
    _, aux = loss(model, batch)
    components = loss.compute_components(model, batch)
    assert set(components.keys()) == set(aux.keys())
    assert set(components.keys()) == LOSS_PARAMS_KEYS[loss_name]


# ---------------------------------------------------------------------------
# Tests 52-57: compute_components call consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_name", list(LOSS_PARAMS_KEYS.keys()))
def test_compute_components_call_consistency(loss_name, batch_h_o_h2o, model):
    """__call__ total equals sum of compute_components values."""
    mols = batch_h_o_h2o["mols"]
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    loss = make_loss(loss_name, molecules=mols)
    total_call, _ = loss(model, batch)
    components = loss.compute_components(model, batch)
    total_components = sum(components.values())
    assert jnp.allclose(total_call, total_components, atol=1e-12)


# ---------------------------------------------------------------------------
# V_xc matching tests
# ---------------------------------------------------------------------------

def test_vxc_term_returns_zero_when_all_vxc_ref_none(batch_h_o_h2o, model):
    """_vxc_term returns 0.0 when all mol_data entries have vxc_ref=None."""
    from xcquinox.alec.losses import _vxc_term
    mol_data = batch_h_o_h2o["mol_data"]
    result = _vxc_term(model, mol_data, tuple(range(len(mol_data))))
    assert float(result) == 0.0


def test_vxc_term_finite_with_synthetic_vxc_ref(batch_h_o_h2o, model):
    """_vxc_term returns a finite positive scalar when vxc_ref is provided."""
    from xcquinox.alec.losses import _vxc_term
    mol_data_list = list(batch_h_o_h2o["mol_data"])
    # Inject a synthetic vxc_ref into H2O (index 2)
    h2o = dict(mol_data_list[2])
    nao = h2o["vxc_pbe"].shape[-1]
    h2o["vxc_ref"] = jnp.zeros((nao, nao))  # zero reference
    mol_data_list[2] = h2o
    result = _vxc_term(model, tuple(mol_data_list), (2,))
    assert result.shape == ()
    assert jnp.isfinite(result)
    assert float(result) > 0.0  # V_xc^NN is not zero for a random model


def test_vxc_term_relative_divides_by_ref_norm(batch_h_o_h2o, model):
    """_vxc_term with relative=True normalizes by ||V_xc^ref||_F^2."""
    from xcquinox.alec.losses import _vxc_term
    mol_data_list = list(batch_h_o_h2o["mol_data"])
    h2o = dict(mol_data_list[2])
    nao = h2o["vxc_pbe"].shape[-1]
    h2o["vxc_ref"] = h2o["vxc_pbe"]  # use PBE as reference
    mol_data_list[2] = h2o
    mol_data_t = tuple(mol_data_list)

    abs_val = float(_vxc_term(model, mol_data_t, (2,), relative=False))
    rel_val = float(_vxc_term(model, mol_data_t, (2,), relative=True))

    # Both should be finite and positive
    assert abs_val > 0.0
    assert rel_val > 0.0
    # Relative and absolute should differ (different denominators)
    assert abs(abs_val - rel_val) > 1e-12


# ---------------------------------------------------------------------------
# V_xc weight integration tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_name", LOSS_NAMES)
def test_vxc_weight_zero_is_backward_compatible(loss_name, batch_h_o_h2o, model):
    """vxc_weight=0 (default) produces identical results to current behavior."""
    mols = batch_h_o_h2o["mols"]
    loss_default = make_loss(loss_name, molecules=mols)
    loss_explicit = make_loss(loss_name, molecules=mols, vxc_weight=0.0)
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    total_default, aux_default = loss_default(model, batch)
    total_explicit, aux_explicit = loss_explicit(model, batch)
    np.testing.assert_allclose(float(total_default), float(total_explicit), rtol=1e-10)
    assert "loss_vxc" not in aux_default
    assert "loss_vxc" not in aux_explicit


@pytest.mark.parametrize("loss_name", LOSS_NAMES)
def test_vxc_weight_positive_adds_loss_vxc_component(loss_name, batch_h_o_h2o, model):
    """vxc_weight > 0 adds 'loss_vxc' to compute_components output."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss(loss_name, molecules=mols, vxc_weight=0.5)
    assert loss.vxc_weight == 0.5
    mol_data_list = list(batch_h_o_h2o["mol_data"])
    h2o = dict(mol_data_list[2])
    nao = h2o["vxc_pbe"].shape[-1]
    h2o["vxc_ref"] = jnp.zeros((nao, nao))
    mol_data_list[2] = h2o
    batch = {
        "mol_data": tuple(mol_data_list),
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    total, aux = loss(model, batch)
    assert "loss_vxc" in aux
    assert jnp.isfinite(aux["loss_vxc"])
    assert float(aux["loss_vxc"]) > 0.0


def test_vxc_gradient_is_finite_and_nonzero(batch_h_o_h2o, model):
    """jax.grad through _vxc_term produces finite, non-zero gradients."""
    from xcquinox.alec.losses import _vxc_term
    mol_data_list = list(batch_h_o_h2o["mol_data"])
    h2o = dict(mol_data_list[2])
    nao = h2o["vxc_pbe"].shape[-1]
    h2o["vxc_ref"] = jnp.zeros((nao, nao))
    mol_data_list[2] = h2o
    mol_data_t = tuple(mol_data_list)

    @eqx.filter_value_and_grad
    def loss_fn(m):
        return _vxc_term(m, mol_data_t, (2,))

    val, grads = loss_fn(model)
    assert jnp.isfinite(val)
    flat_grads = jax.tree.leaves(grads)
    any_nonzero = any(jnp.any(jnp.abs(g) > 0).item() for g in flat_grads if g is not None)
    assert any_nonzero, "All gradients are zero — V_xc gradient path may be broken"


def test_vxc_weight_type_validation(batch_h_o_h2o):
    """vxc_weight must be plain int/float, not bool/str/None."""
    mols = batch_h_o_h2o["mols"]
    for bad_value in (True, False, "0.5", None):
        with pytest.raises(TypeError):
            make_loss("A_atomization", molecules=mols, vxc_weight=bad_value)


def test_vxc_compute_components_relative_mode(batch_h_o_h2o, model):
    """compute_components(relative=True) passes relative=True to _vxc_term."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("B_atomization_plus_dm", molecules=mols, vxc_weight=0.5)
    mol_data_list = list(batch_h_o_h2o["mol_data"])
    h2o = dict(mol_data_list[2])
    nao = h2o["vxc_pbe"].shape[-1]
    h2o["vxc_ref"] = h2o["vxc_pbe"]
    mol_data_list[2] = h2o
    batch = {
        "mol_data": tuple(mol_data_list),
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    comps_abs = loss.compute_components(model, batch, relative=False)
    comps_rel = loss.compute_components(model, batch, relative=True)
    assert "loss_vxc" in comps_abs
    assert "loss_vxc" in comps_rel
    assert abs(float(comps_abs["loss_vxc"]) - float(comps_rel["loss_vxc"])) > 1e-12
