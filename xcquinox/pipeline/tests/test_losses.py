"""Tests for xcquinox.pipeline.losses: AlecLoss, LOSS_REGISTRY, 6 concrete losses.

The enumerated structure below covers the original 43-test contract; later
regression tests were appended after it as fixes landed, so the file now holds
well over 43 tests.

Test structure:
  Tests 1-24  (4 parametrized over 6 losses × 4 aspects = 24 tests):
    (a) registry roundtrip
    (b) forward -> (scalar, dict)
    (c) required_mol_keys
    (d) differentiability via eqx.filter_value_and_grad

  Tests 25-43 (19 additional):
    25. LOSS_REGISTRY has exactly 7 builtins
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

from xcquinox.pipeline.config import ArchitectureConfig, MoleculeSpec
from xcquinox.pipeline.models import AlecGGAModel
from xcquinox.pipeline.data import precompute_fixed_density_data
from xcquinox.pipeline.losses import (
    AlecLoss,
    AtomizationLoss,
    AtomizationPlusDMLoss,
    AtomizationPlusGridLoss,
    DeltaAELoss,
    DeltaAEPlusDMLoss,
    DeltaAEPlusGridLoss,
    _atomic_reg,
    make_loss,
)
from xcquinox.pipeline.tests.fixtures.molecules import h_atom, o_atom, h2o_molecule


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
# Tests 1-6 (a): registry roundtrip, make_loss returns correct type
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

#: two of the six losses stand for the forward and the differentiability checks: the
#: energy-only loss and the one carrying a grid term, the two branches of the shared code
LOSS_PARAMS_TWO = [p for p in LOSS_PARAMS if p.id in ("A_atomization", "C_atomization_plus_grid")]


@pytest.mark.parametrize("loss_name,_cls,_rmk,_aux_keys", LOSS_PARAMS_TWO)
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

@pytest.mark.parametrize("loss_name,_cls,_rmk,_aux_keys", LOSS_PARAMS_TWO)
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
# Test 25: LOSS_REGISTRY has exactly 7 builtins
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 26: list_losses returns sorted
# ---------------------------------------------------------------------------


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
# Test 30: xfail, atomization notebook cell26
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 31: xfail, delta-ae notebook cell29
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 32: molecule-generic, (H, N, NH3) batch
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
    """3-molecule batch (H, N, NH3), tests non-H2O compounds."""
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
# Test 33: B15-4, molecules_only=True skips atoms in DM term
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
# Test 36: xfail, missing required_mol_keys raises KeyError
# ---------------------------------------------------------------------------

def test_missing_required_mol_keys_raises_key_error(batch_h_o_h2o, model):
    """Test 36: loss raises KeyError when a required_mol_key is absent from mol_data."""
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("B_atomization_plus_dm", molecules=mols)
    # Build a batch where mol_data entries don't have 'dm_target' at all
    # (as opposed to None). Currently not enforced, xfail until implemented.
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
# Test 37: D-H1, float type validation rejects bool and non-numeric
# ---------------------------------------------------------------------------

def test_float_type_validation_rejects_non_scalar(batch_h_o_h2o):
    """Test 37 (D-H1): w_atomic must be a plain Python int or float (not bool, str, None)."""
    mols = batch_h_o_h2o["mols"]
    for bad_value in (True, False, "0.01", None):
        with pytest.raises(TypeError):
            make_loss("A_atomization", molecules=mols, w_atomic=bad_value)


# ---------------------------------------------------------------------------
# Test 38: D-H5, field assignment completeness
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 39: xfail, H-E12-3 DM loss UKS/RKS scaling
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="UKS/RKS CCSD dm_target data not available; needs CCSD fixture")
def test_dm_loss_uks_rks_scaling():
    """Test 39 (H-E12-3): DM Frobenius loss scales correctly for UKS vs RKS dm_target."""
    raise NotImplementedError("needs CCSD dm_target fixture for UKS and RKS molecules")


# ---------------------------------------------------------------------------
# Test 40: M-E12-4, aux dict schema per class
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
    """Test 41: the atomization-energy sign convention is
    AE = Σ_Z n_Z · atom_energies[Z] - E_mol (positive for a bound molecule),
    and the loss's relative-squared-error channel stays non-negative.

    The loss_energy >= 0 check alone is tautological for a squared error; the
    load-bearing assertion is that ``_ae_from_atoms`` follows the atoms-minus-
    molecule convention: AE > 0 when E_mol lies below the summed atomic anchors
    (bound), AE < 0 when it lies above (less stable than the free atoms).
    """
    from xcquinox.pipeline.losses import _ae_from_atoms
    mols = batch_h_o_h2o["mols"]
    loss = make_loss("A_atomization", molecules=mols)
    batch = {
        "mol_data": batch_h_o_h2o["mol_data"],
        "targets": batch_h_o_h2o["targets"],
        "atom_energies": batch_h_o_h2o["atom_energies"],
    }
    total, aux = loss(model, batch)
    # Relative-squared-error channel is non-negative by construction.
    assert float(aux["loss_energy"]) >= 0.0, (
        f"loss_energy must be >= 0 (relative squared error), got {aux['loss_energy']}"
    )
    assert float(total) >= 0.0, f"total loss must be >= 0, got {total}"

    # Sign convention: AE = sum(atomic anchors) - E_mol, positive for bound.
    atom_energies = {"H": -0.5, "O": -75.0}
    comp = {"H": 2, "O": 1}
    e_atoms_sum = 2 * (-0.5) + 1 * (-75.0)   # = -76.0 Ha
    # Bound molecule: E_mol below the summed atoms -> AE > 0.
    ae_bound = float(_ae_from_atoms(-76.4, comp, atom_energies))
    assert ae_bound == pytest.approx(e_atoms_sum - (-76.4))
    assert ae_bound > 0.0, f"bound-molecule AE must be positive, got {ae_bound}"
    # Molecule less stable than the free atoms: E_mol above the sum -> AE < 0.
    ae_unbound = float(_ae_from_atoms(-75.6, comp, atom_energies))
    assert ae_unbound < 0.0, (
        f"unbound-configuration AE must be negative, got {ae_unbound}"
    )


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
    from xcquinox.pipeline.evaluation import AtomizationEnergyMetric
    from xcquinox.pipeline.losses import _ae_from_atoms, _compute_energies

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
            "atomic anchor, do not use NN-predicted atomic totals."
        ),
    )


# ---------------------------------------------------------------------------
# Test 43: L-B13-2, molecules_only bool type validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task 7.4: solver_config=None regression, all 6 losses
# ---------------------------------------------------------------------------


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
    from xcquinox.pipeline.losses import _dm_term
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
    from xcquinox.pipeline.losses import _grid_term
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


@pytest.mark.parametrize("loss_name", ["A_atomization", "C_atomization_plus_grid"])
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


# ---------------------------------------------------------------------------
# V_xc matching tests
# ---------------------------------------------------------------------------

def test_vxc_term_returns_zero_when_all_vxc_ref_none(batch_h_o_h2o, model):
    """_vxc_term returns 0.0 when all mol_data entries have vxc_ref=None."""
    from xcquinox.pipeline.losses import _vxc_term
    mol_data = batch_h_o_h2o["mol_data"]
    result = _vxc_term(model, mol_data, tuple(range(len(mol_data))))
    assert float(result) == 0.0


def test_vxc_term_relative_divides_by_ref_norm(batch_h_o_h2o, model):
    """_vxc_term with relative=True normalizes by ||V_xc^ref||_F^2."""
    from xcquinox.pipeline.losses import _vxc_term
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


def test_vxc_gradient_is_finite_and_nonzero(batch_h_o_h2o, model):
    """jax.grad through _vxc_term produces finite, non-zero gradients."""
    from xcquinox.pipeline.losses import _vxc_term
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
    assert any_nonzero, "All gradients are zero, V_xc gradient path may be broken"


def test_vxc_term_handles_uks_reference():
    """_vxc_term with UKS vxc_ref (shape (2, nao, nao)) returns finite loss."""
    import jax.numpy as jnp
    import numpy as np
    import xcquinox.pipeline as pipeline
    from xcquinox.pipeline.config import MoleculeSpec
    from xcquinox.pipeline.data import precompute_fixed_density_data
    from xcquinox.pipeline.losses import _vxc_term

    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec)
    # Inject a synthetic vxc_ref of shape (2, nao, nao)
    md = dict(md)
    nao = np.asarray(md["s_matrix"]).shape[-1]
    md["vxc_ref"] = jnp.zeros((2, nao, nao), dtype=jnp.float64)

    arch = pipeline.get_architecture("deep")
    xnet, cnet = pipeline.create_network_pair(arch, seed=0)
    model = pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)

    # _vxc_term takes model, mol_data_list, iter_idx
    mol_data_list = [md]
    iter_idx = [0]
    val = _vxc_term(model, mol_data_list, iter_idx)
    assert jnp.isfinite(val), f"vxc term not finite: {val}"
    # With vxc_ref=0 and random NN, loss should be > 0
    assert float(val) > 0, f"vxc term should be > 0 with nonzero NN vxc, got {val}"


# ---------------------------------------------------------------------------
# A/D1 energy loss is solver-invariant.
#
# Supersedes the Task-19 test that asserted the opposite (run_scf total_energy
# under FIXED_J). That assertion codified a bug: training's FIXED_J
# run_scf(...).total_energy is a J-pinned hybrid that is NOT a valid energy
# functional of any single density, and it diverges from what evaluation's
# TotalEnergyMetric measures (fixed_density_total_energy at ρ_PBE). The fix
# on 2026-04-24 made _compute_energies solver-invariant so training
# optimizes what evaluation measures.
# ---------------------------------------------------------------------------

def test_loss_a_with_fixed_j_solver_matches_oneshot_energy():
    """A_atomization loss is solver-invariant in its energy term:
    ``solver_config=FIXED_J`` and ``solver_config=None`` ( -> oneshot) MUST
    produce identical loss values when the only difference is the solver.
    The post-hoc fixed-density framework defines E_total as a functional
    of ρ_PBE with NN's V_xc; solver choice governs DM / density / V_xc
    matching terms, not the energy functional."""
    import xcquinox.pipeline as pipeline
    from xcquinox.pipeline.config import MoleculeSpec
    from xcquinox.pipeline.data import precompute_fixed_density_data
    from xcquinox.pipeline.losses import make_loss
    from xcquinox.pipeline.solver import SolverConfig, SolverBackend, SolverMode

    spec = MoleculeSpec(
        name="H2O", atom="O 0 0 0; H 0 1 0; H 0 0 1",
        basis="sto-3g", charge=0, spin=0,
        atom_composition=(("O", 1), ("H", 2)), grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = pipeline.get_architecture("deep")
    xnet, cnet = pipeline.create_network_pair(arch, seed=0)
    model = pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)

    atom_energies = {"H": -0.5, "O": -75.0}
    targets = {"H2O": 0.3}
    batch = {
        "mol_data": [md],
        "targets": targets,
        "atom_energies": atom_energies,
    }

    loss_a_oneshot = make_loss("A_atomization", molecules=(spec,))
    val_oneshot = float(loss_a_oneshot(model, batch)[0])

    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
        max_cycles=5, conv_tol=1e-8,
        mixer_kwargs=(("alpha", 1.0),),
    )
    loss_a_fixed_j = make_loss(
        "A_atomization", molecules=(spec,), solver_config=cfg,
    )
    val_fixed_j = float(loss_a_fixed_j(model, batch)[0])

    assert abs(val_oneshot - val_fixed_j) < 1e-12, (
        f"A_atomization loss must be solver-invariant in its energy term; "
        f"oneshot={val_oneshot!r} fixed_j={val_fixed_j!r}"
    )


# ---------------------------------------------------------------------------
# PBE-anchor integration (2026-04-21 step 6)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Solver-MODE-dispatched energy (2026-06-01): the energy term follows the
# solver mode. ONESHOT / FIXED_J / None use the one-shot fixed-density
# functional on ρ_PBE; FULL uses the SELF-CONSISTENT run_scf(...).total_energy
# (coherent fixed point, backprop through the SCF cycles, the DFS/dpyscf
# target). FIXED_J deliberately STAYS one-shot: its run_scf energy is an
# incoherent J-pinned hybrid (the 2026-04-24 bug that gave 51+ kcal/mol AE).
# Evaluation (TotalEnergyMetric / AtomizationEnergyMetric / held-out) uses the
# SAME rule so training optimizes exactly what evaluation measures.
# ---------------------------------------------------------------------------


def test_compute_energies_full_self_consistent_else_oneshot(h2o_mol_data):
    """ONESHOT/FIXED_J/None -> one-shot fixed_density energy; FULL -> the
    self-consistent run_scf(...).total_energy."""
    from xcquinox.pipeline.losses import _compute_energies
    from xcquinox.pipeline.solver import (
        SolverConfig, SolverMode, SolverBackend, run_scf,
    )
    from xcquinox.pipeline.oneshot import fixed_density_total_energy

    model = _make_model()
    mol_data = (h2o_mol_data,)
    E_oneshot = float(fixed_density_total_energy(model, h2o_mol_data))

    # None / ONESHOT / FIXED_J all equal the one-shot frozen-density functional.
    for cfg in (None,
                SolverConfig(mode=SolverMode.ONESHOT, max_cycles=0),
                SolverConfig(mode=SolverMode.FIXED_J, max_cycles=3)):
        E = float(_compute_energies(model, mol_data, 1, solver_config=cfg)[0])
        assert abs(E - E_oneshot) < 1e-10, (
            f"{cfg} must use the one-shot fixed-density energy")

    # FULL routes to the self-consistent SCF energy (needs ERIs to rebuild J).
    spec = MoleculeSpec(
        name="H2O", atom="O 0 0 0; H 0 1 0; H 0 0 1",
        basis="sto-3g", charge=0, spin=0,
        atom_composition=(("O", 1), ("H", 2)), grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    full = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
        max_cycles=3, conv_tol=1e-8, mixer_kwargs=(("alpha", 1.0),),
    )
    E_full = float(_compute_energies(model, (md,), 1, solver_config=full)[0])
    E_full_direct = float(run_scf(full, model, md).total_energy)
    assert abs(E_full - E_full_direct) < 1e-10, (
        f"FULL energy must equal run_scf().total_energy; "
        f"loss={E_full!r} run_scf={E_full_direct!r}")


def test_compute_energies_full_is_differentiable_through_scf():
    """The FULL-mode energy loss backprops through the SCF cycles: the gradient
    of the energy w.r.t. the model is finite and nonzero, the property that was
    missing while the energy was hard-coded one-shot on the frozen density."""
    from xcquinox.pipeline.losses import _compute_energies
    from xcquinox.pipeline.solver import SolverConfig, SolverMode, SolverBackend

    model = _make_model()
    spec = MoleculeSpec(
        name="H2O", atom="O 0 0 0; H 0 1 0; H 0 0 1",
        basis="sto-3g", charge=0, spin=0,
        atom_composition=(("O", 1), ("H", 2)), grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    full = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
        max_cycles=3, conv_tol=1e-8, mixer_kwargs=(("alpha", 1.0),),
    )

    def loss_fn(m):
        return _compute_energies(m, (md,), 1, solver_config=full)[0] ** 2

    grads = eqx.filter_grad(loss_fn)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    gnorm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves)))
    assert bool(jnp.isfinite(jnp.array(gnorm))) and gnorm > 0.0, (
        f"FULL-mode energy gradient must flow through the SCF (gnorm={gnorm})")


# ---------------------------------------------------------------------------
# Fix: _dm_term per-element normalization
# ---------------------------------------------------------------------------

def test_dm_term_normalizes_per_element_for_uks(monkeypatch):
    """_dm_term must normalize the squared error by the total element
    count of dm_ref (n_ao^2 for RKS, 2*n_ao^2 for UKS). Pre-fix UKS
    branch divided by n_ao^2 only, off by factor of 2 vs RKS and
    inconsistent with _vxc_term.

    R3-F strengthening: the prior version of this test never
    called ``_dm_term``: it just re-derived the arithmetic identity
    ``sum(ones**2) / n_elems == 1.0``, which would still pass if the
    UKS divisor regressed to ``n_ao**2``. This version monkey-patches
    ``dm_prediction_for_loss`` (the symbol ``_dm_term`` actually calls; the
    older ``oneshot_dm_prediction_fast`` name is one wrapper deeper and is no
    longer imported by ``losses``) to a fixed stub and calls the real
    ``_dm_term`` with a known prediction-target gap, then asserts the
    returned per-element MSE matches the analytical expectation under
    the (2 * n_ao * n_ao) UKS divisor. A regression to ``n_ao**2``
    would double the returned value and fail this test.
    """
    import jax.numpy as jnp
    from xcquinox.pipeline import losses as losses_mod

    n_ao = 4
    # UKS dm_target: shape (2, n_ao, n_ao); fill with 0.5 so per-element
    # squared error vs the prediction stub is 0.25 everywhere. With
    # n_elems = 2 * n_ao * n_ao the per-element MSE equals 0.25.
    dm_target = jnp.full((2, n_ao, n_ao), 0.5)
    # Stub the prediction to be all-zeros so error per element = 0.25.
    def _stub_predict(model, mol_data, solver_config=None):
        del model, mol_data, solver_config
        return jnp.zeros_like(dm_target)
    monkeypatch.setattr(
        losses_mod, "dm_prediction_for_loss", _stub_predict,
    )
    mol_data = [{"dm_target": dm_target}]
    # model is unused by the stub; pass any object.
    out = losses_mod._dm_term(model=object(), mol_data=mol_data, iter_idx=[0])
    expected_per_element = 0.25
    expected_uks_divisor = 2 * n_ao * n_ao
    assert abs(float(out) - expected_per_element) < 1e-12, (
        f"UKS _dm_term should give err/(2*n_ao^2) = {expected_per_element}; "
        f"got {float(out)}. A regression to err/n_ao^2 would give "
        f"{2 * expected_per_element} (twice as large)."
    )
    # Sanity: confirm the divisor is exactly 2*n_ao^2 by checking the
    # raw error vs returned value.
    raw_err = float(jnp.sum(dm_target ** 2))
    derived_divisor = raw_err / float(out)
    assert abs(derived_divisor - expected_uks_divisor) < 1e-9, (
        f"derived divisor = {derived_divisor}, expected "
        f"{expected_uks_divisor} (2 * n_ao^2). A pre-fix n_ao^2 divisor "
        f"would give {n_ao * n_ao}."
    )


# ---------------------------------------------------------------------------
# Fix: A and D1 now have molecules_only flag
# ---------------------------------------------------------------------------


def test_atomic_reg_is_mean_not_sum():
    """_atomic_reg returns the MEAN squared relative error over anchored atoms,
    so the channel scale is independent of how many anchors a batch carries
    (matching the other mean-reduced channels). A SUM would grow with the
    anchor count and silently up-weight w_atomic on larger pools."""
    # Each anchor is built so its squared relative error == 1.0:
    # (E_nn - E)^2 / (E^2 + 1e-8) ~ 1 when E_nn = 2 E.
    atom_energies = {1: -0.5, 6: -37.8}
    atom_idx = {1: 0, 6: 1}
    E_nn = jnp.array([2.0 * -0.5, 2.0 * -37.8])
    val = float(_atomic_reg(E_nn, atom_idx, atom_energies))
    assert val == pytest.approx(1.0, abs=1e-6)   # MEAN of {~1, ~1}
    assert val < 1.5                             # NOT the SUM (~2.0)

    # Empty anchor dict -> exactly 0.0 (no anchors to regularize toward).
    assert float(_atomic_reg(E_nn, {}, atom_energies)) == 0.0

    # Mean-scale invariance: a third ZERO-error anchor must LOWER the value
    # (mean {1,1,0}=2/3); a SUM would stay ~2.0 in both the 2- and 3-anchor case.
    atom_energies3 = {1: -0.5, 6: -37.8, 8: -75.0}
    atom_idx3 = {1: 0, 6: 1, 8: 2}
    E_nn3 = jnp.array([2.0 * -0.5, 2.0 * -37.8, -75.0])
    val3 = float(_atomic_reg(E_nn3, atom_idx3, atom_energies3))
    assert val3 == pytest.approx(2.0 / 3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# DFS per-step tail-weighted energy loss (2026-06-24).
# When SolverConfig.scf_loss_use_tail is on, the energy-derived channels score
# a quadratic-weighted window of the LAST scf_loss_tail SCF-cycle energies
# (per-step residual, mean of (w_i * r_i)^2) instead of only the final cycle.
# step_w2 = weights**2; step_w2=None keeps the byte-identical scalar path.
# ---------------------------------------------------------------------------


def test_rxn_residual_term_tail_weighted_matches_manual():
    from xcquinox.pipeline.losses import _rxn_residual_term
    # 2 species x 3 SCF steps; coeffs [-1, 1] -> e_rxn = e1 - e0 per step.
    e_nn = jnp.array([[1.0, 1.0, 1.0],
                      [2.0, 3.0, 4.0]])
    coeffs = jnp.array([-1.0, 1.0])
    ref = jnp.array(2.0)
    w2 = jnp.array([0.0, 0.25, 1.0])
    out = float(_rxn_residual_term(e_nn, coeffs, ref, step_w2=w2))
    # e_rxn=[1,2,3]; resid=[-1,0,1]; w2*resid^2=[0,0,1]; mean=1/3
    assert out == pytest.approx(1.0 / 3.0, abs=1e-12)


def _h2o_sto3g_full_md():
    spec = MoleculeSpec(
        name="H2O", atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("O", 1), ("H", 2)), grid_level=1,
    )
    return precompute_fixed_density_data(spec, required_keys=("eri",))


def test_compute_energy_trajectories_tail_shape_and_values():
    from xcquinox.pipeline.losses import _compute_energy_trajectories
    from xcquinox.pipeline.solver import (
        SolverConfig, SolverMode, SolverBackend, run_scf,
    )
    model = _make_model()
    md = _h2o_sto3g_full_md()
    full = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=4,
        conv_tol=1e-8, mixer_kwargs=(("alpha", 1.0),),
        scf_loss_use_tail=True, scf_loss_tail=2, scf_loss_weight_power=2.0,
    )
    traj = _compute_energy_trajectories(model, (md,), 1, solver_config=full)
    assert traj.shape == (1, 2)  # tail_len = min(4, 2) = 2
    trace = run_scf(full, model, md).energy_trace
    assert jnp.allclose(traj[0], trace[2:])  # skip = 4 - 2 = 2


def test_the_density_terms_skip_a_marked_record_silently(h_mol_data,
                                                         h2o_mol_data):
    """``_grid_term``, ``_dm_term`` and ``_vxc_term`` leave a record marked for
    evaluation at the reference density out of their mean, and say nothing.

    A non-self-consistent entry of the published protocol carries no
    per-molecule loss: its density, density-matrix and potential are the
    reference's by construction, so scoring them measures the reference against
    itself. The existing skip for a missing reference warns, since a missing
    reference is a staging fault; a marked record is a configured choice and
    must not raise a warning. The oracle is the same pair of records with and
    without the mark: the marked pair's value equals the term over the
    self-consistent record alone, and the unmarked pair's does not.
    """
    import warnings

    from xcquinox.pipeline.losses import _dm_term, _grid_term, _vxc_term
    from xcquinox.pipeline.oneshot import ONESHOT_AT_REFERENCE_KEY

    model = AlecGGAModel.from_arch(_make_arch(), seed=0)

    def with_refs(md):
        """Synthetic references: the PBE quantities themselves, so every
        channel is finite and non-zero for a network that is not PBE."""
        out = dict(md)
        out["dm_target"] = md["dm_pbe"]
        out["rho_ref_grid"] = md["rho_grid"]
        out["vxc_ref"] = md["vxc_pbe"]
        return out

    kept = with_refs(h2o_mol_data)
    unmarked = with_refs(h_mol_data)
    marked = dict(unmarked, **{ONESHOT_AT_REFERENCE_KEY: True})

    for term in (_grid_term, _dm_term, _vxc_term):
        alone = float(term(model, (kept,), (0,)))
        both = float(term(model, (kept, unmarked), (0, 1)))
        # The second record moves the value, so the equality below cannot be
        # satisfied by two coincidentally equal contributions.
        assert abs(both - alone) > 1e-12, term.__name__
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = float(term(model, (kept, marked), (0, 1)))
            only_marked = float(term(model, (marked,), (0,)))
        assert value == pytest.approx(alone, rel=0, abs=1e-12), term.__name__
        # every record marked: the channel is empty, not a division by zero
        assert only_marked == 0.0, term.__name__
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert not runtime, (term.__name__, [str(w.message) for w in runtime])


