# Exact Spin Scaling for Every Density-Matrix Feature -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every UKS exchange evaluation in the library receive the descriptor feature block of the symmetric doubled density `diag(P_sigma, P_sigma)` for its own spin channel, in the energy, in the potential, in all three SCF backends, in the losses, and in the open-shell pretraining row footing, with executable oracles O1-O4 that pin the result against libxc and against the archived tree.

**Architecture:** One primitive -- `doubled_spin_dm(dm, spin_channel)` returning `stack([P_sigma, P_sigma])` -- carries the whole convention. Every density-matrix descriptor already computes the right per-channel quantity when handed that doubled DM: `compute_tau_from_dm` sums a 3-D DM so it returns `2 tau_sigma`; `compute_rung35_occupancy` returns `[n_sigma, n_sigma]` in the alpha-major-then-spin layout; `compute_dm_features_array` takes the per-spin idempotency branch on `P_sigma`. So the change is plumbing plus two new access paths: a precomputed per-channel block on `mol_data` (`assemble_descriptor_features(..., spin_channel=0|1)`) and a live per-channel closure factory (`solver.make_uks_feature_fns`) shared by the solver, the loss and the finite-difference oracle. `split_exc_energy_uks` and its potential twins take three blocks (`features_a`, `features_b`, `features_tot`); correlation keeps the total density and the total block. At `rho_a = rho_b` the three blocks are bit-identical by construction, which is why closed-shell byte identity is structural rather than numerical.

**Tech Stack:** Python 3, JAX (`jax_enable_x64`, CPU), equinox, PySCF + libxc, pyscfad, pytest.

**Spec:** `xcquinox/alec/SPEC_pretrain_fidelity_program.md` (this plan implements Section 3.1 and its oracles O1-O4; Sections 3.2-3.5 are other plans)

## Global Constraints

Every task's requirements implicitly include this section.

- Certificate tolerances, copied verbatim from Section 7 of the spec: "tol_AE = 1.0 kcal/mol on atomization energies and tol_atom = 1.0 mHa on atomic E_xc, for every architecture; no override without `fidelity.override_reason`." This plan does not build the certificate (Section 3.3 does), but no oracle here may be written with a bound looser than these.
- Spin-scaling convention, copied verbatim from Section 7 of the spec: "the symmetric doubled density diag(P_sigma, P_sigma) defines the per-channel feature block for EVERY density-matrix descriptor (alpha, rung-3.5 single and multishell, DM statistics); the cusp feature is geometry-only and unchanged; pretraining rows are posed on the same footing."
- Closed-shell invariant, copied verbatim from Section 3.1 of the spec: "Closed shells: rho_a = rho_b gives identical blocks, so RKS and every closed-shell UKS number is unchanged byte for byte (pinned by test against the archived tree)."
- Comments and docstrings are ASCII only, in scientific voice. They state physics, measurements and rationale. They never mention the process by which the code was produced, never mention an assistant or a model, never say "we", "I", "now", "previously", "as requested", "TODO" or "FIXME". Reference literature the way the surrounding code does (author, journal, volume, page, year).
- Run `python -m py_compile <file>` on every Python file immediately after editing it. A task is not finished while any edited file fails to compile.
- Every test run redirects to a log file and the log is read with `Read`. Never pipe a test run through `tail`, `head`, `less`, `grep -m`, or any other truncating filter: the log must be complete. Create the log directory once with `mkdir -p /tmp/xcq-testlogs`.
- Implementers run no git commands: no `git add`, `git commit`, `git push`, `git checkout`, `git branch`, `git stash`, `git rebase`. The single sanctioned exception is the read-only `git archive ae204537e | tar -x -C /tmp/xcq-ae204537e` export in Task 11, which writes only into the scratch directory and touches no tracked file; that export is required by the closed-shell byte-identity oracle. Committing is the controller's job.
- `xcquinox/alec/HISTORY.md` gets an entry for this change (Task 13). It is the canonical development record for the paper.
- Every number quoted in a comment or a docstring must have been measured by the implementer on this machine. Do not copy a number from this plan into a comment without re-measuring it; the plan's tolerances are bounds, not measurements.

---

## File Structure

| File | Responsibility after this plan |
|---|---|
| `xcquinox/alec/descriptors.py` | Owns the doubled-density primitive `doubled_spin_dm`, the per-channel descriptor accessor `Descriptor.compute_for_spin_channel`, the per-spin precompute key names (`Descriptor.spin_mol_keys`), and `assemble_descriptor_features(..., spin_channel=None)` reading precomputed blocks. |
| `xcquinox/alec/data.py` | Precomputes the per-channel blocks and the per-spin tau for open-shell molecules; declares them on `MoleculeData`. |
| `xcquinox/alec/padding.py` | Pads the new grid-shaped per-spin keys. |
| `xcquinox/alec/solver.py` | `_reassemble_features(..., spin_channel=)` (live path) and `make_uks_feature_fns` -- the one closure factory the manual solver, the loss and the FD oracle share. |
| `xcquinox/alec/oneshot.py` | Three-block UKS energy (`split_exc_energy_uks`), three-block UKS potential (`_uks_spin_resolved_vxc`), one-shot Fock. |
| `xcquinox/alec/solver_manual.py` | UKS SCF loop on three live blocks, with three feature-response contractions. |
| `xcquinox/alec/solver_pyscfad.py` | UKS `eval_xc` callback and feature holder on three per-block slices. |
| `xcquinox/alec/losses.py` | `_vxc_term` on three blocks; `_anchor_term` refuses a descriptor architecture. |
| `xcquinox/alec/pretrain_data_gen.py` | `spin_channel_exchange_rows` -- open-shell exchange rows on the exact-spin-scaling footing; `_atom_columns(exchange_footing=...)` switch, default byte-identical. |
| `xcquinox/alec/tests/parent_adapter.py` | O1 support: `LibxcParentModel`, the parent functional wearing the model's evaluation surface. Not a test module (no `test_` prefix), so pytest does not collect it. |
| `xcquinox/alec/tests/test_spin_scaling_oracles.py` | O1 and O4. |
| `xcquinox/alec/tests/test_solv01_split_xc.py` | O2 (finite-difference Fock check) plus the re-pointed split-energy contract tests. |
| `xcquinox/alec/tests/record_closed_shell_reference.py` | O3 recorder: computes the closed-shell record from whichever `xcquinox` is first on `sys.path`, prints JSON. Not a test module. |
| `xcquinox/alec/tests/fixtures/closed_shell_reference_ae204537e.json` | O3 reference numbers, produced by the recorder run against the archived tree. |
| `xcquinox/alec/tests/test_closed_shell_byte_identity.py` | O3 assertion. |

---

## Task 1: The doubled-density primitive and the per-channel descriptor accessor

**Files:**
- Modify: `xcquinox/alec/descriptors.py:17-41` (the `Descriptor` base class), `:73-106` (`CuspDescriptor`), `:109-162` (`DMStatisticsDescriptor`), `:165-210` (`DMRung35Descriptor`), `:213-269` (`DMRung35MultishellDescriptor`), `:272-309` (`MetaGGAAlphaDescriptor`), `:312-317` (`assemble_descriptor_features`)
- Test: `xcquinox/alec/tests/test_descriptors.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `descriptors.doubled_spin_dm(dm: jnp.ndarray, spin_channel: int) -> jnp.ndarray` -- takes `(2, nao, nao)`, returns `(2, nao, nao)` with `P_sigma` in both slots.
  - `Descriptor.spin_mol_keys: ClassVar[tuple[str, ...]]` -- `()` for a geometry-only descriptor, else `(alpha_key, beta_key)`.
  - `Descriptor.compute_for_spin_channel(self, mol_data: dict, spin_channel: int) -> jnp.ndarray`
  - `descriptors.assemble_descriptor_features(descriptors, mol_data, spin_channel=None) -> jnp.ndarray`
  - Per-spin `mol_data` key names, fixed here and consumed by Task 2: `dm_features_a`/`dm_features_b`, `rung35_features_a`/`rung35_features_b`, `rung35ms_features_a`/`rung35ms_features_b`, `metagga_features_a`/`metagga_features_b`.

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_descriptors.py`:

```python
# ---------------------------------------------------------------------------
# Per-spin-channel feature blocks: the symmetric doubled density
# diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)).
# ---------------------------------------------------------------------------

def test_doubled_spin_dm_places_the_channel_in_both_slots():
    from xcquinox.alec.descriptors import doubled_spin_dm
    rng = np.random.default_rng(20260821)
    p = jnp.asarray(rng.standard_normal((2, 4, 4)))
    for s in (0, 1):
        d = doubled_spin_dm(p, s)
        assert d.shape == (2, 4, 4)
        assert bool(jnp.all(d[0] == p[s]))
        assert bool(jnp.all(d[1] == p[s]))


def test_doubled_spin_dm_refuses_a_total_density_matrix():
    from xcquinox.alec.descriptors import doubled_spin_dm
    with pytest.raises(ValueError, match="spin-resolved"):
        doubled_spin_dm(jnp.zeros((4, 4)), 0)


def test_doubled_spin_dm_refuses_an_out_of_range_channel():
    from xcquinox.alec.descriptors import doubled_spin_dm
    with pytest.raises(ValueError, match="spin_channel"):
        doubled_spin_dm(jnp.zeros((2, 4, 4)), 2)


def test_cusp_per_channel_block_equals_the_shared_block():
    from xcquinox.alec.descriptors import CuspDescriptor
    d = CuspDescriptor()
    mol_data = {"cusp_features": jnp.arange(6.0).reshape(3, 2),
                "rho_grid": jnp.ones(3)}
    for s in (0, 1):
        got = d.compute_for_spin_channel(mol_data, s)
        assert bool(jnp.all(got == mol_data["cusp_features"]))


def test_rung35_per_channel_block_reads_its_own_spin_key():
    from xcquinox.alec.descriptors import DMRung35Descriptor
    d = DMRung35Descriptor()
    mol_data = {"rung35_features": jnp.zeros((3, 2)),
                "rung35_features_a": jnp.full((3, 2), 0.25),
                "rung35_features_b": jnp.full((3, 2), 0.75),
                "rho_grid": jnp.ones(3)}
    assert float(d.compute_for_spin_channel(mol_data, 0)[0, 0]) == 0.25
    assert float(d.compute_for_spin_channel(mol_data, 1)[0, 0]) == 0.75


def test_metagga_and_dm_statistics_declare_their_spin_keys():
    from xcquinox.alec.descriptors import (
        DMStatisticsDescriptor, DMRung35MultishellDescriptor,
        MetaGGAAlphaDescriptor, CuspDescriptor)
    assert DMStatisticsDescriptor.spin_mol_keys == (
        "dm_features_a", "dm_features_b")
    assert DMRung35MultishellDescriptor.spin_mol_keys == (
        "rung35ms_features_a", "rung35ms_features_b")
    assert MetaGGAAlphaDescriptor.spin_mol_keys == (
        "metagga_features_a", "metagga_features_b")
    assert CuspDescriptor.spin_mol_keys == ()


def test_per_channel_block_refuses_an_absent_spin_key():
    from xcquinox.alec.descriptors import DMRung35Descriptor
    d = DMRung35Descriptor()
    with pytest.raises(KeyError, match="rung35_features_a"):
        d.compute_for_spin_channel(
            {"rung35_features": jnp.zeros((3, 2)), "rung35_features_a": None}, 0)


def test_assemble_descriptor_features_spin_channel_preserves_column_order():
    from xcquinox.alec.descriptors import (
        assemble_descriptor_features, CuspDescriptor, DMRung35Descriptor)
    descriptors = (CuspDescriptor(), DMRung35Descriptor())
    mol_data = {
        "rho_grid": jnp.ones(3),
        "cusp_features": jnp.full((3, 2), 7.0),
        "rung35_features": jnp.zeros((3, 2)),
        "rung35_features_a": jnp.full((3, 2), 0.25),
        "rung35_features_b": jnp.full((3, 2), 0.75),
    }
    out = assemble_descriptor_features(descriptors, mol_data, spin_channel=0)
    assert out.shape == (3, 4)
    assert bool(jnp.all(out[:, :2] == 7.0))
    assert bool(jnp.all(out[:, 2:] == 0.25))


def test_assemble_descriptor_features_defaults_to_the_total_block():
    from xcquinox.alec.descriptors import (
        assemble_descriptor_features, DMRung35Descriptor)
    mol_data = {
        "rho_grid": jnp.ones(3),
        "rung35_features": jnp.full((3, 2), 0.5),
        "rung35_features_a": jnp.full((3, 2), 0.25),
        "rung35_features_b": jnp.full((3, 2), 0.75),
    }
    out = assemble_descriptor_features((DMRung35Descriptor(),), mol_data)
    assert bool(jnp.all(out == 0.5))


def test_assemble_descriptor_features_empty_descriptors_ignores_spin_channel():
    from xcquinox.alec.descriptors import assemble_descriptor_features
    mol_data = {"rho_grid": jnp.ones(5)}
    assert assemble_descriptor_features((), mol_data, spin_channel=1).shape == (5, 0)
```

If `test_descriptors.py` does not already import `numpy as np`, `pytest` and `jax.numpy as jnp`, add those imports at the top of the file.

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
mkdir -p /tmp/xcq-testlogs
python -m pytest xcquinox/alec/tests/test_descriptors.py -v > /tmp/xcq-testlogs/task01_red.log 2>&1; echo "exit=$?"
```
Expected: the new tests error with `ImportError: cannot import name 'doubled_spin_dm' from 'xcquinox.alec.descriptors'`, and `test_metagga_and_dm_statistics_declare_their_spin_keys` fails with `AttributeError: type object 'DMStatisticsDescriptor' has no attribute 'spin_mol_keys'`. Read the log with `Read`.

- [ ] **Step 3: Add the primitive and the accessor**

In `xcquinox/alec/descriptors.py`, insert after the `Descriptor` class definition's `__post_init__` (currently ending at line 40) -- that is, add the two members inside the class and the module function below the class.

Inside `class Descriptor`, add the class variable next to `required_mol_keys` (line 20):

```python
    # Per-spin-channel precompute keys, (alpha, beta), holding this descriptor's
    # features for the symmetric doubled density diag(P_sigma, P_sigma). Empty
    # for a geometry-only descriptor, whose per-channel block is the shared one.
    spin_mol_keys: ClassVar[tuple[str, ...]] = ()
```

and add this method to the class body, after `describe`:

```python
    def compute_for_spin_channel(self, mol_data: dict,
                                 spin_channel: int) -> jnp.ndarray:
        """Features of the symmetric doubled density ``diag(P_sigma, P_sigma)``.

        The exact exchange spin-scaling relation
        ``E_x[n_a, n_b] = (E_x[2 n_a] + E_x[2 n_b]) / 2`` (Oliver and Perdew,
        Phys. Rev. A 20, 397 (1979)) evaluates each channel on the fictitious
        spin-unpolarized system whose two spin blocks both hold ``P_sigma``.
        That system, not the physical one, is where a density-matrix descriptor
        must be evaluated for the relation to stay exact. A geometry-only
        descriptor has no density-matrix dependence, so its per-channel block is
        the shared block.
        """
        if spin_channel not in (0, 1):
            raise ValueError(
                "spin_channel must be 0 (alpha) or 1 (beta); got "
                f"{spin_channel!r}."
            )
        if not self.spin_mol_keys:
            return self.compute(mol_data)
        key = self.spin_mol_keys[spin_channel]
        value = mol_data.get(key)
        if value is None:
            raise KeyError(
                f"{type(self).__name__}.compute_for_spin_channel requires "
                f"mol_data[{key!r}], which is absent or None. Open-shell "
                "precompute populates the per-channel blocks; a closed-shell "
                "molecule has rho_a = rho_b and therefore one block, which is "
                "reached with spin_channel=None."
            )
        return value
```

Add the module-level primitive immediately after `list_descriptors` (currently line 70):

```python
def doubled_spin_dm(dm: jnp.ndarray, spin_channel: int) -> jnp.ndarray:
    """The symmetric doubled density matrix ``diag(P_sigma, P_sigma)``.

    The exact exchange spin-scaling relation (Oliver and Perdew, Phys. Rev. A
    20, 397 (1979)) refers each spin channel to the spin-unpolarized system
    built by placing ``P_sigma`` in BOTH spin slots. That system has total
    density ``2 rho_sigma``, gradient invariant ``4 sigma_sigma_sigma`` and
    kinetic-energy density ``2 tau_sigma``, and it is the system whose
    density-matrix descriptors define the channel's feature block: the
    iso-orbital indicator becomes
    ``alpha(2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma)``, the rung-3.5
    occupancy becomes ``[n_sigma, n_sigma]`` (still inside the Bessel bound
    ``[0, 1]``), and the density-matrix statistics become those of
    ``diag(P_sigma, P_sigma)``.

    Every descriptor kernel already produces the right quantity when handed
    this matrix: ``metagga.compute_tau_from_dm`` sums the two spin slots of a
    3-D density matrix, ``rung35.compute_rung35_occupancy`` contracts each slot
    separately, and ``features.compute_dm_features`` takes its per-spin
    idempotency branch on a 3-D argument. So this one transform carries the
    whole convention.
    """
    p = jnp.asarray(dm)
    if p.ndim != 3 or p.shape[0] != 2:
        raise ValueError(
            "doubled_spin_dm requires a spin-resolved (2, nao, nao) density "
            f"matrix; got shape {tuple(p.shape)}."
        )
    if spin_channel not in (0, 1):
        raise ValueError(
            f"spin_channel must be 0 (alpha) or 1 (beta); got {spin_channel!r}."
        )
    block = p[spin_channel]
    return jnp.stack([block, block], axis=0)
```

- [ ] **Step 4: Declare each descriptor's spin keys**

In `DMStatisticsDescriptor`, next to `required_mol_keys` (line 147):

```python
    spin_mol_keys: ClassVar[tuple[str, ...]] = ("dm_features_a", "dm_features_b")
```

In `DMRung35Descriptor`, next to `required_mol_keys` (line 197):

```python
    spin_mol_keys: ClassVar[tuple[str, ...]] = ("rung35_features_a",
                                                "rung35_features_b")
```

In `DMRung35MultishellDescriptor`, next to `required_mol_keys` (line 241):

```python
    spin_mol_keys: ClassVar[tuple[str, ...]] = ("rung35ms_features_a",
                                                "rung35ms_features_b")
```

In `MetaGGAAlphaDescriptor`, next to `required_mol_keys` (line 296):

```python
    spin_mol_keys: ClassVar[tuple[str, ...]] = ("metagga_features_a",
                                                "metagga_features_b")
```

`CuspDescriptor` is left alone: it inherits the empty tuple, which is the statement that the nuclear-cusp proximity feature is geometry-only and identical in all three blocks.

- [ ] **Step 5: Give `assemble_descriptor_features` a spin channel**

Replace `xcquinox/alec/descriptors.py:312-317` with:

```python
def assemble_descriptor_features(descriptors: tuple[Descriptor, ...],
                                 mol_data: dict,
                                 spin_channel: int | None = None) -> jnp.ndarray:
    """Concatenate descriptor outputs left-to-right in declaration order.

    ``spin_channel=None`` returns the block of the physical (total) density,
    which the correlation term consumes. ``spin_channel=0`` / ``1`` returns the
    block of the symmetric doubled density ``diag(P_sigma, P_sigma)``, which is
    what the exact exchange spin scaling evaluates for the alpha / beta channel
    (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)). Column order and width are
    identical in all three blocks.
    """
    if not descriptors:
        return jnp.zeros((mol_data["rho_grid"].shape[0], 0))
    if spin_channel is None:
        return jnp.concatenate([d.compute(mol_data) for d in descriptors], axis=1)
    return jnp.concatenate(
        [d.compute_for_spin_channel(mol_data, spin_channel) for d in descriptors],
        axis=1,
    )
```

- [ ] **Step 6: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/descriptors.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_descriptors.py -v > /tmp/xcq-testlogs/task01_green.log 2>&1; echo "exit=$?"
```
Expected: PASS, all tests in the file green (the pre-existing ones must not move).

- [ ] **Step 7: Extend the tau contract test that the doubling relies on**

`xcquinox/alec/tests/test_metagga.py:35-40` pins that `compute_tau_from_dm` on a
3-D density matrix returns the TOTAL kinetic-energy density. That behavior is
unchanged and is precisely what makes `doubled_spin_dm` produce `2 tau_sigma`,
but the comment above it now states only half the contract. Replace the comment
and add the companion case:

```python
def test_tau_from_dm_matches_pyscf_uks_total():
    # OH doublet. compute_tau_from_dm sums the two spin slots of a 3-D density
    # matrix, so on the PHYSICAL matrix it returns the total kinetic-energy
    # density -- the iso-orbital ingredient of the total density. The same
    # summation on the symmetric doubled matrix diag(P_sigma, P_sigma) returns
    # 2 tau_sigma, the ingredient of the channel the exact exchange spin
    # scaling evaluates (test below).
    mol, mf, ao, dm = _scf("O 0 0 0; H 0 0 0.97", 1)
    tau_ref = mf._numint.eval_rho(mol, ao, dm[0] + dm[1], xctype="MGGA")[5]
    tau = np.asarray(compute_tau_from_dm(jnp.asarray(ao[1:4]), jnp.asarray(dm)))
    assert np.allclose(tau, tau_ref, atol=1e-9)


def test_tau_from_doubled_spin_dm_is_twice_the_channel_tau():
    """tau(diag(P_sigma, P_sigma)) = 2 tau_sigma -- the meta-GGA ingredient of
    the spin-unpolarized system the Oliver-Perdew relation refers to (Phys. Rev.
    A 20, 397 (1979))."""
    from xcquinox.alec.descriptors import doubled_spin_dm
    mol, mf, ao, dm = _scf("O 0 0 0; H 0 0 0.97", 1)
    for s in (0, 1):
        tau_ref = mf._numint.eval_rho(mol, ao, dm[s], xctype="MGGA")[5]
        tau_doubled = np.asarray(compute_tau_from_dm(
            jnp.asarray(ao[1:4]), doubled_spin_dm(jnp.asarray(dm), s)))
        assert np.allclose(tau_doubled, 2.0 * tau_ref, atol=1e-9)
```

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_metagga.py -v > /tmp/xcq-testlogs/task01_metagga.log 2>&1; echo "exit=$?"
```
Expected: PASS. Before Step 3 this new test would fail with
`ImportError: cannot import name 'doubled_spin_dm'`.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_descriptors.py xcquinox/alec/tests/test_metagga.py -v > /tmp/xcq-testlogs/task01_green.log 2>&1`

---

## Task 2: Precompute the per-channel blocks and the per-spin tau

**Files:**
- Modify: `xcquinox/alec/data.py:197-256` (`MoleculeData`), `:712-720` (end of the descriptor precompute block, new block inserted after it), `:794-830` (the `MoleculeData(...)` construction)
- Modify: `xcquinox/alec/padding.py:117-120` (`_PAD_GRID_EDGE`)
- Test: `xcquinox/alec/tests/test_spin_scaling_precompute.py` (create)

**Interfaces:**
- Consumes: `descriptors.doubled_spin_dm` and the key names from Task 1.
- Produces: `mol_data` keys `dm_features_a`, `dm_features_b`, `rung35_features_a`, `rung35_features_b`, `rung35ms_features_a`, `rung35ms_features_b`, `metagga_features_a`, `metagga_features_b` (each `(n_grid, k)` or `None`), and `tau_spin_a`, `tau_spin_b` (each `(n_grid,)` or `None`). All ten are `None` for a closed-shell molecule and for a descriptor the architecture does not carry.

- [ ] **Step 1: Write the failing tests**

Create `xcquinox/alec/tests/test_spin_scaling_precompute.py`:

```python
"""Per-channel descriptor blocks on precomputed molecule data.

Every UKS exchange evaluation is posed on the symmetric doubled density
diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)). These
tests pin what precompute stores for that system: the channel occupancy in both
rung-3.5 spin slots, the iso-orbital indicator at (2 rho_sigma, 4 sigma_sigma
sigma, 2 tau_sigma), the density-matrix statistics of diag(P_sigma, P_sigma),
and the per-spin kinetic-energy density itself.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import MoleculeData, precompute_fixed_density_data
from xcquinox.alec.descriptors import (
    DMRung35Descriptor, DMRung35MultishellDescriptor, DMStatisticsDescriptor,
    MetaGGAAlphaDescriptor, assemble_descriptor_features)


_ALL_DM_DESCRIPTORS = (DMStatisticsDescriptor(), DMRung35Descriptor(),
                       DMRung35MultishellDescriptor(), MetaGGAAlphaDescriptor())


def _precompute(name, atom, spin, composition, descriptors, basis="def2-svp",
                grid_level=1):
    keys = tuple(sorted({k for d in descriptors for k in d.required_mol_keys}))
    return precompute_fixed_density_data(
        MoleculeSpec(name=name, atom=atom, basis=basis, charge=0, spin=spin,
                     atom_composition=composition, grid_level=grid_level),
        required_keys=keys, descriptors=descriptors)


def _spin_grid(md, s):
    """(rho_sigma, sigma_sigma_sigma) for one spin channel of an open shell."""
    ao = np.asarray(md["ao_grid_deriv"])
    d = np.asarray(md["dm_pbe"])[s]
    rho = np.einsum("pi,ij,pj->p", ao[0], d, ao[0])
    gx = 2 * np.einsum("pi,ij,pj->p", ao[1], d, ao[0])
    gy = 2 * np.einsum("pi,ij,pj->p", ao[2], d, ao[0])
    gz = 2 * np.einsum("pi,ij,pj->p", ao[3], d, ao[0])
    return rho, gx ** 2 + gy ** 2 + gz ** 2


def test_open_shell_precompute_populates_every_per_channel_block():
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),), _ALL_DM_DESCRIPTORS)
    n = int(np.asarray(md["grid_weights"]).shape[0])
    for key, width in (("dm_features", 2), ("rung35_features", 2),
                       ("rung35ms_features", 6), ("metagga_features", 1)):
        for suffix in ("_a", "_b"):
            block = md[key + suffix]
            assert block is not None, key + suffix
            assert np.asarray(block).shape == (n, width), key + suffix
            assert np.all(np.isfinite(np.asarray(block))), key + suffix
    for key in ("tau_spin_a", "tau_spin_b"):
        assert np.asarray(md[key]).shape == (n,), key


def test_closed_shell_precompute_leaves_every_per_channel_block_none():
    md = _precompute("H2", "H 0 0 0; H 0 0 0.74", 0, (("H", 2),),
                     _ALL_DM_DESCRIPTORS)
    for key in ("dm_features_a", "dm_features_b", "rung35_features_a",
                "rung35_features_b", "rung35ms_features_a",
                "rung35ms_features_b", "metagga_features_a",
                "metagga_features_b", "tau_spin_a", "tau_spin_b"):
        assert md[key] is None, key


def test_rung35_per_channel_block_is_the_channel_occupancy_in_both_slots():
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),), (DMRung35Descriptor(),))
    tot = np.asarray(md["rung35_features"])
    for s, suffix in ((0, "_a"), (1, "_b")):
        block = np.asarray(md["rung35_features" + suffix])
        np.testing.assert_allclose(block[:, 0], block[:, 1], rtol=0, atol=1e-14)
        np.testing.assert_allclose(block[:, 0], tot[:, s], rtol=0, atol=1e-14)
        assert float(np.min(block)) > -1e-12
        assert float(np.max(block)) < 1.0 + 1e-12


def test_rung35_multishell_per_channel_block_keeps_alpha_major_then_spin():
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),),
                     (DMRung35MultishellDescriptor(),))
    tot = np.asarray(md["rung35ms_features"])
    block = np.asarray(md["rung35ms_features_a"])
    assert block.shape[1] == 6
    for w in range(3):
        # alpha-major then spin: columns (2 w, 2 w + 1) are one projector width.
        np.testing.assert_allclose(block[:, 2 * w], block[:, 2 * w + 1],
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(block[:, 2 * w], tot[:, 2 * w],
                                   rtol=0, atol=1e-14)


def test_metagga_per_channel_alpha_uses_the_doubled_ingredients():
    from pyscf import gto, dft
    from xcquinox.alec.metagga import compute_alpha
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),),
                     (MetaGGAAlphaDescriptor(),))
    mol = gto.M(atom="Li 0 0 0", basis="def2-svp", spin=1, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.kernel()
    # Same molecule, basis, spin, functional and grid level as the precompute,
    # so the two grids are the same set of points; assert it rather than assume
    # it, since a shape mismatch would otherwise surface as a broadcast error.
    assert mf.grids.coords.shape[0] == np.asarray(md["grid_weights"]).shape[0]
    ao2 = mf._numint.eval_ao(mol, mf.grids.coords, deriv=2)
    dm = np.asarray(md["dm_pbe"])
    for s, suffix in ((0, "_a"), (1, "_b")):
        tau_ref = mf._numint.eval_rho(mol, ao2, dm[s], xctype="MGGA")[5]
        np.testing.assert_allclose(np.asarray(md["tau_spin" + suffix]), tau_ref,
                                   rtol=0, atol=1e-9)
        rho_s, sigma_ss = _spin_grid(md, s)
        expect = np.asarray(compute_alpha(jnp.asarray(2.0 * rho_s),
                                          jnp.asarray(4.0 * sigma_ss),
                                          jnp.asarray(2.0 * tau_ref)))
        got = np.asarray(md["metagga_features" + suffix])[:, 0]
        np.testing.assert_allclose(got, expect, rtol=0, atol=1e-10)


def test_per_channel_alpha_differs_from_the_total_density_alpha():
    """The defect this change removes: on an open shell the iso-orbital
    indicator of diag(P_a, P_a) is a different function of position than the
    indicator of the physical total density, so feeding the total block into the
    alpha exchange channel evaluates a different functional."""
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),),
                     (MetaGGAAlphaDescriptor(),))
    per_channel = np.asarray(md["metagga_features_a"])[:, 0]
    total = np.asarray(md["metagga_features"])[:, 0]
    assert float(np.max(np.abs(per_channel - total))) > 1e-3


def test_dm_statistics_per_channel_block_is_tiled_and_finite():
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),),
                     (DMStatisticsDescriptor(),))
    block = np.asarray(md["dm_features_a"])
    assert np.all(np.isfinite(block))
    np.testing.assert_allclose(block, block[0][None, :], rtol=0, atol=0)


def test_assemble_descriptor_features_reads_the_precomputed_blocks():
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),), _ALL_DM_DESCRIPTORS)
    n = int(np.asarray(md["grid_weights"]).shape[0])
    width = sum(d.n_features for d in _ALL_DM_DESCRIPTORS)
    for spin in (0, 1):
        block = assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md,
                                             spin_channel=spin)
        assert block.shape == (n, width)
    total = assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md)
    a = assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md, spin_channel=0)
    assert float(np.max(np.abs(np.asarray(total) - np.asarray(a)))) > 1e-6


def test_every_per_spin_grid_key_is_declared_and_padded():
    from xcquinox.alec.padding import _PAD_GRID_EDGE
    per_spin = {k for k in MoleculeData.__annotations__
                if k.endswith(("_features_a", "_features_b"))
                or k in ("tau_spin_a", "tau_spin_b")}
    assert len(per_spin) == 10, sorted(per_spin)
    missing = sorted(per_spin - set(_PAD_GRID_EDGE))
    assert not missing, f"padding._PAD_GRID_EDGE is missing {missing}"
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_precompute.py -v > /tmp/xcq-testlogs/task02_red.log 2>&1; echo "exit=$?"
```
Expected: `KeyError: 'dm_features_a'` from the `MoleculeData` TypedDict lookups, and `assert len(per_spin) == 10` failing with `[]`.

- [ ] **Step 3: Declare the keys on `MoleculeData`**

In `xcquinox/alec/data.py`, insert after `metagga_features: jnp.ndarray | None` (line 246), before `eri`:

```python
    # Per-spin-channel descriptor blocks: this descriptor's features for the
    # symmetric doubled density diag(P_sigma, P_sigma), the spin-unpolarized
    # system the exact exchange spin-scaling relation refers to (Oliver and
    # Perdew, Phys. Rev. A 20, 397 (1979)). Layout matches the total-density
    # twin above column for column. None for a closed-shell molecule, whose
    # rho_a = rho_b makes the per-channel block identical to the total one, and
    # None for a descriptor the architecture does not carry.
    dm_features_a: jnp.ndarray | None
    dm_features_b: jnp.ndarray | None
    rung35_features_a: jnp.ndarray | None
    rung35_features_b: jnp.ndarray | None
    rung35ms_features_a: jnp.ndarray | None
    rung35ms_features_b: jnp.ndarray | None
    metagga_features_a: jnp.ndarray | None
    metagga_features_b: jnp.ndarray | None
    # Per-spin positive kinetic-energy density tau_sigma on the grid, (n_grid,).
    # The doubled system's tau is 2 tau_sigma. Stored alongside the meta-GGA
    # blocks so the open-shell exchange ingredients are inspectable without
    # recontracting the density matrix.
    tau_spin_a: jnp.ndarray | None
    tau_spin_b: jnp.ndarray | None
```

- [ ] **Step 4: Compute the blocks in the precompute**

In `xcquinox/alec/data.py`, insert this block immediately after the `if "metagga_features" in all_needed:` block ends (currently line 719) and before `eri = None` (line 721):

```python
    # --- Per-spin-channel descriptor blocks (open shells only) --------------
    # Every UKS exchange evaluation is posed on the symmetric doubled density
    # diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)):
    # density 2 rho_sigma, gradient invariant 4 sigma_sigma_sigma,
    # kinetic-energy density 2 tau_sigma. The blocks below are that system's
    # descriptor features, one per channel; they are what the exchange term
    # consumes, while correlation keeps the total density and the total block.
    # A closed-shell molecule has rho_a = rho_b, so its per-channel block IS the
    # total block and these keys stay None.
    dm_features_a = None
    dm_features_b = None
    rung35_features_a = None
    rung35_features_b = None
    rung35ms_features_a = None
    rung35ms_features_b = None
    metagga_features_a = None
    metagga_features_b = None
    tau_spin_a = None
    tau_spin_b = None
    if is_unrestricted:
        from xcquinox.alec.descriptors import doubled_spin_dm
        dm_pbe_spin = jnp.array(dm_pbe)
        doubled = [doubled_spin_dm(dm_pbe_spin, s) for s in (0, 1)]
        rho_doubled = []
        sigma_doubled = []
        for s in (0, 1):
            d_s = np.asarray(dm_pbe[s])
            r_s = np.einsum("pi,ij,pj->p", ao[0], d_s, ao[0])
            gx_s = 2 * np.einsum("pi,ij,pj->p", ao[1], d_s, ao[0])
            gy_s = 2 * np.einsum("pi,ij,pj->p", ao[2], d_s, ao[0])
            gz_s = 2 * np.einsum("pi,ij,pj->p", ao[3], d_s, ao[0])
            rho_doubled.append(2.0 * r_s)
            sigma_doubled.append(4.0 * (gx_s ** 2 + gy_s ** 2 + gz_s ** 2))
        if dm_features is not None:
            from xcquinox.features import compute_dm_features_array
            dm_features_a, dm_features_b = [
                jnp.tile(compute_dm_features_array(d, jnp.array(s_matrix)),
                         (len(rho_pbe), 1))
                for d in doubled
            ]
        if rung35_features is not None:
            from xcquinox.alec.rung35 import compute_rung35_occupancy
            # [n_sigma, n_sigma]: the channel's occupancy in BOTH spin slots,
            # each still inside the Bessel bound [0, 1].
            rung35_features_a, rung35_features_b = [
                compute_rung35_occupancy(rung35_proj_ao, d) for d in doubled
            ]
        if rung35ms_features is not None:
            from xcquinox.alec.rung35 import compute_rung35_multishell_occupancy
            # Column order stays ALPHA-MAJOR then spin, as in the total block.
            rung35ms_features_a, rung35ms_features_b = [
                compute_rung35_multishell_occupancy(rung35ms_proj_ao, d)
                for d in doubled
            ]
        if metagga_features is not None:
            from xcquinox.alec.metagga import compute_tau_from_dm, compute_alpha
            ao_grad_j = jnp.array(ao[1:4])
            tau_spin_a, tau_spin_b = [
                compute_tau_from_dm(ao_grad_j, jnp.array(dm_pbe[s]))
                for s in (0, 1)
            ]
            # compute_tau_from_dm sums the two spin slots of a 3-D density
            # matrix, so the doubled matrix supplies tau = 2 tau_sigma directly.
            metagga_features_a, metagga_features_b = [
                compute_alpha(jnp.array(rho_doubled[s]),
                              jnp.array(sigma_doubled[s]),
                              compute_tau_from_dm(ao_grad_j, doubled[s])
                              ).reshape(-1, 1)
                for s in (0, 1)
            ]
```

- [ ] **Step 5: Return the new keys**

In the `MoleculeData(...)` construction in `xcquinox/alec/data.py`, add after `metagga_features=metagga_features,` (line 827):

```python
        dm_features_a=dm_features_a,
        dm_features_b=dm_features_b,
        rung35_features_a=rung35_features_a,
        rung35_features_b=rung35_features_b,
        rung35ms_features_a=rung35ms_features_a,
        rung35ms_features_b=rung35ms_features_b,
        metagga_features_a=metagga_features_a,
        metagga_features_b=metagga_features_b,
        tau_spin_a=tau_spin_a,
        tau_spin_b=tau_spin_b,
```

- [ ] **Step 6: Pad the new grid-shaped keys**

Replace `xcquinox/alec/padding.py:117-120` with:

```python
# grid-only fields holding FINITE per-point data (edge-padded, weight-0 rows)
_PAD_GRID_EDGE = ("rho_grid", "sigma_grid", "nabla_rho_grid", "rho_ref_grid",
                  "cusp_features", "dm_features", "rung35_features",
                  "rung35ms_features", "metagga_features",
                  # Per-spin-channel blocks of diag(P_sigma, P_sigma). Same
                  # (n_grid, k) grid-major layout as their total-density twins,
                  # so the same edge pad applies; padded rows carry zero grid
                  # weight and contribute nothing to energy or Fock.
                  "dm_features_a", "dm_features_b",
                  "rung35_features_a", "rung35_features_b",
                  "rung35ms_features_a", "rung35ms_features_b",
                  "metagga_features_a", "metagga_features_b",
                  # Per-spin kinetic-energy density, (n_grid,).
                  "tau_spin_a", "tau_spin_b")
```

- [ ] **Step 7: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/data.py xcquinox/alec/padding.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_precompute.py xcquinox/alec/tests/test_data.py xcquinox/alec/tests/test_shape_padding.py -v > /tmp/xcq-testlogs/task02_green.log 2>&1; echo "exit=$?"
```
Expected: PASS. `test_data.py`'s `expected_keys = set(MoleculeData.__annotations__.keys())` check picks up the ten new keys automatically because they are both declared and returned.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_spin_scaling_precompute.py xcquinox/alec/tests/test_data.py xcquinox/alec/tests/test_shape_padding.py -v > /tmp/xcq-testlogs/task02_green.log 2>&1`

---

## Task 3: The live per-channel closure factory

**Files:**
- Modify: `xcquinox/alec/solver.py:432-506` (`_reassemble_features`), new function appended after it (before `_oneshot_result` at line 509)
- Test: `xcquinox/alec/tests/test_spin_scaling_precompute.py` (append)

**Interfaces:**
- Consumes: `descriptors.doubled_spin_dm` (Task 1); the per-channel precompute blocks (Task 2) for the agreement test.
- Produces:
  - `solver._reassemble_features(descriptors, dm, s_matrix, cusp_features=None, n_grid=None, rung35_proj_ao=None, rung35ms_proj_ao=None, ao_grad=None, rho=None, sigma=None, spin_channel=None) -> jnp.ndarray`
  - `solver.make_uks_feature_fns(descriptors, ao_deriv, s_matrix, n_grid, cusp_features=None, rung35_proj_ao=None, rung35ms_proj_ao=None) -> tuple[Callable, Callable, Callable]` returning `(features_a_of, features_b_of, features_tot_of)`, each `(2, nao, nao) -> (n_grid, n_features)`.

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_spin_scaling_precompute.py`:

```python
# ---------------------------------------------------------------------------
# The live per-channel closures: the single place the doubled-density
# convention is implemented for a density matrix that is not the precompute's.
# ---------------------------------------------------------------------------

def test_live_uks_feature_closures_reproduce_the_precomputed_blocks():
    """The live map P -> f_sigma(P) evaluated at the PBE density matrix must
    return exactly what precompute stored, or the potential belongs to a
    different functional than the energy."""
    from xcquinox.alec.solver import make_uks_feature_fns
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),), _ALL_DM_DESCRIPTORS)
    fa, fb, ft = make_uks_feature_fns(
        descriptors=_ALL_DM_DESCRIPTORS,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"),
    )
    P0 = jnp.asarray(md["dm_pbe"])
    np.testing.assert_allclose(
        np.asarray(fa(P0)),
        np.asarray(assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md,
                                                spin_channel=0)),
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(fb(P0)),
        np.asarray(assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md,
                                                spin_channel=1)),
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ft(P0)),
        np.asarray(assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md)),
        rtol=0, atol=1e-12)


def test_live_uks_feature_closures_collapse_at_a_closed_shell_density():
    """rho_a = rho_b makes the three blocks identical -- the structural reason
    every closed-shell number is unchanged by the exact spin scaling."""
    from xcquinox.alec.solver import make_uks_feature_fns
    md = _precompute("H2O", "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
                     0, (("O", 1), ("H", 2)), _ALL_DM_DESCRIPTORS)
    fa, fb, ft = make_uks_feature_fns(
        descriptors=_ALL_DM_DESCRIPTORS,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"),
    )
    half = 0.5 * jnp.asarray(md["dm_pbe"])
    P0 = jnp.stack([half, half], axis=0)
    a, b, t = np.asarray(fa(P0)), np.asarray(fb(P0)), np.asarray(ft(P0))
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    np.testing.assert_allclose(a, t, rtol=0, atol=0)


def test_live_uks_feature_closures_are_empty_for_a_descriptor_free_model():
    from xcquinox.alec.solver import make_uks_feature_fns
    fa, fb, ft = make_uks_feature_fns(
        descriptors=(), ao_deriv=jnp.zeros((4, 7, 3)),
        s_matrix=jnp.eye(3), n_grid=7)
    P0 = jnp.zeros((2, 3, 3))
    for fn in (fa, fb, ft):
        assert fn(P0).shape == (7, 0)


def test_reassemble_features_spin_channel_doubles_the_density_matrix():
    """_reassemble_features with a spin channel must feed diag(P_sigma, P_sigma)
    to every density-matrix descriptor, so it equals the same call made with an
    explicitly doubled matrix and no channel."""
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.alec.solver import _reassemble_features
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),), (DMRung35Descriptor(),))
    P0 = jnp.asarray(md["dm_pbe"])
    kw = dict(descriptors=(DMRung35Descriptor(),), s_matrix=jnp.asarray(md["s_matrix"]),
              n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
              rung35_proj_ao=md.get("rung35_proj_ao"))
    channelled = _reassemble_features(dm=P0, spin_channel=0, **kw)
    explicit = _reassemble_features(dm=doubled_spin_dm(P0, 0), **kw)
    np.testing.assert_allclose(np.asarray(channelled), np.asarray(explicit),
                               rtol=0, atol=0)
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
python -m pytest xcquinox/alec/tests/test_spin_scaling_precompute.py -v > /tmp/xcq-testlogs/task03_red.log 2>&1; echo "exit=$?"
```
Expected: `ImportError: cannot import name 'make_uks_feature_fns' from 'xcquinox.alec.solver'` for the three closure tests, and `TypeError: _reassemble_features() got an unexpected keyword argument 'spin_channel'` for the fourth.

- [ ] **Step 3: Give `_reassemble_features` a spin channel**

In `xcquinox/alec/solver.py`, change the signature at line 432-443 by adding the parameter, and extend the docstring:

```python
def _reassemble_features(
    descriptors: tuple,
    dm: jnp.ndarray,
    s_matrix: jnp.ndarray,
    cusp_features: jnp.ndarray | None = None,
    n_grid: int | None = None,
    rung35_proj_ao: jnp.ndarray | None = None,
    rung35ms_proj_ao: jnp.ndarray | None = None,
    ao_grad: jnp.ndarray | None = None,
    rho: jnp.ndarray | None = None,
    sigma: jnp.ndarray | None = None,
    spin_channel: int | None = None,
) -> jnp.ndarray:
```

Append to the docstring, after the `n_grid` paragraph:

```
    ``spin_channel`` selects the per-channel block of the symmetric doubled
    density ``diag(P_sigma, P_sigma)`` (Oliver and Perdew, Phys. Rev. A 20, 397
    (1979)) instead of the physical block: the density matrix is doubled here,
    and ``rho`` / ``sigma`` must ALREADY be that system's ``2 rho_sigma`` and
    ``4 sigma_sigma_sigma``. :func:`make_uks_feature_fns` is the one place that
    forms them, so callers should go through it rather than doubling by hand.
```

Insert the doubling immediately after the `if not descriptors:` early return (currently ending line 462), before `cols = []`:

```python
    if spin_channel is not None:
        from xcquinox.alec.descriptors import doubled_spin_dm
        dm = doubled_spin_dm(dm, spin_channel)
```

- [ ] **Step 4: Add the closure factory**

Append to `xcquinox/alec/solver.py` immediately after `_reassemble_features` (after line 506, before `_oneshot_result`):

```python
def make_uks_feature_fns(descriptors: tuple,
                         ao_deriv: jnp.ndarray,
                         s_matrix: jnp.ndarray,
                         n_grid: int,
                         cusp_features: jnp.ndarray | None = None,
                         rung35_proj_ao: jnp.ndarray | None = None,
                         rung35ms_proj_ao: jnp.ndarray | None = None):
    """Three closures ``P_ab -> (n_grid, n_features)``: alpha block, beta block,
    total block.

    The per-channel closures evaluate every density-matrix descriptor on the
    symmetric doubled density ``diag(P_sigma, P_sigma)``, the spin-unpolarized
    system the exact exchange spin-scaling relation refers to (Oliver and
    Perdew, Phys. Rev. A 20, 397 (1979)): density ``2 rho_sigma``, gradient
    invariant ``4 sigma_sigma_sigma``, kinetic-energy density ``2 tau_sigma``.
    The total closure is the physical block and feeds the correlation term,
    which is spin-interpolated rather than spin-scaled (von Barth and Hedin,
    J. Phys. C 5, 1629 (1972); Perdew and Wang, Phys. Rev. B 45, 13244 (1992))
    and therefore stays on the total density.

    This is the single implementation of the doubled-density convention for a
    live density matrix. The UKS energy, the UKS potential and the
    feature-response term all consume these closures, so the potential cannot
    drift from the functional; the same closures are what
    :func:`oneshot.feature_response_vxc` differentiates, one per channel, since
    ``f_a``, ``f_b`` and ``f_tot`` are three different maps of ``P``.

    ``ao_deriv`` is the ``(4, n_grid, n_ao)`` ``eval_ao(deriv=1)`` tensor.
    At ``rho_a = rho_b`` all three closures return the SAME array bit for bit:
    doubling either channel of ``[D/2, D/2]`` reproduces the matrix itself, and
    ``2 rho_a`` / ``4 sigma_aa`` are then ``rho_tot`` / ``sigma_tot``. That is
    why the change leaves every closed-shell number untouched.
    """
    if not descriptors:
        empty = jnp.zeros((n_grid, 0))

        def _empty(_P_ab):
            return empty

        return _empty, _empty, _empty

    from xcquinox.alec.descriptors import MetaGGAAlphaDescriptor
    # rho / sigma are consumed only by the meta-GGA iso-orbital indicator; the
    # contraction is skipped for every other architecture, matching the cost of
    # the pre-existing UKS feature assembly.
    needs_rho = any(isinstance(d, MetaGGAAlphaDescriptor) for d in descriptors)
    ao0 = ao_deriv[0]
    ao_grad = ao_deriv[1:4]

    def _rho_sigma(D):
        rho = jnp.einsum("ij,gi,gj->g", D, ao0, ao0)
        nabla = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_grad, ao0)
        return rho, jnp.sum(nabla * nabla, axis=1)

    def _call(dm, rho, sigma, spin_channel):
        return _reassemble_features(
            descriptors=descriptors,
            dm=dm,
            s_matrix=s_matrix,
            cusp_features=cusp_features,
            n_grid=n_grid,
            rung35_proj_ao=rung35_proj_ao,
            rung35ms_proj_ao=rung35ms_proj_ao,
            ao_grad=ao_grad,
            rho=rho,
            sigma=sigma,
            spin_channel=spin_channel,
        )

    def _features_spin(P_ab, spin_channel):
        if needs_rho:
            rho_s, sigma_ss = _rho_sigma(P_ab[spin_channel])
            rho, sigma = 2.0 * rho_s, 4.0 * sigma_ss
        else:
            rho, sigma = None, None
        return _call(P_ab, rho, sigma, spin_channel)

    def features_a_of(P_ab):
        return _features_spin(P_ab, 0)

    def features_b_of(P_ab):
        return _features_spin(P_ab, 1)

    def features_tot_of(P_ab):
        if needs_rho:
            rho, sigma = _rho_sigma(P_ab[0] + P_ab[1])
        else:
            rho, sigma = None, None
        return _call(P_ab, rho, sigma, None)

    return features_a_of, features_b_of, features_tot_of
```

- [ ] **Step 5: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/solver.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_precompute.py xcquinox/alec/tests/test_solver.py -v > /tmp/xcq-testlogs/task03_green.log 2>&1; echo "exit=$?"
```
Expected: PASS.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_spin_scaling_precompute.py xcquinox/alec/tests/test_solver.py -v > /tmp/xcq-testlogs/task03_green.log 2>&1`

---

## Task 4: Three-block UKS energy and potential in `oneshot` and `losses`

**Files:**
- Modify: `xcquinox/alec/oneshot.py:458-517` (`split_exc_energy_uks`), `:520-558` (`fixed_density_total_energy`), `:691-761` (`compute_vc_polarized_per_spin` docstring only), `:764-858` (`_uks_spin_resolved_vxc`), `:861-969` (`oneshot_dm_prediction_fast` UKS branch)
- Modify: `xcquinox/alec/losses.py:399-459` (`_vxc_term`)
- Modify: `xcquinox/alec/tests/test_solv01_split_xc.py:76-84` (`_uks_split_energy`), `:86-105` (`_uks_split_vxc`), `:1155-1188` (the two descriptor contract tests), `:1190-1222` (the polarization-flag guard test), `:518-... ` (`test_fd_consistency_live_features_uks_polarized`), `:437-455` (`_live_features_fn`)
- Modify: `xcquinox/alec/tests/test_uks_oneshot.py:71-76`, `xcquinox/alec/tests/test_losses.py:1019-1040`

**Interfaces:**
- Consumes: `assemble_descriptor_features(..., spin_channel=)` (Task 1), the precomputed blocks (Task 2), `make_uks_feature_fns` (Task 3).
- Produces:
  - `oneshot.split_exc_energy_uks(model, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot, features_a, features_b, features_tot, grid_weights) -> jnp.ndarray`
  - `oneshot._uks_spin_resolved_vxc(model, mol_data, features_a, features_b, features_tot) -> tuple[jnp.ndarray, jnp.ndarray]`

- [ ] **Step 1: Write the failing tests**

Replace `test_split_energy_openshell_passes_same_features_both_exchange_terms` in `xcquinox/alec/tests/test_solv01_split_xc.py` (lines 1166-1188) with:

```python
def test_split_energy_openshell_uses_the_per_channel_feature_block():
    """Exact spin scaling: each doubled-spin exchange evaluation receives ITS
    OWN channel's feature block -- the block of diag(P_sigma, P_sigma) -- and
    correlation receives the total-density block. Supersedes the pinned
    approximation in which one molecular block fed both exchange terms."""
    model = _build_descriptor_model()
    n_feat = sum(d.n_features for d in model.descriptors)
    rng = np.random.default_rng(1)
    rho_a = jnp.asarray(rng.uniform(0.05, 1.0, 6))
    rho_b = jnp.asarray(rng.uniform(0.01, 0.4, 6))     # rho_a != rho_b
    sigma_aa = jnp.asarray(rng.uniform(0.01, 0.5, 6))
    sigma_bb = jnp.asarray(rng.uniform(0.01, 0.3, 6))
    sigma_tot = jnp.asarray(rng.uniform(0.02, 0.9, 6))
    f_a = jnp.asarray(rng.standard_normal((6, n_feat)))
    f_b = jnp.asarray(rng.standard_normal((6, n_feat)))
    f_tot = jnp.asarray(rng.standard_normal((6, n_feat)))
    gw = jnp.ones(6)
    got = float(split_exc_energy_uks(
        model, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot,
        f_a, f_b, f_tot, gw))
    ex_a = model.eval_ex(2.0 * rho_a, 4.0 * sigma_aa, f_a)
    ex_b = model.eval_ex(2.0 * rho_b, 4.0 * sigma_bb, f_b)
    ec = model.eval_ec(rho_a + rho_b, sigma_tot, f_tot)
    expected = float(0.5 * jnp.sum(gw * (ex_a + ex_b)) + jnp.sum(gw * ec))
    assert abs(got - expected) < 1e-12
    # The three blocks are genuinely distinguished: exchanging the two channel
    # blocks changes the energy, which the superseded contract could not see.
    swapped = float(split_exc_energy_uks(
        model, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot,
        f_b, f_a, f_tot, gw))
    assert abs(swapped - got) > 1e-8


def test_split_energy_openshell_correlation_ignores_the_channel_blocks():
    """Correlation is spin-interpolated, not spin-scaled, so it must depend on
    the total block alone (von Barth and Hedin, J. Phys. C 5, 1629 (1972))."""
    model = _build_descriptor_model()
    n_feat = sum(d.n_features for d in model.descriptors)
    rng = np.random.default_rng(2)
    args = (jnp.asarray(rng.uniform(0.05, 1.0, 6)),
            jnp.asarray(rng.uniform(0.01, 0.4, 6)),
            jnp.asarray(rng.uniform(0.01, 0.5, 6)),
            jnp.asarray(rng.uniform(0.01, 0.3, 6)),
            jnp.asarray(rng.uniform(0.02, 0.9, 6)))
    f_tot = jnp.asarray(rng.standard_normal((6, n_feat)))
    zeros = jnp.zeros((6, n_feat))
    gw = jnp.ones(6)
    with_tot = float(split_exc_energy_uks(model, *args, zeros, zeros, f_tot, gw))
    with_zero = float(split_exc_energy_uks(model, *args, zeros, zeros, zeros, gw))
    ec_tot = float(jnp.sum(gw * model.eval_ec(args[0] + args[1], args[4], f_tot)))
    ec_zero = float(jnp.sum(gw * model.eval_ec(args[0] + args[1], args[4], zeros)))
    assert abs((with_tot - with_zero) - (ec_tot - ec_zero)) < 1e-12
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_solv01_split_xc.py -k "per_channel_feature_block or correlation_ignores" -v > /tmp/xcq-testlogs/task04_red.log 2>&1; echo "exit=$?"
```
Expected: `TypeError: split_exc_energy_uks() takes 8 positional arguments but 10 were given`.

- [ ] **Step 3: Rewrite `split_exc_energy_uks`**

Replace `xcquinox/alec/oneshot.py:458-517` with:

```python
def split_exc_energy_uks(model, rho_a, rho_b, sigma_aa, sigma_bb,
                         sigma_tot, features_a, features_b, features_tot,
                         grid_weights):
    """Integrated UKS XC energy using the SOLV-01 split (exchange spin-scaled,
    correlation on the total density).

        E_xc = 1/2 sum_g w_g [eps_x(2 rho_a, 4 sigma_aa, f_a)
                              + eps_x(2 rho_b, 4 sigma_bb, f_b)]
             +     sum_g w_g  eps_c(rho_tot, sigma_tot, f_tot)

    where eps_x = model.eval_ex, eps_c = model.eval_ec (the exact split of
    eval_exc with identical tail masking). Exchange spin-scaling: Oliver and
    Perdew, Phys. Rev. A 20, 397 (1979). Correlation on the TOTAL density:
    von Barth and Hedin, J. Phys. C 5, 1629 (1972); Perdew and Wang, Phys. Rev.
    B 45, 13244 (1992). This is the energy whose functional derivative is the
    split V_xc built by ``_uks_spin_resolved_vxc`` / the manual solver (the
    finite-difference consistency test guards this).

    EXACT SPIN SCALING FOR DESCRIPTOR FEATURES. The relation above is an
    identity for an F_x of any ingredient set, provided each channel is
    evaluated on the FICTITIOUS SPIN-UNPOLARIZED SYSTEM the relation refers to:
    the symmetric doubled density ``diag(P_sigma, P_sigma)``, with density
    ``2 rho_sigma``, gradient invariant ``4 sigma_sigma_sigma`` and
    kinetic-energy density ``2 tau_sigma``. ``features_sigma`` is that system's
    descriptor block, so the meta-GGA indicator is
    ``alpha(2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma)``, the rung-3.5
    single and multishell occupancies are the channel's occupancy in BOTH spin
    slots (alpha-major-then-spin column order preserved), and the density-matrix
    statistics are those of ``diag(P_sigma, P_sigma)``; the nuclear-cusp
    proximity feature is geometry-only and identical in all three blocks.
    Correlation is spin-interpolated rather than spin-scaled, so it keeps the
    total density and ``features_tot``.

    Callers build the three blocks with
    ``descriptors.assemble_descriptor_features(..., spin_channel=0 / 1 / None)``
    on precomputed data, or with ``solver.make_uks_feature_fns`` on a live
    density matrix. Passing one block three times is the CLOSED-SHELL case and
    nothing else: at ``rho_a = rho_b`` the three blocks are identical, so RKS
    and every closed-shell UKS number is unchanged byte for byte.

    When the cnet is spin-polarization-aware (``cnet.use_spin_polarization``),
    correlation is evaluated with the real zeta = (rho_a-rho_b)/rho_tot and the
    zeta-dependent PW92 baseline (Dick and Fernandez-Serra, Phys. Rev. B 104,
    L161109 (2021)); this is the energy whose per-spin functional derivative
    ``compute_vc_polarized_per_spin`` builds. Flag False keeps the zeta=0
    total-density correlation. ``rho_tot = rho_a + rho_b`` is implied by
    ``sigma_tot``.
    """
    rho_tot = rho_a + rho_b
    ex_a = model.eval_ex(2.0 * rho_a, 4.0 * sigma_aa, features_a)
    ex_b = model.eval_ex(2.0 * rho_b, 4.0 * sigma_bb, features_b)
    # Explicit attribute read instead of getattr(..., False) silent fallback.
    # A polarized model.eqx that loses use_spin_polarization during
    # (de)serialization would silently drop zeta on the open-shell path,
    # making polarized vs unpolarized indistinguishable at eval. Raises
    # AttributeError if the cnet lacks the attribute entirely (legacy
    # hand-built cnet, not normal flow).
    if not hasattr(model.cnet, "use_spin_polarization"):
        raise AttributeError(
            "model.cnet has no `use_spin_polarization` attribute. This "
            "indicates a model built outside the standard "
            "AlecGGA_CNet / create_network_pair path; the silent-False "
            "fallback was removed 2026-05-29 to surface this class of bug "
            "instead of degrading polarized eval."
        )
    if model.cnet.use_spin_polarization:
        ec = model.eval_ec(rho_tot, sigma_tot, features_tot,
                           zeta=uks_zeta(rho_a, rho_b))
    else:
        ec = model.eval_ec(rho_tot, sigma_tot, features_tot)
    E_x = 0.5 * jnp.sum(grid_weights * (ex_a + ex_b))
    E_c = jnp.sum(grid_weights * ec)
    return E_x + E_c
```

- [ ] **Step 4: Feed three blocks from `fixed_density_total_energy`**

In `xcquinox/alec/oneshot.py`, replace the body of `fixed_density_total_energy` (lines 532-558) with:

```python
    features = assemble_descriptor_features(model.descriptors, mol_data)
    if mol_data["is_unrestricted"]:
        # Each exchange channel is evaluated on its own doubled density
        # diag(P_sigma, P_sigma); correlation keeps the total block.
        features_a = assemble_descriptor_features(model.descriptors, mol_data,
                                                  spin_channel=0)
        features_b = assemble_descriptor_features(model.descriptors, mol_data,
                                                  spin_channel=1)
        dm_pbe = mol_data["dm_pbe"]  # (2, nao, nao)
        ao_grid = mol_data["ao_grid"]
        ao_xyz = mol_data["ao_grid_deriv"][1:4]
        grid_weights = mol_data["grid_weights"]
        rho_a = jnp.einsum("ij,gi,gj->g", dm_pbe[0], ao_grid, ao_grid)
        rho_b = jnp.einsum("ij,gi,gj->g", dm_pbe[1], ao_grid, ao_grid)
        nabla_rho_a = 2.0 * jnp.einsum("ij,dgi,gj->gd", dm_pbe[0], ao_xyz, ao_grid)
        nabla_rho_b = 2.0 * jnp.einsum("ij,dgi,gj->gd", dm_pbe[1], ao_xyz, ao_grid)
        sigma_aa = jnp.sum(nabla_rho_a * nabla_rho_a, axis=1)
        sigma_bb = jnp.sum(nabla_rho_b * nabla_rho_b, axis=1)
        nabla_rho_tot = nabla_rho_a + nabla_rho_b
        sigma_tot = jnp.sum(nabla_rho_tot * nabla_rho_tot, axis=1)
        exc_integrated = split_exc_energy_uks(
            model, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot,
            features_a, features_b, features, grid_weights,
        )
        return mol_data["E_non_xc"] + exc_integrated
    exc_integrated = compute_exc_nn(
        model,
        mol_data["rho_grid"],
        mol_data["sigma_grid"],
        features,
        mol_data["grid_weights"],
    )
    return mol_data["E_non_xc"] + exc_integrated
```

Also replace the docstring's closing paragraph so it names the per-channel blocks:

```
    the UKS branch uses the SPLIT XC energy (exchange spin-scaled per Oliver and
    Perdew, Phys. Rev. A 20, 397 (1979), each channel on its own doubled density
    diag(P_sigma, P_sigma); correlation on the total density per von Barth and
    Hedin 1972 / PW92 1992) so that this energy is consistent with the split
    V_xc used by the SCF solvers. RKS is unchanged (combined eval_exc on the
    total density).
```

- [ ] **Step 5: Feed three blocks into the one-shot potential**

In `xcquinox/alec/oneshot.py`, change the `_uks_spin_resolved_vxc` signature (line 764) to:

```python
def _uks_spin_resolved_vxc(model, mol_data, features_a, features_b, features_tot):
```

Replace its `LIMITATION (descriptor features), P2-02` paragraph (lines 802-808) with:

```
    EXACT SPIN SCALING FOR DESCRIPTOR FEATURES. Each exchange channel is
    evaluated at its OWN feature block ``features_sigma``, the block of the
    symmetric doubled density ``diag(P_sigma, P_sigma)`` that the Oliver-Perdew
    relation refers to; correlation is evaluated at ``features_tot``. Since the
    blocks arrive as concrete arrays here (this is the fixed-density one-shot
    path, whose features are frozen at the precompute density matrix), the
    ``de/df . df/dP`` chain-rule term does not enter; the self-consistent path in
    ``solver_manual`` differentiates each channel's ``P -> f_sigma(P)`` map and
    adds it.
```

Change the three evaluation calls (lines 830-857) to use the per-channel blocks:

```python
    # Exchange: per-spin, spin-scaled (part="x"), each at its own channel block.
    vx_a = compute_vxc_nn(
        model, 2.0 * rho_a, 4.0 * sigma_aa, features_a, ao_grid, grid_weights,
        nabla_rho=2.0 * nabla_rho_a, ao_grad=ao_grid_deriv, part="x",
    )
    vx_b = compute_vxc_nn(
        model, 2.0 * rho_b, 4.0 * sigma_bb, features_b, ao_grid, grid_weights,
        nabla_rho=2.0 * nabla_rho_b, ao_grad=ao_grid_deriv, part="x",
    )
```

and in the correlation branch replace `features` with `features_tot` in both the
`compute_vc_polarized_per_spin` call and the `compute_vxc_nn(..., part="c")` call.

In `compute_vc_polarized_per_spin` (line 691), add this sentence to the docstring after the first paragraph, leaving the signature untouched:

```
    ``features`` is the TOTAL-density block: correlation is spin-interpolated,
    not spin-scaled, so it never sees the per-channel blocks of
    ``diag(P_sigma, P_sigma)``.
```

- [ ] **Step 6: Feed three blocks from the one-shot Fock build**

In `oneshot_dm_prediction_fast`, replace line 892 with:

```python
        # Spin-resolved V_xc^NN: each exchange channel at its own doubled-density
        # block, correlation at the total block.
        vxc_nn_a, vxc_nn_b = _uks_spin_resolved_vxc(
            model, mol_data,
            assemble_descriptor_features(model.descriptors, mol_data,
                                         spin_channel=0),
            assemble_descriptor_features(model.descriptors, mol_data,
                                         spin_channel=1),
            features,
        )
```

- [ ] **Step 7: Feed three blocks from the V_xc loss channel**

In `xcquinox/alec/losses.py`, replace lines 420-425 with:

```python
        features = assemble_descriptor_features(model.descriptors, mol_data[i])

        if vxc_ref_arr.ndim == 3:  # UKS: (2, n_ao, n_ao)
            # Exchange channels take the block of their own doubled density
            # diag(P_sigma, P_sigma); correlation takes the total block.
            vxc_nn_a, vxc_nn_b = _uks_spin_resolved_vxc(
                model, mol_data[i],
                assemble_descriptor_features(model.descriptors, mol_data[i],
                                             spin_channel=0),
                assemble_descriptor_features(model.descriptors, mol_data[i],
                                             spin_channel=1),
                features,
            )
```

and update the `_vxc_term` docstring line 407 to read:

```
    (shape ``(2, n_ao, n_ao)``). For UKS, the NN's spin-resolved V_xc is
    constructed via :func:`_uks_spin_resolved_vxc` with the per-channel feature
    blocks of diag(P_sigma, P_sigma), and the squared error is summed across
    both spin channels.
```

- [ ] **Step 8: Re-point the existing UKS tests to the three-block contract**

In `xcquinox/alec/tests/test_solv01_split_xc.py`:

Replace `_uks_split_energy` (lines 76-84) with:

```python
def _uks_split_energy(model, D_a, D_b, features_a, features_b, features_tot,
                      ao_grid, ao_xyz, grid_weights):
    """SOLV-01 split UKS XC energy from a spin-DM pair.

    ``features_a`` / ``features_b`` are the blocks of the symmetric doubled
    densities diag(P_a, P_a) and diag(P_b, P_b); ``features_tot`` is the
    physical block the correlation term consumes.
    """
    rho_a, nra, sig_aa = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, sig_bb = _grid_quantities(D_b, ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)
    return split_exc_energy_uks(
        model, rho_a, rho_b, sig_aa, sig_bb, sig_tot,
        features_a, features_b, features_tot, grid_weights,
    )
```

Replace `_uks_split_vxc` (lines 86-105) with the same parameter change, using `features_a` in the alpha exchange call, `features_b` in the beta exchange call and `features_tot` in the correlation call.

Every call site of these two helpers uses a model built by `_build_model()` or
`_build_polarized_model()`, both of which use the `deep` architecture, which
carries NO descriptors: `features` there is the empty `(n_grid, 0)` array, so the
three blocks are the same empty array and passing it three times is exact. The
sites are, at the line numbers before this task's edits:

| line | helper | test |
|---|---|---|
| 130 | `_uks_split_energy` | `test_closed_shell_reduction_energy_and_vxc` |
| 144 | `_uks_split_vxc` | `test_closed_shell_reduction_energy_and_vxc` |
| 240 | `_uks_split_energy` | `test_fd_energy_potential_consistency` |
| 284 | `_uks_split_vxc` | `test_fd_energy_potential_consistency` |
| 887 | `_uks_split_energy` | `test_polarized_full_split_vxc_fd_consistency` |
| 1124, 1126 | `_uks_split_energy` | `test_spin_swap_symmetry` |
| 1130, 1132 | `_uks_split_vxc` | `test_spin_swap_symmetry` |

Confirm the list is complete before editing:

```bash
cd /home/awills/Documents/Research/xcquinox && grep -n "_uks_split_energy(\|_uks_split_vxc(" xcquinox/alec/tests/test_solv01_split_xc.py
```

At each site pass `features, features, features` in place of the single
`features` argument, and add this comment above the first such call in each
test:

```python
    # The `deep` architecture carries no descriptors, so `features` is the empty
    # (n_grid, 0) array and the three per-channel blocks are that same array.
```

`_uks_split_vxc_exact` (line 182) calls `_exact_vxc_unmasked` rather than any
changed API, so its signature is left alone; add one line to its docstring:

```
    Single ``features`` argument: this helper is used only with the
    descriptor-free ``deep`` architecture, where the three per-channel blocks
    are the same empty array.
```

Replace `_live_features_fn` (lines 437-455) with a wrapper over the shared factory:

```python
def _live_features_fn(model, md):
    """The exact ``P -> features`` map the RKS solver uses, as a closure."""
    from xcquinox.alec.solver import (
        _reassemble_features, _contract_dm_to_grid_with_nabla)
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    n_grid = int(np.asarray(md["grid_weights"]).shape[0])
    s_matrix = jnp.asarray(md["s_matrix"])
    cusp = md.get("cusp_features")
    proj = md.get("rung35_proj_ao")
    proj_ms = md.get("rung35ms_proj_ao")
    has_mgga = any(type(d).__name__ == "MetaGGAAlphaDescriptor"
                   for d in model.descriptors)

    def features_of(P):
        if not model.descriptors:
            return jnp.zeros((n_grid, 0))
        kw = {}
        if has_mgga:
            total = P if P.ndim == 2 else P[0] + P[1]
            rho_t, _nab, sigma_t = _contract_dm_to_grid_with_nabla(
                total, ao_deriv)
            kw = dict(ao_grad=ao_deriv[1:4], rho=rho_t, sigma=sigma_t)
        return _reassemble_features(
            descriptors=model.descriptors, dm=P, s_matrix=s_matrix,
            cusp_features=cusp, n_grid=n_grid, rung35_proj_ao=proj,
            rung35ms_proj_ao=proj_ms, **kw)
    return features_of


def _live_uks_features_fns(model, md):
    """The three ``P_ab -> features`` maps the UKS solver uses."""
    from xcquinox.alec.solver import make_uks_feature_fns
    return make_uks_feature_fns(
        descriptors=model.descriptors,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"),
    )
```

In `test_fd_consistency_live_features_uks_polarized`, replace `features_of = _live_features_fn(model, md)` with:

```python
    features_a_of, features_b_of, features_tot_of = _live_uks_features_fns(
        model, md)
```

replace the `energy` closure with:

```python
    def energy(P):
        rho_a, nabla_a, sigma_aa = spin_quantities(P[0])
        rho_b, nabla_b, sigma_bb = spin_quantities(P[1])
        nabla_tot = nabla_a + nabla_b
        return split_exc_energy_uks(
            model, rho_a, rho_b, sigma_aa, sigma_bb,
            jnp.sum(nabla_tot * nabla_tot, axis=1),
            features_a_of(P), features_b_of(P), features_tot_of(P), weights)
```

replace `f0 = features_of(P0)` with:

```python
    f0_a, f0_b, f0_tot = features_a_of(P0), features_b_of(P0), features_tot_of(P0)
```

use `f0_a` in the alpha `compute_vxc_nn`, `f0_b` in the beta one and `f0_tot` in
`compute_vc_polarized_per_spin`, and replace the feature-response block with:

```python
    if has_dm_dependent_descriptor(model):
        # f_a, f_b and f_tot are three different maps of P, so the chain-rule
        # term is three contractions rather than one accumulated de/df. Each
        # per-channel map depends on P only through its own P_sigma, so its
        # contraction lands in that spin block.
        v_feat = feature_response_vxc(
            0.5 * feature_energy_derivative(
                model, 2.0 * rho_a, 4.0 * sigma_aa, f0_a, part="x"),
            weights, features_a_of, P0)
        v_feat = v_feat + feature_response_vxc(
            0.5 * feature_energy_derivative(
                model, 2.0 * rho_b, 4.0 * sigma_bb, f0_b, part="x"),
            weights, features_b_of, P0)
        v_feat = v_feat + feature_response_vxc(
            feature_energy_derivative(
                model, rho_a + rho_b, sigma_tot, f0_tot, part="c",
                zeta=uks_zeta(rho_a, rho_b)),
            weights, features_tot_of, P0)
        V_a, V_b = V_a + v_feat[0], V_b + v_feat[1]
```

In `test_split_exc_energy_uks_raises_when_cnet_lacks_polarization_flag` (line 1190), change the final call to pass three blocks:

```python
    with pytest.raises(AttributeError, match="use_spin_polarization"):
        split_exc_energy_uks(_ModelNoFlag(model), rho_a, rho_b,
                             sig_aa, sig_bb, sig_tot,
                             features, features, features, grid_weights)
```

In `xcquinox/alec/tests/test_uks_oneshot.py:74-75` and `xcquinox/alec/tests/test_losses.py:1034-1035`, replace the two-line pattern with:

```python
    features = assemble_descriptor_features(model.descriptors, md)
    vxc_a, vxc_b = _uks_spin_resolved_vxc(
        model, md,
        assemble_descriptor_features(model.descriptors, md, spin_channel=0),
        assemble_descriptor_features(model.descriptors, md, spin_channel=1),
        features)
```

- [ ] **Step 9: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/oneshot.py xcquinox/alec/losses.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_solv01_split_xc.py xcquinox/alec/tests/test_uks_oneshot.py xcquinox/alec/tests/test_oneshot.py -v > /tmp/xcq-testlogs/task04_green.log 2>&1; echo "exit=$?"
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_losses.py -v > /tmp/xcq-testlogs/task04_losses.log 2>&1; echo "exit=$?"
```
Expected: PASS in both logs. In particular `test_fd_consistency_live_features_uks_polarized` must stay under `_TOL_UKS = 5e-7` for every architecture; if a descriptor architecture now exceeds it, the three-contraction feature response is wrong, not the tolerance.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_solv01_split_xc.py xcquinox/alec/tests/test_uks_oneshot.py xcquinox/alec/tests/test_oneshot.py xcquinox/alec/tests/test_losses.py -v > /tmp/xcq-testlogs/task04_green.log 2>&1`

---

## Task 5: Manual-solver UKS loop on three live blocks

**Files:**
- Modify: `xcquinox/alec/solver_manual.py:150-205` (`_compute_total_energy_uks`), `:457-729` (`_run_manual_scf_uks`: `_features_for` at 514-548, `_feature_response_uks` at 554-585, `_vx_nn_spin` at 587-605, `_vc_nn_total` at 607-629, the initial energy at 638-646, `body` at 659-729)
- Test: `xcquinox/alec/tests/test_spin_scaling_solver_manual.py` (create)

**Interfaces:**
- Consumes: `solver.make_uks_feature_fns` (Task 3), `oneshot.split_exc_energy_uks` (Task 4).
- Produces: `solver_manual._compute_total_energy_uks(model, D_a, D_b, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot, features_a, features_b, features_tot, grid_weights, h_core, J_total, e_nuc) -> jnp.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `xcquinox/alec/tests/test_spin_scaling_solver_manual.py`:

```python
"""The manual UKS SCF on per-channel feature blocks.

The exchange term of the UKS energy is evaluated on the symmetric doubled
density diag(P_sigma, P_sigma) for each channel (Oliver and Perdew, Phys. Rev. A
20, 397 (1979)); the correlation term stays on the total density. These tests
pin that the SCF energy is that functional and that the Fock matrices the loop
builds are its exact derivative.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (
    FeaturePolicy, SolverBackend, SolverConfig, SolverMode, run_scf,
    make_uks_feature_fns)
from xcquinox.alec.solver_manual import _compute_total_energy_uks

_FD_EPS = 1e-6


def _model(arch_name, seed=0):
    arch = dataclasses.replace(alec.get_architecture(arch_name),
                               use_polarized_correlation=True,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _md(model, name, atom, spin, composition, basis="def2-svp", grid_level=1):
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys} | {"eri"}))
    return precompute_fixed_density_data(
        MoleculeSpec(name=name, atom=atom, basis=basis, charge=0, spin=spin,
                     atom_composition=composition, grid_level=grid_level),
        required_keys=keys, descriptors=model.descriptors)


def _config(policy):
    return SolverConfig(mode=SolverMode.FULL, backend=SolverBackend.MANUAL,
                        max_cycles=2, feature_policy=policy)


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16",
                                       "deep_rung35ms_3x16", "deep_dm_3x16"])
def test_manual_uks_energy_is_the_three_block_split_energy(arch_name):
    """The SCF's own energy helper must equal oneshot.split_exc_energy_uks fed
    the three live blocks, term for term."""
    from xcquinox.alec.oneshot import split_exc_energy_uks
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    ao = jnp.asarray(md["ao_grid"])
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_deriv[1:4]
    w = jnp.asarray(md["grid_weights"])
    P = jnp.asarray(md["dm_pbe"])
    fa, fb, ft = make_uks_feature_fns(
        descriptors=model.descriptors, ao_deriv=ao_deriv,
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"))

    def grid(D):
        rho = jnp.einsum("ij,gi,gj->g", D, ao, ao)
        nabla = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_xyz, ao)
        return rho, nabla, jnp.sum(nabla * nabla, axis=1)

    rho_a, nab_a, sig_aa = grid(P[0])
    rho_b, nab_b, sig_bb = grid(P[1])
    nab_t = nab_a + nab_b
    sig_tot = jnp.sum(nab_t * nab_t, axis=1)
    h = jnp.asarray(md["h_core"])
    j_tot = jnp.asarray(md["j_matrix"])[0] + jnp.asarray(md["j_matrix"])[1]
    e_nuc = jnp.asarray(md["e_nuc"])
    got = float(_compute_total_energy_uks(
        model, P[0], P[1], rho_a, rho_b, sig_aa, sig_bb, sig_tot,
        fa(P), fb(P), ft(P), w, h, j_tot, e_nuc))
    xc = float(split_exc_energy_uks(model, rho_a, rho_b, sig_aa, sig_bb,
                                    sig_tot, fa(P), fb(P), ft(P), w))
    one = float(jnp.einsum("ij,ij->", h, P[0] + P[1]))
    coul = float(0.5 * jnp.einsum("ij,ij->", j_tot, P[0] + P[1]))
    assert abs(got - (float(e_nuc) + one + coul + xc)) < 1e-12


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16",
                                       "deep_rung35ms_3x16", "deep_dm_3x16"])
def test_manual_uks_scf_runs_and_stays_finite_under_reassemble(arch_name):
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    result = run_scf(_config(FeaturePolicy.REASSEMBLE), model, md)
    assert bool(jnp.isfinite(result.total_energy))
    assert bool(jnp.all(jnp.isfinite(result.density_matrix)))


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16", "deep_dm_3x16"])
def test_manual_uks_frozen_policy_uses_the_precomputed_channel_blocks(arch_name):
    """FROZEN must freeze the three blocks separately, not freeze one block and
    reuse it for all three."""
    from xcquinox.alec.descriptors import assemble_descriptor_features
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    result = run_scf(_config(FeaturePolicy.FROZEN), model, md)
    assert bool(jnp.isfinite(result.total_energy))
    a = assemble_descriptor_features(model.descriptors, md, spin_channel=0)
    tot = assemble_descriptor_features(model.descriptors, md)
    assert float(jnp.max(jnp.abs(a - tot))) > 1e-6, (
        "Li must have distinguishable per-channel and total blocks, otherwise "
        "this test cannot see the difference it is checking")


def test_manual_uks_closed_shell_density_gives_three_identical_blocks():
    """A UKS run at rho_a = rho_b must reduce to the RKS functional exactly."""
    model = _model("deep_rung35_mgga_3x16")
    md = _md(model, "H2O", "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
             0, (("O", 1), ("H", 2)))
    fa, fb, ft = make_uks_feature_fns(
        descriptors=model.descriptors,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"))
    half = 0.5 * jnp.asarray(md["dm_pbe"])
    P = jnp.stack([half, half], axis=0)
    np.testing.assert_allclose(np.asarray(fa(P)), np.asarray(ft(P)),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(fb(P)), np.asarray(ft(P)),
                               rtol=0, atol=0)
```

- [ ] **Step 2: Run and confirm it fails**

```bash
python -m pytest xcquinox/alec/tests/test_spin_scaling_solver_manual.py -v > /tmp/xcq-testlogs/task05_red.log 2>&1; echo "exit=$?"
```
Expected: `TypeError: _compute_total_energy_uks() takes 13 positional arguments but 15 were given` for the first test, and `TypeError: split_exc_energy_uks() takes 8 positional arguments but 10 were given` raised from inside `solver_manual` for the SCF tests.

- [ ] **Step 3: Give `_compute_total_energy_uks` three blocks**

In `xcquinox/alec/solver_manual.py`, change the signature (lines 150-164) so `features` becomes three parameters in the same slot:

```python
def _compute_total_energy_uks(
    model,
    D_a: jnp.ndarray,
    D_b: jnp.ndarray,
    rho_a: jnp.ndarray,
    rho_b: jnp.ndarray,
    sigma_aa: jnp.ndarray,
    sigma_bb: jnp.ndarray,
    sigma_tot: jnp.ndarray,
    features_a: jnp.ndarray,
    features_b: jnp.ndarray,
    features_tot: jnp.ndarray,
    grid_weights: jnp.ndarray,
    h_core: jnp.ndarray,
    J_total: jnp.ndarray,
    e_nuc: jnp.ndarray,
) -> jnp.ndarray:
```

Replace the `LIMITATION (descriptor features), P2-02` paragraph (lines 185-189) with:

```
    EXACT SPIN SCALING FOR DESCRIPTOR FEATURES. ``features_a`` and
    ``features_b`` are the descriptor blocks of the symmetric doubled densities
    diag(P_a, P_a) and diag(P_b, P_b) -- the spin-unpolarized systems the
    Oliver-Perdew relation refers to -- so the relation stays exact with
    density-matrix descriptors active. ``features_tot`` is the physical block
    that the spin-interpolated correlation term consumes. At rho_a = rho_b the
    three coincide and this reduces to the RKS functional exactly. See
    ``oneshot.split_exc_energy_uks``.
```

Change the call at the end of the body (line 201-204) to:

```python
    E_xc_nn = split_exc_energy_uks(
        model, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot,
        features_a, features_b, features_tot, grid_weights,
    )
```

- [ ] **Step 4: Build the three live blocks in the UKS SCF**

In `_run_manual_scf_uks`, change the import line 471-475 to add the factory:

```python
    from xcquinox.alec.descriptors import assemble_descriptor_features
    from xcquinox.alec.solver import make_uks_feature_fns
    from xcquinox.alec.oneshot import (
        compute_vxc_nn, compute_vc_polarized_per_spin,
        feature_energy_derivative, feature_response_vxc,
        has_dm_dependent_descriptor, uks_zeta)
```

Replace line 495 with:

```python
    features_initial = assemble_descriptor_features(model.descriptors, mol_data)
    if model.descriptors:
        # Frozen per-channel blocks: the descriptor features of the symmetric
        # doubled densities diag(P_a, P_a) and diag(P_b, P_b) at the precompute
        # density matrix. Freezing one block and reusing it for all three would
        # reinstate the approximation this evaluation removes.
        features_initial_a = assemble_descriptor_features(
            model.descriptors, mol_data, spin_channel=0)
        features_initial_b = assemble_descriptor_features(
            model.descriptors, mol_data, spin_channel=1)
    else:
        features_initial_a = features_initial
        features_initial_b = features_initial

    _features_a_of, _features_b_of, _features_tot_of = make_uks_feature_fns(
        descriptors=model.descriptors,
        ao_deriv=ao_grid_deriv,
        s_matrix=s_matrix,
        n_grid=grid_weights.shape[0],
        cusp_features=cusp_cached,
        rung35_proj_ao=mol_data.get("rung35_proj_ao"),
        rung35ms_proj_ao=mol_data.get("rung35ms_proj_ao"),
    )
```

Replace `_features_for` (lines 514-548) with:

```python
    def _features_for(D_ab):
        """The three descriptor blocks for the current DM pair.

        Returns ``(features_a, features_b, features_tot)``. The per-channel
        blocks are the features of the symmetric doubled densities
        diag(P_a, P_a) and diag(P_b, P_b), which is what the spin-scaled
        exchange terms evaluate (Oliver and Perdew, Phys. Rev. A 20, 397
        (1979)); the total block is the physical one the spin-interpolated
        correlation term consumes. Every density-matrix descriptor receives the
        SPIN-RESOLVED 3-D density matrix, so DMStatisticsDescriptor keeps its
        per-spin idempotency-projector branch (Pople-Nesbet 1954: D_sigma S
        D_sigma = D_sigma per spin) rather than the RKS branch, whose
        idempotency_error would be non-zero and physically meaningless.
        FROZEN policy reuses the initial blocks.
        """
        if policy == FeaturePolicy.FROZEN or not model.descriptors:
            return features_initial_a, features_initial_b, features_initial
        return (_features_a_of(D_ab), _features_b_of(D_ab),
                _features_tot_of(D_ab))
```

Replace `_feature_response_uks` (lines 554-585) with:

```python
    def _feature_response_uks(D_ab, features_a, features_b, features_tot,
                              rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot):
        """Per-spin V_xc contribution from the descriptors' DM dependence.

        Each term of the split UKS energy carries its OWN feature block,

            E_xc = 1/2 sum_g w_g [e_x(2 rho_a, 4 sigma_aa, f_a(P))
                                  + e_x(2 rho_b, 4 sigma_bb, f_b(P))]
                 +     sum_g w_g  e_c(rho_tot, sigma_tot, f_tot(P) [, zeta]),

        and f_a, f_b, f_tot are three DIFFERENT maps of P, so the chain-rule
        term is three contractions rather than one accumulated de/df followed by
        a single contraction. f_sigma is the block of diag(P_sigma, P_sigma) and
        therefore depends on P only through P_sigma, so its contraction lands
        entirely in that spin block; the total block couples both.
        """
        rho_tot = rho_a + rho_b
        v = feature_response_vxc(
            0.5 * feature_energy_derivative(
                model, 2.0 * rho_a, 4.0 * sigma_aa, features_a, part="x"),
            grid_weights, _features_a_of, D_ab)
        v = v + feature_response_vxc(
            0.5 * feature_energy_derivative(
                model, 2.0 * rho_b, 4.0 * sigma_bb, features_b, part="x"),
            grid_weights, _features_b_of, D_ab)
        if model.cnet.use_spin_polarization:
            dedf_c = feature_energy_derivative(
                model, rho_tot, sigma_tot, features_tot, part="c",
                zeta=uks_zeta(rho_a, rho_b))
        else:
            dedf_c = feature_energy_derivative(
                model, rho_tot, sigma_tot, features_tot, part="c")
        return v + feature_response_vxc(dedf_c, grid_weights,
                                        _features_tot_of, D_ab)
```

In `_vx_nn_spin` (line 587), rename the first parameter to `features_s` and add to its docstring:

```
        ``features_s`` is the channel's own block, the descriptor features of
        diag(P_sigma, P_sigma).
```

Update its body to pass `features_s` to `compute_vxc_nn`.

In `_vc_nn_total` (line 607), rename the first parameter to `features_tot`, pass
it through, and add to the docstring:

```
        ``features_tot`` is the physical block; correlation never sees the
        per-channel blocks.
```

- [ ] **Step 5: Wire the loop**

Replace the initial-energy block (lines 639-646) with:

```python
    (rho_a0, rho_b0, nabla_rho_a0, nabla_rho_b0,
     sigma_aa0, sigma_bb0, _nabla_tot0, sigma_tot0) = _spin_resolved_rho(D0)
    features_0_a, features_0_b, features_0_tot = _features_for(D0)
    J_total_0 = _j_total_for_cycle(D0)
    E0 = _compute_total_energy_uks(
        model, D0[0], D0[1], rho_a0, rho_b0, sigma_aa0, sigma_bb0, sigma_tot0,
        features_0_a, features_0_b, features_0_tot,
        grid_weights, h_core, J_total_0, e_nuc,
    )
```

In `body`, replace line 663 with:

```python
        features_a, features_b, features_tot = _features_for(D_cur)
```

replace lines 667-668 with:

```python
        vx_a = _vx_nn_spin(features_a, rho_a, sigma_aa, nabla_rho_a)
        vx_b = _vx_nn_spin(features_b, rho_b, sigma_bb, nabla_rho_b)
```

replace `features` with `features_tot` in the `compute_vc_polarized_per_spin`
call (line 678) and in the `_vc_nn_total` call (line 684), replace the
feature-response call (lines 687-691) with:

```python
        if _needs_feature_response:
            v_feat = _feature_response_uks(
                D_cur, features_a, features_b, features_tot,
                rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot)
            vxc_nn_a = vxc_nn_a + v_feat[0]
            vxc_nn_b = vxc_nn_b + v_feat[1]
```

and replace the mixed-density energy recompute (lines 705-710) with:

```python
        features_m_a, features_m_b, features_m_tot = _features_for(D_mixed)
        j_total_m = _j_total_for_cycle(D_mixed)
        E_new = _compute_total_energy_uks(
            model, D_mixed[0], D_mixed[1], rho_a_m, rho_b_m, sig_aa_m, sig_bb_m,
            sig_tot_m, features_m_a, features_m_b, features_m_tot,
            grid_weights, h_core, j_total_m, e_nuc,
        )
```

Finally, update the step-2 line of the `_run_manual_scf_uks` docstring (line 463)
to read:

```
      2. three descriptor blocks: the alpha and beta blocks of
         diag(P_sigma, P_sigma) for exchange, the total block for correlation
```

Finally, the reporting value near the end of the function currently reads

```python
    features_final = _features_for(final_state.density_matrix)
```

`_features_for` now returns a triple, and `SCFResult.features_used` is the
physical descriptor record, so take the total block:

```python
    # SCFResult.features_used records the PHYSICAL descriptor block; the two
    # per-channel blocks are internal to the exchange spin scaling.
    _features_final_a, _features_final_b, features_final = _features_for(
        final_state.density_matrix)
```

- [ ] **Step 6: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/solver_manual.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_solver_manual.py xcquinox/alec/tests/test_uks_scf.py xcquinox/alec/tests/test_scf_backends.py -v > /tmp/xcq-testlogs/task05_green.log 2>&1; echo "exit=$?"
```
Expected: PASS.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_spin_scaling_solver_manual.py xcquinox/alec/tests/test_uks_scf.py xcquinox/alec/tests/test_scf_backends.py -v > /tmp/xcq-testlogs/task05_green.log 2>&1`

---

## Task 6: pyscfad backend UKS branch on three per-block slices

**Files:**
- Modify: `xcquinox/alec/solver_pyscfad.py:150-246` (`_reassemble_features_on_grid`), `:249-368` (`_make_alec_eval_xc` prologue and `_features_for_block`), `:399-511` (`eval_xc_alec_gga` UKS and RKS branches), `:677-752` (the feature holder and the `get_veff` wrapper)
- Test: `xcquinox/alec/tests/test_spin_scaling_pyscfad.py` (create)

**Interfaces:**
- Consumes: `descriptors.doubled_spin_dm` (Task 1), `assemble_descriptor_features(..., spin_channel=)` (Task 1).
- Produces: `solver_pyscfad._reassemble_features_on_grid(..., spin_channel=None)`; the feature holder gains `features_full_a` / `features_full_b`; the internal `_features_for_block(block_size)` returns `(features_tot, features_a, features_b)`.

- [ ] **Step 1: Write the failing test**

Create `xcquinox/alec/tests/test_spin_scaling_pyscfad.py`:

```python
"""The pyscfad backend's UKS callback on per-channel feature blocks."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pyscfad = pytest.importorskip("pyscfad")

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.descriptors import doubled_spin_dm
from xcquinox.alec.solver_pyscfad import _reassemble_features_on_grid


def _model(arch_name, seed=0):
    arch = dataclasses.replace(alec.get_architecture(arch_name),
                               use_polarized_correlation=True,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _md(model, name, atom, spin, composition, basis="def2-svp", grid_level=1):
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys}))
    return precompute_fixed_density_data(
        MoleculeSpec(name=name, atom=atom, basis=basis, charge=0, spin=spin,
                     atom_composition=composition, grid_level=grid_level),
        required_keys=keys, descriptors=model.descriptors)


def _mol(atom, spin, basis="def2-svp"):
    from pyscf import gto
    return gto.M(atom=atom, basis=basis, spin=spin, verbose=0)


def _uks_grid(mol, grid_level=1):
    """The parent's own grid for this molecule, built independently of the
    precompute so the on-grid reassembly is exercised on real coordinates."""
    from pyscf import dft
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = grid_level
    mf.grids.build()
    return jnp.asarray(mf.grids.coords)


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16",
                                       "deep_rung35ms_3x16", "deep_dm_3x16"])
def test_on_grid_reassembly_with_a_spin_channel_doubles_the_density_matrix(
        arch_name):
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    mol = _mol("Li 0 0 0", 1)
    P = jnp.asarray(md["dm_pbe"])
    kw = dict(descriptors=model.descriptors, s_matrix=jnp.asarray(md["s_matrix"]),
              grid_coords=_uks_grid(mol), mol=mol)
    channelled = _reassemble_features_on_grid(dm=P, spin_channel=0, **kw)
    explicit = _reassemble_features_on_grid(dm=doubled_spin_dm(P, 0), **kw)
    np.testing.assert_allclose(np.asarray(channelled), np.asarray(explicit),
                               rtol=0, atol=0)
    total = _reassemble_features_on_grid(dm=P, **kw)
    assert float(np.max(np.abs(np.asarray(channelled) - np.asarray(total)))) > 1e-6


def test_on_grid_reassembly_collapses_at_a_closed_shell_density():
    model = _model("deep_rung35_mgga_3x16")
    from pyscf import dft
    mol = _mol("O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", 0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.kernel()
    half = 0.5 * jnp.asarray(mf.make_rdm1())
    P = jnp.stack([half, half], axis=0)
    kw = dict(descriptors=model.descriptors,
              s_matrix=jnp.asarray(mol.intor("int1e_ovlp")),
              grid_coords=jnp.asarray(mf.grids.coords), mol=mol)
    a = np.asarray(_reassemble_features_on_grid(dm=P, spin_channel=0, **kw))
    b = np.asarray(_reassemble_features_on_grid(dm=P, spin_channel=1, **kw))
    t = np.asarray(_reassemble_features_on_grid(dm=P, **kw))
    np.testing.assert_allclose(a, t, rtol=0, atol=0)
    np.testing.assert_allclose(b, t, rtol=0, atol=0)


@pytest.mark.parametrize("arch_name", ["deep_rung35_3x16", "deep_mgga_3x16"])
def test_pyscfad_uks_scf_matches_the_manual_backend_energy(arch_name):
    """Both backends must evaluate the same functional: the same three blocks
    reach the same exchange and correlation terms."""
    from xcquinox.alec.solver import (
        FeaturePolicy, SolverBackend, SolverConfig, SolverMode, run_scf)
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    common = dict(mode=SolverMode.FULL, max_cycles=1,
                  feature_policy=FeaturePolicy.REASSEMBLE)
    e_manual = float(run_scf(SolverConfig(backend=SolverBackend.MANUAL,
                                          **common), model, md).total_energy)
    e_pyscfad = float(run_scf(SolverConfig(backend=SolverBackend.PYSCFAD,
                                           **common), model, md).total_energy)
    # The two backends integrate on different grids (pyscfad applies its own
    # small_rho_cutoff pruning), so the bound is a quadrature bound, not an
    # identity; it is far below the 1.0 mHa atomic tolerance of the fidelity
    # certificate and would be blown open by a wrong feature block.
    assert abs(e_manual - e_pyscfad) < 1e-4, (e_manual, e_pyscfad)
```

If `mol_metadata` does not carry `grid_coords`, the fallback branch above builds
the grid directly; keep both paths so the test does not depend on precompute
internals.

- [ ] **Step 2: Run and confirm it fails**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_pyscfad.py -v > /tmp/xcq-testlogs/task06_red.log 2>&1; echo "exit=$?"
```
Expected: `TypeError: _reassemble_features_on_grid() got an unexpected keyword argument 'spin_channel'`.

- [ ] **Step 3: Give the on-grid reassembly a spin channel**

In `xcquinox/alec/solver_pyscfad.py`, add the parameter to the signature (line 150-159):

```python
    metagga_ao: "jnp.ndarray | None" = None,
    spin_channel: "int | None" = None,
) -> "jnp.ndarray":
```

Replace lines 181-182 with:

```python
    dm_arr = jnp.asarray(dm)
    if spin_channel is not None:
        # Per-channel block: every density-matrix descriptor is evaluated on the
        # symmetric doubled density diag(P_sigma, P_sigma) (Oliver and Perdew,
        # Phys. Rev. A 20, 397 (1979)). Doubling here is sufficient for every
        # branch below: the meta-GGA branch derives rho, sigma and tau from
        # dm_arr itself, so it reads 2 rho_sigma, 4 sigma_sigma_sigma and
        # 2 tau_sigma without further change.
        from xcquinox.alec.descriptors import doubled_spin_dm
        dm_arr = doubled_spin_dm(dm_arr, spin_channel)
    # Keep dm_arr's ndim intact; compute_dm_features dispatches on it.
```

Append to the docstring:

```
    ``spin_channel`` selects the per-channel block of diag(P_sigma, P_sigma)
    instead of the physical block. It is the block the exact exchange spin
    scaling evaluates; correlation keeps ``spin_channel=None``.
```

- [ ] **Step 4: Make the block slicer return all three blocks**

In `_make_alec_eval_xc`, replace lines 294-297 with:

```python
    from xcquinox.alec.descriptors import assemble_descriptor_features

    features_frozen = assemble_descriptor_features(descriptors, mol_data)
    n_features = features_frozen.shape[1]
```

Replace `_features_for_block` (lines 299-367) with a version that slices all
three arrays and advances the offset once:

```python
    def _slice_block(features_full, offset: int, block_size: int) -> jnp.ndarray:
        """One block of a full-grid feature array, zero-padded on the tail.

        pyscfad's ``block_loop`` may emit non-uniform block sizes, the last
        block of an unpadded grid is smaller than NBLK, and ``non0tab`` pruning
        can skip blocks entirely while still advancing the internal cursor. Both
        cases produce a short slice. Padded grid points carry zero weight in the
        downstream numint summation, so the padding value never reaches the
        energy or the Fock matrix; only the shape contract matters.
        """
        features_slice = features_full[offset:offset + block_size]
        slice_n = features_slice.shape[0]
        if slice_n < block_size:
            pad = jnp.zeros((block_size - slice_n, features_slice.shape[1]),
                            dtype=features_slice.dtype)
            features_slice = jnp.concatenate([features_slice, pad], axis=0)
        elif slice_n > block_size:
            # Cannot happen from a Python slice; defensive only.
            raise ValueError(
                "Feature slice oversized: offset="
                f"{offset}, block_size={block_size}, slice={slice_n}, "
                f"full grid={features_full.shape[0]}. This indicates a bug in "
                "the slicing logic."
            )
        return features_slice

    def _features_for_block(block_size: int):
        """Return ``(features_tot, features_a, features_b)`` for one grid block.

        ``features_a`` / ``features_b`` are the descriptor features of the
        symmetric doubled densities diag(P_a, P_a) and diag(P_b, P_b), which the
        spin-scaled exchange terms evaluate (Oliver and Perdew, Phys. Rev. A 20,
        397 (1979)); ``features_tot`` is the physical block the spin-interpolated
        correlation term consumes. On a closed-shell (RKS) run the per-channel
        blocks are ``None`` and the caller uses the total block for all three,
        which is exact because rho_a = rho_b.

        The grid offset advances ONCE per call, so a caller that needs more than
        one block must take them from a single call.
        """
        if n_features == 0:
            empty = jnp.zeros((block_size, 0), dtype=features_frozen.dtype)
            return empty, empty, empty

        if feature_holder is not None:
            offset = int(feature_holder["offset"])
            tot = _slice_block(feature_holder["features_full"], offset,
                               block_size)
            per_spin = []
            for key in ("features_full_a", "features_full_b"):
                full = feature_holder.get(key)
                per_spin.append(None if full is None
                                else _slice_block(full, offset, block_size))
            feature_holder["offset"] = offset + block_size
            return tot, per_spin[0], per_spin[1]

        # Legacy path: use precompute features directly. Reached only when the
        # block loop returns the whole grid as one block AND pyscfad's grid
        # matches the precompute grid; the holder is installed for every
        # descriptor-carrying architecture, so no per-channel block is needed
        # here.
        if block_size == features_frozen.shape[0]:
            return features_frozen, None, None
        raise ValueError(
            "pyscfad backend with FROZEN features requires block_loop to "
            "return the full grid as one block, but got block_size="
            f"{block_size} != full grid {features_frozen.shape[0]}. This "
            "happens under jax.grad/jit tracing with descriptor-ful "
            "architectures; use REASSEMBLE policy to resolve."
        )
```

- [ ] **Step 5: Use the three blocks in the callback**

In `eval_xc_alec_gga`, replace line 432 with:

```python
            features_blk, features_blk_a, features_blk_b = _features_for_block(
                int(rho_a.shape[0]))
            if features_blk_a is None:
                # Closed shell fed through the UKS callback: rho_a = rho_b makes
                # the three blocks identical.
                features_blk_a = features_blk
            if features_blk_b is None:
                features_blk_b = features_blk
```

Replace the two exchange evaluations (lines 435-440) so the alpha call passes
`features_blk_a` and the beta call passes `features_blk_b`. Leave every
correlation evaluation on `features_blk`.

Replace the comment block at lines 419-431 so it states the per-channel rule:

```python
            # SOLV-01 split. EXCHANGE obeys the exact spin-scaling relation
            # (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)):
            #   E_x = 0.5 (E_x[2 rho_a, 4 sigma_aa] + E_x[2 rho_b, 4 sigma_bb]),
            # evaluated per spin, each channel at the descriptor block of its
            # OWN doubled density diag(P_sigma, P_sigma). CORRELATION does not
            # obey it: it is spin-interpolated and evaluated ONCE on the TOTAL
            # density with the total block (von Barth and Hedin, J. Phys. C 5,
            # 1629 (1972); Perdew and Wang, Phys. Rev. B 45, 13244 (1992)). When
            # cnet.use_spin_polarization is set, correlation uses the
            # zeta-dependent PW92 baseline and a per-spin vrho_c (Dick and
            # Fernandez-Serra, Phys. Rev. B 104, L161109 (2021)).
            #
            # Use a block-sized features slice since pyscfad chunks the grid
            # under jax.grad / jit tracing; one call yields all three blocks and
            # advances the grid offset exactly once.
```

In the RKS branch, replace line 519 with:

```python
        features_blk = _features_for_block(int(rho0.shape[0]))[0]
```

- [ ] **Step 6: Fill and refresh the per-channel holders**

In `run_pyscfad_scf`, replace the holder construction (lines 692-704) with:

```python
        def _blocks_on_grid(dm_value, mol_value):
            """Total block plus, for an open shell, the two per-channel blocks."""
            common = dict(
                descriptors=descriptors,
                s_matrix=jnp.asarray(mol_data["s_matrix"]),
                grid_coords=jnp.asarray(mf.grids.coords),
                mol=mol_value,
                rung35_proj_ao=_rung35_proj_ao,
                rung35ms_proj_ao=_rung35ms_proj_ao,
                metagga_ao=_metagga_ao,
            )
            total = _reassemble_features_on_grid(dm=dm_value, **common)
            if not is_uks:
                return total, None, None
            return (
                total,
                _reassemble_features_on_grid(dm=dm_value, spin_channel=0,
                                             **common),
                _reassemble_features_on_grid(dm=dm_value, spin_channel=1,
                                             **common),
            )

        _tot0, _a0, _b0 = _blocks_on_grid(mol_data["dm_pbe"], mol)
        feature_holder = {
            "features_full": _tot0,
            "features_full_a": _a0,
            "features_full_b": _b0,
            "offset": 0,
        }
```

Replace the `_holder_get_veff` refresh (lines 738-748) with:

```python
            if policy == FeaturePolicy.REASSEMBLE and dm is not None:
                (feature_holder["features_full"],
                 feature_holder["features_full_a"],
                 feature_holder["features_full_b"]) = _blocks_on_grid(
                    dm, mol_eff)
```

The `_blocks_on_grid` closure reads `mf.grids.coords` and the three cached
constant precomputes, all of which are fixed across the SCF, so it can be defined
once inside the `if descriptors:` block and reused by the wrapper. Hoist the
`_s_matrix` / `_grid_coords` locals at lines 725-726 into it or delete them if
they become unused.

- [ ] **Step 7: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/solver_pyscfad.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_pyscfad.py xcquinox/alec/tests/test_scf_backends.py xcquinox/alec/tests/test_pyscfad_gradflow.py -v > /tmp/xcq-testlogs/task06_green.log 2>&1; echo "exit=$?"
```
Expected: PASS.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_spin_scaling_pyscfad.py xcquinox/alec/tests/test_scf_backends.py xcquinox/alec/tests/test_pyscfad_gradflow.py -v > /tmp/xcq-testlogs/task06_green.log 2>&1`

---

## Task 7: Oracle O1 -- the parent functional wearing the model's evaluation surface

**Files:**
- Create: `xcquinox/alec/tests/parent_adapter.py`
- Create: `xcquinox/alec/tests/test_spin_scaling_oracles.py`

**Interfaces:**
- Consumes: `oneshot.split_exc_energy_uks` (Task 4), `assemble_descriptor_features(..., spin_channel=)` (Task 1), the precomputed blocks (Task 2).
- Produces:
  - `parent_adapter.LibxcParentModel(x_functional=None, c_functional=None, alpha_column=None, use_spin_polarization=True, descriptors=())` with `eval_ex(rho, sigma, features)`, `eval_ec(rho, sigma, features, zeta=0.0)`, `.cnet.use_spin_polarization`, `.descriptors`.
  - `parent_adapter.gga_rho_row(rho, nabla) -> np.ndarray` shape `(4, N)`.
  - `parent_adapter.tau_from_alpha(rho, sigma, alpha) -> np.ndarray`.

- [ ] **Step 1: Write the failing oracle tests**

Create `xcquinox/alec/tests/test_spin_scaling_oracles.py`:

```python
"""Oracles O1 and O4 of the pretraining-fidelity program, Section 3.1.

O1 replaces the network with the parent functional's own enhancement factors,
taken from libxc, and asks whether the library's UKS code path reproduces
libxc's spin-polarized evaluation on open-shell atoms. Any discrepancy is a
defect in the assembly rather than in a fit, because there is no fit left.

O4 is the H atom: one electron in one orbital, so the symmetric doubled density
diag(P_a, P_a) is a two-electron single-orbital system with tau = tau_W and
alpha identically zero, the rung-3.5 block is the doubled orbital's occupancy in
both spin slots, and the exchange energy is exactly half the model's
spin-unpolarized evaluation on that system.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import dft, gto

jax.config.update("jax_enable_x64", True)

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.descriptors import (
    DMRung35Descriptor, MetaGGAAlphaDescriptor, assemble_descriptor_features,
    doubled_spin_dm)
from xcquinox.alec.oneshot import split_exc_energy_uks, uks_zeta
from xcquinox.alec.solver import make_uks_feature_fns
from xcquinox.alec.tests.parent_adapter import (
    LibxcParentModel, gga_rho_row, tau_from_alpha)

# Open-shell atoms of the pools, in PySCF's 2S spin convention.
_ATOMS = [("H", 1), ("Li", 1), ("N", 3), ("O", 2)]


def _precompute(symbol, spin, descriptors, basis="def2-svp", grid_level=1):
    keys = tuple(sorted({k for d in descriptors for k in d.required_mol_keys}))
    return precompute_fixed_density_data(
        MoleculeSpec(name=symbol, atom=f"{symbol} 0 0 0", basis=basis,
                     charge=0, spin=spin, atom_composition=((symbol, 1),),
                     grid_level=grid_level),
        required_keys=keys, descriptors=descriptors)


def _spin_quantities(md, s):
    """(rho_sigma, nabla_rho_sigma (N, 3), sigma_sigma_sigma) for one channel."""
    ao = np.asarray(md["ao_grid_deriv"])
    d = np.asarray(md["dm_pbe"])[s]
    rho = np.einsum("pi,ij,pj->p", ao[0], d, ao[0])
    grad = np.stack([2 * np.einsum("pi,ij,pj->p", ao[k], d, ao[0])
                     for k in (1, 2, 3)], axis=-1)
    return rho, grad, np.sum(grad * grad, axis=1)


def _positive_mass_weights(md, rho_a, rho_b):
    """Grid weights with quadrature-noise-negative spin densities removed.

    libxc and the adapter each clamp a nonpositive density to zero exchange but
    need not clamp it identically; such points carry no integrand mass, so they
    are dropped from BOTH sides of the comparison rather than absorbed into the
    tolerance.
    """
    keep = (rho_a >= 0.0) & (rho_b >= 0.0)
    return np.asarray(md["grid_weights"]) * keep, int((~keep).sum())


@pytest.mark.parametrize("symbol,spin", _ATOMS)
def test_o1_exchange_path_equals_libxc_pbe_spin1(symbol, spin):
    """O1: with the parent's own F_x in place of the network, the library's UKS
    exchange assembly is libxc's spin-polarized PBE exchange."""
    md = _precompute(symbol, spin, ())
    rho_a, nabla_a, sigma_aa = _spin_quantities(md, 0)
    rho_b, nabla_b, sigma_bb = _spin_quantities(md, 1)
    nabla_tot = nabla_a + nabla_b
    sigma_tot = np.sum(nabla_tot * nabla_tot, axis=1)
    w, n_dropped = _positive_mass_weights(md, rho_a, rho_b)
    empty = jnp.zeros((w.shape[0], 0))
    parent = LibxcParentModel(x_functional="GGA_X_PBE")
    got = float(split_exc_energy_uks(
        parent, jnp.asarray(rho_a), jnp.asarray(rho_b),
        jnp.asarray(sigma_aa), jnp.asarray(sigma_bb), jnp.asarray(sigma_tot),
        empty, empty, empty, jnp.asarray(w)))
    rho_uks = np.stack([gga_rho_row(rho_a, nabla_a),
                        gga_rho_row(rho_b, nabla_b)])
    eps = np.asarray(dft.libxc.eval_xc("GGA_X_PBE", rho_uks, spin=1,
                                       deriv=0)[0])
    ref = float(np.sum(w * (rho_a + rho_b) * eps))
    assert abs(got - ref) < 1e-10, (symbol, got, ref, n_dropped)


@pytest.mark.parametrize("symbol,spin", _ATOMS)
def test_o1_correlation_path_equals_libxc_pbe_on_the_total_density(symbol, spin):
    """O1: unpolarized cnet flag -- correlation is the parent's own correlation
    evaluated once on the total density, exactly."""
    md = _precompute(symbol, spin, ())
    rho_a, nabla_a, _sig_aa = _spin_quantities(md, 0)
    rho_b, nabla_b, _sig_bb = _spin_quantities(md, 1)
    nabla_tot = nabla_a + nabla_b
    sigma_tot = np.sum(nabla_tot * nabla_tot, axis=1)
    w, _n = _positive_mass_weights(md, rho_a, rho_b)
    empty = jnp.zeros((w.shape[0], 0))
    parent = LibxcParentModel(c_functional="GGA_C_PBE",
                              use_spin_polarization=False)
    got = float(split_exc_energy_uks(
        parent, jnp.asarray(rho_a), jnp.asarray(rho_b),
        jnp.zeros_like(jnp.asarray(rho_a)), jnp.zeros_like(jnp.asarray(rho_b)),
        jnp.asarray(sigma_tot), empty, empty, empty, jnp.asarray(w)))
    row = gga_rho_row(rho_a + rho_b, nabla_tot)
    eps = np.asarray(dft.libxc.eval_xc("GGA_C_PBE", row, spin=0, deriv=0)[0])
    ref = float(np.sum(w * (rho_a + rho_b) * eps))
    assert abs(got - ref) < 1e-10, (symbol, got, ref)


@pytest.mark.parametrize("symbol,spin", _ATOMS)
def test_o1_polarized_correlation_tracks_libxc_within_the_zeta_clip(symbol, spin):
    """O1: polarized cnet flag -- correlation is libxc's spin-polarized PBE at
    the library's own zeta. The residual is the documented boundary clip
    ``oneshot._ZETA_BOUNDARY_EPS = 1e-6``, which holds |zeta| strictly inside 1
    so the PW92 spin interpolation stays twice differentiable."""
    from xcquinox.alec.oneshot import _ZETA_BOUNDARY_EPS
    md = _precompute(symbol, spin, ())
    rho_a, nabla_a, _sig_aa = _spin_quantities(md, 0)
    rho_b, nabla_b, _sig_bb = _spin_quantities(md, 1)
    nabla_tot = nabla_a + nabla_b
    sigma_tot = np.sum(nabla_tot * nabla_tot, axis=1)
    w, _n = _positive_mass_weights(md, rho_a, rho_b)
    empty = jnp.zeros((w.shape[0], 0))
    parent = LibxcParentModel(c_functional="GGA_C_PBE",
                              use_spin_polarization=True)
    got = float(split_exc_energy_uks(
        parent, jnp.asarray(rho_a), jnp.asarray(rho_b),
        jnp.zeros_like(jnp.asarray(rho_a)), jnp.zeros_like(jnp.asarray(rho_b)),
        jnp.asarray(sigma_tot), empty, empty, empty, jnp.asarray(w)))
    rho_uks = np.stack([gga_rho_row(rho_a, nabla_a),
                        gga_rho_row(rho_b, nabla_b)])
    eps = np.asarray(dft.libxc.eval_xc("GGA_C_PBE", rho_uks, spin=1,
                                       deriv=0)[0])
    ref = float(np.sum(w * (rho_a + rho_b) * eps))
    assert abs(got - ref) < 1e-6, (symbol, got, ref, _ZETA_BOUNDARY_EPS)


@pytest.mark.parametrize("symbol,spin", _ATOMS)
def test_o1_scan_exchange_path_equals_libxc_through_the_alpha_column(symbol, spin):
    """O1, the discriminating case: a meta-GGA parent reads the iso-orbital
    indicator out of the per-channel feature block. The library's UKS exchange
    assembly equals libxc's spin-polarized SCAN exchange only if that block
    carries alpha(2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma). Feeding the
    total-density block into both channels moves this by far more than 1e-10 Ha
    on every open-shell atom with more than one electron."""
    descriptors = (MetaGGAAlphaDescriptor(),)
    md = _precompute(symbol, spin, descriptors)
    rho_a, nabla_a, sigma_aa = _spin_quantities(md, 0)
    rho_b, nabla_b, sigma_bb = _spin_quantities(md, 1)
    nabla_tot = nabla_a + nabla_b
    sigma_tot = np.sum(nabla_tot * nabla_tot, axis=1)
    w, _n = _positive_mass_weights(md, rho_a, rho_b)
    f_a = assemble_descriptor_features(descriptors, md, spin_channel=0)
    f_b = assemble_descriptor_features(descriptors, md, spin_channel=1)
    f_tot = assemble_descriptor_features(descriptors, md)
    parent = LibxcParentModel(x_functional="MGGA_X_SCAN", alpha_column=0,
                              descriptors=descriptors)
    got = float(split_exc_energy_uks(
        parent, jnp.asarray(rho_a), jnp.asarray(rho_b),
        jnp.asarray(sigma_aa), jnp.asarray(sigma_bb), jnp.asarray(sigma_tot),
        f_a, f_b, f_tot, jnp.asarray(w)))
    # Reference at the tau the alpha column encodes. Inverting alpha rather than
    # recontracting tau keeps the descriptor's [0, _ALPHA_MAX] value clip out of
    # the comparison: the oracle asks whether the assembly is the parent's own,
    # not whether the clip is active in the deep tail.
    tau_a = 0.5 * tau_from_alpha(2.0 * rho_a, 4.0 * sigma_aa,
                                 np.asarray(f_a)[:, 0])
    tau_b = 0.5 * tau_from_alpha(2.0 * rho_b, 4.0 * sigma_bb,
                                 np.asarray(f_b)[:, 0])
    zeros = np.zeros_like(rho_a)
    mgga_a = np.vstack([gga_rho_row(rho_a, nabla_a), zeros, tau_a])
    mgga_b = np.vstack([gga_rho_row(rho_b, nabla_b), zeros, tau_b])
    eps = np.asarray(dft.libxc.eval_xc("MGGA_X_SCAN", (mgga_a, mgga_b),
                                       spin=1, deriv=0)[0])
    ref = float(np.sum(w * (rho_a + rho_b) * eps))
    assert abs(got - ref) < 1e-10, (symbol, got, ref)


@pytest.mark.parametrize("symbol,spin", [("Li", 1), ("N", 3), ("O", 2)])
def test_o1_per_channel_ingredients_are_the_libxc_spin_polarized_ingredients(
        symbol, spin):
    """O1, ingredient form: (2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma) is
    what libxc's spin-polarized meta-GGA reads for the channel, and the stored
    alpha column is exactly that alpha."""
    from xcquinox.alec.metagga import compute_alpha
    descriptors = (MetaGGAAlphaDescriptor(),)
    md = _precompute(symbol, spin, descriptors)
    mol = gto.M(atom=f"{symbol} 0 0 0", basis="def2-svp", spin=spin, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.kernel()
    ao2 = mf._numint.eval_ao(mol, mf.grids.coords, deriv=2)
    dm = np.asarray(md["dm_pbe"])
    for s, suffix in ((0, "_a"), (1, "_b")):
        rho_s, _grad, sigma_ss = _spin_quantities(md, s)
        tau_s = mf._numint.eval_rho(mol, ao2, dm[s], xctype="MGGA")[5]
        expect = np.asarray(compute_alpha(jnp.asarray(2.0 * rho_s),
                                          jnp.asarray(4.0 * sigma_ss),
                                          jnp.asarray(2.0 * tau_s)))
        got = np.asarray(md["metagga_features" + suffix])[:, 0]
        np.testing.assert_allclose(got, expect, rtol=0, atol=1e-10)


def test_o1_total_block_would_break_the_scan_oracle():
    """The superseded contract, exercised on purpose: feeding the total-density
    block into both exchange channels does NOT reproduce libxc, which is the
    measurement that makes the passing oracle meaningful rather than vacuous."""
    descriptors = (MetaGGAAlphaDescriptor(),)
    md = _precompute("N", 3, descriptors)
    rho_a, nabla_a, sigma_aa = _spin_quantities(md, 0)
    rho_b, nabla_b, sigma_bb = _spin_quantities(md, 1)
    nabla_tot = nabla_a + nabla_b
    sigma_tot = np.sum(nabla_tot * nabla_tot, axis=1)
    w, _n = _positive_mass_weights(md, rho_a, rho_b)
    f_tot = assemble_descriptor_features(descriptors, md)
    f_a = assemble_descriptor_features(descriptors, md, spin_channel=0)
    f_b = assemble_descriptor_features(descriptors, md, spin_channel=1)
    parent = LibxcParentModel(x_functional="MGGA_X_SCAN", alpha_column=0,
                              descriptors=descriptors)
    args = (jnp.asarray(rho_a), jnp.asarray(rho_b), jnp.asarray(sigma_aa),
            jnp.asarray(sigma_bb), jnp.asarray(sigma_tot))
    exact = float(split_exc_energy_uks(parent, *args, f_a, f_b, f_tot,
                                       jnp.asarray(w)))
    approx = float(split_exc_energy_uks(parent, *args, f_tot, f_tot, f_tot,
                                        jnp.asarray(w)))
    assert abs(exact - approx) > 1e-3, (exact, approx)
```

- [ ] **Step 2: Run and confirm it fails**

```bash
python -m pytest xcquinox/alec/tests/test_spin_scaling_oracles.py -v > /tmp/xcq-testlogs/task07_red.log 2>&1; echo "exit=$?"
```
Expected: `ModuleNotFoundError: No module named 'xcquinox.alec.tests.parent_adapter'`.

- [ ] **Step 3: Write the adapter**

Create `xcquinox/alec/tests/parent_adapter.py`:

```python
"""The parent functional wearing the model's evaluation surface.

The library's UKS energy path (``oneshot.split_exc_energy_uks``) touches a model
through exactly three names: ``eval_ex(rho, sigma, features)``,
``eval_ec(rho, sigma, features, zeta=...)`` and
``cnet.use_spin_polarization``. Substituting the parent functional for the
network turns that path into a pure quadrature of the parent, so the library's
own assembly can be compared against libxc's spin-polarized evaluation on the
same grid with no fit in the way. A discrepancy is then a defect in the
assembly.

This module lives in the test package and is deliberately NOT named ``test_*``,
so pytest imports it on demand rather than collecting it.
"""
import jax.numpy as jnp
import numpy as np
from pyscf import dft as _pyscf_dft

_LIBXC = _pyscf_dft.libxc


def gga_rho_row(rho, nabla_rho) -> np.ndarray:
    """libxc GGA input row ``(4, N)``: ``[rho, d/dx rho, d/dy rho, d/dz rho]``.

    ``nabla_rho`` is ``(N, 3)``, the layout the library stores.
    """
    r = np.asarray(rho, dtype=np.float64)
    g = np.asarray(nabla_rho, dtype=np.float64).reshape(r.shape[0], 3)
    return np.vstack([r[None, :], g.T])


def _row_from_sigma(rho, sigma, n_components) -> np.ndarray:
    """libxc input row encoding a KNOWN ``sigma`` rather than a real gradient.

    The gradient magnitude is placed in the x component and the other two are
    left at zero, so ``sigma_libxc = dx^2 + dy^2 + dz^2`` is the requested
    value. Only the invariant enters a GGA or meta-GGA, so this encoding is
    exact.
    """
    r = np.asarray(rho, dtype=np.float64)
    row = np.zeros((n_components, r.shape[0]), dtype=np.float64)
    row[0] = r
    row[1] = np.sqrt(np.clip(np.asarray(sigma, dtype=np.float64), 0.0, None))
    return row


def tau_from_alpha(rho, sigma, alpha) -> np.ndarray:
    """Invert the iso-orbital indicator: ``tau = alpha tau_unif + tau_W``.

    ``alpha = (tau - tau_W) / tau_unif`` with ``tau_W = sigma / (8 rho)`` and
    ``tau_unif = (3/10) (3 pi^2)^{2/3} rho^{5/3}`` (Sun, Ruzsinszky and Perdew,
    Phys. Rev. Lett. 115, 036402 (2015), Eq. 2). Inverting rather than
    recontracting the density matrix keeps the descriptor's value clip out of a
    comparison: whatever alpha the library assembled, this is the kinetic-energy
    density that alpha stands for.
    """
    r = np.maximum(np.asarray(rho, dtype=np.float64), 1e-300)
    s = np.asarray(sigma, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    tau_unif = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * r ** (5.0 / 3.0)
    return a * tau_unif + s / (8.0 * r)


class _PolarizationFlag:
    """Stand-in for the model's cnet, carrying only the flag the energy reads."""

    def __init__(self, use_spin_polarization: bool):
        self.use_spin_polarization = bool(use_spin_polarization)


class LibxcParentModel:
    """The parent functional with the model's evaluation surface.

    ``x_functional`` / ``c_functional`` are libxc names ("GGA_X_PBE",
    "GGA_C_PBE", "MGGA_X_SCAN", ...); ``None`` makes that channel return exactly
    zero, so exchange and correlation can be oracled independently.

    ``alpha_column`` (meta-GGA only) is the index of the iso-orbital indicator
    inside the feature block. The adapter inverts it to the kinetic-energy
    density the parent needs, so a meta-GGA parent reads precisely the alpha the
    library assembled. That is what makes the per-channel ingredients testable:
    under the exact spin scaling the alpha column of ``features_a`` is
    ``alpha(2 rho_a, 4 sigma_aa, 2 tau_a)`` and the reconstructed tau is
    ``2 tau_a``, which is the alpha channel's ingredient in libxc's own
    spin-polarized meta-GGA.

    Exchange is evaluated SPIN-UNPOLARIZED at the arguments it is handed. That
    is the correct surface: the caller has already applied the Oliver-Perdew
    doubling (Phys. Rev. A 20, 397 (1979)), so the adapter must not double
    again. Correlation is evaluated through libxc's spin-polarized entry point
    at ``(rho_a, rho_b) = rho (1 +- zeta) / 2`` with the two spin gradients
    taken parallel and proportional to the spin densities, so that
    ``sigma_aa + 2 sigma_ab + sigma_bb`` reproduces the requested total
    invariant exactly. PBE correlation is a functional of the total-density
    gradient alone, so that choice is exact rather than approximate.
    """

    def __init__(self, x_functional: str | None = None,
                 c_functional: str | None = None,
                 alpha_column: int | None = None,
                 use_spin_polarization: bool = True,
                 descriptors: tuple = ()):
        self.x_functional = x_functional
        self.c_functional = c_functional
        self.alpha_column = alpha_column
        self.cnet = _PolarizationFlag(use_spin_polarization)
        self.descriptors = tuple(descriptors)

    def eval_ex(self, rho, sigma, features):
        """``rho * eps_x^parent`` at the arguments handed in, spin-unpolarized."""
        r = np.asarray(rho, dtype=np.float64)
        if self.x_functional is None:
            return jnp.zeros(r.shape[0])
        positive = r > 0.0
        r_safe = np.where(positive, r, 1.0)
        s = np.asarray(sigma, dtype=np.float64)
        if self.alpha_column is None:
            row = _row_from_sigma(r_safe, s, 4)
        else:
            alpha = np.asarray(features, dtype=np.float64)[:, self.alpha_column]
            row = _row_from_sigma(r_safe, s, 6)
            row[5] = tau_from_alpha(r_safe, s, alpha)
        eps = np.asarray(
            _LIBXC.eval_xc(self.x_functional, row, spin=0, deriv=0)[0])
        # A nonpositive density carries no exchange; masking here matches libxc's
        # own clamp on an empty spin channel and keeps a quadrature-noise
        # negative from entering the integrand.
        return jnp.asarray(np.where(positive, r * eps, 0.0))

    def eval_ec(self, rho, sigma, features, zeta=0.0):
        """``rho * eps_c^parent(rho, sigma, zeta)``, spin-polarized in zeta."""
        del features
        r = np.asarray(rho, dtype=np.float64)
        if self.c_functional is None:
            return jnp.zeros(r.shape[0])
        positive = r > 0.0
        z = np.asarray(zeta, dtype=np.float64) * np.ones_like(r)
        r_safe = np.where(positive, r, 1.0)
        rho_a = 0.5 * r_safe * (1.0 + z)
        rho_b = 0.5 * r_safe * (1.0 - z)
        g = np.sqrt(np.clip(np.asarray(sigma, dtype=np.float64), 0.0, None))
        row_a = np.zeros((4, r.shape[0]), dtype=np.float64)
        row_b = np.zeros((4, r.shape[0]), dtype=np.float64)
        row_a[0] = rho_a
        row_b[0] = rho_b
        row_a[1] = g * rho_a / r_safe
        row_b[1] = g * rho_b / r_safe
        eps = np.asarray(_LIBXC.eval_xc(
            self.c_functional, np.stack([row_a, row_b]), spin=1, deriv=0)[0])
        return jnp.asarray(np.where(positive, r * eps, 0.0))
```

- [ ] **Step 4: Run the oracle**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/tests/parent_adapter.py xcquinox/alec/tests/test_spin_scaling_oracles.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_oracles.py -v > /tmp/xcq-testlogs/task07_green.log 2>&1; echo "exit=$?"
```
Expected: PASS. If `test_o1_exchange_path_equals_libxc_pbe_spin1` misses by more than 1e-10, the failure is in the doubling convention or in the grid mask, not in the tolerance: raise the mask, never the bound.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_spin_scaling_oracles.py -v > /tmp/xcq-testlogs/task07_green.log 2>&1`

---

## Task 8: Oracle O4 -- the H atom

**Files:**
- Modify: `xcquinox/alec/tests/test_spin_scaling_oracles.py` (append)

**Interfaces:**
- Consumes: everything Task 7 imports, plus `solver.make_uks_feature_fns` (Task 3) and `descriptors.doubled_spin_dm` (Task 1).
- Produces: nothing consumed downstream.

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_spin_scaling_oracles.py`:

```python
# ---------------------------------------------------------------------------
# O4: the H atom. One electron, one orbital, fully polarized.
# ---------------------------------------------------------------------------

def _live_model(arch_name, seed=0):
    arch = dataclasses.replace(alec.get_architecture(arch_name),
                               use_polarized_correlation=True,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def test_o4_h_atom_alpha_is_zero_at_every_grid_point():
    """diag(P_a, P_a) is a two-electron single-orbital system, so tau = tau_W
    exactly and the iso-orbital indicator vanishes identically (Sun, Ruzsinszky
    and Perdew, Phys. Rev. Lett. 115, 036402 (2015): alpha = 0 marks a single
    orbital)."""
    md = _precompute("H", 1, (MetaGGAAlphaDescriptor(),))
    alpha_a = np.asarray(md["metagga_features_a"])[:, 0]
    assert float(np.max(np.abs(alpha_a))) < 1e-8, float(np.max(np.abs(alpha_a)))


def test_o4_h_atom_rung35_block_is_the_doubled_single_orbital():
    """The alpha channel's block is [n_a, n_a] -- the occupancy of the doubled
    orbital in BOTH spin slots -- while the physical block is [n_a, 0]. The two
    are not the same feature vector, which is the whole content of the fix on a
    one-electron system."""
    md = _precompute("H", 1, (DMRung35Descriptor(),))
    block = np.asarray(md["rung35_features_a"])
    total = np.asarray(md["rung35_features"])
    np.testing.assert_allclose(block[:, 0], block[:, 1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(block[:, 0], total[:, 0], rtol=0, atol=1e-14)
    assert float(np.max(np.abs(total[:, 1]))) < 1e-14, "H has no beta electron"
    assert float(np.max(block[:, 1])) > 1e-3, (
        "the doubled system's second slot must carry the SAME occupancy as the "
        "first, not the empty physical beta channel")
    assert float(np.max(block)) < 1.0 + 1e-12, "Bessel bound"
    assert float(np.min(block)) > -1e-12, "positive semidefinite P"


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16",
                                       "deep_rung35ms_3x16",
                                       "deep_cusp_3x16", "deep_dm_3x16"])
def test_o4_h_atom_exchange_equals_the_spin_scaled_unpolarized_evaluation(
        arch_name):
    """The alpha channel's block IS the block an RKS run on diag(P_a, P_a) would
    assemble, so the H-atom exchange energy is exactly half the model's
    spin-unpolarized evaluation on that system. The beta channel is empty and
    contributes only the model's rho_cutoff floor."""
    model = _live_model(arch_name)
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys}))
    md = precompute_fixed_density_data(
        MoleculeSpec(name="H", atom="H 0 0 0", basis="def2-svp", charge=0,
                     spin=1, atom_composition=(("H", 1),), grid_level=1),
        required_keys=keys, descriptors=model.descriptors)
    rho_a, _grad_a, sigma_aa = _spin_quantities(md, 0)
    rho_b, _grad_b, sigma_bb = _spin_quantities(md, 1)
    w = jnp.asarray(md["grid_weights"])
    features_a_of, features_b_of, features_tot_of = make_uks_feature_fns(
        descriptors=model.descriptors,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"))
    P0 = jnp.asarray(md["dm_pbe"])
    doubled = doubled_spin_dm(P0, 0)
    # The channel block is the doubled system's OWN total block. Under the
    # superseded contract the left side was the physical molecular block and
    # this equality did not hold.
    np.testing.assert_allclose(np.asarray(features_a_of(P0)),
                               np.asarray(features_tot_of(doubled)),
                               rtol=0, atol=1e-14)
    ex_uks = 0.5 * float(jnp.sum(w * (
        model.eval_ex(jnp.asarray(2.0 * rho_a), jnp.asarray(4.0 * sigma_aa),
                      features_a_of(P0))
        + model.eval_ex(jnp.asarray(2.0 * rho_b), jnp.asarray(4.0 * sigma_bb),
                        features_b_of(P0)))))
    ex_rks_doubled = float(jnp.sum(w * model.eval_ex(
        jnp.asarray(2.0 * rho_a), jnp.asarray(4.0 * sigma_aa),
        features_tot_of(doubled))))
    assert abs(ex_uks - 0.5 * ex_rks_doubled) < 1e-12, (
        arch_name, ex_uks, 0.5 * ex_rks_doubled)


def test_o4_h_atom_beta_channel_carries_no_density():
    """The precondition the previous test rests on, stated separately so a
    failure names the cause."""
    md = _precompute("H", 1, ())
    rho_b, _grad, sigma_bb = _spin_quantities(md, 1)
    assert float(np.max(np.abs(rho_b))) < 1e-14
    assert float(np.max(np.abs(sigma_bb))) < 1e-14
```

- [ ] **Step 2: Run and confirm the state before the implementation**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_oracles.py -k "o4" -v > /tmp/xcq-testlogs/task08_red.log 2>&1; echo "exit=$?"
```
Expected: PASS on the current tree, because Tasks 1-7 already installed the behavior these tests describe. This task is a coverage task: its RED evidence is the deliberate-regression check in Step 3, not a failing run here. If any O4 test fails at this point, stop and fix the implementation before continuing.

- [ ] **Step 3: Prove the O4 tests discriminate**

Temporarily change `descriptors.Descriptor.compute_for_spin_channel` to `return self.compute(mol_data)` unconditionally (the superseded contract), re-run, confirm `test_o4_h_atom_rung35_block_is_the_doubled_single_orbital` and `test_o4_h_atom_exchange_equals_the_spin_scaled_unpolarized_evaluation` FAIL, then restore the method exactly.

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_oracles.py -k "o4" -v > /tmp/xcq-testlogs/task08_discrimination.log 2>&1; echo "exit=$?"
```
Expected while the method is reverted: FAIL on both named tests. Restore, re-run, expect PASS.

- [ ] **Step 4: Compile and run the whole oracle module**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/tests/test_spin_scaling_oracles.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_spin_scaling_oracles.py -v > /tmp/xcq-testlogs/task08_green.log 2>&1; echo "exit=$?"
```
Expected: PASS.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_spin_scaling_oracles.py -v > /tmp/xcq-testlogs/task08_green.log 2>&1`

---

## Task 9: Oracle O2 -- central-difference Fock check on H, Li, N, O and at the production identity

**Files:**
- Modify: `xcquinox/alec/tests/test_solv01_split_xc.py` -- `test_fd_consistency_live_features_uks_polarized` (as re-pointed in Task 4) and the tolerance commentary above `_tolerances` (lines 330-395)
- Test: same file

**Interfaces:**
- Consumes: `_live_uks_features_fns` (Task 4), `solver.make_uks_feature_fns` (Task 3), the three-block `split_exc_energy_uks` (Task 4).
- Produces: nothing consumed downstream.

- [ ] **Step 1: Replace the open-shell FD test with the species-parametrized form**

In `xcquinox/alec/tests/test_solv01_split_xc.py`, add above the test:

```python
# Open-shell atoms of the BH76 / W4-11 pools, in PySCF's 2S spin convention,
# with the number of doubly-unoccupied beta orbitals recorded: H is fully
# polarized (nocc_b == 0), so its beta density matrix is identically zero and
# stays zero under the SCF. Perturbing that block would move the beta density
# off zero and straddle every low-density guard, which is why the probe below
# perturbs only the occupied channels.
_UKS_FD_SPECIES = {
    "H": ("H 0 0 0", 1, (("H", 1),)),
    "Li": ("Li 0 0 0", 1, (("Li", 1),)),
    "N": ("N 0 0 0", 3, (("N", 1),)),
    "O": ("O 0 0 0", 2, (("O", 1),)),
}


def _uks_fd_perturbation(P0, md, seed=20260821):
    """Symmetric perturbation of the spin density matrices for the FD probe.

    A spin channel with no occupied orbital carries an identically zero density
    matrix that the SCF never populates, so its Fock block is not part of the
    functional's domain; perturbing it drives the spin density through zero and
    straddles the low-density guards at every point. Such a channel is left
    unperturbed and drops out of both sides of the comparison.
    """
    W = np.asarray(_symmetric_perturbation(P0.shape, seed=seed))
    for s, key in ((0, "nocc_a"), (1, "nocc_b")):
        if int(md[key]) == 0:
            W[s] = 0.0
    return jnp.asarray(W)
```

Then replace the test's parametrize decorator and its molecule/perturbation setup
so the body reads:

```python
@pytest.mark.parametrize("arch_name", sorted(alec.ARCHITECTURES))
@pytest.mark.parametrize("species", sorted(_UKS_FD_SPECIES))
def test_fd_consistency_live_features_uks_polarized(arch_name, species):
    """Oracle O2. Open-shell, polarized correlation -- the production
    configuration -- on every open-shell atom of the pools.

    Exercises all four feature-derivative sites: the two spin-scaled exchange
    channels, each at the block of its own doubled density diag(P_sigma,
    P_sigma), and ``compute_vc_polarized_per_spin`` on the total block, plus the
    three chain-rule contractions that differentiate the three P -> f maps.
    """
    from xcquinox.alec.oneshot import (
        compute_vxc_nn, compute_vc_polarized_per_spin,
        feature_energy_derivative, feature_response_vxc,
        has_dm_dependent_descriptor, uks_zeta,
        _ZETA_BOUNDARY_EPS, _RHO_TOT_FLOOR)
    from xcquinox.alec.models import _NN_TAIL_THRESHOLD

    atom, spin, composition = _UKS_FD_SPECIES[species]
    model = _live_model(arch_name)
    md = _md_with_descriptors(model, species, atom, "def2-svp", spin,
                              composition)
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_deriv[1:4]
    features_a_of, features_b_of, features_tot_of = _live_uks_features_fns(
        model, md)

    dm = np.asarray(md["dm_pbe"])
    assert dm.ndim == 3, f"{species} spin={spin} must precompute a spin-resolved DM"
    P0 = jnp.asarray(dm)
    W = _uks_fd_perturbation(P0, md)

    def spin_quantities(D):
        rho = jnp.einsum("ij,gi,gj->g", D, ao_grid, ao_grid)
        nabla = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_xyz, ao_grid)
        return rho, nabla, jnp.sum(nabla * nabla, axis=1)

    def guard_status(P):
        rho_a = np.asarray(spin_quantities(P[0])[0])
        rho_b = np.asarray(spin_quantities(P[1])[0])
        rho_tot = rho_a + rho_b
        zeta = (rho_a - rho_b) / np.maximum(rho_tot, _RHO_TOT_FLOOR)
        return np.stack([
            np.abs(zeta) >= 1.0 - _ZETA_BOUNDARY_EPS,
            rho_tot <= _RHO_TOT_FLOOR,
            2.0 * rho_a <= _NN_TAIL_THRESHOLD,
            2.0 * rho_b <= _NN_TAIL_THRESHOLD,
            rho_a <= 1e-10, rho_b <= 1e-10, rho_tot <= 1e-10,
        ])

    keep = ~np.any(guard_status(P0 + _FD_EPS * W)
                   != guard_status(P0 - _FD_EPS * W), axis=0)
    assert keep.sum() > 0.9 * keep.size, (
        f"{species}: guard-straddle mask discarded more than 10% of the grid; "
        "the perturbation is too large to probe the smooth part of the "
        "functional"
    )
    weights = jnp.asarray(md["grid_weights"]) * jnp.asarray(keep,
                                                            dtype=jnp.float64)

    def energy(P):
        rho_a, nabla_a, sigma_aa = spin_quantities(P[0])
        rho_b, nabla_b, sigma_bb = spin_quantities(P[1])
        nabla_tot = nabla_a + nabla_b
        return split_exc_energy_uks(
            model, rho_a, rho_b, sigma_aa, sigma_bb,
            jnp.sum(nabla_tot * nabla_tot, axis=1),
            features_a_of(P), features_b_of(P), features_tot_of(P), weights)

    rho_a, nabla_a, sigma_aa = spin_quantities(P0[0])
    rho_b, nabla_b, sigma_bb = spin_quantities(P0[1])
    nabla_tot = nabla_a + nabla_b
    sigma_tot = jnp.sum(nabla_tot * nabla_tot, axis=1)
    f0_a, f0_b, f0_tot = features_a_of(P0), features_b_of(P0), features_tot_of(P0)

    V_a = compute_vxc_nn(model, 2.0 * rho_a, 4.0 * sigma_aa, f0_a, ao_grid,
                         weights, nabla_rho=2.0 * nabla_a, ao_grad=ao_deriv,
                         part="x")
    V_b = compute_vxc_nn(model, 2.0 * rho_b, 4.0 * sigma_bb, f0_b, ao_grid,
                         weights, nabla_rho=2.0 * nabla_b, ao_grad=ao_deriv,
                         part="x")
    vc_a, vc_b = compute_vc_polarized_per_spin(
        model, rho_a, rho_b, sigma_tot, f0_tot, ao_grid, weights, nabla_tot,
        ao_deriv)
    V_a, V_b = V_a + vc_a, V_b + vc_b

    if has_dm_dependent_descriptor(model):
        # f_a, f_b and f_tot are three different maps of P, so the chain-rule
        # term is three contractions rather than one accumulated de/df.
        v_feat = feature_response_vxc(
            0.5 * feature_energy_derivative(
                model, 2.0 * rho_a, 4.0 * sigma_aa, f0_a, part="x"),
            weights, features_a_of, P0)
        v_feat = v_feat + feature_response_vxc(
            0.5 * feature_energy_derivative(
                model, 2.0 * rho_b, 4.0 * sigma_bb, f0_b, part="x"),
            weights, features_b_of, P0)
        v_feat = v_feat + feature_response_vxc(
            feature_energy_derivative(
                model, rho_a + rho_b, sigma_tot, f0_tot, part="c",
                zeta=uks_zeta(rho_a, rho_b)),
            weights, features_tot_of, P0)
        V_a, V_b = V_a + v_feat[0], V_b + v_feat[1]

    assert bool(jnp.all(jnp.isfinite(V_a)) and jnp.all(jnp.isfinite(V_b))), (
        f"{arch_name}/{species}: polarized UKS V_xc has NaN/inf")

    analytic = float(jnp.sum(V_a * W[0]) + jnp.sum(V_b * W[1]))
    fd = float((energy(P0 + _FD_EPS * W) - energy(P0 - _FD_EPS * W))
               / (2.0 * _FD_EPS))
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)

    _rks_tol, tol, blocked_by = _tolerances(model)
    assert rel < tol, (
        f"{arch_name}/{species}: polarized UKS V_xc is not dE_xc/dP with live "
        f"per-channel features (FD={fd:.6e} analytic={analytic:.6e} "
        f"rel={rel:.3e} > {tol:.0e}"
        + (f", bound set by the known {blocked_by} defect)" if blocked_by
           else ")")
    )
```

- [ ] **Step 2: Add the production-identity case**

Append to the same file:

```python
@pytest.mark.slow
@pytest.mark.parametrize("arch_name", sorted(alec.ARCHITECTURES))
def test_fd_consistency_uks_polarized_production_identity(arch_name):
    """Oracle O2 at the production identity: 6-311++G(3df,2pd), grid level 3.

    The def2-svp probe above runs on every architecture and every open-shell
    atom of the pools on each test invocation; this one carries the identity the
    campaign actually reports and is marked slow so it is opt-in
    (``-m slow``). The two differ only in basis and grid; the assertion is the
    same statement that the assembled Fock matrices are the derivative of the
    assembled energy.
    """
    model = _live_model(arch_name)
    md = _md_with_descriptors(model, "N", "N 0 0 0", "6-311++G(3df,2pd)", 3,
                              (("N", 1),), grid_level=3)
    _assert_uks_fd_consistency(model, md, arch_name, "N/production")
```

and factor the body of the def2-svp test from `def spin_quantities(D):` through
the final `assert rel < tol` into a module-level helper
`_assert_uks_fd_consistency(model, md, arch_name, label)` that both tests call,
so the two cannot drift apart. The helper takes `md` already precomputed and
builds `features_a_of / features_b_of / features_tot_of` with
`_live_uks_features_fns(model, md)` and `W` with `_uks_fd_perturbation(P0, md)`.

- [ ] **Step 3: Update the tolerance commentary**

The block comment above `_tolerances` (lines 330-395) records the measured
residuals of the superseded single-block contract. Replace the two paragraphs
that describe the open-shell probe with newly measured numbers from
`/tmp/xcq-testlogs/task09_green.log`. Record, for the def2-svp probe: the worst
observed relative residual across `(architecture, species)`, the species and
architecture that produced it, and the resulting margin against `_TOL_UKS`. Do
not change `_TOL_UKS` unless the measured worst case exceeds it; if it does,
that is a defect in the three-contraction feature response, not a tolerance
problem, and the task stops until it is found. Keep the existing eps-sweep
paragraph (it justifies `_FD_EPS = 1e-6` and is unaffected).

- [ ] **Step 4: Run and record**

```bash
python -m py_compile xcquinox/alec/tests/test_solv01_split_xc.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_solv01_split_xc.py -v > /tmp/xcq-testlogs/task09_green.log 2>&1; echo "exit=$?"
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_solv01_split_xc.py -m slow -v > /tmp/xcq-testlogs/task09_slow.log 2>&1; echo "exit=$?"
```
Expected: PASS in both. Read both logs with `Read`. The residuals move with
evaluation order across architectures (measured up to ~5x, documented in the
file's own block comment), so quote the values produced by the parametrized run
above -- those are what the assertions actually see. To read the numbers out,
temporarily lower `_TOL_UKS` to `1e-30`, re-run into
`/tmp/xcq-testlogs/task09_residuals.log`, read every failure message (each
carries `rel=`), then restore `_TOL_UKS` to `5e-7` and confirm the suite is
green again.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_solv01_split_xc.py -v > /tmp/xcq-testlogs/task09_green.log 2>&1` and `python -m pytest xcquinox/alec/tests/test_solv01_split_xc.py -m slow -v > /tmp/xcq-testlogs/task09_slow.log 2>&1`

---

## Task 10: Oracle O3 -- closed-shell byte identity against the archived tree

**Files:**
- Create: `xcquinox/alec/tests/record_closed_shell_reference.py`
- Create: `xcquinox/alec/tests/fixtures/closed_shell_reference_ae204537e.json`
- Create: `xcquinox/alec/tests/test_closed_shell_byte_identity.py`

**Interfaces:**
- Consumes: `oneshot.fixed_density_total_energy`, `oneshot.compute_vxc_nn`, `oneshot.split_exc_energy_uks` -- the first two unchanged across the two trees, the third handled by arity dispatch.
- Produces: `record_closed_shell_reference.closed_shell_record(arch_name) -> dict[str, float]` with keys `E_rks`, `V_rks_trace`, `V_rks_sq`, `E_uks_closed`, `V_uks_a_trace`, `V_uks_a_sq`.

- [ ] **Step 1: Write the recorder**

Create `xcquinox/alec/tests/record_closed_shell_reference.py`:

```python
"""Closed-shell reference record, computed from whichever xcquinox is importable.

rho_a = rho_b makes the three per-channel feature blocks identical by
construction (doubling either channel of [D/2, D/2] reproduces the matrix, and
2 rho_a / 4 sigma_aa are then rho_tot / sigma_tot), so the exact spin scaling
must leave every closed-shell number untouched. This script computes those
numbers so the SAME script, run against an archived tree and against the working
tree, produces two records that can be compared digit for digit.

Reference values for the fixture are produced against the tree at ae204537e:

    mkdir -p /tmp/xcq-ae204537e
    git archive ae204537e | tar -x -C /tmp/xcq-ae204537e
    PYTHONPATH=/tmp/xcq-ae204537e JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        python xcquinox/alec/tests/record_closed_shell_reference.py \\
        > xcquinox/alec/tests/fixtures/closed_shell_reference_ae204537e.json

The header line printed on stderr names the loaded package; confirm it points
into /tmp/xcq-ae204537e before accepting the output.
"""
import inspect
import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.descriptors import assemble_descriptor_features
from xcquinox.alec.oneshot import (
    compute_vxc_nn, fixed_density_total_energy, split_exc_energy_uks)

# Closed-shell probe: a molecule that visits every descriptor (three nuclei for
# the cusp feature, a genuine density matrix for the rung-3.5 and DM statistics
# columns, a non-uniform iso-orbital indicator for the meta-GGA column).
_SPEC = dict(name="H2O_closed_shell_reference",
             atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
             basis="def2-svp", charge=0, spin=0,
             atom_composition=(("O", 1), ("H", 2)), grid_level=1)


def _build_model(arch_name):
    """Production configuration, fixed seed, identical in both trees."""
    import dataclasses
    arch = dataclasses.replace(alec.get_architecture(arch_name),
                               use_polarized_correlation=True,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _split_energy(model, rho_a, rho_b, sig_aa, sig_bb, sig_tot, features, w):
    """Call the split UKS energy under either arity.

    At rho_a = rho_b the three per-channel blocks are the SAME array, so passing
    one block three times is the closed-shell case rather than a compatibility
    shim; the archived tree took a single block because it had no per-channel
    notion at all.
    """
    n_params = len(inspect.signature(split_exc_energy_uks).parameters)
    if n_params == 8:
        return split_exc_energy_uks(model, rho_a, rho_b, sig_aa, sig_bb,
                                    sig_tot, features, w)
    return split_exc_energy_uks(model, rho_a, rho_b, sig_aa, sig_bb, sig_tot,
                                features, features, features, w)


def closed_shell_record(arch_name) -> dict:
    """The six numbers that pin this architecture's closed-shell behavior."""
    model = _build_model(arch_name)
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys}))
    md = precompute_fixed_density_data(MoleculeSpec(**_SPEC),
                                       required_keys=keys,
                                       descriptors=model.descriptors)
    features = assemble_descriptor_features(model.descriptors, md)
    ao = jnp.asarray(md["ao_grid"])
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_deriv[1:4]
    w = jnp.asarray(md["grid_weights"])

    e_rks = float(fixed_density_total_energy(model, md))
    v_rks = compute_vxc_nn(model, jnp.asarray(md["rho_grid"]),
                           jnp.asarray(md["sigma_grid"]), features, ao, w,
                           nabla_rho=jnp.asarray(md["nabla_rho_grid"]),
                           ao_grad=ao_deriv)

    # The same molecule fed through the UKS helpers as a closed shell:
    # D_a = D_b = D / 2, so rho_a = rho_b and the spin channels coincide.
    D_half = 0.5 * jnp.asarray(md["dm_pbe"])

    def grid(D):
        rho = jnp.einsum("ij,gi,gj->g", D, ao, ao)
        nabla = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_xyz, ao)
        return rho, nabla, jnp.sum(nabla * nabla, axis=1)

    rho_h, nabla_h, sig_h = grid(D_half)
    nabla_tot = 2.0 * nabla_h
    sig_tot = jnp.sum(nabla_tot * nabla_tot, axis=1)
    e_uks = float(_split_energy(model, rho_h, rho_h, sig_h, sig_h, sig_tot,
                                features, w))
    v_uks_a = compute_vxc_nn(model, 2.0 * rho_h, 4.0 * sig_h, features, ao, w,
                             nabla_rho=2.0 * nabla_h, ao_grad=ao_deriv,
                             part="x") \
        + compute_vxc_nn(model, 2.0 * rho_h, sig_tot, features, ao, w,
                         nabla_rho=nabla_tot, ao_grad=ao_deriv, part="c")
    return {
        "E_rks": e_rks,
        "V_rks_trace": float(jnp.sum(v_rks)),
        "V_rks_sq": float(jnp.sum(v_rks * v_rks)),
        "E_uks_closed": e_uks,
        "V_uks_a_trace": float(jnp.sum(v_uks_a)),
        "V_uks_a_sq": float(jnp.sum(v_uks_a * v_uks_a)),
    }


def main():
    print(f"# xcquinox loaded from {sys.modules['xcquinox'].__file__}",
          file=sys.stderr)
    record = {name: closed_shell_record(name)
              for name in sorted(alec.ARCHITECTURES)}
    json.dump(record, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing assertion**

Create `xcquinox/alec/tests/test_closed_shell_byte_identity.py`:

```python
"""Oracle O3: closed-shell results are unchanged, digit for digit.

rho_a = rho_b makes the three per-channel feature blocks identical, so the exact
spin scaling has no closed-shell content at all: RKS and closed-shell UKS
energies and potentials must reproduce the archived tree exactly, not merely
within a tolerance. The reference numbers were produced by
``record_closed_shell_reference.py`` run against the tree at ae204537e.
"""
import json
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import xcquinox.alec as alec
from xcquinox.alec.tests.record_closed_shell_reference import (
    closed_shell_record)

_FIXTURE = (Path(__file__).parent / "fixtures"
            / "closed_shell_reference_ae204537e.json")
_REFERENCE = json.loads(_FIXTURE.read_text())


def test_the_reference_covers_every_architecture():
    assert set(_REFERENCE) == set(alec.ARCHITECTURES), (
        "the archived reference and the live architecture registry disagree; "
        "regenerate the fixture with record_closed_shell_reference.py"
    )


@pytest.mark.parametrize("arch_name", sorted(alec.ARCHITECTURES))
def test_closed_shell_results_are_byte_identical_to_the_archived_tree(arch_name):
    reference = _REFERENCE[arch_name]
    got = closed_shell_record(arch_name)
    assert set(got) == set(reference)
    for key in sorted(reference):
        assert got[key] == reference[key], (
            f"{arch_name}.{key}: {got[key]!r} != archived {reference[key]!r}. "
            "Closed-shell results carry no per-channel content -- rho_a = rho_b "
            "makes the three feature blocks the same array -- so any movement "
            "here is an unintended change to the shared code path."
        )
```

- [ ] **Step 3: Run and confirm it fails**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_closed_shell_byte_identity.py -v > /tmp/xcq-testlogs/task10_red.log 2>&1; echo "exit=$?"
```
Expected: collection error, `FileNotFoundError: ... closed_shell_reference_ae204537e.json`.

- [ ] **Step 4: Produce the reference from the archived tree**

```bash
mkdir -p /tmp/xcq-ae204537e
cd /home/awills/Documents/Research/xcquinox && git archive ae204537e | tar -x -C /tmp/xcq-ae204537e
cd /home/awills/Documents/Research/xcquinox && PYTHONPATH=/tmp/xcq-ae204537e JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python xcquinox/alec/tests/record_closed_shell_reference.py > xcquinox/alec/tests/fixtures/closed_shell_reference_ae204537e.json 2> /tmp/xcq-testlogs/task10_record_archived.log; echo "exit=$?"
```
Read `/tmp/xcq-testlogs/task10_record_archived.log` with `Read` and confirm its
first line reads `# xcquinox loaded from /tmp/xcq-ae204537e/xcquinox/__init__.py`.
If it names the working tree instead, the archived tree did not win the import;
re-run with `PYTHONPATH=/tmp/xcq-ae204537e` set as the ONLY entry and with the
working tree's editable install temporarily irrelevant by invoking from
`/tmp/xcq-ae204537e` as the working directory and passing the script by absolute
path. Do not proceed with a reference taken from the working tree: it would make
the oracle vacuous.

- [ ] **Step 5: Run against the working tree**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_closed_shell_byte_identity.py -v > /tmp/xcq-testlogs/task10_green.log 2>&1; echo "exit=$?"
```
Expected: PASS for every architecture. A mismatch on a descriptor-free
architecture points at a shared-path regression in `compute_vxc_nn` or
`fixed_density_total_energy`; a mismatch only on descriptor architectures points
at `assemble_descriptor_features` or `_reassemble_features` having changed the
total-density path, which this plan must not do.

- [ ] **Step 6: Compile**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/tests/record_closed_shell_reference.py xcquinox/alec/tests/test_closed_shell_byte_identity.py && echo compiled
```

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_closed_shell_byte_identity.py -v > /tmp/xcq-testlogs/task10_green.log 2>&1`

---

## Task 11: The PBE anchor is aligned, not silently reinterpreted

**Files:**
- Modify: `xcquinox/alec/losses.py:462-471` (`_anchor_term`)
- Modify: `xcquinox/alec/oneshot.py:1120-1174` (`_nn_fx_local_uks` docstring)
- Test: `xcquinox/alec/tests/test_pbe_anchor.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_anchor_term(model, sample, weight)` raises `ValueError` when `weight != 0.0` and `model.descriptors` is non-empty.

**Decision recorded here rather than left implicit.** The anchor is ALIGNED, not
retired. Retiring it would touch the config schema (`pbe_anchor_weight` /
`pbe_anchor_sample` at `config.py:775-776` and `:1060-1061`), the rendered
YAMLs, and every spec file that carries the field, for a term whose production
weight is 0.0 -- a blast radius unrelated to spin scaling. Aligning it costs one
guard: the anchor evaluates F_x at synthetic `(rho_alpha, rho_beta, s)` points
with zero descriptor extras, and a synthetic point has no density matrix, so the
symmetric doubled density `diag(P_sigma, P_sigma)` that now defines every
per-channel block is undefined there. The zero-extras slice is therefore not a
slice any physical system visits. The guard closes that path for descriptor
architectures and leaves it exact for the descriptor-free ones, whose F_x has no
extras at all.

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_pbe_anchor.py`:

```python
# ---------------------------------------------------------------------------
# The anchor and the per-channel feature blocks.
# ---------------------------------------------------------------------------

def _anchor_model(arch_name, seed=0):
    import dataclasses
    import xcquinox.alec as alec
    arch = dataclasses.replace(alec.get_architecture(arch_name),
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def test_anchor_term_refuses_a_descriptor_architecture_at_non_zero_weight():
    """A synthetic (rho_alpha, rho_beta, s) point has no density matrix, so the
    per-channel block of diag(P_sigma, P_sigma) is undefined there and the
    zero-extras slice is not one any physical system visits."""
    from xcquinox.alec.losses import _anchor_term
    from xcquinox.alec.pbe_anchor import build_pbe_anchor_sample
    sample = build_pbe_anchor_sample(n_points=8, seed=3)
    model = _anchor_model("deep_rung35_mgga_3x16")
    with pytest.raises(ValueError, match="pbe_anchor_weight"):
        _anchor_term(model, sample, 1e-3)


def test_anchor_term_is_inert_at_zero_weight_for_a_descriptor_architecture():
    """Production weight is 0.0, so the guard changes no production behavior."""
    from xcquinox.alec.losses import _anchor_term
    from xcquinox.alec.pbe_anchor import build_pbe_anchor_sample
    sample = build_pbe_anchor_sample(n_points=8, seed=3)
    model = _anchor_model("deep_rung35_mgga_3x16")
    assert float(_anchor_term(model, sample, 0.0)) == 0.0


def test_anchor_term_still_evaluates_for_a_descriptor_free_architecture():
    from xcquinox.alec.losses import _anchor_term
    from xcquinox.alec.pbe_anchor import build_pbe_anchor_sample
    sample = build_pbe_anchor_sample(n_points=8, seed=3)
    model = _anchor_model("deep_3x16")
    value = float(_anchor_term(model, sample, 1e-3))
    assert np.isfinite(value) and value >= 0.0
```

If `test_pbe_anchor.py` does not already import `pytest` and `numpy as np`, add
those imports at the top of the file.

- [ ] **Step 2: Run and confirm it fails**

```bash
python -m pytest xcquinox/alec/tests/test_pbe_anchor.py -k anchor_term -v > /tmp/xcq-testlogs/task11_red.log 2>&1; echo "exit=$?"
```
Expected: `Failed: DID NOT RAISE <class 'ValueError'>` on the first test.

- [ ] **Step 3: Add the guard**

Replace `xcquinox/alec/losses.py:462-471` with:

```python
def _anchor_term(model, sample, weight: float) -> jnp.ndarray:
    """PBE-anchor loss: weight * mean((F_x_nn - F_x_PBE)^2) on a fixed sample.

    The anchor probes the network's F_x SHAPE at synthetic
    (rho_alpha, rho_beta, s) points with ZERO descriptor extras
    (:func:`oneshot._nn_fx_local_uks`). A synthetic point carries no density
    matrix, so the symmetric doubled density diag(P_sigma, P_sigma) that defines
    every per-channel descriptor block (Oliver and Perdew, Phys. Rev. A 20, 397
    (1979)) is undefined there, and the zero-extras slice is not a slice any
    physical system visits. The anchor is therefore refused for a
    descriptor-carrying architecture instead of being left to pin an arbitrary
    feature slice; it stays exact for the descriptor-free architectures, whose
    F_x takes no extras. Production weight is 0.0, so the refusal is inert on
    every current configuration.
    """
    if sample is None or weight == 0.0:
        return jnp.array(0.0)
    if model.descriptors:
        raise ValueError(
            f"PBE anchor requested at pbe_anchor_weight={weight!r} for an "
            "architecture carrying descriptors "
            f"{tuple(type(d).registry_name for d in model.descriptors)!r}. The "
            "anchor evaluates F_x at synthetic (rho_alpha, rho_beta, s) points "
            "with zero descriptor extras; a synthetic point has no density "
            "matrix, so the per-channel feature block of diag(P_sigma, P_sigma) "
            "is undefined and the zero-extras slice is not one any physical "
            "system visits. Set pbe_anchor_weight=0.0 for descriptor "
            "architectures."
        )
    from xcquinox.alec.pbe_anchor import pbe_anchor_loss
    from xcquinox.alec.oneshot import _nn_fx_local_uks

    def _nn_fx(m, rho_alpha, rho_beta, s_vals):
        return _nn_fx_local_uks(m, rho_alpha, rho_beta, s_vals)
    return pbe_anchor_loss(model, sample, weight, _nn_fx)
```

- [ ] **Step 4: Align the helper's docstring**

Replace the final paragraph of `_nn_fx_local_uks` (lines 1144-1146 of
`xcquinox/alec/oneshot.py`) with:

```
    Uses zero extras (no descriptor features), and is FEATURE-FREE by
    construction. The exact spin scaling gives each channel the descriptor block
    of its own doubled density diag(P_sigma, P_sigma); a synthetic
    (rho_alpha, rho_beta, s) point has no density matrix, so no such block
    exists and the zero-extras row is not a row any physical system visits.
    ``losses._anchor_term`` therefore refuses a descriptor-carrying architecture
    at non-zero weight, and this helper is reached only for the descriptor-free
    ones, where the zero-extras row IS the network's whole input.
```

- [ ] **Step 5: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/losses.py xcquinox/alec/oneshot.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pbe_anchor.py xcquinox/alec/tests/test_losses.py -v > /tmp/xcq-testlogs/task11_green.log 2>&1; echo "exit=$?"
```
Expected: PASS. If a pre-existing loss test constructs a descriptor architecture
with a non-zero anchor weight, that test is asserting the defect the guard
closes: change its weight to 0.0 or its architecture to `deep_3x16` and say why
in a comment.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_pbe_anchor.py xcquinox/alec/tests/test_losses.py -v > /tmp/xcq-testlogs/task11_green.log 2>&1`

---

## Task 12: Open-shell pretraining row footing

**Files:**
- Modify: `xcquinox/alec/pretrain_data_gen.py:62-200` (`_atom_columns`), new module-level function inserted before it
- Test: `xcquinox/alec/tests/test_pretrain_data_gen.py` (append)

**Scope.** Section 3.2 of the spec owns the pretraining SET (which systems,
which weighting, the per-system energy term) and the `.npz` schema. This task
delivers only the ROW FOOTING for open-shell atoms and keeps it independent of
the data-set composition: the default `exchange_footing="total"` returns exactly
today's columns, and the new footing is exposed as an extra `x_rows` entry that
no current consumer reads.

**Relation to the DFS protocol (spec Section 6).** DFS forms the exchange target
as `F_x^sigma - 1 = e_x^ref(rho_sigma, 0) / e_x^LDA(rho_sigma, 0) - 1` with libxc
`spin=1` and the OTHER channel zeroed, with the channel's descriptors built from
`(2 rho_sigma, 4 gamma_sigma, 2 tau_sigma)`. That is the same number this task
computes. For exchange, `E_x[n_sigma, 0] = E_x[2 n_sigma] / 2` (Oliver and
Perdew, Phys. Rev. A 20, 397 (1979)), and libxc's `spin=1` per-electron output is
normalized by the total density `n_sigma`, so
`e_x^ref(rho_sigma, 0) = eps_x^unpolarized(2 rho_sigma, 4 sigma_sigma_sigma)`;
the LDA denominator scales identically, so the ratio is the unpolarized
enhancement factor at the doubled inputs. This task uses the `spin=0` call at the
doubled density instead of the zeroed-channel `spin=1` call because the doubled
form has no empty-channel edge, and Step 1 pins the two against each other. The
`rho_floor` cut is a parameter here (DFS uses `rho_tot > 1e-6`); which value the
pretraining set uses is Section 3.2's call.

**Interfaces:**
- Consumes: `descriptors.doubled_spin_dm` (Task 1).
- Produces:
  - `pretrain_data_gen.spin_channel_exchange_rows(mol, mf, ao, dm_ab, *, descriptors=True, cusp_log_transform=True, rho_floor=_RHO_FLOOR) -> dict[str, np.ndarray]` with keys `rho`, `sigma`, `Fx`, `Fx_scan`, `metagga`, `weights` and, when `descriptors` is True, `cusp`, `dm`, `rung35`, `rung35ms`.
  - `_atom_columns(..., exchange_footing: str = "total")`; with `"spin_channel"` the returned dict gains `x_rows` (the dict above, or `None` for a closed-shell atom).

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_pretrain_data_gen.py`:

```python
# ---------------------------------------------------------------------------
# Open-shell exchange row footing: the inputs the production UKS exchange
# actually evaluates, (2 rho_sigma, 4 sigma_sigma_sigma, features of
# diag(P_sigma, P_sigma)), with the parent's SPIN-UNPOLARIZED enhancement factor
# at those inputs as the target.
# ---------------------------------------------------------------------------

def _open_shell_scf(symbol="O", spin=2, basis="def2-svp", grid_level=1):
    from pyscf import dft, gto
    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, charge=0, spin=spin,
                verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = grid_level
    mf.kernel()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
    return mol, mf, ao, mf.make_rdm1()


def test_spin_channel_rows_reproduce_the_parent_open_shell_exchange_energy():
    """The rows are an exact quadrature of the parent's open-shell exchange:
    summing w_row * rho_row * eps_x^LDA(rho_row) * (1 + Fx_row) reproduces
    libxc's spin-polarized PBE exchange, because 1/2 (E_x[2 rho_a] +
    E_x[2 rho_b]) IS that energy (Oliver and Perdew, Phys. Rev. A 20, 397
    (1979))."""
    from pyscf import dft
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=False)
    c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    ex_lda = c_lda * np.cbrt(np.clip(rows["rho"], 1e-300, None))
    got = float(np.sum(rows["weights"] * rows["rho"] * ex_lda
                       * (1.0 + rows["Fx"])))
    rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
    rho_b_gga = mf._numint.eval_rho(mol, ao, dm_ab[1], xctype="GGA", hermi=True)
    eps = np.asarray(mf._numint.eval_xc(
        "PBE,", np.stack([rho_a_gga, rho_b_gga]), spin=1)[0])
    ref = float(np.sum(np.asarray(mf.grids.weights)
                       * (rho_a_gga[0] + rho_b_gga[0]) * eps))
    # The residual is the rho floor that drops the deep tail from the row set
    # plus the +-5 clip on the stored enhancement factor; both carry negligible
    # exchange mass at this basis and grid.
    assert abs(got - ref) < 1e-6, (got, ref)


def test_spin_channel_rows_match_the_dfs_zeroed_channel_recipe():
    """The DFS protocol (spec Section 6) targets e_x^ref(rho_sigma, 0) with libxc
    spin=1 and the other channel zeroed. For exchange E_x[n_sigma, 0] =
    E_x[2 n_sigma] / 2 (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)) and
    libxc's spin=1 per-electron output is normalized by the total density, so
    that recipe returns the unpolarized enhancement at the doubled inputs -- the
    number this row builder computes through the spin=0 call."""
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=False)
    rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
    zeroed = np.zeros_like(rho_a_gga)
    ex_ref = np.asarray(mf._numint.eval_xc(
        "PBE,", np.stack([rho_a_gga, zeroed]), spin=1)[0])
    ex_lda_ref = np.asarray(mf._numint.eval_xc(
        "LDA_X,", (rho_a_gga[0], zeroed[0]), spin=1)[0])
    safe = np.where(np.abs(ex_lda_ref) > 1e-12, ex_lda_ref, 1e-12)
    fx_dfs = np.clip(ex_ref / safe - 1.0, -5.0, 5.0)
    keep = 2.0 * rho_a_gga[0] > 1e-10
    n_a = int(keep.sum())
    np.testing.assert_allclose(rows["Fx"][:n_a], fx_dfs[keep],
                               rtol=0, atol=1e-9)


def test_spin_channel_rows_carry_the_doubled_ingredients():
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.alec.metagga import compute_alpha, compute_tau_from_dm
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=False)
    rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
    n_a = int(np.sum(2.0 * rho_a_gga[0] > 1e-10))
    # Rows are emitted alpha channel first, so the leading block is the alpha
    # channel's doubled density.
    np.testing.assert_allclose(rows["rho"][:n_a],
                               2.0 * rho_a_gga[0][2.0 * rho_a_gga[0] > 1e-10],
                               rtol=0, atol=1e-12)
    tau_doubled = np.asarray(compute_tau_from_dm(
        jnp.asarray(ao[1:4]), doubled_spin_dm(jnp.asarray(dm_ab), 0)))
    sigma_doubled = 4.0 * (rho_a_gga[1:4] ** 2).sum(axis=0)
    expect = np.asarray(compute_alpha(jnp.asarray(2.0 * rho_a_gga[0]),
                                      jnp.asarray(sigma_doubled),
                                      jnp.asarray(tau_doubled)))
    keep = 2.0 * rho_a_gga[0] > 1e-10
    np.testing.assert_allclose(rows["metagga"][:n_a, 0], expect[keep],
                               rtol=0, atol=1e-12)


def test_spin_channel_rows_rung35_block_is_the_channel_in_both_slots():
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=True)
    r = rows["rung35"]
    np.testing.assert_allclose(r[:, 0], r[:, 1], rtol=0, atol=1e-14)
    assert float(np.max(r)) < 1.0 + 1e-12
    ms = rows["rung35ms"]
    assert ms.shape[1] == 6
    for w in range(3):
        np.testing.assert_allclose(ms[:, 2 * w], ms[:, 2 * w + 1],
                                   rtol=0, atol=1e-14)


def test_atom_columns_default_footing_is_unchanged():
    from xcquinox.alec.pretrain_data_gen import _atom_columns
    cols = _atom_columns("O", 2, "def2-svp", 1, polarized=True,
                         descriptors=True)
    assert "x_rows" not in cols


def test_atom_columns_spin_channel_footing_only_adds_x_rows():
    from xcquinox.alec.pretrain_data_gen import _atom_columns
    base = _atom_columns("O", 2, "def2-svp", 1, polarized=True,
                         descriptors=True)
    extended = _atom_columns("O", 2, "def2-svp", 1, polarized=True,
                             descriptors=True,
                             exchange_footing="spin_channel")
    assert set(extended) - set(base) == {"x_rows"}
    for key in base:
        np.testing.assert_array_equal(np.asarray(base[key]),
                                      np.asarray(extended[key]))
    assert extended["x_rows"] is not None
    assert extended["x_rows"]["rho"].ndim == 1


def test_atom_columns_spin_channel_footing_is_none_for_a_closed_shell_atom():
    from xcquinox.alec.pretrain_data_gen import _atom_columns
    cols = _atom_columns("He", 0, "def2-svp", 1, polarized=True,
                         descriptors=True,
                         exchange_footing="spin_channel")
    assert cols["x_rows"] is None


def test_atom_columns_rejects_an_unknown_footing():
    from xcquinox.alec.pretrain_data_gen import _atom_columns
    with pytest.raises(ValueError, match="exchange_footing"):
        _atom_columns("He", 0, "def2-svp", 1, polarized=True,
                      descriptors=True, exchange_footing="per_orbital")
```

If `test_pretrain_data_gen.py` does not already import `numpy as np`,
`jax.numpy as jnp` and `pytest`, add those imports at the top.

- [ ] **Step 2: Run and confirm it fails**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_data_gen.py -k "spin_channel or footing" -v > /tmp/xcq-testlogs/task12_red.log 2>&1; echo "exit=$?"
```
Expected: `ImportError: cannot import name 'spin_channel_exchange_rows' from 'xcquinox.alec.pretrain_data_gen'` and `TypeError: _atom_columns() got an unexpected keyword argument 'exchange_footing'`.

- [ ] **Step 3: Add the row builder**

Insert into `xcquinox/alec/pretrain_data_gen.py` immediately before
`_atom_columns` (before line 62):

```python
def spin_channel_exchange_rows(mol, mf, ao, dm_ab, *, descriptors=True,
                               cusp_log_transform=True, rho_floor=_RHO_FLOOR):
    """Open-shell exchange rows on the exact-spin-scaling footing.

    The production UKS exchange evaluates, per spin channel, the symmetric
    doubled density diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20,
    397 (1979)): density ``2 rho_sigma``, gradient invariant
    ``4 sigma_sigma_sigma``, kinetic-energy density ``2 tau_sigma``, and the
    descriptor features of that density matrix. Those are the inputs the network
    sees at SCF time on an open shell, so those are the inputs its exchange rows
    must be posed at, with the parent's SPIN-UNPOLARIZED enhancement factor at
    the same inputs as the target -- ``eval_xc(..., spin=0)`` on the doubled
    density, not the spin-polarized call on the physical one.

    Each row carries HALF the grid weight, because
    ``E_x = 1/2 (E_x[2 rho_a] + E_x[2 rho_b])``: summing
    ``w_row rho_row eps_x^LDA(rho_row) (1 + Fx_row)`` over both channels then
    reproduces the parent's open-shell exchange energy exactly.

    Returns 1-D columns (2-D for the descriptor blocks), alpha channel first
    then beta, with points below ``rho_floor`` in the DOUBLED density dropped. A
    channel with no electron (the beta channel of H) contributes no rows.
    Correlation is untouched: it is spin-interpolated rather than spin-scaled
    and keeps the total density with zeta.

    Parameters
    ----------
    mol, mf, ao : the converged parent calculation and its ``deriv=1`` AO
        values on ``mf.grids.coords``.
    dm_ab : array, shape (2, nao, nao). The parent's spin-resolved density
        matrix.
    """
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.alec.metagga import compute_alpha, compute_tau_from_dm
    from xcquinox.alec.rung35 import (
        DEFAULT_RUNG35_ALPHA, DEFAULT_RUNG35_MULTISHELL_ALPHAS,
        compute_projected_ao, compute_projected_ao_multishell,
        compute_rung35_multishell_occupancy, compute_rung35_occupancy)

    dm_j = jnp.asarray(dm_ab)
    ao_grad = jnp.asarray(ao[1:4])
    s_matrix = jnp.asarray(mol.intor("int1e_ovlp"))
    weights = np.asarray(mf.grids.weights)
    c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)

    names = ["rho", "sigma", "Fx", "Fx_scan", "metagga", "weights"]
    if descriptors:
        names += ["cusp", "dm", "rung35", "rung35ms"]
    parts = {k: [] for k in names}

    for s in (0, 1):
        dm_doubled = doubled_spin_dm(dm_j, s)
        rho_gga_s = mf._numint.eval_rho(mol, ao, np.asarray(dm_ab[s]),
                                        xctype="GGA", hermi=True)
        rho_d = 2.0 * rho_gga_s[0]
        grad_d = 2.0 * rho_gga_s[1:4]
        sigma_d = (grad_d ** 2).sum(axis=0)
        tau_d = np.asarray(compute_tau_from_dm(ao_grad, dm_doubled))
        keep = rho_d > rho_floor
        if not bool(keep.any()):
            continue
        gga_row = np.vstack([rho_d, grad_d])
        mgga_row = np.vstack([gga_row, np.zeros_like(rho_d), tau_d])
        # The parent's SPIN-UNPOLARIZED enhancement at the doubled inputs: this
        # is exactly what the spin-scaling relation asks the functional for.
        ex_pbe = mf._numint.eval_xc("PBE,", gga_row, spin=0)[0]
        ex_scan = mf._numint.eval_xc("SCAN,", mgga_row, spin=0)[0]
        ex_lda = c_lda * np.cbrt(np.clip(rho_d, 1e-300, None))
        ex_safe = np.where(np.abs(ex_lda) > 1e-12, ex_lda, 1e-12)
        parts["rho"].append(rho_d[keep])
        parts["sigma"].append(sigma_d[keep])
        parts["Fx"].append(np.clip(ex_pbe / ex_safe - 1.0, -5.0, 5.0)[keep])
        parts["Fx_scan"].append(
            np.clip(ex_scan / ex_safe - 1.0, -5.0, 5.0)[keep])
        parts["metagga"].append(np.asarray(compute_alpha(
            jnp.asarray(rho_d), jnp.asarray(sigma_d),
            jnp.asarray(tau_d)))[keep].reshape(-1, 1))
        # Half the grid weight per channel: E_x = 1/2 (E_x[2 rho_a] + E_x[2 rho_b]).
        parts["weights"].append(0.5 * weights[keep])
        if descriptors:
            coords_v = mf.grids.coords[keep]
            parts["cusp"].append(np.asarray(_features.compute_cusp_descriptor(
                jnp.asarray(coords_v),
                jnp.asarray(mol.atom_coords()),
                jnp.asarray(mol.atom_charges()),
                log_transform=cusp_log_transform,
            )))
            dm_global = _features.compute_dm_features_array(dm_doubled, s_matrix)
            parts["dm"].append(np.tile(np.asarray(dm_global),
                                       (int(keep.sum()), 1)))
            proj = compute_projected_ao(mol, coords_v, DEFAULT_RUNG35_ALPHA)
            parts["rung35"].append(np.asarray(compute_rung35_occupancy(
                jnp.asarray(proj), dm_doubled)))
            proj_ms = compute_projected_ao_multishell(
                mol, coords_v, DEFAULT_RUNG35_MULTISHELL_ALPHAS)
            parts["rung35ms"].append(np.asarray(
                compute_rung35_multishell_occupancy(jnp.asarray(proj_ms),
                                                    dm_doubled)))

    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}
```

- [ ] **Step 4: Add the footing switch to `_atom_columns`**

Change the signature (lines 62-63) to:

```python
def _atom_columns(symbol, spin, basis, grid_level, *, polarized, descriptors,
                  density_fit=False, auxbasis=None, cusp_log_transform=True,
                  exchange_footing="total"):
```

Append to the docstring:

```
    ``exchange_footing`` selects how OPEN-SHELL exchange rows are posed.
    ``"total"`` (default) is unchanged: one row per grid point at the total
    density with spin-resolved libxc targets. ``"spin_channel"`` additionally
    returns ``x_rows``, the per-channel rows of
    :func:`spin_channel_exchange_rows` -- ``(2 rho_sigma, 4 sigma_sigma_sigma,
    features of diag(P_sigma, P_sigma))`` with the parent's spin-unpolarized
    enhancement factor at those inputs as the target, which is what the exact
    spin scaling evaluates at SCF time. ``x_rows`` is ``None`` for a
    closed-shell atom, whose total-density rows already are that footing.
    Correlation rows are untouched under either setting: correlation is
    spin-interpolated rather than spin-scaled and keeps the total density with
    zeta. The composition of the pretraining SET is not decided here.
```

Add validation at the top of the body, immediately after the docstring:

```python
    if exchange_footing not in ("total", "spin_channel"):
        raise ValueError(
            "exchange_footing must be 'total' or 'spin_channel'; got "
            f"{exchange_footing!r}."
        )
```

Add before the `return cols` at line 200:

```python
    if exchange_footing == "spin_channel":
        cols["x_rows"] = (
            spin_channel_exchange_rows(
                mol, mf, ao, dm_ab, descriptors=bool(descriptors),
                cusp_log_transform=cusp_log_transform)
            if is_uks else None
        )
```

- [ ] **Step 5: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/pretrain_data_gen.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_data_gen.py xcquinox/alec/tests/test_pretrain_data_basis.py xcquinox/alec/tests/test_metagga_pretrain.py -v > /tmp/xcq-testlogs/task12_green.log 2>&1; echo "exit=$?"
```
Expected: PASS, with every pre-existing test in those files unchanged (the
default footing is byte-identical).

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_pretrain_data_gen.py xcquinox/alec/tests/test_pretrain_data_basis.py xcquinox/alec/tests/test_metagga_pretrain.py -v > /tmp/xcq-testlogs/task12_green.log 2>&1`

---

## Task 13: Full-suite run and the HISTORY entry

**Files:**
- Modify: `xcquinox/alec/HISTORY.md` (prepend a dated entry in the file's existing format)
- Modify: `xcquinox/alec/SPEC_pretrain_fidelity_program.md:158` (mark step 1 of Section 5 done)

**Interfaces:**
- Consumes: the measured numbers from every task's green log.
- Produces: the development record.

- [ ] **Step 1: Run the whole alec suite**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests -v > /tmp/xcq-testlogs/task13_full.log 2>&1; echo "exit=$?"
```
Read the log with `Read`. Expected: no test fails and no test errors. Any failure
here is a call site of a changed signature that no earlier task's log covered;
fix it and re-run before writing the entry.

- [ ] **Step 2: Run the slow oracles once**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests -m slow -v > /tmp/xcq-testlogs/task13_slow.log 2>&1; echo "exit=$?"
```
Expected: PASS.

- [ ] **Step 3: Write the HISTORY entry**

Prepend to `xcquinox/alec/HISTORY.md`, in the file's existing entry format
(date, short hash placeholder left for the controller to fill at commit time is
NOT acceptable -- write the entry without a hash and let the controller add it,
matching how the file already handles pre-commit entries):

```markdown
### 2026-08-21 -- Exact spin scaling for every density-matrix feature

**What changed.** Every UKS exchange evaluation now receives the descriptor
feature block of the symmetric doubled density diag(P_sigma, P_sigma) for its
own spin channel, in place of one molecular block shared by both channels.
`descriptors.doubled_spin_dm` is the single primitive; the per-channel blocks
reach the code through two paths, `assemble_descriptor_features(...,
spin_channel=0|1)` on precomputed data and `solver.make_uks_feature_fns` on a
live density matrix. `split_exc_energy_uks`, `_uks_spin_resolved_vxc` and
`solver_manual._compute_total_energy_uks` take three blocks (alpha, beta,
total); the pyscfad `eval_xc` callback slices three per grid block. The
feature-response term became three contractions -- f_a, f_b and f_tot are three
different maps of P -- instead of one accumulated de/df. `data.py` precomputes
`{dm,rung35,rung35ms,metagga}_features_{a,b}` and `tau_spin_{a,b}` for open
shells; `padding.py` pads them. `losses._anchor_term` refuses a
descriptor-carrying architecture at non-zero PBE-anchor weight.
`pretrain_data_gen.spin_channel_exchange_rows` poses open-shell exchange rows at
(2 rho_sigma, 4 sigma_sigma_sigma, features of diag(P_sigma, P_sigma)) with the
parent's spin-unpolarized enhancement factor as the target.

**Why.** The exchange spin-scaling relation E_x[n_a, n_b] = (E_x[2 n_a] +
E_x[2 n_b])/2 (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)) is an identity
only when each channel is evaluated on the fictitious spin-unpolarized system it
refers to. The library doubled rho and quadrupled sigma per channel but passed
the PHYSICAL density's descriptor features into both terms, which made the
open-shell functional something other than the one the relation defines -- worst
on the meta-GGA architectures, whose iso-orbital indicator was frozen at the
total density (deep_mgga_3x16 sat 20.8 to 55.9 kcal/mol from SCAN on
atomization energies; correcting alpha alone moved it to 7.6 to 7.9). The
doubled system is well defined for every density-matrix descriptor: alpha
becomes alpha(2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma), the rung-3.5
occupancies become the channel's occupancy in both spin slots (still inside the
Bessel bound), the DM statistics become those of diag(P_sigma, P_sigma), and the
nuclear-cusp feature is geometry-only and unchanged. Correlation is
spin-interpolated rather than spin-scaled (von Barth and Hedin, J. Phys. C 5,
1629 (1972); Perdew and Wang, Phys. Rev. B 45, 13244 (1992)) and keeps the total
density with the total block.

**Verification.** Four oracles, all executable:
O1, the parent functional wearing the model's evaluation surface -- with libxc's
own PBE or SCAN enhancement in place of the network the library's UKS energy
reproduces libxc spin=1 on H, Li, N and O to <MEASURED> Ha, and the per-channel
ingredients equal libxc's spin-polarized ones; feeding the total block into both
channels moves the SCAN oracle by <MEASURED> Ha on N.
O2, the assembled Fock matrices are the central difference of the assembled
energy for every architecture in ARCHITECTURES on H, Li, N and O at def2-svp
(worst relative residual <MEASURED> on <ARCH>/<SPECIES>) and at the production
identity 6-311++G(3df,2pd) grid level 3.
O3, closed-shell results are byte identical to the tree at ae204537e for every
architecture; rho_a = rho_b makes the three blocks the same array, so this is
structural rather than numerical.
O4, the H atom -- alpha vanishes at every grid point, the rung-3.5 block is the
doubled single orbital's occupancy in both slots rather than the physical
[n_a, 0], and the exchange energy is exactly half the spin-unpolarized
evaluation on diag(P_a, P_a).
```

Replace every `<MEASURED>`, `<ARCH>` and `<SPECIES>` with values measured on this
machine. The three slots and how to obtain each:

1. The O1 residual against libxc. Temporarily change the bound in
   `test_o1_exchange_path_equals_libxc_pbe_spin1` and
   `test_o1_scan_exchange_path_equals_libxc_through_the_alpha_column` from
   `1e-10` to `0.0`, run

   ```bash
   python -m pytest xcquinox/alec/tests/test_spin_scaling_oracles.py -k o1 -v > /tmp/xcq-testlogs/task13_o1_residuals.log 2>&1; echo "exit=$?"
   ```

   read every failure message (each carries the `(symbol, got, ref)` tuple),
   take the largest `abs(got - ref)` across the four atoms, then restore both
   bounds to `1e-10` and confirm the module is green again.
2. The SCAN oracle's movement under the superseded contract: the
   `(exact, approx)` tuple asserted by
   `test_o1_total_block_would_break_the_scan_oracle`, obtained the same way by
   temporarily raising its `1e-3` bound to `1e30`.
3. The worst O2 relative residual, architecture and species: from
   `/tmp/xcq-testlogs/task09_residuals.log` produced in Task 9 Step 4.

Do not invent a number and do not copy one from the plan. An entry containing an
unresolved placeholder is a failed task.

- [ ] **Step 4: Mark the spec step**

In `xcquinox/alec/SPEC_pretrain_fidelity_program.md`, change line 158 from

```
1. Spin-scaling fix (3.1) with oracles; commit; two reviews.
```

to

```
1. Spin-scaling fix (3.1) with oracles: DONE (oracles O1-O4 executable in
   `xcquinox/alec/tests/test_spin_scaling_oracles.py`,
   `test_solv01_split_xc.py` and `test_closed_shell_byte_identity.py`);
   commit; two reviews.
```

- [ ] **Step 5: Final compile sweep**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/descriptors.py xcquinox/alec/data.py xcquinox/alec/padding.py xcquinox/alec/solver.py xcquinox/alec/solver_manual.py xcquinox/alec/solver_pyscfad.py xcquinox/alec/oneshot.py xcquinox/alec/losses.py xcquinox/alec/pretrain_data_gen.py xcquinox/alec/tests/parent_adapter.py xcquinox/alec/tests/record_closed_shell_reference.py && echo compiled
```

**Covering test command:** `python -m pytest xcquinox/alec/tests -v > /tmp/xcq-testlogs/task13_full.log 2>&1`

---

## Self-review

Run by the plan author against the spec before handover; recorded so the
executor can see what was already checked and what was decided rather than
inherited.

### Spec coverage

| Spec requirement (Section 3.1 and its oracles) | Task |
|---|---|
| `assemble_descriptor_features` gains a per-channel form; each Descriptor exposes a per-channel accessor for diag(P_sigma, P_sigma) | 1 |
| MetaGGAAlphaDescriptor -> alpha(2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma), new per-spin tau from `ao_grad` and `dm[sigma]` | 1 (`doubled_spin_dm` + the tau contract test), 2 (`metagga_features_{a,b}`, `tau_spin_{a,b}`), 3 (live path) |
| DMRung35 / Multishell -> the channel occupancy in both spin slots, alpha-major-then-spin preserved | 1, 2 (`rung35_features_{a,b}`, `rung35ms_features_{a,b}`) |
| DMStatistics -> statistics of diag(P_sigma, P_sigma) | 1, 2 (`dm_features_{a,b}`) |
| Cusp -> unchanged | 1 (`spin_mol_keys == ()` and its test) |
| Energy: `split_exc_energy_uks(..., features_a, features_b, features_tot)`; exchange at (2 rho_sigma, 4 sigma_sigma_sigma, features_sigma); correlation on the total density with the total features | 4 |
| `fixed_density_total_energy` builds the three blocks | 4 |
| `solver_manual._compute_total_energy_uks` builds the three blocks | 5 |
| Potential: `_uks_spin_resolved_vxc` per channel | 4 |
| Manual-solver loop: `_features_for`, `_vx_nn_spin`, `_feature_response_uks` per channel | 5 |
| The feature response differentiates each channel's P -> f_sigma(P) map | 3 (the three closures), 4 (FD probe), 5 (three contractions in the SCF) |
| `solver_pyscfad`'s UKS branch follows | 6 |
| `_reassemble_features` follows | 3 |
| Closed shells unchanged byte for byte, pinned against the archived tree | 10 (O3); structural proof pinned in 3 and 5 |
| The PBE anchor is retired or aligned | 11 (aligned, with the reason) |
| O1 parent reproduction of the code path, 1e-10 Ha on open-shell atoms; per-channel inputs equal libxc's spin-polarized ingredients | 7 |
| O2 central-difference Fock check, every descriptor active, re-pointed at the new energy, extended from Li/def2-svp to the production basis | 9 |
| O3 closed-shell byte identity for every architecture | 10 |
| O4 H atom: alpha identically 0, rung-3.5 occupancy of the doubled single orbital, exchange equals the spin-scaled unpolarized evaluation | 8 |
| `mol_data` carries per-spin tau, per-spin alpha, per-channel rung-3.5 and DM-statistics blocks | 2 |
| One helper shared by the loss, eval and solver paths | 1 (`assemble_descriptor_features(spin_channel=)` for precomputed data) and 3 (`make_uks_feature_fns` for a live density matrix) |
| Open-shell pretraining ROW FOOTING at (2 rho_sigma, 4 sigma_sigma_sigma, features of diag(P_sigma, P_sigma)) with the parent's spin-unpolarized F_x as target; correlation rows unchanged; independent of the data-set composition | 12 |

Not covered here, by design, and owned by other plans: the pretraining SET and
the per-system energy term (Section 3.2), the fidelity certificate and its
enforcement (Section 3.3), the workflow matrix (Section 3.4), campaign v6
(Section 3.5).

### Ambiguities in the spec, and how they were resolved

1. **`compute_for_spin_channel(mol_data_or_dm, sigma)` takes two kinds of
   argument.** Section 3.1 writes one accessor that might receive either
   precomputed molecule data or a live density matrix. Those are genuinely
   different code paths with different failure modes -- the first is a lookup,
   the second is a contraction that must be differentiable -- so they are split:
   `Descriptor.compute_for_spin_channel(mol_data, spin_channel)` is the lookup
   (Task 1), and `solver.make_uks_feature_fns` is the live path (Task 3). Both
   are implemented on top of the same primitive, `doubled_spin_dm`, and Task 3
   pins that they agree at the precompute density matrix, so there is one
   convention with two entry points rather than two conventions.
2. **"the feature response differentiates each channel's P -> f_sigma(P) map"
   changes the shape of the derivative.** With one shared block the code
   accumulated `de/df` across the three energy terms and contracted once. With
   three blocks that is no longer valid: `f_a`, `f_b` and `f_tot` are three
   different maps, so the plan replaces the single contraction with three
   (Tasks 4, 5, 9). This is the one place the fix is not a pure substitution,
   and the O2 oracle is what catches it if it is done wrong.
3. **"retired or aligned" for the PBE anchor.** Resolved as ALIGNED with a
   refusal (Task 11): retiring reaches the config schema, the rendered YAMLs and
   every spec file for a term whose production weight is 0.0, which is a blast
   radius unrelated to spin scaling. The refusal closes the ill-defined path
   (a synthetic point has no density matrix, so `diag(P_sigma, P_sigma)` does not
   exist there) without touching any production configuration.
4. **O1's "1e-10 Ha" against a path that clips zeta.** `oneshot.uks_zeta` holds
   `|zeta|` inside `1 - 1e-6` so the PW92 spin interpolation stays twice
   differentiable, which puts a floor of order `1e-6 dE_c/dzeta` under any
   comparison of the POLARIZED correlation path with libxc. Resolved by
   oracling the two pieces separately (Task 7): exchange to 1e-10 Ha, correlation
   on the total density to 1e-10 Ha with the polarization flag off, and the
   polarized correlation path to 1e-6 Ha with the clip named as the floor. The
   spin-scaling change lives entirely in the exchange path, which is held to the
   spec's 1e-10.
5. **O1's SCAN adapter and the descriptor's alpha clip.** `compute_alpha` clips
   to `[0, 100]`, so a libxc SCAN reference built from the raw kinetic-energy
   density would disagree with the library wherever the clip is active. Resolved
   by inverting the assembled alpha back to a kinetic-energy density for the
   reference (`parent_adapter.tau_from_alpha`), which asks the question the
   oracle is for -- is the assembly the parent's own -- rather than re-testing
   the clip.
6. **O2's "H, Li, N (and O if cheap)" and the fully polarized H atom.** O is
   included (def2-svp, grid level 2 costs the same order as Li). H's beta
   density matrix is identically zero and the SCF never populates it, so
   perturbing that block drives the beta density through zero and straddles
   every low-density guard at essentially every grid point, which would trip the
   probe's own 10% guard-straddle assertion. Resolved by leaving an
   unoccupied channel unperturbed (`_uks_fd_perturbation`), which is also the
   physically correct domain: a channel with no occupied orbital has no Fock
   block in the functional's domain.
7. **O3's "closed-shell UKS molecule" when precompute never builds one.**
   `precompute_fixed_density_data` routes `spin == 0` to RKS, so no molecule
   arrives with `is_unrestricted` True and `rho_a == rho_b`. Resolved the way the
   existing closed-shell reduction test does: feed `D_a = D_b = D/2` through the
   UKS helpers by hand (Task 10). That is the configuration the byte-identity
   claim is about, and it is the one the reduction is proved on.
8. **"no git commands by implementers" versus O3's `git archive`.** The
   constraint is stated verbatim in the Global Constraints, with one named
   read-only exception: the `git archive ae204537e | tar -x -C /tmp/xcq-ae204537e`
   export in Task 10, which is required by the oracle the spec asks for and
   writes only into the scratch directory. No other git invocation appears in
   the plan.
9. **Where the pretraining footing task sits.** The instruction to keep it last
   is honored for implementation: Task 12 is the last code task. Task 13 is
   bookkeeping (full-suite run, HISTORY entry, spec sequence marker) and carries
   no implementation.
