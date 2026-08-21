# Pretraining That Delivers the Parent -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four-atom-plus-mesh pretraining set with the set Section 7 of the spec binds -- the DFS pretraining set in its entirety, every single-atom species of the BH76 and W4-11 pools, and the synthetic mesh as a regularizer -- generated at the production identity on the PARENT functional's own self-consistent density, posed on the exact-spin-scaling exchange footing for every open shell, fit with a per-system energy term in Hartree beside the point-wise enhancement-factor term, stopped on a held-out-system validation criterion, and recorded in the pretrain metadata the Section 3.3 certificate and HISTORY read. Every new knob defaults to today's value, so an old YAML reproduces today's data and today's loss.

**Architecture:** One composition layer and one column builder carry the whole change. `resolve_pretrain_systems` turns the config flags into an ordered, de-duplicated tuple of `PretrainSystem(name, atom, charge, spin)`; `_system_columns` runs the parent SCF for ONE such system and returns the same column dict `_atom_columns` always returned, plus the two LDA energy-density columns that turn a row of enhancement factors back into Hartrees. `_atom_columns` becomes the single-atom wrapper of `_system_columns`, so the atomic and molecular rows are the same quantity by construction rather than by inspection. The `.npz` grows a per-row `system_all` segment index, a per-system energy table, and -- when the footing is `spin_channel` -- a SECOND row block `*_x` holding the exchange rows, because per-channel exchange rows and total-density correlation rows are no longer the same rows. `_PretrainLoss` gains a `jax.ops.segment_sum` over that index: `w_E * mean_s (sum_{i in s} w_i e_LDA_i F^NN_i - E_parent_s)^2`, exactly zero when the network reproduces the stored targets. Validation holds out MOLECULES (never an atom -- every pool atom is a system the certificate bounds at `tol_atom`), and the validated path runs its own full-batch loop so the optimizer state and the learning-rate schedule survive across validations, which `xcTrainer`'s API cannot express; the unvalidated path still goes through `xcTrainer` unchanged, which is what makes the default byte-identical.

**Tech Stack:** Python 3, JAX 0.7 (`jax_enable_x64`, CPU), equinox 0.13, optax 0.2.6, PySCF + libxc, numpy 2.3, pytest.

**Spec:** `xcquinox/alec/SPEC_pretrain_fidelity_program.md` (this plan implements Section 3.2, the four adopted deviations of Section 6, and the pretraining-set / energy-term / validation / acceptance decisions of Section 7. Section 3.1 is the sibling plan `2026-08-21-spin-scaling-exact-features.md`; Section 3.3 is the sibling plan that owns `dfs_pretrain_set.py` and `cluster/fidelity.py`.)

## Global Constraints

Every task's requirements implicitly include this section.

- Certificate tolerances, copied verbatim from Section 7 of the spec: "tol_AE = 1.0 kcal/mol on atomization energies and tol_atom = 1.0 mHa on atomic E_xc, for every architecture; no override without `fidelity.override_reason`." This plan does not build the certificate (Section 3.3 does); it must not make a choice that puts a floor above those numbers, and Task 3 measures the floor its row set imposes.
- Pretraining set, copied verbatim from Section 7 of the spec: "the DFS pretraining set in its entirety, plus every atom of the BH76 / W4-11 pools (open shells per spin channel), plus the synthetic mesh as a regularizer, with a per-system energy term in the loss beside the point-wise enhancement-factor term."
- Objective, copied verbatim from Section 6 deviation 3: "the point-wise residual is integration-weighted (as today) AND a per-system energy term E_xc^NN - E_xc^parent in Hartree is added, so the H atom and every molecule carry an energy of their own; validation on held-out systems and a stop criterion replace the hand interruption."
- Footing, copied verbatim from Section 3.2 of the spec: "open-shell rows are posed per spin channel at (2 rho_sigma, 4 sigma_sigma, features of diag(P_sigma, P_sigma)) with the parent's spin-unpolarized enhancement factors at those inputs as targets (this is what the exact spin scaling evaluates). Correlation rows keep the total density with zeta (polarized cnet)."
- Every new knob's DEFAULT reproduces today's behavior. A YAML written before this change must produce a `.npz` whose every pre-existing array is bit-identical and a loss whose value is bit-identical. Task 4 pins the data half against a recorded fixture; Task 6 pins the loss half structurally.
- Comments and docstrings are ASCII only, in scientific voice. They state physics, measurements and rationale. They never mention the process by which the code was produced, never mention an assistant or a model, never say "we", "I", "now", "previously", "as requested", "TODO" or "FIXME". Reference literature the way the surrounding code does (author, journal, volume, page, year).
- Run `python -m py_compile <file>` on every Python file immediately after editing it. A task is not finished while any edited file fails to compile.
- Every test run redirects to a log file and the log is read with `Read`. Never pipe a test run through `tail`, `head`, `less`, `grep -m`, or any other truncating filter: the log must be complete. Create the log directory once with `mkdir -p /tmp/xcq-testlogs`.
- Implementers run no git commands: no `git add`, `git commit`, `git push`, `git checkout`, `git branch`, `git stash`, `git rebase`. Committing is the controller's job. This plan contains no sanctioned exception; every reference artifact it needs is produced by a recorder script run against the installed tree.
- Every new config field must appear in BOTH `cluster/grid_config.py::_build_pretrain` (the strict allow-list on the way IN -- a field missing there silently reverts to its default on every stage that re-reads `resolved_config.yaml`) and in `cluster/__main__.py::_config_to_raw_dict` on the way OUT. `_config_to_raw_dict` serializes the whole `pretrain` section with `dataclasses.asdict`, so a new `PretrainConfig` field round-trips automatically THERE; the failure mode is `_build_pretrain`. Every new field is also set explicitly in `cluster/examples/grid_step7.yaml` and asserted in `xcquinox/alec/tests/test_cluster_examples.py`.
- `xcquinox/alec/HISTORY.md` gets an entry for this change (Task 11). It is the canonical development record for the paper.
- Every number quoted in a comment or a docstring must have been measured by the implementer on this machine. Do not copy a number from this plan into a comment without re-measuring it; the plan's tolerances are bounds, not measurements.

## Dependencies: what this plan consumes and does not build

This plan is applied ON TOP of the Section 3.1 plan
(`docs/superpowers/plans/2026-08-21-spin-scaling-exact-features.md`, whose Task
12 has landed) and alongside the Section 3.3 plan
(`docs/superpowers/plans/2026-08-21-fidelity-certificate-gate.md`). It CONSUMES
and must not re-implement:

From Section 3.1:
- `descriptors.doubled_spin_dm(dm, spin_channel)`; `assemble_descriptor_features(descriptors, mol_data, spin_channel=None)`; the per-spin `mol_data` keys `dm_features_{a,b}`, `rung35_features_{a,b}`, `rung35ms_features_{a,b}`, `metagga_features_{a,b}`, `tau_spin_{a,b}`.
- `pretrain_data_gen.spin_channel_exchange_rows(mol, mf, ao, dm_ab, *, descriptors=True, cusp_log_transform=True, rho_floor=_RHO_FLOOR) -> dict` with keys `rho`, `sigma`, `Fx`, `Fx_scan`, `metagga`, `weights` and, with descriptors on, `cusp`, `dm`, `rung35`, `rung35ms`. Rows are alpha channel first then beta, each carrying HALF the grid weight, points below the floor in the DOUBLED density dropped, a channel with no electron contributing no rows. It reads `mf` only for `mf._numint` and `mf.grids`, which is why Task 2 can hand it a grid-only mean field that never ran an SCF.
- `pretrain_data_gen._atom_columns(..., exchange_footing="total"|"spin_channel")` returning `x_rows`.

From Section 3.3 (`xcquinox/alec/dfs_pretrain_set.py` is already in the tree; read it for the exact signatures):
- `dfs_pretrain_set.LEVELS == ("gga", "mgga")` and `dfs_pretrain_set.MGGA_EXCLUDED == ("H2", "N2")`.
- `dfs_pretrain_set.dfs_pretrain_records(level="gga") -> list[dict]`: atoms first then molecules, each `{"kind", "name", "atom", "charge", "spin", "atom_composition", "g2_97_index"}` with `atom` a PySCF geometry string in Angstrom, returned as fresh copies. This is what the composition layer reads, because it is basis-free and the basis and grid level are the run's identity, applied later.
- `dfs_pretrain_set.dfs_pretrain_systems(level, *, basis, grid_level) -> list[MoleculeSpec]`: the same set already carrying an identity. NOT used here -- the composition layer must stay basis-free so one resolved set can be compared against a manifest written at any identity -- but named so the two entry points are not confused.
- `data.precompute_fixed_density_data(mol_spec, ..., reference_xc="pbe"|"scan")`: the ONE place a frozen parent density is produced in this library. `"pbe"` is byte-identical to today; `"scan"` returns the same `MoleculeData` with every density-carrying entry rebuilt from SCAN's own self-consistent density on the SAME grid, AO tables, overlap and constant projectors. This plan's generator obtains every parent density through that keyword and runs no SCF of its own.
- `cluster.fidelity.fidelity_certificate(cfg, run_dir, arch_name)`. The CALL from the pretrain stage is the Section 3.3 plan's task; this plan only writes the metadata that call and HISTORY consume (Tasks 6, 7, 11).

Ordering: Task 2 cannot go green until `reference_xc` exists on
`precompute_fixed_density_data` (Section 3.3 plan, Task 3). Tasks 1 and 3-11 do
not depend on it. If Task 2 is reached before that keyword lands, implement it
against the keyword and leave its `reference_xc="scan"` tests failing with
`TypeError`, recording the blocker in the task log. Do NOT work around it with a
local SCF: the duplicate SCF is exactly what this route removes.

## File Structure

| File | Responsibility after this plan |
|---|---|
| `xcquinox/alec/pretrain_data_gen.py` | The composition layer (`PretrainSystem`, `normalize_system`, `pool_atom_systems`, `resolve_pretrain_systems`, `resolve_parent_density`, `dfs_level_for_reference_xc`); the per-system column builder `_system_columns`, which reads its parent density from `precompute_fixed_density_data(..., reference_xc=)`, with `_atom_columns` / `_molecule_columns` as its two wrappers; the per-system energy targets; the two-block `.npz` schema; the manifest identity. |
| `xcquinox/alec/pretrain.py` | `_assemble_pretrain_descriptors(..., suffix=)` reading either row block; the per-system energy term in `_PretrainLoss`; the validated training loop with early stopping and best-weight retention; the metadata the certificate and HISTORY read. |
| `xcquinox/alec/config.py` | `PretrainSpec` carries `parent_density`, `energy_term_weight`, `validation_fraction`, `validation_seed`, `validate_every`, `patience`. |
| `xcquinox/alec/cluster/grid_config.py` | `PretrainConfig` carries `dfs_set`, `pool_atoms`, `parent_density`, `exchange_footing`, `mesh_fraction`, `energy_term_weight`, `validation_fraction`, `validation_seed`, `validate_every`, `patience`; `_build_pretrain` parses them; `validate_grid_semantics` bounds them. |
| `xcquinox/alec/cluster/_datagen.py` | Routes JAX to fp64 before anything imports it, resolves the distinct `(polarized, reference_xc)` data files the sweep's architectures need, and generates each; logs the resolved system count and the wall estimate. |
| `xcquinox/alec/cluster/inputs.py` | The preflight's belt-and-braces `ensure_pretrain_data` call threads the same knobs. |
| `xcquinox/alec/cluster/_pretrain.py` | Threads the new `PretrainConfig` fields into `PretrainSpec`. |
| `xcquinox/alec/cluster/examples/grid_step7.yaml` | Sets every new field at its default, with the v6 values named in comments. |
| `xcquinox/alec/tests/record_pretrain_data_reference.py` | Recorder for the default-output regression fixture. Not a test module (no `test_` prefix), so pytest does not collect it. |
| `xcquinox/alec/tests/fixtures/pretrain_data_default_reference.npz` | The recorded default output, produced BEFORE the generator is touched. |
| `xcquinox/alec/tests/test_pretrain_set.py` | The composition layer. |
| `xcquinox/alec/tests/test_pretrain_systems.py` | `_system_columns` / `_molecule_columns` and the per-system energy targets. |
| `xcquinox/alec/tests/test_pretrain_schema.py` | The two-block `.npz` schema, the system table, the manifest identity and the default-output regression pin. |
| `xcquinox/alec/tests/test_pretrain_energy_term.py` | The energy term in the loss, the validation split and the stop criterion. |

---

## Task 1: The pretraining-set composition layer

**Files:**
- Modify: `xcquinox/alec/pretrain_data_gen.py` -- a new block immediately after `_RHO_FLOOR` (currently line 59; the Section 3.1 insertion of `spin_channel_exchange_rows` sits just below it, so insert BEFORE that function)
- Test: `xcquinox/alec/tests/test_pretrain_set.py` (create)

**Interfaces:**
- Consumes: `full_benchmark_pools.load_full_held_out_pools`; `dfs_pretrain_set.dfs_pretrain_records` behind the `_dfs_pretrain_records` seam; `rungs.seed_xc_for_arch` (agreement only, pinned by test).
- Produces:
  - `pretrain_data_gen.PretrainSystem` -- `namedtuple("PretrainSystem", ("name", "atom", "charge", "spin"))`
  - `pretrain_data_gen.normalize_system(obj) -> PretrainSystem`
  - `pretrain_data_gen._geometry_key(atom_str) -> tuple`, `_n_atoms(atom_str) -> int`, `_composition_from_atom(atom_str) -> tuple[tuple[str, int], ...]`
  - `pretrain_data_gen._mol_spec_for(system, basis, grid_level) -> MoleculeSpec`
  - `pretrain_data_gen.pool_atom_systems() -> tuple[PretrainSystem, ...]` (14 entries)
  - `pretrain_data_gen.dfs_level_for_reference_xc(reference_xc: str) -> str`
  - `pretrain_data_gen.resolve_parent_density(arch, parent_density: str) -> str`
  - `pretrain_data_gen.resolve_pretrain_systems(*, atoms=None, dfs_set=False, pool_atoms=False, reference_xc="pbe") -> tuple[PretrainSystem, ...]`
  - `pretrain_data_gen.pretrain_data_filename(polarized: bool, reference_xc: str = "pbe") -> str`

**Vocabulary.** The YAML knob is `pretrain.parent_density` -- it names a physical
choice, whose density. The value that reaches the library is the `reference_xc`
keyword of `precompute_fixed_density_data`, and that is the name used everywhere
below the config layer and in the data manifest, so one word follows the value
from the YAML to the SCF.

- [ ] **Step 1: Write the failing tests**

Create `xcquinox/alec/tests/test_pretrain_set.py`:

```python
"""Composition of the pretraining set (spec Sections 3.2, 6 deviation 1, 7).

Section 7 binds the set: "the DFS pretraining set in its entirety, plus every
atom of the BH76 / W4-11 pools (open shells per spin channel), plus the
synthetic mesh as a regularizer". These tests pin the composition layer alone --
no SCF, no libxc, no grid.
"""
import pytest

import xcquinox.alec.pretrain_data_gen as pdg


# ---------------------------------------------------------------------------
# The pool atoms
# ---------------------------------------------------------------------------

def test_pool_atom_systems_are_the_fourteen_single_atom_species():
    """Section 6 deviation 1 says "14 elements". The union of the two pools has
    12 neutral single-atom species plus the two closed-shell anions F- and Cl-
    that carry BH76 barrier heights: 14 distinct (symbol, charge, 2S) triples."""
    got = sorted((s.atom.split()[0], s.charge, s.spin)
                 for s in pdg.pool_atom_systems())
    assert got == sorted([
        ("Al", 0, 1), ("B", 0, 1), ("Be", 0, 0), ("C", 0, 2),
        ("Cl", 0, 1), ("Cl", -1, 0), ("F", 0, 1), ("F", -1, 0),
        ("H", 0, 1), ("N", 0, 3), ("O", 0, 2), ("P", 0, 3),
        ("S", 0, 2), ("Si", 0, 2),
    ])
    assert len(got) == 14


def test_pool_atom_names_are_unique_and_mark_the_anions():
    names = [s.name for s in pdg.pool_atom_systems()]
    assert len(set(names)) == len(names)
    assert {"F-", "Cl-", "Be", "H"} <= set(names)


def test_pool_atoms_sit_at_the_origin_like_the_free_atom_path():
    """A free atom's geometry is a single nucleus at the origin, spelled the
    same way the historical atom path spells it, so a pool atom and a
    ``pretrain.atoms`` entry for the same element deduplicate to one system."""
    for s in pdg.pool_atom_systems():
        assert s.atom.split()[1:] == ["0", "0", "0"], s.atom


# ---------------------------------------------------------------------------
# normalize_system
# ---------------------------------------------------------------------------

def test_normalize_system_accepts_the_historical_symbol_spin_pair():
    assert pdg.normalize_system(("O", 2)) == pdg.PretrainSystem(
        name="O", atom="O 0 0 0", charge=0, spin=2)


def test_normalize_system_accepts_a_dfs_record():
    """The DFS inventory hands out mappings with exactly these keys plus
    ``kind``, ``atom_composition`` and ``g2_97_index``, which are ignored."""
    s = pdg.normalize_system({"kind": "molecule", "name": "H2",
                              "atom": "H 0 0 0; H 0 0 0.74", "charge": 0,
                              "spin": 0, "atom_composition": [["H", 2]],
                              "g2_97_index": 2})
    assert (s.name, s.charge, s.spin) == ("H2", 0, 0)


def test_normalize_system_accepts_a_mol_spec():
    from xcquinox.alec.config import MoleculeSpec
    ms = MoleculeSpec(name="he", atom="He 0 0 0", basis="sto-3g", charge=0,
                      spin=0, atom_composition=(("He", 1),))
    assert pdg.normalize_system(ms).atom == "He 0 0 0"


def test_normalize_system_is_idempotent():
    s = pdg.PretrainSystem("x", "H 0 0 0", 0, 1)
    assert pdg.normalize_system(s) is s


def test_normalize_system_refuses_an_unusable_object():
    with pytest.raises(TypeError, match="pretraining system"):
        pdg.normalize_system(42)


# ---------------------------------------------------------------------------
# Geometry-keyed de-duplication and the MoleculeSpec builder
# ---------------------------------------------------------------------------

def test_geometry_key_collapses_two_spellings_of_the_same_atom():
    assert pdg._geometry_key("H 0 0 0") == pdg._geometry_key(
        "h 0.0 0.0 0.00000000")


def test_geometry_key_separates_two_geometries():
    assert pdg._geometry_key("H 0 0 0; H 0 0 0.74") != pdg._geometry_key(
        "H 0 0 0; H 0 0 1.40")


def test_geometry_key_rejects_a_malformed_geometry():
    with pytest.raises(ValueError, match="geometry"):
        pdg._geometry_key("H 0 0")


def test_composition_and_atom_count_come_from_the_geometry():
    assert pdg._composition_from_atom("C 0 0 0; H 0 0 1.1; H 0 1.1 0") == \
        (("C", 1), ("H", 2))
    assert pdg._n_atoms("C 0 0 0; H 0 0 1.1; H 0 1.1 0") == 3


def test_mol_spec_for_carries_the_identity_and_the_composition():
    """The generator hands this spec to precompute_fixed_density_data, so its
    composition is derived from the geometry rather than trusted from a record:
    a pool entry, a DFS entry and a (symbol, 2S) pair must produce the same spec
    for the same molecule."""
    ms = pdg._mol_spec_for({"name": "h2o",
                            "atom": "O 0 0 0.117; H 0 0.757 -0.469; "
                                    "H 0 -0.757 -0.469",
                            "charge": 0, "spin": 0},
                           "def2-svp", 3)
    assert ms.name == "h2o"
    assert ms.basis == "def2-svp"
    assert ms.grid_level == 3
    assert ms.charge == 0 and ms.spin == 0
    assert ms.atom_composition == (("H", 2), ("O", 1))


# ---------------------------------------------------------------------------
# resolve_pretrain_systems
# ---------------------------------------------------------------------------

def test_resolve_defaults_to_the_historical_four_atoms():
    got = pdg.resolve_pretrain_systems()
    assert tuple((s.name, s.spin) for s in got) == pdg.DEFAULT_PRETRAIN_ATOMS


def test_resolve_honors_an_explicit_atom_list():
    got = pdg.resolve_pretrain_systems(atoms=(("Li", 1), ("C", 2)))
    assert [s.name for s in got] == ["Li", "C"]


def test_resolve_pool_atoms_drops_the_historical_default():
    """Turning an inventory on replaces the four-atom default rather than adding
    to it: He is in neither pool nor the DFS set, and the set the spec binds is
    stated exactly."""
    got = pdg.resolve_pretrain_systems(pool_atoms=True)
    assert len(got) == 14
    assert "He" not in [s.name for s in got]


def test_resolve_keeps_an_explicit_atom_alongside_the_pool():
    got = pdg.resolve_pretrain_systems(atoms=(("He", 0),), pool_atoms=True)
    assert len(got) == 15
    assert got[-1].name == "He"


def test_resolve_deduplicates_by_geometry_charge_and_spin():
    got = pdg.resolve_pretrain_systems(atoms=(("H", 1),), pool_atoms=True)
    assert len(got) == 14


def test_resolve_keeps_an_ion_beside_its_neutral_atom():
    """Same geometry, different charge: two physical systems, both kept."""
    got = pdg.resolve_pretrain_systems(pool_atoms=True)
    fluorines = [s for s in got if s.atom.startswith("F ")]
    assert sorted((s.charge, s.spin) for s in fluorines) == [(-1, 0), (0, 1)]


# ---------------------------------------------------------------------------
# The DFS inventory seam
# ---------------------------------------------------------------------------

def test_dfs_level_maps_the_rung_baseline_to_the_inventory():
    assert pdg.dfs_level_for_reference_xc("pbe") == "gga"
    assert pdg.dfs_level_for_reference_xc("scan") == "mgga"
    with pytest.raises(ValueError, match="reference_xc"):
        pdg.dfs_level_for_reference_xc("blyp")


def test_dfs_levels_agree_with_the_inventory_module():
    from xcquinox.alec.dfs_pretrain_set import LEVELS
    assert set(LEVELS) == {pdg.dfs_level_for_reference_xc("pbe"),
                           pdg.dfs_level_for_reference_xc("scan")}


def test_resolve_dfs_set_asks_the_inventory_for_the_rungs_level(monkeypatch):
    calls = []

    def _fake(level):
        calls.append(level)
        return [{"name": "H2", "atom": "H 0 0 0; H 0 0 0.74", "charge": 0,
                 "spin": 0},
                {"name": "Li", "atom": "Li 0 0 0", "charge": 0, "spin": 1}]

    monkeypatch.setattr(pdg, "_dfs_pretrain_records", _fake)
    got = pdg.resolve_pretrain_systems(dfs_set=True, reference_xc="scan")
    assert calls == ["mgga"]
    assert [s.name for s in got] == ["H2", "Li"]


def test_resolve_dfs_set_reads_the_committed_inventory():
    gga = pdg.resolve_pretrain_systems(dfs_set=True, reference_xc="pbe")
    assert len(gga) == 30, [s.name for s in gga]
    mgga = pdg.resolve_pretrain_systems(dfs_set=True, reference_xc="scan")
    assert len(mgga) == 28, [s.name for s in mgga]


def test_meta_gga_level_drops_exactly_the_excluded_molecules():
    from xcquinox.alec.dfs_pretrain_set import MGGA_EXCLUDED
    gga = {s.name for s in pdg.resolve_pretrain_systems(dfs_set=True,
                                                        reference_xc="pbe")}
    mgga = {s.name for s in pdg.resolve_pretrain_systems(dfs_set=True,
                                                         reference_xc="scan")}
    assert gga - mgga == set(MGGA_EXCLUDED)


def test_the_v6_set_is_the_dfs_set_plus_every_pool_atom():
    """DFS contributes 8 free atoms, 7 of which (H, N, O, P, S, Cl, Al) are also
    pool atoms; the pools add B, Be, C, F, Si and the two anions. 30 + 7 for the
    GGA rung, 28 + 7 for the meta-GGA rung."""
    gga = pdg.resolve_pretrain_systems(dfs_set=True, pool_atoms=True,
                                       reference_xc="pbe")
    names = [s.name for s in gga]
    assert len(set(names)) == len(names)
    assert len(names) == 37, names
    mgga = pdg.resolve_pretrain_systems(dfs_set=True, pool_atoms=True,
                                        reference_xc="scan")
    assert len(mgga) == 35, [s.name for s in mgga]


# ---------------------------------------------------------------------------
# Parent density and filename
# ---------------------------------------------------------------------------

def test_resolve_parent_density_passes_an_explicit_choice_through():
    from xcquinox.alec.config import get_architecture
    arch = get_architecture("deep_3x16")
    assert pdg.resolve_parent_density(arch, "pbe") == "pbe"
    assert pdg.resolve_parent_density(arch, "scan") == "scan"
    with pytest.raises(ValueError, match="parent_density"):
        pdg.resolve_parent_density(arch, "blyp")


def test_resolve_parent_density_auto_is_the_rung_baseline():
    """"auto" must agree with rungs.seed_xc_for_arch under its production
    "mgga_scan" policy for EVERY registered architecture: the pretraining parent
    density and the SCF seed are the same rung baseline (PBE for the GGA rung,
    SCAN for the meta-GGA rung), and a disagreement would pretrain a network
    against a density its own SCF never visits."""
    from xcquinox.alec.config import ARCHITECTURES, get_architecture
    from xcquinox.alec.rungs import seed_xc_for_arch
    for name in ARCHITECTURES:
        assert pdg.resolve_parent_density(get_architecture(name), "auto") == \
            seed_xc_for_arch(name), name


def test_pretrain_data_filename_keeps_the_two_historical_names():
    assert pdg.pretrain_data_filename(False) == "pretrain_data.npz"
    assert pdg.pretrain_data_filename(True) == "pretrain_data_polarized.npz"
    assert pdg.pretrain_data_filename(True, "scan") == \
        "pretrain_data_polarized_scan.npz"
    assert pdg.pretrain_data_filename(False, "scan") == "pretrain_data_scan.npz"
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
mkdir -p /tmp/xcq-testlogs
python -m pytest xcquinox/alec/tests/test_pretrain_set.py -v > /tmp/xcq-testlogs/task01_red.log 2>&1; echo "exit=$?"
```
Expected: every test errors with `AttributeError: module
'xcquinox.alec.pretrain_data_gen' has no attribute 'pool_atom_systems'` and the
sibling names. Read the log with `Read`.

- [ ] **Step 3: Add the composition layer**

Insert into `xcquinox/alec/pretrain_data_gen.py` immediately after the
`_RHO_FLOOR` constant (currently line 59):

```python
#: LDA exchange coefficient, ``eps_x^LDA(rho) = _LDA_X_C rho^(1/3)``. The same
#: constant libxc's ``LDA_X,`` returns at spin=0 and the same one
#: :func:`spin_channel_exchange_rows` divides by, kept here so the per-system
#: energy targets and the stored enhancement factors share one denominator.
_LDA_X_C = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)

#: One pretraining system: a geometry, a charge and a PySCF 2S spin. ``atom`` is
#: a PySCF geometry string in Angstrom. Free atoms are spelled
#: ``"<Sym> 0 0 0"`` so a pool atom and a ``pretrain.atoms`` entry for the same
#: element are one system rather than two.
PretrainSystem = namedtuple("PretrainSystem", ("name", "atom", "charge", "spin"))


def _system_name(symbol, charge):
    """Canonical system name for a free atom: the symbol, with the ion charge
    appended as a run of ``+`` or ``-`` (``F-``, ``Cl-``). Names are labels for
    provenance and for the validation split's record; the physics is carried by
    (geometry, charge, spin)."""
    if charge == 0:
        return str(symbol)
    sign = "+" if charge > 0 else "-"
    return f"{symbol}{sign * abs(int(charge))}"


def _geometry_key(atom_str):
    """Canonical hashable geometry: ``(symbol, x, y, z)`` per nucleus, rounded
    to 1e-8 angstrom and sorted.

    Two spellings of the same structure ("H 0 0 0" and "H 0.0 0.0 0.0", or two
    orderings of the same nuclei) collapse to one key, so the DFS inventory and
    the pool inventory deduplicate against each other and against an explicit
    ``pretrain.atoms`` entry without depending on how each source spells its
    geometry.
    """
    items = []
    for chunk in str(atom_str).replace("\n", ";").split(";"):
        parts = chunk.split()
        if not parts:
            continue
        if len(parts) != 4:
            raise ValueError(
                f"malformed PySCF geometry chunk {chunk!r} in {atom_str!r}: "
                "expected '<symbol> <x> <y> <z>' per nucleus."
            )
        items.append((parts[0].capitalize(), round(float(parts[1]), 8),
                      round(float(parts[2]), 8), round(float(parts[3]), 8)))
    if not items:
        raise ValueError(f"empty PySCF geometry {atom_str!r}.")
    return tuple(sorted(items))


def _n_atoms(atom_str):
    """Number of nuclei in a PySCF geometry string."""
    return len(_geometry_key(atom_str))


def _composition_from_atom(atom_str):
    """``((symbol, count), ...)``, sorted, for a PySCF geometry string.

    Derived from the geometry rather than trusted from a record, so a pool
    entry, a DFS entry and a ``(symbol, 2S)`` pair produce the same composition
    for the same molecule and therefore the same MoleculeSpec.
    """
    counts = {}
    for symbol, _x, _y, _z in _geometry_key(atom_str):
        counts[symbol] = counts.get(symbol, 0) + 1
    return tuple(sorted(counts.items()))


def normalize_system(obj):
    """Coerce a pretraining-system descriptor into a :class:`PretrainSystem`.

    Accepts a ``PretrainSystem``; a mapping carrying ``name``/``atom``/
    ``charge``/``spin`` (the schema of the committed pool JSON and of
    ``dfs_pretrain_set.dfs_pretrain_records``, whose extra ``kind``,
    ``atom_composition`` and ``g2_97_index`` entries are ignored); any object
    exposing those four attributes (``config.MoleculeSpec``); or a
    ``(symbol, 2S)`` pair, the historical ``pretrain.atoms`` form, which names a
    neutral free atom at the origin. Keeping the coercion in one place is what
    lets the set be assembled from three inventories written independently.
    """
    if isinstance(obj, PretrainSystem):
        return obj
    if isinstance(obj, dict):
        return PretrainSystem(name=str(obj["name"]), atom=str(obj["atom"]),
                              charge=int(obj.get("charge", 0)),
                              spin=int(obj.get("spin", 0)))
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        symbol, spin = obj
        return PretrainSystem(name=str(symbol), atom=f"{symbol} 0 0 0",
                              charge=0, spin=int(spin))
    if all(hasattr(obj, a) for a in ("name", "atom", "charge", "spin")):
        return PretrainSystem(name=str(obj.name), atom=str(obj.atom),
                              charge=int(obj.charge), spin=int(obj.spin))
    raise TypeError(
        f"cannot read {obj!r} as a pretraining system: expected a "
        "PretrainSystem, a mapping with name/atom/charge/spin, an object with "
        "those attributes, or a (symbol, 2S) pair."
    )


def _mol_spec_for(system, basis, grid_level):
    """The :class:`~xcquinox.alec.config.MoleculeSpec` for one pretraining
    system at the run's identity.

    This is the spec :func:`data.precompute_fixed_density_data` receives, so the
    pretraining rows and the training features of the same molecule are built
    from the same object.
    """
    from xcquinox.alec.config import MoleculeSpec
    system = normalize_system(system)
    return MoleculeSpec(
        name=system.name, atom=system.atom, basis=basis,
        charge=int(system.charge), spin=int(system.spin),
        atom_composition=_composition_from_atom(system.atom),
        grid_level=grid_level)


def pool_atom_systems():
    """Every single-atom species of the BH76 and W4-11 pools, de-duplicated.

    Fourteen distinct (symbol, charge, 2S) triples: the twelve neutral elements
    the two pools span -- Al, B, Be, C, Cl, F, H, N, O, P, S, Si, with their
    Hund's-rule ground-state spins -- plus the two closed-shell anions F- and
    Cl-, which are reactants of BH76 barrier heights and therefore systems the
    Section 3.3 certificate bounds at tol_atom. All of them are free atoms at
    the origin, so the geometry is the pool's own.

    Read from the committed pool JSON through ``full_benchmark_pools`` rather
    than transcribed, so a pool edit propagates here.
    """
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    mol_specs, _reactions = load_full_held_out_pools()
    seen = {}
    for ms in mol_specs.values():
        composition = dict(ms.atom_composition)
        if sum(composition.values()) != 1:
            continue
        seen.setdefault((str(next(iter(composition))), int(ms.charge),
                         int(ms.spin)), None)
    return tuple(
        PretrainSystem(name=_system_name(symbol, charge),
                       atom=f"{symbol} 0 0 0", charge=charge, spin=spin)
        for symbol, charge, spin in sorted(seen)
    )


def dfs_level_for_reference_xc(reference_xc):
    """Which DFS pretraining inventory a parent density's file uses.

    The DFS notebook ships two variants (spec Section 6): the GGA one with 22
    G2/97 molecules and the meta-GGA one with 20, the difference being H2 and N2
    (``dfs_pretrain_set.MGGA_EXCLUDED``). The parent functional and the
    inventory are the same rung choice, so one maps onto the other.
    """
    if reference_xc == "pbe":
        return "gga"
    if reference_xc == "scan":
        return "mgga"
    raise ValueError(
        f"reference_xc must be 'pbe' or 'scan'; got {reference_xc!r}.")


def resolve_parent_density(arch, parent_density):
    """The ``reference_xc`` whose self-consistent density ``arch`` pretrains on.

    ``pretrain.parent_density`` is the YAML knob; this is where its value becomes
    the ``reference_xc`` keyword of
    :func:`data.precompute_fixed_density_data`. ``"pbe"`` / ``"scan"`` pass
    through. ``"auto"`` is the rung baseline: SCAN for the meta-GGA rung, PBE
    otherwise (spec Section 1, "PBE for GGA-rung architectures, SCAN for
    meta-GGA architectures"). That is the map ``rungs.seed_xc_for_arch`` applies
    under its production ``"mgga_scan"`` policy, computed from the architecture
    OBJECT rather than a registry name so an architecture built ad hoc resolves
    too; the agreement with ``seed_xc_for_arch`` over the whole registry is
    pinned by test.
    """
    if parent_density in ("pbe", "scan"):
        return parent_density
    if parent_density != "auto":
        raise ValueError(
            "parent_density must be 'pbe', 'scan' or 'auto'; got "
            f"{parent_density!r}."
        )
    return "scan" if bool(getattr(arch, "meta_gga", False)) else "pbe"


def _dfs_pretrain_records(level):
    """The DFS pretraining inventory for ``level`` ("gga" / "mgga").

    A named seam so the composition layer can be tested without the inventory
    and so an import failure names the module that supplies it. The RECORD form
    is read rather than ``dfs_pretrain_systems``: the composition layer is
    basis-free, and the basis and grid level are applied later by
    :func:`_mol_spec_for` at the run's own identity.
    """
    try:
        from xcquinox.alec.dfs_pretrain_set import dfs_pretrain_records
    except ImportError as exc:  # pragma: no cover - exercised when absent
        raise ImportError(
            "the DFS pretraining inventory lives in "
            "xcquinox.alec.dfs_pretrain_set (dfs_pretrain_records(level)); "
            "pretrain.dfs_set cannot be honored without it"
        ) from exc
    return dfs_pretrain_records(level)


def resolve_pretrain_systems(*, atoms=None, dfs_set=False, pool_atoms=False,
                             reference_xc="pbe"):
    """The ordered, de-duplicated pretraining set.

    Order is DFS inventory, then pool atoms, then the explicit ``atoms`` list,
    with the first occurrence of a (geometry, charge, spin) winning. ``atoms`` of
    ``None`` means the historical four-atom default when neither inventory is
    requested and NOTHING when one is: the set Section 7 binds is stated exactly
    ("the DFS pretraining set in its entirety, plus every atom of the BH76 /
    W4-11 pools"), and He belongs to neither.
    """
    if atoms is None:
        atoms = () if (dfs_set or pool_atoms) else DEFAULT_PRETRAIN_ATOMS
    ordered = []
    if dfs_set:
        ordered.extend(_dfs_pretrain_records(
            dfs_level_for_reference_xc(reference_xc)))
    if pool_atoms:
        ordered.extend(pool_atom_systems())
    ordered.extend(atoms)
    out = []
    seen = set()
    for entry in ordered:
        system = normalize_system(entry)
        key = (_geometry_key(system.atom), int(system.charge), int(system.spin))
        if key in seen:
            continue
        seen.add(key)
        out.append(system)
    return tuple(out)


def pretrain_data_filename(polarized, reference_xc="pbe"):
    """Canonical pretrain-data filename.

    ``reference_xc="pbe"`` reproduces the two historical names. The SCAN-density
    file carries its own suffix because it is built at a DIFFERENT
    self-consistent density (spec Section 6 deviation 1) and its rows are not
    interchangeable with the PBE file's.
    """
    base = "pretrain_data_polarized" if polarized else "pretrain_data"
    return (f"{base}.npz" if reference_xc == "pbe"
            else f"{base}_{reference_xc}.npz")
```

Add `from collections import namedtuple` to the import block at the top of the
module (currently lines 40-48), immediately after `import os`.

- [ ] **Step 4: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/pretrain_data_gen.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_set.py -v > /tmp/xcq-testlogs/task01_green.log 2>&1; echo "exit=$?"
```
Expected: PASS, all of them -- `dfs_pretrain_set.py` and its committed JSON are
already in the tree.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_pretrain_set.py -v > /tmp/xcq-testlogs/task01_green.log 2>&1`

---

## Task 2: Per-system columns on the parent's precomputed density

**Files:**
- Modify: `xcquinox/alec/cluster/_datagen.py:22-45` (JAX routing preamble and the lazily bound generator seam)
- Modify: `xcquinox/alec/pretrain_data_gen.py` -- `_atom_columns` (the function immediately after `spin_channel_exchange_rows`)
- Test: `xcquinox/alec/tests/test_pretrain_systems.py` (create), `xcquinox/alec/tests/test_cluster_datagen.py` (append)

**Interfaces:**
- Consumes: `PretrainSystem`, `normalize_system`, `_mol_spec_for` (Task 1); `data.precompute_fixed_density_data(..., reference_xc=)` (Section 3.3 plan); `spin_channel_exchange_rows` and the `exchange_footing` switch (Section 3.1 plan).
- Produces:
  - `_datagen._route_jax_env() -> None`
  - `pretrain_data_gen._require_sane_density(mol_data, system, reference_xc, basis, grid_level, n_electrons) -> None`
  - `pretrain_data_gen._system_columns(system, basis, grid_level, *, reference_xc, polarized, descriptors, density_fit=False, auxbasis=None, cusp_log_transform=True, exchange_footing="total") -> dict`
  - `pretrain_data_gen._molecule_columns(mol_spec, reference_xc, basis, grid_level, *, polarized, descriptors, density_fit=False, auxbasis=None, cusp_log_transform=True, exchange_footing="total") -> dict`
  - `_atom_columns` keeps its signature and becomes the single-atom wrapper.
  - Both column dicts gain the keys `e_lda_x` and `e_lda_c`.

**Why the density comes from the precompute.** The Section 3.3 certificate
compares `E_xc^NN` with `E_xc^parent` on a frozen parent density produced by
`precompute_fixed_density_data`, and training evaluates its features on the
density that same function produces. A generator that ran its own SCF would fit
a network on a density that merely OUGHT to equal the one it is later measured
on. Routing the rows through the same function with `reference_xc` makes that
equality structural, and removes the last place in the library that decides for
itself what "the PBE density" means.

- [ ] **Step 1: Route JAX to fp64 on the datagen node**

The datagen worker never sets `JAX_ENABLE_X64`, so on a compute node JAX
defaults to single precision and every `jnp.asarray` in the generator -- the
kinetic-energy density, the iso-orbital indicator, the rung-3.5 occupancies, the
cusp feature -- is computed and stored in float32. The test suite enables fp64
in `conftest.py`, which is why no test has ever seen it. Routing the density
through `precompute_fixed_density_data`, whose `MoleculeData` holds `dm_pbe` /
`ao_grid_deriv` / `grid_weights` as `jnp.array(...)`, would push the DENSITY
ITSELF through that downcast, so this is a prerequisite and not a tidy-up.

Add to `xcquinox/alec/tests/test_cluster_datagen.py`:

```python
def test_datagen_routes_jax_to_double_precision(monkeypatch):
    """The generator's descriptor, tau and alpha columns are JAX computations,
    and the parent density arrives as a jnp array. Without this the datagen node
    computes them in single precision while every test computes them in double,
    so the file the cluster writes is not the file the tests describe."""
    import os
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    _datagen._route_jax_env()
    assert os.environ["JAX_ENABLE_X64"] == "1"


def test_datagen_routes_before_anything_that_imports_jax():
    """The switch is only honored before the first jax import, so the routing
    call must precede every other statement of main() and the module must not
    import the generator at module scope."""
    import inspect
    src = inspect.getsource(_datagen.main)
    assert src.index("_route_jax_env()") < src.index("argv = sys.argv")
    assert src.index("_route_jax_env()") < src.index("load_grid_config")
    module_src = inspect.getsource(_datagen)
    head = module_src.split("def _route_jax_env")[0]
    assert "import pretrain_data_gen" not in head, (
        "importing the generator at module scope pulls in jax.numpy before "
        "_route_jax_env can set the precision flag")
```

In `xcquinox/alec/cluster/_datagen.py`, replace the module-level generator
import and seam (lines 31-38) with a lazily bound seam:

```python
# Mockable heavy-call seam, tests monkeypatch ``_datagen._ensure_pretrain_data``
# to assert the generation calls without running real SCFs. Bound lazily in
# ``main`` rather than at import, because importing the generator pulls in
# jax.numpy and the precision flag below must be set first; a test that patches
# the name still wins, since the rebind only fires while the value is None.
_ensure_pretrain_data = None


def _route_jax_env():
    """Pin JAX to fp64 via the environment, before jax is imported.

    The pretrain-data generator computes the kinetic-energy density, the
    iso-orbital indicator, the rung-3.5 occupancies and the cusp feature through
    JAX, and reads the parent density out of a ``MoleculeData`` whose arrays are
    ``jnp.array``. JAX defaults to float32 and equinox / optax may capture the
    default dtype before a post-import config update runs, so the env-var switch
    is the only reliable one; ``cluster._pretrain`` and
    ``cluster._eval_one_spec`` open the same way. ``JAX_PLATFORMS`` is left
    untouched so the sbatch-requested device is honored.
    """
    os.environ["JAX_ENABLE_X64"] = "1"
```

and open `main` with it:

```python
def main(argv=None) -> int:
    """Datagen-job entrypoint. Returns a process exit code (0 = success)."""
    # Route JAX before any import that pulls it in. The module-level imports are
    # jax-free; xcquinox.alec.pretrain_data_gen imports jax.numpy, so the seam
    # below is bound only after this call.
    _route_jax_env()
    global _ensure_pretrain_data
    if _ensure_pretrain_data is None:
        from xcquinox.alec import pretrain_data_gen as _pdg
        _ensure_pretrain_data = _pdg.ensure_pretrain_data
    if argv is None:
        argv = sys.argv[1:]
```

```bash
python -m py_compile xcquinox/alec/cluster/_datagen.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_datagen.py -v > /tmp/xcq-testlogs/task02_datagen.log 2>&1; echo "exit=$?"
```
Expected: PASS, with the six pre-existing datagen tests unchanged.

- [ ] **Step 2: Write the failing column tests**

Create `xcquinox/alec/tests/test_pretrain_systems.py`:

```python
"""Per-system pretrain columns on the parent functional's own density.

Section 6 deviation 1 puts the pretraining set "at the production identity ...
on the parent functional's own self-consistent densities (PBE for GGA-rung,
SCAN for meta-GGA; DFS used PBE for both)". The density comes from
data.precompute_fixed_density_data(..., reference_xc=...), the one place this
library produces a frozen parent density, so the rows a network is fit on and
the rows the fidelity certificate measures it on are the same quadrature of the
same density by construction. These tests run real SCFs on tiny systems
(sto-3g, grid level 0 or 1, He / H / H2 / OH).
"""
import numpy as np
import pytest

import xcquinox.alec.pretrain_data_gen as pdg
from xcquinox.alec.pretrain_data_gen import (
    PretrainSystem, _atom_columns, _molecule_columns)


_H2 = PretrainSystem(name="h2", atom="H 0 0 0; H 0 0 0.74", charge=0, spin=0)
_OH = PretrainSystem(name="oh", atom="O 0 0 0; H 0 0 0.97", charge=0, spin=1)


def test_molecule_columns_are_aligned_and_finite():
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=True,
                             descriptors=True)
    n = cols["rho"].shape[0]
    assert n > 0
    for key in ("sigma", "Fx", "Fc", "Fx_scan", "Fc_scan", "weights", "zeta",
                "e_lda_x", "e_lda_c"):
        assert np.asarray(cols[key]).shape == (n,), key
        assert np.all(np.isfinite(np.asarray(cols[key]))), key
    assert np.asarray(cols["metagga"]).shape == (n, 1)
    assert np.asarray(cols["cusp"]).shape == (n, 2)
    assert np.asarray(cols["dm"]).shape[0] == n
    assert np.asarray(cols["rung35"]).shape == (n, 2)
    assert np.asarray(cols["rung35ms"]).shape == (n, 6)
    assert np.asarray(cols["rho"]).dtype == np.float64


def test_molecule_columns_reproduce_the_atom_path_for_a_free_atom():
    """A free atom is the single-nucleus case of the molecular builder. A
    divergence would mean the atomic rows and the molecular rows are not the
    same quantity, which is the failure the coverage change exists to remove."""
    a = _atom_columns("H", 1, "sto-3g", 0, polarized=True, descriptors=True)
    m = _molecule_columns(PretrainSystem("H", "H 0 0 0", 0, 1), "pbe",
                          "sto-3g", 0, polarized=True, descriptors=True)
    assert set(a) == set(m)
    for key in a:
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(m[key]),
                                      err_msg=key)


def test_columns_sit_on_the_precomputes_grid_and_density():
    """The rows are a quadrature of the SAME grid and the SAME density matrix
    the training features are built from; that identity is what makes the
    certificate's E_xc^NN - E_xc^parent a statement about the network rather
    than about two pipelines that were supposed to agree."""
    from xcquinox.alec.data import precompute_fixed_density_data
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=False,
                             descriptors=False)
    md = precompute_fixed_density_data(
        pdg._mol_spec_for(_H2, "sto-3g", 0), required_keys=(), descriptors=())
    w = np.asarray(md["grid_weights"])
    rho = np.asarray(md["rho_grid"])
    keep = rho > pdg._RHO_FLOOR
    np.testing.assert_array_equal(cols["weights"], w[keep])
    np.testing.assert_allclose(cols["rho"], rho[keep], rtol=0, atol=1e-12)


def test_energy_density_columns_invert_the_stored_ratio():
    """w * e_lda * (1 + F) is the parent's energy quadrature. Summing it must
    reproduce libxc's own integrated exchange and correlation on the same grid,
    up to the density floor and the +-5 clip on the stored ratio."""
    from pyscf import dft, gto
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=False,
                             descriptors=False)
    mol = gto.M(atom=_H2.atom, basis="sto-3g", charge=0, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.kernel()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
    rho_gga = mf._numint.eval_rho(mol, ao, mf.make_rdm1(), xctype="GGA",
                                 hermi=True)
    w = np.asarray(mf.grids.weights)
    ref_x = float(np.sum(w * rho_gga[0] * np.asarray(
        mf._numint.eval_xc("PBE,", rho_gga, spin=0)[0])))
    ref_c = float(np.sum(w * rho_gga[0] * np.asarray(
        mf._numint.eval_xc(",PBE", rho_gga, spin=0)[0])))
    got_x = float(np.sum(cols["weights"] * cols["e_lda_x"]
                         * (1.0 + cols["Fx"])))
    got_c = float(np.sum(cols["weights"] * cols["e_lda_c"]
                         * (1.0 + cols["Fc"])))
    assert abs(got_x - ref_x) < 1e-9, (got_x, ref_x)
    assert abs(got_c - ref_c) < 1e-9, (got_c, ref_c)


def test_open_shell_energy_density_columns_use_the_spin_resolved_baseline():
    """The open-shell Fx / Fc are libxc spin=1 ratios, so their denominators are
    the SPIN-POLARIZED LDA and PW92 per-electron energies at the total density.
    e_lda_x / e_lda_c must be those same denominators times rho_tot, or the
    energy term would integrate a different functional than the fit."""
    from pyscf import dft, gto
    cols = _molecule_columns(_OH, "pbe", "sto-3g", 0, polarized=True,
                             descriptors=False)
    mol = gto.M(atom=_OH.atom, basis="sto-3g", charge=0, spin=1, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.kernel()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
    dm = mf.make_rdm1()
    ra = mf._numint.eval_rho(mol, ao, dm[0], xctype="GGA", hermi=True)
    rb = mf._numint.eval_rho(mol, ao, dm[1], xctype="GGA", hermi=True)
    rho_uks = np.stack([ra, rb], axis=0)
    w = np.asarray(mf.grids.weights)
    ref_x = float(np.sum(w * (ra[0] + rb[0]) * np.asarray(
        mf._numint.eval_xc("PBE,", rho_uks, spin=1)[0])))
    ref_c = float(np.sum(w * (ra[0] + rb[0]) * np.asarray(
        mf._numint.eval_xc(",PBE", rho_uks, spin=1)[0])))
    got_x = float(np.sum(cols["weights"] * cols["e_lda_x"]
                         * (1.0 + cols["Fx"])))
    got_c = float(np.sum(cols["weights"] * cols["e_lda_c"]
                         * (1.0 + cols["Fc"])))
    assert abs(got_x - ref_x) < 1e-9, (got_x, ref_x)
    assert abs(got_c - ref_c) < 1e-9, (got_c, ref_c)


def test_scan_reference_uses_the_scan_density():
    """reference_xc='scan' must reach precompute_fixed_density_data and come
    back with SCAN's own self-consistent density: the two densities differ, and
    that difference is the whole content of the meta-GGA rung's footing."""
    pbe = _molecule_columns(_H2, "pbe", "sto-3g", 1, polarized=False,
                            descriptors=False)
    scan = _molecule_columns(_H2, "scan", "sto-3g", 1, polarized=False,
                             descriptors=False)
    assert pbe["rho"].shape == scan["rho"].shape
    assert float(np.max(np.abs(pbe["rho"] - scan["rho"]))) > 1e-9


def test_system_columns_refuse_an_unknown_reference_xc():
    with pytest.raises(ValueError, match="reference_xc"):
        _molecule_columns(_H2, "blyp", "sto-3g", 0, polarized=False,
                          descriptors=False)


def test_require_sane_density_catches_a_density_that_lost_electrons():
    """The check that needs no cooperation from the precompute: the quadrature
    of the stored density against the electron count. It catches a stalled SCF,
    a grid too coarse for a diffuse anion, and a density matrix that does not
    belong to the stored grid."""
    s = PretrainSystem("alcl3", "Al 0 0 0", 0, 0)
    good = {"rho_grid": np.full(4, 0.5), "grid_weights": np.full(4, 5.0)}
    pdg._require_sane_density(good, s, "pbe", "def2-svp", 3, 10)
    with pytest.raises(RuntimeError, match="alcl3"):
        pdg._require_sane_density(good, s, "pbe", "def2-svp", 3, 13)


def test_require_sane_density_reports_a_non_converged_scf_when_told():
    s = PretrainSystem("si2", "Si 0 0 0", 0, 0)
    md = {"rho_grid": np.full(4, 0.5), "grid_weights": np.full(4, 5.0),
          "scf_converged": False}
    with pytest.raises(RuntimeError, match="converge"):
        pdg._require_sane_density(md, s, "scan", "def2-svp", 3, 10)


def test_open_shell_molecule_carries_per_channel_exchange_rows():
    """Section 3.2: open-shell rows are posed per spin channel. The molecular
    path must reach the same row builder the atomic path does."""
    cols = _molecule_columns(_OH, "pbe", "sto-3g", 0, polarized=True,
                             descriptors=True,
                             exchange_footing="spin_channel")
    x = cols["x_rows"]
    assert x is not None
    assert x["rho"].ndim == 1
    assert x["rho"].shape[0] > cols["rho"].shape[0]
    np.testing.assert_allclose(x["rung35"][:, 0], x["rung35"][:, 1],
                               rtol=0, atol=1e-14)


def test_closed_shell_molecule_has_no_separate_exchange_rows():
    """rho_a = rho_b makes the doubled density the total one, so a closed
    shell's total-density rows ALREADY are the exact-spin-scaling rows."""
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=True,
                             descriptors=True,
                             exchange_footing="spin_channel")
    assert cols["x_rows"] is None


def test_charged_system_runs_at_its_charge():
    """F- is a BH76 species and a pretraining system; its SCF must carry the
    charge, or the row set is a different atom."""
    cols = _molecule_columns(PretrainSystem("H-", "H 0 0 0", -1, 0), "pbe",
                             "sto-3g", 0, polarized=False, descriptors=False)
    neutral = _atom_columns("H", 1, "sto-3g", 0, polarized=False,
                            descriptors=False)
    assert float(np.sum(cols["weights"] * cols["rho"])) > \
        float(np.sum(neutral["weights"] * neutral["rho"])) + 0.5
```

- [ ] **Step 3: Run the tests and confirm they fail**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_systems.py -v > /tmp/xcq-testlogs/task02_red.log 2>&1; echo "exit=$?"
```
Expected: `ImportError: cannot import name '_molecule_columns' from
'xcquinox.alec.pretrain_data_gen'`.

- [ ] **Step 4: Generalize the column builder onto the precomputed density**

Rename `_atom_columns` to `_system_columns` and replace its head -- the
signature through `mf.kernel()` -- with the block below. Everything from
`ao = mf._numint.eval_ao(...)` onward is retained, with the three substitutions
and two additions listed after it.

```python
def _require_sane_density(mol_data, system, reference_xc, basis, grid_level,
                          n_electrons):
    """Raise unless the parent density is a converged density on this grid.

    The Fx / Fc targets and the per-system energies are properties of the
    CONVERGED parent density; an unconverged one is not a functional's density
    at all and enters the fit as noise no later stage can tell from a fit error.
    ``precompute_fixed_density_data`` reports convergence only if the installed
    version carries the key, so that test is a presence test. The check that
    always runs needs no cooperation: the quadrature of the stored density
    against the electron count, which catches a stalled SCF, a grid too coarse
    to resolve a diffuse anion, and a density matrix that does not belong to the
    stored grid.
    """
    if mol_data.get("scf_converged") is False:
        raise RuntimeError(
            f"the {reference_xc} SCF for pretraining system {system.name!r} "
            f"(geometry {system.atom!r}, charge {system.charge}, 2S "
            f"{system.spin}, basis {basis}, grid level {grid_level}) did not "
            "converge"
        )
    rho = np.asarray(mol_data["rho_grid"])
    weights = np.asarray(mol_data["grid_weights"])
    n_grid = float(np.sum(weights * rho))
    tol = 1e-3 * max(1.0, float(n_electrons))
    if not abs(n_grid - float(n_electrons)) < tol:
        raise RuntimeError(
            f"the {reference_xc} density of pretraining system "
            f"{system.name!r} integrates to {n_grid:.6f} electrons on its own "
            f"grid, against {n_electrons} expected (basis {basis}, grid level "
            f"{grid_level}); the SCF did not converge or the grid does not "
            "resolve this density"
        )


def _system_columns(system, basis, grid_level, *, reference_xc, polarized,
                    descriptors, density_fit=False, auxbasis=None,
                    cusp_log_transform=True, exchange_footing="total"):
    """Pretrain columns for ONE system on the parent functional's own density.

    The general case of :func:`_atom_columns`: an arbitrary geometry, charge and
    spin, and a parent functional that is PBE (the GGA rung's baseline) or SCAN
    (the meta-GGA rung's). The density is NOT computed here: it comes from
    ``data.precompute_fixed_density_data(mol_spec, reference_xc=...)``, the one
    place this library produces a frozen parent density. Training builds its
    features from that function's output and the Section 3.3 certificate
    measures ``E_xc^NN - E_xc^parent`` on it, so obtaining the pretraining rows
    the same way makes "the same density on the same grid" structural instead of
    a coincidence that has to be re-argued whenever the pipeline moves.

    Both the PBE and the SCAN enhancement targets are evaluated on whichever
    density the file was built at, exactly as the single-atom path has always
    done, so the column layout does not depend on the parent; the manifest
    records the ``reference_xc`` and ``run_pretrain`` refuses a file whose
    parent does not match the architecture's rung.

    Returns a dict of column arrays sharing one leading length (the descriptor
    blocks are 2-D; ``x_rows`` has its own row set): ``rho``, ``sigma``, ``Fx``,
    ``Fc``, ``Fx_scan``, ``Fc_scan``, ``metagga``, ``weights``, ``e_lda_x``,
    ``e_lda_c``, optionally ``zeta``, optionally ``cusp`` / ``dm`` / ``rung35``
    / ``rung35ms``, and under the ``spin_channel`` footing ``x_rows``.

    ``e_lda_x`` and ``e_lda_c`` are the LDA energy DENSITIES ``rho eps_x^LDA``
    and ``rho eps_c^PW92`` in the EXACT convention the ``Fx`` / ``Fc`` ratios
    were formed in (libxc ``spin=1`` for an open shell, ``spin=0`` for a closed
    one). Multiplying a stored enhancement factor by them returns Hartree per
    unit volume, which is what makes the per-system energy term integrate the
    same quantity the point-wise term fits.

    ``exchange_footing`` selects how OPEN-SHELL exchange rows are posed.
    ``"total"`` is unchanged: one row per grid point at the total density with
    spin-resolved libxc targets. ``"spin_channel"`` additionally returns
    ``x_rows``, the per-channel rows of :func:`spin_channel_exchange_rows` --
    ``(2 rho_sigma, 4 sigma_sigma_sigma, features of diag(P_sigma, P_sigma))``
    with the parent's spin-unpolarized enhancement factor at those inputs as the
    target, which is what the exact spin scaling evaluates at SCF time (Oliver
    and Perdew, Phys. Rev. A 20, 397 (1979)). ``x_rows`` is ``None`` for a
    closed-shell system, whose total-density rows already are that footing.
    Correlation rows are untouched under either setting: correlation is
    spin-interpolated rather than spin-scaled and keeps the total density with
    zeta.

    ``density_fit`` is recorded in the manifest but no longer changes the parent
    SCF: the density is the precompute's, whose PBE / SCAN baseline is
    deliberately full-ERI so it is a fixed reference-quality anchor shared with
    training. ``auxbasis`` is forwarded for the same identity bookkeeping.
    """
    if reference_xc not in ("pbe", "scan"):
        raise ValueError(
            f"reference_xc must be 'pbe' or 'scan'; got {reference_xc!r}.")
    if exchange_footing not in ("total", "spin_channel"):
        raise ValueError(
            "exchange_footing must be 'total' or 'spin_channel'; got "
            f"{exchange_footing!r}."
        )
    system = normalize_system(system)
    from xcquinox.alec.data import precompute_fixed_density_data

    mol_spec = _mol_spec_for(system, basis, grid_level)
    # No descriptors and no reference keys are requested: the descriptor columns
    # below are built by the same calls the single-atom path has always used, so
    # an existing file's numbers do not move, and the precompute's own blocks
    # (which it would build at the same values) are not paid for twice.
    mol_data = precompute_fixed_density_data(
        mol_spec, required_keys=(), descriptors=(), auxbasis=auxbasis,
        reference_xc=reference_xc)

    mol = gto.M(atom=system.atom, basis=basis, charge=int(system.charge),
                spin=int(system.spin), verbose=0)
    # A mean field for its integration grid and its libxc handle ONLY: the
    # kernel is never run here. A Becke-Lebedev grid is a deterministic function
    # of the geometry and the level, so this rebuild reproduces the precompute's
    # quadrature exactly; the guard refuses to continue if it does not, which
    # turns "the same grid" from an assumption into a check.
    mf = dft.UKS(mol) if system.spin else dft.RKS(mol)
    if grid_level is not None:
        mf.grids.level = grid_level
    mf.grids.build()
    weights = np.asarray(mol_data["grid_weights"])
    coords = mf.grids.coords
    if (np.asarray(mf.grids.weights).shape != weights.shape
            or not np.array_equal(np.asarray(mf.grids.weights), weights)):
        raise RuntimeError(
            f"the rebuilt integration grid for pretraining system "
            f"{system.name!r} is not the one precompute_fixed_density_data "
            "used; the pretrain rows and the training features would be "
            "quadratures of different grids"
        )
    _require_sane_density(mol_data, system, reference_xc, basis, grid_level,
                          int(mol.nelectron))

    ao = np.asarray(mol_data["ao_grid_deriv"])
    dm_ab = np.asarray(mol_data["dm_pbe"])
    is_uks = (dm_ab.ndim == 3)
```

Substitutions in the retained body:

1. `mf.grids.coords[valid]` becomes `coords[valid]` in the descriptor block.
2. `np.asarray(mf.grids.weights)[valid]` in the `cols` dict becomes
   `weights[valid]`.
3. The `x_rows` tail added by the Section 3.1 plan is unchanged: it passes
   `mol`, `mf`, `ao`, `dm_ab`, which are exactly the four names bound above.
   `spin_channel_exchange_rows` reads `mf` only for `mf._numint` and
   `mf.grids`, both of which a grid-only mean field carries.

Additions in the retained body:

4. After the `fx` / `fc` clip block, add:

```python
    # LDA energy densities in the SAME convention the ratios above were formed
    # in: ``ex_safe`` / ``ec_safe`` are the denominators the clips divided by,
    # so ``e_lda * (1 + F)`` returns the parent's energy density exactly
    # wherever the +-5 clip is inactive. These are the columns the per-system
    # energy term contracts with the quadrature weights.
    e_lda_x = rho * ex_safe
    e_lda_c = rho * ec_safe
```

5. In the `cols` dict, after `"weights"`:

```python
        "e_lda_x": e_lda_x[valid],
        "e_lda_c": e_lda_c[valid],
```

- [ ] **Step 5: Re-expose the two wrappers**

Immediately after `_system_columns`, add:

```python
def _atom_columns(symbol, spin, basis, grid_level, *, polarized, descriptors,
                  density_fit=False, auxbasis=None, cusp_log_transform=True,
                  exchange_footing="total"):
    """Per-atom pretrain columns: the single-nucleus case of
    :func:`_system_columns` on the PBE density.

    Kept as a named entry point because the historical pretraining set is a list
    of free atoms and because the atomic rows are the ones every pre-existing
    ``.npz`` was built from; the geometry spelling ``"<Sym> 0 0 0"`` is the one
    those files were generated with.
    """
    return _system_columns(
        PretrainSystem(name=str(symbol), atom=f"{symbol} 0 0 0", charge=0,
                       spin=int(spin)),
        basis, grid_level, reference_xc="pbe", polarized=polarized,
        descriptors=descriptors, density_fit=density_fit, auxbasis=auxbasis,
        cusp_log_transform=cusp_log_transform,
        exchange_footing=exchange_footing)


def _molecule_columns(mol_spec, reference_xc, basis, grid_level, *, polarized,
                      descriptors, density_fit=False, auxbasis=None,
                      cusp_log_transform=True, exchange_footing="total"):
    """Pretrain columns for one molecule of the set, on the parent's density.

    ``mol_spec`` is anything :func:`normalize_system` accepts: a
    ``PretrainSystem``, the mapping form the DFS inventory and the pool JSON
    use, or a ``config.MoleculeSpec``. The basis and grid level come from the
    run's production identity, not from the spec, so every system in a file
    shares one integration identity.
    """
    return _system_columns(
        mol_spec, basis, grid_level, reference_xc=reference_xc,
        polarized=polarized, descriptors=descriptors, density_fit=density_fit,
        auxbasis=auxbasis, cusp_log_transform=cusp_log_transform,
        exchange_footing=exchange_footing)
```

- [ ] **Step 6: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/pretrain_data_gen.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_systems.py xcquinox/alec/tests/test_pretrain_data_gen.py xcquinox/alec/tests/test_pretrain_data_basis.py xcquinox/alec/tests/test_metagga_pretrain.py xcquinox/alec/tests/test_cusp_log_transform_skew.py -v > /tmp/xcq-testlogs/task02_green.log 2>&1; echo "exit=$?"
```
Expected: PASS, with every pre-existing test in those files unchanged. The
Section 3.1 footing tests still pass because `e_lda_x` / `e_lda_c` are added to
BOTH footings, so the set difference is still exactly `{"x_rows"}`.

If `reference_xc` has not yet landed on `precompute_fixed_density_data`, every
test in the file fails with `TypeError: precompute_fixed_density_data() got an
unexpected keyword argument 'reference_xc'`. That is the documented blocker in
the Dependencies section; do not route around it with a local SCF.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_pretrain_systems.py xcquinox/alec/tests/test_pretrain_data_gen.py xcquinox/alec/tests/test_pretrain_data_basis.py xcquinox/alec/tests/test_metagga_pretrain.py xcquinox/alec/tests/test_cusp_log_transform_skew.py xcquinox/alec/tests/test_cluster_datagen.py -v > /tmp/xcq-testlogs/task02_green.log 2>&1`

---

## Task 3: Per-system parent energies

> **Ordering note.** Step 1 records the regression fixture Task 4 pins the
> default generator output against. Run it BEFORE Tasks 1 and 2 if the tasks are
> executed out of order. Neither of those tasks changes a stored column VALUE
> (Task 1 adds functions; Task 2 adds two columns, two guards, and moves the
> density from a local SCF to `precompute_fixed_density_data(...,
> reference_xc="pbe")`, which is the SAME PBE SCF on the same grid), so
> recording after them still captures the pre-change numbers. That last claim is
> exactly what Task 4's fixture comparison checks: if the two routes ever
> disagreed, the regression pin would fail rather than the disagreement being
> baked in. Recording FIRST removes even that argument, so do it first.

**Files:**
- Create: `xcquinox/alec/tests/record_pretrain_data_reference.py`
- Create: `xcquinox/alec/tests/fixtures/pretrain_data_default_reference.npz` (written by the recorder)
- Modify: `xcquinox/alec/pretrain_data_gen.py` -- new function after `_system_columns`
- Test: `xcquinox/alec/tests/test_pretrain_systems.py` (append)

**Interfaces:**
- Consumes: the column dicts of Task 2; `spin_channel_exchange_rows`'s `rho` / `weights` / `Fx` / `Fx_scan` columns.
- Produces: `pretrain_data_gen._system_energy_targets(cols, x_cols) -> tuple[float, float, float, float]` returning `(e_x, e_c, e_x_scan, e_c_scan)` in Hartree.

- [ ] **Step 1: Record the default-output reference**

Create `xcquinox/alec/tests/record_pretrain_data_reference.py`:

```python
"""Record the pretrain-data generator's DEFAULT output as a test fixture.

Not a test module: pytest does not collect it (no ``test_`` prefix). Run it
once, before the pretraining-protocol change touches the generator, to freeze
what the default configuration produced:

    python xcquinox/alec/tests/record_pretrain_data_reference.py

The fixture is a two-atom (He closed shell, H open shell) file at sto-3g and
grid level 0 with descriptors and the zeta column, which exercises the RKS
branch, the UKS branch, every descriptor column and the (r_s, s, alpha) mesh in
a few hundred kilobytes. Stored compressed; the assertions are on array
contents, not on the zip container, whose headers carry write timestamps.
"""
import os
import sys
import tempfile

import numpy as np

from xcquinox.alec.pretrain_data_gen import generate_pretrain_data_npz

_ATOMS = (("He", 0), ("H", 1))
_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                        "pretrain_data_default_reference.npz")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = generate_pretrain_data_npz(
            tmp, atoms=_ATOMS, basis="sto-3g", grid_level=0, polarized=True,
            descriptors=True, density_fit=False)
        with np.load(path) as z:
            payload = {k: np.array(z[k]) for k in z.files}
    os.makedirs(os.path.dirname(_FIXTURE), exist_ok=True)
    np.savez_compressed(_FIXTURE, **payload)
    print(f"wrote {_FIXTURE}")
    for k in sorted(payload):
        print(f"  {k}: shape={payload[k].shape} dtype={payload[k].dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```bash
python xcquinox/alec/tests/record_pretrain_data_reference.py > /tmp/xcq-testlogs/task03_record.log 2>&1; echo "exit=$?"
```
Read the log with `Read`. Expected: the fixture path plus one line per key. The
key list is the pin: `rho_all`, `sigma_all`, `Fx_all`, `Fc_all`, `Fx_scan_all`,
`Fc_scan_all`, `metagga_all`, `weights_all`, `zeta_all`, `cusp_all`, `dm_all`,
`rung35_all`, `rung35ms_all`, `rho_mesh`, `sigma_mesh`, `Fx_scan_mesh`,
`Fc_scan_mesh`, `metagga_mesh`, `weights_mesh`, `zeta_mesh`. Record the row
count of `rho_all` in the Task 11 HISTORY entry.

- [ ] **Step 2: Write the failing tests**

Append to `xcquinox/alec/tests/test_pretrain_systems.py`:

```python
# ---------------------------------------------------------------------------
# Per-system parent energies: the target of the energy term
# ---------------------------------------------------------------------------

def test_system_energy_targets_are_the_row_quadrature():
    """The stored target is the quadrature over the rows the file keeps, not
    libxc's full-grid integral. That is what makes the energy term vanish
    exactly when the network reproduces the stored enhancement factors."""
    cols = _molecule_columns(_H2, "pbe", "sto-3g", 0, polarized=False,
                             descriptors=False)
    e_x, e_c, e_x_scan, e_c_scan = pdg._system_energy_targets(cols, None)
    assert e_x == pytest.approx(float(np.sum(
        cols["weights"] * cols["e_lda_x"] * (1.0 + cols["Fx"]))), rel=0,
        abs=1e-14)
    assert e_c == pytest.approx(float(np.sum(
        cols["weights"] * cols["e_lda_c"] * (1.0 + cols["Fc"]))), rel=0,
        abs=1e-14)
    assert e_x_scan == pytest.approx(float(np.sum(
        cols["weights"] * cols["e_lda_x"] * (1.0 + cols["Fx_scan"]))), rel=0,
        abs=1e-14)
    assert e_c_scan == pytest.approx(float(np.sum(
        cols["weights"] * cols["e_lda_c"] * (1.0 + cols["Fc_scan"]))), rel=0,
        abs=1e-14)
    assert e_x < 0.0 and e_c < 0.0


def test_row_quadrature_tracks_the_full_grid_libxc_integral():
    """The rows the density floor drops are exactly the rows the model clamps
    to F = 1 (models._NN_TAIL_THRESHOLD is the same 1e-10), so the network can
    move no energy there. The gap between the row quadrature and libxc's
    full-grid integral is therefore the floor of what pretraining could reach,
    and it must sit far below the certificate's tol_atom = 1.0 mHa."""
    from pyscf import dft, gto
    from xcquinox.alec.models import _NN_TAIL_THRESHOLD
    assert _NN_TAIL_THRESHOLD == pdg._RHO_FLOOR
    cols = _atom_columns("O", 2, "def2-svp", 1, polarized=True,
                         descriptors=False)
    e_x, e_c, _sx, _sc = pdg._system_energy_targets(cols, None)
    mol = gto.M(atom="O 0 0 0", basis="def2-svp", charge=0, spin=2, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.kernel()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
    dm = mf.make_rdm1()
    ra = mf._numint.eval_rho(mol, ao, dm[0], xctype="GGA", hermi=True)
    rb = mf._numint.eval_rho(mol, ao, dm[1], xctype="GGA", hermi=True)
    rho_uks = np.stack([ra, rb], axis=0)
    w = np.asarray(mf.grids.weights)
    ref_x = float(np.sum(w * (ra[0] + rb[0]) * np.asarray(
        mf._numint.eval_xc("PBE,", rho_uks, spin=1)[0])))
    ref_c = float(np.sum(w * (ra[0] + rb[0]) * np.asarray(
        mf._numint.eval_xc(",PBE", rho_uks, spin=1)[0])))
    assert abs(e_x - ref_x) < 1e-6, (e_x, ref_x)
    assert abs(e_c - ref_c) < 1e-6, (e_c, ref_c)


def test_per_channel_and_total_exchange_energies_agree():
    """The Oliver-Perdew relation as a number: the exchange energy read off the
    per-channel doubled-density rows must equal the one read off the
    total-density spin-resolved rows. Both are E_x^PBE of the same density, so
    a disagreement means one of the two footings is not the parent's exchange.
    (Oliver and Perdew, Phys. Rev. A 20, 397 (1979).)"""
    cols = _atom_columns("O", 2, "def2-svp", 1, polarized=True,
                         descriptors=False,
                         exchange_footing="spin_channel")
    e_total, _c, _sx, _sc = pdg._system_energy_targets(cols, None)
    e_channel, _c2, _sx2, _sc2 = pdg._system_energy_targets(
        cols, cols["x_rows"])
    assert abs(e_channel - e_total) < 1e-6, (e_channel, e_total)


def test_system_energy_targets_use_the_channel_rows_when_given():
    cols = _atom_columns("O", 2, "def2-svp", 1, polarized=True,
                         descriptors=False,
                         exchange_footing="spin_channel")
    x = cols["x_rows"]
    e_x, _c, e_x_scan, _sc = pdg._system_energy_targets(cols, x)
    e_lda = x["rho"] * (pdg._LDA_X_C * np.cbrt(x["rho"]))
    assert e_x == pytest.approx(float(np.sum(
        x["weights"] * e_lda * (1.0 + x["Fx"]))), rel=0, abs=1e-14)
    assert e_x_scan == pytest.approx(float(np.sum(
        x["weights"] * e_lda * (1.0 + x["Fx_scan"]))), rel=0, abs=1e-14)
```

- [ ] **Step 3: Run and confirm it fails**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_systems.py -k "energy_target or quadrature or per_channel_and_total" -v > /tmp/xcq-testlogs/task03_red.log 2>&1; echo "exit=$?"
```
Expected: `AttributeError: module 'xcquinox.alec.pretrain_data_gen' has no attribute '_system_energy_targets'`.

- [ ] **Step 4: Add the energy targets**

Insert into `xcquinox/alec/pretrain_data_gen.py` immediately after
`_molecule_columns`:

```python
def _system_energy_targets(cols, x_cols):
    """Per-system parent energies in Hartree: ``(e_x, e_c, e_x_scan, e_c_scan)``.

    Each is the ROW QUADRATURE over the rows this file stores,
    ``sum_i w_i e_LDA_i (1 + F_i)``, not libxc's full-grid integral. That
    choice is what makes the per-system energy term measure the fit and nothing
    else: the network's own energy on the same rows is
    ``sum_i w_i e_LDA_i F^NN_i``, so the residual vanishes exactly when the
    network reproduces the stored enhancement factors. The two integrals differ
    only by the rows the density floor drops and by the +-5 clip on the stored
    ratio, and the floor is the model's own tail threshold
    (``models._NN_TAIL_THRESHOLD`` = 1e-10), below which the model clamps F to 1
    and the network cannot move the energy at all -- the dropped rows are
    exactly the rows pretraining could not have fitted. Measured on the O atom
    at def2-svp / grid level 1 the gap to libxc is below 1e-6 Ha, three orders
    of magnitude under the certificate's tol_atom = 1.0 mHa.

    ``x_cols`` is the per-channel exchange block of
    :func:`spin_channel_exchange_rows`, or ``None`` when the exchange rows ARE
    the total-density rows (a closed-shell system, or the ``"total"`` footing).
    That block carries no LDA column of its own because its denominator is the
    analytic unpolarized LDA at the DOUBLED density, a function of the stored
    ``rho`` alone.
    """
    if x_cols is None:
        e_x = float(np.sum(cols["weights"] * cols["e_lda_x"]
                           * (1.0 + cols["Fx"])))
        e_x_scan = float(np.sum(cols["weights"] * cols["e_lda_x"]
                                * (1.0 + cols["Fx_scan"])))
    else:
        e_lda_x = x_cols["rho"] * (_LDA_X_C * np.cbrt(x_cols["rho"]))
        e_x = float(np.sum(x_cols["weights"] * e_lda_x
                           * (1.0 + x_cols["Fx"])))
        e_x_scan = float(np.sum(x_cols["weights"] * e_lda_x
                                * (1.0 + x_cols["Fx_scan"])))
    e_c = float(np.sum(cols["weights"] * cols["e_lda_c"] * (1.0 + cols["Fc"])))
    e_c_scan = float(np.sum(cols["weights"] * cols["e_lda_c"]
                            * (1.0 + cols["Fc_scan"])))
    return e_x, e_c, e_x_scan, e_c_scan
```

- [ ] **Step 5: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/pretrain_data_gen.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_systems.py -v > /tmp/xcq-testlogs/task03_green.log 2>&1; echo "exit=$?"
```
Expected: PASS. Record the measured `|e_x - ref_x|` and `|e_channel - e_total|`
from the log for the Task 11 HISTORY entry (temporarily tighten the two bounds
to `0.0` to read the numbers out of the failure messages, then restore them).

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_pretrain_systems.py -v > /tmp/xcq-testlogs/task03_green.log 2>&1`

---

## Task 4: The two-block `.npz` schema

**Files:**
- Modify: `xcquinox/alec/pretrain_data_gen.py:431-520` (`generate_pretrain_data_npz`)
- Test: `xcquinox/alec/tests/test_pretrain_schema.py` (create)

**Interfaces:**
- Consumes: `resolve_pretrain_systems`, `pretrain_data_filename` (Task 1); `_system_columns` (Task 2); `_system_energy_targets` (Task 3).
- Produces: `generate_pretrain_data_npz(out_dir, *, atoms=None, basis, grid_level, polarized=True, descriptors=True, density_fit=False, auxbasis=None, cusp_log_transform=True, progress=False, dfs_set=False, pool_atoms=False, reference_xc="pbe", exchange_footing="total", mesh_fraction=MESH_WEIGHT_FRACTION, systems=None) -> str`, and the schema:
  - correlation / total-density block, unchanged names plus `system_all`, `e_lda_x_all`, `e_lda_c_all`
  - exchange block, present only under the `spin_channel` footing: `rho_x`, `sigma_x`, `Fx_x`, `Fx_scan_x`, `metagga_x`, `weights_x`, `system_x`, and with descriptors `cusp_x`, `dm_x`, `rung35_x`, `rung35ms_x`
  - system table: `e_x_parent_sys`, `e_c_parent_sys`, `e_x_parent_scan_sys`, `e_c_parent_scan_sys`, `system_natoms`
  - `mesh_weight_fraction` (0-d float)

- [ ] **Step 1: Write the failing tests**

Create `xcquinox/alec/tests/test_pretrain_schema.py`:

```python
"""The pretrain-data .npz schema after the pretraining-protocol change.

Two row blocks, because per-channel exchange rows and total-density correlation
rows are no longer the same rows (spec Section 3.2): the historical ``*_all``
block is the correlation / total-density block, and a ``*_x`` block appears
under the ``spin_channel`` footing. A per-row ``system_*`` index and a
per-system energy table carry the energy term of Section 6 deviation 3.
"""
import os

import numpy as np
import pytest

import xcquinox.alec.pretrain_data_gen as pdg


_TINY = (("He", 0), ("H", 1))
_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                        "pretrain_data_default_reference.npz")


def _gen(tmp_path, **kw):
    kw.setdefault("atoms", _TINY)
    kw.setdefault("basis", "sto-3g")
    kw.setdefault("grid_level", 0)
    kw.setdefault("polarized", True)
    kw.setdefault("descriptors", True)
    path = pdg.generate_pretrain_data_npz(str(tmp_path), **kw)
    with np.load(path) as z:
        return path, {k: np.array(z[k]) for k in z.files}


# ---------------------------------------------------------------------------
# The regression pin: the default configuration is unchanged
# ---------------------------------------------------------------------------

def test_default_output_matches_the_recorded_reference(tmp_path):
    """Every column the generator wrote before the pretraining-protocol change
    is bit-identical at the default configuration, so a YAML already in flight
    trains on the same numbers. New keys may appear; old ones may not move.
    (The .npz CONTAINER is a zip whose headers carry write timestamps, so the
    pin is on array contents, not on the file's bytes.)"""
    ref = dict(np.load(_FIXTURE))
    _path, got = _gen(tmp_path)
    missing = sorted(set(ref) - set(got))
    assert not missing, f"the default output lost {missing}"
    for key in sorted(ref):
        assert got[key].dtype == ref[key].dtype, key
        assert got[key].shape == ref[key].shape, key
        np.testing.assert_array_equal(got[key], ref[key], err_msg=key)


def test_default_output_adds_only_the_documented_new_keys(tmp_path):
    ref = dict(np.load(_FIXTURE))
    _path, got = _gen(tmp_path)
    assert sorted(set(got) - set(ref)) == sorted([
        "e_c_parent_scan_sys", "e_c_parent_sys", "e_lda_c_all", "e_lda_x_all",
        "e_x_parent_scan_sys", "e_x_parent_sys", "mesh_weight_fraction",
        "system_all", "system_natoms",
    ])


def test_default_output_writes_no_exchange_block(tmp_path):
    _path, got = _gen(tmp_path)
    assert not [k for k in got if k.endswith("_x")]


# ---------------------------------------------------------------------------
# The system index and the energy table
# ---------------------------------------------------------------------------

def test_system_index_partitions_the_rows_in_declaration_order(tmp_path):
    _path, got = _gen(tmp_path)
    seg = got["system_all"]
    assert seg.dtype == np.int32
    assert seg.shape == got["rho_all"].shape
    assert sorted(set(seg.tolist())) == [0, 1]
    # Rows are emitted system by system, so the index is non-decreasing.
    assert np.all(np.diff(seg) >= 0)
    assert got["system_natoms"].tolist() == [1, 1]


def test_energy_table_is_the_per_system_row_quadrature(tmp_path):
    _path, got = _gen(tmp_path)
    for s in (0, 1):
        rows = got["system_all"] == s
        expect_x = float(np.sum(got["weights_all"][rows]
                                * got["e_lda_x_all"][rows]
                                * (1.0 + got["Fx_all"][rows])))
        expect_c = float(np.sum(got["weights_all"][rows]
                                * got["e_lda_c_all"][rows]
                                * (1.0 + got["Fc_all"][rows])))
        assert got["e_x_parent_sys"][s] == pytest.approx(expect_x, rel=0,
                                                         abs=1e-12)
        assert got["e_c_parent_sys"][s] == pytest.approx(expect_c, rel=0,
                                                         abs=1e-12)


def test_polarized_correlation_baseline_matches_the_model(tmp_path):
    """e_lda_c_all / rho_all is the libxc PW92 baseline the Fc ratio divided by.
    The production correlation path multiplies the network's F_c by
    utils.pw92c_polarized_scalar at the same zeta, so the two must be the same
    function or the pretraining energy target is not the production energy."""
    import jax.numpy as jnp
    from xcquinox.utils import pw92c_polarized_scalar
    _path, got = _gen(tmp_path)
    rho = got["rho_all"]
    zeta = got["zeta_all"]
    half = 0.5 * (1.0 + zeta)
    ours = np.asarray(pw92c_polarized_scalar(jnp.asarray(rho * half),
                                             jnp.asarray(rho * (1.0 - half))))
    np.testing.assert_allclose(got["e_lda_c_all"] / rho, ours, rtol=1e-10,
                               atol=1e-12)


# ---------------------------------------------------------------------------
# The exchange block
# ---------------------------------------------------------------------------

def test_spin_channel_footing_writes_an_exchange_block(tmp_path):
    _path, got = _gen(tmp_path, exchange_footing="spin_channel")
    n_x = got["rho_x"].shape[0]
    for key in ("sigma_x", "Fx_x", "Fx_scan_x", "weights_x", "system_x"):
        assert got[key].shape == (n_x,), key
    assert got["metagga_x"].shape == (n_x, 1)
    assert got["cusp_x"].shape == (n_x, 2)
    assert got["rung35_x"].shape == (n_x, 2)
    assert got["rung35ms_x"].shape == (n_x, 6)
    assert sorted(set(got["system_x"].tolist())) == [0, 1]
    # He is closed-shell: its exchange rows ARE its total-density rows. H is a
    # one-electron open shell: only the alpha channel survives the floor, so
    # its exchange block is the alpha channel alone.
    assert int(np.sum(got["system_x"] == 0)) == int(
        np.sum(got["system_all"] == 0))


def test_closed_shell_exchange_rows_are_the_total_density_rows(tmp_path):
    _path, got = _gen(tmp_path, exchange_footing="spin_channel")
    he_x = got["system_x"] == 0
    he_a = got["system_all"] == 0
    np.testing.assert_array_equal(got["rho_x"][he_x], got["rho_all"][he_a])
    np.testing.assert_array_equal(got["Fx_x"][he_x], got["Fx_all"][he_a])
    np.testing.assert_array_equal(got["weights_x"][he_x],
                                  got["weights_all"][he_a])


def test_exchange_energy_table_uses_the_exchange_block(tmp_path):
    _path, got = _gen(tmp_path, exchange_footing="spin_channel")
    for s in (0, 1):
        rows = got["system_x"] == s
        rho = got["rho_x"][rows]
        e_lda = rho * (pdg._LDA_X_C * np.cbrt(rho))
        expect = float(np.sum(got["weights_x"][rows] * e_lda
                              * (1.0 + got["Fx_x"][rows])))
        assert got["e_x_parent_sys"][s] == pytest.approx(expect, rel=0,
                                                         abs=1e-12)


# ---------------------------------------------------------------------------
# Filename, reference density, mesh fraction
# ---------------------------------------------------------------------------

def test_scan_reference_writes_its_own_file(tmp_path):
    path, got = _gen(tmp_path, reference_xc="scan", grid_level=1)
    assert os.path.basename(path) == "pretrain_data_polarized_scan.npz"
    assert got["rho_all"].shape[0] > 0


def test_mesh_fraction_is_stored_and_scales_the_mesh_weights(tmp_path):
    _path, base = _gen(tmp_path)
    assert float(base["mesh_weight_fraction"]) == pdg.MESH_WEIGHT_FRACTION
    other = tmp_path / "half"
    other.mkdir()
    _p2, got = _gen(other, mesh_fraction=0.5)
    assert float(got["mesh_weight_fraction"]) == 0.5
    share = float(got["weights_mesh"].sum()
                  / (got["weights_mesh"].sum() + got["weights_all"].sum()))
    assert share == pytest.approx(0.5, rel=1e-12)


def test_systems_argument_overrides_the_composition_flags(tmp_path):
    """ensure_pretrain_data resolves the set once and hands the SAME tuple to
    the currency check and to the generator, so the two can never disagree."""
    sysm = (pdg.PretrainSystem("He", "He 0 0 0", 0, 0),)
    _path, got = _gen(tmp_path, systems=sysm, atoms=(("H", 1),))
    assert got["system_natoms"].tolist() == [1]
    assert sorted(set(got["system_all"].tolist())) == [0]
```

- [ ] **Step 2: Run and confirm it fails**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_schema.py -v > /tmp/xcq-testlogs/task04_red.log 2>&1; echo "exit=$?"
```
Expected: `TypeError: generate_pretrain_data_npz() got an unexpected keyword
argument 'exchange_footing'` and `KeyError: 'system_all'`. The two regression
tests (`test_default_output_matches_the_recorded_reference`,
`test_default_output_adds_only_the_documented_new_keys`) fail only on the
missing new keys, which is the RED state that proves the pin is live.

- [ ] **Step 3: Rewrite the generator**

Replace `xcquinox/alec/pretrain_data_gen.py:431-520`
(`generate_pretrain_data_npz`) with:

```python
def generate_pretrain_data_npz(out_dir, *, atoms=None, basis=DEFAULT_BASIS,
                               grid_level=DEFAULT_GRID_LEVEL,
                               polarized=True, descriptors=True,
                               density_fit=False, auxbasis=None,
                               cusp_log_transform=True, progress=False,
                               dfs_set=False, pool_atoms=False,
                               reference_xc="pbe",
                               exchange_footing="total",
                               mesh_fraction=MESH_WEIGHT_FRACTION,
                               systems=None):
    """Generate the pretrain-data ``.npz`` in ``out_dir`` and return its path.

    ``polarized=True`` writes the zeta-carrying file; ``reference_xc="scan"`` writes
    the SCAN-density file under its own name (:func:`pretrain_data_filename`).
    The set is ``resolve_pretrain_systems(atoms=..., dfs_set=..., pool_atoms=...,
    reference_xc=...)`` unless ``systems`` supplies an already-resolved tuple,
    which
    is how :func:`ensure_pretrain_data` guarantees the currency check and the
    generation see the same list.

    TWO ROW BLOCKS. The historical ``*_all`` block is the total-density block:
    it carries the correlation rows always, and the exchange rows too under the
    default ``"total"`` footing. Under ``exchange_footing="spin_channel"`` a
    second ``*_x`` block carries the exchange rows on the exact-spin-scaling
    footing -- per channel at ``(2 rho_sigma, 4 sigma_sigma_sigma, features of
    diag(P_sigma, P_sigma))`` for an open shell (Oliver and Perdew, Phys. Rev. A
    20, 397 (1979)), and the total-density rows for a closed shell, where
    rho_a = rho_b makes the two the same rows. The two blocks have different
    lengths on an open shell, which is why they cannot share one set of names.

    THE SYSTEM TABLE. ``system_all`` / ``system_x`` index each row into the
    system it came from, and ``e_{x,c}_parent[_scan]_sys`` hold that system's
    parent energy in Hartree as the row quadrature (see
    :func:`_system_energy_targets`). Together they are the per-system energy
    term of the pretraining objective: a network can no longer lower the
    point-wise residual while missing a system's energy.

    A sidecar ``<npz>.manifest.json`` records the identity the data was built
    at so :func:`pretrain_data_is_current` can force a regeneration.
    """
    from xcquinox.alec.data import clear_precompute_cache
    systems = (tuple(normalize_system(s) for s in systems)
               if systems is not None
               else resolve_pretrain_systems(atoms=atoms, dfs_set=dfs_set,
                                             pool_atoms=pool_atoms,
                                             reference_xc=reference_xc))
    if not systems:
        raise ValueError(
            "the pretraining set is empty: pass atoms=..., or turn on "
            "dfs_set / pool_atoms."
        )
    per_system = []
    for _i, system in enumerate(systems, 1):
        if progress:
            print(f"  pretrain data: system {_i}/{len(systems)} "
                  f"{system.name} ({reference_xc.upper()} density @ "
                  f"{basis}) ...",
                  flush=True)
        per_system.append(_system_columns(
            system, basis, grid_level, reference_xc=reference_xc,
            polarized=polarized,
            descriptors=descriptors, density_fit=density_fit,
            auxbasis=auxbasis, cusp_log_transform=cusp_log_transform,
            exchange_footing=exchange_footing))
        # precompute_fixed_density_data memoizes its MoleculeData in a
        # process-level dict, and each one holds the (4, n_grid, n_ao) AO
        # derivative tensor -- of order 0.8 GB for a ten-nucleus molecule at
        # 6-311++G(3df,2pd) and grid level 3. Retaining one per system would
        # exhaust the node long before the set is generated, and nothing here
        # revisits a system, so the cache is dropped as each system's columns
        # are extracted.
        clear_precompute_cache()
    save_kwargs = {
        "rho_all": np.concatenate([c["rho"] for c in per_system]),
        "sigma_all": np.concatenate([c["sigma"] for c in per_system]),
        "Fx_all": np.concatenate([c["Fx"] for c in per_system]),
        "Fc_all": np.concatenate([c["Fc"] for c in per_system]),
        # SCAN (meta-GGA) targets + iso-orbital alpha column, always present so
        # meta_gga archs pretrain to SCAN (pretrain.py routes the target by the
        # arch's meta_gga flag); GGA archs ignore these keys.
        "Fx_scan_all": np.concatenate([c["Fx_scan"] for c in per_system]),
        "Fc_scan_all": np.concatenate([c["Fc_scan"] for c in per_system]),
        "metagga_all": np.concatenate([c["metagga"] for c in per_system]),
        "weights_all": np.concatenate([c["weights"] for c in per_system]),
    }
    # Per-row system index and the LDA energy densities that turn a row of
    # enhancement factors into Hartrees: the two ingredients of the per-system
    # energy term. Written unconditionally (4 + 16 bytes per row) so any file,
    # including the historical atoms-only default, can carry the term.
    save_kwargs["system_all"] = np.concatenate(
        [np.full(c["rho"].shape[0], i, dtype=np.int32)
         for i, c in enumerate(per_system)])
    save_kwargs["e_lda_x_all"] = np.concatenate(
        [c["e_lda_x"] for c in per_system])
    save_kwargs["e_lda_c_all"] = np.concatenate(
        [c["e_lda_c"] for c in per_system])
    if exchange_footing == "spin_channel":
        # One exchange block over EVERY system: the per-channel rows of an open
        # shell, and the total-density rows of a closed shell, which ARE the
        # per-channel rows there.
        x_blocks = [c["x_rows"] if c.get("x_rows") is not None else c
                    for c in per_system]
        save_kwargs.update({
            "rho_x": np.concatenate([b["rho"] for b in x_blocks]),
            "sigma_x": np.concatenate([b["sigma"] for b in x_blocks]),
            "Fx_x": np.concatenate([b["Fx"] for b in x_blocks]),
            "Fx_scan_x": np.concatenate([b["Fx_scan"] for b in x_blocks]),
            "metagga_x": np.concatenate([b["metagga"] for b in x_blocks]),
            "weights_x": np.concatenate([b["weights"] for b in x_blocks]),
            "system_x": np.concatenate(
                [np.full(b["rho"].shape[0], i, dtype=np.int32)
                 for i, b in enumerate(x_blocks)]),
        })
        if descriptors:
            for _key in ("cusp", "dm", "rung35", "rung35ms"):
                save_kwargs[f"{_key}_x"] = np.concatenate(
                    [b[_key] for b in x_blocks])
    # Per-system parent energies, Hartree. Both the PBE and the SCAN targets,
    # for the same reason the Fx / Fx_scan columns are both present: the file's
    # density is the parent's, the target is the rung's.
    _targets = [
        _system_energy_targets(
            c, c.get("x_rows") if exchange_footing == "spin_channel" else None)
        for c in per_system
    ]
    save_kwargs.update({
        "e_x_parent_sys": np.array([t[0] for t in _targets], dtype=np.float64),
        "e_c_parent_sys": np.array([t[1] for t in _targets], dtype=np.float64),
        "e_x_parent_scan_sys": np.array([t[2] for t in _targets],
                                        dtype=np.float64),
        "e_c_parent_scan_sys": np.array([t[3] for t in _targets],
                                        dtype=np.float64),
        # Nuclei per system: the validation split holds out MOLECULES only.
        "system_natoms": np.array([_n_atoms(s.atom) for s in systems],
                                  dtype=np.int32),
    })
    # (s, alpha) parameter-space mesh, stored under SEPARATE *_mesh keys so the
    # atomic arrays every GGA arch reads stay byte-identical. pretrain.py
    # concatenates these ONLY for a meta_gga arch whose descriptor set the mesh
    # can actually define (see _mesh_columns).
    mesh = _mesh_columns()
    _w_atom = float(save_kwargs["weights_all"].sum())
    _n_mesh = mesh["rho"].shape[0]
    # Rescale the (weightless) mesh rows to a stated share of the total
    # integration weight: w_mesh_total / (w_atom + w_mesh_total) = FRACTION.
    _w_mesh_total = _w_atom * mesh_fraction / (1.0 - mesh_fraction)
    save_kwargs.update({
        "rho_mesh": mesh["rho"],
        "sigma_mesh": mesh["sigma"],
        "Fx_scan_mesh": mesh["Fx_scan"],
        "Fc_scan_mesh": mesh["Fc_scan"],
        "metagga_mesh": mesh["metagga"],
        "weights_mesh": np.full(_n_mesh, _w_mesh_total / _n_mesh),
        # Stored beside the weights it produced so the loss reads the share the
        # DATA was built at rather than a constant that may have moved.
        "mesh_weight_fraction": np.asarray(float(mesh_fraction)),
    })
    if polarized:
        save_kwargs["zeta_mesh"] = mesh["zeta"]
    if polarized:
        save_kwargs["zeta_all"] = np.concatenate(
            [c["zeta"] for c in per_system])
    if descriptors:
        save_kwargs["cusp_all"] = np.concatenate(
            [c["cusp"] for c in per_system])
        save_kwargs["dm_all"] = np.concatenate([c["dm"] for c in per_system])
        save_kwargs["rung35_all"] = np.concatenate(
            [c["rung35"] for c in per_system])
        save_kwargs["rung35ms_all"] = np.concatenate(
            [c["rung35ms"] for c in per_system])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            pretrain_data_filename(polarized, reference_xc))
    # ATOMIC write (tmp + os.replace): the data dir is SHARED across sweep
    # runs, and two concurrently submitted runs whose datagen stages both see
    # a stale file would otherwise race a plain in-place np.savez -- a torn
    # zip that fails every reader. With the rename, concurrent regenerations
    # merely duplicate compute (last writer wins, both logically identical)
    # and a reader always sees a complete file. The tmp name is pid-tagged so
    # two writers do not collide on the tmp path either.
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    np.savez(tmp_path, **save_kwargs)
    # np.savez appends .npz to a name without it: normalize.
    if not tmp_path.endswith(".npz") and os.path.isfile(tmp_path + ".npz"):
        tmp_path = tmp_path + ".npz"
    os.replace(tmp_path, out_path)
    _write_pretrain_manifest(
        out_path, basis=basis, grid_level=grid_level, density_fit=density_fit,
        auxbasis=_effective_auxbasis(basis, density_fit, auxbasis),
        atoms=tuple((s.name, s.spin) for s in systems), systems=systems,
        reference_xc=reference_xc, exchange_footing=exchange_footing,
        mesh_fraction=mesh_fraction)
    return out_path
```

The manifest signature this calls is added in Task 5; until then the call
raises `TypeError` on the new keywords, which is why Task 5 follows
immediately.

- [ ] **Step 4: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/pretrain_data_gen.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_schema.py xcquinox/alec/tests/test_pretrain_data_gen.py xcquinox/alec/tests/test_pretrain_mesh.py -v > /tmp/xcq-testlogs/task04_green.log 2>&1; echo "exit=$?"
```
Expected after Task 5's manifest change: PASS. Run this command again at the end
of Task 5; a `TypeError` on `_write_pretrain_manifest` here is the expected
intermediate state and is resolved there.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_pretrain_schema.py xcquinox/alec/tests/test_pretrain_data_gen.py xcquinox/alec/tests/test_pretrain_mesh.py -v > /tmp/xcq-testlogs/task04_green.log 2>&1`

---

## Task 5: The data identity -- manifest, currency check, ensure

**Files:**
- Modify: `xcquinox/alec/pretrain_data_gen.py:300-327` (`_write_pretrain_manifest`), `:339-397` (`pretrain_data_is_current`), `:408-428` (`ensure_pretrain_data`)
- Test: `xcquinox/alec/tests/test_pretrain_schema.py` (append)

**Interfaces:**
- Consumes: `resolve_pretrain_systems`, `pretrain_data_filename` (Task 1).
- Produces:
  - `_write_pretrain_manifest(npz_path, *, basis, grid_level, density_fit, auxbasis=None, atoms=DEFAULT_PRETRAIN_ATOMS, systems=None, reference_xc="pbe", exchange_footing="total", mesh_fraction=MESH_WEIGHT_FRACTION) -> None`
  - `pretrain_data_is_current(npz_path, *, basis, grid_level, auxbasis=None, atoms=DEFAULT_PRETRAIN_ATOMS, systems=None, reference_xc="pbe", exchange_footing="total", mesh_fraction=MESH_WEIGHT_FRACTION) -> bool`
  - `ensure_pretrain_data(data_dir, *, atoms=None, basis, grid_level, polarized=True, descriptors=True, density_fit=False, auxbasis=None, cusp_log_transform=True, progress=False, dfs_set=False, pool_atoms=False, reference_xc="pbe", exchange_footing="total", mesh_fraction=MESH_WEIGHT_FRACTION) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_pretrain_schema.py`:

```python
# ---------------------------------------------------------------------------
# Data identity: what forces a regeneration
# ---------------------------------------------------------------------------

def test_manifest_records_the_new_identity(tmp_path):
    path, _got = _gen(tmp_path, exchange_footing="spin_channel",
                      mesh_fraction=0.4)
    meta = pdg.read_pretrain_manifest(path)
    assert meta["reference_xc"] == "pbe"
    assert meta["exchange_footing"] == "spin_channel"
    assert meta["mesh"]["weight_fraction"] == 0.4
    assert [row[0] for row in meta["systems"]] == ["He", "H"]
    assert meta["systems"][1] == ["H", "H 0 0 0", 0, 1]
    # The legacy projection stays, so a manifest reader written before the set
    # became a system list still sees an atom list.
    assert meta["atoms"] == [["He", 0], ["H", 1]]


def test_currency_check_legacy_manifest_stays_current(tmp_path):
    """A file written before this change carries no reference_xc / footing /
    systems
    keys. They must read as the historical values so an existing data
    directory is not invalidated wholesale."""
    p = tmp_path / "pretrain_data.npz"
    np.savez(p, rho_all=np.ones(3), sigma_all=np.ones(3), Fx_all=np.ones(3),
             Fc_all=np.ones(3), weights_all=np.ones(3),
             metagga_all=np.ones((3, 1)), Fx_scan_all=np.ones(3),
             Fc_scan_all=np.ones(3), rho_mesh=np.ones(4),
             sigma_mesh=np.ones(4), Fx_scan_mesh=np.ones(4),
             Fc_scan_mesh=np.ones(4), metagga_mesh=np.ones((4, 1)),
             weights_mesh=np.ones(4))
    import json
    with open(str(p) + ".manifest.json", "w") as f:
        json.dump({"basis": "def2-svp", "grid_level": 1, "density_fit": False,
                   "auxbasis": None, "atoms": [["H", 1]]}, f)
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1, atoms=[("H", 1)]) is True


def test_currency_check_footing_is_part_of_the_identity(tmp_path):
    path, _got = _gen(tmp_path)
    sysm = pdg.resolve_pretrain_systems(atoms=_TINY)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=sysm) is True
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=sysm,
        exchange_footing="spin_channel") is False


def test_currency_check_reference_density_is_part_of_the_identity(tmp_path):
    path, _got = _gen(tmp_path)
    sysm = pdg.resolve_pretrain_systems(atoms=_TINY)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=sysm,
        reference_xc="scan") is False


def test_currency_check_mesh_fraction_is_part_of_the_identity(tmp_path):
    path, _got = _gen(tmp_path)
    sysm = pdg.resolve_pretrain_systems(atoms=_TINY)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=sysm,
        mesh_fraction=0.4) is False


def test_currency_check_system_list_is_part_of_the_identity(tmp_path):
    path, _got = _gen(tmp_path)
    other = pdg.resolve_pretrain_systems(atoms=(("He", 0),))
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=other) is False


def test_currency_check_rejects_a_manifest_without_a_system_list(tmp_path):
    """A file written before the set became a system list cannot be shown to
    hold the requested systems, so it regenerates."""
    import json
    path, _got = _gen(tmp_path)
    meta = pdg.read_pretrain_manifest(path)
    meta.pop("systems")
    with open(str(path) + ".manifest.json", "w") as f:
        json.dump(meta, f)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0,
        systems=pdg.resolve_pretrain_systems(atoms=_TINY)) is False


def test_ensure_resolves_the_set_once(monkeypatch, tmp_path):
    """The currency check and the generation must see the SAME resolved tuple:
    resolving twice would let a non-deterministic inventory silently regenerate
    on every call."""
    seen = []

    def _fake_generate(out_dir, **kw):
        seen.append(kw["systems"])
        p = os.path.join(out_dir, pdg.pretrain_data_filename(
            kw["polarized"], kw["reference_xc"]))
        np.savez(p, rho_all=np.ones(1))
        return p

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", _fake_generate)
    pdg.ensure_pretrain_data(str(tmp_path), basis="sto-3g", grid_level=0,
                             pool_atoms=True)
    assert len(seen) == 1
    assert len(seen[0]) == 14


def test_ensure_uses_the_reference_specific_filename(monkeypatch, tmp_path):
    paths = []

    def _fake_generate(out_dir, **kw):
        p = os.path.join(out_dir, pdg.pretrain_data_filename(
            kw["polarized"], kw["reference_xc"]))
        paths.append(p)
        np.savez(p, rho_all=np.ones(1))
        return p

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", _fake_generate)
    pdg.ensure_pretrain_data(str(tmp_path), basis="sto-3g", grid_level=0,
                             reference_xc="scan", polarized=True)
    assert os.path.basename(paths[0]) == "pretrain_data_polarized_scan.npz"


def test_ensure_is_idempotent_at_the_new_identity(tmp_path):
    p1 = pdg.ensure_pretrain_data(str(tmp_path), atoms=_TINY, basis="sto-3g",
                                  grid_level=0, polarized=True,
                                  descriptors=True,
                                  exchange_footing="spin_channel")
    mtime = os.path.getmtime(p1)
    p2 = pdg.ensure_pretrain_data(str(tmp_path), atoms=_TINY, basis="sto-3g",
                                  grid_level=0, polarized=True,
                                  descriptors=True,
                                  exchange_footing="spin_channel")
    assert p1 == p2
    assert os.path.getmtime(p2) == mtime
```

- [ ] **Step 2: Run and confirm it fails**

```bash
python -m pytest xcquinox/alec/tests/test_pretrain_schema.py -k "manifest or currency or ensure" -v > /tmp/xcq-testlogs/task05_red.log 2>&1; echo "exit=$?"
```
Expected: `TypeError: _write_pretrain_manifest() got an unexpected keyword
argument 'systems'`.

- [ ] **Step 3: Extend the manifest writer**

Replace the signature and the `meta` dict of
`xcquinox/alec/pretrain_data_gen.py:300-321` with:

```python
def _write_pretrain_manifest(npz_path, *, basis, grid_level, density_fit,
                             auxbasis=None, atoms=DEFAULT_PRETRAIN_ATOMS,
                             systems=None, reference_xc="pbe",
                             exchange_footing="total",
                             mesh_fraction=MESH_WEIGHT_FRACTION):
    """Record the identity a pretrain ``.npz`` was built at.

    Written as a sidecar so the ``.npz`` array payload stays byte-identical to
    the pre-manifest format (legacy loaders that ignore the sidecar are
    unaffected). Every key here is something a change of which changes the
    stored VALUES, so :func:`pretrain_data_is_current` treats all of them as
    the file's identity:

    - ``basis`` / ``grid_level`` / ``auxbasis``: the integration identity.
    - ``atoms``: the legacy projection ``[[name, 2S], ...]`` of the set, kept
      so a reader written before the set became a system list still resolves.
    - ``systems``: the set itself, ``[[name, geometry, charge, 2S], ...]``. A
      geometry change is a different physical system and must force a
      regeneration, which the atom-name projection cannot see.
    - ``reference_xc``: the functional whose SELF-CONSISTENT density the rows
      sit on (PBE for the GGA rung, SCAN for the meta-GGA rung).
    - ``exchange_footing``: ``"total"`` or ``"spin_channel"``. The open-shell
      exchange rows are a different row set under the two, so a footing change
      is a data change.
    - ``mesh.weight_fraction``: the share of the total integration weight the
      synthetic mesh carries. Recorded because it is a deliberate choice, not
      an emergent property of a quadrature: mesh rows carry no physical grid
      weight, so their pull on the pretrain loss is set here.
    """
    meta = {"basis": basis, "grid_level": int(grid_level),
            "density_fit": bool(density_fit), "auxbasis": auxbasis,
            "atoms": [[str(s), int(sp)] for s, sp in atoms],
            "systems": (None if systems is None else
                        [[str(s.name), str(s.atom), int(s.charge),
                          int(s.spin)] for s in systems]),
            "reference_xc": str(reference_xc),
            "exchange_footing": str(exchange_footing),
            "mesh": {"rs": list(MESH_RS), "s": list(MESH_S),
                     "alpha": list(MESH_ALPHA),
                     "weight_fraction": float(mesh_fraction)}}
```

The atomic write below it is unchanged.

- [ ] **Step 4: Extend the currency check**

In `pretrain_data_is_current`, replace the signature and the `manifest_ok`
block (currently lines 339-366) with:

```python
def pretrain_data_is_current(npz_path, *, basis, grid_level, auxbasis=None,
                             atoms=DEFAULT_PRETRAIN_ATOMS, systems=None,
                             reference_xc="pbe", exchange_footing="total",
                             mesh_fraction=MESH_WEIGHT_FRACTION):
    """True iff ``npz_path`` exists AND its manifest matches the requested
    identity.

    A missing file OR a missing/mismatched manifest returns ``False`` so the
    harness regenerates rather than silently reusing data built at a different
    identity. Every keyword defaults to the value the historical generator
    used, and every manifest key absent from a legacy file reads as that same
    value, so a data directory written before this change stays current and
    only a real change forces the regeneration.

    ``systems`` is the resolved pretraining set. When given it REPLACES the
    ``atoms`` comparison, and a manifest without a ``systems`` list cannot be
    shown to hold them, so it regenerates.
    """
    if not os.path.isfile(npz_path):
        return False
    meta = read_pretrain_manifest(npz_path)
    if meta is None:
        return False
    if systems is None:
        want_atoms = [[str(s), int(sp)] for s, sp in atoms]
        have_atoms = meta.get(
            "atoms", [[str(s), int(sp)] for s, sp in DEFAULT_PRETRAIN_ATOMS])
        composition_ok = (have_atoms == want_atoms)
    else:
        have_systems = meta.get("systems")
        composition_ok = have_systems is not None and (
            [list(row) for row in have_systems]
            == [[str(s.name), str(s.atom), int(s.charge), int(s.spin)]
                for s in systems])
    manifest_ok = (meta.get("basis") == basis
                   and int(meta.get("grid_level", -1)) == int(grid_level)
                   and meta.get("auxbasis") == auxbasis
                   and composition_ok
                   and str(meta.get("reference_xc", "pbe"))
                   == str(reference_xc)
                   and str(meta.get("exchange_footing", "total"))
                   == str(exchange_footing)
                   and float(meta.get("mesh", {}).get(
                       "weight_fraction", MESH_WEIGHT_FRACTION))
                   == float(mesh_fraction))
    if not manifest_ok:
        return False
```

Everything from the `# A descriptor-bearing file written before rung-3.5
support ...` comment onward is unchanged. Do NOT add a staleness rule for
`system_all`: an existing production file is still valid data for the
point-wise loss, and `run_pretrain` refuses loudly (Task 6) when the energy
term is on and the index is absent. Forcing every archived data directory to
regenerate for a column the default loss does not read would cost hours and
buy nothing.

- [ ] **Step 5: Extend the skip-if-current driver**

Replace `ensure_pretrain_data` (currently lines 408-428) with:

```python
def ensure_pretrain_data(data_dir, *, atoms=None, basis=DEFAULT_BASIS,
                         grid_level=DEFAULT_GRID_LEVEL, polarized=True,
                         descriptors=True, density_fit=False, auxbasis=None,
                         cusp_log_transform=True, progress=False,
                         dfs_set=False, pool_atoms=False, reference_xc="pbe",
                         exchange_footing="total",
                         mesh_fraction=MESH_WEIGHT_FRACTION):
    """Skip-if-current driver for staged pretrain data.

    Returns the canonical ``.npz`` path, (re)generating it ONLY when the file
    is absent or its manifest's identity differs from the requested one.
    Idempotent: a second call at the same settings is a no-op. The set is
    resolved ONCE here and handed to both the currency check and the
    generator, so the file that is checked and the file that is written can
    never be built from different lists.
    """
    eff_aux = _effective_auxbasis(basis, density_fit, auxbasis)
    systems = resolve_pretrain_systems(atoms=atoms, dfs_set=dfs_set,
                                       pool_atoms=pool_atoms,
                                       reference_xc=reference_xc)
    out_path = os.path.join(data_dir,
                            pretrain_data_filename(polarized, reference_xc))
    if pretrain_data_is_current(out_path, basis=basis, grid_level=grid_level,
                                auxbasis=eff_aux, systems=systems,
                                reference_xc=reference_xc,
                                exchange_footing=exchange_footing,
                                mesh_fraction=mesh_fraction):
        return out_path
    return generate_pretrain_data_npz(
        data_dir, systems=systems, basis=basis, grid_level=grid_level,
        polarized=polarized, descriptors=descriptors, density_fit=density_fit,
        auxbasis=auxbasis, cusp_log_transform=cusp_log_transform,
        progress=progress, reference_xc=reference_xc,
        exchange_footing=exchange_footing, mesh_fraction=mesh_fraction)
```

- [ ] **Step 6: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/pretrain_data_gen.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_schema.py xcquinox/alec/tests/test_pretrain_data_gen.py xcquinox/alec/tests/test_pretrain_data_basis.py xcquinox/alec/tests/test_pretrain_mesh.py xcquinox/alec/tests/test_metagga_pretrain.py -v > /tmp/xcq-testlogs/task05_green.log 2>&1; echo "exit=$?"
```
Expected: PASS. `test_pretrain_data_basis.py`'s `test_default_atom_set_*` tests
call `ensure_pretrain_data(str(tmp_path))` with no `atoms`, which the ``None``
sentinel resolves to `DEFAULT_PRETRAIN_ATOMS` because neither inventory is
requested, so they are unchanged.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_pretrain_schema.py xcquinox/alec/tests/test_pretrain_data_gen.py xcquinox/alec/tests/test_pretrain_data_basis.py xcquinox/alec/tests/test_pretrain_mesh.py xcquinox/alec/tests/test_metagga_pretrain.py -v > /tmp/xcq-testlogs/task05_green.log 2>&1`

---

## Task 6: The per-system energy term in the loss

**Files:**
- Modify: `xcquinox/alec/pretrain_data_gen.py` -- `_system_energy_targets` (Task 3) and the exchange block of `generate_pretrain_data_npz` (Task 4)
- Modify: `xcquinox/alec/pretrain.py:139-205` (`_assemble_pretrain_descriptors`), `:456-543` (`_PretrainLoss` and the weighting branch, moved to module scope), `:389-449` (the block/target selection in `run_pretrain`)
- Test: `xcquinox/alec/tests/test_pretrain_energy_term.py` (create)

**Interfaces:**
- Consumes: the `.npz` schema of Task 4; `read_pretrain_manifest`, `resolve_parent_density` (Tasks 1, 5).
- Produces:
  - `pretrain_data_gen._x_block_lda(block) -> np.ndarray` and the stored column `e_lda_x_x`
  - `pretrain._assemble_pretrain_descriptors(arch, pretrain_data, *, for_cnet=False, suffix="_all")`
  - `pretrain._PretrainLoss` at module scope, with `parts(model, descriptors, ref_F) -> (pointwise, energy)` and the fields `weights`, `energy_row_weight`, `energy_segment`, `energy_target`, `energy_weight`, `n_systems`
  - `pretrain._energy_term_inputs(pretrain_data, *, weight_key, lda_key, segment_key, target_key, n_mesh) -> (row_weight, segment, target, n_systems)`

- [ ] **Step 1: Write the failing tests**

Create `xcquinox/alec/tests/test_pretrain_energy_term.py`:

```python
"""The per-system energy term of the pretraining objective.

Spec Section 6 deviation 3: "the point-wise residual is integration-weighted
(as today) AND a per-system energy term E_xc^NN - E_xc^parent in Hartree is
added, so the H atom and every molecule carry an energy of their own". These
tests pin the term's algebra against closed forms and its plumbing against a
real tiny .npz.
"""
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import xcquinox.alec.pretrain_data_gen as pdg
from xcquinox.alec.config import ArchitectureConfig, PretrainSpec
from xcquinox.alec.pretrain import (
    _PretrainLoss, _assemble_pretrain_descriptors, _energy_term_inputs,
    run_pretrain)


class _EchoModel(eqx.Module):
    """A stand-in network whose enhancement factor is the row's first column
    plus a constant, so a test can make it reproduce a target exactly or miss
    it by a stated amount."""
    offset: float = 0.0

    def __call__(self, row):
        return 1.0 + row[0] + self.offset


def _loss_arrays():
    """Two systems, three rows each, with a mesh row belonging to neither."""
    ref = jnp.asarray([0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.0])
    descriptors = jnp.stack([ref, jnp.ones(7)], axis=1)
    row_weight = jnp.asarray([1.0, 2.0, 0.5, 3.0, 1.5, 1.0, 0.0])
    segment = jnp.asarray([0, 0, 0, 1, 1, 1, 2], dtype=jnp.int32)
    return ref, descriptors, row_weight, segment


def _parent_energy(ref, row_weight, segment, n_systems):
    """The parent's own value of the same quadrature: sum w (1 + F_ref)."""
    contrib = np.asarray(row_weight) * (1.0 + np.asarray(ref))
    seg = np.asarray(segment)
    return jnp.asarray([contrib[seg == s].sum() for s in range(n_systems)])


# ---------------------------------------------------------------------------
# The term's algebra
# ---------------------------------------------------------------------------

def test_energy_term_vanishes_for_a_network_that_reproduces_the_target():
    """The stored per-system target is the quadrature of the stored
    enhancement factors, so a network that reproduces them exactly carries no
    energy error. That is what makes the term measure the fit and nothing
    else."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=2)
    pointwise, energy = loss.parts(_EchoModel(0.0), descriptors, ref)
    assert float(pointwise) == pytest.approx(0.0, abs=1e-28)
    assert float(energy) == pytest.approx(0.0, abs=1e-24)
    assert float(loss(_EchoModel(0.0), descriptors, ref)) == \
        pytest.approx(0.0, abs=1e-24)


def test_constant_offset_gives_the_analytic_energy_term():
    """A network uniformly off by c gives per-system energy error c * R_s with
    R_s the system's total row weight, so the term is
    mean_s (c R_s)^2 exactly."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=2)
    c = 0.25
    _pw, energy = loss.parts(_EchoModel(c), descriptors, ref)
    rw = np.asarray(row_weight)
    seg = np.asarray(segment)
    expect = float(np.mean([(c * rw[seg == s].sum()) ** 2 for s in range(2)]))
    assert float(energy) == pytest.approx(expect, rel=1e-12)


def test_mesh_rows_carry_no_energy():
    """A synthetic (r_s, s, alpha) node belongs to no system: its sink segment
    index is asked of segment_sum and dropped, so its enhancement factor can
    never move a system's energy."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=2)
    _pw, base = loss.parts(_EchoModel(0.0), descriptors, ref)
    bumped = descriptors.at[6, 0].add(10.0)
    _pw2, moved = loss.parts(_EchoModel(0.0), bumped, ref)
    assert float(base) == pytest.approx(float(moved), abs=1e-24)


def test_total_loss_is_pointwise_plus_the_weighted_energy_term():
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    for w_e in (0.5, 2.0):
        loss = _PretrainLoss(weights=jnp.ones(7),
                             energy_row_weight=row_weight,
                             energy_segment=segment, energy_target=target,
                             energy_weight=w_e, n_systems=2)
        pw, en = loss.parts(_EchoModel(0.3), descriptors, ref)
        assert float(loss(_EchoModel(0.3), descriptors, ref)) == \
            pytest.approx(float(pw) + w_e * float(en), rel=1e-12)


def test_zero_weight_returns_the_pre_existing_loss_bit_for_bit():
    """Default configuration: the energy term is not merely zero, it is not
    evaluated, so an existing run's loss value does not move."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    w = jnp.asarray([1.0, 2.0, 0.5, 3.0, 1.5, 1.0, 0.25])
    plain = _PretrainLoss(weights=w)
    armed = _PretrainLoss(weights=w, energy_row_weight=row_weight,
                          energy_segment=segment, energy_target=target,
                          energy_weight=0.0, n_systems=2)
    model = _EchoModel(0.4)
    a = float(plain(model, descriptors, ref))
    b = float(armed(model, descriptors, ref))
    assert a == b
    resid = (np.asarray(descriptors)[:, 0] + 0.4 - np.asarray(ref)) ** 2
    expect = float(np.sum(np.asarray(w) * resid) / (np.sum(np.asarray(w))
                                                    + 1e-12))
    assert a == pytest.approx(expect, rel=1e-12)


def test_energy_term_is_differentiable():
    """The term must reach the optimizer: a zero gradient would make it
    decorative."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=2)
    grad = eqx.filter_grad(loss)(_EchoModel(0.3), descriptors, ref)
    assert abs(float(grad.offset)) > 1e-6


# ---------------------------------------------------------------------------
# _energy_term_inputs
# ---------------------------------------------------------------------------

def test_energy_term_inputs_pad_the_mesh_with_the_sink_segment():
    data = {"weights_all": jnp.asarray([1.0, 2.0, 4.0]),
            "e_lda_c_all": jnp.asarray([-1.0, -2.0, -0.5]),
            "system_all": jnp.asarray([0, 0, 1], dtype=jnp.int32),
            "e_c_parent_sys": jnp.asarray([-5.0, -2.0])}
    rw, seg, tgt, ns = _energy_term_inputs(
        data, weight_key="weights_all", lda_key="e_lda_c_all",
        segment_key="system_all", target_key="e_c_parent_sys", n_mesh=2)
    assert ns == 2
    np.testing.assert_allclose(np.asarray(rw), [-1.0, -4.0, -2.0, 0.0, 0.0])
    assert np.asarray(seg).tolist() == [0, 0, 1, 2, 2]
    np.testing.assert_allclose(np.asarray(tgt), [-5.0, -2.0])


# ---------------------------------------------------------------------------
# Row-block selection
# ---------------------------------------------------------------------------

def test_assemble_reads_the_exchange_block_on_request():
    arch = ArchitectureConfig.from_spec("t_plain", 2, 8)
    data = {"rho_all": jnp.ones(3), "sigma_all": jnp.zeros(3),
            "rho_x": jnp.full(5, 2.0), "sigma_x": jnp.full(5, 3.0)}
    assert _assemble_pretrain_descriptors(arch, data).shape == (3, 2)
    got = _assemble_pretrain_descriptors(arch, data, suffix="_x")
    assert got.shape == (5, 2)
    assert float(got[0, 0]) == 2.0


def test_assemble_refuses_a_correlation_row_set_that_is_not_the_total_density():
    """Correlation is spin-interpolated rather than spin-scaled and stays on the
    total density (von Barth and Hedin, J. Phys. C 5, 1629 (1972); Perdew and
    Wang, Phys. Rev. B 45, 13244 (1992)), so the cnet never reads the
    per-channel exchange block."""
    arch = ArchitectureConfig.from_spec("t_plain", 2, 8)
    with pytest.raises(ValueError, match="total density"):
        _assemble_pretrain_descriptors(arch, {"rho_x": jnp.ones(3),
                                              "sigma_x": jnp.ones(3)},
                                       for_cnet=True, suffix="_x")


# ---------------------------------------------------------------------------
# run_pretrain plumbing, on a real tiny .npz
# ---------------------------------------------------------------------------

_TINY = (("He", 0), ("H", 1))


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("energy_term")
    pdg.generate_pretrain_data_npz(
        str(d), atoms=_TINY, basis="sto-3g", grid_level=0, polarized=False,
        descriptors=True, exchange_footing="spin_channel")
    return str(d)


def _spec(tmp_path, data_dir, **kw):
    arch = ArchitectureConfig.from_spec("t_energy", 2, 8)
    return PretrainSpec(arch=arch, data_dir=data_dir,
                        checkpoint_dir=str(tmp_path / "ck"), n_steps=2,
                        seed=0, loss_weighting="integration", **kw)


def test_run_pretrain_records_the_energy_term(tiny_dir, tmp_path):
    md = run_pretrain(_spec(tmp_path, tiny_dir, energy_term_weight=1.0))
    assert md["energy_term_weight"] == 1.0
    assert md["n_systems"] == 2
    assert np.isfinite(md["energy_term_x_final"])
    assert np.isfinite(md["energy_term_c_final"])
    assert md["exchange_footing"] == "spin_channel"
    on_disk = json.load(open(os.path.join(tmp_path / "ck",
                                          "pretrain_metadata.json")))
    assert on_disk["energy_term_weight"] == 1.0


def test_run_pretrain_default_records_a_zero_weight(tiny_dir, tmp_path):
    md = run_pretrain(_spec(tmp_path, tiny_dir))
    assert md["energy_term_weight"] == 0.0


def test_run_pretrain_refuses_the_energy_term_without_a_system_index(tmp_path):
    d = tmp_path / "legacy"
    d.mkdir()
    np.savez(d / "pretrain_data.npz", rho_all=np.ones(4),
             sigma_all=np.zeros(4), Fx_all=np.zeros(4), Fc_all=np.zeros(4),
             Fx_scan_all=np.zeros(4), Fc_scan_all=np.zeros(4),
             metagga_all=np.zeros((4, 1)), weights_all=np.ones(4))
    with pytest.raises(ValueError, match="system_all"):
        run_pretrain(_spec(tmp_path, str(d), energy_term_weight=1.0))


def test_run_pretrain_refuses_a_file_built_on_the_wrong_parent_density(
        tiny_dir, tmp_path):
    """A meta-GGA architecture pretraining on a PBE-density file would be fit
    to a density its SCF never sees; the mismatch fails loudly instead."""
    arch = ArchitectureConfig.from_spec("t_mgga_parent", 2, 8,
                                        descriptors=["metagga"],
                                        meta_gga=True)
    spec = PretrainSpec(arch=arch, data_dir=tiny_dir,
                        checkpoint_dir=str(tmp_path / "ck_p"), n_steps=2,
                        seed=0, parent_density="auto")
    with pytest.raises(ValueError, match="parent"):
        run_pretrain(spec)
```

- [ ] **Step 2: Run and confirm it fails**

```bash
python -m pytest xcquinox/alec/tests/test_pretrain_energy_term.py -v > /tmp/xcq-testlogs/task06_red.log 2>&1; echo "exit=$?"
```
Expected: `ImportError: cannot import name '_PretrainLoss' from
'xcquinox.alec.pretrain'` (the class is still local to `run_pretrain`).

- [ ] **Step 3: Give the exchange block its own LDA column (generator amendment)**

The loss must multiply the network's enhancement factor by the SAME
floating-point number the per-system target was built from, or "zero energy
error" becomes "zero to within the last-ulp difference between `np.cbrt` and
`rho ** (1/3)`". Store the column and share one expression.

In `xcquinox/alec/pretrain_data_gen.py`, add immediately before
`_system_energy_targets`:

```python
def _x_block_lda(block):
    """``rho eps_x^LDA`` for one exchange block, in the block's own convention.

    A closed-shell system's exchange block IS its total-density block, which
    already carries the libxc-derived ``e_lda_x`` (a spin=0 call there, so the
    unpolarized LDA at the total density). An open shell's per-channel block
    carries no LDA column: its denominator is the analytic unpolarized LDA at
    the DOUBLED density ``2 rho_sigma``, a function of the stored ``rho``
    alone. One expression, used by both the per-system target and the stored
    column, so the two are the same floating-point number.
    """
    if "e_lda_x" in block:
        return np.asarray(block["e_lda_x"])
    rho = np.asarray(block["rho"])
    return rho * (_LDA_X_C * np.cbrt(rho))
```

In `_system_energy_targets`, replace the `else` branch's two `e_lda_x` lines
with:

```python
    else:
        e_lda_x = _x_block_lda(x_cols)
```

In `generate_pretrain_data_npz`, inside the `exchange_footing == "spin_channel"`
block, add to the `save_kwargs.update({...})` payload:

```python
            "e_lda_x_x": np.concatenate([_x_block_lda(b) for b in x_blocks]),
```

- [ ] **Step 4: Move the loss to module scope and add the energy term**

In `xcquinox/alec/pretrain.py`, DELETE the `class _PretrainLoss` definition
from inside `run_pretrain` (currently lines 470-490) and insert at module
scope, immediately after `_compute_integration_weights` (currently ending line
96):

```python
class _PretrainLoss(eqx.Module):
    """Pretraining objective: point-wise enhancement-factor residual plus an
    optional per-system energy term.

    Networks return the enhancement factor F; targets are stored as ``F - 1``,
    so ``pred - 1`` aligns with ``ref_F``. ``weights=None`` gives the plain
    mean of squared residuals; a 1-D ``weights`` aligned with the rows gives
    the integration-weighted reduction
    ``sum(w r^2) / (sum(w) + 1e-12)``.

    The energy term is

        w_E * (1 / N_sys) sum_s ( sum_{i in s} w_i e_LDA_i F^NN_i - E_s )^2

    in Hartree^2, with ``E_s`` the parent's own value of the same quadrature
    (``pretrain_data_gen._system_energy_targets``). It exists because the
    point-wise residual alone does not bound a system's energy: measured across
    seven architectures, the one with the LOWEST exchange residual carried the
    LARGEST atomization-energy offset from its parent. The mean over systems
    rather than the sum keeps the term's magnitude independent of how many
    systems the set holds, so ``w_E`` means the same thing for a four-atom file
    and a thirty-seven-system one. Rows belonging to no system -- the synthetic
    (r_s, s, alpha) mesh -- carry zero weight and the sink segment index
    ``n_systems``, which is asked of ``segment_sum`` and then dropped.

    At ``energy_weight == 0`` the term is not evaluated at all, so the returned
    value is the pre-existing loss bit for bit; that short circuit is the
    reason the reduction is written twice.
    """
    weights: jnp.ndarray | None = None
    energy_row_weight: jnp.ndarray | None = None
    energy_segment: jnp.ndarray | None = None
    energy_target: jnp.ndarray | None = None
    energy_weight: float = eqx.field(static=True, default=0.0)
    n_systems: int = eqx.field(static=True, default=0)

    def parts(self, model, descriptors, ref_F):
        """``(pointwise, energy)`` -- the two terms, the second unweighted."""
        pred = jax.vmap(model)(descriptors).squeeze()
        shifted = pred - 1.0
        residual_sq = (shifted - ref_F) ** 2
        if self.weights is None:
            pointwise = jnp.mean(residual_sq)
        else:
            w = self.weights
            pointwise = jnp.sum(w * residual_sq) / (jnp.sum(w) + 1e-12)
        if self.energy_target is None:
            return pointwise, jnp.zeros_like(pointwise)
        e_nn = jax.ops.segment_sum(
            self.energy_row_weight * pred, self.energy_segment,
            num_segments=self.n_systems + 1)[:self.n_systems]
        delta = e_nn - self.energy_target
        return pointwise, jnp.sum(delta * delta) / self.n_systems

    def __call__(self, model, descriptors, ref_F):
        if self.energy_weight == 0.0 or self.energy_target is None:
            pred = jax.vmap(model)(descriptors).squeeze()
            pred = pred - 1.0
            residual_sq = (pred - ref_F) ** 2
            if self.weights is None:
                return jnp.mean(residual_sq)
            w = self.weights
            return jnp.sum(w * residual_sq) / (jnp.sum(w) + 1e-12)
        pointwise, energy = self.parts(model, descriptors, ref_F)
        return pointwise + self.energy_weight * energy


def _energy_term_inputs(pretrain_data, *, weight_key, lda_key, segment_key,
                        target_key, n_mesh):
    """``(row_weight, segment, target, n_systems)`` for one network's energy term.

    ``row_weight_i = w_i e_LDA_i`` is Hartree per unit enhancement factor, so
    ``sum_{i in s} row_weight_i F^NN_i`` is the network's XC energy of system
    ``s`` on the rows the file stores. ``n_mesh`` synthetic rows are appended
    with zero weight and the sink segment index, so the row set matches the
    descriptor tensor the mesh was concatenated onto.
    """
    target = jnp.asarray(pretrain_data[target_key])
    n_systems = int(target.shape[0])
    row_weight = (jnp.asarray(pretrain_data[weight_key])
                  * jnp.asarray(pretrain_data[lda_key]))
    segment = jnp.asarray(pretrain_data[segment_key], dtype=jnp.int32)
    if n_mesh:
        row_weight = jnp.concatenate([row_weight, jnp.zeros(n_mesh)])
        segment = jnp.concatenate(
            [segment, jnp.full(n_mesh, n_systems, dtype=jnp.int32)])
    return row_weight, segment, target, n_systems
```

- [ ] **Step 5: Let the descriptor assembler read either block**

In `_assemble_pretrain_descriptors`, change the signature to

```python
def _assemble_pretrain_descriptors(arch: ArchitectureConfig, pretrain_data: dict,
                                   *, for_cnet: bool = False,
                                   suffix: str = "_all") -> jnp.ndarray:
```

append to its docstring:

```
    ``suffix`` selects the row block. ``"_all"`` (default) is the
    total-density block, which carries the correlation rows always and the
    exchange rows under the historical footing. ``"_x"`` is the per-channel
    exchange block a file built on the exact-spin-scaling footing carries; the
    correlation network never reads it, because correlation is
    spin-interpolated rather than spin-scaled (von Barth and Hedin, J. Phys. C
    5, 1629 (1972); Perdew and Wang, Phys. Rev. B 45, 13244 (1992)) and stays
    on the total density with zeta.
```

and replace the body's first lines (currently 164-173) with:

```python
    from xcquinox.alec.descriptors import make_descriptor
    if for_cnet and suffix != "_all":
        raise ValueError(
            "the correlation network is posed on the total density, so its "
            f"rows are the '_all' block; got suffix={suffix!r}."
        )
    cols = [pretrain_data["rho" + suffix], pretrain_data["sigma" + suffix]]
    if for_cnet and arch.use_polarized_correlation:
        zeta_all = pretrain_data.get("zeta_all")
        if zeta_all is None:
            zeta_all = jnp.zeros_like(pretrain_data["rho_all"])
        cols.append(zeta_all)
    # Map descriptor.name -> the pretrain_data column STEM; the block suffix
    # is appended, so one map serves both row blocks.
    _key_map = {"dm_statistics": "dm", "cusp": "cusp", "rung35": "rung35",
                "rung35_multishell": "rung35ms", "metagga": "metagga"}
    for spec in arch.descriptors:
        stem = _key_map.get(spec.name)
        if stem is None:
            raise KeyError(
                f"_assemble_pretrain_descriptors: no pretrain_data key "
                f"mapping registered for descriptor {spec.name!r}; update "
                f"_key_map in pretrain.py"
            )
        key = stem + suffix
        arr = pretrain_data[key]
```

The rest of the loop is unchanged.

- [ ] **Step 6: Wire the term into `run_pretrain`**

In `run_pretrain`, immediately after `pretrain_data` is lifted into JAX
(currently line 387), insert:

```python
    # Which row block the exchange network reads. A file built on the
    # exact-spin-scaling footing carries the open-shell exchange rows
    # separately, because the per-channel rows of an open shell are not its
    # total-density rows; a file built on the historical footing has one block
    # and the xnet reads it, byte-identically.
    x_suffix = "_x" if "rho_x" in pretrain_data else "_all"
    # The parent whose SELF-CONSISTENT density this architecture must pretrain
    # on. A meta-GGA network fit on a PBE density is fit to a density its SCF
    # never sees.
    from xcquinox.alec.pretrain_data_gen import (
        read_pretrain_manifest, resolve_parent_density)
    want_reference = resolve_parent_density(
        spec.arch, getattr(spec, "parent_density", "pbe"))
    _manifest = read_pretrain_manifest(npz_path)
    file_reference = str((_manifest or {}).get("reference_xc", "pbe"))
    if _manifest is not None and file_reference != want_reference:
        raise ValueError(
            f"run_pretrain: architecture {spec.arch.name!r} resolves to the "
            f"{want_reference!r} parent density, but {npz_path!r} was built on "
            f"the {file_reference!r} density. Point data_dir at the "
            f"{want_reference} file or set pretrain.parent_density explicitly."
        )
    energy_weight = float(getattr(spec, "energy_term_weight", 0.0))
    if energy_weight > 0.0 and "system_all" not in pretrain_data:
        raise ValueError(
            "run_pretrain: pretrain.energy_term_weight > 0 needs the per-row "
            "system index 'system_all' and the per-system energy table, which "
            f"{npz_path!r} predates. Regenerate it with "
            "pretrain_data_gen.ensure_pretrain_data."
        )
```

Change the descriptor assembly (currently lines 393-395) to:

```python
    descriptors = _assemble_pretrain_descriptors(spec.arch, pretrain_data,
                                                 suffix=x_suffix)
    descriptors_c = _assemble_pretrain_descriptors(
        spec.arch, pretrain_data, for_cnet=True)
```

and the target selection (currently lines 402-407) to:

```python
    if bool(getattr(spec.arch, "meta_gga", False)):
        Fx_target = pretrain_data["Fx_scan" + x_suffix]
        Fc_target = pretrain_data["Fc_scan_all"]
        e_x_key, e_c_key = "e_x_parent_scan_sys", "e_c_parent_scan_sys"
    else:
        Fx_target = pretrain_data["Fx" + x_suffix]
        Fc_target = pretrain_data["Fc_all"]
        e_x_key, e_c_key = "e_x_parent_sys", "e_c_parent_sys"
```

In the `"integration"` branch, change the two weight sources (currently line
519) to:

```python
        w_x, _unused = _compute_integration_weights(
            pretrain_data["rho" + x_suffix],
            pretrain_data.get("weights" + x_suffix))
        _unused, w_c = _compute_integration_weights(rho_all, grid_weights)
```

(`rho_all` and `grid_weights` above are unchanged and stay the correlation
block's; at `x_suffix == "_all"` both calls take the same arguments and return
the same arrays, which is why the default is byte-identical.)

In the `mesh_used` block just below it (currently lines 520-538), read the
share from the DATA rather than from the module constant, so the loss can never
weight the mesh differently than the file was built to:

```python
            from xcquinox.alec.pretrain_data_gen import MESH_WEIGHT_FRACTION
            # The share the DATA was built at. A file written before the share
            # became configurable carries no such key and falls back to the
            # constant it was built with.
            mesh_fraction = float(pretrain_data_np.get(
                "mesh_weight_fraction", MESH_WEIGHT_FRACTION))
            n_mesh = int(pretrain_data["rho_mesh"].shape[0])
            scale = mesh_fraction / (1.0 - mesh_fraction)
```

and the mesh-append banner (currently lines 435-440) the same way:

```python
            from xcquinox.alec.pretrain_data_gen import MESH_WEIGHT_FRACTION
            _share = float(pretrain_data_np.get("mesh_weight_fraction",
                                                MESH_WEIGHT_FRACTION))
            print(f"[pretrain] (s, alpha) mesh appended: "
                  f"{pretrain_data['rho_mesh'].shape[0]} nodes "
                  f"({100.0 * _share:.0f}% effective "
                  "loss-weight share per channel, by construction)",
                  flush=True)
```

Replace the two loss constructions (currently lines 539-543) with:

```python
    n_mesh_rows = int(pretrain_data["rho_mesh"].shape[0]) if mesh_used else 0
    energy_kwargs_x = {}
    energy_kwargs_c = {}
    if energy_weight > 0.0:
        _rw, _seg, _tgt, _ns = _energy_term_inputs(
            pretrain_data, weight_key="weights" + x_suffix,
            lda_key="e_lda_x" + x_suffix, segment_key="system" + x_suffix,
            target_key=e_x_key, n_mesh=n_mesh_rows)
        energy_kwargs_x = dict(energy_row_weight=_rw, energy_segment=_seg,
                               energy_target=_tgt, n_systems=_ns,
                               energy_weight=energy_weight)
        _rw, _seg, _tgt, _ns = _energy_term_inputs(
            pretrain_data, weight_key="weights_all", lda_key="e_lda_c_all",
            segment_key="system_all", target_key=e_c_key, n_mesh=n_mesh_rows)
        energy_kwargs_c = dict(energy_row_weight=_rw, energy_segment=_seg,
                               energy_target=_tgt, n_systems=_ns,
                               energy_weight=energy_weight)
    n_systems = int(pretrain_data[e_x_key].shape[0]) \
        if e_x_key in pretrain_data else 0
```

and make both weighting branches pass them:

```python
        loss_fn_x = _PretrainLoss(weights=w_x, **energy_kwargs_x)
        loss_fn_c = _PretrainLoss(weights=w_c, **energy_kwargs_c)
    else:  # "unweighted": validated at construction
        loss_fn_x = _PretrainLoss(**energy_kwargs_x)
        loss_fn_c = _PretrainLoss(**energy_kwargs_c)
```

Finally, add to the metadata dict (currently lines 640-672), after
`"pretrain_mesh"`:

```python
        # Pretraining-set provenance the Section 3.3 certificate and HISTORY
        # read: which systems the fit saw, on which parent density, at which
        # exchange footing, and how hard the per-system energy term pulled.
        "reference_xc": want_reference,
        "exchange_footing": str(
            (_manifest or {}).get("exchange_footing", "total")),
        "energy_term_weight": energy_weight,
        "n_systems": n_systems,
        "n_rows_x": int(descriptors.shape[0]),
        "n_rows_c": int(descriptors_c.shape[0]),
        "energy_term_x_final": float(
            loss_fn_x.parts(xnet_trained, descriptors, Fx_target)[1]),
        "energy_term_c_final": float(
            loss_fn_c.parts(cnet_trained, descriptors_c, Fc_target)[1]),
```

- [ ] **Step 7: Compile and run**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/pretrain.py xcquinox/alec/pretrain_data_gen.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_energy_term.py xcquinox/alec/tests/test_pretrain.py xcquinox/alec/tests/test_pretrain_weighted.py xcquinox/alec/tests/test_pretrain_mesh.py xcquinox/alec/tests/test_pretrain_schema.py -v > /tmp/xcq-testlogs/task06_green.log 2>&1; echo "exit=$?"
```
Expected: PASS. `PretrainSpec` does not yet carry `parent_density` /
`energy_term_weight`, so the two `run_pretrain` tests that pass them fail with
`TypeError` until Task 8 adds the fields; run this command again at the end of
Task 8. Every OTHER test in these files must be green here.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_pretrain_energy_term.py xcquinox/alec/tests/test_pretrain.py xcquinox/alec/tests/test_pretrain_weighted.py xcquinox/alec/tests/test_pretrain_mesh.py xcquinox/alec/tests/test_pretrain_schema.py -v > /tmp/xcq-testlogs/task06_green.log 2>&1`

---

## Task 7: Held-out-system validation and the stop criterion

**Files:**
- Modify: `xcquinox/alec/pretrain.py` -- new helpers after `_energy_term_inputs`; the training section of `run_pretrain` (currently lines 583-627) and the metadata dict
- Test: `xcquinox/alec/tests/test_pretrain_energy_term.py` (append)

**Interfaces:**
- Consumes: `_PretrainLoss` and `_energy_term_inputs` (Task 6).
- Produces:
  - `pretrain._validation_systems(system_natoms, fraction, seed) -> tuple[int, ...]`
  - `pretrain._system_split_arrays(segment, n_systems, held_out) -> tuple`
  - `pretrain._restrict_loss(loss, descriptors, ref_F, mask, remap, kept_ids) -> (loss, descriptors, ref_F)`
  - `pretrain._train_pretrain_network(model, optimizer, loss_train, desc_train, ref_train, loss_val, desc_val, ref_val, *, n_steps, validate_every, patience, monitor, progress_callback=None) -> (model, losses, record)`

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_pretrain_energy_term.py`:

```python
# ---------------------------------------------------------------------------
# Held-out-system validation
# ---------------------------------------------------------------------------

def test_validation_holds_out_molecules_and_never_an_atom():
    """Every pool atom is a system the Section 3.3 certificate bounds at
    tol_atom = 1.0 mHa, and every atomization energy is anchored on atoms. A
    held-out atom would be an atom the fit never saw, so the split draws from
    the MOLECULES only."""
    from xcquinox.alec.pretrain import _validation_systems
    natoms = np.array([1, 1, 1, 2, 3, 5, 4, 2, 3, 10], dtype=np.int32)
    held = _validation_systems(natoms, 0.3, seed=0)
    assert held
    assert all(int(natoms[i]) > 1 for i in held)
    assert len(held) == 2  # round(0.3 * 7)


def test_validation_split_is_seeded_and_reproducible():
    from xcquinox.alec.pretrain import _validation_systems
    natoms = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    a = _validation_systems(natoms, 0.5, seed=7)
    b = _validation_systems(natoms, 0.5, seed=7)
    c = _validation_systems(natoms, 0.5, seed=8)
    assert a == b
    assert a != c
    assert tuple(sorted(a)) == a


def test_validation_split_is_empty_at_zero_fraction():
    from xcquinox.alec.pretrain import _validation_systems
    natoms = np.array([1, 2, 3], dtype=np.int32)
    assert _validation_systems(natoms, 0.0, seed=0) == ()


def test_validation_split_never_takes_every_molecule():
    """A split that held out all the molecules would leave the fit with atoms
    only, which is the coverage failure the set change exists to remove."""
    from xcquinox.alec.pretrain import _validation_systems
    natoms = np.array([1, 2, 3], dtype=np.int32)
    held = _validation_systems(natoms, 1.0, seed=0)
    assert len(held) == 1


def test_validation_split_with_no_molecules_is_empty():
    from xcquinox.alec.pretrain import _validation_systems
    assert _validation_systems(np.ones(4, dtype=np.int32), 0.5, seed=0) == ()


def test_split_arrays_keep_the_mesh_in_training():
    """The synthetic mesh regularizes the functional form; it is not a system
    whose energy is predicted, so holding it out would measure nothing."""
    from xcquinox.alec.pretrain import _system_split_arrays
    seg = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)  # 3 = sink
    train_mask, val_mask, train_remap, val_remap, train_ids, val_ids = \
        _system_split_arrays(seg, 3, (1,))
    assert train_mask.tolist() == [True, True, False, False, True, True,
                                   True, True]
    assert val_mask.tolist() == [False, False, True, True, False, False,
                                 False, False]
    assert train_ids.tolist() == [0, 2]
    assert val_ids.tolist() == [1]
    # Renumbering maps kept systems onto 0..n-1 and everything else onto the
    # sink index.
    assert train_remap[np.array([0, 2, 3])].tolist() == [0, 1, 2]
    assert int(train_remap[1]) == 2
    assert int(val_remap[1]) == 0


def test_restrict_loss_reindexes_the_energy_term():
    from xcquinox.alec.pretrain import _restrict_loss, _system_split_arrays
    ref = jnp.asarray([0.1, 0.2, 0.3, 0.4])
    desc = jnp.stack([ref, jnp.ones(4)], axis=1)
    seg = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    rw = jnp.asarray([1.0, 1.0, 2.0, 2.0])
    tgt = jnp.asarray([10.0, 20.0])
    full = _PretrainLoss(weights=jnp.ones(4), energy_row_weight=rw,
                         energy_segment=seg, energy_target=tgt,
                         energy_weight=1.0, n_systems=2)
    tm, vm, trm, vrm, tid, vid = _system_split_arrays(np.asarray(seg), 2, (1,))
    tr_loss, tr_desc, tr_ref = _restrict_loss(full, desc, ref, tm, trm, tid)
    assert tr_desc.shape == (2, 2)
    assert tr_loss.n_systems == 1
    assert np.asarray(tr_loss.energy_segment).tolist() == [0, 0]
    np.testing.assert_allclose(np.asarray(tr_loss.energy_target), [10.0])
    va_loss, va_desc, _va_ref = _restrict_loss(full, desc, ref, vm, vrm, vid)
    assert va_desc.shape == (2, 2)
    assert np.asarray(va_loss.energy_target).tolist() == [20.0]


def test_training_loop_stops_on_patience_and_returns_the_best_weights():
    """The stop criterion replaces the DFS protocol's hand interruption (spec
    Section 6): training halts when the monitored validation quantity has not
    improved for ``patience`` validations, and the weights that are kept are
    the best ones seen, not the last ones."""
    import optax
    from xcquinox.alec.pretrain import _train_pretrain_network
    ref = jnp.asarray([0.0, 0.0])
    desc = jnp.stack([ref, jnp.ones(2)], axis=1)
    loss = _PretrainLoss(weights=jnp.ones(2))
    model, losses, record = _train_pretrain_network(
        _EchoModel(1.0), optax.sgd(1e-9), loss, desc, ref, loss, desc, ref,
        n_steps=100, validate_every=1, patience=3, monitor="pointwise")
    assert len(losses) < 100
    assert record["stopped_early"] is True
    assert record["best_step"] >= 1
    assert len(record["history"]) == len(losses) // 1
    assert float(record["best_value"]) <= float(record["history"][0][1])


def test_training_loop_runs_to_the_end_without_patience():
    import optax
    from xcquinox.alec.pretrain import _train_pretrain_network
    ref = jnp.asarray([0.0, 0.0])
    desc = jnp.stack([ref, jnp.ones(2)], axis=1)
    loss = _PretrainLoss(weights=jnp.ones(2))
    _m, losses, record = _train_pretrain_network(
        _EchoModel(1.0), optax.sgd(1e-3), loss, desc, ref, loss, desc, ref,
        n_steps=10, validate_every=5, patience=0, monitor="pointwise")
    assert len(losses) == 10
    assert record["stopped_early"] is False


def test_run_pretrain_validation_records_the_held_out_systems(tiny_dir,
                                                              tmp_path):
    """The tiny file is two free atoms, so there is nothing to hold out: the
    split must be empty and the run must still complete."""
    md = run_pretrain(_spec(tmp_path, tiny_dir, validation_fraction=0.5,
                            patience=2, validate_every=1))
    assert md["validation"]["fraction"] == 0.5
    assert md["validation"]["systems"] == []
    assert md["validation"]["monitor"] == "pointwise"
```

- [ ] **Step 2: Run and confirm it fails**

```bash
python -m pytest xcquinox/alec/tests/test_pretrain_energy_term.py -k "validation or split or restrict or training_loop" -v > /tmp/xcq-testlogs/task07_red.log 2>&1; echo "exit=$?"
```
Expected: `ImportError: cannot import name '_validation_systems' from
'xcquinox.alec.pretrain'`.

- [ ] **Step 3: Add the split and the loop**

Insert into `xcquinox/alec/pretrain.py` immediately after
`_energy_term_inputs`:

```python
def _validation_systems(system_natoms, fraction, seed):
    """Indices of the systems held out of the fit, as a sorted tuple.

    The split draws from the MOLECULES only. Every single-atom system is an
    anchor: the Section 3.3 certificate bounds each pool atom's E_xc at
    tol_atom, and every atomization energy is a molecule minus its atoms, so an
    atom the fit never saw would fail the acceptance test by construction.
    What validation is for here is the molecular extrapolation of the
    density-matrix features -- exactly the failure the campaign measured -- and
    that is what the molecules measure.

    ``fraction`` is a fraction of the ELIGIBLE (multi-nucleus) systems, rounded
    to the nearest integer, floored at one and capped at all-but-one so a fit
    is never left with atoms alone. The permutation is seeded so every
    architecture in a sweep holds out the same systems and their validation
    numbers are comparable.
    """
    natoms = np.asarray(system_natoms)
    eligible = [int(i) for i in range(natoms.shape[0]) if int(natoms[i]) > 1]
    if fraction <= 0.0 or len(eligible) < 2:
        return ()
    k = int(round(float(fraction) * len(eligible)))
    k = max(1, min(k, len(eligible) - 1))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(np.asarray(eligible, dtype=np.int64))
    return tuple(sorted(int(i) for i in order[:k]))


def _system_split_arrays(segment, n_systems, held_out):
    """Row masks and segment renumberings for a held-out-system split.

    ``segment`` is the per-row system index, with ``n_systems`` marking a row
    that belongs to no system -- the synthetic (r_s, s, alpha) mesh. Mesh rows
    always train: the mesh is a regularizer of the functional form, not a
    system whose energy is predicted, so holding it out would measure nothing.

    Returns ``(train_mask, val_mask, train_remap, val_remap, train_ids,
    val_ids)``. The remaps carry the kept systems onto ``0..n_kept-1`` and
    everything else, including the sink, onto ``n_kept``, so a restricted
    segment array is still a valid ``segment_sum`` index with its own sink.
    """
    seg = np.asarray(segment)
    held = np.zeros(n_systems + 1, dtype=bool)
    for i in held_out:
        held[int(i)] = True
    val_mask = held[seg]
    train_mask = ~val_mask
    train_ids = np.asarray([s for s in range(n_systems) if not held[s]],
                           dtype=np.int64)
    val_ids = np.asarray([s for s in range(n_systems) if held[s]],
                         dtype=np.int64)
    train_remap = np.full(n_systems + 1, train_ids.shape[0], dtype=np.int32)
    train_remap[train_ids] = np.arange(train_ids.shape[0], dtype=np.int32)
    val_remap = np.full(n_systems + 1, val_ids.shape[0], dtype=np.int32)
    val_remap[val_ids] = np.arange(val_ids.shape[0], dtype=np.int32)
    return (train_mask, val_mask, train_remap, val_remap, train_ids, val_ids)


def _restrict_loss(loss, descriptors, ref_F, mask, remap, kept_ids):
    """Restrict a loss and its rows to one side of a held-out-system split.

    The point-wise weights are sliced, the energy term's row weights are
    sliced, its segment indices are renumbered onto the kept systems, and its
    target vector is sliced to the same systems, so the restricted term is the
    same objective over fewer systems rather than a differently normalized one.
    """
    idx = jnp.asarray(np.flatnonzero(np.asarray(mask)))
    desc = jnp.asarray(descriptors)[idx]
    ref = jnp.asarray(ref_F)[idx]
    kwargs = {"weights": (None if loss.weights is None
                          else jnp.asarray(loss.weights)[idx])}
    if loss.energy_target is not None:
        kept = jnp.asarray(np.asarray(kept_ids, dtype=np.int64))
        kwargs.update(
            energy_row_weight=jnp.asarray(loss.energy_row_weight)[idx],
            energy_segment=jnp.asarray(remap)[
                jnp.asarray(loss.energy_segment)[idx]],
            energy_target=jnp.asarray(loss.energy_target)[kept],
            n_systems=int(np.asarray(kept_ids).shape[0]),
            energy_weight=loss.energy_weight,
        )
    return _PretrainLoss(**kwargs), desc, ref


def _train_pretrain_network(model, optimizer, loss_train, desc_train,
                            ref_train, loss_val, desc_val, ref_val, *,
                            n_steps, validate_every, patience, monitor,
                            progress_callback=None):
    """Full-batch pretraining with held-out-system validation and early stop.

    Returns ``(best_model, losses, record)``. The loop is written here rather
    than driven through ``xcquinox.train.xcTrainer`` because a stop criterion
    needs the optimizer STATE and the learning-rate schedule to survive across
    validations: ``xcTrainer`` initializes its optimizer state in its
    constructor and returns no state, so chunking a run through it would reset
    Adam's moments and restart the schedule at every validation. The
    unvalidated path still goes through ``xcTrainer`` unchanged, which is what
    keeps a run without validation byte-identical.

    ``monitor`` is ``"energy"`` when the per-system energy term is active and
    ``"pointwise"`` otherwise: the quantity the campaign needs bounded is the
    system energy, and the point-wise residual is a poor proxy for it (the
    architecture with the lowest exchange residual carried the largest
    atomization-energy offset). ``patience`` of 0 disables the stop; the loop
    then runs the full schedule and still returns the best weights seen.
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def _step(m, s):
        value, grads = eqx.filter_value_and_grad(loss_train)(
            m, desc_train, ref_train)
        updates, s = optimizer.update(grads, s, m)
        return eqx.apply_updates(m, updates), s, value

    @eqx.filter_jit
    def _evaluate(m):
        return loss_val.parts(m, desc_val, ref_val)

    every = int(validate_every) if int(validate_every) > 0 else int(n_steps)
    losses = []
    history = []
    best_value = float("inf")
    best_step = 0
    best_model = model
    stale = 0
    stopped_early = False
    for step in range(int(n_steps)):
        model, opt_state, value = _step(model, opt_state)
        losses.append(float(value))
        if progress_callback is not None:
            try:
                progress_callback(step + 1, int(n_steps), float(value))
            except Exception:  # noqa: BLE001 - a logging callback never stops a fit
                pass
        if (step + 1) % every and (step + 1) != int(n_steps):
            continue
        pointwise, energy = _evaluate(model)
        monitored = float(energy if monitor == "energy" else pointwise)
        history.append((step + 1, float(pointwise), float(energy)))
        if monitored < best_value:
            best_value = monitored
            best_step = step + 1
            best_model = model
            stale = 0
        else:
            stale += 1
            if int(patience) > 0 and stale >= int(patience):
                stopped_early = True
                break
    record = {"monitor": monitor, "best_step": best_step,
              "best_value": best_value, "stopped_early": stopped_early,
              "history": history}
    return best_model, losses, record
```

- [ ] **Step 4: Branch `run_pretrain` on the split**

In `run_pretrain`, immediately before the `# --- Train xnet ---` block
(currently line 583), insert:

```python
    # --- Held-out-system validation split ---------------------------------
    # A fraction of the MOLECULES is withheld from the fit and scored between
    # optimizer steps; training stops when the monitored quantity has not
    # improved for `patience` validations, and the weights kept are the best
    # ones seen. fraction 0 (the default) reproduces the unvalidated schedule
    # exactly, through the same xcTrainer call as before.
    val_fraction = float(getattr(spec, "validation_fraction", 0.0))
    val_seed = int(getattr(spec, "validation_seed", 0))
    validate_every = int(getattr(spec, "validate_every", 50))
    patience = int(getattr(spec, "patience", 0))
    held_out = ()
    if val_fraction > 0.0 and "system_natoms" in pretrain_data:
        held_out = _validation_systems(
            np.asarray(pretrain_data_np["system_natoms"]), val_fraction,
            val_seed)
    monitor = "energy" if energy_weight > 0.0 else "pointwise"
    system_names = [row[0] for row in (_manifest or {}).get("systems") or []]
    validation_record = {
        "fraction": val_fraction, "seed": val_seed,
        "validate_every": validate_every, "patience": patience,
        "monitor": monitor,
        "systems": [system_names[i] if i < len(system_names) else f"sys{i}"
                    for i in held_out],
    }
```

Replace the two trainer calls. The xnet block (currently lines 584-606)
becomes:

```python
    t0 = time.time()
    optimizer_x = _build_optimizer(
        lr_start=spec.lr_start,
        lr_end=spec.lr_end,
        n_steps=spec.n_steps,
        lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
    )
    if held_out:
        seg_x = jnp.concatenate([
            jnp.asarray(pretrain_data["system" + x_suffix], dtype=jnp.int32),
            jnp.full(n_mesh_rows, n_systems, dtype=jnp.int32)]) \
            if n_mesh_rows else jnp.asarray(
                pretrain_data["system" + x_suffix], dtype=jnp.int32)
        tm, vm, trm, vrm, tid, vid = _system_split_arrays(
            np.asarray(seg_x), n_systems, held_out)
        lx_tr, dx_tr, fx_tr = _restrict_loss(loss_fn_x, descriptors,
                                             Fx_target, tm, trm, tid)
        lx_va, dx_va, fx_va = _restrict_loss(loss_fn_x, descriptors,
                                             Fx_target, vm, vrm, vid)
        xnet_trained, losses_x, record_x = _train_pretrain_network(
            xnet, optimizer_x, lx_tr, dx_tr, fx_tr, lx_va, dx_va, fx_va,
            n_steps=spec.n_steps, validate_every=validate_every,
            patience=patience, monitor=monitor,
            progress_callback=_x_callback)
        validation_record["x"] = record_x
    else:
        trainer_x = xcquinox.train.xcTrainer(
            model=xnet,
            optim=optimizer_x,
            loss=loss_fn_x,
            steps=spec.n_steps,
            do_jit=True,
            serialize_every=max(50, spec.n_steps // 10),
            checkpoint_dir=xnet_ckpt_dir,
            progress_callback=_x_callback,
        )
        xnet_trained, losses_x = trainer_x(1, [descriptors], [Fx_target])
    eqx.tree_serialise_leaves(xnet_path, xnet_trained)
```

and the cnet block (currently lines 608-627) becomes the same shape with
`descriptors_c`, `Fc_target`, `loss_fn_c`, `"system_all"`, `cnet_ckpt_dir`,
`_c_callback`, and `validation_record["c"] = record_c`.

Add to the metadata dict, next to the Task 6 keys:

```python
        "validation": validation_record,
```

- [ ] **Step 5: Compile and run**

```bash
python -m py_compile xcquinox/alec/pretrain.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_pretrain_energy_term.py xcquinox/alec/tests/test_pretrain.py xcquinox/alec/tests/test_pretrain_mesh.py -v > /tmp/xcq-testlogs/task07_green.log 2>&1; echo "exit=$?"
```
Expected: PASS except the tests that pass the not-yet-existing `PretrainSpec`
fields, which Task 8 adds.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_pretrain_energy_term.py xcquinox/alec/tests/test_pretrain.py xcquinox/alec/tests/test_pretrain_mesh.py -v > /tmp/xcq-testlogs/task07_green.log 2>&1`

---

## Task 8: The configuration surface

**Files:**
- Modify: `xcquinox/alec/config.py:693-741` (`PretrainSpec`)
- Modify: `xcquinox/alec/cluster/grid_config.py:234-276` (`PretrainConfig`), `:546-559` (`_build_pretrain`), `:1001-1006` (the `loss_weighting` check in `validate_grid_semantics`)
- Modify: `xcquinox/alec/cluster/examples/grid_step7.yaml:84-92` (the `pretrain` section)
- Test: `xcquinox/alec/tests/test_cluster_grid_config.py` (append), `xcquinox/alec/tests/test_cluster_examples.py` (append), `xcquinox/alec/tests/test_config.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks (`grid_config` deliberately imports neither JAX nor PySCF, so the mesh-fraction default is the literal 0.3 with a test pinning it against `pretrain_data_gen.MESH_WEIGHT_FRACTION`).
- Produces: `PretrainConfig` fields `dfs_set`, `pool_atoms`, `parent_density`, `exchange_footing`, `mesh_fraction`, `energy_term_weight`, `validation_fraction`, `validation_seed`, `validate_every`, `patience`; the same six run-time fields on `PretrainSpec`.

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_config.py`:

```python
# ---------------------------------------------------------------------------
# PretrainSpec: pretraining-protocol fields (spec Sections 3.2, 6, 7)
# ---------------------------------------------------------------------------

def test_pretrain_spec_protocol_defaults_reproduce_the_historical_run(tmp_path):
    from xcquinox.alec.config import PretrainSpec, get_architecture
    spec = PretrainSpec(arch=get_architecture("deep_3x16"),
                        data_dir=str(tmp_path),
                        checkpoint_dir=str(tmp_path / "ck"))
    assert spec.parent_density == "pbe"
    assert spec.energy_term_weight == 0.0
    assert spec.validation_fraction == 0.0
    assert spec.validation_seed == 0
    assert spec.validate_every == 50
    assert spec.patience == 0


def test_pretrain_spec_rejects_an_unknown_parent_density(tmp_path):
    import pytest
    from xcquinox.alec.config import PretrainSpec, get_architecture
    with pytest.raises(ValueError, match="parent_density"):
        PretrainSpec(arch=get_architecture("deep_3x16"),
                     data_dir=str(tmp_path),
                     checkpoint_dir=str(tmp_path / "ck"),
                     parent_density="blyp")


def test_pretrain_spec_validate_bounds_the_protocol_fields(tmp_path):
    import pytest
    from xcquinox.alec.config import PretrainSpec, get_architecture
    base = dict(arch=get_architecture("deep_3x16"), data_dir=str(tmp_path),
                checkpoint_dir=str(tmp_path / "ck"))
    with pytest.raises(ValueError, match="energy_term_weight"):
        PretrainSpec(**base, energy_term_weight=-1.0).validate()
    with pytest.raises(ValueError, match="validation_fraction"):
        PretrainSpec(**base, validation_fraction=1.5).validate()
    with pytest.raises(ValueError, match="validate_every"):
        PretrainSpec(**base, validate_every=0).validate()
    with pytest.raises(ValueError, match="patience"):
        PretrainSpec(**base, patience=-1).validate()
```

Append to `xcquinox/alec/tests/test_cluster_grid_config.py`:

```python
# ---------------------------------------------------------------------------
# PretrainConfig: pretraining-protocol fields
# ---------------------------------------------------------------------------

def test_pretrain_config_protocol_defaults_are_todays_behavior():
    from xcquinox.alec.cluster.grid_config import PretrainConfig
    pt = PretrainConfig(data_dir="/d")
    assert pt.dfs_set is False
    assert pt.pool_atoms is False
    assert pt.parent_density == "pbe"
    assert pt.exchange_footing == "total"
    assert pt.mesh_fraction == 0.3
    assert pt.energy_term_weight == 0.0
    assert pt.validation_fraction == 0.0
    assert pt.validation_seed == 0
    assert pt.validate_every == 50
    assert pt.patience == 0


def test_pretrain_config_mesh_fraction_default_matches_the_generator():
    """grid_config imports neither JAX nor PySCF, so the default is written as
    a literal; this pins it against the constant it must equal."""
    from xcquinox.alec.cluster.grid_config import PretrainConfig
    from xcquinox.alec.pretrain_data_gen import MESH_WEIGHT_FRACTION
    assert PretrainConfig(data_dir="/d").mesh_fraction == MESH_WEIGHT_FRACTION


def test_build_pretrain_parses_every_protocol_field():
    """A field missing from _build_pretrain silently reverts to its default on
    every stage that re-reads resolved_config.yaml."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    pt = _build_pretrain({
        "data_dir": "/d", "dfs_set": True, "pool_atoms": True,
        "parent_density": "auto", "exchange_footing": "spin_channel",
        "mesh_fraction": 0.25, "energy_term_weight": 1.0,
        "validation_fraction": 0.2, "validation_seed": 11,
        "validate_every": 25, "patience": 8,
    })
    assert pt.dfs_set is True and pt.pool_atoms is True
    assert pt.parent_density == "auto"
    assert pt.exchange_footing == "spin_channel"
    assert pt.mesh_fraction == 0.25
    assert pt.energy_term_weight == 1.0
    assert pt.validation_fraction == 0.2
    assert pt.validation_seed == 11
    assert pt.validate_every == 25
    assert pt.patience == 8


def test_config_to_raw_dict_round_trips_every_protocol_field(tmp_path):
    """The resolved_config.yaml round trip is what datagen, pretrain, preflight
    and eval all read; a dropped field is a silently reverted run."""
    import dataclasses
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    pt = _build_pretrain({
        "data_dir": "/d", "dfs_set": True, "pool_atoms": True,
        "parent_density": "auto", "exchange_footing": "spin_channel",
        "mesh_fraction": 0.25, "energy_term_weight": 1.0,
        "validation_fraction": 0.2, "validation_seed": 11,
        "validate_every": 25, "patience": 8,
    })
    raw = dataclasses.asdict(pt)
    assert _build_pretrain(raw) == pt
    for f in dataclasses.fields(pt):
        assert f.name in raw, f.name


def test_validate_grid_semantics_bounds_the_protocol_fields():
    import dataclasses
    import pytest
    from xcquinox.alec.cluster.grid_config import validate_grid_semantics
    cfg = _cfg()                       # the module's GridConfig builder
    domain = _StubDomain(pool_size=40)  # as in test_validate_ok
    for field, value, message in (
            ("parent_density", "blyp", "parent_density"),
            ("exchange_footing", "per_orbital", "exchange_footing"),
            ("mesh_fraction", 1.0, "mesh_fraction"),
            ("energy_term_weight", -1.0, "energy_term_weight"),
            ("validation_fraction", 1.0, "validation_fraction"),
            ("validate_every", 0, "validate_every"),
            ("patience", -1, "patience"),
    ):
        bad = dataclasses.replace(
            cfg, pretrain=dataclasses.replace(cfg.pretrain, **{field: value}))
        with pytest.raises(ValueError, match=message):
            validate_grid_semantics(bad, domain)
```

Append to `xcquinox/alec/tests/test_cluster_examples.py`:

```python
def test_example_sets_every_pretraining_protocol_field():
    """The shipped template names each pretraining-protocol knob explicitly at
    its default, with the v6 value in a comment, so a copy of it is a complete
    statement of the protocol rather than a set of invisible defaults."""
    pytest.importorskip("yaml")
    raw = _raw_yaml(_example_path())["pretrain"]
    for key, value in (("dfs_set", False), ("pool_atoms", False),
                       ("parent_density", "pbe"),
                       ("exchange_footing", "total"),
                       ("mesh_fraction", 0.3), ("energy_term_weight", 0.0),
                       ("validation_fraction", 0.0), ("validation_seed", 0),
                       ("validate_every", 50), ("patience", 0)):
        assert key in raw, f"grid_step7.yaml is missing pretrain.{key}"
        assert raw[key] == value, (key, raw[key], value)
    cfg = load_grid_config(_example_path())
    assert cfg.pretrain.dfs_set is False
    assert cfg.pretrain.parent_density == "pbe"
```

- [ ] **Step 2: Run and confirm it fails**

```bash
python -m pytest xcquinox/alec/tests/test_config.py xcquinox/alec/tests/test_cluster_grid_config.py xcquinox/alec/tests/test_cluster_examples.py -k "protocol or pretrain_config or build_pretrain or round_trips" -v > /tmp/xcq-testlogs/task08_red.log 2>&1; echo "exit=$?"
```
Expected: `TypeError: PretrainSpec.__init__() got an unexpected keyword
argument 'parent_density'` and `AssertionError: grid_step7.yaml is missing
pretrain.dfs_set`.

- [ ] **Step 3: Extend `PretrainSpec`**

In `xcquinox/alec/config.py`, add after `loss_weighting` (line 708):

```python
    # --- Pretraining protocol (spec Sections 3.2, 6, 7) -------------------
    # Every default reproduces the pre-protocol run exactly, so an existing
    # spec is unchanged and the new behavior is opt-in per YAML.
    #
    # The parent functional whose SELF-CONSISTENT density the pretrain data
    # sits on. "auto" resolves to the architecture's rung baseline (SCAN for
    # the meta-GGA rung, PBE otherwise); "pbe" keeps every architecture on the
    # PBE-density file, which is what every file written before this change
    # is.
    parent_density: str = "pbe"
    # Weight of the per-system energy term, in inverse Hartree^2. The term is
    # mean_s (E_xc^NN_s - E_xc^parent_s)^2, so w_E = 1 makes a 1 mHa mean
    # energy error worth 1e-6, the order of the converged point-wise residual.
    # 0.0 = the point-wise objective alone, byte-identical to the prior loss.
    energy_term_weight: float = 0.0
    # Fraction of the MULTI-NUCLEUS systems withheld from the fit and scored
    # between optimizer steps. 0.0 = no split and no stop criterion.
    validation_fraction: float = 0.0
    # Seed of the held-out permutation. Separate from ``seed`` (the network
    # initialization) so every architecture in a sweep holds out the same
    # systems and their validation numbers are comparable.
    validation_seed: int = 0
    # Optimizer steps between validations.
    validate_every: int = 50
    # Validations without improvement before training stops. 0 = no early
    # stop; the best weights are still the ones kept.
    patience: int = 0
```

Extend `__post_init__`:

```python
    def __post_init__(self) -> None:
        if self.loss_weighting not in ("unweighted", "integration"):
            raise ValueError(
                f"loss_weighting must be 'unweighted' or 'integration', "
                f"got {self.loss_weighting!r}"
            )
        if self.parent_density not in ("pbe", "scan", "auto"):
            raise ValueError(
                f"parent_density must be 'pbe', 'scan' or 'auto', got "
                f"{self.parent_density!r}"
            )
```

and `validate`, after the `grad_clip` check:

```python
        if self.energy_term_weight < 0:
            raise ValueError(
                f"energy_term_weight must be >= 0, got "
                f"{self.energy_term_weight}")
        if not (0.0 <= self.validation_fraction < 1.0):
            raise ValueError(
                f"validation_fraction must be in [0, 1), got "
                f"{self.validation_fraction}")
        if self.validate_every <= 0:
            raise ValueError(
                f"validate_every must be > 0, got {self.validate_every}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
```

- [ ] **Step 4: Extend `PretrainConfig` and its parser**

In `xcquinox/alec/cluster/grid_config.py`, add after `atoms: tuple = ()`
(line 275):

```python
    # --- Pretraining protocol (spec Sections 3.2, 6, 7) -------------------
    # The set. ``dfs_set`` adds the DFS pretraining inventory in its entirety
    # (8 free atoms and 22 G2/97 molecules for the GGA rung, 20 for the
    # meta-GGA rung); ``pool_atoms`` adds every single-atom species of the
    # BH76 and W4-11 pools with its production charge and spin. Turning either
    # on REPLACES the historical four-atom default, which ``atoms`` can still
    # extend. Both default False, so an existing YAML is unchanged.
    dfs_set: bool = False
    pool_atoms: bool = False
    # The density the targets sit on: "pbe", "scan", or "auto" for the
    # architecture's rung baseline. "pbe" is every file written before this
    # change; "auto" splits a mixed-rung sweep across two data files.
    parent_density: str = "pbe"
    # How OPEN-SHELL exchange rows are posed. "spin_channel" is the exact
    # spin-scaling footing the production UKS exchange evaluates, per channel
    # at (2 rho_sigma, 4 sigma_sigma_sigma, features of diag(P_sigma,
    # P_sigma)); "total" is the historical total-density footing. The footing
    # is part of the data's identity, so a change regenerates the file.
    exchange_footing: str = "total"
    # Share of the total integration weight carried by the synthetic
    # (r_s, s, alpha) mesh, which is kept as a regularizer only. Must equal
    # pretrain_data_gen.MESH_WEIGHT_FRACTION's historical 0.3 to reproduce
    # existing data; written as a literal because this module deliberately
    # imports neither JAX nor PySCF.
    mesh_fraction: float = 0.3
    # The objective. energy_term_weight is the weight of the per-system energy
    # term in inverse Hartree^2; 0.0 is the point-wise objective alone.
    energy_term_weight: float = 0.0
    # Validation and the stop criterion.
    validation_fraction: float = 0.0
    validation_seed: int = 0
    validate_every: int = 50
    patience: int = 0
```

Replace `_build_pretrain` (lines 546-559) with:

```python
def _build_pretrain(d: dict) -> PretrainConfig:
    ctx = "pretrain"
    return PretrainConfig(
        data_dir=_require(d, "data_dir", ctx),
        n_steps=d.get("n_steps", 1000),
        lr_start=d.get("lr_start", 1e-2),
        lr_end=d.get("lr_end", 1e-5),
        lr_decay_start=d.get("lr_decay_start", 0.2),
        grad_clip=d.get("grad_clip", 1.0),
        seed=d.get("seed", 42),
        loss_weighting=d.get("loss_weighting", "integration"),
        atoms=_parse_pretrain_atoms(d.get("atoms")),
        dfs_set=bool(d.get("dfs_set", False)),
        pool_atoms=bool(d.get("pool_atoms", False)),
        parent_density=str(d.get("parent_density", "pbe")),
        exchange_footing=str(d.get("exchange_footing", "total")),
        mesh_fraction=float(d.get("mesh_fraction", 0.3)),
        energy_term_weight=float(d.get("energy_term_weight", 0.0)),
        validation_fraction=float(d.get("validation_fraction", 0.0)),
        validation_seed=int(d.get("validation_seed", 0)),
        validate_every=int(d.get("validate_every", 50)),
        patience=int(d.get("patience", 0)),
    )
```

- [ ] **Step 5: Bound them in `validate_grid_semantics`**

In `validate_grid_semantics`, after the `loss_weighting` check (line 1006),
insert:

```python
    if pt.parent_density not in ("pbe", "scan", "auto"):
        raise ValueError(
            f"pretrain.parent_density must be 'pbe', 'scan' or 'auto', got "
            f"{pt.parent_density!r}"
        )
    if pt.exchange_footing not in ("total", "spin_channel"):
        raise ValueError(
            f"pretrain.exchange_footing must be 'total' or 'spin_channel', "
            f"got {pt.exchange_footing!r}"
        )
    if not (0.0 <= pt.mesh_fraction < 1.0):
        raise ValueError(
            f"pretrain.mesh_fraction must be in [0, 1), got "
            f"{pt.mesh_fraction}"
        )
    if pt.energy_term_weight < 0:
        raise ValueError(
            f"pretrain.energy_term_weight must be >= 0, got "
            f"{pt.energy_term_weight}"
        )
    if not (0.0 <= pt.validation_fraction < 1.0):
        raise ValueError(
            f"pretrain.validation_fraction must be in [0, 1), got "
            f"{pt.validation_fraction}"
        )
    if pt.validate_every <= 0:
        raise ValueError(
            f"pretrain.validate_every must be > 0, got {pt.validate_every}"
        )
    if pt.patience < 0:
        raise ValueError(
            f"pretrain.patience must be >= 0, got {pt.patience}"
        )
```

- [ ] **Step 6: Name every field in the shipped example**

Replace the `pretrain:` block of
`xcquinox/alec/cluster/examples/grid_step7.yaml` (lines 84-92) with:

```yaml
pretrain:
  data_dir: /gpfs/projects/CHANGE_ME/xcquinox/step7/pretrain_data
  n_steps: 1000
  lr_start: 0.01            # 1e-2
  lr_end: 0.00001           # 1e-5
  lr_decay_start: 0.2       # fraction of n_steps
  grad_clip: 1.0
  seed: 42
  loss_weighting: integration
  # --- pretraining protocol ------------------------------------------------
  # Every value below is the pre-protocol default, so this template reproduces
  # the step-7 pretraining exactly. The campaign-v6 value follows each one.
  #
  # The SET. dfs_set adds the DFS pretraining inventory in its entirety;
  # pool_atoms adds every single-atom species of the BH76 and W4-11 pools with
  # its production charge and spin. Either one replaces the historical
  # four-atom default, which `atoms` can still extend.
  dfs_set: false            # v6: true
  pool_atoms: false         # v6: true
  # The DENSITY the targets sit on. "auto" gives each architecture its rung
  # baseline (SCAN for the meta-GGA rung, PBE otherwise) and splits a
  # mixed-rung sweep across two data files.
  parent_density: pbe       # v6: auto
  # The open-shell exchange FOOTING. "spin_channel" poses each channel at
  # (2 rho_sigma, 4 sigma_sigma_sigma, features of diag(P_sigma, P_sigma)),
  # which is what the production UKS exchange evaluates.
  exchange_footing: total   # v6: spin_channel
  # The synthetic (r_s, s, alpha) mesh's share of the total integration
  # weight. Kept as a regularizer only.
  mesh_fraction: 0.3        # v6: 0.3
  # The OBJECTIVE. energy_term_weight weights the per-system energy term
  # mean_s (E_xc^NN_s - E_xc^parent_s)^2 in inverse Hartree^2, beside the
  # point-wise enhancement-factor residual.
  energy_term_weight: 0.0   # v6: see HISTORY (chosen by measured sweep)
  # VALIDATION and the stop criterion: a seeded fraction of the multi-nucleus
  # systems is withheld and scored every validate_every steps; training stops
  # after `patience` validations without improvement and keeps the best
  # weights.
  validation_fraction: 0.0  # v6: 0.2
  validation_seed: 0        # v6: 0
  validate_every: 50        # v6: 50
  patience: 0               # v6: 10
```

- [ ] **Step 7: Compile and run**

```bash
python -m py_compile xcquinox/alec/config.py xcquinox/alec/cluster/grid_config.py && echo compiled
python -m pytest xcquinox/alec/tests/test_config.py xcquinox/alec/tests/test_cluster_grid_config.py xcquinox/alec/tests/test_cluster_examples.py xcquinox/alec/tests/test_cluster_cli.py xcquinox/alec/tests/test_pretrain_energy_term.py -v > /tmp/xcq-testlogs/task08_green.log 2>&1; echo "exit=$?"
```
Expected: PASS, including the `run_pretrain` tests deferred from Tasks 6 and 7.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_config.py xcquinox/alec/tests/test_cluster_grid_config.py xcquinox/alec/tests/test_cluster_examples.py xcquinox/alec/tests/test_cluster_cli.py xcquinox/alec/tests/test_pretrain_energy_term.py -v > /tmp/xcq-testlogs/task08_green.log 2>&1`

---

## Task 9: Harness threading -- pretrain stage, datagen stage, preflight

**Files:**
- Modify: `xcquinox/alec/cluster/_pretrain.py:268-290` (the `PretrainSpec` build and the log line)
- Modify: `xcquinox/alec/cluster/_datagen.py:48-123` (the required-file resolution and `main`)
- Modify: `xcquinox/alec/cluster/inputs.py:345-355` (the belt-and-braces ensure call)
- Test: `xcquinox/alec/tests/test_cluster_pretrain.py` (append), `xcquinox/alec/tests/test_cluster_datagen.py` (append)

**Interfaces:**
- Consumes: `PretrainConfig` (Task 8); `resolve_parent_density`, `pretrain_data_filename` (Task 1).
- Produces: `_datagen._required_data_specs(cfg) -> list[tuple[bool, str]]` -- the distinct `(polarized, reference_xc)` pairs the sweep's architectures consume.

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_cluster_datagen.py`:

```python
# ---------------------------------------------------------------------------
# Pretraining-protocol plumbing
# ---------------------------------------------------------------------------

def _cfg2(archs, polarized, **pretrain_kw):
    pt = dict(data_dir="/d/pt", atoms=(), dfs_set=False, pool_atoms=False,
              parent_density="pbe", exchange_footing="total",
              mesh_fraction=0.3)
    pt.update(pretrain_kw)
    return _ns(
        sweep=_ns(arch=list(archs)),
        use_polarized_correlation=polarized,
        pretrain=_ns(**pt),
        inputs=_ns(basis="def2-svp", grid_level=3, density_fit=False,
                   auxbasis=None),
    )


def test_required_data_specs_single_parent():
    cfg = _cfg2(["deep_3x16", "deep_mgga_3x16"], True)
    assert _datagen._required_data_specs(cfg) == [(True, "pbe")]


def test_required_data_specs_auto_splits_a_mixed_rung_sweep():
    """With parent_density: auto a GGA-rung arch wants the PBE-density file and
    a meta-GGA-rung arch wants the SCAN-density file, so datagen builds both."""
    cfg = _cfg2(["deep_3x16", "deep_mgga_3x16"], True, parent_density="auto")
    assert _datagen._required_data_specs(cfg) == [(True, "pbe"),
                                                  (True, "scan")]


def test_required_data_specs_auto_single_rung():
    cfg = _cfg2(["deep_mgga_3x16"], True, parent_density="auto")
    assert _datagen._required_data_specs(cfg) == [(True, "scan")]


def test_main_threads_every_protocol_knob(monkeypatch, tmp_path):
    cfg = _cfg2(["deep_3x16"], True, dfs_set=True, pool_atoms=True,
                exchange_footing="spin_channel", mesh_fraction=0.25)
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    assert len(calls) == 1
    _dd, kw = calls[0]
    assert kw["dfs_set"] is True
    assert kw["pool_atoms"] is True
    assert kw["reference_xc"] == "pbe"
    assert kw["exchange_footing"] == "spin_channel"
    assert kw["mesh_fraction"] == 0.25


def test_main_default_call_is_unchanged(monkeypatch, tmp_path):
    """A YAML written before the protocol change must reach the generator with
    exactly the keyword set it always did, so its data file is not
    regenerated."""
    cfg = _cfg(["deep", "deep_attn"], True, basis="def2-svp", grid=2,
               df=False, data_dir="/d/svp")
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    assert calls[0][1] == {"basis": "def2-svp", "grid_level": 2,
                           "density_fit": False, "auxbasis": None,
                           "polarized": True, "descriptors": True}
```

`_cfg` (the existing helper) builds a config whose `pretrain` namespace has
only `data_dir`, which is what a pre-protocol `resolved_config.yaml` reloads
to under `getattr(..., default)` access; that is what
`test_main_default_call_is_unchanged` exercises.

Append to `xcquinox/alec/tests/test_cluster_pretrain.py`:

```python
def test_pretrain_spec_carries_the_protocol_fields(tmp_path, monkeypatch):
    """A field the worker forgets to thread is a knob the YAML sets and the run
    silently ignores."""
    from xcquinox.alec.cluster import _pretrain as pretrain_mod
    d = tmp_path / "run"
    d.mkdir()
    data_dir = tmp_path / "pretrain_data"
    data_dir.mkdir()
    cfg = _config_dict(data_dir=str(data_dir))
    cfg["pretrain"].update({
        "parent_density": "auto", "energy_term_weight": 1.0,
        "validation_fraction": 0.2, "validation_seed": 11,
        "validate_every": 25, "patience": 8})
    _write_config(str(d), cfg)
    captured = {}

    def _fake(spec, progress_callback=None):
        captured["spec"] = spec
        return {}

    monkeypatch.setattr(pretrain_mod, "_run_pretrain", _fake)
    pretrain_mod.main([str(d), "0"])
    spec = captured["spec"]
    assert spec.parent_density == "auto"
    assert spec.energy_term_weight == 1.0
    assert spec.validation_fraction == 0.2
    assert spec.validation_seed == 11
    assert spec.validate_every == 25
    assert spec.patience == 8
```

`_config_dict` and `_write_config` are the module's existing helpers. The
worker's `xnet.eqx` / `cnet.eqx` guard makes `main` return non-zero here, which
is fine: the test reads the captured spec, not the exit code.

- [ ] **Step 2: Run and confirm it fails**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_datagen.py xcquinox/alec/tests/test_cluster_pretrain.py -v > /tmp/xcq-testlogs/task09_red.log 2>&1; echo "exit=$?"
```
Expected: `AttributeError: module 'xcquinox.alec.cluster._datagen' has no
attribute '_required_data_specs'` and `AttributeError: 'PretrainSpec' object
has no attribute ...` in the worker test.

- [ ] **Step 3: Thread the pretrain worker**

In `xcquinox/alec/cluster/_pretrain.py`, extend the `PretrainSpec(...)` call
(lines 274-285) with:

```python
        parent_density=getattr(pt, "parent_density", "pbe"),
        energy_term_weight=getattr(pt, "energy_term_weight", 0.0),
        validation_fraction=getattr(pt, "validation_fraction", 0.0),
        validation_seed=getattr(pt, "validation_seed", 0),
        validate_every=getattr(pt, "validate_every", 50),
        patience=getattr(pt, "patience", 0),
```

and extend the log line (lines 286-290) to:

```python
    _log(
        arch_name,
        f"running run_pretrain: n_steps={pt.n_steps}, "
        f"loss_weighting={pt.loss_weighting!r}, "
        f"parent_density={getattr(pt, 'parent_density', 'pbe')!r}, "
        f"energy_term_weight={getattr(pt, 'energy_term_weight', 0.0)}, "
        f"validation_fraction={getattr(pt, 'validation_fraction', 0.0)}, "
        f"checkpoint_dir={checkpoint_dir}",
    )
```

The `getattr` defaults keep the worker able to read a `resolved_config.yaml`
written before this change, which is what recovery and resubmit paths do.

- [ ] **Step 4: Thread the datagen stage**

In `xcquinox/alec/cluster/_datagen.py`, add after `_required_polarized_flags`
(line 66):

```python
def _required_data_specs(cfg):
    """The distinct ``(polarized, reference_xc)`` pretrain-data files needed.

    The polarization flag decides whether the file carries the zeta column; the
    parent decides which functional's SELF-CONSISTENT density the rows sit on.
    Under ``pretrain.parent_density: auto`` the parent is the architecture's
    rung baseline, so a sweep that mixes GGA-rung and meta-GGA-rung
    architectures needs BOTH files -- they are different densities, not two
    views of one.
    """
    import dataclasses as _dc

    from xcquinox.alec.pretrain_data_gen import resolve_parent_density

    run_polarized = bool(getattr(cfg, "use_polarized_correlation", False))
    requested = getattr(cfg.pretrain, "parent_density", "pbe")
    specs = {}
    for name in cfg.sweep.arch:
        arch = get_architecture(name)
        if run_polarized:
            arch = _dc.replace(arch, use_polarized_correlation=True)
        polarized = _pretrain_data_filename(arch).endswith("_polarized.npz")
        specs.setdefault((polarized, resolve_parent_density(arch, requested)),
                         None)
    return sorted(specs)
```

Replace the generation loop in `main` (lines 94-123) with:

```python
    data_dir = cfg.pretrain.data_dir
    from xcquinox.alec.pretrain_data_gen import pretrain_data_filename
    specs = _required_data_specs(cfg)
    required = [pretrain_data_filename(p, ref) for p, ref in specs]
    pt = cfg.pretrain
    # Only knobs that DIFFER from the generator's defaults are passed, so a
    # configuration written before the pretraining protocol change reaches the
    # generator with exactly the keyword set it always did and its data file is
    # not regenerated.
    extra = {}
    if getattr(pt, "atoms", ()):
        extra["atoms"] = tuple(tuple(a) for a in pt.atoms)
    if getattr(pt, "dfs_set", False):
        extra["dfs_set"] = True
    if getattr(pt, "pool_atoms", False):
        extra["pool_atoms"] = True
    if getattr(pt, "exchange_footing", "total") != "total":
        extra["exchange_footing"] = str(pt.exchange_footing)
    if float(getattr(pt, "mesh_fraction", 0.3)) != 0.3:
        extra["mesh_fraction"] = float(pt.mesh_fraction)
    _log(
        f"archs={list(cfg.sweep.arch)} -> required: {required} | "
        f"basis={cfg.inputs.basis} grid_level={cfg.inputs.grid_level} "
        f"density_fit={cfg.inputs.density_fit} data_dir={data_dir} | "
        f"protocol={extra}"
    )
    try:
        for polarized, reference_xc in specs:
            call = dict(extra)
            # The reference density is named only when the call is not the
            # historical one, so a pre-protocol configuration reaches the
            # generator with exactly the keyword set it always did.
            if call or reference_xc != "pbe":
                call["reference_xc"] = reference_xc
            path = _ensure_pretrain_data(
                data_dir,
                basis=cfg.inputs.basis,
                grid_level=cfg.inputs.grid_level,
                density_fit=cfg.inputs.density_fit,
                auxbasis=cfg.inputs.auxbasis,
                polarized=polarized,
                descriptors=True,
                **call,
            )
            _log(f"ensured pretrain data (polarized={polarized}, "
                 f"reference_xc={reference_xc}): {path}")
    except Exception as exc:  # noqa: BLE001, fail the stage loudly + non-zero.
        _log(f"ERROR: pretrain-data generation failed: "
             f"{type(exc).__name__}: {exc}")
        return 1
```

The per-iteration `call = dict(extra)` copy matters: mutating `extra` in the
loop would leak one iteration's `reference_xc` into the next, which on a
mixed-rung sweep would build the SCAN file twice and never build the PBE one.

- [ ] **Step 5: Thread the preflight's belt-and-braces call**

In `xcquinox/alec/cluster/inputs.py`, replace the `_ensure_pretrain_data(...)`
call (lines 345-355) with the same conditional-keyword construction, resolving
the parent from the run-level `pretrain.parent_density` and the FIRST swept
architecture's rung. When the sweep mixes rungs under `auto`, loop over
`_datagen._required_data_specs(cfg)` instead of duplicating the resolution:

```python
        from xcquinox.alec.cluster._datagen import _required_data_specs
        _pt = cfg.pretrain
        _extra = {}
        if getattr(_pt, "atoms", ()):
            _extra["atoms"] = tuple(tuple(a) for a in _pt.atoms)
        if getattr(_pt, "dfs_set", False):
            _extra["dfs_set"] = True
        if getattr(_pt, "pool_atoms", False):
            _extra["pool_atoms"] = True
        if getattr(_pt, "exchange_footing", "total") != "total":
            _extra["exchange_footing"] = str(_pt.exchange_footing)
        if float(getattr(_pt, "mesh_fraction", 0.3)) != 0.3:
            _extra["mesh_fraction"] = float(_pt.mesh_fraction)
        for _polarized, _reference_xc in _required_data_specs(cfg):
            _call = dict(_extra)
            if _call or _reference_xc != "pbe":
                _call["reference_xc"] = _reference_xc
            _ensure_pretrain_data(
                cfg.pretrain.data_dir,
                basis=cfg.inputs.basis,
                grid_level=cfg.inputs.grid_level,
                density_fit=cfg.inputs.density_fit,
                auxbasis=cfg.inputs.auxbasis,
                polarized=_polarized,
                **_call,
            )
```

The preflight call historically passed `polarized=cfg.use_polarized_correlation`
directly; going through `_required_data_specs` makes the preflight and the
datagen stage agree on which files exist by construction.

- [ ] **Step 6: Compile and run**

```bash
python -m py_compile xcquinox/alec/cluster/_pretrain.py xcquinox/alec/cluster/_datagen.py xcquinox/alec/cluster/inputs.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_datagen.py xcquinox/alec/tests/test_cluster_pretrain.py xcquinox/alec/tests/test_cluster_inputs.py xcquinox/alec/tests/test_cluster_preflight.py -v > /tmp/xcq-testlogs/task09_green.log 2>&1; echo "exit=$?"
```
Expected: PASS, with the three pre-existing `_required_polarized_flags` tests
unchanged.

- [ ] **Step 7: Record the datagen wall estimate**

The measured per-species precompute wall at the PRODUCTION identity
(6-311++G(3df,2pd), grid level 3, from `scratch/probe_pretrain_gga_rungs.log`)
is H 0.7 s, Li 0.6 s, O 1.1 s, C 1.4 s, N 2.0 s, H2O 3.4 s (32136 grid
points), N2 3.5 s (26616), CH4 5.2 s (49408). The v6 set adds molecules well
past CH4 -- C3H8 (11 nuclei), SiCH6 (8), C4H6 (10), AlCl3 (4 heavy nuclei at a
3df basis), Si2 -- whose grids and basis sets are several times larger, and the
SCAN file's SCF is the slower of the two. Add to the `cluster:` section of
`grid_step7.yaml`, next to the other per-stage walls:

```yaml
  # Datagen builds the pretrain-data file(s). At the production identity the
  # measured precompute wall is 0.7 s for H and 5.2 s for CH4 (49408 grid
  # points); the v6 set's largest molecules (C3H8, C4H6, SiCH6, AlCl3, Si2) are
  # several times that, and parent_density: auto builds a second file on SCAN
  # densities, whose SCF is the slower one. A 4 h wall is roughly an order of
  # magnitude of margin on one job that every later stage waits for.
  datagen_time: "04:00:00"
```

and add the assertion to `test_cluster_examples.py`:

```python
def test_example_gives_datagen_a_wall_of_its_own():
    pytest.importorskip("yaml")
    cfg = load_grid_config(_example_path())
    assert cfg.cluster.datagen_time == "04:00:00"
```

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_examples.py -v > /tmp/xcq-testlogs/task09_examples.log 2>&1; echo "exit=$?"
```
Expected: PASS.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_cluster_datagen.py xcquinox/alec/tests/test_cluster_pretrain.py xcquinox/alec/tests/test_cluster_inputs.py xcquinox/alec/tests/test_cluster_preflight.py xcquinox/alec/tests/test_cluster_examples.py -v > /tmp/xcq-testlogs/task09_green.log 2>&1`

---

## Task 10: Choose the energy-term weight by measurement

**Files:**
- Create: `scratch/probe_pretrain_energy_weight.py` (untracked probe; the repository already keeps its measurement probes there)
- Modify: `xcquinox/alec/cluster/examples/grid_step7.yaml` (replace the `energy_term_weight` v6 comment with the measured value)

**Interfaces:**
- Consumes: `ensure_pretrain_data`, `run_pretrain` and the metadata of Tasks 5-9.
- Produces: the measured v6 value of `pretrain.energy_term_weight`, and the numbers the Task 11 HISTORY entry quotes.

**Why a sweep.** The term's weight is dimensionful (inverse Hartree^2) and its
right value is the one that buys the certificate's `tol_atom = 1.0 mHa` without
destroying the point-wise fit. The metadata already reports
`energy_term_{x,c}_final`, which is `mean_s (E^NN_s - E^parent_s)^2` in
Hartree^2, so `sqrt(energy_term_x_final + energy_term_c_final)` is the RMS
per-system E_xc error in Hartree -- the certificate's quantity, measured on the
pretraining rows themselves and free. The certificate of Section 3.3 remains
the arbiter; this sweep only picks the starting point so the certificate is not
run blind.

- [ ] **Step 1: Write the probe**

Create `scratch/probe_pretrain_energy_weight.py`:

```python
"""Sweep the per-system energy-term weight of the pretraining objective.

Reports, per weight, the final point-wise pretrain losses and the RMS
per-system E_xc error in mHa, which is the quantity the Section 3.3 certificate
bounds at tol_atom = 1.0 mHa. Run at a reduced identity so the sweep is
minutes; the chosen weight is confirmed once at the production identity.

    python scratch/probe_pretrain_energy_weight.py > \
        scratch/probe_pretrain_energy_weight.log 2>&1
"""
import math
import os
import sys
import tempfile
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

from xcquinox.alec.config import PretrainSpec, get_architecture
from xcquinox.alec.pretrain import run_pretrain
from xcquinox.alec.pretrain_data_gen import ensure_pretrain_data

BASIS = "def2-svp"
GRID = 1
ARCHS = ("deep_3x16", "deep_cusp_3x16", "deep_rung35_3x16")
WEIGHTS = (0.0, 0.1, 1.0, 10.0, 100.0)
N_STEPS = 1000


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.time()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "probe_energy_weight_data")
    os.makedirs(data_dir, exist_ok=True)
    path = ensure_pretrain_data(
        data_dir, basis=BASIS, grid_level=GRID, polarized=True,
        descriptors=True, dfs_set=True, pool_atoms=True, reference_xc="pbe",
        exchange_footing="spin_channel", progress=True)
    log(f"# data: {path} (t={time.time() - t0:.1f}s)")
    log("# arch                 w_E     loss_x      loss_c    "
        "rms_dE_xc/mHa  steps")
    for arch_name in ARCHS:
        arch = get_architecture(arch_name)
        for w in WEIGHTS:
            with tempfile.TemporaryDirectory() as ck:
                spec = PretrainSpec(
                    arch=arch, data_dir=data_dir, checkpoint_dir=ck,
                    n_steps=N_STEPS, seed=0, loss_weighting="integration",
                    energy_term_weight=w, parent_density="pbe")
                md = run_pretrain(spec)
            rms = 1000.0 * math.sqrt(md["energy_term_x_final"]
                                     + md["energy_term_c_final"])
            log(f"{arch_name:22s} {w:6.1f} {md['final_loss_x']:.4e} "
                f"{md['final_loss_c']:.4e} {rms:12.4f} "
                f"{md['pretrain_steps']:6d}")
    log(f"# TOTAL WALL = {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the sweep**

```bash
python scratch/probe_pretrain_energy_weight.py > /tmp/xcq-testlogs/task10_sweep.log 2>&1; echo "exit=$?"
```
Read the log with `Read`. Expected shape: `rms_dE_xc` falls steeply from
`w_E = 0` and flattens; `loss_x` / `loss_c` rise slowly. If the run exceeds an
hour, drop `ARCHS` to the first two and `N_STEPS` to 500 and re-run -- the
comparison is between weights, not against an absolute loss.

- [ ] **Step 3: Choose the weight**

Rule, stated so the choice is reproducible rather than aesthetic: take the
SMALLEST swept weight for which every architecture's `rms_dE_xc` is at or below
0.5 mHa -- half the certificate's `tol_atom`, so the production-identity
evaluation has margin -- AND no architecture's `loss_x` or `loss_c` has risen by
more than a factor of three from its `w_E = 0` value. If no swept weight
satisfies both, report the trade-off in the log, take the weight that minimizes
`rms_dE_xc` subject to the factor-of-three cap, and say so in HISTORY: a
pretraining that cannot reach 0.5 mHa on its own rows will not pass the
certificate, and that is a finding, not a tuning failure.

Replace the placeholder in `xcquinox/alec/cluster/examples/grid_step7.yaml`:

```yaml
  energy_term_weight: 0.0   # v6: <MEASURED> (see HISTORY 2026-08-21)
```

with the chosen number in place of `<MEASURED>`.

- [ ] **Step 4: Confirm once at the production identity**

Re-run the probe with `BASIS = "6-311++G(3df,2pd)"`, `GRID = 3`,
`ARCHS = ("deep_3x16",)` and `WEIGHTS = (0.0, <chosen>)`, into
`scratch/probe_pretrain_energy_weight_production.log`. This is the wall-clock
measurement the Task 9 `datagen_time` guidance is checked against: record the
data-generation time the probe's first line reports and, if it exceeds 2 h,
raise `datagen_time` and the comment beside it.

```bash
cd /home/awills/Documents/Research/xcquinox && python scratch/probe_pretrain_energy_weight.py > /tmp/xcq-testlogs/task10_production.log 2>&1; echo "exit=$?"
```
Read the log. Record in HISTORY: the data-generation wall, the number of
systems, the row counts (`n_rows_x`, `n_rows_c` in the metadata), and the
`rms_dE_xc` at both weights.

**Covering test command:** this task produces measurements rather than code; its
gate is Task 11's full-suite run. Confirm no tracked file other than the example
YAML changed: `python -m pytest xcquinox/alec/tests/test_cluster_examples.py -v > /tmp/xcq-testlogs/task10_examples.log 2>&1`

---

## Task 11: Full-suite run, the acceptance hook, and the HISTORY entry

**Files:**
- Modify: `xcquinox/alec/HISTORY.md` (prepend a dated entry in the file's existing format)
- Modify: `xcquinox/alec/SPEC_pretrain_fidelity_program.md:169` (mark step 2 of Section 5 done)

**Interfaces:**
- Consumes: every task's green log and Task 10's measurements.
- Produces: the development record, and the statement of which metadata fields the Section 3.3 certificate consumes.

- [ ] **Step 1: Run the whole alec suite**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests -v > /tmp/xcq-testlogs/task11_full.log 2>&1; echo "exit=$?"
```
Read the log with `Read`. Expected: no failures and no errors. Any failure here
is a call site of a changed signature that no earlier task's log covered
(`ensure_pretrain_data`'s `atoms=None` sentinel and
`_assemble_pretrain_descriptors`'s `_key_map` are the two most likely); fix it
and re-run before writing the entry.

- [ ] **Step 2: Run the slow tests once**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests -m slow -v > /tmp/xcq-testlogs/task11_slow.log 2>&1; echo "exit=$?"
```
Expected: PASS. `test_pretrain_mesh.py::test_generator_writes_mesh_keys_and_weight_share`
lives here and exercises the real generator end to end.

- [ ] **Step 3: Confirm the acceptance hook's inputs exist**

The Section 3.3 plan owns the `fidelity_certificate(cfg, run_dir, arch_name)`
call at the end of the pretrain stage. What THIS plan owes it is the provenance
in `<run_dir>/pretrain/<arch>/pretrain_metadata.json`. Confirm every field is
present on a real run:

```bash
cd /home/awills/Documents/Research/xcquinox && python -c "
import json, tempfile, os
from xcquinox.alec.config import PretrainSpec, get_architecture
from xcquinox.alec.pretrain import run_pretrain
from xcquinox.alec.pretrain_data_gen import ensure_pretrain_data
d = tempfile.mkdtemp()
ensure_pretrain_data(d, atoms=(('He', 0), ('H', 1)), basis='sto-3g',
                     grid_level=0, polarized=False, descriptors=True,
                     exchange_footing='spin_channel')
ck = os.path.join(d, 'ck')
md = run_pretrain(PretrainSpec(arch=get_architecture('deep_3x16'), data_dir=d,
                               checkpoint_dir=ck, n_steps=3, seed=0,
                               loss_weighting='integration',
                               energy_term_weight=1.0))
need = ['reference_xc', 'exchange_footing', 'energy_term_weight',
        'n_systems', 'n_rows_x', 'n_rows_c', 'energy_term_x_final',
        'energy_term_c_final', 'validation']
print(json.dumps({k: md[k] for k in need}, indent=2))
" > /tmp/xcq-testlogs/task11_metadata.log 2>&1; echo "exit=$?"
```
Read the log. Every key must be present and finite; `validation` must carry
`fraction`, `seed`, `validate_every`, `patience`, `monitor` and `systems`.

- [ ] **Step 4: Write the HISTORY entry**

Prepend to `xcquinox/alec/HISTORY.md`, in the file's existing entry format:

```markdown
### 2026-08-21 -- Pretraining that delivers the parent

**What changed.** The pretraining set, its footing, its objective and its stop
criterion. `pretrain_data_gen` gained a composition layer (`PretrainSystem`,
`normalize_system`, `pool_atom_systems`, `resolve_pretrain_systems`) and a
general per-system column builder `_system_columns`, of which `_atom_columns`
and `_molecule_columns` are the two wrappers, so atomic and molecular rows are
the same quantity by construction. The set is now configurable:
`pretrain.dfs_set` adds the DFS pretraining inventory in its entirety (8 free
atoms and 22 G2/97 molecules for the GGA rung, 20 for the meta-GGA rung),
`pretrain.pool_atoms` adds all <MEASURED> single-atom species of the BH76 and
W4-11 pools with their production charges and spins, and the synthetic
(r_s, s, alpha) mesh is kept as a regularizer at its stated weight share. The
SCF runs at the architecture's rung parent (`pretrain.parent_density`), so a
meta-GGA network is fit on SCAN's own self-consistent density and a GGA network
on PBE's; the two files are separate because they are separate densities.
Open-shell exchange rows are posed per spin channel
(`pretrain.exchange_footing: spin_channel`), which puts a second row block in
the `.npz`. The objective gained a per-system energy term,
`w_E mean_s (E_xc^NN_s - E_xc^parent_s)^2` in Hartree^2, implemented as a
`segment_sum` over a per-row system index against a per-system parent-energy
table; a seeded fraction of the MOLECULES is held out and scored between
optimizer steps, and training stops after `pretrain.patience` validations
without improvement, keeping the best weights. Every knob defaults to the
pre-change behavior, pinned by a recorded reference file.

**Why.** Section 3.2 of `SPEC_pretrain_fidelity_program.md`, defect D2: the
pretraining set was seven atoms plus a synthetic mesh, so density-matrix
features never saw a molecular environment and the H atom -- one electron,
fully polarized -- was fit to +13.7 mHa by every cusp-carrying network, which
each H in a molecule then multiplied into the atomization energy (CH4 at -25.7
kcal/mol from PBE for deep_cusp_3x16). The integration-weighted point-wise loss
was blind to both: the architecture with the LOWEST exchange residual
(deep_rung35_attn_3x16 at 2.1e-6) carried the LARGEST offset. A per-system
energy term is the term that cannot be satisfied by a low point-wise residual,
and held-out molecules are what detects the molecular extrapolation failing.
The set and the tolerances are the ones Section 7 binds.

**Verification.** <MEASURED> rows over <MEASURED> systems at the production
identity, generated in <MEASURED>. The per-system parent energy is the
quadrature over the stored rows; it tracks libxc's full-grid integral to
<MEASURED> Ha on the O atom, and the density floor that separates them is the
model's own tail threshold, below which the network cannot move the energy at
all. The exchange energy read off the per-channel doubled-density rows equals
the one read off the total-density spin-resolved rows to <MEASURED> Ha, which
is the Oliver-Perdew relation (Phys. Rev. A 20, 397 (1979)) as a number. A
network reproducing the stored enhancement factors carries an energy term of
exactly zero; a network uniformly off by c carries mean_s (c R_s)^2 with R_s the
system's total row weight, both pinned by test. The energy-term weight was
chosen by measured sweep: <MEASURED> (RMS per-system E_xc error <MEASURED> mHa
at w_E = 0, <MEASURED> mHa at the chosen weight, against the certificate's
tol_atom = 1.0 mHa). The default configuration reproduces the recorded
reference file column for column.
```

Replace every `<MEASURED>` with a value read from this machine's logs:

1. Pool single-atom count: from `test_pool_atom_systems_are_the_fourteen_single_atom_species`.
2. Row and system counts, and the generation wall: from the Task 10
   production-identity log and the `n_rows_x` / `n_rows_c` / `n_systems`
   metadata.
3. The row-quadrature-vs-libxc gap and the per-channel-vs-total exchange gap:
   from Task 3 Step 5, read by temporarily tightening the two bounds to `0.0`.
4. The energy-weight sweep numbers: from Task 10's logs.

Do not invent a number and do not copy one from this plan; the plan's values
are bounds, not measurements.

- [ ] **Step 5: Mark the spec's sequence**

In `xcquinox/alec/SPEC_pretrain_fidelity_program.md`, Section 5, change

```
2. Pretraining footing + data set + energy term (3.2); commit; two reviews.
```

to

```
2. Pretraining footing + data set + energy term (3.2); DONE 2026-08-21; commit; two reviews.
```

matching the marker style step 1 uses after the Section 3.1 plan landed.

**Covering test command:** `python -m pytest xcquinox/alec/tests -v > /tmp/xcq-testlogs/task11_full.log 2>&1`

---

## Self-review

Run by the plan author against the spec before handover; recorded so the
executor can see what was already checked and what was decided rather than
inherited.

### Spec coverage

| Spec requirement | Task |
|---|---|
| 3.2 Footing: open-shell rows per spin channel at (2 rho_sigma, 4 sigma_sigma, features of diag(P_sigma, P_sigma)) with the parent's spin-unpolarized F_x as target | 2 (the molecular path reaches `spin_channel_exchange_rows`), 4 (the `*_x` block), 8/9 (`pretrain.exchange_footing`) |
| 3.2 Correlation rows keep the total density with zeta (polarized cnet) | 6 (`_assemble_pretrain_descriptors` refuses `suffix="_x"` for the cnet), 2 (`zeta` column unchanged) |
| 3.2 Coverage: the DFS pretraining set in its entirety | 1 (`resolve_pretrain_systems(dfs_set=True)` through the `_dfs_pretrain_records` seam), 8/9 (`pretrain.dfs_set`) |
| 3.2 Coverage: plus every atom of the BH76 / W4-11 pools | 1 (`pool_atom_systems`, 14 species pinned), 8/9 (`pretrain.pool_atoms`) |
| 3.2 Coverage: generated at the production identity | 9 (basis / grid level / DF come from `cfg.inputs`, as they already did), 10 Step 4 (production-identity confirmation) |
| 3.2 Coverage: with the parent functional's self-consistent densities | 2 (`precompute_fixed_density_data(..., reference_xc=)`, grid guard, electron-count guard), 1 (`resolve_parent_density`, agreement with `rungs.seed_xc_for_arch` pinned), 4 (one file per reference density), 6 (`run_pretrain` refuses a mismatched file) |
| 3.2 Coverage: the synthetic mesh is kept as a regularizer only | 4 (`mesh_fraction` stored beside the weights it produced), 6 (the loss reads the file's share), 6/7 (mesh rows carry zero energy weight and never validate) |
| 3.2 Weighting: an explicit per-system energy term in Hartree beside the point-wise term | 3 (targets), 6 (`_PretrainLoss` segment-sum term) |
| 3.2 "the H atom is one system among many with a term of its own" | 3/4 (every system gets its own table row, atoms included), 7 (atoms are never held out) |
| 3.2 Acceptance inside the pretrain stage: the energy-space check runs after training | owned by the Section 3.3 plan; 6/7 write the metadata it consumes, 11 Step 3 confirms every field exists |
| 6 deviation 1: the DFS set + pool atoms at the production identity on the parent's own densities; mesh a regularizer | 1, 2, 4, 8, 9 |
| 6 deviation 2: footing identical to DFS for the exchange rows, extended to every density-matrix feature | consumed from the Section 3.1 plan (`spin_channel_exchange_rows`); 2/4 put it in the set |
| 6 deviation 3: integration-weighted point-wise residual AND a per-system energy term | 6 |
| 6 deviation 3: validation on held-out systems and a stop criterion replace the hand interruption | 7 |
| 6 deviation 4: acceptance is the certificate with a hard threshold | owned by the Section 3.3 plan; 10's weight rule is stated against `tol_atom` |
| 7: pretraining set = DFS set + pool atoms + mesh, with a per-system energy term | 1, 4, 6, 8 |
| 7: tol_AE 1.0 kcal/mol, tol_atom 1.0 mHa | Global Constraints; 3 (the row set's floor is measured against tol_atom), 10 (the weight rule targets half of tol_atom) |
| 7: campaign v6 resubmits every architecture under the new pretraining | 8/10 (the v6 value of every knob is named in the shipped template) |
| Every new config field in `_build_pretrain` and covered by `test_cluster_examples.py` | 8 |
| Defaults reproduce today's data and today's loss | 3 (recorded fixture), 4 (`test_default_output_matches_the_recorded_reference`), 6 (`test_zero_weight_returns_the_pre_existing_loss_bit_for_bit`), 9 (`test_main_default_call_is_unchanged`) |

Not covered here, by design, and owned by other plans: the exact spin scaling
itself and oracles O1-O4 (Section 3.1), the DFS inventory module
`dfs_pretrain_set.py` and the certificate `cluster/fidelity.py` with its call
site and enforcement (Section 3.3), the workflow matrix (Section 3.4), and the
v6 YAMLs and rendered scripts (Section 3.5).

### Ambiguities in the spec, and how they were resolved

1. **"every atom of the BH76 / W4-11 pools (14 elements)".** The union of the
   two committed pool JSONs has TWELVE neutral single-atom species (Al, B, Be,
   C, Cl, F, H, N, O, P, S, Si) and two closed-shell anions (F-, Cl-) that are
   BH76 reactants: 14 distinct (symbol, charge, 2S) triples. The count in the
   spec matches that reading exactly, so the anions are in. The parenthetical
   "all open shells" does not hold literally -- Be, F- and Cl- are closed
   shells -- and the set is defined by the pools rather than by that phrase.
   Task 1 pins all 14 with their spins.
2. **"config default for new data" versus "an old YAML reproduces today's
   data".** Resolved by making the DATACLASS default `exchange_footing:
   "total"` and naming `spin_channel` as the v6 value in the shipped template,
   while `pretrain_data_is_current` treats the footing as part of the file's
   identity with a legacy default of `"total"`. An existing YAML therefore
   neither changes its data nor regenerates it; a v6 YAML changes the footing
   and the file regenerates because the identity moved. The safety net for a
   forgotten flip is the Section 3.3 certificate, which fails on a network fit
   at the wrong footing.
3. **Whether `pretrain.atoms`' historical default survives the new
   inventories.** Resolved with a `None` sentinel: `atoms=None` means the
   historical four atoms when neither inventory is on and NOTHING when one is.
   The set Section 7 binds is stated exactly, and He is in neither pool nor the
   DFS set; an explicit `atoms` list still extends the set.
4. **Which entry point of the DFS inventory to read.** The Section 3.3 module
   exposes both `dfs_pretrain_records(level)` (basis-free dicts) and
   `dfs_pretrain_systems(level, *, basis, grid_level)` (`MoleculeSpec`s).
   Resolved in favor of the RECORDS: the composition layer must stay basis-free
   so one resolved set can be compared against a manifest written at any
   identity, and the basis and grid level are applied later by `_mol_spec_for`
   at the run's own identity. `normalize_system` accepts a mapping, an
   attribute-carrying object or a `(symbol, 2S)` pair either way, so the only
   coupling left is the function name and the `"gga"` / `"mgga"` level strings,
   which are pinned against `dfs_pretrain_set.LEVELS` by test.
5. **One data file or two.** "The parent functional's own self-consistent
   densities (PBE for GGA-rung, SCAN for meta-GGA)" makes the DENSITY
   rung-dependent, so one file cannot serve both rungs -- and the DFS inventory
   differs between them too (22 molecules against 20). Resolved with a
   reference-density dimension on the filename
   (`pretrain_data[_polarized][_scan].npz`, the PBE names unchanged), a
   `reference_xc` key in the manifest identity, a loud refusal in
   `run_pretrain` when an architecture's rung and the file's reference density
   disagree, and `_datagen` generating one file per distinct
   `(polarized, reference_xc)` pair.
6. **What "E_xc^parent" means as a pretraining target.** Two readings: libxc's
   full-grid integral, or the quadrature over the rows the file stores.
   Resolved in favor of the row quadrature, because the density floor that
   separates them is `models._NN_TAIL_THRESHOLD` = 1e-10 -- below it the model
   clamps F to 1 and the network cannot move the energy at all -- so the
   dropped rows are exactly the rows pretraining could not have fitted. The
   choice makes the energy term exactly zero for a network that reproduces the
   stored enhancement factors, which is the property the required test asserts,
   and Task 3 measures the gap to libxc (bounded at 1e-6 Ha, three orders under
   `tol_atom`).
7. **Which LDA baseline the energy term contracts with.** The stored `Fc` is a
   libxc `spin=1` ratio on an open shell, so its denominator is the
   spin-POLARIZED PW92, while an unpolarized cnet's production baseline is the
   unpolarized PW92 at the total density -- a pre-existing, documented
   asymmetry. Resolved by storing `e_lda_c` as the exact denominator the ratio
   was formed in and contracting the term with it, so the term measures the fit
   without inheriting the asymmetry, and by pinning `e_lda_c / rho` against
   `utils.pw92c_polarized_scalar` on the polarized file (Task 4) so the
   polarized architectures -- the v6 ones -- have the term and the production
   energy be the same quantity by construction.
8. **The DFS protocol's `rho_tot > 1e-6` floor.** NOT adopted, deliberately.
   The library's floor is 1e-10, which is exactly `models._NN_TAIL_THRESHOLD`:
   every row above it is a row the network's output reaches, and every row below
   it is clamped. Raising the floor to the DFS value would leave the interval
   [1e-10, 1e-6] evaluated by a network that was never fit there, which is the
   extrapolation failure mode the campaign measured. The deviation is recorded
   here rather than silently taken.
9. **Which systems validation may hold out.** The spec says "held-out systems"
   without qualification. Resolved to MOLECULES only: every pool atom is a
   system the certificate bounds at `tol_atom` and every atomization energy is
   anchored on atoms, so a held-out atom would fail acceptance by construction,
   while what validation is for -- the molecular extrapolation of the
   density-matrix features -- is what the molecules measure. The split is also
   capped at all-but-one molecule.
10. **A stop criterion inside `xcTrainer`.** `xcTrainer` initializes its
    optimizer state in its constructor, returns no state, swallows callback
    exceptions and returns the LAST model rather than the best, so it can
    express neither early stopping nor best-weight retention, and chunking a run
    through it would reset Adam's moments and restart the learning-rate schedule
    at every validation. Resolved by writing the validated path's loop in
    `pretrain.py` (a faithful copy of `xcTrainer.make_step`) and leaving the
    unvalidated path on `xcTrainer` untouched, which is what makes a run without
    validation byte-identical.
11. **"byte-identical `.npz`".** An `.npz` is a zip whose member headers carry
    write timestamps, so two files with identical contents are never identical
    byte streams. The regression pin is therefore defined on ARRAY CONTENTS:
    every key the pre-change generator wrote is present with the same dtype,
    shape and bitwise values, and new keys may be added. Task 3 records the
    reference with a recorder script, matching the pattern the Section 3.1
    plan's O3 oracle uses.
12. **String-valued provenance in the `.npz`.** `run_pretrain` lifts every
    stored array into JAX with `jnp.array`, which refuses a unicode array, so
    the system NAMES, the footing and the parent live in the JSON manifest
    sidecar and only numeric arrays go in the `.npz`. That also makes the
    provenance human-readable, which the certificate's audit trail wants.
13. **Where the certificate call belongs.** Section 3.2's fourth bullet puts
    acceptance "inside the pretrain stage" while Section 3.3 owns
    `fidelity_certificate`. Resolved by scope: the Section 3.3 plan adds the
    call to `cluster/_pretrain.py`; this plan only guarantees the metadata that
    call and HISTORY read, and Task 11 Step 3 asserts every field exists on a
    real run.
14. **Where a dropped config field actually bites.** `_config_to_raw_dict`
    serializes the whole `pretrain` section with `dataclasses.asdict`, so a new
    `PretrainConfig` field round-trips OUT automatically; the strict allow-list
    is `_build_pretrain` on the way IN, and a field missing there silently
    reverts on every stage that re-reads `resolved_config.yaml`. Task 8 covers
    both directions with a round-trip test that walks `dataclasses.fields`.
15. **The energy-term weight's value.** The spec names no number. Resolved by
    measurement (Task 10) against a stated rule -- the smallest swept weight
    that brings the RMS per-system E_xc error to half of `tol_atom` without
    tripling the point-wise residual -- with 1.0 per Hartree^2 as the starting
    point, which is the weight at which a 1 mHa mean energy error is worth
    1e-6, the order of the converged point-wise residual.
16. **Where the parent density is produced.** An earlier draft of this plan ran
    the parent SCF inside the generator. That is now wrong by construction: the
    Section 3.3 certificate measures `E_xc^NN - E_xc^parent` on the density
    `precompute_fixed_density_data` returns, and training builds its features on
    the same object, so a generator with its own SCF would fit a network on a
    density that merely OUGHT to equal the one it is judged on. Resolved by
    obtaining every parent density through
    `precompute_fixed_density_data(..., reference_xc=)` and keeping only a
    grid-only mean field -- built but never converged -- for the libxc calls,
    the grid coordinates and the `spin_channel_exchange_rows` adapter, with an
    exact-equality guard between that grid and the precompute's. Two
    consequences are recorded rather than hidden: `density_fit` no longer
    changes the parent SCF (the precompute's PBE / SCAN baseline is
    deliberately full-ERI, and that is the density training uses), and the
    generator now inherits the precompute's overlap-conditioning gate, so a
    basis that is near-linearly-dependent on a pretraining molecule fails loudly
    there instead of producing quiet nonsense.
17. **Single precision on the datagen node.** `cluster/_datagen.py` never set
    `JAX_ENABLE_X64`, while `conftest.py` sets it for every test, so the
    descriptor, tau and iso-orbital columns of every production pretrain file
    were computed and stored in float32 while every test computed them in
    double. Routing the DENSITY through a `MoleculeData` whose arrays are
    `jnp.array` would have pushed the density itself through that downcast.
    Resolved in Task 2 Step 1 with a `_route_jax_env()` preamble mirroring
    `cluster/_pretrain.py`, a lazily bound generator seam so the module import
    no longer pulls in `jax.numpy` before the flag is set, and two tests -- one
    on the flag, one on the ordering.
18. **Memory of the precompute cache.** `precompute_fixed_density_data`
    memoizes `MoleculeData` in a process-level dict, and each entry holds a
    `(4, n_grid, n_ao)` AO derivative tensor -- of order 0.8 GB for a
    ten-nucleus molecule at the production identity. Generating a 37-system set
    would retain all of them. Resolved by calling `data.clear_precompute_cache()`
    as each system's columns are extracted: nothing in the generator revisits a
    system, so the cache buys nothing here and costs the node.
