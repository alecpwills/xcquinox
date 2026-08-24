# Per-Architecture Pretraining-Fidelity Certificate and its Enforcement -- Implementation Plan


**Goal:** Make every architecture in a campaign carry a machine-checked certificate that its pretrained networks reproduce their parent functional (PBE for a GGA-rung architecture, SCAN for a meta-GGA one) in energy units on frozen parent densities at the run's production identity, and make the pretrain worker, the train task, the preflight, the in-process model builder, the run validator, the cross-arm merge and the figure suite all refuse to proceed without it.

**Architecture:** One new module, `xcquinox/alec/cluster/fidelity.py`, owns the whole notion. It has two layers with a hard boundary: a CHEAP layer whose module body imports only stdlib plus `grid_config` and `materialize` (pinned by an AST test on the source, since importing any cluster module already pulls the package's own jax-carrying `__init__`) holding the certificate filename, the `certificate_status` and `gate_certificate` predicates every enforcement site calls, the parent resolution and the oracle-set construction; and a PHYSICS layer whose jax/pyscf/`xcquinox.alec.data` imports all live inside functions. The physics is a straight line: build the model exactly as `train._build_model` does, ask the library for a record built on the parent functional's own self-consistent density (`precompute_fixed_density_data(..., reference_xc=parent)` -- a new keyword on the ONE construction path, not a second construction inside the certificate), evaluate `E_xc^NN` through `oneshot.fixed_density_total_energy` and `E_xc^parent` through libxc on the SAME stored grid, difference them, and fold molecules against free atoms into atomization offsets. The verdict, every number, the run identity and the code version go into `<run_dir>/pretrain/<arch>/fidelity_certificate.json`; every enforcement site reads that one file through one of exactly two predicates -- `gate_certificate` for the on-node gates, which honours the run's `fidelity.enforce` flag, and `certificate_status` for the record layers, which requires PASS unconditionally -- so the gate cannot drift between sites. The DFS pretraining set moves out of a trajectory file and into a committed JSON behind a neutral library module (`xcquinox/alec/dfs_pretrain_set.py`) shared with the pretraining plan.

**Tech Stack:** Python 3, JAX (`jax_enable_x64`, CPU), equinox, PySCF + libxc, ASE (build-time only, for the one-off geometry export), pytest, PyYAML.

**Spec:** `xcquinox/alec/SPEC_pretrain_fidelity_program.md` (this plan implements Section 3.3 -- the gate and its enforcement -- plus the Section 7 decisions that bind it. Section 3.1 (spin scaling) and Section 3.2 (pretraining) are separate plans; this plan works against TODAY's energy path and gets stricter results for free once they land.)

## Global Constraints

Every task's requirements implicitly include this section.

- Certificate tolerances, copied verbatim from Section 7 of the spec: "tol_AE = 1.0 kcal/mol on atomization energies and tol_atom = 1.0 mHa on atomic E_xc, for every architecture; no override without `fidelity.override_reason`."
- Certificate scope, copied verbatim from Section 3.3 of the spec: "PASS iff max |dE_xc| per atom <= tol_atom and max |dAE| <= tol_AE and O1-O4 pass. Proposed: tol_AE = 1.0 kcal/mol, tol_atom = 1.0 mHa (configurable in the YAML but never above 2.0 / 2.0 without an explicit `fidelity.override_reason`)." Oracles O1-O4 are unit tests of the spin-scaling plan (Section 3.1) and are NOT re-run inside the certificate; the certificate records the code version (`xcquinox_version`) instead, and the run validator refuses a certificate whose version differs from the run's manifest.
- Enforcement scope, copied verbatim from Section 3.3 of the spec: "the train task refuses to start without a PASS certificate for its architecture; `merge_v4_arms` and the figure loaders refuse a run whose architectures lack one; the certificate's table is rendered into the figure provenance footer."
- Parent functional, copied verbatim from Section 1 of the spec: "the parent functional (PBE for GGA-rung architectures, SCAN for meta-GGA architectures)". This is a property of the ARCHITECTURE's rung (`rungs.seed_xc_for_arch`), not of the run's `inputs.seed_xc` SCF-seeding knob.
- Comments and docstrings are ASCII only, in scientific voice. They state physics, measurements and rationale. They never mention the process by which the code was produced, never mention an assistant or a model, never say "we", "I", "now", "previously", "as requested", "TODO" or "FIXME". Reference literature the way the surrounding code does (author, journal, volume, page, year).
- Run `python -m py_compile <file>` on every Python file immediately after editing it. A task is not finished while any edited file fails to compile.
- Every test run redirects to a log file and the log is read with `Read`. Never pipe a test run through `tail`, `head`, `less`, `grep -m`, or any other truncating filter: the log must be complete. Create the log directory once with `mkdir -p /tmp/xcq-testlogs`.
- Implementers run no git commands: no `git add`, `git commit`, `git push`, `git checkout`, `git branch`, `git stash`, `git rebase`. Committing is the controller's job.
- Every new behaviour is written test-first: the test goes in, the run is executed and the FAILING log is read, and only then is the implementation written. A task that shows no RED run is not finished.
- Every JSON artifact this plan writes goes through `xcquinox.alec.cluster.materialize._write_json_atomic` (mkstemp + `os.replace`). A certificate a reader can observe half-written is a gate that can be passed by a crash.
- `xcquinox/alec/HISTORY.md` gets an entry for this change (Task 14). It is the canonical development record for the paper.
- Every number quoted in a comment or a docstring must have been measured by the implementer on this machine, or explicitly attributed to a recorded measurement (`SPEC_pretrain_fidelity_program.md` Section 2). Do not copy a number from this plan into a comment as a fresh claim; the plan's tolerances are bounds, not measurements.
- Two enforcement layers, and they differ. The ON-NODE gates (pretrain exit, train task, preflight sweep) honour `fidelity.enforce`: when it is False -- permitted only together with a non-empty `fidelity.override_reason` -- the certificate is still computed and written with its TRUE verdict, and the gates log that verdict and continue. The RECORD layers (`validate_run`, `merge_v4_arms`, the figure loaders) require `verdict == "PASS"` unconditionally, so a non-enforcing run can never enter the record. `fidelity.enforce = False` exists for the Section 3.4 workflow matrix, which runs a 50-step pretrain that cannot meet the tolerance yet must exercise the train/eval wiring with the certificate recorded. It is a config field, never an environment variable.
- ONE construction path. The reference density determines eighteen separate `mol_data` fields (the density matrix, four grid quantities, four total-density descriptor blocks, ten per-spin ones, and the three reference energies). No module outside `xcquinox/alec/data.py` may construct any of them. The certificate obtains a parent-density record by ASKING `precompute_fixed_density_data` for one (`reference_xc=`), never by rebuilding fields on a record it was handed. Task 4 carries a source-level guard (`test_fidelity_never_rebuilds_a_precompute_field`) that fails if `fidelity.py` so much as names a descriptor constructor.
- The certificate works against TODAY's energy path and gets stricter results for free once the Section 3.2 pretraining change lands. The Section 3.1 per-spin feature blocks are already in this tree; `reference_xc` populates them through the same open-shell branch with no special-casing, so nothing in this plan re-implements spin scaling.
- SCF reproducibility, measured on this machine at sto-3g / grid level 1, binding on how tests are written: two INDEPENDENT reference SCF runs of the same system are NOT bitwise equal. A closed shell agrees to ~5e-14 Ha in energy and ~5e-8 in the dimensionless meta-GGA alpha (a ratio that amplifies round-off in sigma); a DEGENERATE open shell can converge to a different orientation of its singly occupied shell entirely (free O, C and F atoms moved by 0.2-0.6 point-wise in `rho_grid` between runs of the identical call). The ENERGY consequence is small -- the free-atom `E_xc` spread over three runs was at most 2.2e-3 mHa (O), some 450 times inside `tol_atom = 1.0 mHa` -- which is why the tolerance remains meaningful. But no test in this plan may assert bitwise equality across two separate SCF runs, and no step may claim it.

---

## File Structure

| File | Responsibility after this plan |
|---|---|
| `xcquinox/alec/data/dfs_pretrain_set.json` | The DFS pretraining set's geometries, charges and spins: 8 free atoms + 22 G2/97 molecules, exported once from the Haunschild-Klopper trajectory. Committed data, no ASE at run time. |
| `scripts/generate_dfs_pretrain_set.py` | One-off exporter that builds that JSON from `~/Documents/Research/xcdiff/data/haunschild_g2/g2_97.traj` at the indices of spec Section 6. |
| `xcquinox/alec/dfs_pretrain_set.py` | Neutral library loader: `dfs_pretrain_records(level)` and `dfs_pretrain_systems(level, basis=, grid_level=)`. Shared with the Section 3.2 pretraining plan. |
| `xcquinox/alec/data.py` | Gains `reference_xc` on `precompute_fixed_density_data`, on the memo cache key and on `MoleculeData`: the functional of the reference SCF that produces the density every grid quantity and descriptor block is built from. Default `"pbe"`; the certificate asks for `"scan"` on a meta-GGA architecture. |
| `xcquinox/alec/padding.py` | Strips `reference_xc` in the pad pass (run-level provenance the energy kernel never reads). |
| `xcquinox/alec/cluster/fidelity.py` | The certificate. Cheap layer (Task 2, committed; the oracle-set geometries corrected in Task 3): `CERTIFICATE_FILENAME`, `VERDICT_PASS`, `certificate_status` / `certificate_status_in`, `certificate_enforced_in`, `gate_certificate`, `resolve_parent`, `run_identity`, `build_oracle_set`, `_distinct_archs`. Physics layer: `build_certified_model`, `_parent_exc_on_stored_grid`, `_parent_exc_numint`, `evaluate_system`, `fidelity_certificate`, `main`. Constructs no grid quantity and no descriptor block. |
| `xcquinox/alec/cluster/grid_config.py` | `FidelityConfig` (including `enforce`), the optional `fidelity` block in `load_grid_config`, its bounds in `validate_grid_semantics`. |
| `xcquinox/alec/cluster/__main__.py` | `_config_to_raw_dict` round-trips the `fidelity` block; `_pretrain_status` reports certificate PASS counts. |
| `xcquinox/alec/cluster/examples/grid_step7.yaml` | Ships the `fidelity` block with the binding tolerances. |
| `xcquinox/alec/cluster/_pretrain.py` | Runs the certificate on the pretrain node behind the `_fidelity_certificate` seam; exits non-zero on anything but PASS. |
| `xcquinox/alec/cluster/_train_task.py` | Refuses a spec whose architecture has no PASS certificate (`fidelity_certificate_missing` / `fidelity_certificate_failed`, rc 3, deterministic). |
| `xcquinox/alec/cluster/_preflight.py` | Sweeps every distinct architecture's certificate before releasing the train array. |
| `xcquinox/alec/cluster/validate_run.py` | Refuses a run whose certificates are absent, FAIL, at the wrong identity, or built by different code than the manifest. |
| `xcquinox/alec/train.py` | `_require_fidelity_certificate` gate in `_build_model`, with the `XCQUINOX_ALLOW_UNCERTIFIED=1` escape hatch. |
| `notebooks/analysis/merge_v4_arms.py` | `_validate_arm_fidelity_certificates`; carries each arm's `pretrain/<arch>` into the merged view so the figure layer can see the certificates. |
| `notebooks/analysis/make_ablation_arch_figure.py` | `arch_coverage["uncertified"]`, the `coverage_note` clause, `_FIDELITY_DISCLOSURE` in the footer, the `build_bh76w411_suite` hard fail, certificate numbers in `provenance_footer`. |
| `xcquinox/alec/tests/test_dfs_pretrain_set.py` | Count / names / geometry pins for the committed DFS set. |
| `xcquinox/alec/tests/test_cluster_fidelity.py` | Cheap-layer, seam-mocked and REAL-physics (sto-3g H / H2) certificate tests. |

---

## Certificate JSON schema (fixed once, referenced by every task)

`<run_dir>/pretrain/<arch>/fidelity_certificate.json`:

```json
{
  "verdict": "PASS",
  "arch": "deep_3x16",
  "parent": "pbe",
  "xcquinox_version": "1.0.0+123.gabcdef0",
  "identity": {"basis": "6-311++G(3df,2pd)", "grid_level": 3,
               "density_fit": true, "auxbasis": null,
               "orientation_lock_strength": 0.0},
  "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0, "override_reason": null},
  "enforced": true,
  "per_system": [{"name": "atom_H", "spin": 1, "charge": 0, "is_atom": true,
                  "n_grid": 2336, "reference_xc": "pbe",
                  "E_xc_nn": -0.377701248, "E_xc_parent": -0.385096785,
                  "E_xc_parent_numint": -0.385096785,
                  "parent_grid_diff_Ha": 0.0,
                  "dE_xc_mHa": 7.395537, "duration_s": 8.4}],
  "per_atomization": [{"name": "H2", "dAE_kcalmol": 10.195983}],
  "summary": {"max_atom_mHa": 7.395537, "max_dAE_kcalmol": 10.195983,
              "n_systems": 40, "n_atoms": 16, "n_atomizations": 24,
              "n_failed_systems": 0,
              "max_parent_grid_diff_Ha": 2.6e-11,
              "max_parent_record_diff_Ha": 1.2e-16,
              "failure_reasons": []},
  "timestamp": "2026-08-21T12:00:00Z",
  "duration_s": 1234.5
}
```

(The illustrative numbers above are the measured sto-3g / grid-level-1 values
of the Task 3 real-physics test for a freshly seeded `deep_3x16`, which is the
LDA + PW92 limit; a production certificate carries roughly 40 systems at
6-311++G(3df,2pd) / grid level 3.)

The keys `verdict`, `arch`, `parent`, `xcquinox_version`, `identity`, `tolerances`, `enforced`, `per_system`, `per_atomization`, `summary`, `timestamp`, `duration_s` and the sub-keys listed above are BINDING: Tasks 3, 5, 6, 7, 8, 9, 10, 11 and 12 all read them by exactly these names. `enforced` records whether this run's ON-NODE gates act on the verdict; it never softens the record layers. `verdict` is `"PASS"` or `"FAIL"`. A system whose evaluation raised carries `{"name": ..., "error": "<type>: <message>"}` in place of the numeric keys.

---

## Task 1: The DFS pretraining set as committed data behind a neutral loader

**Files:**
- Create: `scripts/generate_dfs_pretrain_set.py`
- Create: `xcquinox/alec/data/dfs_pretrain_set.json` (produced by running the script)
- Create: `xcquinox/alec/dfs_pretrain_set.py`
- Create: `xcquinox/alec/tests/test_dfs_pretrain_set.py`
- Modify: `pyproject.toml:65-73` (the `[tool.setuptools.package-data]` table)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (the Section 3.2 pretraining plan consumes these too):
  - `xcquinox.alec.dfs_pretrain_set.LEVELS: tuple[str, ...]` == `("gga", "mgga")`
  - `xcquinox.alec.dfs_pretrain_set.MGGA_EXCLUDED: tuple[str, ...]` == `("H2", "N2")`
  - `dfs_pretrain_records(level: str = "gga") -> list[dict]` -- raw records, atoms first then molecules, each `{"kind": "atom"|"molecule", "name": str, "atom": str, "charge": int, "spin": int, "atom_composition": list[[str, int]], "g2_97_index": int|None}`
  - `dfs_pretrain_systems(level: str = "gga", *, basis: str = "6-311++G(3df,2pd)", grid_level: int | None = 3) -> list[MoleculeSpec]`

- [ ] **Step 1: Write the failing test**

Create `xcquinox/alec/tests/test_dfs_pretrain_set.py`:

```python
"""The committed DFS pretraining set (spec Section 6).

The set is the pretraining protocol of the DFS code (Dick and
Fernandez-Serra, Phys. Rev. B 104, L161109 (2021)): eight free atoms with
explicit spins plus 22 G2/97 molecules from the Haunschild and Klopper
trajectory (Theor. Chem. Acc. 131, 1112 (2012)), all molecules run closed
shell. These tests pin the count, the names, the spins and two geometries so
a regenerated JSON that silently changes the set is caught.
"""
import json

import pytest

from xcquinox.alec.dfs_pretrain_set import (
    LEVELS, MGGA_EXCLUDED, dfs_pretrain_records, dfs_pretrain_systems,
)

_MOLECULE_NAMES = (
    "H2", "N2", "LiF", "HCN", "CO2", "Cl2", "F2", "O2", "C2H2", "CO",
    "HCl", "LiH", "Na2", "AlCl3", "PH3", "Si2", "C4H6", "CH4", "SiCH6",
    "C3H8", "CH2", "SiH4",
)
_ATOM_SPINS = {"P": 3, "N": 3, "H": 1, "Li": 1, "O": 2, "Cl": 1,
               "Al": 1, "S": 2}


def test_levels_and_exclusions_are_declared():
    assert LEVELS == ("gga", "mgga")
    assert MGGA_EXCLUDED == ("H2", "N2")


def test_gga_level_is_thirty_systems_eight_atoms_twentytwo_molecules():
    recs = dfs_pretrain_records("gga")
    atoms = [r for r in recs if r["kind"] == "atom"]
    mols = [r for r in recs if r["kind"] == "molecule"]
    assert len(recs) == 30
    assert len(atoms) == 8
    assert len(mols) == 22


def test_mgga_level_drops_h2_and_n2_only():
    gga = {r["name"] for r in dfs_pretrain_records("gga")}
    mgga = {r["name"] for r in dfs_pretrain_records("mgga")}
    assert gga - mgga == {"H2", "N2"}
    assert len(dfs_pretrain_records("mgga")) == 28


def test_molecule_names_and_order_are_the_spec_order():
    mols = [r for r in dfs_pretrain_records("gga") if r["kind"] == "molecule"]
    assert tuple(r["name"] for r in mols) == _MOLECULE_NAMES


def test_every_molecule_is_closed_shell_and_neutral():
    for r in dfs_pretrain_records("gga"):
        if r["kind"] != "molecule":
            continue
        assert r["spin"] == 0, r["name"]
        assert r["charge"] == 0, r["name"]


def test_atom_spins_are_the_hund_ground_states_the_protocol_declares():
    atoms = {r["name"]: r for r in dfs_pretrain_records("gga")
             if r["kind"] == "atom"}
    assert set(atoms) == set(_ATOM_SPINS)
    for name, spin in _ATOM_SPINS.items():
        assert atoms[name]["spin"] == spin
        assert atoms[name]["charge"] == 0
        assert atoms[name]["atom_composition"] == [[name, 1]]


def test_h2_geometry_is_the_g2_97_entry():
    mols = {r["name"]: r for r in dfs_pretrain_records("gga")}
    h2 = mols["H2"]
    assert h2["g2_97_index"] == 2
    lines = [ln.strip() for ln in h2["atom"].split(";")]
    assert lines == ["H 0.0000000000 0.0000000000 0.3713950000",
                     "H 0.0000000000 0.0000000000 -0.3713950000"]


def test_ch4_geometry_is_the_g2_97_entry():
    mols = {r["name"]: r for r in dfs_pretrain_records("gga")}
    ch4 = mols["CH4"]
    assert ch4["g2_97_index"] == 10
    lines = [ln.strip() for ln in ch4["atom"].split(";")]
    assert lines[0] == "C 0.0000000000 0.0000000000 0.0000000000"
    assert lines[1] == "H 0.6303820000 0.6303820000 0.6303820000"
    assert len(lines) == 5


def test_atom_composition_matches_the_geometry_for_every_record():
    for r in dfs_pretrain_records("gga"):
        symbols = [ln.strip().split()[0] for ln in r["atom"].split(";")]
        counts = {}
        for s in symbols:
            counts[s] = counts.get(s, 0) + 1
        assert sorted(tuple(x) for x in r["atom_composition"]) == \
            sorted(counts.items()), r["name"]


def test_systems_are_molecule_specs_carrying_the_requested_identity():
    systems = dfs_pretrain_systems("gga", basis="sto-3g", grid_level=1)
    assert len(systems) == 30
    assert all(ms.basis == "sto-3g" for ms in systems)
    assert all(ms.grid_level == 1 for ms in systems)
    by_name = {ms.name: ms for ms in systems}
    assert by_name["O"].spin == 2
    assert by_name["C4H6"].atom_composition == (("C", 4), ("H", 6))


def test_systems_default_to_the_production_identity():
    systems = dfs_pretrain_systems("gga")
    assert systems[0].basis == "6-311++G(3df,2pd)"
    assert systems[0].grid_level == 3


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="level"):
        dfs_pretrain_records("lda")


def test_records_are_copies_the_caller_cannot_poison():
    a = dfs_pretrain_records("gga")
    a[0]["name"] = "MUTATED"
    b = dfs_pretrain_records("gga")
    assert b[0]["name"] != "MUTATED"


def test_committed_json_declares_its_provenance():
    from xcquinox.alec.dfs_pretrain_set import _DATA_PATH
    with open(_DATA_PATH) as f:
        raw = json.load(f)
    assert "source" in raw
    assert "g2_97" in raw["source"]["trajectory"]
    assert raw["source"]["indices"][:3] == [2, 113, 25]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
mkdir -p /tmp/xcq-testlogs
python -m pytest xcquinox/alec/tests/test_dfs_pretrain_set.py -v \
  > /tmp/xcq-testlogs/t1-red.log 2>&1; echo "rc=$?"
```
Read `/tmp/xcq-testlogs/t1-red.log`. Expected: collection error, `ModuleNotFoundError: No module named 'xcquinox.alec.dfs_pretrain_set'`.

- [ ] **Step 3: Write the exporter script**

Create `scripts/generate_dfs_pretrain_set.py`:

```python
"""Export the DFS pretraining set to xcquinox/alec/data/dfs_pretrain_set.json.

The pretraining protocol of the DFS code (Dick and Fernandez-Serra,
Phys. Rev. B 104, L161109 (2021)) trains on eight free atoms with explicit
spins -- P (2S=3), N (3), H (1), Li (1), O (2), Cl (1), Al (1), S (2) -- plus
22 molecules taken from the Haunschild and Klopper G2/97 trajectory
(Theor. Chem. Acc. 131, 1112 (2012)) at the indices below, all run as closed
shells. The meta-GGA variant of the protocol drops H2 and N2.

The trajectory is an ASE file outside this repository, so the geometries are
exported ONCE into package data: the cluster nodes carry only this package,
and the certificate and the pretraining data generator must resolve the same
geometries with no ASE dependency at run time.

Usage:
    python scripts/generate_dfs_pretrain_set.py [--traj PATH] [--out PATH]
"""
import argparse
import os
import sys

# G2/97 trajectory indices, in the order the DFS notebook lists them.
G2_97_INDICES = (2, 113, 25, 18, 11, 17, 114, 121, 101, 0, 20, 26, 29, 67,
                 28, 110, 125, 10, 115, 89, 105, 50)
# Names in the same order (the trajectory carries formulas, not these names).
MOLECULE_NAMES = ("H2", "N2", "LiF", "HCN", "CO2", "Cl2", "F2", "O2", "C2H2",
                  "CO", "HCl", "LiH", "Na2", "AlCl3", "PH3", "Si2", "C4H6",
                  "CH4", "SiCH6", "C3H8", "CH2", "SiH4")
# The eight free atoms and their 2S values, as the protocol declares them.
ATOM_SPINS = (("P", 3), ("N", 3), ("H", 1), ("Li", 1), ("O", 2), ("Cl", 1),
              ("Al", 1), ("S", 2))

DEFAULT_TRAJ = os.path.expanduser(
    "~/Documents/Research/xcdiff/data/haunschild_g2/g2_97.traj")


def _atom_string(symbols, positions):
    """PySCF geometry string in Angstrom, ten decimals (the trajectory's
    precision), one atom per ';'-separated field."""
    return "; ".join(
        f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}"
        for s, p in zip(symbols, positions))


def _composition(symbols):
    """Sorted (symbol, count) pairs, the MoleculeSpec.atom_composition form."""
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return [[s, counts[s]] for s in sorted(counts)]


def build(traj_path):
    from ase.io import read
    frames = read(traj_path, ":")
    atoms = [{"kind": "atom", "name": sym,
              "atom": f"{sym} 0.0000000000 0.0000000000 0.0000000000",
              "charge": 0, "spin": spin,
              "atom_composition": [[sym, 1]], "g2_97_index": None}
             for sym, spin in ATOM_SPINS]
    molecules = []
    for name, idx in zip(MOLECULE_NAMES, G2_97_INDICES):
        frame = frames[idx]
        symbols = list(frame.get_chemical_symbols())
        molecules.append({
            "kind": "molecule", "name": name,
            "atom": _atom_string(symbols, frame.get_positions()),
            "charge": 0, "spin": 0,
            "atom_composition": _composition(symbols),
            "g2_97_index": int(idx),
        })
    return {
        "source": {
            "protocol": "Dick and Fernandez-Serra, Phys. Rev. B 104, "
                        "L161109 (2021), pretraining notebook",
            "trajectory": "haunschild_g2/g2_97.traj (Haunschild and Klopper, "
                          "Theor. Chem. Acc. 131, 1112 (2012))",
            "indices": [int(i) for i in G2_97_INDICES],
            "units": "angstrom",
        },
        "atoms": atoms,
        "molecules": molecules,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", default=DEFAULT_TRAJ)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out = args.out
    if out is None:
        import xcquinox.alec
        out = os.path.join(os.path.dirname(os.path.abspath(
            xcquinox.alec.__file__)), "data", "dfs_pretrain_set.json")
    payload = build(args.traj)
    from xcquinox.alec.cluster.materialize import _write_json_atomic
    _write_json_atomic(payload, out)
    sys.stdout.write(
        f"wrote {len(payload['atoms'])} atom(s) + "
        f"{len(payload['molecules'])} molecule(s) to {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: `py_compile` and run the exporter**

```bash
python -m py_compile scripts/generate_dfs_pretrain_set.py
python scripts/generate_dfs_pretrain_set.py \
  > /tmp/xcq-testlogs/t1-export.log 2>&1; echo "rc=$?"
```
Read the log. Expected: `wrote 8 atom(s) + 22 molecule(s) to .../xcquinox/alec/data/dfs_pretrain_set.json`.

- [ ] **Step 5: Write the loader module**

Create `xcquinox/alec/dfs_pretrain_set.py`:

```python
"""The DFS pretraining set: eight free atoms and 22 G2/97 molecules.

Source protocol: the pretraining notebook of the DFS code (Dick and
Fernandez-Serra, Phys. Rev. B 104, L161109 (2021)). Eight free atoms with
explicit spins -- P (2S=3), N (3), H (1), Li (1), O (2), Cl (1), Al (1),
S (2) -- plus 22 molecules of the Haunschild and Klopper G2/97 set
(Theor. Chem. Acc. 131, 1112 (2012)), every molecule run as a closed shell
(including O2 and CH2, which are open-shell species physically: the protocol
poses them at 2S = 0 and the pretraining targets follow). The meta-GGA
variant of the protocol drops H2 and N2, giving 28 systems against 30.

Geometries are committed package data (``data/dfs_pretrain_set.json``,
regenerated with ``scripts/generate_dfs_pretrain_set.py``) rather than read
from the ASE trajectory they came from: the compute nodes carry only this
package, and the fidelity certificate and the pretraining data generator must
resolve byte-identical geometries.
"""
from __future__ import annotations

import copy
import json
import os

LEVELS: tuple[str, ...] = ("gga", "mgga")
# The meta-GGA variant of the DFS pretraining notebook omits these two.
MGGA_EXCLUDED: tuple[str, ...] = ("H2", "N2")

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data",
    "dfs_pretrain_set.json")

_CACHE: dict | None = None


def _load() -> dict:
    """Read and memoize the committed JSON."""
    global _CACHE
    if _CACHE is None:
        with open(_DATA_PATH) as f:
            _CACHE = json.load(f)
    return _CACHE


def dfs_pretrain_records(level: str = "gga") -> list[dict]:
    """The set's raw records for ``level``, atoms first then molecules.

    Each record is ``{"kind": "atom"|"molecule", "name", "atom", "charge",
    "spin", "atom_composition", "g2_97_index"}`` with ``atom`` a PySCF
    geometry string in Angstrom. Returns fresh copies so a caller cannot
    poison the module cache.
    """
    if level not in LEVELS:
        raise ValueError(
            f"dfs_pretrain_set level must be one of {LEVELS}, got {level!r}")
    raw = _load()
    excluded = set(MGGA_EXCLUDED) if level == "mgga" else set()
    out = [copy.deepcopy(r) for r in raw["atoms"]]
    out += [copy.deepcopy(r) for r in raw["molecules"]
            if r["name"] not in excluded]
    return out


def dfs_pretrain_systems(level: str = "gga", *,
                         basis: str = "6-311++G(3df,2pd)",
                         grid_level: int | None = 3) -> list:
    """The set as :class:`~xcquinox.alec.config.MoleculeSpec` objects.

    ``basis`` / ``grid_level`` default to the production identity of the
    campaign (6-311++G(3df,2pd), grid level 3); pass a smaller pair for a
    local probe.
    """
    from xcquinox.alec.config import MoleculeSpec
    return [
        MoleculeSpec(
            name=r["name"], atom=r["atom"], basis=basis,
            charge=int(r["charge"]), spin=int(r["spin"]),
            atom_composition=tuple((str(s), int(n))
                                   for s, n in r["atom_composition"]),
            grid_level=grid_level,
        )
        for r in dfs_pretrain_records(level)
    ]
```

- [ ] **Step 6: Declare the package data**

In `pyproject.toml`, inside `[tool.setuptools.package-data]` (currently lines 65-73), add an `xcquinox.alec` entry so the JSON data directory ships with the package (this also covers the existing `bh76_full_pool.json` / `w411_full_pool.json`, which are read the same way):

```toml
"xcquinox.alec" = [
    "data/*.json"
]
```

- [ ] **Step 7: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/dfs_pretrain_set.py
python -m pytest xcquinox/alec/tests/test_dfs_pretrain_set.py -v \
  > /tmp/xcq-testlogs/t1-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: 14 passed.

**Deliverable:** `dfs_pretrain_systems("gga")` returns the 30 DFS systems at the production identity with no ASE import, and `dfs_pretrain_systems("mgga")` returns 28.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_dfs_pretrain_set.py -v > /tmp/xcq-testlogs/t1-green.log 2>&1`

---

## Task 2: `cluster/fidelity.py` -- the cheap layer (predicate, parent, oracle set)

**Files:**
- Create: `xcquinox/alec/cluster/fidelity.py`
- Create: `xcquinox/alec/tests/test_cluster_fidelity.py`

**Interfaces:**
- Consumes: `dfs_pretrain_set.dfs_pretrain_records(level)` (Task 1).
- Produces (read by Tasks 3, 5, 6, 7, 8, 9, 10, 11, 12):
  - `CERTIFICATE_FILENAME: str` == `"fidelity_certificate.json"`
  - `VERDICT_PASS: str` == `"PASS"`, `VERDICT_FAIL: str` == `"FAIL"`
  - `HA_TO_KCAL: float`, `HA_TO_MHA: float`, `PARENT_GRID_TOL_HA: float`
  - `certificate_path_in(pretrain_dir: str) -> str`
  - `certificate_path(run_dir: str, arch: str) -> str`
  - `read_certificate(pretrain_dir: str) -> dict | None`
  - `certificate_status_in(pretrain_dir: str) -> tuple[str, str]` -- status in `{"PASS","FAIL","MISSING","UNREADABLE"}` plus a human-readable reason
  - `certificate_status(run_dir: str, arch: str) -> tuple[str, str]`
  - `certificate_enforced_in(pretrain_dir: str) -> bool` -- the certificate's own `enforced` field; absent/unreadable -> `True`
  - `gate_certificate(run_dir: str, arch: str) -> tuple[bool, str]` -- the ON-NODE predicate: `(allowed, message)`
  - `resolve_parent(arch_name: str) -> str` -- `"pbe"` or `"scan"`
  - `dfs_level_for_parent(parent: str) -> str` -- `"gga"` or `"mgga"`
  - `run_identity(cfg) -> dict`
  - `atom_system_name(symbol: str, charge: int) -> str`
  - `is_atom_system(mol_spec) -> bool`
  - `build_oracle_set(cfg, arch_name: str) -> tuple`  (of `MoleculeSpec`)
  - `_distinct_archs(cfg) -> list[str]`
- Import-weight contract: `fidelity.py`'s MODULE BODY imports only `__future__`, `argparse`, `json`, `os`, `sys`, `time`, `xcquinox.alec.cluster.grid_config` and `xcquinox.alec.cluster.materialize`. Every jax / equinox / pyscf / `xcquinox.alec.data` import lives inside a function. (Importing any `xcquinox.alec.cluster` module already executes the package's jax-carrying `__init__`, so this is a statement about this file's own body and is checked on its source, not on `sys.modules`.)

- [ ] **Step 1: Write the failing tests**

Create `xcquinox/alec/tests/test_cluster_fidelity.py`:

```python
"""Tests for xcquinox.alec.cluster.fidelity -- the per-architecture physics
certificate.

The cheap layer (the certificate predicate, the parent resolution, the oracle
set) is tested directly. The certificate itself is tested twice: once with the
per-system evaluation monkeypatched at the ``evaluate`` seam, so the verdict
arithmetic and the JSON schema are exercised with no SCF at all, and once for
REAL on H and H2 at sto-3g with networks built in the test, so the energy path,
the libxc parent route and the atomization fold are pinned against physics.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from xcquinox.alec.cluster import fidelity as fid


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _cfg(arch=("deep_3x16",), basis="sto-3g", grid_level=1,
         tol_AE=1.0, tol_atom=1.0, override_reason=None, enforce=True,
         polarized=False, pretrain_seed=42):
    """The attribute surface fidelity reads off a GridConfig."""
    return SimpleNamespace(
        sweep=SimpleNamespace(arch=tuple(arch)),
        inputs=SimpleNamespace(basis=basis, grid_level=grid_level,
                               density_fit=False, auxbasis=None,
                               orientation_lock_strength=0.0),
        pretrain=SimpleNamespace(seed=pretrain_seed),
        fidelity=SimpleNamespace(tol_AE=tol_AE, tol_atom=tol_atom,
                                 override_reason=override_reason,
                                 enforce=enforce),
        use_polarized_correlation=polarized,
    )


def _write_certificate(run_dir, arch, verdict="PASS", **extra):
    """Write a certificate; ``extra`` sets any schema key (``enforced``,
    ``summary``, ``tolerances``, ...)."""
    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    payload = {"verdict": verdict, "arch": arch}
    payload.update(extra)
    with open(os.path.join(d, fid.CERTIFICATE_FILENAME), "w") as f:
        json.dump(payload, f)
    return d


# ---------------------------------------------------------------------------
# Import weight: the module body stays a pure reader
# ---------------------------------------------------------------------------

def test_fidelity_module_body_imports_only_cheap_modules():
    """`cluster status`, validate_run, merge and the train task's PARENT
    process all read certificates. fidelity.py's module BODY must therefore
    stay a pure reader -- path helpers, the certificate predicates, constants
    -- with every jax / pyscf / xcquinox.alec.data import inside a function,
    so a reader never triggers a model import or an SCF-capable stack it does
    not use. Checked on the source: importing any cluster module already
    executes the package's own jax-carrying __init__, so sys.modules cannot
    distinguish this file's cost from the package's."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(fid))
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top.append(node.module or "")
    assert sorted(top) == [
        "__future__", "argparse", "json", "os", "sys", "time",
        "xcquinox.alec.cluster.grid_config",
        "xcquinox.alec.cluster.materialize",
    ], sorted(top)


def test_fidelity_imports_in_a_fresh_interpreter():
    """No import cycle: _pretrain imports fidelity, fidelity must not import
    _pretrain (it derives the distinct-arch list from _canon_axis instead)."""
    out = subprocess.run(
        [sys.executable, "-c", "import xcquinox.alec.cluster.fidelity"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------------------
# The one predicate every enforcement site calls
# ---------------------------------------------------------------------------

def test_certificate_status_missing(tmp_path):
    status, reason = fid.certificate_status_in(str(tmp_path))
    assert status == "MISSING"
    assert fid.CERTIFICATE_FILENAME in reason


def test_certificate_status_pass(tmp_path):
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="PASS")
    assert fid.certificate_status_in(d) == ("PASS", "fidelity certificate PASS")


def test_certificate_status_fail_carries_the_summary(tmp_path):
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                           summary={"max_atom_mHa": 13.7,
                                    "max_dAE_kcalmol": 25.7})
    status, reason = fid.certificate_status_in(d)
    assert status == "FAIL"
    assert "13.7" in reason and "25.7" in reason


def test_certificate_status_unreadable(tmp_path):
    d = tmp_path / "pretrain" / "deep_3x16"
    d.mkdir(parents=True)
    (d / fid.CERTIFICATE_FILENAME).write_text("{not json")
    status, reason = fid.certificate_status_in(str(d))
    assert status == "UNREADABLE"
    assert "JSON" in reason


def test_certificate_status_by_run_dir_and_arch_uses_the_harness_layout(tmp_path):
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    _write_certificate(str(tmp_path), "deep_3x16")
    assert fid.certificate_path(str(tmp_path), "deep_3x16") == os.path.join(
        pretrain_checkpoint_dir(str(tmp_path), "deep_3x16"),
        fid.CERTIFICATE_FILENAME)
    assert fid.certificate_status(str(tmp_path), "deep_3x16")[0] == "PASS"


def test_read_certificate_returns_none_when_absent(tmp_path):
    assert fid.read_certificate(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# The ON-NODE gate: fidelity.enforce = False records the verdict and continues
# ---------------------------------------------------------------------------

def test_certificate_enforced_defaults_to_true_when_the_field_is_absent(
        tmp_path):
    """A certificate that does not say otherwise is enforcing; so is an
    absent one, which cannot say anything at all."""
    d = _write_certificate(str(tmp_path), "deep_3x16")
    assert fid.certificate_enforced_in(d) is True
    assert fid.certificate_enforced_in(str(tmp_path / "nowhere")) is True


def test_certificate_enforced_reads_the_recorded_flag(tmp_path):
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                           enforced=False)
    assert fid.certificate_enforced_in(d) is False


def test_gate_allows_a_passing_certificate(tmp_path):
    _write_certificate(str(tmp_path), "deep_3x16", verdict="PASS")
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is True
    assert "PASS" in message


def test_gate_refuses_an_enforced_failure(tmp_path):
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=True,
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7})
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False
    assert "13.7" in message


def test_gate_allows_a_recorded_failure_when_enforcement_is_off(tmp_path):
    """The Section 3.4 workflow matrix: a 50-step pretrain cannot meet the
    tolerance, but train and eval must still be exercised end to end with the
    real verdict on the record."""
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=False,
                       tolerances={"tol_AE": 1.0, "tol_atom": 1.0,
                                   "override_reason": "workflow matrix"},
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7})
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is True
    assert "enforcement is OFF" in message
    assert "workflow matrix" in message


def test_gate_never_allows_a_missing_certificate(tmp_path):
    """Enforcement can only be waived by a certificate that exists to record
    the waiver; an absent one waives nothing."""
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False
    assert "MISSING" in message or "was never checked" in message


def test_gate_never_allows_an_unreadable_certificate(tmp_path):
    d = tmp_path / "pretrain" / "deep_3x16"
    d.mkdir(parents=True)
    (d / fid.CERTIFICATE_FILENAME).write_text("{truncated")
    allowed, _message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False


# ---------------------------------------------------------------------------
# Parent resolution: the arch's RUNG picks the parent, not inputs.seed_xc
# ---------------------------------------------------------------------------

def test_parent_is_pbe_for_gga_rung_and_scan_for_meta_gga():
    assert fid.resolve_parent("deep_3x16") == "pbe"
    assert fid.resolve_parent("deep_cusp_3x16") == "pbe"
    assert fid.resolve_parent("deep_rung35_3x16") == "pbe"
    assert fid.resolve_parent("deep_mgga_3x16") == "scan"


def test_parent_agrees_with_the_rung_seed_policy():
    from xcquinox.alec.rungs import seed_xc_for_arch
    from xcquinox.alec.config import list_architectures
    for name in list_architectures():
        assert fid.resolve_parent(name) == seed_xc_for_arch(name)


def test_dfs_level_follows_the_parent():
    assert fid.dfs_level_for_parent("pbe") == "gga"
    assert fid.dfs_level_for_parent("scan") == "mgga"


def test_distinct_archs_matches_the_pretrain_workers_selector():
    from xcquinox.alec.cluster import _pretrain as pt
    cfg = _cfg(arch=("medium", "deep", "medium", "shallow"))
    assert fid._distinct_archs(cfg) == ["deep", "medium", "shallow"]
    assert fid._distinct_archs(cfg) == pt._distinct_archs(cfg)


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

def test_run_identity_carries_the_five_scf_identity_fields():
    cfg = _cfg(basis="6-311++G(3df,2pd)", grid_level=3)
    cfg.inputs.density_fit = True
    cfg.inputs.auxbasis = "def2-universal-jkfit"
    cfg.inputs.orientation_lock_strength = 0.02
    assert fid.run_identity(cfg) == {
        "basis": "6-311++G(3df,2pd)", "grid_level": 3, "density_fit": True,
        "auxbasis": "def2-universal-jkfit",
        "orientation_lock_strength": 0.02}


# ---------------------------------------------------------------------------
# Oracle set
# ---------------------------------------------------------------------------

def test_atom_system_names_are_canonical():
    assert fid.atom_system_name("O", 0) == "atom_O"
    assert fid.atom_system_name("F", -1) == "atom_F-"
    assert fid.atom_system_name("Na", 1) == "atom_Na+"
    assert fid.atom_system_name("O", -2) == "atom_O-2"


def test_oracle_set_puts_atoms_first_then_molecules_each_sorted():
    systems = fid.build_oracle_set(_cfg(), "deep_3x16")
    names = [ms.name for ms in systems]
    atoms = [n for n in names if n.startswith("atom_")]
    mols = [n for n in names if not n.startswith("atom_")]
    assert names == atoms + mols
    assert atoms == sorted(atoms)
    assert mols == sorted(mols)


def test_oracle_set_carries_every_pool_free_atom_with_its_pool_spin():
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    pool, _ = load_full_held_out_pools(basis="sto-3g", grid_level=1)
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    seen = 0
    for ms in pool.values():
        comp = tuple(ms.atom_composition)
        if len(comp) != 1 or int(comp[0][1]) != 1:
            continue
        seen += 1
        name = fid.atom_system_name(comp[0][0], ms.charge)
        assert name in systems, name
        assert systems[name].spin == ms.spin
        assert systems[name].charge == ms.charge
    assert seen >= 14


def test_oracle_set_carries_the_dfs_molecules_and_the_fixed_three():
    systems = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    for name in ("H2", "LiF", "AlCl3", "C4H6", "SiH4"):
        assert name in systems
    for name in ("H2O", "N2", "CH4"):
        assert name in systems


def test_meta_gga_oracle_set_drops_h2_but_keeps_n2_from_the_fixed_three():
    """The meta-GGA DFS variant omits H2 and N2; the fixed molecule set
    restores N2, so every architecture is measured on a common N2 / H2O / CH4
    core whatever its rung."""
    gga = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    mgga = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_mgga_3x16")}
    assert "H2" in gga and "H2" not in mgga
    assert "N2" in gga and "N2" in mgga


def test_oracle_set_supplies_a_free_atom_for_every_element_it_dissociates():
    systems = fid.build_oracle_set(_cfg(), "deep_3x16")
    names = {ms.name for ms in systems}
    for ms in systems:
        if fid.is_atom_system(ms):
            continue
        for sym, _n in ms.atom_composition:
            assert fid.atom_system_name(sym, 0) in names, (ms.name, sym)


def test_oracle_set_adds_lithium_and_sodium_which_no_pool_carries():
    names = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    assert "atom_Li" in names and "atom_Na" in names


def test_ground_state_spin_table_agrees_with_the_pool_spins():
    """The certificate's Hund ground-state table is the atomization reference;
    it must agree species by species with the spins the BH76 / W4-11 pools
    carry, or a molecule would be folded against a different atom than the
    benchmark uses."""
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    pool, _ = load_full_held_out_pools(basis="sto-3g", grid_level=1)
    for ms in pool.values():
        comp = tuple(ms.atom_composition)
        if len(comp) != 1 or int(comp[0][1]) != 1 or ms.charge != 0:
            continue
        sym = comp[0][0]
        assert fid._ATOM_GROUND_SPIN[sym] == ms.spin, sym


def test_oracle_set_specs_carry_the_run_identity():
    cfg = _cfg(basis="def2-tzvpd", grid_level=2)
    for ms in fid.build_oracle_set(cfg, "deep_3x16"):
        assert ms.basis == "def2-tzvpd"
        assert ms.grid_level == 2
        assert ms.external_data_path is None


def test_is_atom_system():
    from xcquinox.alec.config import MoleculeSpec
    atom = MoleculeSpec(name="atom_O", atom="O 0 0 0", basis="sto-3g", spin=2,
                        atom_composition=(("O", 1),))
    mol = MoleculeSpec(name="OH", atom="O 0 0 0; H 0 0 1", basis="sto-3g",
                       spin=1, atom_composition=(("H", 1), ("O", 1)))
    assert fid.is_atom_system(atom)
    assert not fid.is_atom_system(mol)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_fidelity.py -v \
  > /tmp/xcq-testlogs/t2-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: collection error, `ImportError: cannot import name 'fidelity' from 'xcquinox.alec.cluster'`.

- [ ] **Step 3: Write the cheap layer**

Create `xcquinox/alec/cluster/fidelity.py`:

```python
"""xcquinox.alec.cluster.fidelity -- the per-architecture physics certificate.

Pretrained networks are accepted only when they reproduce their parent
functional in ENERGY units. For one architecture the certificate evaluates

    dE_xc = E_xc^NN[rho_parent] - E_xc^parent[rho_parent]

through the production energy path, on the parent's own self-consistent
density, at the run's SCF identity, for every free atom of the BH76 / W4-11
pools, the DFS pretraining molecules and a fixed molecule set; the molecular
differences are folded against the free atoms into atomization-energy offsets

    dAE(mol) = dE_xc(mol) - sum_atoms n_atom * dE_xc(atom).

PASS requires max |dE_xc| over the free atoms <= tol_atom (mHa) AND max |dAE|
<= tol_AE (kcal/mol). The parent is PBE for a GGA-rung architecture and SCAN
for a meta-GGA one (rungs.seed_xc_for_arch), which is what each rung was
pretrained against.

The verdict, every number, the run identity and the installed code version go
to ``<run_dir>/pretrain/<arch>/fidelity_certificate.json``. The pretrain
worker, the train task, the preflight, the in-process model builder, the run
validator, the cross-arm merge and the figure suite all read that file through
:func:`certificate_status`, so the gate cannot drift between sites.

Invocation on a node::

    python -m xcquinox.alec.cluster.fidelity <RUN_DIR> <ARCH_IDX>

ENFORCEMENT HAS TWO LAYERS. The ON-NODE gates (the pretrain worker's exit
code, the train task, the preflight sweep) call :func:`gate_certificate`,
which honours the certificate's recorded ``enforced`` flag: a run configured
with ``fidelity.enforce: false`` (permitted only with a non-empty
``fidelity.override_reason``) still computes and writes the certificate with
its TRUE verdict, and the gates log it and continue. That exists for the
workflow-verification matrix, whose short pretraining runs cannot meet the
tolerance but must exercise the train and eval wiring with the physics on
record. The RECORD layers -- ``validate_run``, ``merge_v4_arms`` and the
figure loaders -- call :func:`certificate_status` and require PASS
unconditionally, so a non-enforcing run can never become a quantitative
result.

IMPORT WEIGHT (a contract, pinned by an AST test on this file's source): the
MODULE BODY imports only ``__future__``, ``argparse``, ``json``, ``os``,
``sys``, ``time``, ``grid_config`` and ``materialize``. Every jax / equinox /
pyscf / ``xcquinox.alec.data`` import happens INSIDE a function, so the login
node CLI, the run validator, the train task's parent process and the analysis
layer read a certificate without this file pulling a model or an SCF stack.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from xcquinox.alec.cluster.grid_config import (
    _canon_axis, load_grid_config, pretrain_checkpoint_dir,
)
from xcquinox.alec.cluster.materialize import _write_json_atomic


CERTIFICATE_FILENAME = "fidelity_certificate.json"
VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"

# CODATA-consistent conversions, matching the harness domain tables.
HA_TO_KCAL = 627.509474
HA_TO_MHA = 1000.0

# The parent XC energy is computed two independent ways per system: point-wise
# on the stored precompute grid (the grid the network is integrated on, so the
# comparison is grid-exact) and through PySCF's own nr_rks / nr_uks on a
# freshly built grid of the same level. The two routes agreed to 2.6e-11 Ha on
# OH/sto-3g and to 2.0e-10 Ha at the production identity (scratch probes
# 2026-08-20); anything above this bound means the stored grid and the
# molecule no longer describe the same system.
PARENT_GRID_TOL_HA = 1e-6

# libxc names of the two parents.
_PARENT_XC = {"pbe": "PBE", "scan": "SCAN"}

# 2S for the Hund ground state of each neutral element the BH76 / W4-11 pools
# and the DFS pretraining set span. These are the spins the pools themselves
# carry (asserted species by species in the tests) and the spins the DFS
# pretraining protocol declares for its eight free atoms. The table is the
# ATOMIZATION reference: a molecule's offset is folded against these atoms.
_ATOM_GROUND_SPIN: dict[str, int] = {
    "H": 1, "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1,
    "Na": 1, "Mg": 0, "Al": 1, "Si": 2, "P": 3, "S": 2, "Cl": 1,
}

# Molecules the certificate always carries, on top of the pools' free atoms
# and the DFS molecules: the three systems the pre-certificate offsets were
# measured on (SPEC_pretrain_fidelity_program.md Section 2), so every
# architecture is comparable on a common core whatever its rung. Geometry in
# Angstrom. H2O is the experimental r = 0.958 A, 104.5 degree structure; N2 is
# r = 1.0977 A; CH4 is the G2/97 entry (identical to the DFS record, which
# wins by name when the DFS set carries the molecule).
# SUPERSEDED BY TASK 3 STEPS 1-2: these literal geometries let a DFS record of
# the same name win and made dAE(N2) / dAE(CH4) rung-dependent. Task 3 replaces
# this table with _FIXED_MOLECULE_POOL_NAMES, which resolves all three from the
# BH76 / W4-11 pools for every rung. Kept here as the record of what Task 2
# committed.
_FIXED_MOLECULES: tuple[tuple[str, str, int, int], ...] = (
    ("H2O", "O 0.0000000000 0.0000000000 0.0000000000; "
            "H 0.0000000000 0.7570000000 0.5870000000; "
            "H 0.0000000000 -0.7570000000 0.5870000000", 0, 0),
    ("N2", "N 0.0000000000 0.0000000000 0.0000000000; "
           "N 0.0000000000 0.0000000000 1.0977000000", 0, 0),
    ("CH4", "C 0.0000000000 0.0000000000 0.0000000000; "
            "H 0.6303820000 0.6303820000 0.6303820000; "
            "H -0.6303820000 -0.6303820000 0.6303820000; "
            "H 0.6303820000 -0.6303820000 -0.6303820000; "
            "H -0.6303820000 0.6303820000 -0.6303820000", 0, 0),
)


# ---------------------------------------------------------------------------
# The certificate file: one path helper and one predicate, shared by every
# enforcement site so the gate cannot drift between them.
# ---------------------------------------------------------------------------

def certificate_path_in(pretrain_dir: str) -> str:
    """Certificate path inside a pretrain checkpoint directory."""
    return os.path.join(pretrain_dir, CERTIFICATE_FILENAME)


def certificate_path(run_dir: str, arch: str) -> str:
    """Certificate path for one architecture of a run."""
    return certificate_path_in(pretrain_checkpoint_dir(run_dir, arch))


def read_certificate(pretrain_dir: str) -> dict | None:
    """The parsed certificate, or ``None`` when absent or unparseable."""
    try:
        with open(certificate_path_in(pretrain_dir)) as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def certificate_status_in(pretrain_dir: str) -> tuple[str, str]:
    """``(status, reason)`` for the certificate in ``pretrain_dir``.

    ``status`` is ``"PASS"``, ``"FAIL"``, ``"MISSING"`` (no file) or
    ``"UNREADABLE"`` (file present but not a JSON object). Only ``"PASS"``
    releases a gate: an unreadable certificate is unverifiable, and an
    unverifiable certificate is refused.
    """
    path = certificate_path_in(pretrain_dir)
    if not os.path.isfile(path):
        return "MISSING", (
            f"no {CERTIFICATE_FILENAME} in {pretrain_dir}: the architecture "
            "was never checked against its parent functional")
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as exc:
        return "UNREADABLE", (
            f"{path} is not readable JSON ({type(exc).__name__}: {exc})")
    if not isinstance(payload, dict):
        return "UNREADABLE", f"{path} is not a JSON object"
    verdict = payload.get("verdict")
    if verdict == VERDICT_PASS:
        return VERDICT_PASS, "fidelity certificate PASS"
    summary = payload.get("summary") or {}
    return VERDICT_FAIL, (
        f"fidelity certificate verdict {verdict!r} at {path} "
        f"(max_atom_mHa={summary.get('max_atom_mHa')}, "
        f"max_dAE_kcalmol={summary.get('max_dAE_kcalmol')}, "
        f"reasons={summary.get('failure_reasons')})")


def certificate_status(run_dir: str, arch: str) -> tuple[str, str]:
    """``(status, reason)`` for one architecture of a run.

    This is the RECORD-layer predicate: ``validate_run``, ``merge_v4_arms``
    and the figure loaders require ``VERDICT_PASS`` from it unconditionally.
    On-node gates call :func:`gate_certificate` instead.
    """
    return certificate_status_in(pretrain_checkpoint_dir(run_dir, arch))


def certificate_enforced_in(pretrain_dir: str) -> bool:
    """Whether the certificate in ``pretrain_dir`` says its verdict is acted on.

    A certificate written by a run with ``fidelity.enforce: false`` records
    ``"enforced": false``. Absent, unreadable, or written before the field
    existed -> ``True``: enforcement can only be waived by a certificate that
    exists to record the waiver.
    """
    payload = read_certificate(pretrain_dir)
    if not payload:
        return True
    return bool(payload.get("enforced", True))


def gate_certificate(run_dir: str, arch: str) -> tuple[bool, str]:
    """``(allowed, message)`` for an ON-NODE gate.

    ``allowed`` is True when the certificate PASSes, and also when it exists,
    FAILs, and records ``enforced: false`` -- the workflow-verification
    matrix, whose short pretraining runs cannot meet the tolerance yet must
    exercise the train and eval wiring with the real verdict written down. A
    MISSING or UNREADABLE certificate is never allowed: there is then no
    record of what was measured or of any waiver.

    The record layers do NOT call this. ``validate_run``, ``merge_v4_arms``
    and the figure loaders require PASS through :func:`certificate_status`, so
    a non-enforcing run can never enter the record.
    """
    pretrain_dir = pretrain_checkpoint_dir(run_dir, arch)
    status, reason = certificate_status_in(pretrain_dir)
    if status == VERDICT_PASS:
        return True, reason
    if status != VERDICT_FAIL:
        return False, reason
    if certificate_enforced_in(pretrain_dir):
        return False, reason
    payload = read_certificate(pretrain_dir) or {}
    override = ((payload.get("tolerances") or {}).get("override_reason")
                or "(no reason recorded)")
    return True, (
        f"{reason}; enforcement is OFF for this run "
        f"(fidelity.enforce=false, override_reason: {override}) so the "
        "verdict is recorded and the stage continues. This run cannot enter "
        "validate_run, merge_v4_arms or the figure suite.")


# ---------------------------------------------------------------------------
# Parent functional and run identity
# ---------------------------------------------------------------------------

def resolve_parent(arch_name: str) -> str:
    """The parent functional an architecture must reproduce: ``"pbe"`` for a
    GGA-rung architecture, ``"scan"`` for a meta-GGA one.

    Derived from the architecture's RUNG (``rungs.seed_xc_for_arch``), not
    from ``inputs.seed_xc``: the parent is what the networks were pretrained
    against, a property of the architecture, while ``inputs.seed_xc`` selects
    the SCF starting density for training and may be pinned to "pbe" for a
    controlled experiment.
    """
    from xcquinox.alec.rungs import seed_xc_for_arch
    return seed_xc_for_arch(arch_name)


def dfs_level_for_parent(parent: str) -> str:
    """The DFS pretraining-set level matching a parent functional.

    The meta-GGA variant of the DFS protocol drops H2 and N2, so a SCAN-parent
    architecture is certified on the same 28 systems it was pretrained on.
    """
    return "mgga" if parent == "scan" else "gga"


def run_identity(cfg) -> dict:
    """The SCF / grid identity every certificate number is computed at.

    The five fields are exactly the run-level inputs that change an energy:
    basis, grid level, the Coulomb backend (density fitting plus its auxiliary
    basis) and the orientation-lock strength. ``validate_run`` refuses a
    certificate whose identity differs from the config's.
    """
    inp = cfg.inputs
    return {
        "basis": inp.basis,
        "grid_level": int(inp.grid_level),
        "density_fit": bool(getattr(inp, "density_fit", False)),
        "auxbasis": getattr(inp, "auxbasis", None),
        "orientation_lock_strength": float(
            getattr(inp, "orientation_lock_strength", 0.0)),
    }


def _distinct_archs(cfg):
    """The de-duplicated, sorted architecture list of the sweep.

    Uses ``grid_config._canon_axis`` -- the EXACT de-dup + sort ``expand_grid``
    applies to the arch axis -- so ``<arch_idx>`` selects the same
    architecture here as in ``cluster._pretrain``. Deliberately NOT imported
    from ``_pretrain``: that module imports this one for its gate.
    """
    return _canon_axis(cfg.sweep.arch)


# ---------------------------------------------------------------------------
# Oracle set
# ---------------------------------------------------------------------------

def atom_system_name(symbol: str, charge: int) -> str:
    """Canonical oracle-set name for a free atom: ``atom_O``, ``atom_F-``.

    The merged BH76 / W4-11 species dict carries the same oxygen atom under
    both ``O`` and ``o``, so the oracle set renames every free atom by element
    symbol and charge instead of carrying the pool key through.
    """
    if charge == 0:
        return f"atom_{symbol}"
    sign = "-" if charge < 0 else "+"
    magnitude = "" if abs(charge) == 1 else str(abs(charge))
    return f"atom_{symbol}{sign}{magnitude}"


def is_atom_system(mol_spec) -> bool:
    """True for a single free atom (one element, one nucleus)."""
    comp = tuple(mol_spec.atom_composition)
    return len(comp) == 1 and int(comp[0][1]) == 1


def _composition_from_atom_string(atom: str):
    """Sorted ``((symbol, count), ...)`` from a PySCF geometry string."""
    counts: dict[str, int] = {}
    for field in atom.split(";"):
        symbol = field.strip().split()[0]
        counts[symbol] = counts.get(symbol, 0) + 1
    return tuple((s, counts[s]) for s in sorted(counts))


def build_oracle_set(cfg, arch_name: str) -> tuple:
    """The certificate's systems for ``arch_name``, in a byte-stable order.

    Free atoms first (sorted by canonical name), then molecules (sorted by
    name), so the per-system table of two certificates is directly diffable.

    Composition:
      * every free atom of the BH76 / W4-11 pools, at the pool's own charge
        and spin (the species the held-out atomization energies are built from);
      * the DFS pretraining set at the level matching the architecture's
        parent (30 systems for a GGA rung, 28 for a meta-GGA one);
      * H2O, N2 and CH4, the three molecules the pre-certificate offsets were
        measured on, unless the DFS set already carries the name;
      * one neutral ground-state free atom for every element any molecule
        dissociates into (Li and Na appear in the DFS molecules but in neither
        pool), so every atomization offset can be formed.
    """
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    from xcquinox.alec.dfs_pretrain_set import dfs_pretrain_records

    basis = cfg.inputs.basis
    grid_level = cfg.inputs.grid_level
    level = dfs_level_for_parent(resolve_parent(arch_name))

    atom_spin: dict[tuple[str, int], int] = {}

    def _add_atom(symbol, charge, spin, source):
        key = (str(symbol), int(charge))
        previous = atom_spin.get(key)
        if previous is not None and previous != int(spin):
            raise ValueError(
                f"free atom {symbol} charge {charge} carries 2S={previous} "
                f"and 2S={spin} in different oracle sources ({source}); the "
                "certificate needs exactly one spin per free atom")
        atom_spin[key] = int(spin)

    pool_specs, _pool_reactions = load_full_held_out_pools(
        basis=basis, grid_level=grid_level)
    for ms in pool_specs.values():
        if is_atom_system(ms):
            _add_atom(ms.atom_composition[0][0], ms.charge, ms.spin,
                      "BH76/W4-11 pools")

    molecules: dict[str, object] = {}
    for record in dfs_pretrain_records(level):
        if record["kind"] == "atom":
            _add_atom(record["atom_composition"][0][0], record["charge"],
                      record["spin"], "DFS pretraining set")
            continue
        molecules[record["name"]] = MoleculeSpec(
            name=record["name"], atom=record["atom"], basis=basis,
            charge=int(record["charge"]), spin=int(record["spin"]),
            atom_composition=tuple((str(s), int(n))
                                   for s, n in record["atom_composition"]),
            grid_level=grid_level)

    # SUPERSEDED BY TASK 3 STEP 2: `if name in molecules: continue` lets a DFS
    # record win, which is the rung-dependence Task 3 corrects.
    for name, atom, charge, spin in _FIXED_MOLECULES:
        if name in molecules:
            continue
        molecules[name] = MoleculeSpec(
            name=name, atom=atom, basis=basis, charge=int(charge),
            spin=int(spin),
            atom_composition=_composition_from_atom_string(atom),
            grid_level=grid_level)

    for ms in molecules.values():
        for symbol, _count in ms.atom_composition:
            if (symbol, 0) in atom_spin:
                continue
            if symbol not in _ATOM_GROUND_SPIN:
                raise ValueError(
                    f"no ground-state spin recorded for element {symbol!r}; "
                    "add it to fidelity._ATOM_GROUND_SPIN before certifying "
                    "an architecture on a molecule that contains it")
            _add_atom(symbol, 0, _ATOM_GROUND_SPIN[symbol],
                      "ground-state table")

    atoms: dict[str, object] = {}
    for (symbol, charge), spin in atom_spin.items():
        name = atom_system_name(symbol, charge)
        atoms[name] = MoleculeSpec(
            name=name,
            atom=f"{symbol} 0.0000000000 0.0000000000 0.0000000000",
            basis=basis, charge=int(charge), spin=int(spin),
            atom_composition=((symbol, 1),), grid_level=grid_level)

    return (tuple(atoms[k] for k in sorted(atoms))
            + tuple(molecules[k] for k in sorted(molecules)))
```

- [ ] **Step 4: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/cluster/fidelity.py
python -m pytest xcquinox/alec/tests/test_cluster_fidelity.py -v \
  > /tmp/xcq-testlogs/t2-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: 22 passed.

**Deliverable:** `certificate_status(run_dir, arch)` is the one predicate every gate will call, and `build_oracle_set(cfg, arch)` returns the full deterministic system list -- both importable with no JAX.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_cluster_fidelity.py -v > /tmp/xcq-testlogs/t2-green.log 2>&1`

---

## Task 3: The pool geometries for the three common molecules, and `reference_xc` -- the reference density's functional, in the one construction path

**Depends on:** the spec 3.1 Task 2 change (per-spin precompute blocks) being
COMMITTED. Those `data.py` / `padding.py` edits are in this working tree right
now; this task edits the same two files and must land after them, not race
them. Check before starting: `xcquinox/alec/data.py` must already declare
`dm_features_a` ... `tau_spin_b` on `MoleculeData` and populate them in the
`if is_unrestricted:` block of `precompute_fixed_density_data`. If it does not,
stop and wait for that commit.

Two things, both about WHICH density the certificate measures on. Steps 1-2
correct the committed oracle set so the three headline molecules are one fixed
molecule for every rung; Steps 3-12 make the reference functional a parameter
of the library's one construction path.

The certificate needs the parent functional's OWN self-consistent density for a
meta-GGA architecture, and `precompute_fixed_density_data` hard-codes a PBE SCF.
The density is not one quantity: it is `dm_pbe`, `rho_grid`, `sigma_grid`,
`nabla_rho_grid`, `E_pbe` / `E_xc_pbe` / `E_non_xc`, four total-density
descriptor blocks and ten per-spin ones, all built from it. Rebuilding that set
anywhere else would be a second construction that has to mirror this one
forever -- the exact failure class this program exists to remove. So the
functional becomes a parameter of the one construction path instead.

**Files:**
- Modify: `xcquinox/alec/data.py:197-269` (`MoleculeData`: docstring + the `reference_xc` field), `:284-293` (`_precompute_cache_key` signature), `:317-331` (the key tuple), `:491-501` (`precompute_fixed_density_data` signature), `:502-530` (its docstring), `:539-556` (the reference SCF), `:624-633` (the reference XC energy), `:886-931` (the result construction)
- Modify: `xcquinox/alec/cluster/fidelity.py:114-125` (`_FIXED_MOLECULES` -> `_FIXED_MOLECULE_POOL_NAMES`), `:342-343` and `:388-395` (`build_oracle_set`) -- a correction to Task 2's committed code, sequenced here
- Modify: `xcquinox/alec/padding.py:134-137` (`_STRIP_KEYS`)
- Test: `xcquinox/alec/tests/test_data.py`, `xcquinox/alec/tests/test_shape_padding.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (consumed by Task 4):
  - `data.precompute_fixed_density_data(mol_spec, required_keys=(), descriptors=(), auxbasis=None, orientation_lock_strength=0.0, seed_source="pbe", seed_cache_dir=None, seed_density_fit=False, seed_allow_generate=False, reference_xc="pbe") -> MoleculeData`
  - `MoleculeData["reference_xc"]: str` -- the functional whose self-consistent density every grid quantity and descriptor block in the record was built from.
  - `data._precompute_cache_key(..., reference_xc: str = "pbe")` -- the key gains a `reference_xc` slot so a PBE record is never handed to a SCAN caller.
  - `fidelity._FIXED_MOLECULE_POOL_NAMES: tuple[tuple[str, str], ...]` == `(("H2O", "H2O"), ("N2", "n2"), ("CH4", "CH4"))` -- certificate name -> merged-pool species key for the three unconditional molecules.

Naming decision to carry into the code comments: the `dm_pbe` / `rho_grid` /
`E_pbe` / `E_xc_pbe` / `E_non_xc` key names are KEPT. They have always meant
"the reference SCF's", and PBE was merely the only reference available;
renaming them would touch every consumer in the library for no physical gain.
Recording `reference_xc` beside them makes the meaning checkable instead of
implied, which is what the certificate and `validate_run` need.

- [ ] **Step 1: Correct the committed oracle set to use the pool geometries**

This is a correction to Task 2's code (`cluster/fidelity.py`, already
committed), sequenced here because it must land before any certificate is
computed. Task 2 let a DFS record of the same name win over the unconditional
H2O / N2 / CH4, which makes the certificate's three headline numbers
rung-dependent: the DFS set carries N2 only at its GGA level, so a GGA-rung
architecture was certified on the DFS N2 (measured r = 1.0987920 A) while a
meta-GGA architecture, whose DFS level drops N2, fell through to a literal
geometry. `dAE(N2)` was then not the same physical quantity across rungs, and
neither was `dAE(CH4)` (DFS r(CH) = 1.0918537 A against the pool's 1.0874456
A). The three molecules are exactly the ones spec Section 2 tabulates, so they
must be one fixed molecule for every architecture.

Write the failing tests first. Append to
`xcquinox/alec/tests/test_cluster_fidelity.py`:

```python
# ---------------------------------------------------------------------------
# The three unconditional molecules are ONE molecule for every rung
# ---------------------------------------------------------------------------

def test_unconditional_molecules_are_identical_across_rungs():
    """dAE(H2O), dAE(N2) and dAE(CH4) are the certificate's headline numbers
    and the ones spec Section 2 tabulates. They are comparable between a
    GGA-rung and a meta-GGA architecture only if both are computed on the SAME
    molecule, so all three come from the pools whatever the rung."""
    gga = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    mgga = {ms.name: ms
            for ms in fid.build_oracle_set(_cfg(), "deep_mgga_3x16")}
    for name in ("H2O", "N2", "CH4"):
        assert name in gga and name in mgga, name
        assert gga[name].atom == mgga[name].atom, name
        assert gga[name].spin == mgga[name].spin, name
        assert gga[name].charge == mgga[name].charge, name
        assert gga[name].atom_composition == mgga[name].atom_composition, name


def test_unconditional_molecules_carry_the_pool_geometry():
    """The pool species are the ones the held-out atomization energies are
    scored on, so the certificate measures the functional on the geometry the
    campaign reports."""
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    pool, _rxns = load_full_held_out_pools(basis="sto-3g", grid_level=1)
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    assert fid._FIXED_MOLECULE_POOL_NAMES == (
        ("H2O", "H2O"), ("N2", "n2"), ("CH4", "CH4"))
    for name, pool_key in fid._FIXED_MOLECULE_POOL_NAMES:
        source = pool[pool_key]
        assert systems[name].atom == source.atom, name
        assert systems[name].spin == source.spin, name
        assert systems[name].charge == source.charge, name
        # The pool spec may carry a benchmark reference path; the certificate's
        # copy must not, or the precompute would try to load and shape-check it.
        assert systems[name].external_data_path is None, name


def test_a_dfs_record_never_overrides_an_unconditional_molecule():
    """The DFS pretraining set carries N2 (at its GGA level only) and CH4 at
    its own geometries. Neither may win, or the same architecture family is
    certified on two different N2 molecules depending on its rung."""
    from xcquinox.alec.dfs_pretrain_set import dfs_pretrain_records
    dfs = {r["name"]: r for r in dfs_pretrain_records("gga")}
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    for name in ("N2", "CH4"):
        assert name in dfs, f"the DFS set no longer carries {name}"
        assert systems[name].atom != dfs[name]["atom"], (
            f"{name} resolved to the DFS geometry; the pool geometry must win")


def test_the_dfs_molecules_keep_their_own_geometries():
    """Only the three unconditional names are overridden. Every other DFS
    molecule is still certified at the geometry it was pretrained on."""
    from xcquinox.alec.dfs_pretrain_set import dfs_pretrain_records
    dfs = {r["name"]: r for r in dfs_pretrain_records("gga")
           if r["kind"] == "molecule"}
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    overridden = {name for name, _key in fid._FIXED_MOLECULE_POOL_NAMES}
    checked = 0
    for name, record in dfs.items():
        if name in overridden:
            continue
        assert systems[name].atom == record["atom"], name
        checked += 1
    assert checked >= 18
```

Run them:

```bash
mkdir -p /tmp/xcq-testlogs
python -m pytest xcquinox/alec/tests/test_cluster_fidelity.py \
  -k "unconditional or dfs_record or dfs_molecules" -v \
  > /tmp/xcq-testlogs/t3-oracle-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: `AttributeError: ... has no attribute '_FIXED_MOLECULE_POOL_NAMES'`, and
`test_unconditional_molecules_are_identical_across_rungs` failing on N2
(the GGA set resolves the DFS geometry, the meta-GGA set the literal).

- [ ] **Step 2: Resolve the three from the pools and let them override**

In `xcquinox/alec/cluster/fidelity.py`, replace the `_FIXED_MOLECULES` literal
table (`:114-125`) with a pool-name map:

```python
# The three molecules the pre-certificate offsets were measured on
# (SPEC_pretrain_fidelity_program.md Section 2). Every certificate carries all
# three, whatever the architecture's rung, at the BH76 / W4-11 POOL geometry --
# mapped here from the pool's own species key to the certificate's canonical
# name. Resolving them from the pool rather than from a literal is what makes
# dAE(H2O), dAE(N2) and dAE(CH4) the same physical quantity for a GGA-rung and
# a meta-GGA architecture: the DFS pretraining set carries N2 only at its GGA
# level and at a different bond length (1.0987920 A against the pool's
# 1.0971114 A), and CH4 at a different r(CH) (1.0918537 A against 1.0874456 A),
# so a DFS record of one of these three names must never win. The pool species
# are also the ones the held-out atomization energies are scored on.
# The merged pool keeps the BH76 entry when a name appears in both sets
# (load_full_held_out_pools), so all three resolve to one benchmark's
# geometries; the certificate does not invent a second merge policy.
_FIXED_MOLECULE_POOL_NAMES: tuple[tuple[str, str], ...] = (
    ("H2O", "H2O"),
    ("N2", "n2"),
    ("CH4", "CH4"),
)
```

and replace the loop that consumed it in `build_oracle_set` (`:388-395`) with
one that OVERRIDES rather than defers:

```python
    # The three common molecules override any DFS record of the same name, so
    # every rung's certificate measures the same H2O, N2 and CH4. The DFS
    # molecules that are not among the three keep their own geometries, which
    # are the geometries their pretraining rows were generated at.
    for name, pool_key in _FIXED_MOLECULE_POOL_NAMES:
        source = pool_specs.get(pool_key)
        if source is None:
            raise ValueError(
                f"the BH76 / W4-11 pools carry no species {pool_key!r}, so "
                f"the certificate's unconditional molecule {name!r} cannot be "
                "resolved to a pool geometry")
        molecules[name] = MoleculeSpec(
            name=name, atom=source.atom, basis=basis,
            charge=int(source.charge), spin=int(source.spin),
            atom_composition=tuple((str(s), int(n))
                                   for s, n in source.atom_composition),
            grid_level=grid_level)
```

Update the `build_oracle_set` docstring's composition bullet (`:342-343`) from

```
      * H2O, N2 and CH4, the three molecules the pre-certificate offsets were
        measured on, unless the DFS set already carries the name;
```

to

```
      * H2O, N2 and CH4 -- the three molecules the pre-certificate offsets were
        measured on -- at the POOL geometry, for every rung, overriding any DFS
        record of the same name so the three headline atomization offsets are
        the same physical quantity across architectures;
```

Then:

```bash
python -m py_compile xcquinox/alec/cluster/fidelity.py
python -m pytest xcquinox/alec/tests/test_cluster_fidelity.py -v \
  > /tmp/xcq-testlogs/t3-oracle-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass, including Task 2's own oracle-set tests
(`test_oracle_set_carries_the_dfs_molecules_and_the_fixed_three`,
`test_meta_gga_oracle_set_drops_h2_but_keeps_n2_from_the_fixed_three`) -- the
latter still holds, since N2 is now unconditional for both rungs rather than
restored by a literal.

- [ ] **Step 3: Write the failing tests**

Append to `xcquinox/alec/tests/test_data.py`:

```python
# ---------------------------------------------------------------------------
# reference_xc: the functional whose self-consistent density the record holds
# ---------------------------------------------------------------------------
# Reproducibility note, measured on this machine before these tests were
# written: two INDEPENDENT SCF runs of the same closed-shell system agree to
# ~5e-14 Ha in energy but only to ~5e-8 in the dimensionless meta-GGA alpha (a
# ratio that amplifies round-off in sigma), and two runs of a DEGENERATE
# open-shell radical (OH) can converge to different orientations of the singly
# occupied pi orbital, differing by O(100) in sigma_grid point-wise. So the
# "unchanged default" pin below is an OBJECT-IDENTITY pin through the memo
# cache plus the untouched existing suite, not a bitwise comparison of two
# separate SCF runs, which no SCF in this library would pass.

_H2O_ATOM = ("O 0.0000000000 0.0000000000 0.0000000000; "
             "H 0.0000000000 0.7570000000 0.5870000000; "
             "H 0.0000000000 -0.7570000000 0.5870000000")


def _h2o_spec():
    from xcquinox.alec.config import MoleculeSpec
    return MoleculeSpec(name="H2O_refxc", atom=_H2O_ATOM, basis="sto-3g",
                        charge=0, spin=0,
                        atom_composition=(("H", 2), ("O", 1)), grid_level=1)


def _oh_spec():
    from xcquinox.alec.config import MoleculeSpec
    return MoleculeSpec(name="OH_refxc", atom="O 0 0 0; H 0 0 0.97",
                        basis="sto-3g", charge=0, spin=1,
                        atom_composition=(("H", 1), ("O", 1)), grid_level=1)


def test_reference_xc_defaults_to_pbe_and_is_recorded():
    """The record states which functional's density it holds, so a consumer
    can assert it instead of assuming PBE."""
    from xcquinox.alec.data import precompute_fixed_density_data
    md = precompute_fixed_density_data(_h2o_spec())
    assert md["reference_xc"] == "pbe"


def test_explicit_pbe_is_the_same_record_as_the_default():
    """`reference_xc="pbe"` and the default are ONE cache entry and one SCF:
    the default path is unchanged, and no consumer silently pays for a second
    reference SCF by spelling the default out."""
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data,
                                    set_precompute_cache_enabled)
    set_precompute_cache_enabled(True)
    clear_precompute_cache()
    spec = _h2o_spec()
    a = precompute_fixed_density_data(spec)
    b = precompute_fixed_density_data(spec, reference_xc="pbe")
    assert a is b


def test_reference_xc_scan_reproduces_a_standalone_pyscf_scan_scf():
    """The record's total energy IS the reference functional's SCF energy: a
    SCAN record must reproduce a plain PySCF SCAN calculation of the same
    molecule on the same grid."""
    import numpy as np
    from pyscf import dft, gto
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    spec = _h2o_spec()
    md = precompute_fixed_density_data(spec, reference_xc="scan")
    assert md["reference_xc"] == "scan"

    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    mf.kernel()
    assert mf.converged
    assert abs(float(md["E_pbe"]) - float(mf.e_tot)) < 1e-8
    assert np.allclose(np.asarray(md["dm_pbe"]), np.asarray(mf.make_rdm1()),
                       atol=1e-7)


def test_reference_xc_scan_moves_the_density_and_the_energy():
    """A SCAN record is not a relabelled PBE record. H2O/sto-3g has real
    variational freedom (5 occupied orbitals in a 7-function basis); H2 and the
    H atom do NOT -- their densities are fixed by symmetry and normalization,
    so they cannot serve as this pin."""
    import numpy as np
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    spec = _h2o_spec()
    pbe = precompute_fixed_density_data(spec, reference_xc="pbe")
    scan = precompute_fixed_density_data(spec, reference_xc="scan")
    assert np.max(np.abs(np.asarray(pbe["dm_pbe"])
                         - np.asarray(scan["dm_pbe"]))) > 1e-4
    assert np.max(np.abs(np.asarray(pbe["rho_grid"])
                         - np.asarray(scan["rho_grid"]))) > 1e-5
    assert abs(float(pbe["E_pbe"]) - float(scan["E_pbe"])) > 1e-3


def test_reference_xc_scan_rebuilds_every_grid_quantity_from_that_density():
    """Every grid quantity in the record is a contraction of the record's own
    density matrix with its own AO table -- for any reference functional."""
    import numpy as np
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    md = precompute_fixed_density_data(_h2o_spec(), reference_xc="scan")
    ao = np.asarray(md["ao_grid_deriv"])
    dm = np.asarray(md["dm_pbe"])
    dm_tot = dm if dm.ndim == 2 else dm[0] + dm[1]
    rho = np.einsum("pi,ij,pj->p", ao[0], dm_tot, ao[0])
    gx = 2 * np.einsum("pi,ij,pj->p", ao[1], dm_tot, ao[0])
    gy = 2 * np.einsum("pi,ij,pj->p", ao[2], dm_tot, ao[0])
    gz = 2 * np.einsum("pi,ij,pj->p", ao[3], dm_tot, ao[0])
    assert np.allclose(np.asarray(md["rho_grid"]), rho, atol=1e-12)
    assert np.allclose(np.asarray(md["sigma_grid"]),
                       gx ** 2 + gy ** 2 + gz ** 2, atol=1e-10)
    # E_non_xc is the reference SCF's total minus its own XC energy.
    assert abs(float(md["E_non_xc"])
               - (float(md["E_pbe"]) - float(md["E_xc_pbe"]))) < 1e-12


def test_reference_xc_scan_populates_the_per_spin_blocks_by_the_same_path():
    """The per-spin-channel blocks follow the reference density with no
    special-casing: they are built from the record's own density matrix in the
    one open-shell branch, whatever functional produced it."""
    import numpy as np
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    arch = ArchitectureConfig.from_spec(
        "refxc_probe", 3, 16,
        descriptors=["cusp", "dm_statistics", "rung35",
                     "rung35_multishell", "metagga"],
        meta_gga=True)
    desc = arch.materialize_descriptors()
    req = tuple(sorted({k for d in desc for k in d.required_mol_keys}))
    clear_precompute_cache()
    md = precompute_fixed_density_data(_oh_spec(), required_keys=req,
                                       descriptors=desc, reference_xc="scan")
    assert md["reference_xc"] == "scan"
    for key in ("dm_features_a", "dm_features_b",
                "rung35_features_a", "rung35_features_b",
                "rung35ms_features_a", "rung35ms_features_b",
                "metagga_features_a", "metagga_features_b",
                "tau_spin_a", "tau_spin_b"):
        assert md[key] is not None, key
        assert np.all(np.isfinite(np.asarray(md[key]))), key
    # The per-spin tau contracts the record's OWN spin-resolved density matrix.
    from xcquinox.alec.metagga import compute_tau_from_dm
    import jax.numpy as jnp
    dm = jnp.asarray(md["dm_pbe"])
    for slot, key in ((0, "tau_spin_a"), (1, "tau_spin_b")):
        want = compute_tau_from_dm(md["ao_grid_deriv"][1:4], dm[slot])
        assert np.allclose(np.asarray(md[key]), np.asarray(want), atol=1e-12)


def test_cache_key_separates_reference_xc():
    from xcquinox.alec.data import _precompute_cache_key
    spec = _h2o_spec()
    a = _precompute_cache_key(spec, (), (), None, 0.0, "pbe", None, False,
                              reference_xc="pbe")
    b = _precompute_cache_key(spec, (), (), None, 0.0, "pbe", None, False,
                              reference_xc="scan")
    assert a != b


def test_cache_never_hands_a_pbe_record_to_a_scan_caller():
    """The failure a reference_xc-blind cache key would cause: a SCAN
    certificate silently measured against the PBE density."""
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data,
                                    set_precompute_cache_enabled)
    set_precompute_cache_enabled(True)
    clear_precompute_cache()
    spec = _h2o_spec()
    pbe = precompute_fixed_density_data(spec, reference_xc="pbe")
    scan = precompute_fixed_density_data(spec, reference_xc="scan")
    assert scan is not pbe
    assert scan["reference_xc"] == "scan"
    assert abs(float(pbe["E_pbe"]) - float(scan["E_pbe"])) > 1e-3


def test_reference_xc_must_be_a_non_empty_string():
    import pytest
    from xcquinox.alec.data import precompute_fixed_density_data
    with pytest.raises(ValueError, match="reference_xc"):
        precompute_fixed_density_data(_h2o_spec(), reference_xc="")
```

Append to `xcquinox/alec/tests/test_shape_padding.py`:

```python
def test_padding_strips_the_reference_xc_provenance_string():
    """`reference_xc` is run-level provenance the energy kernel never reads.
    Leaving a string leaf in the padded pytree would add a static leaf to the
    per-molecule JIT key for no benefit, so the pad pass strips it -- exactly
    as it strips `name` and `atom_composition`."""
    from xcquinox.alec.padding import _STRIP_KEYS, _pad_mol_data, PadTarget
    import jax.numpy as jnp
    assert "reference_xc" in _STRIP_KEYS
    md = {"reference_xc": "scan",
          "s_matrix": jnp.eye(2),
          "grid_weights": jnp.ones(3)}
    out = _pad_mol_data(md, PadTarget(n_ao=2, n_grid=3, naux=None))
    assert "reference_xc" not in out
```

- [ ] **Step 4: Run the tests to verify they fail**

```bash
mkdir -p /tmp/xcq-testlogs
python -m pytest xcquinox/alec/tests/test_data.py \
  xcquinox/alec/tests/test_shape_padding.py -v \
  > /tmp/xcq-testlogs/t3-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: the new tests fail with
`TypeError: precompute_fixed_density_data() got an unexpected keyword argument 'reference_xc'`
and `KeyError: 'reference_xc'`; every pre-existing test in both files passes.

- [ ] **Step 5: Declare the field on `MoleculeData`**

In `xcquinox/alec/data.py`, extend the `MoleculeData` docstring (`:197-199`) so
the reference-SCF naming is stated rather than implied:

```python
class MoleculeData(TypedDict, total=True):
    """Pre-computed training/test data for one molecule.
    Every key is always present; unused keys are None.

    REFERENCE-SCF FIELDS. ``dm_pbe``, ``rho_grid``, ``sigma_grid``,
    ``nabla_rho_grid``, ``vxc_pbe``, ``E_pbe``, ``E_xc_pbe``, ``E_non_xc`` and
    every descriptor block hold quantities of the REFERENCE self-consistent
    field, whose functional is recorded in ``reference_xc``. The names carry
    ``pbe`` because PBE was the only reference the precompute could produce
    when they were introduced; they are kept so no consumer has to be rewritten
    for a naming change that carries no physics. A consumer that depends on the
    reference being a particular functional asserts ``reference_xc`` rather than
    reading the name.
    """
```

and add the field immediately after `metagga_features` and the per-spin block
(`:268`, before `eri`):

```python
    # The functional of the reference SCF that produced every quantity above:
    # the density matrix, the grid quantities, the total and per-spin
    # descriptor blocks, and E_pbe / E_xc_pbe / E_non_xc. "pbe" for the whole
    # training and evaluation pipeline; the pretraining-fidelity certificate
    # requests "scan" for a meta-GGA architecture, whose parent functional is
    # SCAN, so the network is measured on the density it must reproduce.
    reference_xc: str
```

- [ ] **Step 6: Put `reference_xc` in the cache key**

In `_precompute_cache_key` (`:284-293`), add the parameter last:

```python
def _precompute_cache_key(
    mol_spec: MoleculeSpec,
    required_keys: tuple[str, ...],
    descriptors: tuple[Descriptor, ...],
    auxbasis: str | None = None,
    orientation_lock_strength: float = 0.0,
    seed_source: str = "pbe",
    seed_cache_dir: str | None = None,
    seed_density_fit: bool = False,
    reference_xc: str = "pbe",
) -> tuple:
```

and extend the returned tuple (`:329-331`), keeping every existing slot in
place so no cached key layout shifts meaning:

```python
    # The seed axis: a seed-blind key would hand a "pbe"-seeded record to a
    # "scan"/"minao" caller (or vice versa). seed_cache_dir and the DF flag
    # are part of the loaded file's identity, so they key too.
    # reference_xc keys for the same reason and a stronger one: it selects the
    # SCF that produced the density EVERY field is built from, so a blind key
    # would hand a PBE record to a SCAN caller and the fidelity certificate
    # would silently measure a meta-GGA network against the wrong density.
    return (mol_spec, tuple(sorted(required_keys)), desc_key, ext_key, auxbasis,
            float(orientation_lock_strength),
            (str(seed_source), seed_cache_dir, bool(seed_density_fit)),
            str(reference_xc))
```

- [ ] **Step 7: Thread it through `precompute_fixed_density_data`**

Signature (`:491-501`):

```python
def precompute_fixed_density_data(
    mol_spec: MoleculeSpec,
    required_keys: tuple[str, ...] = (),
    descriptors: tuple[Descriptor, ...] = (),
    auxbasis: str | None = None,
    orientation_lock_strength: float = 0.0,
    seed_source: str = "pbe",
    seed_cache_dir: str | None = None,
    seed_density_fit: bool = False,
    seed_allow_generate: bool = False,
    reference_xc: str = "pbe",
) -> MoleculeData:
```

Docstring first line and a new paragraph (`:502-530`):

```python
    """Run the reference SCF, extract grid data, return a MoleculeData dict.

    ``reference_xc`` selects the functional of that SCF, and therefore the
    density every grid quantity, every descriptor block (total-density and
    per-spin-channel) and ``E_pbe`` / ``E_xc_pbe`` / ``E_non_xc`` are built
    from. It is recorded in the result as ``reference_xc``. The default
    ``"pbe"`` is the whole training and evaluation pipeline; the
    pretraining-fidelity certificate requests ``"scan"`` for a meta-GGA
    architecture, because SCAN is the parent functional those networks were
    pretrained against and the certificate must measure them on the density
    they have to reproduce. This is deliberately a parameter of this one
    construction rather than a second construction elsewhere: the density
    determines eighteen separate fields, and two code paths building them
    would have to be kept identical by hand.
```

Validation and the cache key (`:530-540`, beside the existing `seed_source`
check):

```python
    if seed_source not in ("pbe", "scan", "minao"):
        raise ValueError(
            f"seed_source must be one of 'pbe'/'scan'/'minao', got "
            f"{seed_source!r}")
    if not isinstance(reference_xc, str) or not reference_xc.strip():
        raise ValueError(
            f"reference_xc must be a non-empty pyscf/libxc functional string, "
            f"got {reference_xc!r}")
    cache_key = None
    if _PRECOMPUTE_CACHE_ENABLED:
        try:
            cache_key = _precompute_cache_key(
                mol_spec, required_keys, descriptors, auxbasis,
                orientation_lock_strength, seed_source, seed_cache_dir,
                seed_density_fit, reference_xc)
```

The SCF itself (`:549`):

```python
    mf.xc = reference_xc
```

- [ ] **Step 8: Make the reference XC energy follow the reference functional**

The closed-shell branch evaluates the XC energy density point-wise from a GGA
row set, which cannot carry the kinetic-energy density a meta-GGA needs.
Dispatch on the functional's RUNG -- a physical property, not a string special
case -- and leave the GGA/LDA arm byte-identical to what it was. Replace
`:624-633`:

```python
    # Reference XC energy and E_non_xc
    _xctype = libxc.xc_type(reference_xc)
    if dm_pbe.ndim == 3:  # UKS
        # Use pyscf's veff.exc which already has the correct spin-resolved
        # evaluation. The `veff` object was computed above
        # (mf.get_veff(mol, dm_pbe)); reuse its .exc.
        E_xc_pbe = float(veff.exc)
    elif _xctype == "MGGA":
        # A meta-GGA needs the kinetic-energy density, which the GGA row set
        # below cannot carry, so the closed-shell meta-GGA reference reads the
        # XC energy pyscf already accumulated on this grid. Measured against
        # the point-wise route on a GGA reference the two agree to 1.2e-16 Ha,
        # so this arm is not a different quantity, only a different assembly.
        E_xc_pbe = float(veff.exc)
    else:  # RKS, LDA/GGA reference
        rho_for_xc = mf._numint.eval_rho(mol, ao, dm_pbe_tot, xctype="GGA")
        exc_pbe, _, _, _ = mf._numint.eval_xc(reference_xc, rho_for_xc, spin=0)
        E_xc_pbe = float(np.sum(rho_pbe * exc_pbe * weights))
    E_non_xc = E_pbe - E_xc_pbe
```

and add the import beside the other lazy pyscf imports at the top of the
function body (`:511`, next to `from pyscf import dft, gto`):

```python
    from pyscf import dft, gto
    from pyscf.dft import libxc
```

- [ ] **Step 9: Record it on the result**

In the `MoleculeData(...)` construction (`:886-931`), add after `tau_spin_b`:

```python
        tau_spin_b=tau_spin_b,
        reference_xc=reference_xc,
        eri=eri,
```

- [ ] **Step 10: Strip it in the pad pass**

In `xcquinox/alec/padding.py`, extend `_STRIP_KEYS` (`:134-137`):

```python
# Molecule-identifying leaves the manual-backend energy never reads (verified
# energy-neutral); stripping them stops them keying the per-molecule compile.
# ``reference_xc`` joins them as run-level provenance: the energy kernel never
# reads it, and a string leaf in the padded pytree would key the compile for
# nothing.
_STRIP_KEYS = ("_pyscfad_mol", "name", "atom_composition", "reference_xc")
```

- [ ] **Step 11: `py_compile` and run the new tests GREEN**

```bash
python -m py_compile xcquinox/alec/data.py xcquinox/alec/padding.py
python -m pytest xcquinox/alec/tests/test_data.py \
  xcquinox/alec/tests/test_shape_padding.py -v \
  > /tmp/xcq-testlogs/t3-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass.

- [ ] **Step 12: Prove the default path is unchanged, on the suite that pins it**

The strongest available evidence that `reference_xc="pbe"` is byte-identical to
the previous behaviour is that every existing consumer of
`precompute_fixed_density_data` still produces its pinned numbers. Two
independent SCF runs of the same molecule are NOT bitwise equal in this library
(measured on this machine: ~5e-14 Ha in energy for a closed shell, and a
degenerate open-shell radical can converge to a different orientation of its
singly occupied orbital entirely), so a bitwise round-trip of two separate
calls is not a test any correct implementation would pass. Run instead:

```bash
python -m pytest \
  xcquinox/alec/tests/test_data.py \
  xcquinox/alec/tests/test_data_cderi.py \
  xcquinox/alec/tests/test_shape_padding.py \
  xcquinox/alec/tests/test_solv01_split_xc.py \
  xcquinox/alec/tests/test_descriptors.py \
  -v > /tmp/xcq-testlogs/t3-default.log 2>&1; echo "rc=$?"
```
Read the log in full. Expected: no failures and no new warnings. A failure here
means the default path moved and must be restored before continuing.

**Deliverable:** `precompute_fixed_density_data(spec, reference_xc="scan")` returns a record whose density, grid quantities and every descriptor block are SCAN's, through the same code the PBE path uses, with `reference_xc` recorded and keyed in the cache.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_data.py xcquinox/alec/tests/test_shape_padding.py -v > /tmp/xcq-testlogs/t3-green.log 2>&1`

---

## Task 4: `cluster/fidelity.py` -- the physics layer, the certificate and its entrypoint

**Files:**
- Modify: `xcquinox/alec/cluster/fidelity.py` (append after `build_oracle_set`)
- Modify: `xcquinox/alec/tests/test_cluster_fidelity.py` (append)

**Interfaces:**
- Consumes: everything Task 2 produced, plus `data.precompute_fixed_density_data(..., reference_xc=)` and `MoleculeData["reference_xc"]` (Task 3).
- Produces (read by Tasks 6 and 13):
  - `build_certified_model(cfg, run_dir, arch_name) -> tuple[ArchitectureConfig, AlecGGAModel]`
  - `evaluate_system(model, descriptors, mol_spec, *, parent, auxbasis=None, orientation_lock_strength=0.0) -> dict`
  - `fidelity_certificate(cfg, run_dir, arch_name, *, oracle_set=None, evaluate=None) -> dict` -- writes and returns the payload of the schema above
  - `main(argv=None) -> int` and a `__main__` guard

The parent density is NOT constructed here. `evaluate_system` asks the library
for a record built on the parent functional's own self-consistent density --
`precompute_fixed_density_data(mol_spec, ..., reference_xc=parent)` -- and then
uses the ordinary energy path on it. Nothing in this module rebuilds a grid
quantity, a descriptor block or a density matrix, so there is no second
construction to keep in step with `data.py` by hand. What this module does add
is the PARENT's exchange-correlation energy on that record in a form directly
comparable with the network's, cross-checked against two independent routes per
system:

1. `_parent_exc_on_stored_grid` -- point-wise libxc on the record's stored AO
   table, density matrix and grid weights. This is the PRIMARY number: it uses
   literally the quadrature the network is integrated on, so no grid mismatch
   is possible between the two sides of `dE_xc`. Assembling libxc's input rows
   (including tau for a meta-GGA parent) is an evaluation, not a second
   construction of a `mol_data` field.
2. `_parent_exc_numint` -- PySCF's own `nr_rks` / `nr_uks` on a freshly built,
   unpruned grid of the same level. Recorded as `parent_grid_diff_Ha`.
3. `mol_data["E_xc_pbe"]` -- the XC energy PySCF accumulated during the
   reference SCF itself. Free, and recorded as `parent_record_diff_Ha`.

Both differences are bounded by `PARENT_GRID_TOL_HA`; measured agreement is
2.6e-11 Ha at sto-3g and 2.0e-10 Ha at the production identity.

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_cluster_fidelity.py`:

```python
# ---------------------------------------------------------------------------
# Anti-fork guard: no second construction of a precompute quantity
# ---------------------------------------------------------------------------

def test_fidelity_never_rebuilds_a_precompute_field():
    """Every grid quantity and every descriptor block reaches the certificate
    through data.precompute_fixed_density_data(..., reference_xc=...). A second
    construction here would have to be kept identical to data.py by hand
    forever, which is the failure class this certificate exists to remove.

    Assembling libxc's own input rows inside _parent_exc_on_stored_grid is not
    a construction of a mol_data field and is deliberately not listed."""
    import inspect
    src = inspect.getsource(fid)
    for forbidden in ("compute_rung35_occupancy",
                      "compute_rung35_multishell_occupancy",
                      "compute_dm_features_array",
                      "compute_alpha",
                      "doubled_spin_dm",
                      "nabla_rho_grid",
                      "rung35_proj_ao",
                      "rung35ms_proj_ao"):
        assert forbidden not in src, (
            f"fidelity.py references {forbidden!r}: the parent density's grid "
            "quantities and descriptor blocks must come from "
            "precompute_fixed_density_data(..., reference_xc=...), not from a "
            "second construction in this module")


# ---------------------------------------------------------------------------
# Model construction and the parent-density request
# ---------------------------------------------------------------------------

def test_build_certified_model_loads_the_checkpoint_not_the_skeleton(tmp_path):
    """The skeleton's seed fixes the tree SHAPE only; every array leaf comes
    from the checkpoint. A builder that returned the skeleton would certify a
    randomly initialised network -- exactly the state the gate exists to
    catch."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.networks import create_network_pair

    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_3x16", seed=7)
    arch = get_architecture("deep_3x16")
    _built, model = fid.build_certified_model(_cfg(pretrain_seed=99), run_dir,
                                              "deep_3x16")
    from_checkpoint, _ = create_network_pair(arch, seed=7)
    from_skeleton, _ = create_network_pair(arch, seed=99)
    got = jax.tree_util.tree_leaves(eqx.filter(model.xnet, eqx.is_array))
    want = jax.tree_util.tree_leaves(eqx.filter(from_checkpoint, eqx.is_array))
    other = jax.tree_util.tree_leaves(eqx.filter(from_skeleton, eqx.is_array))
    assert len(got) == len(want) == len(other)
    assert all(bool(jnp.allclose(a, b)) for a, b in zip(got, want))
    # The two seeds really do differ, so the assertion above has content.
    assert any(not bool(jnp.allclose(a, b)) for a, b in zip(want, other))


def test_evaluate_system_requests_the_parent_functionals_density(monkeypatch):
    """The certificate asks the library's one construction path for a record
    built on the PARENT functional's self-consistent density, and forwards the
    run's Coulomb backend and orientation lock unchanged."""
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.models import AlecGGAModel

    seen = {}
    original = data_mod.precompute_fixed_density_data

    def _spy(mol_spec, **kwargs):
        seen.update(kwargs)
        return original(mol_spec, **kwargs)

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _spy)
    arch = get_architecture("deep_3x16")
    model = AlecGGAModel.from_arch(arch, seed=0)
    rec = fid.evaluate_system(model, arch.materialize_descriptors(),
                              _tiny_oracle_set()[0], parent="pbe",
                              auxbasis=None, orientation_lock_strength=0.0)
    assert seen["reference_xc"] == "pbe"
    assert seen["orientation_lock_strength"] == 0.0
    assert rec["reference_xc"] == "pbe"
    assert rec["is_atom"] is True


def test_evaluate_system_refuses_a_record_built_on_another_functional(
        monkeypatch):
    """A record whose reference_xc is not the parent would measure the network
    against the wrong density; that raises rather than entering the table."""
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.models import AlecGGAModel

    original = data_mod.precompute_fixed_density_data

    def _mislabel(mol_spec, **kwargs):
        md = dict(original(mol_spec, **kwargs))
        md["reference_xc"] = "lda,vwn"
        return md

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _mislabel)
    arch = get_architecture("deep_3x16")
    model = AlecGGAModel.from_arch(arch, seed=0)
    with pytest.raises(ValueError, match="reference_xc"):
        fid.evaluate_system(model, arch.materialize_descriptors(),
                            _tiny_oracle_set()[0], parent="pbe")


def test_meta_gga_architecture_is_certified_against_scan(tmp_path, monkeypatch):
    """End to end for the rung that motivated reference_xc: a meta-GGA
    architecture's certificate must be computed against SCAN, on SCAN's own
    density."""
    import xcquinox.alec.data as data_mod
    seen = []
    original = data_mod.precompute_fixed_density_data

    def _spy(mol_spec, **kwargs):
        seen.append(kwargs.get("reference_xc"))
        return original(mol_spec, **kwargs)

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _spy)
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_mgga_3x16", seed=0)
    payload = fid.fidelity_certificate(
        _cfg(arch=("deep_mgga_3x16",), pretrain_seed=0), run_dir,
        "deep_mgga_3x16", oracle_set=_tiny_oracle_set())
    assert payload["parent"] == "scan"
    assert set(seen) == {"scan"}
    assert all(r["reference_xc"] == "scan" for r in payload["per_system"])


# ---------------------------------------------------------------------------
# The certificate, with the per-system evaluation mocked at the seam
# ---------------------------------------------------------------------------

def _fake_evaluate(table):
    """Build an ``evaluate`` seam returning canned dE_xc (mHa) per name."""
    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        d = table[mol_spec.name]
        return {"name": mol_spec.name, "spin": int(mol_spec.spin),
                "charge": int(mol_spec.charge),
                "is_atom": fid.is_atom_system(mol_spec),
                "n_grid": 10, "reference_xc": parent,
                "E_xc_nn": -1.0 + d / fid.HA_TO_MHA, "E_xc_parent": -1.0,
                "E_xc_parent_numint": -1.0, "E_xc_parent_record": -1.0,
                "parent_grid_diff_Ha": 0.0, "parent_record_diff_Ha": 0.0,
                "dE_xc_mHa": d, "duration_s": 0.0}
    return _evaluate


def _tiny_oracle_set(basis="sto-3g", grid_level=1):
    from xcquinox.alec.config import MoleculeSpec
    return (
        MoleculeSpec(name="atom_H", atom="H 0.0 0.0 0.0", basis=basis, spin=1,
                     atom_composition=(("H", 1),), grid_level=grid_level),
        MoleculeSpec(name="H2", atom="H 0 0 0.371395; H 0 0 -0.371395",
                     basis=basis, spin=0, atom_composition=(("H", 2),),
                     grid_level=grid_level),
    )


def _stub_checkpoint(run_dir, arch_name="deep_3x16", seed=42):
    """Write a real xnet.eqx + cnet.eqx pair for ``arch_name``."""
    import equinox as eqx
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    arch = get_architecture(arch_name)
    xnet, cnet = create_network_pair(arch, seed=seed)
    d = pretrain_checkpoint_dir(run_dir, arch_name)
    os.makedirs(d, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(d, "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(os.path.join(d, "cnet.eqx"), cnet)
    return d


def test_certificate_passes_within_tolerance_and_writes_the_schema(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg()
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16",
        oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 0.5, "H2": 1.0}))

    assert payload["verdict"] == "PASS"
    assert payload["arch"] == "deep_3x16"
    assert payload["parent"] == "pbe"
    assert payload["identity"] == fid.run_identity(cfg)
    assert payload["tolerances"] == {"tol_AE": 1.0, "tol_atom": 1.0,
                                     "override_reason": None}
    assert payload["enforced"] is True
    assert isinstance(payload["xcquinox_version"], str)
    assert payload["timestamp"].endswith("Z")
    assert payload["duration_s"] >= 0.0
    assert [r["name"] for r in payload["per_system"]] == ["atom_H", "H2"]
    assert [r["name"] for r in payload["per_atomization"]] == ["H2"]
    s = payload["summary"]
    assert s["n_systems"] == 2 and s["n_atoms"] == 1
    assert s["n_atomizations"] == 1 and s["n_failed_systems"] == 0
    assert s["max_parent_grid_diff_Ha"] == pytest.approx(0.0)
    assert s["max_parent_record_diff_Ha"] == pytest.approx(0.0)
    assert s["max_atom_mHa"] == pytest.approx(0.5)
    # dAE = dE_xc(H2) - 2 dE_xc(H) = 1.0 - 1.0 = 0 mHa.
    assert s["max_dAE_kcalmol"] == pytest.approx(0.0, abs=1e-12)
    assert s["failure_reasons"] == []

    on_disk = json.loads(
        open(fid.certificate_path(run_dir, "deep_3x16")).read())
    assert on_disk == payload
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "PASS"


def test_certificate_fails_on_the_atom_tolerance(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 13.7, "H2": 27.4}))
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] == pytest.approx(13.7)
    assert any("tol_atom" in r for r in payload["summary"]["failure_reasons"])
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"


def test_certificate_fails_on_the_atomization_tolerance(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    # dAE(H2) = (1.0 - 2 * 0.1) mHa = 0.8 mHa = 0.502 kcal/mol -> passes at
    # 1.0; scale it up until it does not.
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 0.1, "H2": 5.0}))
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] == pytest.approx(0.1)
    assert payload["summary"]["max_dAE_kcalmol"] == pytest.approx(
        (5.0 - 0.2) / fid.HA_TO_MHA * fid.HA_TO_KCAL)
    assert any("tol_AE" in r for r in payload["summary"]["failure_reasons"])


def test_certificate_honours_configured_tolerances(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg(tol_AE=2.0, tol_atom=2.0,
               override_reason=None)
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 1.5, "H2": 3.0}))
    assert payload["verdict"] == "PASS"
    assert payload["tolerances"]["tol_atom"] == 2.0


def test_certificate_records_the_override_reason(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg(tol_AE=5.0, tol_atom=5.0,
               override_reason="rung-3.5 control arm, documented in HISTORY")
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 4.0, "H2": 8.0}))
    assert payload["verdict"] == "PASS"
    assert payload["tolerances"]["override_reason"] == (
        "rung-3.5 control arm, documented in HISTORY")


def test_certificate_records_the_enforcement_flag(tmp_path):
    """A non-enforcing run still writes the TRUE verdict; only the gates
    change behaviour, and they read the flag out of the certificate."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg(enforce=False,
               override_reason="workflow matrix: 50-step pretrain")
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 13.7, "H2": 27.4}))
    assert payload["verdict"] == "FAIL"
    assert payload["enforced"] is False
    assert payload["tolerances"]["override_reason"] == (
        "workflow matrix: 50-step pretrain")
    # The record layers still see a FAIL ...
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"
    # ... while an on-node gate is allowed to continue.
    allowed, message = fid.gate_certificate(run_dir, "deep_3x16")
    assert allowed is True
    assert "enforcement is OFF" in message


def test_certificate_records_a_system_that_raised_and_fails(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)

    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        if mol_spec.name == "H2":
            raise RuntimeError("SCF blew up")
        return _fake_evaluate({"atom_H": 0.1})(
            model, descriptors, mol_spec, parent=parent)

    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_evaluate)
    assert payload["verdict"] == "FAIL"
    failed = [r for r in payload["per_system"] if "error" in r]
    assert [r["name"] for r in failed] == ["H2"]
    assert "SCF blew up" in failed[0]["error"]
    assert payload["summary"]["n_failed_systems"] == 1
    assert any("could not be evaluated" in r
               for r in payload["summary"]["failure_reasons"])


def test_certificate_fails_when_the_two_parent_grid_routes_disagree(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)

    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        rec = _fake_evaluate({"atom_H": 0.1, "H2": 0.2})(
            model, descriptors, mol_spec, parent=parent)
        rec["parent_grid_diff_Ha"] = 1e-3
        return rec

    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_evaluate)
    assert payload["verdict"] == "FAIL"
    assert any("grid" in r for r in payload["summary"]["failure_reasons"])


def test_certificate_applies_the_polarized_correlation_patch(tmp_path):
    """The pretrain worker builds a polarized cnet when the run is polarized;
    the certificate must load the checkpoint with the SAME architecture or the
    deserialise would fail on the cnet input width."""
    import dataclasses
    from xcquinox.alec.config import get_architecture
    run_dir = str(tmp_path / "run")
    arch = dataclasses.replace(get_architecture("deep_3x16"),
                               use_polarized_correlation=True)
    import equinox as eqx
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    xnet, cnet = create_network_pair(arch, seed=42)
    d = pretrain_checkpoint_dir(run_dir, "deep_3x16")
    os.makedirs(d, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(d, "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(os.path.join(d, "cnet.eqx"), cnet)

    built_arch, model = fid.build_certified_model(
        _cfg(polarized=True), run_dir, "deep_3x16")
    assert built_arch.use_polarized_correlation is True
    assert model is not None


# ---------------------------------------------------------------------------
# REAL physics: H and H2 at sto-3g, networks built in the test (seconds)
# ---------------------------------------------------------------------------

def test_certificate_real_physics_on_h_and_h2_at_sto3g(tmp_path):
    """The whole energy path, for real, on two tiny systems.

    ``deep_3x16`` is built with ``zero_init_final_layer=True``, so a freshly
    seeded network has Fx = Fc = 1 exactly and its E_xc is the LDA exchange
    plus PW92 correlation. Against PBE on the same frozen PBE density that is
    a large, definite offset, so this pins the sign, the magnitude, the
    atomization fold and the FAIL branch at once. Every number the certificate
    reports is re-derived in the test from an independent PySCF route.
    """
    import numpy as np
    from pyscf import dft, gto
    from pyscf.dft import numint

    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_3x16", seed=0)
    cfg = _cfg(pretrain_seed=0)
    systems = _tiny_oracle_set()

    payload = fid.fidelity_certificate(cfg, run_dir, "deep_3x16",
                                       oracle_set=systems)

    by_name = {r["name"]: r for r in payload["per_system"]}
    assert set(by_name) == {"atom_H", "H2"}
    assert by_name["atom_H"]["is_atom"] is True
    assert by_name["H2"]["is_atom"] is False
    assert by_name["atom_H"]["spin"] == 1 and by_name["H2"]["spin"] == 0

    # (1) Every record was built on the PARENT's own self-consistent density,
    #     and the parent energy is that functional on that density, on the
    #     SAME grid PySCF's own nr_rks / nr_uks uses.
    assert all(r["reference_xc"] == "pbe" for r in by_name.values())
    for ms in systems:
        rec = by_name[ms.name]
        mol = gto.M(atom=ms.atom, basis=ms.basis, charge=ms.charge,
                    spin=ms.spin, verbose=0)
        mf = dft.UKS(mol) if ms.spin else dft.RKS(mol)
        mf.xc = "pbe"
        mf.grids.level = ms.grid_level
        mf.kernel()
        grids = dft.Grids(mol)
        grids.level = ms.grid_level
        grids.build()
        ni = numint.NumInt()
        dm = mf.make_rdm1()
        if ms.spin:
            _v, exc, _ = ni.nr_uks(mol, grids, "PBE", dm)
        else:
            _v, exc, _ = ni.nr_rks(mol, grids, "PBE", dm)
        assert rec["E_xc_parent"] == pytest.approx(float(exc), abs=1e-8)
        assert rec["E_xc_parent_numint"] == pytest.approx(float(exc), abs=1e-8)
        assert abs(rec["parent_grid_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
        # Third independent route: the XC energy PySCF accumulated during the
        # reference SCF itself, carried on the record as E_xc_pbe.
        assert abs(rec["parent_record_diff_Ha"]) < fid.PARENT_GRID_TOL_HA

    # (2) dE_xc is exactly the difference the record carries, in mHa.
    for rec in by_name.values():
        assert rec["dE_xc_mHa"] == pytest.approx(
            (rec["E_xc_nn"] - rec["E_xc_parent"]) * fid.HA_TO_MHA, rel=1e-12)

    # (3) The atomization offset is the molecule minus its atoms, in kcal/mol.
    dae = {r["name"]: r["dAE_kcalmol"] for r in payload["per_atomization"]}
    expected = ((by_name["H2"]["dE_xc_mHa"] - 2 * by_name["atom_H"]["dE_xc_mHa"])
                / fid.HA_TO_MHA * fid.HA_TO_KCAL)
    assert dae["H2"] == pytest.approx(expected, rel=1e-12)

    # (4) An LDA-limit network is nowhere near PBE, so the certificate FAILS
    #     at the binding 1.0 mHa / 1.0 kcal/mol tolerances.
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] > 1.0
    assert abs(dae["H2"]) > 1.0
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"


def test_certificate_real_physics_passes_at_a_loosened_tolerance(tmp_path):
    """The PASS branch on real numbers: the same two systems under a
    deliberately loosened tolerance carrying its override reason."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_3x16", seed=0)
    cfg = _cfg(tol_AE=100.0, tol_atom=100.0, pretrain_seed=0,
               override_reason="unit test: pins the PASS branch on real "
                               "sto-3g numbers")
    payload = fid.fidelity_certificate(cfg, run_dir, "deep_3x16",
                                       oracle_set=_tiny_oracle_set())
    assert payload["verdict"] == "PASS"
    assert payload["summary"]["failure_reasons"] == []
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "PASS"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_selects_the_arch_by_index_and_returns_zero_on_pass(
        tmp_path, monkeypatch):
    import yaml
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw = _minimal_raw_config(archs=["deep", "medium", "shallow"])
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(raw, f)

    seen = {}

    def _fake(cfg, rd, arch_name, **kwargs):
        seen["arch"] = arch_name
        return {"verdict": "PASS", "enforced": True,
                "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                               "override_reason": None},
                "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2,
                            "n_systems": 2, "n_atoms": 1,
                            "n_atomizations": 1,
                            "failure_reasons": []}}

    monkeypatch.setattr(fid, "fidelity_certificate", _fake)
    assert fid.main([str(run_dir), "1"]) == 0
    assert seen["arch"] == "medium"


def test_main_returns_zero_on_a_failed_but_unenforced_certificate(
        tmp_path, monkeypatch):
    import yaml
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(_minimal_raw_config(archs=["deep"]), f)
    monkeypatch.setattr(fid, "fidelity_certificate", lambda *a, **k: {
        "verdict": "FAIL", "enforced": False,
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": "workflow matrix"},
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7,
                    "n_systems": 2, "n_atoms": 1, "n_atomizations": 1,
                    "failure_reasons": ["max_atom_mHa"]}})
    assert fid.main([str(run_dir), "0"]) == 0


def test_main_returns_one_on_a_failed_certificate(tmp_path, monkeypatch):
    import yaml
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(_minimal_raw_config(archs=["deep"]), f)
    monkeypatch.setattr(fid, "fidelity_certificate", lambda *a, **k: {
        "verdict": "FAIL", "enforced": True,
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": None},
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7,
                    "n_systems": 2, "n_atoms": 1, "n_atomizations": 1,
                    "failure_reasons": ["max_atom_mHa"]}})
    assert fid.main([str(run_dir), "0"]) == 1


def test_main_rejects_an_out_of_range_arch_index(tmp_path):
    import yaml
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(_minimal_raw_config(archs=["deep"]), f)
    assert fid.main([str(run_dir), "7"]) == 1


def test_main_reports_a_missing_config(tmp_path):
    assert fid.main([str(tmp_path), "0"]) == 1


def _minimal_raw_config(archs):
    """A complete-but-minimal raw config dict load_grid_config accepts."""
    return {
        "sweep": {"arch": list(archs), "loss": ["l2"], "metric": ["l2"],
                  "subset_size": [1], "solver": ["oneshot"]},
        "solvers": {"oneshot": {"mode": "oneshot", "max_cycles": 1}},
        "hyperparams": {"n_steps": 1, "lr_start": 1e-3, "lr_end": 1e-4,
                        "lr_decay_start": 0.5, "grad_clip": 1.0,
                        "gradnorm_alpha": 1.0, "vxc_weight": 1.0,
                        "density_weight": 1.0},
        "inputs": {"external_refs_dir": "/tmp/refs",
                   "subset_ledger_path": "/tmp/ledger.json",
                   "basis": "sto-3g", "grid_level": 1,
                   "output_root": "/tmp/out"},
        "pretrain": {"data_dir": "/tmp/pretrain_data"},
        "cluster": {"partition": "short", "time": "01:00:00", "mem": "8G",
                    "cpus_per_task": 1, "array_throttle": 1,
                    "eval_array_throttle": 1, "max_concurrent_tasks": 10},
        "domain_profile": "dfs_step7",
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_fidelity.py -v \
  > /tmp/xcq-testlogs/t4-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: the Task 2 tests still pass; the new ones fail with
`AttributeError: module 'xcquinox.alec.cluster.fidelity' has no attribute 'build_certified_model'` (and `evaluate_system`, `fidelity_certificate`, `main`).

- [ ] **Step 3: Append the physics layer to `xcquinox/alec/cluster/fidelity.py`**

```python
# ---------------------------------------------------------------------------
# Model construction -- the production builder, not a second one
# ---------------------------------------------------------------------------

def _build_model(arch, pretrain_dir: str, *, seed: int):
    """Load a pretrained xnet/cnet pair through the production model builder.

    Mirrors ``train._build_model``: ``create_network_pair`` supplies a
    skeleton whose every array leaf ``eqx.tree_deserialise_leaves`` overwrites
    from the checkpoint, so the skeleton's seed never reaches the certified
    model; only the architecture (depth, width, attention, descriptor count)
    has to match, and it does by construction.
    """
    import equinox as eqx
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.networks import create_network_pair
    xnet_skeleton, cnet_skeleton = create_network_pair(arch, seed=seed)
    xnet = eqx.tree_deserialise_leaves(
        os.path.join(pretrain_dir, "xnet.eqx"), xnet_skeleton)
    cnet = eqx.tree_deserialise_leaves(
        os.path.join(pretrain_dir, "cnet.eqx"), cnet_skeleton)
    return AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def build_certified_model(cfg, run_dir: str, arch_name: str):
    """``(arch, model)`` for ``arch_name`` as the run itself would build them.

    The registry entry is patched with the run-level polarized-correlation
    override exactly as ``cluster._pretrain`` patches it before pretraining,
    so the cnet input width matches the checkpoint on disk.
    """
    import dataclasses
    from xcquinox.alec.config import get_architecture
    arch = get_architecture(arch_name)
    if getattr(cfg, "use_polarized_correlation", False):
        arch = dataclasses.replace(arch, use_polarized_correlation=True)
    pretrain_dir = pretrain_checkpoint_dir(run_dir, arch_name)
    return arch, _build_model(arch, pretrain_dir, seed=cfg.pretrain.seed)


# ---------------------------------------------------------------------------
# The parent's exchange-correlation energy on the record's own density
# ---------------------------------------------------------------------------

def _parent_exc_on_stored_grid(mol_data, parent: str) -> float:
    """E_xc^parent on the SAME grid and density the network is evaluated on.

    Built from ``mol_data``'s stored AO derivative table, density matrix and
    grid weights, so the parent and the network see byte-identical quadrature
    with no Grids object in between. Assembling libxc's input rows here is an
    EVALUATION of the parent functional, not a second construction of a
    ``mol_data`` field: nothing computed in this function is ever stored back.
    The exchange-correlation energy density is libxc's, per electron of the
    TOTAL density, so the integral is sum_g w_g rho_g eps_g.
    """
    import numpy as np
    from pyscf.dft import numint
    ao = np.asarray(mol_data["ao_grid_deriv"])
    dm = np.asarray(mol_data["dm_pbe"])
    weights = np.asarray(mol_data["grid_weights"])
    per_spin = [0.5 * dm, 0.5 * dm] if dm.ndim == 2 else [dm[0], dm[1]]
    rows = []
    for d in per_spin:
        r = np.einsum("gi,ij,gj->g", ao[0], d, ao[0])
        gx = 2.0 * np.einsum("gi,ij,gj->g", ao[1], d, ao[0])
        gy = 2.0 * np.einsum("gi,ij,gj->g", ao[2], d, ao[0])
        gz = 2.0 * np.einsum("gi,ij,gj->g", ao[3], d, ao[0])
        tau = 0.5 * np.einsum("dgi,ij,dgj->g", ao[1:4], d, ao[1:4])
        rows.append((np.vstack([r, gx, gy, gz]), tau))
    (rho_a, tau_a), (rho_b, tau_b) = rows
    rho_total = rho_a[0] + rho_b[0]
    ni = numint.NumInt()
    xc = _PARENT_XC[parent]
    if bool(mol_data["is_unrestricted"]):
        if parent == "scan":
            lapl = np.zeros_like(rho_a[0])
            exc = ni.eval_xc(xc, (np.vstack([rho_a, lapl, tau_a]),
                                  np.vstack([rho_b, lapl, tau_b])), spin=1)[0]
        else:
            exc = ni.eval_xc(xc, np.stack([rho_a, rho_b], axis=0), spin=1)[0]
    else:
        gga = rho_a + rho_b
        if parent == "scan":
            lapl = np.zeros_like(gga[0])
            exc = ni.eval_xc(xc, np.vstack([gga, lapl, tau_a + tau_b]),
                             spin=0)[0]
        else:
            exc = ni.eval_xc(xc, gga, spin=0)[0]
    return float(np.sum(weights * rho_total * exc))


def _parent_exc_numint(mol_spec, parent: str, dm) -> float:
    """Independent E_xc^parent through PySCF's own ``nr_rks`` / ``nr_uks``.

    Cross-check of :func:`_parent_exc_on_stored_grid` on a freshly built,
    unpruned grid of the same level. The reference SCF prunes its grid on the
    density (``small_rho_cutoff``), so the two point counts differ; the
    integrals agreed to 2.6e-11 Ha on OH/sto-3g and to 2.0e-10 Ha at the
    production identity (scratch probes, 2026-08-20). The difference is
    recorded per system and bounded by :data:`PARENT_GRID_TOL_HA`.
    """
    import numpy as np
    from pyscf import dft, gto
    from pyscf.dft import numint
    mol = gto.M(atom=mol_spec.atom, basis=mol_spec.basis,
                charge=mol_spec.charge, spin=mol_spec.spin, verbose=0)
    grids = dft.Grids(mol)
    if mol_spec.grid_level is not None:
        grids.level = int(mol_spec.grid_level)
    grids.build()
    ni = numint.NumInt()
    dm = np.asarray(dm)
    if dm.ndim == 3:
        _v, exc, _n = ni.nr_uks(mol, grids, _PARENT_XC[parent], dm)
    else:
        _v, exc, _n = ni.nr_rks(mol, grids, _PARENT_XC[parent], dm)
    return float(exc)


# ---------------------------------------------------------------------------
# Per-system evaluation -- the seam the mocked tests replace
# ---------------------------------------------------------------------------

def evaluate_system(model, descriptors, mol_spec, *, parent: str,
                    auxbasis=None, orientation_lock_strength: float = 0.0
                    ) -> dict:
    """dE_xc for one system, on the parent's own density at the run identity.

    The record comes from the library's ONE construction path with
    ``reference_xc=parent``, so its density matrix, grid quantities and every
    descriptor block -- total-density and per-spin-channel -- are the parent
    functional's, built by exactly the code the training pipeline uses. This
    module constructs none of them.

    ``E_xc^NN`` is ``oneshot.fixed_density_total_energy(model, mol_data)
    - mol_data["E_non_xc"]``: the production energy path, minus a term that
    cancels identically (``fixed_density_total_energy`` returns
    ``E_non_xc + E_xc^NN``, so whatever ``E_non_xc`` holds drops out).
    ``E_xc^parent`` is libxc on the same stored grid and density, cross-checked
    against a fresh-grid ``nr_rks``/``nr_uks`` and against the XC energy the
    reference SCF itself accumulated.

    ``seed_source`` is deliberately left at its default: ``dm_seed`` is the SCF
    starting guess, which the fixed-density energy path never reads, and
    requesting the SCAN seed would demand a seed cache the certificate does not
    need -- the parent density here comes from ``reference_xc``, not from the
    seed axis.
    """
    import numpy as np
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.oneshot import fixed_density_total_energy

    t0 = time.time()
    required = tuple(sorted({k for d in descriptors
                             for k in d.required_mol_keys}))
    mol_data = precompute_fixed_density_data(
        mol_spec, required_keys=required, descriptors=descriptors,
        auxbasis=auxbasis,
        orientation_lock_strength=orientation_lock_strength,
        reference_xc=parent)
    got_reference = mol_data["reference_xc"]
    if got_reference != parent:
        raise ValueError(
            f"the precompute returned a record with reference_xc="
            f"{got_reference!r} for {mol_spec.name!r} but the certificate "
            f"asked for {parent!r}; the network would be measured against a "
            "density its parent functional did not produce")

    dm = np.asarray(mol_data["dm_pbe"])
    e_xc_nn = (float(fixed_density_total_energy(model, mol_data))
               - float(mol_data["E_non_xc"]))
    e_xc_parent = _parent_exc_on_stored_grid(mol_data, parent)
    e_xc_parent_numint = _parent_exc_numint(mol_spec, parent, dm)
    # The XC energy pyscf accumulated during the reference SCF itself. Free,
    # and a third independent route to the same number.
    e_xc_parent_record = float(mol_data["E_xc_pbe"])
    n_grid = int(np.asarray(mol_data["grid_weights"]).shape[0])
    del mol_data

    return {
        "name": mol_spec.name,
        "spin": int(mol_spec.spin),
        "charge": int(mol_spec.charge),
        "is_atom": is_atom_system(mol_spec),
        "n_grid": n_grid,
        "reference_xc": got_reference,
        "E_xc_nn": e_xc_nn,
        "E_xc_parent": e_xc_parent,
        "E_xc_parent_numint": e_xc_parent_numint,
        "E_xc_parent_record": e_xc_parent_record,
        "parent_grid_diff_Ha": e_xc_parent - e_xc_parent_numint,
        "parent_record_diff_Ha": e_xc_parent - e_xc_parent_record,
        "dE_xc_mHa": (e_xc_nn - e_xc_parent) * HA_TO_MHA,
        "duration_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------

def fidelity_certificate(cfg, run_dir: str, arch_name: str, *,
                         oracle_set=None, evaluate=None) -> dict:
    """Certify one architecture and write its certificate; return the payload.

    ``oracle_set`` overrides :func:`build_oracle_set` (a short list for a
    probe or a test); ``evaluate`` overrides :func:`evaluate_system` (the seam
    the schema tests replace so no SCF runs).
    """
    import xcquinox

    t0 = time.time()
    parent = resolve_parent(arch_name)
    arch, model = build_certified_model(cfg, run_dir, arch_name)
    descriptors = arch.materialize_descriptors()
    systems = tuple(oracle_set) if oracle_set is not None \
        else build_oracle_set(cfg, arch_name)
    run = evaluate if evaluate is not None else evaluate_system

    # The precompute memoizes on (spec, keys, descriptors); dozens of
    # production-basis grids would exhaust a node's memory long before the
    # sweep finished, and each system is visited exactly once.
    from xcquinox.alec.data import (clear_precompute_cache,
                                    set_precompute_cache_enabled)
    set_precompute_cache_enabled(False)
    clear_precompute_cache()

    inputs = cfg.inputs
    per_system = []
    try:
        for mol_spec in systems:
            try:
                per_system.append(run(
                    model, descriptors, mol_spec, parent=parent,
                    auxbasis=getattr(inputs, "auxbasis", None),
                    orientation_lock_strength=float(
                        getattr(inputs, "orientation_lock_strength", 0.0))))
            except Exception as exc:  # noqa: BLE001 -- recorded, not raised
                per_system.append({
                    "name": mol_spec.name,
                    "error": f"{type(exc).__name__}: {exc}"})
    finally:
        set_precompute_cache_enabled(True)
        clear_precompute_cache()

    ok = {r["name"]: r for r in per_system if "error" not in r}

    per_atomization = []
    for mol_spec in systems:
        if is_atom_system(mol_spec) or mol_spec.name not in ok:
            continue
        atom_terms = []
        missing = None
        for symbol, count in mol_spec.atom_composition:
            atom_name = atom_system_name(symbol, 0)
            if atom_name not in ok:
                missing = atom_name
                break
            atom_terms.append(int(count) * ok[atom_name]["dE_xc_mHa"])
        if missing is not None:
            per_atomization.append({
                "name": mol_spec.name, "dAE_kcalmol": None,
                "error": f"free atom {missing} is missing from the oracle set"})
            continue
        d_ae_mha = ok[mol_spec.name]["dE_xc_mHa"] - sum(atom_terms)
        per_atomization.append({
            "name": mol_spec.name,
            "dAE_kcalmol": d_ae_mha / HA_TO_MHA * HA_TO_KCAL})

    atom_dev = [abs(r["dE_xc_mHa"]) for r in ok.values() if r["is_atom"]]
    ae_dev = [abs(r["dAE_kcalmol"]) for r in per_atomization
              if r.get("dAE_kcalmol") is not None]
    grid_dev = [abs(r["parent_grid_diff_Ha"]) for r in ok.values()]
    record_dev = [abs(r["parent_record_diff_Ha"]) for r in ok.values()]
    n_failed = sum(1 for r in per_system if "error" in r)


    fid_cfg = cfg.fidelity
    tol_atom = float(fid_cfg.tol_atom)
    tol_ae = float(fid_cfg.tol_AE)
    max_atom = max(atom_dev) if atom_dev else None
    max_ae = max(ae_dev) if ae_dev else None
    max_grid = max(grid_dev) if grid_dev else None
    max_record = max(record_dev) if record_dev else None

    reasons = []
    if n_failed:
        reasons.append(
            f"{n_failed} system(s) could not be evaluated: "
            + ", ".join(r["name"] for r in per_system if "error" in r))
    if not atom_dev:
        reasons.append("no free atom was evaluated, so tol_atom is untested")
    elif max_atom > tol_atom:
        reasons.append(
            f"max |dE_xc| over free atoms {max_atom:.4f} mHa exceeds "
            f"tol_atom {tol_atom} mHa")
    if not ae_dev:
        reasons.append(
            "no atomization offset could be formed, so tol_AE is untested")
    elif max_ae > tol_ae:
        reasons.append(
            f"max |dAE| {max_ae:.4f} kcal/mol exceeds tol_AE "
            f"{tol_ae} kcal/mol")
    if max_grid is not None and max_grid > PARENT_GRID_TOL_HA:
        reasons.append(
            f"the point-wise and fresh-grid parent routes disagree by "
            f"{max_grid:.3e} Ha, above the {PARENT_GRID_TOL_HA:.0e} Ha bound")
    if max_record is not None and max_record > PARENT_GRID_TOL_HA:
        reasons.append(
            f"the point-wise parent energy and the reference SCF's own "
            f"accumulated E_xc disagree by {max_record:.3e} Ha, above the "
            f"{PARENT_GRID_TOL_HA:.0e} Ha bound")

    payload = {
        "verdict": VERDICT_FAIL if reasons else VERDICT_PASS,
        "arch": arch_name,
        "parent": parent,
        "xcquinox_version": getattr(xcquinox, "__version__", "unknown"),
        "identity": run_identity(cfg),
        "tolerances": {"tol_AE": tol_ae, "tol_atom": tol_atom,
                       "override_reason": fid_cfg.override_reason},
        # Whether this run's ON-NODE gates act on the verdict. False belongs
        # to the workflow-verification matrix only; the record layers ignore
        # it and require PASS regardless.
        "enforced": bool(getattr(fid_cfg, "enforce", True)),
        "per_system": per_system,
        "per_atomization": per_atomization,
        "summary": {
            "max_atom_mHa": max_atom,
            "max_dAE_kcalmol": max_ae,
            "n_systems": len(per_system),
            "n_atoms": len(atom_dev),
            "n_atomizations": len(ae_dev),
            "n_failed_systems": n_failed,
            "max_parent_grid_diff_Ha": max_grid,
            "max_parent_record_diff_Ha": max_record,
            "failure_reasons": reasons,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": time.time() - t0,
    }
    _write_json_atomic(payload, certificate_path(run_dir, arch_name))
    return payload


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _route_jax_env():
    """Pin JAX to fp64 via env var, before jax is imported.

    JAX defaults to float32 and equinox may capture the default dtype before a
    post-import config update runs, so the env-var switch is the only reliable
    one. ``JAX_PLATFORMS`` is left untouched so the certificate runs on
    whichever device the sbatch script requested (mirrors
    ``cluster._pretrain``).
    """
    os.environ["JAX_ENABLE_X64"] = "1"


def _log(arch, message):
    """One tagged harness log line to stdout -- the SLURM log."""
    sys.stdout.write(f"[harness fidelity arch={arch}] {message}\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    """Certificate entrypoint. Returns 0 on PASS, non-zero otherwise."""
    _route_jax_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="The materialized run directory.")
    parser.add_argument("arch_idx", type=int,
                        help="Index into the sorted distinct-architecture "
                             "list (the same selector the pretrain array "
                             "uses).")
    args = parser.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)

    cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(cfg_path):
        json_path = os.path.join(run_dir, "resolved_config.json")
        if not os.path.isfile(json_path):
            sys.stdout.write(
                f"[harness fidelity] ERROR: no resolved config at "
                f"{cfg_path}\n")
            sys.stdout.flush()
            return 1
        cfg_path = json_path
    try:
        cfg = load_grid_config(cfg_path)
    except (ValueError, ImportError, OSError) as exc:
        sys.stdout.write(
            f"[harness fidelity] ERROR: failed to load resolved config: "
            f"{exc}\n")
        sys.stdout.flush()
        return 1

    archs = _distinct_archs(cfg)
    if not (0 <= args.arch_idx < len(archs)):
        sys.stdout.write(
            f"[harness fidelity] ERROR: arch_idx {args.arch_idx} is out of "
            f"range; the config has {len(archs)} distinct architecture(s) "
            f"(valid indices 0..{len(archs) - 1}): {archs}\n")
        sys.stdout.flush()
        return 1
    arch_name = archs[args.arch_idx]

    _log(arch_name, f"certifying against parent "
                    f"{resolve_parent(arch_name).upper()} at "
                    f"{run_identity(cfg)}")
    payload = fidelity_certificate(cfg, run_dir, arch_name)
    summary = payload["summary"]
    _log(arch_name,
         f"verdict={payload['verdict']} "
         f"max_atom={summary['max_atom_mHa']} mHa "
         f"max_dAE={summary['max_dAE_kcalmol']} kcal/mol over "
         f"{summary['n_systems']} system(s) "
         f"({summary['n_atoms']} atom(s), {summary['n_atomizations']} "
         f"atomization(s))")
    if payload["verdict"] != VERDICT_PASS:
        for reason in summary["failure_reasons"]:
            _log(arch_name, f"FAIL: {reason}")
        if not payload["enforced"]:
            _log(arch_name,
                 "enforcement is OFF for this run (fidelity.enforce=false, "
                 f"override_reason: "
                 f"{payload['tolerances']['override_reason']!r}); the verdict "
                 "is on record and the stage continues. This run cannot enter "
                 "validate_run, merge_v4_arms or the figure suite.")
            return 0
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
```

- [ ] **Step 4: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/cluster/fidelity.py
python -m pytest xcquinox/alec/tests/test_cluster_fidelity.py -v \
  > /tmp/xcq-testlogs/t4-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all tests pass. The two real-physics tests take roughly
15 s each on a cold JAX cache; the meta-GGA test adds a SCAN SCF on two tiny
systems.

- [ ] **Step 5: Confirm the import-weight and anti-fork contracts survived**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_fidelity.py \
  -k "module_body_imports or never_rebuilds" -v \
  > /tmp/xcq-testlogs/t4-contracts.log 2>&1; echo "rc=$?"
```
Read the log. Expected: 2 passed. A failure in the first means a heavy import
leaked to module scope -- move it inside the function that needs it. A failure
in the second means a precompute quantity is being rebuilt here -- take it from
`precompute_fixed_density_data(..., reference_xc=...)` instead.

**Deliverable:** `python -m xcquinox.alec.cluster.fidelity <run_dir> <arch_idx>` writes a certificate and exits 0 only on PASS (or on a recorded, non-enforcing FAIL), with the parent density supplied entirely by the library's one construction path.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_cluster_fidelity.py -v > /tmp/xcq-testlogs/t4-green.log 2>&1`

---

## Task 5: Config plumbing -- `FidelityConfig`, the optional YAML block, its bounds and its round trip

**Files:**
- Modify: `xcquinox/alec/cluster/grid_config.py:22` (imports), `:229-232` (insert the dataclass before the pretrain section), `:376-425` (`GridConfig` fields), `:546-558` (builders), `:709-726` (`load_grid_config`), `:978-1005` (`validate_grid_semantics`, the pretrain-bounds block)
- Modify: `xcquinox/alec/cluster/__main__.py:135-172` (`_config_to_raw_dict`)
- Modify: `xcquinox/alec/cluster/examples/grid_step7.yaml` (append a `fidelity` block)
- Modify: `xcquinox/alec/tests/test_cluster_examples.py:15-26` (import), `:144-163` (`test_example_structural_completeness`)
- Modify: `xcquinox/alec/tests/test_cluster_grid_config.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (read by Task 3's `fidelity_certificate` and by every later task):
  - `grid_config.FidelityConfig(tol_AE: float = 1.0, tol_atom: float = 1.0, override_reason: str | None = None, enforce: bool = True)` -- frozen dataclass
  - `GridConfig.fidelity: FidelityConfig` -- present on every config, defaulting when the YAML omits the block
  - YAML key names: `fidelity.tol_AE`, `fidelity.tol_atom`, `fidelity.override_reason`, `fidelity.enforce`

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_cluster_grid_config.py`:

```python
# ---------------------------------------------------------------------------
# FidelityConfig: the per-architecture physics-certificate tolerances
# ---------------------------------------------------------------------------

def test_fidelity_defaults_to_the_binding_tolerances(tmp_path):
    """A config with no fidelity block carries tol_AE = 1.0 kcal/mol and
    tol_atom = 1.0 mHa, so every YAML written before the certificate existed
    loads at the binding tolerances rather than at no tolerance."""
    from xcquinox.alec.cluster.grid_config import FidelityConfig
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", _base_config_dict()))
    assert isinstance(cfg.fidelity, FidelityConfig)
    assert cfg.fidelity.tol_AE == 1.0
    assert cfg.fidelity.tol_atom == 1.0
    assert cfg.fidelity.override_reason is None
    assert cfg.fidelity.enforce is True


def test_fidelity_block_parses(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 0.5, "tol_atom": 0.25,
                       "override_reason": None}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    assert cfg.fidelity.tol_AE == 0.5
    assert cfg.fidelity.tol_atom == 0.25


def test_fidelity_block_must_be_a_mapping(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="fidelity"):
        load_grid_config(_write(tmp_path, "grid.yaml", raw))


def test_fidelity_resolved_round_trip(tmp_path):
    """The resolved config is re-read by the pretrain, preflight and eval
    stages; a dropped fidelity block would silently revert a documented
    override to the binding tolerances mid-run."""
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 2.5, "tol_atom": 2.5,
                       "override_reason": "rung-3.5 control arm"}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    cfg2 = load_grid_config(
        _write(tmp_path, "resolved.yaml", _config_to_raw_dict(cfg)))
    assert cfg2.fidelity == cfg.fidelity


def test_validate_rejects_a_loose_tolerance_without_an_override_reason(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 3.0, "tol_atom": 1.0}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    with pytest.raises(ValueError, match="override_reason"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_rejects_a_loose_atom_tolerance_without_an_override_reason(
        tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 2.5,
                       "override_reason": "   "}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    with pytest.raises(ValueError, match="override_reason"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_accepts_a_loose_tolerance_with_an_override_reason(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 3.0, "tol_atom": 3.0,
                       "override_reason": "descriptor-free control arm, "
                                          "documented in HISTORY 2026-08-21"}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_accepts_the_ceiling_without_an_override_reason(tmp_path):
    """2.0 / 2.0 is the ceiling, not past it."""
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 2.0, "tol_atom": 2.0}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_rejects_disabled_enforcement_without_a_reason(tmp_path):
    """Turning the on-node gates off is a documented decision or it does not
    happen: the reason is copied into every certificate the run writes."""
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, "enforce": False}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    with pytest.raises(ValueError, match="override_reason"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_accepts_disabled_enforcement_with_a_reason(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, "enforce": False,
                       "override_reason": "workflow-verification matrix, "
                                          "50-step pretrain"}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    validate_grid_semantics(cfg, _StubDomain(pool_size=100))
    assert cfg.fidelity.enforce is False


def test_fidelity_enforce_round_trips(tmp_path):
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, "enforce": False,
                       "override_reason": "workflow matrix"}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    cfg2 = load_grid_config(
        _write(tmp_path, "resolved2.yaml", _config_to_raw_dict(cfg)))
    assert cfg2.fidelity.enforce is False
    assert cfg2.fidelity == cfg.fidelity


def test_validate_rejects_a_nonpositive_tolerance(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 0.0, "tol_atom": 1.0}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    with pytest.raises(ValueError, match="tol_AE must be > 0"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))
```

Modify `xcquinox/alec/tests/test_cluster_examples.py`. In the import block at
`:15-26`, add `FidelityConfig` to the names imported from
`xcquinox.alec.cluster.grid_config`. Then in
`test_example_structural_completeness` (`:144-163`), after the
`_assert_fields_covered(ClusterResources, raw.get("cluster"), "cluster")` line,
add:

```python
    _assert_fields_covered(FidelityConfig, raw.get("fidelity"), "fidelity")
```

and append a new test to the same file:

```python
def test_example_ships_the_binding_certificate_tolerances():
    """The shipped template must carry the program's binding tolerances --
    tol_AE = 1.0 kcal/mol, tol_atom = 1.0 mHa -- explicitly, so a copy-me user
    sees them rather than inheriting an invisible default."""
    pytest.importorskip("yaml")
    raw = _raw_yaml(_example_path())
    assert raw.get("fidelity") is not None, (
        "grid_step7.yaml must ship a 'fidelity' block")
    assert raw["fidelity"]["tol_AE"] == 1.0
    assert raw["fidelity"]["tol_atom"] == 1.0
    assert raw["fidelity"]["override_reason"] is None
    assert raw["fidelity"]["enforce"] is True
    cfg = load_grid_config(_example_path())
    assert cfg.fidelity.tol_AE == 1.0
    assert cfg.fidelity.tol_atom == 1.0
    assert cfg.fidelity.enforce is True
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_grid_config.py \
  xcquinox/alec/tests/test_cluster_examples.py -v \
  > /tmp/xcq-testlogs/t5-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: `ImportError: cannot import name 'FidelityConfig'` for the examples file and `AttributeError: 'GridConfig' object has no attribute 'fidelity'` for the new grid-config tests.

- [ ] **Step 3: Add the dataclass and the builder to `grid_config.py`**

Change the dataclasses import at `:22` to:

```python
from dataclasses import dataclass, field, fields
```

Insert, immediately before the `# Pretrain stage config` banner comment at `:230-233`:

```python
# ---------------------------------------------------------------------------
# Pretraining-fidelity certificate config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FidelityConfig:
    """Tolerances for the per-architecture physics certificate.

    Every architecture's pretrained networks must reproduce their parent
    functional (PBE for a GGA-rung architecture, SCAN for a meta-GGA one) in
    energy units before the run may train: the certificate
    (``cluster/fidelity.py``) requires max |dE_xc| over free atoms <=
    ``tol_atom`` mHa AND max |dAE| over atomization energies <= ``tol_AE``
    kcal/mol on frozen parent densities at the run's identity.

    The defaults are the program's binding decision (1.0 kcal/mol and 1.0
    mHa). ``validate_grid_semantics`` refuses either tolerance above 2.0
    unless ``override_reason`` is non-empty, so a run can only be loosened
    deliberately and with the reason on the record: the string is copied into
    every certificate the run writes.
    """
    tol_AE: float = 1.0          # kcal/mol, atomization-energy offset
    tol_atom: float = 1.0        # mHa, free-atom E_xc offset
    override_reason: str | None = None
    # When False the certificate is still computed and written with its TRUE
    # verdict, but the ON-NODE gates (the pretrain worker's exit code, the
    # train task, the preflight sweep) log the verdict and continue instead of
    # refusing. Permitted only with a non-empty ``override_reason``. It exists
    # for the per-architecture workflow-verification matrix, whose short
    # pretraining runs cannot meet the tolerance yet must exercise the train
    # and eval wiring with the physics on record. The RECORD layers
    # (``validate_run``, ``merge_v4_arms``, the figure loaders) ignore this
    # field and require PASS regardless, so a non-enforcing run can never
    # become a quantitative result.
    enforce: bool = True


```

Add to `GridConfig` (after `eval_coldstart`, `:424`):

```python
    # Pretraining-fidelity certificate tolerances. Optional in the YAML: a
    # config written before the certificate existed loads at the binding
    # 1.0 kcal/mol / 1.0 mHa defaults rather than at no tolerance.
    fidelity: FidelityConfig = field(default_factory=FidelityConfig)
```

Add the builder next to `_build_pretrain` (after `:558`):

```python
def _build_fidelity(d) -> FidelityConfig:
    """Build FidelityConfig from a raw dict; ``None`` -> the defaults.

    The ``fidelity`` section is OPTIONAL so every YAML authored before the
    certificate existed still loads, at the binding tolerances.
    """
    if d is None:
        return FidelityConfig()
    if not isinstance(d, dict):
        raise ValueError(
            f"grid config section 'fidelity' must be a mapping, got "
            f"{type(d).__name__}")
    reason = d.get("override_reason")
    return FidelityConfig(
        tol_AE=float(d.get("tol_AE", 1.0)),
        tol_atom=float(d.get("tol_atom", 1.0)),
        override_reason=None if reason is None else str(reason),
        enforce=bool(d.get("enforce", True)),
    )
```

Add to the `GridConfig(...)` construction in `load_grid_config` (after the
`eval_coldstart=...` line at `:725`):

```python
        fidelity=_build_fidelity(raw.get("fidelity")),
```

- [ ] **Step 4: Add the bounds to `validate_grid_semantics`**

Insert immediately after the `pretrain.loss_weighting` check that closes the
pretrain-bounds block (`:1004`), before the `# --- resource bounds` banner:

```python
    # --- certificate tolerance bounds --------------------------------------
    # The program's binding decision is tol_AE = 1.0 kcal/mol and tol_atom =
    # 1.0 mHa for every architecture. A looser run is possible but never
    # silent: above 2.0 / 2.0 the config must carry a non-empty
    # override_reason, which the certificate copies into its own record.
    fid = cfg.fidelity
    if fid.tol_AE <= 0:
        raise ValueError(f"fidelity.tol_AE must be > 0, got {fid.tol_AE}")
    if fid.tol_atom <= 0:
        raise ValueError(f"fidelity.tol_atom must be > 0, got {fid.tol_atom}")
    _override = (fid.override_reason or "").strip()
    if (fid.tol_AE > 2.0 or fid.tol_atom > 2.0) and not _override:
        raise ValueError(
            f"fidelity.tol_AE={fid.tol_AE} kcal/mol / "
            f"fidelity.tol_atom={fid.tol_atom} mHa exceed the 2.0 / 2.0 "
            "ceiling; a certificate tolerance above that ceiling requires a "
            "non-empty fidelity.override_reason, which is recorded in every "
            "certificate the run writes")
    if not fid.enforce and not _override:
        raise ValueError(
            "fidelity.enforce=false disables the on-node certificate gates, "
            "so it requires a non-empty fidelity.override_reason; the reason "
            "is recorded in every certificate the run writes. Such a run is "
            "still refused by validate_run, merge_v4_arms and the figure "
            "suite, so it can only be used for workflow verification")
```

- [ ] **Step 5: Round-trip the block in `_config_to_raw_dict`**

In `xcquinox/alec/cluster/__main__.py`, add to the `raw` dict built by
`_config_to_raw_dict` (after the `"pretrain": ...` entry at `:147`):

```python
        # fidelity MUST round-trip for the same reason ae_as_reactions and
        # inline_eval do: the pretrain worker re-reads resolved_config.yaml to
        # get its tolerances, so a dropped block would silently certify at the
        # defaults instead of at the run's documented override.
        "fidelity": dataclasses.asdict(cfg.fidelity),
```

- [ ] **Step 6: Ship the block in the example YAML**

Append to `xcquinox/alec/cluster/examples/grid_step7.yaml`, after the
`use_polarized_correlation: false` block at the end of the file:

```yaml

# --- pretraining-fidelity certificate --------------------------------------
# Before the train array may start, every distinct architecture must carry a
# PASS certificate at <run_dir>/pretrain/<arch>/fidelity_certificate.json: its
# pretrained networks reproduce their parent functional (PBE for a GGA-rung
# architecture, SCAN for a meta-GGA one) in energy units on frozen parent
# densities at this run's identity. PASS requires
#   max |dE_xc| over free atoms         <= tol_atom  (mHa)
#   max |dAE|   over atomization energies <= tol_AE  (kcal/mol)
# Neither may exceed 2.0 without a non-empty override_reason, which is copied
# into every certificate the run writes.
# `enforce: false` (which REQUIRES a non-empty override_reason) still computes
# and writes the certificate with its true verdict, but lets the pretrain
# stage, the train tasks and the preflight continue past a FAIL. It exists for
# workflow verification only: validate_run, merge_v4_arms and the figure suite
# refuse such a run regardless, so it can never become a quantitative result.
fidelity:
  tol_AE: 1.0               # kcal/mol
  tol_atom: 1.0             # mHa
  override_reason: null
  enforce: true
```

- [ ] **Step 7: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/cluster/grid_config.py \
  xcquinox/alec/cluster/__main__.py
python -m pytest xcquinox/alec/tests/test_cluster_grid_config.py \
  xcquinox/alec/tests/test_cluster_examples.py \
  xcquinox/alec/tests/test_cluster_cli.py -v \
  > /tmp/xcq-testlogs/t5-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass, including the pre-existing
`test_resolved_config_round_trip_preserves_every_field`, which iterates
`dataclasses.fields(GridConfig)` and would have failed on a `fidelity` field
missing from `_config_to_raw_dict`.

**Deliverable:** every config carries `cfg.fidelity` with the binding tolerances, the block round-trips through `resolved_config.yaml`, and a tolerance above 2.0 without a reason is refused on the login node.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_cluster_grid_config.py xcquinox/alec/tests/test_cluster_examples.py xcquinox/alec/tests/test_cluster_cli.py -v > /tmp/xcq-testlogs/t5-green.log 2>&1`

---

## Task 6: On-node gate -- the pretrain worker certifies before it reports success

**Files:**
- Modify: `xcquinox/alec/cluster/_pretrain.py:46-49` (imports), `:178-187` (seam block, add the second seam), `:317-327` (the tail of `main`)
- Modify: `xcquinox/alec/tests/test_cluster_pretrain.py` (fixtures + append)

**Interfaces:**
- Consumes: `fidelity.fidelity_certificate(cfg, run_dir, arch_name)`, `fidelity.resolve_parent`, `fidelity.VERDICT_PASS` (Tasks 2-3); `cfg.fidelity` (Task 5).
- Produces: `_pretrain._fidelity_certificate` -- the module-level seam tests monkeypatch, mirroring `_pretrain._run_pretrain`.

- [ ] **Step 1: Write the failing tests**

In `xcquinox/alec/tests/test_cluster_pretrain.py`, every existing happy-path
test asserts `pt.main([...]) == 0`. Add an autouse fixture so those keep
passing against a stubbed certificate, then the new tests. Insert the fixture
right after the `run_dir` fixture (`:109-122`):

```python
@pytest.fixture(autouse=True)
def stub_certificate(request, monkeypatch):
    """Stub the fidelity certificate at its seam for every test in this file.

    The certificate loads the checkpoint and runs PySCF SCFs at the run's
    identity; the pretrain-worker tests are about worker orchestration, so
    they get a PASS payload for free. The tests that exercise the gate
    override this with their own seam. A test whose name ends in
    ``_unstubbed`` opts out entirely, which is how the seam-identity test can
    observe the real module-level binding.
    """
    if request.node.name.endswith("_unstubbed"):
        return
    monkeypatch.setattr(pt, "_fidelity_certificate", lambda cfg, rd, arch: {
        "verdict": "PASS", "enforced": True,
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": None},
        "summary": {"max_atom_mHa": 0.12, "max_dAE_kcalmol": 0.34,
                    "n_systems": 40, "n_atoms": 16, "n_atomizations": 24,
                    "failure_reasons": []}})
```

Append the gate tests:

```python
# ---------------------------------------------------------------------------
# The on-node fidelity gate
# ---------------------------------------------------------------------------

def _stub_pretrain_writes_checkpoint(monkeypatch):
    def fake_run_pretrain(spec, progress_callback=None):
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {}
    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)


def test_pretrain_runs_the_certificate_for_its_own_arch(run_dir, monkeypatch):
    _stub_pretrain_writes_checkpoint(monkeypatch)
    seen = {}

    def fake_cert(cfg, rd, arch):
        seen["args"] = (rd, arch)
        seen["tol"] = (cfg.fidelity.tol_AE, cfg.fidelity.tol_atom)
        return {"verdict": "PASS", "enforced": True,
                "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                               "override_reason": None},
                "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2,
                            "n_systems": 2, "n_atoms": 1, "n_atomizations": 1,
                            "failure_reasons": []}}

    monkeypatch.setattr(pt, "_fidelity_certificate", fake_cert)
    assert pt.main([run_dir, "1"]) == 0
    assert seen["args"] == (os.path.abspath(run_dir), "medium")
    assert seen["tol"] == (1.0, 1.0)


def test_pretrain_exits_nonzero_on_a_failed_certificate(run_dir, monkeypatch,
                                                        capsys):
    _stub_pretrain_writes_checkpoint(monkeypatch)
    monkeypatch.setattr(pt, "_fidelity_certificate", lambda cfg, rd, arch: {
        "verdict": "FAIL", "enforced": True,
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": None},
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7,
                    "n_systems": 40, "n_atoms": 16, "n_atomizations": 24,
                    "failure_reasons": ["max |dE_xc| over free atoms 13.7000 "
                                        "mHa exceeds tol_atom 1.0 mHa"]}})
    assert pt.main([run_dir, "1"]) == 1
    out = capsys.readouterr().out
    assert "fidelity certificate FAILED" in out
    assert "13.7" in out and "25.7" in out
    assert "tol_atom" in out


def test_pretrain_continues_past_a_failure_when_enforcement_is_off(
        run_dir, monkeypatch, capsys):
    """Workflow-verification runs must reach the train stage with a FAIL on
    record; the worker says so in the log and exits 0."""
    _stub_pretrain_writes_checkpoint(monkeypatch)
    monkeypatch.setattr(pt, "_fidelity_certificate", lambda cfg, rd, arch: {
        "verdict": "FAIL", "enforced": False,
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": "workflow matrix: 50-step "
                                          "pretrain"},
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7,
                    "n_systems": 40, "n_atoms": 16, "n_atomizations": 24,
                    "failure_reasons": ["max_atom_mHa"]}})
    assert pt.main([run_dir, "1"]) == 0
    out = capsys.readouterr().out
    assert "fidelity certificate FAILED" in out
    assert "enforcement is OFF" in out
    assert "workflow matrix" in out
    assert "pretrain SUCCEEDED" in out


def test_pretrain_exits_nonzero_when_the_certificate_raises(run_dir,
                                                            monkeypatch,
                                                            capsys):
    _stub_pretrain_writes_checkpoint(monkeypatch)

    def boom(cfg, rd, arch):
        raise RuntimeError("libxc unavailable")

    monkeypatch.setattr(pt, "_fidelity_certificate", boom)
    assert pt.main([run_dir, "1"]) == 1
    out = capsys.readouterr().out
    assert "fidelity certificate RAISED" in out
    assert "libxc unavailable" in out


def test_pretrain_logs_the_passing_summary(run_dir, monkeypatch, capsys):
    _stub_pretrain_writes_checkpoint(monkeypatch)
    assert pt.main([run_dir, "1"]) == 0
    out = capsys.readouterr().out
    assert "fidelity certificate PASSED" in out
    assert "pretrain SUCCEEDED" in out
    # The certificate line precedes the SUCCEEDED line: the job only reports
    # success after the physics has been checked.
    assert out.index("fidelity certificate PASSED") < out.index(
        "pretrain SUCCEEDED")


def test_pretrain_does_not_certify_when_the_checkpoint_is_missing(
        run_dir, monkeypatch):
    """A worker that wrote no checkpoint fails at the existing guard; the
    certificate must not be attempted against an absent xnet.eqx."""
    monkeypatch.setattr(pt, "_run_pretrain", lambda spec, progress_callback=None: {})
    called = []
    monkeypatch.setattr(pt, "_fidelity_certificate",
                        lambda *a, **k: called.append(1))
    assert pt.main([run_dir, "1"]) == 1
    assert called == []


def test_fidelity_certificate_seam_is_the_library_function_unstubbed():
    """One implementation of the certificate, bound as a seam -- not a wrapper
    that could drift from what the library actually runs."""
    from xcquinox.alec.cluster import fidelity
    assert pt._fidelity_certificate is fidelity.fidelity_certificate
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_pretrain.py -v \
  > /tmp/xcq-testlogs/t6-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: every test errors in the autouse fixture with
`AttributeError: <module 'xcquinox.alec.cluster._pretrain'> has no attribute '_fidelity_certificate'`.

- [ ] **Step 3: Add the seam and the gate**

In `xcquinox/alec/cluster/_pretrain.py`, extend the import block at `:46-49`:

```python
from xcquinox.alec.config import get_architecture
from xcquinox.alec.cluster import fidelity
from xcquinox.alec.cluster.grid_config import (
    load_grid_config, _canon_axis, pretrain_checkpoint_dir,
)
```

Add the second seam immediately after `_run_pretrain` (after `:187`). It is a
module-level BINDING, not a wrapper, so `_pretrain._fidelity_certificate is
fidelity.fidelity_certificate` holds and there is exactly one implementation
of the certificate:

```python
# ---------------------------------------------------------------------------
# fidelity_certificate seam
# ---------------------------------------------------------------------------
# Bound at module level -- the same test-seam pattern as _run_pretrain above --
# so a unit test can monkeypatch it and avoid the real SCF sweep. The call
# writes <run_dir>/pretrain/<arch>/fidelity_certificate.json and returns its
# payload.
_fidelity_certificate = fidelity.fidelity_certificate
```

In `main`, insert between the silent-no-checkpoint guard (which ends at `:320`)
and the `pretrain SUCCEEDED` log at `:322`:

```python
    # --- pretraining-fidelity gate -----------------------------------------
    # The checkpoint is on disk; whether it is USABLE is a physics question.
    # The certificate answers it here, on this node, with the checkpoint hot
    # and the production identity available, and gates this job's exit code --
    # so the train array's afterok dependency blocks on an uncertified
    # architecture with no extra job kind and no extra scheduling round trip.
    t_cert = time.time()
    _log(arch_name,
         "running the per-architecture fidelity certificate against parent "
         f"{fidelity.resolve_parent(arch_name).upper()} ...")
    try:
        certificate = _fidelity_certificate(cfg, run_dir, arch_name)
    except Exception as exc:  # any failure must produce a non-zero exit
        _log(arch_name,
             f"fidelity certificate RAISED after "
             f"{_fmt_secs(time.time() - t_cert)}: {type(exc).__name__}: {exc}")
        return 1
    summary = certificate.get("summary") or {}
    line = (f"max_atom={summary.get('max_atom_mHa')} mHa, "
            f"max_dAE={summary.get('max_dAE_kcalmol')} kcal/mol over "
            f"{summary.get('n_systems')} system(s) "
            f"({summary.get('n_atoms')} atom(s), "
            f"{summary.get('n_atomizations')} atomization(s)) in "
            f"{_fmt_secs(time.time() - t_cert)}")
    if certificate.get("verdict") != fidelity.VERDICT_PASS:
        _log(arch_name, f"fidelity certificate FAILED: {line}")
        for reason in summary.get("failure_reasons") or ():
            _log(arch_name, f"  reason: {reason}")
        # The enforcement flag is read back out of the certificate that was
        # just written, so the worker, the train task and the preflight all
        # decide from the same recorded statement rather than from three
        # readings of the config.
        if not certificate.get("enforced", True):
            _reason = (certificate.get("tolerances") or {}).get(
                "override_reason")
            _log(arch_name,
                 "fidelity enforcement is OFF for this run "
                 f"(fidelity.enforce=false, override_reason: {_reason!r}); "
                 "the verdict is on record and the stage continues. This run "
                 "cannot enter validate_run, merge_v4_arms or the figure "
                 "suite.")
        else:
            return 1
    else:
        _log(arch_name, f"fidelity certificate PASSED: {line}")

```

- [ ] **Step 4: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/cluster/_pretrain.py
python -m pytest xcquinox/alec/tests/test_cluster_pretrain.py -v \
  > /tmp/xcq-testlogs/t6-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass.

**Deliverable:** the pretrain array task exits non-zero unless its architecture certifies, so the train array's `afterok` dependency blocks.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_cluster_pretrain.py -v > /tmp/xcq-testlogs/t6-green.log 2>&1`

---

## Task 7: Train-task refusal -- a spec never trains against an uncertified checkpoint

**Files:**
- Modify: `xcquinox/alec/cluster/_train_task.py:124-136` (add `_read_cell_arch` beside `_read_width`), `:456-464` (insert the gate after the `precompute_failed_species` early exit)
- Modify: `xcquinox/alec/tests/test_cluster_train_task.py:25-32` (`_write_manifest`), `:57-63` (the `run_dir` fixture), append new tests
- Verify (no edit): `xcquinox/alec/cluster/__main__.py:386` -- neither classification is added to `_RETRYABLE`, so `_classify_failure` treats both as deterministic.

**Interfaces:**
- Consumes: `fidelity.gate_certificate(run_dir, arch)`, `fidelity.certificate_status(run_dir, arch)`, `fidelity.CERTIFICATE_FILENAME` (Task 2).
- Produces: `_train_task._read_cell_arch(run_dir, idx) -> str | None`; the failure classifications `"fidelity_certificate_missing"` and `"fidelity_certificate_failed"`, both with `rc = 3` and an `"arch"` key in `failure.json`.

- [ ] **Step 1: Write the failing tests**

In `xcquinox/alec/tests/test_cluster_train_task.py`, replace `_write_manifest`
(`:25-32`) with a version that records the grid cell, and extend the `run_dir`
fixture (`:57-63`) to write a PASS certificate:

```python
def _write_manifest(run_dir, width=4, n_specs=4, arch="deep_3x16"):
    payload = {
        "xcquinox_version": "test",
        "python_version": "3.x",
        "width": width,
        "n_specs": n_specs,
        "specs": [{"index": i, "spec_file": f"spec_{i:0{width}d}.spec",
                   "sha256": "x" * 64,
                   "cell": {"arch": arch, "loss": "l2", "metric": "l2",
                            "subset_size": 1, "solver": "oneshot"}}
                  for i in range(n_specs)],
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(payload, f)


def _write_pass_certificate(run_dir, arch="deep_3x16", verdict="PASS"):
    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    payload = {"verdict": verdict, "arch": arch,
               "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2}}
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump(payload, f)
    return d


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    _write_manifest(str(d))
    _write_spec(str(d), 0)
    # Every orchestration test in this file describes a run whose architecture
    # certified; the gate's own tests remove or downgrade the certificate.
    _write_pass_certificate(str(d))
    return str(d)
```

Append the gate tests:

```python
# ---------------------------------------------------------------------------
# The pretraining-fidelity gate
# ---------------------------------------------------------------------------

def test_missing_certificate_refuses_before_the_worker_runs(run_dir,
                                                            monkeypatch):
    os.remove(os.path.join(run_dir, "pretrain", "deep_3x16",
                           "fidelity_certificate.json"))
    calls = []
    monkeypatch.setattr(tt, "_run_worker",
                        lambda s, d: calls.append(1) or (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3
    assert calls == []          # the node is never spent on an uncertified spec
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "fidelity_certificate_missing"
    assert failure["rc"] == 3
    assert failure["arch"] == "deep_3x16"
    assert "fidelity_certificate.json" in failure["log_excerpt"]


def test_failed_certificate_refuses_with_its_own_classification(run_dir,
                                                                monkeypatch):
    _write_pass_certificate(run_dir, verdict="FAIL")
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "fidelity_certificate_failed"
    assert failure["arch"] == "deep_3x16"


def test_unreadable_certificate_is_treated_as_missing(run_dir, monkeypatch):
    path = os.path.join(run_dir, "pretrain", "deep_3x16",
                        "fidelity_certificate.json")
    with open(path, "w") as f:
        f.write("{truncated")
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3
    assert _read_failure(run_dir, 0)["classification"] == \
        "fidelity_certificate_missing"


def test_manifest_without_a_cell_arch_is_refused_not_waved_through(tmp_path,
                                                                   monkeypatch):
    """A manifest with no arch for this index makes the certificate
    unresolvable; an unresolvable certificate is a refusal, never a pass."""
    d = tmp_path / "run"
    d.mkdir()
    with open(d / "manifest.json", "w") as f:
        json.dump({"width": 4, "n_specs": 1}, f)
    _write_spec(str(d), 0)
    monkeypatch.setattr(tt, "_run_worker", lambda s, dev: (0, "ok"))
    assert tt.main([str(d), "0"]) == 3
    failure = _read_failure(str(d), 0)
    assert failure["classification"] == "fidelity_certificate_missing"
    assert failure["arch"] is None


def test_unenforced_failure_lets_the_worker_run(run_dir, monkeypatch,
                                                capsys):
    """A workflow-verification run reaches the train stage with its FAIL on
    record; the log says so."""
    d = os.path.join(run_dir, "pretrain", "deep_3x16")
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump({"verdict": "FAIL", "arch": "deep_3x16", "enforced": False,
                   "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                                  "override_reason": "workflow matrix"},
                   "summary": {"max_atom_mHa": 13.7,
                               "max_dAE_kcalmol": 25.7}}, f)

    def fake_worker(spec_path, device):
        _write_model(run_dir, 0)
        return 0, "ok"

    monkeypatch.setattr(tt, "_run_worker", fake_worker)
    assert tt.main([run_dir, "0"]) == 0
    out = capsys.readouterr().out
    assert "enforcement is OFF" in out


def test_unenforced_but_MISSING_certificate_is_still_refused(run_dir,
                                                             monkeypatch):
    """Enforcement can only be waived by a certificate that exists to record
    the waiver; an absent one waives nothing."""
    os.remove(os.path.join(run_dir, "pretrain", "deep_3x16",
                           "fidelity_certificate.json"))
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3


def test_passing_certificate_lets_the_worker_run(run_dir, monkeypatch):
    def fake_worker(spec_path, device):
        _write_model(run_dir, 0)
        return 0, "ok"
    monkeypatch.setattr(tt, "_run_worker", fake_worker)
    assert tt.main([run_dir, "0"]) == 0


def test_precompute_failed_species_marker_still_wins(run_dir, monkeypatch):
    """The preflight's precise diagnosis is preserved: it exits 0 BEFORE the
    fidelity gate, so a spec already marked unbuildable is not relabelled."""
    ck = os.path.join(run_dir, "checkpoints", "spec_0000")
    os.makedirs(ck, exist_ok=True)
    with open(os.path.join(ck, "failure.json"), "w") as f:
        json.dump({"classification": "precompute_failed_species", "rc": 0}, f)
    os.remove(os.path.join(run_dir, "pretrain", "deep_3x16",
                           "fidelity_certificate.json"))
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 0
    assert _read_failure(run_dir, 0)["classification"] == \
        "precompute_failed_species"


def test_read_cell_arch_resolves_the_index(run_dir):
    assert tt._read_cell_arch(run_dir, 0) == "deep_3x16"
    assert tt._read_cell_arch(run_dir, 99) is None


def test_certificate_classifications_are_deterministic_not_retryable():
    """A blind resubmit cannot make an absent or failed certificate pass, so
    neither classification may enter the retry set."""
    from xcquinox.alec.cluster.__main__ import _RETRYABLE
    assert "fidelity_certificate_missing" not in _RETRYABLE
    assert "fidelity_certificate_failed" not in _RETRYABLE
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_train_task.py -v \
  > /tmp/xcq-testlogs/t7-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: the new gate tests fail (`assert 0 == 3`, no
`failure.json`) and `test_read_cell_arch_resolves_the_index` fails with
`AttributeError: module ... has no attribute '_read_cell_arch'`.

- [ ] **Step 3: Add the arch reader**

In `xcquinox/alec/cluster/_train_task.py`, add immediately after `_read_width`
(after `:136`):

```python
def _read_cell_arch(run_dir, idx):
    """The architecture name grid cell ``idx`` carries, from ``manifest.json``.

    ``None`` when the manifest records no cell for the index (a truncated or
    pre-``specs``-entry manifest). The caller treats an unresolvable
    architecture as an unverifiable certificate: a spec whose pretraining
    provenance cannot be established does not train.
    """
    path = os.path.join(run_dir, _MANIFEST_FILENAME)
    try:
        with open(path) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    for entry in manifest.get("specs") or ():
        try:
            if int(entry.get("index", -1)) == int(idx):
                return (entry.get("cell") or {}).get("arch")
        except (TypeError, ValueError):
            continue
    return None
```

- [ ] **Step 4: Add the gate to `main`**

Insert immediately after the `precompute_failed_species` early-exit block
(after its `return 0` at `:464`) and before the `if not os.path.exists(spec_path)`
check:

```python
    # --- pretraining-fidelity gate -----------------------------------------
    # A spec may not train against networks that were never shown to reproduce
    # their parent functional: the pre-certificate checkpoints were off by 2.3
    # to 56 kcal/mol in atomization energies (SPEC_pretrain_fidelity_program.md
    # Section 2), larger than every effect the training is meant to measure.
    # Neither classification is in ``__main__._RETRYABLE``, so ``resubmit``
    # treats both as deterministic -- a blind retry cannot make an absent or
    # failed certificate pass. ``gate_certificate`` (not ``certificate_status``)
    # is the predicate here: a run configured with ``fidelity.enforce: false``
    # records the FAIL and is allowed through, because the workflow-verification
    # matrix must reach the train stage with a short pretrain that cannot meet
    # the tolerance. Such a run is still refused by ``validate_run``,
    # ``merge_v4_arms`` and the figure suite.
    from xcquinox.alec.cluster.fidelity import (
        CERTIFICATE_FILENAME, certificate_status, gate_certificate)
    arch = _read_cell_arch(run_dir, idx)
    if arch is None:
        excerpt = (
            f"manifest.json in {run_dir} records no cell architecture for "
            f"index {idx}, so this spec's {CERTIFICATE_FILENAME} cannot be "
            "located")
        _log(idx, f"REFUSING to train: {excerpt}")
        _write_failure_json(checkpoint_dir, {
            "classification": "fidelity_certificate_missing",
            "rc": 3,
            "arch": None,
            "log_excerpt": excerpt,
        })
        return 3
    allowed, message = gate_certificate(run_dir, arch)
    if not allowed:
        status, _reason = certificate_status(run_dir, arch)
        classification = ("fidelity_certificate_failed"
                          if status == "FAIL"
                          else "fidelity_certificate_missing")
        _log(idx, f"REFUSING to train arch {arch!r}: {message}")
        _write_failure_json(checkpoint_dir, {
            "classification": classification,
            "rc": 3,
            "arch": arch,
            "log_excerpt": message,
        })
        return 3
    _log(idx, f"fidelity gate for arch {arch!r}: {message}")
```

- [ ] **Step 5: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/cluster/_train_task.py
python -m pytest xcquinox/alec/tests/test_cluster_train_task.py -v \
  > /tmp/xcq-testlogs/t7-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass.

**Deliverable:** a train array task exits 3 with a deterministic `failure.json` rather than training against an uncertified pretrain checkpoint.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_cluster_train_task.py -v > /tmp/xcq-testlogs/t7-green.log 2>&1`

---

## Task 8: Preflight sweep -- the array is not released until every architecture certifies

**Files:**
- Modify: `xcquinox/alec/cluster/_preflight.py:632-646` (insert step 9 after the compile-smoke block, before the SUCCEEDED log)
- Modify: `xcquinox/alec/tests/test_cluster_preflight.py:70-127` (`_write_resolved_config` also writes a PASS certificate), append new tests

**Interfaces:**
- Consumes: `fidelity.gate_certificate(run_dir, arch)` (Task 2).
- Produces: nothing new; the preflight's exit code carries the verdict.

- [ ] **Step 1: Write the failing tests**

In `xcquinox/alec/tests/test_cluster_preflight.py`, add a helper and call it
from `_write_resolved_config` so the many `main([...]) == 0` assertions in the
file keep describing a certified run. Insert right after the
`_write_resolved_config` definition (`:127`):

```python
def _write_pass_certificate(run_dir, arch="shallow", verdict="PASS"):
    """Write a PASS fidelity certificate for ``arch`` under ``run_dir``.

    The preflight runs afterok on the pretrain array, so by preflight time
    every architecture already carries one; these fixtures describe that
    state, and the gate's own tests remove or downgrade it.
    """
    d = os.path.join(str(run_dir), "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    payload = {"verdict": verdict, "arch": arch,
               "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2}}
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump(payload, f)
    return d
```

and add, as the last statement of `_write_resolved_config` before its
`return path`:

```python
    for arch in cfg["sweep"]["arch"]:
        _write_pass_certificate(run_dir, arch)
```

(Python resolves `_write_pass_certificate` at call time, so its definition may
follow `_write_resolved_config` in the file.)

Append the gate tests:

```python
# ---------------------------------------------------------------------------
# The per-architecture fidelity gate
# ---------------------------------------------------------------------------

def test_preflight_blocks_the_array_on_a_missing_certificate(tmp_path,
                                                             patched, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)
    os.remove(os.path.join(str(run_dir), "pretrain", "shallow",
                           "fidelity_certificate.json"))
    assert main([str(run_dir)]) == 1
    out = capsys.readouterr().out
    assert "fidelity gate FAILED" in out
    assert "shallow" in out


def test_preflight_blocks_the_array_on_a_failed_certificate(tmp_path,
                                                            patched, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)
    _write_pass_certificate(run_dir, "shallow", verdict="FAIL")
    assert main([str(run_dir)]) == 1
    out = capsys.readouterr().out
    assert "fidelity gate FAILED" in out


def test_preflight_reports_the_gate_when_every_arch_certifies(tmp_path,
                                                              patched,
                                                              capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)
    assert main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "fidelity gate PASSED" in out
    assert "1/1 architecture certificate(s) released the gate" in out
    assert "preflight SUCCEEDED" in out


def test_preflight_releases_an_unenforced_failure(tmp_path, patched, capsys):
    """A workflow-verification run must reach its train array with the FAIL on
    record; the preflight log says the gate was not enforced."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(run_dir)
    d = os.path.join(str(run_dir), "pretrain", "shallow")
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump({"verdict": "FAIL", "arch": "shallow", "enforced": False,
                   "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                                  "override_reason": "workflow matrix"},
                   "summary": {"max_atom_mHa": 13.7,
                               "max_dAE_kcalmol": 25.7}}, f)
    assert main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "enforcement is OFF" in out
    assert "preflight SUCCEEDED" in out


def test_preflight_checks_every_distinct_arch(tmp_path, patched, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = _write_resolved_config(run_dir)
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["sweep"]["arch"] = ["shallow", "medium"]
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    _write_pass_certificate(run_dir, "shallow")
    # "medium" has no certificate: the sweep must catch it.
    assert main([str(run_dir)]) == 1
    out = capsys.readouterr().out
    assert "medium" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_preflight.py -v \
  > /tmp/xcq-testlogs/t8-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: the four new tests fail (`assert 0 == 1`, and
`"fidelity gate PASSED" in out` is False).

- [ ] **Step 3: Add the sweep**

In `xcquinox/alec/cluster/_preflight.py`, insert between the compile-smoke
block (which ends at `:643`) and `_log(f"preflight SUCCEEDED: ...")` at `:645`:

```python
    # --- 9. per-architecture fidelity certificates -------------------------
    # The preflight is submitted afterok on the pretrain array, so by this
    # point every distinct architecture has been certified on its own node.
    # This sweep is the run-level cross-check: it catches an architecture that
    # was pretrained under a different submission, a certificate that was
    # deleted, and a partial pretrain array that SLURM reported as complete.
    # ``gate_certificate`` honours a run configured with
    # ``fidelity.enforce: false`` (the workflow-verification matrix), which
    # ``validate_run``, ``merge_v4_arms`` and the figure suite still refuse.
    from xcquinox.alec.cluster.fidelity import gate_certificate
    archs = sorted(set(cfg.sweep.arch))
    uncertified = []
    for arch in archs:
        allowed, message = gate_certificate(run_dir, arch)
        if allowed:
            _log(f"fidelity gate for arch {arch}: {message}")
            continue
        uncertified.append(arch)
        _log(f"ERROR: fidelity certificate for arch {arch} does not release "
             f"the gate: {message}")
    if uncertified:
        _log(f"ERROR: fidelity gate FAILED for {len(uncertified)}/"
             f"{len(archs)} architecture(s) ({', '.join(uncertified)}) -- "
             "blocking the train array")
        return 1
    _log(f"fidelity gate PASSED: {len(archs)}/{len(archs)} architecture "
         "certificate(s) released the gate")

```

- [ ] **Step 4: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/cluster/_preflight.py
python -m pytest xcquinox/alec/tests/test_cluster_preflight.py -v \
  > /tmp/xcq-testlogs/t8-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass.

**Deliverable:** the preflight exits non-zero when any swept architecture lacks a PASS certificate, so the train array's `afterok:<preflight>` dependency never fires.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_cluster_preflight.py -v > /tmp/xcq-testlogs/t8-green.log 2>&1`

---

## Task 9: Run validation -- a run's certificates must match the run

**Files:**
- Modify: `xcquinox/alec/cluster/validate_run.py:1-40` (docstring "Checks" list), `:100-105` (read the manifest version once), `:200-247` (per-arch loop: certificate block first)
- Modify: `xcquinox/alec/tests/test_validate_run.py:72-82` (`_write_run` writes matching certificates), append new tests

**Interfaces:**
- Consumes: the certificate JSON keys `verdict`, `identity`, `xcquinox_version` (Task 3 schema).
- Produces: nothing new; `validate_run` returns `(failures, warnings, n_checked)` as before.

- [ ] **Step 1: Write the failing tests**

In `xcquinox/alec/tests/test_validate_run.py`, replace `_write_run` (`:72-82`)
so a synthetic run carries certificates that match its config, and add a
helper the new tests use to perturb one:

```python
_VERSION = "test-version"


def _write_certificate(run_dir, arch, *, verdict="PASS", identity=None,
                       version=_VERSION):
    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    payload = {
        "verdict": verdict,
        "arch": arch,
        "parent": "pbe",
        "xcquinox_version": version,
        "identity": identity if identity is not None else {
            "basis": _BASIS, "grid_level": 1, "density_fit": False,
            "auxbasis": None, "orientation_lock_strength": 0.0},
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": None},
        "per_system": [], "per_atomization": [],
        "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2,
                    "n_systems": 2, "failure_reasons": []},
    }
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump(payload, f)
    return os.path.join(d, "fidelity_certificate.json")


def _write_run(tmp_path, specs, certificates=True):
    run = tmp_path / "run"
    (run / "specs").mkdir(parents=True)
    (run / "resolved_config.yaml").write_text("placeholder: true\n")
    with open(run / "manifest.json", "w") as f:
        json.dump({"width": 4, "xcquinox_version": _VERSION}, f)
    ser = importlib.import_module("pi" + "ckle")
    for i, spec in enumerate(specs):
        with open(run / "specs" / f"spec_{i:04d}.spec", "wb") as f:
            ser.dump(spec, f)
    if certificates:
        for arch in _ARCHS:
            _write_certificate(str(run), arch)
    return str(run)
```

Append the new tests:

```python
# ---------------------------------------------------------------------------
# Pretraining-fidelity certificates
# ---------------------------------------------------------------------------

def test_missing_certificate_is_a_failure(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")],
                     certificates=False)
    failures, _warnings, _n = vr.validate_run(run)
    assert any("no fidelity_certificate.json" in f for f in failures)
    assert sum("fidelity_certificate" in f for f in failures) == 2


def test_failed_certificate_is_a_failure(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", verdict="FAIL")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("verdict 'FAIL'" in f for f in failures)


def test_unreadable_certificate_is_a_failure(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    path = os.path.join(run, "pretrain", "deep_3x16",
                        "fidelity_certificate.json")
    with open(path, "w") as f:
        f.write("{truncated")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("not readable JSON" in f for f in failures)


def test_identity_mismatch_is_a_failure(tmp_path, patched_cfg):
    """A certificate computed at a different basis or grid says nothing about
    this run: the energy differences it bounds are not this run's."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", identity={
        "basis": "def2-tzvpd", "grid_level": 3, "density_fit": True,
        "auxbasis": "def2-universal-jkfit",
        "orientation_lock_strength": 0.02})
    failures, _warnings, _n = vr.validate_run(run)
    assert any("identity basis=" in f for f in failures)
    assert any("identity grid_level=" in f for f in failures)
    assert any("identity density_fit=" in f for f in failures)
    assert any("identity auxbasis=" in f for f in failures)
    assert any("identity orientation_lock_strength=" in f for f in failures)


def test_unenforced_failure_is_still_a_validation_failure(tmp_path,
                                                          patched_cfg):
    """`fidelity.enforce: false` releases the ON-NODE gates only. A run whose
    certificate reads FAIL can never enter the record, whatever it recorded
    about enforcement."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", verdict="FAIL")
    path = os.path.join(run, "pretrain", "deep_3x16",
                        "fidelity_certificate.json")
    with open(path) as f:
        payload = json.load(f)
    payload["enforced"] = False
    payload["tolerances"]["override_reason"] = "workflow matrix"
    with open(path, "w") as f:
        json.dump(payload, f)
    failures, _warnings, _n = vr.validate_run(run)
    assert any("verdict 'FAIL'" in f for f in failures)


def test_version_mismatch_is_a_failure(tmp_path, patched_cfg):
    """The certificate stands in for the O1-O4 oracles: it certifies the
    installed code. A certificate from other code certifies nothing here."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", version="some-other-build")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("xcquinox_version" in f and "manifest" in f for f in failures)


def test_manifest_without_a_version_warns_rather_than_fails(tmp_path,
                                                            patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    with open(os.path.join(run, "manifest.json"), "w") as f:
        json.dump({"width": 4}, f)
    failures, warnings, _n = vr.validate_run(run)
    assert not any("xcquinox_version" in f for f in failures)
    assert any("xcquinox_version" in w for w in warnings)
```

The existing `test_clean_run_validates` now describes a certified run and must
keep asserting `failures == []`; its pretrain-metadata warning assertion is
unchanged.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest xcquinox/alec/tests/test_validate_run.py -v \
  > /tmp/xcq-testlogs/t9-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: the six new tests fail (no certificate failures are
produced); `test_clean_run_validates` still passes.

- [ ] **Step 3: Read the manifest version once**

In `xcquinox/alec/cluster/validate_run.py`, insert after `width = _read_width(run_dir)`
(`:103`):

```python
    # The manifest's version is the run's code identity; a certificate must
    # have been produced by the same build, since the certificate is what
    # stands in for the spin-scaling oracles on the installed code.
    manifest_version = None
    try:
        with open(os.path.join(run_dir, "manifest.json")) as f:
            manifest_version = json.load(f).get("xcquinox_version")
    except (OSError, ValueError):
        manifest_version = None
```

- [ ] **Step 4: Add the certificate block at the TOP of the per-arch loop**

In `validate_run.py`, the per-arch loop currently starts at `:201` with
`for arch_name in sorted(set(cfg.sweep.arch)):` followed by the pretrain-metadata
checks (which `continue` on a missing metadata file). Insert the certificate
block as the FIRST statements of the loop body, so a metadata `continue` can
never skip it:

```python
    for arch_name in sorted(set(cfg.sweep.arch)):
        # --- fidelity certificate ------------------------------------------
        cert_path = os.path.join(run_dir, "pretrain", arch_name,
                                 "fidelity_certificate.json")
        if not os.path.isfile(cert_path):
            failures.append(
                f"pretrain/{arch_name}: no fidelity_certificate.json -- the "
                "architecture was never shown to reproduce its parent "
                "functional")
        else:
            cert = None
            try:
                with open(cert_path) as f:
                    cert = json.load(f)
            except ValueError as exc:
                failures.append(
                    f"pretrain/{arch_name}: fidelity_certificate.json is not "
                    f"readable JSON ({exc})")
            if isinstance(cert, dict):
                if cert.get("verdict") != "PASS":
                    failures.append(
                        f"pretrain/{arch_name}: fidelity certificate verdict "
                        f"{cert.get('verdict')!r}, expected 'PASS' "
                        f"(summary: {cert.get('summary')})")
                identity = cert.get("identity") or {}
                expected_identity = {
                    "basis": cfg.inputs.basis,
                    "grid_level": int(cfg.inputs.grid_level),
                    "density_fit": bool(
                        getattr(cfg.inputs, "density_fit", False)),
                    "auxbasis": getattr(cfg.inputs, "auxbasis", None),
                    "orientation_lock_strength": float(
                        getattr(cfg.inputs, "orientation_lock_strength", 0.0)),
                }
                for key, want in expected_identity.items():
                    got = identity.get(key)
                    if got is not None and key == "grid_level":
                        got = int(got)
                    if got is not None and key == "orientation_lock_strength":
                        got = float(got)
                    if got != want:
                        failures.append(
                            f"pretrain/{arch_name}: certificate identity "
                            f"{key}={got!r} but the config says {want!r} -- "
                            "the certificate was not computed at this run's "
                            "identity")
                cert_version = cert.get("xcquinox_version")
                if manifest_version is None:
                    warnings.append(
                        f"pretrain/{arch_name}: manifest.json records no "
                        "xcquinox_version, so the certificate's code version "
                        f"({cert_version!r}) cannot be cross-checked")
                elif cert_version != manifest_version:
                    failures.append(
                        f"pretrain/{arch_name}: certificate xcquinox_version "
                        f"{cert_version!r} != manifest {manifest_version!r} "
                        "-- the certificate was produced by different code "
                        "than the run")

        meta_path = os.path.join(run_dir, "pretrain", arch_name,
                                 "pretrain_metadata.json")
```

(the rest of the existing loop body is unchanged).

- [ ] **Step 5: Extend the module docstring**

In the `Checks` list of the module docstring (`:15-34`), append a bullet after
the `pretrain metadata` bullet:

```
* fidelity certificate: every swept architecture must carry
  ``pretrain/<arch>/fidelity_certificate.json`` with ``verdict == "PASS"``.
  The certificate's ``enforced`` field releases the ON-NODE gates only and is
  deliberately ignored here: a workflow-verification run must never be
  mistaken for a result, an
  ``identity`` block equal to the config's basis / grid level / density
  fitting / auxiliary basis / orientation-lock strength, and an
  ``xcquinox_version`` equal to the manifest's. The certificate is what stands
  in for the spin-scaling oracles on the installed code, so a certificate from
  a different build or a different identity certifies nothing about this run.
  A manifest with no recorded version is a warning (it cannot be
  cross-checked), everything else is a failure.
```

- [ ] **Step 6: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/cluster/validate_run.py
python -m pytest xcquinox/alec/tests/test_validate_run.py -v \
  > /tmp/xcq-testlogs/t9-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass.

**Deliverable:** `python -m xcquinox.alec.cluster.validate_run <run_dir>` exits non-zero for a run whose certificates are absent, failed, at the wrong identity, or from other code.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_validate_run.py -v > /tmp/xcq-testlogs/t9-green.log 2>&1`

---

## Task 10: Merge -- an uncertified arm never enters the merged view

**Files:**
- Modify: `notebooks/analysis/merge_v4_arms.py:84-123` (add `_validate_arm_fidelity_certificates` after `_validate_arm_seed_policy`), `:178-180` (call it, then carry the arm's `pretrain/` into the merged view)
- Modify: `notebooks/analysis/test_merge_v4_arms.py:12-27` (`_mk_arm` gains a certificate writer), append new tests

**Interfaces:**
- Consumes: `fidelity.certificate_status(run_dir, arch)`, `fidelity.VERDICT_PASS` (Task 2).
- Produces: `merge_v4_arms._validate_arm_fidelity_certificates(run: Path, arch_names) -> None` (raises `SystemExit`); `<out_dir>/pretrain/<arch>` symlinks in the merged view, read by Task 11.

- [ ] **Step 1: Write the failing tests**

In `notebooks/analysis/test_merge_v4_arms.py`, extend `_mk_arm` (`:12-27`) with
an optional certificate writer and add the new tests:

```python
def _mk_arm(root, base, run_name, n_specs, payload="x", arch=None,
            certified=True, verdict="PASS"):
    import json
    run = root / base / "runs" / run_name
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    arch_name = arch or f"{payload}_arch"
    for i in range(n_specs):
        d = ck / f"spec_{i:04d}"
        d.mkdir()
        (d / "completion.json").write_text(payload)
    (run / "manifest.json").write_text(json.dumps({
        "n_specs": n_specs,
        "specs": [{"index": i,
                   "cell": {"arch": arch_name, "subset_size": i + 1}}
                  for i in range(n_specs)]}))
    if certified:
        d = run / "pretrain" / arch_name
        d.mkdir(parents=True)
        (d / "fidelity_certificate.json").write_text(json.dumps(
            {"verdict": verdict, "arch": arch_name,
             "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2}}))
    return ck


def test_merge_refuses_a_registry_arch_with_no_certificate(tmp_path):
    """A registry architecture that was never certified cannot enter the
    grouped figures: its pretrained networks may be arbitrarily far from the
    parent functional every number on the figure is compared against."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=False)
    with pytest.raises(SystemExit, match="fidelity"):
        mv.build_view(tmp_path, tmp_path / "merged")


def test_merge_refuses_a_registry_arch_whose_certificate_failed(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=True, verdict="FAIL")
    with pytest.raises(SystemExit, match="fidelity"):
        mv.build_view(tmp_path, tmp_path / "merged")


def test_merge_refuses_an_unenforced_failure(tmp_path):
    """`fidelity.enforce: false` releases the ON-NODE gates only; the merge is
    a record layer and refuses a FAIL regardless."""
    import json
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=True, verdict="FAIL")
    cert = (tmp_path / "dfs6311_grid3_v5" / "runs" / "run_20260810T193206Z"
            / "pretrain" / "deep_3x16" / "fidelity_certificate.json")
    payload = json.loads(cert.read_text())
    payload["enforced"] = False
    cert.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="fidelity"):
        mv.build_view(tmp_path, tmp_path / "merged")


def test_merge_accepts_a_certified_registry_arch(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=True)
    report = mv.build_view(tmp_path, tmp_path / "merged")
    assert report["dfs6311_grid3_v5"] == ("run_20260810T193206Z", 1)


def test_merge_skips_non_registry_archs(tmp_path):
    """Test fixtures and legacy display names carry no certificate
    expectation, matching the seed-policy guard."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "legacy", arch="not_a_registry_arch", certified=False)
    report = mv.build_view(tmp_path, tmp_path / "merged")
    assert report["dfs6311_grid3_v5"] == ("run_20260810T193206Z", 1)


def test_merged_view_carries_the_arms_pretrain_certificates(tmp_path):
    """The merged directory has no pretrain stage of its own, so the figure
    layer would read every arch as uncertified unless the arms' certificates
    travel with the merge."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16")
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    cert = out / "pretrain" / "deep_3x16" / "fidelity_certificate.json"
    assert cert.is_file()
    import json
    assert json.loads(cert.read_text())["verdict"] == "PASS"
```

Add `import pytest` to the file's imports if it is not already there (it is, at
`:5`).

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest notebooks/analysis/test_merge_v4_arms.py -v \
  > /tmp/xcq-testlogs/t10-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: `DID NOT RAISE SystemExit` for the two refusal tests
and `assert False` (no `pretrain/` in the merged view) for the last.

- [ ] **Step 3: Add the guard**

In `notebooks/analysis/merge_v4_arms.py`, insert after
`_validate_arm_seed_policy` (after `:123`):

```python
def _validate_arm_fidelity_certificates(run: Path, arch_names) -> None:
    """Refuse an arm whose REGISTRY architectures lack a PASS certificate.

    The per-architecture physics certificate
    (``xcquinox.alec.cluster.fidelity``) is the only machine-checked statement
    that an architecture's pretrained networks reproduce their parent
    functional. Without it the arm's held-out numbers cannot be read against
    the parent baselines the grouped figures draw, so an uncertified or failed
    arm is refused here rather than silently merged. This is a RECORD layer:
    it calls ``certificate_status`` and requires PASS, ignoring the
    certificate's ``enforced`` field, which releases the on-node gates of a
    workflow-verification run only. Archs the registry does
    not know (test fixtures, legacy display names) carry no certificate
    expectation and are skipped, matching
    :func:`_validate_arm_seed_policy`; an unreadable certificate is
    unverifiable and therefore refused.
    """
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.cluster.fidelity import (VERDICT_PASS,
                                                certificate_status)
    registry_archs = set()
    for a in arch_names:
        if not a:
            continue
        try:
            get_architecture(a)
        except KeyError:
            continue
        registry_archs.add(a)
    for arch in sorted(registry_archs):
        status, reason = certificate_status(str(run), arch)
        if status != VERDICT_PASS:
            raise SystemExit(
                f"[merge] REFUSING {run}: arch {arch} has no PASS "
                f"pretraining-fidelity certificate ({reason}) -- an "
                "uncertified arm cannot enter the grouped figures")
```

- [ ] **Step 4: Call it and carry the certificates into the merged view**

In `build_view`, replace the single `_validate_arm_seed_policy(...)` call at
`:178-179` with:

```python
        _arm_archs = {(e.get("cell") or {}).get("arch")
                      for e in entries.values()}
        _validate_arm_seed_policy(run, _arm_archs)
        _validate_arm_fidelity_certificates(run, _arm_archs)
        # The merged directory runs no pretrain stage, so the arms' per-arch
        # certificates travel with the merge: the figure layer resolves them
        # through the same <run_dir>/pretrain/<arch> layout it uses for a
        # single-arm run.
        arm_pretrain = run / "pretrain"
        if arm_pretrain.is_dir():
            pt_out = out_dir / "pretrain"
            pt_out.mkdir(exist_ok=True)
            for src in sorted(arm_pretrain.iterdir()):
                if src.is_dir() and not (pt_out / src.name).exists():
                    (pt_out / src.name).symlink_to(src.resolve())
```

- [ ] **Step 5: `py_compile` and run the tests GREEN**

```bash
python -m py_compile notebooks/analysis/merge_v4_arms.py
python -m pytest notebooks/analysis/test_merge_v4_arms.py -v \
  > /tmp/xcq-testlogs/t10-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass, including the pre-existing merge tests whose
fixtures use non-registry arch names.

**Deliverable:** `merge_v4_arms.build_view` raises `SystemExit` for an uncertified arm and the merged view carries each arm's certificates.
**Covering command:** `python -m pytest notebooks/analysis/test_merge_v4_arms.py -v > /tmp/xcq-testlogs/t10-green.log 2>&1`

---

## Task 11: Figures -- coverage, footer disclosure, provenance numbers and the hard fail

**Files:**
- Modify: `notebooks/analysis/make_ablation_arch_figure.py:545-592` (`arch_coverage`), `:595-616` (`coverage_note`), `:937-958` (`provenance_footer`), `:1505-1535` (add `_FIDELITY_*` beside `_VXC_DISCLOSURE`), `:2760-2788` (`_stamp_parity_footer`), `:8139-8145` (`build_bh76w411_suite`)
- Modify: `notebooks/analysis/test_make_ablation_arch_figure.py:46-115` (`_make_run_dir` writes certificates), append new tests

**Interfaces:**
- Consumes: `fidelity.certificate_status(run_dir, arch)`, `fidelity.VERDICT_PASS`, `read_certificate` and the certificate `summary` keys (Tasks 2-3); the merged view's `pretrain/<arch>` symlinks (Task 10).
- Produces:
  - `_arch_uncertified(run_dir: Path, arch: str) -> bool`
  - `arch_coverage(...)["uncertified"] -> List[str]`
  - `fidelity_summary(run_dir: Path, archs=None) -> Dict[str, Any] | None` -- `{"n_archs": int, "max_atom_mHa": float, "max_dAE_kcalmol": float}`; `archs=None` means every architecture in the run's manifest grid. As shipped (commit 73ce49ab4) the dictionary also carries `n_archs_without_numbers` (certificates that state no numbers, excluded from `n_archs`) and `not_pass` (`"arch (STATUS)"` for every contributing certificate that is not PASS), and the footer marks the line when `not_pass` is non-empty.
  - `provenance_footer(baseline, scan_baseline=None, fidelity=None) -> str`
  - `_FIDELITY_DISCLOSURE: str`, `_FIDELITY_GATE_DATE: str`, `_run_predates_fidelity_gate(run_id) -> bool`

- [ ] **Step 1: Write the failing tests**

In `notebooks/analysis/test_make_ablation_arch_figure.py`, extend
`_make_run_dir` (`:46-115`) so its runs are certified. Insert immediately after
`(run_dir / "specs").mkdir()` (`:68`):

```python
    # Every architecture of a real run carries a PASS fidelity certificate;
    # the figure fixtures describe that state, and the gate's own tests
    # remove or downgrade one.
    for arch in sorted({c["arch"] for c in specs}):
        pd = run_dir / "pretrain" / arch
        pd.mkdir(parents=True, exist_ok=True)
        (pd / "fidelity_certificate.json").write_text(json.dumps(
            {"verdict": "PASS", "arch": arch,
             "summary": {"max_atom_mHa": 0.31, "max_dAE_kcalmol": 0.62,
                         "n_systems": 40, "failure_reasons": []}}))
```

Append the new tests:

```python
# ---------------------------------------------------------------------------
# Pretraining-fidelity certificates in the figure layer
# ---------------------------------------------------------------------------

def test_arch_coverage_reports_no_uncertified_arch_for_a_certified_run(
        tmp_path):
    run = _make_run_dir(tmp_path)
    assert fig.arch_coverage(run)["uncertified"] == []


def test_arch_coverage_flags_a_missing_certificate(tmp_path):
    run = _make_run_dir(tmp_path)
    (run / "pretrain" / "deep" / "fidelity_certificate.json").unlink()
    assert fig.arch_coverage(run)["uncertified"] == ["deep"]


def test_arch_coverage_flags_a_failed_certificate(tmp_path):
    run = _make_run_dir(tmp_path)
    (run / "pretrain" / "deep" / "fidelity_certificate.json").write_text(
        json.dumps({"verdict": "FAIL", "arch": "deep",
                    "summary": {"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7}}))
    assert "deep" in fig.arch_coverage(run)["uncertified"]


def test_arch_coverage_flags_an_unenforced_failure(tmp_path):
    """The figure layer is a record layer: `enforced: false` does not make a
    FAIL acceptable on a figure."""
    run = _make_run_dir(tmp_path)
    (run / "pretrain" / "deep" / "fidelity_certificate.json").write_text(
        json.dumps({"verdict": "FAIL", "arch": "deep", "enforced": False,
                    "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                                   "override_reason": "workflow matrix"},
                    "summary": {"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7}}))
    assert "deep" in fig.arch_coverage(run)["uncertified"]


def test_arch_coverage_ignores_non_registry_arch_names(tmp_path):
    """Legacy display names carry no certificate expectation, matching the
    merge guard, or every historical figure would read as uncertified."""
    run = tmp_path / "r"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(
        {"n_specs": 1, "width": 4,
         "specs": [{"index": 0, "spec_file": "spec_0000.spec",
                    "sha256": "x" * 64,
                    "cell": {"arch": "legacy_display_name",
                             "subset_size": 1}}]}))
    (run / "specs").mkdir()
    assert fig.arch_coverage(run)["uncertified"] == []


def test_coverage_note_names_the_uncertified_archs(tmp_path):
    run = _make_run_dir(tmp_path)
    (run / "pretrain" / "deep" / "fidelity_certificate.json").unlink()
    note = fig.coverage_note(run)
    assert "UNCERTIFIED (no PASS fidelity certificate)" in note
    assert "deep" in note


def test_coverage_note_is_silent_when_every_arch_is_certified(tmp_path):
    run = _make_run_dir(tmp_path)
    assert "UNCERTIFIED" not in fig.coverage_note(run)


def test_pre_gate_runs_carry_the_fidelity_disclosure_first(tmp_path):
    import matplotlib.pyplot as plt
    assert fig._run_predates_fidelity_gate("run_20260810T202813Z") is True
    assert fig._run_predates_fidelity_gate("run_20260901T000000Z") is False
    assert fig._run_predates_fidelity_gate("synthetic-id") is False
    f = plt.figure()
    fig._stamp_parity_footer(f, run_id="run_20260810T202813Z", title="t",
                             note="base note", provenance=None, caveat=None)
    texts = [t.get_text() for t in f.texts]
    stamped = [t for t in texts if fig._FIDELITY_DISCLOSURE in t]
    assert stamped, texts
    assert stamped[0].startswith(fig._FIDELITY_DISCLOSURE)
    assert "base note" in stamped[0]
    plt.close(f)


def test_post_gate_runs_carry_no_fidelity_disclosure(tmp_path):
    import matplotlib.pyplot as plt
    f = plt.figure()
    fig._stamp_parity_footer(f, run_id="run_20260901T000000Z", title="t",
                             note="base note", provenance=None, caveat=None)
    assert not any(fig._FIDELITY_DISCLOSURE in t.get_text() for t in f.texts)
    plt.close(f)


def test_fidelity_summary_reads_the_worst_certificate_numbers(tmp_path):
    run = _make_run_dir(tmp_path)
    (run / "pretrain" / "deep_attn" / "fidelity_certificate.json").write_text(
        json.dumps({"verdict": "PASS", "arch": "deep_attn",
                    "summary": {"max_atom_mHa": 0.9,
                                "max_dAE_kcalmol": 0.85}}))
    got = fig.fidelity_summary(run, ["deep", "deep_notransform", "deep_attn"])
    assert got["n_archs"] == 3
    assert got["max_atom_mHa"] == pytest.approx(0.9)
    assert got["max_dAE_kcalmol"] == pytest.approx(0.85)


def test_fidelity_summary_is_none_without_certificates(tmp_path):
    run = _make_run_dir(tmp_path)
    for arch in ("deep", "deep_notransform", "deep_attn"):
        (run / "pretrain" / arch / "fidelity_certificate.json").unlink()
    assert fig.fidelity_summary(run, ["deep"]) is None


def test_provenance_footer_is_byte_identical_without_a_fidelity_argument():
    baseline = {"bh76": 8.0, "w411": 12.0, "combined": 10.0}
    assert fig.provenance_footer(baseline) == fig.provenance_footer(
        baseline, None, None)


def test_provenance_footer_carries_the_certificate_numbers():
    baseline = {"bh76": 8.0, "w411": 12.0, "combined": 10.0}
    s = fig.provenance_footer(baseline, None,
                              {"n_archs": 7, "max_atom_mHa": 0.42,
                               "max_dAE_kcalmol": 0.71})
    assert "Pretraining fidelity" in s
    assert "7 arch" in s
    assert "0.42 mHa" in s
    assert "0.71 kcal/mol" in s


def test_build_bh76w411_suite_refuses_an_uncertified_run(tmp_path):
    root, runs = _make_bh76w411_results(tmp_path)
    (runs["svp_grid2"] / "pretrain" / "deep"
     / "fidelity_certificate.json").unlink()
    with pytest.raises(ValueError, match="fidelity certificate"):
        fig.build_bh76w411_suite(results_root=root, outroot=tmp_path / "f")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest notebooks/analysis/test_make_ablation_arch_figure.py \
  -k "fidelity or uncertified or coverage_note or provenance_footer" -v \
  > /tmp/xcq-testlogs/t11-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: `KeyError: 'uncertified'`,
`AttributeError: ... has no attribute '_run_predates_fidelity_gate'` /
`'fidelity_summary'` / `'_FIDELITY_DISCLOSURE'`, and
`DID NOT RAISE ValueError`.

- [ ] **Step 3: Add the certificate predicate and the coverage entry**

In `notebooks/analysis/make_ablation_arch_figure.py`, insert immediately before
`arch_coverage` (before `:545`):

```python
def _arch_uncertified(run_dir: Path, arch: str) -> bool:
    """True when ``arch`` has no PASS fidelity certificate under ``run_dir``.

    Registry architectures only: a name the registry does not know (legacy
    display name, test fixture) carries no certificate expectation, matching
    ``merge_v4_arms._validate_arm_fidelity_certificates``. This is a RECORD
    layer: it requires PASS and ignores the certificate's ``enforced`` field,
    which releases the on-node gates of a workflow-verification run only. A merged view
    resolves the same ``<run_dir>/pretrain/<arch>`` layout because
    ``merge_v4_arms.build_view`` links each arm's pretrain directory into it.
    """
    try:
        from xcquinox.alec.config import get_architecture
        from xcquinox.alec.cluster.fidelity import (VERDICT_PASS,
                                                    certificate_status)
    except ImportError:      # the analysis layer runs without the package
        return False
    try:
        get_architecture(arch)
    except KeyError:
        return False
    status, _reason = certificate_status(str(run_dir), arch)
    return status != VERDICT_PASS


def fidelity_summary(run_dir: Path,
                     archs=None) -> Optional[Dict[str, Any]]:
    """Worst per-architecture certificate numbers over ``archs``, or ``None``.

    ``{"n_archs", "max_atom_mHa", "max_dAE_kcalmol"}`` -- the largest free-atom
    E_xc offset (mHa) and the largest atomization-energy offset (kcal/mol) any
    certified architecture of the run carries. ``archs=None`` reads every
    architecture in the run's manifest grid. ``None`` when no architecture has
    a readable certificate, which keeps the provenance footer of every
    pre-gate figure byte-identical.
    """
    try:
        from xcquinox.alec.cluster.fidelity import read_certificate
        from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    except ImportError:
        return None
    if archs is None:
        cells = ccp._read_manifest_cells(run_dir)
        archs = sorted({c.get("arch") for c in cells.values() if c.get("arch")})
    atom_devs, ae_devs, n = [], [], 0
    for arch in archs:
        cert = read_certificate(pretrain_checkpoint_dir(str(run_dir), arch))
        if not cert:
            continue
        n += 1
        summary = cert.get("summary") or {}
        if _is_num(summary.get("max_atom_mHa")):
            atom_devs.append(abs(float(summary["max_atom_mHa"])))
        if _is_num(summary.get("max_dAE_kcalmol")):
            ae_devs.append(abs(float(summary["max_dAE_kcalmol"])))
    if not n or not atom_devs or not ae_devs:
        return None
    return {"n_archs": n, "max_atom_mHa": max(atom_devs),
            "max_dAE_kcalmol": max(ae_devs)}
```

In `arch_coverage`, add `uncertified` to the docstring's returned-keys list and
compute it. After the `grid_archs` assignment (`:566`), add:

```python
    uncertified: set = {a for a in grid_archs if _arch_uncertified(run_dir, a)}
```

and add to the returned dict (after `"in_progress": _ordered(in_progress),`):

```python
        # Architectures whose pretrained networks were never shown to
        # reproduce their parent functional. Their held-out numbers are not
        # comparable with the parent baselines the figures draw.
        "uncertified": _ordered(uncertified),
```

In `coverage_note`, insert before the final `return "  ".join(parts)`:

```python
    if cov.get("uncertified"):
        parts.append("UNCERTIFIED (no PASS fidelity certificate): "
                     f"{', '.join(cov['uncertified'])}.")
```

- [ ] **Step 4: Add the footer disclosure**

Insert immediately after the `_VXC_DISCLOSURE` block (after `:1524`):

```python
# ---------------------------------------------------------------------------
# Pretraining-fidelity disclosure. Runs started before the per-architecture
# physics certificate existed were never checked against their parent
# functional; the offsets measured on those checkpoints span 2.3 to 56
# kcal/mol in atomization energies (recorded in
# xcquinox/alec/SPEC_pretrain_fidelity_program.md Section 2), larger than the
# architecture differences the figures resolve. Runs started after the gate
# date carry a PASS certificate for every architecture (enforced by
# build_bh76w411_suite) and draw no disclosure.
# ---------------------------------------------------------------------------
_FIDELITY_GATE_DATE = "20260821"
_FIDELITY_DISCLOSURE = (
    "PRETRAINING FIDELITY: this run predates the per-architecture physics "
    "certificate; its pretrained networks were never checked against their "
    "parent functional (PBE for GGA-rung, SCAN for meta-GGA). Atomization-"
    "energy offsets measured on pre-certificate checkpoints span 2.3 to 56 "
    "kcal/mol.")


def _run_predates_fidelity_gate(run_id: str) -> bool:
    """True when the run's encoded start date predates the certificate gate.

    Run directories encode their start as ``run_YYYYMMDDTHHMMSSZ``; an id
    without that stamp is conservatively treated as post-gate (no disclosure)
    so synthetic and test ids do not acquire one."""
    import re as _re
    m = _re.search(r"run_(\d{8})T", str(run_id))
    return bool(m) and m.group(1) < _FIDELITY_GATE_DATE
```

In `_stamp_parity_footer`, insert as the FIRST statement of the body (before
the `if _run_predates_vxc_fix(run_id):` block at `:2769`):

```python
    if _run_predates_fidelity_gate(run_id):
        # Leads the footer: it bounds every number on the figure, so it must
        # be read before the V_xc provenance and before the panel note.
        note = _FIDELITY_DISCLOSURE + ("  " + note if note else "")
```

- [ ] **Step 5: Carry the certificate numbers in `provenance_footer`**

Change the signature and add the clause. Replace the `def provenance_footer(...)`
line (`:937-938`) with:

```python
def provenance_footer(baseline: Dict[str, float],
                      scan_baseline: Optional[Dict[str, float]] = None,
                      fidelity: Optional[Dict[str, Any]] = None) -> str:
```

extend its docstring with:

```
    ``fidelity`` (from :func:`fidelity_summary`) appends the worst
    per-architecture pretraining-fidelity numbers of the run, so the figure
    states how close the pretrained networks are to their parent functional.
    ``None`` -> the string is byte-identical to the pre-certificate footer.
```

and insert before the final `return s`:

```python
    if fidelity:
        s += (f" Pretraining fidelity (worst of {fidelity['n_archs']} arch):"
              f" |dE_xc| atom <= {fidelity['max_atom_mHa']:.2f} mHa"
              f" / |dAE| <= {fidelity['max_dAE_kcalmol']:.2f} kcal/mol.")
    # Shipped form (commit 73ce49ab4): the same sentence, followed by the
    # number of certificates without numbers when non-zero and by the
    # "arch (STATUS)" list of contributing certificates that did not pass.
```

- [ ] **Step 6: Add the hard fail in `build_bh76w411_suite`**

Insert immediately after the `archs_not_in_order` `raise ValueError(...)` block
(after `:8145`):

```python
            uncertified = cov["coverage"]["uncertified"]
            if uncertified:
                raise ValueError(
                    f"{basis} {cov['run']} carries architectures with no PASS "
                    f"pretraining-fidelity certificate {uncertified}; their "
                    "pretrained networks were never shown to reproduce their "
                    "parent functional, so their held-out numbers cannot be "
                    "read against the parent baselines these figures draw. "
                    "Run `python -m xcquinox.alec.cluster.fidelity <run_dir> "
                    "<arch_idx>` for each and resubmit the arm.")
```

- [ ] **Step 7: Wire the summary into all three provenance-footer call sites**

`provenance_footer` is called at three places, each with ``run_dir`` in scope.
Pass the run's certificate summary at every one so no figure family is left
without it:

- `:3061` -- replace `prov = provenance_footer(baseline)` with
  `prov = provenance_footer(baseline, None, fidelity_summary(run_dir))`
- `:7346` -- replace `prov = provenance_footer(baseline, scan_baseline)` with
  `prov = provenance_footer(baseline, scan_baseline, fidelity_summary(run_dir))`
- `:7994` -- replace `prov = provenance_footer(baseline, scan_baseline)` with
  `prov = provenance_footer(baseline, scan_baseline, fidelity_summary(run_dir))`

Add the covering test to `notebooks/analysis/test_make_ablation_arch_figure.py`:

```python
def test_build_parity_variants_stamps_the_certificate_numbers(tmp_path,
                                                              monkeypatch):
    """The wiring: a certified run's figures carry its worst certificate
    numbers on the grey provenance line."""
    run = _make_run_dir(tmp_path)
    seen = []
    real = fig.provenance_footer

    def _spy(baseline, scan_baseline=None, fidelity=None):
        seen.append(fidelity)
        return real(baseline, scan_baseline, fidelity)

    monkeypatch.setattr(fig, "provenance_footer", _spy)
    fig.build_parity_variants(run, tmp_path / "out")
    assert seen, "provenance_footer was never called"
    assert seen[0] is not None
    assert seen[0]["max_atom_mHa"] == pytest.approx(0.31)
    assert seen[0]["max_dAE_kcalmol"] == pytest.approx(0.62)
```

- [ ] **Step 8: `py_compile` and run the tests GREEN**

```bash
python -m py_compile notebooks/analysis/make_ablation_arch_figure.py
python -m pytest notebooks/analysis/test_make_ablation_arch_figure.py -v \
  > /tmp/xcq-testlogs/t11-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass, including the pre-existing coverage and
suite tests (whose fixtures now write certificates).

**Deliverable:** the figure layer names uncertified architectures in the coverage note, refuses to build the BH76/W4-11 suite for a run that carries one, and renders the certificate's worst numbers in the provenance footer.
**Covering command:** `python -m pytest notebooks/analysis/test_make_ablation_arch_figure.py -v > /tmp/xcq-testlogs/t11-green.log 2>&1`

---

## Task 12: In-process gate -- `train._build_model` refuses an uncertified checkpoint

**Files:**
- Modify: `xcquinox/alec/train.py:305-336` (add `_ALLOW_UNCERTIFIED_ENV` + `_require_fidelity_certificate` above `_build_model`, call it inside)
- Modify: `xcquinox/alec/tests/test_train.py:1090-1150` (`test_pretrain_checkpoint_lower_initial_loss` writes a PASS certificate), append new tests

**Interfaces:**
- Consumes: `fidelity.certificate_status_in(pretrain_dir)` and `fidelity.VERDICT_PASS` (Task 2).
- Produces: `train._ALLOW_UNCERTIFIED_ENV: str` == `"XCQUINOX_ALLOW_UNCERTIFIED"`; `train._require_fidelity_certificate(pretrain_checkpoint: str) -> None` (raises `ValueError`).

Design note on the two existing tests the task instruction names.
`test_train.py:145` (`test_validate_missing_pretrain_checkpoint`) exercises
`TrainingSpec.validate()`, never `_build_model`, so the gate does not touch it
and it is left alone. `test_train.py:1090`
(`test_pretrain_checkpoint_lower_initial_loss`) does call `run_training` with a
synthesised checkpoint, so it gets a PASS certificate written beside the
serialised networks rather than the environment escape hatch: that way the
test exercises the gate's ACCEPT path, and the escape hatch gets a test of its
own instead of being the only thing keeping the suite green.

- [ ] **Step 1: Write the failing tests**

In `xcquinox/alec/tests/test_train.py`, inside
`test_pretrain_checkpoint_lower_initial_loss`, immediately after the two
`eqx.tree_serialise_leaves(...)` calls that write `cnet.eqx` (`:1129`), insert:

```python
        # train._build_model refuses an uncertified pretrain checkpoint; this
        # one is synthesised by the test, so it carries the PASS certificate a
        # real pretrain job would have written beside the networks.
        with open(os.path.join(pretrain_dir,
                               "fidelity_certificate.json"), "w") as f:
            json.dump({"verdict": "PASS", "arch": arch.name}, f)
```

Append the new tests to the same file:

```python
# ---------------------------------------------------------------------------
# In-process pretraining-fidelity gate
# ---------------------------------------------------------------------------

def test_build_model_refuses_an_uncertified_pretrain_checkpoint(tmp_path,
                                                                monkeypatch):
    """A checkpoint with no certificate is refused with an actionable message.

    The pre-certificate checkpoints were 2.3 to 56 kcal/mol away from their
    parent in atomization energies; training from one silently measures that
    offset instead of the architecture."""
    import equinox as eqx
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.networks import create_network_pair

    monkeypatch.delenv(train_mod._ALLOW_UNCERTIFIED_ENV, raising=False)
    arch = _make_arch()
    xnet, cnet = create_network_pair(arch, seed=0)
    d = tmp_path / "pretrain_ckpt"
    d.mkdir()
    eqx.tree_serialise_leaves(str(d / "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(str(d / "cnet.eqx"), cnet)

    spec = _make_training_spec(pretrain_checkpoint=str(d))
    with pytest.raises(ValueError, match="fidelity"):
        train_mod._build_model(spec)


def test_build_model_accepts_a_passing_certificate(tmp_path, monkeypatch):
    import equinox as eqx
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.networks import create_network_pair

    monkeypatch.delenv(train_mod._ALLOW_UNCERTIFIED_ENV, raising=False)
    arch = _make_arch()
    xnet, cnet = create_network_pair(arch, seed=0)
    d = tmp_path / "pretrain_ckpt"
    d.mkdir()
    eqx.tree_serialise_leaves(str(d / "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(str(d / "cnet.eqx"), cnet)
    with open(d / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": arch.name}, f)

    spec = _make_training_spec(pretrain_checkpoint=str(d))
    assert train_mod._build_model(spec) is not None


def test_build_model_refuses_a_failed_certificate(tmp_path, monkeypatch):
    import equinox as eqx
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.networks import create_network_pair

    monkeypatch.delenv(train_mod._ALLOW_UNCERTIFIED_ENV, raising=False)
    arch = _make_arch()
    xnet, cnet = create_network_pair(arch, seed=0)
    d = tmp_path / "pretrain_ckpt"
    d.mkdir()
    eqx.tree_serialise_leaves(str(d / "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(str(d / "cnet.eqx"), cnet)
    with open(d / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "FAIL", "arch": arch.name,
                   "summary": {"max_atom_mHa": 13.7,
                               "max_dAE_kcalmol": 25.7}}, f)

    spec = _make_training_spec(pretrain_checkpoint=str(d))
    with pytest.raises(ValueError, match="FAIL"):
        train_mod._build_model(spec)


def test_env_escape_hatch_allows_an_uncertified_checkpoint(tmp_path,
                                                           monkeypatch):
    """Probes and one-off experiments opt out explicitly, in the environment,
    where it is visible in the job script."""
    import equinox as eqx
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.networks import create_network_pair

    monkeypatch.setenv(train_mod._ALLOW_UNCERTIFIED_ENV, "1")
    arch = _make_arch()
    xnet, cnet = create_network_pair(arch, seed=0)
    d = tmp_path / "pretrain_ckpt"
    d.mkdir()
    eqx.tree_serialise_leaves(str(d / "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(str(d / "cnet.eqx"), cnet)

    spec = _make_training_spec(pretrain_checkpoint=str(d))
    assert train_mod._build_model(spec) is not None


def test_from_scratch_models_are_untouched_by_the_gate(monkeypatch):
    from xcquinox.alec import train as train_mod
    monkeypatch.delenv(train_mod._ALLOW_UNCERTIFIED_ENV, raising=False)
    spec = _make_training_spec()
    assert spec.pretrain_checkpoint is None
    assert train_mod._build_model(spec) is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest xcquinox/alec/tests/test_train.py \
  -k "uncertified or fidelity or from_scratch_models" -v \
  > /tmp/xcq-testlogs/t12-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: `AttributeError: module 'xcquinox.alec.train' has no
attribute '_ALLOW_UNCERTIFIED_ENV'`.

- [ ] **Step 3: Add the gate**

In `xcquinox/alec/train.py`, insert immediately before `def _build_model`
(before `:310`):

```python
# Escape hatch for the in-process pretraining-fidelity gate. Set to "1" to
# train from a checkpoint that carries no PASS certificate (local probes,
# unit tests, deliberate pre-certificate reproductions). Read from the
# environment rather than the spec so it is visible in the job script.
_ALLOW_UNCERTIFIED_ENV = "XCQUINOX_ALLOW_UNCERTIFIED"


def _require_fidelity_certificate(pretrain_checkpoint: str) -> None:
    """Refuse a pretrain checkpoint with no PASS fidelity certificate.

    The certificate (``xcquinox.alec.cluster.fidelity``) is the only
    machine-checked statement that a pretrained xnet/cnet pair reproduces its
    parent functional (PBE for a GGA-rung architecture, SCAN for a meta-GGA
    one). Pre-certificate checkpoints were 2.3 to 56 kcal/mol away from their
    parent in atomization energies (recorded in
    ``xcquinox/alec/SPEC_pretrain_fidelity_program.md`` Section 2), larger than
    every effect the training is meant to resolve, so a training run started
    from one measures the pretraining error rather than the architecture.

    The cluster path already refuses at the array-task level; this in-process
    check also covers notebooks, probes and any direct ``run_training`` call.
    """
    if os.environ.get(_ALLOW_UNCERTIFIED_ENV) == "1":
        return
    from xcquinox.alec.cluster.fidelity import (VERDICT_PASS,
                                                certificate_status_in)
    status, reason = certificate_status_in(pretrain_checkpoint)
    if status == VERDICT_PASS:
        return
    raise ValueError(
        f"refusing to train from pretrain_checkpoint "
        f"{pretrain_checkpoint!r}: {reason}. Produce one with `python -m "
        f"xcquinox.alec.cluster.fidelity <run_dir> <arch_idx>`, or set "
        f"{_ALLOW_UNCERTIFIED_ENV}=1 to train from an uncertified checkpoint "
        "deliberately.")
```

and add the call as the first statement after the from-scratch early return in
`_build_model` (`:312-313`):

```python
def _build_model(spec: TrainingSpec) -> AlecGGAModel:
    """Build model from scratch or pretrain checkpoint.

    A pretrain checkpoint must carry a PASS fidelity certificate; see
    :func:`_require_fidelity_certificate`.
    """
    if spec.pretrain_checkpoint is None:
        return AlecGGAModel.from_arch(spec.arch, seed=spec.seed)
    _require_fidelity_certificate(spec.pretrain_checkpoint)
    xnet_skeleton, cnet_skeleton = create_network_pair(spec.arch, seed=spec.seed)
```

- [ ] **Step 4: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/train.py
python -m pytest xcquinox/alec/tests/test_train.py -v \
  > /tmp/xcq-testlogs/t12-green.log 2>&1; echo "rc=$?"
python -m pytest xcquinox/alec/tests/test_train.py -m slow -v \
  > /tmp/xcq-testlogs/t12-slow.log 2>&1; echo "rc=$?"
```
Read both logs. Expected: all pass, including the slow
`test_pretrain_checkpoint_lower_initial_loss`.

**Deliverable:** `run_training` from an uncertified pretrain checkpoint raises with an actionable message unless the environment opts out explicitly.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_train.py -v > /tmp/xcq-testlogs/t12-green.log 2>&1`

---

## Task 13: `cluster status` reports the certificate count

**Files:**
- Modify: `xcquinox/alec/cluster/__main__.py:51-56` (import), `:957-985` (`_pretrain_status`)
- Modify: `xcquinox/alec/tests/test_cluster_cli.py:1026-1052` (the pretrain-status test), append a new test

**Interfaces:**
- Consumes: `fidelity.certificate_status_in(pretrain_dir)`, `fidelity.VERDICT_PASS` (Task 2).
- Produces: the `_pretrain_status` line format
  `"<D>/<N> architecture checkpoint pair(s) present, <C>/<N> architecture certificate(s) PASS"`.

- [ ] **Step 1: Write the failing tests**

In `xcquinox/alec/tests/test_cluster_cli.py`, update the exact-string assertion
at `:1051` and append a second test:

```python
    line = cli._pretrain_status(str(run_dir))
    assert line == ("1/1 architecture checkpoint pair(s) present, "
                    "0/1 architecture certificate(s) PASS")
```

```python
def test_pretrain_status_counts_passing_certificates(tmp_path):
    """A checkpoint pair on disk is not the same as a certified architecture:
    `status` must show both counts so an operator can see the pretrain array
    finished but the physics gate did not."""
    import json
    from xcquinox.alec.cluster.grid_config import (
        load_grid_config, pretrain_checkpoint_dir,
    )
    run_dir = tmp_path / "run_TESTID"
    run_dir.mkdir()
    d = _base_config_dict()
    d["sweep"]["arch"] = ["medium", "shallow"]
    gp = tmp_path / "_g.json"
    gp.write_text(json.dumps(d))
    cfg = load_grid_config(str(gp))
    cli._write_resolved_config(cfg, str(run_dir))

    for arch in sorted(set(cfg.sweep.arch)):
        ck = pretrain_checkpoint_dir(str(run_dir), arch)
        os.makedirs(ck, exist_ok=True)
        open(os.path.join(ck, "xnet.eqx"), "wb").close()
        open(os.path.join(ck, "cnet.eqx"), "wb").close()
    # Only one of the two certified.
    ck = pretrain_checkpoint_dir(str(run_dir), "medium")
    with open(os.path.join(ck, "fidelity_certificate.json"), "w") as f:
        json.dump({"verdict": "PASS", "arch": "medium"}, f)

    assert cli._pretrain_status(str(run_dir)) == (
        "2/2 architecture checkpoint pair(s) present, "
        "1/2 architecture certificate(s) PASS")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_cli.py \
  -k pretrain_status -v > /tmp/xcq-testlogs/t13-red.log 2>&1; echo "rc=$?"
```
Read the log. Expected: both fail on the string comparison (the current status
line has no certificate clause).

- [ ] **Step 3: Extend `_pretrain_status`**

In `xcquinox/alec/cluster/__main__.py`, add one module-level import beside the
other cluster imports (after `:60`, next to `from
xcquinox.alec.cluster.materialize import write_manifest`):

```python
from xcquinox.alec.cluster.fidelity import (VERDICT_PASS,
                                            certificate_status_in)
```

(`cluster.fidelity`'s module body imports stdlib plus `grid_config` /
`materialize` only, so this adds no import cost to the login-node CLI beyond
what importing any cluster module already carries -- the contract is pinned by
`test_cluster_fidelity.test_fidelity_module_body_imports_only_cheap_modules`.)

Extend the docstring of `_pretrain_status` with:

```
    Alongside the checkpoint-pair count the line reports how many
    architectures carry a PASS fidelity certificate: the pretrain array can
    finish and still leave the campaign blocked, because the train array is
    gated on the certificate, not on the checkpoint files.
```

and replace the counting loop and the return (`:977-985`) with:

```python
    archs = sorted(set(cfg.sweep.arch))
    done = 0
    certified = 0
    for arch in archs:
        d = pretrain_checkpoint_dir(run_dir, arch)
        if (os.path.exists(os.path.join(d, "xnet.eqx"))
                and os.path.exists(os.path.join(d, "cnet.eqx"))):
            done += 1
        status, _reason = certificate_status_in(d)
        if status == VERDICT_PASS:
            certified += 1
    return (f"{done}/{len(archs)} architecture checkpoint pair(s) present, "
            f"{certified}/{len(archs)} architecture certificate(s) PASS")
```

- [ ] **Step 4: `py_compile` and run the tests GREEN**

```bash
python -m py_compile xcquinox/alec/cluster/__main__.py
python -m pytest xcquinox/alec/tests/test_cluster_cli.py -v \
  > /tmp/xcq-testlogs/t13-green.log 2>&1; echo "rc=$?"
```
Read the log. Expected: all pass.

**Deliverable:** `python -m xcquinox.alec.cluster status <run_dir>` shows both the checkpoint count and the certificate count.
**Covering command:** `python -m pytest xcquinox/alec/tests/test_cluster_cli.py -v > /tmp/xcq-testlogs/t13-green.log 2>&1`

---

## Task 14: sbatch surface, walltime note, HISTORY entry and the full-suite run

**Files:**
- Modify: `xcquinox/alec/cluster/_pretrain.py:1-40` (module docstring), no template change
- Modify: `xcquinox/alec/cluster/examples/grid_step7.yaml` (`pretrain_time` comment)
- Modify: `xcquinox/alec/tests/test_cluster_pretrain.py` (append the template-shape pin)
- Modify: `xcquinox/alec/HISTORY.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing importable.

Decision on the sbatch surface, with its justification (the task instruction
asks for a new job kind to be justified or refused): **no new job kind and no
template change.** The certificate must (a) load the checkpoint the pretrain
job just wrote, (b) run PySCF SCFs at the production identity on a node with
the pretrain array's memory, and (c) gate an exit code the train array's
`afterok:<pretrain>` dependency already reads. Running it inside
`_pretrain.main` gives all three for free. A second `python -m ...` line in
`pretrain.sbatch.tmpl` would pay the JAX + PySCF import a second time, would
need its own `set -e` interaction to fail the job, and would still land on the
same node with the same wall clock; a separate SLURM job kind would add a
dependency edge, a submission record and a log family for work that is one
function call. The only real cost is wall time, addressed by the YAML note
below.

- [ ] **Step 1: Write the failing test**

Append to `xcquinox/alec/tests/test_cluster_pretrain.py`:

```python
def test_pretrain_template_invokes_only_the_certifying_worker():
    """The certificate runs INSIDE _pretrain, not as a second command.

    It loads the checkpoint the job just wrote, runs PySCF SCFs at the
    production identity, and gates this job's exit code, which is what the
    train array's afterok dependency reads. A second python invocation would
    pay the JAX/PySCF import twice and would need its own failure semantics
    for that dependency, so the template carries exactly one."""
    text = importlib.resources.files("xcquinox.alec.cluster").joinpath(
        "templates/pretrain.sbatch.tmpl").read_text()
    invocations = [ln.strip() for ln in text.splitlines()
                   if ln.strip().startswith("python -m")]
    assert invocations == [
        "python -m xcquinox.alec.cluster._pretrain "
        "${RUN_DIR} $${SLURM_ARRAY_TASK_ID}"
    ]
```

- [ ] **Step 2: Run it**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_pretrain.py \
  -k template_invokes -v > /tmp/xcq-testlogs/t14-tmpl.log 2>&1; echo "rc=$?"
```
Read the log. Expected: PASS immediately -- this test pins a property the
template already has and that Task 6's design chose to keep. If it fails, the
template was changed and the decision above was not honoured.

- [ ] **Step 3: Record the gate in the `_pretrain` module docstring**

In `xcquinox/alec/cluster/_pretrain.py`, add a bullet to the "For one
architecture index it:" list, after the "Verifies ``xnet.eqx`` + ``cnet.eqx``
landed" bullet (`:26-30`):

```
  - Runs the per-architecture physics certificate
    (:func:`xcquinox.alec.cluster.fidelity.fidelity_certificate`) on this node,
    against PBE for a GGA-rung architecture and SCAN for a meta-GGA one, and
    exits non-zero unless the verdict is PASS. The certificate is run HERE and
    not as a separate job because it needs the checkpoint that was just
    written, this node's memory at the production identity, and this job's
    exit code: the train array is already gated ``afterok`` on the pretrain
    array, so a failed certificate blocks the campaign with no extra
    dependency edge. Budget wall clock for it (order tens of minutes at
    6-311++G(3df,2pd) / grid level 3 over ~40 systems) on top of the
    pretraining itself.
```

- [ ] **Step 4: Note the wall-clock coupling in the example YAML**

In `xcquinox/alec/cluster/examples/grid_step7.yaml`, extend the comment above
`pretrain_time` in the `cluster` block:

```yaml
  # Pretrain runs once per distinct architecture, up front; pretrain_throttle
  # unset means all distinct architectures pretrain concurrently. The pretrain
  # task ALSO runs the per-architecture fidelity certificate on the same node
  # after training the networks (an SCF sweep over the pools' free atoms, the
  # DFS pretraining molecules and H2O/N2/CH4 at the run's identity), so this
  # wall must cover pretraining plus that sweep.
  pretrain_time: "08:00:00"
```

- [ ] **Step 5: Run the whole affected suite**

```bash
python -m pytest \
  xcquinox/alec/tests/test_dfs_pretrain_set.py \
  xcquinox/alec/tests/test_cluster_fidelity.py \
  xcquinox/alec/tests/test_cluster_grid_config.py \
  xcquinox/alec/tests/test_cluster_examples.py \
  xcquinox/alec/tests/test_cluster_cli.py \
  xcquinox/alec/tests/test_cluster_pretrain.py \
  xcquinox/alec/tests/test_cluster_train_task.py \
  xcquinox/alec/tests/test_cluster_preflight.py \
  xcquinox/alec/tests/test_validate_run.py \
  xcquinox/alec/tests/test_train.py \
  notebooks/analysis/test_merge_v4_arms.py \
  notebooks/analysis/test_make_ablation_arch_figure.py \
  -v > /tmp/xcq-testlogs/t14-suite.log 2>&1; echo "rc=$?"
```
Read the log in full. Expected: no failures. Any failure here is a real
regression from an earlier task -- fix it before writing the HISTORY entry.

- [ ] **Step 6: Write the HISTORY entry**

Append to `xcquinox/alec/HISTORY.md`, in the file's existing entry style
(date, short hash placeholder the controller fills at commit time, WHAT, Why):

```markdown
## 2026-08-21 -- Per-architecture pretraining-fidelity certificate and its enforcement

**What:** A new module `xcquinox/alec/cluster/fidelity.py` certifies, per
architecture, that the pretrained exchange and correlation networks reproduce
their parent functional (PBE for a GGA-rung architecture, SCAN for a meta-GGA
one) in energy units. It evaluates `E_xc^NN - E_xc^parent` through the
production energy path on the parent's own self-consistent density at the run's
identity, over every free atom of the BH76 / W4-11 pools, the DFS pretraining
molecules and H2O / N2 / CH4, and folds the molecular differences into
atomization-energy offsets. PASS requires max |dE_xc| over free atoms <= 1.0
mHa and max |dAE| <= 1.0 kcal/mol; the tolerances are configurable
(`fidelity.tol_atom`, `fidelity.tol_AE`) but neither may exceed 2.0 without a
non-empty `fidelity.override_reason`, which is copied into the certificate.
The verdict and every number go to
`<run_dir>/pretrain/<arch>/fidelity_certificate.json`, which also records
`enforced`. Enforcement has two layers. The ON-NODE gates honour
`fidelity.enforce` (False requires an `override_reason`): a
workflow-verification run computes and records the true verdict but is allowed
to proceed, so a deliberately short pretraining run can exercise the train and
eval wiring with the physics on record. The RECORD layers require PASS
unconditionally, so such a run can never become a quantitative result. The
pretrain array task certifies and exits non-zero on anything but PASS; the
train task refuses with a deterministic `fidelity_certificate_missing` /
`fidelity_certificate_failed`; the preflight sweeps every distinct
architecture; `train._build_model` refuses in process (env
`XCQUINOX_ALLOW_UNCERTIFIED=1` opts out); `validate_run` additionally checks
the certificate's identity and code version against the run's; `merge_v4_arms`
refuses an uncertified arm and carries each arm's certificates into the merged
view; the figure suite reports uncertified architectures, refuses to build for
one, stamps a disclosure on pre-gate runs, and renders the certificate's worst
numbers in the provenance footer. The DFS pretraining set moved from an ASE
trajectory outside the repository into committed package data
(`xcquinox/alec/data/dfs_pretrain_set.json`) behind
`xcquinox/alec/dfs_pretrain_set.py`.

**Why:** Nothing in the campaign compared a pretrained network with its parent.
The offsets that had accumulated unmeasured were 2.3 to 56 kcal/mol in
atomization energies depending on architecture (SPEC_pretrain_fidelity_program
Section 2) -- larger than the architecture differences the campaign exists to
resolve, so every arch-to-arch comparison was partly a comparison of
pretraining errors. A tolerance without a gate is a preference; the certificate
is the gate, and it is enforced at every layer that could otherwise let an
uncertified number reach a figure. The certificate also records the installed
`xcquinox_version`, which is what makes it a statement about the code that
actually ran rather than about code in principle.
```

**Deliverable:** the whole affected suite is green, the sbatch surface decision is pinned by a test, and the change is on the development record.
**Covering command:** the Step 5 command.

---

## Self-review against the spec

### Spec 3.3 coverage

| Spec 3.3 clause | Task |
|---|---|
| "Loads the pretrained networks through the production model builder." | 4 (`_build_model` / `build_certified_model`, mirroring `train._build_model`, with the `use_polarized_correlation` patch `_pretrain` applies) |
| "Builds the oracle set: every atom of the pools (all open-shell), the DFS pretraining molecules, and a fixed molecule set spanning the pool's elements" | 2 (`build_oracle_set`), 1 (the DFS set as committed data) |
| "on frozen parent densities at the production identity (PBE for GGA-rung, SCAN for meta-GGA)" | 2 (`resolve_parent`, `run_identity`), 3 (`precompute_fixed_density_data(..., reference_xc=)` -- the ONE construction path), 4 (`evaluate_system` requests it and refuses a record that is not the parent's) |
| "Computes E_xc^NN - E_xc^parent per system (production footing, energy path)" | 4 (`evaluate_system`: `fixed_density_total_energy` minus `E_non_xc`, against libxc on the same stored grid, cross-checked two independent ways) |
| "and the implied atomization-energy offsets" | 4 (`fidelity_certificate`, `per_atomization`) |
| "runs O1-O4 on the installed code" | Out of scope by instruction: O1-O4 are unit tests of the Section 3.1 plan. The certificate records `xcquinox_version` instead, and Task 9 makes a version mismatch against the run's manifest a hard failure -- so a certificate is a statement about the code that actually ran. Stated in Global Constraints. |
| "PASS iff max \|dE_xc\| per atom <= tol_atom and max \|dAE\| <= tol_AE" | 4 (verdict), 5 (tolerances) |
| "Writes `<run_dir>/pretrain/<arch>/fidelity_certificate.json` (inputs, every number, tolerances, code hash, verdict)" | 4. "code hash": `xcquinox_version` is produced by versioningit with the format `{base_version}+{distance}.{vcs}{rev}`, so the version string carries the git revision. |
| "the train task refuses to start without a PASS certificate for its architecture" | 7 |
| "`merge_v4_arms` and the figure loaders refuse a run whose architectures lack one" | 10, 11 |
| "the certificate's table is rendered into the figure provenance footer" | 11 (`fidelity_summary` + `provenance_footer`) |

Beyond the spec's list, because the spec's intent ("no campaign stage may start,
and no result may enter the figure pipeline") is broader than its enumeration:
the pretrain worker itself (Task 6), the preflight sweep (Task 8), the
in-process model builder (Task 12), the run validator (Task 9) and the `status`
signal (Task 13).

### Spec 3.5 and 7 coverage

- 3.5 "Every arm is resubmitted under the corrected code and pretraining (the
  descriptor-free architectures do not meet tol_AE = 1.0 either at 2.3-4.2
  kcal/mol)": this plan does not submit v6 (that is Section 5 step 5), but it
  makes v6 possible and makes the retirement of the pre-v6 record enforceable.
  The gate applies to `sorted(set(cfg.sweep.arch))` with no exemption for the
  descriptor-free architectures (Task 8's
  `test_preflight_checks_every_distinct_arch`), and pre-gate runs acquire the
  `_FIDELITY_DISCLOSURE` footer and are refused by `build_bh76w411_suite`
  (Task 11).
- 7 "tol_AE = 1.0 kcal/mol ... tol_atom = 1.0 mHa, for every architecture; no
  override without `fidelity.override_reason`": Task 5, defaults plus the
  2.0/2.0 ceiling rule, plus the same rule for `enforce: false`.
- 7 "the symmetric doubled density diag(P_sigma, P_sigma) ...": Section 3.1's
  plan, which is landing in this tree now. This plan does not implement it and
  adds no spin-scaling logic of its own: because the parent density is obtained
  through `precompute_fixed_density_data`, the per-spin-channel blocks are
  populated by the SAME open-shell branch that builds them for PBE, with no
  special-casing. Task 3 states the sequencing dependency explicitly (its
  `data.py` edit must land after that commit) and Task 4's
  `test_meta_gga_architecture_is_certified_against_scan` pins the end-to-end
  result.
- 7 "Campaign v6 resubmits every architecture ... the descriptor-free
  architectures are included": as above.
- 7 "Pretraining set: the DFS pretraining set in its entirety, plus every atom
  of the BH76 / W4-11 pools ...": Section 3.2's plan owns the pretraining
  change; Task 1 of this plan owns the shared loader
  (`xcquinox/alec/dfs_pretrain_set.py`) that plan consumes.

### Placeholder scan

Searched the plan for `TBD`, `TODO`, `FIXME`, "implement later", "fill in
details", "add appropriate", "add validation", "handle edge cases", "similar to
Task N", "write tests for the above": the only hits are inside the Global
Constraints sentence that forbids those words in comments. Every step carries
its actual code.

### Name consistency

Certificate JSON keys (`verdict`, `arch`, `parent`, `xcquinox_version`,
`identity`, `tolerances.tol_AE` / `tol_atom` / `override_reason`, `enforced`,
`per_system[].{name,spin,charge,is_atom,n_grid,reference_xc,E_xc_nn,
E_xc_parent,E_xc_parent_numint,E_xc_parent_record,parent_grid_diff_Ha,
parent_record_diff_Ha,dE_xc_mHa,duration_s}`,
`per_atomization[].{name,dAE_kcalmol}`, `summary.{max_atom_mHa,
max_dAE_kcalmol,n_systems,n_atoms,n_atomizations,n_failed_systems,
max_parent_grid_diff_Ha,max_parent_record_diff_Ha,failure_reasons}`,
`timestamp`,
`duration_s`) are spelled identically in Tasks 3, 5, 6, 7, 8, 9, 10, 11, 12 and
in every fixture. `FidelityConfig` field names (`tol_AE`, `tol_atom`,
`override_reason`, `enforce`) are spelled identically in the dataclass, the
builder, the validator, the serializer, the example YAML and every test.
`certificate_status` (record layer) and `gate_certificate` (on-node layer) are
never interchanged: Tasks 5, 6, 7 use the gate, Tasks 8, 9, 10 use the status.

---

## Ambiguities in the spec, and how they were resolved

1. **"on the SAME grid" for the parent energy.** The precompute's grid is the
   SCF-pruned grid (`small_rho_cutoff`), so a freshly built `dft.Grids` of the
   same level has a different point count. Resolved by making the PRIMARY
   parent energy point-wise on the stored precompute grid
   (`_parent_exc_on_stored_grid`), which is grid-exact against the network by
   construction, and keeping PySCF's `nr_rks`/`nr_uks` on a fresh grid as a
   per-system CROSS-CHECK recorded as `parent_grid_diff_Ha` and bounded by
   `PARENT_GRID_TOL_HA = 1e-6`. The two routes agree to 2.6e-11 Ha at sto-3g
   and 2.0e-10 Ha at the production identity (measured; scratch probes
   2026-08-20 and the Task 4 real-physics test).
2. **How the SCAN density enters `mol_data`.** `precompute_fixed_density_data`
   hard-coded a PBE SCF and had no interface for another functional's density.
   An earlier draft of this plan rebuilt the density-carrying entries inside
   `fidelity.py`; that was rejected, correctly, as a second construction that
   would have to mirror `data.py` forever -- eighteen fields, ten of them the
   per-spin blocks the Section 3.1 work had just added. Resolved instead by
   making the functional a PARAMETER of the one construction:
   `precompute_fixed_density_data(..., reference_xc="scan")`, keyed in the memo
   cache and recorded on the record. The certificate then constructs nothing,
   which a source-level guard enforces
   (`test_fidelity_never_rebuilds_a_precompute_field`). `E_non_xc` needs no
   special handling at all now, and in any case cancels identically in
   `fixed_density_total_energy(...) - E_non_xc`.
3. **Which parent for which architecture.** The spec says PBE for GGA-rung and
   SCAN for meta-GGA, while `inputs.seed_xc` also selects a functional. These
   are different things: the parent is a property of the architecture's rung
   (what it was pretrained against), the seed is the run's SCF starting
   density. Resolved by deriving the parent from `rungs.seed_xc_for_arch`
   directly, NOT from `spec_builder.resolve_seed_xc`, with a test asserting the
   two agree for every registered architecture.
4. **"a fixed molecule set spanning the pool's elements."** The union of the
   DFS molecules and H2O/N2/CH4 spans H, C, N, O, F, Na, Al, Si, P, S, Cl and
   Li. Boron and beryllium appear in the pools only as free atoms and in no
   oracle molecule, so their `tol_atom` is checked but no atomization offset
   involves them. Recorded rather than papered over; adding a molecule for
   them would mean inventing a geometry the campaign does not otherwise use.
5. **DFS set level per architecture.** The meta-GGA variant of the DFS protocol
   omits H2 and N2. Resolved by certifying each architecture on the level it
   was pretrained on (`dfs_level_for_parent`), while the unconditional fixed
   set restores N2 -- so every architecture, whatever its rung, is measured on
   a common N2 / H2O / CH4 core and the cross-rung numbers stay comparable.
6. **"the certificate's table" in the figure footer.** A 40-row table cannot be
   rendered in a figure footer. Resolved by rendering the worst numbers -- the
   largest free-atom `|dE_xc|` in mHa and the largest `|dAE|` in kcal/mol over
   the run's architectures -- which is what bounds every number on the figure.
   The full table stays in the certificate JSON beside the run.
7. **SCF convergence and reproducibility of the oracle set.** The reference SCF
   now belongs entirely to `precompute_fixed_density_data`, which reports no
   convergence flag, so the per-system `scf_converged` slot was dropped from
   the certificate schema rather than recorded as a permanent `null`. What
   replaced it is stronger: three independent routes to `E_xc^parent` on the
   record's own density (point-wise on the stored grid, `nr_rks`/`nr_uks` on a
   fresh grid, and the XC energy the reference SCF itself accumulated), whose
   pairwise differences are recorded per system and bounded at
   `PARENT_GRID_TOL_HA`. A density that did not converge shows up there and in
   an absurd `dE_xc`, which is what the verdict acts on. Separately, measured
   on this machine: degenerate free atoms (O, C, F) converge to different
   orientations of their open shell run to run, moving `rho_grid` by 0.2-0.6
   point-wise -- but their `E_xc` by at most 2.2e-3 mHa, some 450 times inside
   `tol_atom = 1.0 mHa`. The tolerance is therefore meaningful; the
   orientation lock in the run identity is what keeps it tight.
8. **`xcquinox_version` as "code hash".** versioningit formats the version as
   `{base_version}+{distance}.{vcs}{rev}`, so the recorded version string
   carries the git revision. `validate_run` compares it against the manifest's,
   which is what makes the certificate a statement about the code that ran.
9. **The import-weight claim.** An earlier draft of this plan asserted that
   importing `cluster.fidelity` imports no jax. That is false: importing any
   `xcquinox.alec.cluster` module executes `xcquinox/__init__.py`, which
   imports `xcquinox.net`, which imports jax. Resolved by restating the
   contract as a property of `fidelity.py`'s own module body and checking it
   with an AST test on the source (Task 2), which is both true and the thing
   that actually matters: a certificate reader must not trigger a model import
   or an SCF stack from this file.
10. **`fidelity.enforce`.** Introduced by an interface decision from the
    controlling plan, not by the spec text. Reconciled with the spec by
    splitting enforcement into an ON-NODE layer that may be waived (with a
    recorded reason) and a RECORD layer that may not, so the spec's binding
    sentence -- "no result may enter the figure pipeline" without a
    certificate -- holds unconditionally while the Section 3.4 workflow matrix
    can still exercise the wiring end to end.
11. **Which geometry for H2O, N2 and CH4.** The spec calls for "a fixed
    molecule set spanning the pool's elements" without naming a source. Task 2
    took the three from a literal table and let a DFS record of the same name
    win, which made them rung-dependent: a GGA-rung architecture was certified
    on the DFS N2 (r = 1.0987920 A) while a meta-GGA one, whose DFS level drops
    N2, fell through to the literal. Resolved by resolving all three from the
    BH76 / W4-11 pools (`_FIXED_MOLECULE_POOL_NAMES`) for every rung,
    overriding any DFS record of the same name, so `dAE(H2O)`, `dAE(N2)` and
    `dAE(CH4)` -- the three numbers spec Section 2 tabulates -- are the same
    physical quantity for every architecture and are measured on the geometries
    the held-out atomization energies are scored on. The trade-off, stated
    rather than hidden: N2 and CH4 are then certified at the pool geometry
    rather than the geometry their DFS pretraining rows were generated at
    (r(CH) 1.0874456 A against 1.0918537 A). The other eighteen-plus DFS
    molecules keep their own geometries, which a test pins.

12. **A literal byte-identity round-trip test for `reference_xc="pbe"`.** The
    instruction was to pin the default by "comparing every key of
    `precompute_fixed_density_data(m)` with and without `reference_xc="pbe"`".
    Measured on this machine, two INDEPENDENT SCF runs of the same system are
    not bitwise equal (closed shell: ~5e-14 Ha in energy, ~5e-8 in the
    meta-GGA alpha; degenerate open shell: a different orientation of the
    singly occupied shell entirely), so a bitwise comparison of two separate
    calls is a test no correct implementation could pass. Resolved by pinning
    what is actually pinnable and equally binding: (a) an OBJECT-IDENTITY test
    through the memo cache -- the default and the explicit `"pbe"` are one
    cache entry and one SCF, so no consumer silently gets a second reference
    calculation; (b) byte-identity BY CONSTRUCTION -- the only edits on the
    default path are `mf.xc = reference_xc` with default `"pbe"` and an
    `xc_type`-dispatched branch whose LDA/GGA arm is the original expression
    verbatim; and (c) the whole existing consumer suite
    (`test_data`, `test_data_cderi`, `test_shape_padding`,
    `test_solv01_split_xc`, `test_descriptors`) re-run and required to be
    unchanged. Task 3 Step 10 carries this, with the reasoning in the test
    file so the next reader does not re-propose the impossible version.


