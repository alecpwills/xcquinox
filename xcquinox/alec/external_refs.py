"""Step-7 CCSD reference pre-compute pipeline.

Generates per-species external_data_path .npz files containing CCSD
reference density matrix, grid density, and OEP-inverted V_xc for the
union of training + held-out probe + HBPT species.

Pipeline stages (each individually cached via np.savez_compressed):
  1. SCF  → _intermediates/<name>_g{grid_level}_scf.npz   (MO coeffs, DM, S)
  2. CCSD → _intermediates/<name>_g{grid_level}_ccsd.npz  (CC density matrix + rho)
  3. OEP  → <name>.npz                                    (vxc_ref + dm_target +
                                                          rho_ref_grid + provenance)

Cache layout was changed in 2026-05-03 (spec sec. 5.6) to bake
grid_level into intermediate filenames; legacy unsuffixed names are
migrated by `_migrate_intermediates_to_grid_suffixed` invoked from the
top of both `precompute_all` and `preflight_uks_oep`.

Reuses step-6 cells 12-13 OEP-cascade pattern verbatim
(_build_step6_notebook.py:728-768, 843-877). 2-tier: svp-jkfit primary
(reg=1e-4, conv_tol=2e-3, max_iter=500), def2-tzvp-jkfit fallback
(reg=1e-4, conv_tol=2e-3, max_iter=1000). On both-tier failure raises
RuntimeError.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciesEntry:
    """Canonical pre-compute species record.

    Dedup key is (name, charge, spin) — `Li` and `Li+` are distinct
    entries with different charges.
    """
    name: str
    charge: int
    spin: int  # PySCF 2S = N_α − N_β convention
    source: str  # one of "dfs_ae", "dfs_atom", "bh76", "ip13",
                 # "probe_a", "probe_b", "probe_c", "probe_d",
                 # "probe_atom_ref", "hbpt"


def _fsync_dir(dir_path) -> None:
    """fsync a directory entry for durability after an atomic os.replace.

    POSIX rename is atomic per-file but the rename is not durable across a
    power loss until the *parent directory* is fsync'd. Only catch the
    AttributeError from a missing O_DIRECTORY (Windows); let real OSErrors
    (ENOSPC, EIO) bubble so durability failures fail loudly (matches the
    `_migrate_intermediates_to_grid_suffixed` policy).
    """
    import os
    if not hasattr(os, "O_DIRECTORY"):
        return
    dir_fd = os.open(str(dir_path), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _fsync_file(path) -> None:
    """fsync a file's CONTENTS to disk (EXTREF-04).

    ``os.replace`` makes the rename atomic, but the renamed file's *contents*
    are not durable across power loss unless the data was fsync'd first — a
    crash between write and the OS flush can leave the cache path pointing at a
    zero-length/partial file. Call this on the temp file BEFORE ``os.replace``.
    Real OSErrors (ENOSPC, EIO) bubble so durability failures fail loudly,
    matching :func:`_fsync_dir`.
    """
    import os
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _build_hf_meanfield(mol, is_uks: bool):
    """Construct an (unconverged) HF mean-field object for ``mol``.

    Factored out so EXTREF-01's "real HF SCF before CCSD" contract is
    unit-testable (tests monkeypatch this to inject a stub mean-field
    without running PySCF).
    """
    from pyscf import scf
    return scf.UHF(mol) if is_uks else scf.RHF(mol)


def _prepare_converged_hf(mol, *, dm0, is_uks: bool):
    """Run a real HF SCF and return the CONVERGED HF mean-field.

    EXTREF-01: CCSD must sit on a self-consistent HF determinant. Earlier
    code grafted PBE Kohn-Sham MO coeff/occ onto an HF object and faked
    ``converged=True`` without re-converging — that runs CCSD on a
    non-canonical, non-self-consistent PBE determinant (Brillouin's
    theorem is violated: the occ-virt Fock block is nonzero), and the
    relaxed 1-RDM then depends on arbitrary PBE start orbitals.

    The PBE DM is used ONLY as the SCF initial guess (``kernel(dm0=...)``)
    to speed convergence; the orbitals CCSD sees are the converged HF
    orbitals. Raises ``RuntimeError`` if HF does not converge so a
    non-self-consistent reference can never silently reach CCSD.

    Sources: single-reference coupled-cluster theory is formulated on a
    converged (canonical) Hartree-Fock reference for which Brillouin's
    theorem (the occupied-virtual block of the Fock matrix vanishes) holds —
    Szabo & Ostlund, *Modern Quantum Chemistry* (Dover, 1996) §3.3, and
    T. D. Crawford & H. F. Schaefer III, "An Introduction to Coupled Cluster
    Theory for Computational Chemists," *Rev. Comput. Chem.* **14**, 33-136
    (2000); see also R. J. Bartlett & M. Musiał, *Rev. Mod. Phys.* **79**, 291
    (2007). Running CCSD on the non-self-consistent PBE determinant leaves a
    nonzero f_ov block and makes the relaxed 1-RDM depend on the arbitrary PBE
    starting orbitals — which is what fed bias into the ML training target.
    """
    mf_hf = _build_hf_meanfield(mol, is_uks)
    mf_hf.kernel(dm0=dm0)
    if not getattr(mf_hf, "converged", False):
        raise RuntimeError(
            "HF SCF did not converge; refusing to run CCSD on a "
            "non-self-consistent reference determinant"
        )
    return mf_hf


def build_species_union() -> list[SpeciesEntry]:
    """Assemble the canonical species set requiring CCSD references.

    Iterates DFS pool, BH76 reactions, IP13 pairs, atom refs (training +
    probe-induced), Probe A/B/C/D, and HBPT pairs.  De-duplicates on
    (name, charge, spin).  Returns a deterministic list (sorted by name
    then charge then spin) so the iteration order is reproducible across
    runs.
    """
    from xcquinox.alec import dfs_pool, eval_probes
    from xcquinox.alec.subset_selection import _make_hb_atoms, _make_pt_atoms
    seen: dict[tuple[str, int, int], SpeciesEntry] = {}

    def _add(name: str, charge: int, spin: int, source: str) -> None:
        key = (name, charge, spin)
        if key not in seen:
            seen[key] = SpeciesEntry(
                name=name, charge=charge, spin=spin, source=source,
            )

    # DFS AE molecules
    pool = dfs_pool.build_dfs_pool()
    for at in pool["ae_molecules"]:
        _add(
            at.info["dfs_hill"],
            int(at.info.get("charge", 0)),
            int(at.info["spin"]),
            "dfs_ae",
        )
    # DFS atom refs (H, Li)
    for at in pool["atom_refs"]:
        sym = at.info["name"]
        _add(sym, int(at.info.get("charge", 0)), int(at.info["spin"]),
             "dfs_atom")

    # BH76 species (need atom-level dispatch)
    for rxn in pool["bh76_reactions"]:
        spins = rxn.get("species_spins", {})
        charges = rxn.get("species_charges", {})
        for sp in (*rxn["reactants"], *rxn["products"]):
            _add(sp, int(charges.get(sp, 0)), int(spins.get(sp, 0)), "bh76")

    # IP13 neutrals + cations
    for pair in pool["ip13_pairs"]:
        _add(pair["neutral"], int(pair["neutral_charge"]),
             int(pair["neutral_spin"]), "ip13")
        _add(pair["cation"], int(pair["cation_charge"]),
             int(pair["cation_spin"]), "ip13")

    # Probe sets — read entries from eval_probes
    for probe_name in eval_probes.ALL_PROBES:
        kind = eval_probes.PROBE_KIND[probe_name]
        entries = eval_probes.ALL_PROBES[probe_name]
        if kind == "ae":
            for entry in entries:
                _add(entry["hill"], int(entry.get("charge", 0)),
                     int(entry["spin"]), f"probe_{probe_name.split('_')[1]}")
        else:  # bh76
            for rxn in entries:
                spins = rxn.get("species_spins", {})
                charges = rxn.get("species_charges", {})
                for sp in (*rxn["reactants"], *rxn["products"]):
                    _add(sp, int(charges.get(sp, 0)),
                         int(spins.get(sp, 0)), f"probe_{probe_name.split('_')[1]}")

    # Probe-induced atom refs (S, Cl, P, Si, Be) for atom_energies anchor
    from xcquinox.alec.dfs_pool import ATOMIC_GROUND_STATE_SPIN
    for sym in ("S", "Cl", "P", "Si", "Be"):
        _add(sym, 0, ATOMIC_GROUND_STATE_SPIN[sym], "probe_atom_ref")

    # HBPT water-dimer pairs
    hb = _make_hb_atoms()
    pt = _make_pt_atoms()
    _add(hb.info["dfs_hill"], int(hb.info["charge"]),
         int(hb.info["spin"]), "hbpt")
    _add(pt.info["dfs_hill"], int(pt.info["charge"]),
         int(pt.info["spin"]), "hbpt")

    return sorted(seen.values(), key=lambda s: (s.name, s.charge, s.spin))


def resolve_geometry(spec: SpeciesEntry):
    """Build an ASE Atoms object for a SpeciesEntry.

    Strategy by source:
      - dfs_ae: lookup by Hill formula in g2_97.traj
      - dfs_atom / bh76 (single-letter or two-letter symbol):
        bare atom at origin
      - ip13: bare atom (cation = bare atom with charge+1)
      - probe_a / probe_b / probe_d (compound):
        lookup by Hill formula in g2_97.traj OR pull from
        eval_probes.build_probe_pool's output entries
      - probe_c (BH76 species): same dispatch as bh76
      - hbpt: call _make_hb_atoms / _make_pt_atoms
    """
    from ase import Atoms
    from xcquinox.alec.dfs_pool import _g297_traj_path
    from xcquinox.alec.subset_selection import _make_hb_atoms, _make_pt_atoms
    from ase.io import read as ase_read

    if spec.source == "hbpt":
        atoms = _make_hb_atoms() if spec.name == "HBWD" else _make_pt_atoms()
        return atoms

    # Atomic species: name is a single chemical symbol (or symbol+"+" for
    # cations). Use ase.data.chemical_symbols as the authoritative element
    # list — the prior `len(sym) <= 2 and sym.isalpha()` check incorrectly
    # treated diatomic Hill formulas like "HF", "HS", "NO" as single atoms
    # (they're 2 chars and alphabetic but NOT elements), causing
    # `Atoms(sym, positions=[(0,0,0)])` to crash with
    # "positions wrong length: 1 != 2" since ASE expands "HF" to 2 atoms.
    from ase.data import chemical_symbols
    sym = spec.name.rstrip("+")
    if sym in chemical_symbols and spec.source in (
        "dfs_atom", "bh76", "ip13", "probe_atom_ref", "probe_c",
    ):
        atoms = Atoms(sym, positions=[(0.0, 0.0, 0.0)])
        atoms.info["name"] = spec.name
        atoms.info["charge"] = spec.charge
        atoms.info["spin"] = spec.spin
        return atoms

    # Compound species: try g2_97.traj first
    traj = ase_read(str(_g297_traj_path()), ":")
    by_hill = {a.get_chemical_formula(): a for a in traj}
    if spec.name in by_hill:
        atoms = by_hill[spec.name].copy()
        atoms.info["dfs_hill"] = spec.name
        atoms.info["charge"] = spec.charge
        atoms.info["spin"] = spec.spin
        return atoms

    # Probe species not in g2_97: pull from eval_probes.build_probe_pool.
    # ``pool["entries"]`` is list[dict] (raw PROBE_* entries); ``pool["molecules"]``
    # is the corresponding list[ASE Atoms] with at.info["name"] set by
    # eval_probes._attach_info.  Match by the dict's "name" against
    # at.info["name"] (T2 spec-review fix — earlier draft iterated entries
    # as if they were Atoms, which is unreachable today but would crash
    # if a probe AE molecule were ever absent from g2_97.traj).
    from xcquinox.alec import eval_probes
    for probe_name in eval_probes.ALL_PROBES:
        if eval_probes.PROBE_KIND[probe_name] != "ae":
            continue
        for entry in eval_probes.ALL_PROBES[probe_name]:
            if entry["hill"] == spec.name:
                pool = eval_probes.build_probe_pool(probe_name)
                for at in pool["molecules"]:
                    if at.info.get("name") == entry["name"]:
                        a = at.copy()
                        a.info["charge"] = spec.charge
                        a.info["spin"] = spec.spin
                        return a
    raise KeyError(
        f"Could not resolve geometry for SpeciesEntry(name={spec.name!r}, "
        f"charge={spec.charge}, spin={spec.spin}, source={spec.source!r})"
    )


def run_scf_with_cache(
    spec: SpeciesEntry,
    atoms,
    *,
    cache_dir,
    basis: str = "def2-svp",
    grid_level: int = 1,
) -> dict:
    """Stage 1: PBE SCF with on-disk cache (np.savez_compressed).

    Returns dict with keys: dm, mo_coeff, mo_occ, mo_energy, S,
    spin_unrestricted, n_ao, n_grid.

    Cache layout:
      <cache_dir>/_intermediates/<name>_g{grid_level}_scf.npz
    (Spec sec. 5.6: the `_g{N}_` infix lets the same `cache_dir` host
    multiple grid_levels of the same species without collision; legacy
    pre-2026-05-03 caches are migrated by
    `_migrate_intermediates_to_grid_suffixed` at top of `precompute_all`
    / `preflight_uks_oep`.)
    """
    import numpy as np
    from pathlib import Path
    from pyscf import dft, gto

    inter = Path(cache_dir) / "_intermediates"
    inter.mkdir(parents=True, exist_ok=True)
    cache_path = inter / f"{spec.name}_g{int(grid_level)}_scf.npz"

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as z:
            return {
                "dm": np.asarray(z["dm"]),
                "mo_coeff": np.asarray(z["mo_coeff"]),
                "mo_occ": np.asarray(z["mo_occ"]),
                "mo_energy": np.asarray(z["mo_energy"]),
                "S": np.asarray(z["S"]),
                "spin_unrestricted": bool(z["spin_unrestricted"]),
                "n_ao": int(z["n_ao"]),
                "n_grid": int(z["n_grid"]),
                "grid_coords": np.asarray(z["grid_coords"]),
                "grid_weights": np.asarray(z["grid_weights"]),
            }

    coords = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    atom_lines = [(s, tuple(coords[i])) for i, s in enumerate(syms)]
    mol = gto.M(atom=atom_lines, basis=basis, charge=spec.charge,
                spin=spec.spin, unit="angstrom", verbose=0)

    is_uks = spec.spin > 0
    mf = dft.UKS(mol) if is_uks else dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = grid_level
    mf.kernel()

    # Build the result dict ONCE — used both for the cache write and the
    # return value.  Avoids redundant PySCF calls (make_rdm1/get_ovlp
    # were called twice in the earlier draft) and removes a DRY violation
    # (T3 code-quality review).
    # grid_coords/grid_weights are stored so Stage 2 (CCSD) can reuse the
    # exact pruned grid from the SCF run without rebuilding (which would
    # give a different grid size due to pruning).
    result = {
        "dm": np.asarray(mf.make_rdm1()),
        "mo_coeff": np.asarray(mf.mo_coeff),
        "mo_occ": np.asarray(mf.mo_occ),
        "mo_energy": np.asarray(mf.mo_energy),
        "S": np.asarray(mf.get_ovlp()),
        "spin_unrestricted": bool(is_uks),
        "n_ao": int(mol.nao),
        "n_grid": int(mf.grids.weights.size),
        "grid_coords": np.asarray(mf.grids.coords),
        "grid_weights": np.asarray(mf.grids.weights),
    }

    # Atomic write: temp file + os.replace so an interrupted SCF cannot
    # leave a corrupt partial .npz that future runs read as a cache hit
    # (T3 code-quality review).
    import os
    import tempfile
    fd, tmp_name = tempfile.mkstemp(dir=str(inter), suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **result)
        # EXTREF-04: fsync content THEN dir so stages 2/3 ("parity with stage 1")
        # are accurate — stage 1 previously had neither.
        _fsync_file(tmp_name)
        os.replace(tmp_name, cache_path)
        _fsync_dir(inter)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return result


def run_ccsd_with_cache(
    spec: SpeciesEntry,
    atoms,
    *,
    scf_payload: dict,
    cache_dir,
    basis: str = "def2-svp",
    grid_level: int = 1,
) -> dict:
    """Stage 2: CCSD on a converged HF reference + spin-summed grid
    density, with on-disk cache.

    CCSD is run on a CONVERGED Hartree-Fock determinant (EXTREF-01): a
    real HF SCF is performed using the cached PBE density matrix only as
    the initial guess (``kernel(dm0=pbe_dm)``), then convergence is
    verified before CCSD. This yields canonical CCSD@HF orbitals — NOT
    CCSD on grafted, non-self-consistent PBE Kohn-Sham orbitals (which
    would violate Brillouin's theorem and bias the relaxed 1-RDM toward
    the arbitrary PBE start orbitals).

    Returns dict with keys: dm_ao, rho_ref_grid (1D spin-summed),
    grid_weights, ao_grid.

    The rho_ref_grid spin-summing is REQUIRED for UKS species — the
    data.py loader expects shape (N_grid,), NOT (2, N_grid). See
    xcquinox/alec/data.py:296-299 for the canonical spin-summing
    pattern (`dm_pbe_tot = dm_pbe[0] + dm_pbe[1]` then einsum).

    Cache layout:
      <cache_dir>/_intermediates/<name>_g{grid_level}_ccsd.npz  (np.savez_compressed)
    (Spec sec. 5.6: the `_g{N}_` infix lets the same `cache_dir` host
    multiple grid_levels of the same species without collision; legacy
    pre-2026-05-03 caches are migrated by
    `_migrate_intermediates_to_grid_suffixed` at top of `precompute_all`
    / `preflight_uks_oep`.)
    """
    import numpy as np
    from pathlib import Path
    from pyscf import dft, gto

    inter = Path(cache_dir) / "_intermediates"
    inter.mkdir(parents=True, exist_ok=True)
    cache_path = inter / f"{spec.name}_g{int(grid_level)}_ccsd.npz"

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as z:
            return {
                "dm_ao": np.asarray(z["dm_ao"]),
                "rho_ref_grid": np.asarray(z["rho_ref_grid"]),
                "grid_weights": np.asarray(z["grid_weights"]),
                "ao_grid": np.asarray(z["ao_grid"]),
            }

    # Build mol for AO evaluation; grid coords/weights are taken directly
    # from the SCF payload so the CCSD grid is identical to the SCF grid
    # (PySCF prunes the grid during kernel(), so rebuilding from scratch
    # yields a different number of points).
    coords = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    atom_lines = [(s, tuple(coords[i])) for i, s in enumerate(syms)]
    mol = gto.M(atom=atom_lines, basis=basis, charge=spec.charge,
                spin=spec.spin, unit="angstrom", verbose=0)
    is_uks = bool(scf_payload["spin_unrestricted"])

    # EXTREF-01: run a REAL HF SCF and converge it before CCSD. The cached
    # PBE density matrix is used ONLY as the initial guess to speed
    # convergence; CCSD then sits on the canonical, self-consistent HF
    # determinant (not on grafted PBE Kohn-Sham orbitals, which would
    # violate Brillouin's theorem and bias the relaxed 1-RDM).
    mf_hf = _prepare_converged_hf(
        mol, dm0=np.asarray(scf_payload["dm"]), is_uks=is_uks
    )

    if is_uks:
        from pyscf.cc import uccsd
        mycc = uccsd.UCCSD(mf_hf)
    else:
        from pyscf.cc import ccsd
        mycc = ccsd.RCCSD(mf_hf)
    mycc.kernel()
    dm_cc = np.asarray(mycc.make_rdm1(ao_repr=True))

    # Spin-sum the AO-basis DM for grid evaluation.  The unrestricted DM
    # may be (2, n_ao, n_ao); we keep both spin channels in dm_ao for
    # the V_xc shape contract but build a SCALAR grid density via the
    # spin-summed total (data.py:296-299 pattern).
    if is_uks and dm_cc.ndim == 3:
        dm_total = dm_cc[0] + dm_cc[1]
    else:
        dm_total = dm_cc

    # Reuse the exact SCF grid (pruned during kernel()) to keep n_grid consistent.
    grid_coords = scf_payload["grid_coords"]
    grid_weights = scf_payload["grid_weights"]
    ao_grid = dft.numint.eval_ao(mol, grid_coords, deriv=0)
    rho_ref_grid = np.einsum("ij,gj,gi->g", dm_total, ao_grid, ao_grid)

    result = {
        "dm_ao": dm_cc,
        "rho_ref_grid": rho_ref_grid,
        "grid_weights": grid_weights,
        "ao_grid": ao_grid,
    }
    # Atomic write (matches T3 pattern): temp file + os.replace.
    import os
    import tempfile
    fd, tmp_name = tempfile.mkstemp(dir=str(inter), suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **result)
        # EXTREF-04: fsync content THEN dir (durability parity with stage 1).
        _fsync_file(tmp_name)
        os.replace(tmp_name, cache_path)
        _fsync_dir(inter)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return result


# OEP cascade tiers — split RKS vs UKS because the achievable density_error
# floor depends on the inner-SCF level shift.
#
# RKS (closed-shell): mirrors step-6 _build_step6_notebook.py:729-730,
# 844-845. conv_tol=2e-3 is tuned against the achievable floor for
# def2-svp/grid_level=1 (~1.17e-3 on H2O/C2H2); gives ~1.7x margin
# (step-6 cell 12). 2-tier cascade verified by R-A in Round-1 review.
_OEP_TIERS_RKS: tuple[dict, ...] = (
    {"aux_basis": "def2-svp-jkfit",  "regularization": 1e-4,
     "max_iter": 500,  "conv_tol": 2e-3},
    {"aux_basis": "def2-tzvp-jkfit", "regularization": 1e-4,
     "max_iter": 1000, "conv_tol": 2e-3},
)

# UKS (open-shell): level_shift=0.5 on the inner SCF (set in run_oep_cascade
# below) suppresses basin-hopping for X²Π / near-degenerate radicals but
# slightly biases the converged inner DM relative to the unshifted minimum
# — bias is small in energy (~mHa) but lifts the density-L2 floor to
# ~6e-3 on HO at def2-svp/grid_level=1. conv_tol=1e-2 gives ~1.7x margin
# above that empirical floor (parity with the RKS margin policy) and
# matches the UKS-acceptable threshold established in
# xcquinox/alec/tests/test_oep_uks.py (which accepts density_error < 0.1
# for Li/sto-3g, calling 6e-3-class results "real progress, not full
# convergence"). Verified empirically on HO 2026-05-02: L-BFGS plateaus
# at ~6e-3 by iter 5 and oscillates 6.1e-3..8.3e-3 thereafter.
_OEP_TIERS_UKS: tuple[dict, ...] = (
    {"aux_basis": "def2-svp-jkfit",  "regularization": 1e-4,
     "max_iter": 500,  "conv_tol": 1e-2},
    {"aux_basis": "def2-tzvp-jkfit", "regularization": 1e-4,
     "max_iter": 1000, "conv_tol": 1e-2},
)

# Required keys in the per-species cache npz; checked by both
# run_oep_cascade's recover-corrupt-cache path and precompute_all's
# skip-if-cached predicate. Keep these two sites in lockstep.
_REQUIRED_NPZ_KEYS: frozenset[str] = frozenset({
    "vxc_ref", "dm_target", "rho_ref_grid", "ref_density_method",
})


# Per-species OEP cascade overrides — populated by the verifier in
# scripts/oep_per_species_emit_overrides.py after the harness sweep.
# Key: (name, charge, spin) tuple matching SpeciesEntry fields.
# Value: tuple of override-tier dicts; each MERGES onto the
# corresponding default per-spin tier (or the last default tier when
# the override has more tiers than the default; see
# `_resolve_tiers_for_species` below).
#
# Override-tier dicts may carry any subset of the keys in
# `_OVERRIDE_TIER_KNOB_ALLOWLIST` below. Per spec sec. 5.1 / 5.2.
_PER_SPECIES_OEP_OVERRIDES: dict[tuple[str, int, int], tuple[dict, ...]] = {
    # ── AUTO-GENERATED by scripts/oep_per_species_emit_overrides.py
    # ── Source: reports_local/oep_tune/2026-05-06/summary.json
    # ── Citations [oep-tdl-1..6] resolve to TO-DOWNLOAD entries in
    #    reports_local/latex/references.bib — AUTHOR-RECALLED, UNVERIFIED.
    #    Verify each via WebFetch + pdftotext before paper write-up.

    # Be winner: density_error_min=4.63e-03, n_iter=3, wall=0.8s
    # Tune log: reports_local/oep_tune/2026-05-06/Be.jsonl trial_idx=1
    ("Be", 0, 0): (
        {
            "aux_basis": "def2-svp-jkfit",
            "grid_level": 1,
            "regularization": 0.001,
            "conv_tol": 0.0079,  # 1.7 * density_error_min
        },
    ),
    # C+ winner: density_error_min=1.40e-02, n_iter=3, wall=1.3s
    # Tune log: reports_local/oep_tune/2026-05-06/C+.jsonl trial_idx=0
    ("C+", 1, 1): (
        {
            "aux_basis": "def2-tzvp-jkfit",
            "grid_level": 1,
            "inner_damp": 0.1,
            "level_shift": 0.5,
            "conv_tol": 0.024,  # 1.7 * density_error_min
        },
    ),
    # F2 winner: density_error_min=9.43e-03, n_iter=10, wall=4.9s
    # Tune log: reports_local/oep_tune/2026-05-06/F2.jsonl trial_idx=0
    ("F2", 0, 0): (
        {
            "aux_basis": "def2-svp-jkfit",
            "grid_level": 1,
            "regularization": 0.0001,
            "conv_tol": 0.016,  # 1.7 * density_error_min
        },
    ),
    # F2O winner: density_error_min=4.84e-03, n_iter=53, wall=409.6s
    # Tune log: reports_local/oep_tune/2026-05-06/F2O.jsonl trial_idx=0
    ("F2O", 0, 0): (
        {
            "aux_basis": "def2-tzvp-jkfit",
            "grid_level": 1,
            "regularization": 0.001,
            "conv_tol": 0.0082,  # 1.7 * density_error_min
        },
    ),
    # HF winner: density_error_min=4.13e-03, n_iter=29, wall=5.7s
    # Tune log: reports_local/oep_tune/2026-05-06/HF.jsonl trial_idx=0
    ("HF", 0, 0): (
        {
            "aux_basis": "def2-svp-jkfit",
            "regularization": 0.0001,
            "conv_tol": 0.007,  # 1.7 * density_error_min
        },
    ),
    # HS winner: density_error_min=1.19e-02, n_iter=2, wall=2.5s
    # Tune log: reports_local/oep_tune/2026-05-06/HS.jsonl trial_idx=0
    ("HS", 0, 1): (
        {
            "aux_basis": "def2-tzvp-jkfit",
            "inner_damp": 0.1,
            "level_shift": 0.5,
            "conv_tol": 0.02,  # 1.7 * density_error_min
        },
    ),
    # N2O winner: density_error_min=4.70e-03, n_iter=12, wall=91.9s
    # Tune log: reports_local/oep_tune/2026-05-06/N2O.jsonl trial_idx=0
    ("N2O", 0, 0): (
        {
            "aux_basis": "def2-tzvp-jkfit",
            "grid_level": 1,
            "regularization": 0.001,
            "conv_tol": 0.008,  # 1.7 * density_error_min
        },
    ),
    # O3 winner: density_error_min=9.22e-03, n_iter=9, wall=89.6s
    # Tune log: reports_local/oep_tune/2026-05-06/O3.jsonl trial_idx=0
    ("O3", 0, 0): (
        {
            "aux_basis": "def2-tzvp-jkfit",
            "grid_level": 1,
            "regularization": 0.001,
            "conv_tol": 0.016,  # 1.7 * density_error_min
        },
    ),
}


# Closed set of recognized override-tier knob names (spec sec. 5.2).
# `_validate_overrides` rejects any override-tier dict containing keys
# outside this allowlist (catches typos like `aux_bais` that would
# otherwise silently no-op via the merge-then-tier.get pattern).
_OVERRIDE_TIER_KNOB_ALLOWLIST: frozenset[str] = frozenset({
    "aux_basis",                     # str
    "regularization",                # float, > 0
    "max_iter",                      # int, >= 1
    "conv_tol",                      # float, > 0
    "grid_level",                    # int, >= 0
    "level_shift",                   # float, |x| <= 5 (Pass-7)
    "inner_damp",                    # float, in [0, 1)
    "inner_diis_start_cycle",        # int, >= 1
})


def _validate_overrides(species_union: list[SpeciesEntry]) -> None:
    """Sanity-check the populated _PER_SPECIES_OEP_OVERRIDES.

    Raises ValueError on any violation. Per spec sec. 5.2 (Pass-8 pin),
    canonical call site is `precompute_all` immediately after
    `build_species_union()` is computed for the run, BEFORE any
    cache-dir migration or preflight. Module import does NOT call this
    (avoids brittling pytest collection on test-mutated dicts) and the
    harness does NOT call it (covered transitively by precompute_all).
    Tests bypassing precompute_all may import this helper directly.

    Validation rules:
    1. Every key is a 3-tuple ``(str, int, int)``; bool excluded.
    2. Every key matches a SpeciesEntry in `species_union`.
    3. Every value is a non-empty tuple of dicts.
    4. Every dict's keys lie within `_OVERRIDE_TIER_KNOB_ALLOWLIST`.
    5. Per-knob bounds: regularization>0, max_iter>=1, conv_tol>0,
       grid_level>=0, inner_damp in [0,1), inner_diis_start_cycle>=1,
       |level_shift|<=5 (Pass-7: negatives allowed; Ziegler-VSO).
    """
    valid_keys = {(s.name, s.charge, s.spin) for s in species_union}
    for key, ovr_tiers in _PER_SPECIES_OEP_OVERRIDES.items():
        # 1. Type-shape check on the key
        if (not isinstance(key, tuple) or len(key) != 3
                or not isinstance(key[0], str)
                or not isinstance(key[1], int) or isinstance(key[1], bool)
                or not isinstance(key[2], int) or isinstance(key[2], bool)):
            raise ValueError(
                f"override key {key!r} must be (str, int, int)"
            )
        # 2. Species existence
        if key not in valid_keys:
            raise ValueError(
                f"override key {key} does not match any species in "
                f"build_species_union(); orphan override"
            )
        # 3. Tier list shape
        if not isinstance(ovr_tiers, tuple) or len(ovr_tiers) == 0:
            raise ValueError(
                f"override for {key}: tier list must be non-empty tuple"
            )
        # 4. Per-tier dict + key allowlist
        for i, tier in enumerate(ovr_tiers):
            if not isinstance(tier, dict):
                raise ValueError(
                    f"override for {key} tier {i}: must be dict"
                )
            unknown = set(tier) - _OVERRIDE_TIER_KNOB_ALLOWLIST
            if unknown:
                raise ValueError(
                    f"override for {key} tier {i}: unknown knobs "
                    f"{sorted(unknown)}; allowed: "
                    f"{sorted(_OVERRIDE_TIER_KNOB_ALLOWLIST)}"
                )
            # 5. Per-knob bounds
            if "regularization" in tier and not (tier["regularization"] > 0):
                raise ValueError(
                    f"override for {key} tier {i}: regularization must be > 0"
                )
            if "max_iter" in tier and not (tier["max_iter"] >= 1):
                raise ValueError(
                    f"override for {key} tier {i}: max_iter must be >= 1"
                )
            if "conv_tol" in tier and not (tier["conv_tol"] > 0):
                raise ValueError(
                    f"override for {key} tier {i}: conv_tol must be > 0"
                )
            if "grid_level" in tier and not (tier["grid_level"] >= 0):
                raise ValueError(
                    f"override for {key} tier {i}: grid_level must be >= 0"
                )
            if ("inner_damp" in tier
                    and not (0.0 <= tier["inner_damp"] < 1.0)):
                raise ValueError(
                    f"override for {key} tier {i}: inner_damp must be "
                    f"in [0, 1)"
                )
            if ("inner_diis_start_cycle" in tier
                    and not (tier["inner_diis_start_cycle"] >= 1)):
                raise ValueError(
                    f"override for {key} tier {i}: "
                    f"inner_diis_start_cycle must be >= 1"
                )
            if "level_shift" in tier and abs(tier["level_shift"]) > 5.0:
                raise ValueError(
                    f"override for {key} tier {i}: "
                    f"|level_shift| > 5 Ha is implausible; check unit/typo"
                )


def _resolve_tiers_for_species(
    name: str, charge: int, spin: int, is_uks: bool,
) -> tuple[dict, ...]:
    """Resolve the cascade tier list for a species.

    Returns the per-spin default tiers verbatim (`is`-equality with
    `_OEP_TIERS_UKS` / `_OEP_TIERS_RKS`) if the species is not in
    `_PER_SPECIES_OEP_OVERRIDES`. Otherwise produces a tier list whose
    i-th entry merges the i-th override-tier dict onto the i-th
    default tier (clamped to the last default tier when the override
    has more tiers than the default). When the override has FEWER
    tiers than the default, the resolved cascade is TRUNCATED to the
    override's length — the override is the authoritative cascade for
    that species, not a per-tier patch on top of the full default
    cascade.

    The merged tier dict is a *partial* merge: it carries default-tier
    knobs (aux_basis, regularization, max_iter, conv_tol) plus any
    override-set new knobs (grid_level, level_shift, inner_damp,
    inner_diis_start_cycle). The cascade-loop caller fills spin-
    default values for any new knob the override didn't set.

    Empty override tuple is normally rejected upstream by
    `_validate_overrides`; this function defensively re-checks and
    raises ValueError if it ever sees one.
    """
    ovr_tiers = _PER_SPECIES_OEP_OVERRIDES.get((name, charge, spin))
    base_tiers = _OEP_TIERS_UKS if is_uks else _OEP_TIERS_RKS
    if ovr_tiers is None:
        return base_tiers
    if len(ovr_tiers) == 0:
        raise ValueError(
            f"override for ({name!r}, {charge}, {spin}) is empty "
            f"— _validate_overrides should have rejected this earlier"
        )
    return tuple(
        {**base_tiers[min(i, len(base_tiers) - 1)], **ovr}
        for i, ovr in enumerate(ovr_tiers)
    )


def _migrate_intermediates_to_grid_suffixed(cache_dir) -> int:
    """Rename pre-2026-05-03 intermediates to grid-suffixed names.

    Scans `_intermediates/` shallow (NOT recursive); renames every
    `<name>_scf.npz` to `<name>_g1_scf.npz` and every `<name>_ccsd.npz`
    to `<name>_g1_ccsd.npz` independently — `_scf` and `_ccsd` are
    scanned in two separate passes so a crash mid-migration leaves
    the remaining files visible to the next migration pass (spec
    sec. 5.6 partial-state recovery).

    Pre-2026-05-03 caches were built at the global default
    grid_level=1 (run_scf_with_cache and run_ccsd_with_cache default
    grid_level=1), so the `_g1_` rename is correct by construction.

    Returns the number of files renamed (0 on second / idempotent
    call, > 0 on first call against a legacy cache).

    Pass-8 single-writer assumption: callers must not invoke this
    helper concurrently against the same `cache_dir` from multiple
    processes. Sane callers (`precompute_all`, `preflight_uks_oep`,
    the harness's startup precaution) all run sequentially.

    Raises:
      FileExistsError if `<name>_g1_scf.npz` already exists alongside
          an unsuffixed `<name>_scf.npz` (refuses to silently overwrite;
          user must resolve the conflict manually).
      OSError (let bubble) if the directory is read-only.
    """
    from pathlib import Path
    import os
    inter = Path(cache_dir) / "_intermediates"
    if not inter.is_dir():
        return 0
    import re
    # Pre-compile patterns that match ANY `_g{N}_<stage>.npz` for N in [0,9]
    # (MoleculeSpec.grid_level range per config.py:370). Used to skip
    # already-grid-suffixed files of any grid_level — not just _g1_.
    # Pass-9 fix (Plan-2 review): the earlier `name.endswith(suffix_new)`
    # only-skipped `_g1_*` files, which would have corrupted future
    # `_g2_*` / `_g3_*` caches written by override species (spec §5.6
    # explicitly says "Override species at grid_level=2 get fresh _g2_*
    # files; their _g1_* caches are retained").
    _re_already_g_scf  = re.compile(r"_g\d+_scf\.npz$")
    _re_already_g_ccsd = re.compile(r"_g\d+_ccsd\.npz$")
    n_renamed = 0
    for suffix_old, suffix_new, already_re in (
        ("_scf.npz",  "_g1_scf.npz",  _re_already_g_scf),
        ("_ccsd.npz", "_g1_ccsd.npz", _re_already_g_ccsd),
    ):
        for p in inter.iterdir():     # SHALLOW: iterdir, not rglob
            name = p.name
            # Pass-8 fix: use regex/endswith — NOT `"_g" in name`
            # substring test (would corrupt Mg/Hg/Ag).
            # Pass-9 fix: skip ANY `_g{N}_<stage>.npz` for any N
            # (Plan-2 review caught that `_g2_scf.npz` would otherwise
            # be re-renamed to `_g2_g1_scf.npz`).
            if already_re.search(name):
                continue              # already migrated at any g{N}
            if name.endswith(suffix_old):
                base = name[:-len(suffix_old)]
                target = inter / f"{base}{suffix_new}"
                if target.exists():
                    raise FileExistsError(
                        f"migration conflict: {p} would overwrite "
                        f"{target}; user must resolve manually"
                    )
                os.replace(str(p), str(target))
                n_renamed += 1
    # Parent-directory fsync for durability across power loss
    # (POSIX rename is atomic per-file but not durable until parent fsync).
    # Plan-2-review fix: only catch AttributeError (Windows lacks
    # O_DIRECTORY); let real OSErrors (ENOSPC, EIO) bubble — durability
    # failures should fail loudly so the user can intervene before SCF
    # results land in a broken filesystem state.
    if hasattr(os, "O_DIRECTORY"):
        dir_fd = os.open(str(inter), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    return n_renamed


def run_oep_cascade(
    spec: SpeciesEntry,
    atoms,
    *,
    ccsd_payload: dict,
    cache_dir,
    basis: str = "def2-svp",
    grid_level: int = 1,
    progress_callback=None,
):
    """Stage 3: OEP inversion with 2-tier cascade + skip-if-cached.

    Tries svp-jkfit primary; on RuntimeError or non-converged inversion
    falls back to def2-tzvp-jkfit. On both-tier failure raises
    RuntimeError listing the species.

    Output: <cache_dir>/<name>.npz with vxc_ref + dm_target +
    rho_ref_grid + ref_density_method + oep_* provenance. Two-phase
    write exploits save_vxc_ref's merge semantics
    (xcquinox/alec/oep.py:696-700).

    ``progress_callback`` (optional) is a callable
    ``fn(tier_idx, aux_basis, iter_int, density_error_float)`` invoked
    once per L-BFGS outer iteration. The cascade adapts its own
    ``progress_callback`` argument from this richer signature so callers
    (e.g. ``scripts/smoke_preflight_uks_oep.py``) can show per-tier +
    per-iter convergence trajectory inside the otherwise-silent
    ``run_oep_inversion`` call.
    """
    from collections import Counter
    from pathlib import Path
    import numpy as np
    from xcquinox.alec import oep as alec_oep
    from xcquinox.alec.config import MoleculeSpec

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{spec.name}.npz"

    if npz_path.is_file():
        # Verify completeness — must have all required keys
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                if _REQUIRED_NPZ_KEYS.issubset(set(z.files)):
                    return npz_path
        except (OSError, ValueError):
            pass  # Corrupt cache — recompute

    # Build a MoleculeSpec for run_oep_inversion
    coords = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    atom_lines = "; ".join(
        f"{s} {coords[i,0]:.6f} {coords[i,1]:.6f} {coords[i,2]:.6f}"
        for i, s in enumerate(syms)
    )
    comp = dict(Counter(syms))
    mol_spec = MoleculeSpec.from_dict(
        name=spec.name, atom=atom_lines, basis=basis,
        charge=spec.charge, spin=spec.spin,
        atom_composition=comp, grid_level=grid_level,
    )

    # UKS species with orbital degeneracy (X²Π radicals like HO/CN/NO,
    # near-degenerate X²A1 like NO2) need a level shift on the inner SCF
    # to keep DIIS in a single broken-symmetry basin under L-BFGS-B
    # perturbations of the OEP coefficients. Without this, density_error
    # plateaus far from conv_tol (HO at def2-svp/grid_level=1 stalls at
    # ~0.17). Closed-shell RKS is unaffected, so level_shift=0 there.
    # See xcquinox/alec/tests/test_oep_uks.py module docstring for
    # background on the basin-hopping failure mode.
    is_uks = spec.spin > 0
    spin_default_level_shift = 0.5 if is_uks else 0.0
    tiers = _resolve_tiers_for_species(
        spec.name, spec.charge, spec.spin, is_uks=is_uks,
    )

    last_err = None
    oep_result = None
    for tier_idx, tier in enumerate(tiers):
        aux_basis = tier["aux_basis"]
        regularization = tier["regularization"]
        max_iter = tier["max_iter"]
        conv_tol = tier["conv_tol"]
        # New per-tier knobs, all with safe defaults that preserve
        # pre-Plan-2 behavior for non-override species:
        tier_grid_level = tier.get("grid_level", grid_level)
        tier_level_shift = tier.get("level_shift", spin_default_level_shift)
        tier_inner_damp = tier.get("inner_damp", 0.1)
        tier_inner_diis = tier.get("inner_diis_start_cycle", 5)

        # grid_level travels through mol_spec (canonical source per
        # spec sec. 5.4). dataclasses.replace returns a NEW frozen
        # instance; the original mol_spec is unchanged.
        if tier_grid_level == mol_spec.grid_level:
            tier_mol_spec = mol_spec
        else:
            tier_mol_spec = dataclasses.replace(
                mol_spec, grid_level=tier_grid_level,
            )

        # Adapt the cascade's richer (tier_idx, aux_basis, iter, err)
        # callback signature down to run_oep_inversion's (iter, err).
        _cb = None
        if progress_callback is not None:
            _aux = aux_basis
            def _cb(it, err, _idx=tier_idx, _aux=_aux):
                progress_callback(_idx, _aux, it, err)

        try:
            oep_result = alec_oep.run_oep_inversion(
                tier_mol_spec,
                ccsd_payload["dm_ao"],
                aux_basis=aux_basis,
                regularization=regularization,
                max_iter=max_iter,
                conv_tol=conv_tol,
                level_shift=tier_level_shift,
                inner_damp=tier_inner_damp,
                inner_diis_start_cycle=tier_inner_diis,
                progress_callback=_cb,
            )
            if oep_result.converged:
                break
            last_err = (
                f"OEP not converged at tier {tier_idx} ({aux_basis}); "
                f"density_error={oep_result.density_error:.3e}"
            )
        except (RuntimeError, ValueError) as e:
            last_err = f"tier {tier_idx} ({aux_basis}) raised: {e}"
            oep_result = None

    if oep_result is None or not oep_result.converged:
        raise RuntimeError(
            f"OEP cascade failed for {spec.name!r} (charge={spec.charge}, "
            f"spin={spec.spin}, source={spec.source}): {last_err}"
        )

    # Two-phase write: phase 1 stores rho_ref_grid (+ grid_level_used
    # provenance); phase 2's save_vxc_ref merges in vxc_ref + dm_target +
    # provenance (its merge semantics preserve any phase-1 key not in its
    # own payload, so grid_level_used survives the merge).
    # Use np.savez_compressed for consistency with stages 1+2 caches
    # (T5 code-quality nit).
    # CFG-03: record the generating grid_level so data.py can assert the
    # consumer's resolved grid_level matches what produced this reference.
    np.savez_compressed(
        npz_path,
        rho_ref_grid=ccsd_payload["rho_ref_grid"],
        ref_density_method=np.array("ccsd"),
        grid_level_used=np.array(int(grid_level)),
    )
    # P4-03: save_vxc_ref records the achieved OEP density error as the
    # ``oep_density_error`` key (a real noise floor on this species' vxc_ref:
    # OEP inversion is ill-posed, so the V_xc-channel error is unbounded by the
    # density-error tolerance). It is already in data._ALLOWED_EXTERNAL_KEYS, so
    # downstream can read it to WEIGHT the per-species vxc loss by the achieved
    # floor (recommended; the weighting itself is a training-design choice not
    # yet wired into losses.L5GradnormVxcStep7 — see review P4-03).
    alec_oep.save_vxc_ref(
        oep_result, str(npz_path),
        dm_target=ccsd_payload["dm_ao"],
        method="ccsd",
    )
    # EXTREF-04: fsync the parent dir for durability parity with stage 1.
    _fsync_dir(cache_dir)
    return npz_path


def preflight_uks_oep(
    *,
    cache_dir,
    basis: str = "def2-svp",
    grid_level: int = 1,
) -> None:
    """Smoke-test UKS OEP on HO (doublet, 2Pi) and HN (triplet, 3Sigma-)
    BEFORE running the full ~58-species pre-compute.

    HO: 9 e-, smallest meaningful UKS doublet.
    HN: 8 e-, smallest UKS triplet (NIST CCCBDB cited at
    dfs_pool.py:175-182 -- Herzberg I VI 3Sigma-).

    Aborts (raises RuntimeError) if either OEP fails or returns
    wrong-shape vxc_ref. Catches the UKS-OEP unknown before burning
    ~hour of CPU on the full set.
    """
    import numpy as np
    # Spec sec. 5.6: migrate pre-2026-05-03 unsuffixed cache filenames
    # for direct callers that bypass precompute_all (e.g.,
    # smoke_preflight_uks_oep.py, tests). Idempotent no-op when
    # already migrated.
    _migrate_intermediates_to_grid_suffixed(cache_dir)
    smoke_specs = [
        SpeciesEntry("HO", 0, 1, "dfs_ae"),  # doublet
        SpeciesEntry("HN", 0, 2, "dfs_ae"),  # triplet
    ]
    for spec in smoke_specs:
        atoms = resolve_geometry(spec)
        scf = run_scf_with_cache(spec, atoms, cache_dir=cache_dir,
                                 basis=basis, grid_level=grid_level)
        if not scf["spin_unrestricted"]:
            raise RuntimeError(
                f"Pre-flight failure: {spec.name} should be UKS but "
                f"SCF dispatched RKS (spin={spec.spin})"
            )
        cc = run_ccsd_with_cache(spec, atoms, scf_payload=scf,
                                 cache_dir=cache_dir,
                                 basis=basis, grid_level=grid_level)
        npz_path = run_oep_cascade(spec, atoms, ccsd_payload=cc,
                                   cache_dir=cache_dir,
                                   basis=basis, grid_level=grid_level)
        # Verify shape contract
        with np.load(npz_path, allow_pickle=False) as z:
            vxc = np.asarray(z["vxc_ref"])
            rho = np.asarray(z["rho_ref_grid"])
        if vxc.ndim != 3 or vxc.shape[0] != 2:
            raise RuntimeError(
                f"Pre-flight UKS shape mismatch for {spec.name}: "
                f"vxc_ref.shape={vxc.shape}, expected (2, n_ao, n_ao)"
            )
        if rho.ndim != 1:
            raise RuntimeError(
                f"Pre-flight UKS rho_ref_grid shape for {spec.name} "
                f"is {rho.shape}; must be 1D spin-summed (data.py:296-299)"
            )


class RunLog:
    """Atomic JSON log for the Cell 0.5 pipeline.

    Writes _run_log_partial.json after every species (kill-safe via
    tempfile.mkstemp + os.replace, matching the T3 atomic-write precedent
    at external_refs.py:264-274). On finalize, renames to
    _run_log_<UTC-timestamp>.json so each run's log is preserved for
    later debugging.
    """

    def __init__(self, *, cache_dir):
        from pathlib import Path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.cache_dir / "_run_log_partial.json"
        self._payload: dict = {
            "started_at_utc": None,
            "ended_at_utc": None,
            "species_count": 0,
            "results": [],
        }

    def start(self, species_names) -> None:
        import datetime
        self._payload["started_at_utc"] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        self._payload["species_count"] = len(list(species_names))
        self._flush()

    def record_result(
        self, *, name, charge, spin, status,
        wall_clock_s, error_msg, **extra,
    ) -> None:
        self._payload["results"].append({
            "name": name, "charge": int(charge), "spin": int(spin),
            "status": status, "wall_clock_s": float(wall_clock_s),
            "error_msg": error_msg, **extra,
        })
        self._flush()

    def finalize(self):
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%SZ")
        self._payload["ended_at_utc"] = now.isoformat()
        final_path = self.cache_dir / f"_run_log_{ts}.json"
        self._flush(path=final_path)
        if self.partial_path.is_file():
            self.partial_path.unlink()
        return final_path

    def _flush(self, *, path=None):
        """Atomic JSON write: tempfile.mkstemp -> write -> os.replace.

        Matches the T3 atomic-write pattern at external_refs.py:264-274
        so a kill mid-flush cannot leave a corrupt partial JSON that the
        next run would mis-parse.
        """
        import json
        import os
        import tempfile
        target = path if path is not None else self.partial_path
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.cache_dir), suffix=".json"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._payload, f, indent=2)
            os.replace(tmp_name, target)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise


def precompute_all(
    species: list["SpeciesEntry"],
    *,
    cache_dir,
    basis: str = "def2-svp",
    grid_level: int = 1,
    run_preflight: bool = True,
) -> None:
    """Top-level Cell 0.5 driver.

    Iterates the species union, runs SCF + CCSD + OEP per species (each
    stage cached). Skip-if-cached for species whose final .npz already
    has all required keys. Logs every result via RunLog. On any
    species-level failure, raises RuntimeError with the failed-species
    list -- does NOT silently skip.

    Parameters
    ----------
    species : list[SpeciesEntry]
    cache_dir : path
        Root for external_refs/. Cell 0.5 passes
        STEP7_ROOT / "external_refs".
    basis, grid_level : floor at def2-svp / 1 to match descriptor
        extraction (data.py shape contract).
    run_preflight : run preflight_uks_oep first (default True). Set
        False in tests that pre-populate caches.
    """
    import time
    import traceback
    from pathlib import Path
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(iterable, **_kw):
            class _Noop:
                def __iter__(self_inner):
                    return iter(iterable)
                def set_postfix(self_inner, **kw):
                    pass
            return _Noop()

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Spec sec. 5.2 (Pass-8 canonical call site, Plan-2-review-corrected
    # ordering): validate the per-species override table BEFORE any
    # disk mutation. Orphan keys / typo'd knobs / out-of-range values
    # raise here, fail-fast before migration touches anything.
    _validate_overrides(species)
    # Spec sec. 5.6: migrate any pre-2026-05-03 unsuffixed cache
    # filenames BEFORE preflight reads from _intermediates/. Idempotent
    # no-op once migration has run.
    _migrate_intermediates_to_grid_suffixed(cache_dir)
    log = RunLog(cache_dir=cache_dir)
    log.start([s.name for s in species])

    if run_preflight:
        preflight_uks_oep(cache_dir=cache_dir,
                          basis=basis, grid_level=grid_level)

    failures: list[str] = []
    bar = tqdm(species, desc="Cell 0.5 CCSD refs", leave=True,
               dynamic_ncols=True)
    for spec in bar:
        bar.set_postfix(name=spec.name, charge=spec.charge, spin=spec.spin)
        npz_path = cache_dir / f"{spec.name}.npz"
        if _npz_is_complete(npz_path):
            log.record_result(
                name=spec.name, charge=spec.charge, spin=spec.spin,
                status="SKIPPED_CACHED", wall_clock_s=0.0, error_msg=None,
            )
            continue
        t0 = time.time()
        try:
            atoms = resolve_geometry(spec)
            scf = run_scf_with_cache(spec, atoms, cache_dir=cache_dir,
                                     basis=basis, grid_level=grid_level)
            cc = run_ccsd_with_cache(spec, atoms, scf_payload=scf,
                                     cache_dir=cache_dir,
                                     basis=basis, grid_level=grid_level)
            run_oep_cascade(spec, atoms, ccsd_payload=cc,
                            cache_dir=cache_dir,
                            basis=basis, grid_level=grid_level)
            dt = time.time() - t0
            log.record_result(
                name=spec.name, charge=spec.charge, spin=spec.spin,
                status="OK", wall_clock_s=dt, error_msg=None,
            )
        except Exception as e:
            dt = time.time() - t0
            tb = traceback.format_exc()
            log.record_result(
                name=spec.name, charge=spec.charge, spin=spec.spin,
                status="FAIL", wall_clock_s=dt, error_msg=tb,
            )
            failures.append(spec.name)

    log.finalize()
    if failures:
        raise RuntimeError(
            f"Cell 0.5 pre-compute failed for {len(failures)} species: "
            f"{failures}. Inspect _run_log_*.json for details."
        )


def _npz_is_complete(npz_path) -> bool:
    """True if the npz exists and carries every key the loss expects."""
    import numpy as np
    if not npz_path.is_file():
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            return _REQUIRED_NPZ_KEYS.issubset(set(z.files))
    except (OSError, ValueError):
        return False
