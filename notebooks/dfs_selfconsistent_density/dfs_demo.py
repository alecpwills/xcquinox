"""DFS-exact self-consistent density-training demo helpers.

Thin orchestration over ``xcquinox.alec``. Every spec this module builds is
assembled by calling the SAME production functions the cluster harness uses for
the ``dfs_step7`` domain (``get_domain_profile``, ``build_dfs_pool_points``,
``species_union_from_points``, ``atoms_to_mol_spec``, ``build_targets``,
``classify_aux_only``, ``DomainProfile.bh76_meta_to_loss_dict``), so a demo
``TrainingSpec`` is field-for-field identical to what ``spec_builder`` produces
-- only the pool is shrunk to a handful of small systems. Nothing here is new
physics.

The training recipe replicated here is the repo's source-audited DFS ("dpyscf")
methodology, documented in ``xcquinox/alec/HISTORY.md`` (Phases 10-13) and
sourced to Dick & Fernandez-Serra, "Highly accurate and constrained density
functional obtained with differentiable programming," Phys. Rev. B 104, L161109
(2021):

  - loss ``L5_gradnorm_vxc_step7`` with ``density_per_electron=True`` (the drho /
    N_e^2 density-loss normalization),
  - ``update_scheme="per_molecule"`` (the dpyscf per-group stochastic loop) with
    ``channel_weights=()`` so the density channel inherits the 20x
    density-dominant ``_DEFAULT_CHANNEL_WEIGHTS`` (train.py),
  - ``ae_as_reactions`` atomization energies (compound -> constituent atoms,
    scored with the NN's own self-consistent atom energies),
  - the ``full_3`` / ``full_25`` FULL differentiable KS-SCF solvers with the
    decaying mixer ``alpha=0.3^step+0.3`` and the tail-weighted energy loss,
  - adamw with the linear-decay LR schedule.

Documented deviations from the PRB paper (all inherited from the repo's
dfs_step7 recipe): plain CCSD (not CCSD(T)) reference densities; ``grid_level=2``
(paper uses 3); adamw (paper uses Adam + ReduceLROnPlateau); spin-summed ``N_e^2``
(paper uses per-spin ``N_sigma^2``). The arch set spans GGA -> rung-3.5 ->
meta-GGA, so the paper's meta-GGA rung IS represented (the ``deep_mgga_*`` nets).
"""
from __future__ import annotations

import dataclasses
import os
from typing import Iterable, Sequence

import numpy as np

from xcquinox.alec import benchmark_refs, get_architecture
from xcquinox.alec.balancing import GradNormConfig
from xcquinox.alec.external_refs import SpeciesEntry, run_scf_with_cache
from xcquinox.alec.cluster.domain import get_domain_profile
from xcquinox.alec.cluster.spec_builder import (
    atoms_to_mol_spec,
    build_targets,
    classify_aux_only,
)
from xcquinox.alec.config import PretrainSpec, TestSpec, TrainingSpec
from xcquinox.alec.pretrain import run_pretrain
from xcquinox.alec.pretrain_data_gen import ensure_pretrain_data
from xcquinox.alec.solver import FeaturePolicy, SolverConfig, SolverMode
from xcquinox.alec.orientation_lock import DEFAULT_STRENGTH as _OL_DEFAULT_STRENGTH
from xcquinox.alec.dfs_pool import ATOMIC_GROUND_STATE_SPIN
from xcquinox.alec.training_points import (
    _atom_anchor_atoms,
    build_dfs_pool_points,
    species_union_from_points,
)

# ---------------------------------------------------------------------------
# Fixed demo configuration
# ---------------------------------------------------------------------------

#: The four spin-diverse training molecules (Hill formulae as used by
#: ``build_dfs_pool()``): closed-shell H2O + LiH, open-shell OH (doublet) +
#: NH (triplet). Their AE-as-reaction points carry along the H/O/Li/N atom
#: anchors, so the species union is {H2O, LiH, OH, NH, H, O, Li, N}.
DEFAULT_MOLECULE_HILLS: tuple[str, ...] = ("H2O", "HLi", "HO", "HN")

#: Smallest end-to-end subset for the SMOKE path (one closed- + one open-shell).
SMOKE_MOLECULE_HILLS: tuple[str, ...] = ("H2O", "HO")

#: The architectures featured by the notebook (swap your own here). Two GGA-ladder
#: nets (plain GGA, GGA+cusp+rung-3.5) plus two meta-GGA nets (pure DFS meta-GGA,
#: and the combined cusp+rung-3.5+meta-GGA). The meta-GGA nets pretrain to SCAN
#: (arch.meta_gga=True routes the target), so the notebook can ask whether a
#: trained meta-GGA net improves on SCAN itself at reproducing the CCSD density.
ARCH_NAMES: tuple[str, ...] = (
    "deep_3x16", "deep_rung35_3x16", "deep_mgga_3x16", "deep_rung35_mgga_3x16")

#: Held-out generalization set (real DFS-pool entries, elements subset {H,O,Li,N},
#: NONE in the training set): N2 (closed-shell triple bond, a classic PBE
#: density-error case), NO (X-2-Pi degenerate radical -- doubles as proof the
#: orientation lock generalizes to an UNSEEN degenerate system), NO2 (bent 2-A1
#: polyatomic radical). Their AE-as-reaction points carry the N/O atom anchors.
HELDOUT_MOLECULE_HILLS: tuple[str, ...] = ("N2", "NO", "NO2")

#: Orientation-lock strength for the whole demo (orientation_lock.py). Applied
#: IDENTICALLY in CCSD ref-generation, the PBE seed, training, and eval (via the
#: solver configs below and generate_ccsd_density_refs), so a degenerate radical
#: (OH in training, NO held-out) always locks the SAME representative of its 2-Pi
#: manifold -> its density is reproducible across processes/machines. 0.0 = off.
ORIENTATION_LOCK_STRENGTH: float = _OL_DEFAULT_STRENGTH

#: DFS reference density basis (Dick & Fernandez-Serra 2021) and repo-recipe grid.
DFS_BASIS: str = "6-311++G(3df,2pd)"
DFS_GRID_LEVEL: int = 2

#: DFS optimizer / schedule hyperparameters (dfs_step7 recipe).
DFS_HYPERPARAMS: dict = {
    "lr_start": 1e-3,
    "lr_end": 1e-5,
    "lr_decay_start": 0.5,
    "grad_clip": 1.0,
    "weight_decay": 1e-4,
    "gradnorm_alpha": 1.5,
    "seed": 42,
    # Pre-balancer scale factors. Under update_scheme="per_molecule" the loop
    # forces both to 1.0 and the 20x density dominance comes from
    # _DEFAULT_CHANNEL_WEIGHTS instead; kept here to match the dfs_step7 YAML.
    "vxc_weight": 0.01,
    "density_weight": 0.1,
}

#: Production epoch counts per solver (CLI ``--n-steps`` on the real runs).
DFS_N_EPOCHS: dict = {"full_3": 150, "full_25": 100}

#: Production pretraining step count (dfs_step7 recipe). Pretraining fits the
#: enhancement factors to PBE per atom; the archs zero-init to LDA (F=1 over
#: lda_x + PW92), so this is the LDA->PBE warm-start the DFS recipe uses. The
#: pretrain ATOMS are derived from the training systems via pretrain_atoms_for()
#: so they always exist at the training basis.
DFS_PRETRAIN_STEPS: int = 2500

#: The DFS domain profile (Chakravorty atom anchors, ("H","Li") regularizer set).
DOMAIN = get_domain_profile("dfs_step7")


# ---------------------------------------------------------------------------
# Pool selection + molecule specs
# ---------------------------------------------------------------------------

def _is_molecule(mol_spec) -> bool:
    """True for a polyatomic (composition sum > 1); False for a single atom."""
    return sum(dict(mol_spec.atom_composition).values()) > 1


def select_dfs_points(hills: Sequence[str] = DEFAULT_MOLECULE_HILLS) -> list:
    """Return the AE-as-reaction ``TrainingPoint``s for the requested molecules.

    Filters the canonical 26-point DFS pool (built with
    ``ae_as_reactions=True``, so each atomization energy is a molecule ->
    constituent-atoms reaction scored with the NN's own self-consistent atom
    energies -- the Dick & Fernandez-Serra 2021 L_RE form). Each returned point
    is ``kind="bh76"`` and carries its compound plus the H/O/Li/N atom anchors
    as species.
    """
    want = set(hills)
    points = build_dfs_pool_points(bh76_mode="reaction_energy", ae_as_reactions=True)
    chosen = [tp for tp in points if tp.name in want]
    found = {tp.name for tp in chosen}
    missing = want - found
    if missing:
        raise ValueError(
            f"requested molecules not in the DFS pool: {sorted(missing)}; "
            f"available AE Hill formulae: {sorted(tp.name for tp in points if tp.kind == 'bh76')}"
        )
    return chosen


def build_mol_specs(chosen_points: Sequence, *, basis: str, grid_level: int,
                    refs_dir: str) -> tuple:
    """Build the deduplicated ``MoleculeSpec`` union for the chosen points.

    Wires ``external_data_path`` to ``<refs_dir>/<name>.npz`` for any species
    whose CCSD reference density already exists on disk (molecules only; atoms
    have no density reference). Call this AFTER
    :func:`generate_ccsd_density_refs` to pick up the reference paths.
    """
    sp_atoms = species_union_from_points(chosen_points)
    specs = tuple(
        atoms_to_mol_spec(at, basis=basis, grid_level=grid_level,
                          external_refs_dir=refs_dir)
        for at in sp_atoms
    )
    # The L5 Dick atomic regularizer anchors on ("H", "Li"). Match
    # spec_builder.build_training_specs: inject the neutral ground-state anchor
    # for any regularizer symbol absent from the species union, via the SAME
    # helper the natural AE path uses (so an injected anchor is byte-identical to
    # a naturally-occurring one). Needed for subsets with no Li-bearing molecule
    # (e.g. the H2O + OH smoke subset).
    present_atoms = {
        next(iter(dict(ms.atom_composition)))
        for ms in specs if not _is_molecule(ms)
    }
    missing_reg = [s for s in DOMAIN.regularize_atom_syms if s not in present_atoms]
    if missing_reg:
        specs = specs + tuple(
            atoms_to_mol_spec(_atom_anchor_atoms(s), basis=basis,
                              grid_level=grid_level, external_refs_dir=refs_dir)
            for s in missing_reg
        )
    return specs


def molecule_specs(mol_specs: Iterable) -> list:
    """The polyatomic subset of ``mol_specs`` (the density-reference targets)."""
    return [ms for ms in mol_specs if _is_molecule(ms)]


# ---------------------------------------------------------------------------
# CCSD reference densities (reuses benchmark_refs -- density-only, no OEP)
# ---------------------------------------------------------------------------

def generate_ccsd_density_refs(mol_specs: Sequence, *, refs_dir: str, basis: str,
                               grid_level: int, progress: bool = True,
                               orientation_lock_strength: float = ORIENTATION_LOCK_STRENGTH
                               ) -> list:
    """Generate per-molecule CCSD reference densities into ``refs_dir``.

    Reuses :func:`xcquinox.alec.benchmark_refs.generate_one` (converged HF ->
    CCSD 1-RDM -> spin-summed density on the PBE-SCF grid; density-only, no OEP).
    Writes ``<refs_dir>/<name>.npz`` with ``rho_ref_grid`` + ``rho_pbe_grid`` on
    the SAME pruned grid ``precompute_fixed_density_data`` builds, so the training
    density and reference align point-for-point. Atoms are skipped (no density
    reference needed -- the loss is molecules-only). Returns ``[(name, status)]``
    where status is ``"OK"``/``"SKIP"``.

    ``orientation_lock_strength`` (>0) biases the reference PBE + HF-for-CCSD core
    Hamiltonians with the SAME operator the eval/training seed uses, so a
    degenerate radical's reference density locks the same component. Turning it on
    (or changing it) auto-regenerates the ref (the strength tags the caches +
    npz), so this is self-healing -- no manual ``refs/`` deletion needed.
    """
    os.makedirs(refs_dir, exist_ok=True)
    mols = molecule_specs(mol_specs)
    results = []
    for i, ms in enumerate(mols, 1):
        if progress:
            print(f"  CCSD ref {i}/{len(mols)}: {ms.name} @ {basis} ...", flush=True)
        # generate_one returns "SKIP" when a complete .npz already exists for this
        # basis/grid/lock (reused), else "OK" (freshly computed). "SKIP" is NOT an error.
        status = benchmark_refs.generate_one(
            ms, out_dir=refs_dir, basis=basis, grid_level=grid_level,
            orientation_lock_strength=orientation_lock_strength)
        if progress:
            human = "cached (reused existing .npz)" if status == "SKIP" else "generated"
            print(f"    {ms.name}: {human}", flush=True)
        results.append((ms.name, status))
    return results


# ---------------------------------------------------------------------------
# SCAN self-consistent baseline (a meta-GGA comparator alongside PBE)
# ---------------------------------------------------------------------------

def _weighted_density_rmse(rho, rho_ref, weights) -> float:
    """Grid-weighted density RMSE vs a reference, mirroring
    ``evaluation.pbe_density_errors``: ``sqrt(sum(w (rho-rho_ref)^2) / sum(w))``.
    """
    rho = np.asarray(rho)
    rho_ref = np.asarray(rho_ref)
    w = np.asarray(weights)
    diff = rho - rho_ref
    return float(np.sqrt(np.sum(w * diff ** 2) / np.sum(w)))


def scan_baseline(mol_specs: Sequence, mol_data_by_name: dict, *, refs_dir: str,
                  basis: str, grid_level: int,
                  orientation_lock_strength: float = ORIENTATION_LOCK_STRENGTH,
                  progress: bool = True) -> dict:
    """Per-species SCAN self-consistent baseline (a meta-GGA comparator to PBE).

    For every species in ``mol_specs`` (molecules AND the AE constituent atoms)
    runs a SCAN KS-SCF via :func:`external_refs.run_scf_with_cache` with
    ``xc="scan"``, reusing the SAME basis/grid/charge/spin/orientation-lock the
    PBE + CCSD references use. The orientation lock matters for the degenerate
    radicals (OH in training, held-out NO): it selects the same representative of
    the degenerate manifold as the CCSD reference, so SCAN's density is scored
    against the matching reference component (identical to how the PBE baseline is
    formed). SCAN caches land next to the PBE/CCSD intermediates under
    ``<refs_dir>/_intermediates/`` with an ``_xcscan`` tag, so they never collide
    with the PBE caches.

    Returns ``{name: {"E_scan", "density_rmse_scan"}}`` where:

      - ``E_scan``            -- SCAN self-consistent total energy (Hartree), for
                                 EVERY species (molecules + atoms), so the demo can
                                 form SCAN's OWN self-consistent atomization energy;
      - ``density_rmse_scan`` -- grid-weighted RMSE of the SCAN self-consistent
                                 density vs the CCSD reference on the eval grid
                                 (molecules only; ``None`` for atoms).

    ``mol_data_by_name`` maps each polyatomic's name to its eval ``mol_data`` (the
    ``precompute_fixed_density_data`` output), supplying ``ao_grid`` (the
    reference-grid AO values), ``grid_weights`` and ``rho_ref_grid``. The SCAN dm
    is in the AO basis (grid-independent), so ``rho_scan_grid = einsum(
    "ij,gj,gi->g", dm_scan_tot, ao_grid, ao_grid)`` evaluates the SCAN density on
    the exact grid the CCSD/PBE references live on. Atoms need no ``mol_data``
    (the density objective is molecules-only).
    """
    out: dict = {}
    n = len(mol_specs)
    for i, ms in enumerate(mol_specs, 1):
        spec = SpeciesEntry(name=ms.name, charge=int(ms.charge),
                            spin=int(ms.spin), source="benchmark")
        atoms = benchmark_refs._mol_spec_to_atoms(ms)
        if progress:
            print(f"  SCAN SCF {i}/{n}: {ms.name} @ {basis} ...", flush=True)
        scf = run_scf_with_cache(
            spec, atoms, cache_dir=refs_dir, basis=basis, grid_level=grid_level,
            orientation_lock_strength=orientation_lock_strength, xc="scan")
        rec = {"E_scan": float(scf["e_tot"]), "density_rmse_scan": None}
        if _is_molecule(ms):
            md = mol_data_by_name.get(ms.name)
            if md is not None and md.get("rho_ref_grid") is not None:
                dm = np.asarray(scf["dm"])
                dm_tot = dm[0] + dm[1] if dm.ndim == 3 else dm
                ao = np.asarray(md["ao_grid"])
                rho_scan = np.einsum("ij,gj,gi->g", dm_tot, ao, ao)
                rec["density_rmse_scan"] = _weighted_density_rmse(
                    rho_scan, md["rho_ref_grid"], md["grid_weights"])
        out[ms.name] = rec
    return out


def attach_scan_baseline(per_molecule_records: Iterable[dict],
                         scan_by_name: dict) -> list:
    """Copies of ``per_molecule_records`` with the SCAN baseline merged in.

    Adds ``E_scan`` (all species) and ``density_rmse_scan`` (molecules) from
    :func:`scan_baseline` output keyed by molecule name, so the three aggregation
    fns can emit their ``*_scan`` series parallel to PBE. A ``None`` SCAN field is
    not attached (atoms carry no density RMSE). Inputs are not mutated.
    """
    out = []
    for rec in per_molecule_records:
        name = rec.get("name") or rec.get("molecule")
        rec = dict(rec)
        s = scan_by_name.get(name)
        if s is not None:
            if s.get("E_scan") is not None:
                rec["E_scan"] = float(s["E_scan"])
            if s.get("density_rmse_scan") is not None:
                rec["density_rmse_scan"] = float(s["density_rmse_scan"])
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Solvers + architectures
# ---------------------------------------------------------------------------

def solver_configs() -> dict:
    """The DFS ``full_3`` and ``full_25`` FULL self-consistent solvers.

    Both are ``mode=FULL`` with the DFS decaying mixer (alpha=0.3^step+0.3) and
    the tail-weighted energy loss (last 10 cycles, quadratic weights). ``full_25``
    additionally enables ``scf_grad_checkpoint`` to keep 25-cycle backprop
    memory-bounded.
    """
    common = dict(
        mode=SolverMode.FULL,
        feature_policy=FeaturePolicy.REASSEMBLE,
        mixer_name="decaying_linear",
        mixer_kwargs=(("base", 0.3), ("floor", 0.3)),
        scf_loss_use_tail=True,
        scf_loss_tail=10,
        scf_loss_weight_power=2.0,
        # Orientation lock on both solvers -> training AND eval precompute bias
        # h_core identically, so OH's (and held-out NO's) density is reproducible.
        orientation_lock_strength=ORIENTATION_LOCK_STRENGTH,
    )
    return {
        "full_3": SolverConfig(max_cycles=3, **common),
        "full_25": SolverConfig(max_cycles=25, scf_grad_checkpoint=True, **common),
    }


def dfs_arch(arch_name: str, *, polarized: bool = True):
    """Return an ``ArchitectureConfig`` with the DFS-recipe polarized correlation.

    The dfs_step7 runs set ``use_polarized_correlation=True`` (the zeta-dependent
    PW92c baseline; adds the x1 spin feature to the correlation net).
    """
    arch = get_architecture(arch_name)
    if polarized:
        arch = dataclasses.replace(arch, use_polarized_correlation=True)
    return arch


def pretrain_atoms_for(mol_specs):
    """Ground-state ``(symbol, 2S)`` atoms for the unique elements in ``mol_specs``.

    Pretraining only needs the elements the functional will encounter, and
    deriving them from the training systems keeps the pretrain-atom set
    consistent with the training basis: the systems were built (and CCSD-ref'd)
    at that basis, so every element is guaranteed available -- unlike a
    hard-coded set that may name an element the basis lacks (e.g. He is absent
    from PySCF's 6-311++G(3df,2pd)).
    """
    syms = sorted({s for ms in mol_specs for s in dict(ms.atom_composition)})
    missing = [s for s in syms if s not in ATOMIC_GROUND_STATE_SPIN]
    if missing:
        raise KeyError(
            f"no ground-state spin for pretrain element(s) {missing}; add them to "
            "xcquinox.alec.dfs_pool.ATOMIC_GROUND_STATE_SPIN (with a citation)."
        )
    return tuple((s, ATOMIC_GROUND_STATE_SPIN[s]) for s in syms)


def pretrain_to_pbe(arch, *, data_dir, checkpoint_dir, basis, grid_level, atoms,
                    n_steps=DFS_PRETRAIN_STEPS, progress_callback=None, force=False):
    """Pretrain ``arch``'s enhancement factors to PBE; return the checkpoint dir.

    The archs zero-initialize to LDA (F_x = F_c = 1 multiply lda_x + PW92, the
    uniform-gas limit); the DFS recipe warm-starts them to PBE first. This
    generates the shared per-atom PBE Fx/Fc target data (``ensure_pretrain_data``,
    idempotent/cached across archs) then runs the pretrain regression
    (``run_pretrain``), writing ``xnet.eqx``/``cnet.eqx`` under ``checkpoint_dir``
    for ``build_dfs_training_spec(pretrain_checkpoint=...)``.

    Checkpoint reuse: if ``checkpoint_dir`` already holds ``xnet.eqx`` +
    ``cnet.eqx`` the pretrain is skipped and the existing checkpoint is reused, so
    reruns are instant. Pass ``force=True`` (or delete the checkpoint dir) to
    re-pretrain -- e.g. after changing the arch, basis, or pretrain atoms.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    xnet_ckpt = os.path.join(checkpoint_dir, "xnet.eqx")
    cnet_ckpt = os.path.join(checkpoint_dir, "cnet.eqx")
    if not force and os.path.isfile(xnet_ckpt) and os.path.isfile(cnet_ckpt):
        print(f"  reusing existing pretrain checkpoint (skip; force=True to redo): {checkpoint_dir}",
              flush=True)
        return checkpoint_dir
    ensure_pretrain_data(
        data_dir, atoms=atoms, basis=basis, grid_level=grid_level,
        polarized=True, descriptors=True, progress=True)
    spec = PretrainSpec(
        arch=arch, data_dir=data_dir, checkpoint_dir=checkpoint_dir,
        n_steps=n_steps, lr_start=1e-2, lr_end=1e-5, lr_decay_start=0.2,
        grad_clip=1.0, seed=42, loss_weighting="integration")
    run_pretrain(spec, progress_callback=progress_callback)
    return checkpoint_dir


# ---------------------------------------------------------------------------
# DFS-exact TrainingSpec + TestSpec assembly
# ---------------------------------------------------------------------------

def _ae_ref_kcalmol(chosen_points: Sequence) -> tuple[dict, set]:
    """Reproduce spec_builder's per-AE-point reference dict + rxn-name set.

    AE-as-reaction points (``kind="bh76"``, ``ae_form="predicted_atom_reaction"``)
    carry a REAL AE reference in ``metadata["e_rxn_ref"]`` (kcal/mol); their names
    are forced into ``aux_only_names`` so the fixed-anchor AE channel stays zero
    for them and they train through the reaction channel instead.
    """
    ae_ref = {
        tp.name: tp.metadata.get("ae_kcalmol")
        for tp in chosen_points if tp.kind == "ae"
    }
    ae_rxn_names = {
        tp.name for tp in chosen_points
        if tp.kind == "bh76" and tp.metadata.get("ae_form") == "predicted_atom_reaction"
    }
    ae_ref.update({
        tp.name: tp.metadata.get("e_rxn_ref")
        for tp in chosen_points if tp.name in ae_rxn_names
    })
    return ae_ref, ae_rxn_names


def build_dfs_training_spec(*, arch, solver_cfg, chosen_points: Sequence,
                            mol_specs: Sequence, checkpoint_dir: str,
                            n_steps: int, pretrain_checkpoint: str | None = None,
                            domain=DOMAIN, hyperparams: dict = DFS_HYPERPARAMS
                            ) -> TrainingSpec:
    """Assemble a DFS-exact ``TrainingSpec`` for the chosen subset.

    Mirrors ``spec_builder.build_training_specs`` (the dfs_step7 wiring) field for
    field, using the same helper functions: only the pool is the handful of
    molecules passed in. The result trains with ``update_scheme="per_molecule"``,
    ``L5_gradnorm_vxc_step7`` + ``density_per_electron=True``, the 20x
    density-dominant channel-weight defaults, and the given FULL solver.
    """
    ae_ref, ae_rxn_names = _ae_ref_kcalmol(chosen_points)
    targets = build_targets(mol_specs, ae_ref, domain)
    aux_only_names = tuple(sorted(
        set(classify_aux_only(mol_specs, ae_ref))
        | {ms.name for ms in mol_specs if ms.name in ae_rxn_names}
    ))
    bh76_ha = [
        domain.bh76_meta_to_loss_dict(tp)
        for tp in chosen_points if tp.kind == "bh76"
    ]
    ip13_ha = [
        domain.ip13_meta_to_loss_dict(tp)
        for tp in chosen_points if tp.kind == "ip13"
    ]
    loss_kwargs = {
        "bh76_reactions": bh76_ha,
        "ip13_pairs": ip13_ha,
        "aux_only_names": aux_only_names,
        "regularize_atom_syms": tuple(domain.regularize_atom_syms),
        "solver_config": solver_cfg,
        "vxc_weight": hyperparams["vxc_weight"],
        "density_weight": hyperparams["density_weight"],
        # Survives the per-molecule loop's *_weight overrides (it copies
        # loss_kwargs and only forces the vxc/density scale knobs to 1.0).
        "density_per_electron": True,
    }
    return TrainingSpec.from_dicts(
        arch=arch,
        molecules=tuple(mol_specs),
        targets=targets,
        atom_energies=dict(domain.atom_energies),
        loss_name="L5_gradnorm_vxc_step7",
        loss_kwargs=loss_kwargs,
        solver_config=solver_cfg,
        pretrain_checkpoint=pretrain_checkpoint,
        checkpoint_dir=checkpoint_dir,
        n_steps=n_steps,
        lr_start=hyperparams["lr_start"],
        lr_end=hyperparams["lr_end"],
        lr_decay_start=hyperparams["lr_decay_start"],
        grad_clip=hyperparams["grad_clip"],
        weight_decay=hyperparams["weight_decay"],
        seed=hyperparams["seed"],
        balancing=GradNormConfig(alpha=hyperparams["gradnorm_alpha"]),
        pbe_anchor_weight=0.0,
        pbe_anchor_sample=None,
        require_atom_anchors=False,
        # THE critical DFS knobs: the dpyscf per-group loop + inherit the 20x
        # density-dominant _DEFAULT_CHANNEL_WEIGHTS (channel_weights left empty).
        update_scheme="per_molecule",
        channel_weights=(),
    )


def build_dfs_test_spec(*, training_spec: TrainingSpec, model_checkpoint: str,
                        solver_cfg, output_dir: str, domain=DOMAIN,
                        metrics: Sequence[str] = (
                            "total_energy", "atomization_energy",
                            "density_rmse", "scf_convergence")) -> TestSpec:
    """Build the evaluation ``TestSpec`` under the SAME solver used to train.

    ``density_rmse`` is solver-aware: with the FULL ``solver_cfg`` it evaluates
    the model's SELF-CONSISTENT density and compares to the CCSD ``rho_ref_grid``,
    and also reports ``density_rmse_pbe`` (the PBE-vs-CCSD baseline on the same
    grid) -- the headline "did density training beat PBE?" comparison.
    """
    # run_test auto-wires atom_energies into the atomization_energy metric
    # (evaluation.py:429-430), so no metric_kwargs are needed.
    return TestSpec.from_dicts(
        model_checkpoint=model_checkpoint,
        arch=training_spec.arch,
        molecules=training_spec.molecules,
        metrics=tuple(metrics),
        atom_energies=dict(domain.atom_energies),
        solver_config=solver_cfg,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Diagnostics aggregation (pure logic)
# ---------------------------------------------------------------------------

def aggregate_density_diagnostics(per_molecule_records: Iterable[dict]) -> list:
    """Flatten per-molecule ``run_test`` records into density-RMSE rows.

    Each input record is one molecule's metric dict (the ``DensityRMSEMetric``
    output merged with identity fields). Returns rows carrying the NN
    self-consistent density RMSE vs CCSD (``density_rmse``) and the model-free
    PBE-vs-CCSD baseline (``density_rmse_pbe``), skipping atomic systems (which
    the metric returns as ``skipped``). ``beats_pbe`` is True when the trained
    functional's self-consistent density is closer to CCSD than PBE's.
    """
    rows = []
    for rec in per_molecule_records:
        rmse = rec.get("density_rmse")
        rmse_pbe = rec.get("density_rmse_pbe")
        if rmse is None or rmse_pbe is None:
            continue  # atomic system or no CCSD reference loaded
        row = {
            "name": rec.get("name") or rec.get("molecule"),
            "density_rmse": float(rmse),
            "density_rmse_pbe": float(rmse_pbe),
            "beats_pbe": float(rmse) < float(rmse_pbe),
            "improvement": float(rmse_pbe) - float(rmse),
        }
        # SCAN comparator (attached by attach_scan_baseline); absent -> PBE-only.
        rmse_scan = rec.get("density_rmse_scan")
        if rmse_scan is not None:
            row["density_rmse_scan"] = float(rmse_scan)
            row["beats_scan"] = float(rmse) < float(rmse_scan)
        rows.append(row)
    return rows


# Hartree -> kcal/mol (matches the notebook's KCAL and utils convention).
_HARTREE_TO_KCAL = 627.5094740631


def self_consistent_ae(per_molecule_records, comp_by_name, ae_ref_kcal):
    """Atomization-energy errors (kcal/mol) from each functional's OWN
    self-consistent atom energies.

    This is the physically correct atomization energy, and exactly what
    ``ae_as_reactions`` trains (compound -> constituent atoms, scored with the
    functional's own self-consistent atom energies). It is NOT the fixed-anchor
    ``AE_nn`` field emitted by the ``atomization_energy`` metric, which subtracts
    the molecule energy from FIXED exact (Chakravorty) atom totals and so reports
    the functional's absolute-energy offset (tens-to-hundreds of kcal/mol for a
    net trained only on reaction energies + density), not its atomization energy.

    Atom energies are read from the atomic-system eval records (``skip_reason ==
    "atomic_system"``); a molecule is emitted only if every constituent atom was
    evaluated. Returns one row per non-atomic molecule with a reference AE:
    ``{name, ae_nn_kcal, ae_pbe_kcal, ref_kcal, err_nn, err_pbe, beats_pbe}``.
    """
    e_atom_nn: dict = {}
    e_atom_pbe: dict = {}
    e_atom_scan: dict = {}
    for rec in per_molecule_records:
        if rec.get("skip_reason") == "atomic_system":
            sym = rec.get("molecule") or rec.get("name")
            e_atom_nn[sym] = float(rec["E_total_nn"])
            e_atom_pbe[sym] = float(rec["E_pbe"])
            if rec.get("E_scan") is not None:
                e_atom_scan[sym] = float(rec["E_scan"])

    rows = []
    for rec in per_molecule_records:
        if rec.get("skip_reason") == "atomic_system":
            continue
        name = rec.get("name") or rec.get("molecule")
        comp = comp_by_name.get(name)
        ref = ae_ref_kcal.get(name)
        if comp is None or ref is None:
            continue
        comp = dict(comp)
        if not all(sym in e_atom_nn for sym in comp):
            continue  # a constituent atom was not evaluated self-consistently
        ae_nn = sum(e_atom_nn[s] * n for s, n in comp.items()) - float(rec["E_total_nn"])
        ae_pbe = sum(e_atom_pbe[s] * n for s, n in comp.items()) - float(rec["E_pbe"])
        ae_nn_kcal = ae_nn * _HARTREE_TO_KCAL
        ae_pbe_kcal = ae_pbe * _HARTREE_TO_KCAL
        ref = float(ref)
        err_nn = ae_nn_kcal - ref
        err_pbe = ae_pbe_kcal - ref
        row = {
            "name": name,
            "ae_nn_kcal": ae_nn_kcal,
            "ae_pbe_kcal": ae_pbe_kcal,
            "ref_kcal": ref,
            "err_nn": err_nn,
            "err_pbe": err_pbe,
            "beats_pbe": abs(err_nn) < abs(err_pbe),
        }
        # SCAN's OWN self-consistent atomization energy -- only when the molecule
        # AND every constituent atom carry a SCAN energy (mirrors the own-atom NN
        # rule); attach_scan_baseline populates E_scan.
        if rec.get("E_scan") is not None and all(s in e_atom_scan for s in comp):
            ae_scan = sum(e_atom_scan[s] * n for s, n in comp.items()) - float(rec["E_scan"])
            ae_scan_kcal = ae_scan * _HARTREE_TO_KCAL
            err_scan = ae_scan_kcal - ref
            row["ae_scan_kcal"] = ae_scan_kcal
            row["err_scan"] = err_scan
            row["beats_scan"] = abs(err_nn) < abs(err_scan)
        rows.append(row)
    return rows


def combined_energy_density(ae_rows, density_rows):
    """DFS energy-density error ``ED`` (kcal/mol), NN vs PBE.

    Follows Dick & Fernandez-Serra (PRB 104, L161109 (2021)) Eq. 21: the harmonic
    mean of an energy error and the density error rescaled to an energy,
    ``ED = 2 / (1/E_MAE + 1/(gamma*D))``. DFS fit the slope ``gamma`` (1084.87
    kcal/mol) across many functionals against WTMAD-2; this pool is tiny (WTMAD
    dropped, per request) and has only the NN + PBE, so ``gamma`` is
    SELF-CALIBRATED from the PBE baseline (``gamma = E_MAE_pbe / D_pbe``). Density
    and energy then share a kcal/mol scale and ``ED_pbe == E_MAE_pbe``; the choice
    of density-error unit (RMSE here, matching the density figure) does not affect
    the NN-vs-PBE ranking because gamma absorbs it.

    ``E_MAE`` is the MAE of the self-consistent AE error (``self_consistent_ae``);
    ``D`` is the mean self-consistent density RMSE vs CCSD
    (``aggregate_density_diagnostics``). Returns
    ``{gamma, E_MAE_nn, E_MAE_pbe, D_nn, D_pbe, ED_nn, ED_pbe, beats_pbe}``.
    """
    ae_rows = list(ae_rows)
    density_rows = list(density_rows)
    if not ae_rows or not density_rows:
        raise ValueError(
            "combined_energy_density needs non-empty ae_rows and density_rows")

    e_mae_nn = sum(abs(r["err_nn"]) for r in ae_rows) / len(ae_rows)
    e_mae_pbe = sum(abs(r["err_pbe"]) for r in ae_rows) / len(ae_rows)
    d_nn = sum(r["density_rmse"] for r in density_rows) / len(density_rows)
    d_pbe = sum(r["density_rmse_pbe"] for r in density_rows) / len(density_rows)

    if d_pbe <= 0.0:
        raise ValueError("PBE density error non-positive; cannot self-calibrate gamma")
    gamma = e_mae_pbe / d_pbe

    def _harmonic(a, b):
        if a <= 0.0 or b <= 0.0:
            return 0.0
        return 2.0 / (1.0 / a + 1.0 / b)

    ed_nn = _harmonic(e_mae_nn, gamma * d_nn)
    ed_pbe = _harmonic(e_mae_pbe, gamma * d_pbe)  # == e_mae_pbe by construction
    result = {
        "gamma": gamma,
        "E_MAE_nn": e_mae_nn,
        "E_MAE_pbe": e_mae_pbe,
        "D_nn": d_nn,
        "D_pbe": d_pbe,
        "ED_nn": ed_nn,
        "ED_pbe": ed_pbe,
        "beats_pbe": ed_nn < ed_pbe,
    }
    # SCAN comparator on the SAME gamma (self-calibrated from PBE) so energy and
    # density share one kcal/mol scale across PBE/NN/SCAN. Emitted only when every
    # row carries the SCAN series (attach_scan_baseline was applied); otherwise
    # the result stays PBE-vs-NN only, for backward compatibility.
    if (all("err_scan" in r for r in ae_rows)
            and all("density_rmse_scan" in r for r in density_rows)):
        e_mae_scan = sum(abs(r["err_scan"]) for r in ae_rows) / len(ae_rows)
        d_scan = sum(r["density_rmse_scan"] for r in density_rows) / len(density_rows)
        ed_scan = _harmonic(e_mae_scan, gamma * d_scan)
        result["E_MAE_scan"] = e_mae_scan
        result["D_scan"] = d_scan
        result["ED_scan"] = ed_scan
        result["beats_scan"] = ed_nn < ed_scan
    return result


# ---------------------------------------------------------------------------
# Held-out generalization (does the tiny functional beat PBE OFF the training set?)
# ---------------------------------------------------------------------------

def heldout_points(hills: Sequence[str] = HELDOUT_MOLECULE_HILLS) -> list:
    """AE-as-reaction ``TrainingPoint``s for the held-out molecules.

    Same machinery as :func:`select_dfs_points` (so the geometry, spin, and AE
    reference all come verbatim from the DFS pool -- no fabricated values); only
    the Hill list differs. The returned points carry the compounds plus their
    constituent atom anchors (N, O), so a held-out eval can compute each
    functional's OWN self-consistent atomization energy.
    """
    return select_dfs_points(hills)


def heldout_comp_and_ae(chosen_points: Sequence, mol_specs: Sequence) -> tuple[dict, dict]:
    """``(comp_by_name, ae_ref_kcal)`` for the held-out molecules.

    ``comp_by_name`` maps each polyatomic's name to its atom composition (for
    :func:`self_consistent_ae`); ``ae_ref_kcal`` maps it to the pool's reference
    atomization energy in kcal/mol (via :func:`_ae_ref_kcalmol`, the same source
    the training path uses -- not a hand-typed literal).
    """
    comp_by_name = {ms.name: dict(ms.atom_composition)
                    for ms in molecule_specs(mol_specs)}
    ae_ref_kcal, _ = _ae_ref_kcalmol(chosen_points)
    ae_ref_kcal = {k: float(v) for k, v in ae_ref_kcal.items()
                   if k in comp_by_name and v is not None}
    return comp_by_name, ae_ref_kcal


def build_heldout_test_spec(*, arch, solver_cfg, mol_specs: Sequence,
                            model_checkpoint: str, output_dir: str, domain=DOMAIN,
                            metrics: Sequence[str] = (
                                "total_energy", "atomization_energy",
                                "density_rmse", "scf_convergence")) -> TestSpec:
    """A ``TestSpec`` that evaluates one trained checkpoint on the held-out pool.

    Like :func:`build_dfs_test_spec` but over an EXPLICIT ``mol_specs`` union
    (the held-out molecules + their N/O atom anchors) rather than the training
    spec's molecules, under the SAME FULL ``solver_cfg`` (so the orientation lock
    rides along and the held-out density is self-consistent + reproducible). No
    retraining: this only consumes the already-trained ``model_checkpoint``.
    """
    return TestSpec.from_dicts(
        model_checkpoint=model_checkpoint,
        arch=arch,
        molecules=tuple(mol_specs),
        metrics=tuple(metrics),
        atom_energies=dict(domain.atom_energies),
        solver_config=solver_cfg,
        output_dir=output_dir,
    )


def heldout_summary(per_model_combined: dict) -> dict:
    """Tally how many trained models beat PBE on the held-out set (pure logic).

    ``per_model_combined`` maps ``model_name -> combined_energy_density(...)``
    (each already computed from that model's held-out ``run_test`` records).
    Returns a per-model table plus the ``k/N`` beat-PBE counts on AE-MAE, mean
    density RMSE, and the combined energy-density error ``ED``.
    """
    rows = []
    for name, c in per_model_combined.items():
        row = {
            "model": name,
            "E_MAE_nn": c["E_MAE_nn"], "E_MAE_pbe": c["E_MAE_pbe"],
            "D_nn": c["D_nn"], "D_pbe": c["D_pbe"],
            "ED_nn": c["ED_nn"], "ED_pbe": c["ED_pbe"],
            "beats_ae": c["E_MAE_nn"] < c["E_MAE_pbe"],
            "beats_density": c["D_nn"] < c["D_pbe"],
            "beats_ed": bool(c["beats_pbe"]),
        }
        # SCAN comparator (present iff combined_energy_density saw a SCAN series).
        if "ED_scan" in c:
            row["E_MAE_scan"] = c["E_MAE_scan"]
            row["D_scan"] = c["D_scan"]
            row["ED_scan"] = c["ED_scan"]
            row["beats_ae_scan"] = c["E_MAE_nn"] < c["E_MAE_scan"]
            row["beats_density_scan"] = c["D_nn"] < c["D_scan"]
            row["beats_ed_scan"] = bool(c["beats_scan"])
        rows.append(row)
    n = len(rows)
    summary = {
        "rows": rows,
        "n_models": n,
        "n_beat_ae": sum(r["beats_ae"] for r in rows),
        "n_beat_density": sum(r["beats_density"] for r in rows),
        "n_beat_ed": sum(r["beats_ed"] for r in rows),
    }
    if rows and all("beats_ed_scan" in r for r in rows):
        summary["n_beat_ae_scan"] = sum(r["beats_ae_scan"] for r in rows)
        summary["n_beat_density_scan"] = sum(r["beats_density_scan"] for r in rows)
        summary["n_beat_ed_scan"] = sum(r["beats_ed_scan"] for r in rows)
    return summary
