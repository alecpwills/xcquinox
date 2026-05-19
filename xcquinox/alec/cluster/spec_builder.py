"""xcquinox.alec.cluster.spec_builder — generic spec assembly for the HPC harness.

This module is the de-notebooked, domain-agnostic extraction of "Cell A" of
``notebooks/_build_step7_notebook.py`` — the logic that assembles
:class:`xcquinox.alec.TrainingSpec` (and the matching :class:`TestSpec`)
objects from a chosen subset of :class:`TrainingPoint` objects.

It deliberately carries no ``step7`` / notebook-stage naming: a
:class:`~xcquinox.alec.cluster.domain.DomainProfile` supplies every physics
table (atomic energies, kcal/Ha, BH76/IP13 extractors), a
:class:`~xcquinox.alec.cluster.grid_config.GridConfig` supplies the swept axes
+ hyperparameters, and the caller supplies the full ``TrainingPoint`` pool plus
a *name-based* subset ledger.

The subset_ledger schema (canonical — the later ``inputs.py`` writer MUST match)
----------------------------------------------------------------------------
``build_training_specs`` expects ``subset_ledger`` to be a dict of the form::

    {
        "pool_fingerprint": "<hex sha256 digest>",
        "entries": {
            (metric, subset_size): {
                "metric": metric,                 # str, e.g. "l2"
                "subset_size": subset_size,        # int
                "point_names": [name, name, ...],  # chosen TrainingPoint names
            },
            ...
        },
    }

Notes:
- The ledger is **NAME-BASED**: each entry stores the chosen training points by
  their ``TrainingPoint.name`` (NOT by index into the pool). Names are resolved
  back to ``TrainingPoint`` objects by identity lookup in ``points``.
- ``pool_fingerprint`` is a stable :func:`pool_fingerprint` digest of the pool
  the subsets were selected from. ``build_training_specs`` asserts it matches
  the fingerprint of the ``points`` it is given, and fails loudly on mismatch.
- The ``entries`` keys MAY be ``(metric, subset_size)`` tuples (the in-memory
  form) or the JSON-string form ``"<metric>:<subset_size>"`` — the lookup
  helper accepts both so a JSON-loaded ledger works without re-keying.
- An entry's ``point_names`` MAY be empty (a valid degenerate subset); the
  resulting spec will simply have whatever ``species_union_from_points`` yields.
"""
import hashlib
import os

from xcquinox.alec.config import MoleculeSpec, TrainingSpec, TestSpec
from xcquinox.alec.training_points import species_union_from_points
from xcquinox.alec.solver import SolverConfig, SolverMode, FeaturePolicy
from xcquinox.alec import get_architecture


# ---------------------------------------------------------------------------
# Pool fingerprinting
# ---------------------------------------------------------------------------

def pool_fingerprint(points) -> str:
    """Stable SHA-256 digest of a ``TrainingPoint`` pool.

    The digest is computed over the *sorted* list of per-point
    ``(name, kind, sorted-species-names)`` triples, so it is independent of
    the order in which the points are presented (a permuted pool hashes the
    same) and depends only on the identity of the points and the species they
    carry.

    Parameters
    ----------
    points : Sequence[TrainingPoint]
        The training-point pool.

    Returns
    -------
    str
        Lower-case hex SHA-256 digest.
    """
    triples = []
    for tp in points:
        species_names = tuple(sorted(s.info["name"] for s in tp.species))
        triples.append((str(tp.name), str(tp.kind), species_names))
    triples.sort()
    h = hashlib.sha256()
    for name, kind, species_names in triples:
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(kind.encode("utf-8"))
        h.update(b"\x00")
        for sn in species_names:
            h.update(sn.encode("utf-8"))
            h.update(b"\x01")
        h.update(b"\x02")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# ASE Atoms -> MoleculeSpec helpers (ports of the notebook's _-prefixed forms)
# ---------------------------------------------------------------------------

def atoms_to_pyscf_str(at) -> str:
    """Convert an ASE ``Atoms`` object's positions to a pyscf-format atom string.

    Positions are emitted in Angstrom (pyscf's default unit).
    """
    syms = at.get_chemical_symbols()
    pos = at.get_positions()  # Angstrom, pyscf default unit
    parts = [
        f"{s} {x:.6f} {y:.6f} {z:.6f}"
        for s, (x, y, z) in zip(syms, pos)
    ]
    return "; ".join(parts)


def atoms_to_mol_spec(at, basis, grid_level, external_refs_dir, name=None) -> MoleculeSpec:
    """Convert an ASE ``Atoms`` entry to a :class:`MoleculeSpec`.

    Faithful port of the notebook's ``_atoms_to_mol_spec``. The
    ``MoleculeSpec.name`` is the Hill formula by default (consistent across the
    training pool, BH76 reaction species, and IP13 ionization pairs); the
    optional ``name`` kwarg overrides it (used e.g. for IP13 cations like
    ``'Li+'`` / ``'C+'`` which carry an explicit ``info['name']``).

    The external CCSD reference ``.npz`` is wired via ``external_data_path``
    when ``<external_refs_dir>/<name>.npz`` exists on disk; otherwise it stays
    ``None`` (species outside the pre-compute set, or a first run before the
    reference files were generated).

    Parameters
    ----------
    at : ase.Atoms
        The molecule geometry. ``at.info`` may carry ``name`` / ``dfs_hill`` /
        ``charge`` / ``spin``.
    basis : str
        pyscf basis-set name.
    grid_level : int | None
        pyscf DFT grid level pinned on the spec.
    external_refs_dir : str | os.PathLike
        Directory holding per-species ``<name>.npz`` external reference files.
    name : str | None
        Explicit MoleculeSpec name override.

    Returns
    -------
    MoleculeSpec
    """
    from collections import Counter

    if name is None:
        # Lookup order matches the notebook: info['name'] (set explicitly by
        # TrainingPoint species — IP13 cations carry 'Li+'/'C+', AE compounds
        # carry their Hill formula) -> dfs_hill -> Hill formula.
        name = (
            at.info.get("name")
            or at.info.get("dfs_hill")
            or at.get_chemical_formula()
        )
    charge = int(at.info.get("charge", 0))
    spin = int(at.info.get("spin", 0))
    atom_str = atoms_to_pyscf_str(at)
    comp_raw = Counter(at.get_chemical_symbols())
    # Wire the external CCSD reference path when present; None otherwise.
    ext_npz = os.path.join(str(external_refs_dir), f"{name}.npz")
    external_data_path = ext_npz if os.path.isfile(ext_npz) else None
    return MoleculeSpec.from_dict(
        name=name,
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=spin,
        atom_composition=dict(comp_raw),
        grid_level=grid_level,
        external_data_path=external_data_path,
    )


# ---------------------------------------------------------------------------
# targets / aux-only classification (ports of the notebook's Cell A logic)
# ---------------------------------------------------------------------------

def classify_aux_only(mol_specs, ae_ref_kcalmol) -> tuple:
    """Return the sorted tuple of polyatomic ``MoleculeSpec`` names that are
    **aux-only** — i.e. present so the BH76 channel can compute reaction
    energies, but NOT members of the AE channel.

    A polyatomic (composition sum > 1) is aux-only iff its name has no entry in
    ``ae_ref_kcalmol`` (the per-AE-point reference dict). Without this
    classification, ``_ae_losses`` would include those species with a 0.0
    target and the relative-error denominator would blow up.
    """
    return tuple(sorted(
        ms.name for ms in mol_specs
        if sum(dict(ms.atom_composition).values()) > 1
        and ms.name not in ae_ref_kcalmol
    ))


def build_targets(mol_specs, ae_ref_kcalmol, domain) -> dict:
    """Build the ``targets`` dict for a :class:`TrainingSpec`.

    Faithful port of the notebook's Cell A ``targets`` construction:

    - **Single atom** (composition sum == 1): the ``domain.atom_energies``
      total energy for that element (same anchor as ``atom_energies``); 0.0
      fallback if the element is absent from the table.
    - **AE compound** (polyatomic with an ``ae_ref_kcalmol`` entry): the
      atomization energy converted Ha = kcal / ``domain.kcal_per_ha``.
    - **Aux polyatomic** (polyatomic with NO ``ae_ref_kcalmol`` entry — a BH76
      reaction species): 0.0 placeholder. ``classify_aux_only`` excludes it
      from the AE channel so the placeholder is never read by the loss.

    Parameters
    ----------
    mol_specs : Sequence[MoleculeSpec]
    ae_ref_kcalmol : dict[str, float | None]
        AE reference values (kcal/mol) keyed by MoleculeSpec name.
    domain : DomainProfile
        Supplies ``atom_energies`` and ``kcal_per_ha``.

    Returns
    -------
    dict[str, float]
    """
    targets: dict = {}
    for ms in mol_specs:
        comp_sum = sum(dict(ms.atom_composition).values())
        if comp_sum == 1:
            sym = next(iter(dict(ms.atom_composition)))
            targets[ms.name] = domain.atom_energies.get(sym, 0.0)
        else:
            ae_kc = ae_ref_kcalmol.get(ms.name)
            targets[ms.name] = (
                ae_kc / domain.kcal_per_ha if ae_kc is not None else 0.0
            )
    return targets


# ---------------------------------------------------------------------------
# Ledger lookup + solver-config materialization
# ---------------------------------------------------------------------------

def _ledger_entries(subset_ledger) -> dict:
    """Return the ``entries`` mapping out of a subset ledger.

    Accepts either the documented ``{"pool_fingerprint": ..., "entries": {...}}``
    form, or — defensively — a bare entries dict (no ``entries`` key).
    """
    if "entries" in subset_ledger:
        return subset_ledger["entries"]
    return {
        k: v for k, v in subset_ledger.items()
        if k != "pool_fingerprint"
    }


def _lookup_entry(entries, metric, subset_size):
    """Look up a ledger entry for ``(metric, subset_size)``.

    Accepts both the in-memory tuple key ``(metric, subset_size)`` and the
    JSON-string key ``"<metric>:<subset_size>"`` so a JSON-loaded ledger works
    without re-keying.
    """
    tuple_key = (metric, subset_size)
    if tuple_key in entries:
        return entries[tuple_key]
    str_key = f"{metric}:{subset_size}"
    if str_key in entries:
        return entries[str_key]
    return None


def _solver_config_from_named(named) -> SolverConfig:
    """Materialize a :class:`SolverConfig` from a :class:`SolverNamed`.

    ``SolverNamed`` stores ``mode`` / ``feature_policy`` as plain strings; this
    coerces them to the ``SolverMode`` / ``FeaturePolicy`` enums.
    """
    mode = SolverMode(named.mode)
    fp = (
        FeaturePolicy(named.feature_policy)
        if named.feature_policy is not None
        else None
    )
    return SolverConfig(
        mode=mode,
        max_cycles=named.max_cycles,
        feature_policy=fp,
    )


def _checkpoint_dir(run_dir: str, idx: int, n: int) -> str:
    """Absolute ``<run_dir>/checkpoints/spec_<idx>`` with the same zero-pad
    scheme as ``materialize._spec_filename`` (``max(4, len(str(N-1)))``)."""
    width = max(4, len(str(n - 1))) if n > 0 else 4
    return os.path.join(
        os.path.abspath(run_dir), "checkpoints", f"spec_{idx:0{width}d}"
    )


# ---------------------------------------------------------------------------
# Main builders
# ---------------------------------------------------------------------------

def build_training_specs(points, subset_ledger, cfg, domain, run_dir, cells=None):
    """Assemble one :class:`TrainingSpec` per :class:`GridCell`.

    Parameters
    ----------
    points : Sequence[TrainingPoint]
        The full training-point pool (e.g. from ``build_dfs_pool_points``).
    subset_ledger : dict
        Name-based subset ledger — see the module docstring for the schema.
        Its ``pool_fingerprint`` MUST match :func:`pool_fingerprint` of
        ``points`` (a mismatch raises :class:`ValueError`).
    cfg : GridConfig
        Harness config — supplies the swept axes, hyperparameters, and named
        solver configs.
    domain : DomainProfile
        Physics tables (atom energies, kcal/Ha, BH76/IP13 extractors,
        regularizer atom symbols).
    run_dir : str
        Absolute run directory; checkpoint dirs are placed under it.
    cells : list[GridCell] | None
        Optional subset of grid cells to build. ``None`` (default) builds the
        full ``expand_grid(cfg)``.

    Returns
    -------
    list[tuple[GridCell, TrainingSpec]]
        In index order. Construction is side-effect-free — ``from_dicts`` does
        not call ``validate()``; a later module runs ``spec.validate()``.

    Raises
    ------
    ValueError
        If the ledger's ``pool_fingerprint`` does not match ``points``, or if
        a ``(metric, subset_size)`` grid cell has no ledger entry.
    """
    from xcquinox.alec.cluster.grid_config import expand_grid
    from xcquinox.alec.balancing import GradNormConfig

    # --- fingerprint guard --------------------------------------------------
    actual_fp = pool_fingerprint(points)
    ledger_fp = subset_ledger.get("pool_fingerprint")
    if ledger_fp is None:
        raise ValueError(
            "subset_ledger is missing the required 'pool_fingerprint' key; "
            "the ledger writer must record pool_fingerprint(points) so the "
            "spec builder can verify the ledger matches the pool it is given."
        )
    if ledger_fp != actual_fp:
        raise ValueError(
            "subset_ledger pool_fingerprint mismatch: the ledger was built "
            f"against a pool with fingerprint {ledger_fp!r}, but the `points` "
            f"passed to build_training_specs fingerprint to {actual_fp!r}. "
            "The ledger and the training-point pool are out of sync — "
            "regenerate the subset ledger against the current pool."
        )

    entries = _ledger_entries(subset_ledger)

    # --- name -> TrainingPoint resolution ----------------------------------
    points_by_name: dict = {}
    for tp in points:
        if tp.name in points_by_name:
            raise ValueError(
                f"training-point pool has a duplicate name {tp.name!r}; "
                "names must be unique so the name-based ledger resolves "
                "unambiguously."
            )
        points_by_name[tp.name] = tp

    # --- per-AE-point reference dict (kcal/mol) ----------------------------
    ae_ref_kcalmol = {
        tp.name: tp.metadata.get("ae_kcalmol")
        for tp in points
        if tp.kind == "ae"
    }

    if cells is None:
        cells = expand_grid(cfg)
    n = len(cells)

    hp = cfg.hyperparams
    out: list = []
    for idx, cell in enumerate(cells):
        entry = _lookup_entry(entries, cell.metric, cell.subset_size)
        if entry is None:
            raise ValueError(
                f"subset_ledger has no entry for (metric={cell.metric!r}, "
                f"subset_size={cell.subset_size}); every grid cell's "
                "(metric, subset_size) pair must be present in the ledger."
            )
        point_names = entry.get("point_names", [])
        missing = [pn for pn in point_names if pn not in points_by_name]
        if missing:
            raise ValueError(
                f"subset_ledger entry for (metric={cell.metric!r}, "
                f"subset_size={cell.subset_size}) names training points not "
                f"present in the pool: {missing}."
            )
        chosen_points = [points_by_name[pn] for pn in point_names]

        # mol_specs = deduped species union of every chosen point.
        sp_atoms = species_union_from_points(chosen_points)
        mol_specs = tuple(
            atoms_to_mol_spec(
                at,
                basis=cfg.inputs.basis,
                grid_level=cfg.inputs.grid_level,
                external_refs_dir=cfg.inputs.external_refs_dir,
            )
            for at in sp_atoms
        )

        targets = build_targets(mol_specs, ae_ref_kcalmol, domain)
        aux_only_names = classify_aux_only(mol_specs, ae_ref_kcalmol)

        # BH76 / IP13 loss inputs come ONLY from the chosen points.
        bh76_ha = [
            domain.bh76_meta_to_loss_dict(tp)
            for tp in chosen_points
            if tp.kind == "bh76"
        ]
        ip13_ha = [
            domain.ip13_meta_to_loss_dict(tp)
            for tp in chosen_points
            if tp.kind == "ip13"
        ]

        solver_cfg = _solver_config_from_named(cfg.solvers[cell.solver])

        loss_kwargs = {
            "bh76_reactions": bh76_ha,
            "ip13_pairs": ip13_ha,
            "aux_only_names": aux_only_names,
            "regularize_atom_syms": tuple(domain.regularize_atom_syms),
            "solver_config": solver_cfg,
            "vxc_weight": hp.vxc_weight,
            "density_weight": hp.density_weight,
        }

        spec = TrainingSpec.from_dicts(
            arch=get_architecture(cell.arch),
            molecules=mol_specs,
            targets=targets,
            atom_energies=dict(domain.atom_energies),
            loss_name=cell.loss,
            loss_kwargs=loss_kwargs,
            solver_config=solver_cfg,
            pretrain_checkpoint=cfg.inputs.pretrain_checkpoint,
            checkpoint_dir=_checkpoint_dir(run_dir, idx, n),
            n_steps=hp.n_steps,
            lr_start=hp.lr_start,
            lr_end=hp.lr_end,
            lr_decay_start=hp.lr_decay_start,
            grad_clip=hp.grad_clip,
            seed=hp.seed,
            balancing=GradNormConfig(alpha=hp.gradnorm_alpha),
            pbe_anchor_weight=hp.pbe_anchor_weight,
            pbe_anchor_sample=None,
            require_atom_anchors=False,
        )
        out.append((cell, spec))
    return out


def build_test_spec(training_spec, run_dir, idx, domain) -> TestSpec:
    """Build the :class:`TestSpec` matching a trained :class:`TrainingSpec`.

    Eval molecules are taken **directly** from ``training_spec.molecules`` (post
    the mixed-pool refactor, that IS the chosen species union — no ``subset.traj``
    is read). The ``reference_ae_kcalmol`` metric kwarg is built from
    ``training_spec.targets_dict`` (Ha) × ``domain.kcal_per_ha`` for compound
    molecules only.

    Parameters
    ----------
    training_spec : TrainingSpec
    run_dir : str
        Absolute run directory; checkpoint/eval dirs are placed under it.
    idx : int
        The spec's array-task index — selects ``spec_<idx>``.
    domain : DomainProfile
        Supplies ``atom_energies`` and ``kcal_per_ha``.

    Returns
    -------
    TestSpec
    """
    spec_dir = os.path.dirname(os.path.abspath(training_spec.checkpoint_dir))
    # Reconstruct the zero-padded spec dir from the same scheme materialize
    # uses, but anchored on run_dir so the path is absolute & deterministic
    # regardless of how the TrainingSpec's checkpoint_dir was set.
    base = os.path.basename(training_spec.checkpoint_dir.rstrip("/"))
    ckpt_dir = os.path.join(
        os.path.abspath(run_dir), "checkpoints", base
    )
    model_checkpoint = os.path.join(ckpt_dir, "model.eqx")
    output_dir = os.path.join(ckpt_dir, "eval")

    targets = training_spec.targets_dict
    reference_ae_kcalmol: dict = {}
    for ms in training_spec.molecules:
        comp_sum = sum(dict(ms.atom_composition).values())
        if comp_sum > 1 and ms.name in targets:
            reference_ae_kcalmol[ms.name] = targets[ms.name] * domain.kcal_per_ha

    return TestSpec.from_dicts(
        arch=training_spec.arch,
        model_checkpoint=model_checkpoint,
        molecules=training_spec.molecules,
        metrics=("total_energy", "atomization_energy",
                 "density_rmse", "scf_convergence"),
        metric_kwargs={
            "atomization_energy": {"reference_ae_kcalmol": reference_ae_kcalmol},
        },
        atom_energies=dict(domain.atom_energies),
        output_dir=output_dir,
        solver_config=training_spec.solver_config,
    )
