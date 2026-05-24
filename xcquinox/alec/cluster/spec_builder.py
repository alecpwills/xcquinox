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
the EXISTING subset ledger.

The subset_ledger schema
------------------------
``build_training_specs`` consumes the EXISTING ``subset_index_log.json`` dict
produced by the (already-finished) subset-selection pre-process — handed
through verbatim by :func:`xcquinox.alec.cluster.inputs.prepare_inputs`. Its
schema is::

    {
        "<metric>/<subset_size>": {            # e.g. "l2/3"
            "chosen_indices": [<int>, ...],    # provenance only
            "metric_value": <float>,           # provenance only
            "point_kinds": [<kind>, ...],      # provenance only
            "point_names": [<name>, ...],      # the chosen TrainingPoint names
            "tag": "bin<NN>"                   # provenance only
        },
        ...
    }

Notes:
- The harness resolves a cell's training points **by name** from
  ``entry["point_names"]`` against the supplied ``points`` pool. It does NOT
  use ``chosen_indices`` — those are positional into a pool list and are not
  robust to pool reordering; ``point_names`` is the stable key.
- There is NO ``pool_fingerprint`` and NO top-level wrapper — the ledger is a
  bare dict of ``"<metric>/<r>"`` entries. The safety net against a stale
  ledger is name resolution itself: if the pool genuinely differs, a
  ``point_name`` will not resolve and ``build_training_specs`` fails loudly.
- An entry's ``point_names`` MUST be present and non-empty. A missing key or
  an empty list is treated as a malformed ledger entry and raises ``ValueError``
  immediately — a real subset always has ≥1 point.
"""
import os
import warnings

from xcquinox.alec.config import MoleculeSpec, TrainingSpec, TestSpec
from xcquinox.alec.training_points import species_union_from_points
from xcquinox.alec.solver import SolverConfig, SolverMode, FeaturePolicy
from xcquinox.alec import get_architecture


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

def _ledger_key(metric: str, subset_size: int) -> str:
    """The ``subset_index_log.json`` entry key for a ``(metric, r)`` pair."""
    return f"{metric}/{int(subset_size)}"


def _coerce_enum(enum_cls, token):
    """Resolve ``token`` to an ``enum_cls`` member by VALUE or by NAME.

    Config files spell solver mode / feature policy as the uppercase enum NAME
    (e.g. ``ONESHOT``, ``FULL``, ``REASSEMBLE`` — matching the notebook's
    ``SolverMode.ONESHOT`` references), while unit tests and the enums' own
    ``__call__`` use the lowercase VALUE (``oneshot``/``full``/``reassemble``).
    Accept either so a config's name-form string does not blow up at
    spec-build time (the preflight stage). Raises a clear ``ValueError`` naming
    the enum and the valid options when ``token`` is neither.
    """
    try:
        return enum_cls(token)          # by value, e.g. "oneshot"
    except ValueError:
        pass
    try:
        return enum_cls[token]          # by name, e.g. "ONESHOT"
    except KeyError:
        valid = [f"{m.name}/{m.value}" for m in enum_cls]
        raise ValueError(
            f"{token!r} is not a valid {enum_cls.__name__} — expected one of "
            f"(name/value): {valid}"
        )


def _solver_config_from_named(named) -> SolverConfig:
    """Materialize a :class:`SolverConfig` from a :class:`SolverNamed`.

    ``SolverNamed`` stores ``mode`` / ``feature_policy`` as plain strings (the
    config's uppercase enum NAME or the lowercase VALUE); this coerces them to
    the ``SolverMode`` / ``FeaturePolicy`` enums, accepting either spelling.
    """
    mode = _coerce_enum(SolverMode, named.mode)
    fp = (
        _coerce_enum(FeaturePolicy, named.feature_policy)
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
        The EXISTING ``subset_index_log.json`` dict — see the module docstring
        for the schema. Keys are ``"<metric>/<subset_size>"`` strings; the
        cell's training points are resolved by name from ``point_names``.
    cfg : GridConfig
        Harness config — supplies the swept axes, hyperparameters, named
        solver configs, and the ``pretrain`` stage config.
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
        If a ``(metric, subset_size)`` grid cell has no ledger entry, or if a
        ledger entry names a training point absent from the ``points`` pool.
    """
    from xcquinox.alec.cluster.grid_config import (
        expand_grid, pretrain_checkpoint_dir,
    )
    from xcquinox.alec.balancing import GradNormConfig

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
        ledger_key = _ledger_key(cell.metric, cell.subset_size)
        if ledger_key not in subset_ledger:
            raise ValueError(
                f"subset_ledger has no entry for (metric={cell.metric!r}, "
                f"subset_size={cell.subset_size}) — key {ledger_key!r} is "
                "absent. Every grid cell's (metric, subset_size) pair must be "
                "present in the existing subset_index_log.json ledger."
            )
        entry = subset_ledger[ledger_key]
        if "point_names" not in entry or not entry["point_names"]:
            raise ValueError(
                f"subset_ledger entry {ledger_key!r} is malformed: "
                "'point_names' key is absent or empty. Every ledger entry "
                "must carry a non-empty 'point_names' list — a real subset "
                "always has ≥1 point."
            )
        point_names = entry["point_names"]
        missing = [pn for pn in point_names if pn not in points_by_name]
        if missing:
            raise ValueError(
                f"subset_ledger entry {ledger_key!r} names training points "
                f"not present in the pool: {missing}. The ledger and the "
                "training-point pool are out of sync — the ledger was "
                "selected against a different pool; regenerate it (or pass "
                "the matching pool)."
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
            # The pretrain stage writes one checkpoint per architecture to the
            # job-scoped ``<pretrain_root>/<run_id>/<arch>/``; that directory IS
            # this cell's pretrained checkpoint. Derived through the SAME helper
            # the pretrain worker uses so the two sides cannot drift. validate()
            # only checks the path when the dir exists, so building specs before
            # the pretrain stage runs is fine — the preflight runs
            # pretrain-then-validate.
            pretrain_checkpoint=pretrain_checkpoint_dir(
                cfg.pretrain.pretrain_root, run_dir, cell.arch
            ),
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


def build_test_spec(
    training_spec,
    run_dir,
    idx,
    domain,
    *,
    holdout_molecule_names: "tuple | None" = None,
) -> TestSpec:
    """Build the :class:`TestSpec` matching a trained :class:`TrainingSpec`.

    By default, eval molecules are taken **directly** from
    ``training_spec.molecules`` (post the mixed-pool refactor, that IS the chosen
    species union — no ``subset.traj`` is read).  **This is in-distribution
    evaluation** — the eval set equals the training set.  It is not a
    generalization estimate.  A :class:`RuntimeWarning` is emitted whenever this
    default path is used, so the in-distribution nature is never silently
    mistaken for held-out performance.

    To evaluate on a held-out or external molecule set, pass
    ``holdout_molecule_names`` — a tuple of :class:`~xcquinox.alec.config.MoleculeSpec`
    objects.  When provided, the returned :class:`TestSpec` uses those molecules
    instead of the training set, and no warning is emitted.

    The ``reference_ae_kcalmol`` metric kwarg is built from
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
    holdout_molecule_names : tuple[MoleculeSpec, ...] | None, optional
        When provided, the returned :class:`TestSpec` evaluates on these
        molecules instead of the training set.  When ``None`` (default), the
        training molecules are used and a :class:`RuntimeWarning` is emitted to
        flag the in-distribution nature of the evaluation.

    Returns
    -------
    TestSpec
    """
    # Reconstruct the zero-padded spec dir from the same scheme materialize
    # uses, but anchored on run_dir so the path is absolute & deterministic
    # regardless of how the TrainingSpec's checkpoint_dir was set.
    base = os.path.basename(training_spec.checkpoint_dir.rstrip("/"))
    ckpt_dir = os.path.join(
        os.path.abspath(run_dir), "checkpoints", base
    )
    model_checkpoint = os.path.join(ckpt_dir, "model.eqx")
    output_dir = os.path.join(ckpt_dir, "eval")

    if holdout_molecule_names is None:
        eval_molecules = training_spec.molecules
        warnings.warn(
            "build_test_spec: eval molecules are the TRAINING molecules "
            "(in-distribution evaluation — not a held-out generalization "
            "estimate). Pass holdout_molecule_names to evaluate on an "
            "external set.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        eval_molecules = holdout_molecule_names

    targets = training_spec.targets_dict
    # Exclude AUX-ONLY species (BH76/IP13 reaction polyatomics with no real AE
    # reference) — they carry a 0.0 placeholder target. Mirrors the training
    # loss, which drops them via classify_aux_only; without this the eval AE
    # metric scores their full atomization energy against a 0.0 reference (the
    # CH4 ~+440 / HF ~+150 kcal/mol artifact). aux_only_names is carried in the
    # TrainingSpec's loss_kwargs by build_training_specs.
    aux_only = set(training_spec.loss_kwargs_dict.get("aux_only_names", ()))
    reference_ae_kcalmol: dict = {}
    for ms in training_spec.molecules:
        comp_sum = sum(dict(ms.atom_composition).values())
        if comp_sum > 1 and ms.name in targets and ms.name not in aux_only:
            reference_ae_kcalmol[ms.name] = targets[ms.name] * domain.kcal_per_ha

    return TestSpec.from_dicts(
        arch=training_spec.arch,
        model_checkpoint=model_checkpoint,
        molecules=eval_molecules,
        metrics=("total_energy", "atomization_energy",
                 "density_rmse", "scf_convergence"),
        metric_kwargs={
            "atomization_energy": {"reference_ae_kcalmol": reference_ae_kcalmol},
        },
        atom_energies=dict(domain.atom_energies),
        output_dir=output_dir,
        solver_config=training_spec.solver_config,
    )
