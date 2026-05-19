"""xcquinox.alec.cluster — HPC (SLURM) training-harness sub-package.

Thin re-export of the config, domain and materialize layers. Other modules
(spec_builder, preflight, ...) are not built yet; only grid_config + domain +
materialize names are exported.
"""
from xcquinox.alec.cluster.grid_config import (
    GridConfig,
    GridCell,
    SweepAxes,
    SolverNamed,
    HyperParams,
    InputPaths,
    ClusterResources,
    load_grid_config,
    expand_grid,
    validate_grid_semantics,
    VALID_METRICS,
)
from xcquinox.alec.cluster.domain import (
    DomainProfile,
    get_domain_profile,
    DOMAIN_PROFILES,
    ATOMIC_ENERGIES_CHAKRAVORTY,
    KCAL_PER_HA,
    bh76_meta_to_loss_dict,
    ip13_meta_to_loss_dict,
)
from xcquinox.alec.cluster.materialize import (
    write_spec_atomic,
    materialize_specs,
    write_manifest,
)
from xcquinox.alec.cluster.spec_builder import (
    build_training_specs,
    build_test_spec,
    pool_fingerprint,
    atoms_to_pyscf_str,
    atoms_to_mol_spec,
    build_targets,
    classify_aux_only,
)
from xcquinox.alec.cluster.inputs import (
    prepare_inputs,
    StagedInputs,
)
from xcquinox.alec.cluster.job_tracking import (
    reduce_outcomes,
    append_job_record,
    read_job_records,
    mark_superseded,
    SlurmTransientError,
    _run_slurm,
)
from xcquinox.alec.cluster.submit import (
    render_sbatch,
    submit_jobs,
)
from xcquinox.alec.cluster._preflight import (
    main as preflight_main,
)
from xcquinox.alec.cluster._train_task import (
    main as train_task_main,
)

__all__ = [
    "GridConfig",
    "GridCell",
    "SweepAxes",
    "SolverNamed",
    "HyperParams",
    "InputPaths",
    "ClusterResources",
    "load_grid_config",
    "expand_grid",
    "validate_grid_semantics",
    "VALID_METRICS",
    "DomainProfile",
    "get_domain_profile",
    "DOMAIN_PROFILES",
    "ATOMIC_ENERGIES_CHAKRAVORTY",
    "KCAL_PER_HA",
    "bh76_meta_to_loss_dict",
    "ip13_meta_to_loss_dict",
    "write_spec_atomic",
    "materialize_specs",
    "write_manifest",
    "build_training_specs",
    "build_test_spec",
    "pool_fingerprint",
    "atoms_to_pyscf_str",
    "atoms_to_mol_spec",
    "build_targets",
    "classify_aux_only",
    "prepare_inputs",
    "StagedInputs",
    "reduce_outcomes",
    "append_job_record",
    "read_job_records",
    "mark_superseded",
    "SlurmTransientError",
    "_run_slurm",
    "render_sbatch",
    "submit_jobs",
    "preflight_main",
    "train_task_main",
]
