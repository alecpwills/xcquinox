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
]
