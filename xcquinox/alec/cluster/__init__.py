"""xcquinox.alec.cluster — HPC (SLURM) training-harness sub-package.

Thin re-export of the config layer. Other modules (domain, spec_builder,
preflight, ...) are not built yet; only grid_config names are exported.
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
]
