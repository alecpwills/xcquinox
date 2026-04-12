"""xcquinox.alec — step3b-notebook functionality as a tested library."""

# Registries + helpers
from xcquinox.alec.descriptors import (
    DESCRIPTOR_REGISTRY, register_descriptor, make_descriptor, list_descriptors,
    Descriptor, CuspDescriptor, DMStatisticsDescriptor,
    assemble_descriptor_features,
)
from xcquinox.alec.constraints import (
    CONSTRAINT_REGISTRY, register_constraint, make_constraint, list_constraints,
    Constraint, LiebOxfordBound, UEGLimit, NonNegativeCorrelation, ScalingSymmetric,
)
from xcquinox.alec.losses import (
    LOSS_REGISTRY, register_loss, make_loss, list_losses,
    AlecLoss,
    AtomizationLoss, AtomizationPlusDMLoss, AtomizationPlusGridLoss,
    DeltaAELoss, DeltaAEPlusDMLoss, DeltaAEPlusGridLoss,
)
from xcquinox.alec.evaluation import (
    METRIC_REGISTRY, register_metric, make_metric, list_metrics,
    Metric,
    TotalEnergyMetric, AtomizationEnergyMetric, DensityRMSEMetric, ConstraintViolationsMetric,
    run_test,
)

# Config + architectures
from xcquinox.alec.config import (
    FeatureSpec, ArchitectureConfig,
    ARCHITECTURES, get_architecture, list_architectures,
    MoleculeSpec,
    PretrainSpec, TrainingSpec, TestSpec,
)

# Data / oneshot
from xcquinox.alec.data import (
    MoleculeData, precompute_fixed_density_data,
)
from xcquinox.alec.oneshot import (
    fixed_density_total_energy,
    oneshot_dm_prediction_fast,
    oneshot_grid_density,
    oneshot_total_energy,
    compute_exc_nn,
    compute_vxc_nn,
)

# Networks + model
from xcquinox.alec.networks import (
    AlecGGA_XNet, AlecGGA_CNet, create_network_pair,
)
from xcquinox.alec.models import AlecGGAModel

# Training / pretraining
from xcquinox.alec.pretrain import run_pretrain, from_legacy_step3b
from xcquinox.alec.train import run_training

# Parallel orchestration
from xcquinox.alec.parallel import (
    WorkerJob, WorkerResult, run_workers,
    build_pretrain_jobs, build_training_jobs,
)
