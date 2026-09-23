"""xcquinox.pipeline: step3b-notebook functionality as a tested library."""

# Registries + helpers
from xcquinox.pipeline.descriptors import (
    DESCRIPTOR_REGISTRY, register_descriptor, make_descriptor, list_descriptors,
    Descriptor, CuspDescriptor, DMStatisticsDescriptor,
    assemble_descriptor_features,
)
from xcquinox.pipeline.constraints import (
    CONSTRAINT_REGISTRY, register_constraint, make_constraint, list_constraints,
    Constraint, LiebOxfordBound, UEGLimit, NonNegativeCorrelation, ScalingSymmetric,
)
from xcquinox.pipeline.losses import (
    LOSS_REGISTRY, register_loss, make_loss, list_losses,
    AlecLoss,
    AtomizationLoss, AtomizationPlusDMLoss, AtomizationPlusGridLoss,
    DeltaAELoss, DeltaAEPlusDMLoss, DeltaAEPlusGridLoss,
)
from xcquinox.pipeline.balancing import (
    LossMetric,
    BalancingConfig,
    LossNormConfig,
    TwoPhaseConfig,
    GradNormConfig,
)
from xcquinox.pipeline.evaluation import (
    METRIC_REGISTRY, register_metric, make_metric, list_metrics,
    Metric,
    TotalEnergyMetric, AtomizationEnergyMetric, DensityRMSEMetric, ConstraintViolationsMetric,
    PBEReferenceMetric, SCFConvergenceMetric,
    run_test,
)

# Config + architectures
from xcquinox.pipeline.config import (
    FeatureSpec, ArchitectureConfig,
    ARCHITECTURES, get_architecture, list_architectures,
    MoleculeSpec,
    PretrainSpec, TrainingSpec, TestSpec,
)

# Data / oneshot
from xcquinox.pipeline.data import (
    MoleculeData, precompute_fixed_density_data,
)
from xcquinox.pipeline.oneshot import (
    fixed_density_total_energy,
    oneshot_dm_prediction_fast,
    oneshot_grid_density,
    oneshot_total_energy,
    compute_exc_nn,
    compute_vxc_nn,
)

# Networks + model
from xcquinox.pipeline.networks import (
    AlecGGA_XNet, AlecGGA_CNet, create_network_pair,
)
from xcquinox.pipeline.models import AlecGGAModel

# Training / pretraining
from xcquinox.pipeline.pretrain import run_pretrain, from_legacy_step3b
from xcquinox.pipeline.train import run_training

# OEP utility
from xcquinox.pipeline.oep import OEPResult, run_oep_inversion, save_vxc_ref

# PBE-anchor regularization
from xcquinox.pipeline.pbe_anchor import (
    PBEAnchorSample,
    build_pbe_anchor_sample,
    pbe_anchor_loss,
)

# Parallel orchestration
from xcquinox.pipeline.parallel import (
    WorkerJob, WorkerResult, run_workers,
    build_pretrain_jobs, build_training_jobs,
)
