"""Optional XPolicyLab experiment orchestration integration."""

from interactive_training.integrations.xpolicylab.contracts import (
    ArtifactManifest,
    CheckpointArtifact,
    DatasetBinding,
    DatasetManifest,
    EvaluationSpec,
    EvaluationSummary,
    ExperimentSpec,
    TrainSpec,
    TrialResult,
)
from interactive_training.integrations.xpolicylab.datasets import (
    load_dataset_manifest,
    validate_dataset_manifest_profile,
    verify_dataset_files,
)
from interactive_training.integrations.xpolicylab.experiment import (
    EmbodiedRoundResult,
    XPolicyExperiment,
)
from interactive_training.integrations.xpolicylab.policies import (
    ControlPolicy,
    PromotionStage,
)
from interactive_training.integrations.xpolicylab.profiles.gr3 import (
    GR3_DAGGER_PROFILE_ID,
    Gr3DaggerEpisodeProfile,
    validate_gr3_dagger_manifest,
)
from interactive_training.integrations.xpolicylab.profiles.gr3_anygrasp import (
    GR3_ANYGRASP_PROFILE_ID,
    Gr3AnygraspProfile,
    validate_gr3_anygrasp_manifest,
)
from interactive_training.integrations.xpolicylab.profiles.robotwin import (
    ROBOTWIN_CLEAN_PROFILE_ID,
    RoboTwinCleanTaskProfile,
    validate_robotwin_clean_manifest,
)
from interactive_training.integrations.xpolicylab.reporter import (
    TrialAggregator,
    load_trial_results,
)
from interactive_training.integrations.xpolicylab.runner import (
    RunnerPolicy,
    XPolicyExperimentRunner,
)

__all__ = [
    "GR3_ANYGRASP_PROFILE_ID",
    "GR3_DAGGER_PROFILE_ID",
    "ROBOTWIN_CLEAN_PROFILE_ID",
    "ArtifactManifest",
    "CheckpointArtifact",
    "ControlPolicy",
    "DatasetBinding",
    "DatasetManifest",
    "EmbodiedRoundResult",
    "EvaluationSpec",
    "EvaluationSummary",
    "ExperimentSpec",
    "Gr3AnygraspProfile",
    "Gr3DaggerEpisodeProfile",
    "PromotionStage",
    "RoboTwinCleanTaskProfile",
    "RunnerPolicy",
    "TrainSpec",
    "TrialAggregator",
    "TrialResult",
    "XPolicyExperiment",
    "XPolicyExperimentRunner",
    "load_dataset_manifest",
    "load_trial_results",
    "validate_dataset_manifest_profile",
    "validate_gr3_anygrasp_manifest",
    "validate_gr3_dagger_manifest",
    "validate_robotwin_clean_manifest",
    "verify_dataset_files",
]
