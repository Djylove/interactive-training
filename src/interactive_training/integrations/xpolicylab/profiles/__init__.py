"""Robot- and recorder-specific dataset profile validators."""

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

__all__ = [
    "GR3_ANYGRASP_PROFILE_ID",
    "GR3_DAGGER_PROFILE_ID",
    "ROBOTWIN_CLEAN_PROFILE_ID",
    "Gr3AnygraspProfile",
    "Gr3DaggerEpisodeProfile",
    "RoboTwinCleanTaskProfile",
    "validate_gr3_anygrasp_manifest",
    "validate_gr3_dagger_manifest",
    "validate_robotwin_clean_manifest",
]
