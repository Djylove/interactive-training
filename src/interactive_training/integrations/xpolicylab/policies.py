"""Safety policy and artifact promotion rules for embodied experiments."""

from __future__ import annotations

from enum import Enum


class PromotionStage(str, Enum):
    TRAINED = "trained"
    OFFLINE_VALIDATED = "offline_validated"
    DEBUG_VALIDATED = "debug_validated"
    SIMULATOR_CANDIDATE = "simulator_candidate"
    SIMULATOR_APPROVED = "simulator_approved"
    ROBOT_SHADOW_CANDIDATE = "robot_shadow_candidate"
    ROBOT_SHADOW_APPROVED = "robot_shadow_approved"
    ROBOT_ENFORCE_CANDIDATE = "robot_enforce_candidate"


_NEXT_STAGE = {
    PromotionStage.TRAINED: PromotionStage.OFFLINE_VALIDATED,
    PromotionStage.OFFLINE_VALIDATED: PromotionStage.DEBUG_VALIDATED,
    PromotionStage.DEBUG_VALIDATED: PromotionStage.SIMULATOR_CANDIDATE,
    PromotionStage.SIMULATOR_CANDIDATE: PromotionStage.SIMULATOR_APPROVED,
    PromotionStage.SIMULATOR_APPROVED: PromotionStage.ROBOT_SHADOW_CANDIDATE,
    PromotionStage.ROBOT_SHADOW_CANDIDATE: PromotionStage.ROBOT_SHADOW_APPROVED,
    PromotionStage.ROBOT_SHADOW_APPROVED: PromotionStage.ROBOT_ENFORCE_CANDIDATE,
}


class ControlPolicy:
    """Explicit controller-side guard; independent of an agent's blocklist."""

    def __init__(
        self,
        *,
        training_knobs: set[str] | frozenset[str] = frozenset(),
        shadow_knobs: set[str] | frozenset[str] = frozenset(),
        safety_knobs: set[str] | frozenset[str] = frozenset(),
    ):
        self.training_knobs = frozenset(training_knobs)
        self.shadow_knobs = frozenset(shadow_knobs)
        self.safety_knobs = frozenset(safety_knobs)
        overlap = (
            (self.training_knobs & self.shadow_knobs)
            | (self.training_knobs & self.safety_knobs)
            | (self.shadow_knobs & self.safety_knobs)
        )
        if overlap:
            raise ValueError(f"knob classes must not overlap: {sorted(overlap)}")

    @staticmethod
    def next_stage(stage: PromotionStage) -> PromotionStage | None:
        return _NEXT_STAGE.get(stage)

    @staticmethod
    def can_promote(current: PromotionStage, target: PromotionStage) -> bool:
        return _NEXT_STAGE.get(current) == target

    def authorize_knob(self, name: str, source: str, environment: str) -> bool:
        if (
            name in self.safety_knobs
            or name not in self.training_knobs | self.shadow_knobs
        ):
            return False
        is_agent = source.startswith("agent")
        if name in self.training_knobs:
            return environment not in {"robot_shadow", "robot_enforce"}
        # Deployment-quality knobs are human-only in v0.1 and never mutable in enforce.
        return not is_agent and environment != "robot_enforce"

    def require_knob(self, name: str, source: str, environment: str) -> None:
        if not self.authorize_knob(name, source, environment):
            raise PermissionError(
                f"source {source!r} cannot change knob {name!r} in {environment!r}"
            )
