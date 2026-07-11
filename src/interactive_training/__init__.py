"""Public API for Interactive Training 2."""

from interactive_training.agents import LLMAgent, Observation, OpenAIClient, Plan
from interactive_training.core import (
    Accuracy,
    Action,
    ActionResult,
    AverageReward,
    Goal,
    Memory,
    RoundContext,
    StepControl,
    TrainingSession,
    ValidationLoss,
)
from interactive_training.integrations import autopatch, make_interactive, unpatch
from interactive_training.transport import Client, HttpTransport, aim_frontend

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "Accuracy",
    "Action",
    "ActionResult",
    "AverageReward",
    "Client",
    "Goal",
    "HttpTransport",
    "LLMAgent",
    "Memory",
    "Observation",
    "OpenAIClient",
    "Plan",
    "RoundContext",
    "StepControl",
    "TrainingSession",
    "ValidationLoss",
    "aim_frontend",
    "autopatch",
    "make_interactive",
    "unpatch",
]
