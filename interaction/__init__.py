"""Interaction package for Milestone M6 downstream consumer.

Provides mock interaction receivers, event subscribers, and action dispatchers
that consume outputs from the Vision-Based First Contact Readiness Pipeline.
"""

from __future__ import annotations

__all__ = [
    "InteractionAction",
    "InteractionState",
    "MockInteractionReceiver",
    "RobotActionType",
]


def __getattr__(name: str):
    if name in __all__:
        from interaction.mock_interaction_receiver import (
            InteractionAction,
            InteractionState,
            MockInteractionReceiver,
            RobotActionType,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
