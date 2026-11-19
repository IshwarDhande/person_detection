"""Readiness decision state machine for Milestone M6.

Maintains persistent readiness state (NOT_READY, PENDING, READY), manages revocable
readiness, and emits structured one-shot FIRST_CONTACT_READY events per visit
according to Section 6 and 7 of the M6 specification.
"""

from __future__ import annotations

import math
from typing import Optional
import uuid

from perception.contact.contact_types import (
    DistanceInfo,
    DistanceObservation,
    FirstContactReadyEvent,
    ReadinessState,
)
from perception.contact.eligibility_timer import EligibilityTimer
from perception.contact.visit_manager import VisitManager


class ReadinessStateMachine:
    """Evaluates readiness transitions and generates one-shot contact events."""

    def __init__(
        self,
        eligibility_duration_seconds: float = 2.0,
        max_inter_frame_gap_seconds: float = 0.5,
    ) -> None:
        """Initialize readiness state machine.

        Args:
            eligibility_duration_seconds: Required continuous valid time in seconds (default: 2.0).
            max_inter_frame_gap_seconds: Maximum allowed gap between consecutive valid frames (default: 0.5s).
        """
        self._timer = EligibilityTimer(
            required_duration_seconds=eligibility_duration_seconds,
            max_inter_frame_gap_seconds=max_inter_frame_gap_seconds,
        )
        self._state: ReadinessState = ReadinessState.NOT_READY
        self._eligible_since: Optional[float] = None
        self._ready_since: Optional[float] = None
        self._current_visit_id: Optional[str] = None

    @property
    def state(self) -> ReadinessState:
        """Current externally visible readiness state."""
        return self._state

    @property
    def eligible_since(self) -> Optional[float]:
        """Timestamp when current continuous eligibility began, or None."""
        return self._eligible_since

    @property
    def ready_since(self) -> Optional[float]:
        """Timestamp when READY was achieved in current attempt, or None."""
        return self._ready_since

    @property
    def elapsed_eligibility(self) -> float:
        """Elapsed continuous eligibility seconds."""
        return self._timer.elapsed_seconds

    def reset(self) -> None:
        """Full reset of state machine and eligibility timer."""
        self._state = ReadinessState.NOT_READY
        self._eligible_since = None
        self._ready_since = None
        self._timer.reset()
        self._current_visit_id = None

    def update(
        self,
        is_all_valid: bool,
        visit_manager: VisitManager,
        track_id: Optional[int],
        distance_obs: Optional[DistanceObservation],
        timestamp: float,
    ) -> tuple[ReadinessState, Optional[FirstContactReadyEvent], float]:
        """Update readiness state machine for current frame.

        Args:
            is_all_valid: True if all seven readiness predicates are simultaneously True.
            visit_manager: Active VisitManager instance.
            track_id: Active M3 track identifier, or None.
            distance_obs: Current DistanceObservation, or None.
            timestamp: Monotonic frame timestamp in seconds.

        Returns:
            (state, event, elapsed_eligibility) tuple.
        """
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            self.reset()
            raise ValueError(f"timestamp must be a finite float, got {timestamp}")

        active_visit_id = visit_manager.active_visit_id

        # Visit boundary management: if visit_id changed, reset internal state
        if active_visit_id != self._current_visit_id:
            self._state = ReadinessState.NOT_READY
            self._eligible_since = None
            self._ready_since = None
            self._timer.reset()
            self._current_visit_id = active_visit_id

        # If no active visit or track_id is None, remain NOT_READY
        if active_visit_id is None or track_id is None:
            self._state = ReadinessState.NOT_READY
            self._eligible_since = None
            self._ready_since = None
            self._timer.reset()
            return ReadinessState.NOT_READY, None, 0.0

        event: Optional[FirstContactReadyEvent] = None

        # Condition broken / unsatisfied: reset timer, transition to NOT_READY (revocable)
        if not is_all_valid:
            self._timer.reset()
            self._eligible_since = None
            self._ready_since = None
            self._state = ReadinessState.NOT_READY
            return ReadinessState.NOT_READY, None, 0.0

        # All 7 conditions are valid
        try:
            is_complete, elapsed = self._timer.update(is_eligible=True, timestamp=timestamp)
        except ValueError:
            self.reset()
            raise

        if elapsed == 0.0 or self._eligible_since is None:
            self._eligible_since = timestamp

        if self._state == ReadinessState.NOT_READY:
            if is_complete:
                self._state = ReadinessState.READY
                self._ready_since = timestamp
            else:
                self._state = ReadinessState.PENDING

        elif self._state == ReadinessState.PENDING:
            if is_complete:
                self._state = ReadinessState.READY
                self._ready_since = timestamp

        elif self._state == ReadinessState.READY:
            # Remains in live READY state
            pass

        # Check for one-shot event generation on first transition to READY
        if self._state == ReadinessState.READY and not visit_manager.event_emitted:
            dist_val = distance_obs.distance_m if distance_obs is not None else 0.0
            dist_src = distance_obs.source.value if distance_obs is not None else "MOCK"

            event = FirstContactReadyEvent(
                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                visit_id=active_visit_id,
                track_id=track_id,
                timestamp=timestamp,
                readiness_duration=self._timer.required_duration_seconds,
                distance=DistanceInfo(
                    value_m=dist_val,
                    source=dist_src,
                ),
            )
            visit_manager.mark_event_emitted()

        return self._state, event, elapsed

