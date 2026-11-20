"""Visit lifecycle management module for Milestone M6.

Manages single-active-person locking, visit identifiers, tracking-loss grace periods,
reliable zone exit termination, and approach/initial-presence persistence according to
Section 8 of the M6 specification.
"""

from __future__ import annotations

import math
from typing import Optional
import uuid

from perception.readiness.observation_types import (
    FrameObservation,
    InteractionZoneValue,
    ObservationStatus,
    PersonPresenceValue,
)
from perception.temporal.observation_types import ApproachValue, TemporalObservation
from perception.tracking.single_person_tracker import Track

DEFAULT_TRACKING_LOSS_TIMEOUT_SECONDS: float = 5.0


class VisitManager:
    """Manages the single-person visit lifecycle and target locking."""

    def __init__(
        self,
        tracking_loss_timeout_seconds: float = DEFAULT_TRACKING_LOSS_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize visit manager.

        Args:
            tracking_loss_timeout_seconds: Grace period in seconds to retain a visit
                                           during tracking loss (default: 5.0s).
        """
        if tracking_loss_timeout_seconds <= 0.0:
            raise ValueError(
                f"tracking_loss_timeout_seconds must be positive, got {tracking_loss_timeout_seconds}"
            )
        self.tracking_loss_timeout_seconds: float = float(tracking_loss_timeout_seconds)

        self._active_visit_id: Optional[str] = None
        self._active_track_id: Optional[int] = None
        self._approach_or_presence_established: bool = False
        self._has_been_in_zone: bool = False
        self._event_emitted: bool = False
        self._tracking_loss_start_time: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._last_termination_reason: Optional[str] = None

    @property
    def has_active_visit(self) -> bool:
        """True if a visit is currently active."""
        return self._active_visit_id is not None

    @property
    def active_visit_id(self) -> Optional[str]:
        """Unique identifier of the currently active visit, or None."""
        return self._active_visit_id

    @property
    def active_track_id(self) -> Optional[int]:
        """Track ID of the locked active target, or None."""
        return self._active_track_id

    @property
    def approach_or_presence_established(self) -> bool:
        """True if confirmed approach or initial presence has been satisfied for this visit."""
        return self._approach_or_presence_established

    @property
    def event_emitted(self) -> bool:
        """True if FIRST_CONTACT_READY event has already been emitted for this visit."""
        return self._event_emitted

    @property
    def is_in_loss_grace_period(self) -> bool:
        """True if tracking is currently lost but within the 5.0s grace period."""
        return self._tracking_loss_start_time is not None

    @property
    def tracking_loss_duration(self) -> float:
        """Elapsed seconds since tracking was lost during current grace period."""
        if self._tracking_loss_start_time is None or self._last_timestamp is None:
            return 0.0
        return max(0.0, self._last_timestamp - self._tracking_loss_start_time)

    @property
    def last_termination_reason(self) -> Optional[str]:
        """Reason code/string for the most recent visit termination."""
        return self._last_termination_reason

    def mark_event_emitted(self) -> None:
        """Mark that FIRST_CONTACT_READY has been emitted for the active visit."""
        self._event_emitted = True

    def terminate_visit(self, reason: str) -> None:
        """Terminate the active visit and clear target locking.

        Args:
            reason: Explanatory reason string (e.g. 'RELIABLE_ZONE_EXIT', 'TRACKING_LOSS_TIMEOUT').
        """
        self._last_termination_reason = reason
        self._active_visit_id = None
        self._active_track_id = None
        self._approach_or_presence_established = False
        self._has_been_in_zone = False
        self._event_emitted = False
        self._tracking_loss_start_time = None
        self._last_timestamp = None

    def update(
        self,
        track: Optional[Track],
        frame_observation: Optional[FrameObservation],
        temporal_observation: Optional[TemporalObservation],
        timestamp: float,
        matched_detection: Optional[PersonDetection] = None,
    ) -> tuple[Optional[str], bool]:
        """Update visit lifecycle for the current frame.

        Args:
            track: Current M3 Track from SinglePersonTracker, or None.
            frame_observation: Current M4 FrameObservation, or None.
            temporal_observation: Current M5 TemporalObservation, or None.
            timestamp: Monotonic frame timestamp in seconds.
            matched_detection: Raw PersonDetection associated with the track, or None.

        Returns:
            (active_visit_id, is_tracking_active) tuple where:
                - active_visit_id: Current visit ID (or None if no visit active).
                - is_tracking_active: True if active target is currently observed.
        """
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite float, got {timestamp}")

        self._last_timestamp = float(timestamp)

        m4_person = frame_observation.person if frame_observation is not None else None

        # Check whether the current track matches the locked active target
        is_active_target_observed = (
            track is not None
            and track.observed
            and (self._active_track_id is None or track.track_id == self._active_track_id)
        )

        # Check whether M4 frame observation belongs to the active target
        is_m4_active_person = False
        if is_active_target_observed and m4_person is not None and m4_person.bbox_metrics is not None:
            if matched_detection is not None:
                is_m4_active_person = (
                    matched_detection.bbox_xyxy == m4_person.bbox_metrics.original_bbox_xyxy
                )
            # Fail closed: Do NOT use track.bbox_xyxy alone without verified matched_detection

        # Update zone entry for the active target
        if is_active_target_observed and is_m4_active_person and m4_person is not None:
            if (
                m4_person.interaction_zone.value == InteractionZoneValue.INSIDE
                and m4_person.interaction_zone.status == ObservationStatus.VALID
            ):
                self._has_been_in_zone = True

        # Check 1: Reliable Zone Exit terminates active visit immediately
        # Strictly applies ONLY when the observation is verified to belong to the locked active target!
        if (
            self._active_visit_id is not None
            and self._has_been_in_zone
            and is_active_target_observed
            and is_m4_active_person
            and m4_person is not None
            and m4_person.interaction_zone.value == InteractionZoneValue.OUTSIDE
            and m4_person.interaction_zone.status == ObservationStatus.VALID
        ):
            self.terminate_visit("RELIABLE_ZONE_EXIT")
            return None, False

        # Check 2: No active visit -> check if a new visit should be initiated
        if self._active_visit_id is None:
            if track is not None and track.observed:
                # Initiate new visit
                self._active_visit_id = f"visit_{uuid.uuid4().hex[:8]}"
                self._active_track_id = track.track_id
                self._event_emitted = False
                self._tracking_loss_start_time = None
                self._last_termination_reason = None
                self._has_been_in_zone = False

                # Check for Initial Presence path (startup / arrival directly in zone)
                if (
                    is_m4_active_person
                    and m4_person is not None
                    and m4_person.presence.value == PersonPresenceValue.DETECTED
                    and m4_person.presence.status == ObservationStatus.VALID
                    and m4_person.interaction_zone.value == InteractionZoneValue.INSIDE
                    and m4_person.interaction_zone.status == ObservationStatus.VALID
                ):
                    self._approach_or_presence_established = True
                    self._has_been_in_zone = True
                else:
                    self._approach_or_presence_established = False
            else:
                return None, False

        # Verify active target tracking status for the established visit
        is_active_target_observed = (
            track is not None
            and track.observed
            and track.track_id == self._active_track_id
        )

        # Check 3: Tracking Loss Management
        if not is_active_target_observed:
            if self._tracking_loss_start_time is None:
                self._tracking_loss_start_time = timestamp

            loss_elapsed = timestamp - self._tracking_loss_start_time
            if loss_elapsed >= self.tracking_loss_timeout_seconds:
                self.terminate_visit("TRACKING_LOSS_TIMEOUT")
                return None, False

            # In grace period: retain visit
            return self._active_visit_id, False

        # Active target is observed: clear tracking loss timer (reacquisition)
        self._tracking_loss_start_time = None

        # Check 4: Update Approach / Initial Presence if not yet satisfied
        if not self._approach_or_presence_established:
            # Path 1: Confirmed approach from M5 (must belong to active target)
            if (
                temporal_observation is not None
                and temporal_observation.track_id == self._active_track_id
                and temporal_observation.approach.value == ApproachValue.APPROACHING
                and temporal_observation.approach.status == ObservationStatus.VALID
            ):
                self._approach_or_presence_established = True

            # Path 2: Reliable observation inside zone (must belong to active target)
            elif (
                is_m4_active_person
                and m4_person.presence.value == PersonPresenceValue.DETECTED
                and m4_person.presence.status == ObservationStatus.VALID
                and m4_person.interaction_zone.value == InteractionZoneValue.INSIDE
                and m4_person.interaction_zone.status == ObservationStatus.VALID
            ):
                self._approach_or_presence_established = True

        return self._active_visit_id, True
