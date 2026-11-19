"""Readiness Engine for Milestone M6: Readiness-Based Single-Person First Contact.

Coordinates predicate evaluation, visit lifecycle management, eligibility timing,
and one-shot contact event generation into a unified decision layer.

Architecture & Execution Flow:
+-----------------------------------------------------------------------------+
|                         M6 READINESS ENGINE PIPELINE                        |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Input: M3 Track, M4 FrameObservation, M5 TemporalObservation, Distance     |
|       |                                                                     |
|       v                                                                     |
|  [VisitManager.update()]                                                    |
|       +-- Target Locking (single-person active)                             |
|       +-- 5.0s Tracking-Loss Grace Period & Reacquisition                   |
|       +-- Reliable Zone Exit Termination                                    |
|       +-- Approach / Initial Presence Flag (A)                              |
|       |                                                                     |
|       v                                                                     |
|  [PredicateEvaluator.evaluate()]                                            |
|       +-- E = P & B & A & Z & D & S & O                                     |
|       +-- Explainable Reason Codes (NO_PERSON, TOO_FAR, etc.)               |
|       |                                                                     |
|       v                                                                     |
|  [ReadinessStateMachine.update()]                                           |
|       +-- EligibilityTimer (2.0s continuous valid time)                     |
|       +-- State Transitions: NOT_READY <-> PENDING <-> READY                |
|       +-- One-Shot FIRST_CONTACT_READY Event Emission                       |
|       |                                                                     |
|       v                                                                     |
|  FirstContactResult                                                         |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

import time
from typing import Optional

from perception.contact.contact_config import ContactConfig, load_contact_config
from perception.contact.contact_types import (
    ContactTimings,
    DistanceObservation,
    FirstContactReadyEvent,
    FirstContactResult,
    ReadinessPredicates,
    ReadinessState,
    ReasonCode,
)
from perception.contact.distance_provider import DistanceProvider, create_distance_provider
from perception.contact.predicates import PredicateEvaluator
from perception.contact.readiness_state_machine import ReadinessStateMachine
from perception.contact.visit_manager import VisitManager
from perception.detection.person_detector import PersonDetection
from perception.readiness.observation_types import FrameObservation
from perception.temporal.observation_types import TemporalObservation
from perception.tracking.single_person_tracker import Track


class ReadinessEngine:
    """Orchestrates single-person first-contact readiness decisions."""

    def __init__(
        self,
        config: Optional[ContactConfig] = None,
        distance_provider: Optional[DistanceProvider] = None,
    ) -> None:
        """Initialize the readiness engine.

        Args:
            config: ContactConfig instance (or None to load defaults).
            distance_provider: DistanceProvider instance (or None to instantiate from config).
        """
        self.config = config if config is not None else load_contact_config()
        self.config.validate()

        self.distance_provider = (
            distance_provider
            if distance_provider is not None
            else create_distance_provider(self.config)
        )

        self.visit_manager = VisitManager(
            tracking_loss_timeout_seconds=self.config.tracking_loss_timeout_seconds
        )
        self.predicate_evaluator = PredicateEvaluator(config=self.config)
        self.state_machine = ReadinessStateMachine(
            eligibility_duration_seconds=self.config.eligibility_duration_seconds,
            max_inter_frame_gap_seconds=self.config.max_inter_frame_gap_seconds,
        )

        self._last_result: Optional[FirstContactResult] = None

    @property
    def last_result(self) -> Optional[FirstContactResult]:
        """Most recent FirstContactResult produced by evaluate()."""
        return self._last_result

    def reset(self) -> None:
        """Full reset of visit state, timers, and state machine."""
        self.visit_manager.terminate_visit("MANUAL_ENGINE_RESET")
        self.state_machine.reset()
        self._last_result = None

    def evaluate(
        self,
        track: Optional[Track],
        frame_observation: FrameObservation,
        temporal_observation: Optional[TemporalObservation] = None,
        matched_detection: Optional[PersonDetection] = None,
        distance_observation: Optional[DistanceObservation] = None,
        timestamp: Optional[float] = None,
    ) -> FirstContactResult:
        """Evaluate first-contact readiness for the current frame.

        Args:
            track: Current M3 Track from SinglePersonTracker, or None.
            frame_observation: Current M4 FrameObservation.
            temporal_observation: Current M5 TemporalObservation, or None.
            matched_detection: Raw PersonDetection matched to active track, or None.
            distance_observation: Explicit DistanceObservation, or None to query provider.
            timestamp: Monotonic frame timestamp in seconds (defaults to frame metadata).

        Returns:
            FirstContactResult containing state, event, predicates, reasons, and timings.
        """
        t_start = time.perf_counter()

        if timestamp is None:
            if frame_observation is not None:
                timestamp = frame_observation.frame_metadata.timestamp
            else:
                timestamp = time.monotonic()

        # Step 1: Distance observation acquisition
        if distance_observation is None:
            bbox = track.bbox_xyxy if track is not None else None
            tid = track.track_id if track is not None else None
            distance_observation = self.distance_provider.get_distance(
                timestamp=timestamp, bbox_xyxy=bbox, track_id=tid
            )

        # Step 2: Visit lifecycle management
        t_v0 = time.perf_counter()
        visit_id, is_active_target_observed = self.visit_manager.update(
            track=track,
            frame_observation=frame_observation,
            temporal_observation=temporal_observation,
            timestamp=timestamp,
            matched_detection=matched_detection,
        )
        t_v_ms = (time.perf_counter() - t_v0) * 1000.0

        # Step 3: Predicate evaluation
        t_p0 = time.perf_counter()
        predicates, reasons = self.predicate_evaluator.evaluate(
            track=track,
            matched_detection=matched_detection,
            frame_observation=frame_observation,
            temporal_observation=temporal_observation,
            distance_observation=distance_observation,
            visit_manager=self.visit_manager,
            timestamp=timestamp,
        )
        t_p_ms = (time.perf_counter() - t_p0) * 1000.0

        # Step 4: Readiness state machine update
        t_s0 = time.perf_counter()
        state, event, elapsed = self.state_machine.update(
            is_all_valid=predicates.is_all_valid,
            visit_manager=self.visit_manager,
            track_id=self.visit_manager.active_track_id,
            distance_obs=distance_observation,
            timestamp=timestamp,
        )
        t_s_ms = (time.perf_counter() - t_s0) * 1000.0

        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        timings = ContactTimings(
            predicate_eval_ms=t_p_ms,
            visit_management_ms=t_v_ms,
            state_machine_ms=t_s_ms,
            total_ms=t_total_ms,
        )

        result = FirstContactResult(
            state=state,
            event=event,
            predicates=predicates,
            reasons=reasons,
            visit_id=self.visit_manager.active_visit_id,
            track_id=self.visit_manager.active_track_id,
            elapsed_eligibility=elapsed,
            distance=distance_observation,
            timings=timings,
        )

        self._last_result = result
        return result

