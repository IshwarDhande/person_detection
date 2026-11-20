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

import math
import time
from typing import Optional, Sequence, Union

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
from perception.contact.criteria_selection import (
    ALL_CRITERIA_NUMBERS,
    CriteriaSelection,
    filter_reasons_for_criteria,
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
        criteria_selection: Optional[
            Union[CriteriaSelection, frozenset[int], set[int], Sequence[int]]
        ] = None,
    ) -> None:
        """Initialize the readiness engine.

        Args:
            config: ContactConfig instance (or None to load defaults).
            distance_provider: DistanceProvider instance (or None to instantiate from config).
            criteria_selection: CriteriaSelection instance or collection of criterion numbers 1..7.
                Defaults to {1, 2, 7} (Person presence, Bbox validity, Facing robot).
        """
        self.config = config if config is not None else load_contact_config()
        self.config.validate()

        if criteria_selection is None:
            self.criteria_selection = CriteriaSelection.default()
        elif isinstance(criteria_selection, CriteriaSelection):
            self.criteria_selection = criteria_selection
        else:
            self.criteria_selection = CriteriaSelection(frozenset(criteria_selection))

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

        self._last_accepted_timestamp: Optional[float] = None
        self._last_result: Optional[FirstContactResult] = None

    @property
    def last_accepted_timestamp(self) -> Optional[float]:
        """Most recent valid frame timestamp accepted by the engine."""
        return self._last_accepted_timestamp

    @property
    def last_result(self) -> Optional[FirstContactResult]:
        """Most recent FirstContactResult produced by evaluate()."""
        return self._last_result

    def reset(self) -> None:
        """Full reset of visit state, timers, state machine, and timestamp history."""
        self.visit_manager.terminate_visit("MANUAL_ENGINE_RESET")
        self.state_machine.reset()
        self._last_accepted_timestamp = None
        self._last_result = None

    def evaluate(
        self,
        track: Optional[Track],
        frame_observation: FrameObservation,
        temporal_observation: Optional[TemporalObservation] = None,
        matched_detection: Optional[PersonDetection] = None,
        distance_observation: Optional[DistanceObservation] = None,
        timestamp: Optional[float] = None,
        criteria_selection: Optional[
            Union[CriteriaSelection, frozenset[int], set[int], Sequence[int]]
        ] = None,
    ) -> FirstContactResult:
        """Evaluate first-contact readiness for the current frame.

        Args:
            track: Current M3 Track from SinglePersonTracker, or None.
            frame_observation: Current M4 FrameObservation.
            temporal_observation: Current M5 TemporalObservation, or None.
            matched_detection: Raw PersonDetection matched to active track, or None.
            distance_observation: Explicit DistanceObservation, or None to query provider.
            timestamp: Monotonic frame timestamp in seconds (defaults to frame metadata).
            criteria_selection: Optional per-frame CriteriaSelection override.

        Returns:
            FirstContactResult containing state, event, predicates, reasons, and timings.
        """
        t_start = time.perf_counter()

        active_selection: CriteriaSelection
        if criteria_selection is None:
            active_selection = self.criteria_selection
        elif isinstance(criteria_selection, CriteriaSelection):
            active_selection = criteria_selection
        else:
            active_selection = CriteriaSelection(frozenset(criteria_selection))

        if timestamp is None:
            if (
                frame_observation is not None
                and hasattr(frame_observation, "frame_metadata")
                and frame_observation.frame_metadata is not None
            ):
                timestamp = frame_observation.frame_metadata.timestamp

        # Fail closed at boundary if timestamp is missing, invalid, negative, non-finite,
        # or backward/duplicate relative to last accepted frame timestamp.
        is_invalid_ts = (
            timestamp is None
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0
            or (self._last_accepted_timestamp is not None and timestamp <= self._last_accepted_timestamp)
        )
        if is_invalid_ts:
            self.state_machine.reset()
            result = FirstContactResult(
                state=ReadinessState.NOT_READY,
                event=None,
                predicates=ReadinessPredicates(
                    p_presence=False,
                    b_bbox=False,
                    a_approach=False,
                    z_zone=False,
                    d_distance=False,
                    s_stationary=False,
                    o_orientation=False,
                ),
                reasons=[ReasonCode.INVALID_TIMESTAMP],
                visit_id=self.visit_manager.active_visit_id,
                track_id=self.visit_manager.active_track_id,
                elapsed_eligibility=0.0,
                distance=None,
                timings=ContactTimings(),
                selected_criteria=active_selection.sorted_numbers,
            )
            self._last_result = result
            return result

        self._last_accepted_timestamp = timestamp

        try:
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

            # Step 3: Predicate evaluation (all 7 predicates evaluated)
            t_p0 = time.perf_counter()
            predicates, raw_reasons = self.predicate_evaluator.evaluate(
                track=track,
                matched_detection=matched_detection,
                frame_observation=frame_observation,
                temporal_observation=temporal_observation,
                distance_observation=distance_observation,
                visit_manager=self.visit_manager,
                timestamp=timestamp,
                is_active_target_observed=is_active_target_observed,
            )
            t_p_ms = (time.perf_counter() - t_p0) * 1000.0

            # Filter reasons according to active criteria selection
            reasons = filter_reasons_for_criteria(raw_reasons, active_selection.selected_numbers)

            # Step 4: Readiness state machine update
            # Demo gate: conjunction of selected criteria AND active target observed
            is_gate_valid = active_selection.is_satisfied(predicates)
            effective_valid = is_gate_valid and is_active_target_observed
            t_s0 = time.perf_counter()
            state, event, elapsed = self.state_machine.update(
                is_all_valid=effective_valid,
                visit_manager=self.visit_manager,
                track_id=self.visit_manager.active_track_id,
                distance_obs=distance_observation,
                timestamp=timestamp,
                is_active_target_observed=is_active_target_observed,
                selected_criteria=active_selection.selected_numbers,
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
                selected_criteria=active_selection.sorted_numbers,
            )
            self._last_result = result
            return result
        except ValueError:
            # Boundary protection: catch non-monotonic or invalid timestamp errors
            self.state_machine.reset()
            t_total_ms = (time.perf_counter() - t_start) * 1000.0
            result = FirstContactResult(
                state=ReadinessState.NOT_READY,
                event=None,
                predicates=ReadinessPredicates(
                    p_presence=False,
                    b_bbox=False,
                    a_approach=False,
                    z_zone=False,
                    d_distance=False,
                    s_stationary=False,
                    o_orientation=False,
                ),
                reasons=[ReasonCode.INVALID_TIMESTAMP],
                visit_id=self.visit_manager.active_visit_id,
                track_id=self.visit_manager.active_track_id,
                elapsed_eligibility=0.0,
                distance=distance_observation,
                timings=ContactTimings(total_ms=t_total_ms),
                selected_criteria=active_selection.sorted_numbers,
            )
            self._last_result = result
            return result
        return result

