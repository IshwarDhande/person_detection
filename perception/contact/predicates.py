"""Readiness predicate evaluator and reason-code generator for Milestone M6.

Evaluates the seven required readiness conditions:
    E_i(t) = P_i(t) & B_i(t) & A_i(t) & Z_i(t) & D_i(t) & S_i(t) & O_i(t)
according to Section 3, 4, and 9 of the M6 specification.
"""

from __future__ import annotations

import math
from typing import Optional

from perception.contact.contact_config import ContactConfig
from perception.contact.contact_types import (
    DistanceObservation,
    DistanceSource,
    ReadinessPredicates,
    ReasonCode,
)
from perception.contact.visit_manager import VisitManager
from perception.detection.person_detector import PersonDetection
from perception.readiness.observation_types import (
    BboxValidityValue,
    FrameObservation,
    HeadOrientationValue,
    InteractionZoneValue,
    ObservationStatus,
    PersonPresenceValue,
)
from perception.temporal.observation_types import MovementValue, TemporalObservation
from perception.tracking.single_person_tracker import Track


class PredicateEvaluator:
    """Evaluates the seven readiness predicates and derives deterministic reason codes."""

    def __init__(self, config: ContactConfig) -> None:
        """Initialize predicate evaluator.

        Args:
            config: ContactConfig instance.
        """
        self.config = config

    def evaluate(
        self,
        track: Optional[Track],
        matched_detection: Optional[PersonDetection],
        frame_observation: Optional[FrameObservation],
        temporal_observation: Optional[TemporalObservation],
        distance_observation: Optional[DistanceObservation],
        visit_manager: VisitManager,
        timestamp: float,
    ) -> tuple[ReadinessPredicates, list[ReasonCode]]:
        """Evaluate the seven readiness predicates and produce explainable reason codes.

        Args:
            track: Current active M3 track, or None.
            matched_detection: YOLO PersonDetection matched to track, or None.
            frame_observation: Current M4 FrameObservation, or None.
            temporal_observation: Current M5 TemporalObservation, or None.
            distance_observation: Current M6 DistanceObservation, or None.
            visit_manager: Active VisitManager instance.
            timestamp: Monotonic frame evaluation timestamp in seconds.

        Returns:
            (ReadinessPredicates, list[ReasonCode]) tuple.
        """
        reasons: list[ReasonCode] = []
        active_track_id = visit_manager.active_track_id

        if frame_observation is None:
            reasons.append(ReasonCode.NO_PERSON)
            if not math.isfinite(timestamp):
                reasons.append(ReasonCode.INVALID_TIMESTAMP)
            if visit_manager.has_active_visit:
                reasons.append(ReasonCode.TRACKING_LOST)
            return (
                ReadinessPredicates(
                    p_presence=False,
                    b_bbox=False,
                    a_approach=False,
                    z_zone=False,
                    d_distance=False,
                    s_stationary=False,
                    o_orientation=False,
                ),
                reasons,
            )

        m4_person = frame_observation.person
        m4_orient = frame_observation.orientation

        # ---------------------------------------------------------------------
        # 1. Timestamp Validity and Finiteness Checks
        # ---------------------------------------------------------------------
        frame_ts = frame_observation.frame_metadata.timestamp
        ts_valid = math.isfinite(timestamp) and math.isfinite(frame_ts)
        if not ts_valid:
            reasons.append(ReasonCode.INVALID_TIMESTAMP)

        # ---------------------------------------------------------------------
        # 2. Frame Freshness & Timestamp Consistency Checks
        # ---------------------------------------------------------------------
        is_frame_stale = not ts_valid or abs(timestamp - frame_ts) > self.config.observation_freshness_seconds
        if is_frame_stale and ts_valid:
            reasons.append(ReasonCode.STALE_OBSERVATION)

        # ---------------------------------------------------------------------
        # 3. Active-Target Identity & Tracking Checks
        # ---------------------------------------------------------------------
        is_track_active = (
            track is not None
            and (active_track_id is None or track.track_id == active_track_id)
        )
        if active_track_id is not None and track is not None and track.track_id != active_track_id:
            reasons.append(ReasonCode.TARGET_MISMATCH)

        is_tracking_observed = (
            is_track_active
            and track is not None
            and track.observed
        )
        if not is_tracking_observed:
            reasons.append(ReasonCode.TRACKING_LOST)

        # Check track timestamp consistency with frame
        if track is not None:
            if not math.isfinite(track.timestamp) or abs(frame_ts - track.timestamp) > self.config.observation_freshness_seconds:
                reasons.append(ReasonCode.TIMESTAMP_MISMATCH)
                is_tracking_observed = False

        # ---------------------------------------------------------------------
        # Predicate P — Person presence
        # ---------------------------------------------------------------------
        p_valid = (
            not is_frame_stale
            and is_tracking_observed
            and m4_person.presence.value == PersonPresenceValue.DETECTED
            and m4_person.presence.status == ObservationStatus.VALID
        )
        if not p_valid:
            reasons.append(ReasonCode.NO_PERSON)

        # ---------------------------------------------------------------------
        # Predicate B — Bounding-box validity & same-person association
        # ---------------------------------------------------------------------
        b_box_valid = (
            not is_frame_stale
            and m4_person.bounding_box.value == BboxValidityValue.VALID
            and m4_person.bounding_box.status == ObservationStatus.VALID
            and m4_person.bbox_metrics is not None
        )
        # Strict association: matched detection must equal M4 selected bbox of active person
        is_same_person = False
        if b_box_valid and is_tracking_observed:
            if matched_detection is not None:
                is_same_person = (
                    matched_detection.bbox_xyxy == m4_person.bbox_metrics.original_bbox_xyxy
                )
            elif track is not None:
                is_same_person = (
                    track.bbox_xyxy == m4_person.bbox_metrics.original_bbox_xyxy
                )

        if not is_same_person and is_tracking_observed:
            reasons.append(ReasonCode.TARGET_MISMATCH)

        b_valid = b_box_valid and is_same_person
        if not b_valid:
            reasons.append(ReasonCode.INVALID_BBOX)

        # ---------------------------------------------------------------------
        # Predicate A — Approach or initial presence
        # ---------------------------------------------------------------------
        a_valid = visit_manager.approach_or_presence_established
        if not a_valid:
            reasons.append(ReasonCode.APPROACH_NOT_CONFIRMED)

        # ---------------------------------------------------------------------
        # Predicate Z — Interaction-zone membership
        # ---------------------------------------------------------------------
        z_valid = (
            not is_frame_stale
            and is_same_person
            and m4_person.interaction_zone.value == InteractionZoneValue.INSIDE
            and m4_person.interaction_zone.status == ObservationStatus.VALID
        )
        if not z_valid:
            reasons.append(ReasonCode.OUTSIDE_INTERACTION_ZONE)

        # ---------------------------------------------------------------------
        # Predicate D — Distance condition
        # ---------------------------------------------------------------------
        d_valid = False
        if distance_observation is None or distance_observation.status == ObservationStatus.INVALID:
            reasons.append(ReasonCode.DISTANCE_UNAVAILABLE)
        elif not math.isfinite(distance_observation.timestamp):
            reasons.append(ReasonCode.INVALID_TIMESTAMP)
        elif abs(timestamp - distance_observation.timestamp) > self.config.observation_freshness_seconds:
            reasons.append(ReasonCode.STALE_OBSERVATION)
            reasons.append(ReasonCode.DISTANCE_UNAVAILABLE)
        elif distance_observation.status == ObservationStatus.UNCERTAIN:
            reasons.append(ReasonCode.DISTANCE_UNRELIABLE)
        elif (
            distance_observation.track_id is not None
            and active_track_id is not None
            and distance_observation.track_id != active_track_id
        ):
            reasons.append(ReasonCode.TARGET_MISMATCH)
            reasons.append(ReasonCode.DISTANCE_UNAVAILABLE)
        elif self.config.require_metric_distance_in_production and distance_observation.source == DistanceSource.MOCK:
            reasons.append(ReasonCode.DISTANCE_UNRELIABLE)
        elif distance_observation.distance_m > self.config.distance_max_m:
            reasons.append(ReasonCode.TOO_FAR)
        else:
            d_valid = (distance_observation.status == ObservationStatus.VALID)

        # ---------------------------------------------------------------------
        # Predicate S — Stationary behaviour
        # ---------------------------------------------------------------------
        s_valid = False
        if temporal_observation is not None:
            temp_ts = temporal_observation.timestamp
            if not math.isfinite(temp_ts):
                reasons.append(ReasonCode.INVALID_TIMESTAMP)
            elif abs(timestamp - temp_ts) > self.config.observation_freshness_seconds or abs(frame_ts - temp_ts) > self.config.observation_freshness_seconds:
                reasons.append(ReasonCode.STALE_OBSERVATION)
                reasons.append(ReasonCode.TIMESTAMP_MISMATCH)
            elif (
                temporal_observation.track_id is not None
                and active_track_id is not None
                and temporal_observation.track_id != active_track_id
            ):
                reasons.append(ReasonCode.TARGET_MISMATCH)
            elif not temporal_observation.observed:
                reasons.append(ReasonCode.TRACKING_LOST)
            elif (
                temporal_observation.movement.value == MovementValue.STATIONARY
                and temporal_observation.movement.status == ObservationStatus.VALID
            ):
                s_valid = True

        if not s_valid:
            reasons.append(ReasonCode.PERSON_MOVING)

        # ---------------------------------------------------------------------
        # Predicate O — Facing the robot
        # ---------------------------------------------------------------------
        o_valid = (
            not is_frame_stale
            and is_same_person
            and m4_orient.head_orientation.value == HeadOrientationValue.FACING_ROBOT
            and m4_orient.head_orientation.status == ObservationStatus.VALID
        )
        if not o_valid:
            reasons.append(ReasonCode.NOT_FACING_ROBOT)

        # ---------------------------------------------------------------------
        # Observation Uncertainty Check
        # ---------------------------------------------------------------------
        has_uncertainty = (
            m4_person.presence.status == ObservationStatus.UNCERTAIN
            or m4_person.bounding_box.status == ObservationStatus.UNCERTAIN
            or m4_person.interaction_zone.status == ObservationStatus.UNCERTAIN
            or m4_orient.head_orientation.status == ObservationStatus.UNCERTAIN
            or (temporal_observation is not None and temporal_observation.movement.status == ObservationStatus.UNCERTAIN)
            or (distance_observation is not None and distance_observation.status == ObservationStatus.UNCERTAIN)
        )
        if has_uncertainty and ReasonCode.PERCEPTION_UNCERTAIN not in reasons:
            reasons.append(ReasonCode.PERCEPTION_UNCERTAIN)

        # Deduplicate reasons preserving order
        unique_reasons: list[ReasonCode] = []
        for r in reasons:
            if r not in unique_reasons:
                unique_reasons.append(r)

        predicates = ReadinessPredicates(
            p_presence=p_valid,
            b_bbox=b_valid,
            a_approach=a_valid,
            z_zone=z_valid,
            d_distance=d_valid,
            s_stationary=s_valid,
            o_orientation=o_valid,
        )

        return predicates, unique_reasons

