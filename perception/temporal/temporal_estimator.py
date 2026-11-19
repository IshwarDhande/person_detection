"""Temporal behaviour estimator module for Milestone M5.

Aggregates history management, motion estimation, approach detection, stationary
classification, and stability timing into a unified temporal perception interface
according to Milestone M5 specification.

Architecture & Execution Flow:
+-----------------------------------------------------------------------------+
|                        M5 TEMPORAL ESTIMATOR PIPELINE                       |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Input: M3 Track, M4 FrameObservation, Matched Detection                    |
|       |                                                                     |
|       v                                                                     |
|  [Track Active & Observed?]                                                  |
|       |                                                                     |
|       +-- NO ---> Immediate Reset & Emit UNKNOWN [INVALID]                  |
|       +-- YES                                                               |
|            |                                                                |
|            v                                                                |
|  [Same-Person Association Valid?]                                           |
|       |                                                                     |
|       +-- NO ---> Reject mismatched observation; hard reset                 |
|       +-- YES                                                               |
|            |                                                                |
|            v                                                                |
|  [Extract Observed Center c_t = (cx, cy)]                                   |
|       |                                                                     |
|       v                                                                     |
|  [ObservationHistory.add_observation(t, c_t)]                               |
|       |                                                                     |
|       v                                                                     |
|  [MotionEstimator]                                                          |
|       +-- Velocity Vector (vx, vy) & Speed                                  |
|       +-- 1.0s Net Displacement D_t                                         |
|       |                                                                     |
|       +-----------------------------+                                       |
|       |                             |                                       |
|       v                             v                                       |
|  [ApproachEstimator]        [StationaryEstimator]                           |
|       |                             |                                       |
|       +--------------+--------------+                                       |
|                      |                                                      |
|                      v                                                      |
|             [StabilityTimer]                                                |
|                      |                                                      |
|                      v                                                      |
|             TemporalObservation                                             |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence

from perception.detection.person_detector import PersonDetection
from perception.readiness.observation_types import (
    BboxValidityValue,
    FrameObservation,
    InteractionZoneValue,
    Observation,
    ObservationStatus,
    PersonPresenceValue,
)
from perception.temporal.approach_estimator import ApproachEstimator
from perception.temporal.centroid import compute_polygon_centroid
from perception.temporal.history import ObservationHistory
from perception.temporal.motion_estimator import MotionEstimator
from perception.temporal.observation_types import (
    ApproachValue,
    InitializationState,
    MotionMetrics,
    MovementValue,
    StabilityObservation,
    StabilityValue,
    TemporalObservation,
    TemporalTimings,
)
from perception.temporal.stationary_estimator import StationaryEstimator
from perception.temporal.stability_timer import StabilityTimer
from perception.temporal.temporal_config import TemporalConfig, load_temporal_config
from perception.tracking.single_person_tracker import Track


class TemporalEstimator:
    """Orchestrates temporal human behaviour estimation for Milestone M5."""

    def __init__(
        self,
        config: Optional[TemporalConfig] = None,
        zone_vertices_normalized: Optional[Sequence[tuple[float, float]]] = None,
    ) -> None:
        """Initialize the temporal behaviour estimator.

        Args:
            config: TemporalConfig instance (or None for defaults).
            zone_vertices_normalized: Interaction zone polygon vertices in [0, 1] normalized
                                      coordinates (from M4 configuration).
        """
        self.config = config if config is not None else load_temporal_config()
        self.config.validate()

        self._zone_vertices_normalized = (
            list(zone_vertices_normalized) if zone_vertices_normalized is not None else []
        )

        # Internal estimators and state
        self._history = ObservationHistory(
            history_window_seconds=self.config.history_window_seconds,
            max_observation_gap_seconds=self.config.max_observation_gap_seconds,
            initial_valid_observations=self.config.initial_valid_observations,
        )
        self._motion_estimator = MotionEstimator(
            min_displacement_window_seconds=self.config.stationary_displacement_window_seconds,
        )
        self._approach_estimator = ApproachEstimator(
            min_direction_cosine=self.config.approach_min_direction_cosine,
            min_closing_speed_px_s=self.config.approach_min_closing_speed_px_s,
            confirmation_seconds=self.config.approach_confirmation_seconds,
        )
        self._stationary_estimator = StationaryEstimator(
            velocity_threshold_px_s=self.config.stationary_velocity_threshold_px_s,
            displacement_threshold_px=self.config.stationary_displacement_threshold_px,
        )
        self._stability_timer = StabilityTimer(
            required_duration_seconds=self.config.stability_duration_seconds,
            violation_tolerance_seconds=self.config.stability_violation_tolerance_seconds,
        )

        self._active_track_id: Optional[int] = None
        self._last_timings: Optional[TemporalTimings] = None

    @property
    def history(self) -> ObservationHistory:
        """Access underlying observation history buffer."""
        return self._history

    @property
    def stability_timer(self) -> StabilityTimer:
        """Access underlying stability timer."""
        return self._stability_timer

    @property
    def last_timings(self) -> Optional[TemporalTimings]:
        """Component latency timings from the most recent estimate() call."""
        return self._last_timings

    def reset(self) -> None:
        """Full reset of all temporal history, timers, and active track association."""
        self._history.reset()
        self._approach_estimator.reset()
        self._stability_timer.reset()
        self._active_track_id = None
        self._last_timings = None

    def get_zone_centroid_px(
        self,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        """Compute pixel coordinates of the interaction-zone centroid.

        Args:
            image_width: Current frame width in pixels.
            image_height: Current frame height in pixels.

        Returns:
            (x_centroid_px, y_centroid_px) as float tuple.

        Raises:
            ValueError: If interaction zone is empty or degenerate.
        """
        if len(self._zone_vertices_normalized) < 3:
            raise ValueError(
                "Cannot compute zone centroid: interaction zone polygon has < 3 vertices."
            )
        px_vertices = [
            (xn * image_width, yn * image_height)
            for xn, yn in self._zone_vertices_normalized
        ]
        return compute_polygon_centroid(px_vertices)

    def estimate(
        self,
        track: Optional[Track],
        frame_observation: FrameObservation,
        matched_detection: Optional[PersonDetection] = None,
        timestamp: Optional[float] = None,
    ) -> TemporalObservation:
        """Estimate temporal human behaviour from consecutive observations.

        Args:
            track: Current M3 Track from SinglePersonTracker, or None if no track.
            frame_observation: Current M4 FrameObservation for the frame.
            matched_detection: YOLO PersonDetection matched to track, or None.
            timestamp: Monotonic timestamp of the frame. Defaults to frame metadata timestamp.

        Returns:
            TemporalObservation conforming to Milestone M5 specification.
        """
        if timestamp is None:
            timestamp = frame_observation.frame_metadata.timestamp

        t_start = time.perf_counter()
        w_img = frame_observation.frame_metadata.image_width
        h_img = frame_observation.frame_metadata.image_height

        # --- Check 1: Active Track and Observed Verification ---
        if track is None or not track.observed:
            # Tracking loss or coasting/predicted position
            self.reset()
            t_total_ms = (time.perf_counter() - t_start) * 1000.0
            self._last_timings = TemporalTimings(
                history_ms=0.0,
                motion_ms=0.0,
                stationary_ms=0.0,
                approach_ms=0.0,
                stability_ms=0.0,
                total_ms=t_total_ms,
            )
            return TemporalObservation(
                timestamp=timestamp,
                track_id=track.track_id if track is not None else None,
                observed=False,
                approach=Observation(
                    value=ApproachValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                movement=Observation(
                    value=MovementValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                stability=StabilityObservation(
                    value=StabilityValue.RESET,
                    elapsed_seconds=0.0,
                    violation_seconds=0.0,
                    status=ObservationStatus.VALID,
                ),
                motion=MotionMetrics(),
                initialization_state=self._history.state,
                initialization_count=0,
            )

        # Track is active and observed: check track identity
        if self._active_track_id is not None and track.track_id != self._active_track_id:
            # New person track assigned: reset previous track's temporal state
            self.reset()
        self._active_track_id = track.track_id

        # --- Check 2: Same-Person Association Verification ---
        is_same_person = False
        m4_person = frame_observation.person

        # Check basic M4 person prerequisites and matched detection
        if (
            m4_person.presence.value == PersonPresenceValue.DETECTED
            and m4_person.presence.status == ObservationStatus.VALID
            and m4_person.bounding_box.value == BboxValidityValue.VALID
            and m4_person.bounding_box.status == ObservationStatus.VALID
            and m4_person.bbox_metrics is not None
            and matched_detection is not None
        ):
            # Strict association contract: M3 matched detection must exactly equal M4 selected bbox
            det_box = matched_detection.bbox_xyxy
            m4_box = m4_person.bbox_metrics.original_bbox_xyxy
            is_same_person = (det_box == m4_box)
        else:
            is_same_person = False

        if not is_same_person:
            # Mismatched observations, missing detection, or invalid bounding box: do NOT combine
            self.reset()
            t_total_ms = (time.perf_counter() - t_start) * 1000.0
            self._last_timings = TemporalTimings(
                history_ms=0.0,
                motion_ms=0.0,
                stationary_ms=0.0,
                approach_ms=0.0,
                stability_ms=0.0,
                total_ms=t_total_ms,
            )
            return TemporalObservation(
                timestamp=timestamp,
                track_id=track.track_id,
                observed=True,
                approach=Observation(
                    value=ApproachValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                movement=Observation(
                    value=MovementValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                stability=StabilityObservation(
                    value=StabilityValue.RESET,
                    elapsed_seconds=0.0,
                    violation_seconds=0.0,
                    status=ObservationStatus.VALID,
                ),
                motion=MotionMetrics(),
                initialization_state=self._history.state,
                initialization_count=0,
            )

        # --- Step 3: Extract Center of Observed Bounding Box ---
        assert m4_person.bbox_metrics is not None
        orig_box = m4_person.bbox_metrics.original_bbox_xyxy
        cx = (orig_box[0] + orig_box[2]) * 0.5
        cy = (orig_box[1] + orig_box[3]) * 0.5
        center = (cx, cy)

        # --- Step 4: Update Observation History ---
        t_hist_start = time.perf_counter()
        is_continuous = self._history.add_observation(timestamp, center)
        if not is_continuous:
            # Temporal gap discontinuity reset occurred
            self._approach_estimator.reset()
            self._stability_timer.reset()
        t_hist_ms = (time.perf_counter() - t_hist_start) * 1000.0

        # --- Step 5: Motion Estimation (Velocity & Displacement) ---
        t_motion_start = time.perf_counter()
        velocity_xy, speed, vel_status = self._motion_estimator.estimate_velocity(self._history)
        disp, disp_status = self._motion_estimator.estimate_displacement(self._history)
        t_motion_ms = (time.perf_counter() - t_motion_start) * 1000.0

        # --- Step 6: Stationary Estimation ---
        t_stat_start = time.perf_counter()
        movement_obs = self._stationary_estimator.evaluate(
            speed_px_s=speed,
            speed_status=vel_status,
            displacement_px=disp,
            displacement_status=disp_status,
        )
        t_stat_ms = (time.perf_counter() - t_stat_start) * 1000.0

        # --- Step 7: Approach Estimation ---
        t_app_start = time.perf_counter()
        zone_centroid: Optional[tuple[float, float]] = None
        dir_cosine: Optional[float] = None
        closing_speed: Optional[float] = None
        dist_to_centroid: Optional[float] = None

        if len(self._zone_vertices_normalized) >= 3:
            try:
                zone_centroid = self.get_zone_centroid_px(w_img, h_img)
            except ValueError:
                zone_centroid = None

        if zone_centroid is not None:
            approach_obs, dir_cosine, closing_speed, dist_to_centroid = (
                self._approach_estimator.evaluate(
                    history=self._history,
                    zone_centroid=zone_centroid,
                    velocity_xy_px_s=velocity_xy,
                    speed_px_s=speed,
                    velocity_status=vel_status,
                    timestamp=timestamp,
                )
            )
        else:
            approach_obs = Observation(
                value=ApproachValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.INVALID,
            )
        t_app_ms = (time.perf_counter() - t_app_start) * 1000.0

        # --- Step 8: Stability Timer ---
        t_stab_start = time.perf_counter()
        # Eligibility Predicate E_t:
        # P_t (Person DETECTED & VALID) AND B_t (BBox VALID & VALID)
        # AND Z_t (Zone INSIDE & VALID) AND S_t (Stationary & VALID)
        p_valid = (
            m4_person.presence.value == PersonPresenceValue.DETECTED
            and m4_person.presence.status == ObservationStatus.VALID
        )
        b_valid = (
            m4_person.bounding_box.value == BboxValidityValue.VALID
            and m4_person.bounding_box.status == ObservationStatus.VALID
        )
        z_valid = (
            m4_person.interaction_zone.value == InteractionZoneValue.INSIDE
            and m4_person.interaction_zone.status == ObservationStatus.VALID
        )
        s_valid = (
            movement_obs.value == MovementValue.STATIONARY
            and movement_obs.status == ObservationStatus.VALID
        )

        is_eligible = p_valid and b_valid and z_valid and s_valid

        # Hard reset check for stability timer:
        # Case B: Unavailable or unreliable evidence:
        # 1. Temporal gap discontinuity (not is_continuous)
        # 2. Presence not DETECTED or status not VALID
        # 3. Bounding box not VALID or status not VALID
        # 4. Interaction zone status not VALID or value is UNKNOWN
        # 5. Movement status not VALID (UNCERTAIN or INVALID) or value is UNKNOWN
        is_hard_reset = (
            not is_continuous
            or m4_person.presence.status != ObservationStatus.VALID
            or m4_person.presence.value != PersonPresenceValue.DETECTED
            or m4_person.bounding_box.status != ObservationStatus.VALID
            or m4_person.bounding_box.value != BboxValidityValue.VALID
            or m4_person.interaction_zone.status != ObservationStatus.VALID
            or m4_person.interaction_zone.value == InteractionZoneValue.UNKNOWN
            or movement_obs.status != ObservationStatus.VALID
            or movement_obs.value == MovementValue.UNKNOWN
        )

        stability_obs = self._stability_timer.update(
            is_eligible=is_eligible,
            is_hard_reset=is_hard_reset,
            timestamp=timestamp,
        )
        t_stab_ms = (time.perf_counter() - t_stab_start) * 1000.0

        t_total_ms = (time.perf_counter() - t_start) * 1000.0
        self._last_timings = TemporalTimings(
            history_ms=t_hist_ms,
            motion_ms=t_motion_ms,
            stationary_ms=t_stat_ms,
            approach_ms=t_app_ms,
            stability_ms=t_stab_ms,
            total_ms=t_total_ms,
        )

        # --- Step 9: Assemble Output Contract ---
        motion_metrics = MotionMetrics(
            velocity_xy_px_s=velocity_xy,
            speed_px_s=speed,
            displacement_px=disp,
            distance_to_zone_centroid_px=dist_to_centroid,
            direction_cosine=dir_cosine,
            closing_speed_px_s=closing_speed,
            history_count=self._history.count,
            history_duration_seconds=self._history.duration_seconds,
        )

        return TemporalObservation(
            timestamp=timestamp,
            track_id=track.track_id,
            observed=True,
            approach=approach_obs,
            movement=movement_obs,
            stability=stability_obs,
            motion=motion_metrics,
            initialization_state=self._history.state,
            initialization_count=self._history.consecutive_valid_count,
        )

