"""Unit and integration tests for Configurable Readiness Criteria Selection.

Tests:
1. CLI parsing and validation (defaults, aliases, explicit lists, invalid tokens, duplicates, conflicts).
2. Selected gate evaluation (selected satisfied, selected unsatisfied blocks, unselected unsatisfied does not block).
3. Event isolation (all 7 criteria emit FIRST_CONTACT_READY, subset emits DEMO_CRITERIA_READY, never leaks).
4. One-shot event emission per visit for DEMO_CRITERIA_READY.
5. Invariant preservation (active target loss and invalid timestamp block regardless of criteria selection).
"""

from __future__ import annotations

import unittest

from perception.contact.contact_config import ContactConfig
from perception.contact.contact_types import (
    DemoCriteriaReadyEvent,
    DistanceObservation,
    DistanceSource,
    FirstContactReadyEvent,
    ReadinessPredicates,
    ReadinessState,
    ReasonCode,
)
from perception.contact.criteria_selection import (
    ALL_CRITERIA_NUMBERS,
    CRITERION_DETAILS,
    DEFAULT_CRITERIA_NUMBERS,
    STATIONARY_ALIAS_NUMBERS,
    TEMPORAL_ALIAS_NUMBERS,
    CriteriaSelection,
    filter_reasons_for_criteria,
    parse_criteria_string,
)
from perception.contact.readiness_engine import ReadinessEngine
from perception.contact.run_first_contact_demo import parse_args
from perception.detection.person_detector import PersonDetection
from perception.readiness.observation_types import (
    BboxMetrics,
    BboxValidityValue,
    FaceObservations,
    FaceVisibilityValue,
    FrameMetadata,
    FrameObservation,
    HeadOrientationValue,
    InteractionZoneValue,
    Observation,
    ObservationStatus,
    OrientationObservations,
    PersonObservations,
    PersonPresenceValue,
)
from perception.temporal.observation_types import (
    ApproachValue,
    InitializationState,
    MotionMetrics,
    MovementValue,
    StabilityObservation,
    StabilityValue,
    TemporalObservation,
)
from perception.tracking.single_person_tracker import Track, TrackState


def _make_track(
    track_id: int = 1,
    bbox: tuple[float, float, float, float] = (300.0, 200.0, 500.0, 600.0),
    observed: bool = True,
    timestamp: float = 1.0,
) -> Track:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return Track(
        track_id=track_id,
        bbox_xyxy=bbox,
        center=(cx, cy),
        velocity=(0.0, 0.0, 0.0, 0.0),
        timestamp=timestamp,
        creation_timestamp=timestamp,
        last_seen_timestamp=timestamp,
        hit_count=3,
        miss_count=0,
        state=TrackState.CONFIRMED,
        observed=observed,
    )


def _make_frame_obs(
    bbox: tuple[float, float, float, float] = (300.0, 200.0, 500.0, 600.0),
    presence_val: PersonPresenceValue = PersonPresenceValue.DETECTED,
    presence_stat: ObservationStatus = ObservationStatus.VALID,
    bbox_val: BboxValidityValue = BboxValidityValue.VALID,
    bbox_stat: ObservationStatus = ObservationStatus.VALID,
    zone_val: InteractionZoneValue = InteractionZoneValue.INSIDE,
    zone_stat: ObservationStatus = ObservationStatus.VALID,
    orient_val: HeadOrientationValue = HeadOrientationValue.FACING_ROBOT,
    orient_stat: ObservationStatus = ObservationStatus.VALID,
    timestamp: float = 1.0,
) -> FrameObservation:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    metrics = BboxMetrics(
        original_bbox_xyxy=bbox,
        clipped_bbox_xyxy=bbox,
        width=w,
        height=h,
        area=w * h,
        area_ratio=(w * h) / (1280 * 720),
        image_width=1280,
        image_height=720,
    )
    return FrameObservation(
        person=PersonObservations(
            presence=Observation(value=presence_val, confidence=0.9, status=presence_stat),
            bounding_box=Observation(value=bbox_val, confidence=None, status=bbox_stat),
            bbox_metrics=metrics,
            interaction_zone=Observation(value=zone_val, confidence=None, status=zone_stat),
        ),
        face=FaceObservations(
            visibility=Observation(value=FaceVisibilityValue.SUFFICIENT, confidence=0.9, status=ObservationStatus.VALID),
            metrics=None,
        ),
        orientation=OrientationObservations(
            head_orientation=Observation(value=orient_val, confidence=None, status=orient_stat),
            metrics=None,
        ),
        frame_metadata=FrameMetadata(
            timestamp=timestamp,
            image_width=1280,
            image_height=720,
        ),
    )


def _make_temporal_obs(
    movement_val: MovementValue = MovementValue.STATIONARY,
    movement_stat: ObservationStatus = ObservationStatus.VALID,
    approach_val: ApproachValue = ApproachValue.APPROACHING,
    approach_stat: ObservationStatus = ObservationStatus.VALID,
    timestamp: float = 1.0,
    track_id: int | None = 1,
) -> TemporalObservation:
    return TemporalObservation(
        timestamp=timestamp,
        track_id=track_id,
        observed=True,
        approach=Observation(value=approach_val, confidence=None, status=approach_stat),
        movement=Observation(value=movement_val, confidence=None, status=movement_stat),
        stability=StabilityObservation(
            value=StabilityValue.ACCUMULATING,
            elapsed_seconds=1.0,
            violation_seconds=0.0,
            status=ObservationStatus.VALID,
        ),
        motion=MotionMetrics(),
        initialization_state=InitializationState.READY,
        initialization_count=5,
    )


class TestCriteriaParsingAndValidation(unittest.TestCase):
    """Test CLI parsing and validation logic for criteria selection."""

    def test_default_selection_is_1_2_7(self) -> None:
        """When no criteria argument is passed, selection defaults to {1, 2, 7}."""
        parsed = parse_args([])
        self.assertIsNotNone(parsed.criteria_selection)
        self.assertEqual(parsed.criteria_selection.selected_numbers, frozenset({1, 2, 7}))
        self.assertEqual(parsed.criteria_selection.sorted_numbers, (1, 2, 7))
        self.assertTrue(parsed.criteria_selection.is_subset)
        self.assertFalse(parsed.criteria_selection.is_all)
        self.assertEqual(parsed.criteria_selection.selection_name, "default")

    def test_all_flag_selects_all_seven(self) -> None:
        """--all flag selects all criteria {1, 2, 3, 4, 5, 6, 7}."""
        parsed = parse_args(["--all"])
        self.assertEqual(parsed.criteria_selection.selected_numbers, ALL_CRITERIA_NUMBERS)
        self.assertEqual(parsed.criteria_selection.sorted_numbers, (1, 2, 3, 4, 5, 6, 7))
        self.assertTrue(parsed.criteria_selection.is_all)
        self.assertFalse(parsed.criteria_selection.is_subset)
        self.assertEqual(parsed.criteria_selection.selection_name, "all")

    def test_stationary_alias_selects_1_2_4_7(self) -> None:
        """--stationary alias selects {1, 2, 4, 7} (presence, bbox, zone, facing)."""
        parsed = parse_args(["--stationary"])
        self.assertEqual(parsed.criteria_selection.selected_numbers, STATIONARY_ALIAS_NUMBERS)
        self.assertEqual(parsed.criteria_selection.sorted_numbers, (1, 2, 4, 7))
        self.assertTrue(parsed.criteria_selection.is_subset)
        self.assertEqual(parsed.criteria_selection.selection_name, "stationary")

    def test_temporal_alias_selects_3_6(self) -> None:
        """--temporal alias selects {3, 6} (approach and stationary)."""
        parsed = parse_args(["--temporal"])
        self.assertEqual(parsed.criteria_selection.selected_numbers, TEMPORAL_ALIAS_NUMBERS)
        self.assertEqual(parsed.criteria_selection.sorted_numbers, (3, 6))
        self.assertTrue(parsed.criteria_selection.is_subset)
        self.assertEqual(parsed.criteria_selection.selection_name, "temporal")

    def test_explicit_numeric_selection(self) -> None:
        """--criteria 1,2,6 selects {1, 2, 6}."""
        parsed = parse_args(["--criteria", "1,2,6"])
        self.assertEqual(parsed.criteria_selection.selected_numbers, frozenset({1, 2, 6}))
        self.assertEqual(parsed.criteria_selection.sorted_numbers, (1, 2, 6))
        self.assertTrue(parsed.criteria_selection.is_subset)

    def test_explicit_full_selection(self) -> None:
        """--criteria 1,2,3,4,5,6,7 selects all seven criteria and is recognized as all."""
        parsed = parse_args(["--criteria", "1,2,3,4,5,6,7"])
        self.assertEqual(parsed.criteria_selection.selected_numbers, ALL_CRITERIA_NUMBERS)
        self.assertTrue(parsed.criteria_selection.is_all)
        self.assertEqual(parsed.criteria_selection.selection_name, "all")

    def test_rejection_of_unknown_criterion_number(self) -> None:
        """Criterion numbers outside 1..7 (e.g. 8) are rejected with error."""
        with self.assertRaises(SystemExit):
            parse_args(["--criteria", "1,2,8"])

    def test_rejection_of_duplicate_numbers(self) -> None:
        """Duplicate numbers (e.g. 1,2,1) are rejected."""
        with self.assertRaises(SystemExit):
            parse_args(["--criteria", "1,2,1"])

    def test_rejection_of_empty_selection(self) -> None:
        """Empty criteria string is rejected."""
        with self.assertRaises(SystemExit):
            parse_args(["--criteria", ""])

    def test_rejection_of_malformed_selection(self) -> None:
        """Malformed criteria string (e.g. '1,,2' or '1,a,3') is rejected."""
        with self.assertRaises(SystemExit):
            parse_args(["--criteria", "1,,2"])
        with self.assertRaises(SystemExit):
            parse_args(["--criteria", "1,a,3"])
        with self.assertRaises(SystemExit):
            parse_args(["--criteria", "1,2,"])

    def test_rejection_of_conflicting_options(self) -> None:
        """Mutually exclusive options (e.g. --all and --stationary) are rejected."""
        with self.assertRaises(SystemExit):
            parse_args(["--all", "--stationary"])
        with self.assertRaises(SystemExit):
            parse_args(["--all", "--criteria", "1,2"])
        with self.assertRaises(SystemExit):
            parse_args(["--stationary", "--temporal"])


class TestCriteriaGateEvaluation(unittest.TestCase):
    """Test selected criteria gate evaluation semantics: E_demo(t) = \\bigwedge_{i in C} P_i(t)."""

    def setUp(self) -> None:
        self.config = ContactConfig(
            distance_mode="mock",
            mock_distance_m=1.2,
            distance_max_m=1.5,
            eligibility_duration_seconds=2.0,
            tracking_loss_timeout_seconds=5.0,
        )
        self.box = (300.0, 200.0, 500.0, 600.0)
        self.det = PersonDetection(bbox_xyxy=self.box, confidence=0.92, class_id=0)
        self.track = _make_track(track_id=1, bbox=self.box, timestamp=1.0)

    def test_selected_criteria_all_satisfied_enters_pending(self) -> None:
        """When selected criteria are satisfied, gate passes and state transitions to PENDING."""
        # Default criteria {1, 2, 7}
        engine = ReadinessEngine(config=self.config, criteria_selection={1, 2, 7})
        result = engine.evaluate(
            track=self.track,
            frame_observation=_make_frame_obs(timestamp=1.0),
            temporal_observation=_make_temporal_obs(timestamp=1.0),
            matched_detection=self.det,
            timestamp=1.0,
        )
        self.assertEqual(result.state, ReadinessState.PENDING)
        self.assertEqual(result.selected_criteria, (1, 2, 7))

    def test_unsatisfied_selected_criterion_blocks_readiness(self) -> None:
        """Unsatisfied selected criterion blocks readiness, keeps NOT_READY, and produces reason."""
        # Selection {1, 2, 7}: Person turns away -> criterion 7 (Orientation) unsatisfied
        engine = ReadinessEngine(config=self.config, criteria_selection={1, 2, 7})
        result = engine.evaluate(
            track=self.track,
            frame_observation=_make_frame_obs(orient_val=HeadOrientationValue.NOT_FACING_ROBOT, timestamp=1.0),
            temporal_observation=_make_temporal_obs(timestamp=1.0),
            matched_detection=self.det,
            timestamp=1.0,
        )
        self.assertEqual(result.state, ReadinessState.NOT_READY)
        self.assertFalse(result.predicates.O)
        self.assertIn(ReasonCode.NOT_FACING_ROBOT, result.reasons)

    def test_unsatisfied_unselected_criterion_does_not_block_readiness(self) -> None:
        """Unsatisfied UNSELECTED criterion does NOT block readiness."""
        # Selection {1, 2, 7} (P, B, O only).
        # Set Criterion 4 (Zone = OUTSIDE), Criterion 6 (Stationary = MOVING), Criterion 3 (Approach = UNCONFIRMED).
        # All three unselected criteria are False, but selected {1, 2, 7} are True.
        engine = ReadinessEngine(config=self.config, criteria_selection={1, 2, 7})

        frame_obs = _make_frame_obs(
            zone_val=InteractionZoneValue.OUTSIDE,
            timestamp=1.0,
        )
        temp_obs = _make_temporal_obs(
            approach_val=ApproachValue.UNKNOWN,
            movement_val=MovementValue.MOVING,
            timestamp=1.0,
        )

        result = engine.evaluate(
            track=self.track,
            frame_observation=frame_obs,
            temporal_observation=temp_obs,
            matched_detection=self.det,
            timestamp=1.0,
        )

        # Unselected criteria 3, 4, 6 are False in predicates
        self.assertFalse(result.predicates.A)
        self.assertFalse(result.predicates.Z)
        self.assertFalse(result.predicates.S)

        # But selected criteria 1, 2, 7 are True
        self.assertTrue(result.predicates.P)
        self.assertTrue(result.predicates.B)
        self.assertTrue(result.predicates.O)

        # Readiness gate passes into PENDING because all SELECTED criteria are valid!
        self.assertEqual(result.state, ReadinessState.PENDING)
        # Unselected predicate failure reasons must not pollute result reasons
        self.assertNotIn(ReasonCode.OUTSIDE_INTERACTION_ZONE, result.reasons)
        self.assertNotIn(ReasonCode.PERSON_MOVING, result.reasons)
        self.assertNotIn(ReasonCode.APPROACH_NOT_CONFIRMED, result.reasons)


class TestEventIsolation(unittest.TestCase):
    """Test strict event isolation between production FIRST_CONTACT_READY and demo DEMO_CRITERIA_READY."""

    def setUp(self) -> None:
        self.config = ContactConfig(
            distance_mode="mock",
            mock_distance_m=1.2,
            distance_max_m=1.5,
            eligibility_duration_seconds=2.0,
            tracking_loss_timeout_seconds=5.0,
        )
        self.box = (300.0, 200.0, 500.0, 600.0)
        self.det = PersonDetection(bbox_xyxy=self.box, confidence=0.92, class_id=0)

    def test_full_selection_emits_first_contact_ready(self) -> None:
        """When all seven criteria {1..7} are selected, emit production FirstContactReadyEvent."""
        engine = ReadinessEngine(config=self.config, criteria_selection=ALL_CRITERIA_NUMBERS)
        t0 = 1.0
        dt = 0.1

        # Run 2.1 seconds (21 frames)
        res = None
        for i in range(21):
            t = t0 + i * dt
            trk = _make_track(track_id=1, bbox=self.box, timestamp=t)
            res = engine.evaluate(
                trk,
                _make_frame_obs(timestamp=t),
                _make_temporal_obs(timestamp=t),
                self.det,
                timestamp=t,
            )

        self.assertIsNotNone(res)
        self.assertEqual(res.state, ReadinessState.READY)
        self.assertIsNotNone(res.event)
        self.assertIsInstance(res.event, FirstContactReadyEvent)
        self.assertEqual(res.event.event_type, "FIRST_CONTACT_READY")
        self.assertNotIsInstance(res.event, DemoCriteriaReadyEvent)

    def test_default_selection_emits_first_contact_ready(self) -> None:
        """Default {1, 2, 7} is production first-contact gate and emits FirstContactReadyEvent."""
        engine = ReadinessEngine(config=self.config)  # Default is {1, 2, 7}
        t0 = 1.0
        dt = 0.1

        res = None
        for i in range(21):
            t = t0 + i * dt
            trk = _make_track(track_id=1, bbox=self.box, timestamp=t)
            res = engine.evaluate(
                trk,
                _make_frame_obs(timestamp=t),
                _make_temporal_obs(timestamp=t),
                self.det,
                timestamp=t,
            )

        self.assertIsNotNone(res)
        self.assertEqual(res.state, ReadinessState.READY)
        self.assertIsNotNone(res.event)
        self.assertIsInstance(res.event, FirstContactReadyEvent)
        self.assertEqual(res.event.event_type, "FIRST_CONTACT_READY")
        self.assertNotIsInstance(res.event, DemoCriteriaReadyEvent)

    def test_subset_selection_emits_demo_criteria_ready_never_first_contact(self) -> None:
        """When an experimental subset of criteria is selected (e.g. {3, 6}), emit DemoCriteriaReadyEvent, NEVER FirstContactReadyEvent."""
        # Test with experimental subset {3, 6} (temporal alias)
        engine = ReadinessEngine(config=self.config, criteria_selection={3, 6})
        t0 = 1.0
        dt = 0.1

        res = None
        for i in range(21):
            t = t0 + i * dt
            trk = _make_track(track_id=1, bbox=self.box, timestamp=t)
            res = engine.evaluate(
                trk,
                _make_frame_obs(timestamp=t),
                _make_temporal_obs(timestamp=t),
                self.det,
                timestamp=t,
            )

        self.assertIsNotNone(res)
        self.assertEqual(res.state, ReadinessState.READY)
        self.assertIsNotNone(res.event)
        # MUST BE DemoCriteriaReadyEvent
        self.assertIsInstance(res.event, DemoCriteriaReadyEvent)
        self.assertEqual(res.event.event_type, "DEMO_CRITERIA_READY")
        self.assertEqual(res.event.selected_criteria, (3, 6))
        # MUST NEVER be FirstContactReadyEvent
        self.assertNotIsInstance(res.event, FirstContactReadyEvent)

        # Verify event dict serialization
        event_dict = res.event.to_dict()
        self.assertEqual(event_dict["event_type"], "DEMO_CRITERIA_READY")
        self.assertEqual(event_dict["selected_criteria"], [3, 6])
        self.assertEqual(event_dict["track_id"], 1)

    def test_one_shot_emission_per_visit_for_demo_criteria_ready(self) -> None:
        """DEMO_CRITERIA_READY must fire exactly once per visit and not repeat on subsequent frames."""
        engine = ReadinessEngine(config=self.config, criteria_selection={3, 6})
        t0 = 1.0
        dt = 0.1

        emitted_events = []
        for i in range(30):
            t = t0 + i * dt
            trk = _make_track(track_id=1, bbox=self.box, timestamp=t)
            res = engine.evaluate(
                trk,
                _make_frame_obs(timestamp=t),
                _make_temporal_obs(timestamp=t),
                self.det,
                timestamp=t,
            )
            if res.event is not None:
                emitted_events.append(res.event)

        # Exactly 1 event emitted
        self.assertEqual(len(emitted_events), 1)
        self.assertEqual(emitted_events[0].event_type, "DEMO_CRITERIA_READY")


class TestNonCriterionSafetyInvariants(unittest.TestCase):
    """Test safety invariant preservation across criteria selection modes."""

    def setUp(self) -> None:
        self.config = ContactConfig(
            distance_mode="mock",
            mock_distance_m=1.2,
            distance_max_m=1.5,
            eligibility_duration_seconds=2.0,
            tracking_loss_timeout_seconds=5.0,
        )
        self.box = (300.0, 200.0, 500.0, 600.0)
        self.det = PersonDetection(bbox_xyxy=self.box, confidence=0.92, class_id=0)

    def test_active_target_loss_blocks_readiness_regardless_of_selection(self) -> None:
        """Loss of active target track blocks readiness even when selected criteria evaluate True."""
        # Criteria selection {1, 2} (presence and bbox)
        engine = ReadinessEngine(config=self.config, criteria_selection={1, 2})

        # Frame 1: Target observed -> PENDING
        res1 = engine.evaluate(
            _make_track(track_id=1, bbox=self.box, timestamp=1.0),
            _make_frame_obs(timestamp=1.0),
            _make_temporal_obs(timestamp=1.0),
            self.det,
            timestamp=1.0,
        )
        self.assertEqual(res1.state, ReadinessState.PENDING)

        # Frame 2: Target disappears (track is None)
        res2 = engine.evaluate(
            None,
            _make_frame_obs(timestamp=1.1),
            _make_temporal_obs(timestamp=1.1),
            None,
            timestamp=1.1,
        )
        self.assertEqual(res2.state, ReadinessState.NOT_READY)
        self.assertIsNone(res2.event)
        self.assertIn(ReasonCode.TRACKING_LOST, res2.reasons)

    def test_backward_timestamp_resets_eligibility_regardless_of_selection(self) -> None:
        """Backward timestamp triggers fail-closed behavior regardless of criteria selection."""
        engine = ReadinessEngine(config=self.config, criteria_selection={1, 2, 7})

        # Frame 1 at t = 1.0 -> PENDING
        res1 = engine.evaluate(
            _make_track(track_id=1, bbox=self.box, timestamp=1.0),
            _make_frame_obs(timestamp=1.0),
            _make_temporal_obs(timestamp=1.0),
            self.det,
            timestamp=1.0,
        )
        self.assertEqual(res1.state, ReadinessState.PENDING)

        # Frame 2 at t = 0.9 (backward timestamp) -> fail-closed with INVALID_TIMESTAMP
        res2 = engine.evaluate(
            _make_track(track_id=1, bbox=self.box, timestamp=0.9),
            _make_frame_obs(timestamp=0.9),
            _make_temporal_obs(timestamp=0.9),
            self.det,
            timestamp=0.9,
        )
        self.assertEqual(res2.state, ReadinessState.NOT_READY)
        self.assertIsNone(res2.event)
        self.assertIn(ReasonCode.INVALID_TIMESTAMP, res2.reasons)
        self.assertEqual(res2.elapsed_eligibility, 0.0)


if __name__ == "__main__":
    unittest.main()
