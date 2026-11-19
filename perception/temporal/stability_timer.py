"""Stability duration and cumulative tolerance timer module for Milestone M5.

Tracks continuous stability duration (2.0s required) subject to a cumulative
noise tolerance budget (200 ms) for brief valid threshold excursions according
to Milestone M5 specification Section 5.13.

Architecture & State Transitions:
+-----------------------------------------------------------------------------+
|                          M5 STABILITY TIMER                                 |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Eligibility Predicate E_t:                                                 |
|    E_t = Person(DETECTED) AND BBox(VALID) AND Zone(INSIDE) AND Stationary   |
|    (All four predicates must be VALID and True)                             |
|                                                                             |
|  Accumulation & Tolerance:                                                  |
|    - E_t True: Timer accumulates elapsed time (t - T_start).                |
|    - E_t False (valid threshold excursion):                                 |
|        T_violation += Delta t                                               |
|        If T_violation <= 0.2s: attempt continues accumulating elapsed time. |
|        If T_violation > 0.2s:  budget exhausted -> RESET.                   |
|    - Elapsed >= 2.0s (within budget): COMPLETE (one-shot per attempt).      |
|                                                                             |
|  Immediate Hard Resets:                                                     |
|    - Tracking loss / track unobserved / ID change.                          |
|    - Required M4 observation UNKNOWN or INVALID.                            |
|    - Observation gap Delta t > 0.2s or non-monotonic timestamp.             |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

import math
from typing import Optional

from perception.readiness.observation_types import ObservationStatus
from perception.temporal.observation_types import StabilityObservation, StabilityValue

DEFAULT_STABILITY_DURATION_SECONDS: float = 2.0  # Frozen (2.0s continuous stability)
DEFAULT_STABILITY_TOLERANCE_SECONDS: float = 0.2  # Frozen (200 ms cumulative budget)


class StabilityTimer:
    """Timestamp-based stability timer with cumulative noise tolerance."""

    def __init__(
        self,
        required_duration_seconds: float = DEFAULT_STABILITY_DURATION_SECONDS,
        violation_tolerance_seconds: float = DEFAULT_STABILITY_TOLERANCE_SECONDS,
    ) -> None:
        """Initialize the stability timer.

        Args:
            required_duration_seconds: Continuous duration in seconds to achieve COMPLETE.
            violation_tolerance_seconds: Cumulative allowance in seconds for valid excursions.
        """
        if required_duration_seconds <= 0.0:
            raise ValueError(
                f"required_duration_seconds must be positive, got {required_duration_seconds}"
            )
        if violation_tolerance_seconds < 0.0:
            raise ValueError(
                f"violation_tolerance_seconds must be non-negative, got {violation_tolerance_seconds}"
            )
        if violation_tolerance_seconds >= required_duration_seconds:
            raise ValueError(
                f"violation_tolerance_seconds ({violation_tolerance_seconds}) must be strictly "
                f"less than required_duration_seconds ({required_duration_seconds})"
            )

        self.required_duration_seconds = float(required_duration_seconds)
        self.violation_tolerance_seconds = float(violation_tolerance_seconds)

        self._state: StabilityValue = StabilityValue.NOT_STARTED
        self._start_time: Optional[float] = None
        self._last_time: Optional[float] = None
        self._violation_duration: float = 0.0
        self._in_excursion: bool = False
        self._excursion_start_time: Optional[float] = None
        self._completed: bool = False

    @property
    def state(self) -> StabilityValue:
        """Current stability lifecycle state."""
        return self._state

    @property
    def elapsed_seconds(self) -> float:
        """Accumulated continuous attempt duration in seconds."""
        if self._start_time is None or self._last_time is None:
            return 0.0
        return max(0.0, float(self._last_time - self._start_time))

    @property
    def violation_seconds(self) -> float:
        """Cumulative duration of tolerated threshold excursions in seconds."""
        if self._in_excursion and self._excursion_start_time is not None and self._last_time is not None:
            return float(self._violation_duration + max(0.0, self._last_time - self._excursion_start_time))
        return float(self._violation_duration)

    @property
    def in_excursion(self) -> bool:
        """True if an active threshold excursion is currently underway."""
        return self._in_excursion

    def reset(self) -> None:
        """Hard reset: clear accumulated time and violation budget."""
        self._state = StabilityValue.RESET
        self._start_time = None
        self._last_time = None
        self._violation_duration = 0.0
        self._in_excursion = False
        self._excursion_start_time = None
        self._completed = False

    def update(
        self,
        is_eligible: bool,
        is_hard_reset: bool,
        timestamp: float,
    ) -> StabilityObservation:
        """Update stability timer for the current frame.

        Args:
            is_eligible: True if all four eligibility criteria (Person, BBox, Zone,
                         Stationary) are VALID and satisfied.
            is_hard_reset: True if tracking was lost, observations were UNKNOWN/INVALID/UNCERTAIN,
                           or temporal continuity was broken.
            timestamp: Monotonic timestamp of the frame in seconds.

        Returns:
            StabilityObservation containing current state, elapsed time, violation time,
            and reliability status.
        """
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite float, got {timestamp}")

        # Check for immediate hard reset
        if is_hard_reset:
            if self._state != StabilityValue.NOT_STARTED:
                self.reset()
            return StabilityObservation(
                value=self._state,
                elapsed_seconds=0.0,
                violation_seconds=0.0,
                status=ObservationStatus.VALID,
            )

        # Monotonicity check
        if self._last_time is not None and timestamp < self._last_time:
            self.reset()
            return StabilityObservation(
                value=StabilityValue.RESET,
                elapsed_seconds=0.0,
                violation_seconds=0.0,
                status=ObservationStatus.INVALID,
            )

        # Case 1: NOT_STARTED or RESET
        if self._state in (StabilityValue.NOT_STARTED, StabilityValue.RESET):
            if is_eligible:
                self._state = StabilityValue.ACCUMULATING
                self._start_time = timestamp
                self._last_time = timestamp
                self._violation_duration = 0.0
                self._in_excursion = False
                self._excursion_start_time = None
                self._completed = False
            else:
                self._last_time = timestamp

        # Case 2: ACCUMULATING
        elif self._state == StabilityValue.ACCUMULATING:
            assert self._start_time is not None
            self._last_time = timestamp
            elapsed = timestamp - self._start_time

            if is_eligible:
                # If recovering from an excursion, charge the complete sampled excursion duration
                # including the recovery interval (from first ineligible sample up to this eligible timestamp)
                if self._in_excursion and self._excursion_start_time is not None:
                    excursion_span = timestamp - self._excursion_start_time
                    self._violation_duration += excursion_span
                    self._in_excursion = False
                    self._excursion_start_time = None

                # Check if tolerance was exceeded upon recovery (with 1e-6 floating-point epsilon)
                if self._violation_duration > (self.violation_tolerance_seconds + 1e-6):
                    self.reset()
                elif elapsed >= self.required_duration_seconds and not self._completed:
                    self._state = StabilityValue.COMPLETE
                    self._completed = True
            else:
                # Frame is ineligible (valid threshold excursion)
                if not self._in_excursion:
                    self._in_excursion = True
                    self._excursion_start_time = timestamp

                # Check ongoing cumulative violation duration
                assert self._excursion_start_time is not None
                current_violation = self._violation_duration + (timestamp - self._excursion_start_time)
                if current_violation > (self.violation_tolerance_seconds + 1e-6):
                    # Cumulative violation budget exhausted!
                    self.reset()
                elif elapsed >= self.required_duration_seconds and not self._completed:
                    self._state = StabilityValue.COMPLETE
                    self._completed = True

        # Case 3: COMPLETE
        elif self._state == StabilityValue.COMPLETE:
            self._last_time = timestamp
            if not is_eligible:
                # Eligibility ended after completion: reset for next attempt
                self.reset()

        status = ObservationStatus.VALID if self._state != StabilityValue.UNKNOWN else ObservationStatus.INVALID

        return StabilityObservation(
            value=self._state,
            elapsed_seconds=self.elapsed_seconds,
            violation_seconds=self.violation_seconds,
            status=status,
        )

