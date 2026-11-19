"""Eligibility timer module for Milestone M6.

Tracks continuous satisfaction of all seven readiness predicates using monotonic
frame timestamps according to Section 5 of the M6 specification.
"""

from __future__ import annotations

import math
from typing import Optional

DEFAULT_ELIGIBILITY_DURATION_SECONDS: float = 2.0
DEFAULT_MAX_INTER_FRAME_GAP_SECONDS: float = 0.5
FLOATING_POINT_EPSILON: float = 1e-6


class EligibilityTimer:
    """Evaluates continuous multi-frame readiness eligibility."""

    def __init__(
        self,
        required_duration_seconds: float = DEFAULT_ELIGIBILITY_DURATION_SECONDS,
        max_inter_frame_gap_seconds: float = DEFAULT_MAX_INTER_FRAME_GAP_SECONDS,
    ) -> None:
        """Initialize eligibility timer.

        Args:
            required_duration_seconds: Required continuous valid time in seconds (default: 2.0).
            max_inter_frame_gap_seconds: Maximum allowed gap between consecutive valid frames (default: 0.5s).
        """
        if not math.isfinite(required_duration_seconds) or required_duration_seconds <= 0.0:
            raise ValueError(
                f"required_duration_seconds must be a positive finite float, got {required_duration_seconds}"
            )
        if not math.isfinite(max_inter_frame_gap_seconds) or max_inter_frame_gap_seconds <= 0.0:
            raise ValueError(
                f"max_inter_frame_gap_seconds must be a positive finite float, got {max_inter_frame_gap_seconds}"
            )
        self.required_duration_seconds: float = float(required_duration_seconds)
        self.max_inter_frame_gap_seconds: float = float(max_inter_frame_gap_seconds)
        self._start_time: Optional[float] = None
        self._last_time: Optional[float] = None
        self._is_complete: bool = False

    @property
    def is_running(self) -> bool:
        """True if the timer is actively accumulating and not yet complete."""
        return self._start_time is not None and not self._is_complete

    @property
    def is_complete(self) -> bool:
        """True if continuous eligibility duration has been achieved."""
        return self._is_complete

    @property
    def start_time(self) -> Optional[float]:
        """Monotonic timestamp when current continuous eligibility began."""
        return self._start_time

    @property
    def elapsed_seconds(self) -> float:
        """Accumulated continuous elapsed valid time in seconds."""
        if self._start_time is None or self._last_time is None:
            return 0.0
        return max(0.0, float(self._last_time - self._start_time))

    def reset(self) -> None:
        """Hard reset: clear accumulated eligibility and start timestamp."""
        self._start_time = None
        self._last_time = None
        self._is_complete = False

    def update(
        self,
        is_eligible: bool,
        timestamp: float,
    ) -> tuple[bool, float]:
        """Update eligibility timer with current frame evaluation.

        Args:
            is_eligible: True if all seven readiness predicates are simultaneously valid.
            timestamp: Monotonic capture timestamp of the frame in seconds.

        Returns:
            (is_complete, elapsed_seconds) tuple.

        Raises:
            ValueError: If timestamp is non-finite or non-monotonic.
        """
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite float, got {timestamp}")

        if self._last_time is not None and timestamp < self._last_time:
            self.reset()
            raise ValueError(
                f"Non-monotonic timestamp detected: previous={self._last_time}, current={timestamp}"
            )

        if not is_eligible:
            # Condition broken: reset timer immediately
            self.reset()
            self._last_time = timestamp
            return False, 0.0

        # Enforce continuous observation: reject inter-frame gaps larger than max_inter_frame_gap_seconds
        if self._last_time is not None:
            gap = timestamp - self._last_time
            if gap > self.max_inter_frame_gap_seconds:
                # Observation gap violated: reset continuity and start anew
                self.reset()
                self._start_time = timestamp
                self._last_time = timestamp
                return False, 0.0

        # Eligibility condition is met
        if self._start_time is None:
            self._start_time = timestamp
        self._last_time = timestamp

        elapsed = float(timestamp - self._start_time)
        if elapsed >= (self.required_duration_seconds - FLOATING_POINT_EPSILON):
            self._is_complete = True
            return True, elapsed

        return False, elapsed

