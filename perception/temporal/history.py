"""Temporal observation history buffer and continuity management for Milestone M5.

Maintains a bounded, timestamp-ordered buffer of observed person center positions
and enforces the 5-consecutive-observation initialization rule and 200 ms maximum
observation gap constraint according to Milestone M5 specification.

Architecture & State Transitions:
+-----------------------------------------------------------------------------+
|                     M5 OBSERVATION HISTORY & INITIALIZATION                 |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Initial: UNINITIALIZED (count=0, history=[])                               |
|       |                                                                     |
|       +-- First valid observed center (t_0)                                 |
|       v                                                                     |
|  COLLECTING (count=1..4)                                                    |
|       |                                                                     |
|       +-- Delta t <= 0.2s: count += 1                                       |
|       +-- count >= 5 ------> READY (motion estimators active)               |
|       |                                                                     |
|  Reset Events (any state):                                                  |
|       +-- Tracking loss (track lost or unobserved)                          |
|       +-- Track ID change                                                   |
|       +-- Delta t > 0.2s (temporal gap discontinuity)                       |
|       +-- Delta t <= 0.0s (timestamp not strictly increasing)               |
|       +-- Invalid observation / Bbox validity failed                        |
|       |                                                                     |
|       v                                                                     |
|     RESET (history cleared, count=0)                                        |
|       |                                                                     |
|       +-- Next valid observed center ---> COLLECTING (count=1)              |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

from collections import deque
import math
from typing import Optional, Sequence

from perception.temporal.observation_types import InitializationState

DEFAULT_HISTORY_WINDOW_SECONDS: float = 1.0
DEFAULT_MAX_OBSERVATION_GAP_SECONDS: float = 0.2
DEFAULT_INITIAL_VALID_OBSERVATIONS: int = 5


class ObservationHistory:
    """Bounded, timestamp-ordered observation history buffer for a single track.

    Enforces:
        - Timestamp monotonicity: Delta t > 0.
        - Maximum observation gap: Delta t <= 0.2s.
        - Bounded history window: t - W <= t_i <= t (default W = 1.0s).
        - 5-consecutive-observation initialization requirement.
    """

    def __init__(
        self,
        history_window_seconds: float = DEFAULT_HISTORY_WINDOW_SECONDS,
        max_observation_gap_seconds: float = DEFAULT_MAX_OBSERVATION_GAP_SECONDS,
        initial_valid_observations: int = DEFAULT_INITIAL_VALID_OBSERVATIONS,
    ) -> None:
        """Initialize the observation history buffer.

        Args:
            history_window_seconds: Maximum time span of retained observations in seconds.
            max_observation_gap_seconds: Maximum allowed gap between consecutive observations.
            initial_valid_observations: Number of consecutive observations required to reach READY.

        Raises:
            ValueError: If parameters violate domain constraints.
        """
        if history_window_seconds <= 0.0:
            raise ValueError(
                f"history_window_seconds must be positive, got {history_window_seconds}"
            )
        if max_observation_gap_seconds <= 0.0:
            raise ValueError(
                f"max_observation_gap_seconds must be positive, got {max_observation_gap_seconds}"
            )
        if initial_valid_observations < 1:
            raise ValueError(
                f"initial_valid_observations must be >= 1, got {initial_valid_observations}"
            )

        self.history_window_seconds = float(history_window_seconds)
        self.max_observation_gap_seconds = float(max_observation_gap_seconds)
        self.initial_valid_observations = int(initial_valid_observations)

        # Buffer of (timestamp, (cx, cy))
        self._buffer: deque[tuple[float, tuple[float, float]]] = deque()
        self._consecutive_valid_count: int = 0
        self._state: InitializationState = InitializationState.UNINITIALIZED
        self._last_timestamp: Optional[float] = None

    @property
    def state(self) -> InitializationState:
        """Current initialization state."""
        return self._state

    @property
    def consecutive_valid_count(self) -> int:
        """Number of consecutive valid observations accumulated."""
        return self._consecutive_valid_count

    @property
    def is_ready(self) -> bool:
        """True if the history has collected at least the required initial observations."""
        return self._state == InitializationState.READY

    @property
    def observations(self) -> list[tuple[float, tuple[float, float]]]:
        """Current list of buffered observations in chronological order."""
        return list(self._buffer)

    @property
    def count(self) -> int:
        """Number of observations currently in the buffer."""
        return len(self._buffer)

    @property
    def duration_seconds(self) -> float:
        """Time span between oldest and newest observations in seconds."""
        if len(self._buffer) < 2:
            return 0.0
        return float(self._buffer[-1][0] - self._buffer[0][0])

    @property
    def latest_timestamp(self) -> Optional[float]:
        """Timestamp of the most recent observation, or None if buffer is empty."""
        return self._last_timestamp

    @property
    def latest_center(self) -> Optional[tuple[float, float]]:
        """Center coordinate of the most recent observation, or None if buffer is empty."""
        if len(self._buffer) == 0:
            return None
        return self._buffer[-1][1]

    def reset(self) -> None:
        """Hard reset: clear all buffered observations and reset initialization counter."""
        self._buffer.clear()
        self._consecutive_valid_count = 0
        self._state = InitializationState.RESET
        self._last_timestamp = None

    def add_observation(
        self,
        timestamp: float,
        center: tuple[float, float],
    ) -> bool:
        """Add a new valid observed center position to the history.

        Args:
            timestamp: Monotonic timestamp of the frame in seconds.
            center: Observed bounding-box center (cx, cy) in pixels.

        Returns:
            True if the observation was accepted and sequence remained continuous;
            False if an observation gap exceeded max_observation_gap_seconds (triggering reset).

        Raises:
            ValueError: If timestamp is non-finite or non-monotonic (timestamp <= last_timestamp),
                        or if center coordinates are non-finite.
        """
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite float, got {timestamp}")

        cx, cy = center
        if not (math.isfinite(cx) and math.isfinite(cy)):
            raise ValueError(f"center coordinates must be finite floats, got ({cx}, {cy})")

        # Monotonicity check
        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                raise ValueError(
                    f"Non-monotonic timestamp: current={timestamp} <= prev={self._last_timestamp}"
                )

            dt = timestamp - self._last_timestamp
            if dt > self.max_observation_gap_seconds:
                # Temporal discontinuity! Clear history, reset counter, transition to RESET
                self.reset()
                # Fresh start with this observation
                self._buffer.append((timestamp, (cx, cy)))
                self._last_timestamp = timestamp
                self._consecutive_valid_count = 1
                self._state = InitializationState.COLLECTING
                return False

        # Continuous observation
        self._buffer.append((timestamp, (cx, cy)))
        self._last_timestamp = timestamp
        self._consecutive_valid_count += 1

        # Prune observations older than history window
        cutoff = timestamp - self.history_window_seconds
        while len(self._buffer) > 0 and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

        # Update initialization state
        if self._consecutive_valid_count >= self.initial_valid_observations:
            self._state = InitializationState.READY
        else:
            self._state = InitializationState.COLLECTING

        return True

