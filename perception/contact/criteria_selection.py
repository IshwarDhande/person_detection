"""Configurable readiness criteria selection for Milestone M6.

Provides stable numerical identifiers (1..7), aliases (--all, --stationary, --temporal),
CLI parsing, validation, and criteria-gate evaluation for first-contact readiness.

Fixed Identifier Mapping:
| Number | ID | Criterion                    | Predicate Attribute |
|--------|----|------------------------------|---------------------|
| 1      | P  | Person presence              | p_presence          |
| 2      | B  | Bounding-box validity        | b_bbox              |
| 3      | A  | Approach / initial presence  | a_approach          |
| 4      | Z  | Interaction-zone membership  | z_zone              |
| 5      | D  | Distance                     | d_distance          |
| 6      | S  | Stationary                   | s_stationary        |
| 7      | O  | Facing robot                 | o_orientation       |
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from perception.contact.contact_types import ReadinessPredicates, ReasonCode

# ---------------------------------------------------------------------------
# Stable Criterion Constants
# ---------------------------------------------------------------------------

ALL_CRITERIA_NUMBERS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7})
DEFAULT_CRITERIA_NUMBERS: frozenset[int] = frozenset({1, 2, 7})
STATIONARY_ALIAS_NUMBERS: frozenset[int] = frozenset({1, 2, 4, 7})
TEMPORAL_ALIAS_NUMBERS: frozenset[int] = frozenset({3, 6})

CRITERION_DETAILS: dict[int, dict[str, Any]] = {
    1: {
        "id": "P",
        "name": "Person presence",
        "attr": "p_presence",
        "reasons": {
            ReasonCode.NO_PERSON,
            ReasonCode.TRACKING_LOST,
            ReasonCode.TARGET_MISMATCH,
            ReasonCode.STALE_OBSERVATION,
            ReasonCode.TIMESTAMP_MISMATCH,
        },
    },
    2: {
        "id": "B",
        "name": "Bounding-box validity",
        "attr": "b_bbox",
        "reasons": {
            ReasonCode.INVALID_BBOX,
            ReasonCode.TARGET_MISMATCH,
        },
    },
    3: {
        "id": "A",
        "name": "Approach / initial presence",
        "attr": "a_approach",
        "reasons": {
            ReasonCode.APPROACH_NOT_CONFIRMED,
        },
    },
    4: {
        "id": "Z",
        "name": "Interaction-zone membership",
        "attr": "z_zone",
        "reasons": {
            ReasonCode.OUTSIDE_INTERACTION_ZONE,
        },
    },
    5: {
        "id": "D",
        "name": "Distance",
        "attr": "d_distance",
        "reasons": {
            ReasonCode.DISTANCE_UNAVAILABLE,
            ReasonCode.DISTANCE_UNRELIABLE,
            ReasonCode.TOO_FAR,
            ReasonCode.TARGET_MISMATCH,
            ReasonCode.STALE_OBSERVATION,
            ReasonCode.TIMESTAMP_MISMATCH,
        },
    },
    6: {
        "id": "S",
        "name": "Stationary",
        "attr": "s_stationary",
        "reasons": {
            ReasonCode.PERSON_MOVING,
            ReasonCode.TARGET_MISMATCH,
            ReasonCode.STALE_OBSERVATION,
            ReasonCode.TIMESTAMP_MISMATCH,
            ReasonCode.TRACKING_LOST,
        },
    },
    7: {
        "id": "O",
        "name": "Facing robot",
        "attr": "o_orientation",
        "reasons": {
            ReasonCode.NOT_FACING_ROBOT,
            ReasonCode.PERCEPTION_UNCERTAIN,
        },
    },
}

VALID_CRITERIA_NUMBERS_STR = "1:P, 2:B, 3:A, 4:Z, 5:D, 6:S, 7:O"


# ---------------------------------------------------------------------------
# Data Structure for Active Criteria Selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CriteriaSelection:
    """Represents a validated subset or complete set of readiness criteria."""

    selected_numbers: frozenset[int]
    selection_name: str = "custom"  # "default", "all", "stationary", "temporal", "explicit", "custom"

    def __post_init__(self) -> None:
        if not self.selected_numbers:
            raise ValueError("CriteriaSelection cannot have an empty set of selected criteria.")
        invalid = self.selected_numbers - ALL_CRITERIA_NUMBERS
        if invalid:
            raise ValueError(
                f"Invalid criterion numbers: {sorted(invalid)}. Valid choices are 1..7 ({VALID_CRITERIA_NUMBERS_STR})."
            )

    @property
    def is_all(self) -> bool:
        """True if all seven criteria {1..7} are selected (full M6 production mode)."""
        return self.selected_numbers == ALL_CRITERIA_NUMBERS

    @property
    def is_subset(self) -> bool:
        """True if a strict subset (< 7 criteria) is selected (demo/experimental mode)."""
        return not self.is_all

    @property
    def sorted_numbers(self) -> tuple[int, ...]:
        """Tuple of selected criterion numbers in ascending order."""
        return tuple(sorted(self.selected_numbers))

    @classmethod
    def all(cls) -> CriteriaSelection:
        """Create selection containing all 7 criteria {1..7} (full M6 mode)."""
        return cls(ALL_CRITERIA_NUMBERS, selection_name="all")

    @classmethod
    def default(cls) -> CriteriaSelection:
        """Create default demo selection {1, 2, 7}."""
        return cls(DEFAULT_CRITERIA_NUMBERS, selection_name="default")

    def is_selected(self, number: int) -> bool:
        """Check if a criterion number is among the active selection."""
        return number in self.selected_numbers

    def is_satisfied(self, predicates: ReadinessPredicates) -> bool:
        """Evaluate whether all selected criteria are satisfied for the given predicates.

        E_demo(t) = \\bigwedge_{i \\in C} P_i(t)
        """
        for num in self.selected_numbers:
            attr = CRITERION_DETAILS[num]["attr"]
            if not getattr(predicates, attr):
                return False
        return True

    def get_criterion_status(
        self, number: int, predicates: ReadinessPredicates
    ) -> tuple[str, str, bool, bool]:
        """Return (id, name, is_selected, is_satisfied) for a specific criterion."""
        info = CRITERION_DETAILS[number]
        is_sel = number in self.selected_numbers
        is_sat = bool(getattr(predicates, info["attr"]))
        return info["id"], info["name"], is_sel, is_sat

    def format_summary(self) -> str:
        """Format a human-readable summary of the selection."""
        ids = [CRITERION_DETAILS[num]["id"] for num in sorted(self.selected_numbers)]
        nums_str = ",".join(str(n) for n in sorted(self.selected_numbers))
        ids_str = ",".join(ids)
        if self.is_all:
            return f"FULL_M6 [ALL 7] ({{{nums_str}}} -> [{ids_str}])"
        if self.selected_numbers == DEFAULT_CRITERIA_NUMBERS:
            return f"DEFAULT_M6 [P,B,O] ({{{nums_str}}} -> [{ids_str}])"
        return f"DEMO_SUBSET [{self.selection_name}] ({{{nums_str}}} -> [{ids_str}])"


# ---------------------------------------------------------------------------
# CLI Parsing & Validation Helpers
# ---------------------------------------------------------------------------

def parse_criteria_string(criteria_str: str) -> frozenset[int]:
    """Parse and validate a comma-separated criterion numbers string.

    Args:
        criteria_str: Comma-separated string, e.g. "1,2,6" or "1,2,3,4,5,6,7".

    Returns:
        frozenset of valid criterion integers.

    Raises:
        ValueError: If input is empty, malformed, contains duplicates, or invalid numbers.
    """
    if criteria_str is None:
        raise ValueError("Criteria string cannot be None.")

    stripped = criteria_str.strip()
    if not stripped:
        raise ValueError("Criteria selection cannot be empty.")

    # Check for empty components e.g. "1,,2" or trailing commas "1,2,"
    parts = stripped.split(",")
    parsed_numbers: list[int] = []

    for i, part in enumerate(parts):
        token = part.strip()
        if not token:
            raise ValueError(
                f"Malformed criteria list '{criteria_str}': empty item at position {i + 1}."
            )
        try:
            num = int(token)
        except ValueError:
            raise ValueError(
                f"Invalid criterion '{token}' in '{criteria_str}'. Must be an integer 1..7 ({VALID_CRITERIA_NUMBERS_STR})."
            )

        if num not in ALL_CRITERIA_NUMBERS:
            raise ValueError(
                f"Unknown criterion number {num}. Valid choices are 1..7 ({VALID_CRITERIA_NUMBERS_STR})."
            )

        if num in parsed_numbers:
            raise ValueError(
                f"Duplicate criterion number {num} in '{criteria_str}'. Each criterion must be specified at most once."
            )

        parsed_numbers.append(num)

    return frozenset(parsed_numbers)


def add_criteria_arguments(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive criteria selection arguments to an ArgumentParser.

    Args:
        parser: The argparse.ArgumentParser instance to extend.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        dest="criteria_all",
        help="Select all seven criteria {1, 2, 3, 4, 5, 6, 7} for full M6 production first-contact readiness.",
    )
    group.add_argument(
        "--stationary",
        action="store_true",
        dest="criteria_stationary",
        help="Select stationary readiness criteria {1, 2, 4, 7} (presence, bbox, interaction zone, facing robot).",
    )
    group.add_argument(
        "--temporal",
        action="store_true",
        dest="criteria_temporal",
        help="Select temporal behaviour criteria {3, 6} (approach and stationary movement).",
    )
    group.add_argument(
        "--criteria",
        type=str,
        default=None,
        dest="criteria",
        metavar="LIST",
        help=(
            "Explicit comma-separated criterion numbers 1..7 (e.g. '1,2,6' or '1,2,3,4,5,6,7'). "
            f"Valid numbers: {VALID_CRITERIA_NUMBERS_STR}. Default when omitted: {{1, 2, 7}}."
        ),
    )


def resolve_criteria_selection(
    args: argparse.Namespace,
    parser: Optional[argparse.ArgumentParser] = None,
) -> CriteriaSelection:
    """Resolve and validate the active CriteriaSelection from parsed CLI arguments.

    Args:
        args: Parsed argparse.Namespace.
        parser: Optional ArgumentParser for raising clean user-facing CLI errors.

    Returns:
        Validated CriteriaSelection instance.

    Raises:
        SystemExit: If parser is provided and validation fails.
        ValueError: If parser is None and validation fails.
    """
    def _fail(msg: str) -> None:
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)

    # Count how many criteria arguments were supplied
    is_all = getattr(args, "criteria_all", False)
    is_stationary = getattr(args, "criteria_stationary", False)
    is_temporal = getattr(args, "criteria_temporal", False)
    explicit_str = getattr(args, "criteria", None)

    active_modes = sum([
        bool(is_all),
        bool(is_stationary),
        bool(is_temporal),
        explicit_str is not None,
    ])

    if active_modes > 1:
        _fail("Conflicting criteria selection options. Exactly one selection mode is allowed per invocation.")

    if is_all:
        return CriteriaSelection(ALL_CRITERIA_NUMBERS, selection_name="all")

    if is_stationary:
        return CriteriaSelection(STATIONARY_ALIAS_NUMBERS, selection_name="stationary")

    if is_temporal:
        return CriteriaSelection(TEMPORAL_ALIAS_NUMBERS, selection_name="temporal")

    if explicit_str is not None:
        try:
            numbers = parse_criteria_string(explicit_str)
        except ValueError as exc:
            _fail(str(exc))
            return None  # Unreachable after _fail

        name = "all" if numbers == ALL_CRITERIA_NUMBERS else "explicit"
        return CriteriaSelection(numbers, selection_name=name)

    # Default selection when no argument is provided: {1, 2, 7}
    return CriteriaSelection(DEFAULT_CRITERIA_NUMBERS, selection_name="default")


def filter_reasons_for_criteria(
    reasons: Sequence[ReasonCode],
    selected_numbers: frozenset[int],
) -> list[ReasonCode]:
    """Filter reasons to those relevant to selected criteria and global safety checks.

    Global reasons (INVALID_TIMESTAMP, PIPELINE_FAILURE) are always retained.
    """
    relevant_reasons: set[ReasonCode] = {
        ReasonCode.INVALID_TIMESTAMP,
        ReasonCode.PIPELINE_FAILURE,
    }
    for num in selected_numbers:
        relevant_reasons.update(CRITERION_DETAILS[num]["reasons"])

    return [r for r in reasons if r in relevant_reasons]
