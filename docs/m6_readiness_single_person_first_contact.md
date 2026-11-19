# M6 — Readiness-Based Single-Person First Contact

**Document:** `m6_readiness_single_person_first_contact.md`  
**Milestone:** M6  
**Status:** Requirements Frozen — Approved for Implementation Planning  
**System:** Vision-Based First Contact  
**Scope:** Complete single-person perception-to-first-contact pipeline

---

## 1. Objective

M6 integrates the previously implemented milestones M1–M5 into a complete single-person first-contact pipeline for the tabletop social robot.

The system must continuously evaluate whether the currently tracked person is ready for first contact and emit a structured `FIRST_CONTACT_READY` event when all required conditions remain satisfied for two continuous seconds.

M6 introduces the readiness decision state machine, eligibility timer, visit lifecycle management, distance-provider integration, and event generation.

M6 must reuse existing M1–M5 capabilities rather than reimplementing their perception algorithms.

### 1.1 Scope

M6 includes:

1. Camera acquisition.
2. Person detection and single-person tracking.
3. Frame-level readiness observations.
4. Temporal behaviour estimation.
5. Single-person readiness state machine.
6. Two-second continuous eligibility logic.
7. Confidence-aware, fail-closed decisions.
8. Visit lifecycle management.
9. Structured `FIRST_CONTACT_READY` event.
10. End-to-end demo and validation.

### 1.2 Out of scope

- Multi-person simultaneous readiness evaluation.
- Person identification or facial recognition.
- Re-identification across independent visits.
- New person-detection models.
- New tracking algorithms.
- New face-orientation models.
- New temporal behaviour estimation algorithms.
- Downstream conversational or interaction execution.
- Treating mock distance as validated camera-derived distance.

---

## 2. System Integration Architecture

M6 integrates the existing perception pipeline with a new readiness decision layer.

```text
                 CAMERA ACQUISITION
                        |
                        v
                M1: PERSON DETECTION
                        |
                        v
                M2: BBOX SMOOTHING
                        |
                        v
                M3: PERSON TRACKING
                        |
                        v
             M4: FRAME-LEVEL OBSERVATIONS
                        |
                        v
             M5: TEMPORAL BEHAVIOUR
                        |
                        v
             DISTANCE PROVIDER
                        |
                        v
              M6: READINESS ENGINE
                        |
              +---------+---------+
              |                   |
              v                   v
        READINESS STATE     VISIT MANAGEMENT
              |                   |
              +---------+---------+
                        |
                        v
              FIRST_CONTACT_READY
                        |
                        v
                DOWNSTREAM SYSTEM
```

### 2.1 Integration requirements

- Preserve the existing M1–M5 implementations and frozen design decisions.
- Use the same captured-frame timestamp throughout the pipeline.
- Ensure M3, M4 and M5 evidence refers to the same person.
- Use current-frame detections rather than predicted-only tracking positions to establish presence.
- M6 must not independently estimate movement, face orientation, or person location.
- M6 consumes observations and performs deterministic decision-making.
- Preserve the single-active-track policy.
- Do not silently combine observations belonging to different people.

**Integration prerequisite:** Before M6 implementation, verify the M5 corrections and the actual M3–M4–M5 interfaces in the current repository. The implementation must use verified interfaces rather than assumed method names or data structures.

---

## 3. Readiness Predicate

The readiness predicate contains seven required conditions.

For active person \(i\):

\[
\boxed{
E_i(t)=P_i(t)\land B_i(t)\land A_i(t)
\land Z_i(t)\land D_i(t)\land S_i(t)\land O_i(t)
}
\]

Where:

| Predicate | Meaning | Required evidence |
|---|---|---|
| \(P\) | Person presence | Current-frame person detection is valid |
| \(B\) | Bounding-box validity | Valid bounding box associated with the active M3 track |
| \(A\) | Approach or initial presence | Confirmed approach OR reliable initial presence |
| \(Z\) | Interaction-zone membership | Person is inside the configured zone |
| \(D\) | Proximity | Reliable distance within the configured threshold |
| \(S\) | Stationary behaviour | M5 reports `STATIONARY + VALID` |
| \(O\) | Robot-facing orientation | M4 reports `FACING_ROBOT + VALID` |

All seven predicates must be true simultaneously.

### 3.1 Overall reliability predicate removed

An independent overall reliability predicate `Q` is not part of the M6 design.

Reliability is evaluated locally for each observation. An unreliable required observation cannot satisfy its corresponding predicate.

High confidence in one observation must never compensate for unreliable evidence in another.

---

## 4. Predicate Definitions

### 4.1 P — Person presence

`P = true` only when M4 provides valid current-frame person-presence evidence.

Requirements:

- A person must be detected in the current frame.
- Predicted-only Kalman positions cannot establish presence.
- Missing detection makes `P` unsatisfied.
- Tracking loss resets pending eligibility.

### 4.2 B — Bounding-box validity

`B = true` only when:

- The current detection has a valid bounding box.
- The detection is associated with the active M3 track.
- The detection and associated M4 observations refer to the same person.

A predicted-only bounding box cannot satisfy `B`.

### 4.3 A — Approach or initial presence

There are two valid ways to satisfy `A`.

**Path 1: Confirmed approach**

M5 confirms that the person approached the interaction zone.

**Path 2: Initial presence**

At system startup, a person is already inside the interaction zone and is reliably observed there.

The initial-presence path prevents a person already waiting in front of the robot from being permanently excluded simply because the system did not observe their approach.

Once either path establishes `A`, it remains satisfied for the current visit, subject to visit lifecycle termination.

A temporary tracking loss within the visit grace period does not create a new visit or independently invalidate an already established approach/initial-presence condition.

### 4.4 Z — Interaction-zone membership

M6 consumes M4's zone-membership observation.

| M4 observation | Predicate |
|---|---|
| `INSIDE + VALID` | `Z = true` |
| `OUTSIDE + VALID` | `Z = false` |
| `UNKNOWN` | Unsatisfied |
| `UNCERTAIN` | Unsatisfied |
| `INVALID` | Unsatisfied |

A reliable `OUTSIDE + VALID` observation terminates the current visit immediately.

An uncertain or invalid zone observation resets pending eligibility but does not, by itself, terminate the visit.

### 4.5 D — Distance condition

The distance predicate is:

\[
D_i(t)=
\begin{cases}
1 & \text{if distance is valid and } d_i(t)\leq d_{\max}\\
0 & \text{otherwise}
\end{cases}
\]

Initial configured threshold:

\[
d_{\max}=1.5\text{ m}
\]

The threshold must be configurable.

M6 does not estimate distance itself. It consumes an observation from an independent distance provider.

#### Distance modes

**MOCK mode**

- Used for laptop prototype and decision-logic testing.
- Supplies an explicit mock distance value.
- Must be clearly labelled `MOCK`.
- Must never be described as validated camera-derived distance.

**METRIC mode**

- Used when a validated metric-distance estimator is available.
- Supplies camera-derived metric distance with reliability information.
- Must be used for production operation.

Production mode must reject mock observations.

Unavailable, stale, invalid, or unreliable distance cannot satisfy `D`.

### 4.6 S — Stationary behaviour

M6 consumes M5's movement estimate.

`S = true` only when:

```text
movement_state = STATIONARY
reliability = VALID
```

The existing M5 thresholds remain unchanged:

- Velocity below 20 px/s.
- Displacement below 20 pixels over a 1-second window.

All other states make `S` unsatisfied:

- `MOVING`
- `UNKNOWN`
- `UNCERTAIN`
- `INVALID`

M6 must not independently calculate stationary behaviour.

### 4.7 O — Facing the robot

M6 consumes M4's head-orientation observation.

| M4 observation | Predicate |
|---|---|
| `FACING_ROBOT + VALID` | `O = true` |
| `NOT_FACING_ROBOT + VALID` | `O = false` |
| `UNKNOWN` | Unsatisfied |
| `UNCERTAIN` | Unsatisfied |
| `INVALID` | Unsatisfied |

Unknown orientation must never be interpreted as facing the robot.

---

## 5. Continuous Eligibility Logic

### 5.1 Eligibility duration

The person must satisfy all seven predicates continuously for:

\[
T_{\text{eligible}}=2.0\text{ seconds}
\]

Readiness is achieved when:

\[
E_i(t)=1
\]

continuously over an interval of at least two seconds.

### 5.2 Timer implementation

Requirements:

- Use monotonic frame timestamps.
- Use captured-frame timestamps consistently across M1–M6.
- Start the eligibility interval when all seven predicates become valid.
- Complete eligibility when elapsed valid time reaches two seconds.
- Do not use wall-clock time for eligibility calculations.
- Reject non-monotonic timestamps.
- Reject stale observations according to the configured pipeline observation-freshness policy.
- Do not silently substitute a different timestamp when one stage fails to provide the expected frame timestamp.

### 5.3 Reset policy

Immediately reset the pending eligibility timer when:

- Person presence becomes false.
- Bounding box becomes invalid.
- Approach/initial-presence requirement is unsatisfied.
- Person leaves the interaction zone.
- Distance becomes unavailable or unreliable.
- Person is too far away.
- Person is moving.
- Person is not facing the robot.
- Any required observation becomes uncertain.
- Any required observation becomes invalid.
- An observation is missing or stale.
- Tracking is lost.
- Required pipeline integration fails.

After a reset, all seven conditions must become valid again before a new two-second interval starts.

Previously accumulated eligibility time must never be resumed.

---

## 6. Readiness State Machine

M6 maintains a persistent readiness state.

Three externally visible readiness states are required.

| State | Meaning |
|---|---|
| `NOT_READY` | One or more required predicates are unsatisfied |
| `PENDING` | All predicates are valid; two-second timer is running |
| `READY` | Two continuous seconds have completed |

### 6.1 State transitions

```text
                 ALL PREDICATES VALID
        +----------------------------------+
        |                                  |
        v                                  |
    NOT_READY -----------------------> PENDING
        ^                                  |
        |                                  |
        |  ANY REQUIRED CONDITION          |
        |  FALSE / UNKNOWN / INVALID       |
        |                                  |
        +----------------------------------+
                                           |
                                           | 2 CONTINUOUS SECONDS
                                           v
                                         READY
                                           |
                                           |
                       REQUIRED EVIDENCE BECOMES INVALID
                                           |
                                           v
                                      NOT_READY
```

### 6.2 Persistent state fields

Maintain at least:

```python
readiness_state = {
    "state": "NOT_READY",
    "eligible_since": None,
    "ready_since": None,
    "event_emitted": False,
    "visit_id": None,
}
```

Implementation may use an appropriate class or typed structure, provided the semantics remain unchanged.

### 6.3 Revocable readiness

`READY` is a live state, not a permanent state.

If any required predicate becomes false, uncertain, invalid, missing, or unavailable:

- Transition from `READY` to `NOT_READY`.
- Keep the visit active unless a visit termination condition occurs.
- Preserve `event_emitted = True`.
- Do not emit another event for the same visit.

If all conditions become valid again, start a fresh two-second interval.

---

## 7. Event Semantics

### 7.1 Event definition

M6 emits:

```text
FIRST_CONTACT_READY
```

when the person first transitions into `READY` during the current visit.

The event is emitted exactly once per visit.

### 7.2 State versus event

These are separate concepts:

**Readiness state**

Answers:

"Is this person ready right now?"

It is continuously evaluated and updated.

**FIRST_CONTACT_READY event**

Answers:

"Has this person achieved first-contact readiness for this visit?"

It is a one-shot notification.

### 7.3 Event-emission rules

1. Emit the event on the first transition into `READY`.
2. Set `event_emitted = True`.
3. Do not emit duplicate events during the same visit.
4. Continue evaluating readiness after event emission.
5. Allow `READY` to transition back to `NOT_READY`.
6. A subsequent transition to `READY` in the same visit must not emit another event.
7. A new visit receives a new event-emission allowance.

### 7.4 Event structure

The structured event must contain:

```python
{
    "event_type": "FIRST_CONTACT_READY",
    "event_id": "...",
    "visit_id": "...",
    "track_id": "...",
    "timestamp": 0.0,
    "readiness_duration": 2.0,
    "distance": {
        "value_m": 1.2,
        "source": "MOCK"
    }
}
```

Field semantics:

| Field | Meaning |
|---|---|
| `event_type` | Event identifier |
| `event_id` | Unique event identifier |
| `visit_id` | Current visit identifier |
| `track_id` | Active M3 track identifier |
| `timestamp` | Monotonic event/frame timestamp |
| `readiness_duration` | Completed eligibility duration |
| `distance.value_m` | Distance at event generation |
| `distance.source` | `MOCK` or `METRIC` |

The event must preserve the actual distance source.

Downstream interaction execution and completion are outside M6.

---

## 8. Visit Lifecycle Management

M6 maintains one active visit for the selected person.

### 8.1 Visit creation

A visit may be created when:

- A person is selected as the active target.
- A person is already present at startup and reliable initial-presence evidence is available.

Each visit receives a unique `visit_id`.

The visit identifier is distinct from the M3 track identifier.

### 8.2 Single-active-person policy

- Only one person is evaluated for first-contact readiness at a time.
- The active target remains locked throughout the visit.
- No target switching occurs during an active visit.
- Other people are not evaluated by M6.
- Once the visit ends, the existing M3/M4 selection policy determines the next eligible person.

M6 must not introduce a new multi-person selection or tracking algorithm.

### 8.3 Tracking-loss timeout

Configured tracking-loss grace period:

\[
T_{\text{loss}}=5.0\text{ seconds}
\]

When tracking is lost:

1. Immediately reset pending eligibility.
2. Start the five-second tracking-loss timeout.
3. Retain the current visit during the grace period.
4. Do not terminate the visit immediately.

#### Reacquisition within five seconds

- Retain the same `visit_id`.
- Do not create a new visit.
- If the previous state was `PENDING`, start a fresh two-second interval.
- If the previous state was `READY`, retain `event_emitted = True`.
- Re-evaluate current readiness from fresh observations.
- Do not emit another event for the same visit.

#### No reacquisition within five seconds

- End the visit.
- Clear active visit state.
- The next selected person receives a new visit.

### 8.4 Reliable zone exit

A reliable `OUTSIDE + VALID` zone observation ends the current visit immediately.

This rule takes precedence over the tracking-loss grace policy.

### 8.5 Uncertain zone evidence

If zone membership becomes uncertain or invalid:

- Reset pending eligibility.
- Keep the visit active.
- Do not terminate the visit solely because of uncertainty.

### 8.6 Visit termination

When a visit ends:

- Clear the active visit.
- Stop readiness evaluation for that visit.
- Clear its pending eligibility timer.
- Retain event history as needed for logging and audit.
- Permit selection of the next eligible person.

A new visit receives:

- New `visit_id`.
- `readiness_state = NOT_READY`.
- `event_emitted = False`.
- Fresh eligibility timing state.

---

## 9. Reason Codes and Observability

M6 must make its decisions explainable.

Maintain reason codes describing why readiness is not currently satisfied.

Suggested reason codes:

```text
NO_PERSON
INVALID_BBOX
APPROACH_NOT_CONFIRMED
OUTSIDE_INTERACTION_ZONE
DISTANCE_UNAVAILABLE
DISTANCE_UNRELIABLE
TOO_FAR
PERSON_MOVING
NOT_FACING_ROBOT
PERCEPTION_UNCERTAIN
TRACKING_LOST
STALE_OBSERVATION
INVALID_TIMESTAMP
PIPELINE_FAILURE
```

Multiple reason codes may be active simultaneously.

The reason-code set should be deterministic and derived from the actual predicate evaluation.

### 9.1 State-transition logging

Log at least:

- Timestamp.
- Active `track_id`.
- Current `visit_id`.
- Previous readiness state.
- New readiness state.
- Predicate values.
- Active reason codes.
- Eligibility start time.
- Eligibility elapsed duration.
- Event-emission status.
- Visit lifecycle transitions.

Logging must not modify readiness decisions.

---

## 10. Distance Provider Interface

Distance estimation is independent of M6.

M6 consumes a distance observation with the following conceptual contract:

```python
DistanceObservation(
    distance_m=1.2,
    source="MOCK",
    status="VALID",
    timestamp=...
)
```

Required fields:

| Field | Description |
|---|---|
| `distance_m` | Estimated or supplied distance in metres |
| `source` | `MOCK` or `METRIC` |
| `status` | Observation reliability/status |
| `timestamp` | Associated frame timestamp |

Requirements:

- Distance threshold defaults to 1.5 m.
- Threshold must be configurable.
- Mock distance must be explicit.
- Production configuration must reject mock observations.
- Missing, stale, invalid, or uncertain distance cannot satisfy `D`.
- M6 must not calculate metric distance from bounding-box size or image position.

The actual interface and types must be aligned with existing repository conventions during implementation.

---

## 11. Failure Handling

### 11.1 Camera failure

- Do not make new readiness decisions without fresh frames.
- Reset pending eligibility.
- Log the failure.
- Do not emit an event from stale observations.

### 11.2 Detection failure

- Reset pending eligibility.
- Do not treat predicted-only tracking positions as current person presence.

### 11.3 M3/M4/M5 interface failure

- Fail closed.
- Reset pending eligibility.
- Do not combine observations from different people or frames.
- Log the failure.

### 11.4 Distance-provider failure

- `D` becomes unsatisfied.
- Reset pending eligibility.
- Do not substitute a fabricated or stale distance.

### 11.5 Invalid configuration

Invalid configuration must cause a clear startup failure.

Validate at least:

- Distance threshold.
- Distance-provider mode.
- Required observation configuration.
- Interaction-zone configuration.
- Eligibility duration.
- Tracking-loss timeout.
- Observation freshness configuration.

### 11.6 Application shutdown

- Terminate the active visit.
- Clear pending eligibility.
- Do not emit a readiness event as part of shutdown.
- Release camera and pipeline resources safely.

---

## 12. Demo Requirements

The complete M6 demo must demonstrate the following scenario:

1. Start the camera and pipeline.
2. Detect one person.
3. Establish the active track and visit.
4. Confirm approach or initial presence.
5. Verify interaction-zone membership.
6. Receive a valid distance observation.
7. Observe the person becoming stationary.
8. Observe the person facing the robot.
9. Satisfy all seven readiness predicates simultaneously.
10. Maintain continuous eligibility for two seconds.
11. Emit exactly one `FIRST_CONTACT_READY` event.
12. Continue evaluating readiness after event emission.

### 12.1 Required visual/debug information

The demo should display or log:

- Current frame.
- Person detection.
- Active tracking ID.
- Interaction-zone polygon.
- Zone-membership state.
- Distance value and source.
- Approach state.
- Movement/stationary state.
- Orientation state.
- All seven readiness predicates.
- Current readiness state.
- Eligibility timer.
- Active reason codes.
- Current visit ID.
- Event-emission status.

### 12.2 Mock distance demonstration

If validated metric distance is unavailable on the laptop:

- Use the explicit `MOCK` distance provider.
- Display a clear `MOCK DISTANCE` indicator.
- Allow deterministic distance values for replay/testing.
- Do not describe the demonstration as validated physical-distance estimation.

---

## 13. Testing Requirements

M6 must have unit, integration, replay, and end-to-end tests.

### 13.1 Predicate tests

Test every predicate independently:

- Valid person presence.
- Missing person presence.
- Valid bounding box.
- Invalid bounding box.
- Confirmed approach.
- Initial presence at startup.
- Inside interaction zone.
- Outside interaction zone.
- Valid distance within threshold.
- Distance beyond threshold.
- Missing or unreliable distance.
- Valid stationary observation.
- Moving observation.
- Valid facing observation.
- Not-facing observation.
- Unknown or uncertain orientation.

### 13.2 Eligibility timer tests

Test:

- All conditions valid for less than two seconds.
- All conditions valid for exactly two seconds.
- All conditions valid for more than two seconds.
- One predicate fails before completion.
- One predicate becomes uncertain.
- One predicate becomes invalid.
- Missing observation.
- Stale observation.
- Non-monotonic timestamp.
- Fresh eligibility interval after reset.
- No preservation of previously accumulated time.

### 13.3 State-machine tests

Test:

```text
NOT_READY -> PENDING
PENDING -> READY
PENDING -> NOT_READY
READY -> NOT_READY
NOT_READY -> PENDING
```

Verify that readiness can be revoked and re-established within the same visit.

### 13.4 Event tests

Verify:

- Event emitted exactly once per visit.
- Event not emitted before two seconds.
- No duplicate event after readiness revocation.
- No duplicate event after reacquisition within the grace period.
- New visit allows a new event.
- Event contains correct visit, track, timestamp, and distance-source information.

### 13.5 Visit lifecycle tests

Test:

- Initial presence at startup.
- New person arrival.
- Reliable zone exit.
- Tracking loss during `PENDING`.
- Tracking loss during `READY`.
- Reacquisition before five seconds.
- Reacquisition after five seconds.
- No reacquisition.
- Uncertain zone observation.
- Active-target locking.
- New visit creation after termination.

### 13.6 Integration and replay tests

Verify:

- M1–M5 integration.
- Same-person association across M3/M4/M5.
- Consistent frame timestamps.
- No predicted-only person presence.
- Correct distance-provider integration.
- End-to-end readiness transition.
- One-shot event generation.
- Correct reset and reacquisition semantics.

Synthetic tests must not be presented as real-world accuracy measurements.

---

## 14. Performance and Evaluation

Measure:

- Camera acquisition FPS.
- End-to-end pipeline latency.
- M1–M5 component latency where measurable.
- M6 decision latency.
- Total CPU usage.
- Memory usage.
- Readiness event timing.

Report average, p50, p95, and p99 latency where sufficient samples are available.

Use a sufficiently long benchmark for meaningful tail-latency estimates.

Do not invent performance targets or claim a performance improvement without measurements.

---

## 15. Exit Criteria

M6 is complete only when the following criteria are satisfied.

### Functional

- [ ] Complete single-person M1–M5 pipeline integrated.
- [ ] Seven-predicate readiness condition implemented.
- [ ] Two-second continuous eligibility logic implemented.
- [ ] Approach and initial-presence paths verified.
- [ ] Mock and metric distance-provider modes supported.
- [ ] Fail-closed uncertainty handling implemented.
- [ ] Readiness state continuously evaluated.
- [ ] Readiness can be revoked.
- [ ] One-shot event emitted per visit.
- [ ] Five-second tracking-loss policy implemented.
- [ ] Reliable zone exit terminates the visit.
- [ ] Single-active-person locking implemented.

### Reliability

- [ ] Same-person association verified.
- [ ] Same-frame timestamp consistency verified.
- [ ] Predicted-only positions cannot trigger readiness.
- [ ] Missing or uncertain evidence cannot trigger readiness.
- [ ] Tracking-loss and reacquisition behaviour verified.
- [ ] No duplicate events within a visit.

### Testing

- [ ] Unit tests pass.
- [ ] State-machine tests pass.
- [ ] Timer tests pass.
- [ ] Visit lifecycle tests pass.
- [ ] Integration tests pass.
- [ ] Replay tests pass.
- [ ] End-to-end demo passes.

### Documentation

- [ ] Architecture and interface details documented.
- [ ] Configuration documented.
- [ ] Event schema documented.
- [ ] Test results reported accurately.
- [ ] Performance measurements reported.
- [ ] Limitations documented.
- [ ] Mock-distance limitations explicitly stated.

---

## 16. Implementation Constraints

The implementation must follow the project's existing governance and engineering standards.

Before coding:

1. Inspect the complete repository tree.
2. Inspect `git status --short`.
3. Read all applicable governance documents.
4. Inspect existing M1–M5 source, configuration, tests, demos, and reports.
5. Verify the actual M3/M4/M5 interfaces.
6. Verify M5 correction and integration status.
7. Identify the correct locations for new M6 files.

During implementation:

- Preserve existing user changes.
- Do not modify frozen M1–M5 behaviour without documenting a blocking defect.
- Do not introduce unrelated refactoring.
- Do not introduce new perception models.
- Do not silently alter any of the 21 frozen M6 decisions.
- Do not claim tests or benchmarks that were not executed.
- Do not commit, push, or modify Git history.
- Update governance or milestone-status documents only when explicitly requested.
- Stop after M6 implementation and validation for review.

---

## 17. Final Design Summary

M6 provides a deterministic, interpretable first-contact readiness layer on top of the existing M1–M5 perception pipeline.

The central decision rule is:

\[
\boxed{
\text{FIRST\_CONTACT\_READY}
\iff
\bigwedge_{x\in\{P,B,A,Z,D,S,O\}}x
\text{ remains valid for 2 continuous seconds}
}
\]

The system continuously maintains the current readiness state while emitting a single `FIRST_CONTACT_READY` event per visit.

The design supports both people who approach after startup and people already waiting in front of the robot, distinguishes prototype mock distance from validated metric distance, handles uncertainty conservatively, and preserves visit continuity during brief tracking loss.

**M6 is intended to deliver a complete, testable, observable single-person first-contact pipeline without expanding the project into multi-person readiness or new perception research.**

