# Round 3 --- Multi-Person Tracking, Target Selection, and Visit Lifecycle

**Project:** Vision-Based First Contact for a Reception Social Robot\
**Status:** Requirements gathering complete; technical validation and
final system integration remain pending\
**Scope:** Vision-only person tracking, target selection, candidate
management, and lifecycle coordination with the downstream interaction
module.

------------------------------------------------------------------------

## 1. Purpose and scope

Round 3 defines how the vision system manages multiple people in the
reception camera view, maintains temporary tracking identities, selects
a first-contact target, manages other eligible visitors, and coordinates
target lifecycle with a separate interaction module.

### In scope

-   Tracking all visible people simultaneously.
-   Maintaining temporary tracking IDs across frames.
-   Handling partial occlusion and temporary track loss.
-   Selecting and locking a first-contact target.
-   Evaluating other visitors while one target is selected.
-   Maintaining an ordered queue of eligible candidates.
-   Handling queue overflow, candidate invalidation, and reacquisition.
-   Separating track, target, and visit identities.
-   Coordinating target lifecycle with the downstream interaction
    module.

### Out of scope

-   Speech recognition, dialogue generation, and conversation policy.
-   Gesture generation and nonverbal interaction.
-   Face recognition or persistent biometric identity recognition.
-   Selection of a particular detector, tracker, re-identification
    model, or tracking library.
-   Final numerical calibration of distance tolerance, association
    confidence, or tracking thresholds.

Round 3 builds on Round 1 (hardware/environment) and Round 2 (detection,
approach, proximity, readiness, and first-contact trigger).

------------------------------------------------------------------------

## 2. Dependencies on Round 1 and Round 2

The following earlier requirements remain applicable:

-   Stationary tabletop social robot in an office reception.
-   Forward-facing, head-mounted RGB camera.
-   1280×720 resolution at 30 FPS.
-   RGB-only perception; no depth stream.
-   Environment includes normal artificial lighting, daylight, low
    lighting, crowds, and people carrying bags or objects.
-   Partially visible people should be detected whenever possible with
    sufficient confidence.
-   Approach must be observed using position and movement over multiple
    frames.
-   Interaction zone uses a hybrid image-space polygon and estimated
    physical distance.
-   Initial near-distance target is 1.5 m; this is not yet validated for
    RGB-only distance estimation.
-   Candidate must have an observed approach, be in the interaction
    zone, be within the distance threshold with reliable estimation, be
    stationary, and be facing the robot.
-   All readiness conditions must hold continuously for 2 seconds.
-   A person already waiting nearby does not qualify without an observed
    approach.
-   A passerby is tracked but must not trigger first contact unless all
    readiness conditions are met.
-   Leaving the interaction zone during the 2-second readiness interval
    resets that interval.
-   Uncertain distance or orientation causes HOLD; unreliable perception
    does not trigger.
-   A temporarily disappearing selected person causes
    PAUSE/reacquisition rather than immediate reset.
-   If multiple people qualify, select the person who qualified first,
    subject to the same-frame tie-break specified in Round 3.
-   First contact is intended once per visit.
-   `FIRST_CONTACT_READY` is the conceptual handoff event to the
    downstream interaction module.

------------------------------------------------------------------------

## 3. Round 3 decisions

### 3.1 Tracking scope and identity

  -----------------------------------------------------------------------
  ID                      Requirement             Decision
  ----------------------- ----------------------- -----------------------
  R3-01                   Track scope             Track all visible
                                                  people simultaneously,
                                                  subject to measured
                                                  system capacity.

  R3-02                   Track identity          Maintain persistent
                                                  temporary tracking IDs
                                                  across frames whenever
                                                  continuity can be
                                                  maintained.

  R3-03                   Partial occlusion       Continue tracking
                                                  through partial
                                                  occlusion whenever
                                                  possible; do not
                                                  discard a track solely
                                                  because part of a
                                                  person is hidden.

  R3-04                   Return after track      Assign a new tracking
                          declared lost           ID if the old track has
                                                  already passed its loss
                                                  timeout and been
                                                  retired.
  -----------------------------------------------------------------------

Tracking identity is temporary visual continuity, not real-world
identity. The system is not required to recognize a person across
separate visits.

### 3.2 Target selection and locking

  -----------------------------------------------------------------------
  ID                      Requirement             Decision
  ----------------------- ----------------------- -----------------------
  R3-05                   Selection policy        Select the candidate
                                                  who first satisfies all
                                                  Round 2 readiness
                                                  conditions.

  R3-06                   Target locking          Once selected, retain
                                                  that person as the
                                                  primary target until
                                                  the target lifecycle
                                                  permits release.

  R3-07                   Target switching        Do not switch merely
                                                  because another person
                                                  becomes closer or
                                                  appears more suitable.

  R3-08                   Background evaluation   Continue tracking and
                                                  evaluating other
                                                  visible people while
                                                  one target is locked.
  -----------------------------------------------------------------------

This preserves the Round 2 first-qualified policy and avoids target
oscillation. Other candidates can be evaluated and queued, but they do
not pre-empt the locked target.

### 3.3 Candidate queue

  -----------------------------------------------------------------------
  ID                      Requirement             Decision
  ----------------------- ----------------------- -----------------------
  R3-09                   Queue ordering          Order eligible
                                                  candidates by the time
                                                  they first satisfy all
                                                  readiness conditions.

  R3-10                   Queue capacity          Maximum of five queued
                                                  eligible candidates.

  R3-11                   Queue overflow          Retain the five
                                                  earliest-qualified
                                                  candidates. People
                                                  outside the queue
                                                  remain detected/tracked
                                                  but are not queued.

  R3-12                   Queue invalidation      Remove a queued
                                                  candidate when they are
                                                  confidently observed
                                                  leaving the interaction
                                                  zone.

  R3-13                   Next candidate          When the current target
                                                  is released, select the
                                                  next currently eligible
                                                  candidate without
                                                  repeating an already
                                                  completed 2-second
                                                  stability check,
                                                  provided their
                                                  eligibility remains
                                                  valid.

  R3-14                   Queue position after    Preserve the
                          temporary loss          candidate's queue
                                                  position if
                                                  reacquisition occurs
                                                  within the five-second
                                                  recovery period and
                                                  association with the
                                                  prior visit/track
                                                  context is sufficiently
                                                  reliable.
  -----------------------------------------------------------------------

When a queue slot becomes available, the system may consider tracked
people outside the queue. Their readiness must be revalidated before
insertion. Queue ordering uses original qualification time, not the time
at which a slot becomes available.

### 3.4 Simultaneous qualification tie-break

  -----------------------------------------------------------------------
  ID                      Requirement             Decision
  ----------------------- ----------------------- -----------------------
  R3-15                   Same-frame              If two or more
                          qualification           candidates qualify in
                                                  the same camera frame
                                                  and their qualification
                                                  times cannot be
                                                  ordered, prefer the
                                                  candidate with the
                                                  smaller estimated
                                                  distance to the robot.

  R3-16                   Distance uncertainty    Apply a calibrated
                                                  distance tolerance. If
                                                  the distance difference
                                                  is within the
                                                  tolerance, use a stable
                                                  deterministic
                                                  tracking-ID-based
                                                  fallback.
  -----------------------------------------------------------------------

The numerical distance tolerance is not yet specified. Since distance is
estimated from RGB, the tie-break must not overreact to insignificant
differences in noisy estimates.

### 3.5 Track loss, target release, and visit lifecycle

  -----------------------------------------------------------------------
  ID                      Requirement             Decision
  ----------------------- ----------------------- -----------------------
  R3-17                   Track-loss recovery     Retain a temporarily
                          period                  lost track for 5
                                                  seconds while
                                                  attempting recovery.

  R3-18                   Recovery within timeout If the person returns
                                                  within 5 seconds and
                                                  continuity can be
                                                  established, recover
                                                  the original tracking
                                                  ID where possible.

  R3-19                   Expiry of track-loss    If the person does not
                          timeout                 return within 5
                                                  seconds, declare the
                                                  track lost and retire
                                                  its tracking ID. A
                                                  later detection
                                                  receives a new tracking
                                                  ID.

  R3-20                   Target release before   If the selected person
                          handoff                 is temporarily lost,
                                                  wait up to 5 seconds.
                                                  If not reacquired,
                                                  release the target and
                                                  allow another currently
                                                  eligible candidate to
                                                  be selected, subject to
                                                  lifecycle state.

  R3-21                   Visit identity          Maintain visit-level
                                                  state separately from
                                                  temporary tracking IDs
                                                  and selected-target
                                                  identity.

  R3-22                   New visit               After the previous
                                                  visit is completed, a
                                                  later approach is
                                                  treated as a new visit
                                                  with a new visit ID.

  R3-23                   Completion authority    The downstream
                                                  interaction module
                                                  signals interaction
                                                  completion. Vision must
                                                  not infer
                                                  conversational
                                                  completion merely from
                                                  disappearance.
  -----------------------------------------------------------------------

#### Clarification: temporary occlusion versus declared track loss

The five-second timeout is a recovery grace period, not an instruction
to assign a new ID as soon as a person becomes invisible.

Example:

``` text
t = 0 s:  P01 visible and tracked
t = 1 s:  P01 becomes occluded / temporarily lost
t = 3 s:  Person returns; attempt to recover P01
          If continuity is reliable, resume P01
```

If the person does not return before the five-second recovery period
expires:

``` text
t = 0 s:  P01 becomes temporarily lost
t = 5 s:  P01 declared LOST and retired
t = 8 s:  Person detected again; assign new tracking ID P02
```

A new track may be associated with an existing visit only if short-term
spatial-temporal evidence is sufficiently reliable. This is not face
recognition or persistent appearance-based identity recognition. If
association is ambiguous, do not confidently merge the tracks.

#### Physical departure versus temporary invisibility

The five-second recovery period applies to temporary visibility loss,
such as brief occlusion or camera-view interruption. It does not
override the queue invalidation rule.

-   If the person is confidently observed leaving the interaction zone,
    remove them from the eligible queue.
-   If the person is temporarily hidden and their physical departure
    cannot be established, preserve their candidate state during the
    five-second recovery period.
-   If reacquisition is reliable within that period, preserve their
    queue position.
-   If reacquisition does not occur within the period, retire the track
    and remove/expire its candidate state as appropriate.

### 3.6 Shared ownership after first-contact handoff

The user selected **shared ownership** for target lifecycle after
`FIRST_CONTACT_READY`.

Responsibilities:

**Vision module owns perception facts and actions** - Detects and tracks
people. - Maintains temporary tracking IDs. - Maintains target
visibility and tracking confidence. - Attempts target reacquisition. -
Reports target visibility loss, reacquisition, and confirmed zone
departure. - Does not declare a conversation complete based solely on
visual disappearance.

**Downstream interaction module owns interaction state and decisions** -
Manages the interaction/conversation. - Decides whether an interaction
should continue, pause, or be cancelled after receiving perception
updates. - Signals interaction completion or cancellation to vision.

**Shared lifecycle contract** - Vision reports perception events. - The
interaction module decides the interaction-level response. - Target
release is coordinated through explicit lifecycle messages. - The
five-second perception recovery timeout does not, by itself, prove that
the interaction has ended. - Vision must not start a new first-contact
interaction with another candidate while the previous target's
interaction remains active, unless the interaction module has explicitly
cancelled/completed or otherwise released it.

Conceptual event names (not a finalized API):

  ---------------------------------------------------------------------------
  Direction               Event                       Meaning
  ----------------------- --------------------------- -----------------------
  Vision → Interaction    `FIRST_CONTACT_READY`       A candidate has
                                                      satisfied first-contact
                                                      readiness.

  Vision → Interaction    `TARGET_TEMPORARILY_LOST`   Selected target is
                                                      temporarily
                                                      unavailable; recovery
                                                      is in progress.

  Vision → Interaction    `TARGET_REACQUIRED`         Selected target has
                                                      been reacquired.

  Vision → Interaction    `TARGET_LOST`               Five-second perception
                                                      recovery period
                                                      expired.

  Vision → Interaction    `TARGET_LEFT_ZONE`          Selected target was
                                                      confidently observed
                                                      leaving the interaction
                                                      zone.

  Interaction → Vision    `INTERACTION_STARTED`       Interaction has
                                                      started.

  Interaction → Vision    `INTERACTION_PAUSED`        Interaction is paused.

  Interaction → Vision    `INTERACTION_CANCELLED`     Interaction is
                                                      cancelled.

  Interaction → Vision    `INTERACTION_COMPLETED`     Interaction is
                                                      completed.
  ---------------------------------------------------------------------------

These names are illustrative. Event payloads, acknowledgements, retries,
deduplication, ordering, and failure handling remain to be designed.

------------------------------------------------------------------------

## 4. Conceptual state model

Three identifiers must remain distinct:

  -----------------------------------------------------------------------
  Identifier              Meaning                 Lifetime
  ----------------------- ----------------------- -----------------------
  `track_id`              Temporary visual        From track creation
                          tracking identity       until declared
                                                  lost/retired

  `target_id`             Identity of the         Until target lifecycle
                          currently selected      is released
                          target within the       
                          vision/interaction      
                          handoff                 

  `visit_id`              Operational identity    Across temporary
                          for one reception visit tracking interruptions;
                                                  ends when visit is
                                                  completed/released
  -----------------------------------------------------------------------

A visit can be associated with more than one temporary track ID if a
track is lost and a new track is reliably associated with the same
ongoing visit. This association must not be confused with face
recognition.

### Conceptual example

``` text
visit_id = V01
    ├── track_id = P01  (initial track)
    └── track_id = P02  (new track after P01 was retired,
                         only associated with V01 if reliable)

target_id = T01        (selected target during the handoff lifecycle)
```

The exact data model and association mechanism are not yet selected.

------------------------------------------------------------------------

## 5. Conceptual processing flow

``` text
Camera frames
    |
    v
Detect all visible people
    |
    v
Multi-person tracking
    |
    +--> Maintain temporary track IDs
    +--> Handle partial occlusion
    +--> Apply 5-second recovery grace period
    |
    v
Evaluate each person's Round 2 readiness
    |
    v
Maintain eligible candidate queue (capacity 5)
    |
    v
If no target is locked:
    select earliest-qualified valid candidate
    |
    v
Lock selected target
    |
    v
Emit FIRST_CONTACT_READY
    |
    v
Shared lifecycle with interaction module
    |
    +--> Vision reports visibility / loss / reacquisition
    +--> Interaction decides continue / pause / cancel / complete
    |
    v
Explicit target release
    |
    v
Select next currently eligible candidate, if permitted
```

This is a requirements-level flow, not a final algorithm or
implementation design.

------------------------------------------------------------------------

## 6. Evaluation and validation plan

The following metrics are recommended for later system evaluation.
Numerical acceptance thresholds have not been selected.

### Detection and tracking

-   Person detection precision, recall, and mAP.
-   Tracking HOTA and IDF1.
-   ID switches and track fragmentation.
-   Track recovery rate after temporary occlusion.
-   Track-loss false alarm rate.
-   Tracking latency and frame rate under multiple-person load.

### Candidate and target management

-   Correct first-qualified selection rate.
-   Target-switch count (should be zero except through explicit
    lifecycle release).
-   Candidate queue ordering accuracy.
-   Queue overflow behaviour and capacity adherence.
-   Candidate invalidation correctness.
-   Same-frame tie-break determinism.

### Lifecycle and interaction integration

-   Correct `FIRST_CONTACT_READY` delivery.
-   Correct target-loss and reacquisition reporting.
-   Duplicate, delayed, and out-of-order event handling.
-   Time from target loss to notification.
-   Time from interaction completion/cancellation to target release.
-   False release rate and stale-target rate.
-   Correct prevention of a second first-contact trigger within the same
    visit.

### Robustness scenarios

-   Partial and full occlusion.
-   Crowded reception with more than five eligible people.
-   Several people qualifying in the same frame.
-   Candidate leaving the zone while queued.
-   Target temporarily disappearing and returning within five seconds.
-   Target returning after the five-second timeout.
-   Interaction still active when the target becomes visually
    unavailable.
-   Interaction module delayed, unavailable, or not responding.

------------------------------------------------------------------------

## 7. Unresolved technical and integration parameters

Requirements gathering is complete, but the following engineering
details remain open. They must not be treated as user decisions already
made.

### Perception and tracking

1.  Camera model, field of view, intrinsics, mounting height, pitch, and
    whether the head/camera pose is truly fixed.
2.  Maximum expected number of visible people and measured throughput.
3.  Detector operating point (precision/recall preference) and
    confidence thresholds.
4.  Tracker architecture and association method.
5.  Exact partial-occlusion handling and track-management thresholds.
6.  Definition of when a track becomes temporarily lost versus confirmed
    lost.
7.  Whether the five-second timeout is measured from last reliable
    observation and how brief intermittent detections affect the timer.
8.  Short-term spatial-temporal association method for a new track to an
    existing visit, including confidence threshold and
    ambiguous-association behaviour.
9.  Maximum number of tracks retained during recovery and
    memory/resource limits.

### Candidate and queue semantics

10. Exact definition and calibration of the interaction-zone boundary
    and departure confidence.
11. Whether queue candidates outside the top five are continuously
    evaluated and how they enter when a slot opens.
12. Exact eligibility revalidation policy at dequeue time.
13. Distance-estimation tolerance and uncertainty model for same-frame
    tie-breaking.
14. Deterministic tracking-ID fallback ordering and behaviour if IDs are
    not comparable across tracker resets.
15. Exact semantics when an eligible candidate becomes uncertain while
    queued.

### Lifecycle and module integration

16. Whether the five-second recovery period is identical before and
    after `FIRST_CONTACT_READY`, or whether the interaction module can
    request a different recovery policy.
17. Exact target-release handshake and the event that authorizes vision
    to release the target after handoff.
18. What happens if the interaction module does not respond to
    `TARGET_LOST` or `TARGET_LEFT_ZONE`.
19. Whether a new candidate can be selected while the previous
    interaction is paused.
20. Event schema, identifiers, timestamps, sequence numbers,
    acknowledgements, retries, deduplication, and ordering.
21. Visit completion and new-visit creation semantics, including how a
    later approach is distinguished from the still-active visit without
    persistent person recognition.
22. How to prevent duplicate `FIRST_CONTACT_READY` events for the same
    visit.
23. Behaviour after process restart, camera interruption, or module
    restart while a target/visit is active.

### System-level validation

24. Numerical acceptance thresholds for detection, tracking, target
    selection, false first-contact triggers, latency, and queue
    correctness.
25. Test dataset composition, annotation protocol, scenario coverage,
    and reproducibility requirements.

------------------------------------------------------------------------

## 8. Requirements status

**Round 3 user decisions:** complete through Batch 5 and the subsequent
clarification of the five-second recovery semantics and shared lifecycle
ownership.

**Round 3 specification status:** consolidated requirements draft; not
yet frozen as an implementation specification.

**Next engineering stage:** review the open parameters, define the
interface contract and state transitions, then select and benchmark
perception/tracking methods. No detector or tracker has been selected by
these requirements.

------------------------------------------------------------------------

## 9. Decision summary

The resulting policy can be summarized as:

> Track all visible people and maintain temporary identities across
> frames. Select and lock the first person who satisfies the
> first-contact readiness conditions. Continue evaluating other visitors
> and maintain a queue of up to five eligible candidates ordered by
> qualification time. Preserve track and queue state during temporary
> visibility loss for up to five seconds, recover the original track ID
> when possible, and retire it after timeout. Keep track, target, and
> visit identities separate. After first-contact handoff, vision reports
> perception state while the interaction module owns interaction
> decisions; target release is coordinated explicitly. When a target is
> released, select the next currently eligible queued candidate, subject
> to current validity and interaction lifecycle state.
