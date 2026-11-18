# Round 2 --- Detection, Proximity, and First-Contact Readiness

**Project:** Vision-Based First Contact for a Tabletop Social Robot\
**Document:** `round_2.md`\
**Status:** Requirements gathered; technical validation pending\
**Scope:** Vision-based determination of when a detected person is
eligible for first contact. Multi-person target management is reserved
for Round 3.

------------------------------------------------------------------------

## 1. Purpose

This document formalizes the Round 2 requirements for the reception
robot's vision-based first-contact system.

The system must distinguish between:

1.  A person who is merely visible to the camera.
2.  A person who is approaching the reception area.
3.  A person who has entered the intended interaction area.
4.  A person who satisfies the conditions for first contact.
5.  A situation in which the perception system is too uncertain to
    initiate contact safely.

The objective is to trigger a single `FIRST_CONTACT_READY` event only
when the system has sufficient visual evidence that a visitor has
approached the reception point and is ready to interact.

This project covers the visual perception and readiness decision only.
Dialogue, speech generation, conversational turn-taking, and nonverbal
response generation belong to separate downstream projects.

## 2. Scope and dependencies

### 2.1 In scope

-   Person visibility and approach assessment.
-   Image-space interaction-zone membership.
-   Approximate physical-distance estimation using RGB-camera geometry.
-   Movement and stationary-state assessment.
-   Person orientation assessment.
-   Temporal stability validation.
-   False-trigger prevention.
-   Uncertainty and perception-failure handling.
-   Emission of a structured first-contact event.

### 2.2 Out of scope

-   Speech recognition and speech generation.
-   Dialogue management and conversational policy.
-   Gesture or facial-expression generation.
-   Full multi-person target-selection and handover policy.
-   Persistent real-world identity recognition.

Multi-person tracking is required as an upstream capability for this
system, but detailed tracking and target-selection requirements are
deferred to Round 3.

## 3. Hardware and operating environment

The following configuration was supplied during Round 1.

  -----------------------------------------------------------------------
  Parameter                           Current configuration
  ----------------------------------- -----------------------------------
  Robot configuration                 Stationary tabletop social robot

  Camera mounting                     Head-mounted, facing forward

  Camera model                        Not yet identified

  Image resolution                    720p (1280 × 720)

  Frame rate                          30 FPS

  Camera modality                     RGB only

  Operating environment               Office reception

  Lighting conditions                 Normal artificial lighting,
                                      daylight, and low lighting

  Scene conditions                    Crowded environment; people may
                                      carry bags or objects

  Maximum detection distance          Not yet specified
  -----------------------------------------------------------------------

### 3.1 Implications

1.  The camera does not provide direct metric depth.
2.  Bounding-box size alone must not be treated as a reliable
    physical-distance measurement.
3.  The camera's mounting height, pitch, field of view, and intrinsic
    calibration are not yet known.
4.  Lighting variation, crowding, carried objects, and partial occlusion
    must be included in evaluation.
5.  A fixed-camera geometric calibration may be useful if the camera
    remains rigidly mounted relative to the reception area.

## 4. Agreed behavioural requirements

### 4.1 Person visibility and detection

-   Partially visible people should be detected whenever the detector
    can provide sufficiently reliable evidence.
-   Full-body visibility is not mandatory.
-   The system should distinguish detection confidence from the later
    decision that a person is eligible for interaction.

### 4.2 Approach recognition

Approach recognition should combine spatial and temporal evidence:

-   Spatial evidence: the person's position relative to the configured
    reception interaction area.
-   Temporal evidence: the person's movement trajectory over multiple
    frames.

A person merely crossing the camera's field of view should not be
treated as approaching the reception point solely because they are
visible or nearby.

### 4.3 Interaction-area definition

The interaction area should use a hybrid representation:

1.  A manually calibrated image-space region, initially represented as a
    polygon.
2.  An approximate physical-distance constraint.

The initial physical-distance target is **1.5 metres** from the
robot/reception interaction reference point.

The 1.5 m value is a behavioural requirement to validate experimentally.
It is not yet a demonstrated measurement capability of the RGB camera.

### 4.4 Movement and stopping behaviour

-   A person must stop before first contact is triggered.
-   The system must estimate movement over time rather than infer a stop
    from a single frame.
-   The person must remain within the interaction area during the
    stability-check period.

### 4.5 Orientation

-   The person must be facing the robot for first contact to be
    eligible.
-   Orientation is a required condition in the current specification,
    not merely a supporting cue.
-   The exact visual definition of "facing the robot" remains to be
    selected and validated.

### 4.6 Stability duration

The person must satisfy the required readiness conditions for **2
continuous seconds** before the first-contact event is emitted.

The conditions include:

-   Valid approach history.
-   Membership in the interaction area.
-   Distance within the 1.5 m target.
-   Stationary state.
-   Required orientation.
-   Sufficient perception and tracking confidence.

If a required condition becomes invalid, the pending stability interval
is reset unless the event is specifically classified as temporary
tracking loss, in which case evaluation is paused as described in
Section 6.

## 5. False-trigger prevention

### 5.1 Passersby

-   People passing through the scene may be detected and tracked.
-   Detection or proximity alone must not trigger first contact.
-   The full first-contact eligibility conditions must be satisfied.

### 5.2 People already waiting nearby

A person who is already standing near the reception area must not
automatically become eligible merely because they satisfy proximity and
orientation conditions.

The system must first have observed an approach towards the reception
area.

This requirement helps distinguish a potential visitor from someone who
is waiting nearby or talking to another person.

### 5.3 Leaving before readiness

If a candidate leaves the interaction area before completing the
two-second stability interval:

-   Cancel the pending first-contact evaluation.
-   Reset the stability timer.
-   Return the candidate to the appropriate observation state.

### 5.4 Repeated contact

The robot should initiate first contact **once per visit**.

A 30-second cooldown was discussed but is not active in the current
policy. A cooldown and a once-per-visit policy are different: a cooldown
permits another trigger after a fixed interval, whereas once-per-visit
prevents another trigger until the visit is considered complete.

The once-per-visit policy is the current requirement.

Implementation note: visit-level interaction state must be managed
separately from a temporary tracker ID. A tracking ID is not a
persistent real-world identity and may change after occlusion or track
loss.

## 6. Uncertainty and failure handling

  -----------------------------------------------------------------------
  Situation                           Required behaviour
  ----------------------------------- -----------------------------------
  Distance estimate is uncertain      Hold evaluation until the estimate
                                      becomes sufficiently stable

  Orientation estimate is uncertain   Hold evaluation until orientation
                                      is sufficiently clear

  Selected person is temporarily lost Pause readiness evaluation and
                                      attempt reacquisition

  Two people qualify simultaneously   Prefer the person who qualified
                                      first

  Perception cannot establish         Do not emit a first-contact event;
  readiness reliably                  continue observing
  -----------------------------------------------------------------------

### 6.1 Hold, pause, and reset semantics

These behaviours must remain distinct:

-   **HOLD:** A required estimate is uncertain. Do not advance the
    readiness decision until reliable evidence is available.
-   **PAUSE:** The selected person is temporarily unavailable due to
    tracking loss. Preserve the pending evaluation state while
    attempting reacquisition.
-   **RESET:** A behavioural requirement has been violated, such as the
    person leaving the interaction area before readiness. Clear the
    pending stability interval.

The maximum duration for which a candidate may remain paused during
tracking loss is not yet specified and must be determined
experimentally.

### 6.2 Simultaneous candidates

If multiple candidates satisfy all first-contact conditions, select the
person with the earliest valid qualification time.

Formally, for the set of eligible candidates (`\mathcal{E}`{=tex}):

\[ i\^\* = `\arg`{=tex}`\min`{=tex}*{i `\in `{=tex}`\mathcal{E}`{=tex}}
t*{`\mathrm{qualified}`{=tex},i} \]

A deterministic tie-breaking rule is still required for candidates that
qualify within the same processing timestamp or frame.

## 7. Formal first-contact readiness model

For each tracked person (i), maintain a per-frame observation:

\[ X_i(t) = `\left[
ID_i,\ B_i(t),\ Z_i(t),\ D_i(t),\
V_i(t),\ O_i(t),\ Q_i(t)
\right]`{=tex}`\right`{=tex}\] \]

where:

-   (ID_i): temporary tracking identifier.
-   (B_i(t)): person bounding box or associated visual localization.
-   (Z_i(t)): interaction-zone membership.
-   (D_i(t)): estimated distance to the robot.
-   (V_i(t)): estimated movement state.
-   (O_i(t)): estimated orientation relative to the robot.
-   (Q_i(t)): relevant detection, tracking, and estimation confidence.

Define the following predicates:

\[
```{=tex}
\begin{aligned}
A_i(t) &= \text{approach towards reception has been observed},\\
Z_i(t) &= \text{person is inside the interaction area},\\
D_i(t) &= \text{estimated distance is within 1.5 metres},\\
S_i(t) &= \text{person is stationary},\\
O_i(t) &= \text{person is facing the robot},\\
Q_i(t) &= \text{perception evidence is sufficiently reliable}.
\end{aligned}
```
\]

The instantaneous eligibility condition is:

\[ E_i(t) = A_i(t) `\land `{=tex}Z_i(t) `\land `{=tex}D_i(t)
`\land `{=tex}S_i(t) `\land `{=tex}O_i(t) `\land `{=tex}Q_i(t) \]

A candidate becomes ready only after the eligibility condition has
remained valid for two continuous seconds:

\[ t - t\_{`\mathrm{eligible}`{=tex},i}
`\geq 2`{=tex} `\mathrm{s}`{=tex} \]

This is a requirements-level formulation. The numerical thresholds for
stationary motion, confidence, distance stability, and orientation are
not yet defined.

### 7.1 Important temporal detail

The two-second interval should not be interpreted as a requirement that
every raw measurement be identical. Measurements will naturally
fluctuate. The implementation must define filtering, hysteresis, and
confidence thresholds so that harmless frame-to-frame noise does not
cause repeated resets.

The current behavioural decision is that required conditions must remain
valid throughout the stability interval. The precise signal-processing
implementation remains open.

## 8. Proposed state machine

The following state machine expresses the agreed high-level behaviour.
It does not yet define the detailed multi-person tracking and
target-management logic reserved for Round 3.

``` mermaid
stateDiagram-v2
    [*] --> OBSERVING

    OBSERVING --> APPROACH_DETECTED: Approach observed
    APPROACH_DETECTED --> CANDIDATE_EVALUATION: Candidate enters interaction area

    CANDIDATE_EVALUATION --> STABILITY_CHECK: Readiness conditions satisfied
    CANDIDATE_EVALUATION --> OBSERVING: Candidate invalidated

    STABILITY_CHECK --> FIRST_CONTACT_READY: Conditions sustained for 2 seconds
    STABILITY_CHECK --> OBSERVING: Required condition violated / candidate leaves
    STABILITY_CHECK --> PAUSED: Temporary tracking loss

    PAUSED --> STABILITY_CHECK: Candidate reacquired
    PAUSED --> OBSERVING: Reacquisition timeout / candidate abandoned

    FIRST_CONTACT_READY --> CONTACT_LOCKED: Downstream module acknowledges event
    CONTACT_LOCKED --> OBSERVING: Visit completed and target released
```

`CONTACT_LOCKED` and visit completion require coordination with the
downstream interaction module. The vision system should not assume that
a conversation has started or ended merely because it emitted
`FIRST_CONTACT_READY`.

## 9. Proposed software architecture

The initial software architecture should separate perception, temporal
reasoning, and event generation.

  -----------------------------------------------------------------------
  Module                              Responsibility
  ----------------------------------- -----------------------------------
  Camera interface                    Acquire RGB frames and timestamps

  Person detector                     Produce person detections and
                                      confidence scores

  Person tracker                      Associate detections across frames
                                      and maintain temporary IDs

  Spatial estimator                   Estimate image-space position, zone
                                      membership, and approximate
                                      distance

  Motion estimator                    Estimate trajectory, approach
                                      behaviour, and stationary state

  Orientation estimator               Estimate whether a person is facing
                                      the robot

  Confidence evaluator                Assess reliability of the relevant
                                      estimates

  First-contact decision engine       Apply eligibility, temporal
                                      stability, and failure-handling
                                      rules

  Event publisher                     Emit a structured event to the
                                      downstream interaction system
  -----------------------------------------------------------------------

### 9.1 Conceptual data flow

``` text
RGB Camera
    |
    v
Person Detection
    |
    v
Person Tracking
    |
    +-----------------------------+
    |                             |
    v                             v
Spatial Analysis             Temporal Analysis
- Interaction zone          - Approach history
- Distance estimate         - Movement state
- Relative position         - Stability duration
    |                             |
    +---------------+-------------+
                    |
                    v
             Orientation Analysis
                    |
                    v
             Confidence Evaluation
                    |
                    v
          First-Contact Decision Engine
                    |
                    v
             FIRST_CONTACT_READY
                    |
                    v
          Downstream Interaction Module
```

## 10. Event interface

The vision system should communicate readiness through a structured
event rather than directly invoking speech or conversation behaviour.

Illustrative payload:

``` json
{
  "event": "FIRST_CONTACT_READY",
  "person_id": 3,
  "timestamp": 12345.67,
  "zone": "INTERACTION",
  "estimated_distance_m": 1.42,
  "stationary": true,
  "facing_robot": true,
  "tracking_confidence": 0.94,
  "readiness_duration_s": 2.0
}
```

The values above are examples, not measured results or finalized
thresholds.

The production interface should additionally define:

-   Event schema and version.
-   Timestamp clock and synchronization convention.
-   Event identifier and deduplication behaviour.
-   Whether the distance estimate is valid and its uncertainty.
-   How the downstream module acknowledges, rejects, or releases a
    target.
-   How visit completion is communicated back to the vision system.

## 11. Unresolved technical questions

The behavioural requirements have been gathered, but the following
engineering questions must be resolved before implementation is
considered complete.

  -----------------------------------------------------------------------
  Area                    Unresolved question     Initial investigation
  ----------------------- ----------------------- -----------------------
  Camera geometry         What are the camera     Inspect hardware and
                          intrinsics, mounting    calibrate camera
                          height, pitch, and      
                          field of view?          

  Distance estimation     Can the system reliably Fixed-camera geometry
                          estimate the 1.5 m      and floor-plane
                          threshold?              calibration

  Ground-point            How should distance be  Compare visible
  localization            estimated when feet are footpoint, lower-body,
                          occluded?               and alternative
                                                  localization methods

  Interaction polygon     How should the          Manual calibration
                          reception polygon be    procedure
                          calibrated and          
                          maintained?             

  Orientation             What constitutes        Evaluate body
                          "facing the robot"?     orientation and
                                                  head-pose alternatives

  Stationary state        What displacement or    Measure person motion
                          velocity threshold      in representative
                          defines a stop?         videos

  Distance stability      What uncertainty or     Calibrate against
                          fluctuation is          ground-truth distances
                          acceptable around 1.5   
                          m?                      

  Confidence              What thresholds are     Validation-set
                          required for detection, calibration
                          tracking, distance, and 
                          orientation?            

  Temporal filtering      How should noisy        Compare smoothing and
                          measurements be         hysteresis strategies
                          filtered without        
                          excessive latency?      

  Tracking loss           What is the maximum     Occlusion experiments
                          pause/reacquisition     
                          duration?               

  Visit management        How is once-per-visit   Define visit lifecycle
                          state maintained when   and downstream release
                          tracker IDs change?     protocol

  Simultaneous            How are same-frame      Deterministic tie-break
  qualification           qualification ties      rule
                          resolved?               

  Event lifecycle         How are event           Define interface
                          acknowledgement and     contract with
                          target release          interaction module
                          represented?            
  -----------------------------------------------------------------------

## 12. Evaluation plan

Evaluation should cover the perception components and the end-to-end
first-contact decision.

### 12.1 Person detection

-   Precision.
-   Recall.
-   Mean Average Precision (mAP).
-   Performance under partial occlusion and different lighting
    conditions.

### 12.2 Tracking

-   HOTA.
-   IDF1.
-   ID switches.
-   Track fragmentation.
-   Recovery after temporary occlusion.

### 12.3 Proximity and approach estimation

-   Distance-estimation error against measured ground truth.
-   Interaction-zone classification accuracy.
-   Approach-versus-passerby classification performance.
-   Stationary-state classification accuracy.
-   Orientation classification accuracy.

### 12.4 First-contact decision

-   First-contact trigger precision.
-   Genuine-visitor missed-contact rate.
-   False triggers per operating hour.
-   Trigger latency.
-   Percentage of candidates cancelled due to uncertainty.
-   Performance when multiple people are present.

Two particularly important metrics are:

\[ `\text{False Trigger Rate}`{=tex} =
`\frac{\text{Unwanted first-contact triggers}}`{=tex}
{`\text{Operating hours}`{=tex}} \]

and

\[ `\text{Trigger Latency}`{=tex} = t\_{`\mathrm{trigger}`{=tex}} -
t\_{`\mathrm{visitor\ ready}`{=tex}} \]

The definition of ground-truth visitor readiness must be established in
the annotation protocol before latency is reported.

## 13. Current decisions versus implementation assumptions

The following distinction must be maintained during implementation.

### Behavioural decisions supplied and agreed

-   Partial person visibility is acceptable.
-   Approach recognition combines position and movement.
-   The interaction area uses a hybrid image-space and distance
    representation.
-   The initial near-distance target is 1.5 m.
-   The person must stop.
-   The person must face the robot.
-   Readiness conditions must be sustained for 2 seconds.
-   Passersby should not trigger contact.
-   A waiting person must have an observed approach.
-   Leaving during the observation interval resets the candidate.
-   First contact occurs once per visit.
-   Uncertain distance or orientation causes evaluation to be held.
-   Temporary tracking loss pauses evaluation and initiates
    reacquisition.
-   The first person to qualify is preferred.
-   Unreliable perception must not trigger contact.

### Not yet finalized

-   Detection and tracking algorithms.
-   Camera model and calibration parameters.
-   Exact distance-estimation method and accuracy.
-   Stationary velocity threshold.
-   Orientation-estimation method and threshold.
-   Confidence thresholds.
-   Distance uncertainty tolerance.
-   Tracking-loss timeout.
-   Same-frame tie-breaking.
-   Visit-completion and event-acknowledgement protocol.

## 14. Project status and next steps

  -----------------------------------------------------------------------
  Requirements round                  Status
  ----------------------------------- -----------------------------------
  Round 1 --- Hardware and            Completed
  environment                         

  Round 2 --- Detection, proximity,   Requirements gathered; technical
  and contact readiness               validation pending

  Round 3 --- Multi-person tracking   On hold
  and target selection                
  -----------------------------------------------------------------------

Recommended next steps:

1.  Complete Round 3 requirements gathering.
2.  Consolidate all three rounds into the overall system requirements
    specification.
3.  Resolve the camera and geometric-calibration questions.
4.  Define measurable thresholds and annotation rules.
5.  Design the detailed architecture and event interface.
6.  Implement and evaluate the system incrementally.

------------------------------------------------------------------------

**Scope boundary:** Round 2 defines how the vision system determines
whether a person is ready for first contact. It does not finalize how
the robot manages multiple simultaneous people, preserves target
identity, switches targets, or handles handover. Those requirements
belong to Round 3.
