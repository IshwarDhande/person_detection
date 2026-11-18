# Round 2 --- Questions, Options, Selected Answers, and Rationale

**Project:** Vision-Based First Contact for a Tabletop Social Robot\
**Document:** `r2_ques.md`\
**Status:** Round 2 requirements gathered; technical validation pending\
**Purpose:** Preserve the questions asked during Round 2, all offered
options, the user's selected answers, and the reasoning associated with
each decision.

> **Interpretation note:** The "Reasoning" sections explain the
> engineering implications of the selected option. Unless explicitly
> stated in the conversation, they are design interpretations rather
> than verbatim explanations supplied by the user.

------------------------------------------------------------------------

## 1. Summary of selected decisions

  -----------------------------------------------------------------------
  ID                      Topic                   Selected answer
  ----------------------- ----------------------- -----------------------
  Q1                      Person visibility       Partial visibility
                                                  accepted

  Q2                      Approach recognition    Combine position and
                                                  movement

  Q3                      Interaction-area        Hybrid: image-space
                          representation          polygon and estimated
                                                  distance

  Q4                      Near-distance target    1.5 metres

  Q5                      Stopping requirement    Person must stop

  Q6                      Orientation             Person must face the
                                                  robot

  Q7                      Stability duration      2 seconds

  Q8                      Passersby               Detect/track, but do
                                                  not initiate contact

  Q9                      Person already waiting  Require observed
                          nearby                  approach

  Q10                     Leaving during          Cancel and reset
                          observation             

  Q11                     Repeated first contact  Once per visit

  Q12                     Cooldown                30 seconds selected as
                                                  a value, but inactive
                                                  under once-per-visit
                                                  policy

  Q13                     Uncertain distance      Hold evaluation

  Q14                     Uncertain orientation   Hold evaluation

  Q15                     Temporary tracking loss Pause and attempt
                                                  reacquisition

  Q16                     Multiple simultaneous   Select the first person
                          candidates              to qualify

  Q17                     Unreliable perception   Do not trigger;
                                                  continue observing
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 2. Detailed question and answer record

## Q1. What should count as a detected person?

**Context:** A person may enter the camera frame while part of their
body is hidden behind a reception table, another person, or an object.

**Options offered:**

-   A. Yes --- detect partially visible people whenever possible.
-   B. Only when most of the body is visible.
-   C. Upper-body visibility is sufficient.

**Selected answer:** **A --- Detect partially visible people whenever
possible.**

**Reasoning and implications:**

-   Requiring a full or mostly visible body could cause the system to
    miss visitors in realistic reception scenes.
-   Partial visibility is common in crowded environments and when people
    carry bags or objects.
-   Detection should still be supported by sufficient detector
    confidence; partial visibility does not mean accepting arbitrary
    visual evidence.
-   The system must distinguish successful person detection from later
    interaction eligibility.

**Implementation implication:** Use a person detector that can handle
partial occlusion and evaluate its precision and recall under
representative occlusion conditions.

------------------------------------------------------------------------

## Q2. How should the robot recognize that someone is approaching?

**Context:** The system must distinguish a person walking across the
reception area from a person moving towards the reception desk.

**Options offered:**

-   A. Position only --- entering a predefined region.
-   B. Movement only --- detecting movement towards the robot.
-   C. Both position and movement.

**Selected answer:** **C --- Combine position and movement.**

**Reasoning and implications:**

-   Position alone may classify a passerby as an approaching visitor if
    they cross the configured region.
-   Movement alone may detect motion towards the robot without
    establishing that the person is in the reception interaction area.
-   Combining spatial and temporal evidence provides a stronger basis
    for identifying an approach.

**Implementation implication:** Evaluate a person's position relative to
the reception region together with their trajectory over multiple
frames.

------------------------------------------------------------------------

## Q3. How should the interaction area initially be defined?

**Context:** The camera is RGB-only and does not directly provide metric
depth.

**Options offered:**

-   A. Manually define an image-space polygon representing the reception
    area.
-   B. Use estimated physical distance in metres.
-   C. Use both the polygon and estimated distance.
-   D. Use a learned or adaptive interaction zone.

**Selected answer:** **C --- Hybrid: image-space polygon and estimated
physical distance.**

**Reasoning and implications:**

-   An image-space polygon provides a practical way to represent the
    visible reception area.
-   Approximate physical distance adds a metric constraint to the
    decision.
-   The combination avoids relying exclusively on either image position
    or an uncertain monocular distance estimate.
-   A learned or adaptive zone is not required for the initial version.

**Implementation implication:** Begin with a manually calibrated polygon
and investigate fixed-camera geometry or another validated monocular
distance-estimation method. Do not treat bounding-box size alone as a
reliable distance measurement.

------------------------------------------------------------------------

## Q4. What should be the approximate near distance?

**Context:** A visitor should be sufficiently close to the reception
robot before first contact is considered.

**Options offered:**

-   A. Approximately 1 metre.
-   B. Approximately 1.5 metres.
-   C. Approximately 2 metres.
-   D. Determine the optimal distance through experiments.

**Selected answer:** **B --- Approximately 1.5 metres.**

**Reasoning and implications:**

-   The selected value gives the initial system a concrete proximity
    target.
-   It makes the first implementation and evaluation criteria more
    specific.
-   The value is a behavioural requirement, not a claim that the RGB
    camera can already measure 1.5 metres accurately.
-   The actual threshold must be validated against measured ground-truth
    distances and the physical reception layout.

**Implementation implication:** Use 1.5 m as the initial target
threshold, subject to camera calibration and experimental validation.

------------------------------------------------------------------------

## Q5. Must the person stop before the robot initiates contact?

**Context:** A person may walk towards the desk, slow down, or continue
moving through the area.

**Options offered:**

-   A. Yes, wait until the person completely stops.
-   B. No, slowing down near the desk is sufficient.
-   C. Initiate immediately upon entering the interaction area.
-   D. Accept both stationary and moving visitors.

**Selected answer:** **A --- The person must stop.**

**Reasoning and implications:**

-   A stationary visitor provides stronger evidence of an opportunity to
    interact than a person who is still walking.
-   This conservative condition helps prevent greetings while someone is
    merely passing through.
-   It may increase trigger latency and may miss visitors who expect the
    robot to greet them while they are still moving.
-   The stationary-state threshold must be defined using measured
    motion, not a single-frame observation.

**Implementation implication:** Estimate person movement over a temporal
window and define a measurable stationary threshold experimentally.

------------------------------------------------------------------------

## Q6. Should the person's orientation matter?

**Context:** A person may be near the desk but facing the robot, looking
elsewhere, or approaching sideways.

**Options offered:**

-   A. Facing the robot must be mandatory.
-   B. Facing the robot is supporting evidence, not mandatory.
-   C. Ignore orientation completely.
-   D. Consider body orientation only, not facial orientation.

**Selected answer:** **A --- Facing the robot is mandatory.**

**Reasoning and implications:**

-   The selected policy requires visual evidence that the visitor is
    oriented towards the robot before first contact.
-   This is intended to reduce unwanted greetings to people who are near
    the desk but occupied with something else.
-   A strict orientation requirement may reject genuine visitors who are
    looking at a notice, talking to another person, or temporarily
    turning away.
-   The exact meaning of "facing the robot" is not yet operationally
    defined.

**Implementation implication:** Evaluate body-orientation and head-pose
alternatives. The final orientation estimator and threshold remain
unresolved.

------------------------------------------------------------------------

## Q7. How long should the robot observe someone before initiating contact?

**Context:** A person may briefly enter the interaction area or appear
stationary for only a moment. A temporal observation period helps avoid
decisions based on transient visual evidence.

**Options offered:**

-   A. Immediately.
-   B. 1 second.
-   C. 2 seconds.
-   D. 3 seconds.
-   E. Adapt the duration according to movement and confidence.

**Selected answer:** **C --- 2 seconds.**

**Additional user requirement:** Stability --- the person should have
remained in the area.

**Reasoning and implications:**

-   A two-second interval provides a short temporal validation period.
-   It reduces the likelihood of triggering on a brief crossing or
    transient detection.
-   Requiring stability makes the decision depend on sustained evidence
    rather than a single frame.
-   The system must define how measurement noise is handled during this
    interval.

**Implementation implication:** Maintain a stability timer for each
candidate. The required conditions should remain valid for two
continuous seconds. Filtering and hysteresis may be needed to avoid
unnecessary resets from small frame-to-frame fluctuations.

------------------------------------------------------------------------

## Q8. What should happen when someone walks past the reception?

**Context:** A person may pass close to the reception desk without
intending to interact.

**Options offered:**

-   A. Detect and track, but never initiate contact unless all readiness
    conditions are met.
-   B. Ignore passersby completely.
-   C. Greet anyone entering the near zone.

**Selected answer:** **A --- Detect and track, but do not initiate
contact unless all readiness conditions are met.**

**Reasoning and implications:**

-   Detection and tracking are useful for understanding the scene even
    when a person is not an interaction candidate.
-   Proximity alone must not cause a greeting.
-   This preserves scene awareness while reducing false first-contact
    triggers.

**Implementation implication:** Keep person detection/tracking separate
from the first-contact decision engine.

------------------------------------------------------------------------

## Q9. What if someone is standing near the reception but not approaching?

**Context:** A person may be waiting for a colleague, talking to another
visitor, or standing near the desk for a reason unrelated to the robot.

**Options offered:**

-   A. Do not initiate unless approach behaviour was observed first.
-   B. Initiate if proximity, orientation, and stability conditions are
    satisfied.
-   C. Initiate after the person remains in the interaction zone for a
    longer period.

**Selected answer:** **A --- Require observed approach behaviour
first.**

**Reasoning and implications:**

-   A person already standing nearby should not automatically be treated
    as a new visitor.
-   Requiring an observed approach provides additional evidence that the
    person has moved towards the reception.
-   This may miss visitors who were already present before the system
    began observing, or whose approach was occluded.

**Implementation implication:** Maintain an approach-history flag or
equivalent temporal evidence for each candidate. Define how the system
handles people first detected while already inside the interaction area.

------------------------------------------------------------------------

## Q10. What should happen if a person leaves the interaction zone during the two-second observation period?

**Context:** A candidate may enter the interaction area but leave before
satisfying the stability requirement.

**Options offered:**

-   A. Cancel the pending contact and reset the observation timer.
-   B. Pause the timer and resume if the person returns quickly.
-   C. Continue the timer regardless of the person's position.

**Selected answer:** **A --- Cancel the pending contact and reset the
timer.**

**Reasoning and implications:**

-   Leaving the interaction area means the candidate no longer satisfies
    a required readiness condition.
-   Resetting prevents separate periods of presence from being
    incorrectly accumulated into one continuous stability interval.
-   Continuing the timer would allow a trigger even though the person is
    no longer in the intended interaction area.

**Implementation implication:** Clear the pending stability interval
when the person leaves the interaction area before first-contact
readiness is reached.

------------------------------------------------------------------------

## Q11. How many times should the robot initiate first contact with the same person?

**Context:** A person may remain in front of the robot after the
first-contact event. The robot should not repeatedly generate the same
greeting trigger.

**Options offered:**

-   A. Only once during the person's current visit.
-   B. Allow another trigger after a cooldown period.
-   C. Trigger again only after an explicit reset from the interaction
    module.

**Selected answer:** **A --- Only once during the person's current
visit.**

**Reasoning and implications:**

-   The robot should not repeatedly initiate first contact while the
    same visit is ongoing.
-   A visit-level policy is more meaningful than a simple short timer.
-   A temporary tracking-ID change must not automatically be interpreted
    as a new visit.

**Implementation implication:** Maintain visit-level interaction state
separately from the temporary multi-object-tracking ID. Define visit
completion and release through coordination with the downstream
interaction module.

------------------------------------------------------------------------

## Q12. If a cooldown is used, what duration would be preferred?

**Context:** This question was conditional on choosing the
cooldown-based policy in Q11.

**Options offered:**

-   A. 15 seconds.
-   B. 30 seconds.
-   C. 60 seconds.
-   D. Until the person leaves and returns.

**Selected answer:** **B --- 30 seconds.**

**Status:** **Recorded, but inactive under the selected once-per-visit
policy.**

**Reasoning and implications:**

-   Thirty seconds was selected as the cooldown value.
-   However, a cooldown and a once-per-visit rule are different
    policies.
-   A cooldown permits another trigger after the interval;
    once-per-visit prevents another trigger until the visit is
    considered complete.
-   The current active requirement is once per visit, so the 30-second
    value must not be used to re-enable repeated greetings.

**Implementation implication:** Keep the cooldown parameter disabled
unless the policy is explicitly revised.

------------------------------------------------------------------------

## Q13. What if the distance estimate is uncertain?

**Context:** The RGB-based distance estimate may fluctuate around the
1.5 m threshold.

**Options offered:**

-   A. Wait until the distance estimate becomes stable.
-   B. Use the calibrated image-space zone as the primary decision.
-   C. Use the best available estimate and proceed.

**Selected answer:** **A --- Hold evaluation until the estimate becomes
sufficiently stable.**

**Reasoning and implications:**

-   The system should not make a proximity decision from an unstable or
    unreliable distance estimate.
-   Holding evaluation is safer than triggering based on a potentially
    incorrect measurement.
-   This can increase latency or prevent a trigger when distance remains
    uncertain.

**Implementation implication:** Define a distance-validity criterion,
uncertainty tolerance, and stability window. The exact thresholds remain
unresolved.

------------------------------------------------------------------------

## Q14. What if the robot cannot confidently determine whether the person is facing it?

**Context:** Orientation estimation may be unreliable because of
occlusion, camera angle, head movement, or poor visibility.

**Options offered:**

-   A. Wait until orientation becomes sufficiently clear.
-   B. Ignore orientation uncertainty and use other conditions.
-   C. Cancel that person's candidacy immediately.

**Selected answer:** **A --- Hold evaluation until orientation becomes
sufficiently clear.**

**Reasoning and implications:**

-   Facing the robot is a mandatory readiness condition.
-   Uncertain orientation should not be treated as proof that the person
    is facing the robot.
-   Holding avoids both an unjustified trigger and an immediate
    rejection of a potentially valid visitor.

**Implementation implication:** Use an explicit orientation-validity
state. Do not silently replace an uncertain orientation estimate with a
positive result.

------------------------------------------------------------------------

## Q15. What if the selected person temporarily disappears from the camera?

**Context:** Another person or object may temporarily block the selected
visitor.

**Options offered:**

-   A. Pause readiness evaluation and attempt to reacquire the person.
-   B. Immediately cancel the candidate.
-   C. Continue evaluation using the last known position.

**Selected answer:** **A --- Pause evaluation and attempt
reacquisition.**

**Reasoning and implications:**

-   Temporary occlusion should not immediately destroy the candidate's
    pending state.
-   Reacquisition may allow the system to continue tracking the same
    person.
-   Continuing with only the last known position could produce an
    incorrect readiness decision.
-   A maximum pause duration is still required.

**Implementation implication:** Preserve the pending state during
temporary loss, attempt tracker reacquisition, and define an
experimentally validated timeout. The timeout has not yet been selected.

------------------------------------------------------------------------

## Q16. What if two people simultaneously satisfy all first-contact conditions?

**Context:** Multiple visitors may approach the reception at nearly the
same time.

**Options offered:**

-   A. Select the person who qualified first.
-   B. Select the person closest to the robot.
-   C. Defer selection until one person has clear priority.
-   D. Maintain a queue and select the first person.

**Selected answer:** **A --- Select the person who qualified first.**

**Reasoning and implications:**

-   The first-qualified policy provides a simple and deterministic
    ordering principle.
-   It avoids repeatedly changing the selected candidate based only on
    small distance fluctuations.
-   It is compatible with a first-come-first-served interpretation of
    reception interaction.
-   A tie-breaking rule is still needed if two people qualify in the
    same processing frame or timestamp.

**Implementation implication:** Record each candidate's qualification
timestamp and select the minimum. Define deterministic tie-breaking for
equal timestamps.

------------------------------------------------------------------------

## Q17. What should happen if the perception system cannot confidently establish contact readiness?

**Context:** Detection, distance, orientation, tracking, or other
estimates may be unreliable.

**Options offered:**

-   A. Do not initiate contact; continue observing.
-   B. Use a relaxed set of conditions as fallback.
-   C. Send an uncertainty notification to the downstream module.
-   D. Request human intervention.

**Selected answer:** **A --- Do not initiate contact; continue
observing.**

**Reasoning and implications:**

-   The system should not emit a first-contact event without sufficient
    evidence.
-   A relaxed fallback could undermine the conservative readiness
    policy.
-   Continuing observation allows the system to gather better evidence.
-   The selected answer defines the trigger behaviour; it does not
    prohibit internal logging or diagnostic reporting of uncertainty.

**Implementation implication:** The decision engine must fail closed
with respect to first-contact triggering: insufficient confidence means
no trigger.

------------------------------------------------------------------------

# 3. Consolidated decision logic

The selected answers imply the following initial readiness condition for
each candidate person (i).

Let:

-   (A_i(t)): approach towards reception has been observed.
-   (Z_i(t)): person is inside the calibrated interaction region.
-   (D_i(t)): estimated distance is at or below 1.5 m and sufficiently
    reliable.
-   (S_i(t)): person is stationary.
-   (O_i(t)): person is facing the robot and orientation is sufficiently
    reliable.
-   (Q_i(t)): the relevant perception and tracking evidence is
    sufficiently reliable.

The instantaneous eligibility condition is:

\[ E_i(t) = A_i(t) `\land `{=tex}Z_i(t) `\land `{=tex}D_i(t)
`\land `{=tex}S_i(t) `\land `{=tex}O_i(t) `\land `{=tex}Q_i(t) \]

A candidate qualifies only after the required conditions have remained
valid for two continuous seconds:

\[ t - t\_{`\mathrm{eligible}`{=tex},i}
`\geq 2`{=tex} `\mathrm{s}`{=tex} \]

If multiple candidates qualify, select the earliest qualification time:

\[ i\^\* = `\arg`{=tex}`\min`{=tex}*{i `\in `{=tex}`\mathcal{E}`{=tex}}
t*{`\mathrm{qualified}`{=tex},i} \]

The selected person should receive no more than one first-contact
trigger during the current visit.

**Important:** This is a requirements-level model. It does not yet
specify numerical thresholds for stationary motion, orientation,
confidence, distance uncertainty, or tracking-loss timeout.

## 4. Hold, pause, and reset semantics

The selected answers imply three different temporal behaviours:

  -----------------------------------------------------------------------
  Behaviour               Meaning                 Example
  ----------------------- ----------------------- -----------------------
  HOLD                    Wait because an         Distance or orientation
                          estimate is uncertain   is unreliable

  PAUSE                   Temporarily suspend     Person is occluded or
                          evaluation while        temporarily lost
                          preserving candidate    
                          state                   

  RESET                   Clear pending readiness Person leaves the
                          progress because a      interaction area
                          required behavioural    
                          condition was violated  
  -----------------------------------------------------------------------

These semantics should be implemented explicitly rather than represented
by one generic failure state.

## 5. Unanswered questions and open requirements

The initial broad Round 2 questionnaire also contained questions that
were not answered when the discussion shifted to the guided question
sequence. They remain open and must not be treated as decided.

  -----------------------------------------------------------------------
  Open item               Original question       Status
  ----------------------- ----------------------- -----------------------
  Detection operating     Should detection        Unanswered
  point                   prioritize high         
                          precision, balanced     
                          precision/recall, or    
                          high recall?            

  Simultaneous tracking   Track one person, a     Unanswered
  capacity                limited number, or all  
                          visible people?         

  Occlusion policy at     Ignore partially        Unanswered
  tracker level           occluded people, track  
                          whenever possible, or   
                          reacquire after loss?   

  Detection confidence    What detector           Not specified
  threshold               confidence is           
                          acceptable?             

  Stationary threshold    What speed/displacement Not specified
                          constitutes a stop?     

  Orientation threshold   What angular range      Not specified
                          counts as facing the    
                          robot?                  

  Distance uncertainty    What error or variance  Not specified
                          is acceptable around    
                          1.5 m?                  

  Tracking-loss timeout   How long may evaluation Not specified
                          remain paused?          

  Same-frame tie-break    How should equal        Not specified
                          qualification           
                          timestamps be resolved? 

  Visit completion        What event marks the    Not specified
                          end of a visit?         

  Camera calibration      What are camera         Not available
                          intrinsics, mounting    
                          height, pitch, and      
                          field of view?          

  Maximum detection       What is the required    Not specified
  distance                maximum                 
                          person-detection range? 
  -----------------------------------------------------------------------

## 6. Design trade-offs to validate experimentally

The selected requirements favour avoiding false greetings, but they may
increase missed contacts or delay interaction.

  -----------------------------------------------------------------------
  Selected constraint     Benefit                 Potential cost
  ----------------------- ----------------------- -----------------------
  Partial visibility      Better tolerance to     More ambiguous
  accepted                occlusion               detections

  Position + movement     Better passerby         Requires temporal
                          discrimination          tracking

  Hybrid interaction zone Combines scene geometry Calibration and
                          and approximate metric  distance uncertainty
                          proximity               

  1.5 m target            Concrete initial        May not fit every
                          proximity requirement   reception layout

  Must stop               Reduces greetings to    May delay or miss
                          people passing through  moving visitors

  Must face robot         Stronger evidence of    Orientation estimation
                          availability            may be unreliable

  2-second stability      Reduces transient       Adds latency
                          triggers                

  Require observed        Avoids greeting people  May miss people first
  approach                already waiting nearby  observed inside the
                                                  zone

  Once per visit          Prevents repeated       Requires visit
                          greetings               lifecycle management

  Hold on uncertainty     Avoids unsupported      May increase missed
                          decisions               triggers

  Pause on tracking loss  Tolerates temporary     Requires a timeout and
                          occlusion               robust reacquisition

  First-qualified         Simple and              Requires tie-breaking
  selection               deterministic           
  -----------------------------------------------------------------------

These trade-offs should be assessed using representative reception
videos and controlled experiments. In particular, evaluate false-trigger
rate, genuine-visitor missed-contact rate, and trigger latency together
rather than optimizing only one metric.

## 7. Round 2 completion status

  -----------------------------------------------------------------------
  Area                                Status
  ----------------------------------- -----------------------------------
  Person visibility                   Behaviour selected

  Approach recognition                Behaviour selected

  Interaction-zone representation     Behaviour selected

  Near-distance target                Initial value selected; validation
                                      pending

  Stop and orientation requirements   Behaviour selected; estimator
                                      thresholds pending

  Stability interval                  2 seconds selected

  Passerby handling                   Behaviour selected

  Waiting-person handling             Behaviour selected

  Cancellation and reset              Behaviour selected

  Repeat-contact policy               Once per visit selected

  Distance/orientation uncertainty    Hold selected

  Temporary tracking loss             Pause and reacquire selected

  Multiple simultaneous candidates    First-qualified selected

  Perception failure                  No trigger selected

  Technical thresholds and            Open
  calibration                         
  -----------------------------------------------------------------------

**Round 2 status:** Behavioural requirements gathered. Technical
parameters, calibration, and validation criteria remain open.

**Round 3 status:** On hold. Multi-person tracking, identity
persistence, target switching, and handover decisions have not been
finalized in this document.

------------------------------------------------------------------------

## 8. Scope boundary

Round 2 defines how the vision system determines whether an approaching
person is ready for first contact. It does not finalize how the robot
manages multiple simultaneous people, preserves a selected target across
occlusion, switches between candidates, or handles handover. Those
topics belong to Round 3.
