# Round 3 --- Questions, Options, Answers, and Rationale Record

**Project:** Vision-Based First Contact for a Reception Social Robot\
**Status:** Requirements gathering complete; technical parameters remain
open\
**Purpose:** Preserve the questions asked, available options, user's
selected answers, explanations, recommendations, and unresolved details.

------------------------------------------------------------------------

## How to read this record

-   **Selected answer** records the user's explicit choice.
-   **Recommendation** records the assistant's suggested default, not an
    additional user decision.
-   **Clarification** captures later messages that refined an earlier
    answer.
-   **Open technical detail** means the requirement is known but its
    numerical value or implementation is not yet decided.

------------------------------------------------------------------------

# Batch 1 --- Tracking scope, identity, and occlusion

## Q1. How many people should the robot track simultaneously?

**Options presented** - A. Track only one person --- selected
candidate. - B. Track up to 3--5 people. - C. Track all visible people
simultaneously. (Recommended)

**User answer:** C --- Track all visible people simultaneously.

**Interpretation:** The robot should detect and maintain tracks for all
visible people, not only the current first-contact candidate.

**Trade-offs discussed:** This gives a complete view of the reception
area and supports background evaluation and multiple-person selection.
It increases computational load and the possibility of tracking errors
in crowds. Maximum practical capacity must be measured.

## Q2. Should a person retain the same tracking ID as they move?

**Options presented** - A. Yes, maintain the same ID across frames
whenever possible. (Recommended) - B. Assign a new ID whenever the
person's position changes significantly. - C. Need more explanation.

**User answer:** A --- Persistent tracking identity.

**Interpretation:** Maintain a temporary ID across frames while visual
tracking continuity can be maintained.

**Rationale:** Per-person approach, proximity, stationary duration, and
qualification time require temporal association.

## Q3. What should happen when a tracked person is partially hidden by another person?

**Options presented** - A. Continue tracking whenever possible.
(Recommended) - B. Temporarily pause that person's track. - C. Discard
the track and detect again later.

**User answer:** A --- Continue tracking through partial occlusion.

**Interpretation:** Partial occlusion alone should not cause immediate
track deletion.

**Trade-off:** Longer or complex occlusions can make association
unreliable. Track-management thresholds remain open.

## Q4. Person A disappears behind Person B and becomes visible again. What should happen?

**Options presented** - A. Attempt to recover A's previous tracking ID.
(Recommended) - B. Assign A a new tracking ID. - C. Consider A lost and
do not attempt recovery.

**User answer:** B --- Assign a new ID after the previous track is lost.

**Clarification added later:** This does not mean assigning a new ID
immediately after disappearance. The five-second recovery grace period
applies first. If the person returns within five seconds and continuity
can be established, recover the original tracking ID. A new ID is
assigned only after the old track has been declared lost/retired and the
person is detected again later.

**Important distinction:** Track identity is not persistent person
identity. No face recognition is required.

------------------------------------------------------------------------

# Batch 2 --- Target selection and locking

## Q5. When should the robot select a person as its target?

**Options presented** - A. First qualified --- select the person who
first satisfies all Round 2 readiness conditions. (Recommended) - B.
Closest person. - C. Most central person in the interaction zone. - D.
Other policy.

**User answer:** A --- First qualified.

**Interpretation:** Select the candidate with the earliest qualification
timestamp.

**Rationale:** Consistent with Round 2, predictable, and avoids
continually prioritizing whoever is closest.

## Q6. Once a target is selected, should the robot keep tracking that person as its primary target?

**Options presented** - A. Yes, lock onto the selected target until the
contact process ends. (Recommended) - B. Continuously reconsider and
switch to whoever appears most suitable. - C. Need more explanation.

**User answer:** A --- Lock the selected target.

**Interpretation:** Retain the selected target as primary until the
target lifecycle permits release.

**Trade-off:** Prevents target oscillation, but requires explicit
release and failure handling so that a target is not held indefinitely.

## Q7. Person A is selected, but Person B subsequently approaches and stands closer. What should happen?

**Options presented** - A. Keep Person A as target; do not switch merely
because B is closer. (Recommended) - B. Switch to Person B if they
become a better candidate. - C. Re-evaluate all candidates and restart
target selection.

**User answer:** A --- No switching merely because another person is
closer.

**Interpretation:** Target lock takes precedence over later proximity
changes. Other people can still be tracked and evaluated in the
background.

------------------------------------------------------------------------

# Batch 3 --- Track loss, target release, and visit lifecycle

## Q8. How long should the system wait before declaring a selected person's track lost?

**Options presented** - A. 2 seconds. - B. 5 seconds --- balanced
initial choice. (Recommended) - C. 10 seconds. - D. Adaptive timeout.

**User answer:** B --- 5 seconds.

**Interpretation:** Retain the track context and attempt recovery for
five seconds after temporary loss.

**Open technical detail:** Exact timer semantics (last reliable
observation, intermittent detections, and reset behaviour) remain to be
specified.

## Q9. What should happen if the selected person leaves the interaction zone before first contact?

**Options presented** - A. Release the target and allow another eligible
person to be selected. (Recommended) - B. Keep waiting for the same
person to return. - C. Cancel the current attempt and wait for a
completely new visit.

**User answer:** A --- Release the target and allow another eligible
person to be selected.

**Additional clarification from user:** Wait five seconds after
temporary disappearance; if the person has not returned, only then
release the target and allow another eligible person to be selected.

**Interpretation:** The five-second grace period applies to temporary
visibility loss. A confidently observed physical departure from the
interaction zone is a separate event and can invalidate a queued
candidate immediately, as clarified in Batch 5.

## Q10. Who should signal that first-contact interaction has finished?

**Options presented** - A. Downstream interaction module sends a
completion signal to vision. (Recommended) - B. Vision decides based on
visual disappearance. - C. Vision releases after a fixed time. - D. Need
explanation.

**User answer:** A --- Downstream interaction module signals completion.

**Interpretation:** Vision must not infer conversational completion
merely because a person disappears or stops moving.

## Q11. Should visit identity be maintained separately from tracking identity?

**Options presented** - A. Yes, maintain separate visit-level state,
independent of temporary tracking IDs. (Recommended) - B. No, use the
tracking ID to represent the visit. - C. Need explanation.

**User answer:** A --- Separate visit identity.

**Interpretation:** Maintain distinct `track_id`, `target_id`, and
`visit_id` concepts.

**Rationale:** A temporary tracking ID can change without necessarily
ending the operational visit. This does not require biometric
recognition.

------------------------------------------------------------------------

# Batch 4 --- Candidate queue and returning visitors

## Q12. After a target is released, what should happen to the next eligible person?

**Options presented** - A. Select the next eligible person immediately.
(Recommended) - B. Start a fresh two-second stability observation for
every person. - C. Require a newly observed approach before selecting
anyone. - D. Need explanation.

**User answer:** A --- Select the next eligible person immediately.

**Interpretation:** An already-qualified candidate does not need to
repeat the two-second stability check, provided their readiness
conditions remain valid at dequeue time.

**Important dependency:** If the candidate has left the zone or their
perception state is uncertain, they are not currently eligible and must
not be selected.

## Q13. Should the robot independently track and evaluate all other visitors while one target is locked?

**Options presented** - A. Yes, continue tracking and evaluating others
in the background, but do not switch the locked target. (Recommended) -
B. Track others but defer readiness evaluation. - C. Stop processing
other people after target selection.

**User answer:** A --- Independent background evaluation.

**Trade-off:** Improves responsiveness after release, but requires
sufficient compute and careful state management.

## Q14. If several people are waiting, how should the system manage them?

**Options presented** - A. Maintain an ordered candidate queue based on
qualification time. (Recommended) - B. Recompute the best candidate
whenever the target is released. - C. Do not maintain a queue; evaluate
everyone again from scratch.

**User answer:** A --- Maintain an ordered candidate queue.

**Interpretation:** Queue order follows the time at which each person
first satisfies all readiness conditions.

## Q15. After a visit is completed and the target is released, how should a returning person be treated?

**Options presented** - A. Treat a later approach as a new visit, with a
new visit ID. (Recommended) - B. Attempt to associate the returning
person with the previous visit. - C. Require an external signal from the
interaction module to establish a new visit.

**User answer:** A --- New visit.

**Interpretation:** After completion of the previous visit, a later
approach is a new operational visit with a new `visit_id`. No long-term
person recognition is required.

------------------------------------------------------------------------

# Batch 5 --- Queue consistency and lifecycle edge cases

## Q16. A queued person leaves the interaction zone. What should happen?

**Options presented** - A. Immediately remove them from the eligible
queue. (Recommended) - B. Temporarily suspend and wait for return. - C.
Keep them queued for a short timeout before removal.

**User answer:** A --- Remove.

**Interpretation:** A candidate confidently observed leaving the
interaction zone is removed from the eligible queue.

**Clarification:** This is distinct from temporary occlusion or
uncertain visibility loss. The five-second recovery period applies to
temporary loss, not confirmed physical departure.

## Q17. A queued person is temporarily occluded or disappears but returns within five seconds. Should they retain their queue position?

**Options presented** - A. Yes, preserve queue position if visit state
can be reliably associated. (Recommended) - B. Put them at the end of
the queue. - C. Remove them and require a new approach observation.

**User answer:** A --- Retain queue position.

**Interpretation:** Preserve queue priority when the person is reliably
reacquired within the five-second recovery period.

**Dependency:** If association is ambiguous, do not silently merge the
returning track with the previous visit/candidate.

## Q18. The selected target disappears after `FIRST_CONTACT_READY` has been emitted. Who controls the target lifecycle?

**Options initially presented** - A. Downstream interaction module owns
active interaction lifecycle and informs vision about completion or
cancellation. (Recommended) - B. Vision applies five-second timeout and
releases independently. - C. Shared ownership: vision reports track
loss; downstream decides whether to continue or cancel.

**User's initial answer:** C --- Shared ownership.

**User asked for clarification. Explanation provided:** - Vision owns
observable perception facts: visibility, tracking, loss, reacquisition,
and zone departure. - Interaction owns interaction/conversation state
and decides whether to continue, pause, cancel, or complete. - Target
release is coordinated through explicit lifecycle events. - Five-second
perception timeout does not automatically mean the conversation has
ended. - Vision must not initiate a new first-contact interaction while
the prior interaction remains active unless the interaction module has
explicitly released/cancelled/completed it.

**Final user clarification:** Shared ownership confirmed.

**Interpretation:** Vision reports perception status; interaction makes
interaction-level decisions; target release is coordinated. Exact
handshake and timeout/failure behaviour remain open.

## Q19. Two people qualify in the same camera frame. What should determine their order?

**Options presented** - A. Stable deterministic tie-break using tracking
ID. (Recommended) - B. Select the person closest to the robot. - C.
Select the person closest to the centre of the interaction zone. - D.
Select arbitrarily.

**User answer:** B --- Distance.

**Later refinement accepted by user:** Use a distance tolerance. If the
estimated distances are indistinguishable within the tolerance, use
tracking ID as a deterministic fallback.

**Interpretation:** 1. Use qualification time when it distinguishes
candidates. 2. For same-frame ties, prefer the smaller estimated
distance. 3. If the distance difference is within calibrated tolerance,
use deterministic tracking-ID fallback.

**Open technical detail:** Numerical tolerance and uncertainty model
remain to be calibrated.

## Q20. How should candidate queue capacity be managed?

**Options presented** - A. Maintain all currently eligible candidates,
subject to tested capacity. (Recommended) - B. Limit queue to five. - C.
Limit queue to ten. - D. Need explanation.

**User answer:** B --- Limit queue to five.

**Interpretation:** Maximum of five queued eligible candidates. This
does not limit person detection or tracking.

------------------------------------------------------------------------

# Additional decisions resolving earlier ambiguity

## Five-second track recovery semantics

The user questioned how a person could receive a new tracking ID after
returning at three seconds if the system waits five seconds before
discarding a track.

**Resolution:** - The five-second interval is a recovery grace period. -
If the person returns within five seconds and continuity can be
established, recover the original tracking ID. - Do not assign a new ID
merely because the person was invisible for three seconds. - If the
person does not return within five seconds, declare the old track lost
and retire its ID. - A later detection receives a new tracking ID. - A
new track may be associated with an existing visit only when short-term
spatial-temporal evidence is sufficiently reliable. This is not face
recognition. - If association is ambiguous, do not confidently merge the
tracks.

### Example

``` text
t = 0 s: P01 visible
t = 1 s: P01 temporarily lost
t = 3 s: person returns; attempt recovery of P01
         If continuity is reliable, resume P01

Alternative:
t = 0 s: P01 temporarily lost
t = 5 s: recovery period expires; retire P01
t = 8 s: person detected again; assign P02
```

## Physical departure versus temporary invisibility

-   Confidently observed departure from the interaction zone: remove
    candidate from queue.
-   Temporary occlusion or visibility loss without confirmed departure:
    preserve candidate state during the five-second recovery interval.
-   Reliable reacquisition within five seconds: preserve queue position.
-   No reacquisition within five seconds: retire track and expire
    candidate state as appropriate.

------------------------------------------------------------------------

# Round 3 decision summary

  -----------------------------------------------------------------------
  Area                                Final user decision
  ----------------------------------- -----------------------------------
  Tracking scope                      All visible people

  Track identity                      Persistent while continuity is
                                      maintained

  Partial occlusion                   Continue tracking

  Reappearance after declared track   New tracking ID
  loss                                

  Target selection                    First qualified

  Target locking                      Lock

  Target switching                    Do not switch merely because
                                      another person is closer

  Background evaluation               Continue independently

  Track-loss recovery                 5 seconds

  Target release after pre-handoff    Release after five-second recovery
  loss                                expires, subject to lifecycle state

  Visit identity                      Separate from tracking identity

  Next eligible candidate             Select immediately if still
                                      eligible

  Candidate management                Ordered queue

  Queue invalidation                  Remove on confirmed departure from
                                      interaction zone

  Queue position after temporary loss Retain if reliably reacquired
                                      within five seconds

  Post-handoff lifecycle              Shared ownership

  Same-frame tie-break                Distance, then tolerance, then
                                      tracking ID

  Queue capacity                      Five

  Queue overflow                      Five earliest-qualified candidates

  Returning after completed visit     New visit ID
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Open technical details --- not yet decided

1.  Camera model, intrinsics, field of view, mounting geometry, and
    camera/head pose stability.
2.  Maximum expected visible-person count and measured tracking
    capacity.
3.  Detector operating point and confidence thresholds.
4.  Tracker architecture and association method.
5.  Track state transition thresholds and exact five-second timer
    semantics.
6.  Short-term spatial-temporal association method and confidence
    threshold for connecting a new track to an existing visit.
7.  Behaviour when association is ambiguous.
8.  Exact interaction-zone departure confidence and boundary tolerance.
9.  Queue admission policy for people outside the five-person queue when
    a slot becomes free.
10. Eligibility revalidation at dequeue time.
11. Numerical distance tolerance and uncertainty model for same-frame
    tie-breaking.
12. Target-release handshake after first-contact handoff.
13. Interaction-module response timeout and failure behaviour.
14. Whether a new target may be selected while the previous interaction
    is paused.
15. Event schema, payload, timestamps, sequence numbers,
    acknowledgements, retries, and deduplication.
16. Visit completion/new-visit semantics and duplicate first-contact
    prevention.
17. Behaviour after process restart, camera interruption, or module
    restart.
18. Evaluation thresholds, dataset design, and reproducibility protocol.

**Status:** All Round 3 user-facing questions and decisions are
recorded. Round 3 requirements gathering is complete, but the technical
specification is not frozen until the open engineering details are
reviewed.
