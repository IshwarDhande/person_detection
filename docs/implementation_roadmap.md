# Vision-Based First Contact — Incremental Implementation Roadmap

**Strategy:** Demo-driven incremental development  
**Initial platform:** Laptop built-in webcam  
**Final platform:** Stationary reception robot  
**Status:** Proposed implementation order

## 1. Purpose

Reorganize implementation into small, demonstrable milestones. The project requirements remain unchanged; only the implementation order changes.

The sequence is:

1. Basic person detection.
2. Reliable single-person detection.
3. Single-person tracking.
4. Frame-level readiness perception.
5. Temporal behaviour estimation.
6. Complete single-person first-contact pipeline.
7. Multi-person detection and tracking.
8. Multi-person readiness and target selection.
9. Lifecycle and downstream integration.
10. Robot deployment and calibration.

**Core principle:** Complete, demonstrate, test, and evaluate one milestone before starting the next.

## 2. Two-stage development strategy

### Stage A — Laptop development

Use the laptop's built-in webcam to implement and demonstrate the software pipeline. The laptop camera is a development device; its geometry and calibration must not be treated as the robot's geometry.

Develop hardware-independent perception, behavioural reasoning, readiness, scheduling, and integration interfaces where practical.

### Stage B — Robot deployment

After the software pipeline is sufficiently implemented and validated:

- Integrate the robot-mounted camera.
- Characterize its capture configuration.
- Measure mounting geometry.
- Perform robot-specific intrinsic and extrinsic calibration.
- Establish the physical interaction zone and validate distance estimation.
- Evaluate reception lighting, crowding, and occlusion.
- Tune deployment-specific configuration.

Robot-specific calibration is deferred during initial laptop development.

## 3. Milestones

### M1 — Basic person detection

**Goal:** First visible working demo.

Scope:
- Open the laptop webcam.
- Run a suitable pretrained person detector.
- Filter detections to the person class.
- Draw bounding boxes and confidence scores.
- Display FPS and inference latency.
- Handle camera/model initialization errors.
- Support clean shutdown.

**Demo:** A person enters the webcam view and a bounding box with confidence appears.

**Exit criteria:**
- Live person detection works.
- Bounding box and confidence are visible.
- FPS and latency are visible.
- Application exits cleanly.
- Basic detector and camera smoke tests pass.
- A repeatable execution command is provided.

**Out of scope:** Tracking, localization, approach/stationary estimation, readiness, multi-person scheduling, robot calibration.

### M2 — Reliable single-person detection

**Goal:** Establish stable detection behaviour for one person.

Scope:
- Continuous real-time inference.
- Configurable confidence threshold.
- Detection filtering and basic smoothing if needed.
- Handle intermittent missed detections.
- Test distance, pose, and lighting variations.
- Measure detection latency and throughput.

**Demo:** One person remains in view, moves, changes pose, and becomes partially occluded.

**Exit criteria:** Continuous operation, bounded resource use, tested confidence filtering, repeatable evaluation, and documented failure cases.

### M3 — Single-person tracking

**Goal:** Maintain a temporary track for one detected person.

Scope:
- Associate successive detections.
- Maintain a temporary `track_id`.
- Track bounding-box movement.
- Display recent trajectory.
- Handle temporary detection loss and attempt reacquisition.
- Measure tracking continuity and latency.

**Demo:** One person moves, stops, briefly leaves the frame, and returns; track continuity is visualized.

**Exit criteria:** Track creation/update/loss/recovery are testable; normal movement is tracked; temporary loss is handled explicitly; limitations are documented.

This milestone validates single-person continuity, not the final multi-person association policy.

### M4 — Frame-level readiness perception (Part A)

**Goal:** Infer properties available from an individual frame.

Input: One RGB frame and its single-person detection.

Potential observations:
- Person detected and detection confidence.
- Bounding box and visibility.
- Image-space interaction-zone membership.
- Pose/keypoint availability and quality.
- Robot-facing orientation estimate and confidence, if supported.
- Per-property validity and uncertainty.

Output: Structured frame-level observation with confidence/validity. Use `UNKNOWN` when evidence is insufficient.

**Demo:** Overlay frame-level properties and indicate which are valid, invalid, or uncertain.

**Exit criteria:** Observation contract defined; confidence/validity exposed; uncertainty is preserved; tests cover valid, invalid, and uncertain cases.

**Important:** Image-space zone membership is not metric distance. Laptop geometry is a prototype; physical distance and robot-specific zone calibration are deferred.

### M5 — Temporal behaviour estimation (Part B)

**Goal:** Infer behaviour from a sequence of tracked observations.

Input: Timestamped single-person track history and M4 observations.

Scope:
- Maintain timestamped observation history.
- Estimate image-space movement and trajectory.
- Estimate movement direction.
- Develop approach-to-reception predicate.
- Estimate velocity and stationary state.
- Apply temporal smoothing/hysteresis where needed.
- Implement timestamp-based two-second stability timing once predicates are available.
- Preserve uncertainty explicitly.

**Demo:** A person approaches, stops, and remains in the relevant region; indicators show approach, movement, stationary state, and readiness timing.

**Exit criteria:** Timestamp-based logic; controlled/labelled sequence evaluation; false positives/negatives measured; deterministic timer tests; HOLD/uncertainty treatment explicitly decided before production readiness is finalized.

Laptop image-space movement is useful for prototyping but does not validate metric movement or physical distance.

### M6 — Complete single-person first-contact pipeline

**Goal:** Integrate M1–M5.

Scope:
- Camera acquisition.
- Person detection and single-person tracking.
- Frame-level readiness observations.
- Temporal behaviour estimates.
- Per-person readiness state machine.
- Two-second continuous eligibility logic.
- Confidence-aware fail-closed decisions.
- Structured `FIRST_CONTACT_READY` event.

The agreed readiness predicate remains:

\[
E_i(t)=A_i(t)\land Z_i(t)\land D_i(t)\land S_i(t)\land O_i(t)\land Q_i(t)
\]

Where:
- `A`: approach observed.
- `Z`: inside interaction zone.
- `D`: within configured distance threshold with reliable estimation.
- `S`: stationary.
- `O`: facing the robot.
- `Q`: perception evidence sufficiently reliable.

All required predicates must remain valid continuously for two seconds.

**Demo:** One person enters, approaches, stops, faces the robot direction, satisfies the configured zone/proximity evidence, and triggers `FIRST_CONTACT_READY`.

**Exit criteria:** End-to-end single-person pipeline works; transitions are observable; uncertainty does not trigger; leaving the zone resets pending stability; unit/replay tests pass; event is emitted once per relevant visit; limitations are documented.

If laptop metric distance is unavailable, use a clearly labelled prototype/mock distance input to test decision logic. Never represent it as validated camera-derived distance.

### M7 — Multi-person detection and tracking

**Goal:** Extend perception to multiple simultaneous people.

Scope:
- Detect multiple people per frame.
- Introduce a multi-object tracker.
- Maintain independent temporary `track_id` values.
- Handle crossing paths, partial occlusion, fragmentation, and recovery.
- Implement the five-second track-recovery grace period using timestamps.
- Evaluate ID switches and track recovery.

**Demo:** Multiple people enter, leave, cross, and re-enter with independent track IDs.

**Exit criteria:** Multiple tracks maintained; lifecycle testable; five-second recovery validated; tracking metrics (e.g. IDF1/HOTA, ID switches) measured; performance measured at increasing person counts.

### M8 — Multi-person readiness and target selection

**Goal:** Apply single-person readiness independently to multiple tracks and implement scheduling.

Scope:
- Maintain per-track readiness state.
- Evaluate all visible people in the background.
- Select first-qualified candidate.
- Lock selected target; do not pre-empt merely because another person is closer.
- Maintain qualification-ordered queue with capacity five.
- Same-frame tie-break: smaller estimated distance; within calibrated tolerance, deterministic tracking-ID fallback.
- Remove candidates on confidently observed departure.
- Preserve queue position during reliable short-term reacquisition.
- On release, revalidate and select the next eligible candidate without repeating the two-second interval if eligibility remains valid.

**Demo:** Multiple people approach; the first-qualified candidate is selected and others are queued.

**Exit criteria:** Independent readiness, deterministic selection, queue ordering/capacity, tie-breaking, target locking, and release are tested.

### M9 — Lifecycle and downstream integration

**Goal:** Integrate with a separate interaction module.

Scope:
- Define versioned event contract.
- Publish `FIRST_CONTACT_READY`.
- Report target loss, reacquisition, and departure where required.
- Receive interaction lifecycle events.
- Coordinate explicit target release.
- Prevent a new first-contact event while the previous interaction remains active unless release is confirmed.
- Handle duplicate, delayed, or missing acknowledgements.
- Test with a simulated downstream interaction module.

**Demo:** Simulated interaction starts, receives target-loss/reacquisition events, completes or cancels, and explicitly releases the target.

**Exit criteria:** Handoff, acknowledgement, release, timeout, and error paths tested. A five-second perception timeout must not imply conversation completion. Duplicate first-contact events are prevented within the defined lifecycle.

### M10 — Robot deployment and calibration

**Goal:** Transfer the validated pipeline to the actual reception robot.

Scope:
- Identify/configure robot camera and adapter.
- Measure actual mounting height, pitch, yaw, and roll.
- Calibrate intrinsics and distortion.
- Establish camera-to-floor extrinsics where appropriate.
- Validate floor visibility and ground-contact estimation.
- Establish physical interaction-zone geometry.
- Validate metric distance around the 1.5 m target.
- Evaluate lighting, crowding, occlusion, and partial visibility.
- Profile robot compute performance and tune deployment configuration.

**Demo:** Complete first-contact pipeline runs on the robot in the reception environment.

**Exit criteria:** Robot-specific calibration recorded; distance/zone accuracy measured against physical ground truth; representative deployment evaluation completed; acceptance criteria met or remaining gaps documented.

## 4. Readiness split: Part A and Part B

### Part A — Frame-level evidence (M4)

For frame \(I_t\):

\[
O_t=f_\theta(I_t)
\]

Estimate person detection, visibility, image-space zone membership, pose/keypoints, facing-direction evidence, and per-property confidence/validity.

This answers: **What can be inferred from this frame?**

### Part B — Temporal behaviour (M5)

Given observation history:

\[
H_t=\{O_{t-k},\ldots,O_t\}
\]

Estimate approach history, movement direction/velocity, stationary state, temporal consistency, and readiness stability duration.

This answers: **What behaviour is the person exhibiting over time?**

M6 combines both parts into the single-person readiness state machine. Do not put temporal approach/stationary logic into the frame-level model merely for convenience.

## 5. How progress is measured

A milestone is not complete merely because code has been written. Each milestone must have:

1. **Working feature:** Requested behaviour runs.
2. **Demonstration:** Repeatable live or recorded demo.
3. **Tests:** Relevant automated tests pass.
4. **Evaluation:** Relevant metrics or measured observations are recorded.
5. **Known limitations:** Failures and unvalidated assumptions are stated.

| Milestone | Feature | Demo | Evaluation |
|---|---|---|---|
| M1 | Basic person detection | Live webcam box | FPS/latency |
| M2 | Reliable detection | Continuous session | Detection performance |
| M3 | Single-person tracking | Track ID/trajectory | Continuity/recovery |
| M4 | Frame-level readiness | Observation overlay | Per-property quality |
| M5 | Temporal behaviour | Approach/stop demo | Behaviour metrics |
| M6 | Single-person pipeline | First-contact event | Trigger correctness |
| M7 | Multi-person tracking | Multiple track IDs | IDF1/HOTA |
| M8 | Multi-person scheduling | Queue/target demo | Selection correctness |
| M9 | Lifecycle integration | Simulated interaction | Handoff/release correctness |
| M10 | Robot deployment | Reception demo | Physical accuracy/robustness |

Initial status for all milestones: **Not started**. Update progress only when supported by an actual implementation and demonstration.

## 6. Development discipline

- Work on one milestone at a time.
- Do not automatically proceed to the next milestone.
- Stop after the current milestone and provide a concise implementation/demo/test report.
- Wait for review and authorization before starting the next milestone.
- Preserve existing user changes; do not commit unless explicitly authorized.
- Do not create or update governance files, `WORKLOG.md`, `STATUS.md`, architecture documents, or repository maps unless explicitly asked.
- Inspect only repository files relevant to the current milestone.
- Avoid speculative abstractions and unnecessary dependencies.
- Use timestamps for temporal logic, not fixed frame counts.
- Never claim laptop-camera measurements validate robot-camera geometry.

## 7. Immediate next action

Start with **M1 — Basic person detection only**.

The first deliverable is a working laptop webcam demo showing:
- Live camera feed.
- Person bounding box.
- Detection confidence.
- FPS.
- Inference latency.
- Clean shutdown.

Do not implement tracking, readiness, multi-person scheduling, or robot calibration in M1.

After M1 is demonstrated and reviewed, proceed to M2.
