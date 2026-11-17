---
title: Vision-Based First Contact — System Architecture
created: 2026-11-16
last_updated: 2026-11-17
status: active
---

# System Architecture

## Purpose

Determine when a person approaching a stationary tabletop reception robot is ready for first contact. The system uses a forward-facing RGB camera and emits a structured event to a separate interaction module.

## Scope

Included: camera acquisition; person detection/tracking; spatial localisation; approach, stationary, and orientation estimation; readiness and candidate management; target/visit lifecycle; event handoff; replay, testing, evaluation, monitoring.

Excluded: speech, dialogue, gestures/actuation, emotion recognition, face recognition/biometric identity, and navigation.

## Development and deployment strategy

Implementation is incremental and demo-driven.

- **Initial development:** laptop built-in webcam.
- **Later deployment:** actual robot-mounted camera, with separate robot-specific calibration and configuration.

The laptop camera is used to develop and validate software behaviour. Its intrinsics, field of view, mounting geometry, image-space zones, and distance estimates must not be reused as robot calibration.

## Incremental implementation order

1. **M1 — Basic person detection:** webcam input, person box, confidence, FPS, latency, clean shutdown.
2. **M2 — Reliable single-person detection:** continuous operation, filtering, missed-detection handling and evaluation.
3. **M3 — Single-person tracking:** temporary ID, trajectory, loss and reacquisition.
4. **M4 — Frame-level readiness perception (Part A):** per-frame observations, visibility, image-space zone, pose/orientation evidence, confidence and unknown states.
5. **M5 — Temporal behaviour estimation (Part B):** approach, movement, velocity, stationary state, temporal filtering and timestamp-based stability.
6. **M6 — Complete single-person pipeline:** readiness state machine and `FIRST_CONTACT_READY` event.
7. **M7 — Multi-person detection and tracking.**
8. **M8 — Multi-person readiness, candidate queue and target selection.**
9. **M9 — Downstream lifecycle and event integration.**
10. **M10 — Robot camera integration, robot-specific calibration and deployment validation.**

Each milestone must have a working feature, repeatable demo, relevant tests, evaluation evidence, and stated limitations. Work on one milestone at a time; progression requires review/authorization.

## Logical components

- **Camera/perception:** timestamped RGB frames, person detections, temporary tracks.
- **Spatial understanding:** camera calibration, geometric mapping, robot-relative location, distance, interaction-zone membership.
- **Behaviour understanding:** temporal approach, motion/stationarity, robot-facing orientation, confidence.
- **Decision engine:** readiness predicates, qualification timestamps, queue, target selection. Must remain independent of a specific neural model.
- **Lifecycle management:** track, visit, target, and recovery state machines.
- **Integration:** versioned event contract and explicit downstream lifecycle coordination.
- **Runtime:** pipeline assembly, scheduling, monitoring, and error handling.

## Readiness decomposition

Readiness is implemented in two distinct parts before integration.

### Part A — Frame-level perception (M4)

For an RGB frame \(I_t\):

\[
O_t=f_	heta(I_t)
\]

Produce structured observations such as person detection/confidence, bounding box, visibility, image-space zone membership, pose/keypoint evidence, facing-direction evidence, and per-property validity/confidence.

This stage answers: **What can be inferred from this frame?**

Use explicit unknown/uncertain states. Do not force unsupported binary estimates.

### Part B — Temporal behaviour (M5)

Given timestamped observation history:

\[
H_t=\{O_{t-k},\ldots,O_t\}
\]

Estimate approach history, movement direction/velocity, stationary state, temporal consistency, and readiness stability duration.

This stage answers: **What behaviour is the person exhibiting over time?**

Use timestamps/elapsed time rather than fixed frame counts. Keep frame-level observation extraction separate from temporal reasoning.

### Complete readiness (M6 onward)

For candidate \(i\):

\[
E_i(t)=A_i(t)\land Z_i(t)\land D_i(t)\land S_i(t)\land O_i(t)\land Q_i(t)
\]

where \(A\) is observed approach, \(Z\) interaction-zone membership, \(D\) distance within the initial 1.5 m threshold, \(S\) stationary state, \(O\) robot-facing orientation, and \(Q\) reliable required perception.

All predicates must remain true continuously for 2 seconds. A person already waiting nearby is not eligible without an observed approach. Uncertain required evidence must not trigger first contact. The precise semantics of uncertainty/HOLD during the stability interval remain unresolved and must be specified and tested before production readiness is finalized.

During laptop development, image-space zone and movement estimates may be used to prototype logic. They are not metric robot-relative distance. If metric distance is unavailable, use an explicitly labelled test/mock input for decision-logic tests; do not claim camera-derived distance is validated.

## Multi-person policy (M7–M8)

- Track all visible people; continue through partial occlusion where association is reliable.
- Use a 5-second temporary track-loss reacquisition grace period.
- Keep `track_id`, `visit_id`, and `target_id` distinct.
- Select and lock the first-qualified candidate; do not switch simply because another person is closer.
- Continue evaluating other candidates in the background.
- Queue up to five eligible candidates in qualification order.
- Remove a candidate on confidently observed departure from the interaction zone.
- Preserve queue position after reliable reacquisition within the grace period.
- When target is released, choose the next currently eligible candidate without repeating the 2-second interval if eligibility remains valid.
- Same-frame tie: smaller estimated distance; if within calibrated tolerance, deterministic tracking-ID fallback.

## Handoff and lifecycle (M9)

After `FIRST_CONTACT_READY`, interaction continuation/pause/cancel/completion is coordinated with the downstream interaction module. A 5-second perception timeout is a track-recovery rule, not proof that interaction ended. Do not activate another target until explicit release unless the agreed lifecycle contract says otherwise. Event schema, acknowledgement, retries, deduplication, and failure policy remain open.

## Invariants

1. Uncertain required evidence cannot trigger first contact.
2. Approach is mandatory; proximity alone is insufficient.
3. Readiness predicates satisfy the continuous-time requirement.
4. Locked target is not replaced merely because another person is closer.
5. Background candidate evaluation continues.
6. Temporary invisibility differs from confirmed departure.
7. Track, visit, and target identities are distinct.
8. Perception timeout does not terminate interaction.
9. RGB distance requires calibration and uncertainty handling.
10. Decision policy is testable independently of model inference.
11. Laptop calibration is not robot calibration.
12. Milestones are demonstrated and evaluated individually.

## Deployment assumptions and unresolved details

The development camera was previously reported to support a 1280×720 MJPG stream at approximately 30 FPS. Treat those values as the measured development-camera configuration, not as guaranteed robot hardware specifications.

Robot camera identity, intrinsics, FOV, pose, maximum range, distance error, orientation/stationary thresholds, detector/tracker operating point, and middleware remain to be validated for deployment. Ground-plane homography is viable only when camera pose, visible ground geometry, and planar-ground assumptions are appropriate.
