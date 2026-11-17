---
title: Vision-Based First Contact — System Architecture
created: 2026-11-16
last_updated: 2026-11-16
status: active
---

# System Architecture

## Purpose
Determine when a person approaching a stationary tabletop reception robot is ready for first contact. The system uses a forward-facing RGB camera and emits a structured event to a separate interaction module.

## Scope
Included: camera acquisition; person detection/tracking; spatial localisation; approach, stationary, and orientation estimation; readiness and candidate management; target/visit lifecycle; event handoff; replay, testing, evaluation, monitoring.

Excluded: speech, dialogue, gestures/actuation, emotion recognition, face recognition/biometric identity, and navigation.

## Logical components
- **Camera/perception:** timestamped RGB frames, person detections, multi-person temporary tracks.
- **Spatial understanding:** camera calibration, geometric mapping, robot-relative location, distance, interaction-zone membership.
- **Behaviour understanding:** temporal approach, motion/stationarity, robot-facing orientation, confidence.
- **Decision engine:** readiness predicates, qualification timestamps, queue, target selection. Must remain independent of a specific neural model.
- **Lifecycle management:** track, visit, target, and recovery state machines.
- **Integration:** versioned event contract and explicit downstream lifecycle coordination.
- **Runtime:** pipeline assembly, scheduling, monitoring, and error handling.

## Data flow
```text
RGB camera -> frame acquisition -> person detection -> multi-object tracking
                                         |                    |
                                         v                    v
                                spatial estimates      behaviour estimates
                                         |                    |
                                         +---------+----------+
                                                   v
                                          readiness evaluation
                                                   v
                                         candidate queue/scheduler
                                                   v
                                          target/visit lifecycle
                                                   v
                                          FIRST_CONTACT_READY
                                                   v
                                         interaction-module handshake
```

## Readiness contract
For candidate \(i\):
\[
E_i(t)=A_i(t)\land Z_i(t)\land D_i(t)\land S_i(t)\land O_i(t)\land Q_i(t)
\]
where \(A\) is observed approach, \(Z\) interaction-zone membership, \(D\) distance within the initial 1.5 m threshold, \(S\) stationary state, \(O\) robot-facing orientation, and \(Q\) reliable required perception.

All predicates must remain true continuously for 2 seconds. Use timestamps/elapsed time, not fixed frame counts. A person already waiting nearby is not eligible without an observed approach. Uncertain required evidence must not trigger first contact. The precise semantics of uncertainty/HOLD during the stability interval remain unresolved and must be specified and tested before implementation.

## Multi-person policy
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

## Handoff and lifecycle
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

## Deployment assumptions and unresolved details
Nominal stream: 1280×720 RGB at 30 FPS. Camera model, intrinsics, FOV, pose, head stability, maximum range, distance error, orientation/stationary thresholds, detector/tracker, and middleware are not yet confirmed. Floor-plane homography is viable only if camera pose is stable and planar-ground assumptions hold.
