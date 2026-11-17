# Project Status

## Objective
Build a vision-only first-contact subsystem for a stationary tabletop social robot at an office reception. It observes people through a forward-facing RGB camera, estimates spatial/temporal behaviour, selects an eligible person, and hands off a structured event to a separate interaction module.

## Current phase
Requirements consolidation and repository/architecture setup. Implementation stack and camera calibration are not frozen.

## Confirmed requirements
- Stationary tabletop robot; forward-facing, head-mounted RGB camera.
- Hardware verified stream: 1280×720 at 30 FPS (MJPG format required; YUYV throttles to 10 FPS). Camera model confirmed as Luxvisions Innotech Integrated RGB Camera (0x30c9:0x00a8, uvcvideo).
- Mounting geometry confirmed: Lens height 115 cm, pitch ~0°, yaw ~0°, roll ~0°, fixed mounting.
- Framing observation confirmed: Feet not visible at 1.5 m (occluded by reception desk and vertical FOV cut-off below 2.78 m). Ground-plane homography from bounding-box bottom is infeasible; upper-body keypoint distance estimation required.
- Camera calibration acquisition utility implemented in tools/calibration/camera_capture_utility.py supporting diagnostic stream test, Mode A (scene inspection), and Mode B (intrinsic pattern capture).
- Office reception conditions include artificial light, daylight, low light, crowds, carried objects, and partial visibility.
- Track all visible people simultaneously; maintain temporary IDs through reliable continuity and partial occlusion.
- Attempt reacquisition for up to 5 seconds; retire a track after the grace period if not reacquired.
- Keep `track_id`, `visit_id`, and `target_id` separate.
- Readiness requires observed approach, interaction-zone membership, distance within initial 1.5 m threshold, stationary behaviour, robot-facing orientation, and reliable perception.
- All readiness predicates must remain true continuously for 2 seconds.
- Uncertain evidence means HOLD/no trigger.
- Select and lock the first-qualified candidate; continue background tracking/evaluation.
- Candidate queue capacity five, ordered by qualification time.
- Remove candidates on confidently observed departure from the interaction zone.
- Same-frame tie: smaller estimated distance; within calibrated tolerance, deterministic tracking-ID fallback.
- Downstream interaction module participates in lifecycle decisions and explicitly coordinates release.
- Perception timeout does not by itself terminate an interaction.

## Completed project artefacts
Requirements/question records from rounds 1–3 and a consolidated implementation blueprint have been prepared separately. Active repository directory structure confirmed; empty skeleton directories pruned in favor of just-in-time package creation.

## Current work
Establish project governance documents, confirm the repository tree, and resolve hardware/calibration and state-machine ambiguities before implementation.

## Blockers and open engineering questions
1. Camera intrinsics, FOV, mounting height/pitch/yaw, and head stability (camera model and 720p@30fps MJPG capability confirmed in M0 Batch 1).
2. Maximum reliable detection range and image coverage.
3. Calibration method and distance error near the 1.5 m boundary.
4. Ground-contact estimation when feet are occluded.
5. Robot-facing orientation definition/estimator.
6. Stationary threshold and temporal filtering.
7. Detector operating point and maximum simultaneous people.
8. Tracker association/occlusion policy and confidence thresholds.
9. Exact interaction between uncertainty/HOLD and the 2-second timer.
10. Zone boundary tolerance and departure confidence.
11. Queue admission when capacity becomes available.
12. Event schema, acknowledgement, retries, ordering, deduplication.
13. Interaction-module failure, restart, and target-release behaviour.

## Next priorities
1. Integrate documentation without overwriting existing project material.
2. Conduct camera/site survey and collect calibration data.
3. Define typed interfaces for observations, tracks, spatial/behaviour estimates, candidates, and lifecycle events.
4. Build deterministic tests for readiness timing, queue policy, and lifecycle.
5. Implement offline video replay before live deployment.

## Critical invariants
No trigger under uncertain required evidence; approach is mandatory; readiness is continuous for 2 seconds; locked target is not replaced merely by a closer person; background evaluation continues; temporary invisibility differs from confirmed departure; identities remain separate; perception timeout does not end interaction; RGB distance requires calibration and uncertainty.

## References
- [Architecture](research/ARCHITECTURE.md)
- [Repository map](research/REPOSITORY_MAP.md)
- [Engineering standards](research/ENGINEERING_STANDARDS.md)
- [Worklog](WORKLOG.md)
