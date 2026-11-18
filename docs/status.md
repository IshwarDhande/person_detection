# Project Status

## Objective

Build a vision-only first-contact subsystem for a stationary tabletop social robot at an office reception. It observes people through a forward-facing RGB camera, estimates spatial and temporal behaviour, selects an eligible person, and hands off a structured event to a separate interaction module.

## Current development strategy

**Incremental, demo-driven implementation.**

Develop the software first using the laptop's built-in webcam. Complete and demonstrate one milestone at a time. After the single-person pipeline and subsequent multi-person/lifecycle functionality are validated, transfer the software to the reception robot and perform robot-specific camera calibration.

The laptop camera is a development device. Its geometry and calibration must not be treated as the robot camera's geometry or calibration.

## Current phase

**M1 — Basic person detection: next implementation milestone.**

The camera discovery/capture utility and initial camera inspection work have been reported in the worklog. The first person-detection demo has not yet been recorded as completed in this status file.

## Implementation milestones

| Milestone | Scope | Status |
|---|---|---|
| M1 | Basic person detection: live webcam, person box, confidence, FPS, latency | Not started |
| M2 | Reliable single-person detection | Not started |
| M3 | Single-person tracking | Not started |
| M4 | Frame-level readiness perception (Part A) | Not started |
| M5 | Temporal behaviour estimation (Part B) | Not started |
| M6 | Complete single-person first-contact pipeline | Not started |
| M7 | Multi-person detection and tracking | Not started |
| M8 | Multi-person readiness, queue and target selection | Not started |
| M9 | Downstream lifecycle/event integration | Not started |
| M10 | Robot integration, calibration and deployment validation | Deferred |

A milestone is complete only when the feature works, a repeatable demo is available, relevant tests pass, evaluation evidence is reported, and limitations are documented. Do not mark a milestone complete solely because code exists.

## Confirmed system requirements

- Stationary tabletop reception robot; intended robot camera is forward-facing RGB.
- The development camera has been identified in prior camera discovery work as a Luxvisions Innotech integrated RGB camera, with a reported 1280×720 MJPG stream at approximately 30 FPS. Treat this as the current development-camera characterization, not as confirmation of the eventual robot camera.
- The laptop webcam is used for initial software development. Robot camera identity and geometry remain to be characterized during deployment.
- Reception conditions may include artificial light, daylight, low light, crowds, carried objects, and partial visibility.
- The eventual system must track all visible people simultaneously and maintain temporary IDs through reliable continuity and partial occlusion.
- Attempt reacquisition for up to 5 seconds; retire a track after the grace period if it is not reliably reacquired.
- Keep `track_id`, `visit_id`, and `target_id` separate.
- Readiness requires observed approach, interaction-zone membership, distance within the initial 1.5 m threshold, stationary behaviour, robot-facing orientation, and reliable perception.
- All readiness predicates must remain true continuously for 2 seconds.
- Uncertain required evidence must not trigger first contact. HOLD/timer semantics require an explicit decision before final readiness implementation.
- Select and lock the first-qualified candidate; continue background tracking/evaluation.
- Candidate queue capacity is five, ordered by qualification time.
- Remove candidates on confidently observed departure from the interaction zone.
- Same-frame tie: smaller estimated distance; within calibrated tolerance, deterministic tracking-ID fallback.
- Downstream interaction module participates in lifecycle decisions and explicitly coordinates release.
- Perception timeout does not by itself terminate an interaction.

## Camera and calibration notes

Prior camera discovery and capture work is recorded in `WORKLOG.md` and the calibration documents. The current development-camera stream was reported as 1280×720 MJPG at approximately 30 FPS.

A scene image has been captured for inspection. The reported observation that feet were not visible at approximately 1.5 m applies to the inspected development-camera setup only; it must not be generalized to the eventual robot camera.

Do not treat the previously suggested ground-plane/upper-body localization approach as validated. Camera FOV, intrinsics, distortion, actual mounting pose, and distance-estimation accuracy require measurement. In particular, do not infer a precise floor visibility cutoff or metric distance accuracy without measured FOV/calibration evidence.

## Immediate next action

Implement **M1 only**:
- Open the laptop webcam.
- Run a suitable pretrained person detector.
- Display person bounding box and confidence.
- Display FPS and inference latency.
- Support clean shutdown and clear initialization errors.
- Run basic detector/camera smoke tests and demonstrate the live application.

Do not implement tracking, readiness, multi-person scheduling, or robot calibration in M1.

## Critical invariants

No trigger under uncertain required evidence; approach is mandatory; readiness is continuous for 2 seconds; locked target is not replaced merely by a closer person; background evaluation continues; temporary invisibility differs from confirmed departure; identities remain separate; perception timeout does not end interaction; RGB distance requires calibration and uncertainty.

## References

- [Architecture](architecture.md)
- [Worklog](worklog.md)
- [Incremental implementation roadmap](implementation_roadmap.md)
