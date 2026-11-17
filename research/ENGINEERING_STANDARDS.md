---
title: Engineering Standards
created: 2026-11-16
last_updated: 2026-11-16
status: active
---

# Engineering Standards

## Architecture
- Start as a modular monolith with explicit package boundaries and typed interfaces.
- Keep inference replaceable and decision policy model-agnostic.
- Separate observations, estimates, decisions, and externally visible events.
- Keep state transitions explicit and testable.
- Avoid premature distributed infrastructure.

## Time and state
- Use monotonic time for elapsed durations and deadlines.
- Preserve source timestamps with frames and derived observations.
- Do not equate frame counts with elapsed time.
- Distinguish track loss, confirmed departure, target release, and interaction completion.
- Keep `track_id`, `visit_id`, and `target_id` distinct.

## Uncertainty and geometry
- Expose validity/confidence for estimates used by decisions.
- Fail closed when required evidence is uncertain.
- Do not treat detector confidence as a calibrated probability without measurement.
- Document camera intrinsics, distortion, pose, and calibration.
- Floor-plane homography requires stable camera pose and a valid planar ground assumption.
- Estimate location from a defensible ground-contact point; define behaviour when feet are occluded.
- Measure distance error near the 1.5 m decision boundary. Bounding-box height alone is not reliable physical distance.

## Interfaces
- Use typed, versioned contracts.
- Include timestamps, identifiers, validity/confidence, and coordinate-frame definitions where needed.
- Define acknowledgement, retry, timeout, and duplicate handling for event delivery.
- Do not silently reinterpret existing fields or states.

## Testing
- Unit-test geometry, readiness timing, queue ordering, tie-breaks, and lifecycle transitions.
- Test just below/at/above temporal and spatial thresholds.
- Test dropped frames, temporary occlusion, reacquisition, stale observations, and departure.
- Use deterministic fixtures and recorded-video replay.
- Separate hardware-dependent tests from ordinary CI tests.

## Evaluation
Report subsystem results separately: detection precision/recall; tracking ID switches/fragmentation and suitable HOTA/IDF1 metrics; localisation distance error; behaviour precision/recall; decision false-trigger rate, missed eligible candidates, and latency; end-to-end handoff success and duplicate rate. Define acceptance thresholds before deployment claims.

## Reproducibility
Record code revision, dirty-tree state, configuration, model/checkpoint, dataset/video fixture, dependencies, and random seed where applicable. Preserve useful failed runs.

## Privacy
Treat reception video as potentially sensitive personal data. Follow organisational access, retention, consent, and security policies. Do not commit raw video, face crops, identifying data, or credentials. Do not add biometric identity.

## Code quality
Use clear names, cohesive functions, type annotations on public interfaces, structured logging, startup configuration validation, and actionable errors.
