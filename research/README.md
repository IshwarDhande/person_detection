# Vision-Based First Contact

A vision-only first-contact subsystem for a stationary tabletop social robot at an office reception. It detects and tracks people, estimates spatial and temporal behaviour, decides first-contact readiness, and emits a structured event to a separate interaction module.

## Scope
**In scope:** RGB acquisition; multi-person detection/tracking; calibrated person localisation; approach, stationary, and robot-facing estimation; interaction-zone/readiness logic; candidate queue and target selection; visit/target lifecycle; event handoff; replay, evaluation, testing, and deployment monitoring.

**Out of scope:** speech, dialogue, gestures/actuation, emotion recognition, face recognition/biometric identity, and navigation.

## Initial behavioural contract
A candidate is eligible only after an observed approach, interaction-zone membership, estimated distance within the initial 1.5 m threshold, stationary behaviour, robot-facing orientation, and reliable perception have all remained true continuously for 2 seconds. Uncertain evidence must not trigger first contact. A person already waiting nearby is not eligible without an observed approach.

Select and lock the first-qualified candidate. Continue background tracking/evaluation. Maintain at most five eligible candidates in qualification order. Remove a queued candidate on confidently observed departure. Use a 5-second temporary track-loss reacquisition grace period. Keep `track_id`, `visit_id`, and `target_id` distinct. The downstream interaction module participates in post-handoff lifecycle control; perception timeout alone does not end an interaction.

These are requirements, not validated performance claims. Camera geometry, thresholds, confidence calibration, and operating limits remain to be established.

## Repository structure
- `configs/runtime/`: runtime environment configuration (`environment.yaml`).
- `data/raw/calibration/`: raw camera calibration frames and scene inspection baselines.
- `docs/calibration/`: hardware surveys, framing reports, and capture guides.
- `research/`: engineering requirements, architecture, standards, and decision records.
- `tools/calibration/`: hardware discovery and calibration capture utilities.

Planned subsystem directories (`perception/`, `spatial_understanding/`, `behaviour_understanding/`, `decision_engine/`, `lifecycle_management/`, `integration/`, `runtime/`, `evaluation/`, `experiments/`, `tests/`, `checkpoints/`, `logs/`, `deployment/`) are introduced just-in-time as their implementation and test code is developed. See `research/REPOSITORY_MAP.md` and `research/ARCHITECTURE.md`.

## Development state
Requirements and architecture are being consolidated. Detector, tracker, pose estimator, localisation method, and runtime stack are provisional until evaluated on the target hardware. See `STATUS.md`.

## Engineering principles
Keep modules replaceable and decision logic model-independent. Use timestamps, explicit uncertainty, fail-closed triggering, offline replay, deterministic tests, and traceable configuration/model/data provenance. Do not commit private reception video or large artifacts without approval.
