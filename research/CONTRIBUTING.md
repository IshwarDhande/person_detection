# Contributing

## Workflow

1. Read `AGENTS.md` and only the relevant status/architecture/requirements references.
2. Identify the current approved implementation milestone.
3. Inspect `git status --short` and the target code before editing.
4. Make a focused, minimal change limited to the current milestone.
5. Add or update tests for the changed behaviour.
6. Run the smallest relevant test set, then broader tests only when warranted.
7. Demonstrate the feature using the appropriate live or replay workflow.
8. Report measured evaluation results and known limitations.
9. Review `git diff` and `git diff --check`.
10. Do not begin the next milestone without user review/authorization.

## Incremental milestone workflow

The project follows a demo-driven sequence:

1. M1 — Basic person detection.
2. M2 — Reliable single-person detection.
3. M3 — Single-person tracking.
4. M4 — Frame-level readiness perception (Part A).
5. M5 — Temporal behaviour estimation (Part B).
6. M6 — Complete single-person first-contact pipeline.
7. M7 — Multi-person detection and tracking.
8. M8 — Multi-person readiness and target selection.
9. M9 — Lifecycle and downstream integration.
10. M10 — Robot integration, calibration and deployment.

A milestone is complete only when the feature works, a repeatable demo is available, relevant tests pass, evaluation evidence is reported, and limitations are stated. Code completion alone is insufficient.

## Development platform and calibration

Use the laptop built-in webcam for initial software development. Do not treat laptop camera calibration, image-space zones, or distance estimates as robot calibration. Robot-specific camera integration and physical calibration belong to M10.

## Branches and commits

Use descriptive branches such as `feature/person-detection` or `feature/single-person-tracking`. Keep commits focused and use imperative subjects. Do not commit raw reception video, personal data, credentials, temporary logs, or large checkpoints without approval. Do not commit unless authorized.

## Review expectations

Explain the milestone addressed, design choice, changed interfaces/configuration, tests actually run, demo result, measured metrics, assumptions, and known limitations. Do not claim real-world performance from unit tests or a small set of videos.

## Testing

Test the behaviour introduced by the current milestone. For later milestones, include temporal boundaries, dropped frames, occlusion/reacquisition, queue ordering, lifecycle transitions, and event handoff as applicable. Keep hardware-independent tests separate from hardware-in-the-loop validation. Use recorded-video replay for repeatability.

## Documentation ownership

- `README.md`: project introduction, roadmap summary and setup/navigation.
- `research/STATUS.md`: current project state and milestone status.
- `research/WORKLOG.md`: engineering session history, when requested/required by project convention.
- `research/ARCHITECTURE.md`: system design and interfaces.
- `research/REPOSITORY_MAP.md`: actual code navigation.
- `research/IMPLEMENTATION_ROADMAP.md`: milestone sequence and exit criteria.
- `research/ENGINEERING_STANDARDS.md`: engineering and validation standards.

Do not automatically update governance documents after ordinary implementation tasks. Update only the documents explicitly requested or materially affected by an authorized change.
