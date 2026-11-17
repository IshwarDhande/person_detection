# Contributing

## Workflow
1. Read `AGENTS.md`, `STATUS.md`, and relevant architecture documentation.
2. Inspect `git status` and existing code before editing.
3. Make a focused, minimal change.
4. Add or update tests.
5. Run relevant tests and inspect failures.
6. Review `git diff` and `git diff --check`.
7. Update status, worklog, architecture, repository map, or ADRs when warranted.

## Branches and commits
Use descriptive branches such as `feature/person-tracking` or `fix/readiness-timer`. Keep commits focused and use imperative subjects. Do not commit raw reception video, personal data, credentials, temporary logs, or large checkpoints without approval.

## Review expectations
Explain the requirement addressed, design choice, changed interfaces/configuration, tests actually run, assumptions, and known limitations. Do not claim real-world performance from unit tests or a small set of videos.

## Testing
Test temporal boundaries, dropped frames, occlusion/reacquisition, queue ordering, lifecycle transitions, and event handoff. Keep hardware-independent tests separate from hardware-in-the-loop validation. Use recorded-video replay for repeatability.

## Documentation ownership
- `README.md`: project introduction and setup.
- `STATUS.md`: current project state.
- `WORKLOG.md`: session history.
- `research/ARCHITECTURE.md`: system design.
- `research/REPOSITORY_MAP.md`: code navigation.
- `research/ENGINEERING_STANDARDS.md`: engineering and validation standards.
- `research/decisions/`: consequential decisions.
