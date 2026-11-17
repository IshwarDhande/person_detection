# AI Coding Agent Instructions

These instructions apply to AI coding agents working on the Vision-Based First Contact project.

## 1. Execution principle: focused progress, minimal overhead

- Prioritize the user's requested implementation over administrative work.
- Do not perform a broad repository audit for every task. Inspect only the code, tests, and configuration relevant to the requested change.
- Do not read every governance document on every run. Start with this file and consult only the relevant references listed below.
- Do not create extra reports, plans, summaries, inventories, diagrams, or duplicate documentation unless explicitly requested or essential to the deliverable.
- Avoid repeated scans, redundant commands, unnecessary abstractions, and unrelated refactoring.
- If the task is clear, proceed without asking questions. Ask only when missing information blocks safe or correct implementation.
- Keep scope limited to the user's request.

## 2. Repository and governance references

Project governance/reference documents are maintained under `research/`. Identify the actual repository root; do not assume the current working directory is the root.

For every task:
1. Read this `AGENTS.md`.
2. Inspect `git status --short` before editing.
3. Inspect the target code and nearby tests/configuration.

Read other `research/` documents selectively:
- `research/ARCHITECTURE.md` — only when changing architecture, module boundaries, interfaces, or data flow.
- `research/REPOSITORY_MAP.md` — only when adding/moving files or when file placement is unclear.
- `research/ENGINEERING_STANDARDS.md` — only when coding/testing conventions are unclear from nearby code.
- `research/STATUS.md` and relevant recent `research/WORKLOG.md` entries — only when the task depends on current progress, blockers, or previous attempts.
- `research/round_1.md`, `round_2.md`, `round_3.md` and associated question/decision files — only when changing requirements covered by those rounds.
- `research/project_overview_implementation.md` — only when consolidated requirements or roadmap context is needed.
- `research/CONTRIBUTING.md` — only when changing development workflow, setup, formatting, or Git practices.

Resolve paths relative to the actual repository root. Do not assume proposed directories exist. If relevant documents conflict, report the specific conflict rather than silently choosing.

## 3. Project purpose and scope

This is an engineering project, not a research-study repository. Do not create `research/RESEARCH_PLAN.md`.

The project builds a vision-only first-contact subsystem for a stationary tabletop social robot at an office reception. It detects and tracks people, estimates spatial and temporal behaviour, determines first-contact readiness, selects a target, and sends a structured event to a separate interaction module.

Out of scope:
- Speech recognition/synthesis, dialogue, or conversation generation.
- Gesture generation or robot actuation.
- Emotion recognition.
- Face recognition or persistent biometric identity.
- Robot navigation.

Keep responsibilities separated: perception, spatial understanding, behaviour understanding, decision engine, lifecycle management, integration, and runtime. Keep decision policy independent of detector/tracker implementations. Prefer a modular monolith initially. Do not introduce ROS 2 nodes, microservices, databases, or other infrastructure without a demonstrated requirement.

## 4. Behavioural requirements to preserve

- Readiness requires observed approach, interaction-zone membership, distance within the configurable initial 1.5 m threshold, stationary state, robot-facing orientation, and reliable evidence.
- All required predicates must remain true continuously for 2 seconds.
- Uncertain required evidence must not trigger first contact. HOLD/timer semantics remain unresolved; do not silently decide them.
- Track all visible people and continue background candidate evaluation while a target is locked.
- Select the first-qualified candidate; do not switch merely because another person is closer.
- Candidate queue capacity is five; preserve qualification order and documented tie-break.
- Temporary track-loss reacquisition grace is 5 seconds. This does not terminate an interaction.
- Keep `track_id`, `visit_id`, and `target_id` distinct.
- Interaction release/completion requires explicit lifecycle coordination with the downstream module.
- A person already waiting nearby is not eligible without an observed approach.

These are requirements, not proof of validated performance. Hardware geometry, thresholds, confidence calibration, and operating limits must be measured and validated.

## 5. Implementation, testing, and file placement

- Follow existing code style, package structure, naming, and test patterns.
- Before creating a file, inspect the relevant directory and reuse an existing module/utility where appropriate.
- Use `research/REPOSITORY_MAP.md` only when adding/moving files or when placement is unclear.
- Do not create directories or abstractions unless justified by the requested change.
- Use monotonic timestamps for elapsed durations and deadlines.
- Expose validity/confidence for estimates used by decisions.
- Prefer deterministic unit tests and offline replay before live-hardware tests.
- Add/update tests directly covering changed behavior, especially temporal boundaries, state transitions, queue ordering, reacquisition, and event lifecycle where applicable.
- Run the smallest relevant test set first; run broader tests only when warranted. Report exactly what ran and its result.
- Do not add dependencies unless necessary; explain additions.
- Do not assume unmeasured hardware properties.

## 6. Governance/documentation updates — only when requested

**Do not automatically update governance or project-status files after ordinary implementation tasks.**

Unless the user explicitly asks for governance/documentation updates, leave these files untouched:
- `research/AGENTS.md`
- `research/STATUS.md`
- `research/WORKLOG.md`
- `research/ARCHITECTURE.md`
- `research/REPOSITORY_MAP.md`
- `research/ENGINEERING_STANDARDS.md`
- Requirements, decision records, and roadmap documents under `research/`

Update governance files only when:
1. The user explicitly requests it; or
2. The requested deliverable is itself a governance/documentation change; or
3. The user authorizes synchronization because a source change makes an existing interface/architecture document materially false.

When an update is authorized, edit only the specific relevant document(s). Do not cascade updates across every governance file. Do not create a worklog entry, status update, ADR, report, or summary as a routine completion ritual. Do not create duplicate sources of truth.

## 7. Git and data safety

- Inspect Git status before editing and preserve all pre-existing user changes.
- Never overwrite, discard, or revert unrelated user work.
- Do not commit unless explicitly requested.
- Review the diff for changed files and run `git diff --check` when practical.
- Do not commit private reception video, identifying data, credentials, or large model artifacts without approval.
- Do not install packages, use `sudo`, change system configuration, or terminate processes unless explicitly authorized and necessary.
- Do not capture/save camera images or video unless the user explicitly requests or initiates capture.

## 8. Final response

Keep the final response concise and task-focused. Report:
1. What changed.
2. Exact files created/modified.
3. Relevant tests/commands actually run and outcomes.
4. Any blocker or manual action required.

Do not provide a lengthy repository audit or repeat project documentation unless requested. Never claim validation that was not performed.
