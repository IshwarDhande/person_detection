# Agent Instructions — Vision-Based First Contact

## 1. Work style
- Implement only the milestone or task the user explicitly requests.
- Keep changes focused; avoid broad audits, unrelated refactors, speculative abstractions, and unnecessary reports.
- Inspect the relevant code, tests, configuration, and `git status --short` before editing.
- Preserve existing user changes. Never commit unless explicitly authorized.

## 2. Project references
Project documentation files are under `docs/`. Read this file first, then consult only relevant documents:
- `docs/status.md` — current milestone and blockers.
- `docs/implementation_roadmap.md` — milestone scope and exit criteria.
- `docs/architecture.md` — interfaces, data flow, or architecture changes.
- `docs/worklog.md` and requirement records — only when prior decisions or history matter.

Identify the actual repository root; do not assume proposed paths exist.

## 3. Project scope
Build a vision-based first-contact subsystem for a stationary reception robot. Out of scope: speech/dialogue, gestures/actuation, emotion recognition, face recognition/biometric identity, and navigation. Do not create `docs/research_plan.md`.

## 4. Implementation order
Follow the approved incremental roadmap:
1. M1 Basic person detection.
2. M2 Reliable single-person detection.
3. M3 Single-person tracking.
4. M4 Frame-level readiness perception.
5. M5 Temporal behaviour estimation.
6. M6 Complete single-person pipeline.
7. M7 Multi-person detection and tracking.
8. M8 Multi-person readiness and target selection.
9. M9 Lifecycle and downstream integration.
10. M10 Robot integration and calibration.

Work on one milestone at a time. Stop after it; do not start the next milestone without user authorization. Laptop webcam is for initial development only; its calibration is not robot calibration.

## 5. Engineering requirements
- Reuse existing structure and dependencies; create files/directories only when needed.
- Use timestamps for temporal logic and expose confidence/validity for decision evidence.
- Preserve core requirements: observed approach; 2-second continuous readiness; fail-closed uncertainty; first-qualified selection; queue capacity five; 5-second track reacquisition; distinct `track_id`, `visit_id`, `target_id`; explicit downstream target release.
- Do not treat laptop image-space estimates as validated robot-relative distance.
- Add focused tests. Run the smallest relevant test set and report actual results.
- A milestone is complete only with a working feature, repeatable demo, relevant passing tests, evaluation evidence, and known limitations.

## 6. Governance and safety
- Do not update `STATUS.md`, `WORKLOG.md`, architecture, repository map, or other governance files during ordinary implementation unless the user asks.
- When documentation updates are requested, edit only the relevant files.
- Do not install dependencies, use `sudo`, alter system settings, or capture/save camera media unless authorized.
- Do not delete or overwrite user data. Do not commit.

## 7. Final response
Briefly report what changed, exact files, commands/tests and results, demo/metrics, limitations, and any required manual action. Never claim work or validation that was not performed.
