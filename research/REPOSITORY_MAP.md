---
title: Repository Map
created: 2026-11-16
last_updated: 2026-11-17
status: active
---

# Repository Map

This document reflects the actual active filesystem structure. Planned subsystem packages are introduced just-in-time as their implementations and tests are added, rather than pre-creating empty skeleton directories.

| Directory | Responsibility | Active Contents |
|---|---|---|
| `configs/runtime/` | Python runtime and environment specifications | `environment.yaml` |
| `data/raw/calibration/` | Raw camera calibration frames and scene inspection baselines | `calibration_metadata.json`, `scene_inspection/scene_empty.png` |
| `docs/calibration/` | Camera hardware survey, framing reports, capture guides, and inventories | `camera_capture_guide.md`, `m0_camera_hardware_inventory.md`, `m0_camera_hardware_inventory.json`, `m0_framing_and_calibration_report.md` |
| `research/` | Requirements, architecture specifications, engineering standards, and governance records | `AGENTS.md`, `README.md`, `STATUS.md`, `WORKLOG.md`, `ARCHITECTURE.md`, `round_*.md`, etc. |
| `tools/calibration/` | Hardware discovery and calibration image acquisition utilities | `camera_hardware_discovery.py`, `camera_capture_utility.py` |

## Intended architectural layout (to be created as implemented)
When pipeline modules are implemented, they will be organized under the following architectural boundaries (see `research/ARCHITECTURE.md`):
- `perception/`: Camera adapters, person detection, multi-person tracking, pose estimation.
- `spatial_understanding/`: Camera calibration, geometric mapping, distance estimation, interaction zone.
- `behaviour_understanding/`: Approach detection, motion/stationarity estimation, body orientation.
- `decision_engine/`: Readiness evaluation, candidate queue, target selection, temporal logic.
- `lifecycle_management/`: Track lifecycle, target lifecycle, visit state, recovery management.
- `integration/`: Event system, interaction module interface, messaging, robot middleware.
- `runtime/`: Pipeline assembly, orchestration, scheduling, monitoring.
- `evaluation/`, `experiments/`, `tests/`: Subsystem benchmarking, experiments, and unit/integration test suites.
- `checkpoints/`, `logs/`, `deployment/`: Model weights, execution logs, and robot/container deployment manifests.

## Entry points
Verified and implemented entry points:
- Calibration / Hardware discovery: `tools/calibration/camera_hardware_discovery.py`
- Calibration capture utility: `tools/calibration/camera_capture_utility.py`
- Environment activation: `activate_env.sh`

## Maintenance
Keep raw and processed data separate; configuration outside source; model artifacts separate from source; no credentials or private video in Git.
