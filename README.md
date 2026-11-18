# Vision-Based First Contact

A vision-only first-contact subsystem for a stationary tabletop social robot at an office reception. It detects and tracks people, estimates spatial and temporal behaviour, decides first-contact readiness, and emits a structured event to a separate interaction module.

Please refer to [`docs/readme.md`](docs/readme.md) for full project scope, architecture, and documentation.

## Quickstart

```bash
# activate runtime virtual environment
source activate_env.sh

# verify camera stream and fps
python3 tools/calibration/camera_capture_utility.py --mode test
```

## Documentation Map

- [Project Overview & Navigation](docs/readme.md)
- [System Architecture](docs/architecture.md)
- [Project Status](docs/status.md)
- [Engineering Worklog](docs/worklog.md)
- [Incremental Implementation Roadmap](docs/implementation_roadmap.md)
- [Camera Capture User Guide](docs/calibration/camera_capture_guide.md)
- [M0 Hardware Inventory Report](docs/calibration/m0_camera_hardware_inventory.md)
- [Framing & Calibration Report](docs/calibration/m0_framing_and_calibration_report.md)
