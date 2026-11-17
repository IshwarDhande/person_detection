# Vision-Based First Contact

A vision-only first-contact subsystem for a stationary tabletop social robot at an office reception. It detects and tracks people, estimates spatial and temporal behaviour, decides first-contact readiness, and emits a structured event to a separate interaction module.

Please refer to [`research/README.md`](research/README.md) for full project scope, architecture, and governance documentation.

## Quickstart

```bash
# activate runtime virtual environment
source activate_env.sh

# verify camera stream and fps
python3 tools/calibration/camera_capture_utility.py --mode test
```

## Documentation Map

- [Project Overview & Navigation](research/README.md)
- [System Architecture](research/ARCHITECTURE.md)
- [Repository Map](research/REPOSITORY_MAP.md)
- [Project Status](research/STATUS.md)
- [Engineering Worklog](research/WORKLOG.md)
- [Camera Capture User Guide](docs/calibration/camera_capture_guide.md)
- [M0 Hardware Inventory Report](docs/calibration/m0_camera_hardware_inventory.md)
- [Framing & Calibration Report](docs/calibration/m0_framing_and_calibration_report.md)
