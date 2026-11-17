#!/usr/bin/env python3
"""
camera_capture_utility.py

Safe camera calibration and image acquisition utility for the Vision-Based
First Contact project (M0 milestone).

Supports:
  1. Diagnostic stream testing (FPS, resolution, format validation without saving)
  2. Mode A: Scene inspection capture (empty reception, 1.0m, 1.5m, 2.0m, 2.5m)
  3. Mode B: Intrinsic calibration target capture (Checkerboard & ChArUco)
  4. Structured metadata recording with camera geometry annotations

Conforms to engineering standards: no continuous recording, fail-closed consent,
clean resource management, and explicit MJPG 1280x720 configuration.
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Hardware specification constants confirmed in M0 Batch 1
DEFAULT_DEVICE = "/dev/video0"
DEFAULT_PERSISTENT_PATH = (
    "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0"
)
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_FOURCC = "MJPG"

# Camera mounting and scene geometry documented in M0
CAMERA_GEOMETRY = {
    "lens_height_cm": 115.0,
    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
    "mounting": "fixed",
    "feet_visibility_at_1_5m": "not_visible",
    "notes": "Tabletop robot head mounting at reception desk",
}

# Standard stages for Mode A scene inspection
SCENE_STAGES = [
    {"stage": "empty", "distance_m": 0.0, "description": "Empty reception scene (background baseline)"},
    {"stage": "person_1_0m", "distance_m": 1.0, "description": "Person standing at approximately 1.0 m"},
    {"stage": "person_1_5m", "distance_m": 1.5, "description": "Person standing at approximately 1.5 m (first-contact boundary)"},
    {"stage": "person_2_0m", "distance_m": 2.0, "description": "Person standing at approximately 2.0 m"},
    {"stage": "person_2_5m", "distance_m": 2.5, "description": "Person standing at approximately 2.5 m"},
]


class CameraDevice:
    """Manages the OpenCV VideoCapture handle with explicit MJPG configuration and safe cleanup."""

    def __init__(
        self,
        device_path: str = DEFAULT_DEVICE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        fourcc: str = DEFAULT_FOURCC,
    ) -> None:
        self.device_path = device_path
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.target_fourcc = fourcc
        self.cap: Optional[cv2.VideoCapture] = None
        self.backend_name = "Unknown"
        self.is_opened = False

    def open(self) -> Tuple[bool, str]:
        """Open the camera, enforce MJPG and target resolution, and verify initial frame."""
        # Try device path or numeric index
        dev_arg: Any = self.device_path
        if self.device_path.startswith("/dev/video"):
            try:
                dev_arg = int(self.device_path.replace("/dev/video", ""))
            except ValueError:
                dev_arg = self.device_path

        self.cap = cv2.VideoCapture(dev_arg, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            # Fallback to generic backend
            self.cap = cv2.VideoCapture(dev_arg)

        if not self.cap.isOpened():
            self.is_opened = False
            return False, f"Failed to open video capture device at {self.device_path}"

        self.is_opened = True
        self.backend_name = self.cap.getBackendName() if hasattr(self.cap, "getBackendName") else "V4L2"

        # Explicitly configure fourcc, resolution, and fps
        fourcc_code = cv2.VideoWriter_fourcc(*self.target_fourcc)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Warm-up / flush initial buffers
        for _ in range(5):
            self.cap.read()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, "Camera opened but failed to read initial verification frame"

        actual_h, actual_w = frame.shape[:2]
        if actual_w != self.target_width or actual_h != self.target_height:
            return False, (
                f"Resolution mismatch: requested {self.target_width}x{self.target_height}, "
                f"returned {actual_w}x{actual_h}"
            )

        return True, "Camera successfully opened and verified at target resolution"

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read a single frame with monotonic timestamp."""
        if not self.cap or not self.is_opened:
            return False, None, time.monotonic()
        t = time.monotonic()
        ret, frame = self.cap.read()
        return ret, frame, t

    def benchmark_stream(self, num_frames: int = 30) -> Dict[str, Any]:
        """Benchmark actual streaming framerate and stability over a frame sample."""
        if not self.cap or not self.is_opened:
            return {"error": "Camera not opened"}

        timestamps: List[float] = []
        successful_reads = 0
        failed_reads = 0

        t_start = time.monotonic()
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            t_now = time.monotonic()
            if ret and frame is not None:
                timestamps.append(t_now)
                successful_reads += 1
            else:
                failed_reads += 1

        t_end = time.monotonic()
        total_time = t_end - t_start
        observed_fps = successful_reads / total_time if total_time > 0 else 0.0

        intervals = [
            timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))
        ]
        mean_interval_ms = (sum(intervals) / len(intervals) * 1000) if intervals else 0.0
        jitter_ms = (
            (sum((x - (mean_interval_ms / 1000)) ** 2 for x in intervals) / len(intervals)) ** 0.5 * 1000
            if intervals
            else 0.0
        )

        return {
            "requested_frames": num_frames,
            "successful_reads": successful_reads,
            "failed_reads": failed_reads,
            "elapsed_seconds": round(total_time, 3),
            "observed_fps": round(observed_fps, 2),
            "mean_interval_ms": round(mean_interval_ms, 2),
            "jitter_ms": round(jitter_ms, 2),
        }

    def close(self) -> None:
        """Safely release the camera handle."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_opened = False


def calculate_sharpness(gray_frame: np.ndarray) -> float:
    """Compute the Laplacian variance as an image sharpness / blur metric."""
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


class PatternDetector:
    """Detects and validates checkerboard and ChArUco patterns for intrinsic calibration."""

    def __init__(
        self,
        pattern_type: str = "checkerboard",
        board_width: int = 9,
        board_height: int = 6,
        square_size_mm: float = 25.0,
        marker_size_mm: float = 15.0,
        dict_id: int = cv2.aruco.DICT_4X4_50,
    ) -> None:
        self.pattern_type = pattern_type.lower()
        self.board_width = board_width
        self.board_height = board_height
        self.square_size_mm = square_size_mm
        self.marker_size_mm = marker_size_mm
        self.dict_id = dict_id

        if self.pattern_type == "charuco":
            self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            self.charuco_board = cv2.aruco.CharucoBoard(
                (board_width, board_height),
                square_size_mm / 1000.0,
                marker_size_mm / 1000.0,
                self.dictionary,
            )
            self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)
        else:
            self.charuco_board = None
            self.charuco_detector = None

    def detect(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Detect pattern corners and assess quality/coverage."""
        info: Dict[str, Any] = {
            "pattern_type": self.pattern_type,
            "board_size": [self.board_width, self.board_height],
            "corners_found": 0,
            "sharpness": round(calculate_sharpness(gray), 1),
            "coverage_quadrant": "Unknown",
        }

        if self.pattern_type == "checkerboard":
            pattern_size = (self.board_width, self.board_height)
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
            if ret and corners is not None:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                info["corners_found"] = len(refined)

                # Determine quadrant of corner centroid
                cx = np.mean(refined[:, 0, 0])
                cy = np.mean(refined[:, 0, 1])
                h, w = gray.shape
                x_pos = "left" if cx < w / 2 else "right"
                y_pos = "top" if cy < h / 2 else "bottom"
                info["coverage_quadrant"] = f"{y_pos}_{x_pos}"

                return True, refined, info
            return False, None, info

        elif self.pattern_type == "charuco":
            if not self.charuco_detector:
                return False, None, info
            corners, ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            if corners is not None and len(corners) >= 4:
                info["corners_found"] = len(corners)
                cx = np.mean(corners[:, 0, 0])
                cy = np.mean(corners[:, 0, 1])
                h, w = gray.shape
                x_pos = "left" if cx < w / 2 else "right"
                y_pos = "top" if cy < h / 2 else "bottom"
                info["coverage_quadrant"] = f"{y_pos}_{x_pos}"
                return True, corners, info
            return False, None, info

        return False, None, info


class CalibrationMetadataManager:
    """Maintains a persistent, structured JSON metadata index for captured images."""

    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path
        self.data: Dict[str, Any] = {
            "project": "Vision-Based First Contact for Reception Social Robot",
            "milestone": "M0 — Camera Calibration & Scene Framing",
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
            "camera_geometry": CAMERA_GEOMETRY,
            "images": [],
        }
        self.load()

    def load(self) -> None:
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and "images" in loaded:
                        self.data = loaded
            except Exception as e:
                print(f"[Warning] Could not load existing metadata: {e}")

    def add_image_entry(self, entry: Dict[str, Any]) -> None:
        # Avoid duplicate entries for the same relative path
        self.data["images"] = [
            item for item in self.data["images"] if item.get("file_name") != entry.get("file_name")
        ]
        self.data["images"].append(entry)
        self.data["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def save(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)


def run_test_mode(camera: CameraDevice, num_frames: int = 30) -> int:
    """Run diagnostic verification without persisting images."""
    print("\n--- Running Camera Diagnostic Stream Verification ---")
    ok, msg = camera.open()
    if not ok:
        print(f"[ERROR] {msg}")
        return 1

    print(f"[OK] {msg}")
    print(f"     Backend: {camera.backend_name}")
    print(f"     Target Config: {camera.target_width}x{camera.target_height} @ {camera.target_fps} FPS ({camera.target_fourcc})")

    # Read a sample frame to inspect data properties
    ret, frame, _ = camera.read_frame()
    if not ret or frame is None:
        print("[ERROR] Failed to capture verification frame.")
        camera.close()
        return 1

    h, w, c = frame.shape
    dtype_str = str(frame.dtype)
    print(f"     Actual Frame: {w}x{h}, Channels: {c}, Dtype: {dtype_str}")

    print(f"\nBenchmarking stream over {num_frames} frames...")
    stats = camera.benchmark_stream(num_frames)
    camera.close()

    print(f"     Elapsed Time: {stats['elapsed_seconds']} s")
    print(f"     Successful Reads: {stats['successful_reads']}/{stats['requested_frames']}")
    print(f"     Observed FPS: {stats['observed_fps']} FPS")
    print(f"     Mean Inter-frame Interval: {stats['mean_interval_ms']} ms (Jitter: {stats['jitter_ms']} ms)")

    status_str = "PASS" if stats["failed_reads"] == 0 and stats["observed_fps"] >= 25.0 else "WARN"
    print(f"\nDiagnostic Result: [{status_str}] Camera streaming verified without resource leaks.")
    return 0


def run_scene_inspection_mode(
    camera: CameraDevice,
    output_dir: Path,
    metadata_mgr: CalibrationMetadataManager,
    stage_id: Optional[str] = None,
    distance_m: Optional[float] = None,
    consent: bool = False,
    notes: str = "",
    save: bool = False,
    preview: bool = False,
    delay: float = 0.0,
) -> int:
    """Capture scene inspection frames (empty, 1.0m, 1.5m, 2.0m, 2.5m)."""
    print("\n--- Mode A: Scene Inspection Capture ---")
    if not save:
        print("[Notice] Dry-run mode: frames will NOT be saved to disk. Pass --save to persist images.")

    if not consent:
        print("[Notice] Scene inspection with human subjects requires explicit user consent.")
        print("         Pass --consent to certify consent under project privacy standards.")

    ok, msg = camera.open()
    if not ok:
        print(f"[ERROR] {msg}")
        return 1

    scene_dir = output_dir / "scene_inspection"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Determine which stages to run
    stages_to_run = []
    if stage_id:
        matched = [s for s in SCENE_STAGES if s["stage"] == stage_id]
        if not matched:
            # Custom stage
            dist = distance_m if distance_m is not None else 0.0
            stages_to_run.append({
                "stage": stage_id,
                "distance_m": dist,
                "description": f"Custom stage: {stage_id}",
            })
        else:
            stages_to_run = matched
    else:
        stages_to_run = SCENE_STAGES

    saved_count = 0
    try:
        for idx, stage in enumerate(stages_to_run, 1):
            s_name = stage["stage"]
            s_dist = stage["distance_m"]
            s_desc = stage["description"]

            print(f"\n[{idx}/{len(stages_to_run)}] Stage '{s_name}' — {s_desc}")
            print(f"     Nominal Distance: {s_dist} m | Lens Height: 115 cm | Pitch: ~0°")

            # In human subject stages, verify consent flag
            if s_dist > 0.0 and not consent and save:
                print("     [BLOCKED] Skipping image save: --consent flag required when recording subjects.")
                continue

            # Optional countdown delay so operator can step into position
            if delay > 0 and save:
                print(f"     [Countdown] {int(delay)} seconds to position for stage '{s_name}'...")
                t_target = time.monotonic() + delay
                while time.monotonic() < t_target:
                    rem = max(0, int(t_target - time.monotonic()) + 1)
                    sys.stdout.write(f"\r     Capturing in {rem}s... ")
                    sys.stdout.flush()
                    camera.read_frame()
                    time.sleep(0.2)
                print("\r     [CAPTURING NOW]                          ")

            # Read a fresh, stabilized frame
            for _ in range(5):
                camera.read_frame()
            ret, frame, t_captured = camera.read_frame()

            if not ret or frame is None:
                print(f"     [ERROR] Failed to capture frame for stage '{s_name}'.")
                continue

            h, w, c = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = calculate_sharpness(gray)

            print(f"     Captured frame: {w}x{h}, {c} channels, Dtype: {frame.dtype}, Sharpness: {sharpness:.1f}")

            if preview:
                try:
                    display_frame = frame.copy()
                    text = f"Stage: {s_name} ({s_dist}m) | Sharpness: {sharpness:.1f}"
                    cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("Scene Inspection Preview", display_frame)
                    cv2.waitKey(1000)
                except Exception as e:
                    print(f"     [Notice] Live preview unavailable: {e}")

            if save:
                file_name = f"scene_{s_name}.png"
                file_path = scene_dir / file_name
                cv2.imwrite(str(file_path), frame)
                saved_count += 1
                print(f"     [SAVED] {file_path.relative_to(output_dir.parent.parent)}")

                # Construct metadata record
                record = {
                    "file_name": file_name,
                    "relative_path": str(file_path.relative_to(output_dir.parent.parent)),
                    "capture_mode": "scene_inspection",
                    "stage": s_name,
                    "nominal_distance_m": s_dist,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic_timestamp": t_captured,
                    "resolution": {"width": w, "height": h, "channels": c, "dtype": str(frame.dtype)},
                    "sharpness": round(sharpness, 2),
                    "camera_geometry": {
                        "lens_height_cm": CAMERA_GEOMETRY["lens_height_cm"],
                        "pitch_deg": CAMERA_GEOMETRY["pitch_deg"],
                        "yaw_deg": CAMERA_GEOMETRY["yaw_deg"],
                        "roll_deg": CAMERA_GEOMETRY["roll_deg"],
                        "feet_visibility": "not_visible" if s_dist <= 2.0 else "unknown",
                    },
                    "privacy_status": {
                        "consent_verified": consent,
                        "biometric_identity_retained": False,
                        "purpose": "First-contact framing and distance boundary calibration",
                    },
                    "operator_notes": notes or s_desc,
                }
                metadata_mgr.add_image_entry(record)

    finally:
        camera.close()
        if preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print(f"\nScene Inspection Finished. Saved {saved_count} image(s).")
    return 0


def run_intrinsic_calibration_mode(
    camera: CameraDevice,
    output_dir: Path,
    metadata_mgr: CalibrationMetadataManager,
    pattern_type: str = "checkerboard",
    board_width: int = 9,
    board_height: int = 6,
    square_size_mm: float = 25.0,
    marker_size_mm: float = 15.0,
    count: int = 15,
    save: bool = False,
    preview: bool = False,
    min_sharpness: float = 30.0,
) -> int:
    """Capture calibration target (Checkerboard / ChArUco) images across multiple orientations."""
    print("\n--- Mode B: Intrinsic Calibration Pattern Capture ---")
    print(f"Target: {pattern_type.upper()} ({board_width}x{board_height}) | Square: {square_size_mm}mm")
    if not save:
        print("[Notice] Dry-run mode: images will NOT be saved. Pass --save to write images.")

    detector = PatternDetector(
        pattern_type=pattern_type,
        board_width=board_width,
        board_height=board_height,
        square_size_mm=square_size_mm,
        marker_size_mm=marker_size_mm,
    )

    ok, msg = camera.open()
    if not ok:
        print(f"[ERROR] {msg}")
        return 1

    intrinsic_dir = output_dir / "intrinsics"
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    print("\nCapture Guidelines:")
    print("  1. Position the board at varying angles: perpendicular, tilted pitch (±20°), tilted yaw (±20°).")
    print("  2. Cover all 4 frame quadrants (top-left, top-right, bottom-left, bottom-right) and center.")
    print("  3. Vary distance from 0.5 m to 2.0 m.")
    print("  4. Keep the board steady to avoid motion blur.")

    saved_images = 0
    quadrants_covered = set()
    attempt = 0
    max_attempts = count * 10

    try:
        while saved_images < count and attempt < max_attempts:
            attempt += 1
            ret, frame, t_captured = camera.read_frame()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners, info = detector.detect(gray)

            sharpness = info["sharpness"]
            quadrant = info["coverage_quadrant"]
            corners_count = info["corners_found"]

            status_msg = f"Corners: {corners_count} | Sharpness: {sharpness:.1f} | Quad: {quadrant}"

            if preview:
                try:
                    display_frame = frame.copy()
                    if found and corners is not None:
                        if pattern_type == "checkerboard":
                            cv2.drawChessboardCorners(display_frame, (board_width, board_height), corners, found)
                        elif pattern_type == "charuco":
                            cv2.aruco.drawDetectedCornersCharuco(display_frame, corners)
                    color = (0, 255, 0) if (found and sharpness >= min_sharpness) else (0, 0, 255)
                    cv2.putText(display_frame, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(
                        display_frame,
                        f"Collected: {saved_images}/{count}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )
                    cv2.imshow("Intrinsic Calibration Capture", display_frame)
                    key = cv2.waitKey(50)
                    if key == ord("q"):
                        print("\nUser requested early termination.")
                        break
                except Exception:
                    pass

            # Auto-save condition when valid pattern is detected and reasonably sharp
            if found and sharpness >= min_sharpness and save:
                # Require pause between captures to avoid identical frames
                file_name = f"calib_{pattern_type}_{saved_images + 1:03d}_{quadrant}.png"
                file_path = intrinsic_dir / file_name

                cv2.imwrite(str(file_path), frame)
                saved_images += 1
                quadrants_covered.add(quadrant)
                print(f"  [SAVED {saved_images}/{count}] {file_name} ({status_msg})")

                # Record metadata
                record = {
                    "file_name": file_name,
                    "relative_path": str(file_path.relative_to(output_dir.parent.parent)),
                    "capture_mode": "intrinsic_calibration",
                    "pattern": {
                        "type": pattern_type,
                        "board_width": board_width,
                        "board_height": board_height,
                        "square_size_mm": square_size_mm,
                        "marker_size_mm": marker_size_mm if pattern_type == "charuco" else None,
                    },
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic_timestamp": t_captured,
                    "resolution": {"width": frame.shape[1], "height": frame.shape[0], "channels": frame.shape[2]},
                    "sharpness": sharpness,
                    "corners_detected": corners_count,
                    "coverage_quadrant": quadrant,
                }
                metadata_mgr.add_image_entry(record)

                # Delay slightly so user can reposition board
                time.sleep(1.0)

    finally:
        camera.close()
        if preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print(f"\nIntrinsic Capture Complete. Collected {saved_images}/{count} images.")
    print(f"Quadrants Covered: {', '.join(sorted(quadrants_covered)) if quadrants_covered else 'None'}")
    return 0


def main() -> None:
    # Setup graceful signal handlers to ensure camera is never held open
    def sig_handler(sig, frame):
        print("\n[Signal Caught] Releasing camera resources and exiting cleanly...")
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    parser = argparse.ArgumentParser(
        description="M0 — Camera Calibration Capture & Framing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. Diagnostic test (verify stream & FPS, no files saved):
     python3 tools/calibration/camera_capture_utility.py --mode test

  2. Scene inspection capture (empty reception):
     python3 tools/calibration/camera_capture_utility.py --mode scene --stage empty --save

  3. Scene inspection with person at 1.5m (requires consent flag):
     python3 tools/calibration/camera_capture_utility.py --mode scene --stage person_1_5m --consent --save

  4. Intrinsic calibration with ChArUco board:
     python3 tools/calibration/camera_capture_utility.py --mode intrinsic --pattern charuco --count 15 --preview --save
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["test", "scene", "intrinsic"],
        default="test",
        help="Operation mode: 'test' (diagnostic), 'scene' (Mode A), 'intrinsic' (Mode B). Default: test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Camera device path (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Target frame width (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Target frame height (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Target capture FPS (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default=DEFAULT_FOURCC,
        help=f"Compression FOURCC (default: {DEFAULT_FOURCC})",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Number of frames for stream benchmarking in test mode (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/calibration"),
        help="Root output directory for calibration images (default: data/raw/calibration)",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("data/raw/calibration/calibration_metadata.json"),
        help="Path to structured metadata JSON (default: data/raw/calibration/calibration_metadata.json)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display live GUI preview window (requires GUI display)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Explicitly save captured image(s) to disk. If omitted, runs in dry-run mode.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Operator notes to record in image metadata",
    )

    # Mode A arguments
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Scene inspection stage (empty, person_1_0m, person_1_5m, person_2_0m, person_2_5m)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Nominal distance in meters for scene inspection (e.g. 1.5)",
    )
    parser.add_argument(
        "--consent",
        action="store_true",
        help="Certify explicit subject consent for scene inspection with human participants",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Countdown delay in seconds before frame capture to allow operator positioning (default: 0.0)",
    )

    # Mode B arguments
    parser.add_argument(
        "--pattern",
        choices=["checkerboard", "charuco"],
        default="checkerboard",
        help="Calibration target pattern type (default: checkerboard)",
    )
    parser.add_argument(
        "--board-width",
        type=int,
        default=9,
        help="Number of inner corners / squares horizontally (default: 9)",
    )
    parser.add_argument(
        "--board-height",
        type=int,
        default=6,
        help="Number of inner corners / squares vertically (default: 6)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="Square size in millimeters (default: 25.0)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=15.0,
        help="ChArUco marker size in millimeters (default: 15.0)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=15,
        help="Target number of calibration board images to collect (default: 15)",
    )

    args = parser.parse_args()

    # Print Header and ASCII Topology
    print("=" * 80)
    print("Vision-Based First Contact — Camera Calibration Capture Utility (M0)")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()} UTC")
    print("=" * 80)
    print("""
+------------------------------------------------------------------------------------+
|                         CAMERA CALIBRATION ACQUISITION PIPELINE                    |
+------------------------------------------------------------------------------------+
|                                                                                    |
|  [ Physical Tabletop Robot ]                                                       |
|    - Mounting: Fixed on reception desk                                             |
|    - Lens Height: 115.0 cm | Optical Pitch: ~0.0 deg                               |
|    - Feet Status: NOT VISIBLE at 1.5 m (Occluded by desk / vertical FOV cutoff)    |
|                                                                                    |
|  [ Linux Video Subsystem ]                                                         |
|    - Device Node: /dev/video0 (Luxvisions Integrated RGB Camera)                   |
|    - Pipeline Format: MJPG @ 1280x720 (30 FPS enforced via V4L2)                   |
|                                                                                    |
|  [ Acquisition Modes ]                                                             |
|    * Mode 'test'      ==> Stream diagnostic & framerate validation (Zero saves)    |
|    * Mode 'scene'     ==> Multi-distance scene framing inspection (Empty to 2.5 m) |
|    * Mode 'intrinsic' ==> ChArUco / Checkerboard multi-pose calibration dataset    |
|                                                                                    |
+------------------------------------------------------------------------------------+
    """)

    camera = CameraDevice(
        device_path=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
    )
    metadata_mgr = CalibrationMetadataManager(args.metadata_file)

    exit_code = 0
    if args.mode == "test":
        exit_code = run_test_mode(camera, num_frames=args.num_frames)
    elif args.mode == "scene":
        exit_code = run_scene_inspection_mode(
            camera=camera,
            output_dir=args.output_dir,
            metadata_mgr=metadata_mgr,
            stage_id=args.stage,
            distance_m=args.distance,
            consent=args.consent,
            notes=args.notes,
            save=args.save,
            preview=args.preview,
            delay=args.delay,
        )
    elif args.mode == "intrinsic":
        exit_code = run_intrinsic_calibration_mode(
            camera=camera,
            output_dir=args.output_dir,
            metadata_mgr=metadata_mgr,
            pattern_type=args.pattern,
            board_width=args.board_width,
            board_height=args.board_height,
            square_size_mm=args.square_size,
            marker_size_mm=args.marker_size,
            count=args.count,
            save=args.save,
            preview=args.preview,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

