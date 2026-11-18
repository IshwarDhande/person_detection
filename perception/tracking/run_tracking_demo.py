"""Standalone Live Person Tracking Demonstration Runner for Milestone M3.

This utility captures live video from a configurable camera device (default: /dev/video0),
runs YOLOv8n person detection, tracks a single person using an 8-D constant-velocity Kalman
filter with IoU and prediction-uncertainty gating, and renders minimal real-time tracking
overlays according to Milestone M3 specification.

Architecture & Execution Flow:
+-----------------------------------------------------------------------------+
|                      M3 LIVE TRACKING RUNNER PIPELINE                       |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [Camera: /dev/video0]                                                      |
|       |                                                                     |
|       | V4L2 MJPG Capture (1280x720 @ 30 FPS requested)                     |
|       v                                                                     |
|  [Frame Acquisition] -------------------------------+                       |
|       |                                             |                       |
|       v (perf_counter start)                        | (Loop Timestamps)     |
|  [PersonDetector.detect_raw()]                      v                       |
|       |                                     [Application FPS]               |
|       v Raw Person Detections                       |                       |
|  [SinglePersonTracker.update(dets, t_mono)]         |                       |
|       |                                             |                       |
|       v Active Track / Predicted Box                |                       |
|  [Visual Rendering] <-------------------------------+                       |
|       | - Observed Confirmed: Green bounding box, "ID: 1 [CONFIRMED]"       |
|       | - Observed Tentative: Amber bounding box, "ID: 1 [TENTATIVE] (1/3)" |
|       | - Unobserved / Coast: Magenta box, "ID: 1 [LOST/COAST] [PREDICTED]" |
|       | - Minimal HUD: FPS, Latency, Track ID, Lifecycle, Miss Duration     |
|       v                                                                     |
|  [cv2.imshow / Key Poll] --> Clean exit on 'q', 'Esc', or window close      |
|       |                                                                     |
|       v (finally)                                                           |
|  [Resource Release & Metrics Summary]                                       |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional, Tuple

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from perception.detection.person_detector import (
    PersonDetection,
    PersonDetector,
)
from perception.tracking.single_person_tracker import (
    SinglePersonTracker,
    Track,
    TrackState,
)

DEFAULT_CAMERA_DEVICE = "/dev/video0"
DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = SinglePersonTracker.DEFAULT_IOU_THRESHOLD
DEFAULT_RETENTION_TIMEOUT = SinglePersonTracker.DEFAULT_RETENTION_TIMEOUT
DEFAULT_CAPTURE_WIDTH = 1280
DEFAULT_CAPTURE_HEIGHT = 720
DEFAULT_CAPTURE_FPS = 30
DEFAULT_FOURCC = "MJPG"
WINDOW_NAME = "Vision-Based First Contact — M3 Single-Person Tracking"

# Distinct Color Scheme (BGR)
COLOR_CONFIRMED = (0, 255, 0)     # Bright Green for verified real observations
COLOR_TENTATIVE = (0, 215, 255)   # Amber / Gold for tentative tracks
COLOR_PREDICTED = (255, 0, 255)   # Magenta / Purple for predicted/coasting boxes
COLOR_TEXT_BG = (20, 20, 20)      # Dark grey backing for text readability
COLOR_TEXT_FG = (255, 255, 255)   # White text


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate command-line arguments for the tracking demo runner.

    Args:
        args: Optional list of CLI arguments (defaults to sys.argv[1:]).

    Returns:
        Validated argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="M3 Live Single-Person Tracking Demonstration Runner (YOLOv8n + Kalman /dev/video0)"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA_DEVICE,
        help=f"Camera device path or numeric index (default: {DEFAULT_CAMERA_DEVICE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to YOLOv8 weights file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for person detection in [0.0, 1.0] (default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU association threshold in (0.0, 1.0] (default: {DEFAULT_IOU_THRESHOLD})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_RETENTION_TIMEOUT,
        help=f"Track retention timeout in seconds during loss (default: {DEFAULT_RETENTION_TIMEOUT}s)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_CAPTURE_WIDTH,
        help=f"Requested camera capture width in pixels (default: {DEFAULT_CAPTURE_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_CAPTURE_HEIGHT,
        help=f"Requested camera capture height in pixels (default: {DEFAULT_CAPTURE_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_CAPTURE_FPS,
        help=f"Requested camera stream framerate (default: {DEFAULT_CAPTURE_FPS})",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default=DEFAULT_FOURCC,
        help=f"FourCC pixel format code (default: {DEFAULT_FOURCC})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for YOLO model (e.g., 'cpu', 'cuda:0', default: auto)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before exiting",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying a visual GUI window (for benchmarking and smoke profiling)",
    )

    parsed = parser.parse_args(args)

    # Validate parameters
    if not (0.0 <= parsed.conf <= 1.0):
        parser.error(f"--conf must be between 0.0 and 1.0, got {parsed.conf}")
    if not (0.0 < parsed.iou_thresh <= 1.0):
        parser.error(f"--iou-thresh must be in (0.0, 1.0], got {parsed.iou_thresh}")
    if parsed.timeout <= 0.0:
        parser.error(f"--timeout must be a positive float, got {parsed.timeout}")
    if parsed.width <= 0:
        parser.error(f"--width must be positive, got {parsed.width}")
    if parsed.height <= 0:
        parser.error(f"--height must be positive, got {parsed.height}")
    if parsed.fps <= 0:
        parser.error(f"--fps must be positive, got {parsed.fps}")
    if parsed.max_frames is not None and parsed.max_frames <= 0:
        parser.error(f"--max-frames must be positive, got {parsed.max_frames}")

    return parsed


def initialize_camera(
    device: str,
    width: int,
    height: int,
    fps: int,
    fourcc: str,
) -> tuple[cv2.VideoCapture, int, int, float]:
    """Open and configure the video capture stream."""
    if device.isdigit():
        cap_index = int(device)
        cap = cv2.VideoCapture(cap_index, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(int(device) if device.isdigit() else device)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera device '{device}'. Verify that the camera is connected, "
            f"not in use by another process, and accessible under permissions."
        )

    if len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))  # type: ignore[attr-defined]

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        cap.release()
        raise RuntimeError(
            f"Camera opened successfully on '{device}', but initial frame read failed. "
            f"Verify stream compatibility and lighting."
        )

    return cap, actual_w, actual_h, actual_fps


def compute_application_fps(
    timestamps: deque[float] | list[float],
    frame_count: Optional[int] = None,
) -> float:
    """Calculate steady-state application throughput in FPS from monotonic timestamps."""
    n = len(timestamps) if frame_count is None else frame_count
    if len(timestamps) < 2 or n < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0.0:
        return 0.0
    return float((n - 1) / elapsed)


def render_tracking(
    frame: np.ndarray,
    track: Optional[Track],
) -> np.ndarray:
    """Render minimal bounding-box and temporary ID overlays conforming to D11.

    Display elements:
        - Bounding box.
        - Temporary track ID.
        - Lifecycle state (TENTATIVE, CONFIRMED, or LOST / COASTING).
        - Explicit [PREDICTED] marker when unobserved to avoid mistaking prediction for detection.
    """
    if track is None or track.state == TrackState.DELETED:
        return frame

    x1, y1, x2, y2 = [int(round(v)) for v in track.bbox_xyxy]
    h_img, w_img = frame.shape[:2]
    x1 = max(0, min(w_img - 1, x1))
    x2 = max(0, min(w_img - 1, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(0, min(h_img - 1, y2))

    if track.observed:
        if track.state == TrackState.CONFIRMED:
            color = COLOR_CONFIRMED
            label = f"ID: {track.track_id} [CONFIRMED]"
        else:
            color = COLOR_TENTATIVE
            label = f"ID: {track.track_id} [TENTATIVE] ({track.hit_count}/3)"
        thickness = 2
    else:
        # Predicted box during LOST / COASTING
        color = COLOR_PREDICTED
        label = f"ID: {track.track_id} [LOST/COAST] [PREDICTED] ({track.miss_duration:.1f}s)"
        thickness = 1

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Render label with background pill
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    label_y1 = max(0, y1 - th - baseline - 4)
    label_y2 = y1
    label_x2 = min(w_img - 1, x1 + tw + 6)

    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), COLOR_TEXT_BG, -1)
    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, 1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, label_y2 - baseline - 2),
        font,
        font_scale,
        color if not track.observed else COLOR_TEXT_FG,
        text_thickness,
        cv2.LINE_AA,
    )

    return frame


def render_hud(
    frame: np.ndarray,
    app_fps: float,
    inference_ms: float,
    track: Optional[Track],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Render a clean, minimal HUD diagnostic overlay in the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    color_white = (255, 255, 255)
    color_cyan = (255, 255, 0)
    color_green = (0, 255, 0)
    color_amber = (0, 215, 255)
    color_magenta = (255, 0, 255)
    line_h = 22
    y_start = 32

    # Line 1: Header
    cv2.putText(frame, "M3 SINGLE-PERSON TRACKER", (20, y_start), font, 0.50, color_cyan, 1, cv2.LINE_AA)

    # Line 2: Throughput & Inference Latency
    fps_text = f"Loop Throughput: {app_fps:5.1f} FPS"
    lat_text = f"YOLO Latency: {inference_ms:5.1f} ms"
    cv2.putText(frame, f"{fps_text} | {lat_text}", (20, y_start + line_h), font, font_scale, color_white, 1, cv2.LINE_AA)

    # Line 3: Stream Resolution
    res_text = f"Resolution: {frame_width}x{frame_height} (MJPG)"
    cv2.putText(frame, res_text, (20, y_start + line_h * 2), font, font_scale, (180, 180, 180), 1, cv2.LINE_AA)

    # Line 4: Tracking State
    if track is None or track.state == TrackState.DELETED:
        track_status = "No Active Track (searching)"
        status_color = (180, 180, 180)
    elif track.observed:
        if track.state == TrackState.CONFIRMED:
            track_status = f"Track #{track.track_id}: CONFIRMED (obs)"
            status_color = color_green
        else:
            track_status = f"Track #{track.track_id}: TENTATIVE (hits: {track.hit_count}/3)"
            status_color = color_amber
    else:
        track_status = f"Track #{track.track_id}: LOST / COASTING (loss: {track.miss_duration:.1f}s)"
        status_color = color_magenta

    cv2.putText(frame, track_status, (20, y_start + line_h * 3), font, 0.48, status_color, 1, cv2.LINE_AA)

    # Line 5: Track Age & Velocity
    if track is not None and track.state != TrackState.DELETED:
        vx, vy = track.velocity[0], track.velocity[1]
        v_mag = math.sqrt(vx * vx + vy * vy)
        info_text = f"Age: {track.age:4.1f}s | Miss: {track.miss_duration:3.1f}s | Speed: {v_mag:4.0f} px/s"
    else:
        info_text = "Age: -- | Miss: -- | Speed: --"
    cv2.putText(frame, info_text, (20, y_start + line_h * 4), font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def run_tracking_pipeline(args: argparse.Namespace) -> int:
    """Execute the live single-person tracking demonstration pipeline.

    Guarantees resource release (camera, detector, windows) in try/finally blocks.
    """
    print("=" * 78)
    print(" Vision-Based First Contact — M3 Single-Person Tracking Runner")
    print("=" * 78)
    print(f" Camera Device:       {args.camera}")
    print(f" Model Weights:       {args.model}")
    print(f" Confidence Filter:   >= {args.conf:.2f} (fixed)")
    print(f" IoU Threshold:       >= {args.iou_thresh:.2f}")
    print(f" Retention Timeout:   {args.timeout:.1f} s")
    print(f" Requested Capture:   {args.width}x{args.height} @ {args.fps} FPS ({args.fourcc})")
    print(f" Compute Device:      {args.device or 'Default (Auto)'}")
    print(f" Headless Mode:       {args.headless}")
    print("-" * 78)

    detector: Optional[PersonDetector] = None
    tracker: Optional[SinglePersonTracker] = None
    cap: Optional[cv2.VideoCapture] = None
    total_frames = 0
    t_start = 0.0

    try:
        # Step 1: Initialize YOLO PersonDetector
        print("Initializing PersonDetector...")
        detector = PersonDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            device=args.device,
        )
        print("PersonDetector initialized successfully.")

        # Step 2: Initialize SinglePersonTracker
        print("Initializing SinglePersonTracker...")
        tracker = SinglePersonTracker(
            iou_threshold=args.iou_thresh,
            confirmation_hits=3,
            retention_timeout_seconds=args.timeout,
        )
        print("SinglePersonTracker initialized successfully.")

        # Step 3: Open Camera Stream
        print(f"Opening camera stream on '{args.camera}'...")
        cap, actual_w, actual_h, actual_fps = initialize_camera(
            device=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc,
        )
        print(f"Camera opened. Negotiated stream: {actual_w}x{actual_h} (reported {actual_fps:.1f} FPS)")

        if not args.headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WINDOW_NAME, actual_w, actual_h)
            print("\nLive visual window active. Press 'q', 'Esc', or close the window to exit.")

        # Performance measurement structures
        rolling_timestamps: deque[float] = deque(maxlen=60)
        inference_latencies: deque[float] = deque(maxlen=60)
        all_inference_latencies: list[float] = []

        first_frame_timestamp: Optional[float] = None
        current_app_fps = 0.0

        print("\nStarting frame acquisition and tracking loop...")
        t_start = time.perf_counter()

        while True:
            # Check frame bound
            if args.max_frames is not None and total_frames >= args.max_frames:
                print(f"\nReached requested frame limit of {args.max_frames} frames. Stopping.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[WARNING] Video capture returned empty frame or stream ended.")
                break

            total_frames += 1
            t_now = time.monotonic()

            if first_frame_timestamp is None:
                first_frame_timestamp = t_now

            rolling_timestamps.append(t_now)
            current_app_fps = compute_application_fps(rolling_timestamps)

            # Step 4: YOLO Person Detection
            t_inf_start = time.perf_counter()
            # Raw detections (without M2 single-person smoothing, since M3 Kalman handles motion)
            raw_detections = detector.detect_raw(frame)
            t_inf_end = time.perf_counter()
            lat_ms = (t_inf_end - t_inf_start) * 1000.0
            inference_latencies.append(lat_ms)
            all_inference_latencies.append(lat_ms)

            # Step 5: Single-Person Tracking Update
            active_track = tracker.update(raw_detections, timestamp=t_now)

            # Step 6: Visual Rendering
            if not args.headless:
                display_frame = frame.copy()
                display_frame = render_tracking(display_frame, active_track)
                display_frame = render_hud(
                    display_frame,
                    app_fps=current_app_fps,
                    inference_ms=lat_ms,
                    track=active_track,
                    frame_width=actual_w,
                    frame_height=actual_h,
                )
                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or Esc
                    print("\nUser requested exit via keyboard.")
                    break

                # Check if window was closed via OS close button
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("\nUser closed the visual display window.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Cleaning up...")
    except Exception as exc:
        print(f"\n[FATAL ERROR] An unexpected exception occurred in tracking pipeline: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Step 7: Guaranteed Resource Cleanup
        print("\nShutting down and releasing resources...")
        if cap is not None:
            try:
                cap.release()
                print("Camera capture released.")
            except Exception as e:
                print(f"Error releasing camera: {e}")

        if detector is not None:
            try:
                detector.close()
                print("PersonDetector closed.")
            except Exception as e:
                print(f"Error closing detector: {e}")

        if tracker is not None:
            try:
                tracker.reset()
                print("SinglePersonTracker reset.")
            except Exception as e:
                print(f"Error resetting tracker: {e}")

        try:
            cv2.destroyAllWindows()
            print("OpenCV windows destroyed.")
        except Exception:
            pass

    # Step 8: Final Metrics Summary
    t_end = time.perf_counter()
    elapsed_total = t_end - t_start

    print("\n" + "=" * 78)
    print(" Performance & Execution Summary")
    print("=" * 78)
    print(f" Total frames processed:      {total_frames}")
    print(f" Total runtime:               {elapsed_total:.2f} s")

    if total_frames > 1 and first_frame_timestamp is not None:
        steady_state_elapsed = rolling_timestamps[-1] - first_frame_timestamp
        steady_state_fps = compute_application_fps([first_frame_timestamp, rolling_timestamps[-1]], frame_count=total_frames)
        print(f" Steady-state Application FPS: {steady_state_fps:.2f} FPS")
    else:
        print(" Steady-state Application FPS: Insufficient frames captured")

    if all_inference_latencies:
        mean_lat = float(np.mean(all_inference_latencies))
        med_lat = float(np.median(all_inference_latencies))
        min_lat = float(np.min(all_inference_latencies))
        max_lat = float(np.max(all_inference_latencies))
        print(f" YOLO Latency (Mean):          {mean_lat:.2f} ms")
        print(f" YOLO Latency (Median):        {med_lat:.2f} ms")
        print(f" YOLO Latency (Min / Max):     {min_lat:.2f} ms / {max_lat:.2f} ms")
    print("=" * 78)

    return 0


def main() -> None:
    """Entry point for standalone execution."""
    args = parse_args()
    exit_code = run_tracking_pipeline(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

