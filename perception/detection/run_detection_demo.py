"""Standalone Live Person Detection Demonstration Runner for Milestone M1.

This utility captures live video from a configurable camera device (default: /dev/video0),
runs frame-level person detection using PersonDetector (YOLOv8n), and renders real-time
bounding boxes, confidence scores, inference latency, and application throughput.

Architecture & Execution Flow:
+-----------------------------------------------------------------------------+
|                     M1 LIVE DETECTION RUNNER PIPELINE                       |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [Camera: /dev/video0]                                                      |
|       |                                                                     |
|       | V4L2 MJPG Capture (1280x720 @ 30 FPS requested)                     |
|       v                                                                     |
|  [Frame Acquisition] -------------------------------+                       |
|       |                                             |                       |
|       v (perf_counter start)                        | (Loop Timestamps)     |
|  [PersonDetector.detect()]                          v                       |
|       |                                     [Application FPS]               |
|       v (perf_counter end -> Latency ms)            |                       |
|  [Visual Rendering] <-------------------------------+                       |
|       | - Bounding boxes: (x1, y1) -> (x2, y2)                              |
|       | - Confidence label: "Person: 0.85"                                  |
|       | - HUD Overlay: FPS, Latency ms, Count, Resolution                   |
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
    SinglePersonSmoother,
)

DEFAULT_CAMERA_DEVICE = "/dev/video0"
DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_CAPTURE_WIDTH = 1280
DEFAULT_CAPTURE_HEIGHT = 720
DEFAULT_CAPTURE_FPS = 30
DEFAULT_FOURCC = "MJPG"
DEFAULT_EMA_ALPHA = SinglePersonSmoother.DEFAULT_ALPHA
WINDOW_NAME = "Vision-Based First Contact — M1/M2 Person Detection"


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate command-line arguments for the detection demo runner.

    Args:
        args: Optional list of CLI arguments (defaults to sys.argv[1:]).

    Returns:
        Validated argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="M1 Live Person Detection Demonstration Runner (YOLOv8n /dev/video0)"
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
        help=f"Requested video FourCC codec (default: {DEFAULT_FOURCC})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch compute device override (e.g. 'cpu', 'cuda:0')",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before terminating (default: unlimited)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying a graphical window (useful for automated profiling)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help=f"EMA coefficient in (0.0, 1.0] for single-person bounding-box smoothing (default: {DEFAULT_EMA_ALPHA})",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable single-person EMA bounding-box smoothing",
    )

    parsed = parser.parse_args(args)

    # Validate argument bounds
    if not (0.0 <= parsed.conf <= 1.0):
        parser.error(f"--conf must be between 0.0 and 1.0, got {parsed.conf}")
    if not (0.0 < parsed.ema_alpha <= 1.0):
        parser.error(f"--ema-alpha must be in (0.0, 1.0], got {parsed.ema_alpha}")
    if parsed.width <= 0:
        parser.error(f"--width must be positive, got {parsed.width}")
    if parsed.height <= 0:
        parser.error(f"--height must be positive, got {parsed.height}")
    if parsed.fps <= 0:
        parser.error(f"--fps must be positive, got {parsed.fps}")
    if parsed.max_frames is not None and parsed.max_frames <= 0:
        parser.error(f"--max-frames must be positive, got {parsed.max_frames}")

    return parsed


def open_camera_stream(
    device_arg: str,
    width: int = DEFAULT_CAPTURE_WIDTH,
    height: int = DEFAULT_CAPTURE_HEIGHT,
    fps: int = DEFAULT_CAPTURE_FPS,
    fourcc: str = DEFAULT_FOURCC,
) -> Tuple[cv2.VideoCapture, int, int, float]:
    """Open the camera device, request format settings, and verify frame delivery.

    Args:
        device_arg: Path to device or numeric index.
        width: Requested pixel width.
        height: Requested pixel height.
        fps: Requested capture FPS.
        fourcc: Requested 4-character codec string (e.g. 'MJPG').

    Returns:
        Tuple of (cv2.VideoCapture object, actual_width, actual_height, actual_fps).

    Raises:
        RuntimeError: If device cannot be opened or fails to read an initial frame.
    """
    dev_target: Any = device_arg
    if isinstance(device_arg, str):
        if device_arg.startswith("/dev/video"):
            try:
                dev_target = int(device_arg.replace("/dev/video", ""))
            except ValueError:
                dev_target = device_arg
        elif device_arg.isdigit():
            dev_target = int(device_arg)

    # Attempt V4L2 backend first on Linux
    cap = cv2.VideoCapture(dev_target, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(dev_target)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video capture device '{device_arg}'. Check device existence and user permissions."
        )

    # Configure FourCC codec (crucial for USB 2.0 bandwidth to avoid 10 FPS YUYV throttling)
    if len(fourcc) == 4:
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Verify initial frame capture
    ret, test_frame = cap.read()
    if not ret or test_frame is None or test_frame.size == 0:
        cap.release()
        raise RuntimeError(
            f"Camera '{device_arg}' opened successfully but failed to yield a valid video frame."
        )

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    # Fallback to test_frame dimensions if CAP_PROP returns 0
    if actual_w <= 0 or actual_h <= 0:
        actual_h, actual_w = test_frame.shape[:2]

    return cap, actual_w, actual_h, actual_fps


def render_detections(
    frame: np.ndarray,
    detections: list[PersonDetection],
    app_fps: float,
    mean_app_fps: float,
    latency_ms: float,
    mean_latency_ms: float,
    actual_w: int,
    actual_h: int,
) -> np.ndarray:
    """Draw bounding boxes, confidence labels, and HUD overlay on the display frame.

    Args:
        frame: BGR image frame (modified in-place or returned).
        detections: List of PersonDetection objects to draw.
        app_fps: Current instantaneous application FPS.
        mean_app_fps: Cumulative mean application FPS.
        latency_ms: Current frame inference latency in milliseconds.
        mean_latency_ms: Cumulative mean inference latency in milliseconds.
        actual_w: Actual frame width.
        actual_h: Actual frame height.

    Returns:
        Annotated BGR image frame.
    """
    # 1. Draw person bounding boxes and confidence labels
    for det in detections:
        conf_pct = det.confidence * 100.0

        if det.is_smoothed and det.smoothed_bbox_xyxy is not None:
            # Subtle raw bounding box (Cyan thin line)
            rx1, ry1, rx2, ry2 = [int(round(coord)) for coord in det.bbox_xyxy]
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 1)

            # Stabilized smoothed bounding box (Bright Green solid line)
            sx1, sy1, sx2, sy2 = [int(round(coord)) for coord in det.smoothed_bbox_xyxy]
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            label = f"Person (EMA): {conf_pct:.1f}%"
            draw_x, draw_y = sx1, sy1
        else:
            # Unsmoothed detection (multi-person or smoothing disabled)
            x1, y1, x2, y2 = [int(round(coord)) for coord in det.bbox_xyxy]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {conf_pct:.1f}%"
            draw_x, draw_y = x1, y1

        # Text label background plate
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_y = max(draw_y - 6, text_h + 4)
        cv2.rectangle(
            frame,
            (draw_x, label_y - text_h - 4),
            (draw_x + text_w + 6, label_y + 2),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (draw_x + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # 2. Render HUD Overlay Panel in top-left corner
    hud_x, hud_y = 12, 12
    hud_w, hud_h = 340, 105
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (25, 25, 25), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (80, 80, 80), 1)

    if len(detections) == 1 and detections[0].is_smoothed:
        det_label = "Detections: 1 person (EMA smoothed)"
    elif len(detections) > 1:
        det_label = f"Detections: {len(detections)} persons (multi-person, raw)"
    else:
        det_label = f"Detections: {len(detections)} persons"

    hud_lines = [
        f"App FPS: {app_fps:5.1f} (avg: {mean_app_fps:4.1f})",
        f"Inference: {latency_ms:5.1f} ms (avg: {mean_latency_ms:4.1f} ms)",
        det_label,
        f"Resolution: {actual_w}x{actual_h}",
    ]

    for idx, line in enumerate(hud_lines):
        line_y = hud_y + 20 + (idx * 21)
        # Highlight latency or count conditionally
        color = (0, 255, 255) if idx == 0 else (220, 220, 220)
        if idx == 2 and len(detections) > 0:
            color = (0, 255, 0)
        cv2.putText(
            frame,
            line,
            (hud_x + 10, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def compute_application_fps(
    timestamps: Sequence[float], frame_count: Optional[int] = None
) -> float:
    """Calculate steady-state application FPS from frame completion timestamps.

    Throughput is defined as:
        FPS = (N - 1) / (tau_N - tau_1)
    where tau_i are monotonic timestamps recorded at the completion of processed frames.

    Args:
        timestamps: Sequence of monotonically increasing timestamps (in seconds).
        frame_count: Optional total frame count N across the timestamp window.
                     If None, defaults to len(timestamps).

    Returns:
        Computed FPS as a float, or 0.0 if fewer than 2 frames or elapsed time <= 0.
    """
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0.0:
        return 0.0
    n_frames = frame_count if frame_count is not None else len(timestamps)
    if n_frames < 2:
        return 0.0
    return (n_frames - 1) / elapsed


def run_detection_pipeline(
    camera: str = DEFAULT_CAMERA_DEVICE,
    model: str = DEFAULT_MODEL_PATH,
    conf: float = DEFAULT_CONFIDENCE_THRESHOLD,
    width: int = DEFAULT_CAPTURE_WIDTH,
    height: int = DEFAULT_CAPTURE_HEIGHT,
    fps: int = DEFAULT_CAPTURE_FPS,
    fourcc: str = DEFAULT_FOURCC,
    device: Optional[str] = None,
    max_frames: Optional[int] = None,
    headless: bool = False,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    no_smoothing: bool = False,
) -> dict[str, Any]:
    """Execute the live detection demonstration loop.

    Guarantees resource cleanup across detector, camera, and GUI windows on any
    initialization failure, loop exception, or normal exit.

    Args:
        camera: Device path or index.
        model: Weights path.
        conf: Confidence threshold (fixed at 0.5 for M1/M2).
        width: Requested capture width.
        height: Requested capture height.
        fps: Requested capture FPS.
        fourcc: Codec FourCC.
        device: Torch compute device.
        max_frames: Optional frame limit.
        headless: Headless flag.
        ema_alpha: EMA smoothing coefficient in (0.0, 1.0] for single person bounding box.
        no_smoothing: If True, disables EMA bounding-box smoothing.

    Returns:
        Dictionary containing summary metrics and execution statistics.
    """
    print("=" * 78)
    print(" Vision-Based First Contact — M1/M2 Person Detection Runner")
    print("=" * 78)
    print(f" Camera Device:       {camera}")
    print(f" Model Weights:       {model}")
    print(f" Confidence Filter:   >= {conf:.2f} (fixed)")
    print(f" Smoothing (EMA):     {'Disabled' if no_smoothing else f'Enabled (alpha={ema_alpha:.2f})'}")
    print(f" Requested Capture:   {width}x{height} @ {fps} FPS ({fourcc})")
    print(f" Compute Device:      {device if device else 'Default (Auto)'}")
    print(f" Headless Mode:       {headless}")
    if max_frames:
        print(f" Frame Limit:         {max_frames} frames")
    print("-" * 78)

    detector: Optional[PersonDetector] = None
    cap: Optional[cv2.VideoCapture] = None
    window_created: bool = False

    session_start_time = time.perf_counter()

    # Metrics tracking
    frame_count = 0
    total_detections_count = 0
    latencies_ms: list[float] = []
    loop_timestamps: deque[float] = deque(maxlen=30)
    first_frame_timestamp: Optional[float] = None
    last_frame_timestamp: Optional[float] = None
    consecutive_read_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10
    actual_w = 0
    actual_h = 0
    actual_fps = 0.0

    try:
        # 1. Initialize Smoother and PersonDetector
        smoother = None if no_smoothing else SinglePersonSmoother(alpha=ema_alpha)
        print("Initializing PersonDetector...")
        detector = PersonDetector(
            model_path=model,
            confidence_threshold=conf,
            device=device,
            smoother=smoother,
        )
        print("PersonDetector initialized successfully.")

        # 2. Open Camera Stream
        print(f"Opening camera stream on '{camera}'...")
        cap, actual_w, actual_h, actual_fps = open_camera_stream(
            device_arg=camera,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
        )
        print(f"Camera opened. Negotiated stream: {actual_w}x{actual_h} (reported {actual_fps:.1f} FPS)")

        if not headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            window_created = True
            print("Live preview active. Press 'q' or 'Esc' to exit.")

        while True:
            # Check frame cap
            if max_frames is not None and frame_count >= max_frames:
                print(f"Reached configured limit of {max_frames} frames. Terminating.")
                break

            # A. Acquire frame
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_read_failures += 1
                print(f"[WARN] Camera frame read failed (attempt {consecutive_read_failures}/{MAX_CONSECUTIVE_FAILURES})")
                if consecutive_read_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"Camera stream terminated unexpectedly: failed to read {MAX_CONSECUTIVE_FAILURES} consecutive frames."
                    )
                time.sleep(0.01)
                continue

            consecutive_read_failures = 0

            # B. Run detection and measure inference latency (isolated from loop FPS)
            t_infer_start = time.perf_counter()
            detections = detector.detect(frame)
            t_infer_end = time.perf_counter()

            infer_lat_ms = (t_infer_end - t_infer_start) * 1000.0
            latencies_ms.append(infer_lat_ms)
            total_detections_count += len(detections)
            frame_count += 1

            # C. Track application throughput timestamps (starts after first frame)
            t_completed = time.perf_counter()
            loop_timestamps.append(t_completed)
            last_frame_timestamp = t_completed
            if first_frame_timestamp is None:
                first_frame_timestamp = t_completed

            # Rolling FPS over recent frames
            app_fps = compute_application_fps(loop_timestamps)

            # Cumulative steady-state FPS over all processed frames (excluding startup/warmup)
            if frame_count > 1 and first_frame_timestamp is not None:
                mean_app_fps = compute_application_fps(
                    [first_frame_timestamp, t_completed], frame_count=frame_count
                )
            else:
                mean_app_fps = 0.0

            mean_lat_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

            # D. Render and display
            if not headless:
                rendered_frame = render_detections(
                    frame=frame,
                    detections=detections,
                    app_fps=app_fps,
                    mean_app_fps=mean_app_fps,
                    latency_ms=infer_lat_ms,
                    mean_latency_ms=mean_lat_ms,
                    actual_w=actual_w,
                    actual_h=actual_h,
                )
                cv2.imshow(WINDOW_NAME, rendered_frame)

                # Poll keys
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q', 'Q', or Esc
                    print("User requested exit via keypress.")
                    break

                # Check if window was closed via OS 'X' button
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user.")
                    break
            else:
                # Periodic console progress in headless mode
                if frame_count % 30 == 0 or frame_count == 1:
                    print(
                        f"Frame {frame_count:4d}: App FPS={app_fps:5.1f} (avg: {mean_app_fps:5.1f}), "
                        f"Latency={infer_lat_ms:5.1f} ms (avg: {mean_lat_ms:5.1f} ms), "
                        f"Detections={len(detections)}"
                    )

    finally:
        # Resource cleanup - guaranteed execution for all acquired resources
        print("-" * 78)
        print("Releasing camera and graphics resources...")
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
        if window_created:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            window_created = False
        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass
            detector = None

    total_session_time = time.perf_counter() - session_start_time

    # Steady-state application throughput calculation
    if frame_count > 1 and first_frame_timestamp is not None and last_frame_timestamp is not None:
        steady_state_duration = last_frame_timestamp - first_frame_timestamp
        final_fps = compute_application_fps(
            [first_frame_timestamp, last_frame_timestamp], frame_count=frame_count
        )
    else:
        steady_state_duration = 0.0
        final_fps = 0.0

    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    min_latency = float(np.min(latencies_ms)) if latencies_ms else 0.0
    max_latency = float(np.max(latencies_ms)) if latencies_ms else 0.0
    p50_latency = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    print("=" * 78)
    print(" M1/M2 EXECUTION & PERFORMANCE BASELINE SUMMARY")
    print("=" * 78)
    print(f" Processed Frames:                  {frame_count}")
    print(f" Steady-State Loop Duration:        {steady_state_duration:.2f} s (frame 1 to {frame_count})")
    print(f" Total Session Duration:            {total_session_time:.2f} s (including initialization)")
    print(f" Mean Application Throughput:       {final_fps:.2f} FPS (steady-state loop)")
    print(f" Camera Stream Negotiated FPS:      {actual_fps:.1f} FPS (sensor hardware target)")
    print(f" Mean Inference Latency:            {mean_latency:.2f} ms (per-frame detect() call)")
    print(f" Median (P50) Latency:              {p50_latency:.2f} ms")
    print(f" 95th Percentile (P95):             {p95_latency:.2f} ms")
    print(f" Latency Min / Max:                 {min_latency:.2f} ms / {max_latency:.2f} ms")
    print(f" Total Person Detections:           {total_detections_count}")
    print(f" Stream Resolution:                 {actual_w}x{actual_h}")
    print("=" * 78)

    return {
        "processed_frames": frame_count,
        "steady_state_duration_s": steady_state_duration,
        "total_session_time_s": total_session_time,
        "mean_app_fps": final_fps,
        "mean_inference_latency_ms": mean_latency,
        "p50_inference_latency_ms": p50_latency,
        "p95_inference_latency_ms": p95_latency,
        "min_inference_latency_ms": min_latency,
        "max_inference_latency_ms": max_latency,
        "total_person_detections": total_detections_count,
        "frame_width": actual_w,
        "frame_height": actual_h,
        "smoothing_enabled": not no_smoothing,
        "ema_alpha": ema_alpha if not no_smoothing else None,
    }


def main() -> None:
    """Entry point for live demonstration script."""
    args = parse_args()
    try:
        run_detection_pipeline(
            camera=args.camera,
            model=args.model,
            conf=args.conf,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc,
            device=args.device,
            max_frames=args.max_frames,
            headless=args.headless,
            ema_alpha=args.ema_alpha,
            no_smoothing=args.no_smoothing,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Terminating cleanly.")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
