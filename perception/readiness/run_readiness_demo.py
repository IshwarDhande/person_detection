"""Standalone Live Readiness Perception Demonstration Runner for Milestone M4.

Captures live video, detects people using YOLOv8n, evaluates bounding-box geometry,
assesses face visibility with MediaPipe Face Landmarker, computes image-space interaction-zone
membership, and estimates head orientation according to Milestone M4 specifications.

Visual Indicators:
  - Interaction Zone Polygon: Cyan boundary line
  - Person Bounding Box:
      * Green if VALID & INSIDE zone
      * Blue if VALID & OUTSIDE zone
      * Red if INVALID
  - Face Bounding Box: Yellow rectangle when face is detected
  - Status Indicators:
      * [VALID]: Green / text indication
      * [UNCERTAIN]: Amber text indication
      * [INVALID]: Red text indication
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from perception.detection.person_detector import PersonDetector
from perception.readiness.face_analyzer import FaceAnalyzer
from perception.readiness.frame_observer import FrameObserver
from perception.readiness.interaction_zone import InteractionZoneEvaluator
from perception.readiness.observation_types import (
    FaceVisibilityValue,
    FrameObservation,
    HeadOrientationValue,
    InteractionZoneValue,
    ObservationStatus,
    PersonPresenceValue,
)
from perception.readiness.orientation_estimator import OrientationEstimator
from perception.readiness.readiness_config import ReadinessConfig, load_readiness_config

DEFAULT_CONFIG_PATH = "configs/m4/stationary_readiness.yaml"
WINDOW_NAME = "Vision-Based First Contact — M4 Readiness Perception"

# Color Palette (BGR)
COLOR_ZONE = (255, 255, 0)         # Cyan for interaction zone polygon
COLOR_INSIDE = (0, 255, 0)         # Green for person inside interaction zone
COLOR_OUTSIDE = (255, 128, 0)      # Blue/Orange for person outside zone
COLOR_INVALID = (0, 0, 255)        # Red for invalid / failed state
COLOR_UNCERTAIN = (0, 215, 255)    # Amber/Gold for uncertain state
COLOR_FACE = (0, 255, 255)         # Yellow for face box
COLOR_TEXT_BG = (20, 20, 20)       # Dark charcoal for HUD backplate
COLOR_TEXT_FG = (255, 255, 255)    # White text


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate CLI arguments for the M4 readiness demo runner."""
    parser = argparse.ArgumentParser(
        description="M4 Live Frame-Level Readiness Perception Demonstration Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to M4 YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Override camera device path or index from config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override YOLO model path from config",
    )
    parser.add_argument(
        "--face-model",
        type=str,
        default=None,
        help="Override MediaPipe face landmarker model path from config",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override detection confidence threshold from config",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override capture width from config",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override capture height from config",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override capture framerate from config",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default=None,
        help="Override capture FourCC format from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for YOLO (e.g., 'cpu', 'cuda:0', default: auto)",
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
        help="Run in headless benchmark mode without GUI windows",
    )

    parsed = parser.parse_args(args)

    if parsed.conf is not None and not (0.0 <= parsed.conf <= 1.0):
        parser.error(f"--conf must be between 0.0 and 1.0, got {parsed.conf}")
    if parsed.width is not None and parsed.width <= 0:
        parser.error(f"--width must be positive, got {parsed.width}")
    if parsed.height is not None and parsed.height <= 0:
        parser.error(f"--height must be positive, got {parsed.height}")
    if parsed.fps is not None and parsed.fps <= 0:
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
        cap = cv2.VideoCapture(int(device), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(int(device) if device.isdigit() else device)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera device '{device}'. Verify that the device is connected."
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
        raise RuntimeError(f"Camera opened on '{device}', but initial test frame read failed.")

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


def render_readiness_overlays(
    frame: np.ndarray,
    obs: FrameObservation,
    zone_evaluator: Optional[InteractionZoneEvaluator],
) -> np.ndarray:
    """Render bounding boxes, interaction zone, and face overlays on the frame."""
    h_img, w_img = frame.shape[:2]

    # 1. Render Interaction Zone Polygon
    if zone_evaluator is not None and zone_evaluator.enabled:
        pts = zone_evaluator.to_pixel_coords(w_img, h_img)
        pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_np], isClosed=True, color=COLOR_ZONE, thickness=2, lineType=cv2.LINE_AA)
        
        # Zone label
        if len(pts) > 0:
            first_pt = (int(pts[0][0]), int(pts[0][1]))
            cv2.putText(
                frame,
                "INTERACTION ZONE (Image-Space)",
                (first_pt[0] + 5, first_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_ZONE,
                1,
                cv2.LINE_AA,
            )

    # 2. Render Person Bounding Box
    if obs.person.presence.value == PersonPresenceValue.DETECTED and obs.person.bbox_metrics is not None:
        metrics = obs.person.bbox_metrics
        x1, y1, x2, y2 = [int(round(v)) for v in metrics.original_bbox_xyxy]
        x1, x2 = max(0, min(w_img - 1, x1)), max(0, min(w_img - 1, x2))
        y1, y2 = max(0, min(h_img - 1, y1)), max(0, min(h_img - 1, y2))

        # Color based on zone membership
        if obs.person.interaction_zone.value == InteractionZoneValue.INSIDE:
            box_color = COLOR_INSIDE
            zone_tag = "INSIDE ZONE"
        else:
            box_color = COLOR_OUTSIDE
            zone_tag = "OUTSIDE ZONE"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Label tag
        conf = obs.person.presence.confidence or 0.0
        label = f"Person {conf * 100:.0f}% [{zone_tag}]"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = y1
        label_x2 = min(w_img - 1, x1 + tw + 6)
        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), COLOR_TEXT_BG, -1)
        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), box_color, 1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT_FG,
            1,
            cv2.LINE_AA,
        )

    # 3. Render Face Bounding Box
    if obs.face.metrics is not None and obs.face.metrics.face_bbox_xyxy is not None:
        fx1, fy1, fx2, fy2 = [int(round(v)) for v in obs.face.metrics.face_bbox_xyxy]
        fx1, fx2 = max(0, min(w_img - 1, fx1)), max(0, min(w_img - 1, fx2))
        fy1, fy2 = max(0, min(h_img - 1, fy1)), max(0, min(h_img - 1, fy2))

        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), COLOR_FACE, 1)

        orient_tag = obs.orientation.head_orientation.value.value
        cv2.putText(
            frame,
            f"Face: {orient_tag}",
            (fx1, max(0, fy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            COLOR_FACE,
            1,
            cv2.LINE_AA,
        )

    return frame


def render_hud(
    frame: np.ndarray,
    obs: FrameObservation,
    app_fps: float,
    yolo_ms: float,
    pipeline_ms: float,
) -> np.ndarray:
    """Render structured HUD showing the 5 observation conditions and performance."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 210), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    line_h = 22
    y = 32

    # Helper for status color
    def status_col(st: ObservationStatus) -> tuple[int, int, int]:
        if st == ObservationStatus.VALID:
            return (0, 255, 0)
        elif st == ObservationStatus.UNCERTAIN:
            return (0, 215, 255)
        return (0, 0, 255)

    # Header
    cv2.putText(frame, "M4 READINESS PERCEPTION (FRAME-LEVEL)", (20, y), font, 0.50, (255, 255, 0), 1, cv2.LINE_AA)
    y += line_h

    # FPS & Latency
    cv2.putText(
        frame,
        f"FPS: {app_fps:4.1f} | YOLO: {yolo_ms:4.1f}ms | Total M4: {pipeline_ms:4.1f}ms",
        (20, y),
        font,
        font_scale,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    y += line_h

    # Cond 1: Person Presence
    p_val = obs.person.presence.value.value
    p_st = obs.person.presence.status
    cv2.putText(
        frame,
        f"1. Person: {p_val} [{p_st.value}]",
        (20, y),
        font,
        font_scale,
        status_col(p_st),
        1,
        cv2.LINE_AA,
    )
    y += line_h

    # Cond 2: BBox Validity
    b_val = obs.person.bounding_box.value.value
    b_st = obs.person.bounding_box.status
    ratio_str = f"({obs.person.bbox_metrics.area_ratio * 100:.1f}% frame)" if obs.person.bbox_metrics else ""
    cv2.putText(
        frame,
        f"2. BBox Valid: {b_val} {ratio_str} [{b_st.value}]",
        (20, y),
        font,
        font_scale,
        status_col(b_st),
        1,
        cv2.LINE_AA,
    )
    y += line_h

    # Cond 3: Face Visibility
    f_val = obs.face.visibility.value.value
    f_st = obs.face.visibility.status
    face_dim = (
        f"({obs.face.metrics.face_width:.0f}x{obs.face.metrics.face_height:.0f}px)"
        if obs.face.metrics and obs.face.metrics.face_width
        else ""
    )
    cv2.putText(
        frame,
        f"3. Face: {f_val} {face_dim} [{f_st.value}]",
        (20, y),
        font,
        font_scale,
        status_col(f_st),
        1,
        cv2.LINE_AA,
    )
    y += line_h

    # Cond 4: Interaction Zone
    z_val = obs.person.interaction_zone.value.value
    z_st = obs.person.interaction_zone.status
    cv2.putText(
        frame,
        f"4. Zone: {z_val} [{z_st.value}]",
        (20, y),
        font,
        font_scale,
        status_col(z_st),
        1,
        cv2.LINE_AA,
    )
    y += line_h

    # Cond 5: Head Orientation
    o_val = obs.orientation.head_orientation.value.value
    o_st = obs.orientation.head_orientation.status
    yaw_str = f"(yaw: {obs.orientation.metrics.yaw_ratio:.2f})" if obs.orientation.metrics and obs.orientation.metrics.yaw_ratio is not None else ""
    cv2.putText(
        frame,
        f"5. Orientation: {o_val} {yaw_str} [{o_st.value}]",
        (20, y),
        font,
        font_scale,
        status_col(o_st),
        1,
        cv2.LINE_AA,
    )

    return frame


def run_readiness_pipeline(args: argparse.Namespace) -> int:
    """Execute the live readiness perception demonstration pipeline.

    Guarantees resource release (detector, landmarker, camera, windows) in try/finally blocks.
    """
    # Step 1: Load Configuration
    config = load_readiness_config(args.config)

    # Apply CLI overrides
    camera_dev = args.camera or config.camera.device
    model_path = args.model or config.detector.model_path
    face_model_path = args.face_model or config.face_analyzer.model_path
    conf_thresh = args.conf if args.conf is not None else config.detector.confidence_threshold
    cap_w = args.width if args.width is not None else config.camera.width
    cap_h = args.height if args.height is not None else config.camera.height
    cap_fps = args.fps if args.fps is not None else config.camera.fps
    cap_fourcc = args.fourcc or config.camera.fourcc

    print("=" * 78)
    print(" Vision-Based First Contact — M4 Readiness Perception Runner")
    print("=" * 78)
    print(f" Camera Device:       {camera_dev}")
    print(f" YOLO Model:          {model_path} (conf >= {conf_thresh:.2f})")
    print(f" MediaPipe Model:     {face_model_path}")
    print(f" Capture Format:      {cap_w}x{cap_h} @ {cap_fps} FPS ({cap_fourcc})")
    print(f" Headless Mode:       {args.headless}")
    print("-" * 78)

    detector: Optional[PersonDetector] = None
    face_analyzer: Optional[FaceAnalyzer] = None
    observer: Optional[FrameObserver] = None
    zone_evaluator: Optional[InteractionZoneEvaluator] = None
    cap: Optional[cv2.VideoCapture] = None
    total_frames = 0
    t_start = 0.0

    # Latency tracking for reporting
    yolo_latencies: list[float] = []
    bbox_latencies: list[float] = []
    zone_latencies: list[float] = []
    face_latencies: list[float] = []
    orient_latencies: list[float] = []
    total_latencies: list[float] = []
    rolling_timestamps: deque[float] = deque(maxlen=60)

    try:
        # Step 2: Initialize YOLO Detector
        print("Initializing YOLO PersonDetector...")
        detector = PersonDetector(
            model_path=model_path,
            confidence_threshold=conf_thresh,
            device=args.device,
        )
        print("PersonDetector ready.")

        # Step 3: Initialize MediaPipe Face Analyzer
        print(f"Initializing FaceAnalyzer from '{face_model_path}'...")
        face_analyzer = FaceAnalyzer(
            model_path=face_model_path,
            min_face_width=config.face_analyzer.min_face_width,
            min_face_height=config.face_analyzer.min_face_height,
            min_detection_confidence=config.face_analyzer.min_detection_confidence,
            min_tracking_confidence=config.face_analyzer.min_tracking_confidence,
        )
        print("FaceAnalyzer ready.")

        # Step 4: Initialize Interaction Zone & Orientation Estimator
        zone_evaluator = InteractionZoneEvaluator(
            vertices_normalized=config.interaction_zone.vertices,
            enabled=config.interaction_zone.enabled,
        )
        orient_estimator = OrientationEstimator(
            yaw_threshold=config.orientation.yaw_threshold,
            pitch_threshold=config.orientation.pitch_threshold,
            uncertain_multiplier=config.orientation.uncertain_multiplier,
        )

        # Step 5: Initialize FrameObserver
        observer = FrameObserver(
            face_analyzer=face_analyzer,
            zone_evaluator=zone_evaluator,
            orientation_estimator=orient_estimator,
        )

        # Step 6: Initialize Camera
        print(f"Opening camera on '{camera_dev}'...")
        cap, actual_w, actual_h, actual_fps = initialize_camera(
            device=camera_dev,
            width=cap_w,
            height=cap_h,
            fps=cap_fps,
            fourcc=cap_fourcc,
        )
        print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        if not args.headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WINDOW_NAME, actual_w, actual_h)
            print("Live window active. Press 'q' or 'Esc' to exit.")

        t_start = time.perf_counter()
        first_frame_t: Optional[float] = None
        current_app_fps = 0.0

        # Step 7: Main Acquisition Loop
        while True:
            if args.max_frames is not None and total_frames >= args.max_frames:
                print(f"\nReached max-frames limit of {args.max_frames}. Exiting.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[WARNING] Camera stream returned empty frame.")
                break

            total_frames += 1
            t_now = time.monotonic()
            if first_frame_t is None:
                first_frame_t = t_now
            rolling_timestamps.append(t_now)
            current_app_fps = compute_application_fps(rolling_timestamps)

            # Measure YOLO inference
            t_yolo_start = time.perf_counter()
            raw_detections = detector.detect_raw(frame)
            t_yolo_end = time.perf_counter()
            yolo_ms = (t_yolo_end - t_yolo_start) * 1000.0
            yolo_latencies.append(yolo_ms)

            # Measure Full M4 Readiness Observation
            t_obs_start = time.perf_counter()
            observation = observer.observe(frame, raw_detections, timestamp=t_now)
            t_obs_end = time.perf_counter()
            m4_total_ms = (t_obs_end - t_yolo_start) * 1000.0
            total_latencies.append(m4_total_ms)

            # Record fine-grained component timings from observer
            if observer.last_timings is not None:
                bbox_latencies.append(observer.last_timings.bbox_ms)
                zone_latencies.append(observer.last_timings.zone_ms)
                if observer.last_timings.face_ms is not None:
                    face_latencies.append(observer.last_timings.face_ms)
                if observer.last_timings.orient_ms is not None:
                    orient_latencies.append(observer.last_timings.orient_ms)

            # Visual Rendering
            if not args.headless:
                display_frame = frame.copy()
                display_frame = render_readiness_overlays(display_frame, observation, zone_evaluator)
                display_frame = render_hud(
                    display_frame,
                    obs=observation,
                    app_fps=current_app_fps,
                    yolo_ms=yolo_ms,
                    pipeline_ms=m4_total_ms,
                )
                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\nUser requested exit via keyboard.")
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("\nUser closed the display window.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Cleaning up...")
    except Exception as exc:
        print(f"\n[FATAL ERROR] An unexpected exception occurred: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Step 8: Guaranteed Resource Cleanup
        print("\nReleasing resources...")
        if cap is not None:
            try:
                cap.release()
                print("Camera released.")
            except Exception as e:
                print(f"Error releasing camera: {e}")

        if observer is not None:
            try:
                observer.close()
                print("Observer closed.")
            except Exception as e:
                print(f"Error closing observer: {e}")
        elif face_analyzer is not None:
            try:
                face_analyzer.close()
                print("FaceAnalyzer closed.")
            except Exception as e:
                print(f"Error closing face analyzer: {e}")

        if detector is not None:
            try:
                detector.close()
                print("PersonDetector closed.")
            except Exception as e:
                print(f"Error closing detector: {e}")

        try:
            cv2.destroyAllWindows()
            print("OpenCV windows destroyed.")
        except Exception:
            pass

    # Step 9: Metrics Summary
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    def _print_metric_row(name: str, values: list[float]) -> None:
        if not values:
            print(f" {name:<29} N/A (no executions)")
            return
        mean_val = float(np.mean(values))
        p50 = float(np.percentile(values, 50))
        p95 = float(np.percentile(values, 95))
        p99 = float(np.percentile(values, 99))
        print(f" {name:<29} Avg: {mean_val:6.2f} ms | p50: {p50:6.2f} ms | p95: {p95:6.2f} ms | p99: {p99:6.2f} ms")

    print("\n" + "=" * 78)
    print(" Performance & Execution Summary (Milestone M4)")
    print("=" * 78)
    print(f" Total frames processed:      {total_frames}")
    print(f" Total runtime:               {elapsed:.2f} s")
    if total_frames > 1 and first_frame_t is not None:
        ss_fps = compute_application_fps([first_frame_t, rolling_timestamps[-1]], frame_count=total_frames)
        print(f" Steady-state Application FPS: {ss_fps:.2f} FPS")
    print("-" * 78)
    print(" Component Execution Latencies:")
    _print_metric_row("YOLOv8n Inference", yolo_latencies)
    _print_metric_row("BBox Validation", bbox_latencies)
    _print_metric_row("Zone Evaluation", zone_latencies)
    _print_metric_row("MediaPipe Face Landmarker", face_latencies)
    _print_metric_row("Orientation Post-Processing", orient_latencies)
    print("-" * 78)
    _print_metric_row("Total M4 Pipeline Latency", total_latencies)
    print("-" * 78)

    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f" Process Memory RSS:          {mem_mb:.2f} MB")
        cpu_usage = process.cpu_percent(interval=None)
        print(f" Process CPU Utilization:     {cpu_usage:.1f}%")
    except Exception:
        pass
    print("=" * 78)

    return 0


def main() -> None:
    """Entry point for standalone execution."""
    args = parse_args()
    exit_code = run_readiness_pipeline(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

