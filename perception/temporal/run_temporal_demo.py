"""Standalone Live Temporal Behaviour Estimation Demonstration Runner for Milestone M5.

Captures live video, detects people via YOLOv8n, tracks the target person via M3 Kalman
tracker, computes M4 frame-level readiness observations, and evaluates M5 temporal behaviour
(velocity regression, displacement, centroid-directed approach, stationary classification,
and 2.0s stability timing with 200 ms noise tolerance).

Visual Overlays & HUD:
  - Interaction Zone Polygon: Cyan boundary line
  - Interaction Zone Centroid: Cyan crosshair and circle
  - Person Bounding Box:
      * Green if VALID & INSIDE zone
      * Orange if VALID & OUTSIDE zone
      * Magenta if COASTING / PREDICTED
  - Line from person center to zone centroid (yellow dotted / solid)
  - Motion Trajectory Breadcrumbs: Cyan history trail
  - Velocity Vector: Yellow arrow pointing in movement direction
  - Real-Time Diagnostic HUD:
      * FPS, Latency breakdown (YOLO, Tracker, Observer, Temporal)
      * Track ID, Status, Initialization (e.g. READY 5/5)
      * Velocity, Displacement, Distance to Centroid, Direction Cosine, Closing Speed
      * Approach State [Status], Movement State [Status]
      * Stability State, Elapsed / 2.00s, Violation / 0.20s
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional, Sequence

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import psutil

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
from perception.temporal.observation_types import (
    ApproachValue,
    InitializationState,
    MovementValue,
    StabilityValue,
    TemporalObservation,
)
from perception.temporal.temporal_config import TemporalConfig, load_temporal_config
from perception.temporal.temporal_estimator import TemporalEstimator
from perception.tracking.single_person_tracker import SinglePersonTracker, Track

DEFAULT_M4_CONFIG_PATH = "configs/m4/stationary_readiness.yaml"
DEFAULT_M5_CONFIG_PATH = "configs/m5/temporal_behaviour.yaml"
WINDOW_NAME = "Vision-Based First Contact — M5 Temporal Behaviour Estimation"

# Color Palette (BGR)
COLOR_ZONE = (255, 255, 0)          # Cyan for interaction zone polygon & centroid
COLOR_INSIDE = (0, 255, 0)          # Green for person inside interaction zone
COLOR_OUTSIDE = (255, 128, 0)       # Blue/Orange for person outside zone
COLOR_COASTING = (255, 0, 255)      # Magenta for coasting / predicted boxes
COLOR_TRAIL = (255, 255, 0)         # Cyan trail for center trajectory
COLOR_ARROW = (0, 255, 255)         # Yellow arrow for velocity vector
COLOR_LINE_CENTROID = (0, 215, 255) # Amber line to centroid
COLOR_TEXT_BG = (20, 20, 20)        # Dark charcoal for HUD backplate
COLOR_TEXT_FG = (255, 255, 255)     # White text
COLOR_VALID = (0, 255, 0)           # Green for VALID
COLOR_UNCERTAIN = (0, 215, 255)     # Amber for UNCERTAIN
COLOR_INVALID = (0, 0, 255)         # Red for INVALID


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate CLI arguments for the M5 temporal demo runner."""
    parser = argparse.ArgumentParser(
        description="M5 Live Temporal Behaviour Estimation Demonstration Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_M4_CONFIG_PATH,
        help=f"Path to M4 YAML config file (default: {DEFAULT_M4_CONFIG_PATH})",
    )
    parser.add_argument(
        "--temporal-config",
        type=str,
        default=DEFAULT_M5_CONFIG_PATH,
        help=f"Path to M5 YAML config file (default: {DEFAULT_M5_CONFIG_PATH})",
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
        help="Override YOLO detection confidence threshold",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before automatic termination",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference compute device ('cpu', 'cuda', etc.)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without graphical window display",
    )

    parsed = parser.parse_args(args)

    if parsed.conf is not None and not (0.0 <= parsed.conf <= 1.0):
        parser.error(f"--conf must be in [0.0, 1.0], got {parsed.conf}")
    if parsed.max_frames is not None and parsed.max_frames <= 0:
        parser.error(f"--max-frames must be positive, got {parsed.max_frames}")

    return parsed


def compute_application_fps(timestamps: deque) -> float:
    """Compute rolling FPS over timestamps deque."""
    if len(timestamps) < 2:
        return 0.0
    dt = timestamps[-1] - timestamps[0]
    if dt <= 0.0:
        return 0.0
    return float((len(timestamps) - 1) / dt)


def initialize_camera(
    device: str,
    width: int,
    height: int,
    fps: int,
    fourcc: str = "MJPG",
) -> tuple[cv2.VideoCapture, int, int, float]:
    """Open and negotiate camera stream capture properties."""
    device_arg: Any = int(device) if device.isdigit() else device
    cap = cv2.VideoCapture(device_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video capture device on '{device}'")

    if len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        cap.release()
        raise RuntimeError(f"Camera opened but failed initial frame read on '{device}'")

    return cap, actual_w, actual_h, actual_fps


def draw_text_with_background(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.45,
    color: tuple[int, int, int] = COLOR_TEXT_FG,
    bg_color: tuple[int, int, int] = COLOR_TEXT_BG,
    thickness: int = 1,
    padding: int = 2,
) -> None:
    """Draw text with solid background rectangle for legibility."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def render_temporal_overlays(
    canvas: np.ndarray,
    frame_observation: FrameObservation,
    temporal_obs: TemporalObservation,
    track: Optional[Track],
    zone_vertices_px: list[tuple[float, float]],
    zone_centroid_px: Optional[tuple[float, float]],
    trajectory: Sequence[tuple[float, float]],
    app_fps: float,
    yolo_ms: float,
    tracker_ms: float,
    observer_ms: float,
    temporal_ms: float,
) -> None:
    """Draw comprehensive M5 visual overlays and HUD diagnostics on frame."""
    h_img, w_img = canvas.shape[:2]

    # 1. Draw Interaction Zone Polygon
    if len(zone_vertices_px) >= 3:
        pts = np.array(zone_vertices_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=COLOR_ZONE, thickness=2, lineType=cv2.LINE_AA)

    # 2. Draw Interaction Zone Centroid
    if zone_centroid_px is not None:
        zx, zy = int(round(zone_centroid_px[0])), int(round(zone_centroid_px[1]))
        cv2.circle(canvas, (zx, zy), 6, COLOR_ZONE, -1, lineType=cv2.LINE_AA)
        cv2.drawMarker(canvas, (zx, zy), (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        draw_text_with_background(canvas, "ZONE CENTROID", (zx + 10, zy - 5), scale=0.4, color=COLOR_ZONE)

    # 3. Draw Track Bounding Box & Center
    if track is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox_xyxy]
        cx, cy = int(round(track.center[0])), int(round(track.center[1]))

        # Box Color based on state
        if not track.observed:
            box_color = COLOR_COASTING
            label = f"ID: {track.track_id} [COAST/PREDICTED]"
        elif frame_observation.person.interaction_zone.value == InteractionZoneValue.INSIDE:
            box_color = COLOR_INSIDE
            label = f"ID: {track.track_id} [INSIDE ZONE]"
        else:
            box_color = COLOR_OUTSIDE
            label = f"ID: {track.track_id} [OUTSIDE ZONE]"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, lineType=cv2.LINE_AA)
        draw_text_with_background(canvas, label, (x1, max(20, y1 - 8)), scale=0.5, color=box_color)

        # Draw Center Point
        cv2.circle(canvas, (cx, cy), 5, box_color, -1, lineType=cv2.LINE_AA)

        # 4. Draw Trajectory Breadcrumbs
        if len(trajectory) >= 2:
            for i in range(1, len(trajectory)):
                pt1 = (int(round(trajectory[i - 1][0])), int(round(trajectory[i - 1][1])))
                pt2 = (int(round(trajectory[i][0])), int(round(trajectory[i][1])))
                cv2.line(canvas, pt1, pt2, COLOR_TRAIL, 2, lineType=cv2.LINE_AA)
                cv2.circle(canvas, pt2, 2, COLOR_TRAIL, -1)

        # 5. Draw Line from Person Center to Centroid
        if zone_centroid_px is not None:
            zx, zy = int(round(zone_centroid_px[0])), int(round(zone_centroid_px[1]))
            cv2.line(canvas, (cx, cy), (zx, zy), COLOR_LINE_CENTROID, 1, lineType=cv2.LINE_AA)

        # 6. Draw Estimated Velocity Vector Arrow
        if (
            temporal_obs.motion.velocity_xy_px_s is not None
            and temporal_obs.motion.speed_px_s is not None
            and temporal_obs.motion.speed_px_s > 5.0
        ):
            vx, vy = temporal_obs.motion.velocity_xy_px_s
            arrow_scale = 0.5  # scale pixels/second to screen arrow pixels
            tip_x = int(round(cx + vx * arrow_scale))
            tip_y = int(round(cy + vy * arrow_scale))
            cv2.arrowedLine(
                canvas,
                (cx, cy),
                (tip_x, tip_y),
                COLOR_ARROW,
                2,
                tipLength=0.25,
                line_type=cv2.LINE_AA,
            )

    # 7. Render Multi-Section HUD Panel on Left Edge
    hud_x = 12
    hud_y = 22
    line_h = 17

    # Section A: Header & System Performance
    draw_text_with_background(
        canvas,
        "=== M5 TEMPORAL BEHAVIOUR PERCEPTION ===",
        (hud_x, hud_y),
        scale=0.5,
        color=(0, 255, 255),
    )
    hud_y += line_h

    fps_txt = f"FPS: {app_fps:4.1f} | Latency: {yolo_ms:.1f}ms YOLO, {tracker_ms:.1f}ms Trk, {observer_ms:.1f}ms M4, {temporal_ms:.1f}ms M5"
    draw_text_with_background(canvas, fps_txt, (hud_x, hud_y), scale=0.4, color=(200, 200, 200))
    hud_y += line_h + 3

    # Section B: Tracking & Initialization
    init_state = temporal_obs.initialization_state.value
    init_count = temporal_obs.initialization_count
    trk_id = temporal_obs.track_id if temporal_obs.track_id is not None else "-"
    trk_txt = f"Track ID: {trk_id} | Init: {init_state} ({init_count}/5) | Observed: {temporal_obs.observed}"
    draw_text_with_background(canvas, trk_txt, (hud_x, hud_y), scale=0.42, color=(255, 255, 255))
    hud_y += line_h

    # Section C: Motion Metrics
    m = temporal_obs.motion
    spd_txt = f"Speed: {m.speed_px_s:.1f} px/s" if m.speed_px_s is not None else "Speed: N/A"
    disp_txt = f"Disp: {m.displacement_px:.1f} px/1s" if m.displacement_px is not None else "Disp: N/A"
    dist_txt = f"Dist to Zone: {m.distance_to_zone_centroid_px:.1f} px" if m.distance_to_zone_centroid_px is not None else "Dist: N/A"
    motion_line = f"Motion: {spd_txt} | {disp_txt} | {dist_txt}"
    draw_text_with_background(canvas, motion_line, (hud_x, hud_y), scale=0.42, color=(255, 255, 255))
    hud_y += line_h

    cos_txt = f"cos(theta): {m.direction_cosine:+.2f}" if m.direction_cosine is not None else "cos(theta): N/A"
    close_txt = f"Closing: {m.closing_speed_px_s:+.1f} px/s" if m.closing_speed_px_s is not None else "Closing: N/A"
    draw_text_with_background(canvas, f"Vector: {cos_txt} | {close_txt}", (hud_x, hud_y), scale=0.42, color=(220, 220, 220))
    hud_y += line_h + 3

    # Section D: Approach & Movement Indicators
    app_val = temporal_obs.approach.value.value
    app_status = temporal_obs.approach.status.value
    app_color = COLOR_VALID if app_val == ApproachValue.APPROACHING else (COLOR_UNCERTAIN if app_status == "UNCERTAIN" else COLOR_TEXT_FG)
    draw_text_with_background(canvas, f"Approach: {app_val} [{app_status}]", (hud_x, hud_y), scale=0.45, color=app_color)
    hud_y += line_h

    mov_val = temporal_obs.movement.value.value
    mov_status = temporal_obs.movement.status.value
    mov_color = COLOR_VALID if mov_val == MovementValue.STATIONARY else (COLOR_UNCERTAIN if mov_status == "UNCERTAIN" else (255, 128, 0))
    draw_text_with_background(canvas, f"Movement: {mov_val} [{mov_status}]", (hud_x, hud_y), scale=0.45, color=mov_color)
    hud_y += line_h + 3

    # Section E: Stability Timer
    stab = temporal_obs.stability
    stab_val = stab.value.value
    stab_color = COLOR_VALID if stab_val == StabilityValue.COMPLETE else (COLOR_UNCERTAIN if stab_val == StabilityValue.ACCUMULATING else (180, 180, 180))
    stab_line = f"Stability: {stab_val} | Elapsed: {stab.elapsed_seconds:4.2f}s/2.00s | Excursion: {stab.violation_seconds:4.2f}s/0.20s"
    draw_text_with_background(canvas, stab_line, (hud_x, hud_y), scale=0.45, color=stab_color)


def run_temporal_pipeline(args: argparse.Namespace) -> int:
    """Run full live capture, M1 detection, M3 tracking, M4 readiness, and M5 temporal behaviour."""
    # 1. Load Configurations
    m4_config = load_readiness_config(args.config)
    temporal_config = load_temporal_config(args.temporal_config)

    camera_dev = args.camera if args.camera is not None else m4_config.camera.device
    model_path = args.model if args.model is not None else m4_config.detector.model_path
    face_model_path = args.face_model if args.face_model is not None else m4_config.face_analyzer.model_path
    conf_thresh = args.conf if args.conf is not None else m4_config.detector.confidence_threshold

    print("==============================================================================")
    print(" Vision-Based First Contact — M5 Temporal Behaviour Estimation Runner")
    print("==============================================================================")
    print(f" Camera Device:       {camera_dev}")
    print(f" YOLO Model:          {model_path} (conf >= {conf_thresh:.2f})")
    print(f" Face Landmarker:     {face_model_path}")
    print(f" M5 Config:           {args.temporal_config}")
    print(f" History Window:      {temporal_config.history_window_seconds:.1f} s")
    print(f" Max Obs Gap:         {temporal_config.max_observation_gap_seconds:.2f} s")
    print(f" Init Requirement:    {temporal_config.initial_valid_observations} valid observations")
    print(f" Stationary Bounds:   < {temporal_config.stationary_velocity_threshold_px_s:.1f} px/s, < {temporal_config.stationary_displacement_threshold_px:.1f} px")
    print(f" Stability Duration:  {temporal_config.stability_duration_seconds:.1f} s (tolerance: {temporal_config.stability_violation_tolerance_seconds:.2f} s)")
    print(f" Headless Mode:       {args.headless}")
    print("------------------------------------------------------------------------------")

    # 2. Initialize Pipeline Components
    print("Initializing PersonDetector...")
    detector = PersonDetector(
        model_path=model_path,
        confidence_threshold=conf_thresh,
        device=args.device,
    )

    print("Initializing SinglePersonTracker (M3)...")
    tracker = SinglePersonTracker()

    print(f"Initializing FaceAnalyzer from '{face_model_path}'...")
    face_analyzer = FaceAnalyzer(
        model_path=face_model_path,
        min_face_width=m4_config.face_analyzer.min_face_width,
        min_face_height=m4_config.face_analyzer.min_face_height,
        min_detection_confidence=m4_config.face_analyzer.min_detection_confidence,
        min_tracking_confidence=m4_config.face_analyzer.min_tracking_confidence,
    )

    zone_evaluator = InteractionZoneEvaluator(
        vertices_normalized=m4_config.interaction_zone.vertices,
        enabled=m4_config.interaction_zone.enabled,
    )
    orient_estimator = OrientationEstimator(
        yaw_threshold=m4_config.orientation.yaw_threshold,
        pitch_threshold=m4_config.orientation.pitch_threshold,
        uncertain_multiplier=m4_config.orientation.uncertain_multiplier,
    )

    print("Initializing FrameObserver (M4)...")
    observer = FrameObserver(
        face_analyzer=face_analyzer,
        zone_evaluator=zone_evaluator,
        orientation_estimator=orient_estimator,
    )

    print("Initializing TemporalEstimator (M5)...")
    temporal_estimator = TemporalEstimator(
        config=temporal_config,
        zone_vertices_normalized=m4_config.interaction_zone.vertices,
    )

    # 3. Open Camera
    print(f"Opening camera stream on '{camera_dev}'...")
    cap, actual_w, actual_h, actual_fps = initialize_camera(
        device=camera_dev,
        width=m4_config.camera.width,
        height=m4_config.camera.height,
        fps=m4_config.camera.fps,
        fourcc=m4_config.camera.fourcc,
    )
    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, actual_w, actual_h)
        print("Live window active. Press 'q' or 'Esc' to exit.")

    # 4. Processing Metrics Buffers
    rolling_timestamps: deque[float] = deque(maxlen=60)
    yolo_latencies: list[float] = []
    tracker_latencies: list[float] = []
    observer_latencies: list[float] = []
    temporal_latencies: list[float] = []
    m5_hist_latencies: list[float] = []
    m5_motion_latencies: list[float] = []
    m5_stat_latencies: list[float] = []
    m5_app_latencies: list[float] = []
    m5_stab_latencies: list[float] = []
    cpu_samples: list[float] = []
    rss_samples: list[float] = []
    trajectory: deque[tuple[float, float]] = deque(maxlen=30)

    process = psutil.Process()
    try:
        process.cpu_percent()
    except Exception:
        pass

    total_frames = 0
    t_pipeline_start = time.perf_counter()

    # Precalculate zone pixel vertices and centroid
    zone_vertices_px = zone_evaluator.to_pixel_coords(actual_w, actual_h)
    try:
        zone_centroid_px = temporal_estimator.get_zone_centroid_px(actual_w, actual_h)
    except ValueError:
        zone_centroid_px = None

    try:
        while True:
            if args.max_frames is not None and total_frames >= args.max_frames:
                print(f"\nReached max-frames limit of {args.max_frames}. Exiting.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[WARNING] Camera stream returned empty frame.")
                break

            total_frames += 1
            t_frame = time.monotonic()
            rolling_timestamps.append(t_frame)
            current_app_fps = compute_application_fps(rolling_timestamps)

            # Stage 1: YOLO Person Detection (M1)
            t_yolo_0 = time.perf_counter()
            raw_detections = detector.detect_raw(frame)
            t_yolo_ms = (time.perf_counter() - t_yolo_0) * 1000.0
            yolo_latencies.append(t_yolo_ms)

            # Stage 2: Single-Person Tracking (M3)
            t_trk_0 = time.perf_counter()
            track = tracker.update(raw_detections, timestamp=t_frame)
            matched_det = tracker.last_matched_detection
            t_trk_ms = (time.perf_counter() - t_trk_0) * 1000.0
            tracker_latencies.append(t_trk_ms)

            # Stage 3: Frame-Level Readiness Observation (M4)
            t_obs_0 = time.perf_counter()
            frame_observation = observer.observe(frame, raw_detections, timestamp=t_frame)
            t_obs_ms = (time.perf_counter() - t_obs_0) * 1000.0
            observer_latencies.append(t_obs_ms)

            # Stage 4: Temporal Behaviour Estimation (M5)
            t_temp_0 = time.perf_counter()
            temporal_obs = temporal_estimator.estimate(
                track=track,
                frame_observation=frame_observation,
                matched_detection=matched_det,
                timestamp=t_frame,
            )
            t_temp_ms = (time.perf_counter() - t_temp_0) * 1000.0
            temporal_latencies.append(t_temp_ms)

            # Record M5 fine-grained component breakdown
            if temporal_estimator.last_timings is not None:
                lt = temporal_estimator.last_timings
                m5_hist_latencies.append(lt.history_ms)
                m5_motion_latencies.append(lt.motion_ms)
                m5_stat_latencies.append(lt.stationary_ms)
                m5_app_latencies.append(lt.approach_ms)
                m5_stab_latencies.append(lt.stability_ms)

            # Sample CPU and memory utilization
            try:
                cpu_samples.append(process.cpu_percent())
                rss_samples.append(process.memory_info().rss / (1024.0 * 1024.0))
            except Exception:
                pass

            # Update trajectory breadcrumbs for active observed track
            if track is not None and track.observed:
                trajectory.append(track.center)
            else:
                trajectory.clear()

            # Stage 5: Rendering & Display (if not headless)
            if not args.headless:
                render_temporal_overlays(
                    canvas=frame,
                    frame_observation=frame_observation,
                    temporal_obs=temporal_obs,
                    track=track,
                    zone_vertices_px=zone_vertices_px,
                    zone_centroid_px=zone_centroid_px,
                    trajectory=list(trajectory),
                    app_fps=current_app_fps,
                    yolo_ms=t_yolo_ms,
                    tracker_ms=t_trk_ms,
                    observer_ms=t_obs_ms,
                    temporal_ms=t_temp_ms,
                )
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\nExit key pressed by user.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        total_duration = time.perf_counter() - t_pipeline_start
        print("\nReleasing resources...")
        cap.release()
        detector.close()
        face_analyzer.close()
        cv2.destroyAllWindows()

        avg_fps = total_frames / total_duration if total_duration > 0 else 0.0

        def compute_stats(data: list[float]) -> tuple[float, float, float, float]:
            if not data:
                return (0.0, 0.0, 0.0, 0.0)
            arr = np.array(data)
            return (
                float(np.mean(arr)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 95)),
                float(np.percentile(arr, 99)),
            )

        print("\n==============================================================================")
        print(" M5 Demonstration Execution Summary & Performance Breakdown")
        print("==============================================================================")
        print(f" Total Frames Processed: {total_frames}")
        print(f" Total Elapsed Time:     {total_duration:.2f} s")
        print(f" Average Application FPS:{avg_fps:6.1f} FPS")
        print("------------------------------------------------------------------------------")
        print(" Pipeline Latency (ms):")
        print("   Component                     Avg       p50       p95       p99")
        breakdown_items = [
            ("M1 YOLOv8n Person Det", yolo_latencies),
            ("M3 Kalman Tracker", tracker_latencies),
            ("M4 Readiness Observer", observer_latencies),
            ("M5 Temporal Estimator", temporal_latencies),
            ("  -- History Management", m5_hist_latencies),
            ("  -- Motion Estimation", m5_motion_latencies),
            ("  -- Stationary Classifier", m5_stat_latencies),
            ("  -- Approach Estimator", m5_app_latencies),
            ("  -- Stability Timer", m5_stab_latencies),
        ]
        for name, lat_list in breakdown_items:
            mean_v, p50_v, p95_v, p99_v = compute_stats(lat_list)
            print(f"   {name:<26} {mean_v:7.2f}   {p50_v:7.2f}   {p95_v:7.2f}   {p99_v:7.2f}")
        print("------------------------------------------------------------------------------")
        if cpu_samples:
            mean_cpu = float(np.mean(cpu_samples))
            peak_cpu = float(np.max(cpu_samples))
            print(f" CPU Utilization:        Avg: {mean_cpu:5.1f}% | Peak: {peak_cpu:5.1f}%")
        if rss_samples:
            mean_rss = float(np.mean(rss_samples))
            peak_rss = float(np.max(rss_samples))
            print(f" Resident Memory (RSS):  Avg: {mean_rss:5.1f} MB | Peak: {peak_rss:5.1f} MB")
        print("==============================================================================")

    return 0


def main() -> None:
    """CLI entry point for M5 temporal behaviour demo."""
    args = parse_args()
    sys.exit(run_temporal_pipeline(args))


if __name__ == "__main__":
    main()
