"""Standalone Live First-Contact Readiness Demonstration Runner for Milestone M6.

Integrates camera capture, M1 YOLOv8n detection, M3 Kalman tracking, M4 frame-level
readiness, M5 temporal behaviour estimation, distance-provider observations, and
M6 readiness state machine into an interactive live demonstration.

Visual Overlays & HUD:
  - Interaction Zone Polygon: Cyan boundary line & centroid crosshair
  - Person Bounding Box:
      * Bright Green if READY
      * Amber / Yellow if PENDING
      * Cyan if NOT_READY and valid
      * Magenta if unobserved / coasting
  - MOCK DISTANCE Badge: Amber notification box when using mock distance
  - 7-Predicate Live Checklist: Real-time display of all 7 conditions (P, B, A, Z, D, S, O)
  - Readiness State Banner: NOT_READY, PENDING (elapsed/2.00s), or READY (one-shot event)
  - Active Reason Codes: Explaining any unsatisfied conditions
  - Performance HUD & Breakdown:
      * FPS, latency percentiles across M1-M6, CPU and memory utilization
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

from perception.contact.contact_config import ContactConfig, load_contact_config
from perception.contact.contact_types import (
    DistanceObservation,
    DistanceSource,
    FirstContactReadyEvent,
    FirstContactResult,
    ReadinessState,
    ReasonCode,
)
from perception.contact.distance_provider import (
    DistanceProvider,
    MockDistanceProvider,
    create_distance_provider,
)
from perception.contact.readiness_engine import ReadinessEngine
from perception.detection.person_detector import PersonDetector
from perception.readiness.face_analyzer import FaceAnalyzer
from perception.readiness.frame_observer import FrameObserver
from perception.readiness.interaction_zone import InteractionZoneEvaluator
from perception.readiness.orientation_estimator import OrientationEstimator
from perception.readiness.readiness_config import ReadinessConfig, load_readiness_config
from perception.temporal.temporal_config import TemporalConfig, load_temporal_config
from perception.temporal.temporal_estimator import TemporalEstimator
from perception.tracking.single_person_tracker import SinglePersonTracker, Track

DEFAULT_M4_CONFIG_PATH = "configs/m4/stationary_readiness.yaml"
DEFAULT_M5_CONFIG_PATH = "configs/m5/temporal_behaviour.yaml"
DEFAULT_M6_CONFIG_PATH = "configs/m6/first_contact.yaml"
WINDOW_NAME = "Vision-Based First Contact — M6 Single-Person Readiness Pipeline"

# Color Palette (BGR)
COLOR_ZONE = (255, 255, 0)          # Cyan for interaction zone polygon & centroid
COLOR_READY = (0, 255, 0)           # Bright Green for READY state
COLOR_PENDING = (0, 215, 255)       # Amber / Yellow for PENDING state
COLOR_NOT_READY = (200, 200, 200)   # Gray / Cyan for NOT_READY state
COLOR_COASTING = (255, 0, 255)      # Magenta for coasting / predicted boxes
COLOR_TEXT_BG = (20, 20, 20)        # Dark charcoal for HUD backplate
COLOR_TEXT_FG = (255, 255, 255)     # White text
COLOR_VALID = (0, 255, 0)           # Green
COLOR_INVALID = (0, 0, 255)         # Red
COLOR_BADGE_BG = (0, 140, 255)      # Orange badge for mock mode


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate CLI arguments for the M6 first-contact runner."""
    parser = argparse.ArgumentParser(
        description="M6 Live First-Contact Readiness Demonstration Runner"
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
        "--contact-config",
        type=str,
        default=DEFAULT_M6_CONFIG_PATH,
        help=f"Path to M6 YAML config file (default: {DEFAULT_M6_CONFIG_PATH})",
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
        "--distance",
        type=float,
        default=None,
        help="Override mock distance in metres (e.g. 1.2)",
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
    if parsed.distance is not None and parsed.distance < 0.0:
        parser.error(f"--distance must be non-negative, got {parsed.distance}")
    if parsed.max_frames is not None and parsed.max_frames <= 0:
        parser.error(f"--max-frames must be positive, got {parsed.max_frames}")

    return parsed


def initialize_camera(
    device: str | int,
    width: int,
    height: int,
    fps: float,
    fourcc: str,
) -> tuple[cv2.VideoCapture, int, int, float]:
    """Open camera and negotiate format settings."""
    dev_index: str | int = int(device) if str(device).isdigit() else device
    cap = cv2.VideoCapture(dev_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera on device '{device}'")

    if len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if actual_fps <= 0.0:
        actual_fps = fps

    return cap, actual_w, actual_h, actual_fps


def compute_application_fps(rolling_timestamps: deque[float]) -> float:
    """Calculate steady-state rolling application throughput."""
    if len(rolling_timestamps) < 2:
        return 0.0
    dt = rolling_timestamps[-1] - rolling_timestamps[0]
    return (len(rolling_timestamps) - 1) / dt if dt > 0 else 0.0


def draw_text_with_background(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.45,
    color: tuple[int, int, int] = COLOR_TEXT_FG,
    thickness: int = 1,
    bg_color: tuple[int, int, int] = COLOR_TEXT_BG,
    padding: int = 3,
) -> None:
    """Render antialiased text with a dark charcoal backplate."""
    x, y = pos
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1 = max(0, x - padding)
    y1 = max(0, y - text_h - padding)
    x2 = min(img.shape[1] - 1, x + text_w + padding)
    y2 = min(img.shape[0] - 1, y + baseline + padding)

    sub_img = img[y1:y2, x1:x2]
    rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
    cv2.addWeighted(rect, 0.75, sub_img, 0.25, 0.0, sub_img)

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def render_contact_overlays(
    canvas: np.ndarray,
    result: FirstContactResult,
    track: Optional[Track],
    zone_vertices_px: list[tuple[int, int]],
    zone_centroid_px: Optional[tuple[float, float]],
    app_fps: float,
    yolo_ms: float,
    tracker_ms: float,
    observer_ms: float,
    temporal_ms: float,
    contact_ms: float,
    distance_obs: Optional[DistanceObservation],
) -> None:
    """Render full diagnostic visual overlays, checklists, and state banners."""
    h_canvas, w_canvas = canvas.shape[:2]

    # 1. Draw Interaction Zone Polygon
    if len(zone_vertices_px) >= 3:
        pts = np.array(zone_vertices_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=COLOR_ZONE, thickness=2, lineType=cv2.LINE_AA)

    # 2. Draw Zone Centroid Crosshair
    if zone_centroid_px is not None:
        zx, zy = int(round(zone_centroid_px[0])), int(round(zone_centroid_px[1]))
        cv2.drawMarker(canvas, (zx, zy), COLOR_ZONE, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    # 3. Draw Tracked Person Bounding Box
    if track is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox_xyxy]
        if not track.observed:
            box_color = COLOR_COASTING
        elif result.state == ReadinessState.READY:
            box_color = COLOR_READY
        elif result.state == ReadinessState.PENDING:
            box_color = COLOR_PENDING
        else:
            box_color = COLOR_NOT_READY

        thickness = 3 if result.state == ReadinessState.READY else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness, lineType=cv2.LINE_AA)

        # Label above bounding box
        label_txt = f"ID:{track.track_id} | {result.state.value}"
        draw_text_with_background(canvas, label_txt, (x1, max(18, y1 - 6)), scale=0.45, color=box_color)

    # 4. Render Mock Distance Badge (if MOCK source)
    if distance_obs is not None and distance_obs.source == DistanceSource.MOCK:
        badge_txt = f"[!] MOCK DISTANCE ({distance_obs.distance_m:.1f}m)"
        draw_text_with_background(
            canvas,
            badge_txt,
            (w_canvas - 230, 25),
            scale=0.48,
            color=(0, 255, 255),
            bg_color=(0, 60, 120),
        )

    # 5. Render HUD Panel on Left Side
    hud_x = 12
    hud_y = 22
    line_h = 17

    # Section A: Header & System Throughput
    draw_text_with_background(
        canvas,
        "=== M6 FIRST-CONTACT READINESS PIPELINE ===",
        (hud_x, hud_y),
        scale=0.5,
        color=(0, 255, 255),
    )
    hud_y += line_h

    lat_txt = f"FPS: {app_fps:4.1f} | Latency: {yolo_ms:.1f}ms Y, {tracker_ms:.1f}ms Trk, {observer_ms:.1f}ms M4, {temporal_ms:.1f}ms M5, {contact_ms:.2f}ms M6"
    draw_text_with_background(canvas, lat_txt, (hud_x, hud_y), scale=0.38, color=(200, 200, 200))
    hud_y += line_h + 3

    # Section B: Readiness State Banner
    v_id = result.visit_id if result.visit_id is not None else "-"
    t_id = result.track_id if result.track_id is not None else "-"
    state_str = result.state.value

    if result.state == ReadinessState.READY:
        banner_txt = f"STATE: [ READY ] | Visit: {v_id} | Track: {t_id}"
        banner_col = COLOR_READY
    elif result.state == ReadinessState.PENDING:
        banner_txt = f"STATE: [ PENDING ] Elapsed: {result.elapsed_eligibility:4.2f}s/2.00s | Visit: {v_id}"
        banner_col = COLOR_PENDING
    else:
        banner_txt = f"STATE: [ NOT_READY ] | Visit: {v_id} | Track: {t_id}"
        banner_col = COLOR_NOT_READY

    draw_text_with_background(canvas, banner_txt, (hud_x, hud_y), scale=0.48, color=banner_col)
    hud_y += line_h + 3

    # Section C: 7-Predicate Checklist
    preds = result.predicates
    draw_text_with_background(canvas, "-- Seven Required Readiness Predicates --", (hud_x, hud_y), scale=0.42, color=(220, 220, 220))
    hud_y += line_h

    def _fmt_p(flag: bool, label: str) -> tuple[str, tuple[int, int, int]]:
        sym = "[+]" if flag else "[-]"
        col = COLOR_VALID if flag else COLOR_INVALID
        return f" {sym} {label}", col

    checklist = [
        _fmt_p(preds.p_presence, "P: Person Presence Valid"),
        _fmt_p(preds.b_bbox, "B: BBox Matched to Active Track"),
        _fmt_p(preds.a_approach, "A: Confirmed Approach / Init Presence"),
        _fmt_p(preds.z_zone, "Z: Interaction Zone (INSIDE)"),
        _fmt_p(preds.d_distance, f"D: Distance Valid (<= 1.5m)"),
        _fmt_p(preds.s_stationary, "S: Stationary Behaviour (< 20px/s)"),
        _fmt_p(preds.o_orientation, "O: Facing Robot (< yaw threshold)"),
    ]

    for item_txt, col in checklist:
        draw_text_with_background(canvas, item_txt, (hud_x, hud_y), scale=0.40, color=col)
        hud_y += line_h

    # Section D: Reasons when NOT_READY
    if result.state == ReadinessState.NOT_READY and result.reasons:
        hud_y += 3
        reasons_str = ", ".join(r.value for r in result.reasons[:3])
        draw_text_with_background(canvas, f"Unsatisfied: {reasons_str}", (hud_x, hud_y), scale=0.40, color=(0, 165, 255))
        hud_y += line_h

    # Section E: Event Notification Banner
    if result.event is not None:
        evt_txt = f"*** EVENT EMITTED: {result.event.event_type} (id={result.event.event_id}) ***"
        draw_text_with_background(
            canvas,
            evt_txt,
            (hud_x, h_canvas - 25),
            scale=0.52,
            color=(0, 255, 0),
            bg_color=(0, 80, 0),
        )


def run_first_contact_pipeline(args: argparse.Namespace) -> int:
    """Run full live capture, M1 detection, M3 tracking, M4 readiness, M5 temporal, and M6 decision."""
    # 1. Load Configurations
    m4_config = load_readiness_config(args.config)
    temporal_config = load_temporal_config(args.temporal_config)
    contact_config = load_contact_config(args.contact_config)

    # Apply overrides
    camera_dev = args.camera if args.camera is not None else m4_config.camera.device
    model_path = args.model if args.model is not None else m4_config.detector.model_path
    face_model_path = args.face_model if args.face_model is not None else m4_config.face_analyzer.model_path
    conf_thresh = args.conf if args.conf is not None else m4_config.detector.confidence_threshold

    if args.distance is not None:
        contact_config = ContactConfig(
            distance_max_m=contact_config.distance_max_m,
            distance_mode=contact_config.distance_mode,
            mock_distance_m=args.distance,
            eligibility_duration_seconds=contact_config.eligibility_duration_seconds,
            tracking_loss_timeout_seconds=contact_config.tracking_loss_timeout_seconds,
            observation_freshness_seconds=contact_config.observation_freshness_seconds,
            require_metric_distance_in_production=contact_config.require_metric_distance_in_production,
        )

    print("==============================================================================")
    print(" Vision-Based First Contact — M6 Complete Single-Person Pipeline Runner")
    print("==============================================================================")
    print(f" Camera Device:       {camera_dev}")
    print(f" YOLO Model:          {model_path} (conf >= {conf_thresh:.2f})")
    print(f" Face Landmarker:     {face_model_path}")
    print(f" Distance Mode:       {contact_config.distance_mode.upper()} (threshold <= {contact_config.distance_max_m:.2f} m)")
    print(f" Eligibility Window:  {contact_config.eligibility_duration_seconds:.1f} s continuous")
    print(f" Tracking Loss Grace: {contact_config.tracking_loss_timeout_seconds:.1f} s")
    print(f" Headless Mode:       {args.headless}")
    print("------------------------------------------------------------------------------")

    # 2. Initialize Pipeline Components
    print("Initializing PersonDetector (M1)...")
    detector = PersonDetector(
        model_path=model_path,
        confidence_threshold=conf_thresh,
        device=args.device,
    )

    print("Initializing SinglePersonTracker (M3)...")
    tracker = SinglePersonTracker()

    print(f"Initializing FaceAnalyzer (M4) from '{face_model_path}'...")
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

    print("Initializing DistanceProvider (M6)...")
    distance_provider = create_distance_provider(contact_config)

    print("Initializing ReadinessEngine (M6)...")
    readiness_engine = ReadinessEngine(
        config=contact_config,
        distance_provider=distance_provider,
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

    # 4. Processing Buffers & Profiling
    rolling_timestamps: deque[float] = deque(maxlen=60)
    yolo_latencies: list[float] = []
    tracker_latencies: list[float] = []
    observer_latencies: list[float] = []
    temporal_latencies: list[float] = []
    contact_latencies: list[float] = []
    cpu_samples: list[float] = []
    rss_samples: list[float] = []

    process = psutil.Process()
    try:
        process.cpu_percent()
    except Exception:
        pass

    total_frames = 0
    emitted_events_count = 0
    t_pipeline_start = time.perf_counter()

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
                print("\n[WARNING] Camera stream returned empty frame / read failure.")
                readiness_engine.reset()
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

            # Stage 5: Distance Observation & M6 Readiness Engine
            t_m6_0 = time.perf_counter()
            tid = track.track_id if track is not None else None
            dist_obs = distance_provider.get_distance(
                timestamp=t_frame,
                bbox_xyxy=track.bbox_xyxy if track is not None else None,
                track_id=tid,
            )
            result = readiness_engine.evaluate(
                track=track,
                frame_observation=frame_observation,
                temporal_observation=temporal_obs,
                matched_detection=matched_det,
                distance_observation=dist_obs,
                timestamp=t_frame,
            )
            t_m6_ms = (time.perf_counter() - t_m6_0) * 1000.0
            contact_latencies.append(t_m6_ms)

            if result.event is not None:
                emitted_events_count += 1
                print(f"\n[FIRST_CONTACT_READY EVENT] Frame {total_frames}: {result.event.to_dict()}")

            # Resource utilization sampling
            try:
                cpu_samples.append(process.cpu_percent())
                rss_samples.append(process.memory_info().rss / (1024.0 * 1024.0))
            except Exception:
                pass

            # Rendering
            if not args.headless:
                render_contact_overlays(
                    canvas=frame,
                    result=result,
                    track=track,
                    zone_vertices_px=zone_vertices_px,
                    zone_centroid_px=zone_centroid_px,
                    app_fps=current_app_fps,
                    yolo_ms=t_yolo_ms,
                    tracker_ms=t_trk_ms,
                    observer_ms=t_obs_ms,
                    temporal_ms=t_temp_ms,
                    contact_ms=t_m6_ms,
                    distance_obs=dist_obs,
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
        readiness_engine.reset()
        cap.release()
        detector.close()
        face_analyzer.close()
        cv2.destroyAllWindows()

        avg_fps = total_frames / total_duration if total_duration > 0 else 0.0

        def compute_stats(data: list[float]) -> tuple[float, float, float, float]:
            if not data:
                return 0.0, 0.0, 0.0, 0.0
            arr = np.array(data)
            return (
                float(np.mean(arr)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 95)),
                float(np.percentile(arr, 99)),
            )

        print("\n==============================================================================")
        print(" M6 First-Contact Pipeline Execution Summary & Performance Breakdown")
        print("==============================================================================")
        print(f" Total Frames Processed: {total_frames}")
        print(f" Total Elapsed Time:     {total_duration:.2f} s")
        print(f" Average Application FPS:{avg_fps:6.1f} FPS")
        print(f" First Contact Events:   {emitted_events_count}")
        print("------------------------------------------------------------------------------")
        print(" Pipeline Latency (ms):")
        print("   Component                     Avg       p50       p95       p99")
        breakdown_items = [
            ("M1 YOLOv8n Person Det", yolo_latencies),
            ("M3 Kalman Tracker", tracker_latencies),
            ("M4 Readiness Observer", observer_latencies),
            ("M5 Temporal Estimator", temporal_latencies),
            ("M6 Readiness Engine", contact_latencies),
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
    """CLI entry point for M6 demo runner."""
    args = parse_args()
    sys.exit(run_first_contact_pipeline(args))


if __name__ == "__main__":
    main()

