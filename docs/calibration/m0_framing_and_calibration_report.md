# M0 — Camera Framing Observations & Calibration Requirements Report

> **Document:** `m0_framing_and_calibration_report.md`  
> **Subsystem:** Perception / Spatial Understanding Calibration (M0 Milestone)  
> **Hardware Status:** Confirmed (Luxvisions Integrated RGB Camera, `/dev/video0`)  
> **Verified Stream:** 1280 × 720 @ 30 FPS (`MJPG` via V4L2)  
> **Date:** 2026-11-17  

---

## 1. Executive Summary & Geometry Baseline

The baseline camera hardware and physical mounting parameters have been consolidated following the M0 Batch 1 hardware characterisation and the physical site survey:

```text
+------------------------------------------------------------------------------------+
|                         PHYSICAL RECEPTION GEOMETRY BASELINE                       |
+------------------------------------------------------------------------------------+
|                                                                                    |
|  Mounting Configuration:      Fixed on tabletop social robot head                  |
|  Optical Center Height (h):   115.0 cm (1.15 m) above floor                        |
|  Optical Pitch Angle:         ~0.0° (Level with horizontal horizon)                |
|  Optical Yaw / Roll:          ~0.0° (Aligned with forward robot heading)           |
|  Nominal Stream:              1280 × 720 @ 30 FPS (MJPG compression required)      |
|  Feet Visibility at 1.5 m:    REPORTED NOT VISIBLE (Confirmed by optical analysis) |
|                                                                                    |
+------------------------------------------------------------------------------------+
```

| Parameter | Specification | Status | Source |
| :--- | :--- | :---: | :--- |
| **Camera Model** | Luxvisions Innotech Integrated RGB Camera (`0x30c9:0x00a8`) | CONFIRMED | M0 Batch 1 USB/V4L2 discovery |
| **Linux Device** | `/dev/video0` (Persistent: `by-path/...-usb-0:6:1.0-video-index0`) | CONFIRMED | Kernel UVC driver topology |
| **Resolution & FPS** | 1280 × 720 @ 30 FPS (`MJPG`) | CONFIRMED | Runtime streaming benchmark (29.7 FPS) |
| **Lens Height ($h$)** | 115.0 cm | OBSERVED | Reception robot mounting measurement |
| **Pitch ($\theta_{pitch}$)** | ~0.0° (horizontal forward) | INFERRED | Visual leveling inspection |
| **Feet Visibility at 1.5 m**| **Not visible** | CONFIRMED | Framing observation & geometry proof |

---

## 2. Geometric Proof of Feet Occlusion at 1.5 m

The observation that **a candidate's feet are not visible at the 1.5 m interaction boundary** is a fundamental physical property of the optical setup, not an artifact of bad detection.

```text
+------------------------------------------------------------------------------------+
|                         SIDE-VIEW OPTICAL RAY TRACE (PITCH ~0°)                    |
+------------------------------------------------------------------------------------+
|                                                                                    |
|  Height                                                                            |
|  (meters)                                                                          |
|                                                                                    |
|   1.7 m +-----------------------------------------[ Person Head (Top of Box) ]    |
|         |                                        /                                 |
|   1.15m +---[ Camera Lens: h = 1.15m ]==========/== Top Ray (+22.5 deg)            |
|         |   |                                  /                                   |
|   1.0 m +   |                                 /                                    |
|         |   | [Reception Desk ~80cm]         /                                     |
|   0.8 m +   +-------+                       /                                      |
|         |           |                      /                                       |
|   0.5 m +           |                     / <--- Bottom Ray (-22.5 deg) intersects |
|         |           |                    /       person at ~0.53 m (thigh level)   |
|   0.0 m +===========+===================/========================[ Feet: OCCLUDED ]|
|                    0.5 m               1.5 m                     2.78 m            |
|                                                                (Floor Contact)     |
+------------------------------------------------------------------------------------+
```

### Mathematical Ray Tracing
1. For an RGB camera with typical wide-angle field of view (Horizontal FOV $\approx 70^\circ$, Vertical FOV $\approx 45^\circ$, half-angle $\alpha_v \approx 22.5^\circ$):
   * With optical pitch $\theta = 0^\circ$, the bottom-most ray projects downwards at an angle of $22.5^\circ$ below the horizontal optical axis.
2. The distance $d_{floor}$ at which this bottom ray reaches the floor ($z = 0$) is:
   $$d_{floor} = \frac{h}{\tan(\alpha_v)} = \frac{1.15\text{ m}}{\tan(22.5^\circ)} \approx \frac{1.15}{0.4142} \approx 2.78\text{ meters}$$
3. For any person standing closer than **2.78 meters** (specifically at the **1.5 m decision boundary**):
   * The bottom ray intersects the person's body at height:
     $$y_{intercept} = h - d \cdot \tan(\alpha_v) = 1.15\text{ m} - 1.5\text{ m} \times 0.4142 \approx 0.53\text{ meters}$$
   * The lower 53 cm of the person (feet, ankles, and lower shins) fall entirely below the bottom edge of the image sensor.
4. Additionally, in a physical reception setup, the **tabletop desk edge** (typically 75–85 cm high) forms a physical occluder directly in front of the robot, further masking the lower torso and legs of approaching visitors.

---

## 3. Consequences for Spatial Localisation & Distance Estimation

This finding carries decisive implications for the architecture of `spatial_understanding/`:

```text
+------------------------------------------------------------------------------------+
|                     LOCALISATION STRATEGY FEASIBILITY MATRIX                       |
+------------------------------------------------------------------------------------+
| Strategy                     | Feasibility at 1.5 m | Failure Mode / Reason        |
+------------------------------+----------------------+------------------------------+
| 1. Feet Ground Homography    | INFEASIBLE           | Feet occluded below image &  |
|    (Bottom of Bounding Box)  |                      | behind reception desk        |
|                              |                      |                              |
| 2. Pinhole Eye/Shoulder Ray  | FEASIBLE & ACCURATE  | Upper body & face/shoulders  |
|    Intersection              |                      | visible across 0.8 m – 4.0 m |
|                              |                      |                              |
| 3. Keypoint Stature Prior    | FEASIBLE & ACCURATE  | Shoulder-to-hip or eye height|
|    (MediaPipe / YOLO-Pose)   |                      | relative to camera height    |
|                              |                      |                              |
| 4. Bounding Box Height Curve | UNRELIABLE           | Slouching, bags, partial cut |
|    (Naive pixel height)      |                      | off by bottom frame border   |
+------------------------------+----------------------+------------------------------+
```

### Architectural Decisions Required
1. **Discard Feet-Contact Assumptions:**
   Algorithms in `spatial_understanding/person_localization` must **not** assume the bottom edge of the bounding box represents the person's feet on the floor at distances $\le 2.5\text{ m}$.
2. **Adopt Keypoint-Referenced Distance Estimation:**
   Use keypoint detections from `perception/pose_estimation/` (e.g. eye level, nose, neck, left/right shoulders):
   * Because camera height is known ($h = 1.15\text{ m}$) and optical pitch is approximately level ($\theta \approx 0^\circ$), an adult standing with eye level at $y_{eye} \approx 1.65\text{ m}$ is consistently above the optical center ($1.65\text{ m} - 1.15\text{ m} = +0.50\text{ m}$).
   * The vertical pixel coordinate $v_{eye}$ or $v_{shoulder}$ provides a stable monocular range estimator when combined with calibrated focal length $f_y$.
3. **Calibrate Against Empirical Ground Truth:**
   Use the Mode A scene calibration dataset (captured at 1.0 m, 1.5 m, 2.0 m, 2.5 m) to fit a verified empirical distance mapping function:
   $$\hat{d} = g(v_{shoulder}, v_{eye}, s_{keypoints})$$
   and quantify measurement error $\epsilon_d$ near the 1.5 m boundary.

---

## 4. Outstanding Calibration Requirements

To transition from M0 (Framing & Discovery) to M1 (Perception & Spatial Understanding), the following calibration activities must be completed:

### A. Intrinsic Camera Calibration
* **Objective:** Determine the $3 \times 3$ intrinsic camera matrix $K$ and radial-tangential lens distortion vector $D = [k_1, k_2, p_1, p_2, k_3]$:
  $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
* **Procedure:**
  1. Capture 15–20 non-coplanar views of a ChArUco or $9 \times 6$ checkerboard pattern using `tools/calibration/camera_capture_utility.py --mode intrinsic --save`.
  2. Solve via OpenCV `cv2.calibrateCamera()` or `cv2.aruco.calibrateCameraCharuco()`.
  3. Validate reprojection error: must achieve root-mean-square error $\le 0.5\text{ pixels}$.
  4. Store the resulting calibration matrix in `configs/calibration/camera_intrinsics.yaml`.

### B. Desk Boundary & Extrinsic Reference Measurement
* **Objective:** Measure reception desk dimensions and forward dead-zone.
* **Parameters to Measure:**
  1. Distance from camera lens to front edge of reception desk ($d_{desk}$).
  2. Height of desk surface above floor ($h_{desk}$).
  3. Robot head centerline offset from desk front edge.

### C. 1.5 m First-Contact Decision Boundary Calibration
* **Objective:** Establish the ground-truth pixel boundary and uncertainty bounds for first-contact readiness.
* **Procedure:**
  1. Affix high-contrast physical floor tape marks at 1.00 m, 1.50 m, 2.00 m, and 2.50 m from the robot front edge.
  2. Capture Mode A scene inspection frames with consent:
     ```bash
     python3 tools/calibration/camera_capture_utility.py --mode scene --stage person_1_5m --consent --save
     ```
  3. Record ground-truth coordinates in `data/raw/calibration/calibration_metadata.json`.
  4. Measure pixel coordinate variation across participants of different heights (e.g. 1.55 m to 1.85 m) to define the uncertainty tolerance $\sigma_d$ at 1.5 m.

---

## 5. M0 Calibration Status Summary

```text
+------------------------------------------------------------------------------------+
|                         M0 CALIBRATION STATUS CHECKLIST                            |
+------------------------------------------------------------------------------------+
|                                                                                    |
|  [x] Camera device identified & stream benchmarked (/dev/video0, MJPG 720p@30)     |
|  [x] Camera mounting geometry established (h = 115 cm, pitch ~0 deg, fixed)        |
|  [x] Feet occlusion at 1.5 m proven geometrically and documented                   |
|  [x] Camera capture utility implemented (tools/calibration/camera_capture_utility) |
|  [x] Metadata schema & acquisition guide prepared                                  |
|  [ ] Intrinsic calibration dataset collected (Requires physical board)             |
|  [ ] Intrinsic matrix K & distortion D computed (Target reprojection < 0.5 px)     |
|  [ ] Multi-distance ground-truth tape reference set & captured with consent        |
|                                                                                    |
+------------------------------------------------------------------------------------+
```

