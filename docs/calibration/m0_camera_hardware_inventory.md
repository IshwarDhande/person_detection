# M0 Batch 1 — Embedded Camera Hardware Inventory Report

> **Document:** `m0_camera_hardware_inventory.md`  
> **Inspection Date (UTC):** 2026-11-16T12:18:10.775146+00:00  
> **Diagnostic Script Version:** 1.0.0  
> **Subsystem:** Perception / Embedded RGB Camera Interface  
> **Status:** Completed  

---

## 1. Inspection Summary

An automated hardware discovery and streaming benchmark was executed against all local video devices to identify and characterize the robot's embedded forward-facing camera.

```text
+------------------------------------------------------------------------------------+
|                             CAMERA HARDWARE TOPOLOGY                               |
+------------------------------------------------------------------------------------+
|                                                                                    |
|  Host System: Intel 13th Gen Core i5 (x86_64)                                      |
|  Linux Kernel: 7.0.0-42-generic                                            |
|                                                                                    |
|  [ USB 2.0 Host Controller: pci-0000:00:14.0-usb-0:6 ]                             |
|        |                                                                           |
|        +---> [ Luxvisions Innotech Integrated RGB Camera (ID 30c9:00a8) ]          |
|                    |                                                               |
|                    +---> /dev/video0 : Primary RGB Stream (MJPG/YUYV Capture)      |
|                    +---> /dev/video1 : UVC Metadata Capture (UVCH Header)          |
|                    +---> /dev/video2 : Auxiliary IR Stream (GREY 640x360 Capture)  |
|                    +---> /dev/video3 : UVC Metadata Capture (UVCH Header)          |
|                                                                                    |
+------------------------------------------------------------------------------------+
```

| Parameter | Value | Status |
| :--- | :--- | :--- |
| Hostname | `01HW2570473` | CONFIRMED |
| Operating System | `Ubuntu 24.04.5 LTS` | CONFIRMED |
| Kernel Version | `7.0.0-42-generic` | CONFIRMED |
| Python Version | `3.12.3` | CONFIRMED |
| OpenCV Version | `5.0.0` | CONFIRMED |
| Diagnostic Tool Version | `1.0.0` | CONFIRMED |

### Available System Utilities

| Utility | Present | Path | Version |
| :--- | :---: | :--- | :--- |
| `v4l2-ctl` | Yes | `/usr/bin/v4l2-ctl` | `v4l2-ctl 1.26.1` |
| `media-ctl` | Yes | `/usr/bin/media-ctl` | `media-ctl 1.26.1` |
| `udevadm` | Yes | `/usr/bin/udevadm` | `255` |
| `lsusb` | Yes | `/usr/bin/lsusb` | `lsusb (usbutils) 017` |
| `lspci` | Yes | `/usr/bin/lspci` | `lspci version 3.10.0` |

---

## 2. Camera Device Identification & Classification

Discovered **4** device nodes under `/dev/video*`. Through V4L2 capability flags and sysfs metadata, each device was conclusively classified:

| Device Path | Classification | Driver | Bus Address | Permissions |
| :--- | :--- | :--- | :--- | :--- |
| `/dev/video0` | **RGB Video Capture Device** | `uvcvideo` | `usb-0000:00:14.0-6` | crw-rw---- (0660) [User ACL RW] |
| `/dev/video1` | **Video Capture Device** | `uvcvideo` | `usb-0000:00:14.0-6` | crw-rw---- (0660) [User ACL RW] |
| `/dev/video2` | **Greyscale/IR Video Capture Device** | `uvcvideo` | `usb-0000:00:14.0-6` | crw-rw---- (0660) [User ACL RW] |
| `/dev/video3` | **Video Capture Device** | `uvcvideo` | `usb-0000:00:14.0-6` | crw-rw---- (0660) [User ACL RW] |

### Hardware Descriptors (Primary RGB Camera: `/dev/video0`)

- **Manufacturer / Serial:** `Luxvisions Innotech Limited` (Product ID: `0x00a8`, Vendor ID: `0x30c9`)
- **Device String:** `Integrated RGB Camera: Integrat`
- **Bus Connection:** Internal USB 2.01 (`usb-0000:00:14.0-6`)
- **Kernel Driver:** `uvcvideo` (Linux standard UVC driver v7.0.14)
- **UVC Specification:** UVC 1.50 compliant
- **Symlinks:**
  - `/dev/v4l/by-id/usb-8SSC21K12297V1SR52D22YG_Integrated_RGB_Camera_01.00.00-video-index0`
  - `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0`

---

## 3. Supported Camera Capabilities & Formats

### Primary RGB Device (`/dev/video0`)

The primary sensor exposes two capture formats with discrete resolution profiles:

| Format Code | Compression | Resolution | Max FPS | USB 2.0 Bandwidth Feasibility |
| :--- | :--- | :--- | :---: | :--- |
| `MJPG` | Motion-JPEG (Compressed) | **1920 × 1080** (1080p) | 30.0 FPS | Highly feasible |
| `MJPG` | Motion-JPEG (Compressed) | **1280 × 720** (720p) | **30.0 FPS** | **Optimal / Target Specification** |
| `MJPG` | Motion-JPEG (Compressed) | 960 × 540 | 30.0 FPS | Feasible |
| `MJPG` | Motion-JPEG (Compressed) | 640 × 480 | 30.0 FPS | Feasible |
| `MJPG` | Motion-JPEG (Compressed) | 640 × 360 | 30.0 FPS | Feasible |
| `YUYV` | Uncompressed (4:2:2) | 1920 × 1080 | 5.0 FPS | Bottlenecked by USB 2.0 (~166 MB/s uncompressed) |
| `YUYV` | Uncompressed (4:2:2) | **1280 × 720** | **10.0 FPS** | **Bottlenecked by USB 2.0 (~55 MB/s uncompressed)** |
| `YUYV` | Uncompressed (4:2:2) | 640 × 480 | 30.0 FPS | Feasible uncompressed |
| `YUYV` | Uncompressed (4:2:2) | 640 × 360 | 30.0 FPS | Feasible uncompressed |

> [!IMPORTANT]
> **Bandwidth Insight:** Uncompressed YUYV at 1280×720 cannot exceed 10 FPS over this internal USB 2.0 connection. To achieve 30 FPS at 720p, the pipeline **must request the MJPG pixel format** from V4L2 / OpenCV.

### Exposed Hardware Camera Controls

| Control | Type | Hardware Range | Default | Current Value |
| :--- | :---: | :---: | :---: | :---: |
| `brightness` | Integer | 0 – 255 | 128 | 128 |
| `contrast` | Integer | 0 – 100 | 32 | 32 |
| `saturation` | Integer | 0 – 100 | 64 | 64 |
| `hue` | Integer | -180 – 180 | 0 | 0 |
| `white_balance_automatic` | Boolean | 0 or 1 | 1 (Enabled) | 1 |
| `white_balance_temperature` | Integer | 2800 – 6500 K | 4600 K | 4600 K |
| `gamma` | Integer | 90 – 150 | 120 | 120 |
| `power_line_frequency` | Menu | 0 (Off), 1 (50Hz), 2 (60Hz) | 1 (50 Hz) | 1 (50 Hz) |
| `sharpness` | Integer | 0 – 7 | 3 | 3 |
| `backlight_compensation` | Integer | 0 – 2 | 1 | 1 |
| `auto_exposure` | Menu | 1 (Manual), 3 (Aperture Priority) | 3 | 3 |
| `exposure_time_absolute` | Integer | 2 – 1250 | 156 | 156 |
| `exposure_dynamic_framerate` | Boolean | 0 or 1 | 0 | 1 |

---

## 4. Runtime Verification & Benchmark Results

An OpenCV V4L2 streaming test was executed for `/dev/video0` configured to `1280x720 @ 30 FPS` with `MJPG` compression:

| Metric | Requested | Observed Result | Status |
| :--- | :--- | :--- | :---: |
| Camera Open / Init | `cv2.CAP_V4L2` | Successfully opened `/dev/video0` | PASS |
| Returned Frame Dimensions | `1280 × 720` | `1280x720` | PASS |
| Channel Count & Dtype | `3 channels, uint8` | `3 channels (BGR), uint8` | PASS |
| Test Duration | `5.00 s` | `5.019 s` | PASS |
| Total Frames Acquired | Target ~150 | `149` | PASS |
| Failed / Corrupt Reads | `0` | `0` | PASS |
| Measured Frame Rate | `30.00 FPS` | **`29.69 FPS`** | **PASS** |
| Mean Inter-frame Interval | `33.33 ms` | `33.69 ms` | PASS |
| Inter-frame Jitter (Std Dev) | `< 5.0 ms` | `2.78 ms` | PASS |
| Stream Stability | Continuous | Stable, zero drops | PASS |

---

## 5. Verification of Assumed Specification (1280×720 @ 30 FPS)

### Classification: **VERIFIED RUNTIME CONFIGURATION & CONFIRMED HARDWARE CAPABILITY**

```text
+------------------------------------------------------------------------------------+
|                 ASSUMED SPECIFICATION VERIFICATION BREAKDOWN                      |
+------------------------------------------------------------------------------------+
|                                                                                    |
|  Assumed Parameter        Hardware Capability           Runtime Observed Status    |
|  -----------------        -------------------           -----------------------    |
|  Resolution: 1280x720  => Discrete MJPG & YUYV entry => (720, 1280, 3) confirmed   |
|  Frame Rate: 30 FPS    => Discrete 0.033s (30 fps)   => Measured: 29.69  FPS   |
|                                                                                    |
|  CRITICAL ARCHITECTURAL REQUIREMENT:                                               |
|  The pipeline MUST explicitly request cv2.CAP_PROP_FOURCC = 'MJPG'.                |
|  If defaulted to uncompressed YUYV, the driver throttles to 10 FPS over USB 2.0.   |
|                                                                                    |
+------------------------------------------------------------------------------------+
```

**Evidence Details:**
1. V4L2 capability table directly reports discrete interval `0.033s (30.000 fps)` for `1280x720` under `MJPG` format.
2. Runtime benchmark captured 149 frames in 5.019 seconds with **zero failed reads** and a measured frame rate of **29.69 FPS**.
3. Frame buffers returned by OpenCV match the exact dimensions `(720, 1280, 3)` in `uint8` color format.

---

## 6. Unknown Physical & Optical Properties

The following parameters **cannot be discovered via software inspection** and remain strictly `UNKNOWN`. They require physical measurement and optical calibration:

| Unknown Property | Impact on Pipeline | Required Next Action |
| :--- | :--- | :--- |
| **Camera Height above Floor ($h$)** | Critical for ground-plane projection and metric distance calculation | Physical measurement with laser/tape measure on reception desk |
| **Optical Pitch Angle ($\theta_{pitch}$)** | Governs horizontal distance estimation near 1.5 m interaction zone | Measure physical mounting angle + solve via floor homography |
| **Optical Yaw & Roll ($\theta_{yaw}, \theta_{roll}$)** | Governs left/right corridor symmetry and horizon tilt | Ground-plane calibration with ChArUco / AprilTag pattern |
| **Focal Length ($f_x, f_y$) & Optical Center ($c_x, c_y$)** | Determines pixel-to-angle conversion and 3D ray projection | Pinhole intrinsic calibration (OpenCV `calibrateCamera`) |
| **Lens Distortion ($k_1, k_2, p_1, p_2, k_3$)** | Straight lines appear curved; distorts peripheral distances | Radial-tangential distortion model optimization |
| **Field of View (H-FOV, V-FOV)** | Determines detection coverage at desk boundaries | Derived from intrinsic matrix ($2 \arctan(w / 2f_x)$) |
| **Head Actuation / Mobility** | If head moves, extrinsic calibration parameters become dynamic | Hardware inspection of robot neck servos / joint locks |

---

## 7. Issues and Recommendations

### Key Findings & Recommendations
1. **Enforce MJPG in Camera Adapter:**
   The camera adapter in `perception/camera/` must explicitly set `CAP_PROP_FOURCC` to `cv2.VideoWriter_fourcc(*'MJPG')` before configuring frame dimensions. Relying on default uncompressed YUYV will limit the stream to 10 FPS.
2. **Ignore Auxiliary Device Nodes (`/dev/video1`, `/dev/video2`, `/dev/video3`):**
   - `/dev/video1` and `/dev/video3` are metadata nodes and cannot capture frames.
   - `/dev/video2` is a low-resolution (640×360) IR camera. Unless multimodal IR night-vision is planned, `perception/camera/` should bind specifically to `/dev/video0` or by persistent path `/dev/v4l/by-id/usb-8SSC21K12297V1SR52D22YG_Integrated_RGB_Camera_01.00.00-video-index0`.
3. **Power Line Frequency Filter:**
   The driver is currently set to `power_line_frequency = 1 (50 Hz)`. If operating in a 60 Hz electrical environment (e.g. North America / parts of Asia), this should be adjusted to prevent 60 Hz fluorescent lighting flicker.

---

## 8. M0 Batch 1 Conclusion

| Requirement | Finding | Status |
| :--- | :--- | :---: |
| Embedded camera discovered | Discovered `/dev/video0` (Luxvisions Integrated RGB Camera) | **PASS** |
| Camera model identified | Luxvisions Innotech Integrated RGB Camera (`0x30c9:0x00a8`) | **PASS** |
| Connection interface identified | Internal USB 2.0 (`usb-0000:00:14.0-6`) via `uvcvideo` | **PASS** |
| Supported resolutions identified | Native 1080p, 720p, 540p, 480p, 360p under MJPG & YUYV | **PASS** |
| FPS capability identified | 30.0 FPS under MJPG across all resolutions | **PASS** |
| Runtime streaming verified | 720p @ 30.0 FPS verified with 0 dropped frames and low jitter | **PASS** |

> **Overall M0 Batch 1 Verdict: PASS**  
> The embedded camera hardware has been fully characterized. The project can safely proceed to physical site measurements and camera intrinsic/extrinsic calibration.
