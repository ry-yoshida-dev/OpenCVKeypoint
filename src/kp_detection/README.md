
# kp_detection

## Overview

Unified interface for keypoint detection using various OpenCV-backed algorithms. `KPDetectionParameters.build_detector()` (or method-specific parameter classes) produces detectors; results are typed as `KPDetectionResult` or `ArrayKPDetectionResult`.

## Components

| Component | Description |
| --------- | ----------- |
| [`detector.py`](./detector.py) | Detector abstract base |
| [`type_alias.py`](./type_alias.py) | `KPDetector`, `KPDetectionResult` aliases |
| [`parameter.py`](./parameter.py) | Detection parameters and `build_detector()` |
| [`result.py`](./result.py) | Result abstract base and related types |
| [`results/`](./results/) | `KPDetectionResult`, `ArrayKPDetectionResult` |
| [`method.py`](./method.py) | `KPDetectionMethod` enum |
| [`detectors/`](./detectors/README.md) | Per-algorithm detector classes |


## Supported detectors

Enum values (string labels) are shown in the **Method** column.  
Descriptor column reflects `KPDetectionMethod.has_descriptor` and binary/float support where applicable.

| Method | Type | Descriptor | Notes |
| ------ | ---- | ---------- | ----- |
| MSER | Blob | — | Maximally stable extremal regions |
| SimpleBlob | Blob | — | `cv2.SimpleBlobDetector` |
| AGAST | Corner | — | Adaptive/generic accelerated segment test |
| FAST | Corner | — | Features from accelerated segment test |
| Harris | Corner | — | Use `HarrisParameters` + `build_detector()` |
| ShiTomashi | Corner | — | goodFeaturesToTrack; use `ShiTomashiParameters` + `build_detector()` |
| AKAZE (MLDB) | Feature | Binary | Accelerated KAZE; `cv2.AKAZE_create()` default descriptor (MLDB) |
| KAZE | Feature | Float | Nonlinear scale space; float descriptor only |
| BRISK | Feature | Binary | Binary robust invariant scalable keypoints |
| ORB | Feature | Binary | Oriented FAST and rotated BRIEF |
| SIFT | Feature | Float | Scale-invariant feature transform |

Optional **BRIEF** post-descriptor (`KPDetectionParameters.is_brief_applied`) is only valid when `KPDetectionMethod.is_brief_supported()` is true (ORB, BRISK, AKAZE).

`KPDetectionParameters` also exposes `scale_factor` (default `1.0`), `interpolation` (`OpenCVInterpolationFlag.LINEAR`), and `image_scaler` (built once in `__post_init__`: identity when `scale_factor == 1.0`, otherwise `cv2.resize` with `fx`/`fy`).

Detection results (`KPDetectionResult`, `ArrayKPDetectionResult`) implement `scale_coordinates(factor)` to multiply coordinates in place (and `cv2.KeyPoint.size` where applicable). After detection on a rescaled image, call `scale_coordinates(1.0 / params.scale_factor)` to map back to the original image space.
