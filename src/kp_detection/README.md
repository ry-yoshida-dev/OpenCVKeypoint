
# kp_detection

## Overview

Unified interface for keypoint detection using various OpenCV-backed algorithms. `KPDetectionParameters.build_detector()` (or method-specific parameter classes) produces detectors; results are typed as `KPDetectionResult` or `ArrayKPDetectionResult`.

## Components

| Component | Description |
| --------- | ----------- |
| [`detector.py`](./detector.py) | Detector abstract base |
| [`type_alias.py`](./type_alias.py) | `KPDetector` type alias |
| [`parameter.py`](./parameter.py) | Detection parameters and `build_detector()` |
| [`result.py`](./result.py) | Result abstract base and related types |
| [`results/`](./results/) | `KPDetectionResult`, `ArrayKPDetectionResult` |
| [`method.py`](./method.py) | `KPDetectionMethod` enum |
| [`detectors/`](./detectors/README.md) | Per-algorithm detector classes |

## Examples

### Standard feature detector (SIFT, ORB, …)

```python
import cv2

from kp_detection import KPDetectionMethod, KPDetectionParameters, KPDetectionResult
from kp_detection.detectors import StandardKPDetector

image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("image.png was not found")

params = KPDetectionParameters(method=KPDetectionMethod.SIFT, scale_factor=0.5)
detector = StandardKPDetector(params=params)

result: KPDetectionResult = detector.detect(image)
descriptors = result.descriptors
assert descriptors is not None
print(f"keypoints: {len(result)}, descriptors: {descriptors.shape}")
```

`KPDetectionParameters.build_detector()` and `method.detector_class` return `KPDetector`, so `detect()` is typed as `DetectionResultUnion`. Instantiate a concrete detector class (for example `StandardKPDetector`) when you need a specific result type such as `KPDetectionResult`.

### Shi–Tomasi corners (`ArrayKPDetectionResult`)

Harris and Shi–Tomasi require method-specific parameter classes; call `build_detector()` on those classes.

```python
import cv2

from kp_detection import ArrayKPDetectionResult, ShiTomashiParameters

image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("image.png was not found")

detector = ShiTomashiParameters(max_corners=200, min_distance=5).build_detector()
result: ArrayKPDetectionResult = detector.detect(image)

print(f"corners: {len(result)}")
print(f"x range: {result.x.min():.1f} – {result.x.max():.1f}")
print(f"y range: {result.y.min():.1f} – {result.y.max():.1f}")
```

### Harris corners

```python
import cv2

from kp_detection import HarrisParameters, KPDetectionResult

image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("image.png was not found")

detector = HarrisParameters(corner_th=0.02).build_detector()
result: KPDetectionResult = detector.detect(image)
print(f"keypoints: {len(result)}")
```


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

`KPDetectionParameters` also exposes `scale_factor` (default `1.0`), `interpolation` (`OpenCVInterpolationFlag.LINEAR`), `image_scaler`, and `mask_rescaler` (both built once in `__post_init__`: identity when `scale_factor == 1.0`; otherwise `cv2.resize` with `fx`/`fy`. Masks use `INTER_NEAREST`).

Detectors that run on a scaled working image pass `_scaled_detection_mask(mask)` into OpenCV. `FASTDetector` applies the original-space mask after `_remap_result_to_original_coordinates` via `apply_mask`. `AGASTDetector`, `MSERDetector`, and `SimpleBlobDetector` ignore `mask` and emit `UserWarning` when it is not `None`.

Detection results (`KPDetectionResult`, `ArrayKPDetectionResult`) implement `scale_coordinates(factor)` to multiply coordinates in place (and `cv2.KeyPoint.size` where applicable). Detectors call `_remap_result_to_original_coordinates` to map keypoints back to the input image space.
