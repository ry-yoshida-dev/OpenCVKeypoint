# kp_detection

## Overview

Keypoint detection with OpenCV-backed algorithms. A `KPDetectionParameters` (or a
method-specific parameter class) builds a detector, which returns a `KPDetectionResult`
(`cv2.KeyPoint` based) or an `ArrayKPDetectionResult` (coordinate array based) in
input-image coordinates.

## Components

| Component | Description |
| --------- | ----------- |
| [`detector.py`](./detector.py) | `KeyPointDetector`, the detector abstract base with scaling and masking helpers |
| [`type_alias.py`](./type_alias.py) | `KPDetector` type alias |
| [`parameter.py`](./parameter.py) | `KPDetectionParameters` and `build_detector()` |
| [`result.py`](./result.py) | `KeyPointDetectionResult`, the result abstract base |
| [`results/`](./results/README.md) | Concrete detection result containers |
| [`method.py`](./method.py) | `KPDetectionMethod` enum |
| [`detectors/`](./detectors/README.md) | Per-algorithm detector classes |

## Examples

```python
from kp_detection import (
    HarrisParameters,
    KPDetectionMethod,
    KPDetectionParameters,
    KPDetectionResult,
    ShiTomashiParameters,
)
from kp_detection.detectors import StandardKPDetector

params = KPDetectionParameters(method=KPDetectionMethod.SIFT, scale_factor=0.5)
result: KPDetectionResult = StandardKPDetector(params=params).detect(image)

corners = ShiTomashiParameters(max_corners=200, min_distance=5).build_detector().detect(image)
harris = HarrisParameters(corner_th=0.02).build_detector().detect(image)
```
