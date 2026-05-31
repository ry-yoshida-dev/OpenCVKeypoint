# OpenCVKeypoint

## Overview

OpenCVKeypoint is a unified repository that combines:

- `kp_detection`: keypoint detection interfaces and implementations
- `kp_matching`: keypoint matching interfaces and implementations
- `optical_flow`: optical flow interfaces and implementations

All packages are provided under the same `src` layout so you can use detection, matching, and optical flow in a single project without managing separate repositories.

For package/module-level details, see:

- [`src/kp_detection/README.md`](src/kp_detection/README.md)
- [`src/kp_matching/README.md`](src/kp_matching/README.md)
- [`src/optical_flow/README.md`](src/optical_flow/README.md)

## Repository Structure

```text
OpenCVKeypoint/
├─ src/
│  ├─ kp_detection/
│  ├─ kp_matching/
│  └─ optical_flow/
├─ tests/
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

## Installation

From the repository root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

To install only dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is required.

## Example

### Keypoint detection and matching

```python
import cv2

from kp_detection import KPDetectionMethod, KPDetectionResult
from kp_matching import (
    DrawMatchFlags,
    FLANNParameters,
    KPMatchCommonParameters,
    KPMatchMethod,
    KPMatchingParameters,
    KPMatchingProcessor,
    MatchingVisualizer,
    PairedDetectionResult,
    RatioTestParameters,
)

query_image = cv2.imread("query.png", cv2.IMREAD_GRAYSCALE)
gallery_image = cv2.imread("gallery.png", cv2.IMREAD_GRAYSCALE)
if query_image is None or gallery_image is None:
    raise FileNotFoundError("query.png or gallery.png was not found")

detection_method = KPDetectionMethod.SIFT
params = detection_method.parameter_class(method=detection_method)
detector = detection_method.detector_class(params=params)

query_det_result: KPDetectionResult = detector.detect(query_image)
gallery_det_result: KPDetectionResult = detector.detect(gallery_image)

common_params = KPMatchCommonParameters(
    detection_method=detection_method,
    method=KPMatchMethod.KNN,
    is_cross_check_enabled=False,
    knn=2,
)
ratio_test_params = RatioTestParameters(
    is_enabled=True,
    threshold=0.75,
)
flann_params = FLANNParameters(
    checks=50,
    trees=5,
)
matching_params = KPMatchingParameters(
    common_params=common_params,
    ratio_test_params=ratio_test_params,
    flann_params=flann_params,
)
processor = KPMatchingProcessor(params=matching_params)

paired: PairedDetectionResult = processor.run_pipeline(query_det_result, gallery_det_result)
print(f"number of matches: {len(paired.match_result.matches)}")

# Visualize matched keypoints and save as an image.
visualizer = MatchingVisualizer(flags=DrawMatchFlags.NOT_DRAW_SINGLE_POINTS)
matched_image = visualizer.draw_matches(
    query_image,
    paired.query_det_result.keypoints,
    gallery_image,
    paired.gallery_det_result.keypoints,
    paired.match_result.matches,
)

output_path = "matched_result.jpg"
cv2.imwrite(output_path, matched_image)
print(f"saved visualization: {output_path}")
```

### Optical flow

```python
import cv2

from kp_detection.detectors import ShiTomashiParameters
from optical_flow import FarnebackFlow, FarnebackParameters, PyrLKFlow, PyrLKParameters

source_image = cv2.imread("frame0.png", cv2.IMREAD_GRAYSCALE)
target_image = cv2.imread("frame1.png", cv2.IMREAD_GRAYSCALE)
if source_image is None or target_image is None:
    raise FileNotFoundError("frame0.png or frame1.png was not found")

dense_flow = FarnebackFlow(params=FarnebackParameters(scale_factor=0.5))
flow_result = dense_flow.run(source_image, target_image)
print(f"dense flow shape: {flow_result.flow.shape}")

sparse_flow = PyrLKFlow(
    params=PyrLKParameters(scale_factor=0.5),
    keypoint_params=ShiTomashiParameters(max_corners=100),
)
sparse_result = sparse_flow.run(source_image, target_image)
print(f"tracked keypoints: {len(sparse_result.target_keypoints)}")
```
