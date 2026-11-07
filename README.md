# OpenCVKeypoint

## Overview

OpenCVKeypoint wraps OpenCV keypoint detection, descriptor matching and optical flow in
typed Python interfaces, provided as three packages under `src/`:

| Package | Description |
| ------- | ----------- |
| [`kp_detection`](src/kp_detection/README.md) | Keypoint detection |
| [`kp_matching`](src/kp_matching/README.md) | Descriptor matching, geometric filtering and match visualization |
| [`optical_flow`](src/optical_flow/README.md) | Dense (Farneback) and sparse (PyrLK, SparseRLOF) optical flow |

Supported detectors (`KPDetectionMethod`):

| Method | Type | Descriptor |
| ------ | ---- | ---------- |
| SIFT | Feature | Float |
| KAZE | Feature | Float |
| ORB | Feature | Binary |
| BRISK | Feature | Binary |
| AKAZE (MLDB) | Feature | Binary |
| FAST | Corner | — |
| AGAST | Corner | — |
| Harris | Corner | — |
| ShiTomashi | Corner | — |
| MSER | Blob | — |
| SimpleBlob | Blob | — |

## Installation

Python 3.11 or newer and opencv-contrib-python 5.0 or newer.

```bash
pip install .
```

## Usage

### Keypoint detection and matching

```python
import cv2

from kp_detection import KPDetectionMethod
from kp_matching import (
    KPMatchCommonParameters,
    KPMatchingParameters,
    KPMatchingProcessor,
    MatchingVisualizer,
)

query_image = cv2.imread("query.png", cv2.IMREAD_GRAYSCALE)
gallery_image = cv2.imread("gallery.png", cv2.IMREAD_GRAYSCALE)

method = KPDetectionMethod.SIFT
detector = method.detector_class(params=method.parameter_class(method=method))
query = detector.detect(query_image)
gallery = detector.detect(gallery_image)

processor = KPMatchingProcessor(
    params=KPMatchingParameters(
        common_params=KPMatchCommonParameters(detection_method=method)
    )
)
paired = processor.run_pipeline(query, gallery)  # kNN with ratio test by default

matched_image = MatchingVisualizer().draw_matches(
    query_image,
    paired.query_det_result.keypoints,
    gallery_image,
    paired.gallery_det_result.keypoints,
    paired.match_result.matches,
)
cv2.imwrite("matches.jpg", matched_image)
```

### Optical flow

```python
from kp_detection.detectors import ShiTomashiParameters
from optical_flow import FarnebackFlow, FarnebackParameters, PyrLKFlow, PyrLKParameters

flow = (
    FarnebackFlow(params=FarnebackParameters(scale_factor=0.5))
    .run(source_image, target_image)
    .flow
)

sparse = PyrLKFlow(
    params=PyrLKParameters(scale_factor=0.5),
    keypoint_params=ShiTomashiParameters(max_corners=100),
).run(source_image, target_image)
sparse.target_keypoints
```
