# lucas_kanade

## Overview

Sparse Lucas-Kanade optical flow processor and result container.
Feature seeds are detected with Shi-Tomasi from `kp_detection`.

## Components

| Component | Description |
| --------- | ----------- |
| [`parameter.py`](./parameter.py) | Lucas-Kanade-specific OpenCV parameters |
| [`processor.py`](./processor.py) | `LucasKanadeFlow` processor implementation |
| [`result.py`](./result.py) | `LucasKanadeResult` sparse flow container with coordinate restoration and zero-track fallbacks |

## Examples

```python
from optical_flow import LucasKanadeFlow, LucasKanadeParameters
from kp_detection.detectors import ShiTomashiParameters

processor = LucasKanadeFlow(
    params=LucasKanadeParameters(is_SparseRLOF=False, scale_factor=0.5),
    keypoint_params=ShiTomashiParameters(),
)
result = processor.run(source_image, target_image)
tracked = result.target_keypoints
```
