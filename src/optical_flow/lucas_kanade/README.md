# lucas_kanade

## Overview

Sparse Lucas-Kanade optical flow with a shared abstract processor and two concrete
implementations: PyrLK and SparseRLOF. Feature seeds are detected with Shi-Tomasi
from `kp_detection`.

## Components

| Component | Description |
| --------- | ----------- |
| [`parameter.py`](./parameter.py) | Base `LucasKanadeParameters` with keypoint shape helpers |
| [`protocol.py`](./protocol.py) | `LucasKanadeFlowFunction` protocol for sparse flow callables |
| [`processor.py`](./processor.py) | Abstract `LucasKanadeFlow` processor |
| [`result.py`](./result.py) | `LucasKanadeResult` sparse flow container |
| [`pyr_lk/`](./pyr_lk/README.md) | `PyrLKFlow` backed by `cv2.calcOpticalFlowPyrLK` |
| [`sparse_rlof/`](./sparse_rlof/README.md) | `SparseRLOFFlow` backed by `cv2.optflow.calcOpticalFlowSparseRLOF` |

## Examples

```python
from kp_detection.detectors import ShiTomashiParameters
from optical_flow import PyrLKFlow, PyrLKParameters

processor = PyrLKFlow(
    params=PyrLKParameters(scale_factor=0.5),
    keypoint_params=ShiTomashiParameters(),
)
tracked = processor.run(source_image, target_image).target_keypoints
```

`SparseRLOFFlow` with `SparseRLOFParameters` is used the same way.
