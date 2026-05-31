# sparse_rlof

## Overview

Sparse optical flow using OpenCV `calcOpticalFlowSparseRLOF`. PyrLK on grayscale
images initializes `nextPts` before RLOF refinement on BGR inputs.

## Components

| Component | Description |
| --------- | ----------- |
| [`parameter.py`](./parameter.py) | `SparseRLOFParameters` with `get_flow_func()` |
| [`processor.py`](./processor.py) | `SparseRLOFFlow` processor with BGR preprocessing |

## Examples

```python
from optical_flow import SparseRLOFFlow, SparseRLOFParameters
from kp_detection.detectors import ShiTomashiParameters

processor = SparseRLOFFlow(
    params=SparseRLOFParameters(scale_factor=0.5),
    keypoint_params=ShiTomashiParameters(),
)
result = processor.run(source_image, target_image)
```
