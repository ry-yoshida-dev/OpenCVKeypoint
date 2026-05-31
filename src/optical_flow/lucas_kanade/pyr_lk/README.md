# pyr_lk

## Overview

Sparse optical flow using OpenCV `calcOpticalFlowPyrLK` on grayscale images.

## Components

| Component | Description |
| --------- | ----------- |
| [`parameter.py`](./parameter.py) | `PyrLKParameters` with `get_flow_func()` |
| [`processor.py`](./processor.py) | `PyrLKFlow` processor with grayscale preprocessing |

## Examples

```python
from optical_flow import PyrLKFlow, PyrLKParameters
from kp_detection.detectors import ShiTomashiParameters

processor = PyrLKFlow(
    params=PyrLKParameters(scale_factor=0.5),
    keypoint_params=ShiTomashiParameters(),
)
result = processor.run(source_image, target_image)
```
