# optical_flow

## Overview

Unified interface for optical flow computation using OpenCV-backed algorithms.
Processors inherit from `OpticalFlowProcessor`, share `OpticalFlowParameters`, and return
result containers that restore coordinates to the original image via ``scale()``.

## Components

| Component | Description |
| --------- | ----------- |
| [`parameter.py`](./parameter.py) | Base parameters including scale factor and interpolation flag |
| [`processor.py`](./processor.py) | Abstract base class and shared scaling logic |
| [`validator.py`](./validator.py) | Input validation and preprocessing helpers |
| [`farneback/`](./farneback/) | Farneback dense optical flow |
| [`lucas_canade/`](./lucas_canade/) | Lucas-Kanade sparse optical flow |
| [`method.py`](./method.py) | `OpticalFlowMethod` enum |

## Examples

```python
from kp_detection.detectors import ShiTomashiParameters
from optical_flow import (
    FarnebackFlow,
    FarnebackParameters,
    LucasKanadeFlow,
    LucasKanadeParameters,
    OpticalFlowMethod,
    OpticalFlowProcessor,
)

processors: list[OpticalFlowProcessor] = [
    FarnebackFlow(params=FarnebackParameters(scale_factor=0.5)),
    LucasKanadeFlow(
        params=LucasKanadeParameters(is_SparseRLOF=False, scale_factor=0.5),
        keypoint_params=ShiTomashiParameters(),
    ),
]

for processor in processors:
    match processor.method:
        case OpticalFlowMethod.FARNEBACK:
            result = processor.run(source_image, target_image)
            flow = result.flow
        case OpticalFlowMethod.LUCAS_KANADE:
            result = processor.run(source_image, target_image)
            keypoints = result.target_keypoints
```
