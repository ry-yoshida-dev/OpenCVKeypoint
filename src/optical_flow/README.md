# optical_flow

## Overview

Unified interface for optical flow computation using OpenCV-backed algorithms.
Processors inherit from `OpticalFlowProcessor`, share `OpticalFlowParameters`, and return
result containers that restore coordinates to the original image via `scale()`.

## Components

| Component | Description |
| --------- | ----------- |
| [`parameter.py`](./parameter.py) | Base parameters including scale factor and interpolation flag |
| [`processor.py`](./processor.py) | Abstract base class and shared scaling logic |
| [`utils/`](./utils/README.md) | Input validation and image preprocessing helpers |
| [`farneback/`](./farneback/README.md) | Farneback dense optical flow |
| [`lucas_kanade/`](./lucas_kanade/README.md) | Lucas-Kanade sparse optical flow |
| [`method.py`](./method.py) | `OpticalFlowMethod` enum |

## Examples

```python
from optical_flow import FarnebackFlow, FarnebackParameters, OpticalFlowMethod

processor = FarnebackFlow(params=FarnebackParameters(scale_factor=0.5))
assert processor.method is OpticalFlowMethod.FARNEBACK
flow = processor.run(source_image, target_image).flow
```
