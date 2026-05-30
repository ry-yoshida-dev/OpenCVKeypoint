# farneback

## Overview

Dense Farneback optical flow processor and result container.

## Components

| Component | Description |
| --------- | ----------- |
| [`parameter.py`](./parameter.py) | Farneback-specific OpenCV parameters |
| [`processor.py`](./processor.py) | `FarnebackFlow` processor implementation |
| [`result.py`](./result.py) | `FarnebackResult` dense flow container with coordinate restoration |

## Examples

```python
from optical_flow import FarnebackFlow, FarnebackParameters

processor = FarnebackFlow(params=FarnebackParameters(scale_factor=0.5))
result = processor.run(source_image, target_image)
flow = result.flow
```
