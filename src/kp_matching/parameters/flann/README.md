# flann

## Overview

FLANN parameters for approximate nearest-neighbour matching.

## Components

| Component | Description |
| --------- | ----------- |
| [`index_type.py`](./index_type.py) | FLANN algorithm/index enum values |
| [`parameter.py`](./parameter.py) | Dataclass that validates FLANN options and builds a `cv2.FlannBasedMatcher` |
