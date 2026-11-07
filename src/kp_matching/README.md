# kp_matching

## Overview

Keypoint matching between two detection results with brute-force, kNN or FLANN matchers,
optional ratio test, cross-check and geometric filtering, plus match visualization.

## Components

| Component | Description |
| --------- | ----------- |
| [`processor.py`](./processor.py) | `KPMatchingProcessor`, the matching pipeline |
| [`parameter.py`](./parameter.py) | `KPMatchingParameters`, parameter integration and matcher construction |
| [`method.py`](./method.py) | `KPMatchMethod` enum (BF, kNN, FLANN) |
| [`match_container.py`](./match_container.py) | `MatchResult`, container for managing and filtering matches |
| [`paired_detection_result.py`](./paired_detection_result.py) | `PairedDetectionResult`, query and gallery detection results with their matches |
| [`parameters/`](./parameters/README.md) | Configuration classes for matching |
| [`utils/`](./utils/README.md) | Geometric constraints for outlier filtering |
| [`visualizer/`](./visualizer/README.md) | Drawing matches between two images |
