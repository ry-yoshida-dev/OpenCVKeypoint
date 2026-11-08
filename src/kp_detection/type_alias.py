from __future__ import annotations

from typing import TypeAlias

import cv2

from .detector import KeyPointDetector
from .parameter import KPDetectionParameters
from .results import DetectionResultUnion

KPDetector: TypeAlias = KeyPointDetector[
    cv2.Feature2D | None,
    cv2.Feature2D | None,
    DetectionResultUnion,
    KPDetectionParameters,
]

__all__ = ["KPDetector"]
