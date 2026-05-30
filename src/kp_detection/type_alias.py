from __future__ import annotations

from typing import Any, TypeAlias

from .detector import KeyPointDetector
from .parameter import KPDetectionParameters
from .results import DetectionResultUnion

KPDetector: TypeAlias = KeyPointDetector[
    Any,
    Any,
    DetectionResultUnion,
    KPDetectionParameters,
]

__all__ = ["KPDetector"]
