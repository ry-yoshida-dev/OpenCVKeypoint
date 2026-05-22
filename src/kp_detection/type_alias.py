from __future__ import annotations

from typing import Any, TypeAlias

from .detector import KeyPointDetector
from .parameter import KPDetectionParameters
from .results import KPDetectionResult

KPDetector: TypeAlias = KeyPointDetector[
    Any,
    Any,
    KPDetectionResult,
    KPDetectionParameters,
]

__all__ = ["KPDetector"]
