from __future__ import annotations

from typing import Any, TypeAlias

from .detector import KeyPointDetector
from .results import KeyPointDetectionResult

# Union of the two concrete result types (cv2 KeyPoint vs ndarray).
KPDetectionResult: TypeAlias = KeyPointDetectionResult

KPDetector: TypeAlias = KeyPointDetector[Any, Any, KPDetectionResult, Any]

__all__ = ["KPDetector", "KPDetectionResult"]
