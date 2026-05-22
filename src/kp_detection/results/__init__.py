from __future__ import annotations

from typing import TypeAlias

from ..result import KPT, KPElem, KPDetectionStep
from .array import ArrayKPDetectionResult
from .cv2_keypoint import KPDetectionResult

KeyPointDetectionResult: TypeAlias = KPDetectionResult | ArrayKPDetectionResult

__all__ = [
    "KPT",
    "KPElem",
    "KeyPointDetectionResult",
    "KPDetectionStep",
    "KPDetectionResult",
    "ArrayKPDetectionResult",
]
