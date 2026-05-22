from __future__ import annotations

from typing import TypeAlias

from ..result import KPT, KPElem, KeyPointDetectionResult, KPDetectionStep
from .array import ArrayKPDetectionResult
from .cv2_keypoint import KPDetectionResult

DetectionResultUnion: TypeAlias = KPDetectionResult | ArrayKPDetectionResult

__all__ = [
    "KPT",
    "KPElem",
    "KeyPointDetectionResult",
    "KPDetectionStep",
    "DetectionResultUnion",
    "KPDetectionResult",
    "ArrayKPDetectionResult",
]
