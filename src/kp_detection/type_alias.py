from __future__ import annotations

from typing import Any, TypeAlias

from .detector import KeyPointDetector
from .results import DetectionResultUnion

KPDetector: TypeAlias = KeyPointDetector[Any, Any, DetectionResultUnion, Any]

__all__ = ["KPDetector"]
