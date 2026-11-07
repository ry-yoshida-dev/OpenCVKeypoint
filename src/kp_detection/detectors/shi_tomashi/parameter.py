from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from typing_extensions import override

if TYPE_CHECKING:
    from .detector import ShiTomashiDetector

from ...method import KPDetectionMethod
from ...parameter import KPDetectionParameters


@dataclass
class ShiTomashiParameters(KPDetectionParameters):
    """
    Parameters for Shi–Tomasi keypoint detection (base fields plus detector-specific options).

    Attributes
    ----------
    max_corners: int
        Maximum number of corners to return.
        When increased: More corners can be returned (up to this limit).
    quality_level: float
        Quality level parameter.
        When increased: Only higher quality corners are detected (fewer but better corners).
    min_distance: int
        Minimum distance between corners.
        When increased: Corners must be further apart (fewer corners, more spread out).
    blocksize: int
        Block size for corner detection.
        When increased: Uses larger blocks for detection (may detect larger features, slower computation).
    useHarrisDetector: bool
        If True, use Harris detector to detect corners.
    k: float
        Harris detector free parameter in the equation: R = det(M) - k(trace(M)^2).
        When increased: Harris detector becomes more selective (fewer corners detected).
        If useHarrisDetector is False, this parameter is not used.
    """

    method: KPDetectionMethod = KPDetectionMethod.SHI_TOMASHI
    max_corners: int = 1000
    quality_level: float = 0.01
    min_distance: int = 1
    blocksize: int = 3
    useHarrisDetector: bool = False
    k: float = 0.04

    def __post_init__(self) -> None:
        if self.method != KPDetectionMethod.SHI_TOMASHI:
            warnings.warn(
                f"method was {self.method!r}; using method=KPDetectionMethod.SHI_TOMASHI "
                + "for ShiTomashiParameters.",
                stacklevel=2,
            )
            self.method = KPDetectionMethod.SHI_TOMASHI
        super().__post_init__()
        if self.max_corners <= 0:
            raise ValueError("max_corners must be a positive integer")
        if self.quality_level <= 0 or self.quality_level >= 1:
            raise ValueError("quality_level must be a float strictly between 0 and 1")
        if self.min_distance <= 0:
            raise ValueError("min_distance must be a positive integer")
        if self.blocksize <= 0:
            raise ValueError("blocksize must be a positive integer")
        if self.k <= 0:
            raise ValueError("k must be a positive number")

    @property
    def shi_tomashi_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
        """
        Define the ShiTomashi function.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray | None], np.ndarray]
            (image, mask) -> corners from cv2.goodFeaturesToTrack. Pass mask=None for the full image.
        """

        def detect_corners(
            img: np.ndarray, mask: np.ndarray | None = None
        ) -> np.ndarray:
            return cv2.goodFeaturesToTrack(
                image=img,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                mask=mask,
                blockSize=self.blocksize,
                useHarrisDetector=self.useHarrisDetector,
                k=self.k,
            )

        return detect_corners

    @override
    def build_detector(self) -> ShiTomashiDetector:
        """
        Build a Shi–Tomasi detector instance based on the parameters.

        Returns
        -------
        ShiTomashiDetector:
            Shi–Tomasi detector instance.
        """
        from .detector import ShiTomashiDetector

        return ShiTomashiDetector(params=self)
