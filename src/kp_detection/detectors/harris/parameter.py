from __future__ import annotations

import warnings
import cv2
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .detector import HarrisDetector

from ...method import KPDetectionMethod
from ...parameter import KPDetectionParameters


@dataclass
class HarrisParameters(KPDetectionParameters):
    """
    Parameters for Harris keypoint detection (base fields plus detector-specific options).

    Attributes:
    ----------
    block_size: int
        Neighborhood size for corner detection.
        It is the size of the window considered for computing the local autocorrelation matrix.
    ksize: int
        Aperture parameter for the Sobel operator.
        Must be a small odd number (e.g., 1, 3, 5, 7). Larger values are more robust to noise.
    k: float
        Harris detector free parameter in the equation: R = det(M) - k(trace(M)^2).
        Smaller values (e.g., 0.04) are more sensitive and detect more corners.
    corner_th: float
        Relative threshold for detection.
        Only pixels with a response higher than (corner_th * max_response) are retained.
        Larger values filter out weaker corners.
    """
    method: KPDetectionMethod = KPDetectionMethod.HARRIS
    block_size: int = 2
    ksize: int = 3
    k: float = 0.04
    corner_th: float = 0.01

    def __post_init__(self) -> None:
        if self.method != KPDetectionMethod.HARRIS:
            warnings.warn(
                f"method was {self.method!r}; using method=KPDetectionMethod.HARRIS "
                "for HarrisParameters.",
                stacklevel=2,
            )
            self.method = KPDetectionMethod.HARRIS
        super().__post_init__()
        if self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if self.ksize <= 0:
            raise ValueError("ksize must be a positive integer")
        if self.ksize % 2 == 0:
            raise ValueError("ksize must be an odd number (1, 3, 5, ...)")
        if self.k <= 0:
            raise ValueError("k must be a positive number")
        if not (0 < self.corner_th < 1):
            raise ValueError("corner_th must be a float strictly between 0 and 1")

    @property
    def harris_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Define the Harris function.

        Returns:
        ----------
        Callable[[np.ndarray], np.ndarray]
            Function mapping a single-channel ``(H, W)`` image to a float32
            corner response map of the same shape (output of ``cv2.cornerHarris``).
        """
        return lambda img: cv2.cornerHarris(
            img,
            blockSize=self.block_size,
            ksize=self.ksize,
            k=self.k,
        )

    def build_detector(self) -> HarrisDetector:
        """
        Build a Harris detector instance based on the parameters.

        Returns:
        ----------
        HarrisDetector:
            Harris detector instance.
        """
        from .detector import HarrisDetector
        return HarrisDetector(params=self)
