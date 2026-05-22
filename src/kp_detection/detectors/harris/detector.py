import cv2
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Callable

from .parameter import HarrisParameters
from ...detector import KeyPointDetector
from ...results import KPDetectionResult


@dataclass(repr=False, eq=False)
class HarrisDetector(KeyPointDetector[None, None, KPDetectionResult, HarrisParameters]):
    """
    Harris detector.

    Attributes:
    ----------
    params: HarrisParameters
        Parameters for this detector (method, BRIEF flag, and Harris options).
    harris_function: Callable[[np.ndarray], np.ndarray]
        The Harris corner response function.
    """
    params: HarrisParameters
    harris_function: Callable[[np.ndarray], np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.harris_function = self.params.harris_function

    def _define_detector(self) -> tuple[None, None]:
        """
        Define the detector.
        NOTE: Harris detector does not need a detector and extractor.

        Returns:
        ----------
        tuple[None, None]
            The detector and the extractor.
        """
        return None, None

    def detect(
        self,
        img: np.ndarray,
        mask: np.ndarray | None = None
        ) -> KPDetectionResult:
        """
        Detect keypoints (single-channel image expected by the Harris response).

        Parameters:
        ----------
        img: np.ndarray
            Grayscale image with shape ``(H, W)``.
        mask: np.ndarray | None
            Boolean mask (0 = ignore, 1 = include).

        Returns:
        ----------
        KPDetectionResult
            OpenCV keypoints; the descriptors field is an empty array (no float descriptors).
        """
        if mask is not None and mask.ndim != 2:
            raise ValueError(f"mask must be a 2D array, got shape {mask.shape}")

        if img.ndim == 2:
            working = self.image_scaler(img)
        elif img.ndim == 3:
            warnings.warn("3D array is input as image, converting to grayscale")
            working = self.image_scaler(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            raise ValueError(f"img must be a 2D or 3D array, got shape {img.shape}")

        corner_response = self.harris_function(working)

        if mask is not None:
            corner_response *= mask

        keypoints = np.argwhere(
            corner_response > self.params.corner_th * corner_response.max()
            )
        keypoints = [
            cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size=3.0)
            for pt in keypoints
        ]
        result = KPDetectionResult(
            method=self.params.method,
            keypoints=keypoints,
            descriptors=None,
        )
        return self._remap_result_to_original_coordinates(result)

    def __str__(self) -> str:
        return f"HarrisDetector(params={self.params!r})"
