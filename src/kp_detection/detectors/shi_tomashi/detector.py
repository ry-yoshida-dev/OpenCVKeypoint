import cv2
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Callable

from .parameter import ShiTomashiParameters
from ...detector import KeyPointDetector
from ...results import ArrayKPDetectionResult


@dataclass(repr=False, eq=False)
class ShiTomashiDetector(KeyPointDetector[None, None, ArrayKPDetectionResult, ShiTomashiParameters]):
    """
    ShiTomashi detector.

    Attributes:
    ----------
    params: ShiTomashiParameters
        Parameters for this detector (method, BRIEF flag, and Shi-Tomasi options).
    function: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
        The ShiTomashi function (image, optional mask).
    """
    params: ShiTomashiParameters
    function: Callable[[np.ndarray, np.ndarray | None], np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.function = self.params.shi_tomashi_function

    def _define_detector(self) -> tuple[None, None]:
        """
        Define the detector.

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
        ) -> ArrayKPDetectionResult:
        """
        Detect keypoints (2D grayscale or 3D BGR; the latter is converted to grayscale).

        Parameters:
        ----------
        img: np.ndarray
            Grayscale ``(H, W)`` or color ``(H, W, C)``.
        mask: np.ndarray | None
            Boolean mask (0 = ignore, 1 = include).

        Returns:
        ----------
        ArrayKPDetectionResult
            The result of keypoint detection.
        """
        detection_mask = self._scaled_detection_mask(mask)

        keypoints: np.ndarray
        if img.ndim == 2:
            keypoints = self.function(self.image_scaler(img), detection_mask)
        elif img.ndim == 3:
            warnings.warn("3D array is input as image, converting to grayscale")
            keypoints = self.function(
                self.image_scaler(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                detection_mask,
            )
        else:
            raise ValueError(f"img must be a 2D or 3D array, got shape {img.shape}")
        result = ArrayKPDetectionResult(
            method=self.params.method,
            keypoints=keypoints,
            descriptors=None,
        )
        self._remap_result_to_original_coordinates(result)
        return result

    def __str__(self) -> str:
        return f"ShiTomashiDetector(params={self.params!r})"
