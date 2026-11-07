import cv2
import numpy as np
from typing_extensions import override

from ..detector import KeyPointDetector
from ..parameter import KPDetectionParameters
from ..results import KPDetectionResult


class FASTDetector(
    KeyPointDetector[cv2.Feature2D, None, KPDetectionResult, KPDetectionParameters]
):
    """
    FAST detector.

    Attributes
    ----------
    detector: cv2.FastFeatureDetector
    """

    @override
    def _define_detector(self) -> tuple[cv2.Feature2D, None]:
        """
        Define the detector.

        Returns
        -------
        tuple[cv2.Feature2D, None]
            FAST detector and None (no extractor; detect() returns descriptors=None).
        """
        return cv2.FastFeatureDetector.create(), None

    @override
    def detect(
        self, img: np.ndarray, mask: np.ndarray | None = None
    ) -> KPDetectionResult:
        """
        Detect keypoints in an image.

        Parameters
        ----------
        img: np.ndarray
            Input image ``(H, W)`` or ``(H, W, C)`` as accepted by OpenCV FAST.
        mask: np.ndarray | None
            Boolean mask (0 = ignore, 1 = include).

        Returns
        -------
        KPDetectionResult
            The result of keypoint detection.
        """
        keypoints = self.detector.detect(self.image_scaler(img), None)

        result = KPDetectionResult(
            method=self.params.method,
            keypoints=keypoints,
            descriptors=None,
        )
        self._remap_result_to_original_coordinates(result)
        if mask is not None:
            result.apply_mask(mask)
        return result

    @override
    def __str__(self) -> str:
        return f"FASTDetector(method={self.params.method})"
