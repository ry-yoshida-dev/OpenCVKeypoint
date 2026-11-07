import warnings
from dataclasses import dataclass

import cv2

from kp_detection.method import KPDetectionMethod

from ..method import KPMatchMethod


@dataclass
class KPMatchCommonParameters:
    """
    Common parameters for keypoint matching.

    Attributes
    ----------
    method: KPMatchMethod
        The method for keypoint matching.
    detection_method: KPDetectionMethod
        The keypoint detection method associated with descriptors.
    is_cross_check_enabled: bool
        Whether to use cross-check matching.
    knn: int
        The number of nearest neighbors to use for matching.
    """

    detection_method: KPDetectionMethod
    method: KPMatchMethod = KPMatchMethod.KNN
    is_cross_check_enabled: bool = False
    knn: int = 2

    def __post_init__(self) -> None:
        if self.knn < 1:
            raise ValueError("knn must be at least 1")
        if self.knn > 2:
            raise ValueError("knn must be less than or equal to 2")
        if self.method == KPMatchMethod.KNN and self.is_cross_check_enabled:
            warnings.warn("kNN matching cannot be used with cross-check", stacklevel=2)
            self.is_cross_check_enabled = False

    @property
    def distance_norm(self) -> int:
        """
        OpenCV norm type for descriptor matching.

        Returns
        -------
        int
            cv2 norm constant used by BFMatcher.
        """
        if self.detection_method.is_binary_descriptor_supported:
            return cv2.NORM_HAMMING
        if self.detection_method.is_float_descriptor_supported:
            return cv2.NORM_L2
        raise ValueError(
            f"{self.detection_method.value} does not support descriptor matching."
        )
