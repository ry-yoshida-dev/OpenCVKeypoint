import cv2
import numpy as np
from typing_extensions import override

from ..detector import KeyPointDetector
from ..method import KPDetectionMethod
from ..parameter import KPDetectionParameters
from ..results import KPDetectionResult


class StandardKPDetector(
    KeyPointDetector[cv2.Feature2D, None, KPDetectionResult, KPDetectionParameters]
):
    """
    StandardKPDetector.

    Attributes
    ----------
    detector: cv2.Feature2D
    """

    @override
    def detect(
        self, img: np.ndarray, mask: np.ndarray | None = None
    ) -> KPDetectionResult:
        """
        Detect keypoints in an image.

        Parameters
        ----------
        img: np.ndarray
            Input image ``(H, W)`` or ``(H, W, C)`` as accepted by OpenCV Feature2D.
        mask: np.ndarray | None
            Boolean mask (0 = ignore, 1 = include).

        Returns
        -------
        KPDetectionResult
            OpenCV keypoints plus optional descriptor rows.
        """
        scaled_image = self.image_scaler(img)
        scaled_mask = self._scaled_detection_mask(mask)
        keypoints, descriptors = self.detector.detectAndCompute(
            image=scaled_image,
            mask=scaled_mask,
        )
        if self.brief is not None:
            keypoints, descriptors = self.brief.compute(scaled_image, keypoints)

        result = KPDetectionResult(
            method=self.params.method,
            keypoints=keypoints,
            descriptors=descriptors,
        )
        self._remap_result_to_original_coordinates(result)
        return result

    @override
    def __str__(self) -> str:
        return f"StandardKPDetector(method={self.params.method})"

    @override
    def _define_detector(self) -> tuple[cv2.Feature2D, None]:
        """
        Define the detector.

        Returns
        -------
        tuple[cv2.Feature2D, None]
            The detector and None.
            NOTE: StandardKPDetector does not use an extractor.
        """
        match self.params.method:
            case KPDetectionMethod.ORB:
                detector: cv2.Feature2D = cv2.ORB.create()
            case KPDetectionMethod.SIFT:
                detector = cv2.SIFT.create()
            case KPDetectionMethod.BRISK:
                detector = cv2.xfeatures2d.BRISK.create()
            case KPDetectionMethod.AKAZE:
                detector = cv2.xfeatures2d.AKAZE.create()
            case KPDetectionMethod.KAZE:
                detector = cv2.xfeatures2d.KAZE.create()
            case _:
                raise ValueError(f"Invalid method: {self.params.method}")
        return detector, None
