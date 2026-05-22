import cv2
import numpy as np

from ..detector import KeyPointDetector
from ..parameter import KPDetectionParameters
from ..results import KPDetectionResult

class MSERDetector(KeyPointDetector[cv2.Feature2D, cv2.Feature2D, KPDetectionResult, KPDetectionParameters]):
    """
    MSER detector.

    Attributes:
    ----------
    detector: cv2.MSER
    extractor: cv2.Feature2D
        BRIEF extractor instance (not used in detect(); descriptors are returned as an empty array).
    """

    def _define_detector(self) -> tuple[cv2.Feature2D, cv2.Feature2D]:
        """
        Define the detector.

        Returns:
        ----------
        tuple[cv2.MSER, cv2.Feature2D]
            MSER region detector and BRIEF extractor.
        """
        detector = cv2.MSER_create() # type: ignore
        extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create() # type: ignore
        return detector, extractor # type: ignore

    def detect(
        self,
        img: np.ndarray,
        mask: np.ndarray | None = None
        ) -> KPDetectionResult:
        """
        Detect keypoints in an image.

        Parameters:
        ----------
        img: np.ndarray
            Input image ``(H, W)`` or ``(H, W, C)`` as accepted by OpenCV MSER.
        mask: np.ndarray | None
            Boolean mask (0 = ignore, 1 = include).

        Returns:
        ----------
        KPDetectionResult
            The result of keypoint detection.
        """
        self._warn_if_mask_unused(mask)
        scaled_image = self.image_scaler(img)
        regions: list[np.ndarray]  # each region has shape ``(N, 2)``
        regions, _ = self.detector.detectRegions(scaled_image) # type: ignore

        keypoints = [
            cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=3.0) # type: ignore
            for region in regions # type: ignore
            for pt in region # type: ignore
        ]
        result = KPDetectionResult(
            method=self.params.method,
            keypoints=keypoints,
            descriptors=np.array([]),
        )
        self._remap_result_to_original_coordinates(result)
        return result

    def __str__(self) -> str:
        return f"MSERDetector(method={self.params.method})"