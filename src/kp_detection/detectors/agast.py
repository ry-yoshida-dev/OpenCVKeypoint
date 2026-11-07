import cv2
import numpy as np
from typing_extensions import override

from ..detector import KeyPointDetector
from ..parameter import KPDetectionParameters
from ..results import KPDetectionResult


class AGASTDetector(
    KeyPointDetector[
        cv2.xfeatures2d.AgastFeatureDetector,
        cv2.xfeatures2d.BriefDescriptorExtractor,
        KPDetectionResult,
        KPDetectionParameters,
    ]
):
    """
    AGAST detector.

    Attributes
    ----------
    detector: cv2.xfeatures2d.AgastFeatureDetector
    extractor: cv2.xfeatures2d.BriefDescriptorExtractor
        BRIEF descriptor extractor.
    """

    @override
    def _define_detector(
        self,
    ) -> tuple[
        cv2.xfeatures2d.AgastFeatureDetector, cv2.xfeatures2d.BriefDescriptorExtractor
    ]:
        """
        Define the detector.

        Returns
        -------
        tuple[cv2.xfeatures2d.AgastFeatureDetector, cv2.xfeatures2d.BriefDescriptorExtractor]
            AGAST detector and BRIEF extractor.
        """
        return (
            cv2.xfeatures2d.AgastFeatureDetector.create(),
            cv2.xfeatures2d.BriefDescriptorExtractor.create(),
        )

    @override
    def detect(
        self, img: np.ndarray, mask: np.ndarray | None = None
    ) -> KPDetectionResult:
        """
        Detect keypoints in an image.

        Parameters
        ----------
        img: np.ndarray
            Input image ``(H, W)`` or ``(H, W, C)`` as accepted by OpenCV.
        mask: np.ndarray | None
            Boolean mask (0 = ignore, 1 = include).
            Ignored by this detector; a ``UserWarning`` is emitted when it is not ``None``.

        Returns
        -------
        KPDetectionResult
            The result of keypoint detection.
        """
        self._warn_if_mask_unused(mask)
        scaled_image = self.image_scaler(img)
        keypoints = self.detector.detect(scaled_image, None)
        keypoints, descriptors = self.extractor.compute(scaled_image, keypoints)

        result = KPDetectionResult(
            method=self.params.method,
            keypoints=keypoints,
            descriptors=descriptors,
        )
        self._remap_result_to_original_coordinates(result)
        return result

    @override
    def __str__(self) -> str:
        return f"AGASTDetector(method={self.params.method})"
