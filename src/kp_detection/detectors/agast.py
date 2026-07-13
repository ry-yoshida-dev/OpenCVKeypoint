import cv2
import numpy as np


from ..detector import KeyPointDetector
from ..parameter import KPDetectionParameters
from ..results import KPDetectionResult

# cv2.AgastFeatureDetector was moved to opencv_contrib's xfeatures2d module in
# OpenCV 5.0, and some opencv-contrib-python 5.x wheels do not compile that
# module in at all. Fall back to the always-present Feature2D base for the
# generic type parameter so this class can still be imported; _define_detector
# raises a clear error if the detector is actually instantiated on such builds.
_AgastFeatureDetector = getattr(cv2, "AgastFeatureDetector", cv2.Feature2D)

class AGASTDetector(KeyPointDetector[_AgastFeatureDetector, cv2.Feature2D, KPDetectionResult, KPDetectionParameters]):
    """
    AGAST detector.

    Attributes:
    ----------
    detector: cv2.AgastFeatureDetector
    extractor: cv2.Feature2D
        BRIEF descriptor extractor (cv2.xfeatures2d.BriefDescriptorExtractor_create).

    Notes:
    ----------
    Requires an OpenCV build exposing ``cv2.AgastFeatureDetector_create`` and
    ``cv2.xfeatures2d.BriefDescriptorExtractor_create``. Both moved to
    opencv_contrib's xfeatures2d module in OpenCV 5.0, and some
    opencv-contrib-python 5.x wheels do not compile that module in; on such
    builds this class can still be imported, but instantiation raises
    RuntimeError.
    """

    def _define_detector(self) -> tuple[cv2.Feature2D, cv2.Feature2D]:
        """
        Define the detector.

        Returns:
        ----------
        tuple[cv2.Feature2D, cv2.Feature2D]
            AGAST detector and BRIEF extractor.

        Raises:
        ----------
        RuntimeError
            If the installed OpenCV build does not expose AGAST and/or BRIEF.
        """
        agast_create = getattr(cv2, "AgastFeatureDetector_create", None)
        brief_create = getattr(getattr(cv2, "xfeatures2d", None), "BriefDescriptorExtractor_create", None)
        if agast_create is None or brief_create is None:
            raise RuntimeError(
                f"AGAST/BRIEF are unavailable in this OpenCV build (cv2 {cv2.__version__}). "
                "They live in opencv_contrib's xfeatures2d module, which this build does "
                "not compile in. Install an OpenCV build that provides "
                "cv2.AgastFeatureDetector_create/cv2.xfeatures2d.BriefDescriptorExtractor_create, "
                "or select a different KPDetectionMethod."
            )
        detector = agast_create()
        extractor = brief_create()
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
            Input image ``(H, W)`` or ``(H, W, C)`` as accepted by OpenCV.
        mask: np.ndarray | None
            Boolean mask (0 = ignore, 1 = include).

        Returns:
        ----------
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

    def __str__(self) -> str:
        return f"AGASTDetector(method={self.params.method})"