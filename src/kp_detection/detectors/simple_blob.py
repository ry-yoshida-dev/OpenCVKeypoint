import numpy as np
import cv2
from ..detector import KeyPointDetector
from ..parameter import KPDetectionParameters
from ..results import KPDetectionResult


class SimpleBlobDetector(KeyPointDetector[cv2.SimpleBlobDetector, None, KPDetectionResult, KPDetectionParameters]):
    """
    SimpleBlob detector.

    Attributes:
    ----------
    detector: cv2.SimpleBlobDetector
    """

    def _define_detector(self) -> tuple[cv2.SimpleBlobDetector, None]:
        """
        Define the detector.

        Returns:
        ----------
        tuple[cv2.SimpleBlobDetector, None]
            The detector and None (no extractor).
        """
        detector = cv2.SimpleBlobDetector_create() # type: ignore
        return detector, None # type: ignore

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

        result = KPDetectionResult(
            method=self.params.method,
            keypoints=keypoints,
            descriptors=np.array([]),
        )
        self._remap_result_to_original_coordinates(result)
        return result

    def __str__(self) -> str:
        return f"SimpleBlobDetector(method={self.params.method})"