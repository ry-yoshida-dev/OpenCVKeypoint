import numpy as np

from .paired_detection_result import PairedDetectionResult
from .match_container import MatchResult
from .parameter import KPMatchingParameters
from kp_detection import DetectionResultUnion


class KPMatchingProcessor:
    """Class for keypoint matching.

    This class implements various methods for keypoint matching, including brute-force,
    FLANN-based, and k-Nearest Neighbors (kNN) matchers. It supports applying Lowe's
    ratio test to filter good matches based on a threshold.

    Attributes:
    ----------
    params: KPMatchingParameters
        The parameters for the keypoint matching processor.
    function: Callable[[np.ndarray, np.ndarray], list[cv2.DMatch]]
        The function for matching descriptors.
    """
    def __init__(
        self,
        params: KPMatchingParameters
        ) -> None:
        """
        Initialize the KPMatchingProcessor.

        Parameters:
        ----------
        params: KPMatchingParameters
            The parameters for the keypoint matching processor.
        """
        self.params = params
        self.function = self.params.define_matching_function()

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
        ) -> MatchResult:
        """
        Performs keypoint matching based on the specified method.
        This method matches descriptors from two sets using the chosen matching technique.

        Parameters:
        ----------
        desc1: np.ndarray
            Descriptors of keypoints in the first image.
        desc2: np.ndarray
            Descriptors of keypoints in the second image.

        Returns:
        ---------
        MatchResult: A sorted list of matches based on their distance.

        Raises:
        --------
        ValueError: If an unsupported matching method is specified.
        """

        matches = MatchResult(matches=self.function(desc1, desc2)) # type: ignore

        if self.params.is_ratio_test_enabled:
            matches = matches.apply_ratio_test(threshold=self.params.ratio_test_threshold)
        return matches

    def run_pipeline(
        self,
        query_det_result: DetectionResultUnion,
        gallery_det_result: DetectionResultUnion,
        ) -> PairedDetectionResult:
        """
        Match keypoints from two detection results and wrap the outcome in a PairedDetectionResult.

        Descriptors are read from each detection result; coordinates and matches stay aligned
        with the original keypoint indices.

        Parameters:
        ----------
        query_det_result: DetectionResultUnion
            Detection result for the query image (must include descriptors).
        gallery_det_result: DetectionResultUnion
            Detection result for the gallery image (must include descriptors).

        Returns:
        --------
        PairedDetectionResult:
            Query and gallery detections plus the ``MatchResult`` from this processor.

        Raises:
        --------
        ValueError: If either detection result has no descriptors.
        """
        query_descriptors = query_det_result.descriptors
        gallery_descriptors = gallery_det_result.descriptors
        if query_descriptors is None or gallery_descriptors is None:
            raise ValueError("Descriptors are not available.")

        matches: MatchResult = self.match(query_descriptors, gallery_descriptors)

        return PairedDetectionResult(
            query_det_result=query_det_result,
            gallery_det_result=gallery_det_result,
            match_result=matches
        )

