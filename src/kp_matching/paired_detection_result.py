from __future__ import annotations

import cv2
import warnings
import numpy as np
from dataclasses import dataclass
from typing import cast

from opencv_utility import OpenCVOutlierFilteringFlag
from .match_container import KNNMatchGroup, MatchResult
from .utils import GeometricConstraint
from kp_detection import KPDetectionResult


@dataclass
class PairedDetectionResult:
    """
    Container for keypoint matching result, including query and gallery keypoint detection results and match result.

    Attributes:
    ----------
    query_det_result: KPDetectionResult
        The query keypoint detection result.
    gallery_det_result: KPDetectionResult
        The gallery keypoint detection result.
    match_result: MatchResult
        The match result.
    """
    query_det_result: KPDetectionResult
    gallery_det_result: KPDetectionResult
    match_result: MatchResult

    @property
    def query_matched_coordinates(self) -> np.ndarray:
        """
        The query matched coordinates.

        Returns:
        -------
        np.ndarray:
            The query matched coordinates with shape ``(n, 2)``.
        """
        query_keypoints = self.query_det_result.coordinates
        match_indices = [match.queryIdx for match in self.match_result]
        return query_keypoints[match_indices]
    
    @property
    def gallery_matched_coordinates(self) -> np.ndarray:
        """
        The gallery matched coordinates.

        Returns:
        -------
        np.ndarray:
            The gallery matched coordinates with shape ``(n, 2)``.
        """
        gallery_keypoints = self.gallery_det_result.coordinates
        match_indices = [match.trainIdx for match in self.match_result]
        return gallery_keypoints[match_indices]

    @property
    def query_descriptors(self) -> np.ndarray:
        """
        The query descriptors.

        Returns:
        -------
        np.ndarray:
            The query descriptors.
        """
        if self.query_det_result.descriptors is None:
            raise ValueError("Query keypoint detection result has no descriptors.")
        return np.array([self.query_det_result.descriptors[match.queryIdx] for match in self.match_result])

    @property
    def gallery_descriptors(self):
        """
        The gallery descriptors.

        Returns:
        -------
        np.ndarray:
            The gallery descriptors.
        """
        if self.gallery_det_result.descriptors is None:
                raise ValueError("Gallery descriptors are not available.")
        return np.array([self.gallery_det_result.descriptors[match.trainIdx] for match in self.match_result])

    def filter_outliers(
        self, 
        outlier_th: float = 3.0,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.MAGSAC,
        geometric_constraint: GeometricConstraint = GeometricConstraint.FUNDAMENTAL,
        ) -> PairedDetectionResult:
        """
        Filter outliers from the matches.

        Parameters:
        ----------
        outlier_th: float
            Outlier threshold.
        outlier_filtering_flag: OpenCVOutlierFilteringFlag
            Outlier filtering flag.
        geometric_constraint: GeometricConstraint
            Geometric model used for outlier filtering.

        Returns:
        ----------
        PairedDetectionResult: Filtered matches by outlier filtering process.
        """        
        mask = geometric_constraint.estimate_mask(
            query_matched_coordinates=self.query_matched_coordinates,
            gallery_matched_coordinates=self.gallery_matched_coordinates,
            outlier_filtering_flag=outlier_filtering_flag,
            outlier_th=outlier_th,
        )

        if mask is None or mask.size == 0:
            warnings.warn(
                f"Outlier filtering failed with {geometric_constraint.value} constraint. "
                "Returning the original matches."
            )
            return self
        inlier_indices: list[int] = [int(i) for i in np.flatnonzero(mask.ravel() == 1)]
        raw_matches = self.match_result.matches
        filtered = [raw_matches[i] for i in inlier_indices]
        homogenous_matches = cast(list[cv2.DMatch] | list[KNNMatchGroup], filtered)
        return self.__class__(
            query_det_result=self.query_det_result,
            gallery_det_result=self.gallery_det_result,
            match_result=MatchResult(matches=homogenous_matches),
        )

    def __str__(self) -> str:
        return f"PairedDetectionResult(query_det_result={self.query_det_result}, \
        gallery_det_result={self.gallery_det_result}, \
        match_result={self.match_result})"
