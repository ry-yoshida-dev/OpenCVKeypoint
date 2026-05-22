
from __future__ import annotations

from enum import Enum
from typing import Callable

import cv2
import numpy as np
from opencv_utility import OpenCVOutlierFilteringFlag

class GeometricConstraint(Enum):
    """
    Geometric constraint.

    Attributes:
    ----------
    FUNDAMENTAL: Fundamental matrix constraint for non-planar (3D) scenes.
    HOMOGRAPHY: Homography constraint for planar(2D) scenes.
    """
    FUNDAMENTAL = "fundamental"
    HOMOGRAPHY = "homography"

    @property
    def estimator(self) -> Callable[..., tuple[np.ndarray | None, np.ndarray | None]]:
        """
        OpenCV estimator function associated with this geometric constraint.

        Returns:
        ----------
        Callable[..., tuple[np.ndarray | None, np.ndarray | None]]:
            OpenCV estimator function.
        """
        match self:
            case GeometricConstraint.FUNDAMENTAL:
                return cv2.findFundamentalMat
            case GeometricConstraint.HOMOGRAPHY:
                return cv2.findHomography

    def cv2_flag(self, outlier_filtering_flag: OpenCVOutlierFilteringFlag) -> int:
        """
        CV2 flag for each estimator.

        Parameters:
        ----------
        outlier_filtering_flag: OpenCVOutlierFilteringFlag
            Outlier filtering flag.

        Returns:
        ----------
        int:
            CV2 flag.
        """
        match self:
            case GeometricConstraint.FUNDAMENTAL:
                return outlier_filtering_flag.fundamental_matrix_flag
            case GeometricConstraint.HOMOGRAPHY:
                return outlier_filtering_flag.cv2_flag

    def estimate_mask(
        self,
        query_matched_coordinates: np.ndarray,
        gallery_matched_coordinates: np.ndarray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag,
        outlier_th: float,
    ) -> np.ndarray | None:
        """
        Estimate inlier/outlier mask for matched points.

        Parameters:
        ----------
        query_matched_coordinates: np.ndarray
            Query matched coordinates.
        gallery_matched_coordinates: np.ndarray
            Gallery matched coordinates.
        outlier_filtering_flag: OpenCVOutlierFilteringFlag
            Outlier filtering flag.
        outlier_th: float
            Outlier threshold.

        Returns:
        ----------
        np.ndarray | None:
            Inlier/outlier mask with shape ``(n, 1)`` (0: outlier, 1: inlier).
        """
        match self:
            case GeometricConstraint.FUNDAMENTAL:
                _, mask = self.estimator(
                    points1=query_matched_coordinates,
                    points2=gallery_matched_coordinates,
                    method=self.cv2_flag(outlier_filtering_flag),
                    ransacReprojThreshold=outlier_th,
                )
            case GeometricConstraint.HOMOGRAPHY:
                _, mask = self.estimator(
                    srcPoints=query_matched_coordinates,
                    dstPoints=gallery_matched_coordinates,
                    method=self.cv2_flag(outlier_filtering_flag),
                    ransacReprojThreshold=outlier_th,
                )
        return mask

