from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass
from typing import Sequence

from ..result import KeyPointDetectionResult, KPDetectionStep


@dataclass
class KPDetectionResult(
    KeyPointDetectionResult[
        Sequence[cv2.KeyPoint],
        cv2.KeyPoint,
    ],
):
    """
    Keypoint detection result backed by a tuple or list of cv2.KeyPoint.
    Typical for detectors that return OpenCV keypoints (e.g. ORB, SIFT).


    Attributes:
    ----------
    method: KPDetectionMethod
        The detection method that produced this result (from the base class).
    keypoints: Sequence[cv2.KeyPoint]
        OpenCV keypoints (tuple, list, or other sequence from e.g. ``detect()``).
    descriptors: np.ndarray | None
        Descriptor rows aligned with keypoints, or None if not computed.
    """

    keypoints: Sequence[cv2.KeyPoint]

    def __len__(self) -> int:
        """
        Number of keypoints.

        Returns:
        ----------
        int:
            Length of the keypoints sequence.
        """
        return len(self.keypoints)

    def __getitem__(self, index: int) -> KPDetectionStep[cv2.KeyPoint]:
        """
        Return the cv2.KeyPoint and optional descriptor at index.

        Parameters:
        ----------
        index: int
            Zero-based index into the keypoint sequence.

        Returns:
        ----------
        KPDetectionStep[cv2.KeyPoint]:
            (keypoint, descriptor_row_or_none).
        """
        if self.descriptors is None:
            return (self.keypoints[index], None)
        return (self.keypoints[index], self.descriptors[index])

    def __str__(self) -> str:
        """
        Short summary with keypoint count and descriptor shape.

        Returns:
        ----------
        str:
            String representation for logging or display.
        """
        if self.descriptors is None:
            return (
                f"KPDetectionResult(keypoints.length={len(self.keypoints)}, "
                "descriptors=None)"
            )
        return (
            f"KPDetectionResult(keypoints.length={len(self.keypoints)}, "
            f"descriptors.shape={self.descriptors.shape})"
        )

    @property
    def x(self) -> np.ndarray:
        """
        X coordinates from each keypoint's pt[0].

        Returns:
        ----------
        np.ndarray:
            1-D array with shape ``(N,)`` (dtype inferred by NumPy).
        """
        return np.array([kp.pt[0] for kp in self.keypoints])

    @property
    def y(self) -> np.ndarray:
        """
        Y coordinates from each keypoint's pt[1].

        Returns:
        ----------
        np.ndarray:
            1-D array with shape ``(N,)`` (dtype inferred by NumPy).
        """
        return np.array([kp.pt[1] for kp in self.keypoints])

    @property
    def coordinates(self) -> np.ndarray:
        """
        Stacked (x, y) pairs from each keypoint's pt.

        Returns:
        ----------
        np.ndarray:
            Shape ``(N, 2)``.
        """
        return np.array([kp.pt for kp in self.keypoints])

    @property
    def angles(self) -> np.ndarray:
        """
        Orientation in degrees from each keypoint's angle attribute.

        Returns:
        ----------
        np.ndarray:
            1-D array with shape ``(N,)``.
        """
        return np.array([kp.angle for kp in self.keypoints])

    @property
    def sizes(self) -> np.ndarray:
        """
        Feature diameter / scale from each keypoint's size attribute.

        Returns:
        ----------
        np.ndarray:
            1-D array with shape ``(N,)``.
        """
        return np.array([kp.size for kp in self.keypoints])

    def scale_coordinates(self, factor: float) -> None:
        """
        Scale cv2.KeyPoint positions and sizes in place.

        Parameters:
        ----------
        factor: float
            Multiplier for ``pt`` and ``size`` of each keypoint.

        Raises:
        ----------
        ValueError:
            If factor is not positive.
        """
        if factor <= 0.0:
            raise ValueError(f"factor must be positive, got {factor}")
        self.keypoints = [
            cv2.KeyPoint(
                x=kp.pt[0] * factor,
                y=kp.pt[1] * factor,
                size=kp.size * factor,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id,
            )
            for kp in self.keypoints
        ]

    def apply_mask(self, mask: np.ndarray) -> None:
        """
        Apply a mask to the keypoints.

        Parameters:
        ----------
        mask: np.ndarray
            A boolean mask (0 = ignore, 1 = include).
        """
        kp_array = np.array([kp.pt for kp in self.keypoints])
        
        ix = np.round(kp_array[:, 0]).astype(int)
        iy = np.round(kp_array[:, 1]).astype(int)
        h, w = mask.shape[:2]
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        
        keep = np.zeros(len(self.keypoints), dtype=bool)
        keep[valid] = mask[iy[valid], ix[valid]] > 0
        
        indices = np.flatnonzero(keep)
        
        self.keypoints = [self.keypoints[int(i)] for i in indices]
        if self.descriptors is not None:
            self.descriptors = self.descriptors[indices]
