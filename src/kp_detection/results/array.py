from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterator

from ..result import KeyPointDetectionResult, KPDetectionStep


@dataclass
class ArrayKPDetectionResult(KeyPointDetectionResult[np.ndarray, np.ndarray]):
    """
    Keypoint detection result backed by a coordinate NumPy ndarray.

    Typical sources include cv2.goodFeaturesToTrack (often ``(N, 1, 2)`` float32).
    Orientation and patch size are not represented; angles and sizes are filled
    with -1.0.

    Attributes:
    ----------
    method: KPDetectionMethod
        The detection method that produced this result (from the base class).
    keypoints: np.ndarray
        Coordinates as ``(N, 2)`` or ``(N, 1, 2)``.
    descriptors: np.ndarray | None
        Descriptor rows aligned with keypoints, or None if not computed.
    """

    keypoints: np.ndarray

    def __len__(self) -> int:
        """
        Number of keypoints (rows in the coordinate array).

        Returns:
        ----------
        int:
            Zero when the array is empty; otherwise the leading dimension size.
        """
        if self.keypoints.size == 0:
            return 0
        return int(self.keypoints.shape[0])

    def __getitem__(self, index: int) -> KPDetectionStep[np.ndarray]:
        """
        Return the coordinate row and optional descriptor at index.

        Parameters:
        ----------
        index: int
            Zero-based index into the keypoint sequence.

        Returns:
        ----------
        KPDetectionStep[np.ndarray]:
            (xy_row, descriptor_row_or_none); xy row has shape ``(2,)``.
        """
        row = self._as_xy_pairs()[index]
        if self.descriptors is None:
            return (row, None)
        return (row, self.descriptors[index])

    def __iter__(self) -> Iterator[KPDetectionStep[np.ndarray]]:
        """
        Iterate over (xy row, descriptor or None) pairs.

        Returns:
        ----------
        Iterator[KPDetectionStep[np.ndarray]]:
            Each keypoint is a length-2 coordinate vector; descriptor is None
            when absent or when descriptors are an empty array.
        """
        xy = self._as_xy_pairs()
        if self.descriptors is None:
            return iter((xy[i], None) for i in range(len(xy)))
        if len(self.descriptors) != 0:
            return iter((xy[i], self.descriptors[i]) for i in range(len(xy)))
        return iter((xy[i], None) for i in range(len(xy)))

    def __str__(self) -> str:
        """
        Short summary with keypoints and descriptors shapes.

        Returns:
        ----------
        str:
            String representation for logging or display.
        """
        if self.descriptors is None:
            return (
                f"ArrayKPDetectionResult(keypoints.shape={self.keypoints.shape}, "
                "descriptors=None)"
            )
        return (
            f"ArrayKPDetectionResult(keypoints.shape={self.keypoints.shape}, "
            f"descriptors.shape={self.descriptors.shape})"
        )

    @property
    def x(self) -> np.ndarray:
        """
        X coordinates as a 1-D float64 array (length N).

        Returns:
        ----------
        np.ndarray:
            Column 0 of the normalized ``(N, 2)`` coordinates.
        """
        return self._as_xy_pairs()[:, 0]

    @property
    def y(self) -> np.ndarray:
        """
        Y coordinates as a 1-D float64 array (length N).

        Returns:
        ----------
        np.ndarray:
            Column 1 of the normalized ``(N, 2)`` coordinates.
        """
        return self._as_xy_pairs()[:, 1]

    @property
    def coordinates(self) -> np.ndarray:
        """
        All keypoints as ``(N, 2)`` float64 with x and y columns.

        Returns:
        ----------
        np.ndarray:
            Normalized coordinate array.
        """
        return self._as_xy_pairs()

    @property
    def angles(self) -> np.ndarray:
        """
        Placeholder angles: array of -1.0 (no angle data for plain arrays).

        Returns:
        ----------
        np.ndarray:
            Shape ``(N,)``, dtype float64.
        """
        n = len(self)
        return np.full(n, -1.0, dtype=np.float64)

    @property
    def sizes(self) -> np.ndarray:
        """
        Placeholder sizes: array of -1.0 (no scale data for plain arrays).

        Returns:
        ----------
        np.ndarray:
            Shape ``(N,)``, dtype float64.
        """
        n = len(self)
        return np.full(n, -1.0, dtype=np.float64)

    def scale_coordinates(self, factor: float) -> None:
        """
        Scale stored coordinate rows in place.

        Parameters:
        ----------
        factor: float
            Multiplier for each (x, y) pair. Array layout ``(N, 2)`` or
            ``(N, 1, 2)`` is preserved.

        Raises:
        ----------
        ValueError:
            If factor is not positive.
        """
        if factor <= 0.0:
            raise ValueError(f"factor must be positive, got {factor}")
        if self.keypoints.size == 0:
            return
        original = self.keypoints
        scaled_xy = self._as_xy_pairs() * factor
        if original.ndim == 3 and original.shape[1] == 1:
            self.keypoints = scaled_xy.reshape(-1, 1, 2).astype(
                original.dtype, copy=False
            )
        else:
            self.keypoints = scaled_xy.astype(original.dtype, copy=False)

    def _as_xy_pairs(self) -> np.ndarray:
        """
        Normalize self.keypoints to a float ``(N, 2)`` array.

        Returns:
        ----------
        np.ndarray
            Float64 array with shape ``(N, 2)`` and x/y columns.

        Raises:
        ----------
        ValueError:
            If keypoints is not ``(N, 2)`` or ``(N, 1, 2)``.
        """
        keypoints = self.keypoints
        if keypoints.size == 0:
            return keypoints.reshape(0, 2)
        coords = np.asarray(keypoints, dtype=np.float64)
        if coords.ndim == 3 and coords.shape[1] == 1:
            coords = coords.reshape(-1, 2)
        elif coords.ndim == 2 and coords.shape[1] == 2:
            pass
        else:
            raise ValueError(
                "keypoints must have shape (N, 2) or (N, 1, 2), "
                f"got {tuple(coords.shape)}"
            )
        return coords