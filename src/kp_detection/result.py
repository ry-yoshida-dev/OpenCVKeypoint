from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar


from .method import KPDetectionMethod

KPT = TypeVar("KPT", covariant=True)
KPElem = TypeVar("KPElem", covariant=True)

type KPDetectionStep[T] = tuple[T, np.ndarray | None]


@dataclass
class KeyPointDetectionResult(Generic[KPT, KPElem], ABC):
    """
    Abstract base container for keypoint detection results.

    Subclasses set KPT (batch storage) and KPElem (one element from that batch).

    Attributes:
    ----------
    method: KPDetectionMethod
        The detection method that produced this result.
    keypoints: KPT
        Keypoint data in the subclass-specific format.
    descriptors: np.ndarray | None
        Descriptor rows aligned with keypoints, or None if not computed.
    """

    method: KPDetectionMethod
    keypoints: KPT
    descriptors: np.ndarray | None

    def __post_init__(self) -> None:
        """
        Validate that keypoints and descriptors have compatible lengths.

        Raises:
        ----------
        ValueError:
            If descriptors are present and non-empty but their row count does not
            match the number of keypoints.
        """
        if self.descriptors is None:
            return
        if len(self.descriptors) != 0 and len(self) != len(self.descriptors):
            raise ValueError("keypoints and descriptors must have the same length")

    @abstractmethod
    def __len__(self) -> int:
        """
        Number of keypoints in this result.

        Returns:
        ----------
        int:
            The count of keypoints.
        """

    @abstractmethod
    def __getitem__(self, index: int) -> KPDetectionStep[KPElem]:
        """
        Return the keypoint and descriptor at the given index.

        Parameters:
        ----------
        index: int
            Zero-based index into the keypoint sequence.

        Returns:
        ----------
        KPDetectionStep[KPElem]:
            (keypoint, descriptor_row_or_none); same shape as each step from
            __iter__.
        """

    def __iter__(self) -> Iterator[KPDetectionStep[KPElem]]:
        """
        Iterate over (keypoint, descriptor_or_none) pairs.

        The default implementation indexes via __getitem__. Subclasses whose
        __getitem__ is expensive per index should override this for linear-time
        iteration.

        Returns:
        ----------
        Iterator[KPDetectionStep[KPElem]]:
            Yields the same pairs as repeated __getitem__ calls in order.
        """
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def __str__(self) -> str:
        """
        Short human-readable summary (counts / array shapes).

        Returns:
        ----------
        str:
            String representation for logging or display.
        """

    @property
    @abstractmethod
    def x(self) -> np.ndarray:
        """
        X coordinates for each keypoint (1-D array with shape ``(N,)``).

        Returns:
        ----------
        np.ndarray:
            One value per keypoint, aligned with iteration order.
        """

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        """
        Y coordinates for each keypoint (1-D array with shape ``(N,)``).

        Returns:
        ----------
        np.ndarray:
            One value per keypoint, aligned with iteration order.
        """

    @property
    @abstractmethod
    def coordinates(self) -> np.ndarray:
        """
        Stacked (x, y) pairs for all keypoints.

        Returns:
        ----------
        np.ndarray:
            Shape ``(N, 2)`` (or subclass-equivalent layout).
        """

    @property
    @abstractmethod
    def angles(self) -> np.ndarray:
        """
        Orientation per keypoint (degrees), or placeholders when unknown.

        Returns:
        ----------
        np.ndarray:
            1-D array with shape ``(N,)``.
        """

    @property
    @abstractmethod
    def sizes(self) -> np.ndarray:
        """
        Feature scale / diameter per keypoint, or placeholders when unknown.

        Returns:
        ----------
        np.ndarray:
            1-D array with shape ``(N,)``.
        """

    @abstractmethod
    def scale_coordinates(self, factor: float) -> None:
        """
        Scale keypoint positions (and feature sizes where applicable) in place.

        Parameters:
        ----------
        factor: float
            Multiplier applied to each (x, y) and to feature size when present.
            Use ``1.0 / detection_scale_factor`` to map coordinates from a
            downscaled working image back to the original image space.

        Raises:
        ----------
        ValueError:
            If factor is not positive.
        """