from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..parameter import OpticalFlowParameters
from .protocol import LucasKanadeFlowFunction


@dataclass
class LucasKanadeParameters(OpticalFlowParameters, ABC):
    """
    Base parameters shared by sparse Lucas-Kanade optical flow processors.

    Concrete variants extend this class in ``pyr_lk`` and ``sparse_rlof``.
    """

    @abstractmethod
    def get_flow_func(self) -> LucasKanadeFlowFunction:
        """
        Return the sparse flow function for this parameter set.

        Returns
        -------
        LucasKanadeFlowFunction
            Callable that tracks keypoints between two preprocessed images.
        """
        ...

    @staticmethod
    def expand_keypoints(keypoints: np.ndarray) -> np.ndarray:
        """
        Expand keypoints to shape ``(N, 1, 2)`` for OpenCV sparse flow APIs.

        Parameters
        ----------
        keypoints: np.ndarray
            Keypoints with shape ``(N, 2)`` or ``(N, 1, 2)``.

        Returns
        -------
        np.ndarray
            Keypoints with shape ``(N, 1, 2)`` and dtype ``float32``.
        """
        return np.ascontiguousarray(keypoints.reshape(-1, 1, 2), dtype=np.float32)

    @staticmethod
    def squeeze_keypoints(keypoints: np.ndarray) -> np.ndarray:
        """
        Remove the singleton tracking axis and return keypoints with shape ``(N, 2)``.

        OpenCV sparse flow APIs use ``(N, 1, 2)``; ``LucasKanadeResult`` uses ``(N, 2)``.

        Parameters
        ----------
        keypoints: np.ndarray
            Keypoints with shape ``(N, 2)`` or ``(N, 1, 2)``.

        Returns
        -------
        np.ndarray
            Keypoints with shape ``(N, 2)`` and dtype ``float32``.
        """
        return np.ascontiguousarray(keypoints.reshape(-1, 2), dtype=np.float32)
