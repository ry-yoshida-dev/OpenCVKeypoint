from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from ..parameter import LucasKanadeParameters
from ..protocol import LucasKanadeFlowFunction


@dataclass
class PyrLKParameters(LucasKanadeParameters):
    """Parameters for ``cv2.calcOpticalFlowPyrLK`` sparse flow."""

    def get_flow_func(self) -> LucasKanadeFlowFunction:
        """
        Return the PyrLK sparse flow function.

        Returns
        -------
        LucasKanadeFlowFunction
            ``cv2.calcOpticalFlowPyrLK`` tracker.
        """
        return self._track_keypoints

    def _track_keypoints(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track keypoints with ``cv2.calcOpticalFlowPyrLK``.

        Parameters
        ----------
        source_image: np.ndarray
            Preprocessed grayscale source image.
        target_image: np.ndarray
            Preprocessed grayscale target image.
        source_keypoints: np.ndarray
            Source keypoints with shape ``(N, 2)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Target keypoints, status, and error arrays from OpenCV.
        """
        tracking_keypoints = self.expand_keypoints(source_keypoints)
        next_keypoints = np.empty_like(tracking_keypoints)
        target_keypoints, status, error = cv2.calcOpticalFlowPyrLK(
            source_image,
            target_image,
            tracking_keypoints,
            next_keypoints,
        )
        return self.squeeze_keypoints(target_keypoints), status, error
