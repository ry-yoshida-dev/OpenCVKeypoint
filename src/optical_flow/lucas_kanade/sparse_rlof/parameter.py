from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from ..parameter import LucasKanadeParameters
from ..protocol import LucasKanadeFlowFunction


@dataclass
class SparseRLOFParameters(LucasKanadeParameters):
    """Parameters for ``cv2.optflow.calcOpticalFlowSparseRLOF`` sparse flow."""

    def get_flow_func(self) -> LucasKanadeFlowFunction:
        """
        Return the SparseRLOF sparse flow function.

        Returns
        -------
        LucasKanadeFlowFunction
            ``cv2.optflow.calcOpticalFlowSparseRLOF`` tracker with PyrLK initialization.
        """
        return self._track_keypoints

    def _track_keypoints(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track keypoints with ``cv2.optflow.calcOpticalFlowSparseRLOF``.

        SparseRLOF requires an initialized ``nextPts`` buffer. PyrLK on grayscale
        images provides that initial estimate before RLOF refinement on BGR inputs.

        Parameters
        ----------
        source_image: np.ndarray
            Preprocessed BGR source image.
        target_image: np.ndarray
            Preprocessed BGR target image.
        source_keypoints: np.ndarray
            Source keypoints with shape ``(N, 2)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Target keypoints, status, and error arrays from OpenCV.
        """
        tracking_keypoints = self.expand_keypoints(source_keypoints)
        source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        next_keypoints = np.empty_like(tracking_keypoints)
        initial_target_keypoints, _, _ = cv2.calcOpticalFlowPyrLK(
            source_gray,
            target_gray,
            tracking_keypoints,
            next_keypoints,
        )
        refined_next_keypoints = initial_target_keypoints.copy()

        target_keypoints, status, error = cv2.optflow.calcOpticalFlowSparseRLOF(
            source_image,
            target_image,
            tracking_keypoints,
            refined_next_keypoints,
        )
        return self.squeeze_keypoints(target_keypoints), status, error
