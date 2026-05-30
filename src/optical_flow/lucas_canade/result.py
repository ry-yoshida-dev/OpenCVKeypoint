from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..parameter import OpticalFlowParameters


@dataclass
class LucasKanadeResult:
    """
    Result container for sparse Lucas-Kanade optical flow (``cv2.calcOpticalFlowPyrLK``).

    Each row index ``i`` across all fields refers to the same feature:
    ``source_keypoints[i]`` was tracked to ``target_keypoints[i]`` with
    ``status[i]`` and ``error[i]``.

    Attributes:
    ----------
    source_keypoints: np.ndarray
        Feature locations in the source (previous) image, shape ``(N, 1, 2)`` or ``(N, 2)``,
        dtype ``float32``. Each point is ``(x, y)`` in pixel coordinates.
    target_keypoints: np.ndarray
        Estimated feature locations in the target (next) image, same shape and dtype as
        ``source_keypoints``. Failed tracks may contain unreliable values; filter with
        ``status``.
    status: np.ndarray
        Per-feature tracking success flag from OpenCV, shape ``(N, 1)``, dtype ``uint8``.
        ``1`` means the feature was successfully tracked; ``0`` means tracking failed
        (e.g. out of frame, insufficient texture, or motion too large).
    error: np.ndarray
        Per-feature tracking error from OpenCV, shape ``(N, 1)``, dtype ``float32``.
        Lower values indicate a more reliable match. Useful for ranking or filtering
        tracks in addition to ``status``.
    """
    source_keypoints: np.ndarray
    target_keypoints: np.ndarray
    status: np.ndarray
    error: np.ndarray

    def scale(self, params: OpticalFlowParameters) -> None:
        """
        Map keypoint coordinates back to the original image in place.

        Parameters
        ----------
        params: OpticalFlowParameters
            Parameters that define the scale factor applied before computation.
        """
        if params.scale_factor == 1.0:
            return

        inverse_scale_factor = params.inverse_scale_factor
        self.source_keypoints *= inverse_scale_factor
        self.target_keypoints *= inverse_scale_factor

    @property
    def flow(self) -> np.ndarray:
        """
        Flow field between source and target keypoints.

        Returns
        -------
        np.ndarray
            Flow field with shape (N, 2) between source and target keypoints.
        """
        return self.target_keypoints - self.source_keypoints

    @property
    def mean_flow(self) -> np.ndarray:
        """
        Mean flow field between source and target keypoints.

        Returns
        -------
        np.ndarray
            Mean flow field with shape (2,) between source and target keypoints.
        """
        return np.mean(self.flow, axis=0)

    def __len__(self) -> int:
        """
        Number of tracked keypoints.

        Returns
        -------
        int
            Number of tracked keypoints.
        """
        return len(self.source_keypoints)
