from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

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
    filtered_source_keypoints: np.ndarray = field(init=False)
    filtered_target_keypoints: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        is_tracked = self.status.ravel() == 1
        self.filtered_source_keypoints = self.source_keypoints[is_tracked]
        self.filtered_target_keypoints = self.target_keypoints[is_tracked]
        if len(self.filtered_source_keypoints) == 0:
            warnings.warn(
                "No keypoints were successfully tracked (all status values are 0). "
                "Returning zero-valued flow statistics.",
                stacklevel=2,
            )
            zero_keypoints = np.array([[0.0, 0.0]], dtype=np.float32)
            self.filtered_source_keypoints = zero_keypoints
            self.filtered_target_keypoints = zero_keypoints.copy()

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
        is_tracked = self.status.ravel() == 1
        self.filtered_source_keypoints = self.source_keypoints[is_tracked]
        self.filtered_target_keypoints = self.target_keypoints[is_tracked]
        if len(self.filtered_source_keypoints) == 0:
            zero_keypoints = np.array([[0.0, 0.0]], dtype=np.float32)
            self.filtered_source_keypoints = zero_keypoints
            self.filtered_target_keypoints = zero_keypoints.copy()

    @property
    def flow(self) -> np.ndarray:
        """
        Flow field between successfully tracked source and target keypoints.

        Returns
        -------
        np.ndarray
            Flow field with shape (N, 2) between filtered source and target keypoints.
        """
        return self.filtered_target_keypoints - self.filtered_source_keypoints

    @property
    def mean_flow(self) -> np.ndarray:
        """
        Mean flow field between successfully tracked keypoints.

        Returns
        -------
        np.ndarray
            Mean flow field with shape (2,) between filtered source and target keypoints.
        """
        return np.mean(self.flow, axis=0)

    @property
    def l2_norm(self) -> np.ndarray:
        """
        L2 norm of flow field between successfully tracked keypoints.

        Returns
        -------
        np.ndarray
            L2 norm of flow field with shape (N,) between filtered source and target keypoints.
        """
        return np.linalg.norm(self.flow, axis=1)

    @property
    def mean_l2_norm(self) -> float:
        """
        Mean L2 norm of flow field between successfully tracked keypoints.

        Returns
        -------
        float
            Mean L2 norm of flow field between filtered source and target keypoints.
        """
        return float(np.mean(self.l2_norm))

    @property
    def flow_std(self) -> np.ndarray:
        """
        Standard deviation of flow field between successfully tracked keypoints.

        Returns
        -------
        np.ndarray
            Standard deviation of flow field with shape (2,) between filtered source and target keypoints.
        """
        return np.std(self.flow, axis=0)

    def __len__(self) -> int:
        """
        Number of successfully tracked keypoints.

        Returns
        -------
        int
            Number of keypoints with ``status == 1``.
        """
        return int(np.count_nonzero(self.status.ravel() == 1))
