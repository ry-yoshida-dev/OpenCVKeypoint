from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from ..method import OpticalFlowMethod
from ..processor import OpticalFlowProcessor
from ..validator import OpticalFlowValidator
from .parameter import LucasKanadeParameters
from .result import LucasKanadeResult
from kp_detection.detectors import ShiTomashiDetector, ShiTomashiParameters


@dataclass(repr=False, eq=False)
class LucasKanadeFlow(OpticalFlowProcessor[LucasKanadeParameters, LucasKanadeResult]):
    """
    LucasKanadeFlow is the processor for the Lucas-Kanade optical flow.

    Attributes:
    ----------
    params: LucasKanadeParameters
        The parameters for the Lucas-Kanade optical flow.
    keypoint_params: ShiTomashiParameters
        The parameters for Shi-Tomasi keypoint detection.
    shi_tomashi_detector: ShiTomashiDetector
        The detector for Shi-Tomasi keypoint detection.
    flow_function: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]
        OpenCV optical flow callable. The fourth argument is ``nextPts`` for PyrLK.
    """
    params: LucasKanadeParameters
    keypoint_params: ShiTomashiParameters
    shi_tomashi_detector: ShiTomashiDetector = field(init=False)
    flow_function: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = field(init=False)

    def __post_init__(self) -> None:
        self.shi_tomashi_detector = self.keypoint_params.build_detector()
        self.flow_function = self.params.get_flow_func()

    def run(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> LucasKanadeResult:
        """
        Run Lucas-Kanade optical flow.

        Parameters
        ----------
        source_image : np.ndarray
            The source image.
        target_image : np.ndarray
            The target image.
        mask : np.ndarray, optional
            Optional mask in original-image coordinates. Allowed regions are non-zero.
            The mask constrains Shi-Tomasi keypoint detection only.

        Returns
        ----------
        LucasKanadeResult
            The result of Lucas-Kanade optical flow in original-image coordinates.
        """
        OpticalFlowValidator.validate_run_inputs(source_image, target_image, mask=mask)

        preprocessed_source = self.preprocess_image(source_image)
        preprocessed_target = self.preprocess_image(target_image)
        preprocessed_mask = self.preprocess_mask(mask)
        detection_mask = OpticalFlowValidator.prepare_detection_mask(preprocessed_mask)

        source_keypoints = self._get_source_keypoints(
            preprocessed_source,
            mask=detection_mask,
        )
        result = self.compute_flow(
            source_image=preprocessed_source,
            target_image=preprocessed_target,
            source_keypoints=source_keypoints,
        )
        result.scale(self.params)
        return result

    def compute_flow(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_keypoints: np.ndarray,
    ) -> LucasKanadeResult:
        """
        Compute Lucas-Kanade flow on preprocessed inputs.

        Parameters
        ----------
        source_image: np.ndarray
            Preprocessed source image.
        target_image: np.ndarray
            Preprocessed target image.
        source_keypoints: np.ndarray
            Source keypoints in preprocessed-image coordinates.

        Returns
        -------
        LucasKanadeResult
            The result of Lucas-Kanade optical flow in preprocessed-image coordinates.
        """
        target_keypoints, status, error = self.flow_function(
            source_image,
            target_image,
            source_keypoints,
            None,  # nextPts: let OpenCV allocate tracked points
        )
        return LucasKanadeResult(
            source_keypoints=source_keypoints,
            target_keypoints=target_keypoints,
            status=status,
            error=error,
        )

    def _get_source_keypoints(
        self,
        source_image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Get the source keypoints from a preprocessed source image.

        Parameters:
        ----------
        source_image: np.ndarray
            Preprocessed source image.
        mask: np.ndarray | None
            Preprocessed mask.

        Returns:
        ----------
        np.ndarray
            Source keypoints in preprocessed-image coordinates.
        """
        source_keypoints = np.array(
            self.shi_tomashi_detector.detect(source_image, mask=mask).keypoints
        )

        if np.any(np.isnan(source_keypoints)):
            raise NotImplementedError("No previous keypoints found.")

        source_keypoints = source_keypoints.reshape(-1, 2)

        return source_keypoints.astype(np.float32)

    def _preprocess_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Preprocess the image for Lucas-Kanade optical flow.

        Parameters
        ----------
        image: np.ndarray
            The image to preprocess.

        Returns
        -------
        np.ndarray
            The preprocessed image.
        """
        grayscale_image = OpticalFlowValidator.to_grayscale(image)
        if self.params.is_SparseRLOF:
            return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        return grayscale_image

    def __str__(self) -> str:
        return f"LucasKanadeFlow(params={self.params!r}, keypoint_params={self.keypoint_params!r})"

    @property
    def method(self) -> OpticalFlowMethod:
        match self.params.is_SparseRLOF:
            case True:
                return OpticalFlowMethod.SPARSE_LUCAS_KANADE
            case False:
                return OpticalFlowMethod.LUCAS_KANADE