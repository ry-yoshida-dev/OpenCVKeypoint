from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ..method import OpticalFlowMethod
from ..processor import OpticalFlowProcessor
from ..utils import OpticalFlowPreprocessor, OpticalFlowValidator
from .parameter import LucasKanadeParameters
from .protocol import LucasKanadeFlowFunction
from .result import LucasKanadeResult
from kp_detection.detectors import ShiTomashiDetector, ShiTomashiParameters

ParamsT = TypeVar("ParamsT", bound=LucasKanadeParameters)


@dataclass(repr=False, eq=False)
class LucasKanadeFlow(
    OpticalFlowProcessor[ParamsT, LucasKanadeResult],
    ABC,
    Generic[ParamsT],
):
    """
    Abstract base processor for sparse Lucas-Kanade optical flow.

    Attributes
    ----------
    params: ParamsT
        Parameters for the selected sparse flow variant.
    keypoint_params: ShiTomashiParameters
        Parameters for Shi-Tomasi keypoint detection.
    shi_tomashi_detector: ShiTomashiDetector
        Detector used to seed sparse flow keypoints.
    flow_function: LucasKanadeFlowFunction
        Sparse flow callable selected by ``params.get_flow_func()``.
    """
    params: ParamsT
    keypoint_params: ShiTomashiParameters
    shi_tomashi_detector: ShiTomashiDetector = field(init=False)
    flow_function: LucasKanadeFlowFunction = field(init=False)

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
        Run sparse Lucas-Kanade optical flow.

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
        -------
        LucasKanadeResult
            Sparse flow result in original-image coordinates.
        """
        OpticalFlowValidator.validate_run_inputs(source_image, target_image, mask=mask)

        preprocessed_source = self.preprocess_image(source_image)
        preprocessed_target = self.preprocess_image(target_image)
        preprocessed_mask = self.preprocess_mask(mask)
        detection_mask = OpticalFlowPreprocessor.prepare_detection_mask(preprocessed_mask)

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
        Compute sparse flow on preprocessed inputs.

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
            Sparse flow result in preprocessed-image coordinates.
        """
        target_keypoints, status, error = self.flow_function(
            source_image,
            target_image,
            source_keypoints,
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
        Get source keypoints from a preprocessed source image.

        Parameters
        ----------
        source_image: np.ndarray
            Preprocessed source image.
        mask: np.ndarray | None
            Preprocessed mask.

        Returns
        -------
        np.ndarray
            Source keypoints with shape ``(N, 2)`` and dtype ``float32``.
        """
        source_keypoints = np.array(
            self.shi_tomashi_detector.detect(source_image, mask=mask).keypoints
        )

        if np.any(np.isnan(source_keypoints)):
            raise NotImplementedError("No previous keypoints found.")

        return source_keypoints.reshape(-1, 2).astype(np.float32)

    @abstractmethod
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply method-specific preprocessing to a rescaled image.

        Parameters
        ----------
        image: np.ndarray
            Rescaled input image.

        Returns
        -------
        np.ndarray
            Image ready for sparse flow computation.
        """

    @property
    @abstractmethod
    def method(self) -> OpticalFlowMethod:
        """Return the optical flow method enum for this processor."""

    def __str__(self) -> str:
        return f"{type(self).__name__}(params={self.params!r}, keypoint_params={self.keypoint_params!r})"
