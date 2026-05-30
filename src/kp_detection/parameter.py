from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from opencv_utility import OpenCVInterpolationFlag

from .method import KPDetectionMethod

if TYPE_CHECKING:
    from .type_alias import KPDetector

@dataclass
class KPDetectionParameters:
    """
    Base parameters for keypoint detection (subclasses add method-specific fields).

    Attributes:
    ----------
    method: KPDetectionMethod
        The method for keypoint detection.
    is_brief_applied: bool
        Whether to use brief descriptor extractor.
    scale_factor: float
        Multiplier for resizing the input image before detection (1.0 = no scaling).
        Detection runs on the rescaled working image; returned keypoint coordinates
        are remapped to the original input image space by each detector.
    interpolation: OpenCVInterpolationFlag
        OpenCV interpolation flag used when scaling the input image.
    image_scaler: Callable[[np.ndarray], np.ndarray]
        Built once at init: identity when ``scale_factor`` is ``1.0``,
        otherwise ``cv2.resize`` with ``fx``/``fy`` = ``scale_factor``.
    mask_rescaler: Callable[[np.ndarray], np.ndarray]
        Built once at init: identity when ``scale_factor`` is ``1.0``,
        otherwise ``cv2.resize`` with ``fx``/``fy`` = ``scale_factor`` and
        ``interpolation=cv2.INTER_NEAREST`` (preserves binary masks).
    """
    method: KPDetectionMethod = KPDetectionMethod.SIFT
    is_brief_applied: bool = False
    scale_factor: float = 1.0
    interpolation: OpenCVInterpolationFlag = OpenCVInterpolationFlag.LINEAR
    brief: Any = field(init=False, default=None)
    image_scaler: Callable[[np.ndarray], np.ndarray] = field(init=False)
    mask_rescaler: Callable[[np.ndarray], np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        """
        Post-initialization validation.
        """
        self._validate()
        self.image_scaler = self._build_image_scaler()
        self.mask_rescaler = self._build_mask_rescaler()
        if self.is_brief_applied:
            self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() # type: ignore

    def _validate(self) -> None:
        """
        Validate the parameters.
        """
        if self.scale_factor <= 0.0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")
        if self.is_brief_applied and not self.method.is_brief_supported():
            brief_methods = "\n".join(
                f"  - {m.value}" for m in KPDetectionMethod if m.is_brief_supported()
            )
            warnings.warn(
                "BRIEF descriptor extractor is not supported for "
                f"{self.method.value}. is_brief_applied is set to False.\n"
                "BRIEF is supported for:\n"
                f"{brief_methods}"
            )
            self.is_brief_applied = False

    def _build_image_scaler(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Build ``image_scaler`` from current ``scale_factor`` and ``interpolation``.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Identity when ``scale_factor`` is ``1.0``; otherwise resize via ``cv2.resize``.
        """
        if self.scale_factor == 1.0:
            return lambda img: img
        interpolation = self.interpolation.cv2_flag
        scale = self.scale_factor
        return lambda img: cv2.resize(
            img,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=interpolation,
        )

    def _build_mask_rescaler(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Build ``mask_rescaler`` from current ``scale_factor``.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Identity when ``scale_factor`` is ``1.0``; otherwise nearest-neighbor resize.
        """
        if self.scale_factor == 1.0:
            return lambda mask: mask
        scale = self.scale_factor
        return lambda mask: cv2.resize(
            mask,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

    def build_detector(self) -> KPDetector:
        """
        Build a keypoint detector instance based on the parameters.

        For SHI_TOMASHI use ShiTomashiParameters; for HARRIS use HarrisParameters.
        Calling this on base KPDetectionParameters with those methods raises ValueError.
        Other methods use this class as-is.

        Returns:
        ----------
        KPDetector
            Detector instance for methods handled by KPDetectionParameters.
            Shi-Tomasi and Harris are built from ShiTomashiParameters and HarrisParameters.
        """

        if self.method in [KPDetectionMethod.SHI_TOMASHI, KPDetectionMethod.HARRIS]:
            raise ValueError(f"Invalid method: {self.method}, Shi-Tomasi and Harris are built from ShiTomashiParameters and HarrisParameters.")
        detector_class = self.method.detector_class
        return detector_class(params=self)
    


