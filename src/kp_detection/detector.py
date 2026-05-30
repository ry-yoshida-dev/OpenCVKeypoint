import warnings

import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from .parameter import KPDetectionParameters
from .method import KPDetectionMethod
from .result import KeyPointDetectionResult as DetectionResultBase

DetectorT = TypeVar("DetectorT", bound=cv2.Feature2D | None)
ExtractorT = TypeVar("ExtractorT", bound=cv2.Feature2D | None)
ResultT = TypeVar(
    "ResultT",
    bound=DetectionResultBase[Any, Any],
    covariant=True,
)
ParamsT = TypeVar("ParamsT", bound=KPDetectionParameters, covariant=True)

@dataclass(repr=False, eq=False)
class KeyPointDetector(ABC, Generic[DetectorT, ExtractorT, ResultT, ParamsT]):
    """
    Base class for keypoint detectors.

    Type parameter ResultT is the concrete KeyPointDetectionResult subclass (see
    kp_detection.result) returned by detect().
    Type parameter ParamsT is the concrete KPDetectionParameters subclass for this detector.

    Attributes:
    ----------
    detector: DetectorT
        The detector.
    extractor: ExtractorT
        The extractor.
    params: ParamsT
        Keypoint detection parameters for this detector.
    """
    params: ParamsT
    detector: DetectorT = field(init=False)
    extractor: ExtractorT = field(init=False)
    brief: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.detector, self.extractor = self._define_detector()
        if self.params.is_brief_applied:
            self.brief = self.params.brief

    @abstractmethod
    def detect(
        self,
        img: np.ndarray,
        mask: np.ndarray | None = None
        ) -> ResultT:
        """
        Detect keypoints in an image.

        Parameters:
        ----------
        img: np.ndarray
            Input image, typically ``(H, W)`` or ``(H, W, C)`` as required by the detector.
        mask: np.ndarray | None
            Boolean mask (0 = ignore, 1 = include).
            
        Returns
        ----------
        ResultT
            The result of keypoint detection (subclass-specific).
        """

    @abstractmethod
    def _define_detector(self) -> tuple[DetectorT, ExtractorT]:
        """
        Define the detector.

        Returns
        -------
        tuple[DetectorT, ExtractorT]
            The detector and the extractor (or None if not used).
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the detector.

        Returns:
        ----------
        str: The string representation of the detector.
        """

    @property
    def method(self) -> KPDetectionMethod:
        """
        Get the method of the detector.

        Returns:
        ----------
        KPDetectionMethod: The method of the detector.
        """
        return self.params.method

    @property
    def is_brief_applied(self) -> bool:
        """
        Get whether the detector uses brief descriptor extractor.

        Returns:
        ----------
        bool: The whether the detector uses brief descriptor extractor.
        """
        return self.params.is_brief_applied

    @property
    def image_scaler(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get the image scaler function.

        Returns:
        ----------
        Callable[[np.ndarray], np.ndarray]: The image scaler function.
        """
        return self.params.image_scaler

    @property
    def mask_rescaler(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get the mask rescaler function.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Resizes a 2D mask to match ``image_scaler`` output dimensions.
        """
        return self.params.mask_rescaler

    @property
    def scale_factor(self) -> float:
        """
        Get the scale factor.

        Returns:
        ----------
        float: The scale factor.
        """
        return self.params.scale_factor

    def _warn_if_mask_unused(self, mask: np.ndarray | None) -> None:
        """
        Warn when ``mask`` is passed to a detector that does not apply it.

        Parameters
        ----------
        mask : np.ndarray | None
            Mask argument from ``detect()``.
        """
        if mask is not None:
            warnings.warn(
                f"{type(self).__name__} does not use mask; the argument is ignored.",
                UserWarning,
                stacklevel=3,
            )

    def _scaled_detection_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        """
        Map an original-space mask into working (scaled) image coordinates.

        Parameters
        ----------
        mask : np.ndarray | None
            Boolean mask aligned with the input ``img`` passed to ``detect()``.

        Returns
        -------
        np.ndarray | None
            Mask with the same spatial extent as ``image_scaler(img)``, or None.
        """
        if mask is None:
            return None
        if mask.ndim != 2:
            raise ValueError(f"mask must be a 2D array, got shape {mask.shape}")
        return self.mask_rescaler(mask)

    def _remap_result_to_original_coordinates(
        self,
        result: DetectionResultBase[Any, Any],
    ) -> None:
        """
        Map detection coordinates from the working (scaled) image to the input image.

        Parameters
        ----------
        result : DetectionResultBase[Any, Any]
            Detection result produced on ``image_scaler(img)``; updated in place.
        """
        scale_factor = self.scale_factor
        if scale_factor != 1.0:
            result.scale_coordinates(1.0 / scale_factor)
