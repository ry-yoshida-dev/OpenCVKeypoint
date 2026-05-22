import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .parameter import KPDetectionParameters
from .method import KPDetectionMethod
from .results import KeyPointDetectionResult as DetectionResultUnion

DetectorT = TypeVar("DetectorT", bound=cv2.Feature2D | None)
ExtractorT = TypeVar("ExtractorT", bound=cv2.Feature2D | None)
ResultT = TypeVar("ResultT", bound=DetectionResultUnion, covariant=True)
ParamsT = TypeVar("ParamsT", bound=KPDetectionParameters)

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
            Input image, typically (H, W) or (H, W, C) as required by the detector.
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
