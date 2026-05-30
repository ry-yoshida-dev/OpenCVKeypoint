from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detectors.harris.parameter import HarrisParameters
    from .detectors.shi_tomashi.parameter import ShiTomashiParameters
    from .parameter import KPDetectionParameters
    from .type_alias import KPDetector

    type DetectionParameterClass = (
        type[KPDetectionParameters]
        | type[ShiTomashiParameters]
        | type[HarrisParameters]
    )

class KPDetectionMethod(Enum):
    """
    Enum for keypoint detection methods.

    Supported method labels (each line matches the corresponding member's string value):
    - MSER
    - SimpleBlob
    - AGAST
    - FAST
    - Harris
    - ShiTomashi
    - AKAZE
    - BRISK
    - KAZE
    - ORB
    - SIFT
    """
    MSER = "MSER"
    SIMPLE_BLOB = "SimpleBlob"
    AGAST = "AGAST"
    FAST = "FAST"
    HARRIS = "Harris"
    SHI_TOMASHI = "ShiTomashi"
    AKAZE = "AKAZE"
    BRISK = "BRISK"
    KAZE = "KAZE"
    ORB = "ORB"
    SIFT = "SIFT"

    def is_brief_supported(self) -> bool:
        """
        Check if the method supports BRIEF-like descriptor extractors.
        (Binary local descriptors)

        Returns:
        ----------
        bool:
            True if the method supports BRIEF-like descriptor extractors, False otherwise.
        """
        return self in [
            self.ORB,     # ORB uses rBRIEF
            self.BRISK,   # Binary descriptor
            self.AKAZE,   # MLDB(binary) and KAZE(float) are supported
           ]

    @property
    def is_binary_descriptor_supported(self) -> bool:
        """
        Check if the method supports binary descriptors.

        Returns:
        ----------
        bool:
            True if the method supports binary descriptors, False otherwise.
        """
        return self in [
            KPDetectionMethod.ORB,
            KPDetectionMethod.BRISK,
            KPDetectionMethod.AKAZE,  # MLDB mode only (Binary descriptor)
            ]
    
    @property
    def is_float_descriptor_supported(self) -> bool:
        """
        Check if the method supports float descriptors.

        Returns:
        ----------
        bool:
            True if the method supports float descriptors, False otherwise.
        """
        return self in [
            self.SIFT,
            self.KAZE,
            self.AKAZE,   # KAZE mode uses float descriptor
            ]

    @property
    def has_descriptor(self) -> bool:
        """
        Check if the method supports descriptor extraction.

        Returns:
        ----------
        bool:
            True if the method supports descriptor extraction, False otherwise.
        """
        return self in [
            self.SIFT,
            self.KAZE,
            self.AKAZE,
            self.ORB,
            self.BRISK,
            ]

    @property
    def is_scale_invariant(self) -> bool:
        """
        Check if the method is scale invariant.

        Returns:
        ----------
        bool:
            True if the method is scale invariant, False otherwise.
        """
        return self in [
            self.SIFT,
            self.KAZE,
            self.AKAZE,
            self.ORB,
            self.BRISK,
            ]

    @property
    def is_rotation_invariant(self) -> bool:
        """
        Check if the method is rotation invariant.

        Returns:
        ----------
        bool:
            True if the method is rotation invariant, False otherwise.
        """
        return self in [
            self.SIFT,
            self.ORB,
            self.BRISK,
            self.KAZE,
            self.AKAZE,
            ]

    @property
    def detector_class(self) -> type[KPDetector]:
        """
        Get the detector class for the method.

        Returns:
        ----------
        type[KPDetector]:
            The detector class for the method.
        """
        match self:
            case KPDetectionMethod.ORB:
                from .detectors import StandardKPDetector
                return StandardKPDetector
            case KPDetectionMethod.SIFT:
                from .detectors import StandardKPDetector
                return StandardKPDetector
            case KPDetectionMethod.BRISK:
                from .detectors import StandardKPDetector
                return StandardKPDetector
            case KPDetectionMethod.AKAZE:
                from .detectors import StandardKPDetector
                return StandardKPDetector
            case KPDetectionMethod.KAZE:
                from .detectors import StandardKPDetector
                return StandardKPDetector
            case KPDetectionMethod.AGAST:
                from .detectors import AGASTDetector
                return AGASTDetector
            case KPDetectionMethod.FAST:
                from .detectors import FASTDetector
                return FASTDetector
            case KPDetectionMethod.MSER:
                from .detectors import MSERDetector
                return MSERDetector
            case KPDetectionMethod.SIMPLE_BLOB:
                from .detectors import SimpleBlobDetector
                return SimpleBlobDetector
            case KPDetectionMethod.SHI_TOMASHI:
                from .detectors import ShiTomashiDetector
                return ShiTomashiDetector
            case KPDetectionMethod.HARRIS:
                from .detectors import HarrisDetector
                return HarrisDetector

    @property
    def parameter_class(
        self,
    ) -> (
        type[KPDetectionParameters]
        | type[ShiTomashiParameters]
        | type[HarrisParameters]
    ):
        """
        Get the parameter class for the method.

        Returns:
        ----------
        type[KPDetectionParameters] | type[ShiTomashiParameters] | type[HarrisParameters]:
            The parameter class for the method.
        """
        match self:
            case KPDetectionMethod.SHI_TOMASHI:
                from .detectors.shi_tomashi.parameter import ShiTomashiParameters

                return ShiTomashiParameters
            case KPDetectionMethod.HARRIS:
                from .detectors.harris.parameter import HarrisParameters

                return HarrisParameters
            case _:
                from .parameter import KPDetectionParameters

                return KPDetectionParameters