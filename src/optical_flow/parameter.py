from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from opencv_utility import OpenCVInterpolationFlag


@dataclass
class OpticalFlowParameters:
    """
    Base parameters shared by all optical flow processors.

    Attributes:
    ----------
    scale_factor: float
        Image scale multiplier applied before optical flow computation.
    interpolation: OpenCVInterpolationFlag
        OpenCV interpolation flag used when scaling images and restoring flow fields.
    """
    scale_factor: float = 1.0
    interpolation: OpenCVInterpolationFlag = OpenCVInterpolationFlag.LINEAR

    def __post_init__(self) -> None:
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be a positive number")

    @cached_property
    def inverse_scale_factor(self) -> float:
        """
        Multiplier that maps scaled coordinates back to the original image.

        Returns
        -------
        float
            ``1.0 / scale_factor`` when scaling is enabled, otherwise ``1.0``.
        """
        if self.scale_factor == 1.0:
            return 1.0
        return 1.0 / self.scale_factor

    @cached_property
    def interpolation_flag(self) -> int:
        """
        OpenCV interpolation constant for ``cv2.resize``.

        Returns
        -------
        int
            Value passed to the ``interpolation`` argument of ``cv2.resize``.
        """
        return self.interpolation.cv2_flag
