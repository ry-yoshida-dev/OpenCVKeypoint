from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from ..parameter import OpticalFlowParameters


@dataclass
class FarnebackResult:
    """
    Result container for dense Farneback optical flow (``cv2.calcOpticalFlowFarneback``).

    Attributes:
    ----------
    flow: np.ndarray
        Dense flow field with shape ``(H, W, 2)`` containing ``[dx, dy]`` displacements
        computed on the preprocessed (rescaled) image.
    """
    flow: np.ndarray

    def scale(
        self,
        original_shape: tuple[int, int],
        params: OpticalFlowParameters,
    ) -> None:
        """
        Map the flow field back to the original image resolution and coordinates in place.

        Parameters
        ----------
        original_shape: tuple[int, int]
            ``(height, width)`` of the original input image before rescaling.
        params: OpticalFlowParameters
            Parameters that define the scale factor applied before computation.
        """
        if params.scale_factor == 1.0:
            return

        inverse_scale_factor = params.inverse_scale_factor
        self.flow *= inverse_scale_factor

        original_height, original_width = original_shape
        self.flow = cv2.resize(
            self.flow,
            (original_width, original_height),
            interpolation=params.interpolation_flag,
        )
