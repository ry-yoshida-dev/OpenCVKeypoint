from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Callable

from ..parameter import OpticalFlowParameters


@dataclass
class FarnebackParameters(OpticalFlowParameters):
    """
    FarnebackParameters is the parameters for the Farneback optical flow processor.

    Attributes:
    ----------
    pyr_scale: float
        The pyramid scale.
    levels: int
        The number of levels.
    winsize: int
        The window size.
    iterations: int
        The number of iterations.
    poly_n: int
        The polynomial order.
    poly_sigma: float
        The polynomial sigma.
    flags: int
        The flags.
    """
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    def define_function(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return lambda source_image, target_image: cv2.calcOpticalFlowFarneback(
            prev=source_image,
            next=target_image,
            flow=None,  # type: ignore[arg-type]
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags,
        )
