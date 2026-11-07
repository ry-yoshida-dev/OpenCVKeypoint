from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from ..parameter import OpticalFlowParameters


@dataclass
class FarnebackParameters(OpticalFlowParameters):
    """
    FarnebackParameters is the parameters for the Farneback optical flow processor.

    Attributes
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
        """Build the dense flow function bound to these parameters."""

        def compute_flow(
            source_image: np.ndarray, target_image: np.ndarray
        ) -> np.ndarray:
            # OpenCV allocates the flow field when flow is None, but its stubs do
            # not accept None there.
            flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
                prev=source_image,
                next=target_image,
                flow=None,  # pyright: ignore[reportArgumentType]
                pyr_scale=self.pyr_scale,
                levels=self.levels,
                winsize=self.winsize,
                iterations=self.iterations,
                poly_n=self.poly_n,
                poly_sigma=self.poly_sigma,
                flags=self.flags,
            )
            return cast(np.ndarray, flow)

        return compute_flow
