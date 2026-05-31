from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from ..method import OpticalFlowMethod
from ..processor import OpticalFlowProcessor
from ..utils import OpticalFlowPreprocessor, OpticalFlowValidator
from .parameter import FarnebackParameters
from .result import FarnebackResult


@dataclass(repr=False, eq=False)
class FarnebackFlow(OpticalFlowProcessor[FarnebackParameters, FarnebackResult]):
    """
    FarnebackFlow is the processor for the Farneback optical flow.

    Attributes:
    ----------
    params: FarnebackParameters
        The parameters for the Farneback optical flow.
    function: Callable[[np.ndarray, np.ndarray], np.ndarray]
        The function for the optical flow.
    """
    params: FarnebackParameters
    function: Callable[[np.ndarray, np.ndarray], np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.function = self.params.define_function()

    @property
    def method(self) -> OpticalFlowMethod:
        return OpticalFlowMethod.FARNEBACK

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        return OpticalFlowPreprocessor.to_grayscale(image)

    def run(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> FarnebackResult:
        """
        Run Farneback optical flow.

        Parameters
        ----------
        source_image : np.ndarray
            The source image.
        target_image : np.ndarray
            The target image.
        mask : np.ndarray, optional
            Optional mask validated against the input shape. Farneback does not consume
            this mask during computation; filter the returned flow with
            ``result.flow[mask]`` after ``run()``.

        Returns
        -------
        FarnebackResult
            Dense flow result with ``flow`` of shape ``(H, W, 2)`` in original-image
            coordinates. Apply an optional ``mask`` with ``result.flow[mask]`` to extract
            vectors at selected locations.
        """
        OpticalFlowValidator.validate_run_inputs(source_image, target_image, mask=mask)
        original_shape = source_image.shape[:2]

        preprocessed_source = self.preprocess_image(source_image)
        preprocessed_target = self.preprocess_image(target_image)

        flow = self.function(preprocessed_source, preprocessed_target)
        result = FarnebackResult(flow=flow)
        result.scale(
            original_shape=original_shape,
            params=self.params,
        )
        return result

    def __str__(self) -> str:
        return f"FarnebackFlow(params={self.params!r})"
