from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from typing_extensions import override

from ...method import OpticalFlowMethod
from ...utils import OpticalFlowPreprocessor
from ..processor import LucasKanadeFlow
from .parameter import SparseRLOFParameters


@dataclass(repr=False, eq=False)
class SparseRLOFFlow(LucasKanadeFlow[SparseRLOFParameters]):
    """
    Sparse optical flow processor backed by ``cv2.optflow.calcOpticalFlowSparseRLOF``.

    Attributes
    ----------
    params: SparseRLOFParameters
        SparseRLOF flow parameters including image scale factor.
    """

    params: SparseRLOFParameters

    @override
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        grayscale_image = OpticalFlowPreprocessor.to_grayscale(image)
        return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    @property
    @override
    def method(self) -> OpticalFlowMethod:
        """Optical flow method implemented by this processor."""
        return OpticalFlowMethod.SPARSE_LUCAS_KANADE
