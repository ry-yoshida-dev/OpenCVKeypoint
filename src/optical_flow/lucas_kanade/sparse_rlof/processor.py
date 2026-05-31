from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

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

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        grayscale_image = OpticalFlowPreprocessor.to_grayscale(image)
        return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    @property
    def method(self) -> OpticalFlowMethod:
        return OpticalFlowMethod.SPARSE_LUCAS_KANADE
