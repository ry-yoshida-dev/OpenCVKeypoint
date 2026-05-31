from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ...method import OpticalFlowMethod
from ...utils import OpticalFlowPreprocessor
from ..processor import LucasKanadeFlow
from .parameter import PyrLKParameters


@dataclass(repr=False, eq=False)
class PyrLKFlow(LucasKanadeFlow[PyrLKParameters]):
    """
    Sparse optical flow processor backed by ``cv2.calcOpticalFlowPyrLK``.

    Attributes
    ----------
    params: PyrLKParameters
        PyrLK flow parameters including image scale factor.
    """
    params: PyrLKParameters

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        return OpticalFlowPreprocessor.to_grayscale(image)

    @property
    def method(self) -> OpticalFlowMethod:
        return OpticalFlowMethod.LUCAS_KANADE
