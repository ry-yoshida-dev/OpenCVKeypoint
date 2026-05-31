from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Callable

from ..parameter import OpticalFlowParameters


@dataclass
class LucasKanadeParameters(OpticalFlowParameters):
    """
    LucasKanadeParameters is the parameters for the Lucas-Kanade optical flow processor.

    Attributes:
    ----------
    is_SparseRLOF: bool
        Whether to use SparseRLOF.
    """
    is_SparseRLOF: bool = True

    def get_flow_func(
        self,
    ) -> Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        match self.is_SparseRLOF:
            case True:
                raise RuntimeError("SparseRLOF produced an invalid matrix. Debug required or disable SparseRLOF.")
                return cv2.optflow.calcOpticalFlowSparseRLOF  # type: ignore[return-value]
            case False:
                return cv2.calcOpticalFlowPyrLK  # type: ignore[return-value]
