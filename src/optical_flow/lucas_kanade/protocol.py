from __future__ import annotations

from typing import Protocol

import numpy as np


class LucasKanadeFlowFunction(Protocol):
    """Callable that tracks source keypoints from one image to the next."""

    def __call__(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track source keypoints between two preprocessed images.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Target keypoints ``(N, 2)``, status ``(N, 1)``, and error ``(N, 1)``.
        """
        ...
