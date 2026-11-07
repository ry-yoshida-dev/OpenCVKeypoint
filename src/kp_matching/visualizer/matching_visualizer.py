from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from ..match_container import KNNMatchGroup
from .flags import DrawMatchFlags


@dataclass
class MatchingVisualizer:
    """
    Visualize keypoint matches between two images.

    Parameters
    ----------
    flags: DrawMatchFlags
        The draw match flags.
    """

    flags: DrawMatchFlags = DrawMatchFlags.NOT_DRAW_SINGLE_POINTS

    def draw_matches(
        self,
        img1: np.ndarray,
        kps1: Sequence[cv2.KeyPoint],
        img2: np.ndarray,
        kps2: Sequence[cv2.KeyPoint],
        matches: Sequence[cv2.DMatch | KNNMatchGroup],
    ) -> np.ndarray:
        """
        Draw matches between keypoints in two images.

        This method draws lines connecting matched keypoints in two images.

        Parameters
        ----------
        img1: np.ndarray
            First image.
        kps1: Sequence[cv2.KeyPoint]
            Keypoints in the first image.
        img2: np.ndarray
            Second image.
        kps2: Sequence[cv2.KeyPoint]
            Keypoints in the second image.
        matches: Sequence[cv2.DMatch | KNNMatchGroup]
            Matches between keypoints. For kNN groups, the best match is drawn.

        Returns
        -------
        np.ndarray
            Image with matches drawn.
        """
        best_matches = [m if isinstance(m, cv2.DMatch) else m[0] for m in matches]
        # OpenCV allocates the output image when outImg is None, but its stubs
        # do not accept None there.
        drawn = cv2.drawMatches(  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            img1,
            kps1,
            img2,
            kps2,
            best_matches,
            None,  # pyright: ignore[reportArgumentType]
            flags=self.flags.to_cv2,
        )
        return cast(np.ndarray, drawn)
