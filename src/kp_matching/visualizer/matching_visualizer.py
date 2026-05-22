from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ..match_container import KNNMatchGroup
from .flags import DrawMatchFlags


@dataclass
class MatchingVisualizer:
    """
    Visualize keypoint matches between two images.

    Parameters:
    ----------
    flags: DrawMatchFlags
        The draw match flags.
    """

    flags: DrawMatchFlags = DrawMatchFlags.NOT_DRAW_SINGLE_POINTS

    def draw_matches(
        self,
        img1: np.ndarray, 
        kps1: tuple[cv2.KeyPoint, ...] | Sequence[cv2.KeyPoint] | np.ndarray,
        img2: np.ndarray,
        kps2: tuple[cv2.KeyPoint, ...] | Sequence[cv2.KeyPoint] | np.ndarray, 
        matches: list[cv2.DMatch] | list[KNNMatchGroup]
        ) -> np.ndarray:
        """
        Draws matches between keypoints in two images.

        This method draws lines connecting matched keypoints in two images.

        Parameters:
        ----------
        img1: np.ndarray
            First image.
        kps1: tuple[cv2.KeyPoint, ...] | Sequence[cv2.KeyPoint] | np.ndarray
            Keypoints in the first image.
        img2: np.ndarray
            Second image.
        kps2: tuple[cv2.KeyPoint, ...] | Sequence[cv2.KeyPoint] | np.ndarray
            Keypoints in the second image.
            matches (list[cv2.DMatch]): List of matches between keypoints.

        Returns:
            ndarray: Image with matches drawn.
        """
        m = [m for match in matches if (m := match[0] if isinstance(match, (tuple, list)) else match)]
        img: np.ndarray = cv2.drawMatches(img1, kps1, img2, kps2, m, None, flags=self.flags.to_cv2) # type: ignore
        return img # type: ignore