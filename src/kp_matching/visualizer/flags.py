from enum import Enum

import cv2


class DrawMatchFlags(Enum):
    """
    Draw match flags.

    Attributes
    ----------
    DEFAULT: int
        The default draw match flags.
    DRAW_OVER_OUTIMG: int
        The draw over outimg draw match flags.
    NOT_DRAW_SINGLE_POINTS: int
        The not draw single points draw match flags.
    DRAW_RICH_KEYPOINTS: int
        The draw rich keypoints draw match flags.
    """

    DEFAULT = "default"
    DRAW_OVER_OUTIMG = "draw_over_outimg"
    NOT_DRAW_SINGLE_POINTS = "not_draw_single_points"
    DRAW_RICH_KEYPOINTS = "draw_rich_keypoints"

    @property
    def to_cv2(self) -> int:
        """
        Convert the draw match flags to cv2 draw match flags.

        Returns
        -------
        int
            The cv2 draw match flags.
        """
        match self:
            case self.DEFAULT:
                return cv2.DrawMatchesFlags_DEFAULT
            case self.DRAW_OVER_OUTIMG:
                return cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
            case self.NOT_DRAW_SINGLE_POINTS:
                return cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            case self.DRAW_RICH_KEYPOINTS:
                return cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
