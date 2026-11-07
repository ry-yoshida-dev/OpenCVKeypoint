from enum import Enum


class KPMatchMethod(Enum):
    """
    KPMatchMethod is the method of the keypoint matching processor.

    Attributes
    ----------
    BF: Brute-force matching.
    KNN: k-Nearest Neighbors matching.
    FLANN: FLANN matching.
    """

    BF = "BF"
    KNN = "kNN"
    FLANN = "FLANN"
