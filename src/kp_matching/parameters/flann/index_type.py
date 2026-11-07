from enum import IntEnum


class FLANNIndexType(IntEnum):
    """
    FLANN index type.

    Attributes
    ----------
    LINEAR: int
        The linear index type.
    KDTREE: int
        The KD-tree index type.
    KMEANS: int
        The K-means index type.
    COMPOSITE: int
        The composite index type.
    KDTREE_SINGLE: int
        The single KD-tree index type.
    HIERARCHICAL: int
        The hierarchical index type.
    LSH: int
        The LSH index type.
    SAVED: int
        The saved index type.
    AUTOTUNED: int
        The autotuned index type.
    """

    LINEAR = 0
    KDTREE = 1
    KMEANS = 2
    COMPOSITE = 3
    KDTREE_SINGLE = 4
    HIERARCHICAL = 5
    LSH = 6  # For binary descriptors
    SAVED = 7
    AUTOTUNED = 8
