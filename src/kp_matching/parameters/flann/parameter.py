from dataclasses import dataclass, field

import cv2
from cv2.typing import IndexParams, SearchParams

from .index_type import FLANNIndexType


@dataclass
class FLANNParameters:
    """
    FLANN parameters.

    FLANN is a library for performing fast approximate nearest neighbor searches in high dimensional spaces.

    Attributes
    ----------
    algorithm: FLANNIndexType
        The algorithm to use for the FLANN matcher.
    table_number: int
        The number of tables to use for the FLANN matcher.
    key_size: int
        The key size to use for the FLANN matcher.
    multi_probe_level: int
        The multi probe level to use for the FLANN matcher.
    checks: int
        The checks to use for the FLANN matcher.
    is_binary: bool
        Whether to use binary descriptors for the FLANN matcher.
    trees: int
        The number of trees to use for the FLANN matcher.
    """

    algorithm: FLANNIndexType = FLANNIndexType.KDTREE
    table_number: int = 6
    key_size: int = 12
    multi_probe_level: int = 1
    checks: int = 50
    is_binary: bool = False
    trees: int = 5
    metric: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Post-initialization validation.
        """
        self.metric = cv2.NORM_L2 if not self.is_binary else cv2.NORM_HAMMING
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """
        Validate the parameters.

        Raises
        ------
        ValueError:
            If the parameters are not valid.
        """
        if self.table_number <= 0:
            raise ValueError("table_number must be a positive integer")
        if self.key_size <= 0:
            raise ValueError("key_size must be a positive integer")
        if self.multi_probe_level <= 0:
            raise ValueError("multi_probe_level must be a positive integer")
        if self.checks <= 0:
            raise ValueError("checks must be a positive integer")
        if self.trees <= 0:
            raise ValueError("trees must be a positive integer")

    def define_matcher(self) -> cv2.FlannBasedMatcher:
        """
        Define the FLANN matcher.

        Returns
        -------
        cv2.FlannBasedMatcher
            The FLANN matcher.
        """
        flann_index_params = self._define_flann_index_params()
        flann_search_params = self._define_flann_search_params()
        return cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    def _define_flann_index_params(self) -> IndexParams:
        """
        Define the FLANN index parameters.

        Returns
        -------
        IndexParams
            The FLANN index parameters.
        """
        flann_index_params: IndexParams = {}
        flann_index_params["algorithm"] = self.algorithm.value
        if self.is_binary:
            flann_index_params["table_number"] = self.table_number
            flann_index_params["key_size"] = self.key_size
            flann_index_params["multi_probe_level"] = self.multi_probe_level
        else:
            flann_index_params["trees"] = self.trees
        return flann_index_params

    def _define_flann_search_params(self) -> SearchParams:
        """
        Define the FLANN search parameters.

        Returns
        -------
        SearchParams
            The FLANN search parameters.
        """
        flann_search_params: SearchParams = {}
        flann_search_params["checks"] = self.checks
        return flann_search_params
