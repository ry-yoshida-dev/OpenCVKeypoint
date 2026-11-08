from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from typing_extensions import override

KNNMatchGroup = Sequence[cv2.DMatch]


@dataclass
class MatchResult:
    """
    Container for keypoint matches (cv2.DMatch).

    Supports Lowe's ratio test and convenient access methods.

    Attributes
    ----------
    matches: Sequence[cv2.DMatch | KNNMatchGroup]
        Keypoint matches. Each item is a single DMatch or a group of kNN
        matches (for ratio test).
    """

    matches: Sequence[cv2.DMatch | KNNMatchGroup]

    def __getitem__(self, index: int) -> cv2.DMatch:
        """
        Return a single match (if it is a list of kNN matches, returns the best one).

        Parameters
        ----------
        index: int
            Index of the match to retrieve.

        Returns
        -------
        cv2.DMatch: The match at the specified index.
        """
        m = self.matches[index]
        if isinstance(m, cv2.DMatch):
            return m
        return m[0]

    def __iter__(self) -> Iterator[cv2.DMatch]:
        """
        Iterate over the matches.

        Returns
        -------
        Iterator[cv2.DMatch]: The iterator over the matches.
        """
        for m in self.matches:
            if isinstance(m, cv2.DMatch):
                yield m
            else:
                yield m[0]

    def apply_ratio_test(self, threshold: float = 0.75) -> MatchResult:
        """
        Apply Lowe's ratio test for kNN matches.

        Parameters
        ----------
        threshold: float
            Ratio threshold. Default is 0.75.

        Returns
        -------
        MatchResult: Filtered matches after ratio test.
        """
        knn_matches: list[KNNMatchGroup] = []
        for match in self.matches:
            if isinstance(match, cv2.DMatch):
                continue
            if len(match) < 2:
                warnings.warn(
                    "Ratio test skipped: Found kNN match with k<2.", stacklevel=2
                )
                return MatchResult(matches=self.matches)
            knn_matches.append(match)

        if not knn_matches:
            warnings.warn(
                "Ratio test skipped: No valid kNN pairs (k>=2) found.", stacklevel=2
            )
            return MatchResult(matches=self.matches)

        distances = np.array([(m[0].distance, m[1].distance) for m in knn_matches])
        ratios = distances[:, 0] / (distances[:, 1] + np.finfo(float).eps)
        indices = np.flatnonzero(ratios < threshold)
        good_matches: list[KNNMatchGroup] = [knn_matches[i] for i in indices]
        return MatchResult(matches=good_matches)

    def sort_by_distance(self) -> None:
        """
        Sort the matches by distance.
        """
        distances = [match.distance for match in self]
        order = sorted(range(len(distances)), key=distances.__getitem__)
        self.matches = [self.matches[i] for i in order]

    def filter_suboptimal_pair(self) -> MatchResult:
        """
        Filter out suboptimal pairs by keeping only mutual optimal matches.

        Returns
        -------
        MatchResult: Filtered matches after suboptimal pair filtering.
        """
        delete_indices: list[int | float] = []
        query_match_dict: dict[int, dict[str, int | float]] = {}
        gallery_match_dict: dict[int, dict[str, int | float]] = {}
        for i, match in enumerate(self):
            query_index = match.queryIdx
            train_index = match.trainIdx
            distance = match.distance

            if query_index not in query_match_dict:
                query_match_dict[query_index] = {"distance": distance, "index": i}
            else:
                past_distance = query_match_dict[query_index]["distance"]
                if distance < past_distance:
                    delete_indices.append(query_match_dict[query_index]["index"])
                    query_match_dict[query_index] = {"distance": distance, "index": i}
                else:
                    delete_indices.append(i)

            if train_index not in gallery_match_dict:
                gallery_match_dict[train_index] = {"distance": distance, "index": i}
            else:
                past_distance = gallery_match_dict[train_index]["distance"]
                if distance < past_distance:
                    delete_indices.append(gallery_match_dict[train_index]["index"])
                    gallery_match_dict[train_index] = {"distance": distance, "index": i}
                else:
                    delete_indices.append(i)
        filtered_matches = [
            match for j, match in enumerate(self.matches) if j not in delete_indices
        ]
        return MatchResult(matches=filtered_matches)

    @override
    def __str__(self) -> str:
        """
        Return the string representation of the MatchResult.

        Returns
        -------
        str: The string representation of the MatchResult.
        """
        return f"MatchResult(matches.pair={len(self.matches)})"

    def __len__(self) -> int:
        """
        Return the number of matches.

        Returns
        -------
        int: The number of matches.
        """
        return len(self.matches)
