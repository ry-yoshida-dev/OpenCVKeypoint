from dataclasses import dataclass


@dataclass
class RatioTestParameters:
    """
    Parameters for ratio test.

    Attributes
    ----------
    is_enabled: bool
        Whether to use ratio test.
    threshold: float
        The threshold for ratio test.
        - Closer to 0.0: Stricter (fewer, higher quality matches).
        - Closer to 1.0: Looser (more matches, including potential false positives).
    """

    is_enabled: bool = True
    threshold: float = 0.75

    def __post_init__(self) -> None:
        """
        Post-initialization validation.

        Raises
        ------
        ValueError:
            If the threshold is not a float strictly between 0 and 1.
        """
        if self.threshold <= 0 or self.threshold >= 1:
            raise ValueError("threshold must be a float strictly between 0 and 1")
