from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VALID_MODES = ("complete", "overview")


@dataclass(frozen=True)
class SynthEvalConfig:
    """Evaluation configuration. Immutable after creation."""

    title: str = "SynthEval Report"
    mode: Literal["complete", "overview"] = "complete"
    scoring: bool = True
    n_bins: int = 10
    max_sample_size: int = 5000
    max_bivariate_pairs: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            msg = f"mode must be one of {VALID_MODES}, got '{self.mode}'"
            raise ValueError(msg)
        if self.n_bins < 2:
            msg = f"n_bins must be >= 2, got {self.n_bins}"
            raise ValueError(msg)
        if self.max_sample_size < 1:
            msg = f"max_sample_size must be >= 1, got {self.max_sample_size}"
            raise ValueError(msg)
        if self.max_bivariate_pairs is not None and self.max_bivariate_pairs < 1:
            msg = f"max_bivariate_pairs must be >= 1 or None, got {self.max_bivariate_pairs}"
            raise ValueError(msg)
