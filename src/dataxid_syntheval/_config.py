from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VALID_MODES = ("complete", "overview")


@dataclass(frozen=True)
class SynthEvalConfig:
    """Evaluation configuration. Immutable after creation."""

    title: str = "SynthEval Report"
    mode: Literal["complete", "overview"] = "complete"

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            msg = f"mode must be one of {VALID_MODES}, got '{self.mode}'"
            raise ValueError(msg)
