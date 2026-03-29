from __future__ import annotations

from typing import Any

import polars as pl
from dataxid_profiling import ProfileReport

from dataxid_syntheval._config import SynthEvalConfig  # noqa: TC001 — used at runtime


def ingest(source: Any, config: SynthEvalConfig) -> ProfileReport:
    """Accept a DataFrame or ProfileReport and return a ProfileReport.

    - ProfileReport → passthrough (no re-profiling)
    - pl.DataFrame → profile with dataxid-profiling
    """
    if isinstance(source, ProfileReport):
        return source

    if isinstance(source, pl.DataFrame):
        return ProfileReport(source, mode=config.mode)

    type_name = type(source).__qualname__
    msg = (
        f"Unsupported input type: {type_name}. "
        "Expected pl.DataFrame or ProfileReport."
    )
    raise TypeError(msg)
