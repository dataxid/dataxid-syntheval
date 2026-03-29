from __future__ import annotations

import random

import polars as pl
import pytest
from dataxid_profiling import ProfileReport


def _make_original_df() -> pl.DataFrame:
    """100-row DataFrame simulating real-world data."""
    rng = random.Random(42)
    n = 100

    ages: list[int | None] = [rng.randint(18, 65) for _ in range(n)]
    incomes: list[float | None] = [round(rng.gauss(55_000, 15_000), 2) for _ in range(n)]
    cities: list[str | None] = [
        rng.choice(["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"]) for _ in range(n)
    ]
    genders = [rng.choice(["M", "F"]) for _ in range(n)]
    is_active = [rng.choice([True, False]) for _ in range(n)]

    ages[3] = None
    ages[17] = None
    incomes[7] = None
    cities[12] = None

    return pl.DataFrame(
        {
            "age": pl.Series(ages, dtype=pl.Int64),
            "income": pl.Series(incomes, dtype=pl.Float64),
            "city": cities,
            "gender": genders,
            "is_active": is_active,
        }
    )


def _make_synthetic_df() -> pl.DataFrame:
    """100-row DataFrame simulating synthetic data — slightly shifted distributions."""
    rng = random.Random(99)
    n = 100

    ages: list[int | None] = [rng.randint(20, 60) for _ in range(n)]
    incomes: list[float | None] = [round(rng.gauss(58_000, 12_000), 2) for _ in range(n)]
    cities: list[str | None] = [
        rng.choice(["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"]) for _ in range(n)
    ]
    genders = [rng.choice(["M", "F"]) for _ in range(n)]
    is_active = [rng.choice([True, False]) for _ in range(n)]

    ages[5] = None
    incomes[22] = None
    cities[41] = None

    return pl.DataFrame(
        {
            "age": pl.Series(ages, dtype=pl.Int64),
            "income": pl.Series(incomes, dtype=pl.Float64),
            "city": cities,
            "gender": genders,
            "is_active": is_active,
        }
    )


@pytest.fixture
def original_df() -> pl.DataFrame:
    return _make_original_df()


@pytest.fixture
def synthetic_df() -> pl.DataFrame:
    return _make_synthetic_df()


@pytest.fixture(scope="session")
def original_report() -> ProfileReport:
    return ProfileReport(_make_original_df(), title="Original")


@pytest.fixture(scope="session")
def synthetic_report() -> ProfileReport:
    return ProfileReport(_make_synthetic_df(), title="Synthetic")
