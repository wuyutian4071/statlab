"""Shared pytest fixtures.

Reproducibility contract: every test that needs randomness pulls a *seeded* generator from
the ``rng`` fixture, so the suite is deterministic across machines and runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statlab.data import PointInTimeUniverse, SyntheticSource


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic NumPy generator seeded for reproducible tests."""
    return np.random.default_rng(20240101)


@pytest.fixture
def synthetic_source() -> SyntheticSource:
    """A small offline source with a known cointegration ground truth."""
    return SyntheticSource(n=400, n_pairs=2, n_noise=2, seed=42)


@pytest.fixture
def synthetic_bars(synthetic_source: SyntheticSource) -> pd.DataFrame:
    """Canonical long-form bars from the synthetic source."""
    return synthetic_source.fetch()


@pytest.fixture
def universe(synthetic_bars: pd.DataFrame) -> PointInTimeUniverse:
    """A point-in-time universe built from the synthetic bars."""
    return PointInTimeUniverse.from_bars(synthetic_bars)
