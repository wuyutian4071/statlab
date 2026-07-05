"""Shared pytest fixtures.

Reproducibility contract: every test that needs randomness pulls a *seeded* generator from
the ``rng`` fixture, so the suite is deterministic across machines and runs.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic NumPy generator seeded for reproducible tests."""
    return np.random.default_rng(20240101)
