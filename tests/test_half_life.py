"""Tests for half-life estimation, validated against the known OU half-life."""

from __future__ import annotations

import numpy as np
import pytest

from statlab.data import OUParams, simulate_ou, simulate_random_walk
from statlab.signals import half_life


class TestHalfLife:
    def test_recovers_known_ou_half_life(self, rng: np.random.Generator) -> None:
        # theta chosen so the theoretical half-life ln(2)/theta is ~35 steps.
        theta = np.log(2.0) / 35.0
        params = OUParams(theta=theta, mu=0.0, sigma=1.0)
        spread = simulate_ou(20_000, params, rng)
        estimated = half_life(spread)
        assert estimated == pytest.approx(params.half_life, rel=0.15)

    def test_faster_reversion_gives_shorter_half_life(self, rng: np.random.Generator) -> None:
        fast = simulate_ou(20_000, OUParams(theta=0.2, mu=0.0, sigma=1.0), rng)
        slow = simulate_ou(20_000, OUParams(theta=0.02, mu=0.0, sigma=1.0), rng)
        assert half_life(fast) < half_life(slow)

    def test_random_walk_has_long_half_life(self, rng: np.random.Generator) -> None:
        # A random walk has no real mean reversion, so its estimated lambda is ~0 and the
        # half-life is very long or infinite. Any single path is noisy, so we summarise a
        # batch by its median (robust to the occasional spuriously-finite estimate).
        hls = [half_life(simulate_random_walk(4000, rng)) for _ in range(15)]
        capped = [h if np.isfinite(h) else 1e9 for h in hls]
        assert float(np.median(capped)) > 200.0

    def test_rejects_too_short(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            half_life(np.array([1.0, 2.0]))

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            half_life(np.ones((5, 2)))
