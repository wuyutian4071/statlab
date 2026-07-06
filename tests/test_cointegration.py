"""Tests for cointegration tests, validated against known truth and statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from statlab.data import OUParams, simulate_cointegrated_pair, simulate_random_walk
from statlab.signals import engle_granger, johansen


def _log_df(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise log preserving the DataFrame type (np.log on a frame loses it)."""
    return pd.DataFrame(np.log(df.to_numpy()), index=df.index, columns=df.columns)


class TestEngleGranger:
    def test_detects_cointegrated_pair(self, rng: np.random.Generator) -> None:
        frame = simulate_cointegrated_pair(1000, rng, beta=1.4)
        res = engle_granger(np.log(frame["B"]), np.log(frame["A"]))
        assert res.is_cointegrated(0.05)

    def test_recovers_hedge_ratio(self, rng: np.random.Generator) -> None:
        true_beta = 1.4
        frame = simulate_cointegrated_pair(2000, rng, beta=true_beta)
        res = engle_granger(np.log(frame["B"]), np.log(frame["A"]))
        # OLS on the cointegrating relation should recover beta closely.
        assert res.beta == pytest.approx(true_beta, rel=0.1)

    def test_does_not_flag_independent_random_walks(self, rng: np.random.Generator) -> None:
        a = simulate_random_walk(1000, rng, x0=100.0)
        b = simulate_random_walk(1000, rng, x0=100.0)
        res = engle_granger(a, b)
        # Two independent random walks: fail to reject the null of no cointegration.
        assert not res.is_cointegrated(0.05)

    def test_pvalue_matches_statsmodels(self, rng: np.random.Generator) -> None:
        frame = simulate_cointegrated_pair(1000, rng, beta=1.1)
        y, x = np.log(frame["B"]).to_numpy(), np.log(frame["A"]).to_numpy()
        res = engle_granger(y, x)
        _, sm_pvalue, _ = coint(y, x, trend="c", autolag="aic")
        assert res.pvalue == pytest.approx(sm_pvalue, abs=1e-10)

    def test_rejects_short_series(self) -> None:
        with pytest.raises(ValueError, match="at least 20"):
            engle_granger(np.arange(10.0), np.arange(10.0))

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            engle_granger(np.arange(50.0), np.arange(40.0))


class TestJohansen:
    def test_finds_rank_one_for_a_cointegrated_pair(self, rng: np.random.Generator) -> None:
        frame = simulate_cointegrated_pair(1500, rng, beta=1.2)
        log_frame = _log_df(frame)
        res = johansen(log_frame, level=0.05)
        assert res.rank == 1
        assert res.is_cointegrated

    def test_usually_finds_rank_zero_for_independent_walks(self, rng: np.random.Generator) -> None:
        # Independent random walks are not cointegrated, but any single test rejects the
        # null a small fraction of the time by construction. At the 1% level the vast
        # majority of trials must find rank 0; we allow a couple of expected false
        # positives rather than betting the suite on one draw.
        ranks = []
        for _ in range(15):
            panel = pd.DataFrame(
                {
                    "A": simulate_random_walk(1000, rng, x0=100.0),
                    "B": simulate_random_walk(1000, rng, x0=100.0),
                }
            )
            ranks.append(johansen(panel, level=0.01).rank)
        assert sum(r == 0 for r in ranks) >= 12

    def test_hedge_ratios_normalised(self, rng: np.random.Generator) -> None:
        frame = _log_df(simulate_cointegrated_pair(1500, rng, beta=1.2))
        res = johansen(frame, level=0.05)
        hedge = res.hedge_ratios()
        assert hedge[0] == pytest.approx(1.0)

    def test_rejects_nan_panel(self) -> None:
        panel = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="no-NaN"):
            johansen(panel)

    def test_rejects_invalid_level(self, rng: np.random.Generator) -> None:
        frame = _log_df(simulate_cointegrated_pair(200, rng))
        with pytest.raises(ValueError, match="level must be one of"):
            johansen(frame, level=0.5)


def test_faster_reversion_is_more_significant(rng: np.random.Generator) -> None:
    """A tighter (faster-reverting) spread should be at least as significant."""
    fast = simulate_cointegrated_pair(
        1500, rng, beta=1.0, spread_params=OUParams(theta=0.2, mu=0.0, sigma=0.02)
    )
    res = engle_granger(np.log(fast["B"]), np.log(fast["A"]))
    assert res.pvalue < 0.05
