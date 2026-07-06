"""Tests for the z-score computation and the entry/exit/stop signal state machine."""

from __future__ import annotations

import numpy as np
import pytest

from statlab.signals import SignalParams, generate_positions, rolling_zscore


class TestRollingZScore:
    def test_is_causal_warmup_is_nan(self) -> None:
        z = rolling_zscore(np.arange(10.0), window=4)
        assert z.iloc[:3].isna().all()
        assert z.iloc[3:].notna().all()

    def test_matches_manual_computation(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 10.0])
        z = rolling_zscore(data, window=4)
        w = data
        expected = (w[-1] - w.mean()) / w.std(ddof=0)
        assert z.iloc[-1] == pytest.approx(expected)

    def test_zero_variance_window_is_nan_not_inf(self) -> None:
        z = rolling_zscore(np.ones(5), window=3)
        assert z.iloc[2:].isna().all()

    def test_rejects_small_window(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            rolling_zscore(np.arange(5.0), window=1)


class TestSignalParams:
    def test_valid(self) -> None:
        SignalParams(entry=2.0, exit=0.5, stop=4.0)

    @pytest.mark.parametrize(
        ("entry", "exit_", "stop"),
        [(2.0, 2.0, 4.0), (2.0, 0.5, 1.0), (0.5, 1.0, 4.0)],
    )
    def test_invalid_ordering_rejected(self, entry: float, exit_: float, stop: float) -> None:
        with pytest.raises(ValueError, match="exit < entry < stop"):
            SignalParams(entry=entry, exit=exit_, stop=stop)


class TestGeneratePositions:
    def test_enters_long_when_spread_cheap(self) -> None:
        z = np.array([0.0, -2.5, -1.0, -0.1])
        pos = generate_positions(z, SignalParams(entry=2.0, exit=0.5, stop=4.0))
        assert pos[0] == 0
        assert pos[1] == 1  # z <= -entry -> long spread
        assert pos[2] == 1  # still beyond exit band -> hold
        assert pos[3] == 0  # reverted into exit band -> close

    def test_enters_short_when_spread_rich(self) -> None:
        z = np.array([0.0, 2.5, 1.0, 0.1])
        pos = generate_positions(z, SignalParams(entry=2.0, exit=0.5, stop=4.0))
        assert pos[1] == -1
        assert pos[2] == -1
        assert pos[3] == 0

    def test_stop_loss_closes_on_further_divergence(self) -> None:
        z = np.array([0.0, -2.5, -4.5])  # opens long, then blows through the stop
        pos = generate_positions(z, SignalParams(entry=2.0, exit=0.5, stop=4.0))
        assert pos[1] == 1
        assert pos[2] == 0  # |z| >= stop -> stopped out

    def test_holds_through_nan_warmup(self) -> None:
        z = np.array([np.nan, np.nan, -2.5, np.nan, -0.1])
        pos = generate_positions(z, SignalParams(entry=2.0, exit=0.5, stop=4.0))
        assert pos[0] == 0
        assert pos[2] == 1
        assert pos[3] == 1  # NaN -> hold prior state
        assert pos[4] == 0

    def test_no_position_when_never_crosses_entry(self) -> None:
        z = np.array([0.1, -0.5, 1.0, -1.9, 1.9])
        pos = generate_positions(z, SignalParams(entry=2.0, exit=0.5, stop=4.0))
        assert (pos == 0).all()

    def test_positions_are_in_valid_set(self) -> None:
        rng = np.random.default_rng(0)
        z = rng.normal(0, 2.5, size=500)
        pos = generate_positions(z)
        assert set(np.unique(pos)).issubset({-1, 0, 1})
