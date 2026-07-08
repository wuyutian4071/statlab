r"""Walk-forward validation: discover a pair on a train window, trade it out-of-sample on
the following test window, repeat rolling forward.

This addresses the multiple-comparisons caveat already flagged in ``signals/discovery.py``:
pair discovery is in-sample selection, so a pair's in-sample cointegration p-value overstates
how tradable it actually is. This module never lets discovery see the data it is about to be
graded on — the pair chosen from window *i*'s train slice is backtested strictly over window
*i*'s test dates, which discovery never touched.

Note this is a different (and complementary) causality guarantee from ``PointInTimeUniverse``'s
own: the universe already guarantees no query at date *t* ever sees data dated after *t*. What
walk-forward adds on top is that the *selection* of which pair to trade is itself blind to the
period it will be graded on, not just blind to the literal future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from statlab.backtest import BacktestEngine, BacktestResult, PairsStrategy, Portfolio, sharpe_ratio
from statlab.data.universe import PointInTimeUniverse
from statlab.signals import PairCandidate, discover_pairs


@dataclass(frozen=True)
class WalkForwardWindow:
    """One rolling train/test split. Train immediately precedes test; both are non-overlapping."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_windows(
    trading_days: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[WalkForwardWindow]:
    """Generate rolling train/test windows over ``trading_days``.

    Test windows are non-overlapping by default (``step_days`` defaults to ``test_days``); the
    train window immediately precedes each test window and rolls forward with it (not an
    expanding window). Returns as many windows as fit entirely within ``trading_days``.
    """
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days and test_days must be positive")
    step = step_days if step_days is not None else test_days
    if step <= 0:
        raise ValueError("step_days must be positive")

    windows: list[WalkForwardWindow] = []
    n = len(trading_days)
    train_start_idx = 0
    while True:
        train_end_idx = train_start_idx + train_days - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_days - 1
        if test_end_idx >= n:
            break
        windows.append(
            WalkForwardWindow(
                train_start=trading_days[train_start_idx],
                train_end=trading_days[train_end_idx],
                test_start=trading_days[test_start_idx],
                test_end=trading_days[test_end_idx],
            )
        )
        train_start_idx += step
    return windows


@dataclass(frozen=True)
class WalkForwardResult:
    """One window's outcome: the pair discovered on the train slice (``None`` if none cleared
    the thresholds) and its out-of-sample :class:`~statlab.backtest.BacktestResult`."""

    window: WalkForwardWindow
    pair: PairCandidate | None
    result: BacktestResult | None


def run_walk_forward(
    universe: PointInTimeUniverse,
    windows: list[WalkForwardWindow],
    *,
    cash: float = 1_000_000.0,
    notional: float = 200_000.0,
    min_correlation: float = 0.3,
    max_pvalue: float = 0.1,
    max_half_life: float = 252.0,
    strategy_kwargs: dict[str, Any] | None = None,
) -> list[WalkForwardResult]:
    """Run walk-forward discovery+backtest.

    For each window: discover pairs using only the train slice, then — if a pair clears the
    thresholds — backtest the top-ranked candidate with :class:`PairsStrategy` strictly over
    the test dates. The strategy is still free to read pre-test-window price history when
    fitting its Kalman filter at test time (that's ordinary causal information, no different
    from a live system that doesn't amnesia-wipe its history at an arbitrary backtest
    boundary) — what's blind to the test window is only the *choice of which pair to trade*.
    """
    strategy_kwargs = strategy_kwargs or {}
    results: list[WalkForwardResult] = []

    for window in windows:
        train_prices = universe.as_of(window.train_end)
        train_prices = train_prices.loc[train_prices.index >= window.train_start]
        candidates = discover_pairs(
            train_prices,
            min_correlation=min_correlation,
            max_pvalue=max_pvalue,
            max_half_life=max_half_life,
        )
        if not candidates:
            results.append(WalkForwardResult(window=window, pair=None, result=None))
            continue

        top = candidates[0]
        strategy = PairsStrategy(top.y, top.x, notional, **strategy_kwargs)
        engine = BacktestEngine(universe, strategy, Portfolio(cash))
        result = engine.run(window.test_start, window.test_end)
        results.append(WalkForwardResult(window=window, pair=top, result=result))

    return results


def combined_oos_sharpe(results: list[WalkForwardResult]) -> float:
    """Concatenate every window's out-of-sample return series into one out-of-sample track
    record and report its Sharpe — the single most honest number this milestone produces.
    """
    pieces = [
        r.result.returns() for r in results if r.result is not None and len(r.result.returns()) > 0
    ]
    if not pieces:
        return 0.0
    combined = pd.concat(pieces)
    return sharpe_ratio(combined)
