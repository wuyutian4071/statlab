"""Bar data sources.

A :class:`BarSource` is anything that can produce canonical long-form OHLCV bars. Two
implementations ship:

* :class:`SyntheticSource` — wraps the offline OU/cointegration generators and expands
  them into realistic OHLCV bars. The entire test suite and the demo use this, so nothing
  ever *requires* a network connection.
* :class:`YFinanceSource` — downloads real daily bars from Yahoo Finance. Network-bound
  and therefore only exercised by tests marked ``network``.

Both return the same schema, so downstream code (validation, storage, universe) is
source-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from statlab.data.schema import (
    ADJ_CLOSE,
    BAR_COLUMNS,
    CLOSE,
    DATE,
    HIGH,
    LOW,
    OPEN,
    TICKER,
    VOLUME,
)
from statlab.data.synthetic import simulate_correlated_ou_panel


@runtime_checkable
class BarSource(Protocol):
    """Anything that yields canonical long-form OHLCV bars."""

    def fetch(self) -> pd.DataFrame:
        """Return bars conforming to :data:`statlab.data.schema.BAR_COLUMNS`."""
        ...


def _panel_to_bars(panel: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Expand a wide close-price panel into long-form OHLCV bars.

    Intraday structure is synthesised around each close so the bars satisfy the usual
    consistency relations (``low <= open, close <= high``) and validation has something
    to chew on. Adjusted close equals close here (the synthetic data has no corporate
    actions); a split can be injected separately for split-detection tests.
    """
    frames: list[pd.DataFrame] = []
    for ticker in panel.columns:
        close = panel[ticker].to_numpy()
        n = len(close)
        # Open gaps slightly from the previous close; first open equals its own close.
        gap = rng.normal(0.0, 0.002, size=n)
        open_ = np.empty(n)
        open_[0] = close[0]
        open_[1:] = close[:-1] * (1.0 + gap[1:])
        # A symmetric intraday range wide enough to always bracket open and close.
        spread = np.abs(rng.normal(0.0, 0.004, size=n)) * close
        hi = np.maximum(open_, close) + spread
        lo = np.minimum(open_, close) - spread
        vol = rng.integers(1_000_000, 10_000_000, size=n).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    DATE: panel.index,
                    TICKER: ticker,
                    OPEN: open_,
                    HIGH: hi,
                    LOW: lo,
                    CLOSE: close,
                    ADJ_CLOSE: close,
                    VOLUME: vol,
                }
            )
        )
    bars = pd.concat(frames, ignore_index=True)
    return bars[list(BAR_COLUMNS)]


class SyntheticSource:
    """Offline source producing bars with a *known* cointegration ground truth.

    Wraps :func:`~statlab.data.synthetic.simulate_correlated_ou_panel`. The
    :attr:`truth` attribute records which ticker pairs are genuinely cointegrated, so
    tests can assert both discovery power and specificity.
    """

    def __init__(
        self,
        n: int = 1000,
        *,
        n_pairs: int = 3,
        n_noise: int = 4,
        seed: int = 7,
    ) -> None:
        self.n = n
        self.n_pairs = n_pairs
        self.n_noise = n_noise
        self.seed = seed
        rng = np.random.default_rng(seed)
        panel, self.truth = simulate_correlated_ou_panel(n, rng, n_pairs=n_pairs, n_noise=n_noise)
        self._panel = panel
        # A separate RNG stream for the OHLCV expansion keeps price paths reproducible
        # regardless of how many bars we decorate.
        self._bars = _panel_to_bars(panel, np.random.default_rng(seed + 1))

    def fetch(self) -> pd.DataFrame:
        return self._bars.copy()

    @property
    def tickers(self) -> list[str]:
        return list(self._panel.columns)


class YFinanceSource:
    """Real daily bars from Yahoo Finance (network-bound).

    Deliberately thin: it standardises yfinance's frame into the canonical schema and
    nothing more. Kept import-light so importing the module never pulls in yfinance until
    :meth:`fetch` is actually called.
    """

    def __init__(self, tickers: list[str], start: str, end: str | None = None) -> None:
        if not tickers:
            raise ValueError("tickers must be non-empty")
        self.tickers = tickers
        self.start = start
        self.end = end

    def fetch(self) -> pd.DataFrame:  # pragma: no cover - network path
        import yfinance as yf

        raw = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError("yfinance returned no data for the requested tickers")

        frames: list[pd.DataFrame] = []
        for ticker in self.tickers:
            sub = raw[ticker] if len(self.tickers) > 1 else raw
            frame = pd.DataFrame(
                {
                    DATE: pd.to_datetime(sub.index),
                    TICKER: ticker,
                    OPEN: sub["Open"].to_numpy(),
                    HIGH: sub["High"].to_numpy(),
                    LOW: sub["Low"].to_numpy(),
                    CLOSE: sub["Close"].to_numpy(),
                    ADJ_CLOSE: sub["Adj Close"].to_numpy(),
                    VOLUME: sub["Volume"].to_numpy().astype(float),
                }
            ).dropna(subset=[CLOSE])
            frames.append(frame)

        bars = pd.concat(frames, ignore_index=True)
        return bars[list(BAR_COLUMNS)]
