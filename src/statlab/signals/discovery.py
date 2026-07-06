r"""Pair discovery: from a price panel to a ranked list of tradable pairs.

The funnel mirrors how a researcher actually narrows a large universe down to a handful of
candidates, cheap tests first:

1. **Correlation pre-filter** — a fast screen on log-return correlation. Cointegration
   testing every pair in a big universe is :math:`O(N^2)` ADF regressions; the pre-filter
   discards obviously-unrelated pairs before paying that cost.
2. **Engle-Granger cointegration** — keep pairs whose spread is stationary at ``max_pvalue``.
3. **Half-life filter** — keep pairs whose mean reversion is fast enough to trade but not
   so fast it is just microstructure noise.

Candidates are ranked by cointegration p-value (most significant first). Cointegration is
run on **log prices** (positive prices, multiplicative relationships), matching how the
synthetic ground-truth pairs are constructed.

Caveat surfaced deliberately: testing many pairs invites multiple-comparisons bias — some
pairs will look cointegrated by chance. Discovery is in-sample selection; M6 addresses this
with walk-forward validation and a deflated Sharpe ratio. Do not read a low p-value here as
out-of-sample tradability.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from statlab.signals.cointegration import engle_granger
from statlab.signals.half_life import half_life


@dataclass(frozen=True)
class PairCandidate:
    """A discovered candidate pair and its in-sample statistics."""

    y: str
    x: str
    beta: float
    correlation: float
    pvalue: float
    half_life: float

    def __str__(self) -> str:
        return (
            f"{self.y}~{self.x} beta={self.beta:.3f} corr={self.correlation:.2f} "
            f"p={self.pvalue:.4f} hl={self.half_life:.1f}"
        )


def discover_pairs(
    prices: pd.DataFrame,
    *,
    min_correlation: float = 0.7,
    max_pvalue: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 252.0,
) -> list[PairCandidate]:
    """Discover cointegrated, tradable pairs from a wide price panel.

    Parameters
    ----------
    prices:
        Wide price *level* panel (index=date, columns=tickers). Columns with missing
        values are intersected pairwise on their overlapping, complete dates.
    min_correlation:
        Minimum absolute log-return correlation to survive the pre-filter.
    max_pvalue:
        Maximum Engle-Granger p-value to be considered cointegrated.
    min_half_life, max_half_life:
        Acceptable mean-reversion half-life band (in observations/days).

    Returns
    -------
    list[PairCandidate]
        Ranked by ascending p-value (strongest cointegration first).
    """
    log_prices = pd.DataFrame(np.log(prices.to_numpy()), index=prices.index, columns=prices.columns)
    log_returns = log_prices.diff()

    candidates: list[PairCandidate] = []
    for a, b in combinations(prices.columns, 2):
        pair = log_prices[[a, b]].dropna()
        if len(pair) < 60:
            continue

        rets = log_returns[[a, b]].dropna()
        corr = float(rets[a].corr(rets[b]))
        if not np.isfinite(corr) or abs(corr) < min_correlation:
            continue

        # Fix the regression direction as (b ~ a); a fuller scheme would test both.
        result = engle_granger(pair[b].to_numpy(), pair[a].to_numpy())
        if result.pvalue > max_pvalue:
            continue

        hl = half_life(result.resid)
        if not (min_half_life <= hl <= max_half_life):
            continue

        candidates.append(
            PairCandidate(
                y=str(b),
                x=str(a),
                beta=result.beta,
                correlation=corr,
                pvalue=result.pvalue,
                half_life=hl,
            )
        )

    candidates.sort(key=lambda c: c.pvalue)
    return candidates
