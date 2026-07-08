"""Validation: walk-forward out-of-sample testing, sensitivity grids, and the deflated Sharpe
ratio — the tools that correct for the multiple-comparisons bias `signals/discovery.py` flags.
"""

from __future__ import annotations

from statlab.validation.deflated_sharpe import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from statlab.validation.sensitivity import sensitivity_grid
from statlab.validation.walkforward import (
    WalkForwardResult,
    WalkForwardWindow,
    combined_oos_sharpe,
    run_walk_forward,
    walk_forward_windows,
)

__all__ = [
    "DeflatedSharpeResult",
    "WalkForwardResult",
    "WalkForwardWindow",
    "combined_oos_sharpe",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "run_walk_forward",
    "sensitivity_grid",
    "walk_forward_windows",
]
