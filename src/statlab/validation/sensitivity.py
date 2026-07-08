"""A generic parameter-sensitivity grid runner.

Not specific to any strategy — takes a dict of parameter-name -> candidate values and a
callable that scores one combination, and returns a tidy DataFrame of every combination's
parameters and score. A strategy that only performs well for one razor-thin parameter
combination is more likely overfit than one that's robust across a neighborhood of
reasonable choices; this is the tool for checking which kind you have. Paired with
:mod:`statlab.validation.deflated_sharpe`, the N scores this produces are exactly the "N
trials" input the deflated Sharpe ratio needs to correctly discount the best cell for having
been selected as the max of N.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Any

import pandas as pd


def sensitivity_grid(
    grid: dict[str, list[Any]], run_fn: Callable[..., float], *, metric_name: str = "metric"
) -> pd.DataFrame:
    """Run ``run_fn(**combo)`` for every combination in the cartesian product of ``grid``.

    Returns a tidy DataFrame with one row per combination: the parameter columns plus a
    ``metric_name`` column holding ``run_fn``'s return value for that combination.
    """
    if not grid:
        raise ValueError("grid must not be empty")

    keys = list(grid.keys())
    rows: list[dict[str, Any]] = []
    for values in product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values, strict=True))
        score = run_fn(**combo)
        rows.append({**combo, metric_name: score})

    return pd.DataFrame(rows)
