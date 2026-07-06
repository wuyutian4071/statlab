"""Canonical schema for OHLCV bar data and helpers to reshape it.

The whole data layer speaks one long-form ("tidy") bar format: one row per
``(date, ticker)`` with open/high/low/close/adj_close/volume columns. Keeping a single
canonical schema means ingestion sources, validation, storage, and the point-in-time
universe never disagree about column names or dtypes.
"""

from __future__ import annotations

import pandas as pd

# Column names — referenced everywhere instead of string literals.
DATE = "date"
TICKER = "ticker"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
ADJ_CLOSE = "adj_close"
VOLUME = "volume"

#: Price/volume value columns, in canonical order.
VALUE_COLUMNS: tuple[str, ...] = (OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE, VOLUME)
#: Full long-form bar column set, in canonical order.
BAR_COLUMNS: tuple[str, ...] = (DATE, TICKER, *VALUE_COLUMNS)
#: Columns that must hold strictly positive prices.
PRICE_COLUMNS: tuple[str, ...] = (OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE)


class SchemaError(ValueError):
    """Raised when a bar DataFrame does not conform to the canonical schema."""


def validate_schema(bars: pd.DataFrame) -> None:
    """Assert ``bars`` conforms to the canonical long-form bar schema.

    Checks presence of all required columns and that ``date`` is datetime-typed and the
    value columns are numeric. This is a *structural* check only; semantic validation
    (positive prices, no gaps, split sanity) lives in :mod:`statlab.data.validation`.

    Raises
    ------
    SchemaError
        If a column is missing or has the wrong dtype.
    """
    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(bars[DATE]):
        raise SchemaError(f"column '{DATE}' must be datetime-typed, got {bars[DATE].dtype}")

    for col in VALUE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(bars[col]):
            raise SchemaError(f"column '{col}' must be numeric, got {bars[col].dtype}")


def to_price_panel(bars: pd.DataFrame, field: str = ADJ_CLOSE) -> pd.DataFrame:
    """Pivot long-form bars into a wide price panel (index=date, columns=ticker).

    Parameters
    ----------
    bars:
        Long-form bars conforming to the canonical schema.
    field:
        Which value column to place in the panel (default: adjusted close, the right
        series for return/signal research).

    Returns
    -------
    pandas.DataFrame
        Wide panel indexed by an ascending :class:`~pandas.DatetimeIndex`, columns sorted
        by ticker. Missing ``(date, ticker)`` combinations are ``NaN``.
    """
    if field not in VALUE_COLUMNS:
        raise SchemaError(f"field must be one of {VALUE_COLUMNS}, got {field!r}")
    validate_schema(bars)
    panel = bars.pivot_table(index=DATE, columns=TICKER, values=field, aggfunc="last")
    panel = panel.sort_index().sort_index(axis=1)
    panel.index.name = DATE
    panel.columns.name = TICKER
    return panel
