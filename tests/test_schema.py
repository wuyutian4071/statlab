"""Tests for the canonical bar schema and reshaping helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from statlab.data.schema import (
    ADJ_CLOSE,
    BAR_COLUMNS,
    CLOSE,
    DATE,
    TICKER,
    SchemaError,
    to_price_panel,
    validate_schema,
)


def _minimal_bars() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    rows = []
    for tkr, base in [("AAA", 100.0), ("BBB", 50.0)]:
        for i, d in enumerate(dates):
            px = base + i
            rows.append(
                {
                    DATE: d,
                    TICKER: tkr,
                    "open": px,
                    "high": px + 1,
                    "low": px - 1,
                    CLOSE: px,
                    ADJ_CLOSE: px,
                    "volume": 1_000.0,
                }
            )
    return pd.DataFrame(rows)[list(BAR_COLUMNS)]


class TestValidateSchema:
    def test_accepts_valid_frame(self) -> None:
        validate_schema(_minimal_bars())  # should not raise

    def test_rejects_missing_column(self) -> None:
        bars = _minimal_bars().drop(columns=[CLOSE])
        with pytest.raises(SchemaError, match="missing required columns"):
            validate_schema(bars)

    def test_rejects_non_datetime_date(self) -> None:
        bars = _minimal_bars()
        bars[DATE] = bars[DATE].astype(str)
        with pytest.raises(SchemaError, match="datetime"):
            validate_schema(bars)

    def test_rejects_non_numeric_value(self) -> None:
        bars = _minimal_bars()
        bars[CLOSE] = bars[CLOSE].astype(str)
        with pytest.raises(SchemaError, match="numeric"):
            validate_schema(bars)


class TestToPricePanel:
    def test_pivots_to_wide(self) -> None:
        panel = to_price_panel(_minimal_bars())
        assert list(panel.columns) == ["AAA", "BBB"]
        assert isinstance(panel.index, pd.DatetimeIndex)
        assert panel.shape == (2, 2)
        assert panel.loc[pd.Timestamp("2020-01-02"), "AAA"] == 101.0

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(SchemaError, match="field must be one of"):
            to_price_panel(_minimal_bars(), field="nope")

    def test_index_is_sorted(self) -> None:
        bars = _minimal_bars().sort_values(DATE, ascending=False)
        panel = to_price_panel(bars)
        assert panel.index.is_monotonic_increasing
