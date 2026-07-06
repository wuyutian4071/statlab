"""Tests for semantic bar validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statlab.data import SyntheticSource, validate_bars
from statlab.data.schema import ADJ_CLOSE, CLOSE, DATE, HIGH, LOW, VOLUME
from statlab.data.validation import Severity


def _clean_bars() -> pd.DataFrame:
    return SyntheticSource(n=250, n_pairs=1, n_noise=1, seed=3).fetch()


class TestValidateBars:
    def test_clean_data_has_no_errors(self) -> None:
        report = validate_bars(_clean_bars())
        assert report.ok
        assert not report.errors

    def test_detects_duplicate_rows(self) -> None:
        bars = _clean_bars()
        bars = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
        report = validate_bars(bars)
        assert not report.ok
        assert any(i.code == "duplicate_rows" for i in report.errors)

    def test_detects_nonpositive_price(self) -> None:
        bars = _clean_bars()
        bars.loc[0, CLOSE] = -1.0
        report = validate_bars(bars)
        assert any(i.code == "nonpositive_price" for i in report.errors)

    def test_detects_negative_volume(self) -> None:
        bars = _clean_bars()
        bars.loc[0, VOLUME] = -5.0
        report = validate_bars(bars)
        assert any(i.code == "negative_volume" for i in report.errors)

    def test_detects_ohlc_inconsistency(self) -> None:
        bars = _clean_bars()
        # Force high below the low on one row (violating the OHLC bound).
        high = bars[HIGH].to_numpy(dtype=float, copy=True)
        low = bars[LOW].to_numpy(dtype=float, copy=True)
        high[0] = low[0] - 1.0
        bars[HIGH] = high
        report = validate_bars(bars)
        assert any(i.code == "ohlc_inconsistent" for i in report.errors)

    def test_flags_possible_corporate_action_as_warning(self) -> None:
        bars = _clean_bars()
        tkr = bars["ticker"].iloc[0]
        mask = bars["ticker"] == tkr
        idx = bars.index[mask]
        # Halve the adjusted close from the midpoint onward: a 50% single-day drop.
        half = idx[len(idx) // 2 :]
        bars.loc[half, ADJ_CLOSE] = bars.loc[half, ADJ_CLOSE] / 2.0
        report = validate_bars(bars)
        codes = {i.code for i in report.warnings}
        assert "possible_corporate_action" in codes
        # A warning must not by itself make the dataset "not ok".
        assert report.ok

    def test_flags_calendar_gap(self) -> None:
        bars = _clean_bars()
        tkr = bars["ticker"].iloc[0]
        one = bars[bars["ticker"] == tkr].sort_values(DATE).copy()
        others = bars[bars["ticker"] != tkr]
        # Push every date from the 2nd row onward forward by 30 days, opening a big gap
        # right after the first bar. Adding a timedelta64 array to a DatetimeIndex is
        # both correct at runtime and cleanly typed.
        idx = pd.DatetimeIndex(one[DATE])
        delta = np.zeros(len(idx), dtype="timedelta64[D]")
        delta[1:] = np.timedelta64(30, "D")
        one[DATE] = idx + delta
        merged = (
            pd.concat([others, one], ignore_index=True)
            .sort_values(["ticker", DATE])
            .reset_index(drop=True)
        )
        report = validate_bars(merged)
        assert any(i.code == "calendar_gap" for i in report.warnings)

    def test_report_len_and_severity_partitions(self) -> None:
        bars = _clean_bars()
        bars.loc[0, CLOSE] = -1.0
        report = validate_bars(bars)
        assert len(report) == len(report.errors) + len(report.warnings)
        assert all(i.severity is Severity.ERROR for i in report.errors)
