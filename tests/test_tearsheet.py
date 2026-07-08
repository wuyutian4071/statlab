"""Tests for the HTML tear sheet: compute_stats against independently-computed expected
values on a small hand-constructed BacktestResult, and render_tearsheet's HTML output.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest

from statlab.backtest import BacktestResult
from statlab.report import compute_stats, render_tearsheet


def _result(equity_values: list[float]) -> BacktestResult:
    idx = pd.bdate_range("2020-01-01", periods=len(equity_values))
    equity = pd.Series(equity_values, index=idx, name="equity")
    return BacktestResult(
        equity_curve=equity,
        fills=[],
        initial_cash=equity_values[0] if equity_values else 0.0,
        final_positions={},
        total_costs=12.5,
    )


class TestComputeStats:
    def test_matches_independently_computed_values(self) -> None:
        # equity = [100, 110, 99, 105]; every value below computed independently (see the
        # session's verification transcript), not by re-deriving this module's own formula.
        result = _result([100.0, 110.0, 99.0, 105.0])
        stats = compute_stats(result)

        assert stats.total_return == pytest.approx(0.05)
        assert stats.max_drawdown == pytest.approx(-0.1)
        assert stats.volatility == pytest.approx(1.6818263718064306)
        assert stats.win_rate == pytest.approx(2.0 / 3.0)
        assert stats.sharpe == pytest.approx(3.0270122863164577)
        assert stats.n_days == 4
        assert stats.n_fills == 0
        assert stats.total_costs == pytest.approx(12.5)

    def test_empty_equity_curve_reports_zeroed_stats_not_nan(self) -> None:
        result = _result([])
        stats = compute_stats(result)
        assert stats.total_return == 0.0
        assert stats.max_drawdown == 0.0
        assert stats.volatility == 0.0
        assert stats.win_rate == 0.0
        assert stats.sharpe == 0.0
        assert stats.n_days == 0

    def test_single_point_equity_curve_reports_zeroed_stats(self) -> None:
        result = _result([100.0])
        stats = compute_stats(result)
        assert stats.total_return == 0.0
        assert stats.max_drawdown == 0.0
        assert stats.volatility == 0.0
        assert stats.n_days == 1

    def test_monotonically_rising_equity_has_zero_drawdown(self) -> None:
        result = _result([100.0, 101.0, 103.0, 108.0])
        stats = compute_stats(result)
        assert stats.max_drawdown == pytest.approx(0.0)
        assert stats.win_rate == pytest.approx(1.0)


class TestRenderTearsheet:
    def test_html_contains_title_and_formatted_stats(self) -> None:
        result = _result([100.0, 110.0, 99.0, 105.0])
        html = render_tearsheet(result, title="My Pair Backtest")

        assert "My Pair Backtest" in html
        assert "5.00%" in html  # total return
        assert "-10.00%" in html  # max drawdown
        assert "<html" in html and "</html>" in html

    def test_html_embeds_three_base64_png_charts(self) -> None:
        result = _result([100.0, 110.0, 99.0, 105.0])
        html = render_tearsheet(result)

        assert html.count('src="data:image/png;base64,') == 3
        # Every embedded payload must actually be valid, non-trivial base64-encoded PNG bytes.
        start = 0
        found = 0
        marker = 'src="data:image/png;base64,'
        while True:
            i = html.find(marker, start)
            if i == -1:
                break
            j = html.find('"', i + len(marker))
            payload = html[i + len(marker) : j]
            decoded = base64.b64decode(payload)
            assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
            assert len(decoded) > 500
            found += 1
            start = j
        assert found == 3

    def test_writes_to_out_path_and_creates_parent_dirs(self, tmp_path: Path) -> None:
        result = _result([100.0, 105.0])
        out_path = tmp_path / "nested" / "report.html"
        html = render_tearsheet(result, out_path=out_path)

        assert out_path.exists()
        assert out_path.read_text() == html

    def test_does_not_crash_on_empty_equity_curve(self) -> None:
        result = _result([])
        html = render_tearsheet(result)
        assert "<html" in html
        assert html.count('src="data:image/png;base64,') == 3
