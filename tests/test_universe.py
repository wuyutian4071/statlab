"""Tests for the PointInTimeUniverse behaviour (correctness of the API surface).

Anti-lookahead *guarantees* are proven separately and exhaustively in
``test_no_lookahead.py``; this module covers ordinary functional behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statlab.data import Membership, PointInTimeUniverse


def _panel() -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-04", periods=10, name="date")
    return pd.DataFrame(
        {
            "AAA": np.arange(10.0) + 100.0,
            "BBB": np.arange(10.0) + 200.0,
        },
        index=idx,
    )


class TestConstruction:
    def test_rejects_non_datetime_index(self) -> None:
        bad = _panel().reset_index(drop=True)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            PointInTimeUniverse(bad)

    def test_rejects_unsorted_index(self) -> None:
        panel = _panel().iloc[::-1]
        with pytest.raises(ValueError, match="sorted ascending"):
            PointInTimeUniverse(panel)

    def test_rejects_duplicate_dates(self) -> None:
        panel = _panel()
        dup = pd.concat([panel, panel.iloc[[0]]]).sort_index()
        with pytest.raises(ValueError, match="duplicate dates"):
            PointInTimeUniverse(dup)

    def test_from_bars(self, universe: PointInTimeUniverse) -> None:
        assert len(universe.tickers) == 2 * 2 + 2
        assert universe.trading_days("2015-01-01", "2100-01-01").size > 0


class TestPointInTimeReads:
    def test_as_of_clips_to_date(self) -> None:
        u = PointInTimeUniverse(_panel())
        t = pd.Timestamp("2021-01-08")
        frame = u.as_of(t)
        assert frame.index.max() <= t

    def test_as_of_returns_copy(self) -> None:
        u = PointInTimeUniverse(_panel())
        frame = u.as_of("2021-01-15")
        frame.iloc[0, 0] = -1.0
        assert u.as_of("2021-01-15").iloc[0, 0] != -1.0

    def test_window_size(self) -> None:
        u = PointInTimeUniverse(_panel())
        w = u.window("2021-01-15", 3)
        assert len(w) == 3
        assert w.index.max() <= pd.Timestamp("2021-01-15")

    def test_window_rejects_nonpositive_size(self) -> None:
        u = PointInTimeUniverse(_panel())
        with pytest.raises(ValueError, match="size must be positive"):
            u.window("2021-01-15", 0)

    def test_asof_date(self) -> None:
        u = PointInTimeUniverse(_panel())
        # A weekend query resolves to the prior Friday present in the panel.
        assert u.asof_date("2021-01-10") == pd.Timestamp("2021-01-08")

    def test_asof_date_before_history_is_none(self) -> None:
        u = PointInTimeUniverse(_panel())
        assert u.asof_date("2000-01-01") is None

    def test_price_as_of(self) -> None:
        u = PointInTimeUniverse(_panel())
        assert u.price_as_of("2021-01-15", "AAA") == pytest.approx(109.0)

    def test_price_as_of_unknown_ticker(self) -> None:
        u = PointInTimeUniverse(_panel())
        assert u.price_as_of("2021-01-15", "ZZZ") is None

    def test_volume_as_of_from_bars(self, universe: PointInTimeUniverse) -> None:
        tkr = universe.tickers[0]
        last_day = universe.trading_days("2015-01-01", "2100-01-01")[-1]
        adv = universe.volume_as_of(last_day, tkr, window=20)
        assert adv is not None and adv > 0


class TestMembership:
    def test_from_panel_uses_first_last_valid(self) -> None:
        panel = _panel()
        panel.loc[panel.index[:3], "AAA"] = np.nan  # AAA lists later
        m = Membership.from_panel(panel)
        assert not m.is_member("AAA", panel.index[0])
        assert m.is_member("AAA", panel.index[3])

    def test_members_as_of_respects_delisting(self) -> None:
        idx = pd.bdate_range("2021-01-04", periods=5, name="date")
        m = Membership(
            intervals={
                "AAA": (idx[0], idx[4]),
                "BBB": (idx[0], idx[2]),  # BBB delists after idx[2]
            }
        )
        assert set(m.members_as_of(idx[1])) == {"AAA", "BBB"}
        assert set(m.members_as_of(idx[3])) == {"AAA"}

    def test_as_of_hides_non_members(self) -> None:
        idx = pd.bdate_range("2021-01-04", periods=5, name="date")
        panel = pd.DataFrame({"AAA": range(5), "BBB": range(5)}, index=idx, dtype=float)
        m = Membership(intervals={"AAA": (idx[0], idx[4]), "BBB": (idx[3], idx[4])})
        u = PointInTimeUniverse(panel, membership=m)
        # Before BBB lists, it must not appear as a column.
        assert list(u.as_of(idx[1]).columns) == ["AAA"]
        assert set(u.as_of(idx[3]).columns) == {"AAA", "BBB"}
