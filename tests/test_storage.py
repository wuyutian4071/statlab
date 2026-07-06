"""Tests for partitioned Parquet storage round-trips."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from statlab.data import SyntheticSource, read_bars, write_bars
from statlab.data.schema import BAR_COLUMNS, DATE, TICKER


def _sorted(bars: pd.DataFrame) -> pd.DataFrame:
    return bars[list(BAR_COLUMNS)].sort_values([TICKER, DATE]).reset_index(drop=True)


class TestStorage:
    def test_round_trip_preserves_data(self, tmp_path: Path) -> None:
        bars = SyntheticSource(n=150, n_pairs=2, n_noise=2, seed=5).fetch()
        write_bars(bars, tmp_path / "ds")
        back = read_bars(tmp_path / "ds")
        pd.testing.assert_frame_equal(_sorted(bars), back, check_dtype=False)

    def test_partition_pruning_reads_subset(self, tmp_path: Path) -> None:
        bars = SyntheticSource(n=100, n_pairs=2, n_noise=1, seed=6).fetch()
        write_bars(bars, tmp_path / "ds")
        tickers = sorted(bars[TICKER].unique())[:2]
        subset = read_bars(tmp_path / "ds", tickers=tickers)
        assert sorted(subset[TICKER].unique()) == tickers

    def test_creates_per_ticker_partitions(self, tmp_path: Path) -> None:
        bars = SyntheticSource(n=60, n_pairs=1, n_noise=1, seed=7).fetch()
        root = write_bars(bars, tmp_path / "ds")
        partition_dirs = sorted(p.name for p in Path(root).glob("ticker=*"))
        assert len(partition_dirs) == bars[TICKER].nunique()

    def test_read_missing_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_bars(tmp_path / "does-not-exist")

    def test_overwrite_same_ticker(self, tmp_path: Path) -> None:
        bars = SyntheticSource(n=80, n_pairs=1, n_noise=0, seed=8).fetch()
        write_bars(bars, tmp_path / "ds")
        write_bars(bars, tmp_path / "ds")  # rewrite must not duplicate rows
        back = read_bars(tmp_path / "ds")
        assert len(back) == len(bars)
