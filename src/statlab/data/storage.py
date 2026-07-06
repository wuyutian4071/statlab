"""Partitioned Parquet storage for bar data.

Bars are written as a Parquet dataset partitioned by ``ticker`` (one directory per
symbol). This gives cheap per-ticker reads (predicate pushdown / partition pruning) and
scales to a large universe without loading everything into memory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from statlab.data.schema import BAR_COLUMNS, DATE, TICKER, validate_schema


def write_bars(bars: pd.DataFrame, root: str | Path) -> Path:
    """Write long-form bars to a Parquet dataset partitioned by ticker.

    Parameters
    ----------
    bars:
        Canonical long-form bars.
    root:
        Destination directory (created if absent). Existing partitions for the same
        tickers are overwritten.

    Returns
    -------
    pathlib.Path
        The dataset root.
    """
    validate_schema(bars)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(bars[list(BAR_COLUMNS)], preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=[TICKER],
        existing_data_behavior="delete_matching",
    )
    return root


def read_bars(root: str | Path, tickers: list[str] | None = None) -> pd.DataFrame:
    """Read bars back from a partitioned Parquet dataset.

    Parameters
    ----------
    root:
        Dataset root written by :func:`write_bars`.
    tickers:
        If given, only these partitions are read (partition pruning). Otherwise all.

    Returns
    -------
    pandas.DataFrame
        Long-form bars in canonical column order, sorted by ``(ticker, date)``.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"no dataset at {root}")

    dataset = ds.dataset(str(root), format="parquet", partitioning="hive")
    table = (
        dataset.to_table(filter=ds.field(TICKER).isin(tickers))
        if tickers is not None
        else dataset.to_table()
    )
    bars: pd.DataFrame = table.to_pandas()
    # Partition column comes back as a categorical/string; normalise to plain object.
    bars[TICKER] = bars[TICKER].astype(str)
    result: pd.DataFrame = (
        bars[list(BAR_COLUMNS)].sort_values([TICKER, DATE]).reset_index(drop=True)
    )
    return result
