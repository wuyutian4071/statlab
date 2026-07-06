"""Data layer: ingestion, validation, point-in-time storage, and simulation."""

from __future__ import annotations

from statlab.data.schema import (
    ADJ_CLOSE,
    BAR_COLUMNS,
    CLOSE,
    DATE,
    TICKER,
    VOLUME,
    SchemaError,
    to_price_panel,
    validate_schema,
)
from statlab.data.sources import BarSource, SyntheticSource, YFinanceSource
from statlab.data.storage import read_bars, write_bars
from statlab.data.synthetic import (
    OUParams,
    simulate_cointegrated_pair,
    simulate_correlated_ou_panel,
    simulate_ou,
    simulate_random_walk,
)
from statlab.data.universe import Membership, PointInTimeUniverse
from statlab.data.validation import (
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_bars,
)

__all__ = [
    "ADJ_CLOSE",
    "BAR_COLUMNS",
    "CLOSE",
    "DATE",
    "TICKER",
    "VOLUME",
    "BarSource",
    "Membership",
    "OUParams",
    "PointInTimeUniverse",
    "SchemaError",
    "Severity",
    "SyntheticSource",
    "ValidationIssue",
    "ValidationReport",
    "YFinanceSource",
    "read_bars",
    "simulate_cointegrated_pair",
    "simulate_correlated_ou_panel",
    "simulate_ou",
    "simulate_random_walk",
    "to_price_panel",
    "validate_bars",
    "validate_schema",
    "write_bars",
]
