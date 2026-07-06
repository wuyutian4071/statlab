"""Semantic validation of bar data.

Structural conformance is checked by :func:`statlab.data.schema.validate_schema`; this
module checks that the *values* make sense: positive prices, OHLC consistency, no
duplicate or unsorted dates, no suspicious gaps, and split/adjustment sanity.

Validation returns a :class:`ValidationReport` (a list of issues) rather than raising, so
a caller can decide what is fatal. Real market data is messy; the point is to *surface*
problems, not to pretend they don't exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from statlab.data.schema import (
    ADJ_CLOSE,
    CLOSE,
    DATE,
    HIGH,
    LOW,
    OPEN,
    PRICE_COLUMNS,
    TICKER,
    VOLUME,
    validate_schema,
)


class Severity(Enum):
    """How bad an issue is. ``ERROR`` means the data is unfit for research as-is."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    """A single problem found in a bar dataset."""

    severity: Severity
    code: str
    ticker: str | None
    message: str


@dataclass
class ValidationReport:
    """Collected validation issues with convenience accessors."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, severity: Severity, code: str, message: str, ticker: str | None = None) -> None:
        self.issues.append(ValidationIssue(severity, code, ticker, message))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """True if there are no ERROR-severity issues."""
        return not self.errors

    def __len__(self) -> int:
        return len(self.issues)


def validate_bars(
    bars: pd.DataFrame,
    *,
    max_gap_days: int = 5,
    split_ratio_threshold: float = 0.4,
) -> ValidationReport:
    """Validate long-form bars and return a report of all issues found.

    Checks performed
    ----------------
    * **schema** — structural conformance (delegated; raises on failure).
    * **duplicates** — no repeated ``(date, ticker)`` rows.
    * **ordering** — dates strictly increasing per ticker.
    * **positivity** — all price columns strictly positive; volume non-negative.
    * **OHLC consistency** — ``low <= min(open, close)`` and ``high >= max(open, close)``.
    * **calendar gaps** — business-day gaps larger than ``max_gap_days`` (WARNING).
    * **split sanity** — a raw-close jump not mirrored by the adjusted close suggests an
      unhandled split/dividend (WARNING). A day-over-day adjusted move beyond
      ``split_ratio_threshold`` is flagged as a possible unadjusted corporate action.

    Parameters
    ----------
    max_gap_days:
        Business-day gap above which a WARNING is raised.
    split_ratio_threshold:
        Fractional adjusted-close move above which a possible corporate action is flagged.
    """
    validate_schema(bars)
    report = ValidationReport()

    dup = bars.duplicated(subset=[DATE, TICKER]).sum()
    if dup:
        report.add(Severity.ERROR, "duplicate_rows", f"{dup} duplicate (date, ticker) rows")

    for ticker, group in bars.groupby(TICKER, sort=True):
        tkr = str(ticker)
        g = group.sort_values(DATE)

        dates = g[DATE]
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            report.add(Severity.ERROR, "unsorted_dates", "dates not strictly increasing", tkr)

        for col in PRICE_COLUMNS:
            if (g[col] <= 0).any():
                report.add(Severity.ERROR, "nonpositive_price", f"non-positive {col}", tkr)
        if (g[VOLUME] < 0).any():
            report.add(Severity.ERROR, "negative_volume", "negative volume", tkr)

        lo_ok = (g[LOW] <= g[[OPEN, CLOSE]].min(axis=1) + 1e-9).all()
        hi_ok = (g[HIGH] >= g[[OPEN, CLOSE]].max(axis=1) - 1e-9).all()
        if not (lo_ok and hi_ok):
            report.add(Severity.ERROR, "ohlc_inconsistent", "OHLC bounds violated", tkr)

        gaps = dates.diff().dt.days.dropna()
        big_gaps = int((gaps > max_gap_days).sum())
        if big_gaps:
            report.add(
                Severity.WARNING,
                "calendar_gap",
                f"{big_gaps} gaps > {max_gap_days} days",
                tkr,
            )

        adj = g[ADJ_CLOSE].to_numpy()
        if len(adj) > 1:
            ret = np.abs(np.diff(adj) / adj[:-1])
            n_jumps = int((ret > split_ratio_threshold).sum())
            if n_jumps:
                report.add(
                    Severity.WARNING,
                    "possible_corporate_action",
                    f"{n_jumps} adj-close moves > {split_ratio_threshold:.0%} "
                    "(possible unadjusted split/dividend)",
                    tkr,
                )

    return report
