r"""HTML tear-sheet rendering for a single :class:`~statlab.backtest.BacktestResult`.

Self-contained output — the three charts (equity curve, drawdown, return distribution) are
rendered with matplotlib's headless ``Agg`` backend and embedded as base64 PNGs, so a single
``.html`` file is the whole report: easy to archive, email, or open straight from disk with no
folder of image assets alongside it.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from statlab.backtest.engine import BacktestResult, sharpe_ratio

_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         background: #0d1117; color: #c9d1d9; margin: 0; padding: 2rem; }
  h1 { font-size: 1.4rem; color: #f0f6fc; margin: 0 0 1.5rem 0; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
           background: #21262d; border: 1px solid #21262d; margin-bottom: 2rem; }
  .stat { background: #161b22; padding: 0.9rem 1rem; }
  .stat .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
                 color: #8b949e; margin-bottom: 0.3rem; }
  .stat .value { font-size: 1.15rem; font-variant-numeric: tabular-nums; color: #f0f6fc; }
  .stat .value.pos { color: #3fb950; }
  .stat .value.neg { color: #f85149; }
  .chart { margin-bottom: 1.5rem; }
  .chart img { max-width: 100%; display: block; border: 1px solid #21262d; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div class="stats">
  <div class="stat"><div class="label">Total Return</div>
    <div class="value {{ 'pos' if stats.total_return >= 0 else 'neg' }}">
      {{ "%.2f"|format(stats.total_return * 100) }}%
    </div></div>
  <div class="stat"><div class="label">Sharpe</div>
    <div class="value">{{ "%.2f"|format(stats.sharpe) }}</div></div>
  <div class="stat"><div class="label">Max Drawdown</div>
    <div class="value neg">{{ "%.2f"|format(stats.max_drawdown * 100) }}%</div></div>
  <div class="stat"><div class="label">Volatility (ann.)</div>
    <div class="value">{{ "%.2f"|format(stats.volatility * 100) }}%</div></div>
  <div class="stat"><div class="label">Win Rate</div>
    <div class="value">{{ "%.1f"|format(stats.win_rate * 100) }}%</div></div>
  <div class="stat"><div class="label">Fills</div>
    <div class="value">{{ stats.n_fills }}</div></div>
  <div class="stat"><div class="label">Transaction Costs</div>
    <div class="value">{{ "%.0f"|format(stats.total_costs) }}</div></div>
  <div class="stat"><div class="label">Trading Days</div>
    <div class="value">{{ stats.n_days }}</div></div>
</div>
<div class="chart">
  <img src="data:image/png;base64,{{ equity_chart }}" alt="Equity curve">
</div>
<div class="chart">
  <img src="data:image/png;base64,{{ drawdown_chart }}" alt="Drawdown">
</div>
<div class="chart">
  <img src="data:image/png;base64,{{ returns_chart }}" alt="Return distribution">
</div>
</body>
</html>
""")


@dataclass(frozen=True)
class TearSheetStats:
    """Summary statistics computed once and shared by the stats table and the charts."""

    total_return: float
    sharpe: float
    max_drawdown: float
    volatility: float
    win_rate: float
    n_fills: int
    total_costs: float
    n_days: int


def compute_stats(result: BacktestResult) -> TearSheetStats:
    """Summary stats for ``result``. Mirrors ``BacktestResult.total_return``'s own guard: a
    too-short equity curve reports zeroed stats rather than NaN from an empty rolling window.
    """
    equity = result.equity_curve
    returns = result.returns()

    if len(equity) < 2:
        max_drawdown = 0.0
        volatility = 0.0
        win_rate = 0.0
    else:
        drawdown = equity / equity.cummax() - 1.0
        max_drawdown = float(drawdown.min())
        volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
        win_rate = float((returns > 0).mean())

    return TearSheetStats(
        total_return=result.total_return,
        sharpe=sharpe_ratio(returns),
        max_drawdown=max_drawdown,
        volatility=volatility,
        win_rate=win_rate,
        n_fills=len(result.fills),
        total_costs=result.total_costs,
        n_days=len(equity),
    )


def _fig_to_base64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style_axes(ax: Any) -> None:
    ax.set_facecolor("#161b22")
    ax.grid(alpha=0.25, color="#30363d")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.title.set_color("#f0f6fc")
    ax.yaxis.label.set_color("#8b949e")
    ax.xaxis.label.set_color("#8b949e")


def _equity_chart(equity: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(equity.index, equity.to_numpy(), color="#58a6ff", linewidth=1.2)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity")
    _style_axes(ax)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _drawdown_chart(equity: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.4))
    if len(equity) >= 2:
        drawdown = (equity / equity.cummax() - 1.0) * 100
        ax.fill_between(drawdown.index, drawdown.to_numpy(), 0, color="#f85149", alpha=0.35)
        ax.plot(drawdown.index, drawdown.to_numpy(), color="#f85149", linewidth=0.8)
    ax.set_title("Drawdown")
    ax.set_ylabel("%")
    _style_axes(ax)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _returns_hist(returns: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.4))
    if len(returns) > 0:
        ax.hist(returns.to_numpy() * 100, bins=40, color="#3fb950", alpha=0.8)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("%")
    _style_axes(ax)
    fig.tight_layout()
    return _fig_to_base64(fig)


def render_tearsheet(
    result: BacktestResult,
    *,
    title: str = "Backtest Tear Sheet",
    out_path: str | Path | None = None,
) -> str:
    """Render a self-contained HTML tear sheet for ``result``: a summary stats table plus
    equity, drawdown, and return-distribution charts, all embedded as base64 PNGs.

    Returns the HTML string; if ``out_path`` is given, also writes it there (creating any
    missing parent directories).
    """
    stats = compute_stats(result)
    html = _TEMPLATE.render(
        title=title,
        stats=stats,
        equity_chart=_equity_chart(result.equity_curve),
        drawdown_chart=_drawdown_chart(result.equity_curve),
        returns_chart=_returns_hist(result.returns()),
    )
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
    return html
