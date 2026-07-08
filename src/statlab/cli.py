"""Command-line entry point for statlab.

Subcommands are added milestone by milestone. In M1 only ``version`` and ``gen-synth``
exist; ``ingest`` / ``research`` / ``backtest`` / ``backtest-pair`` / ``validate`` /
``sensitivity`` / ``report`` arrive with their layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from statlab import __version__
from statlab.backtest import (
    BacktestEngine,
    BuyAndHoldStrategy,
    PairsStrategy,
    Portfolio,
    sharpe_ratio,
)
from statlab.data import (
    PointInTimeUniverse,
    SyntheticSource,
    YFinanceSource,
    read_bars,
    simulate_correlated_ou_panel,
    to_price_panel,
    validate_bars,
    write_bars,
)
from statlab.data.sources import BarSource
from statlab.signals import SignalParams, discover_pairs


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"statlab {__version__}")
    return 0


def _cmd_ingest(ns: argparse.Namespace) -> int:
    """Fetch bars from a source, validate them, and write a partitioned Parquet dataset."""
    source: BarSource
    if ns.source == "synthetic":
        source = SyntheticSource(n=ns.n, n_pairs=ns.pairs, n_noise=ns.noise, seed=ns.seed)
    else:
        if not ns.tickers:
            print("error: --tickers is required for the yfinance source")
            return 2
        source = YFinanceSource(ns.tickers.split(","), start=ns.start, end=ns.end)

    bars = source.fetch()
    report = validate_bars(bars)
    for issue in report.issues:
        print(f"  [{issue.severity.value}] {issue.code} {issue.ticker or ''}: {issue.message}")
    if not report.ok:
        print(f"validation failed with {len(report.errors)} error(s); not writing")
        return 1

    root = write_bars(bars, ns.out)
    n_tickers = bars["ticker"].nunique()
    print(f"wrote {len(bars)} bars across {n_tickers} tickers to {root}")
    return 0


def _cmd_gen_synth(ns: argparse.Namespace) -> int:
    """Generate a synthetic price panel and write it to Parquet (offline demo data)."""
    rng = np.random.default_rng(ns.seed)
    panel, truth = simulate_correlated_ou_panel(ns.n, rng, n_pairs=ns.pairs, n_noise=ns.noise)
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    print(f"wrote {panel.shape[0]} rows x {panel.shape[1]} cols to {out}")
    print(f"ground-truth cointegrated pairs: {truth}")
    return 0


def _cmd_research(ns: argparse.Namespace) -> int:
    """Discover cointegrated, tradable pairs from an ingested bar dataset."""
    bars = read_bars(ns.dataset)
    panel = to_price_panel(bars)
    candidates = discover_pairs(
        panel,
        min_correlation=ns.min_corr,
        max_pvalue=ns.max_pvalue,
        max_half_life=ns.max_half_life,
    )
    if not candidates:
        print("no cointegrated pairs found under the given thresholds")
        return 0
    print(f"discovered {len(candidates)} pair(s), ranked by cointegration p-value:")
    for c in candidates[: ns.top]:
        print(f"  {c}")
    return 0


def _cmd_backtest(ns: argparse.Namespace) -> int:
    """Run a buy-and-hold backtest over an ingested dataset (engine demo)."""
    universe = PointInTimeUniverse.from_bars(read_bars(ns.dataset))
    days = universe.trading_days("1900-01-01", "2100-01-01")
    if len(days) < 2:
        print("not enough history to backtest")
        return 1
    start, end = days[0], days[-1]
    tickers = universe.members_as_of(start)[: ns.max_names]

    engine = BacktestEngine(universe, BuyAndHoldStrategy(tickers, ns.notional), Portfolio(ns.cash))
    result = engine.run(start, end)

    print(f"backtest {start.date()} -> {end.date()}  ({len(days)} days, {len(tickers)} names)")
    print(f"  initial equity : {result.initial_cash:,.0f}")
    print(f"  final equity   : {result.equity_curve.iloc[-1]:,.0f}")
    print(f"  total return   : {result.total_return:+.2%}")
    print(f"  ann. Sharpe    : {sharpe_ratio(result.returns()):.2f}")
    print(f"  transaction cost: {result.total_costs:,.0f}")
    print(f"  fills          : {len(result.fills)}")
    return 0


def _cmd_backtest_pair(ns: argparse.Namespace) -> int:
    """Run the M5 PairsStrategy over a single (auto-discovered or given) pair."""
    bars = read_bars(ns.dataset)
    universe = PointInTimeUniverse.from_bars(bars)

    y, x = ns.y, ns.x
    if not y or not x:
        candidates = discover_pairs(
            to_price_panel(bars), min_correlation=ns.min_corr, max_pvalue=ns.max_pvalue
        )
        if not candidates:
            print("no cointegrated pairs found to auto-select; pass --y/--x explicitly")
            return 1
        top = candidates[0]
        y, x = top.y, top.x
        print(f"auto-selected pair: {top}")

    days = universe.trading_days("1900-01-01", "2100-01-01")
    if len(days) < 2:
        print("not enough history to backtest")
        return 1
    start, end = days[0], days[-1]

    params = SignalParams(entry=ns.entry, exit=ns.exit, stop=ns.stop)
    strategy = PairsStrategy(y, x, ns.notional, params=params, delta=ns.delta)
    engine = BacktestEngine(universe, strategy, Portfolio(ns.cash))
    result = engine.run(start, end)

    print(f"backtest-pair {y}~{x}  {start.date()} -> {end.date()}  ({len(days)} days)")
    print(f"  initial equity : {result.initial_cash:,.0f}")
    print(f"  final equity   : {result.equity_curve.iloc[-1]:,.0f}")
    print(f"  total return   : {result.total_return:+.2%}")
    print(f"  ann. Sharpe    : {sharpe_ratio(result.returns()):.2f}")
    print(f"  transaction cost: {result.total_costs:,.0f}")
    print(f"  fills          : {len(result.fills)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="statlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_version = sub.add_parser("version", help="print the statlab version")
    p_version.set_defaults(func=_cmd_version)

    p_gen = sub.add_parser("gen-synth", help="generate a synthetic price panel")
    p_gen.add_argument("--out", default="data/synthetic/panel.parquet", help="output path")
    p_gen.add_argument("--n", type=int, default=1000, help="number of days")
    p_gen.add_argument("--pairs", type=int, default=3, help="cointegrated pairs")
    p_gen.add_argument("--noise", type=int, default=4, help="independent random walks")
    p_gen.add_argument("--seed", type=int, default=7, help="RNG seed")
    p_gen.set_defaults(func=_cmd_gen_synth)

    p_ing = sub.add_parser("ingest", help="ingest bars into a partitioned Parquet dataset")
    p_ing.add_argument(
        "--source", choices=["synthetic", "yfinance"], default="synthetic", help="data source"
    )
    p_ing.add_argument("--out", default="data/bars", help="output dataset root")
    p_ing.add_argument("--tickers", default="", help="comma-separated tickers (yfinance)")
    p_ing.add_argument("--start", default="2015-01-01", help="start date (yfinance)")
    p_ing.add_argument("--end", default=None, help="end date (yfinance)")
    p_ing.add_argument("--n", type=int, default=1000, help="days (synthetic)")
    p_ing.add_argument("--pairs", type=int, default=3, help="cointegrated pairs (synthetic)")
    p_ing.add_argument("--noise", type=int, default=4, help="random walks (synthetic)")
    p_ing.add_argument("--seed", type=int, default=7, help="RNG seed (synthetic)")
    p_ing.set_defaults(func=_cmd_ingest)

    p_res = sub.add_parser("research", help="discover cointegrated pairs from a dataset")
    p_res.add_argument("--dataset", default="data/bars", help="bar dataset root")
    p_res.add_argument("--min-corr", type=float, default=0.7, help="min return correlation")
    p_res.add_argument("--max-pvalue", type=float, default=0.05, help="max cointegration p")
    p_res.add_argument("--max-half-life", type=float, default=252.0, help="max half-life")
    p_res.add_argument("--top", type=int, default=20, help="how many pairs to print")
    p_res.set_defaults(func=_cmd_research)

    p_bt = sub.add_parser("backtest", help="run a buy-and-hold backtest (engine demo)")
    p_bt.add_argument("--dataset", default="data/bars", help="bar dataset root")
    p_bt.add_argument("--cash", type=float, default=1_000_000.0, help="initial cash")
    p_bt.add_argument("--notional", type=float, default=500_000.0, help="invested notional")
    p_bt.add_argument("--max-names", type=int, default=5, help="max tickers to hold")
    p_bt.set_defaults(func=_cmd_backtest)

    p_btp = sub.add_parser("backtest-pair", help="backtest the M5 pairs-trading strategy")
    p_btp.add_argument("--dataset", default="data/bars", help="bar dataset root")
    p_btp.add_argument("--y", default="", help="dependent-leg ticker (auto-discovers if omitted)")
    p_btp.add_argument("--x", default="", help="independent-leg ticker (auto-discovers if omitted)")
    p_btp.add_argument(
        "--min-corr", type=float, default=0.7, help="auto-discovery: min correlation"
    )
    p_btp.add_argument("--max-pvalue", type=float, default=0.05, help="auto-discovery: max p-value")
    p_btp.add_argument("--cash", type=float, default=1_000_000.0, help="initial cash")
    p_btp.add_argument("--notional", type=float, default=200_000.0, help="per-trade notional")
    p_btp.add_argument("--entry", type=float, default=2.0, help="entry z-score threshold")
    p_btp.add_argument("--exit", type=float, default=0.5, help="exit z-score threshold")
    p_btp.add_argument("--stop", type=float, default=4.0, help="stop z-score threshold")
    p_btp.add_argument("--delta", type=float, default=1e-6, help="Kalman drift parameter")
    p_btp.set_defaults(func=_cmd_backtest_pair)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    result: int = ns.func(ns)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
