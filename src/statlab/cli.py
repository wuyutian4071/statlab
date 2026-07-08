"""Command-line entry point for statlab.

Subcommands are added milestone by milestone. In M1 only ``version`` and ``gen-synth``
exist; ``ingest`` / ``research`` / ``backtest`` / ``backtest-pair`` / ``validate`` /
``sensitivity`` arrive with their layers. HTML tear sheets (M7) are an output option
(``--report``) on ``backtest-pair`` rather than a separate subcommand.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

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
from statlab.report import render_tearsheet
from statlab.signals import SignalParams, discover_pairs
from statlab.validation import (
    combined_oos_sharpe,
    deflated_sharpe_ratio,
    run_walk_forward,
    sensitivity_grid,
    walk_forward_windows,
)


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

    if ns.report:
        render_tearsheet(result, title=f"{y}~{x} Pairs Backtest", out_path=ns.report)
        print(f"  tear sheet written to {ns.report}")

    return 0


def _cmd_validate(ns: argparse.Namespace) -> int:
    """M6: walk-forward discovery+backtest — discover a pair on each train window, trade it
    strictly out-of-sample on the following test window, repeat rolling forward."""
    universe = PointInTimeUniverse.from_bars(read_bars(ns.dataset))
    days = universe.trading_days("1900-01-01", "2100-01-01")

    windows = walk_forward_windows(days, ns.train_days, ns.test_days, ns.step_days)
    if not windows:
        print("not enough history for even one walk-forward window")
        return 1

    results = run_walk_forward(
        universe,
        windows,
        cash=ns.cash,
        notional=ns.notional,
        min_correlation=ns.min_corr,
        max_pvalue=ns.max_pvalue,
        max_half_life=ns.max_half_life,
    )

    print(f"walk-forward: {len(windows)} windows, train={ns.train_days}d test={ns.test_days}d")
    for r in results:
        label = f"{r.window.test_start.date()} -> {r.window.test_end.date()}"
        if r.pair is None or r.result is None:
            print(f"  {label}  no pair discovered")
            continue
        sharpe = sharpe_ratio(r.result.returns())
        print(
            f"  {label}  {r.pair.y}~{r.pair.x}  return={r.result.total_return:+.2%}  "
            f"sharpe={sharpe:.2f}  fills={len(r.result.fills)}"
        )

    print(f"combined out-of-sample Sharpe: {combined_oos_sharpe(results):.2f}")
    return 0


def _cmd_sensitivity(ns: argparse.Namespace) -> int:
    """M6: a small entry/delta sensitivity grid for a pair, reporting the deflated Sharpe of
    the best cell next to its naive (undeflated) Sharpe so the discount is visible."""
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

    entries = [float(v) for v in ns.entries.split(",")]
    deltas = [float(v) for v in ns.deltas.split(",")]
    returns_by_combo: dict[tuple[float, float], np.ndarray] = {}

    def run_fn(entry: float, delta: float) -> float:
        params = SignalParams(entry=entry, exit=entry * 0.25, stop=entry * 2.0)
        strategy = PairsStrategy(y, x, ns.notional, params=params, delta=delta)
        engine = BacktestEngine(universe, strategy, Portfolio(ns.cash))
        result = engine.run(start, end)
        returns = result.returns()
        returns_by_combo[(entry, delta)] = returns.to_numpy()
        return sharpe_ratio(returns)

    grid = sensitivity_grid({"entry": entries, "delta": deltas}, run_fn)
    print(f"sensitivity grid: {y}~{x}  {len(grid)} combinations")
    print(grid.to_string(index=False))

    best_row = grid.loc[grid["metric"].idxmax()]
    best_entry = cast(float, best_row["entry"])
    best_delta = cast(float, best_row["delta"])
    best_returns = returns_by_combo[(best_entry, best_delta)]
    dsr_result = deflated_sharpe_ratio(grid["metric"].tolist(), best_returns=best_returns)
    print(f"\nbest cell: entry={best_entry} delta={best_delta}")
    print(f"  naive Sharpe   : {best_row['metric']:.2f}")
    print(
        f"  deflated Sharpe: {dsr_result.dsr:.4f}  "
        f"(SR_0={dsr_result.sr_0:.2f}, N={dsr_result.n_trials})"
    )
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
    p_btp.add_argument("--report", default=None, help="write an HTML tear sheet to this path")
    p_btp.set_defaults(func=_cmd_backtest_pair)

    p_val = sub.add_parser("validate", help="M6: walk-forward discovery+backtest")
    p_val.add_argument("--dataset", default="data/bars", help="bar dataset root")
    p_val.add_argument("--train-days", type=int, default=200, help="train window length")
    p_val.add_argument("--test-days", type=int, default=100, help="test window length")
    p_val.add_argument("--step-days", type=int, default=None, help="roll step (default test-days)")
    p_val.add_argument("--cash", type=float, default=1_000_000.0, help="initial cash per window")
    p_val.add_argument("--notional", type=float, default=200_000.0, help="per-trade notional")
    p_val.add_argument("--min-corr", type=float, default=0.3, help="discovery: min correlation")
    p_val.add_argument("--max-pvalue", type=float, default=0.1, help="discovery: max p-value")
    p_val.add_argument("--max-half-life", type=float, default=252.0, help="discovery: max HL")
    p_val.set_defaults(func=_cmd_validate)

    p_sens = sub.add_parser("sensitivity", help="M6: sensitivity grid + deflated Sharpe")
    p_sens.add_argument("--dataset", default="data/bars", help="bar dataset root")
    p_sens.add_argument("--y", default="", help="dependent-leg ticker (auto-discover if omitted)")
    p_sens.add_argument("--x", default="", help="independent-leg ticker (auto-discover if omitted)")
    p_sens.add_argument("--min-corr", type=float, default=0.3, help="auto-discovery: min corr")
    p_sens.add_argument("--max-pvalue", type=float, default=0.1, help="auto-discovery: max p-value")
    p_sens.add_argument("--cash", type=float, default=1_000_000.0, help="initial cash")
    p_sens.add_argument("--notional", type=float, default=200_000.0, help="per-trade notional")
    p_sens.add_argument("--entries", default="1.0,1.5,2.0", help="comma-separated entry thresholds")
    p_sens.add_argument("--deltas", default="1e-6,1e-5", help="comma-separated Kalman deltas")
    p_sens.set_defaults(func=_cmd_sensitivity)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    result: int = ns.func(ns)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
