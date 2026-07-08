# statlab

> Statistical-arbitrage research platform & event-driven backtester — built to research
> a cointegration-based pairs-trading strategy with **research rigor**: no lookahead bias,
> realistic transaction costs, and honest out-of-sample evaluation.

[![CI](https://github.com/wuyutian4071/statlab/actions/workflows/ci.yml/badge.svg)](https://github.com/wuyutian4071/statlab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**What / Why / Results (30-second version)**

- **What:** an end-to-end quant workflow — data pipeline → signal research (cointegration
  + Kalman-filter hedge ratios) → event-driven backtester with realistic costs →
  walk-forward validation → HTML tear sheet.
- **Why it's different:** most candidate "backtests" are toy vectorized loops that ignore
  costs and leak future information. This one treats sloppy backtesting as a bug:
  point-in-time data, next-bar fills, square-root market-impact costs, and strict
  train/test separation.
- **Results:** _(populated as milestones land — see `BENCHMARKS.md` / the tear sheet)._

## Status

Built milestone by milestone. Current: **M5 — the pairs-trading strategy**: the M3 signals
(Kalman hedge ratio, z-score state machine) wired into an actual `Strategy` that trades a
cointegrated pair through the M4 engine, verified against independently-computed known answers.

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 | Repo skeleton, uv/ruff/mypy/pytest CI, synthetic OU data generator | ✅ |
| M2 | Data layer (schema, sources, validation, Parquet storage) + `PointInTimeUniverse` + `test_no_lookahead.py` | ✅ |
| M3 | Cointegration (Engle-Granger, Johansen), half-life, hand-rolled Kalman hedge ratio, z-score signal, pair discovery | ✅ |
| M4 | Event-driven backtester + portfolio-accounting invariants + cost model | ✅ |
| M5 | Strategy wiring + single-pair known-answer backtests | ✅ |
| M6 | Walk-forward + sensitivity grid + deflated Sharpe | ⬜ |
| M7 | HTML tear sheet + notebooks + `make reproduce` | ⬜ |
| M8 | Polished docs, honest results discussion, architecture diagram | ⬜ |

## Quickstart

```bash
# Install uv (https://docs.astral.sh/uv/) then:
make install        # sync deps into a local .venv
make check          # lint + typecheck + test (the full CI gate)

# Ingest bars into a partitioned Parquet dataset (offline synthetic source by default;
# use --source yfinance --tickers AAPL,MSFT,... for real data):
uv run statlab ingest --source synthetic --out data/bars --n 1000 --seed 7

# Run the buy-and-hold engine demo over an ingested dataset:
uv run statlab backtest --dataset data/bars --cash 1000000 --max-names 5

# Run the M5 pairs-trading strategy (auto-discovers a pair if --y/--x aren't given —
# the default discovery thresholds are conservative, so a smaller/looser demo dataset
# tends to actually find one; see the M5 section below for a worked example):
uv run statlab backtest-pair --dataset data/bars --min-corr 0.3 --max-pvalue 0.1
```

### The point-in-time universe (M2)

`PointInTimeUniverse` is the project's structural defence against lookahead bias. Prices
are stored privately; every read goes through an `as_of(t)` / `window(t, size)` method that
clips to `date <= t` and to the tickers that are *members* at `t` (survivorship-aware).

```python
from statlab.data import PointInTimeUniverse, SyntheticSource

u = PointInTimeUniverse.from_bars(SyntheticSource(n=1000).fetch())
u.as_of("2018-06-01")          # only rows dated <= 2018-06-01
u.window("2018-06-01", 60)     # the trailing 60 observations, never any future bar
u.members_as_of("2018-06-01")  # tickers actually listed on that date
```

The guarantee is proven, not asserted by inspection: `tests/test_no_lookahead.py` checks
the clipping invariant for *every* date and — the decisive test — verifies that a universe
built on a *prefix* of history returns byte-for-byte the same thing as one built on full
history, for every date in the prefix. If future rows could influence a past read, that
equality would break.

### Signal research (M3)

```bash
# Discover cointegrated, tradable pairs from an ingested dataset:
uv run statlab research --dataset data/bars --min-corr 0.7 --max-pvalue 0.05
```

The `statlab.signals` package provides, each validated against a known closed-form or
statsmodels reference:

- **Cointegration** — `engle_granger` (two-step; hedge ratio by OLS, p-value from the
  correctly-tabulated MacKinnon critical values) and `johansen` (system trace test / rank).
- **`half_life`** — mean-reversion half-life from an AR(1) fit; recovers `ln(2)/θ` on a
  synthetic OU spread and returns `inf` for a random walk.
- **`kalman_hedge`** — a hand-rolled Kalman filter tracking a *dynamic* hedge ratio
  `[β_t, α_t]` causally. The innovation is the tradable spread; in the constant-coefficient
  limit it reduces to recursive least squares and converges to OLS (a tested property).
- **`rolling_zscore` + `generate_positions`** — a causal z-score and an entry/exit/stop
  state machine with hysteresis.
- **`discover_pairs`** — the funnel: correlation pre-filter → cointegration → half-life
  band, ranked by p-value. (Multiple-comparisons caveat: this is in-sample selection;
  M6's walk-forward + deflated Sharpe address it.)

### The event-driven backtester (M4)

Per trading day `t`, the engine runs a strict three-step loop — **execute → mark → decide**
— and it's the ordering, not just the components, that makes the backtest lookahead-free:

```python
from statlab.backtest import BacktestEngine, BuyAndHoldStrategy, Portfolio, sharpe_ratio
from statlab.data import PointInTimeUniverse, SyntheticSource

u = PointInTimeUniverse.from_bars(SyntheticSource(n=1000, seed=7).fetch())
start, end = u.trading_days("2000-01-01", "2100-01-01")[[0, -1]]
tickers = u.members_as_of(start)[:5]

engine = BacktestEngine(u, BuyAndHoldStrategy(tickers, notional=500_000), Portfolio(1_000_000))
result = engine.run(start, end)
sharpe_ratio(result.returns())
```

1. **Execute** — orders decided on bar `t-1` fill at bar `t`'s (adjusted) open, priced
   through `CostModel`: `max(c_min, c_ps·Q)` commission + half-spread in bps of notional +
   square-root market impact `η·σ·√(Q/ADV)`, with `σ`/ADV read causally as of `t`.
2. **Mark** — the portfolio is marked to `t`'s close and the equity curve records that point.
3. **Decide** — only now does the strategy see bar `t` and return orders, which will
   themselves fill on `t+1`'s open.

A decision at `t` can therefore only ever transact at `t+1` — combined with the point-in-time
universe from M2, there is structurally no path for future information to reach a trade.

**The accounting invariant**: `Portfolio` tracks trade cashflow (`cash`) and transaction
costs (`costs`) separately, so `equity = cash + Σ qᵢpᵢ - costs` is independently checkable —
`tests/test_engine.py::TestInvariantEndToEnd` reconstructs final equity from the raw fill log
by hand and asserts it matches the engine's own number bit for bit, and a second test proves
a costed run always ends with strictly less equity than an identical frictionless one. Real
pairs-trading strategy wiring lands in M5; this milestone's `BuyAndHoldStrategy` exists to
exercise and benchmark the engine itself (`statlab backtest` on the CLI).

### The pairs-trading strategy (M5)

`PairsStrategy` wires the M3 signals into the `Strategy` protocol — no new signal math, purely
the integration: turning a causal hedge ratio and z-score into correctly-signed, correctly-sized,
correctly-lagged orders. Each bar it re-runs the batch `kalman_hedge` filter over the full
history available so far rather than maintaining incremental filter state; because the filter and
the z-score state machine are both causal (step `t` depends only on data through `t`), this is
mathematically identical to an incremental filter, just `O(n)` per bar instead of `O(1)` — a
deliberate simplicity-over-performance tradeoff, since it means the live strategy calls the exact
same, already-validated M3 function the same way discovery does, with zero risk of a second,
subtly-divergent re-implementation of the recursion.

```bash
uv run statlab ingest --source synthetic --out data/bars --n 400 --pairs 2 --noise 2 --seed 42
uv run statlab backtest-pair --dataset data/bars --min-corr 0.3 --max-pvalue 0.1
```

```
auto-selected pair: P0b~P0a beta=1.810 corr=0.58 p=0.0004 hl=7.6
backtest-pair P0b~P0a  2015-01-02 -> 2016-07-14  (400 days)
  initial equity : 1,000,000
  final equity   : 1,015,561
  total return   : +1.56%
  ann. Sharpe    : 0.94
  transaction cost: 956
  fills          : 32
```

**A real calibration finding worth keeping** (the same spirit as M4's sigma/spread-vs-cost
finding): the Kalman filter's `delta` parameter (how much the hedge ratio is allowed to drift
per bar) defaults to `1e-6` here, tighter than `kalman_hedge`'s own general-purpose default of
`1e-4`. With a larger `delta`, the filter's state uncertainty — and so the innovation variance
the z-score divides by — keeps growing over a long series, which *dampens* z-score sensitivity
late in a long backtest even while the underlying spread keeps mean-reverting completely
normally. The failure mode isn't a crash or a wrong number; it's a strategy that silently never
trades. Caught by actually running the CLI demo end-to-end and asking why `fills = 0`, not by
any test in isolation — a reminder that integration-level manual verification finds a different
class of bug than unit tests do, however thorough the unit tests are.

**Performance, measured, not assumed**: 400 days runs in ~2s. A 3000-day (~12-year) backtest on
this machine took 37.5s — the `O(n²)` cost is real at that scale, so M6's walk-forward and
sensitivity-grid tools (many shorter backtests, not one long one) are the more practical way to
explore a longer history, not a single giant `backtest-pair` run.

Known-answer verification lives in `tests/test_pairs_strategy.py`: fills are checked against a
state sequence computed independently via the same M3 batch functions (proving the wiring and
the one-bar execution lag are both correct, not just "it ran"); a small deterministic step-shock
scenario with exactly-known transition bars and directions (a sign-convention check the
statistical test couldn't fail loudly on by chance); and an equity reconciliation reusing M4's
"reconstruct final equity from the raw fill log by hand" pattern for this two-leg case.

## Design principles

1. **No lookahead bias.** Any query as of date *T* only sees data available at *T*
   (`PointInTimeUniverse`, arriving in M2), and a dedicated `test_no_lookahead.py` proves
   future data cannot leak through any public API.
2. **Realistic costs.** Fills at the *next* bar; cost = commission + half-spread +
   square-root market impact `η·σ·√(Q/ADV)`.
3. **Reproducibility.** Seeded RNG passed explicitly (never global state), pinned deps via
   `uv.lock`, and `make reproduce` regenerates every artifact.
4. **Honesty.** An honest marginal/negative result with correct methodology beats an
   inflated backtest. Limitations are documented, not hidden.

## Package layout

```
src/statlab/
  data/        ingestion, validation, point-in-time storage, synthetic generators
  signals/     cointegration tests, half-life, Kalman hedge ratio, z-scores
  backtest/    event-driven engine, execution/cost model, portfolio accounting
  validation/  walk-forward, sensitivity grids, deflated Sharpe
  report/      HTML tear-sheet rendering
```

## License

MIT — see [LICENSE](LICENSE).
