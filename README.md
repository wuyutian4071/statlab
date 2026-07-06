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

Built milestone by milestone. Current: **M3 — the signal-research layer**: cointegration
tests, half-life, a hand-rolled Kalman hedge ratio, the z-score signal, and pair discovery.

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 | Repo skeleton, uv/ruff/mypy/pytest CI, synthetic OU data generator | ✅ |
| M2 | Data layer (schema, sources, validation, Parquet storage) + `PointInTimeUniverse` + `test_no_lookahead.py` | ✅ |
| M3 | Cointegration (Engle-Granger, Johansen), half-life, hand-rolled Kalman hedge ratio, z-score signal, pair discovery | ✅ |
| M4 | Event-driven backtester + portfolio-accounting invariants + cost model | ⬜ |
| M5 | Strategy wiring + single-pair known-answer backtests | ⬜ |
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
