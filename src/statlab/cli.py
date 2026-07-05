"""Command-line entry point for statlab.

Subcommands are added milestone by milestone. In M1 only ``version`` and ``gen-synth``
exist; ``ingest`` / ``research`` / ``backtest`` / ``report`` arrive with their layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from statlab import __version__
from statlab.data import simulate_correlated_ou_panel


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"statlab {__version__}")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    result: int = ns.func(ns)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
