"""Smoke tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from statlab import __version__
from statlab.cli import main
from statlab.data import read_bars


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    assert rc == 0
    assert __version__ in capsys.readouterr().out


def test_gen_synth_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "panel.parquet"
    # fmt: off
    rc = main([
        "gen-synth", "--out", str(out),
        "--n", "200", "--pairs", "2", "--noise", "1", "--seed", "3",
    ])
    # fmt: on
    assert rc == 0
    assert out.exists()
    panel = pd.read_parquet(out)
    assert panel.shape == (200, 2 * 2 + 1)


def test_gen_synth_is_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    for path in (a, b):
        main(["gen-synth", "--out", str(path), "--n", "150", "--seed", "11"])
    pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))


def test_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_ingest_synthetic_writes_dataset(tmp_path: Path) -> None:
    root = tmp_path / "bars"
    # fmt: off
    rc = main([
        "ingest", "--source", "synthetic", "--out", str(root),
        "--n", "120", "--pairs", "2", "--noise", "1", "--seed", "4",
    ])
    # fmt: on
    assert rc == 0
    bars = read_bars(root)
    assert bars["ticker"].nunique() == 2 * 2 + 1
    assert len(bars) == 120 * (2 * 2 + 1)


def test_ingest_yfinance_without_tickers_errors(tmp_path: Path) -> None:
    rc = main(["ingest", "--source", "yfinance", "--out", str(tmp_path / "x")])
    assert rc == 2


def test_research_discovers_pairs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "bars"
    # fmt: off
    main([
        "ingest", "--source", "synthetic", "--out", str(root),
        "--n", "800", "--pairs", "2", "--noise", "1", "--seed", "21",
    ])
    # fmt: on
    rc = main(["research", "--dataset", str(root), "--min-corr", "0.3", "--max-pvalue", "0.05"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "discovered" in out
