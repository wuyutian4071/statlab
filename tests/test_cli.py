"""Smoke tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from statlab import __version__
from statlab.cli import main


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
