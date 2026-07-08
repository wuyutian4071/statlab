"""Tests for the generic sensitivity-grid runner."""

from __future__ import annotations

import pytest

from statlab.validation import sensitivity_grid


class TestSensitivityGrid:
    def test_rejects_empty_grid(self) -> None:
        with pytest.raises(ValueError, match="grid must not be empty"):
            sensitivity_grid({}, lambda: 0.0)

    def test_produces_the_full_cartesian_product(self) -> None:
        grid = {"a": [1, 2, 3], "b": [10, 20]}
        df = sensitivity_grid(grid, lambda a, b: float(a + b))
        assert len(df) == 6
        assert set(zip(df["a"], df["b"], strict=True)) == {
            (1, 10),
            (1, 20),
            (2, 10),
            (2, 20),
            (3, 10),
            (3, 20),
        }

    def test_metric_column_holds_run_fn_output(self) -> None:
        grid = {"a": [1, 2], "b": [3, 4]}
        df = sensitivity_grid(grid, lambda a, b: float(a * b))
        for _, row in df.iterrows():
            assert row["metric"] == pytest.approx(row["a"] * row["b"])

    def test_custom_metric_name(self) -> None:
        df = sensitivity_grid({"x": [1, 2]}, lambda x: float(x), metric_name="score")
        assert "score" in df.columns
        assert "metric" not in df.columns

    def test_finds_the_known_optimum_of_a_deterministic_function(self) -> None:
        # f(x, y) = -(x - 3)^2 - (y + 1)^2, maximized exactly at x=3, y=-1.
        grid = {"x": [0, 1, 2, 3, 4, 5], "y": [-3, -2, -1, 0, 1]}

        def run_fn(x: int, y: int) -> float:
            return -((x - 3) ** 2) - (y + 1) ** 2

        df = sensitivity_grid(grid, run_fn)
        best = df.loc[df["metric"].idxmax()]
        assert best["x"] == 3
        assert best["y"] == -1
        assert best["metric"] == pytest.approx(0.0)

    def test_single_parameter_grid(self) -> None:
        df = sensitivity_grid({"n": [1, 2, 3, 4]}, lambda n: float(n))
        assert len(df) == 4
        assert list(df["n"]) == [1, 2, 3, 4]
