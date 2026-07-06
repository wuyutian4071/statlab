"""Tests for the pair-discovery pipeline against a known ground truth."""

from __future__ import annotations

import numpy as np

from statlab.data import SyntheticSource
from statlab.data.schema import to_price_panel
from statlab.signals import discover_pairs
from statlab.signals.discovery import PairCandidate


class TestDiscovery:
    def test_finds_all_ground_truth_pairs(self) -> None:
        src = SyntheticSource(n=1200, n_pairs=3, n_noise=3, seed=11)
        panel = to_price_panel(src.fetch())
        found = discover_pairs(panel, min_correlation=0.3, max_pvalue=0.05)

        found_sets = {frozenset((c.y, c.x)) for c in found}
        for a, b in src.truth:
            assert frozenset((a, b)) in found_sets, f"missed cointegrated pair {a},{b}"

    def test_does_not_flag_pure_noise_pairs(self) -> None:
        src = SyntheticSource(n=1200, n_pairs=2, n_noise=3, seed=12)
        panel = to_price_panel(src.fetch())
        found = discover_pairs(panel, min_correlation=0.3, max_pvalue=0.05)

        found_sets = {frozenset((c.y, c.x)) for c in found}
        # The N* columns are independent random walks; no all-noise pair should appear.
        assert frozenset(("N0", "N1")) not in found_sets
        assert frozenset(("N0", "N2")) not in found_sets

    def test_ranked_by_pvalue(self) -> None:
        src = SyntheticSource(n=1000, n_pairs=3, n_noise=2, seed=13)
        panel = to_price_panel(src.fetch())
        found = discover_pairs(panel, min_correlation=0.0, max_pvalue=0.10)
        pvalues = [c.pvalue for c in found]
        assert pvalues == sorted(pvalues)

    def test_recovers_reasonable_hedge_ratio(self) -> None:
        src = SyntheticSource(n=1500, n_pairs=1, n_noise=1, seed=14)
        panel = to_price_panel(src.fetch())
        found = discover_pairs(panel, min_correlation=0.3)
        assert found, "expected at least the one ground-truth pair"
        top = found[0]
        assert isinstance(top, PairCandidate)
        assert top.beta > 0
        assert np.isfinite(top.half_life)

    def test_correlation_prefilter_reduces_candidates(self) -> None:
        src = SyntheticSource(n=1000, n_pairs=2, n_noise=4, seed=15)
        panel = to_price_panel(src.fetch())
        loose = discover_pairs(panel, min_correlation=0.0, max_pvalue=1.0)
        strict = discover_pairs(panel, min_correlation=0.95, max_pvalue=1.0)
        assert len(strict) <= len(loose)
