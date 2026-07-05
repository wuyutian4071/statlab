"""Data layer: ingestion, validation, point-in-time storage, and simulation."""

from __future__ import annotations

from statlab.data.synthetic import (
    OUParams,
    simulate_cointegrated_pair,
    simulate_correlated_ou_panel,
    simulate_ou,
    simulate_random_walk,
)

__all__ = [
    "OUParams",
    "simulate_cointegrated_pair",
    "simulate_correlated_ou_panel",
    "simulate_ou",
    "simulate_random_walk",
]
