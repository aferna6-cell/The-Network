"""Tests for the statistical strategy gate (src/verify.py).

Pure numerics — no network, no model. Validates each statistic against a known
property and proves the gate passes strong edge / fails noise.
"""
from __future__ import annotations

import numpy as np
import pytest

import config
from src import verify


def _rng(seed):
    return np.random.default_rng(seed)


def test_sharpe_matches_manual_formula():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert verify.sharpe(r, 252) == pytest.approx(expected)


def test_sharpe_zero_variance_is_zero():
    assert verify.sharpe([0.01, 0.01, 0.01], 252) == 0.0


def test_max_drawdown_simple_case():
    # +10% then -50% from the peak.
    r = [0.10, -0.50, 0.0]
    assert verify.max_drawdown(r) == pytest.approx(-0.50)


def test_max_drawdown_monotonic_up_is_zero():
    assert verify.max_drawdown([0.01, 0.02, 0.03]) == pytest.approx(0.0)


def test_newey_west_tstat_grows_with_signal():
    strong = np.full(250, 0.01) + _rng(1).normal(0, 0.001, 250)
    weak = _rng(2).normal(0, 0.02, 250)
    assert verify.newey_west_tstat(strong) > 5
    assert abs(verify.newey_west_tstat(weak)) < 2


def test_psr_high_for_strong_sharpe_half_for_zero_mean():
    strong = np.full(300, 0.01) + _rng(3).normal(0, 0.002, 300)
    flat = _rng(4).normal(0, 0.02, 300)
    flat = flat - flat.mean()        # exactly zero mean => SR 0 => PSR ~ 0.5
    assert verify.probabilistic_sharpe_ratio(strong) > 0.99
    assert verify.probabilistic_sharpe_ratio(flat) == pytest.approx(0.5, abs=0.02)


def test_deflated_sharpe_drops_as_trials_rise():
    r = np.full(200, 0.004) + _rng(5).normal(0, 0.01, 200)
    dsr_1 = verify.deflated_sharpe_ratio(r, n_trials=1)
    dsr_many = verify.deflated_sharpe_ratio(r, n_trials=500)
    assert dsr_1 > dsr_many                 # more trials => harder to clear
    assert 0.0 <= dsr_many <= dsr_1 <= 1.0


def test_expected_max_sharpe_increases_with_trials():
    assert verify.expected_max_sharpe(0.1, 100) > verify.expected_max_sharpe(0.1, 10)
    assert verify.expected_max_sharpe(0.1, 1) == 0.0


def test_gate_passes_strong_edge():
    # Consistent positive drift, low noise => should clear every threshold.
    r = np.full(120, 0.012) + _rng(6).normal(0, 0.004, 120)
    v = verify.gate(r, periods_per_year=12, n_trials=1)
    assert v.passed
    assert all(c.passed for c in v.checks)
    assert "PASS" in v.render()


def test_gate_fails_pure_noise():
    r = _rng(7).normal(0, 0.02, 120)
    v = verify.gate(r, periods_per_year=12, n_trials=50)
    assert not v.passed
    assert "FAIL" in v.render()


def test_gate_fails_when_too_few_samples():
    r = np.full(5, 0.05)
    v = verify.gate(r, periods_per_year=12, n_trials=1)
    assert not v.passed
    oos = next(c for c in v.checks if c.name == "oos_obs")
    assert not oos.passed


def test_gate_thresholds_come_from_config():
    assert set(config.STRATEGY_GATE) == {
        "min_sharpe", "min_nw_tstat", "min_dsr", "max_drawdown", "min_oos_obs"}
