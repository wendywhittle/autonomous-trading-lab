"""Tests for the default research configuration."""

from atlab.research.config import DEFAULTS, ResearchConfig


def test_defaults_match_documented_values():
    config = ResearchConfig.defaults()
    assert (config.train_size, config.validate_size, config.test_size) == (63, 21, 21)
    assert config.step == 21
    assert config.expanding is True
    assert config.notional_per_trade == 10.0


def test_base_and_stress_costs_differ():
    config = ResearchConfig.defaults()
    assert config.base_costs.name == "retail-base"
    assert config.stress_costs.name == "retail-stress"
    assert config.stress_costs.per_trade_cost(10.0) > config.base_costs.per_trade_cost(10.0)


def test_gate_thresholds_present():
    config = ResearchConfig.defaults()
    assert config.min_window_fraction == 1.0
    assert config.min_neighbor_fraction == 0.8
    assert config.min_regimes == 2


def test_module_defaults_singleton():
    assert DEFAULTS == ResearchConfig.defaults()
