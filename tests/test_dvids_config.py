"""Focused config tests for DVIDS stock-footage support."""

from __future__ import annotations

from src.config.sections.duration import StockFootageConfig
from src.config.sections.infrastructure import APIKeysConfig


def test_stock_footage_defaults_dvids_disabled():
    """DVIDS should be opt-in by default."""
    config = StockFootageConfig()

    assert config.dvids_enabled is False


def test_api_keys_config_loads_dvids_from_environment(monkeypatch):
    """APIKeysConfig should hydrate the DVIDS key from the environment."""
    monkeypatch.setenv("DVIDS_API_KEY", "dvids_test")

    config = APIKeysConfig()

    assert config.dvids_api_key == "dvids_test"
